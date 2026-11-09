"""Host-to-device copy staging and a semi-convergence angle unit guard.

Dataset input coercion lives in :mod:`scatterem.data.validation`. This module
keeps the two helpers that are not container-specific: a pipelined pinned H2D
copy used when large cubes land on CUDA, and a warning when a probe angle looks
like it was supplied in the other unit.
"""

from __future__ import annotations

from warnings import warn

import torch

from scatterem.utils.env import env_flag

# Copies at or above this size take the pinned double-buffer path when the
# destination is CUDA. Smaller transfers are cheaper as a plain ``.to``.
_H2D_STAGE_MIN_BYTES = 256 << 20

# Bytes per staging slot. Two slots are live at once (host fill overlaps DMA).
_H2D_STAGE_CHUNK_BYTES = 16 << 20


def _h2d_staging_enabled() -> bool:
    """Whether ``SCATTEREM_H2D_STAGING`` allows the pinned pipeline (default on)."""
    return env_flag("SCATTEREM_H2D_STAGING", default=True)


def _can_stage_h2d(tensor: torch.Tensor, device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and tensor.device.type == "cpu"
        and tensor.is_contiguous()
        and tensor.numel() * tensor.element_size() >= _H2D_STAGE_MIN_BYTES
        and _h2d_staging_enabled()
        and torch.cuda.is_available()
    )


def _pinned_pair(chunk_elems: int, dtype: torch.dtype) -> list[torch.Tensor] | None:
    try:
        return [
            torch.empty(chunk_elems, dtype=dtype, pin_memory=True) for _ in range(2)
        ]
    except RuntimeError:
        return None


def to_device_staged(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move ``tensor`` onto ``device``, optionally via pinned host staging.

    Result is bit-identical to ``tensor.to(device)``. Staging runs only for a
    large contiguous host tensor going to CUDA when the env switch allows it;
    otherwise this is a plain ``.to``. Pinning failures fall back the same way.
    DMAs use the current CUDA stream of ``device``.
    """
    if not _can_stage_h2d(tensor, device):
        return tensor.to(device=device)

    elems_per_chunk = max(1, _H2D_STAGE_CHUNK_BYTES // tensor.element_size())
    bufs = _pinned_pair(elems_per_chunk, tensor.dtype)
    if bufs is None:
        return tensor.to(device=device)

    flat_src = tensor.reshape(-1)
    result = torch.empty_like(tensor, device=device)
    flat_dst = result.reshape(-1)
    events = [torch.cuda.Event() for _ in range(2)]
    used = [False, False]
    slot = 0
    for start in range(0, flat_src.numel(), elems_per_chunk):
        stop = min(start + elems_per_chunk, flat_src.numel())
        if used[slot]:
            events[slot].synchronize()
        host_view = bufs[slot][: stop - start]
        host_view.copy_(flat_src[start:stop])
        flat_dst[start:stop].copy_(host_view, non_blocking=True)
        events[slot].record(torch.cuda.current_stream(flat_dst.device))
        used[slot] = True
        slot ^= 1
    for event, was_used in zip(events, used, strict=True):
        if was_used:
            event.synchronize()
    return result


# Probe semi-angles sit near 1e-3..0.1 rad (1..100 mrad). Values near 0.5 are
# empty for both units, so a crossed-unit call is detectable without converting.
_ANGLE_CROSS_UNIT_THRESHOLD = 0.5


def check_semiconvergence_angle_units(
    value: float, unit: str, *, where: str = ""
) -> float:
    """Return ``value`` unchanged; warn if it looks like the other angle unit.

    ``unit`` is the unit the calling API documents (``"rad"`` or ``"mrad"``).
    This never converts; it only surfaces an implausible magnitude.
    """
    if unit not in ("rad", "mrad"):
        raise ValueError(f"unit must be 'rad' or 'mrad', got {unit!r}")
    angle = float(value)
    if angle <= 0.0:
        return angle

    prefix = f"{where}: " if where else ""
    if unit == "mrad" and angle < _ANGLE_CROSS_UNIT_THRESHOLD:
        warn(
            f"{prefix}semiconvergence_angle={angle!r} is in MILLIradians here, and "
            f"{angle} mrad is far smaller than any real probe. Did you pass radians? "
            f"({angle} rad = {angle * 1000.0} mrad.)",
            UserWarning,
            stacklevel=3,
        )
    elif unit == "rad" and angle > _ANGLE_CROSS_UNIT_THRESHOLD:
        warn(
            f"{prefix}semiconvergence_angle={angle!r} is in RADIANS here, and "
            f"{angle} rad is far larger than any real probe. Did you pass "
            f"milliradians? ({angle} mrad = {angle / 1000.0} rad.)",
            UserWarning,
            stacklevel=3,
        )
    return angle
