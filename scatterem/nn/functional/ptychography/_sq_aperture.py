"""The bisected threshold for the squared-radius aperture, and its launches.

The squared-radius spelling of ``probe_cone_pass()``
(see ``nn/functional/warp/ptychography/_aperture_sq.py`` for why it is
BIT-IDENTICAL, not close) is taken by ``correct_aberrations_kouter``,
the ``correct_aberrations_ksum_planes`` mask and the ``ik``-tiled
zero-aberration k-SUM; the ``_direct_ptychography_kouter`` C2R store
takes the half-spectrum twin of the first, and the two-for-one packed
store takes its packed twin.

Their kill switches (``SCATTEREM_KOUTER_SQ_APERTURE``,
``SCATTEREM_KOUTER_HALF_C2R``, ``SCATTEREM_MASK_SQ_APERTURE``,
``SCATTEREM_KSUM_SQ_APERTURE``) were deleted: the
squared spelling is taken whenever its threshold exists.  The original kernels
stay, because they run when it does not (degenerate geometry, see
:func:`_pctf_sq_threshold`), and the full-spectrum store wherever
:func:`.kouter_half.kouter_half_supported` declines.
"""

import math

import torch
import warp as wp

from scatterem.nn._backends_kouter import KOUTER_HALF_ROWS
from scatterem.nn.functional.warp.ptychography import (
    KOUTER_HALF_ROWS as _ROW_COLUMNS,
)
from scatterem.nn.functional.warp.ptychography import (
    MASK_IK_BLOCK,
    _aperture_sq_threshold,
    _direct_ptychography_aperture_mask_planes,
    _direct_ptychography_aperture_mask_planes_sq,
    _direct_ptychography_forward_precomputed_A_kouter,
    _direct_ptychography_forward_precomputed_A_kouter_defocus_only,
    _direct_ptychography_forward_precomputed_A_kouter_sq,
    _direct_ptychography_forward_precomputed_A_kouter_sq_half,
    _direct_ptychography_forward_precomputed_A_kouter_sq_half_rows,
    _direct_ptychography_forward_precomputed_A_kouter_sq_packed,
)
from scatterem.physics import _warp_interop as wi
from scatterem.utils.env import resolve_backend

#: ``(wavelength, semiconvergence_angle, device)`` -> bisected float32 threshold.
#: The bisection is a one-thread launch plus a ``.item()`` (a device sync), so it
#: must not run per call: the geometry is fixed for a whole reconstruction and the
#: same keys recur.  Values are plain floats, so the memo pins no device memory.
_PCTF_SQ_THRESHOLD_CACHE: dict = {}


def _pctf_sq_threshold(
    wavelength: float, semiconvergence_angle: float, device
) -> float | None:
    """The float32 ``s`` threshold equivalent to ``probe_cone_pass()``'s predicate.

    Returns ``None`` when the geometry is degenerate (non-finite, or a wavelength
    that cannot bracket the bisection), in which case the caller keeps the
    original kernel -- a fallback rather than a raise, because this is a
    performance spelling and not a physics change.
    """
    if not (
        math.isfinite(wavelength)
        and math.isfinite(semiconvergence_angle)
        and wavelength > 0.0
    ):
        return None
    key = (wavelength, semiconvergence_angle, str(device))
    hit = _PCTF_SQ_THRESHOLD_CACHE.get(key)
    if hit is not None:
        return hit
    # ``hi0`` must FAIL the predicate.  ``s = (1/wavelength)**2`` gives
    # ``q * wavelength = 1``, i.e. a scattering angle of pi/2, and every
    # physical semiconvergence angle is below that.
    hi0 = (1.0 / wavelength) ** 2
    if not math.isfinite(hi0) or semiconvergence_angle >= math.pi / 2:
        return None
    out = torch.zeros(1, dtype=torch.float32, device=device)
    wp.launch(
        kernel=_aperture_sq_threshold,
        dim=1,
        inputs=[float(wavelength), float(semiconvergence_angle), float(hi0)],
        outputs=[wi.from_torch(out)],
        device=wi.device_from_torch(out.device),
    )
    val = float(out.item())
    if not math.isfinite(val) or val <= 0.0:
        return None
    _PCTF_SQ_THRESHOLD_CACHE[key] = val
    return val


#: The ``half`` value that selects the two-for-one packed store.  It is
#: also the mode of that store in the recorded-command key (:func:`_half_mode`).
HALF_PACKED = 3


def _half_mode(half: int) -> int:
    """0 = full grid, 1 = the half store per element, 2 = its row loop.

    :data:`HALF_PACKED` (3) is the two-for-one packed store.  The mode
    is part of the k-outer op's recorded-command KEY: each mode is a different
    kernel (and launch grid), so a replay must never cross one.
    """
    if not half:
        return 0
    if half == HALF_PACKED:
        return HALF_PACKED
    return 2 if resolve_backend(KOUTER_HALF_ROWS) == "rows" else 1


def _launch_dim(kern, dim: tuple) -> tuple:
    """The launch grid of ``kern`` over the output grid ``dim``.

    Only the row-loop half kernel differs: each thread covers ``KOUTER_HALF_ROWS``
    columns, so its last axis is ``ceil(Wh / KOUTER_HALF_ROWS)``.
    """
    if kern is _direct_ptychography_forward_precomputed_A_kouter_sq_half_rows:
        return (dim[0], dim[1], -(-dim[2] // int(_ROW_COLUMNS)))
    return dim


def _kouter_kernel(
    defocus_only: bool,
    sq: bool,
    half: int,
    wavelength: float,
    semiconvergence_angle: float,
    device,
):
    """``(kernel, tenth argument)`` for the k-outer op's RECORD path.

    The tenth kernel argument changes MEANING with the kernel -- an angle for the
    original, a squared-radius threshold for the ``_sq`` twins -- so the two are
    chosen together and never separately.  A ``None`` threshold (degenerate
    geometry) keeps the original kernel; the caller records it under the same key,
    so the decision replays consistently.  ``half`` without a threshold raises:
    :func:`.kouter_half.kouter_half_supported` has checked that it exists.
    ``half`` is :func:`_half_mode`'s value: 2 picks the row-loop twin
    and :data:`HALF_PACKED` the two-for-one packed twin.
    ``_defocus_only`` has no ``_sq`` twin, so the caller never sets ``sq`` with it.
    """
    if sq:
        s_thresh = _pctf_sq_threshold(
            float(wavelength), float(semiconvergence_angle), device
        )
        if s_thresh is not None:
            if half == HALF_PACKED:
                return (
                    _direct_ptychography_forward_precomputed_A_kouter_sq_packed,
                    s_thresh,
                )
            if half == 2:
                return (
                    _direct_ptychography_forward_precomputed_A_kouter_sq_half_rows,
                    s_thresh,
                )
            if half:
                return (
                    _direct_ptychography_forward_precomputed_A_kouter_sq_half,
                    s_thresh,
                )
            return _direct_ptychography_forward_precomputed_A_kouter_sq, s_thresh
        if half:
            raise ValueError("half-spectrum correction needs the squared aperture")
    if defocus_only:
        return (
            _direct_ptychography_forward_precomputed_A_kouter_defocus_only,
            float(semiconvergence_angle),
        )
    return _direct_ptychography_forward_precomputed_A_kouter, float(
        semiconvergence_angle
    )


def _launch_aperture_mask(
    mask: torch.Tensor,
    coords: tuple,
    sin_rot: float,
    cos_rot: float,
    semiconvergence_angle: float,
    wavelength: float,
    device,
) -> None:
    """Fill ``correct_aberrations_ksum_planes``' ``[Nqy, Nqx, Nk]`` aperture mask.

    ``coords`` is ``(Qx_wp, Qy_wp, Kx_wp, Ky_wp)``.  Unless the threshold
    is undefined, the squared-aperture kernel with its ``ik`` fold of ``MASK_IK_BLOCK`` builds a
    BIT-IDENTICAL mask -- the bisected threshold reproduces the ``probe_cone_pass()``
    predicate for every float32 input.  Measured 1.0846x on
    a benchmark row.
    """
    mask_wp = wi.from_torch(mask, dtype=wp.uint8)
    s_thresh = _pctf_sq_threshold(
        float(wavelength), float(semiconvergence_angle), mask.device
    )
    if s_thresh is None:
        wp.launch(
            kernel=_direct_ptychography_aperture_mask_planes,
            dim=tuple(int(v) for v in mask.shape),
            inputs=[*coords, sin_rot, cos_rot, semiconvergence_angle, wavelength],
            outputs=[mask_wp],
            device=device,
        )
        return
    blk = int(MASK_IK_BLOCK)
    wp.launch(
        kernel=_direct_ptychography_aperture_mask_planes_sq,
        dim=(
            int(mask.shape[0]),
            int(mask.shape[1]),
            (int(mask.shape[2]) + blk - 1) // blk,
        ),
        inputs=[*coords, sin_rot, cos_rot, float(s_thresh)],
        outputs=[mask_wp],
        device=device,
    )
