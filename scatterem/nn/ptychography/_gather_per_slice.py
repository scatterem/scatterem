"""Fused gather + transmit multiply for the plain (non-Warp) multislice route.

Main expands the ``(NZ, U)`` per-unique-pixel stack one slice at
a time (:func:`._transmit.gather_slice`), which already keeps at most one
gathered slice alive.
Where the fused Triton
:func:`scatterem.nn.functional._gather_cmul_triton.gather_cmul` accepts a slice,
the gather and the transmit multiply run as ONE kernel, so the gathered patches
are never written at all: 1.1496x on ``r3_simulator/medium/simulate --no-warp``,
peak 932.1 -> 804.1 MiB, bit-identical to ``index_select`` followed by
``patches.unsqueeze(0) * waves`` (``SCATTEREM_GATHER_CMUL=0`` forces the torch
ops).  It declines -- returns ``None`` -- under autograd (reverse mode, and forward
mode: a dual operand), off CUDA, off complex64,
on non-contiguous operands, without Triton and under ``torch.compile``, and the
caller then takes the unfused ops unchanged.  The compile guard sits HERE, before
the kernel is reached: ``gather_cmul`` bumps module-level call counters that
Dynamo guards on (a recompile on every call), and a raw Triton launch inside the
``cudagraphs`` backend fails to trace.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scatterem.nn.functional._forward_ad import carries_tangent

#: Lazily resolved fused gather + transmit multiply, or ``False`` without Triton.
_GATHER_CMUL = None


def _gather_cmul_fn():
    """The fused Triton ``gather_cmul``, or ``False`` when it cannot be imported."""
    global _GATHER_CMUL
    if _GATHER_CMUL is None:
        try:
            from scatterem.nn.functional._gather_cmul_triton import gather_cmul

            _GATHER_CMUL = gather_cmul
        except Exception:  # noqa: BLE001 - Triton is optional
            _GATHER_CMUL = False
    return _GATHER_CMUL


def fused_gather_transmit(
    sample_patches: Tensor, gather_index: Tensor, waves: Tensor
) -> Tensor | None:
    """``gather_slice(sample_patches, gather_index).unsqueeze(0) * waves``, fused.

    Returns ``None`` when the fused kernel does not apply; the result is then
    exactly what the caller's unfused ops produce.
    """
    if torch.compiler.is_compiling():
        return None
    fused = _gather_cmul_fn()
    if not fused or sample_patches.dim() != 1:
        return None
    if carries_tangent(sample_patches, waves):
        return None
    if tuple(waves.shape[1:]) != tuple(gather_index.shape):
        return None
    return fused(sample_patches, gather_index.reshape(-1), waves)
