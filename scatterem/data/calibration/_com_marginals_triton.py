"""One-pass detector marginals for the vectorised centre of mass.

Private, lazily imported by :func:`._com._measure_com_from_intensities` so that
``import scatterem`` never requires Triton.

Why this exists
---------------
The vectorised centre of mass needs only the two detector marginals of the
(optionally masked) cube, ``(I * m).sum(-1)`` and ``(I * m).sum(-2)``.  The torch
spelling reads the cube twice when unmasked, and when masked writes ``I * m``
through a block buffer and reads it back twice (``_masked_marginals``): ~4 cube
passes where 1 is compulsory.  The function is bandwidth-bound, so the pass count
is the cost.

:func:`com_marginals` runs one program per scan position: it streams that
position's ``(Qx, Qy)`` pattern in ``(BX, next_pow2(Qy))`` tiles, applies the mask
in registers, stores each tile's row sums into ``marginal_x`` and accumulates the
column sums in registers into ``marginal_y``.  The cube is read once and nothing
cube-sized is written.

Numerics
--------
NOT bit-exact to the torch spelling: both sums are reassociated (row sums are one
tree reduction; each column sum adds ``ceil(Qx / BX)`` tile partials in order).
The masked product ``I * m`` itself is the same single rounding.  At
``d27_com_calibration/large`` the marginals are within 1.6e-7 (x) / 3.6e-7 (y) of
float64 where torch's are within 1.3e-7 / 1.5e-7.

Gating
------
The ``com_marginals`` row of :mod:`scatterem.data._backends`, whose predicate is
:func:`com_marginals_supported`: Triton is importable; ``intensities`` is a
C-contiguous 4-D CUDA ``float32`` tensor with at least one scan position; the
mask, if any, is ``(Qx, Qy)`` on the same device and ``float32`` or ``bool`` (so
the product is ``float32``, as torch would promote it); ``Qy <= 1024``; and no
gradient is being recorded through either input (the kernel has no backward).
Integer cubes, other dtypes and CPU stay on the torch path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from scatterem.data._backends import COM_MARGINALS
from scatterem.utils.env import resolve_backend

__all__ = ["CALLS", "com_marginals", "com_marginals_supported"]

#: ``[ran, declined]``, host-side only -- proves the fast path was TAKEN.
CALLS = [0, 0]

#: Largest detector width held in one register row.
_MAX_QY = 1024
_NUM_WARPS = 4


@triton.jit
def _com_marginals_kernel(
    X,  # (N, QX, QY) f32 -- the cube, one pattern per scan position     [in]
    M,  # (QX, QY) f32 -- the detector mask (unused unless HAS_MASK)     [in]
    MX,  # (N, QX) f32 -- sum over qy                                    [out]
    MY,  # (N, QY) f32 -- sum over qx                                    [out]
    QX,
    QY,
    HAS_MASK: tl.constexpr,
    BX: tl.constexpr,
    BY: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BY)
    cmask = cols < QY
    acc_y = tl.zeros([BY], dtype=tl.float32)
    base = X + n * QX * QY
    for qx0 in range(0, QX, BX):
        rows = qx0 + tl.arange(0, BX)
        rmask = rows < QX
        off = rows[:, None] * QY + cols[None, :]
        lm = rmask[:, None] & cmask[None, :]
        v = tl.load(base + off, mask=lm, other=0.0, eviction_policy="evict_first")
        if HAS_MASK:
            v = v * tl.load(M + off, mask=lm, other=0.0, eviction_policy="evict_last")
        tl.store(MX + n * QX + rows, tl.sum(v, axis=1), mask=rmask)
        acc_y += tl.sum(v, axis=0)
    tl.store(MY + n * QY + cols, acc_y, mask=cmask)


def com_marginals_supported(
    intensities: torch.Tensor, mask: torch.Tensor | None = None
) -> bool:
    """True if :func:`com_marginals` may replace the torch marginals."""
    shape = tuple(intensities.shape)
    if (
        not intensities.is_cuda
        or intensities.dtype is not torch.float32
        or len(shape) != 4
        or not intensities.is_contiguous()
        or shape[0] * shape[1] == 0
        or shape[2] == 0
        or not 0 < shape[3] <= _MAX_QY
        or (torch.is_grad_enabled() and intensities.requires_grad)
    ):
        return False
    return mask is None or (
        tuple(mask.shape) == shape[2:]
        and mask.device == intensities.device
        and mask.dtype in (torch.float32, torch.bool)
        and not (torch.is_grad_enabled() and mask.requires_grad)
    )


def com_marginals(
    intensities: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``((I * m).sum(-1), (I * m).sum(-2))`` in one pass, or ``None`` if declined.

    ``intensities`` is ``(Rx, Ry, Qx, Qy)``; ``mask`` is ``(Qx, Qy)`` or ``None``
    (no mask).  ``None`` means the backend row declined and nothing was launched.
    """
    if resolve_backend(COM_MARGINALS, None, intensities, mask) != "triton":
        CALLS[1] += 1
        return None
    CALLS[0] += 1
    Rx, Ry, Qx, Qy = intensities.shape
    dev = intensities.device
    marginal_x = torch.empty((Rx, Ry, Qx), dtype=torch.float32, device=dev)
    marginal_y = torch.empty((Rx, Ry, Qy), dtype=torch.float32, device=dev)
    # The mask is (Qx, Qy): a float32 contiguous copy is a few hundred KiB at most.
    m = marginal_x if mask is None else mask.to(torch.float32).contiguous()
    BY = triton.next_power_of_2(Qy)
    _com_marginals_kernel[(Rx * Ry,)](  # pyright: ignore[reportIndexIssue]
        intensities,
        m,
        marginal_x,
        marginal_y,
        Qx,
        Qy,
        HAS_MASK=mask is not None,
        BX=max(1, min(8, 2048 // BY)),
        BY=BY,
        num_warps=_NUM_WARPS,
    )
    return marginal_x, marginal_y
