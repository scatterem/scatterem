"""The block loop of the torch PRISM projection, and its two fused shortcuts.

:func:`_contract_blocks` is ``_prism_project_torch``'s blocked body: per block of
parents and positions, ramp multiply -> ``ifft2`` -> cyclic crop -> product ->
sum over the block's parents, added into the result.  Two perf-lab shortcuts
replace parts of it on CUDA, each a backend row that declines to the eager
spelling, and each ``torch.equal`` to what it replaces:

* ``crop_contract_exact``: crop gather, product, ``p``-sum and the
  running add as one Triton pass that sums in TensorIterator's order;
* ``prism_torch_ramp_mul``: where that pass consumes the block, the ramp
  multiply is the Warp ``_ramp_mul`` (K-outer, the same products) instead of the
  eager stride-0 broadcast, and the pass reads the permuted ``ifft2`` output by
  stride.

``prism_torch_ramp_mul_grad``: the frozen-probe GRAD block (only ``Sdt``
needs a gradient) takes the same Warp multiply; its consumer,
``_CropContractFrozenBasis``, reads the K-outer view by stride.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scatterem.nn._backends import PRISM_TORCH_RAMP_MUL, PRISM_TORCH_RAMP_MUL_GRAD
from scatterem.utils.env import resolve_backend

from ._torch_adjoint import _RampMulKSum

#: ATen's outer reduce splits across CTAs from here (``crop_contract_exact``).
_MAX_NB_EXACT = 1021
#: Threads per block of ``_prism_ramp_mul``'s ``(M1, M2)`` launch (Warp's default).
_RAMP_MUL_LAUNCH_BLOCK = 256
_SM_COUNT: dict[int, int] = {}


class _Crop:
    """The cyclic crop ``Sdt[p, (py_k + y) % SY, (px_k + x) % SX]`` of a block."""

    def __init__(self, Sdt, Sdt_flat, crop_idx, gy, gx, in_bounds):
        self.Sdt, self.flat, self.idx = Sdt, Sdt_flat, crop_idx
        self.gy, self.gx, self.in_bounds = gy, gx, in_bounds

    def __call__(self, psl: slice, ksl: slice, shape) -> Tensor:
        if self.flat is None:
            out = self.Sdt[psl][:, self.gy[ksl], self.gx[ksl]]
        else:
            out = self.flat[psl].index_select(1, self.index(ksl, shape)).view(shape)
        if self.in_bounds is not None:
            out = out * self.in_bounds[:, ksl]  # zero out-of-bounds positions
        return out

    def index(self, ksl: slice, shape) -> Tensor:
        """The flat crop index of positions ``ksl`` (needs ``self.idx``)."""
        m = shape[-2] * shape[-1]
        return self.idx.view(-1, m)[ksl].reshape(-1)

    def exact_ok(self) -> bool:
        """A flat, periodic, int32-indexed crop: what ``crop_contract_exact`` reads."""
        return (
            self.flat is not None
            and self.in_bounds is None
            and self.idx.dtype is torch.int32
        )


def _ramp_mul(basis: Tensor, ramp: Tensor) -> Tensor:
    """``basis[:, :, None] * ramp``; graph-building, with a fused ``basis`` adjoint.

    when ``basis`` needs a gradient, :class:`._torch_adjoint._RampMulKSum`
    computes the same product (bit-identical forward) and returns autograd's
    ``basis`` adjoint ``(grad * conj(ramp)).sum over K`` in one pass, without
    writing the ``(Nmodes, Bp, K, M1, M2)`` product it would otherwise reduce.
    """
    if torch.is_grad_enabled() and basis.requires_grad:
        return _RampMulKSum.apply(basis, ramp)
    return basis[:, :, None, :, :] * ramp


def _ramp_mul_tile(K: int) -> int | None:
    from ._ramp_kernels import _ramp_mul_tile as tile  # imports Warp

    return tile(K)


def _warp_ramp_mul_kouter(basis: Tensor, ramp: Tensor) -> Tensor:
    """``basis[:, :, None] * ramp`` by the Warp ``_ramp_mul``, as a K-outer view.

    ``ramp`` is ``(1, 1, K, M1, M2)``; the result indexes as ``(Nm, Bp, K, M1,
    M2)`` but is laid out ``(Nm, K, Bp, M1, M2)``.  Same products, bit for bit
    (``ramp._ramp_mul``).
    """
    from .ramp import _ramp_mul as warp_ramp_mul  # ramp imports torch_oracle

    return warp_ramp_mul(basis, ramp[0, 0]).permute(0, 2, 1, 3, 4)


def _ramp_mul_fills_device(basis: Tensor) -> bool:
    """Whether ``_prism_ramp_mul``'s ``(M1, M2)`` grid has one block per SM.

    The kernel measured 1.62-1.77x the eager broadcast from 100 blocks
    up (M >= 160 on 84 SMs) but 1.105x at 64 blocks (M = 128), where its device
    saving is less than the Warp launch's fixed host cost
    (``r2_prism_project/small/project_fwd_backward`` lost 0.9645x wall).
    """
    idx = basis.device.index
    if idx is None:
        idx = torch.cuda.current_device()
    n_sm = _SM_COUNT.get(idx)
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT[idx] = n_sm
    M1, M2 = basis.shape[-2:]
    return M1 * M2 >= _RAMP_MUL_LAUNCH_BLOCK * n_sm


def _ramp_mul_operands_ok(basis: Tensor, ramp: Tensor) -> bool:
    return (
        basis.is_cuda
        and basis.dtype == torch.complex64
        and ramp.dtype == torch.complex64
        and basis.ndim == 4
        and basis.shape[0] == 1
        and ramp.ndim == 5
        and not torch.compiler.is_compiling()
        and _ramp_mul_tile(int(ramp.shape[2])) is not None
    )


def torch_ramp_mul_supported(basis: Tensor, ramp: Tensor, crop: _Crop) -> bool:
    """True if the no-grad block takes the Warp ramp multiply.

    Only where ``crop_contract_exact`` will consume the K-outer result (no
    gradient, one mode, a flat periodic int32 crop, ``1 <= nb < 1021``) and a
    ramp tile divides the block's ``K``.
    """
    return (
        not (torch.is_grad_enabled() and (basis.requires_grad or ramp.requires_grad))
        and crop.exact_ok()
        and 1 <= basis.shape[1] < _MAX_NB_EXACT
        and _ramp_mul_operands_ok(basis, ramp)
    )


def torch_ramp_mul_grad_supported(basis: Tensor, ramp: Tensor, Sdt: Tensor) -> bool:
    """True if the frozen-probe grad block takes the Warp ramp multiply.

    Nothing differentiates through the product there (only ``Sdt`` needs a
    gradient), so it needs no adjoint; the plane must fill the device.
    """
    return (
        Sdt.requires_grad
        and not basis.requires_grad
        and not ramp.requires_grad
        and _ramp_mul_operands_ok(basis, ramp)
        and _ramp_mul_fills_device(basis)
    )


def _grad_block_basis(basis_fft: Tensor, ramp: Tensor, Sdt: Tensor) -> Tensor:
    """The grad path's per-position real-space basis ``(Nm, Bp, K, M1, M2)``."""
    if resolve_backend(PRISM_TORCH_RAMP_MUL_GRAD, None, basis_fft, ramp, Sdt) == (
        "warp"
    ):
        shifted = _warp_ramp_mul_kouter(basis_fft, ramp)
    else:
        shifted = _ramp_mul(basis_fft, ramp)
    return torch.fft.ifft2(shifted, norm="forward")


def _crop_contract_exact():
    try:
        from scatterem.nn.functional._crop_contract_exact_triton import (
            crop_contract_exact,
        )
    except ImportError:  # pragma: no cover - a CUDA build without Triton
        return None
    return crop_contract_exact


def _block_basis(basis_blk: Tensor, ramp_k: Tensor, crop: _Crop) -> Tensor:
    """``ifft2(basis_blk[:, :, None] * ramp_k)``, K-outer when the Warp row takes it."""
    if resolve_backend(PRISM_TORCH_RAMP_MUL, None, basis_blk, ramp_k, crop) == "warp":
        shifted = _warp_ramp_mul_kouter(basis_blk, ramp_k)
    else:
        shifted = _ramp_mul(basis_blk, ramp_k)
    # Unnormalised inverse DFT -- the amplitude ``_prism_project_torch`` specifies,
    # with NO trailing multiply by (M1*M2)**0.5.
    return torch.fft.ifft2(shifted, norm="forward")


def _add_block(out, basis_k, crop: _Crop, psl: slice, ksl: slice, shape) -> None:
    """``out[:, ksl] += (basis_k * crop[None]).sum(1)``, fused where exact."""
    nk = shape[1]
    if out.is_cuda and out.shape[0] == 1 and crop.exact_ok():
        fn = _crop_contract_exact()
        if fn is not None:
            sdt = crop.flat[psl]
            if (
                fn(basis_k, sdt, crop.index(ksl, shape), nk, out=out[:, ksl])
                is not None
            ):
                return
    if not basis_k.is_contiguous() and basis_k.stride(2) != 0:
        # ``.sum(1)``'s reduce order follows the operand strides, so the
        # K-outer view is restored to the p-outer layout first.
        basis_k = basis_k.contiguous()
    out[:, ksl] += (basis_k * crop(psl, ksl, shape)[None]).sum(1)


def _contract_blocks(basis_fft, ramp, crop: _Crop, shape, PB: int, KB: int) -> Tensor:
    """``sum_p basis_k[m, p] * crop[p, k]`` in parent blocks of ``PB`` and position
    blocks of ``KB``; ``ramp is None`` is the ramp-identity route.

    The result starts at zero and every parent block is ADDED to it in parent
    order: ``0 + x`` is ``x`` exactly, so one block (any case in which a single
    position fits the budget) is bit for bit the unblocked contraction.
    """
    Nmodes, Bp, K, M1, M2 = shape
    out = crop.Sdt.new_zeros((Nmodes, K, M1, M2))
    for p0 in range(0, Bp, PB):
        psl = slice(p0, min(p0 + PB, Bp))
        npb = psl.stop - psl.start
        # A factored basis scatters just this block; a tensor is sliced.
        basis_blk = (
            basis_fft[:, psl]
            if isinstance(basis_fft, Tensor)
            else basis_fft.block(p0, psl.stop)
        )  # (Nmodes, npb, M1, M2)
        if ramp is None:
            # The ramp-identity IFFT does not depend on k at all, so it is taken
            # ONCE per parent block; the expand below is a stride-0 view.
            basis_blk = torch.fft.ifft2(basis_blk, norm="forward")
        for k0 in range(0, K, KB):
            ksl = slice(k0, min(k0 + KB, K))
            nk = ksl.stop - ksl.start
            if ramp is None:
                # (Nmodes, npb, nk, M1, M2) as a stride-0 view over k -- the
                # broadcast the multiply below consumes without materialising it.
                basis_k = basis_blk[:, :, None, :, :].expand(Nmodes, npb, nk, M1, M2)
            else:
                basis_k = _block_basis(basis_blk, ramp[:, :, ksl], crop)
            # The sum over THIS block's parents, whole in every position block.
            _add_block(out, basis_k, crop, psl, ksl, (npb, nk, M1, M2))
    return out
