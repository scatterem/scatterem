"""A hand adjoint for the torch oracle's crop + parent-beam contraction.

Pure torch, so :func:`~.torch_oracle._prism_project_torch` stays the reference
the Warp route is graded against.  Only the BACKWARD differs from the autograd
body it replaces; the forward is the same statements, so it is ``torch.equal``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scatterem.nn._backends import PRISM_WINDOW_BWD
from scatterem.utils.env import switched_on


def _periodic_segments(p0: int, M: int, S: int) -> list[tuple[int, int, int, int]]:
    """Split the cyclic range ``(p0 + arange(M)) % S`` into plain slices.

    Parameters
    ----------
    p0 : int
        Start of the range; any integer, it is wrapped into ``[0, S)``.
    M : int
        Length of the range.  Requires ``M <= S``.
    S : int
        Period.

    Returns
    -------
    list of tuple of int
        ``[(dst_lo, dst_hi, src_lo, src_hi)]``: one segment, or two when the range
        wraps past ``S``.
    """
    p0 %= S
    if p0 + M <= S:
        return [(p0, p0 + M, 0, M)]
    n1 = S - p0
    return [(p0, S, 0, n1), (0, M - n1, n1, M)]


def _window_scatter(
    cg: Tensor,  # (Nmodes, K, M1, M2) conj(grad)
    basis_k: Tensor,  # (Nmodes, Bp, K, M1, M2)
    pos: list[list[int]],
    bp: int,
    grid: tuple[int, int],
) -> Tensor:
    """``sum_k`` of the per-position crop adjoint, as in-place window adds.

    Returns ``conj`` of the ``(Bp, SY, SX)`` gradient; the caller applies the
    outer ``conj`` once.  One position's periodic crop is at most 2x2 plain
    rectangles of the grid with no index repeated, so its adjoint is a strided
    add into a view, and ``addcmul_`` fuses the product into that add.
    """
    SY, SX = grid
    M1, M2 = basis_k.shape[-2:]
    acc = cg.new_zeros((bp, SY, SX))
    for k, (py, px) in enumerate(pos):
        for dy0, dy1, sy0, sy1 in _periodic_segments(int(py), M1, SY):
            for dx0, dx1, sx0, sx1 in _periodic_segments(int(px), M2, SX):
                win = acc[:, dy0:dy1, dx0:dx1]
                for m in range(cg.shape[0]):
                    win.addcmul_(
                        cg[m, k, sy0:sy1, sx0:sx1],
                        basis_k[m, :, k, sy0:sy1, sx0:sx1],
                    )
    return acc


class _CropContractFrozenBasis(torch.autograd.Function):
    """Crop + parent-beam contraction with a hand-written ``Sdt`` adjoint.

    Taken whenever ``Sdt`` needs a gradient (frozen probe, or joint with
    ``basis_k``).  Forward is the same statements as ``_prism_project_torch``'s
    grad-path body.  The backward is where it differs.

    autograd's ``mul`` adjoint is ``grad * basis_k.conj()``, and ``mul``
    resolves the conj bit with a full copy of ``basis_k`` (2 GiB read + 2 GiB
    written at ``r2_prism_project/medium``).  Here the conjugation is moved onto
    the small ``grad`` instead: ``g * conj(b) == conj(conj(g) * b)`` elementwise,
    bit for bit, because negating an operand's imaginary part is exact and
    commutes with every rounding in the complex product.  The outer ``conj`` is
    carried through the scatter (a sum, which negation also commutes with) and
    applied once, in place, to the ``(Bp, SY*SX)`` gradient.  Only saved INPUTS
    are used, so double backward stays correct.

    the scatter itself is then done per scan position
    (:func:`_window_scatter`).  The ``basis_k``-sized product is never written and
    no atomic is issued (1.67x on ``medium/project_backward --no-warp``).  It is
    NOT bit-exact against the ``index_add_`` body: ``addcmul``'s complex product
    rounds differently from ``mul``'s (same accuracy, 1.176e-07 vs 1.170e-07 max
    relative), and the positions are summed in ``k`` order rather than in atomic
    order.  In exchange the gradient is deterministic.  It needs the positions on
    the host (one sync in backward), so it is skipped under CUDA-graph capture and
    ``torch.compile``, when a crop is larger than the grid, and when
    ``SCATTEREM_PRISM_WINDOW_BWD=0`` (the ``prism_window_bwd`` backend row's
    switch); the ``index_add_`` body then runs.

    the JOINT case (``basis_k`` also requires grad) takes this Function
    too.  ``grad_basis_k`` is computed with the expression autograd's ``mul``
    adjoint uses, ``grad * conj(S_dt_crop)``, from the same saved crop, so it is
    unchanged element for element.  Under ``create_graph`` the crop is re-gathered
    from the saved ``Sdt_flat`` so double backward still sees its dependence on
    ``Sdt``.

    Triton kernels (CUDA only; each kernel is a backend row and declines to
    the eager spelling): the window scatter is one per-output-pixel gather
    (``prism_window_gather``); ``basis_k`` may arrive unexpanded,
    ``(Nmodes, Bp, 1, M1, M2)``, so ``grad_basis`` comes back K-summed
    (``bcast_mul_conj_ksum``); the forward's parent-beam sum
    ``(basis_k * S_dt_crop[None]).sum(1)`` is one fused pass
    (``crop_contract``; NOT bit-exact, a two-level ``p``-order sum) that reads the
    crop straight from ``Sdt_flat`` through ``crop_idx``, so the crop is neither
    written nor saved, and ``grad_basis`` gathers it the same way.
    """

    @staticmethod
    def forward(ctx, basis_k, Sdt_flat, crop_idx, crop_shape, positions, grid):
        joint = ctx.needs_input_grad[0]
        ctx.Sdt_flat = Sdt_flat if joint else None
        ctx.crop_shape = crop_shape
        ctx.sdt_shape = Sdt_flat.shape
        ctx.grid = grid
        if Sdt_flat.is_cuda:
            # the frozen-probe backward never reads the crop, so it is not
            # materialised at all (a 2 GiB ``index_select`` write plus its
            # read-back at ``r2_prism_project/medium``): the kernel gathers each
            # term's value from ``Sdt_flat`` through ``crop_idx``.  Bit-identical
            # to the crop path below -- the gathered values are the same floats.
            # the joint backward gathers ``grad_basis``'s crop values the
            # same way (``bcast_mul_conj(_ksum)``'s gather mode), so the joint
            # forward does not write or save the crop either.
            out = _crop_contract(
                basis_k, None, sdt=Sdt_flat, idx=crop_idx, crop_shape=crop_shape
            )
            if out is not None:
                ctx.save_for_backward(basis_k, crop_idx, positions, None)
                return out
        S_dt_crop = Sdt_flat.index_select(1, crop_idx).view(crop_shape)
        # ``basis_k`` may be ``(Nmodes, Bp, 1, M1, M2)``, broadcast over the
        # K positions by the product below rather than by the caller, so the
        # backward can return its gradient already summed over K.
        # The crop is what autograd's ``mul`` saved; saved tensors are released
        # after a non-retained backward, a ``ctx`` attribute would not be.
        ctx.save_for_backward(
            basis_k, crop_idx, positions, S_dt_crop if joint else None
        )
        if S_dt_crop.is_cuda:
            # the product and the sum over ``p`` in one pass; the
            # (Nmodes, Bp, K, M1, M2) product (2 GiB at ``r2_prism_project/medium``)
            # is never written.  Not bit-exact (two-level ``p``-order sum).
            out = _crop_contract(basis_k, S_dt_crop)
            if out is not None:
                return out
        if not basis_k.is_contiguous() and basis_k.stride(2) != 0:
            # ``.sum(1)``'s reduce order follows the operand strides, so a
            # K-outer view is restored to the p-outer layout first.
            basis_k = basis_k.contiguous()
        return (basis_k * S_dt_crop[None]).sum(1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0]
        basis_k, crop_idx, positions, S_dt_crop = ctx.saved_tensors
        SY, SX = ctx.grid
        M1, M2 = basis_k.shape[-2:]
        ksum = basis_k.shape[2] != ctx.crop_shape[1]
        if ksum:
            basis_k = basis_k.expand(*basis_k.shape[:2], *ctx.crop_shape[1:])
        window = (
            ctx.needs_input_grad[1]
            and switched_on(PRISM_WINDOW_BWD, "window")
            and M1 <= SY
            and M2 <= SX
            and not torch.compiler.is_compiling()
            and not (grad.is_cuda and torch.cuda.is_current_stream_capturing())
        )
        # Read the positions before anything is queued: in the joint case the
        # sync would otherwise drain the ``grad_basis`` product, and the device
        # would then idle while the host issues the window adds.
        pos = positions.tolist() if window else None
        grad_basis = None
        if ctx.needs_input_grad[0]:
            grad_basis = _grad_basis(ctx, grad, S_dt_crop, crop_idx, ksum)
        if not ctx.needs_input_grad[1]:
            return grad_basis, None, None, None, None, None
        if pos is not None:
            grad_Sdt = _window_grad_sdt(grad, basis_k, pos, ctx.sdt_shape, (SY, SX))
            return grad_basis, grad_Sdt, None, None, None, None
        prod = torch.conj_physical(grad).unsqueeze(1) * basis_k
        prod = prod.sum(0) if prod.shape[0] > 1 else prod[0]
        grad_Sdt = grad.new_zeros(ctx.sdt_shape)
        grad_Sdt.index_add_(1, crop_idx, prod.reshape(prod.shape[0], -1))
        return grad_basis, grad_Sdt.conj_physical_(), None, None, None, None


def _crop_contract(basis_k: Tensor, crop: Tensor | None, **gather) -> Tensor | None:
    """The Triton crop contraction, or ``None`` when it declines."""
    try:
        from scatterem.nn.functional._crop_contract_triton import crop_contract
    except ImportError:  # pragma: no cover - a CUDA build without Triton
        return None
    return crop_contract(basis_k, crop, **gather)


def _grad_basis(
    ctx, grad: Tensor, S_dt_crop: Tensor | None, crop_idx: Tensor, ksum: bool
) -> Tensor:
    """``grad_basis_k``: autograd's ``mul`` adjoint, K-summed for a broadcast basis.

    ``S_dt_crop`` is ``None`` when the forward took the crop-free gather mode; the
    crop is then read through ``crop_idx`` or re-gathered for the eager
    expression, the same floats either way.
    """
    if torch.is_grad_enabled():
        # create_graph: re-gather so double backward sees the crop's dependence
        # on ``Sdt``.
        S_dt_crop = None
    elif grad.is_cuda:
        # ``mul`` resolves the conj with a full copy of the crop
        # (2 GiB at ``r2_prism_project/medium``) before reading it back; the
        # kernel negates the imaginary part in registers instead, bit for bit the
        # same product (see its module docstring).  A K-broadcast basis
        # takes the product summed over K in registers -- the K-sized product is
        # never written.
        try:
            from scatterem.nn.functional import _bcast_mul_conj_triton as bmc
        except ImportError:  # pragma: no cover - a CUDA build without Triton
            bmc = None
        if bmc is not None:
            fn = bmc.bcast_mul_conj_ksum if ksum else bmc.bcast_mul_conj
            if S_dt_crop is None:
                out = fn(grad, None, sdt=ctx.Sdt_flat, idx=crop_idx)
            else:
                out = fn(grad, S_dt_crop)
            if out is not None:
                return out
    crop: Tensor = (
        ctx.Sdt_flat.index_select(1, crop_idx).view(ctx.crop_shape)
        if S_dt_crop is None
        else S_dt_crop
    )
    grad_basis = grad.unsqueeze(1) * crop[None].conj()
    return grad_basis.sum(2, keepdim=True) if ksum else grad_basis


def _window_grad_sdt(
    grad: Tensor,
    basis_k: Tensor,
    pos: list[list[int]],
    sdt_shape: torch.Size,
    grid: tuple[int, int],
) -> Tensor:
    """``grad_Sdt`` by the per-position window scatter or its gather.

    on CUDA outside grad mode the fill, the ``K`` window read-modify-writes
    and the in-place conj are one per-output-pixel Triton gather that adds the
    same terms in the same order and writes each element once (one rounding per
    term from the loop: the loop's broadcast ``addcmul_`` takes ATen's strided
    kernel).  Otherwise, or when the gather declines, the loop runs.
    """
    if grad.is_cuda and not torch.is_grad_enabled():
        try:
            from scatterem.nn.functional._prism_window_gather_triton import (
                window_gather_grad_sdt,
            )
        except ImportError:  # pragma: no cover - a CUDA build without Triton
            window_gather_grad_sdt = None
        if window_gather_grad_sdt is not None:
            out = window_gather_grad_sdt(grad, basis_k, pos, grid)
            if out is not None:
                return out.view(sdt_shape)
    cg = torch.conj_physical(grad)  # (Nmodes, K, M1, M2)
    acc = _window_scatter(cg, basis_k, pos, sdt_shape[0], grid)
    return acc.view(sdt_shape).conj_physical_()


class _RampMulKSum(torch.autograd.Function):
    """``basis[:, :, None] * ramp`` whose ``basis`` adjoint is one fused pass.

    Forward is the eager multiply itself (bit-identical).  Autograd's adjoint
    w.r.t. ``basis`` is ``(grad * ramp.conj()).sum_to_size(...)``, which writes the
    whole ``(Nmodes, Bp, K, M1, M2)`` product only for the sum over ``K`` to read
    it back (``mul`` 9.44 + ``sum`` 3.45 ms at ``r2_prism_project/medium``).
    On CUDA outside grad mode the Triton kernel forms the same products and sums
    them over ``K`` in TensorIterator's order, so the result is ``torch.equal``
    to the eager chain (see ``_ramp_adjoint_ksum_triton``).  Under
    ``create_graph``, and for the ``ramp`` adjoint, the eager expressions are used.
    """

    @staticmethod
    def forward(ctx, basis, ramp):
        ctx.save_for_backward(basis, ramp)
        return basis[:, :, None] * ramp

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0]
        basis, ramp = ctx.saved_tensors
        grad_basis = grad_ramp = None
        if ctx.needs_input_grad[0]:
            if grad.is_cuda and not torch.is_grad_enabled():
                try:
                    from scatterem.nn.functional._ramp_adjoint_ksum_triton import (
                        ramp_adjoint_ksum,
                    )
                except ImportError:  # pragma: no cover - CUDA build without Triton
                    ramp_adjoint_ksum = None
                if ramp_adjoint_ksum is not None:
                    grad_basis = ramp_adjoint_ksum(grad, ramp)
            if grad_basis is None:
                grad_basis = (grad * ramp.conj()).sum(2)
        if ctx.needs_input_grad[1]:
            grad_ramp = (grad * basis[:, :, None].conj()).sum_to_size(ramp.shape)
        return grad_basis, grad_ramp
