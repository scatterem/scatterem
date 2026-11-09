"""The position-chunked Warp projection under grad, as ONE autograd node."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

# Imported for their REGISTRATION side effects: ``torch.ops.scatterem`` gains
# ``prism_project_fwd`` / ``prism_project_bwd`` from ``ops`` and
# ``prism_project_bwd_sdt_accum`` from ``_sdt_launch``.
from . import _sdt_launch as _sdt_launch
from . import ops as ops
from .ramp import _build_basis_rs


class _PrismProjectChunked(torch.autograd.Function):
    """The position-chunked Warp projection as ONE autograd node.

    ``prism_project(chunk=...)`` used to build one ``scatterem::prism_project_fwd``
    node PER CHUNK and ``torch.cat`` them.  Every one of those nodes produced its
    own full-size ``grad_Sdt`` -- ``(Bp, SY, SX)`` complex64, 1024 MiB at
    ``r2/medium`` and INDEPENDENT of ``chunk`` -- and autograd summed ``K/chunk``
    of them, so the memory-bounded route cost 1.87x the time of the unchunked one
    at ``chunk = K/4`` and 3.02x at ``chunk = 1`` (measured).

    One node instead means the backward owns the chunk loop, so it can carry ONE
    accumulator: the ``K/chunk - 1`` gradient adds vanish, and the per-chunk
    full-size writes collapse into one write plus a read-modify-write over only
    the S-matrix elements a chunk's own positions actually cover.

    The per-chunk ``_build_basis_rs`` (ramp + ifft2) stays OUTSIDE in the ordinary
    torch graph -- it is the same work either way and already carries its own
    adjoint.  What crosses the boundary is each chunk's ``basis_rs`` (as a real
    view) and the one shared ``Sdt``.  The forward is the same per-chunk
    ``prism_project_fwd`` calls as before, so its parent-beam sums are unchanged;
    only the order in which chunk gradients are added into ``grad_Sdt`` differs.
    """

    @staticmethod
    def forward(ctx, Sdt_real, positions, periodic, k_bcasts, bounds, *basis_rs_reals):
        outs = [
            torch.ops.scatterem.prism_project_fwd(
                b,
                Sdt_real,
                positions[i0:i1].contiguous(),
                periodic,
                False,  # mul_fp16: the half-precision routes are fp32 under grad
                kb,
            )
            for b, kb, (i0, i1) in zip(basis_rs_reals, k_bcasts, bounds, strict=True)
        ]
        ctx.save_for_backward(Sdt_real, positions, *basis_rs_reals)
        ctx.periodic = periodic
        ctx.k_bcasts = k_bcasts
        ctx.bounds = bounds
        return torch.cat(outs, dim=1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not ctx.periodic:
            raise NotImplementedError(
                "prism_project backward with periodic=False is not implemented. "
                "The backward kernel uses cyclic crop only."
            )
        Sdt_real, positions = ctx.saved_tensors[0], ctx.saved_tensors[1]
        basis_rs_reals = ctx.saved_tensors[2:]
        need_sdt = bool(ctx.needs_input_grad[0])
        grad_out_real = grad_outputs[0].contiguous()

        # empty(), not zeros(): chunk 0 runs with ``accumulate=False`` and so
        # writes every element exactly once, zeros included.  Only chunks 1..n-1
        # read it back, and then only where their own positions land.
        grad_Sdt = torch.empty_like(Sdt_real) if need_sdt else None

        grad_bases = []
        chunks = zip(basis_rs_reals, ctx.k_bcasts, ctx.bounds, strict=True)
        for j, (b, kb, (i0, i1)) in enumerate(chunks):
            go = grad_out_real[:, i0:i1].contiguous()
            pos = positions[i0:i1].contiguous()
            if ctx.needs_input_grad[5 + j]:
                gb, _ = torch.ops.scatterem.prism_project_bwd(
                    go, b, Sdt_real, pos, True, False, kb
                )
                grad_bases.append(gb)
            else:
                grad_bases.append(None)
            if need_sdt:
                torch.ops.scatterem.prism_project_bwd_sdt_accum(
                    go, b, pos, kb, j > 0, grad_Sdt
                )
        # grads for (Sdt_real, positions, periodic, k_bcasts, bounds, *basis_rs)
        return (grad_Sdt, None, None, None, None, *grad_bases)


def _prism_project_warp_chunked(
    Sdt: Tensor,
    basis_fft: Tensor,
    positions: Tensor,
    dr: Tensor,
    chunk: int,
    periodic: bool,
    ident: Callable[[int], bool],
) -> Tensor:
    """Chunked Warp projection under grad through :class:`_PrismProjectChunked`.

    Parameters
    ----------
    Sdt, basis_fft, positions, dr : Tensor
        As in :func:`~.dispatch.prism_project`.
    chunk : int
        Positions per chunk (``< K``).
    periodic : bool
        Cyclic crop; the backward requires ``True``.
    ident : callable
        ``ident(i)`` -- whether the chunk starting at position ``i`` may fold its
        sub-pixel ramp (see ``_ramp_is_identity``).

    Returns
    -------
    Tensor
        ``(Nmodes, K, M1, M2)`` complex64 exit waves.
    """
    # Backward stays fp32 (the bwd kernels are fp32-only), matching
    # _prism_project_warp's own guard.
    Sdt_c = Sdt.to(torch.complex64) if Sdt.dtype == torch.complex32 else Sdt
    Sdt_real = torch.view_as_real(Sdt_c).contiguous()
    pos_i32 = positions.to(torch.int32).contiguous()
    K = positions.shape[0]
    basis_list: list[Tensor] = []
    kb_list: list[int] = []
    bounds: list[tuple[int, int]] = []
    for i in range(0, K, chunk):
        j = min(i + chunk, K)
        basis_rs = _build_basis_rs(
            basis_fft, dr[i:j], "fp32", fold_k=True, ramp_identity=ident(i)
        )
        kpos = j - i
        basis_list.append(torch.view_as_real(basis_rs).contiguous())
        kb_list.append(kpos if (basis_rs.shape[1] == 1 and kpos > 1) else 0)
        bounds.append((i, j))
    out_real = _PrismProjectChunked.apply(
        Sdt_real, pos_i32, periodic, tuple(kb_list), tuple(bounds), *basis_list
    )
    return torch.view_as_complex(out_real)
