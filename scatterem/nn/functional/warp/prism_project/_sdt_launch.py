"""The ``grad_Sdt`` half of the PRISM adjoint: launch, and a shared-buffer op.

:func:`_launch_grad_sdt` was lifted verbatim out of ``ops._prism_project_bwd``
so that the chunked grad path (``_chunked._PrismProjectChunked``) can drive
the same two routes -- gather or atomic scatter -- with ONE shared accumulator
instead of one full-size ``(Bp, SY, SX)`` buffer per chunk.
``accumulate=False`` reproduces the original behaviour exactly.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.nn._backends_launch import PRISM_GRAD_SDT_LAUNCH
from scatterem.physics import _warp_interop as wi
from scatterem.utils.env import resolve_backend

from ._bwd_kernels import _prism_project_bwd_sdt_kernel
from ._sdt_gather_kernels import (
    _PRISM_BWD_SDT_TILE_P,
    _PRISM_BWD_SDT_TILE_P_KB0,
    _SDT_GATHER_MAX_WASTE,
    _prism_project_bwd_sdt_gather_kernel,
    _prism_project_bwd_sdt_gather_tiled_kb0_kernel,
    _prism_project_bwd_sdt_gather_tiled_kernel,
)
from ._sdt_recorded import launch_tiled_gather_recorded, wrapped_positions


def _launch_gather(
    grad_out_wp,
    basis_rs_wp,
    positions: Tensor,
    grad_Sdt: Tensor,
    dims: tuple[int, int, int, int, int],
    k_bcast: int,
    accumulate: bool,
    device,
) -> None:
    """Gather route: one thread per output element, no atomics."""
    K, Nmodes, Bp, SY, SX = dims
    # the ``prism_grad_sdt_launch`` row lets the tiled gather issue from a
    # reusable command (``_sdt_recorded``).  Same kernel, same arguments.
    recorded = Bp > 1 and (
        resolve_backend(PRISM_GRAD_SDT_LAUNCH, None, positions) == "recorded"
    )
    # Wrap once, on (K, 2), so the kernel's inner loop needs no division.
    if recorded:
        pos_n = wrapped_positions(positions, SY, SX)
    else:
        pos_n = torch.stack(
            (
                torch.remainder(positions[:, 0], SY),
                torch.remainder(positions[:, 1], SX),
            ),
            dim=1,
        ).contiguous()
    kb = 1 if k_bcast > 0 else 0
    if Bp > 1:
        # p-TILED: the window inversion, its two bounds tests and the
        # cotangent load are all p-INDEPENDENT, and the untiled kernel
        # repeats them once per parent beam.  ~1 fp32 ulp from the untiled
        # kernel rather than bit-identical (see the kernel's own comment),
        # gated against complex128 truth.  Bp == 1 has nothing to tile and
        # keeps the original launch.
        #
        # The two basis LAYOUTS want different tiles -- 4 when the basis is
        # stored K-broadcast, 32 when it is not -- so pick the instantiation
        # that matches the ``kb`` being passed.  Same body, same maths; the
        # only thing that differs is how many beams a thread owns.
        if kb == 1:
            tiled_kernel = _prism_project_bwd_sdt_gather_tiled_kernel
            tile_p = _PRISM_BWD_SDT_TILE_P
        else:
            tiled_kernel = _prism_project_bwd_sdt_gather_tiled_kb0_kernel
            tile_p = _PRISM_BWD_SDT_TILE_P_KB0
        ntile = -(-Bp // tile_p)  # ceil
        kernel, dim = tiled_kernel, (ntile, SY, SX)
        scalars = [int(K), int(Nmodes), int(SY), int(SX), kb, int(Bp)]
        if recorded and launch_tiled_gather_recorded(
            kernel,
            dim,
            grad_out_wp,
            basis_rs_wp,
            pos_n,
            (*scalars, int(accumulate)),
            grad_Sdt,
            device,
        ):
            return
    else:
        kernel, dim = _prism_project_bwd_sdt_gather_kernel, (Bp, SY, SX)
        scalars = [int(K), int(Nmodes), int(SY), int(SX), kb]
    pos_n_wp = wi.from_torch(pos_n, dtype=wp.int32, requires_grad=False)
    grad_Sdt_wp = wi.from_torch(grad_Sdt, dtype=wp.vec2f, requires_grad=False)
    wp.launch(
        kernel=kernel,
        dim=dim,
        inputs=[grad_out_wp, basis_rs_wp, pos_n_wp, *scalars, int(accumulate)],
        outputs=[grad_Sdt_wp],
        device=device,
    )


def _launch_grad_sdt(
    grad_out_wp,
    basis_rs_real: Tensor,  # (Nmodes, Kb, Bp, M1, M2, 2) float32
    positions: Tensor,  # (K, 2) int32
    pos_wp,
    *,
    K: int,
    k_bcast: int,
    grid: tuple[int, int],
    device,
    out: Tensor | None,
    accumulate: bool,
) -> Tensor:
    """Write the ``grad_Sdt`` half of the PRISM adjoint INTO ``out``.

    ``accumulate=False`` reproduces the original behaviour exactly (allocate when
    ``out`` is None, write every element once); ``accumulate=True`` requires
    ``out`` and adds into it.  Returns the ``(Bp, SY, SX, 2)`` float32 buffer.
    """
    Nmodes, Kb, Bp, M1, M2, _ = basis_rs_real.shape
    SY, SX = grid
    basis_rs_flat = basis_rs_real.reshape(Nmodes * Kb, Bp, M1, M2, 2)
    basis_rs_wp = wi.from_torch(basis_rs_flat, dtype=wp.vec2f, requires_grad=False)
    # Gather when the window inverts uniquely (M1 <= SY, M2 <= SX) and the grid
    # is not so oversampled that testing all K positions per output element
    # costs more than the atomics it removes.  Outside that envelope, fall back
    # to the original scatter — which also stays as the parity reference.
    if M1 <= SY and M2 <= SX and SY * SX <= _SDT_GATHER_MAX_WASTE * M1 * M2:
        # empty(), not zeros(): the gather writes every element exactly once.
        grad_Sdt = out
        if grad_Sdt is None:
            grad_Sdt = torch.empty(
                (Bp, SY, SX, 2), dtype=torch.float32, device=basis_rs_real.device
            )
        dims = (K, Nmodes, Bp, SY, SX)
        _launch_gather(
            grad_out_wp,
            basis_rs_wp,
            positions,
            grad_Sdt,
            dims,
            k_bcast,
            accumulate,
            device,
        )
        return grad_Sdt
    # The scatter route needs no ``accumulate`` flag: it already adds into the
    # buffer with ``wp.atomic_add``, so sharing an accumulator is just a matter of
    # zeroing it once and then not re-zeroing it per chunk.
    grad_Sdt = out
    if grad_Sdt is None:
        grad_Sdt = torch.zeros(
            (Bp, SY, SX, 2), dtype=torch.float32, device=basis_rs_real.device
        )
    elif not accumulate:
        grad_Sdt.zero_()
    wp.launch(
        kernel=_prism_project_bwd_sdt_kernel,
        dim=(Nmodes * K, Bp, M1 * M2),
        inputs=[
            grad_out_wp,
            basis_rs_wp,
            pos_wp,
            int(K),
            int(SY),
            int(SX),
            int(1 if k_bcast > 0 else 0),
        ],
        outputs=[wi.from_torch(grad_Sdt, dtype=wp.vec2f, requires_grad=False)],
        device=device,
    )
    return grad_Sdt


@torch.library.custom_op(
    "scatterem::prism_project_bwd_sdt_accum", mutates_args={"grad_Sdt"}
)
def _prism_project_bwd_sdt_accum(
    grad_out_real: Tensor,  # (Nmodes, kc, M1, M2, 2) float32
    basis_rs_real: Tensor,  # (Nmodes, Kb, Bp, M1, M2, 2) float32
    positions: Tensor,  # (kc, 2) int32
    k_bcast: int,
    accumulate: bool,
    grad_Sdt: Tensor,  # (Bp, SY, SX, 2) float32 -- MUTATED
) -> None:
    """Add one chunk's ``grad_Sdt`` contribution into a shared accumulator.

    The sibling of ``ops._prism_project_bwd``'s ``grad_Sdt`` half, differing only
    in where the answer goes: ``_prism_project_bwd`` allocates a fresh full-size
    ``(Bp, SY, SX)`` buffer per call, which is what made the chunked grad path
    1.87x SLOWER than the unchunked one -- autograd then had to reduce
    ``K/chunk`` of those.  Here every chunk adds into one buffer.

    Parameters
    ----------
    grad_out_real : Tensor
        ``(Nmodes, kc, M1, M2, 2)`` float32 cotangent of this chunk.
    basis_rs_real : Tensor
        ``(Nmodes, Kb, Bp, M1, M2, 2)`` float32 real-space basis of this chunk.
    positions : Tensor
        ``(kc, 2)`` int32 scan positions of this chunk.
    k_bcast : int
        As in ``prism_project_fwd``: ``> 0`` means the basis is K-broadcast.
    accumulate : bool
        ``False`` overwrites ``grad_Sdt`` (the first chunk), ``True`` adds.
    grad_Sdt : Tensor
        ``(Bp, SY, SX, 2)`` float32 accumulator, mutated in place.
    """
    Nmodes, Kb, _Bp, M1, M2, _ = basis_rs_real.shape
    K = k_bcast if k_bcast > 0 else Kb
    _, SY, SX, _ = grad_Sdt.shape
    device = wi.device_from_torch(basis_rs_real.device)
    grad_out_flat = grad_out_real.reshape(Nmodes * K, M1, M2, 2).contiguous()
    grad_out_wp = wi.from_torch(grad_out_flat, dtype=wp.vec2f, requires_grad=False)
    pos_wp = wi.from_torch(positions, dtype=wp.int32, requires_grad=False)
    _launch_grad_sdt(
        grad_out_wp,
        basis_rs_real,
        positions,
        pos_wp,
        K=K,
        k_bcast=k_bcast,
        grid=(SY, SX),
        device=device,
        out=grad_Sdt,
        accumulate=bool(accumulate),
    )


@_prism_project_bwd_sdt_accum.register_fake
def _(grad_out_real, basis_rs_real, positions, k_bcast, accumulate, grad_Sdt):
    return None
