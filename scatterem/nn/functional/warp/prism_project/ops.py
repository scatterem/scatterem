"""The two ``torch.library.custom_op``s and the autograd wiring between them."""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.physics import _warp_interop as wi
from scatterem.utils.warp.launch import block_dim_kwargs

from ._bwd_kernels import (
    _PRISM_BWD_BASIS_BLOCK_DIM,
    _PRISM_BWD_BASIS_TILE_P,
    _prism_project_bwd_basis_bcast_kernel,
    _prism_project_bwd_basis_bcast_tiled_kernel,
    _prism_project_bwd_basis_kernel,
)
from ._fwd_kernels import (
    _PRISM_FWD_TILE_K,
    _PRISM_FWD_TILE_K_NARROW,
    _prism_project_fwd_kernel,
    _prism_project_fwd_tiled_kernel,
    _prism_project_fwd_tiled_narrow_kernel,
)
from ._sdt_launch import _launch_grad_sdt

#: Launch block size of both tiled forward instantiations.  Warp's
#: default is 256.  An in-situ census measured 128 at 1.343x on
#: this kernel, bit-exact, and 1.0617x on ``r2/medium/project_fwd_backward``
#: across separately launched processes.  Only the thread grouping changes; each
#: thread runs the same body, so the output is bit-identical.  ``None`` keeps Warp's
#: default (the reference the tests compare against).
#:
#: Passed to ``wp.launch`` directly, not through ``block_dim_kwargs``: the tiled
#: kernels are generic (``vec2`` / ``vec2h`` overloads), so the helper cannot read
#: their thread ceiling and would drop the request.  No clamp is needed: 128
#: threads at the 255-register maximum use 32 640 registers, under every CUDA
#: block's 64 K.
_PRISM_FWD_TILED_BLOCK_DIM: int | None = 128


def _fwd_tiled_launch_kw(device: torch.device) -> dict:
    """``wp.launch``'s ``device`` and, unless it is ``None``, ``block_dim``."""
    kw: dict = {"device": wi.device_from_torch(device)}
    if _PRISM_FWD_TILED_BLOCK_DIM is not None:
        kw["block_dim"] = _PRISM_FWD_TILED_BLOCK_DIM
    return kw


@torch.library.custom_op("scatterem::prism_project_fwd", mutates_args=())
def _prism_project_fwd(
    basis_rs_real: Tensor,  # (Nmodes, Kb, Bp, M1, M2, 2) float32/float16
    Sdt_real: Tensor,  # (Bp, SY, SX, 2) float32/float16
    positions: Tensor,  # (K, 2) int32
    periodic: bool = True,
    mul_fp16: bool = True,
    k_bcast: int = 0,
) -> Tensor:  # (Nmodes, K, M1, M2, 2) float32 (exit waves)
    # Generic over input precision: real views are float32 (complex64, wp.vec2)
    # or float16 (complex32, wp.vec2h); the matching Warp value type is picked at
    # runtime. The output is ALWAYS float32 — the parent-beam sum accumulates in
    # an fp32 register inside the kernel and the exit wave is written fp32.
    #
    # ``mul_fp16`` selects the per-beam multiply precision for fp16 (vec2h)
    # inputs: True multiplies in fp16 (the "fp16" preset); False casts each
    # operand to fp32 before the multiply (the "mixed" preset — fp32 arithmetic
    # on the complex32-stored S-matrix, with NO full-tensor upcast). For fp32
    # (vec2) inputs it is a no-op (the multiply is fp32 either way).
    #
    # ``k_bcast`` > 0 means ``basis_rs_real`` stores a SINGLE k slice per mode
    # (Kb == 1) and ``k_bcast`` is the true number of positions; the kernel then
    # reads row ``mk // K`` so every position shares one basis. 0 keeps the
    # historical layout, where the basis carries all K rows.
    if basis_rs_real.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            "prism_project supports float16/float32 real views, got "
            f"{basis_rs_real.dtype}"
        )
    Nmodes, Kb, Bp, M1, M2, _ = basis_rs_real.shape
    K = k_bcast if k_bcast > 0 else Kb
    _, SY, SX, _ = Sdt_real.shape
    vt = wp.vec2h if basis_rs_real.dtype == torch.float16 else wp.vec2
    # Warp arrays max out at 4 dims (the vec type consumes the trailing 2-axis);
    # fuse (Nmodes, Kb) -> Nmodes*Kb so the gather array is 4D and the output 3D.
    basis_rs_flat = basis_rs_real.reshape(Nmodes * Kb, Bp, M1, M2, 2)
    # Output exit waves stay fp32 (fp32 accumulator -> fp32 write), regardless of
    # input precision.
    out_flat = torch.zeros(
        (Nmodes * K, M1, M2, 2), dtype=torch.float32, device=basis_rs_real.device
    )
    # Broadcast basis (identity sub-pixel ramp): the basis load is k-independent,
    # so tile the k axis in registers and pay it once per _PRISM_FWD_TILE_K
    # positions instead of once per position. Bit-identical (same per-output
    # accumulation order); measured 7.46x on this kernel at the production shape.
    # Below K=4 there is too little redundancy to be worth the tail lanes, and
    # the untiled kernel is the only one that handles the non-broadcast layout
    # (there each k has its OWN basis row, so there is nothing to reuse).
    if k_bcast > 0 and K >= 4:
        # Pick the tile width from the number of positions that actually exist.
        # A TK-wide tile with K < TK clamps its tail lanes to k = K-1 and throws
        # their work away, so the narrow instantiation is strictly less waste
        # there; at K >= _PRISM_FWD_TILE_K nothing changes and the launch is the
        # bit-identical one measured at the production shape.
        if K >= _PRISM_FWD_TILE_K:
            tile_k = _PRISM_FWD_TILE_K
            tiled_kernel = _prism_project_fwd_tiled_kernel
        else:
            tile_k = _PRISM_FWD_TILE_K_NARROW
            tiled_kernel = _prism_project_fwd_tiled_narrow_kernel
        ntile = (K + tile_k - 1) // tile_k
        wp.launch(
            kernel=tiled_kernel,
            dim=(Nmodes * ntile, M1, M2),
            inputs=[
                wi.from_torch(basis_rs_flat, dtype=vt, requires_grad=False),
                wi.from_torch(Sdt_real, dtype=vt, requires_grad=False),
                wi.from_torch(positions, dtype=wp.int32, requires_grad=False),
                int(K),
                int(ntile),
                int(SY),
                int(SX),
                int(1 if periodic else 0),
                int(1 if mul_fp16 else 0),
            ],
            outputs=[wi.from_torch(out_flat, dtype=wp.vec2, requires_grad=False)],
            **_fwd_tiled_launch_kw(basis_rs_real.device),
        )
        return out_flat.reshape(Nmodes, K, M1, M2, 2)
    wp.launch(
        kernel=_prism_project_fwd_kernel,
        dim=(Nmodes * K, M1, M2),
        inputs=[
            wi.from_torch(basis_rs_flat, dtype=vt, requires_grad=False),
            wi.from_torch(Sdt_real, dtype=vt, requires_grad=False),
            wi.from_torch(positions, dtype=wp.int32, requires_grad=False),
            int(K),
            int(SY),
            int(SX),
            int(1 if periodic else 0),
            int(1 if mul_fp16 else 0),
            int(1 if k_bcast > 0 else 0),
        ],
        # Output binds to the fixed fp32 wp.vec2 declared on the kernel.
        outputs=[wi.from_torch(out_flat, dtype=wp.vec2, requires_grad=False)],
        device=wi.device_from_torch(basis_rs_real.device),
    )
    return out_flat.reshape(Nmodes, K, M1, M2, 2)


@_prism_project_fwd.register_fake
def _(basis_rs_real, Sdt_real, positions, periodic=True, mul_fp16=True, k_bcast=0):
    Nmodes, Kb, _Bp, M1, M2, _ = basis_rs_real.shape
    K = k_bcast if k_bcast > 0 else Kb
    # Exit waves are ALWAYS fp32 (fp32 accumulator), even when inputs are fp16.
    return basis_rs_real.new_empty((Nmodes, K, M1, M2, 2), dtype=torch.float32)


@torch.library.custom_op("scatterem::prism_project_bwd", mutates_args=())
def _prism_project_bwd(
    grad_out_real: Tensor,  # (Nmodes, K, M1, M2, 2) float32
    basis_rs_real: Tensor,  # (Nmodes, Kb, Bp, M1, M2, 2) float32
    Sdt_real: Tensor,  # (Bp, SY, SX, 2) float32
    positions: Tensor,  # (K, 2) int32
    need_basis_grad: bool = True,
    need_sdt_grad: bool = True,
    k_bcast: int = 0,
) -> tuple[Tensor, Tensor]:  # (grad_basis_rs, grad_Sdt) in *_real layout
    # ``need_basis_grad`` / ``need_sdt_grad`` mirror autograd's own
    # ``needs_input_grad``: each selects whether the corresponding kernel runs at
    # all.  A gradient the caller did not ask for is returned as an EMPTY tensor,
    # never as zeros -- zeros would be indistinguishable from a real zero gradient
    # and would let a mis-set flag pass silently.  The autograd wrapper turns the
    # empty result into ``None``, which is what a non-requiring input expects.
    # ``k_bcast`` > 0 mirrors the forward: the basis stores a SINGLE k slice per
    # mode (Kb == 1) and k_bcast is the true number of positions.  ``grad_out``
    # always carries all K rows; only the basis (and hence its gradient) is folded.
    Nmodes, Kb, Bp, M1, M2, _ = basis_rs_real.shape
    K = k_bcast if k_bcast > 0 else Kb
    _, SY, SX, _ = Sdt_real.shape
    device = wi.device_from_torch(basis_rs_real.device)

    # Fuse (Nmodes, K) -> Nmodes*K so arrays are 4D / 3D (Warp 4-dim cap).
    grad_out_flat = grad_out_real.reshape(Nmodes * K, M1, M2, 2).contiguous()

    grad_out_wp = wi.from_torch(grad_out_flat, dtype=wp.vec2f, requires_grad=False)
    pos_wp = wi.from_torch(positions, dtype=wp.int32, requires_grad=False)

    # grad_basis_rs: pure gather (one write per element).  Skipped entirely when
    # the basis needs no gradient -- both the launch and the
    # (Nmodes*K, Bp, M1, M2, 2) fp32 buffer, which is the largest allocation on
    # this path (2.0 GiB at r2/medium) and was previously filled and discarded.
    if need_basis_grad:
        grad_basis_rs_flat = torch.empty(
            (Nmodes * Kb, Bp, M1, M2, 2),
            dtype=torch.float32,
            device=grad_out_real.device,
        )
        Sdt_wp = wi.from_torch(Sdt_real, dtype=wp.vec2f, requires_grad=False)
        if k_bcast > 0:
            # One thread per stored element, reducing over k in a register.  The
            # gradient buffer and the launch are both K times smaller, and the
            # sum over K that torch would otherwise do on the (expand) adjoint
            # disappears with the K-sized buffer it would have read.
            #
            # p-TILED when there is more than one parent beam: the cotangent load
            # and the crop arithmetic inside the k loop are both p-INDEPENDENT, and
            # the untiled kernel repeats them once per beam.  Bp == 1 has nothing to
            # tile and keeps the original launch (which is also the parity
            # reference).
            if Bp > 1:
                ntile_b = -(-Bp // _PRISM_BWD_BASIS_TILE_P)  # ceil
                wp.launch(
                    kernel=_prism_project_bwd_basis_bcast_tiled_kernel,
                    dim=(Nmodes, ntile_b, M1 * M2),
                    inputs=[
                        grad_out_wp,
                        Sdt_wp,
                        pos_wp,
                        int(K),
                        int(SY),
                        int(SX),
                        int(Bp),
                    ],
                    outputs=[
                        wi.from_torch(
                            grad_basis_rs_flat, dtype=wp.vec2f, requires_grad=False
                        )
                    ],
                    device=device,
                    **block_dim_kwargs(
                        _prism_project_bwd_basis_bcast_tiled_kernel,
                        device,
                        _PRISM_BWD_BASIS_BLOCK_DIM,
                    ),
                )
            else:
                wp.launch(
                    kernel=_prism_project_bwd_basis_bcast_kernel,
                    dim=(Nmodes, Bp, M1 * M2),
                    inputs=[grad_out_wp, Sdt_wp, pos_wp, int(K), int(SY), int(SX)],
                    outputs=[
                        wi.from_torch(
                            grad_basis_rs_flat, dtype=wp.vec2f, requires_grad=False
                        )
                    ],
                    device=device,
                )
        else:
            wp.launch(
                kernel=_prism_project_bwd_basis_kernel,
                dim=(Nmodes * K, Bp, M1 * M2),
                inputs=[grad_out_wp, Sdt_wp, pos_wp, int(K), int(SY), int(SX)],
                outputs=[
                    wi.from_torch(
                        grad_basis_rs_flat, dtype=wp.vec2f, requires_grad=False
                    )
                ],
                device=device,
            )
        grad_basis_rs = grad_basis_rs_flat.reshape(Nmodes, Kb, Bp, M1, M2, 2)
    else:
        grad_basis_rs = basis_rs_real.new_empty((0,))

    # grad_Sdt: the cyclic crop, run one of two ways.  Skipped entirely when the
    # S-matrix needs no gradient.
    if need_sdt_grad:
        grad_Sdt = _launch_grad_sdt(
            grad_out_wp,
            basis_rs_real,
            positions,
            pos_wp,
            K=K,
            k_bcast=k_bcast,
            grid=(SY, SX),
            device=device,
            out=None,
            accumulate=False,
        )
    else:
        grad_Sdt = Sdt_real.new_empty((0,))

    return grad_basis_rs, grad_Sdt


@_prism_project_bwd.register_fake
def _(
    grad_out_real,
    basis_rs_real,
    Sdt_real,
    positions,
    need_basis_grad=True,
    need_sdt_grad=True,
    k_bcast=0,
):
    # grad_basis_rs always has the SAME shape as the stored basis, folded or not.
    return (
        (
            torch.empty_like(basis_rs_real)
            if need_basis_grad
            else basis_rs_real.new_empty((0,))
        ),
        torch.empty_like(Sdt_real) if need_sdt_grad else Sdt_real.new_empty((0,)),
    )


def _prism_project_fwd_setup_context(ctx, inputs, output):
    basis_rs_real, Sdt_real, positions, periodic, mul_fp16, k_bcast = inputs
    ctx.save_for_backward(basis_rs_real, Sdt_real, positions)
    ctx.periodic = periodic  # store for backward guard
    ctx.k_bcast = k_bcast  # store for backward guard
    # Which gradients autograd will actually consume.  Read off the inputs here
    # rather than from ``ctx.needs_input_grad`` in the backward: this op has six
    # inputs but ``needs_input_grad`` reports five entries (measured), so relying
    # on its positions would be guesswork.  ``requires_grad`` on the input tensor
    # is the same information and is unambiguous -- an input that does not require
    # grad has its returned gradient discarded by autograd either way.
    ctx.needs_basis_grad = bool(basis_rs_real.requires_grad)
    ctx.needs_sdt_grad = bool(Sdt_real.requires_grad)


def _prism_project_fwd_backward(ctx, grad_out):
    if not ctx.periodic:
        raise NotImplementedError(
            "prism_project_fwd backward with periodic=False is not implemented. "
            "The backward kernel uses cyclic crop only. "
            "Forward simulation runs under torch.no_grad(); "
            "if you need gradients through a finite-specimen projection, "
            "use periodic=True or implement a non-periodic backward kernel."
        )
    basis_rs_real, Sdt_real, positions = ctx.saved_tensors
    need_basis = ctx.needs_basis_grad
    need_sdt = ctx.needs_sdt_grad
    grad_basis_rs, grad_Sdt = torch.ops.scatterem.prism_project_bwd(
        grad_out.contiguous(),
        basis_rs_real,
        Sdt_real,
        positions,
        need_basis,
        need_sdt,
        ctx.k_bcast,
    )
    # grads for (basis_rs_real, Sdt_real, positions, periodic, mul_fp16, k_bcast);
    # positions is int -> None; periodic / mul_fp16 / k_bcast are non-Tensor -> None.
    # A gradient that was not requested comes back empty and is reported as None.
    return (
        grad_basis_rs if need_basis else None,
        grad_Sdt if need_sdt else None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "scatterem::prism_project_fwd",
    _prism_project_fwd_backward,
    setup_context=_prism_project_fwd_setup_context,
)
