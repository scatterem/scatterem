"""Fused multislice transmission: ``waves_out = exp_patches[inverse_index] * waves``.

The MSPT forward gathers the per-unique-pixel (exponentiated) potential to
per-scan-position patches and multiplies it into the exit wave, once per slice.
Done as separate ``index_select`` + complex ``mul`` ops this is the dominant
elementwise cost (the gather materializes a [M,K,ph,pw] temporary and the mul
re-reads it). This Warp op fuses gather + complex-multiply into a single kernel
(no temporary), with an atomic-scatter backward — replacing the slow
sort-based advanced-index backward entirely.

Forward:  out[m,k,y,x] = waves[m,k,y,x] * exp_patches[inv[k,y,x]]
Backward (PyTorch complex convention, out = a*b -> grad_a = grad_out*conj(b)):
  grad_waves[m,k,y,x]   = grad_out[m,k,y,x] * conj(exp_patches[inv[k,y,x]])
  grad_exp_patches[u]   = sum_{inv[k,y,x]=u} grad_out[m,k,y,x] * conj(waves[m,k,y,x])

Precision: the kernels are generic over the complex value type — wp.vec2
(float32 components, complex64) or wp.vec2h (float16 components, complex32),
selected at runtime from the input dtype. The forward output and grad_waves
follow the input precision. The grad_exp_patches scatter, however, accumulates
many contributions per unique pixel, so it ALWAYS reduces in float32 (the
kernel promotes fp16 inputs to float32 and atomic-adds into a float32
accumulator) to preserve gradient precision; the fully-reduced grad is narrowed
back to the leaf complex dtype at the autograd boundary.
"""

from typing import Any

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp._complex_funcs import _cconj, _cmul
from scatterem.physics import _warp_interop as wi


@wp.kernel
def _transmit_fwd_blocked_kernel(
    exp_patches: wp.array(dtype=Any, ndim=1),  # [U]  # pyright: ignore
    inv: wp.array(dtype=wp.int32, ndim=3),  # [K, ph, pw]  # pyright: ignore
    waves: wp.array(dtype=Any, ndim=4),  # [M, K, ph, pw]  # pyright: ignore
    n_modes: int,
    out: wp.array(dtype=Any, ndim=4),  # [M, K, ph, pw]  # pyright: ignore
) -> None:
    # The mode-BLOCKED forward: the same product, with the mode-independent work
    # hoisted out of the grid and into a register.
    #
    # The per-thread-mode kernel it replaced (deleted, with
    # its ``SCATTEREM_TRANSMIT_FWD_MODE_BLOCK=0`` switch) launched over ``(M, K,
    # ph, pw)``, so both ``inv[k,y,x]`` and the gathered ``exp_patches[u]`` --
    # neither of which reads ``m`` -- were re-fetched once per thread, i.e. M
    # times per pixel.  This is the same defect that was fused away in the
    # backward (``_transmit_bwd_fused_kernel``), ported to the forward here.  At
    # M=4 modes, K=256 and 128x128 complex64:
    #
    #   grid  : waves 128 + out 128 + inv 64 MiB  = 320 MiB
    #   block : waves 128 + out 128 + inv 16 MiB  = 272 MiB
    #
    # and, not counted in that arithmetic but measured to matter more, the random
    # ``exp_patches`` gather drops from M loads per pixel to one.
    #
    # BIT-EXACT, unlike the backward's fusion: there is no scatter here, so the
    # thread body is the unchanged expression ``_cmul(waves[m,k,y,x], t)`` with a
    # loop-invariant ``t``.  Nothing is reassociated -- the backward's ~1 ULP
    # ``grad_waves`` perturbation came from nvcc re-contracting the product once
    # the scatter's arithmetic shared the thread body, which has no analogue in a
    # pure elementwise map.
    k, y, x = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    t = exp_patches[inv[k, y, x]]
    for m in range(n_modes):
        out[m, k, y, x] = _cmul(waves[m, k, y, x], t)


@wp.kernel
def _transmit_bwd_waves_kernel(
    grad_out: wp.array(dtype=Any, ndim=4),  # pyright: ignore[reportInvalidTypeForm]
    exp_patches: wp.array(dtype=Any, ndim=1),  # pyright: ignore[reportInvalidTypeForm]
    inv: wp.array(dtype=wp.int32, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    grad_waves: wp.array(dtype=Any, ndim=4),  # pyright: ignore[reportInvalidTypeForm]
) -> None:
    # Generic over the complex value type; grad_waves follows the value type.
    m, k, y, x = wp.tid()  # pyright: ignore
    t = exp_patches[inv[k, y, x]]
    grad_waves[m, k, y, x] = _cmul(grad_out[m, k, y, x], _cconj(t))


@wp.kernel
def _transmit_bwd_expp_kernel(
    grad_out: wp.array(dtype=Any, ndim=4),  # generic INPUT value type
    waves: wp.array(dtype=Any, ndim=4),  # generic INPUT value type  # pyright: ignore
    inv: wp.array(dtype=wp.int32, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    grad_exp_patches: wp.array(dtype=wp.vec2, ndim=1),  # FIXED float32 accumulator
) -> None:
    # The scatter accumulator stays float32 (wp.vec2) even when the input value
    # type is fp16 (wp.vec2h): many contributions land on the same unique pixel,
    # and fp16 atomic accumulation would destroy gradient precision. So we
    # promote the input components to float32, do the complex product in float32,
    # and atomic-add into the float32 accumulator.
    m, k, y, x = wp.tid()  # pyright: ignore
    go = grad_out[m, k, y, x]
    wv = waves[m, k, y, x]
    # Promote components to float32 before the product (inputs may be fp16).
    go32 = wp.vec2(wp.float32(go[0]), wp.float32(go[1]))
    wv32 = wp.vec2(wp.float32(wv[0]), wp.float32(wv[1]))
    g = _cmul(go32, _cconj(wv32))
    wp.atomic_add(grad_exp_patches, inv[k, y, x], g)


@wp.kernel
def _transmit_bwd_fused_kernel(
    grad_out: wp.array(dtype=Any, ndim=4),  # generic INPUT value type
    exp_patches: wp.array(dtype=Any, ndim=1),  # generic INPUT value type
    waves: wp.array(dtype=Any, ndim=4),  # generic INPUT value type  # pyright: ignore
    inv: wp.array(dtype=wp.int32, ndim=3),  # pyright: ignore[reportInvalidTypeForm]
    n_modes: int,
    grad_waves: wp.array(dtype=Any, ndim=4),  # follows the INPUT value type
    grad_exp_patches: wp.array(dtype=wp.vec2, ndim=1),  # FIXED float32 accumulator
) -> None:
    # Both adjoints in one pass, blocked over the MODE axis.
    #
    # The two split kernels above launch over ``(M, K, ph, pw)`` and BOTH read the
    # whole of ``grad_out``; ``inv`` and the gathered ``exp_patches`` value are
    # mode-INDEPENDENT yet are re-fetched once per thread, i.e. M times per pixel.
    # At M=4 modes, K=256 and 128x128 complex64 that
    # is 640 MiB of compulsory traffic where 400 suffices:
    #
    #   split : grad_out 128 + grad_waves 128 + inv 64   (waves kernel)
    #         + grad_out 128 + waves      128 + inv 64   (expp  kernel) = 640 MiB
    #   fused : grad_out 128 + waves 128 + grad_waves 128 + inv 16      = 400 MiB
    #
    # Measured 1.2584 -> 0.7674 ms = 1.640x on the pair at that shape.
    #
    # WHY THE MODE LOOP IS LOAD-BEARING FOR EXACTNESS, not just for traffic.  The
    # straight fusion at ``dim=(M, K, ph, pw)`` is 1.522x but perturbs
    # ``grad_waves`` by ~1 ULP: nvcc contracts the complex product differently
    # once the scatter's arithmetic shares the thread body (four hand-written
    # spellings all give byte-identical deviations, so it is a codegen decision,
    # not statement order).  Hoisting the conjugated transmission into a
    # loop-invariant register restores the split kernel's exact expression --
    # ``grad_waves`` is ``torch.equal`` to the split form, which is what keeps the
    # ptychography gate's bit-exact ``grad_probe`` digest intact.
    #
    # ``grad_exp_patches`` DOES change: the M per-pixel contributions are summed in
    # a register and land as ONE atomic instead of M.  That is a reassociation of a
    # sum whose order is already nondeterministic (measured deviation 1.1e-06 of
    # max|grad|, against 3.6e-07 of launch-to-launch atomic jitter in the split
    # form), and it is strictly the more accurate order.
    k, y, x = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    u = inv[k, y, x]
    tc = _cconj(exp_patches[u])
    acc = wp.vec2(0.0, 0.0)
    for m in range(n_modes):
        go = grad_out[m, k, y, x]
        grad_waves[m, k, y, x] = _cmul(go, tc)
        wv = waves[m, k, y, x]
        # Promote components to float32 before the product (inputs may be fp16).
        go32 = wp.vec2(wp.float32(go[0]), wp.float32(go[1]))
        wv32 = wp.vec2(wp.float32(wv[0]), wp.float32(wv[1]))
        acc = acc + _cmul(go32, _cconj(wv32))
    wp.atomic_add(grad_exp_patches, u, acc)


# Pre-declare both concrete instantiations at module load so the kernels are not
# JIT-recompiled per launch (which would break CUDA-graph capture). wp.vec2 is
# the fp32 (complex64) value type; wp.vec2h is the fp16 (complex32) value type.
for _vt in (wp.vec2h, wp.vec2):
    wp.overload(
        _transmit_fwd_blocked_kernel,
        {
            "exp_patches": wp.array(dtype=_vt, ndim=1),
            "waves": wp.array(dtype=_vt, ndim=4),
            "out": wp.array(dtype=_vt, ndim=4),
        },
    )
    wp.overload(
        _transmit_bwd_waves_kernel,
        {
            "grad_out": wp.array(dtype=_vt, ndim=4),
            "exp_patches": wp.array(dtype=_vt, ndim=1),
            "grad_waves": wp.array(dtype=_vt, ndim=4),
        },
    )
    # expp kernel: overload only the INPUT value type; the accumulator stays
    # fixed at the float32 wp.vec2 declared in the signature.
    wp.overload(
        _transmit_bwd_expp_kernel,
        {
            "grad_out": wp.array(dtype=_vt, ndim=4),
            "waves": wp.array(dtype=_vt, ndim=4),
        },
    )
    # fused kernel: same rule -- inputs and grad_waves follow the value type, the
    # scatter accumulator stays fixed at the float32 wp.vec2.
    wp.overload(
        _transmit_bwd_fused_kernel,
        {
            "grad_out": wp.array(dtype=_vt, ndim=4),
            "exp_patches": wp.array(dtype=_vt, ndim=1),
            "waves": wp.array(dtype=_vt, ndim=4),
            "grad_waves": wp.array(dtype=_vt, ndim=4),
        },
    )
del _vt


@torch.library.custom_op("scatterem::transmit_fwd", mutates_args=())
def _transmit_fwd(
    exp_patches_real: Tensor,  # [U, 2] float32/float16 (view_as_real of complex)
    inv: Tensor,  # [K, ph, pw] int32
    waves_real: Tensor,  # [M, K, ph, pw, 2] float32/float16
) -> Tensor:
    # Generic over precision: real views are float32 (complex64) or float16
    # (complex32); the matching Warp value type (wp.vec2 / wp.vec2h) is picked
    # at runtime and the output follows the input real dtype.
    if waves_real.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            f"transmit supports float16/float32 real views, got {waves_real.dtype}"
        )
    M, K, ph, pw, _ = waves_real.shape
    vt = wp.vec2h if waves_real.dtype == torch.float16 else wp.vec2
    out = torch.empty_like(waves_real)
    device = wi.device_from_torch(waves_real.device)
    exp_wp = wi.from_torch(exp_patches_real, dtype=vt, requires_grad=False)
    inv_wp = wi.from_torch(inv, requires_grad=False)
    waves_wp = wi.from_torch(waves_real, dtype=vt, requires_grad=False)
    out_wp = wi.from_torch(out, dtype=vt, requires_grad=False)
    # Mode-blocked launch: one ``inv`` load and one ``exp_patches`` gather per
    # PIXEL instead of per (mode, pixel).  Bit-exact to the per-thread-mode grid
    # it replaced -- see the kernel's own comment.
    wp.launch(
        kernel=_transmit_fwd_blocked_kernel,
        dim=(K, ph, pw),
        inputs=[exp_wp, inv_wp, waves_wp, M],
        outputs=[out_wp],
        device=device,
    )
    return out


@_transmit_fwd.register_fake
def _(exp_patches_real, inv, waves_real):
    return torch.empty_like(waves_real)


@torch.library.custom_op("scatterem::transmit_bwd", mutates_args=())
def _transmit_bwd(
    grad_out_real: Tensor,  # [M, K, ph, pw, 2] float32/float16
    exp_patches_real: Tensor,  # [U, 2] float32/float16
    inv: Tensor,  # [K, ph, pw] int32
    waves_real: Tensor,  # [M, K, ph, pw, 2] float32/float16
) -> tuple[Tensor, Tensor]:
    # Generic over precision. ``grad_waves`` follows the input real dtype
    # (wp.vec2 / wp.vec2h). ``grad_exp`` is ALWAYS allocated float32: the
    # scatter accumulates many contributions per unique pixel, so the reduction
    # runs in float32 (the kernel promotes fp16 inputs internally) regardless of
    # input precision. The fp32 grad_exp is cast back to the leaf dtype at the
    # autograd boundary in ``_backward``.
    if waves_real.dtype not in (torch.float16, torch.float32):
        raise TypeError(
            f"transmit supports float16/float32 real views, got {waves_real.dtype}"
        )
    M, K, ph, pw, _ = waves_real.shape
    device = wi.device_from_torch(waves_real.device)
    vt = wp.vec2h if waves_real.dtype == torch.float16 else wp.vec2

    grad_waves = torch.empty_like(waves_real)
    # FIXED float32 scatter accumulator (even in fp16 mode).
    grad_exp = torch.zeros(
        exp_patches_real.shape, dtype=torch.float32, device=exp_patches_real.device
    )

    grad_out_wp = wi.from_torch(grad_out_real, dtype=vt, requires_grad=False)
    exp_wp = wi.from_torch(exp_patches_real, dtype=vt, requires_grad=False)
    inv_wp = wi.from_torch(inv, requires_grad=False)
    waves_wp = wi.from_torch(waves_real, dtype=vt, requires_grad=False)

    # One fused launch, blocked over the mode axis: the two split kernels above
    # both stream the whole of ``grad_out`` and both re-read the mode-independent
    # ``inv``/``exp_patches`` once per mode.  See
    # ``_transmit_bwd_fused_kernel``'s comment for the traffic arithmetic and for
    # why the mode LOOP (rather than a mode grid axis) is what keeps
    # ``grad_waves`` bit-identical to the split spelling.  The split kernels are
    # kept above as the reference spelling for A/B comparisons.
    wp.launch(
        kernel=_transmit_bwd_fused_kernel,
        dim=(K, ph, pw),
        # grad_out / exp_patches / waves are the (possibly fp16) value type; the
        # accumulator binds to the fixed float32 wp.vec2 declared on the kernel.
        inputs=[grad_out_wp, exp_wp, waves_wp, inv_wp, M],
        outputs=[
            wi.from_torch(grad_waves, dtype=vt, requires_grad=False),
            wi.from_torch(grad_exp, dtype=wp.vec2, requires_grad=False),
        ],
        device=device,
    )
    return grad_exp, grad_waves


@_transmit_bwd.register_fake
def _(grad_out_real, exp_patches_real, inv, waves_real):
    # grad_exp is always float32 (fp32 scatter accumulation); grad_waves matches
    # the input real dtype.
    grad_exp = torch.empty(
        exp_patches_real.shape, dtype=torch.float32, device=exp_patches_real.device
    )
    return grad_exp, torch.empty_like(waves_real)


def _setup_context(ctx, inputs, output):
    exp_patches_real, inv, waves_real = inputs
    ctx.save_for_backward(exp_patches_real, inv, waves_real)


def _backward(ctx, grad_out):
    exp_patches_real, inv, waves_real = ctx.saved_tensors
    grad_exp, grad_waves = torch.ops.scatterem.transmit_bwd(
        grad_out.contiguous(), exp_patches_real, inv, waves_real
    )
    # ``grad_exp`` is ALWAYS float32 (fp32 scatter accumulation), but the
    # ``exp_patches_real`` leaf is float16 in fp16 (complex32) mode. PyTorch's
    # autograd actually ACCEPTS a mismatched grad dtype here and auto-casts it
    # to the leaf dtype when accumulating into ``.grad`` (verified on torch
    # 2.7.1: returning a float32 grad for a float16 custom-op input yields a
    # float16 ``.grad``). We nonetheless cast EXPLICITLY at this boundary so the
    # returned grad deterministically matches the leaf dtype and the result does
    # not depend on autograd's implicit-cast behaviour. The accumulation itself
    # stayed in float32 inside the kernel; only the final, fully-reduced grad is
    # narrowed to float16 here.
    if grad_exp.dtype != exp_patches_real.dtype:
        grad_exp = grad_exp.to(exp_patches_real.dtype)
    # grads for (exp_patches_real, inv, waves_real)
    return grad_exp, None, grad_waves


torch.library.register_autograd(
    "scatterem::transmit_fwd", _backward, setup_context=_setup_context
)


def fused_transmit(
    exp_patches: Tensor,  # [U] complex64 / complex32
    inverse_index: Tensor,  # [K, ph, pw] int
    waves: Tensor,  # [M, K, ph, pw] complex64 / complex32
) -> Tensor:
    """Fused ``exp_patches[inverse_index].unsqueeze(0) * waves`` (complex).

    Equivalent to gathering ``exp_patches`` through ``inverse_index`` and
    multiplying into ``waves`` (broadcast over the leading mode axis), in one
    Warp kernel with an atomic-scatter backward.

    Generic over precision: ``exp_patches``/``waves`` may be complex64 (fp32,
    wp.vec2) or complex32 (fp16, wp.vec2h); the output matches the input
    complex dtype. ``view_as_real`` yields the right real dtype (float32 /
    float16) for the custom op.

    NOTE (fp32 grad accumulation in fp16 mode): the backward scatter for
    ``exp_patches`` accumulates many contributions per unique pixel in float32
    even when the value type is fp16, so its raw gradient is complex64. To match
    the leaf dtype, that fully-reduced grad is narrowed back to complex32 at the
    autograd boundary (see ``_backward``); accumulation stays fp32, only the
    final returned grad is fp16.

    Parameters
    ----------
    exp_patches : Tensor
        ``(U,)`` complex64 or complex32 per-unique-pixel exponentiated
        potential.
    inverse_index : Tensor
        ``(K, ph, pw)`` integer index into ``exp_patches`` for each
        per-scan-position patch pixel.
    waves : Tensor
        ``(M, K, ph, pw)`` complex64 or complex32 exit waves (mode axis M).

    Returns
    -------
    Tensor
        ``(M, K, ph, pw)`` complex64 or complex32 (matching the inputs),
        equal to ``exp_patches[inverse_index].unsqueeze(0) * waves``.
    """
    exp_real = torch.view_as_real(exp_patches).contiguous()
    waves_real = torch.view_as_real(waves).contiguous()
    inv = inverse_index.to(torch.int32).contiguous()
    out_real = torch.ops.scatterem.transmit_fwd(exp_real, inv, waves_real)
    return torch.view_as_complex(out_real)
