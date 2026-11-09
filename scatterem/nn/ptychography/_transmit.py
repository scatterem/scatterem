"""Per-slice transmission and its custom adjoint.

The broadcast-transmit autograd Function, the compiled reductions it picks
between, and the eligibility test that routes into them. Module level so the
forward model and ``scan_exit_wave.multislice`` share one implementation.
"""

import weakref

import torch
from torch import Tensor

from scatterem.nn.modules.propagator import Propagator
from scatterem.nn._backends import CONJ_MUL_REDUCE_BCAST, SEPARABLE_GATHER
from scatterem.nn.functional._forward_ad import carries_tangent
from scatterem.nn.functional.warp.transmit import fused_transmit
from scatterem.utils.env import switched_on

from ._gather_per_slice import fused_gather_transmit


def _separable_gather(src_i64: Tensor, y_idx: Tensor, x_idx: Tensor, H: int, W: int):
    """``src[:, y_idx[:,:,None]*W + x_idx[:,None,:]]`` without the index tensor.

    The payload is passed as ``int64``, NOT ``complex64``: Inductor refuses to
    generate code for complex operators (it warns and silently falls back to
    eager, which is how this kernel stayed hidden), and a pure data movement
    does not care about the dtype -- ``complex64`` and ``int64`` are both 8
    bytes.  Viewed as ``int64`` the same expression compiles to a generated
    gather that computes the flat offset from two small tables instead of
    loading it from a full-size index.
    """
    return src_i64.view(src_i64.shape[0], H, W)[:, y_idx[:, :, None], x_idx[:, None, :]]


_SEPARABLE_GATHER_COMPILED = None
#: ``(token of the last index examined, a weak reference to that index, its
#: factor tables or None)``
_SEPARABLE_PROOF = None


def _separable_expand(src: Tensor, inverse_index: Tensor) -> Tensor | None:
    """Fast path for ``src.index_select(1, inverse_index.reshape(-1))``.

    ``inverse_index[n, iy, ix]`` is built by
    the batched unique-with-inverse step as ``(dy_n + iy) * W + (dx_n + ix)`` and is
    therefore **separable**: a rank-1 sum of two ``(K, ph)`` / ``(K, pw)``
    tables of a few hundred KiB, materialised as an ``N*ph*pw`` tensor.  The
    ``index_select`` kernel streams that index once per slice, so at
    ``r3_simulator/medium`` it reads 32 x 64 MiB = **2.00 GiB of index against
    a 4.00 GiB output** -- measured, not modelled: an int64 index costs
    1.3464x an int32 one there, against a 1.3333x traffic bound, and the int32
    kernel already runs at 96 % of this card's measured ``copy_`` roof.  The
    only way left is to stop loading the index at all.

    Returns ``None`` (caller keeps ``index_select``) unless every condition
    below holds.  The last one is the load-bearing one: the two recovered
    factor tables are required to **reproduce ``inverse_index`` exactly**, so
    this is a verified gate rather than an argued one -- a producer whose
    index is not of this form (a sparse scan, the ``torch.unique`` fallback
    when the union box is not fully covered, a fixed-size unique, or an index
    handed in by a dataset) simply fails the compare and takes the old path.
    The check costs one 64 MiB build plus a 128 MiB compare, ~3 % of what it
    is buying.

    Forward-only: viewing a complex payload as ``int64`` has no adjoint, so
    the gate refuses whenever grad is live and the gradient path is
    untouched.

    **OPT-IN**, and the reason is measured, not cautious: the generated kernel
    costs one ``torch.compile`` of **3.12 s** on the first call that reaches
    it, against a **4.24 ms** per-call saving -- it pays back after **736
    calls**.  A benchmark with warm-ups never sees that; a single
    ``simulate()`` would pay 3 s for nothing.  So the default is OFF and the
    library default path is byte-identical to before; set
    ``SCATTEREM_SEPARABLE_GATHER=1`` (the ``SEPARABLE_GATHER`` backend row's
    opt-in switch) for a long forward-only sweep (many
    tilts / frozen phonons / scan chunks in one process), where it is worth
    1.0257x on ``r3_simulator/medium/simulate --no-warp``.
    """
    if not switched_on(SEPARABLE_GATHER, "separable"):
        return None
    if (
        src.dim() != 2
        or src.dtype != torch.complex64
        or not src.is_contiguous()
        or inverse_index is None
        or inverse_index.dim() != 3
        or inverse_index.shape[1] < 2
        or torch.is_grad_enabled()
        or src.requires_grad
        or src.device.type != "cuda"
        or torch._dynamo.is_compiling()
    ):
        return None

    # W is the stride between two consecutive rows of the SAME patch, which the
    # construction fixes at exactly the box width; H then follows from the
    # unique count. One 2-element host read.
    row = inverse_index[0, :2, 0].tolist()
    W = int(row[1]) - int(row[0])
    U = src.shape[1]
    if W <= 0 or U % W != 0:
        return None
    H = U // W

    # The proof is per INDEX, not per payload: a slice-by-slice caller hands in
    # the same index NZ times, and re-proving it each time would cost more than
    # the gather it buys (one 64 MiB build + a 128 MiB compare against a
    # 128 MiB slice).
    # The weak reference makes the memo per tensor OBJECT: a freed index whose
    # block the caching allocator hands to the next batch's index has the same
    # data_ptr, _version and shape, and must not reuse the old tables.
    global _SEPARABLE_PROOF
    token = (inverse_index.data_ptr(), inverse_index._version, inverse_index.shape, W)
    if (
        _SEPARABLE_PROOF is None
        or _SEPARABLE_PROOF[0] != token
        or _SEPARABLE_PROOF[1]() is not inverse_index
    ):
        idx = (
            inverse_index if inverse_index.dtype == torch.int32 else inverse_index.int()
        )
        y_idx = torch.div(idx[:, :, 0], W, rounding_mode="floor").contiguous()
        x_idx = torch.remainder(idx[:, 0, :], W).contiguous()
        proven = torch.equal(idx, y_idx[:, :, None] * W + x_idx[:, None, :])
        tables = (y_idx, x_idx) if proven else None
        _SEPARABLE_PROOF = (token, weakref.ref(inverse_index), tables)
    if _SEPARABLE_PROOF[2] is None:
        return None
    y_idx, x_idx = _SEPARABLE_PROOF[2]

    global _SEPARABLE_GATHER_COMPILED
    try:
        if _SEPARABLE_GATHER_COMPILED is None:
            _SEPARABLE_GATHER_COMPILED = torch.compile(_separable_gather, dynamic=False)
        out = _SEPARABLE_GATHER_COMPILED(src.view(torch.int64), y_idx, x_idx, H, W)
    except Exception:  # noqa: BLE001 - any compile/runtime failure -> old path
        return None
    return out.view(torch.complex64)


def _r_mul(a: Tensor, b: Tensor) -> Tensor:
    """Complex multiply on real-interleaved ``(..., 2)`` tensors.

    For compiled callers: Inductor cannot codegen complex
    dtypes (every complex op falls back to an eager kernel with a full HBM
    round-trip), but expressed on real tensors the whole chain fuses.
    """
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return torch.stack((ar * br - ai * bi, ar * bi + ai * br), dim=-1)


def _r_exp_i(v: Tensor) -> Tensor:
    """``exp(1j * z)`` for real-interleaved ``z = (..., 2)``.

    ``exp(1j*(re + 1j*im)) = exp(-im) * (cos re + 1j sin re)`` — identical to
    the ``view_as_complex``-trick transmission in the complex path.
    """
    vr, vi = v.unbind(-1)
    amp = torch.exp(-vi)
    return torch.stack((amp * torch.cos(vr), amp * torch.sin(vr)), dim=-1)


#: Lazily resolved fused ``sum_m g[m]*conj(w[m])``, or ``False`` when Triton is absent.
#: Resolved once, not per call: this sits inside the multislice loop.
_CONJ_MUL_REDUCE = None


def _conj_mul_reduce_fn():
    global _CONJ_MUL_REDUCE
    if _CONJ_MUL_REDUCE is None:
        try:
            from scatterem.nn.functional._conj_mul_reduce_triton import conj_mul_reduce

            _CONJ_MUL_REDUCE = conj_mul_reduce
        except Exception:  # noqa: BLE001 - Triton is optional, same as _bcast_cmul
            _CONJ_MUL_REDUCE = False
    return _CONJ_MUL_REDUCE


#: Same, for the dual-output kernel used when BOTH gradients are wanted.
_CONJ_MUL_REDUCE_BCAST = None


#: Same, for the products-only kernel used above ``_MAX_MODES``.
_CONJ_MUL_BCAST_PRODUCTS = None


def _conj_mul_bcast_products_fn():
    global _CONJ_MUL_BCAST_PRODUCTS
    if _CONJ_MUL_BCAST_PRODUCTS is None:
        try:
            from scatterem.nn.functional._conj_mul_reduce_triton import (
                conj_mul_bcast_products,
            )

            _CONJ_MUL_BCAST_PRODUCTS = conj_mul_bcast_products
        except Exception:  # noqa: BLE001 - Triton is optional, same as _bcast_cmul
            _CONJ_MUL_BCAST_PRODUCTS = False
    return _CONJ_MUL_BCAST_PRODUCTS


#: Is the fused adjoint reachable from a TRACED region?  Set by the caller
#: before tracing: it resolves the lazy Triton import + registers the custom op, which must
#: happen before tracing starts (doing it inside a traced region is the untraceable thing
#: this flag exists to avoid).  Left False -> the compiled route keeps its MulBackward0
#: chain, i.e. the behaviour before the custom op.
_COMPILED_FUSED_READY = False


def _arm_compiled_fused_adjoint() -> bool:
    """Resolve the fused dual kernel and let the compiled route use it. Returns success.

    Honours ``SCATTEREM_CONJ_MUL_DUAL=0`` here rather than at the call site: this runs
    OUTSIDE any traced region, which is the only place an ``os.environ`` read is safe, and
    it keeps that switch meaning the same thing on both routes.  (It is also the knob the
    A/B in a benchmark script turns.)
    """
    global _COMPILED_FUSED_READY
    enabled = switched_on(CONJ_MUL_REDUCE_BCAST, "triton")
    _COMPILED_FUSED_READY = enabled and bool(_conj_mul_reduce_bcast_fn())
    return _COMPILED_FUSED_READY


def _conj_mul_reduce_bcast_fn():
    global _CONJ_MUL_REDUCE_BCAST
    if _CONJ_MUL_REDUCE_BCAST is None:
        try:
            from scatterem.nn.functional._conj_mul_reduce_triton import (
                conj_mul_reduce_bcast,
            )

            _CONJ_MUL_REDUCE_BCAST = conj_mul_reduce_bcast
        except Exception:  # noqa: BLE001 - Triton is optional, same as _bcast_cmul
            _CONJ_MUL_REDUCE_BCAST = False
    return _CONJ_MUL_REDUCE_BCAST


def _torch_chain_adjoint(
    ctx, grad: Tensor, sample_patches: Tensor, waves: Tensor
) -> tuple[Tensor | None, Tensor | None]:
    """The ``MulBackward0`` chain of ``sample_patches.unsqueeze(0) * waves``.

    Plain torch operations, so autograd records them under a ``create_graph``
    backward. Returns ``(grad_patches, grad_waves)``; a gradient that ``ctx`` does
    not need is ``None``.
    """
    grad_patches = grad_waves = None
    if ctx.needs_input_grad[1]:
        grad_waves = torch.mul(grad, sample_patches.unsqueeze(0).conj())
    if ctx.needs_input_grad[0]:
        grad_patches = torch.mul(grad, waves.conj()).sum(0, keepdim=True).squeeze(0)
    return grad_patches, grad_waves


class _BroadcastTransmit(torch.autograd.Function):
    """``sample_patches.unsqueeze(0) * waves`` with a fused ``grad_sample_patches``.

    The FORWARD is the identical torch expression, so the exit wave cannot move by a bit.
    Only the backward differs from what ``MulBackward0`` would do, and only in the
    ``sample_patches`` gradient, where the three eager passes
    (``resolve_conj(waves)`` -> ``mul`` -> ``sum(0)``, ~800 MiB of traffic for 288 MiB of
    compulsory work) collapse into one kernel.

    ``grad_waves`` is fused too, but it was NOT at first, and the reason is worth
    keeping: that tensor propagates back through the slice chain to the probe, so it
    carries ``grad_probe``, and a fused transmit adjoint that moved it by one ulp
    changed it bit for bit.  What made it safe
    to fuse is that ``grad_waves`` is pure *elementwise* -- the "bit-exact only in a
    regime" caveat on the reduced half is a statement about torch's summation order, and
    there is no summation here.  It is ``torch.equal`` to ``mul(grad, patches.conj())``
    unconditionally, and the ``grad_probe`` digest is what proves it end to end.

    When only one of the two gradients is wanted, each half runs on its own path (the
    fused reduction, or the verbatim eager multiply) rather than computing the other
    half to throw it away.

    A ``create_graph`` backward (grad mode on inside ``backward``, as in a
    Hessian-vector product or the BH solver's exact curvature) always takes the
    differentiable torch chain. A Triton result has no graph, so it would set every
    second-order term through this op to zero, with no error.
    """

    @staticmethod
    def forward(ctx, sample_patches: Tensor, waves: Tensor) -> Tensor:
        # Exactly the operands MulBackward0 would have saved, so peak memory is unchanged.
        ctx.save_for_backward(sample_patches, waves)
        return sample_patches.unsqueeze(0) * waves

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        sample_patches, waves = ctx.saved_tensors
        grad = grad_outputs[0]
        grad_patches = grad_waves = None
        if torch.is_grad_enabled():
            # create_graph backward: the fused kernels write graph-less tensors.
            return _torch_chain_adjoint(ctx, grad, sample_patches, waves)
        if torch.compiler.is_compiling():
            # same fused kernel, reached through the opaque custom op because the
            # eager entry point's env read / host counter / Optional return are the parts
            # Dynamo cannot trace.  The gate is the eager gate (pure tensor metadata, so
            # Dynamo settles it at trace time as guards), NOT a weaker one.  Only the dual
            # case is routed here: it is the one this model actually takes and the one that
            # was measured, and a one-sided compiled backward falling through to the torch
            # chain below is exactly what it did before this change.
            #
            # ``_COMPILED_FUSED_READY`` is checked HERE as well as in
            # ``_fused_transmit_adjoint_eligible``, because this Function is reachable
            # without going through ``multislice_transmit_slice`` -- a direct ``.apply``,
            # which is what the tests do.  The flag means "the lazy Triton import has
            # already been resolved OUTSIDE a traced region"; without it the import below
            # is an unguarded import of an OPTIONAL dependency, and CPU torch wheels ship
            # no Triton at all, so on a CPU-only install this backward raised
            # ``ModuleNotFoundError: No module named 'triton'`` instead of falling back to
            # the torch chain it documents as its fallback.  Every other Triton entry
            # point in this module goes through a memoized ``try/except`` resolver for
            # exactly that reason; this one did not.
            if (
                _COMPILED_FUSED_READY
                and ctx.needs_input_grad[0]
                and ctx.needs_input_grad[1]
            ):
                from scatterem.nn.functional._conj_mul_reduce_triton import (
                    conj_mul_reduce_bcast_declines,
                )

                if not conj_mul_reduce_bcast_declines(grad, waves, sample_patches):
                    grad_patches, grad_waves = (
                        torch.ops.scatterem.conj_mul_reduce_bcast(
                            grad, waves, sample_patches
                        )
                    )
                    return grad_patches.squeeze(0), grad_waves
            return _torch_chain_adjoint(ctx, grad, sample_patches, waves)
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            # Both halves need `grad`, the largest operand, so computing them in one
            # kernel reads it once: 448 MiB instead of 576 at the benchmark shape, and
            # it also makes `sample_patches` loop-invariant over the mode axis instead
            # of re-read once per mode.  1.635x on the pair; see conj_mul_reduce_bcast.
            fn = _conj_mul_reduce_bcast_fn()
            dual = fn(grad, waves, sample_patches) if fn else None
            if dual is not None:
                grad_patches, grad_waves = dual
                return grad_patches.squeeze(0), grad_waves
            # Above ``_MAX_MODES`` the dual kernel declines on reduction ORDER; the
            # products alone are still fused and torch keeps the mode sum.
            fn = _conj_mul_bcast_products_fn()
            prods = fn(grad, waves, sample_patches) if fn else None
            if prods is not None:
                prod, grad_waves = prods
                return prod.sum(0, keepdim=True).squeeze(0), grad_waves
        if ctx.needs_input_grad[1]:
            # Verbatim MulBackward0 -- bit-exact by construction, see the class docstring.
            grad_waves = torch.mul(grad, sample_patches.unsqueeze(0).conj())
        if ctx.needs_input_grad[0]:
            fn = _conj_mul_reduce_fn()
            fused = fn(grad, waves) if fn else None
            if fused is None:  # declined (dtype/shape/size gate) -> the torch chain
                fused = torch.mul(grad, waves.conj()).sum(0, keepdim=True)
            grad_patches = fused.squeeze(0)
        return grad_patches, grad_waves


def _fused_transmit_adjoint_eligible(sample_patches: Tensor, waves: Tensor) -> bool:
    """Is the custom-adjoint path worth taking at all?

    Deliberately narrow.  When this is ``False`` the caller runs the plain
    ``sample_patches.unsqueeze(0) * waves``, i.e. byte-for-byte the pre-existing code
    path -- so inference, the ``torch.compile``d forward and every non-complex64 /
    non-CUDA / fp16 route are untouched rather than merely equivalent.  The only case
    routed through the custom Function is the one it speeds up: a live graph that needs
    the transmission gradient at the shapes where the fused kernel wins.

    Under ``is_compiling()`` the fused adjoint is still taken, but through the opaque
    ``scatterem::conj_mul_reduce_bcast`` custom op -- what Dynamo cannot trace is the
    host Python around the kernel (an ``os.environ`` read, a mutated counter list, an
    ``Optional`` return), not the kernel.  That route needs ``_COMPILED_FUSED_READY``,
    which the caller sets *before* tracing starts, because
    resolving the lazy Triton import inside a traced region is itself untraceable.  When it
    is unset (a bare ``torch.compile`` on the model, or no Triton) the compiled route falls
    back to exactly the ``MulBackward0`` chain it had before.
    """
    if torch.compiler.is_compiling():
        return (
            _COMPILED_FUSED_READY
            and torch.is_grad_enabled()
            and sample_patches.requires_grad
            and waves.requires_grad
            and waves.is_cuda
            and waves.dtype is torch.complex64
            and sample_patches.shape == waves.shape[1:]
        )
    return (
        torch.is_grad_enabled()
        and sample_patches.requires_grad
        and waves.is_cuda
        and waves.dtype is torch.complex64
        and sample_patches.shape == waves.shape[1:]
        # The Function has no jvp: forward-mode autograd takes the plain multiply.
        and not carries_tangent(sample_patches, waves)
        and bool(_conj_mul_reduce_fn())
    )


def gather_slice(sample_patches: Tensor, gather_index: Tensor) -> Tensor:
    """Per-position patches ``(K, ph, pw, ...)`` of ONE slice ``(U, ...)``.

    ``sample_patches[gather_index]``, spelled as ``index_select`` so the backward
    is an atomic ``index_add`` rather than a sort-based scatter, with the opt-in
    separable gather (:func:`_separable_expand`) tried first.
    """
    fast = _separable_expand(sample_patches[None], gather_index)
    if fast is not None:
        return fast[0]
    return sample_patches.index_select(0, gather_index.reshape(-1)).reshape(
        *gather_index.shape, *sample_patches.shape[1:]
    )


def multislice_transmit_slice(
    sample_patches: Tensor,
    waves: Tensor,
    propagator: "Propagator | None",
    inverse_index: Tensor | None = None,
    gather_index: Tensor | None = None,
) -> Tensor:
    """One multislice transmission + (optional) propagation step.

    Module-level so every ``MultisliceScanOp`` route shares the EXACT same
    per-slice routine.

    Parameters
    ----------
    sample_patches : Tensor
        Per-slice transmission. When `inverse_index` is given this is the
        *ungathered* per-unique-pixel (exponentiated) potential ``[U]`` and
        the gather + complex multiply is fused in one Warp kernel; otherwise
        it is already gathered ``[K, ph, pw]`` and a plain broadcast multiply
        is used.
    waves : Tensor
        Complex exit waves, shape ``Nmodes x K x M1 x M2``.
    propagator : Propagator or None
        Fresnel propagator, or ``None`` to skip inter-slice propagation (last
        slice in a stack / single-slice degenerate case).
    inverse_index : Tensor, optional
        Per-pixel -> unique-pixel map, shape ``[K, ph, pw]``, for the fused
        Warp transmit path.
    gather_index : Tensor, optional
        The same map for the torch route: ``sample_patches`` is then the
        ungathered ``[U]`` slice and is expanded HERE, one slice at a time, so
        no ``(NZ, K, ph, pw)`` copy of the transmission stack ever exists.

    Returns
    -------
    Tensor
        Complex exit waves, shape ``Nmodes x K x M1 x M2``.
    """
    if inverse_index is None and gather_index is not None:
        # the gather and the multiply as one kernel where it applies.
        fused = fused_gather_transmit(sample_patches, gather_index, waves)
        if fused is not None:
            return fused if propagator is None else propagator(fused)
        sample_patches = gather_slice(sample_patches, gather_index)
    if inverse_index is not None:
        waves = fused_transmit(sample_patches, inverse_index, waves)
    elif _fused_transmit_adjoint_eligible(sample_patches, waves):
        waves = _BroadcastTransmit.apply(sample_patches, waves)
    else:
        waves = sample_patches.unsqueeze(0) * waves
    if propagator is not None:
        waves = propagator(waves)
    return waves
