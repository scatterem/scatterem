"""Fused correct-and-reduce: the ``ik`` axis is summed inside the kernel."""

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_backward_analytic_ksum,
    _direct_ptychography_build_A,
    _direct_ptychography_forward_ksum_planes,
    _direct_ptychography_forward_ksum_planes_defocus_only,
)
from scatterem.physics import _warp_interop as wi

from ._caches import (
    _contiguous_memo,
    _functorch,
    _sin_cos_floats,
    _sin_cos_rot,
)
from ._ksum_tiled import _ksum_kernel, _ksum_plan
from ._sq_aperture import _launch_aperture_mask
from .autograd import _correct_aberrations_bwd_fake

# ---------------------------------------------------------------------------

# A RECORDED ``wp.launch`` for the k-SUM correction kernel -- ``_KOUTER_CMD``'s
# mechanism, one op over, with the operand key of the k-outer op.
#
# WHY THE KEY IS NOT ``is``-IDENTITY.  ``correct_aberrations_ksum`` hands this op
# ``torch.view_as_real(Gprime).contiguous()`` and a freshly allocated ``out_real``
# every call, so an identity test can NEVER hit on those slots (the k-outer op had
# the same defect).  The key is therefore
# ``(data_ptr, shape, stride, dtype)`` with ``is`` kept only as a fast path, and
# the entry holds a STRONG REFERENCE to every bound tensor -- which is exactly
# what makes the pointer key sound: while the previously bound tensor is alive
# the caching allocator cannot hand its block to anything else, so an equal
# ``data_ptr`` implies the same live storage and the same words.
#
# WHY IT PAYS HERE AND DID NOT ON THE K-OUTER ROW.  This mechanism measured
# at 1.0003x on a benchmark row, which is 6.45 %
# GPU-IDLE -- a host removal there cashes at ~1 % of its nominal size.  The k-sum
# row is **42.6 % GPU-idle**
# (ceiling 1.74x) and spends 35.8 % of its wall inside this op body, so the same
# host removal has somewhere to go.
#
# Keyed on the launch grid and device, because those are what the recorded command
# fixes; the five scalars are compared and force a re-record, every array operand
# is compared and rebound.
_KSUM_CMD: dict = {}
_KSUM_CMD_MAX = 4

#: Param index of every array argument of ``_direct_ptychography_forward_ksum``,
#: in the order ``wp.launch`` packs them: inputs 0..11 then the single output 12.
_KSUM_ARRAY_SLOTS = (0, 1, 2, 3, 4, 5, 6, 12)
#: The subset that is ``wp.vec2``-typed (``G``, ``A_all``, ``S_out``).
_KSUM_VEC2_SLOTS = frozenset((0, 5, 12))


def _launch_recorded_command(cmd: wp.Launch | None) -> None:
    """Launch a recorded Warp command, rejecting a missing command."""
    if cmd is None:
        raise ValueError("record_cmd=True must return a Warp launch command")
    cmd.launch()


def _ksum_bind_key(t: Tensor):
    return (t, t.data_ptr(), t.shape, t.stride(), t.dtype)


#: ``[taken, refused]`` -- so the dispatcher bypass in
#: :func:`correct_aberrations_ksum` can be AUDITED as reached rather than
#: asserted.  Two integer stores per call.
_KSUM_BYPASS_STATS = [0, 0]


def _ksum_fwd_bypass_ok(args) -> bool:
    """May this call reach :func:`_ksum_fwd_body` without the dispatcher?

    Only if nothing else needs to see the op: no Dynamo/AOTAutograd trace in
    progress, no interpreter interposed between the caller and the dispatcher,
    and every tensor operand an ordinary dense ``torch.Tensor`` -- not a
    ``FakeTensor``, a functional tensor or any ``__torch_dispatch__`` subclass,
    all of which the body would hand straight to ``wi.from_torch``; and no
    operand requiring grad while grad is enabled, since the body returns a leaf
    and the op returns a node.  ``args[:8]`` is exactly the tensor prefix of the
    op's schema -- the remaining four are python scalars.

    ``type(t) is not Tensor`` catches every operand whose wrapper is a PYTHON
    subclass, and that is where it stops: functorch's wrappers are C++ tensor
    impls behind an unchanged ``type()``, so a ``BatchedTensor`` under
    :func:`torch.vmap` passes all three per-operand clauses -- ``type()`` is
    ``Tensor``, ``requires_grad`` is ``False``, ``is_compiling()`` is ``False``
    -- and would reach ``wi.from_torch`` with a batch dimension the kernel
    cannot see.  The two clauses below are therefore not belt-and-braces; they
    are the ones that actually refuse it:

    * ``peek_interpreter_stack()`` is non-``None`` inside ANY functorch
      transform (``vmap``, ``grad``, ``jvp``, ``functionalize``).  ``grad``
      happens to be refused anyway by the ``requires_grad`` clause; ``vmap`` is
      not refused by anything else.
    * a ``TorchDispatchMode`` or ``TorchFunctionMode`` on the stack is an
      interposition that exists precisely to observe or rewrite the op, and the
      bypass would make it observe nothing.

    Both are host-side stack peeks on tensors this function already touches, so
    they cost no sync and no allocation.
    """
    if torch.compiler.is_compiling():
        return False
    if _functorch is not None and _functorch.peek_interpreter_stack() is not None:
        return False
    if torch._C._len_torch_dispatch_stack() > 0:
        return False
    if torch._C._is_torch_function_mode_enabled():
        return False
    grad_on = torch.is_grad_enabled()
    for t in args[:8]:
        if type(t) is not Tensor or (grad_on and t.requires_grad):
            return False
    return True


def _ksum_fwd_body(
    Gprime_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- view_as_real(Gprime)
    aberrations: Tensor,
    sin_cos_rot: Tensor,  # length-2 float32 [sin, cos]
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    A: Tensor,  # (Nk, 2) hoisted ik-only factor; empty = build it here
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
    n_grad_coeffs: int,
    chi_is_zero: bool = False,
) -> Tensor:
    """Forward of :func:`correct_aberrations_ksum`.

    Returns a freshly-allocated ``(Nqy, Nqx, 2)`` float32 tensor -- the
    ``ik``-reduced corrected ``G``; the public wrapper applies
    ``view_as_complex`` on the way out.  The unfused pair returns
    ``(Nqy, Nqx, Nk, 2)``, which at the published ``Fig1_Gd2O3`` fit chunk is
    86.0 MiB against this 1.0 MiB, and every caller reduced it immediately.

    ``empty``, not ``zeros``: the launch is one thread per ``(iqy, iqx)`` and
    the kernel's last statement is an unconditional ``S_out[iqy, iqx] = acc``
    OUTSIDE the aperture guard, so the destination is covered exactly once.
    (The guard skips the accumulation, never the store -- the same contract
    ``_correct_aberrations_fwd`` relies on for its ``empty_like``.)

    ``chi_is_zero`` selects ``_direct_ptychography_forward_ksum_chi_zero``, which
    is BIT-IDENTICAL whenever every aberration coefficient is exactly ``+0.0``
    (measured: 0 of 960 400 words as INT32) and **1.4264x** on the kernel.  Like
    ``defocus_only`` on the ``kouter`` and ``ksum_planes`` siblings it is a plain
    ``bool`` and is NOT derived here on purpose: deriving it costs a host sync on
    ``aberrations``, worth 1.0007 ms behind this row's queued chunk kernels
    against the ~0.45 ms the fold is worth, and
    the only caller that can amortise one is ``_iter_chunk_images``, which
    derives it once above its chunk loop.  Defaults to ``False`` = exactly the
    shipped kernel, not one instruction different.

    When ``chi_is_zero`` holds, :func:`._ksum_tiled._ksum_plan` may further split
    the ``ik`` axis over a leading tile axis (reassociates the float32 ``ik``
    sum, 5.3e-07 relative to peak; the aperture is then folded, bit-exact).  The
    partial-sum scratch is PERSISTENT, held by the cache entry beside the command
    that binds it: the un-memoised form measured (fresh scratch + plain
    launch) at +0.113 ms of host per call against 0.174 ms of device saved.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(Gprime_real.device)
    out_real = torch.empty(
        (Gprime_real.shape[0], Gprime_real.shape[1], 2),
        dtype=Gprime_real.dtype,
        device=Gprime_real.device,
    )
    # ``A`` is the ``ik``-only aperture-times-aberration factor, the same one
    # ``_correct_aberrations_inplace_fwd`` takes; an empty tensor is the sentinel
    # for "not hoisted", in which case it is built here, one launch per chunk.
    # Callers that drive this in a loop over bright-field chunks should build it
    # ONCE above the loop and pass a slice, which is what makes the hoist pay --
    # per chunk the ~0.042 ms launch gives back 60 % of the 0.070 ms saving. See
    # ``build_aberration_bf_factor``.
    if A.numel() == 0:
        A = torch.empty(
            (Kx.shape[0], 2),
            dtype=torch.float32,
            device=Gprime_real.device,
        )
        # ``_direct_ptychography_build_A`` declares ``kx``/``ky`` as untyped
        # ``wp.array(ndim=1)``, so Warp must infer
        # their element type at launch.
        wp.launch(
            kernel=_direct_ptychography_build_A,
            dim=(Kx.shape[0],),
            inputs=[
                wi.from_torch(Kx),
                wi.from_torch(Ky),
                wi.from_torch(aberrations),
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[wi.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    dim = (int(Gprime_real.shape[0]), int(Gprime_real.shape[1]))
    n_tile, sq = _ksum_plan(
        dim, int(Gprime_real.shape[2]), Gprime_real.is_cuda, chi_is_zero
    )
    # The five scalars are recorded INTO the command, so a change in any of them
    # is a re-record and not a rebind -- cheaper to compare than to set, and it
    # keeps the scalar path impossible to get wrong.
    scalars = (
        float(sin_rot),
        float(cos_rot),
        float(semiconvergence_angle),
        float(eps),
        float(wavelength),
    )
    # ``chi_is_zero``, ``n_tile`` and ``sq`` are part of the KEY, not of
    # ``scalars``: each selects a different kernel or launch grid, and a recorded
    # command carries both, so replaying one across either would launch the wrong
    # thing.  The same reason ``defocus_only`` is in ``_KOUTER_CMD``'s key.  The
    # TILE WIDTH is not here: the kernel derives it from ``G.shape[2]`` and
    # ``S_part.shape[0]``, so a chunk of a different length is a pure rebind.
    key = (dim, str(device), bool(chi_is_zero), n_tile, sq)
    entry = _KSUM_CMD.get(key)
    if entry is not None and entry[1] == scalars:
        cmd, _, bound, part = entry
        dst = out_real if part is None else part
        arrays = (Gprime_real, Qx, Qy, Kx, Ky, A, aberrations, dst)
        for slot, t in zip(_KSUM_ARRAY_SLOTS, arrays, strict=True):
            prev = bound[slot]
            if prev[0] is not t and (
                prev[1] != t.data_ptr()
                or prev[2] != t.shape
                or prev[3] != t.stride()
                or prev[4] is not t.dtype
            ):
                cmd.set_param_at_index_from_ctype(
                    slot,
                    (
                        wi.from_torch(t, dtype=wp.vec2, return_ctype=True)
                        if slot in _KSUM_VEC2_SLOTS
                        else wi.from_torch(t, dtype=wp.float32, return_ctype=True)
                    ),
                )
                bound[slot] = _ksum_bind_key(t)
        cmd.launch()
        if part is not None:
            torch.sum(part, dim=0, out=out_real)
        return out_real
    part = (
        torch.empty((n_tile, *dim, 2), dtype=Gprime_real.dtype, device=Kx.device)
        if n_tile > 1
        else None
    )
    dst = out_real if part is None else part
    arrays = (Gprime_real, Qx, Qy, Kx, Ky, A, aberrations, dst)
    kern, angle_arg = _ksum_kernel(
        n_tile, chi_is_zero, sq, wavelength, semiconvergence_angle, Kx.device
    )
    cmd = wp.launch(
        kernel=kern,
        dim=(n_tile, *dim) if n_tile > 1 else dim,
        inputs=[
            wi.from_torch(Gprime_real, dtype=wp.vec2),
            wi.from_torch(Qx),
            wi.from_torch(Qy),
            wi.from_torch(Kx),
            wi.from_torch(Ky),
            wi.from_torch(A, dtype=wp.vec2),
            wi.from_torch(aberrations),
            sin_rot,
            cos_rot,
            angle_arg,
            eps,
            wavelength,
        ],
        outputs=[wi.from_torch(dst, dtype=wp.vec2)],
        device=device,
        record_cmd=True,
    )
    # ``record_cmd=True`` RECORDS and does NOT launch.
    _launch_recorded_command(cmd)
    if part is not None:
        torch.sum(part, dim=0, out=out_real)
    if len(_KSUM_CMD) >= _KSUM_CMD_MAX:
        _KSUM_CMD.clear()
    _KSUM_CMD[key] = (
        cmd,
        scalars,
        {
            slot: _ksum_bind_key(t)
            for slot, t in zip(_KSUM_ARRAY_SLOTS, arrays, strict=True)
        },
        part,
    )
    return out_real


@torch.library.custom_op(
    "scatterem::correct_aberrations_ksum_fwd",
    mutates_args=(),
)
def _correct_aberrations_ksum_fwd(
    Gprime_real: Tensor,
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    A: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
    n_grad_coeffs: int,
    chi_is_zero: bool = False,
) -> Tensor:
    """``custom_op`` face of :func:`_ksum_fwd_body`.

    The body is a plain module-level function so that a caller which provably
    needs no autograd can reach it WITHOUT the dispatcher.  Measured on
    ``r15_direct_ptychography/medium/direct_ptycho_sumfirst``, per call
    on one captured real argument set: the public wrapper costs 62.2 us of
    HOST-ISSUE time, ``torch.ops...correct_aberrations_ksum_fwd`` 47.6 us and
    this body 19.6 us -- so 28.0 us, **45 % of what the caller pays**, is
    torch's ``custom_op`` machinery (autograd key, storage-aliasing check,
    schema/redispatch) and not work.  On a row that is 29.6 % GPU-idle that is
    on the critical path.  See the gate in :func:`correct_aberrations_ksum`.
    """
    return _ksum_fwd_body(
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        A,
        semiconvergence_angle,
        eps,
        wavelength,
        n_grad_coeffs,
        chi_is_zero,
    )


@_correct_aberrations_ksum_fwd.register_fake
def _(
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    A,
    semiconvergence_angle,
    eps,
    wavelength,
    n_grad_coeffs,
    chi_is_zero=False,
):
    return Gprime_real.new_empty((Gprime_real.shape[0], Gprime_real.shape[1], 2))


@torch.library.custom_op(
    "scatterem::correct_aberrations_ksum_bwd",
    mutates_args=(),
)
def _correct_aberrations_ksum_bwd(
    grad_out_real: Tensor,  # (Nqy, Nqx, 2) float32 -- view_as_real(adj_S)
    Gprime_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- saved input
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    n_coeffs: int,
) -> Tensor:
    """Analytic gradient w.r.t. ``aberrations`` for the fused forward.

    Identical to :func:`_correct_aberrations_bwd` -- including
    ``block_dim=256``, which the cooperative ``wp.tile`` / ``wp.tile_atomic_add``
    reductions require -- except that the upstream adjoint arrives REDUCED, at
    ``(Nqy, Nqx, 2)``.  The unfused op received the ``ik``-broadcast of exactly
    these values and had to materialise it (``expand(...).contiguous()``, 86.0
    MiB per chunk at the Fig1 fit config) before Warp could index it.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(grad_out_real.device)
    out_grad = torch.zeros(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device
    )
    G_wp = wi.from_torch(Gprime_real, dtype=wp.vec2)
    adj_wp = wi.from_torch(grad_out_real, dtype=wp.vec2)
    Qx_wp = wi.from_torch(Qx)
    Qy_wp = wi.from_torch(Qy)
    Kx_wp = wi.from_torch(Kx)
    Ky_wp = wi.from_torch(Ky)
    ab_wp = wi.from_torch(aberrations)
    out_grad_wp = wi.from_torch(out_grad)
    wp.launch(
        kernel=_direct_ptychography_backward_analytic_ksum,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            adj_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
            n_coeffs,
            out_grad_wp,
        ],
        device=device,
        block_dim=256,
    )
    return out_grad


# The fused op returns the same gradient shape as the unfused one.
_correct_aberrations_ksum_bwd.register_fake(_correct_aberrations_bwd_fake)


def _correct_aberrations_ksum_setup_context(ctx, inputs, output):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        A,
        semiconvergence_angle,
        eps,
        wavelength,
        n_grad_coeffs,
        # ``chi_is_zero`` selects a BIT-IDENTICAL forward kernel and is not
        # differentiable, so it is unpacked only to keep this in step with the
        # op's schema; nothing below reads it.  The adjoint must NOT fold on it:
        # d(chi)/d(c_j) is nonzero at c == 0, which is the whole point of fitting.
        _chi_is_zero,
    ) = inputs
    # ``A`` is deliberately NOT saved: the exact adjoint recomputes ``A0``
    # in-kernel with the forward's own arithmetic (aperture and guard), and only on the pairs that pass the guard, so the hoist changes the
    # backward neither in value nor in cost.
    ctx.save_for_backward(
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
    )
    ctx.semiconvergence_angle = float(semiconvergence_angle)
    ctx.wavelength = float(wavelength)
    # Same leading-count contract as ``_correct_aberrations_setup_context``;
    # read the comment there for why a count rather than a mask.
    n_all = int(min(12, aberrations.shape[0]))
    ctx.n_coeffs = n_all if int(n_grad_coeffs) < 0 else min(n_all, int(n_grad_coeffs))


def _correct_aberrations_ksum_backward(ctx, grad_out_real):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
    ) = ctx.saved_tensors
    ab_grad = torch.ops.scatterem.correct_aberrations_ksum_bwd(
        grad_out_real.contiguous(),
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        ctx.semiconvergence_angle,
        ctx.wavelength,
        ctx.n_coeffs,
    )
    # Same sign convention as the unfused op; 13 slots, one per forward input
    # (the extra ones are the hoisted ``A``, which is never differentiated, and
    # the ``chi_is_zero`` kernel selector, which is a bool).
    return (
        None,
        -ab_grad,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "scatterem::correct_aberrations_ksum_fwd",
    _correct_aberrations_ksum_backward,
    setup_context=_correct_aberrations_ksum_setup_context,
)


def correct_aberrations_ksum(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    n_grad_coeffs: int | None = None,
    A: torch.Tensor | None = None,
    chi_is_zero: bool = False,
) -> torch.Tensor:
    """``correct_aberrations(...).sum(dim=-1)``, fused into one kernel.

    Returns the ``(Nqy, Nqx)`` complex bright-field-summed corrected ``G``.
    Use this wherever the corrected chunk is only ever reduced over ``ik``: it
    never allocates the ``(Nqy, Nqx, Nk)`` intermediate, and its backward takes
    the reduced ``(Nqy, Nqx)`` adjoint instead of an ``ik``-broadcast copy of it.

    ``A`` is the optional pre-built ``ik``-only factor from
    :func:`build_aberration_bf_factor`, sliced in step with ``Kx``/``Ky``.
    Passing it hoists one aperture, one aberration polynomial and one ``cexp``
    out of the kernel (1.29x on the captured Fig1 chunk, bit-exactly); omitting
    it builds the same values with one extra launch per call, which for a caller
    looping over bright-field chunks costs ~60 % of the saving. Build it ONCE
    above that loop.

    NOT bit-identical to the unfused pair -- the ``ik`` reduction is
    reassociated, and a non-finite ``G`` outside the aperture no longer
    propagates.  Both are quantified in the kernel's own block comment.  (The
    ``A`` hoist itself IS bit-exact; it is a memoisation of arithmetic the
    kernel would otherwise repeat.)

    ``chi_is_zero`` asserts that every coefficient of ``aberrations`` is exactly
    ``+0.0``, which makes both of the kernel's exponentials the constant
    ``(1.0, -0.0)`` and selects ``_direct_ptychography_forward_ksum_chi_zero`` --
    BIT-IDENTICAL (0 of 960 400 output words as INT32) and **1.4264x** on the
    kernel.  The CALLER establishes it, exactly as ``defocus_only`` works on the
    ``kouter`` and ``ksum_planes`` siblings: deriving it here would put a host
    sync on every call.  ``False``, the default, is the shipped kernel unchanged.
    The predicate is ``+0.0``, not ``|c| == 0`` -- a single ``-0.0`` coefficient
    breaks the chain; see :func:`_direct_ptychography_chi_is_zero`, and use
    :func:`chi_is_zero_host` rather than writing the test again.
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
    # ``Gprime`` may be STRIDED.  Warp only requires the INNER stride to be
    # dense, and ``view_as_real`` of any complex tensor has last stride 1 by
    # construction, so ``wi.from_torch(..., dtype=wp.vec2)`` takes an arbitrary
    # outer layout; a ``.contiguous()`` here would be a full read+write pass to
    # buy nothing.  It matters because the one hot caller -- the chunk-sum-first
    # branch of ``_iter_chunk_images`` -- reads ``vBF.G[..., s:e]``, and ``G`` is
    # an ``fft2`` output, i.e. bright-field-OUTERMOST in memory: compacting that
    # slice is a TRANSPOSE, measured at 1.116 ms against a 3.40 ms row --
    # 32.6 % of it.  The strided read costs the kernel nothing measurable in situ
    # (1.5376 -> 1.5213 ms), so the copy is pure loss: 1.59x on that loop and
    # 1.1235x on the whole row, plus 82 MiB off its peak.  Bit-exact:
    # the kernel reads the same values in the same ``ik``-ascending order and
    # accumulates in one register, so only the ADDRESSES move.
    Gprime_real = torch.view_as_real(Gprime)
    aberrations_c = aberrations.contiguous()
    A_c = Gprime_real.new_empty((0, 2)) if A is None else A.contiguous()
    args = (
        Gprime_real,
        aberrations_c,
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        A_c,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        -1 if n_grad_coeffs is None else int(n_grad_coeffs),
    )
    # THE DISPATCHER IS HALF THE PRICE OF THIS CALL.  Measured per call on one
    # captured real argument set from ``r15_direct_ptychography/medium/
    # direct_ptycho_sumfirst`` (host-issue time, no sync inside the timed
    # region): this wrapper 62.2 us, ``torch.ops...ksum_fwd`` 47.6 us,
    # ``_ksum_fwd_body`` 19.6 us.  28.0 us -- 45 % -- is the ``custom_op``
    # machinery: the autograd key, the output-aliases-input storage check, the
    # schema coercion and the redispatch.  That machinery buys exactly one
    # thing, the analytic ``aberrations`` adjoint registered above, and the
    # hottest caller never asks for it: ``_direct_ptychography``'s
    # ``_CHUNK_SUM_FIRST`` chunk loop is a reconstruction and differentiates
    # nothing, so on
    # ``r15_direct_ptychography/medium/direct_ptycho_sumfirst`` the gate is
    # taken 8 of 8 times per row and the row goes 2.561 -> 2.1945 ms (1.1670x,
    # disjoint, bit-exact, peak identical).  ``determine_aberrations``' TV loop
    # is the OPPOSITE case and is why the gate exists rather than the branch
    # being deleted: 27 taken / 45 refused per fit, the 45 being the LBFGS
    # objective, where this op's whole point is the adjoint.
    #
    # The gate is deliberately narrow, and refuses rather than guesses:
    #   * ``is_compiling`` -- Dynamo/AOTAutograd must see the op, not the body,
    #     or the graph loses the Warp launch and the registered fake kernel.
    #   * ``type(...) is Tensor`` -- a FakeTensor, a functional tensor or any
    #     __torch_dispatch__ subclass must go through the dispatcher; the body
    #     hands its operands straight to ``wi.from_torch``.
    #   * any operand requiring grad, with grad enabled -- taking the fast path
    #     would silently drop ``out.requires_grad`` and turn a later
    #     ``.backward()`` from "zero gradient" into "does not require grad".
    bypass = _ksum_fwd_bypass_ok(args)
    _KSUM_BYPASS_STATS[0 if bypass else 1] += 1
    if bypass:
        out_real = _ksum_fwd_body(*args, bool(chi_is_zero))
    else:
        out_real = torch.ops.scatterem.correct_aberrations_ksum_fwd(
            *args, bool(chi_is_zero)
        )
    return torch.view_as_complex(out_real)


def _correct_aberrations_ksum_planes_impl(
    Gprime_real: Tensor,
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    A: Tensor,
    out_real: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
    defocus_only: bool,
) -> None:
    """``scatterem::correct_aberrations_ksum_fwd`` for ``n_planes`` aberration
    vectors at once: ONE launch instead of ``n_planes``.

    ``aberrations`` is ``[n_planes, n_ab]``, ``A`` is ``[Nk, n_planes, 2]`` and
    ``out_real`` is ``[n_planes, Nqy, Nqx, 2]``; ``Gprime_real`` stays the single
    ``[Nqy, Nqx, Nk, 2]`` chunk every plane reduces.  No ``A``-building fallback
    and no recorded-launch memo: the one caller hoists ``A`` above its chunk loop
    and issues this ``n_chunks`` times per sweep, where the per-plane op it
    replaces was issued ``n_chunks * n_planes`` times -- the memo on
    that op exists precisely because it is on the hot path, and this one is not.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wi.device_from_torch(Gprime_real.device)
    Qx_wp = wi.from_torch(Qx)
    Qy_wp = wi.from_torch(Qy)
    Kx_wp = wi.from_torch(Kx)
    Ky_wp = wi.from_torch(Ky)
    # THE APERTURE IS ``ip``-INDEPENDENT, so build it once per chunk instead of
    # once per plane.  The two shifted-aperture tests are functions of
    # ``(iqy, iqx, ik)`` alone; the planes kernel's grid is
    # ``[n_planes, Nqy, Nqx]``, so it evaluated 2 x ``n_planes`` x Nqy x Nqx x Nk
    # of them for Nqy x Nqx x Nk distinct answers.  Measured on the row's own
    # captured arguments (a benchmark script, medians of 20
    # after 3 warm-ups, 5 alternating in-process cycles): the correction kernel
    # 0.7611 -> 0.6309 ms with the mask already built, and 0.7078 ms including this
    # build launch = **1.0753x on the kernel**, +0.427 ms on the 8-chunk row.
    #
    # The mask, not a float array: ``probe_cone_pass`` returns exactly ``+0.0``/``+1.0``,
    # so one bit each is lossless and the extra traffic is
    # ``Nqy * Nqx * Nk`` BYTES (5.5 MiB at the registered ``medium`` chunk) read
    # ``n_planes`` times, against the ~44 MiB a ``vec2`` float32 table would cost.
    #
    # WHY NOT the other two arms, both measured and both worse:
    #   * moving the plane axis INSIDE the thread (grid ``[Nqy, Nqx]``, an inner
    #     ``ip`` loop) shares the aperture AND the ``G`` load, and is **0.683x** --
    #     60 025 threads do not fill 84 SMs, and ``n_planes`` accumulators cannot
    #     live in registers when ``n_planes`` is a runtime value, so the fold turns
    #     the register accumulator into a global read-modify-write.  BIT-EXACT
    #     (0 of 3 361 400 words differ), just slow.
    #   * a closed-form squared-radius threshold needs no mask, but a point
    #     within an ULP of the aperture edge can flip side.  The predicate is
    #     MONOTONE in ``qx^2 + qy^2``, hence exactly a threshold test, so the
    #     float32 threshold is BISECTED on the device instead; the mask is
    #     built with it, bit-identically (see ``_launch_aperture_mask``).
    mask = torch.empty(
        (Gprime_real.shape[0], Gprime_real.shape[1], Kx.shape[0]),
        dtype=torch.uint8,
        device=Gprime_real.device,
    )
    _launch_aperture_mask(
        mask,
        (Qx_wp, Qy_wp, Kx_wp, Ky_wp),
        sin_rot,
        cos_rot,
        semiconvergence_angle,
        wavelength,
        device,
    )
    mask_wp = wi.from_torch(mask, dtype=wp.uint8)
    # THE POLYNOMIAL IS NOT FREE, and this is where the row's time was.
    # ``lens_wavefront_f32`` evaluates a 12-term expansion twice per live
    # ``(ip, iqy, iqx, ik)``. For its predecessor (same expansion), a
    # runtime-``mode`` ablation of the kernel below priced that
    # polynomial at **0.2118 ms against the two transcendental pairs' 0.2108**,
    # together 43.0 % of a kernel that is 77.91 % of
    # a benchmark row's device time
    # (a benchmark script).  The shipped block comment on the
    # SIBLING ``ksum`` kernel records the same ablation there as "FREE, saves
    # NOTHING" -- a per-piece attribution taken on one kernel does not transfer.
    #
    # A former per-plane correction sweep, its first caller, built its
    # stack as ``ab_i = aberrations_array.clone(); ab_i[0] += depth``, so on a
    # plain defocus sweep C10 is the ONLY coefficient that is ever nonzero and the
    # specialised kernel drops eleven terms whose coefficients are ``+-0.0``.
    # BIT-IDENTICAL, not close: ``ndiff = 0 of 3 361 400`` output words over all 8
    # captured calls of the row (a benchmark script).  Kernel
    # **0.7738 -> 0.5702 ms = 1.3570x**, predicted 1.2232x on the row.
    #
    # ``defocus_only`` is decided by the CALLER, once, above its chunk loop, and
    # handed down -- deriving it here would cost a device sync per chunk on a row
    # that is already 10.94 % GPU-idle.  It must never become a per-thread branch:
    # the identical fold behind a dynamic ``if/else`` is wrong by 100x the signal
    # and no golden array catches it (see the specialised kernel's docstring).
    wp.launch(
        kernel=(
            _direct_ptychography_forward_ksum_planes_defocus_only
            if defocus_only
            else _direct_ptychography_forward_ksum_planes
        ),
        dim=tuple(int(v) for v in out_real.shape[:-1]),
        inputs=[
            wi.from_torch(Gprime_real, dtype=wp.vec2),
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            wi.from_torch(A, dtype=wp.vec2),
            wi.from_torch(aberrations),
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
            mask_wp,
        ],
        outputs=[wi.from_torch(out_real, dtype=wp.vec2)],
        device=device,
    )


# The op stays REGISTERED and unchanged, exactly as for
# ``scatterem::correct_aberrations_kouter``: same name, same
# ``mutates_args``, same body, and still what Dynamo sees.  The body is only named
# separately so the eager wrapper below can reach it without the dispatcher.
@torch.library.custom_op(
    "scatterem::correct_aberrations_ksum_planes",
    mutates_args=("out_real",),
)
def _correct_aberrations_ksum_planes_op(
    Gprime_real: Tensor,
    aberrations: Tensor,
    sin_cos_rot: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    A: Tensor,
    out_real: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
    defocus_only: bool,
) -> None:
    _correct_aberrations_ksum_planes_impl(
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        A,
        out_real,
        semiconvergence_angle,
        eps,
        wavelength,
        defocus_only,
    )


@torch.no_grad()
def correct_aberrations_ksum_planes(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor,
    out: torch.Tensor,
    defocus_only: bool | None = None,
) -> torch.Tensor:
    """:func:`correct_aberrations_ksum` for a STACK of aberration vectors.

    Identical arithmetic to ``n_planes`` separate :func:`correct_aberrations_ksum`
    calls: the kernel behind this is the ``ksum`` kernel with a leading plane axis
    added to its ``wp.tid()`` and two of its subscripts, and each thread still
    accumulates over ``ik`` ascending in one float32 register, so plane ``ip`` of
    ``out`` is bit-identical to what the per-plane call returns.

    Why it exists: the ``_CHUNK_SUM_FIRST`` branch of a former per-plane
    correction sweep paid the correction's fixed per-call host cost
    ``n_chunks * n_depths`` times; the TV autofocus uses it now.

    Args:
        Gprime: ``[Nqy, Nqx, Nk]`` complex chunk, left UNTOUCHED, shared by every
            plane.
        aberrations: ``[n_planes, n_ab]`` aberration coefficients.
        rotation: scan/detector rotation in degrees (float or device tensor).
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Qx, Qy, Kx, Ky: coordinate arrays, as for :func:`correct_aberrations_ksum`.
        A: ``[Nk, n_planes, 2]`` float32 factor from
            :func:`build_aberration_bf_factor`, plane axis INNERMOST so that a
            caller holding ``[n_bf_total, n_planes, 2]`` slices a chunk for free.
        out: pre-allocated contiguous ``[n_planes, Nqy, Nqx]`` complex tensor.
        defocus_only: ``True`` iff ``aberrations[:, 1:]`` is entirely zero, which
            selects a BIT-IDENTICAL kernel that folds the eleven dead terms of the
            aberration polynomial away -- 1.3570x on the kernel, see the block
            comment at the launch.  Left ``None`` it is derived here, which costs
            ONE device sync per call; a hot caller with a stack built above its
            chunk loop should pass it, and must pass ``False`` if it cannot
            establish the condition.

    Returns:
        torch.Tensor - ``out``.
    """
    n_planes = int(aberrations.shape[0])
    want = (n_planes, int(Gprime.shape[0]), int(Gprime.shape[1]))
    if tuple(out.shape) != want or out.dtype != Gprime.dtype:
        raise ValueError(
            f"out must be [n_planes, Nqy, Nqx] with Gprime's dtype; got "
            f"{tuple(out.shape)}/{out.dtype}, want {want}/{Gprime.dtype}"
        )
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous():
        raise ValueError("out must be contiguous")
    # ``Gprime`` may be STRIDED -- see the identical note on
    # :func:`correct_aberrations_ksum`.  Warp requires only the INNER stride to be
    # dense and ``view_as_real`` of any complex tensor has last stride 1 by
    # construction, so ``wi.from_torch(..., dtype=wp.vec2)`` takes an arbitrary
    # outer layout; a ``.contiguous()`` here is a full read+write pass buying
    # nothing.  The hot caller was the ``_CHUNK_SUM_FIRST`` branch of a
    # former per-plane correction sweep, which handed the read-only
    # ``G[..., s:e]`` view straight through: G is an ``fft2`` output, i.e.
    # bright-field-OUTERMOST in memory, so compacting that slice is a TRANSPOSE --
    # 1.156 ms of a 9.68 ms row, 12 % of it.  ``out`` is validated above and is
    # untouched by this; only the INPUT's layout requirement relaxes.  Bit-exact:
    # each thread still reads the same values in the same ``ik``-ascending order
    # into one register, so only the ADDRESSES move.
    if defocus_only is None:
        defocus_only = bool(aberrations[:, 1:].abs().sum() == 0)
    # measured on THIS op at 48.42 us/call (A-vs-A spread 2.09 us) = 6.52 % of
    # a benchmark row (the former sweep).
    # Same reason as ``correct_aberrations_kouter``'s eager branch.
    if torch.compiler.is_compiling():
        call = torch.ops.scatterem.correct_aberrations_ksum_planes
    else:
        call = _correct_aberrations_ksum_planes_impl
    call(
        torch.view_as_real(Gprime),
        aberrations.contiguous(),
        _sin_cos_rot(rotation, Gprime.device),
        _contiguous_memo(Qx),
        _contiguous_memo(Qy),
        _contiguous_memo(Kx),
        _contiguous_memo(Ky),
        A.contiguous(),
        out_real,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        bool(defocus_only),
    )
    return out
