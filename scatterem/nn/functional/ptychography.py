"""
PyTorch functional operations for direct ptychography.
"""
import math

import torch
from torch import Tensor
import warp as wp

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_backward_analytic,
    _direct_ptychography_backward_analytic_ksum,
    _direct_ptychography_build_A,
    _direct_ptychography_build_A_planes,
    _direct_ptychography_chi_is_zero,
    _direct_ptychography_forward,
    _direct_ptychography_forward_ksum,
    _direct_ptychography_aperture_mask_planes,
    _direct_ptychography_forward_ksum_planes,
    _direct_ptychography_forward_precomputed_A,
    _direct_ptychography_forward_precomputed_A_kouter,
    _direct_ptychography_forward_precomputed_A_planes,
    _phase_contrast_transfer_function_forward,
)


# ``[sin, cos]`` arrives as a DEVICE tensor (see the ``_sin_cos_rot`` comment
# below), and every op body that launches an aberration kernel needs the two
# values as Python floats to pass them as kernel scalars.  ``.item()`` is how you
# do that, and ``.item()`` is a **blocking device synchronisation**: it drains
# everything queued on the stream before the host may continue.
#
# Measured in situ on the published ``Fig1_Gd2O3`` fit (``benchmarks/lab/
# _probe_d12_insitu.py``): 3459 ``Tensor.item()`` calls -- 2 per forward op and 2
# per analytic-backward op -- costing **970 ms of host time in a 1975 ms
# ``determine_aberrations``, 280 us each**.  They are not slow in themselves; each
# one waits for the PREVIOUS chunk's kernels, which serialises the whole fit: the
# host cannot enqueue chunk i+1 until chunk i has finished on the device, so the
# GPU idles through every op's host bookkeeping and the device never runs ahead.
#
# The rotation is CONSTANT over a whole reconstruction, so those two floats are
# the same two floats 1725 times. Memoise them on the tensor's identity and
# version, exactly as ``_sin_cos_rot`` memoises the tensor itself, and the sync
# happens ONCE per fit instead of twice per op call. On a hit the caller gets the
# same Python floats the first call read out of that tensor, so the reuse is
# BIT-EXACT BY CONSTRUCTION -- no arithmetic is repeated and none can differ.
#
# Why this is safe where a plain cache would not be:
#   * the strong reference to the tensor lives in the value, so its ``id`` cannot
#     be recycled while the entry is live, and ``hit[0] is sin_cos_rot`` catches
#     the evicted-then-reused-id case (the same argument as ``_sin_cos_rot``);
#   * ``_version`` is part of the key, so an in-place write to the tensor misses;
#   * it is called only from inside ``torch.library.custom_op`` bodies, which
#     Dynamo does not trace -- which is exactly why the ``.item()`` calls were
#     moved in here in the first place, so this changes nothing about
#     ``torch.compile(fullgraph=True)`` safety.
_SIN_COS_FLOAT_CACHE: dict = {}
_SIN_COS_FLOAT_CACHE_MAX = 8


def _sin_cos_floats(sin_cos_rot: Tensor) -> tuple[float, float]:
    """``(sin, cos)`` as Python floats, without a D2H sync on the common path."""
    key = (id(sin_cos_rot), sin_cos_rot._version)
    hit = _SIN_COS_FLOAT_CACHE.get(key)
    if hit is not None and hit[0] is sin_cos_rot:
        return hit[1], hit[2]
    sin_rot = float(sin_cos_rot[0].item())
    cos_rot = float(sin_cos_rot[1].item())
    if len(_SIN_COS_FLOAT_CACHE) >= _SIN_COS_FLOAT_CACHE_MAX:
        _SIN_COS_FLOAT_CACHE.clear()
    _SIN_COS_FLOAT_CACHE[key] = (sin_cos_rot, sin_rot, cos_rot)
    return sin_rot, cos_rot


@torch.library.custom_op(
    "scatterem::correct_aberrations_inplace",
    mutates_args=("Gprime_real", "out_real"),
)
def _correct_aberrations_inplace_op(
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
) -> None:
    # ``sin_cos_rot`` is a length-2 float32 tensor [sin(theta), cos(theta)]
    # carried as a tensor so the wrapper does not need ``.item()`` (which
    # graph-breaks under ``torch.compile(fullgraph=True)``). The conversion
    # happens here, inside the op body, where Dynamo does not trace -- memoised,
    # because it is a device sync; see ``_sin_cos_floats``.
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(Gprime_real.device)
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    # ``A`` is the ``ik``-only aperture-times-aberration factor. An empty tensor
    # is the sentinel for "not hoisted": build it here, one launch for the whole
    # chunk. Callers that drive this in a loop over bright-field chunks pass a
    # slice of one precomputed ``[n_bright_field, 2]`` buffer instead, which is
    # what makes the hoist pay -- see ``build_aberration_bf_factor``. The test is
    # on ``numel``, inside the op body, where Dynamo does not trace.
    if A.numel() == 0:
        A = torch.empty(
            (Kx.shape[0], 2), dtype=torch.float32, device=Gprime_real.device,
        )
        wp.launch(
            kernel=_direct_ptychography_build_A,
            dim=(Kx.shape[0],),
            inputs=[
                Kx_wp,
                Ky_wp,
                ab_wp,
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[wp.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    A_wp = wp.from_torch(A, dtype=wp.vec2)
    # ``out_real`` is the out-of-place destination; an empty tensor is the sentinel
    # for "correct in place", which aims the kernel's output at its own input as
    # before. The kernel already reads ``G`` and writes ``G_out`` as separate
    # arrays, so the only difference between the two modes is which array this
    # launch is pointed at. Out-of-place spares a caller that must preserve its
    # input the separate ``clone()`` pass -- see ``correct_aberrations_inplace``.
    out_wp = G_wp if out_real.numel() == 0 else wp.from_torch(out_real, dtype=wp.vec2)
    wp.launch(
        kernel=_direct_ptychography_forward_precomputed_A,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            A_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[out_wp],
        device=device,
    )


# ``sin_cos_rot`` is a length-2 float32 device tensor ``[sin(theta), cos(theta)]``
# built from a scalar rotation that is CONSTANT over a whole reconstruction:
# ``determine_aberrations`` creates one ``rotation_t`` up front and hands it to
# every (objective evaluation x bright-field chunk) call -- 1800 of them for one
# ``Fig2_carbon`` fit -- and each call rebuilt it with deg2rad + sin + cos +
# stack + to + contiguous, i.e. six torch ops and three tiny device launches to
# turn one unchanged scalar into the same two floats.
#
# Memoised on the rotation's IDENTITY and version counter (a strong reference to
# the rotation lives in the value, so its ``id`` cannot be recycled while the
# entry is live) or, for a plain Python scalar, on its exact value. That makes
# the reuse BIT-EXACT BY CONSTRUCTION rather than by agreement: on a hit the
# caller gets literally the same tensor object the first call built, so no
# arithmetic is repeated and none can differ.
#
# Bypassed when the rotation requires grad (it must stay on the autograd graph;
# note the registered backward returns ``None`` for ``sin_cos_rot``, so nothing
# differentiates through it today) and under ``torch.compile`` (a dict keyed on
# ``id()`` is not traceable). Both fall through to the original construction.
_SIN_COS_ROT_CACHE: dict = {}
_SIN_COS_ROT_CACHE_MAX = 8


def _sin_cos_rot(rotation, device) -> torch.Tensor:
    """``[sin, cos]`` of ``rotation`` (degrees) as a contiguous float32 tensor."""
    key = None
    if isinstance(rotation, torch.Tensor):
        if not (rotation.requires_grad or torch.compiler.is_compiling()):
            key = ("t", id(rotation), rotation._version, str(rotation.device))
            hit = _SIN_COS_ROT_CACHE.get(key)
            # ``hit[0] is rotation`` re-checks identity: the strong reference
            # makes id reuse impossible for a LIVE entry, and this catches the
            # case where the entry was evicted and a new object took the id.
            if hit is not None and hit[0] is rotation:
                return hit[1]
    else:
        if not torch.compiler.is_compiling():
            key = ("f", float(rotation), str(device))
            hit = _SIN_COS_ROT_CACHE.get(key)
            if hit is not None:
                return hit[1]
        rotation = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    rotation_rad = torch.deg2rad(rotation)
    sin_cos_rot = (
        torch.stack([torch.sin(rotation_rad), torch.cos(rotation_rad)])
        .to(dtype=torch.float32)
        .contiguous()
    )
    if key is not None:
        if len(_SIN_COS_ROT_CACHE) >= _SIN_COS_ROT_CACHE_MAX:
            _SIN_COS_ROT_CACHE.clear()
        _SIN_COS_ROT_CACHE[key] = (rotation, sin_cos_rot)
    return sin_cos_rot


# ``correct_aberrations_kouter`` calls ``.contiguous()`` on its four coordinate
# operands on EVERY call, and ``direct_ptychography_depth_section`` calls it
# ``n_chunks * n_depths`` times with the same coordinates.  Measured on
# ``r20_depth_section/small/depth_section`` (D275): 40 calls per row, and
# ``get_q_1d``'s ``Qx``/``Qy`` are NOT contiguous, so each call allocated and filled
# a fresh copy of both -- 80 device allocations and 80 copy kernels per row for two
# vectors that never change.  ``Kx``/``Ky`` arrive as ``vBF.k[s:e, 1]`` / ``[s:e, 0]``,
# genuinely strided, but constant across a chunk's ``n_depths`` planes, so four of
# every five of those copies were redundant too.
#
# Memoised on the source tensor's IDENTITY, version counter, shape and strides, with
# a strong reference to the source in the value -- the same argument as
# ``_SIN_COS_ROT_CACHE``: the strong reference makes ``id`` reuse impossible for a
# live entry, and ``hit[0] is t`` catches the evicted-then-reused-id case.  The
# result is BIT-EXACT BY CONSTRUCTION: ``.contiguous()`` relocates bits, it does not
# compute anything, so a reused copy holds the same words a fresh one would.
#
# This is also what makes ``_KOUTER_CMD`` below pay: a bound launch whose operand
# pointers move on every call has to rebind every operand, and rebinding eight
# arrays costs about what recording the launch again costs.
_CONTIG_CACHE: dict = {}
_CONTIG_CACHE_MAX = 16


def _contiguous_memo(t: Tensor) -> Tensor:
    """``t.contiguous()``, reusing the previous copy of an unchanged source."""
    if t.is_contiguous():
        return t
    if torch.compiler.is_compiling():
        return t.contiguous()
    key = (id(t), t._version, tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
    hit = _CONTIG_CACHE.get(key)
    if hit is not None and hit[0] is t:
        return hit[1]
    out = t.contiguous()
    if len(_CONTIG_CACHE) >= _CONTIG_CACHE_MAX:
        _CONTIG_CACHE.clear()
    _CONTIG_CACHE[key] = (t, out)
    return out


# A RECORDED ``wp.launch`` for the k-outer correction kernel, replayed with only the
# operands that actually moved rebound.  D274-run measured ``wp.launch`` at a FIXED
# 32-124 us of host time per call that does not shrink with problem size, and
# ``r20_depth_section/small/depth_section`` pays it 45 times inside an 18 ms wall
# that is 91 % GPU-idle.  Warp's own fast path for that shape is
# ``wp.launch(record_cmd=True)`` once plus ``set_param_at_index`` + ``.launch()``
# per call -- the same arrangement ``FusedNormalizeScaleInto`` (D271) ships and
# D250 measured bit-exact over 60 calls.
#
# Keyed on the launch grid and device, because those are what the recorded command
# fixes; every other operand is compared and rebound.
#
# WHY THE ENTRY HOLDS THE TENSORS.  ``set_param_at_index`` packs a raw pointer into
# the recorded parameter block.  If the tensor behind that pointer is freed, the
# block dangles and the kernel silently reads whatever the caching allocator handed
# out next -- which is not a crash and not a wrong-looking number, it is a
# reproducibility bug that appears only when the allocator happens to reuse the
# block.  (Measured: a first draft of this that let ``Qx.contiguous()``'s temporary
# die produced a different image digest in one probe arm ordering and the correct
# one in another.)  So the entry keeps a strong reference to every bound tensor,
# which both keeps the pointer valid and makes ``is``-identity a sound cheap test.
_KOUTER_CMD: dict = {}
_KOUTER_CMD_MAX = 4

#: ``(param index, wp dtype or None)`` for every array argument of
#: ``_direct_ptychography_forward_precomputed_A_kouter``.  Inputs occupy 0..12 --
#: 0..6 are the arrays, 7..11 the scalars and 12 the one-element ``chi_is_zero``
#: predicate (D322) -- and the single output is 13, which is the order
#: ``wp.launch`` packs them in.
_KOUTER_ARRAY_SLOTS = (0, 1, 2, 3, 4, 5, 6, 12, 13)

#: ``(id, data_ptr, _version, n, device) -> (flag tensor, aberrations)`` for
#: :func:`_chi_is_zero_flag`.  Bounded like ``_KOUTER_CMD``.
_CHI_ZERO_FLAG: dict = {}
_CHI_ZERO_FLAG_MAX = 4


def _chi_is_zero_flag(aberrations: Tensor, ab_wp, device) -> Tensor:
    """A ONE-ELEMENT DEVICE tensor holding D322's identity predicate.

    ``chi`` is identically ``+0.0`` when every aberration coefficient is ``+0.0``,
    so ``_direct_ptychography_forward_precomputed_A_kouter``'s two ``cexp`` calls
    collapse to the constant ``(1.0, -0.0)`` -- worth **1.32x on that kernel**.
    See ``_direct_ptychography_chi_is_zero`` for the term-by-term proof and for
    why ``-0.0`` coefficients are excluded.

    TWO things about this are measured rather than chosen, and both are the whole
    reason the fold is worth anything on a real row.

    (1) THE ANSWER STAYS ON THE DEVICE.  Reading the predicate on the host is the
    obvious spelling and it is a ``cudaStreamSynchronize``: measured on
    ``r15_direct_ptychography/medium``, ``bool((ab != 0).any())`` costs 0.0245 ms
    on an empty queue and **1.0007 ms** behind eight queued chunk kernels
    (``benchmarks/lab/_probe_d322_gate.py``), against the ~0.34 ms the fold is
    worth.  D313 removed a sync from this exact row; this must not put one back.

    (2) THE FLAG IS MEMOISED, and without that the fold is BELOW the lab's 5 %
    bar.  ``aberrations`` is loop-invariant across a reconstruction's chunks --
    ``_iter_chunk_images`` builds ``A_all`` once for the same reason -- but the op
    is entered once per chunk, so an un-memoised build costs one extra launch per
    chunk on a row that is 12.3 % GPU-idle.  Measured end to end over six
    alternating separately-launched arms: **1.0431x** un-memoised against
    **1.0619x** predicted from the kernel alone; the missing 0.10 ms is 8 launches
    at ~12.5 us of host gap.

    KEYED ON ``(id, data_ptr, _version, n, device)`` AND HOLDING A STRONG
    REFERENCE, for the reason ``_KOUTER_CMD`` records above: an ``id()``- or
    pointer-keyed cache is only sound if the key cannot be recycled, and keeping
    the tensor alive is what guarantees that.  ``_version`` catches in-place
    mutation of the coefficients, which is how a fit updates them.
    """
    key = (
        id(aberrations), aberrations.data_ptr(), aberrations._version,
        int(aberrations.shape[0]), str(device),
    )
    hit = _CHI_ZERO_FLAG.get(key)
    if hit is not None:
        return hit[0]
    flag = torch.empty(1, dtype=torch.int32, device=aberrations.device)
    wp.launch(
        kernel=_direct_ptychography_chi_is_zero,
        dim=(1,),
        inputs=[ab_wp],
        outputs=[wp.from_torch(flag)],
        device=device,
    )
    if len(_CHI_ZERO_FLAG) >= _CHI_ZERO_FLAG_MAX:
        _CHI_ZERO_FLAG.clear()
    _CHI_ZERO_FLAG[key] = (flag, aberrations)
    return flag


@torch.no_grad()
def build_aberration_bf_factor(
    aberrations: torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """Precompute the ``ik``-only factor ``A = aperture(K) * exp(-i chi(K))``.

    ``_direct_ptychography_forward`` evaluates this per ``(Qy, Qx, ik)`` even
    though it varies only with the bright-field pixel. Building it once for all
    bright-field pixels and handing chunk slices to
    :func:`correct_aberrations_inplace` removes one of the kernel's three
    aberration-polynomial evaluations per element.

    Hoist it ABOVE the loop over bright-field chunks. Called per chunk it is a
    LOSS: the launch costs ~30 us of host time for ~2 us of device work, which
    is more than the device saving it buys back (measured: a 2.4 % end-to-end
    regression when built per chunk, a 1.15x win when built once).

    Args:
        aberrations: aberration coefficients, on the same device as ``Kx``.
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Kx: ``[n_bright_field]`` bright-field pixel x coordinates.
        Ky: ``[n_bright_field]`` bright-field pixel y coordinates.

    Returns:
        torch.Tensor - ``[n_bright_field, 2]`` float32 (real, imag), sliceable
        along dim 0 in step with ``Kx``/``Ky``.
    """
    Kx = Kx.contiguous()
    Ky = Ky.contiguous()
    A = torch.empty((Kx.shape[0], 2), dtype=torch.float32, device=Kx.device)
    wp.launch(
        kernel=_direct_ptychography_build_A,
        dim=(Kx.shape[0],),
        inputs=[
            wp.from_torch(Kx),
            wp.from_torch(Ky),
            wp.from_torch(aberrations.contiguous()),
            float(semiconvergence_angle),
            float(wavelength),
        ],
        outputs=[wp.from_torch(A, dtype=wp.vec2)],
        device=wp.device_from_torch(Kx.device),
    )
    return A


@torch.no_grad()
def build_aberration_bf_factor_planes(
    aberrations: torch.Tensor,
    semiconvergence_angle: float,
    wavelength: float,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
) -> torch.Tensor:
    """:func:`build_aberration_bf_factor` for ``n_planes`` aberration vectors AT ONCE.

    ``direct_ptychography_depth_section`` calls the per-plane form once per depth
    plane above its chunk loop.  That kernel is ``len(Kx)`` threads -- 729 at the
    registered ``r20/medium`` geometry -- so its device half is ~2 us and the
    108 us/call it costs is ALL HOST: ``wp.launch`` 49.7 us, the four
    ``wp.from_torch`` views 30.1, ``torch.empty`` 7.8, and 18.7 in the call plus
    the ``no_grad`` guard (``benchmarks/lab/_probe_d346_split.py``, in situ on the
    row's own arguments).  None of it shrinks with problem size, and the sweep
    paid it ``n_depths`` times for what is one launch's worth of device work.

    So the lever here is LAUNCH COUNT, not arithmetic: an ablation that deletes
    six of ``r20/medium``'s seven calls outright bounds the row at **1.0836x**
    (``benchmarks/lab/_probe_d346_roof.py``, in-process alternating arms), while
    hoisting every host-side operand and keeping seven launches bounds it at only
    1.0603x.  A form that exploits the fact that the planes differ ONLY in
    ``aberrations[0]`` -- factoring ``A_plane = A_base * exp(-i pi lambda dz |K|^2)``
    -- would buy the ~2 us of device work, forfeit bit-exactness, and is the
    reason this is spelled as a batch and not as a factorisation.

    Args:
        aberrations: ``[n_planes, n_ab]`` float32 coefficients, one row per plane,
            on the same device as ``Kx``.
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Kx: ``[n_bright_field]`` bright-field pixel x coordinates.
        Ky: ``[n_bright_field]`` bright-field pixel y coordinates.

    Returns:
        torch.Tensor - ``[n_planes, n_bright_field, 2]`` float32 (real, imag),
        plane OUTERMOST so that ``out[ip]`` is a contiguous ``[n_bf, 2]`` tensor
        interchangeable with :func:`build_aberration_bf_factor`'s return value.
        Bit-identical to it row by row -- the kernel is the same six lines with a
        leading plane axis on its ``wp.tid()`` and no reduction.
    """
    Kx = Kx.contiguous()
    Ky = Ky.contiguous()
    aberrations = aberrations.contiguous()
    A = torch.empty(
        (aberrations.shape[0], Kx.shape[0], 2),
        dtype=torch.float32,
        device=Kx.device,
    )
    wp.launch(
        kernel=_direct_ptychography_build_A_planes,
        dim=(aberrations.shape[0], Kx.shape[0]),
        inputs=[
            wp.from_torch(Kx),
            wp.from_torch(Ky),
            wp.from_torch(aberrations),
            float(semiconvergence_angle),
            float(wavelength),
        ],
        outputs=[wp.from_torch(A, dtype=wp.vec2)],
        device=wp.device_from_torch(Kx.device),
    )
    return A


@torch.no_grad()
def correct_aberrations_inplace(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    """
    Correct aberrations in place using direct ptychography.

    Args:
        Gprime: torch.Tensor - input G tensor
        aberrations: torch.Tensor - aberrations array
        rotation: float - rotation in degrees
        semiconvergence_angle: float - semiconvergence angle
        wavelength: float - wavelength
        Qx: torch.Tensor - Qx coordinates
        Qy: torch.Tensor - Qy coordinates
        Kx: torch.Tensor - Kx coordinates
        Ky: torch.Tensor - Ky coordinates
        A: optional ``[len(Kx), 2]`` float32 tensor from
            :func:`build_aberration_bf_factor` -- the ``ik``-only
            ``aperture(K) * exp(-i chi(K))`` factor, hoisted out of a loop over
            bright-field chunks. Omit it and the kernel's own launch builds it;
            the result is identical either way (the hoisted buffer is a
            memoisation of the same float32 arithmetic), only the number of
            launches differs.
        out: optional pre-allocated contiguous complex tensor shaped like
            ``Gprime``. Given one, the correction is computed OUT OF PLACE:
            ``Gprime`` is left untouched and the corrected values land in
            ``out``, which is what is returned. A caller that must preserve
            ``Gprime`` (because it corrects the same chunk again under
            different aberrations) would otherwise have to ``clone()`` first,
            and that clone is a whole extra read+write pass over the chunk on
            top of the kernel's own -- four passes where two will do. The
            arithmetic is identical: the kernel already reads ``G`` and writes
            ``G_out`` as separate arrays, so only the destination changes.

    Returns:
        torch.Tensor - corrected G tensor
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
    # In eager mode, alias ``Gprime``'s storage so the kernel mutates the
    # caller's complex tensor in place (preserved API behavior).
    # Under ``torch.compile``, that aliasing path trips inductor's complex
    # codegen + auto_functionalize interaction (verified: scheduler
    # ``get_buf_bytes`` AssertionError when a mutating custom op writes
    # through a real view of a complex base tensor). We fall back to an
    # out-of-place real buffer in the compile path; the returned complex
    # tensor still carries the corrected values, but the caller's
    # ``Gprime`` is not updated under compile. All real callers consume
    # only the return value, so this is observable only by tests that
    # explicitly check post-call ``Gprime`` -- those tests run eager.
    if torch.compiler.is_compiling():
        Gprime_real = torch.view_as_real(Gprime).contiguous().clone()
    else:
        Gprime_real = torch.view_as_real(Gprime).contiguous()
    if A is None:
        # Empty sentinel: the op body builds the factor itself. Kept as a tensor
        # rather than an Optional so the custom-op schema stays a plain Tensor
        # argument under ``torch.compile(fullgraph=True)``.
        A = torch.empty((0, 2), dtype=torch.float32, device=Gprime.device)
    if out is None:
        # Same empty-sentinel convention as ``A``: "no destination" means correct
        # in place, and the schema stays a plain Tensor argument.
        out_real = torch.empty((0, 2), dtype=torch.float32, device=Gprime.device)
    else:
        if out.shape != Gprime.shape or out.dtype != Gprime.dtype:
            raise ValueError(
                f"out must match Gprime in shape and dtype; got {tuple(out.shape)}/"
                f"{out.dtype} for {tuple(Gprime.shape)}/{Gprime.dtype}"
            )
        out_real = torch.view_as_real(out)
        if not out_real.is_contiguous():
            raise ValueError("out must be contiguous")
    torch.ops.scatterem.correct_aberrations_inplace(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        A.contiguous(),
        out_real,
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
    )
    return torch.view_as_complex(Gprime_real if out is None else out_real)


@torch.library.custom_op(
    "scatterem::correct_aberrations_kouter",
    mutates_args=("out_real",),
)
def _correct_aberrations_kouter_op(
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
) -> None:
    """``scatterem::correct_aberrations_inplace`` on a ``[Nk, Nqy, Nqx]`` chunk.

    Always OUT OF PLACE -- the caller of the k-outer layout is by construction one
    that corrects the same chunk repeatedly under different aberrations, so the
    in-place sentinel of the ``[Nqy, Nqx, Nk]`` op has no user here and is not
    carried over.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(Gprime_real.device)
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    if A.numel() == 0:
        A = torch.empty(
            (Kx.shape[0], 2), dtype=torch.float32, device=Gprime_real.device,
        )
        wp.launch(
            kernel=_direct_ptychography_build_A,
            dim=(Kx.shape[0],),
            inputs=[Kx_wp, Ky_wp, ab_wp, semiconvergence_angle, wavelength],
            outputs=[wp.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    chi_is_zero = _chi_is_zero_flag(aberrations, ab_wp, device)
    dim = tuple(int(v) for v in Gprime_real.shape[:-1])
    # The five scalars are recorded INTO the command, so a change in any of them is
    # a re-record and not a rebind -- cheaper to compare than to set, and it keeps
    # the scalar path impossible to get wrong.
    scalars = (
        float(sin_rot), float(cos_rot), float(semiconvergence_angle),
        float(eps), float(wavelength),
    )
    arrays = (Gprime_real, Qx, Qy, Kx, Ky, A, aberrations, chi_is_zero, out_real)
    key = (dim, str(device))
    entry = _KOUTER_CMD.get(key)
    if entry is not None and entry[1] == scalars:
        cmd, _, bound = entry
        for slot, t in zip(_KOUTER_ARRAY_SLOTS, arrays):
            prev = bound[slot]
            if prev[0] is not t or prev[1] != t.data_ptr() or prev[2] != t.shape:
                cmd.set_param_at_index(
                    slot,
                    wp.from_torch(t, dtype=wp.vec2) if slot in (0, 5, 13)
                    else wp.from_torch(t),
                )
                bound[slot] = (t, t.data_ptr(), t.shape)
        cmd.launch()
        return
    cmd = wp.launch(
        kernel=_direct_ptychography_forward_precomputed_A_kouter,
        dim=dim,
        inputs=[
            G_wp,
            wp.from_torch(Qx),
            wp.from_torch(Qy),
            Kx_wp,
            Ky_wp,
            wp.from_torch(A, dtype=wp.vec2),
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
            wp.from_torch(chi_is_zero),
        ],
        outputs=[wp.from_torch(out_real, dtype=wp.vec2)],
        device=device,
        record_cmd=True,
    )
    # ``record_cmd=True`` RECORDS and does NOT launch.
    cmd.launch()
    if len(_KOUTER_CMD) >= _KOUTER_CMD_MAX:
        _KOUTER_CMD.clear()
    _KOUTER_CMD[key] = (
        cmd,
        scalars,
        {slot: (t, t.data_ptr(), t.shape) for slot, t in zip(_KOUTER_ARRAY_SLOTS, arrays)},
    )


@torch.no_grad()
def correct_aberrations_kouter(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor,
    out: torch.Tensor,
):
    """:func:`correct_aberrations_inplace` for a ``[Nk, Nqy, Nqx]`` chunk.

    Identical arithmetic -- the Warp kernel behind this is
    ``_direct_ptychography_forward_precomputed_A`` with its ``wp.tid()`` unpacking
    and its three subscripts permuted and nothing else changed, and the correction
    is a per-element multiply, so every output word is bit-identical to what the
    ``[Nqy, Nqx, Nk]`` op writes for the same input.

    Why the layout exists: ``ifft2`` over a chunk's two Q axes is **1.66x** faster
    when they are the INNER two (a natural batched transform) than when they are
    the outer two of a ``[Nqy, Nqx, Nk]`` tensor, and bit-for-bit identical
    (D49/D224).  Only a caller that transforms the same chunk several times can
    afford the transposing copy that gets it there, so this is deliberately not
    the default path -- see ``direct_ptychography_depth_section``.

    Args:
        Gprime: ``[Nk, Nqy, Nqx]`` complex chunk, left UNTOUCHED.
        aberrations: aberration coefficients.
        rotation: scan/detector rotation in degrees (float or device tensor).
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Qx, Qy, Kx, Ky: coordinate arrays, as for
            :func:`correct_aberrations_inplace`.
        A: ``[Nk, 2]`` float32 factor from :func:`build_aberration_bf_factor`.
            Pass an empty ``(0, 2)`` tensor to have the op build it.
        out: pre-allocated contiguous complex tensor shaped like ``Gprime``; the
            corrected values land here and it is what is returned.

    Returns:
        torch.Tensor - ``out``, as a complex view.
    """
    if out.shape != Gprime.shape or out.dtype != Gprime.dtype:
        raise ValueError(
            f"out must match Gprime in shape and dtype; got {tuple(out.shape)}/"
            f"{out.dtype} for {tuple(Gprime.shape)}/{Gprime.dtype}"
        )
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous():
        raise ValueError("out must be contiguous")
    torch.ops.scatterem.correct_aberrations_kouter(
        torch.view_as_real(Gprime).contiguous(),
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
    )
    return out


@torch.library.custom_op(
    "scatterem::correct_aberrations_kouter_planes",
    mutates_args=("out_real",),
)
def _correct_aberrations_kouter_planes_op(
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
) -> None:
    """``scatterem::correct_aberrations_kouter`` for ``n_planes`` aberration
    vectors at once: ONE launch instead of ``n_planes``.

    ``aberrations`` is ``[n_planes, n_ab]``, ``A`` is ``[Nk, n_planes, 2]`` and
    ``out_real`` is ``[n_planes, Nk, Nqy, Nqx, 2]``; ``Gprime_real`` stays the
    single ``[Nk, Nqy, Nqx, 2]`` chunk every plane reads.  No ``A``-building
    fallback and no in-place sentinel: the one caller of this layout hoists ``A``
    above its chunk loop and corrects out of place by construction.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    # ``return_ctype=True``, exactly as ``_correct_aberrations_fwd`` uses it: hand
    # Warp the low-level ``array_t`` descriptor instead of a ``wp.array`` wrapper.
    # ``pack_arg`` accepts the descriptor verbatim and it is the SAME BYTES the
    # wrapper would have been packed into, so the launch is bit-identical; what it
    # saves is the wrapper construction, ~6 us -> ~2 us per array over the eight
    # arrays this op builds per call.  That matters here because the depth sweep is
    # 86 % GPU-IDLE at the registered ``small`` size -- the host is the critical
    # path, not the kernel.
    #
    # Gated on float32 for every array, and the gate is a guard rather than a
    # supported path: ``pack_arg`` does no dtype check on a bare descriptor, so the
    # now-typed kernel would misread a non-float32 buffer.  Non-float32 never
    # worked here either (it failed at Warp codegen before the annotation), so the
    # fallback exists only so a wrong dtype fails loudly in Warp's own type check.
    ct = (
        Gprime_real.dtype is torch.float32
        and aberrations.dtype is torch.float32
        and A.dtype is torch.float32
        and out_real.dtype is torch.float32
        and Qx.dtype is torch.float32
        and Qy.dtype is torch.float32
        and Kx.dtype is torch.float32
        and Ky.dtype is torch.float32
    )
    wp.launch(
        kernel=_direct_ptychography_forward_precomputed_A_planes,
        dim=tuple(int(v) for v in out_real.shape[:-1]),
        inputs=[
            wp.from_torch(Gprime_real, dtype=wp.vec2, return_ctype=ct),
            wp.from_torch(Qx, return_ctype=ct),
            wp.from_torch(Qy, return_ctype=ct),
            wp.from_torch(Kx, return_ctype=ct),
            wp.from_torch(Ky, return_ctype=ct),
            wp.from_torch(A, dtype=wp.vec2, return_ctype=ct),
            wp.from_torch(aberrations, return_ctype=ct),
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[wp.from_torch(out_real, dtype=wp.vec2, return_ctype=ct)],
        device=wp.device_from_torch(Gprime_real.device),
    )


@torch.no_grad()
def correct_aberrations_kouter_planes(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    A: torch.Tensor,
    out: torch.Tensor,
):
    """:func:`correct_aberrations_kouter` for a STACK of aberration vectors.

    Identical arithmetic to ``n_planes`` separate :func:`correct_aberrations_kouter`
    calls -- the kernel behind this is the ``kouter`` kernel with a leading plane
    axis added to its ``wp.tid()`` and its subscripts, and the correction is a
    per-element multiply, so plane ``ip`` of ``out`` is bit-identical to what the
    per-plane call writes.  Verified by digest on
    ``r20_depth_section/{small,medium}/depth_section`` (D276).

    Why it exists: the correction's per-CALL host cost is a fixed 100-150 us
    (66 us of custom-op dispatch + ~54 us of Warp launch machinery, D275's split of
    D274-run's constant) that does not shrink with problem size, and the depth
    sweep pays it ``n_chunks * n_depths`` times.  One launch per chunk is the only
    way to remove the dispatch share without deleting the custom op.

    Args:
        Gprime: ``[Nk, Nqy, Nqx]`` complex chunk, left UNTOUCHED, shared by every
            plane.
        aberrations: ``[n_planes, n_ab]`` aberration coefficients.
        rotation: scan/detector rotation in degrees (float or device tensor).
        semiconvergence_angle: semiconvergence angle.
        wavelength: wavelength.
        Qx, Qy, Kx, Ky: coordinate arrays, as for
            :func:`correct_aberrations_kouter`.
        A: ``[Nk, n_planes, 2]`` float32 factor -- the plane axis INNERMOST, so
            that a caller holding ``[n_bf_total, n_planes, 2]`` can slice a chunk
            out of it for free.
        out: pre-allocated contiguous ``[n_planes, Nk, Nqy, Nqx]`` complex tensor.

    Returns:
        torch.Tensor - ``out``.
    """
    n_planes = int(aberrations.shape[0])
    if tuple(out.shape) != (n_planes, *Gprime.shape) or out.dtype != Gprime.dtype:
        raise ValueError(
            f"out must be [n_planes, *Gprime.shape] with Gprime's dtype; got "
            f"{tuple(out.shape)}/{out.dtype} for {n_planes} planes of "
            f"{tuple(Gprime.shape)}/{Gprime.dtype}"
        )
    out_real = torch.view_as_real(out)
    if not out_real.is_contiguous():
        raise ValueError("out must be contiguous")
    torch.ops.scatterem.correct_aberrations_kouter_planes(
        torch.view_as_real(Gprime).contiguous(),
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
    )
    return out


@torch.library.custom_op(
    "scatterem::phase_contrast_transfer_function_fwd",
    mutates_args=(),
)
def _phase_contrast_transfer_function_fwd(
    aberrations: Tensor,
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    sin_rot: float,
    cos_rot: float,
    semiconvergence_angle: float,
    wavelength: float,
) -> Tensor:
    """Compute the un-normalized phase contrast transfer function.

    The returned tensor has shape ``[Nqy, Nqx]`` (float32) and is the
    per-(Qy, Qx) sum of ``|gamma_complex|`` over ``ik``. The bright-field
    normalization ``2 * A.sum()`` is applied by the public wrapper so the
    op body stays pure Warp + a single output allocation.

    This op takes **no** ``G``: the kernel declares one but never indexes it
    (gamma is a function of Q, K, the aberrations, the aperture and the
    wavelength alone), so ``G`` only ever supplied the launch grid
    ``[Nqy, Nqx, Nk]`` and the output shape -- and both are already carried by
    the coordinate arrays as ``(len(Qy), len(Qx), len(Kx))``. Passing a
    ``G``-shaped array instead forced callers to materialize a ``U^2``-tiled
    copy of it (up to 7.7 GiB per call at the Fig1 FF-STEM config) that the
    device then never read.
    """
    device = wp.device_from_torch(Kx.device)
    n_qy, n_qx, n_k = int(Qy.shape[0]), int(Qx.shape[0]), int(Kx.shape[0])
    pctf = torch.zeros((n_qy, n_qx), dtype=torch.float32, device=Kx.device)
    # The kernel's ``G`` parameter is unindexed; Warp packs ``None`` as a null
    # array, so nothing is allocated or read for it.
    G_wp = None
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    pctf_wp = wp.from_torch(pctf)
    wp.launch(
        kernel=_phase_contrast_transfer_function_forward,
        dim=(n_qy, n_qx, n_k),
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
        ],
        outputs=[pctf_wp],
        device=device,
    )
    return pctf


@_phase_contrast_transfer_function_fwd.register_fake
def _(
    aberrations,
    Qx,
    Qy,
    Kx,
    Ky,
    sin_rot,
    cos_rot,
    semiconvergence_angle,
    wavelength,
):
    return torch.empty(
        (Qy.shape[0], Qx.shape[0]), dtype=torch.float32, device=Kx.device,
    )


@torch.no_grad()
def phase_contrast_transfer_function(
    G: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qx: torch.Tensor,
    Qy: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
):
    """
    Compute the phase contrast transfer function.

    Args:
        G: torch.Tensor | None - the ``[Nqy, Nqx, Nk]`` complex64 G tensor.
            **Its values are not read and it may be None.** The kernel's gamma
            depends only on Q, K, the aberrations, the aperture and the
            wavelength; G only ever supplied the launch grid, which is
            ``(len(Qy), len(Qx), len(Kx))``. It is kept in the signature for
            API compatibility and, when given, is checked against that shape.
        aberrations: torch.Tensor - aberrations array
        rotation: float | 0-d torch.Tensor - rotation in degrees. NOTE:
            passing a tensor here will cause a graph-break under
            ``torch.compile(fullgraph=True)`` because the wrapper calls
            ``rotation.item()`` to materialize a Python float for the
            ``math.sin/cos`` call. Pass a Python ``float`` if you intend
            to compile this function.
        semiconvergence_angle: float - semiconvergence angle
        wavelength: float - wavelength
        Qx: torch.Tensor - Qx coordinates
        Qy: torch.Tensor - Qy coordinates
        Kx: torch.Tensor - Kx coordinates
        Ky: torch.Tensor - Ky coordinates

    Returns:
        torch.Tensor - phase contrast transfer function (float32, [Nqy, Nqx])
    """
    if torch.is_tensor(rotation):
        # OK in eager; graph-breaks under torch.compile(fullgraph=True).
        # Compile-safe callers must pass a Python float (see docstring).
        rotation = float(rotation.item())
    sin_rot = math.sin(math.radians(rotation))
    cos_rot = math.cos(math.radians(rotation))
    if G is not None:
        expected = (Qy.shape[0], Qx.shape[0], Kx.shape[0])
        if tuple(G.shape) != expected:
            raise ValueError(
                f"G has shape {tuple(G.shape)}, expected {expected} = "
                "(len(Qy), len(Qx), len(Kx)); the kernel used to take its "
                "launch grid from G, so a mismatch read the coordinate "
                "arrays out of bounds"
            )
    pctf = torch.ops.scatterem.phase_contrast_transfer_function_fwd(
        aberrations.contiguous(),
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        sin_rot,
        cos_rot,
        float(semiconvergence_angle),
        float(wavelength),
    )
    K = torch.sqrt(Ky[None, :] ** 2 + Kx[None, :] ** 2)
    A = K < semiconvergence_angle / wavelength
    pctf_denominator = 2 * A.sum()
    return pctf / pctf_denominator


# ---------------------------------------------------------------------------
# CorrectAberrations -- Pattern H (autograd via hand-written backward kernel)
# ---------------------------------------------------------------------------
#
# The forward op evaluates the production direct-ptychography forward
# (``out = G * conj(gamma_phase)`` with unit-magnitude ``gamma_phase``). The
# backward op invokes ``_direct_ptychography_backward_analytic`` -- the same
# hand-written analytic kernel the legacy ``torch.autograd.Function`` used --
# which differentiates the *unnormalized* effective loss
# (``out_eff = G * conj(gamma_complex)``). This documented forward-vs-backward
# semantic mismatch is preserved verbatim from the original implementation
# (see ``TestCorrectAberrationsAutograd`` for the FD pin against the matched
# effective loss).


@torch.library.custom_op(
    "scatterem::correct_aberrations_fwd",
    mutates_args=(),
)
def _correct_aberrations_fwd(
    Gprime_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- view_as_real(Gprime)
    aberrations: Tensor,
    sin_cos_rot: Tensor,  # length-2 float32 [sin, cos]
    Qx: Tensor,
    Qy: Tensor,
    Kx: Tensor,
    Ky: Tensor,
    semiconvergence_angle: float,
    eps: float,
    wavelength: float,
    n_grad_coeffs: int,
) -> Tensor:
    """Forward of ``CorrectAberrations``.

    Returns a freshly-allocated ``(Nqy, Nqx, Nk, 2)`` float32 tensor; the
    public wrapper applies ``view_as_complex`` on the way out.

    ``n_grad_coeffs`` is unused by the forward -- it is carried here only so
    ``_correct_aberrations_setup_context`` can see it, because the backward's
    cost is linear in it (see the comment on that function).  Negative means
    "all of them", which is what every caller got before it existed.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(Gprime_real.device)
    # ``empty_like``, NOT ``zeros_like``: the launch below covers the destination
    # EXACTLY once and stores unconditionally.  ``dim=Gprime_real.shape[:-1]``
    # gives one thread per ``(iqy, iqx, ik)`` and the last line of
    # ``_direct_ptychography_forward`` is a plain
    # ``G_out[iqy, iqx, ik] = cmul(...)`` OUTSIDE that kernel's aperture guard --
    # the guard skips the aberration polynomials, never the store -- so every
    # ``vec2`` (both floats) is written before anything can read it.  A zero-fill
    # is therefore dead stores, and they are not cheap: the destination is the
    # same size as the input, which at the published ``Fig1_Gd2O3`` fit config is
    # 86.0 MiB per bright-field chunk, and this op runs once per (objective
    # evaluation x chunk) -- 1775-2625 times in ONE ``determine_aberrations``.
    # Measured at that shape: the memset alone is 0.128 ms, and the op goes
    # 0.927 -> 0.796 ms = 1.165x, worth ~4.8 % of the fit stage.
    #
    # Do NOT copy this to a kernel that skips stores.  The sibling
    # ``_phase_contrast_transfer_function_forward`` DOES early-out (R9c) and
    # accumulates with ``wp.atomic_add``, so its destination must stay zeroed;
    # its caller allocates separately and is untouched here.  If
    # ``_direct_ptychography_forward`` ever gains a guarded store, this must go
    # back to ``zeros_like`` in the same commit.
    out_real = torch.empty_like(Gprime_real)
    # ``return_ctype=True`` hands Warp the low-level ``array_t`` descriptor
    # (ptr/grad_ptr/ndim/shape/strides) instead of a ``wp.array`` wrapper.
    # ``pack_arg`` accepts a descriptor verbatim, and it is the SAME BYTES the
    # wrapper would have been packed into -- ``wp.array.__ctype__()`` builds the
    # identical struct, field for field, including the gradient pointer Warp
    # allocates for a ``requires_grad`` input (verified over every array this op
    # wraps in ``benchmarks/lab/_probe_r9f_exact.py``). What it saves is the
    # wrapper construction: 6.1 us -> 2.0 us per 1-D array, 7.2 -> 2.7 for the
    # vec2 views, i.e. ~30 us of the ~100 us this op spends on the host per call
    # -- and this op runs 1800 times in one aberration fit, at 9 % GPU
    # utilisation, so host time is the critical path.
    #
    # This requires the launched kernel to be non-generic (see the dtype
    # annotations on ``_direct_ptychography_forward``): Warp cannot infer an
    # argument type from a bare descriptor. It also means ``pack_arg`` does no
    # dtype check, so anything but float32 falls back to the wrapper, which the
    # now-typed kernel rejects with a clear error rather than misreading the
    # buffer. Non-float32 never worked here -- it failed at Warp codegen before
    # this change -- so the fallback is a guard, not a supported path.
    ct = (
        Gprime_real.dtype is torch.float32
        and aberrations.dtype is torch.float32
        and Qx.dtype is torch.float32
        and Qy.dtype is torch.float32
        and Kx.dtype is torch.float32
        and Ky.dtype is torch.float32
    )
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2, return_ctype=ct)
    out_wp = wp.from_torch(out_real, dtype=wp.vec2, return_ctype=ct)
    Qx_wp = wp.from_torch(Qx, return_ctype=ct)
    Qy_wp = wp.from_torch(Qy, return_ctype=ct)
    Kx_wp = wp.from_torch(Kx, return_ctype=ct)
    Ky_wp = wp.from_torch(Ky, return_ctype=ct)
    ab_wp = wp.from_torch(aberrations, return_ctype=ct)
    wp.launch(
        kernel=_direct_ptychography_forward,
        dim=Gprime_real.shape[:-1],
        inputs=[
            G_wp,
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            ab_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[out_wp],
        device=device,
    )
    return out_real


@_correct_aberrations_fwd.register_fake
def _(
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    semiconvergence_angle,
    eps,
    wavelength,
    n_grad_coeffs,
):
    return torch.empty_like(Gprime_real)


@torch.library.custom_op(
    "scatterem::correct_aberrations_bwd",
    mutates_args=(),
)
def _correct_aberrations_bwd(
    grad_out_real: Tensor,  # (Nqy, Nqx, Nk, 2) float32 -- view_as_real(adj_G)
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
    """Analytic gradient w.r.t. ``aberrations``.

    Mirrors the legacy wrapper's launch -- crucially with ``block_dim=256``
    so the cooperative ``wp.tile`` / ``wp.tile_atomic_add`` reductions
    inside ``_direct_ptychography_backward_analytic`` work correctly.

    ``n_coeffs`` is the number of LEADING coefficients the kernel's
    ``for j in range(n_coeffs)`` loop covers; the returned gradient always has
    ``aberrations``' own length, with the coefficients beyond ``n_coeffs``
    left at the zero they are initialised to.  Those two used to be the same
    number, and separating them is what lets a caller that has frozen the
    high-order coefficients stop paying for them -- see
    ``_correct_aberrations_setup_context``.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(grad_out_real.device)
    out_grad = torch.zeros(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device
    )
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    adj_wp = wp.from_torch(grad_out_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    out_grad_wp = wp.from_torch(out_grad)
    wp.launch(
        kernel=_direct_ptychography_backward_analytic,
        dim=grad_out_real.shape[:-1],
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


@_correct_aberrations_bwd.register_fake
def _(
    grad_out_real,
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    semiconvergence_angle,
    wavelength,
    n_coeffs,
):
    return torch.empty(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device,
    )


def _correct_aberrations_setup_context(ctx, inputs, output):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
        semiconvergence_angle,
        eps,
        wavelength,
        n_grad_coeffs,
    ) = inputs
    ctx.save_for_backward(
        Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky,
    )
    ctx.semiconvergence_angle = float(semiconvergence_angle)
    ctx.wavelength = float(wavelength)
    # ``_direct_ptychography_backward_analytic``'s cost is LINEAR in this
    # number: per coefficient it evaluates ``dchi_cartesian_aberrations``
    # three times and then does a block-wide ``wp.tile_sum`` plus a
    # ``wp.tile_atomic_add``.  Measured on the published ``Fig1_Gd2O3`` fit
    # chunk (512x512x43): 1.098 ms at ``n_coeffs=1``, 3.530 ms at 12, i.e. a
    # ~0.93 ms floor (the 172 MiB of ``G``/``dL_dG`` loads and the three chi
    # evaluations) plus ~0.216 ms per coefficient.
    #
    # Callers that have FROZEN the high-order coefficients multiply those
    # gradients by exactly zero, so computing them is dead work -- and it is
    # the majority of the kernel: ``determine_aberrations`` at
    # ``correct_order=1`` frees three of twelve.  ``n_grad_coeffs`` lets such
    # a caller say so; it is a LEADING count rather than a mask because
    # ``_build_gradient_mask`` freezes a suffix, and a leading count stays
    # CORRECT (merely less aggressive) under an arbitrary user mask.
    #
    # Skipping a coefficient cannot perturb the ones that remain: each ``j``
    # accumulates into its own ``out_grad[j]`` slot, so this is dead-work
    # removal, not a reassociation.
    n_all = int(min(12, aberrations.shape[0]))
    ctx.n_coeffs = n_all if int(n_grad_coeffs) < 0 else min(n_all, int(n_grad_coeffs))


def _correct_aberrations_backward(ctx, grad_out_real):
    (
        Gprime_real,
        aberrations,
        sin_cos_rot,
        Qx,
        Qy,
        Kx,
        Ky,
    ) = ctx.saved_tensors
    ab_grad = torch.ops.scatterem.correct_aberrations_bwd(
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
    # Preserve original-wrapper sign convention: returned ``-ab_grad``.
    # The 11 returns line up with the 11 forward inputs of the custom op
    # (Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky, semi, eps,
    # wavelength, n_grad_coeffs). Only ``aberrations`` is differentiable.
    return (None, -ab_grad, None, None, None, None, None, None, None, None, None)


torch.library.register_autograd(
    "scatterem::correct_aberrations_fwd",
    _correct_aberrations_backward,
    setup_context=_correct_aberrations_setup_context,
)


def correct_aberrations(
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
) -> torch.Tensor:
    """Direct-ptychography aberration correction with analytic backward.

    Differentiable input: ``aberrations``. The backward uses the analytic
    ``_direct_ptychography_backward_analytic`` kernel (Pattern H). See the
    forward-vs-backward semantic note at the top of this op block.

    ``n_grad_coeffs`` (default: all of them) is the number of LEADING
    aberration coefficients the backward computes a gradient for; the rest
    come back as exact zeros. A caller that has frozen the high-order
    coefficients should pass its own count -- the adjoint's cost is linear in
    it. It cannot change the gradients that are computed; see
    ``_correct_aberrations_setup_context``.
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
    Gprime_real = torch.view_as_real(Gprime).contiguous()
    out_real = torch.ops.scatterem.correct_aberrations_fwd(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        -1 if n_grad_coeffs is None else int(n_grad_coeffs),
    )
    return torch.view_as_complex(out_real)


# ---------------------------------------------------------------------------
# Fused correct-and-reduce.  Same op pair, same analytic-adjoint contract, but
# the ``ik`` axis is summed inside the kernel.  See the block comment above
# ``_direct_ptychography_forward_ksum`` for what changes numerically and why the
# ``G`` load may move inside the aperture guard.
# ---------------------------------------------------------------------------

# A RECORDED ``wp.launch`` for the k-SUM correction kernel -- ``_KOUTER_CMD``'s
# mechanism, one op over, with the operand key D280 built.
#
# WHY THE KEY IS NOT ``is``-IDENTITY.  ``correct_aberrations_ksum`` hands this op
# ``torch.view_as_real(Gprime).contiguous()`` and a freshly allocated ``out_real``
# every call, so an identity test can NEVER hit on those slots (D280 measured the
# same defect on the k-outer op).  The key is therefore
# ``(data_ptr, shape, stride, dtype)`` with ``is`` kept only as a fast path, and
# the entry holds a STRONG REFERENCE to every bound tensor -- which is exactly
# what makes the pointer key sound: while the previously bound tensor is alive
# the caching allocator cannot hand its block to anything else, so an equal
# ``data_ptr`` implies the same live storage and the same words.
#
# WHY IT PAYS HERE AND DID NOT ON THE K-OUTER ROW.  D280 measured this mechanism
# at 1.0003x on ``r20_depth_section/medium/depth_section``, which is 6.45 %
# GPU-IDLE -- a host removal there cashes at ~1 % of its nominal size.  The k-sum
# row ``r20_depth_section/medium/depth_section_sumfirst`` is **42.6 % GPU-idle**
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


def _ksum_bind_key(t: Tensor):
    return (t, t.data_ptr(), t.shape, t.stride(), t.dtype)


@torch.library.custom_op(
    "scatterem::correct_aberrations_ksum_fwd",
    mutates_args=(),
)
def _correct_aberrations_ksum_fwd(
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
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(Gprime_real.device)
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
            (Kx.shape[0], 2), dtype=torch.float32, device=Gprime_real.device,
        )
        # ``_direct_ptychography_build_A`` declares ``Kx_all``/``Ky_all``/
        # ``aberrations`` as untyped ``wp.array(ndim=1)``, so Warp must infer
        # their element type at launch.
        wp.launch(
            kernel=_direct_ptychography_build_A,
            dim=(Kx.shape[0],),
            inputs=[
                wp.from_torch(Kx),
                wp.from_torch(Ky),
                wp.from_torch(aberrations),
                semiconvergence_angle,
                wavelength,
            ],
            outputs=[wp.from_torch(A, dtype=wp.vec2)],
            device=device,
        )
    dim = (int(Gprime_real.shape[0]), int(Gprime_real.shape[1]))
    # The five scalars are recorded INTO the command, so a change in any of them
    # is a re-record and not a rebind -- cheaper to compare than to set, and it
    # keeps the scalar path impossible to get wrong.
    scalars = (
        float(sin_rot), float(cos_rot), float(semiconvergence_angle),
        float(eps), float(wavelength),
    )
    arrays = (Gprime_real, Qx, Qy, Kx, Ky, A, aberrations, out_real)
    key = (dim, str(device))
    entry = _KSUM_CMD.get(key)
    if entry is not None and entry[1] == scalars:
        cmd, _, bound = entry
        for slot, t in zip(_KSUM_ARRAY_SLOTS, arrays):
            prev = bound[slot]
            if prev[0] is not t and (
                prev[1] != t.data_ptr()
                or prev[2] != t.shape
                or prev[3] != t.stride()
                or prev[4] is not t.dtype
            ):
                cmd.set_param_at_index_from_ctype(
                    slot,
                    wp.from_torch(t, dtype=wp.vec2, return_ctype=True)
                    if slot in _KSUM_VEC2_SLOTS
                    else wp.from_torch(t, dtype=wp.float32, return_ctype=True),
                )
                bound[slot] = _ksum_bind_key(t)
        cmd.launch()
        return out_real
    cmd = wp.launch(
        kernel=_direct_ptychography_forward_ksum,
        dim=dim,
        inputs=[
            wp.from_torch(Gprime_real, dtype=wp.vec2),
            wp.from_torch(Qx),
            wp.from_torch(Qy),
            wp.from_torch(Kx),
            wp.from_torch(Ky),
            wp.from_torch(A, dtype=wp.vec2),
            wp.from_torch(aberrations),
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
        ],
        outputs=[wp.from_torch(out_real, dtype=wp.vec2)],
        device=device,
        record_cmd=True,
    )
    # ``record_cmd=True`` RECORDS and does NOT launch.
    cmd.launch()
    if len(_KSUM_CMD) >= _KSUM_CMD_MAX:
        _KSUM_CMD.clear()
    _KSUM_CMD[key] = (
        cmd,
        scalars,
        {slot: _ksum_bind_key(t) for slot, t in zip(_KSUM_ARRAY_SLOTS, arrays)},
    )
    return out_real


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
):
    return Gprime_real.new_empty(
        (Gprime_real.shape[0], Gprime_real.shape[1], 2)
    )


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
    device = wp.device_from_torch(grad_out_real.device)
    out_grad = torch.zeros(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device
    )
    G_wp = wp.from_torch(Gprime_real, dtype=wp.vec2)
    adj_wp = wp.from_torch(grad_out_real, dtype=wp.vec2)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    ab_wp = wp.from_torch(aberrations)
    out_grad_wp = wp.from_torch(out_grad)
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


@_correct_aberrations_ksum_bwd.register_fake
def _(
    grad_out_real,
    Gprime_real,
    aberrations,
    sin_cos_rot,
    Qx,
    Qy,
    Kx,
    Ky,
    semiconvergence_angle,
    wavelength,
    n_coeffs,
):
    return torch.empty(
        (aberrations.shape[0],), dtype=torch.float32, device=grad_out_real.device,
    )


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
    ) = inputs
    # ``A`` is deliberately NOT saved: the analytic adjoint does not consume it
    # (it evaluates ``A0`` with UNIT amplitude and no aperture -- the deliberate
    # forward/adjoint mismatch recorded as DISCOVERED D7), so the hoist changes
    # the backward neither in value nor in cost.
    ctx.save_for_backward(
        Gprime_real, aberrations, sin_cos_rot, Qx, Qy, Kx, Ky,
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
    # Same sign convention as the unfused op; 12 slots, one per forward input
    # (the extra one is the hoisted ``A``, which is never differentiated).
    return (
        None, -ab_grad, None, None, None, None, None, None, None, None, None, None,
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
) -> torch.Tensor:
    """``correct_aberrations(...).sum(dim=-1)``, fused into one kernel.

    Returns the ``(Nqy, Nqx)`` complex bright-field-summed corrected ``G``.
    Use this wherever the corrected chunk is only ever reduced over ``ik``: it
    never allocates the ``(Nqy, Nqx, Nk)`` intermediate, and its backward takes
    the reduced ``(Nqy, Nqx)`` adjoint instead of an ``ik``-broadcast copy of
    it.

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
    """
    sin_cos_rot = _sin_cos_rot(rotation, Gprime.device)
    # ``Gprime`` may be STRIDED.  Warp only requires the INNER stride to be
    # dense, and ``view_as_real`` of any complex tensor has last stride 1 by
    # construction, so ``wp.from_torch(..., dtype=wp.vec2)`` takes an arbitrary
    # outer layout; a ``.contiguous()`` here would be a full read+write pass to
    # buy nothing.  It matters because the one hot caller -- the chunk-sum-first
    # branch of ``_iter_chunk_images`` -- reads ``vBF.G[..., s:e]``, and ``G`` is
    # an ``fft2`` output, i.e. bright-field-OUTERMOST in memory: compacting that
    # slice is a TRANSPOSE, measured at 1.116 ms against a 3.40 ms row (D324) --
    # 32.6 % of it.  The strided read costs the kernel nothing measurable in situ
    # (1.5376 -> 1.5213 ms), so the copy is pure loss: 1.59x on that loop and
    # 1.1235x on the whole row, plus 82 MiB off its peak.  Bit-exact:
    # the kernel reads the same values in the same ``ik``-ascending order and
    # accumulates in one register, so only the ADDRESSES move.
    Gprime_real = torch.view_as_real(Gprime)
    out_real = torch.ops.scatterem.correct_aberrations_ksum_fwd(
        Gprime_real,
        aberrations.contiguous(),
        sin_cos_rot,
        Qx.contiguous(),
        Qy.contiguous(),
        Kx.contiguous(),
        Ky.contiguous(),
        (
            Gprime_real.new_empty((0, 2))
            if A is None
            else A.contiguous()
        ),
        float(semiconvergence_angle),
        1e-3,
        float(wavelength),
        -1 if n_grad_coeffs is None else int(n_grad_coeffs),
    )
    return torch.view_as_complex(out_real)


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
) -> None:
    """``scatterem::correct_aberrations_ksum_fwd`` for ``n_planes`` aberration
    vectors at once: ONE launch instead of ``n_planes``.

    ``aberrations`` is ``[n_planes, n_ab]``, ``A`` is ``[Nk, n_planes, 2]`` and
    ``out_real`` is ``[n_planes, Nqy, Nqx, 2]``; ``Gprime_real`` stays the single
    ``[Nqy, Nqx, Nk, 2]`` chunk every plane reduces.  No ``A``-building fallback
    and no recorded-launch memo: the one caller hoists ``A`` above its chunk loop
    and issues this ``n_chunks`` times per sweep, where the per-plane op it
    replaces was issued ``n_chunks * n_planes`` times -- the memo D306 added to
    that op exists precisely because it is on the hot path, and this one is not.
    """
    sin_rot, cos_rot = _sin_cos_floats(sin_cos_rot)
    device = wp.device_from_torch(Gprime_real.device)
    Qx_wp = wp.from_torch(Qx)
    Qy_wp = wp.from_torch(Qy)
    Kx_wp = wp.from_torch(Kx)
    Ky_wp = wp.from_torch(Ky)
    # THE APERTURE IS ``ip``-INDEPENDENT, so build it once per chunk instead of
    # once per plane.  ``a2``/``a3`` are functions of ``(iqy, iqx, ik)`` alone and
    # each costs a ``wp.sqrt`` plus a ``wp.asin``; the planes kernel's grid is
    # ``[n_planes, Nqy, Nqx]``, so it evaluated 2 x ``n_planes`` x Nqy x Nqx x Nk
    # of them for Nqy x Nqx x Nk distinct answers.  Measured on the row's own
    # captured arguments (``benchmarks/lab/_probe_d348_kernel.py``, medians of 20
    # after 3 warm-ups, 5 alternating in-process cycles): the correction kernel
    # 0.7611 -> 0.6309 ms with the mask already built, and 0.7078 ms including this
    # build launch = **1.0753x on the kernel**, +0.427 ms on the 8-chunk row.
    #
    # The mask, not a float array: ``aperture`` returns exactly ``+0.0``/``+1.0``,
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
    #   * the algebraic rewrite ``asin(q*lambda) < alpha  <=>  q^2 < (sin(alpha)
    #     /lambda)^2`` deletes the sqrt and the asin outright and needs no mask,
    #     but it is a HARD-THRESHOLD predicate on a different expression, so a
    #     point within an ULP of the aperture edge can flip side.  Not taken.
    mask = torch.empty(
        (Gprime_real.shape[0], Gprime_real.shape[1], Kx.shape[0]),
        dtype=torch.uint8,
        device=Gprime_real.device,
    )
    mask_wp = wp.from_torch(mask, dtype=wp.uint8)
    wp.launch(
        kernel=_direct_ptychography_aperture_mask_planes,
        dim=tuple(int(v) for v in mask.shape),
        inputs=[
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            wavelength,
        ],
        outputs=[mask_wp],
        device=device,
    )
    wp.launch(
        kernel=_direct_ptychography_forward_ksum_planes,
        dim=tuple(int(v) for v in out_real.shape[:-1]),
        inputs=[
            wp.from_torch(Gprime_real, dtype=wp.vec2),
            Qx_wp,
            Qy_wp,
            Kx_wp,
            Ky_wp,
            wp.from_torch(A, dtype=wp.vec2),
            wp.from_torch(aberrations),
            sin_rot,
            cos_rot,
            semiconvergence_angle,
            eps,
            wavelength,
            mask_wp,
        ],
        outputs=[wp.from_torch(out_real, dtype=wp.vec2)],
        device=device,
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
) -> torch.Tensor:
    """:func:`correct_aberrations_ksum` for a STACK of aberration vectors.

    Identical arithmetic to ``n_planes`` separate :func:`correct_aberrations_ksum`
    calls: the kernel behind this is the ``ksum`` kernel with a leading plane axis
    added to its ``wp.tid()`` and two of its subscripts, and each thread still
    accumulates over ``ik`` ascending in one float32 register, so plane ``ip`` of
    ``out`` is bit-identical to what the per-plane call returns.

    Why it exists: :func:`correct_aberrations_kouter_planes` is this same fold on
    the UNREDUCED kernel, built for the ``k``-outer branch of
    ``direct_ptychography_depth_section``; the ``_CHUNK_SUM_FIRST`` branch of that
    same sweep still paid the correction's fixed per-call host cost
    ``n_chunks * n_depths`` times.

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
    # construction, so ``wp.from_torch(..., dtype=wp.vec2)`` takes an arbitrary
    # outer layout; a ``.contiguous()`` here is a full read+write pass buying
    # nothing.  The hot caller is the ``_CHUNK_SUM_FIRST`` branch of
    # ``direct_ptychography_depth_section``, which now hands the read-only
    # ``G[..., s:e]`` view straight through: G is an ``fft2`` output, i.e.
    # bright-field-OUTERMOST in memory, so compacting that slice is a TRANSPOSE --
    # 1.156 ms of a 9.68 ms row, 12 % of it.  ``out`` is validated above and is
    # untouched by this; only the INPUT's layout requirement relaxes.  Bit-exact:
    # each thread still reads the same values in the same ``ik``-ascending order
    # into one register, so only the ADDRESSES move.
    torch.ops.scatterem.correct_aberrations_ksum_planes(
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
    )
    return out


# Preserve the legacy ``CorrectAberrations.apply(...)`` API so existing
# callers (``scatterem/reconstruction/direct_ptychography.py``, downstream
# scripts) keep working without edits.
class CorrectAberrations:
    """Compatibility shim exposing the legacy ``.apply`` entry point.

    The original ``torch.autograd.Function`` was replaced by a pair of
    ``torch.library.custom_op`` ops (``scatterem::correct_aberrations_fwd``
    and ``scatterem::correct_aberrations_bwd``) wired together via
    ``torch.library.register_autograd``. ``CorrectAberrations.apply`` now
    forwards to the public :func:`correct_aberrations` callable, which is
    compile-safe under ``torch.compile(fullgraph=True, dynamic=False)``.
    """

    apply = staticmethod(correct_aberrations)




