"""Process-wide memos for the aberration-correction ops.

Every helper here is a pure memo of a deterministic computation, kept in a
BOUNDED module-level dict that holds a strong reference to what it caches; the
per-cache comments below say why each one exists, what it is keyed on and why
reuse is bit-exact.
"""

import torch
import warp as wp
from torch import Tensor

from scatterem.nn.functional.warp.ptychography import (
    _direct_ptychography_chi_is_zero,
)
from scatterem.physics import _warp_interop as wi

try:  # pragma: no cover - present in every torch this package supports
    import torch._C._functorch as _functorch
except ImportError:  # pragma: no cover - a torch built without functorch
    # ``torch._C._functorch`` is the C++ module that IMPLEMENTS ``torch.vmap``
    # and friends, so a build without it is a build in which no functorch
    # transform can be active and the check ``_ksum_fwd_bypass_ok`` makes with
    # it is vacuously true.
    _functorch = None


# ``[sin, cos]`` arrives as a DEVICE tensor (see the ``_sin_cos_rot`` comment
# below), and every op body that launches an aberration kernel needs the two
# values as Python floats to pass them as kernel scalars.  ``.item()`` is how you
# do that, and ``.item()`` is a **blocking device synchronisation**: it drains
# everything queued on the stream before the host may continue.
#
# Measured in situ on the published ``Fig1_Gd2O3`` fit (a
# benchmark script): 3459 ``Tensor.item()`` calls -- 2 per forward op and 2
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
# operands on EVERY call, and a former per-plane correction sweep called
# it ``n_chunks * n_depths`` times with the same coordinates.  Measured on
# a benchmark row: 40 calls per row, and
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


#: ``(id, data_ptr, _version, n, device) -> (flag tensor, aberrations)`` for
#: :func:`_chi_is_zero_flag`.  Bounded like ``_KOUTER_CMD``.
_CHI_ZERO_FLAG: dict = {}
_CHI_ZERO_FLAG_MAX = 4


def _chi_is_zero_flag(aberrations: Tensor, device) -> Tensor:
    """A ONE-ELEMENT DEVICE tensor holding the identity predicate.

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
    (a benchmark script), against the ~0.34 ms the fold is
    worth.  A sync was removed from this exact row before; this must not put one back.

    (2) THE FLAG IS MEMOISED, and without that the fold is BELOW a 5 %
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

    (3) THE ``wi.from_torch`` HANDLE IS BUILT HERE, ON THE MISS.  It used to be a
    parameter, which obliged the caller to build one on every entry in order to
    feed a function that, by (2), needs it on approximately none of them.
    """
    key = (
        id(aberrations),
        aberrations.data_ptr(),
        aberrations._version,
        int(aberrations.shape[0]),
        str(device),
    )
    hit = _CHI_ZERO_FLAG.get(key)
    if hit is not None:
        return hit[0]
    flag = torch.empty(1, dtype=torch.int32, device=aberrations.device)
    wp.launch(
        kernel=_direct_ptychography_chi_is_zero,
        dim=(1,),
        inputs=[wi.from_torch(aberrations)],
        outputs=[wi.from_torch(flag)],
        device=device,
    )
    if len(_CHI_ZERO_FLAG) >= _CHI_ZERO_FLAG_MAX:
        _CHI_ZERO_FLAG.clear()
    _CHI_ZERO_FLAG[key] = (flag, aberrations)
    return flag


def chi_is_zero_host(aberrations) -> bool:
    """:func:`_chi_is_zero_flag`'s predicate, on the HOST, in one sync.

    For the caller that can amortise a sync -- one above a chunk loop rather than
    one per call -- and therefore wants to select a specialised KERNEL rather than
    branch inside a general one.  ``correct_aberrations_ksum``'s ``chi_is_zero``
    is that caller's; ``_direct_ptychography_forward_ksum_chi_zero``'s docstring
    prices the difference (1.4264x host-selected vs 1.3600x branched, and the
    branch costs the general path 0.9750x).

    EXACTLY the predicate ``_direct_ptychography_chi_is_zero`` computes on the
    device, and it must stay that way: ``+0.0`` only.  ``signbit`` is what
    excludes ``-0.0``, whose ``exp(-i chi)`` is ``(1.0, +0.0)`` and not
    ``(1.0, -0.0)`` -- the device kernel must exclude it too.  NaN answers
    ``False`` in both.

    Returns ``False`` for anything that is not a floating-point tensor, so a
    caller may hand it whatever it holds.
    """
    if not torch.is_tensor(aberrations) or not aberrations.is_floating_point():
        return False
    if torch.compiler.is_compiling():
        return False
    bad = (aberrations != 0) | torch.signbit(aberrations)
    return not bool(bad.any())
