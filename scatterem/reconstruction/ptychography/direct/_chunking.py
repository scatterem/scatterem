"""The chunked single-side-band accumulation loop and its k-outer variants."""

import math
import weakref

import numpy as np
import torch

from scatterem.nn.functional.ptychography import (
    build_aberration_bf_factor,
    chi_is_zero_host,
    correct_aberrations_inplace,
    correct_aberrations_kouter,
    correct_aberrations_ksum,
)

from . import _config
from ._chunk_count import _effective_n_batches
from ._common import _chunk_ranges
from ._kouter_half import _kouter_sum_half

# ``get_q_1d`` is three device launches per axis (``fftfreq`` is two ``arange_out``
# plus a ``mul_``) and 24 us of host time per call, and the grid it builds depends
# only on the vBF's shape, sampling and device -- none of which changes inside a
# reconstruction.  ``determine_aberrations`` calls ``direct_ptychography`` ~1800
# times per fit, so those six launches were paid 1800 times for one answer.  Worse,
# the fresh ``Qx``/``Qy`` objects also miss ``_contiguous_memo`` and force operand
# rebinds in the recorded aberration launch, exactly as ``vBF.k[s:e, 1]`` did before
# (see ``_contiguous_memo``'s comment in
# ``nn/functional/ptychography/_caches.py``).
#
# Keyed BY VALUE on every input the computation reads, plus an identity re-check on
# the vBF held as a WEAK reference -- ``hit[0]() is vBF`` catches the
# evicted-then-reused-id case exactly as the previous strong reference did, because a
# weakref to a dead object dereferences to ``None`` and ``None is vBF`` is False, so a
# recycled ``id`` can only ever MISS.  A change to shape, sampling, device or upsample
# factor therefore MISSES.  Bit-exact by construction: nothing is recomputed, the
# caller gets the tensors the first call built.
#
# WHY WEAK AND NOT STRONG.  The strong reference this used to keep was a
# process-lifetime leak of the whole DATASET, not of the cache's payload: a vBF holds
# its bright-field stack and (lazily) its ``G``, and the ``Dataset4DStem`` it was built
# from holds the 4D cube.  So up to ``_UPSAMPLED_GRID_CACHE_MAX = 8`` entries, each
# carrying two 1-D q grids of a few KiB, pinned entire multi-GiB datasets that nothing
# else referred to.  Measured: on ``r14_ffstem/gd2o3`` (a 12.25 GiB device cube)
# this made the benchmark check of ``r14_ffstem/gd2o3`` read
# ``2 CHECK(S) FAILED`` on UNMODIFIED code -- the 3rd of 4 variants raised
# ``torch.OutOfMemoryError`` with 37.02 GiB still allocated after the first two, while
# each variant on its own reads ``ALL PASS``.  A cache whose retention exceeds its
# payload by six orders of magnitude is a leak wearing a memo's clothes; the identity
# guarantee it was bought for costs nothing to keep.
#
# The returned ``Qy``/``Qx`` are READ-ONLY, deliberately, which is what makes this
# free -- the same contract ``_g_chunk_kouter_view`` documents below.  Every consumer
# on these paths passes them to an aberration kernel as an input (the ops declare
# only ``Gprime_real``/``out_real`` in ``mutates_args``).  Do not hand them to
# anything that writes in place.
_UPSAMPLED_GRID_CACHE: dict = {}
_UPSAMPLED_GRID_CACHE_MAX = 8


def _upsampled_grid(vBF, upsample_int):
    """``(new_shape, Qy, Qx)`` of the reconstruction grid — derived from the
    vBF STACK shape (the streaming provider may keep no resident G)."""
    new_shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    key = (
        id(vBF),
        new_shape,
        tuple(int(s) for s in vBF.shape[:2]),
        tuple(float(s) for s in vBF.sampling[:2]),
        str(vBF.device),
    )
    hit = _UPSAMPLED_GRID_CACHE.get(key)
    if hit is not None and hit[0]() is vBF:
        return new_shape, hit[1], hit[2]
    Qy, Qx = vBF.get_q_1d(new_shape)
    if len(_UPSAMPLED_GRID_CACHE) >= _UPSAMPLED_GRID_CACHE_MAX:
        _UPSAMPLED_GRID_CACHE.clear()
    try:
        ref = weakref.ref(vBF)
    except TypeError:
        # A provider that does not support weak references simply is not cached,
        # which is the pre-cache behaviour and costs only the six launches this
        # memo exists to remove.  Never fall back to a strong reference: that is
        # the leak above.
        return new_shape, Qy, Qx
    _UPSAMPLED_GRID_CACHE[key] = (ref, Qy, Qx)
    return new_shape, Qy, Qx


def _g_chunk_kouter_view(vBF, s: int, e: int):
    """``G[..., s:e]`` as a ``[Nk, Nqy, Nqx]`` view -- **or None** when that view
    is not already contiguous, i.e. when reaching it would cost a real copy.

    ``vBF.G`` is ``torch.fft.fft2(array, dim=(0, 1))``, and torch/cuFFT works in
    the ``[B, ny, nx]`` layout and hands back a permuted VIEW: the resident G has
    strides ``(nx, 1, ny*nx)`` and ``is_contiguous() == False``, so it is ALREADY
    bright-field-outermost in memory.  ``get_G_chunk``'s ``.contiguous()``
    therefore does not "compact a non-contiguous slice" -- it TRANSPOSES G into
    the ``[Nqy, Nqx, Nk]`` order, which is the order the ``ifft2`` that follows
    then has to transpose back out of (measured: the copy 0.150 ms and the
    transform 1.66x slower, at the registered 245^2 x 90 chunk).  Asking for the
    k-outer view instead costs **nothing at all** -- the permuted slice is
    already dense, so ``.contiguous()`` returns it unchanged.

    Returned READ-ONLY, deliberately, which is what makes it free.  The caller
    ``get_G_chunk`` serves needs an OWNED chunk because
    ``correct_aberrations_inplace`` mutates it; ``correct_aberrations_kouter`` is
    always out of place and never touches its input, so no ownership copy is
    required on this path.  Do not hand this tensor to anything that mutates.

    None is returned -- and the caller falls back to the shipped
    ``[Nqy, Nqx, Nk]`` body -- whenever the layout is not free:

    * an alignment overlay is active (``_align_shifts``): ``phase_ramp`` builds its
      ramp ``[Nk, Nqy, Nqx]`` and a copy transposes it to ``[Nqy, Nqx, Nk]``; it
      only runs during a bright-field-shift fit, so no second spelling here;
    * the provider keeps no resident ``G`` (``DatasetVirtualBrightFieldStreaming``
      at ``g_residency`` "vbf_gpu"/"cpu_pinned" raises on ``.G``).  Those two
      residencies *also* produce dense-but-k-outer chunks (an ``fft2`` output and
      a ``movedim(0, -1)`` of a ``[Nk, ny, nx]`` host buffer respectively), so the
      same win is available there through a provider-side fetch -- unmeasured, and
      out of scope here;
    * the permuted view is not contiguous for any other reason, in which case
      reaching the k-outer layout would cost a transposing copy (0.525 ms
      measured) that is larger than the transform saving it buys (0.349 ms) --
      a measured refutation, and this guard is what keeps it from applying.
    """
    if getattr(vBF, "_align_shifts", None) is not None:
        return None
    try:
        G = vBF.G
    except (RuntimeError, AttributeError):
        return None
    view = G[..., s:e].permute(2, 0, 1)
    return view if view.is_contiguous() else None


def _direct_ptychography_kouter(
    Gprime_k,
    aberrations,
    ds_rotation,
    semiconvergence_angle,
    wavelength,
    Qy,
    Qx,
    Kx,
    Ky,
    upsample,
    reduce,
    A,
    hoist=None,
):
    """:func:`_direct_ptychography` for a ``[Nk, Nqy, Nqx]`` chunk.

    Identical arithmetic, and every stage of it is BIT-EXACT against the
    ``[Nqy, Nqx, Nk]`` form at the registered chunk shapes (``torch.equal`` on all
    three: the correction is a per-element multiply with permuted subscripts; the
    transform is the same cuFFT batched plan reached from the layout it wants; and
    the chunk reduction does not reassociate because ``ifft2(dim=(0, 1))`` returns
    a *permuted view* whose outer physical axis is already ``Nk``, so
    ``sum(dim=-1)`` on it and ``sum(dim=0)`` here walk memory in the same order --
    0 of 60025 words differ, a one-off probe).

    ``Gprime_k`` is only ever READ (the k-outer correction op is out of place), so
    it may be a view of the caller's G.  ``hoist`` is the chunk loop's per-call
    dict for the half body (see ``_kouter_sum_half``); ``None`` decides
    per chunk.
    """
    factor = np.sqrt(upsample[0] * upsample[1])
    if reduce == "sum":
        # the C2R half-spectrum form of the branch below; see ``_kouter_half``.
        img = _kouter_sum_half(
            Gprime_k, aberrations, ds_rotation, semiconvergence_angle, wavelength,
            Qy, Qx, Kx, Ky, upsample, A, factor, hoist,
        )  # fmt: skip
        if img is not None:
            return img
    # ``Gprime_k`` arrives UN-TILED: when ``upsample_int > 1`` the caller used to
    # hand this body ``torch.tile(Gc_k, (1, uy, ux))``, a 4x replication of a chunk
    # whose only consumer is the correction kernel below, which reads every element
    # exactly once.  The kernel now takes its launch grid from ``out`` and indexes
    # ``G`` modulo ``G``'s own Q extents, which is the DEFINITION of the tile, so
    # allocating the corrected chunk at the upsampled grid is the whole of what the
    # tile was for.  BIT-EXACT: same words read in the same order, same arithmetic.
    corrected = torch.empty(
        (
            int(Gprime_k.shape[0]),
            int(Gprime_k.shape[1]) * int(upsample[0]),
            int(Gprime_k.shape[2]) * int(upsample[1]),
        ),
        dtype=Gprime_k.dtype,
        device=Gprime_k.device,
    )
    correct_aberrations_kouter(
        Gprime_k,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
        A
        if A is not None
        else torch.empty((0, 2), dtype=torch.float32, device=Gprime_k.device),
        corrected,
    )
    if reduce == "sum":
        # The argument about ``factor``, applied to the ONE remaining full-size
        # constant multiply in this body -- cuFFT's C2C plan is unnormalised, so
        # ``norm="ortho"`` is not free inside the transform: torch applies the
        # 1/sqrt(Nqy*Nqx) as its OWN elementwise kernel over the [Nk, H, W] complex
        # output.  Measured on the registered medium chunk (90, 245, 245): 1.015 ms
        # for 8 chunks = 15.0 % of the row's whole device time, for a scalar the
        # linear reduction that follows can carry.  ``norm="forward"`` puts the
        # normalisation on the FORWARD transform, so the inverse hands back the raw
        # cuFFT output and the pass disappears; the constant then rides along on the
        # multiply the reduced [H, W] image was already paying, so this costs no
        # extra dispatch at any shape (1.28x / 1.24x on the medium / small chunk).
        #
        # NOT bit-exact, and deliberately so: ``torch.equal`` says
        # ``ifft2(norm="forward") * 1/sqrt(Nqy*Nqx)`` IS ``ifft2(norm="ortho")``
        # elementwise, so the only difference is scale-then-sum vs sum-then-scale
        # over the chunk axis -- the same float32 reassociation this path's oracle
        # already grades against an INDEPENDENT complex128 reference (measured
        # 1.9e-07 peak-normalised against a 1e-5 gate), and strictly FEWER roundings.
        # That is also why the ``_is_exact_power_of_two`` split is gone from this
        # branch: the folded constant is 1/245 at the registered grid, never a power
        # of two, so there is no exponent-shift case left to preserve.
        G_bf = torch.fft.ifft2(corrected, dim=(-2, -1), norm="forward")
        scale = factor / math.sqrt(corrected.shape[-2] * corrected.shape[-1])
        return torch.sum(G_bf.imag, dim=0) * scale
    G_bf = torch.fft.ifft2(corrected, dim=(-2, -1), norm="ortho")
    if reduce == "none":
        # The yielded SHAPE contract is [Nqy, Nqx, Nk] and the empirical-SSNR
        # consumer's output is digested bit-exact, so hand back a permuted VIEW
        # (free) rather than change the contract.  Its consumer indexes the chunk
        # axis (``imag[..., pm]``), which copies to a dense [Nqy, Nqx, m] tensor of
        # the same values either way, so the half-sums are unchanged.
        return (G_bf * factor).imag.permute(1, 2, 0)
    raise ValueError(f"invalid reduce: {reduce!r}")


def _iter_chunk_images(
    vBF, aberrations, rotation_t, Qy, Qx, upsample_int, n_batches, reduce
):
    """Yield ``(s, e, image)`` per BF-pixel chunk: fetch (mutation-safe, from
    vBF.get_G_chunk) → tile when upsampling → aberration-correct → ifft2 →
    ``reduce`` over the chunk axis. ``rotation_t`` may be a pre-built device tensor
    or a python float; either way the hot loop launches no per-chunk H2D copies,
    because ``_sin_cos_rot`` memoises the conversion — but only a FLOAT reaches its
    value-keyed branch, so a caller that rebuilds the tensor per top-level call is
    strictly worse than one that passes the scalar (see ``direct_ptychography``).

    Upsampled, every chunk image still reads each pixel's scan DC ``G(K, 0)`` on
    the scan-lattice frequencies (the tile, or the k-outer kernels' modulo read):
    a caller that sums the images takes the scan lattice out of the sum
    (``_common.remove_scan_lattice``), as ``direct_ptychography`` does."""
    device = vBF.device
    semiconvergence_angle = vBF.meta.semiconvergence_angle
    wavelength = vBF.meta.wavelength
    # ``vBF.k`` is (Nk, 2), so ``vBF.k[s:e, 1]`` is a COLUMN slice -- stride 2,
    # not dense.  A 1-D ``wp.array`` needs a dense inner stride, so every
    # aberration op ``.contiguous()``-copies BOTH coordinate operands on EVERY
    # call: at the registered medium geometry that is 16 tiny device copies and
    # ~25 us of host time per pass of this loop, and because each copy is a FRESH
    # tensor it also forces a rebind of those two slots in the ops' recorded-
    # command memo.  The ``_contiguous_memo`` those ops call cannot help: it keys
    # on tensor IDENTITY and ``vBF.k[s:e, 1]`` is a new Python object every call,
    # so it misses every time.  Compact the two columns ONCE here instead -- a
    # slice of a contiguous 1-D tensor is itself dense, so the ops' own
    # ``.contiguous()`` then returns it unchanged.  BIT-EXACT: ``.contiguous()``
    # relocates bits, it does not compute, so the kernels read the same words.
    Kx_all = vBF.k[:, 1].contiguous()
    Ky_all = vBF.k[:, 0].contiguous()
    # The aberration kernel's ``A = aperture(K) * exp(-i chi(K))`` factor varies
    # only with the bright-field pixel, so it is constant across every chunk.
    # Built ONCE here for all bright-field pixels and sliced per chunk exactly as
    # ``vBF.k[s:e]`` is; per chunk the build's ~30 us of host time would cost more
    # than the device work it saves.
    A_all = build_aberration_bf_factor(
        aberrations,
        semiconvergence_angle,
        wavelength,
        Kx_all,
        Ky_all,
    )
    # The chunk-sum-first collapse fuses the correction and the ik reduction and
    # is spelled for the [Nqy, Nqx, Nk] layout only, so the k-outer route is not
    # taken when it is enabled -- that variant keeps its shipped body verbatim.
    use_kouter = not (reduce == "sum" and _config._CHUNK_SUM_FIRST)
    # ``n_batches`` is a COUNT; where it buys nothing but host dispatch it is
    # coarsened (never refined) -- the sum-first BF-pixel cap and the k-outer
    # ``reduce="sum"`` byte floor, both in ``_chunk_count``.
    n_batches_eff = _effective_n_batches(
        vBF, n_batches, reduce, use_kouter, upsample_int, Qy, Qx, _g_chunk_kouter_view
    )
    # ONE host sync per reconstruction, not one per chunk, and ONLY on the branch
    # that can use the answer.  ``correct_aberrations_ksum``'s kernel evaluates
    # the twelve-term aberration polynomial and two ``cexp`` per live
    # ``(iqy, iqx, ik)``; when every coefficient is exactly ``+0.0`` both
    # exponentials are the constant ``(1.0, -0.0)`` and a specialised,
    # BIT-IDENTICAL kernel drops them -- 1.4264x on a kernel that is 68.99 % of
    # ``r15_direct_ptychography/medium/direct_ptycho_sumfirst``'s device time.
    # The predicate has to be a HOST bool because it selects a kernel, and the
    # sync it costs is 0.0245 ms on the empty queue this point of the function
    # sees, against ~0.45 ms of device work; putting it inside the op instead
    # would pay 1.0007 ms behind the queued chunk kernels (measured,
    # which is why the ``kouter`` sibling reads its predicate from a device
    # array).  Same shape as the hoist that a former
    # correction sweep did for ``ab_defocus_only`` above its chunk loop.
    # The k-outer branch below is untouched: it never reaches this op, and
    # ``False`` here would be one sync it pays for nothing.
    chi_is_zero = False if use_kouter else chi_is_zero_host(aberrations)
    # per-call state for the k-outer half body (see ``_kouter_sum_half``).
    hoist = {} if use_kouter else None
    for s, e in _chunk_ranges(vBF.n_bright_field, n_batches_eff):
        # G is stored bright-field-OUTERMOST (an fft2 output is a permuted view),
        # so ``get_G_chunk``'s ownership copy is a transposition into the layout
        # the ifft2 then has to transpose back out of.  When the k-outer view is
        # free -- resident G, no alignment overlay -- take it: no copy at all, and
        # the transform runs as the natural batched form (1.66x, bit-exact).
        Gc_k = _g_chunk_kouter_view(vBF, s, e) if use_kouter else None
        if Gc_k is not None:
            # NO ``torch.tile`` HERE ANY MORE.  The upsampling tile was materialised
            # for a single consumer -- ``correct_aberrations_kouter`` -- which reads
            # each of its elements exactly once, so the replication was pure
            # traffic: ``aten::copy_ (1,2,2,33,512,512)`` x31 = 17.356 ms = 9.05 %
            # of ``r14_ffstem/gd2o3/fused``.  The op now takes its
            # launch grid from its ``out`` and reads ``G`` modulo ``G``'s own Q
            # extents, which is what ``torch.tile`` means, so the un-tiled chunk and
            # an upsampled ``out`` give the same words with none of the copy.
            # ``upsample_int`` reaches the callee already, so nothing new is passed.
            yield (
                s,
                e,
                _direct_ptychography_kouter(
                    Gc_k,
                    aberrations,
                    rotation_t,
                    semiconvergence_angle,
                    wavelength,
                    Qy,
                    Qx,
                    Kx_all[s:e],
                    Ky_all[s:e],
                    upsample_int,
                    reduce,
                    A_all[s:e],
                    hoist,
                ),
            )
            continue
        # ``get_G_chunk``'s copy exists for OWNERSHIP -- the default body's
        # ``correct_aberrations_inplace`` mutates the chunk.  The chunk-sum-first
        # collapse does NOT: its only consumer is ``correct_aberrations_ksum``,
        # which is out of place and never touches its input, exactly as
        # ``correct_aberrations_kouter`` is on the branch above.  So take the same
        # read-only view that branch takes and pay no copy: G is bright-field-
        # OUTERMOST in memory, so the copy this skips is a TRANSPOSE, 1.116 ms of
        # a 3.40 ms row at the registered medium geometry.  ``_g_chunk_kouter_view``
        # already refuses (returns None) for an alignment overlay, a provider with
        # no resident G, or a slice whose permuted form is not dense -- the three
        # cases where the view is not free -- and this falls back to the owned
        # chunk there.  Permuting it back to [Nqy, Nqx, Nk] is a relabelling of
        # the same storage, so what reaches the kernel is ``G[..., s:e]`` itself.
        Gc_view = None if use_kouter else _g_chunk_kouter_view(vBF, s, e)
        Gc = vBF.get_G_chunk(s, e) if Gc_view is None else Gc_view.permute(1, 2, 0)
        if (upsample_int > 1).any():
            Gc = torch.tile(Gc, (int(upsample_int[0]), int(upsample_int[1]), 1))
        yield (
            s,
            e,
            _direct_ptychography(
                Gc,
                aberrations,
                rotation_t,
                semiconvergence_angle,
                wavelength,
                Qy,
                Qx,
                Kx_all[s:e],
                Ky_all[s:e],
                device,
                upsample_int,
                reduce=reduce,
                A=A_all[s:e],
                chi_is_zero=chi_is_zero,
            ),
        )


def _direct_ptychography(
    Gprime: torch.Tensor,
    aberrations: torch.Tensor,
    ds_rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    Qy: torch.Tensor,
    Qx: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    device: torch.device,
    upsample: np.ndarray,
    reduce: str = "sum",
    A: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    chi_is_zero: bool = False,
) -> torch.Tensor:
    factor = np.sqrt(upsample[0] * upsample[1])
    if reduce == "sum" and _config._CHUNK_SUM_FIRST:
        # One transform per chunk instead of one per bright-field pixel, and the
        # factor lands on the [H, W] real result rather than an [H, W, chunk]
        # complex one. Reassociates the float32 chunk sum: opt-in, see
        # enable_chunk_sum_first.
        #
        # The aberration correction and the ik reduction are FUSED here, so the
        # (Nqy, Nqx, Nk) corrected chunk is never materialised: on the registered
        # 245**2 x 128**2 reconstruction that deletes a 42.1 MiB write followed
        # immediately by a 42.1 MiB read of the same block, per chunk.  This is the
        # fused op for exactly this pair -- ``determine_aberrations`` has used it
        # since; this loop never got it.  ``out`` is deliberately unused on this
        # branch: the fused op does not mutate ``Gprime``, which is the only reason
        # a caller needed a scratch buffer at all.
        Gprime_summed = correct_aberrations_ksum(
            Gprime,
            aberrations,
            ds_rotation,
            semiconvergence_angle,
            wavelength,
            Qx,
            Qy,
            Kx,
            Ky,
            None,
            A,
            chi_is_zero,
        )
        return torch.fft.ifft2(Gprime_summed, dim=(0, 1), norm="ortho").imag * factor
    # correct_aberrations_inplace accepts a float OR a pre-built device tensor;
    # hot loops pass a tensor so no per-chunk H2D copy is launched.  ``A`` is the
    # bright-field-only aberration factor, hoisted above the chunk loop by
    # _iter_chunk_images; None means the op builds its own.  ``out`` is a scratch
    # buffer for callers that must keep ``Gprime`` intact: with it the correction
    # is out of place and they need no clone.
    Gprime_corrected = correct_aberrations_inplace(
        Gprime,
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
        A=A,
        out=out,
    )
    if reduce == "sum":
        # The same normalisation fold as _direct_ptychography_kouter, applied here
        # too so the two layouts stay numerically interchangeable: the streaming /
        # alignment-overlay callers reach THIS body while the resident-G ones reach
        # the k-outer twin, and a test compares the
        # two at atol=1e-7.  Folding one and not the other would put a 1.9e-07
        # difference straight into that comparison.
        G_bf = torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="forward")
        scale = factor / math.sqrt(
            Gprime_corrected.shape[0] * Gprime_corrected.shape[1]
        )
        return torch.sum(G_bf.imag, dim=(-1)) * scale
    # reduce="none" keeps ``norm="ortho"``: its consumer (the empirical-SSNR
    # checkerboard half-split) is digested BIT-EXACT, so nothing about a
    # normalisation fold may reach it.
    G_bf = torch.fft.ifft2(Gprime_corrected, dim=(0, 1), norm="ortho")
    if reduce == "none":
        return (G_bf * factor).imag
    raise ValueError(f"invalid reduce: {reduce!r}")
