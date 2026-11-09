"""Tunable module-level knobs for direct ptychography.

Every name here is read *through this module* by its callers
(``_config._CHUNK_SUM_FIRST``), never imported by value. ``_CHUNK_SUM_FIRST``
is rebound at runtime by :func:`enable_chunk_sum_first`, and the memory budgets
are monkeypatched by the test suite; a ``from ._config import X`` at the call
site would bind the value once at import time and silently ignore both.
"""

import torch

# radial power of k for each of the 12 cartesian aberration coefficients
_ABERRATION_RADIAL_POWER = torch.tensor(
    [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4], dtype=torch.float32
)


# Retained-activation budget for the TV-autofocus chunk loop in
# :func:`determine_aberrations` (see ``use_ckpt`` there).  Gradient-checkpointing
# each chunk buys peak memory by paying a SECOND forward pass inside the
# backward, so it is only worth doing when the memory it saves actually matters.
# What it saves is exactly ``scatterem::correct_aberrations_fwd``'s saved input:
# ``n_k_tv * new_shape[0] * new_shape[1] * 8`` bytes summed over the chunks, i.e.
# one upsampled copy of ``vBF.G``.  Below this budget the checkpoint is skipped
# and the fit runs one forward per objective evaluation instead of two.
#
# The switch is numerically INERT -- the recompute is a deterministic
# re-execution of the same ops on the same inputs, so the aberration gradient is
# identical either way (gated in a one-off probe against the adjoint's own
# ``wp.tile_atomic_add`` reproducibility floor, R9d).  Only peak memory and
# runtime move, which is why a budget is a safe thing to condition on.
#
# 4 GiB keeps the case the loop's docstring was written for on the checkpointed
# path: a 1024^2 scan needs ~9 GB here (17 GB with the ifft2 output the sum-first
# collapse has since deleted), and reintroducing that is the OOM the chunking
# exists to prevent.  The published ``Fig1_Gd2O3`` fit config needs 2150 MiB.
_TV_CHUNK_CKPT_BUDGET_BYTES = 4 * 1024**3

# Budget for the TV-autofocus chunk CACHE (see ``_g_chunks`` in
# :func:`determine_aberrations`).  ``vBF.G`` is fixed for the whole fit, but
# ``G[..., s:e]`` is a non-contiguous last-axis slice, so ``get_G_chunk`` compacts
# the same bytes on every chunk body -- ~1250-1700 times per fit at
# ``n_batches=25``.  Fetching each distinct chunk once instead costs one extra
# copy of G, ``n_k_tv * ny * nx * 8`` bytes, which is the SAME quantity the
# checkpoint budget above is written in; hence the same default, so the two
# decisions cannot disagree about what "a copy of G is affordable" means.
#
# Like the checkpoint switch this is numerically INERT: a cached chunk is the
# tensor ``get_G_chunk`` returns, byte for byte (verified by ``torch.equal`` in
# a one-off probe), and its one consumer in the loop,
# ``correct_aberrations``, is out-of-place (measured).  So only
# peak memory and runtime move, which is what makes a budget safe to condition on.
_TV_CHUNK_CACHE_BUDGET_BYTES = 4 * 1024**3


# Largest transient the TV chunk loop is allowed to hold, in bytes -- i.e. the
# quantity ``n_batches`` was always trying to name.  ``n_batches`` is a COUNT,
# which is not a memory unit: the same 25 means a 15.5 GiB working set per chunk
# at the 1024^2 scan the loop's docstring was written for and 5.4 MB per chunk
# at the published ``Fig2_carbon`` fit (96x96 scan, 1845 BF pixels).  Below this
# budget the split is buying memory nobody needed and paying a FIXED host issue
# per chunk body: measured on one serial ``evaluate_tv`` at ``Fig2_carbon``,
# 5.822 ms at 25 chunks against 2.683 ms at 10 and 2.607 ms at 5, flat below 10,
# with the Warp kernel's DEVICE time unchanged (2.230 vs 2.188 ms) and the row
# 55 % GPU-IDLE at 25 chunks against 96 % device-busy at 5.  The per-chunk cost
# is host dispatch -- ~90 us of self-CPU per ``correct_aberrations_ksum`` call,
# independent of chunk width, because the fused kernel reduces over ``ik`` in a
# register and returns ``[Nqy, Nqx]`` whatever the chunk holds.
#
# Used ONLY to coarsen, never to refine: ``min(n_batches, ...)`` below, so a
# caller who asks for a finer split than this budget requires still gets it, and
# the 1024^2 case is untouched (15.5 GiB / 16 MiB = 927 chunks, so ``min`` picks
# the requested 25).  The peak it can add is bounded BY this constant, which is
# the point of stating the budget in bytes.
#
# NOT numerically inert, unlike the two budgets above -- the loop sums per-chunk
# partial images, so the chunk count reassociates a float32 sum.  Measured at
# ``Fig2_carbon``: the objective is BIT-IDENTICAL at 1, 3, 5, 15 and 25 chunks
# and one float32 ulp away at 10, and the FIT'S OWN RETURN moves 4.768e-06
# (10 chunks) / 9.298e-06 (5 chunks) against a same-code same-process repeat
# floor of 4.768e-07 - 9.537e-07 and against the ``gate_point_is_the_fit``
# tolerance of 1e-3, i.e. two orders of magnitude of headroom, with C1
# bit-identical to the pinned gate vector.  These numbers were measured
# with the earlier surrogate aberration adjoint.  The exact
# adjoint moved the fit, and the gate vector in
# the benchmark targets were re-pinned to the new fit; the chunk-count
# sensitivity has not been re-measured since.
_TV_CHUNK_MIN_BYTES = 16 * 1024**2


# Minimum number of BF pixels per half-split side for the checkerboard SSNR
# estimate to be meaningful.  Below this threshold we fall back to analytical.
_MIN_HALFSPLIT_PIXELS = 4


# Opt-in fast path for the bright-field chunk reduction in ``_direct_ptychography``.
# OFF by default: the collapse is algebraically exact but reassociates the float32
# chunk sum, so it does not reproduce the default path bit for bit.  See
# :func:`enable_chunk_sum_first`.
_CHUNK_SUM_FIRST = False


# Cap on bright-field pixels reduced per chunk on the chunk-sum-first branch of
# ``_iter_chunk_images``, where the chunk count has no memory role at all (see the
# block inside that function for the three conditions and the measurements).  The
# unit is BF pixels and not bytes on purpose: nothing is allocated per chunk
# there, so what bounds the coarsening is the length of the float32 accumulation
# chain ``correct_aberrations_ksum`` runs in a register, not a working set.
_SUMFIRST_CHUNK_MAX_BF = 1024


# Per-chunk BYTE floor for the DEFAULT k-outer ``reduce="sum"`` route of
# ``_iter_chunk_images``: a split whose complex64
# corrected chunks would be smaller than this is coarsened, never refined.  NOT
# numerically inert -- the chunk count reassociates a float32 sum (<= 2.3e-07
# peak-normalised at ``r14_ffstem/carbon``).  See ``_chunk_count``.
_KOUTER_SUM_CHUNK_MIN_BYTES = 16 * 1024**2


def enable_chunk_sum_first(enabled: bool = True) -> bool:
    """Enable the chunk-axis-first reduction in the direct-ptychography loop.

    ``_direct_ptychography`` inverse-transforms a whole bright-field chunk and then
    sums the chunk axis away.  ``ifft2`` is linear and ``Im()`` is R-linear, so

        sum_k Im(ifft2(G_k)) == Im(ifft2(sum_k G_k))

    and reducing the chunk axis *before* the transform replaces one
    two-dimensional transform per bright-field pixel with a single transform per
    chunk.  It also applies the ``sqrt(upsample)`` factor to the ``[H, W]`` real
    result instead of to an ``[H, W, chunk]`` complex tensor whose real half is
    discarded.

    Measured **2.26x** end to end (13.652 -> 6.034 ms, range 2.23-2.29x over
    separate process launches) on a 245**2 scan x 128**2 detector reconstruction,
    717 bright-field pixels, ``n_batches=8``, RTX A6000 -- with peak CUDA memory
    41 MiB *lower*, since the ``[H, W, chunk]`` complex intermediates are never
    materialised.  The win scales with bright-field pixels per chunk, so it is
    **no win at all** on a 62**2 scan (0.997x, ~8 px per chunk and launch-bound)
    and grows with scan size and with small ``n_batches``.

    Default OFF, deliberately.  The identity is exact in exact arithmetic (checked
    to 4.6e-16 of peak in float64 at the live chunk shape), moving the real factor
    across ``Im()`` is bit-exact, and against a complex128 reference the collapsed
    form is *closer* to the truth than the default (2.637e-07 vs 2.716e-07).  The
    two float32 images agree to 3.2e-07 of the image peak -- less than either
    one's own distance from that reference.  But summing before rather than after
    the transform does reassociate the float32 sum, so it is not bit-exact.
    The figure scripts under ``experiments/ff_stem/`` run the default path,
    and the release gate pins the output of ``Fig2_carbon.py``. Flipping the
    default is a decision for a person, not for this switch.

    ``reduce="none"`` (the empirical-SSNR checkerboard half-split) has no chunk sum
    to collapse and is unaffected either way.

    Parameters
    ----------
    enabled : bool, optional
        Whether to take the collapsed path. Default is True.

    Returns
    -------
    bool
        The previous setting, so a caller can restore it.
    """
    global _CHUNK_SUM_FIRST
    previous = _CHUNK_SUM_FIRST
    _CHUNK_SUM_FIRST = bool(enabled)
    return previous
