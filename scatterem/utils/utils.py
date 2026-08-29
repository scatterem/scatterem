"""Module for various convenient utilities."""

from __future__ import annotations

import os
from importlib.util import find_spec

import torch


if find_spec("cv2") is not None:
    pass





def _robust_minmax(im: torch.Tensor, clip_quantile: float):
    """Robust [0,1]-style normalization: returns (normalized, offset, scale) where
    normalized = (im - offset) / scale. Uses low/high quantiles instead of min/max so
    a single hot/dead pixel cannot set the scale. clip_quantile <= 0 reproduces
    min/max exactly (backward-compatible)."""
    if clip_quantile <= 0.0:
        offset = im.min()
        scale = (im.max() - offset).clamp_min(1e-12)
        return (im - offset) / scale, offset, scale
    flat = im.flatten().float()

    def _rank(n: int, quantile: float) -> int:
        """Exact quantile by rank selection, at any input size.

        ``torch.quantile`` caps how many elements it accepts. The usual
        workaround subsamples, which was what this did -- but with an UNSEEDED
        draw, so the returned quantiles, and therefore the displayed image,
        differed between runs of identical code. Measured on a 2048x2048 input:
        0.00504738 then 0.00488388 for the same array. Every image past 1M pixels
        was affected, which includes any 1024^2 reconstruction.

        Selecting the k-th smallest value instead has no size cap, is exact rather
        than estimated, and is deterministic.
        """
        return min(max(int(round(quantile * (n - 1))) + 1, 1), n)

    n = flat.numel()
    k_lo = _rank(n, clip_quantile)
    k_hi = _rank(n, 1.0 - clip_quantile)
    # Take BOTH order statistics out of one sort on CUDA.  ``Tensor.kthvalue``
    # reduces with ONE CUDA BLOCK per reduced slice, so a single flattened image
    # is a one-block radix select over the whole array and its cost is set by
    # that block, not by the card: measured on an A6000, the two selections cost
    # 2.33 ms at 512^2, 9.50 ms at 1024^2 and 45.70 ms at 2048^2, against 0.09 /
    # 0.24 / 0.80 ms for one stable sort (25x / 39x / 57x).  Sorting is the
    # oversized hammer for two order statistics, but it is the parallel one, and
    # it amortises across both of them.  CPU keeps ``kthvalue``: there it is an
    # O(n) introselect and a sort would be strictly worse.
    #
    # A sort-select returns the SAME ELEMENT as ``kthvalue`` -- verified over 48
    # randomised trials, sides 3 to 1024, including heavy ties, +-inf and a
    # constant image, all identical on the whole returned triple -- EXCEPT when
    # the input contains NaN, where the two disagree about where a sign-bit-set
    # NaN ranks.  So NaN falls back to the original path rather than being argued
    # about; the check is one reduction plus one host sync (~0.15 ms) against the
    # 9.50 ms it guards.
    if flat.is_cuda and not bool(torch.isnan(flat).any()):
        ordered = torch.sort(flat, stable=True).values
        offset = ordered[k_lo - 1]
        hi = ordered[k_hi - 1]
    else:
        offset = flat.kthvalue(k_lo).values
        hi = flat.kthvalue(k_hi).values
    scale = (hi - offset).clamp_min(1e-12)
    normalized = ((im - offset) / scale).clamp(0.0, 1.0)
    return normalized, offset, scale


def fuse_images_fourier_weighted(
    im1: torch.Tensor,
    im2: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    verbosity: int = 0,
    clip_quantile: float = 0.005,
    return_filtered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Fuse two images by fourier filtering im2 with weight2 and adding it to im1 with weight1.

    Each input is robustly normalized to [0, 1] using quantile-based clipping (controlled
    by ``clip_quantile``) so that a single hot or dead pixel cannot set the normalization
    scale. For clean data the quantiles ≈ min/max, so results are virtually unchanged
    relative to the previous min/max behavior. Set ``clip_quantile=0`` to reproduce the
    exact legacy min/max normalization.

    The fused image is a normalized band-composite: low-frequency / DC content comes
    exclusively from ``im2`` (the dark-field channel) because ptychography has no DC
    transfer; ``im1`` (ptychographic phase) contributes only at higher spatial frequencies.
    The output is restored to the scale of ``im1`` so its absolute values are meaningful.

    Args:
        im1: torch.Tensor, first image (ptychographic reconstruction)
        im2: torch.Tensor, second image (dark-field / TCDF reconstruction)
        weight1: torch.Tensor, Fourier-domain weight for im1 (high-frequency band)
        weight2: torch.Tensor, Fourier-domain weight for im2 (low-frequency band)
        verbosity: int, if > 0 print weight statistics
        clip_quantile: float, quantile used for robust normalization (default 0.005);
            set to 0 for legacy min/max behavior.
        return_filtered: bool, if True compute and return the per-channel filtered
            images (two extra inverse FFTs); if False (default) those slots are None,
            saving peak memory in production callers that discard them.
    Returns:
        fused: torch.Tensor, fused image
        ptycho_filter: torch.Tensor or None, filtered first image (None unless
            ``return_filtered=True``)
        tcdf_filter: torch.Tensor or None, filtered second image (None unless
            ``return_filtered=True``)
    """
    if verbosity > 0:
        print(
            f"Max weight1: {weight1.max().item():.4f}, "
            f"min weight1: {weight1.min().item():.4f}"
        )
        print(
            f"Max weight2: {weight2.max().item():.4f}, "
            f"min weight2: {weight2.min().item():.4f}"
        )
    im1, _, im1_scale = _robust_minmax(im1.clone(), clip_quantile)
    im2, _, _ = _robust_minmax(im2.clone(), clip_quantile)

    im1_fft = torch.fft.fft2(im1, dim=(0, 1), norm="ortho")
    im2_fft = torch.fft.fft2(im2, dim=(0, 1), norm="ortho")
    im1_fft *= weight1  # in-place: reuse the forward-transform buffers
    im2_fft *= weight2
    im_fused_fft = im1_fft + im2_fft
    im_fused = torch.fft.ifft2(im_fused_fft, dim=(0, 1), norm="ortho").real
    im_fused *= im1_scale

    if return_filtered:
        ptycho_filter = torch.fft.ifft2(im1_fft, dim=(0, 1), norm="ortho").real
        tcdf_filter = torch.fft.ifft2(im2_fft, dim=(0, 1), norm="ortho").real
        return im_fused, ptycho_filter, tcdf_filter
    return im_fused, None, None




































#: Threads used for the parallel phases of :func:`fit_circle_ransac_levels`.  A
#: pure tuning constant -- the result is bit-identical at every value, so unlike a
#: format parameter this one may move freely.  ``n`` counts the CALLER too, so the
#: pool holds ``n - 1``; nt=2 is the one value to avoid, because it leaves a single
#: pool worker doing 99 of the 100 tasks while the caller waits on one.
#:
#: It is still NOT sized from ``os.cpu_count()``: four of the six ops in the fit
#: HOLD the GIL, and D187 measured the whole surface falling to 0.63x once the pool
#: follows the core count.  It is capped at 8 instead, with the box's own core
#: count as the only thing that can lower it, so no machine gets more threads than
#: it has CPUs.
#:
#: The ceiling was 4 while this constant sized ONE phase (scoring).  It now sizes
#: TWO -- the fused Warp geometry pass takes it as well -- and 32.6 ms of a 57.6 ms
#: call is behind it, so the value was re-swept in situ on ``d13_preprocess/medium``
#: (``benchmarks/lab/_probe_d192_nt.py``, 9 reps after a per-arm warm-up,
#: interleaved, median / min ms of ``_circle_method``): nt=1 101.0/93.1, nt=2
#: 99.4/95.2, nt=4 71.5/63.7, nt=5 60.6/54.7, nt=6 59.3/53.1, **nt=8 53.8/48.2**,
#: nt=10 55.4/51.8, nt=12 60.5/54.5, nt=16 65.9/61.0, nt=20 73.5/66.8.  8 is the
#: optimum on BOTH statistics (1.329x / 1.321x over nt=4), 5-12 is one shelf, and
#: 20 = ``cpu_count()`` on this box is still worse than 4.  D188's reason for
#: staying at 4 -- that nt=6's samples were bimodal -- no longer reproduces: the
#: nt=8 samples are a clean unimodal 48.2-60.5 spread.  The answer's independence
#: from this value is pinned by
#: ``tests/test_ransac_levels.py::test_thread_count_does_not_change_the_answer``.
try:
    _RANSAC_SCORE_THREADS = min(8, max(1, len(os.sched_getaffinity(0))))
except AttributeError:  # not linux
    _RANSAC_SCORE_THREADS = min(8, max(1, os.cpu_count() or 1))





























