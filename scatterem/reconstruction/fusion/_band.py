"""SSNR band-composite fusion of two images -- the published FF-STEM fusion."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _robust_minmax(im: torch.Tensor, clip_quantile: float, assume_finite: bool = False):
    """Robust [0,1]-style normalization: returns (normalized, offset, scale) where
    normalized = (im - offset) / scale. Uses low/high quantiles instead of min/max so
    a single hot/dead pixel cannot set the scale. clip_quantile <= 0 reproduces
    min/max exactly (backward-compatible).

    ``assume_finite`` is a caller's promise that ``im`` contains no NaN, so the
    CUDA sort-select can be taken without first asking the device (see the
    comment at the branch below -- that question is a BLOCKING host sync).  It
    is an assertion about the input, never a change of numerics: on a NaN-free
    input both branches return the same triple element for element."""
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
    #
    # That trade inverts on a SMALL image: at the 245^2 registered shape the sort
    # it guards is ~0.05 ms, while ``bool(...)`` is a BLOCKING host sync whose
    # price is set by what is queued behind it, not by the 60k-element reduction
    # -- measured 0.0406 ms x2 per fusion, 8.7 % of
    # ``r15_direct_ptychography/medium/fused``.  A caller that has already made
    # its input finite (``fused_full_field`` ``nan_to_num``s all four arrays
    # immediately before handing them over) knows the answer without asking the
    # device, so it can say so with ``assume_finite=True`` and skip the sync.
    if flat.is_cuda and (assume_finite or not bool(torch.isnan(flat).any())):
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
    assume_finite: bool = False,
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
        assume_finite: bool, caller's promise that ``im1``/``im2`` contain no NaN.
            Skips the two blocking host syncs the robust normalization otherwise
            pays to decide whether the CUDA sort-select is safe. Numerics are
            unchanged on a NaN-free input; leave it False unless the caller has
            just sanitized the images (``fused_full_field`` has).
    Returns:
        fused: torch.Tensor, fused image
        ptycho_filter: torch.Tensor or None, filtered first image (None unless
            ``return_filtered=True``)
        tcdf_filter: torch.Tensor or None, filtered second image (None unless
            ``return_filtered=True``)
    """
    if verbosity > 0:
        logger.info(
            "Max weight1: %.4f, min weight1: %.4f",
            weight1.max().item(),
            weight1.min().item(),
        )
        logger.info(
            "Max weight2: %.4f, min weight2: %.4f",
            weight2.max().item(),
            weight2.min().item(),
        )
    im1, _, im1_scale = _robust_minmax(im1.clone(), clip_quantile, assume_finite)
    im2, _, _ = _robust_minmax(im2.clone(), clip_quantile, assume_finite)

    if not return_filtered and im1.is_cuda:
        # Real images and real weights: the same image from the half spectrum, with
        # the weights symmetrised so it holds for ANY real weight.  Gates (size,
        # dtype, layout) and the numbers are in the module; None means "declined".
        # Kill switch: SCATTEREM_FUSE_HALF_C2R=0.
        try:
            from scatterem.reconstruction.fusion._band_triton import fuse_half
        except ImportError:
            fuse_half = None
        if fuse_half is not None:
            im_fused = fuse_half(im1, im2, weight1, weight2)
            if im_fused is not None:
                im_fused *= im1_scale
                return im_fused, None, None

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
