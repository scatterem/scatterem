"""Module for various convenient utilities."""

from __future__ import annotations

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

    def _nearest_rank(values: torch.Tensor, quantile: float) -> torch.Tensor:
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
        n = values.numel()
        k = min(max(int(round(quantile * (n - 1))) + 1, 1), n)
        return values.kthvalue(k).values

    offset = _nearest_rank(flat, clip_quantile)
    hi = _nearest_rank(flat, 1.0 - clip_quantile)
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




































