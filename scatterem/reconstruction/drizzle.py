"""Drizzle (area-overlap) sub-pixel shift + upsample for shift-and-sum imaging.

The tilt-corrected dark field (and any dithered shift-and-sum reconstruction)
registers a stack of sub-pixel-shifted frames onto a common, optionally
up-sampled, output grid. The default path does this in Fourier space (zero-pad
upsample + phase-ramp shift). That is exact for dense, band-limited signals, but
at **very low dose** the frames are sparse — a handful of single electron counts
— and Fourier shifting turns every isolated count into a sinc: ~20% negative
side-lobes and energy delocalised across the whole frame (classic Gibbs
ringing).

Drizzle avoids this. Each input pixel is treated as a small square "drop"
(``pixfrac`` of an input pixel) that is dropped onto the output grid at its
shifted position; its flux is distributed to the output pixels it overlaps, by
overlap *area*. Two grids are accumulated: the flux ``accum`` and the summed
weights ``hits``; the normalised image ``accum / hits`` is a coverage-weighted
average. Because every weight is non-negative, the result is non-negative for
non-negative input — no ringing, no negative counts — and flux is conserved
exactly. This is the classic Fruchter & Hook (2002) drizzle, specialised to
pure translations on a regular grid.

The resampler is forward-only (``@torch.no_grad``): tilt-corrected dark field is
a direct, non-iterative reconstruction, so no gradient flows through it. It is
device/dtype-portable (plain ``index_add_``); no Warp/CUDA requirement.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(
    sigma: float, radius: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    x = torch.arange(2 * radius + 1, dtype=dtype, device=device) - radius
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def _blur2d_sep(
    t: torch.Tensor, sigma: float, pad_mode: str = "reflect"
) -> torch.Tensor:
    """Separable Gaussian blur of an ``(N, C, H, W)`` tensor (Nadaraya-Watson).

    The per-axis kernel radius is clamped to ``dim - 1`` so ``reflect`` padding
    never exceeds the (possibly small) grid — a truncated Gaussian is used on
    tiny grids rather than raising."""
    if sigma <= 0:
        return t
    C, H, W = t.shape[1], t.shape[-2], t.shape[-1]
    base_r = int(math.ceil(3.0 * float(sigma)))

    if W > 1:
        rw = max(1, min(base_r, W - 1))
        gx = _gaussian_kernel1d(sigma, rw, t.dtype, t.device).view(1, 1, 1, -1)
        t = F.pad(t, (rw, rw, 0, 0), mode=pad_mode)
        t = F.conv2d(t, gx.expand(C, 1, 1, -1), groups=C)
    if H > 1:
        rh = max(1, min(base_r, H - 1))
        gy = _gaussian_kernel1d(sigma, rh, t.dtype, t.device).view(1, 1, -1, 1)
        t = F.pad(t, (0, 0, rh, rh), mode=pad_mode)
        t = F.conv2d(t, gy.expand(C, 1, -1, 1), groups=C)
    return t


@torch.no_grad()
def drizzle_resample(
    images: torch.Tensor,
    shifts: torch.Tensor,
    upsample: int,
    *,
    pixfrac: float = 1.0,
    kde_sigma: float = 0.0,
    eps: float = 1e-12,
    return_parts: bool = False,
):
    """Drizzle a stack of shifted frames onto a common up-sampled grid.

    Args:
        images: ``(N, H, W)`` real, the ``N`` dithered frames of the same scene
            (e.g. dark-field azimuthal-segment images). Non-negative input gives
            non-negative output.
        shifts: ``(N, 2)`` per-frame ``(dy, dx)`` shift in **output (HR) pixels**.
            Frame ``n``'s input pixel ``(i, j)`` is deposited at HR position
            ``(i*U + dy_n, j*U + dx_n)`` — i.e. input pixel ``i`` maps to HR pixel
            ``i*U`` (matching Fourier zero-pad upsampling), plus the sub-pixel shift.
        upsample: integer output magnification ``U`` (output is ``H*U`` x ``W*U``).
        pixfrac: drizzle drop size as a fraction of one input pixel, ``0 < pixfrac <= 1``.
            The deposited footprint is a box of side ``pixfrac*U`` HR pixels.
            Smaller ``pixfrac`` gives sharper results but needs denser coverage to
            avoid holes; ``1.0`` is the safe default.
        kde_sigma: if ``> 0``, Nadaraya-Watson smoothing — a Gaussian of this
            std (HR px) is applied to numerator *and* denominator before dividing.
            Fills small holes and denoises at the cost of resolution.
        eps: denominator floor for uncovered output pixels (they read 0, not NaN).
        return_parts: also return the raw ``(accum, hits)`` grids.

    Returns:
        ``(H*U, W*U)`` hit-normalised HR image; or ``(image, accum, hits)`` if
        ``return_parts``. ``accum`` is the flux grid (conserves ``images.sum()``
        for in-frame content); ``hits`` is the summed coverage weight.
    """
    if images.dim() != 3:
        raise ValueError(f"images must be (N, H, W); got shape {tuple(images.shape)}")
    if shifts.shape != (images.shape[0], 2):
        raise ValueError(
            f"shifts must be (N, 2) matching images N={images.shape[0]}; got {tuple(shifts.shape)}"
        )
    U = int(upsample)
    if U < 1:
        raise ValueError(f"upsample must be a positive integer; got {upsample}")
    if not (0.0 < pixfrac <= 1.0):
        raise ValueError(f"pixfrac must be in (0, 1]; got {pixfrac}")

    N, H, W = images.shape
    device = images.device
    out_dtype = images.dtype if images.is_floating_point() else torch.float32
    # Accumulate in >= float32: many index_add_ into a fp16/bf16 grid loses
    # precision; the normalised result is cast back to the input dtype at the end.
    dtype = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
    images = images.to(dtype)
    shifts = shifts.to(dtype=dtype, device=device)

    Hh, Wh = H * U, W * U

    # Input-pixel centres mapped to HR coordinates: LR (i, j) -> HR (i*U, j*U).
    iy = torch.arange(H, device=device, dtype=dtype)[None, :, None]  # (1,H,1)
    ix = torch.arange(W, device=device, dtype=dtype)[None, None, :]  # (1,1,W)
    cy = iy * U + shifts[:, 0].view(N, 1, 1)  # (N,H,W) HR row centre of each drop
    cx = ix * U + shifts[:, 1].view(N, 1, 1)  # (N,H,W) HR col centre of each drop

    # Half-width of the drop footprint in HR pixels; drop area = (2*hw)^2.
    hw = max(pixfrac * U / 2.0, 1e-6)
    inv2hw = 1.0 / (2.0 * hw)  # normalise so total weight per fully-in-frame drop = 1

    # Nearest HR pixel to each drop centre; a symmetric window covers the footprint.
    ncy = torch.round(cy)
    ncx = torch.round(cx)
    radius = int(math.ceil(hw + 0.5)) + 1

    vals = images.reshape(N * H * W)  # (P,)
    accum = torch.zeros(Hh * Wh, device=device, dtype=dtype)
    hits = torch.zeros(Hh * Wh, device=device, dtype=dtype)

    for oy in range(-radius, radius + 1):
        ty = ncy + oy  # (N,H,W) candidate HR row (float)
        # 1-D overlap of drop [cy-hw, cy+hw] with output pixel [ty-0.5, ty+0.5]
        oy_len = (
            torch.minimum(ty + 0.5, cy + hw) - torch.maximum(ty - 0.5, cy - hw)
        ).clamp_min(0.0)
        wy = oy_len * inv2hw
        ty_l = ty.long()
        valid_y = (ty_l >= 0) & (ty_l < Hh) & (wy > 0)
        if not bool(valid_y.any()):
            continue
        for ox in range(-radius, radius + 1):
            tx = ncx + ox
            ox_len = (
                torch.minimum(tx + 0.5, cx + hw) - torch.maximum(tx - 0.5, cx - hw)
            ).clamp_min(0.0)
            wx = ox_len * inv2hw
            tx_l = tx.long()
            valid = valid_y & (tx_l >= 0) & (tx_l < Wh) & (wx > 0)
            if not bool(valid.any()):
                continue
            w = (wy * wx).reshape(-1)
            flat = (ty_l * Wh + tx_l).reshape(-1)
            # zero-weight the out-of-bounds entries and clamp their index to a valid slot
            vmask = valid.reshape(-1)
            w = torch.where(vmask, w, torch.zeros_like(w))
            flat = torch.where(vmask, flat, torch.zeros_like(flat))
            accum.index_add_(0, flat, vals * w)
            hits.index_add_(0, flat, w)

    accum = accum.reshape(1, 1, Hh, Wh)
    hits = hits.reshape(1, 1, Hh, Wh)
    if kde_sigma and kde_sigma > 0:
        accum = _blur2d_sep(accum, kde_sigma)
        hits = _blur2d_sep(hits, kde_sigma)

    out = (accum / hits.clamp_min(eps)).reshape(Hh, Wh).to(out_dtype)
    if return_parts:
        return (
            out,
            accum.reshape(Hh, Wh).to(out_dtype),
            hits.reshape(Hh, Wh).to(out_dtype),
        )
    return out
