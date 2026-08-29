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

    # Half-width of the drop footprint in HR pixels; drop area = (2*hw)^2.
    hw = max(pixfrac * U / 2.0, 1e-6)
    inv2hw = 1.0 / (2.0 * hw)  # normalise so total weight per fully-in-frame drop = 1
    radius = int(math.ceil(hw + 0.5)) + 1

    vals = images.reshape(N * H * W)  # (P,)
    accum = torch.zeros(Hh * Wh, device=device, dtype=dtype)
    hits = torch.zeros(Hh * Wh, device=device, dtype=dtype)

    # Everything that varies with the window offset is tabulated ONCE, per axis,
    # for every offset -- instead of being rebuilt inside the (2*radius+1)^2 loop.
    #
    # Input-pixel centres map to HR coordinates as LR (i, j) -> HR (i*U, j*U), so
    # the drop centre is `cy = i*U + dy_n`: the row weight and row bounds depend
    # on (n, i) only, and the column ones on (n, j) only. Each axis therefore
    # tabulates in (N, n_in, 2*radius+1), and the expressions below are the ones
    # the loop used to evaluate per offset -- same operands, same order -- so
    # every weight, index and mask is bit-for-bit what it computed.
    #
    # Two things this buys. (1) The loop was HOST-DISPATCH bound, not bandwidth
    # bound: its per-offset operands are (N,H,1)/(N,1,W) -- a few thousand
    # elements -- so each of the ~25 torch ops per iteration cost ~10 us of
    # dispatch against ~2 us of device time. The body is now two views, two
    # elementwise ops, one `where` and the two `index_add_`. (2) Which offsets
    # can carry any weight at all is decided ONCE on the host:
    #   any(valid) = any_n( any_i valid_y[n,i] and any_j valid_x[n,j] )
    # reproduces the old per-iteration `bool(valid.any())` exactly, but as a
    # single D2H transfer taken before any `index_add_` is enqueued, rather than
    # one sync per iteration each stalling the host until the splat drained.
    zero = torch.zeros((), device=device, dtype=dtype)
    offs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)

    def _axis_tables(n_in: int, n_out: int, shift1: torch.Tensor):
        """``(weight, index, in-bounds, live)`` for every offset on one axis."""
        c = torch.arange(n_in, device=device, dtype=dtype) * U + shift1.view(N, 1)
        t = torch.round(c)[:, :, None] + offs  # (N, n_in, 2*radius+1)
        c = c[:, :, None]
        # 1-D overlap of drop [c-hw, c+hw] with output pixel [t-0.5, t+0.5]
        seg = (
            torch.minimum(t + 0.5, c + hw) - torch.maximum(t - 0.5, c - hw)
        ).clamp_min(0.0)
        t_l = t.long()
        inb = (t_l >= 0) & (t_l < n_out)
        # Zeroing the weight on the AXIS is exactly what the old full-size
        # `torch.where(inb_y & inb_x, wy*wx, 0)` did: both factors are
        # non-negative, so `0 * x` is `+0.0` for either factor being zeroed.
        w = torch.where(inb, seg * inv2hw, zero)
        return w, t_l, inb, (w > 0).any(dim=1)  # live: (N, 2*radius+1)

    wy_t, ty_t, inby_t, live_y = _axis_tables(H, Hh, shifts[:, 0])
    wx_t, tx_t, inbx_t, live_x = _axis_tables(W, Wh, shifts[:, 1])
    live = (live_y[:, :, None] & live_x[:, None, :]).any(dim=0).tolist()  # the 1 sync
    ty_t = ty_t * Wh
    # offset-major on the x axis, so one slice yields the (N,1,W) operand
    wx_t, tx_t, inbx_t = (t.transpose(1, 2) for t in (wx_t, tx_t, inbx_t))

    for oy in range(2 * radius + 1):
        if not any(live[oy]):
            continue
        wy = wy_t[:, :, oy : oy + 1]  # (N,H,1)
        row = ty_t[:, :, oy : oy + 1]  # (N,H,1) HR row index * Wh
        inb_y = inby_t[:, :, oy : oy + 1]
        for ox in range(2 * radius + 1):
            if not live[oy][ox]:
                continue
            w = (wy * wx_t[:, ox : ox + 1, :]).reshape(-1)
            flat = (row + tx_t[:, ox : ox + 1, :]).reshape(-1)
            # Park only the genuinely OUT-OF-BOUNDS entries at slot 0 (their
            # weight is already exactly +0.0, zeroed on the axis above). Entries
            # that are in bounds but carry zero weight keep their own index:
            # `seg` is `clamp_min(0.0)`, so `w` is already exactly +0.0 for them,
            # and adding +-0.0 to a grid that starts at +0.0 and only ever
            # receives non-negative weights leaves every bit unchanged
            # (x + 0.0 == x, and +0.0 + -0.0 == +0.0). Clamping them to slot 0
            # instead would serialise up to ~10^6 `atomicAdd(0.0)` onto a single
            # address -- at U=1 the zero-weight entries are the majority of all
            # entries, and out-of-bounds is only the border.
            #
            # The out-of-bounds entries must keep going to slot 0 rather than to
            # a clamped border slot: both add exactly +-0.0, but a deterministic
            # `index_add_` reduces a slot's contributions as a TREE, so moving
            # ~10^5 zero terms out of slot 0's segment re-pairs the nonzero ones
            # and shifts that one pixel by 1 ULP (measured 5.4e-08 relative).
            inb = (inb_y & inbx_t[:, ox : ox + 1, :]).reshape(-1)
            flat = torch.where(inb, flat, 0)
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
