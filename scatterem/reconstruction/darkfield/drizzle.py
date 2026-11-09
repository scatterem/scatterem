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
device/dtype-portable (plain ``index_add_``); on CUDA the float32 splat runs as one
Warp gather kernel instead (``_drizzle_fused``; ``SCATTEREM_DRIZZLE_FUSED=0``
restores the ``index_add_`` splat).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from scatterem.reconstruction._backends import DRIZZLE
from scatterem.utils.env import resolve_backend

#: Bytes one batched ``(oy, ox)`` offset pair costs per input element, used to
#: size :data:`_DRIZZLE_PAIR_BATCH_BYTES` into a pair count.
#:
#: Counted, not fitted: the two operands the ``index_add_`` pair consumes are
#: ``w`` (float32, 4 B) and ``flat`` (int64, 8 B), and the broadcast
#: ``(P,N,H,1)`` / ``(P,N,1,W)`` tables they are built from are smaller than
#: either by a factor of ``min(H, W)``, so 12 B/element is the whole of it to
#: within a few percent.
_DRIZZLE_PAIR_ENTRY_BYTES = 12

#: Byte budget for the batched offset-pair windows of :func:`drizzle_resample`.
#:
#: A RESOURCE gate rather than a pair count, for the usual reason: a count stops
#: being the decision it was measured as the moment the caller's ``N`` changes,
#: and ``N`` here is whatever the caller passes.
#:
#: 256 MiB is chosen to be ABOVE every shape the tcDF caller produces and below
#: the point where the pair axis is the largest object in the process.  Measured
#: (``N = 8``, the ``frame_chunk`` that caller fixes): 9 live pairs at ``U = 2``
#: and 25 at ``U = 4``, i.e. 14 MiB at ``128^2``, 54 MiB at ``256^2`` and 9 MiB
#: at ``64^2, U = 4`` -- all one batch, so that path is byte for byte AND
#: dispatch for dispatch what it was.  What the budget bounds is the other
#: caller: :func:`drizzle_resample` is public, and at ``N = 200, 256^2`` those
#: same 9 pairs are 1.4 GiB, which this caps at 157 MiB.
#:
#: The result does not depend on the value.  Each pair's ``index_add_`` runs on
#: its own operands in the same order at every batch size, so ``PB = 1`` and
#: ``PB = P`` return identical bits;
#: the budget buys memory and costs only the dispatches of an extra round.
_DRIZZLE_PAIR_BATCH_BYTES = 256 * 1024**2


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


# Box (pixfrac) splat kept local (public-release module); the library's linear
# insert is a different kernel.
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

    Runs under ``torch.no_grad``: this is a resampling step, not part of an
    autograd path. Accumulation is forced to at least float32 -- many
    ``index_add_`` into a half-precision grid loses precision -- and the
    normalized result is cast back to the input dtype.

    Parameters
    ----------
    images : Tensor
        Shape ``(N, H, W)``, real. The ``N`` dithered frames of the same scene,
        e.g. dark-field azimuthal-segment images. Non-negative input gives
        non-negative output.
    shifts : Tensor
        Shape ``(N, 2)``, the per-frame ``(dy, dx)`` shift in *output* (HR)
        pixels. Frame ``n``'s input pixel ``(i, j)`` is deposited at HR position
        ``(i * U + dy_n, j * U + dx_n)``: input pixel ``i`` maps to HR pixel
        ``i * U``, matching Fourier zero-pad upsampling, plus the sub-pixel
        shift. Moved to the device and accumulation dtype of ``images``.
    upsample : int
        Integer output magnification ``U``; the output is ``H * U`` by
        ``W * U``.
    pixfrac : float
        Drizzle drop size as a fraction of one input pixel, in ``(0, 1]``. The
        deposited footprint is a box of side ``pixfrac * U`` HR pixels. A
        smaller value is sharper but needs denser coverage to avoid holes.
        Default 1.0.
    kde_sigma : float
        When positive, Nadaraya-Watson smoothing: a Gaussian of this standard
        deviation in HR pixels is applied to numerator *and* denominator before
        dividing. Fills small holes and denoises, at the cost of resolution.
        Default 0.0.
    eps : float
        Denominator floor, so uncovered output pixels read 0 rather than NaN.
        Default 1e-12.
    return_parts : bool
        Also return the raw accumulation grids.
        Default False.

    Returns
    -------
    Tensor or tuple of Tensor
        The hit-normalised ``(H * U, W * U)`` image, in the dtype and on the
        device of ``images``. With ``return_parts=True``, the triple
        ``(image, accum, hits)``: ``accum`` is the flux grid, which conserves
        ``images.sum()`` for in-frame content, and ``hits`` is the summed
        coverage weight.

    Raises
    ------
    ValueError
        If ``images`` is not 3-dimensional, if ``shifts`` is not ``(N, 2)``
        matching it, if ``upsample`` is below 1, or if ``pixfrac`` is outside
        ``(0, 1]``.
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

    if images.is_cuda:
        # The same terms, gathered per output pixel in one launch instead of
        # scattered per window offset -- see ``_drizzle_fused`` for what is and is
        # not identical.
        from ._drizzle_fused import drizzle_splat_fused

        if resolve_backend(DRIZZLE, None, images, shifts) == "warp":
            accum, hits = drizzle_splat_fused(images, shifts, U, hw, inv2hw, radius)
            return _normalise_parts(
                accum, hits, Hh, Wh, kde_sigma, eps, out_dtype, return_parts
            )

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

    def _axis_tables(n_in: int, n_out: int, shift1: torch.Tensor, nax: int = 0):
        """``(weight, index, in-bounds, live)`` for every offset on one axis.

        ``nax`` batches the axes themselves: with ``nax = 0`` ``shift1`` is
        ``(N,)`` and the tables come back ``(N, n_in, 2*radius+1)``; with
        ``nax = 2`` it is ``(2, N)`` and they come back with a leading axis of
        2, which is legal only when the two axes agree on ``(n_in, n_out)``.
        Every operation below is ELEMENTWISE except the final ``any``, which is
        a boolean OR over the same logical axis in both spellings, so the
        batched tables are bit-for-bit the per-axis ones.
        """
        lead = (N, 1) if nax == 0 else (nax, N, 1)
        c = torch.arange(n_in, device=device, dtype=dtype) * U + shift1.reshape(lead)
        t = torch.round(c)[..., None] + offs  # (..., N, n_in, 2*radius+1)
        c = c[..., None]
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
        return w, t_l, inb, (w > 0).any(dim=-2)  # live: (..., N, 2*radius+1)

    if H == W:
        # Then Hh == Wh too, so the y and x tables are the same computation on
        # two different shift columns -- one call instead of two, which on a
        # host-dispatch-bound path is ~24 fewer ATen dispatches per call.
        _w, _t, _i, _l = _axis_tables(H, Hh, shifts.transpose(0, 1), nax=2)
        wy_t, ty_t, inby_t, live_y = _w[0], _t[0], _i[0], _l[0]
        wx_t, tx_t, inbx_t, live_x = _w[1], _t[1], _i[1], _l[1]
    else:
        wy_t, ty_t, inby_t, live_y = _axis_tables(H, Hh, shifts[:, 0])
        wx_t, tx_t, inbx_t, live_x = _axis_tables(W, Wh, shifts[:, 1])
    live = (live_y[:, :, None] & live_x[:, None, :]).any(dim=0).tolist()  # the 1 sync
    ty_t = ty_t * Wh
    # offset-major on the x axis, so one slice yields the (N,1,W) operand
    wx_t, tx_t, inbx_t = (t.transpose(1, 2) for t in (wx_t, tx_t, inbx_t))

    # The (oy, ox) offset pairs that can carry weight, in the SAME oy-major,
    # ox-minor order the nested loop visited them.  Their windows are BUILT in
    # one batch over a leading pair axis -- the body was ~14 ATen dispatches per
    # pair, of which only the two `index_add_` do any real work, so on a
    # host-dispatch-bound path the construction is most of the cost.  The
    # `index_add_` calls themselves stay one per pair, in the same order, on
    # bit-identical operands, so each output slot's contributions are reduced in
    # exactly the groups the shipped loop left them in.
    pairs = [
        (oy, ox)
        for oy in range(2 * radius + 1)
        for ox in range(2 * radius + 1)
        if live[oy][ox]
    ]
    # The pair axis is built under a BYTE BUDGET, not all at once.  One batched
    # pair costs `N*H*W` entries of `w` (4 B) and of `flat` (8 B) plus the two
    # broadcast operands they are built from, and `N` is the caller's -- the tcDF
    # path chunks at 8 frames, but `drizzle_resample` is public and a direct
    # caller with a large `N` would otherwise turn the shipped `(N,H,W)`
    # transient into `(P,N,H,W)` with `P` up to `(2*radius+1)**2 = 49`.  The
    # budget bounds that product instead: `PB` pairs are materialised at a time,
    # the `index_add_` calls still run one per pair in the same order on the same
    # operands, so the result is byte-identical at every `PB` -- including
    # `PB = 1`, which is the shipped loop's own footprint.
    per_pair = max(1, N * H * W) * _DRIZZLE_PAIR_ENTRY_BYTES
    PB = max(1, _DRIZZLE_PAIR_BATCH_BYTES // per_pair)
    for lo in range(0, len(pairs), PB):
        chunk = pairs[lo : lo + PB]
        P = len(chunk)
        oy_i = torch.tensor([p[0] for p in chunk], device=device)
        ox_i = torch.tensor([p[1] for p in chunk], device=device)
        # (P,N,H,1) and (P,N,1,W): each slice `[p]` is the (N,H,1)/(N,1,W)
        # operand the loop sliced out for that pair.
        wy = wy_t.index_select(2, oy_i).permute(2, 0, 1).unsqueeze(-1)
        row = ty_t.index_select(2, oy_i).permute(2, 0, 1).unsqueeze(-1)
        inb_y = inby_t.index_select(2, oy_i).permute(2, 0, 1).unsqueeze(-1)
        wx = wx_t.index_select(1, ox_i).permute(1, 0, 2).unsqueeze(2)
        col = tx_t.index_select(1, ox_i).permute(1, 0, 2).unsqueeze(2)
        inb_x = inbx_t.index_select(1, ox_i).permute(1, 0, 2).unsqueeze(2)

        w = (wy * wx).reshape(P, -1)
        flat = (row + col).reshape(P, -1)
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
        flat = torch.where((inb_y & inb_x).reshape(P, -1), flat, 0)
        for p in range(P):
            flat_p, w_p = flat[p], w[p]
            accum.index_add_(0, flat_p, vals * w_p)
            hits.index_add_(0, flat_p, w_p)

    return _normalise_parts(
        accum, hits, Hh, Wh, kde_sigma, eps, out_dtype, return_parts
    )


def _normalise_parts(accum, hits, Hh, Wh, kde_sigma, eps, out_dtype, return_parts):
    """The shared tail of both splats: optional Nadaraya-Watson blur, then
    ``accum / hits``."""
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


__all__ = [
    "drizzle_resample",
]
