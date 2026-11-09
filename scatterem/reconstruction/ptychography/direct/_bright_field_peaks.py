"""Cross-correlation, peak choice and sub-pixel step for the BF shift estimators.

Registration by upsampled cross-correlation (Guizar-Sicairos, Thurman and
Fienup 2008), with an added peak-choice rule for periodic specimens.

Fourier convention: ``torch.fft.fft2``, kernel ``exp(−2πi k·r)``, zero
frequency at index 0, ``k`` in cycles per scan pixel. For a moving spectrum
``I`` and a fixed spectrum ``R`` the correlation is
``c(u) = Σ_k I(k) R*(k) exp(−2πi k·u)``. If the moving image is the fixed
image translated by ``t``, the peak of ``|c|`` is at ``u = −t``: the
translation that registers the moving image onto the fixed one.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

__all__ = ["condition_spectra", "lowpass_weight", "match_to_template"]

#: Largest number of candidate peaks refined per image.
_CANDIDATE_CAP = 8
#: Candidates must reach this fraction of the coarse maximum.
_CANDIDATE_FRACTION = 0.5
#: Half width of the refinement window, in scan pixels.
_WINDOW_HALF_PX = 0.75


def lowpass_weight(
    scan_grid_shape: tuple[int, int], lowpass_sigma_px: float, device
) -> Tensor:
    """The Gaussian low-pass ``W(k) = exp(−2π² σ² |k|²)``, float64 ``(n0, n1)``.

    ``σ`` is the real-space standard deviation of the blur in scan pixels;
    ``k`` is in cycles per scan pixel, in ``fft2`` order.
    """
    n0, n1 = scan_grid_shape
    k0 = torch.fft.fftfreq(n0, dtype=torch.float64, device=device)[:, None]
    k1 = torch.fft.fftfreq(n1, dtype=torch.float64, device=device)[None, :]
    return torch.exp(-2.0 * math.pi**2 * lowpass_sigma_px**2 * (k0 * k0 + k1 * k1))


def condition_spectra(image_spectra: Tensor, lowpass_sigma_px: float | None) -> Tensor:
    """Remove the mean and apply the optional Gaussian low-pass.

    The DC bin is set to zero: the mean intensity only adds a constant to the
    correlation. The low-pass is :func:`lowpass_weight`.
    """
    out = image_spectra.clone()
    out[..., 0, 0] = 0
    if lowpass_sigma_px is not None:
        n0, n1 = out.shape[-2:]
        w = lowpass_weight((n0, n1), lowpass_sigma_px, out.device)
        out = out * w.to(out.real.dtype)
    return out


def _pad_axis(x: Tensor, axis: int) -> Tensor:
    """Zero-pad one axis of a corner-origin spectrum from ``n`` to ``2n``.

    An even-size Nyquist bin is split equally between ``+n/2`` and ``−n/2``.
    """
    n = x.shape[axis]
    h = (n + 1) // 2
    shape = list(x.shape)
    shape[axis] = 2 * n
    out = x.new_zeros(shape)
    out.narrow(axis, 0, h).copy_(x.narrow(axis, 0, h))
    if n - h > 0:
        out.narrow(axis, 2 * n - (n - h), n - h).copy_(x.narrow(axis, h, n - h))
    if n % 2 == 0:
        half = x.narrow(axis, n // 2, 1) * 0.5
        out.narrow(axis, 3 * n // 2, 1).copy_(half)
        out.narrow(axis, n // 2, 1).copy_(half)
    return out


def _dft_matrix(u: Tensor, n: int, dtype: torch.dtype) -> Tensor:
    """Rows ``exp(−2πi m u / n)`` for positions ``u`` (float64) and all ``m``.

    The Nyquist column of an even ``n`` holds ``cos(π u)``: the split bin of
    :func:`_pad_axis`, so window values agree with the coarse grid.
    """
    m = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64, device=u.device)
    arg = torch.remainder(u[..., None] * m / n, 1.0)
    e = torch.exp(-2j * math.pi * arg)
    if n % 2 == 0:
        e[..., n // 2] = torch.cos(math.pi * u).to(e.dtype)
    return e.to(dtype)


def _crest_mask(mag: Tensor) -> Tensor:
    """Periodic 8-neighbour local maxima (``>=`` every neighbour)."""
    keep = torch.ones_like(mag, dtype=torch.bool)
    for d0 in (-1, 0, 1):
        for d1 in (-1, 0, 1):
            if d0 or d1:
                keep &= mag >= torch.roll(mag, shifts=(d0, d1), dims=(-2, -1))
    return keep


def _parabola(cm: Tensor, c0: Tensor, cp: Tensor) -> Tensor:
    """Vertex offset of the parabola through three samples, in grid steps.

    Used only where the curvature is negative; clipped to ``[−½, ½]``.
    """
    den = cm - 2.0 * c0 + cp
    ok = den < 0
    delta = (cm - cp) / (2.0 * torch.where(ok, den, -1.0))
    return torch.where(ok, delta.clamp(-0.5, 0.5), 0.0)


def _candidates(mag: Tensor, step: float, n0: int, n1: int, tol: float):
    """Indices ``(B, K)`` of the candidate peaks and their validity.

    Candidates are periodic local maxima of at least half the coarse maximum.
    Those whose coarse height can still pass the final tolerance (with a 10 %
    margin for the coarse sampling loss) come first, nearest to zero shift
    first; the rest follow by height. Reason: on a lattice there are more
    equal peaks than the cap of 8, and the peak nearest to zero must be kept.
    """
    b = mag.shape[0]
    g1 = mag.shape[-1]
    flat = mag.reshape(b, -1)
    maxv = flat.amax(dim=1, keepdim=True)
    mask = _crest_mask(mag).reshape(b, -1)
    mask &= (flat >= _CANDIDATE_FRACTION * maxv) & (maxv > 0)
    score = torch.where(mask, flat, torch.full_like(flat, -1.0))
    _, idx = torch.sort(score, dim=1, descending=True, stable=True)
    pos = torch.arange(flat.shape[1], device=mag.device, dtype=torch.float64)
    u0 = torch.div(pos, g1, rounding_mode="floor") * step
    u1 = torch.remainder(pos, g1) * step
    w0 = u0 - n0 * torch.ceil(u0 / n0 - 0.5)
    w1 = u1 - n1 * torch.ceil(u1 / n1 - 0.5)
    dist = (w0 * w0 + w1 * w1).expand(b, -1)
    near = mask & (flat >= 0.9 * (1.0 - tol) * maxv)
    big = torch.full_like(dist, 1e300)
    key = torch.where(
        near, dist, torch.where(mask, big, torch.full_like(dist, float("inf")))
    )
    key = torch.gather(key, 1, idx)
    _, sub = torch.sort(key, dim=1, stable=True)
    idx = torch.gather(idx, 1, sub)
    k = min(_CANDIDATE_CAP, flat.shape[1])
    idx = idx[:, :k]
    return idx, torch.gather(score, 1, idx) >= 0


def _coarse(p: Tensor, kappa: int):
    """Correlation magnitude on the coarse grid and its step in pixels."""
    if kappa == 1:
        return torch.fft.fft2(p).abs(), 1.0
    return torch.fft.fft2(_pad_axis(_pad_axis(p, -2), -1)).abs(), 0.5


def _refine_on_grid(mag: Tensor, idx: Tensor, step: float):
    """Heights and positions of candidates on the coarse grid (``κ <= 2``)."""
    g0, g1 = mag.shape[-2:]
    i0, i1 = idx // g1, idx % g1
    bi = torch.arange(mag.shape[0], device=mag.device)[:, None]

    def at(a, b):
        return mag[bi, a % g0, b % g1].to(torch.float64)

    c = at(i0, i1)
    d0 = _parabola(at(i0 - 1, i1), c, at(i0 + 1, i1))
    d1 = _parabola(at(i0, i1 - 1), c, at(i0, i1 + 1))
    pos = torch.stack([(i0 + d0) * step, (i1 + d1) * step], dim=-1)
    return c, pos


def _refine_window(p: Tensor, centres: Tensor, kappa: int):
    """Heights and positions after a matrix-DFT on a window around each centre.

    The window DFT runs in complex128. Reason: near the top of a peak,
    neighbouring samples differ by about 1e-4 of the height; float32 rounding
    there moved the parabolic step by up to 6e-4 px between CPU and CUDA.
    """
    n0, n1 = p.shape[-2:]
    r = math.ceil(_WINDOW_HALF_PX * kappa)
    offs = (torch.arange(2 * r + 1, dtype=torch.float64, device=p.device) - r) / kappa
    u0 = centres[..., 0, None] + offs
    u1 = centres[..., 1, None] + offs
    e0 = _dft_matrix(u0, n0, torch.complex128)
    e1 = _dft_matrix(u1, n1, torch.complex128)
    # One bmm over the K·w rows of each image: a broadcast ``e0 @ p[:, None]``
    # made K complex128 copies of ``p`` (2.6 GiB extra for 64 x 512 x 512).
    b, k = e0.shape[:2]
    rows = torch.bmm(e0.reshape(b, -1, n0), p.to(torch.complex128))
    c = rows.reshape(b, k, -1, n1) @ e1.transpose(-1, -2)
    mag = c.abs().to(torch.float64)
    w = 2 * r + 1
    flat = mag.reshape(*mag.shape[:2], -1)
    best = flat.argmax(dim=-1)
    j0, j1 = best // w, best % w
    height = torch.gather(flat, -1, best[..., None])[..., 0]

    def at(a, b):
        ok = (a >= 0) & (a < w) & (b >= 0) & (b < w)
        v = torch.gather(
            flat, -1, (a.clamp(0, w - 1) * w + b.clamp(0, w - 1))[..., None]
        )
        return v[..., 0], ok

    cm0, ok_m0 = at(j0 - 1, j1)
    cp0, ok_p0 = at(j0 + 1, j1)
    cm1, ok_m1 = at(j0, j1 - 1)
    cp1, ok_p1 = at(j0, j1 + 1)
    d0 = torch.where(ok_m0 & ok_p0, _parabola(cm0, height, cp0), 0.0)
    d1 = torch.where(ok_m1 & ok_p1, _parabola(cm1, height, cp1), 0.0)
    pos = torch.stack(
        [
            centres[..., 0] + (j0 - r + d0) / kappa,
            centres[..., 1] + (j1 - r + d1) / kappa,
        ],
        dim=-1,
    )
    return height, pos


def _wrap(u: np.ndarray, n: int) -> np.ndarray:
    """Wrap positions into ``(−n/2, n/2]``."""
    return u - n * np.ceil(u / n - 0.5)


def match_to_template(
    shifted_spectra: Tensor,
    anchor_spectrum: Tensor,
    *,
    peak_grid_divisor: int,
    peak_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Register each moving spectrum onto the fixed spectrum.

    Steps: (1) correlation on a coarse grid of 0.5 px by zero-padding to
    ``(2 n0, 2 n1)``; (2) candidates: periodic local maxima of at least half
    the coarse maximum, at most 8; near-top peaks nearest to zero first, then
    the rest by height; (3) matrix-multiply DFT on
    a ``±0.75 px`` window with step ``1/κ`` around each candidate; for
    ``κ = 1`` the grid is the integer grid and for ``κ = 2`` the coarse grid
    is final; (4) a 1-D parabola per axis through the three samples around
    the maximum, clipped to half a grid step; (5) among candidates within
    ``peak_tolerance`` of the highest refined peak, the one with the smallest
    ``|u|`` (wrapped into ``(−n/2, n/2]``) wins; ties: higher peak, then
    lower candidate index. On a scan-periodic specimen the lattice copies of
    the true peak have equal height; rule 5 takes the copy nearest to zero.

    Parameters
    ----------
    shifted_spectra : Tensor
        Complex ``(B, n0, n1)`` pre-processed spectra.
    anchor_spectrum : Tensor
        Complex ``(n0, n1)`` or ``(B, n0, n1)`` pre-processed spectra.
    peak_grid_divisor : int
        ``κ >= 1``; the final grid step is ``1/κ`` px.
    peak_tolerance : float
        In ``[0, 1)``.

    Returns
    -------
    shifts : numpy.ndarray
        float64 ``(B, 2)``, in scan pixels.
    empty : numpy.ndarray
        bool ``(B,)``: True where the correlation is zero everywhere; the
        shift is then 0.
    """
    n0, n1 = shifted_spectra.shape[-2:]
    p = shifted_spectra * anchor_spectrum.conj()
    mag, step = _coarse(p, peak_grid_divisor)
    idx, valid = _candidates(mag, step, n0, n1, peak_tolerance)
    if peak_grid_divisor <= 2:
        height, pos = _refine_on_grid(mag, idx, step)
    else:
        g1 = mag.shape[-1]
        centres = torch.stack([(idx // g1) * step, (idx % g1) * step], dim=-1)
        height, pos = _refine_window(p, centres.to(torch.float64), peak_grid_divisor)
    height = height.cpu().numpy()
    pos = pos.cpu().numpy()
    valid = valid.cpu().numpy()
    b = shifted_spectra.shape[0]
    shifts = np.zeros((b, 2))
    empty = np.zeros(b, dtype=bool)
    for i in range(b):
        ks = np.flatnonzero(valid[i])
        if ks.size == 0:
            empty[i] = True
            continue
        hmax = height[i, ks].max()
        if not hmax > 0:
            empty[i] = True
            continue
        best = None
        for k in ks:
            if height[i, k] < (1.0 - peak_tolerance) * hmax:
                continue
            w = np.array([_wrap(pos[i, k, 0], n0), _wrap(pos[i, k, 1], n1)])
            key = (float(w @ w), -float(height[i, k]), int(k))
            if best is None or key < best[0]:
                best = (key, w)
        if best is None:  # unreachable: the highest peak always qualifies
            empty[i] = True
            continue
        shifts[i] = best[1]
    return shifts, empty
