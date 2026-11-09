"""Choosing the right cross-correlation peak for periodic bright-field images.

A bright-field image of a crystal is periodic, so its cross-correlation with a
shifted copy has a peak at the true shift ``s`` and one at every ``s + n*a``
(``a`` a lattice vector), all of nearly the same height. The global maximum is
then an arbitrary one of them, and once the parallax at the rim of the bright
field exceeds half a lattice period the measured shifts jump by lattice vectors
from pixel to pixel and the aberration fit returns nonsense.

``peak_selection="guided"`` uses the one thing the lattice cannot fake: the
shift field of a bright-field series is smooth in the detector coordinate (it is
the gradient of the aberration function). Pixels are registered in order of
increasing radius, where the parallax is small and the nearest peak is the true
one; each later pixel's shift is predicted from those already measured
(:class:`ShiftModel`) and the highest local maximum within ``radius`` of the
prediction is taken (:func:`nearest_peak`).

The radius is ``RADIUS_FRACTION`` of the lattice period ``p`` estimated from the
reference image's autocorrelation (:func:`lattice_period`). Every other copy of
the true peak lies at least ``p`` from it, so with a prediction error ``e`` the
true peak is inside the disc and every copy outside it whenever
``e < RADIUS_FRACTION * p`` and ``e < (1 - RADIUS_FRACTION) * p``; 0.4 balances
the tolerated prediction error against a margin for broad peaks. A non-periodic
image has no secondary peak, ``p`` is infinite, and the choice is the global
maximum -- the same as ``peak_selection="argmax"``.

The correlation is circular, so a peak is reported at its periodic image nearest
the prediction: a shift beyond half the field of view is recoverable when the
prediction says so, which the global maximum alone never is.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from scatterem.utils.subpixel import (
    _PAIRWISE_CHUNK_ELEMENTS,
    _peak_offsets,
    relative_shifts,
)

PEAK_SELECTIONS = ("argmax", "guided")

#: Search radius as a fraction of the lattice period; see the module docstring.
RADIUS_FRACTION = 0.4

#: A secondary autocorrelation maximum at least this fraction of the central one
#: marks the image as periodic. A perfect commensurate lattice gives ~1; the
#: random-specimen fixtures in the tests stay well below.
PERIODIC_THRESHOLD = 0.5

#: Points at which :class:`ShiftModel` switches from an affine to a cubic fit
#: (10 coefficients per component, so this leaves it well over-determined).
CUBIC_MIN_POINTS = 30


def validate_peak_selection(peak_selection: str) -> None:
    """Raise ``ValueError`` unless ``peak_selection`` is one of ``PEAK_SELECTIONS``."""
    if peak_selection not in PEAK_SELECTIONS:
        raise ValueError(
            f"peak_selection must be one of {PEAK_SELECTIONS}, got {peak_selection!r}"
        )


def _signed_grid(n: int, device) -> torch.Tensor:
    """Shift, in pixels, of each FFT bin: bins past the halfway point are negative."""
    idx = torch.arange(n, device=device)
    return torch.where(idx > n // 2, idx - n, idx)


def _local_maxima(values: torch.Tensor) -> torch.Tensor:
    """Mask of points not exceeded by any of their 8 circular neighbours, ``(N, H, W)``."""
    padded = F.pad(values[:, None], (1, 1, 1, 1), mode="circular")
    pooled = F.max_pool2d(padded, kernel_size=3, stride=1)[:, 0]
    return values >= pooled


def lattice_period(spectrum: torch.Tensor) -> float:
    """Length of the shortest lattice vector of an image, in pixels.

    Parameters
    ----------
    spectrum : torch.Tensor
        ``fft2`` of the image, ``(H, W)``.

    Returns
    -------
    float
        Distance from the origin to the nearest local maximum of the image's
        (circular, mean-free) autocorrelation that reaches
        ``PERIODIC_THRESHOLD`` of the central value; ``inf`` if there is none,
        i.e. the image is not periodic within its field of view.
    """
    power = spectrum.abs() ** 2
    power[0, 0] = 0.0  # mean-free: a constant background is not a lattice
    auto = torch.fft.ifft2(power).real
    if not float(auto[0, 0]) > 0.0:
        return math.inf
    auto = auto / auto[0, 0]
    height, width = auto.shape
    rows = _signed_grid(height, auto.device).to(auto.dtype)[:, None]
    cols = _signed_grid(width, auto.device).to(auto.dtype)[None, :]
    dist = torch.sqrt(rows**2 + cols**2)
    candidate = _local_maxima(auto[None])[0] & (auto >= PERIODIC_THRESHOLD) & (dist > 0)
    if not bool(candidate.any()):
        return math.inf
    return float(dist[candidate].min())


def nearest_peak(
    coarse: torch.Tensor, prediction: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer peak of each correlation near its predicted position.

    Parameters
    ----------
    coarse : torch.Tensor
        Correlation magnitudes on the integer grid, ``(N, H, W)``, in FFT bin
        order.
    prediction : torch.Tensor
        Predicted shift of each correlation, ``(N, 2)`` pixels ``(row, col)``.
    radius : float
        Search radius in pixels; ``inf`` searches the whole grid.

    Returns
    -------
    tuple of torch.Tensor
        ``(row, col)`` long tensors ``(N,)``: the highest local maximum within
        ``radius`` of the prediction (the highest value in the disc when it holds
        no local maximum), reported at the periodic image of that bin nearest
        the prediction.
    """
    nbatch, height, width = coarse.shape
    device = coarse.device
    pred = prediction.to(device=device, dtype=torch.float64)
    rows = _signed_grid(height, device).to(torch.float64)
    cols = _signed_grid(width, device).to(torch.float64)
    # minimum-image offset of every bin from the prediction
    d_row = torch.remainder(rows[None] - pred[:, :1] + height / 2, height) - height / 2
    d_col = torch.remainder(cols[None] - pred[:, 1:] + width / 2, width) - width / 2
    disc = d_row[:, :, None] ** 2 + d_col[:, None, :] ** 2 <= radius**2
    if radius < 1.0:  # always keep the bin nearest the prediction
        disc |= (d_row.abs() <= 0.5)[:, :, None] & (d_col.abs() <= 0.5)[:, None, :]
    lowest = torch.finfo(coarse.dtype).min
    in_disc = torch.where(disc, coarse, lowest)
    peaks = torch.where(disc & _local_maxima(coarse), coarse, lowest)
    has_peak = (disc & _local_maxima(coarse)).reshape(nbatch, -1).any(dim=1)
    flat = torch.where(
        has_peak,
        peaks.reshape(nbatch, -1).argmax(dim=1),
        in_disc.reshape(nbatch, -1).argmax(dim=1),
    )
    items = torch.arange(nbatch, device=device)
    i_row = torch.div(flat, width, rounding_mode="floor")
    i_col = flat % width
    row = pred[:, 0] + d_row[items, i_row]
    col = pred[:, 1] + d_col[items, i_col]
    return torch.round(row).long(), torch.round(col).long()


def guided_shifts(
    reference_spectrum: torch.Tensor,
    spectra: torch.Tensor,
    upsample_factor: float,
    prediction: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """:func:`~scatterem.utils.subpixel.relative_shifts`, peak chosen near a prediction.

    Parameters
    ----------
    reference_spectrum : torch.Tensor
        ``fft2`` of the reference image, ``(H, W)``.
    spectra : torch.Tensor
        ``fft2`` of the images to register, ``(N, H, W)``.
    upsample_factor : float
        Samples per original pixel in the refinement grid.
    prediction : torch.Tensor
        Predicted shift of each image, ``(N, 2)`` pixels ``(row, col)``.
    radius : float
        Search radius around the prediction, pixels.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` shifts in pixels, ``(row, col)``, in the real dtype of
        ``spectra``, with the sign convention of ``relative_shifts``.
    """
    cross_power = reference_spectrum.unsqueeze(0) * spectra.conj()
    peak = _chunked_nearest_peak(cross_power, prediction, radius)
    return _peak_offsets(cross_power, upsample_factor, integer_peak=peak).to(
        spectra.real.dtype
    )


def _chunked_nearest_peak(
    cross_power: torch.Tensor, prediction: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`nearest_peak` of each cross-power spectrum, in chunks of images.

    The pick is made per image, so the chunks change no result. They hold the
    correlation magnitudes, the float64 search disc and the local-maximum
    masks to ``_PAIRWISE_CHUNK_ELEMENTS`` pixels at a time. Unchunked, these
    took 37 bytes per scan pixel per image, against 28 for ``"argmax"``
    (762 x 758 scan, CUDA, 2026-10-09); chunked, a batch of 47 images peaks
    at 26 bytes, below ``"argmax"``.
    """
    nbatch, height, width = cross_power.shape
    chunk = max(1, _PAIRWISE_CHUNK_ELEMENTS // (height * width))
    rows, cols = [], []
    for start in range(0, nbatch, chunk):
        stop = min(nbatch, start + chunk)
        coarse = torch.fft.ifft2(cross_power[start:stop]).abs()
        row, col = nearest_peak(coarse, prediction[start:stop], radius)
        rows.append(row)
        cols.append(col)
    return torch.cat(rows), torch.cat(cols)


def guided_pairwise_shifts(
    G_stack: torch.Tensor, pairs: torch.Tensor, upsample_factor: float, radius: float
) -> torch.Tensor:
    """``pairwise_relative_shifts`` with each pair's peak taken within ``radius`` of 0.

    Neighbouring bright-field pixels differ by one detector pixel, so their
    relative parallax is small and the peak nearest zero is the true one.

    Parameters
    ----------
    G_stack : torch.Tensor
        ``(N, H, W)`` Fourier-domain bright-field images.
    pairs : torch.Tensor
        ``(M, 2)`` neighbour index pairs.
    upsample_factor : float
        Subpixel registration upsample factor.
    radius : float
        Search radius around zero relative shift, pixels.

    Returns
    -------
    torch.Tensor
        ``(M, 2)`` relative shifts, as ``pairwise_relative_shifts`` returns them.
    """
    height, width = G_stack.shape[-2:]
    chunk = max(1, _PAIRWISE_CHUNK_ELEMENTS // (height * width))
    tail = pairs[:, 0].to(device=G_stack.device, dtype=torch.long)
    head = pairs[:, 1].to(device=G_stack.device, dtype=torch.long)
    deltas = torch.zeros((pairs.shape[0], 2), device=G_stack.device)
    for start in range(0, pairs.shape[0], chunk):
        stop = min(pairs.shape[0], start + chunk)
        cross_power = G_stack[tail[start:stop]] * G_stack[head[start:stop]].conj()
        coarse = torch.fft.ifft2(cross_power).abs()
        peak = nearest_peak(coarse, torch.zeros(stop - start, 2), radius)
        deltas[start:stop] = _peak_offsets(
            cross_power, upsample_factor, integer_peak=peak
        ).to(G_stack.real.dtype)
    return deltas


class ShiftModel:
    """Smooth model of a bright-field shift field, for predicting further pixels.

    A least-squares polynomial in the detector coordinate per shift component:
    affine (defocus, astigmatism, rotation) until ``CUBIC_MIN_POINTS`` points
    are recorded, then cubic (adding coma and the third-order terms). Fitted in
    float64 on coordinates scaled to the recorded extent, for conditioning.
    """

    def __init__(self) -> None:
        self._coords: list[torch.Tensor] = []
        self._shifts: list[torch.Tensor] = []

    def record(self, coords: torch.Tensor, shifts: torch.Tensor) -> None:
        """Add measured shifts ``(n, 2)`` at detector coordinates ``(n, 2)``."""
        self._coords.append(coords.detach().to("cpu", torch.float64))
        self._shifts.append(shifts.detach().to("cpu", torch.float64))

    @staticmethod
    def _design(coords: torch.Tensor, scale: float, degree: int) -> torch.Tensor:
        y, x = (coords / scale).unbind(dim=1)
        terms = [torch.ones_like(y), y, x]
        if degree >= 3:
            terms += [y * y, y * x, x * x, y**3, y * y * x, y * x * x, x**3]
        return torch.stack(terms, dim=1)

    def _fit(self) -> tuple[torch.Tensor, float, int] | None:
        if not self._coords:
            return None
        coords, shifts = torch.cat(self._coords), torch.cat(self._shifts)
        if coords.shape[0] < 3:
            return None
        degree = 3 if coords.shape[0] >= CUBIC_MIN_POINTS else 1
        scale = float(coords.abs().max()) or 1.0
        design = self._design(coords, scale, degree)
        if int(torch.linalg.matrix_rank(design)) < design.shape[1]:
            return None
        return torch.linalg.lstsq(design, shifts).solution, scale, degree

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        """Predicted shifts ``(n, 2)`` at ``coords``; the mean (or 0) until fittable."""
        coords64 = coords.detach().to("cpu", torch.float64)
        fit = self._fit()
        if fit is None:
            mean = torch.cat(self._shifts).mean(0) if self._shifts else None
            out = torch.zeros(coords64.shape[0], 2, dtype=torch.float64)
            return out if mean is None else out + mean
        coef, scale, degree = fit
        return self._design(coords64, scale, degree) @ coef

    def residual_rms(self) -> float:
        """RMS distance of the recorded shifts from the fitted model, pixels."""
        fit = self._fit()
        if fit is None:
            return 0.0
        coef, scale, degree = fit
        coords, shifts = torch.cat(self._coords), torch.cat(self._shifts)
        resid = shifts - self._design(coords, scale, degree) @ coef
        return float(torch.sqrt((resid**2).sum(1).mean()))


class Registrar:
    """Register bright-field images against a reference, by either peak rule.

    ``"argmax"`` is exactly ``relative_shifts`` (the global maximum). ``"guided"``
    predicts each batch from the shifts registered so far and takes the peak
    near the prediction; see the module docstring.

    Parameters
    ----------
    peak_selection : str
        ``"argmax"`` or ``"guided"``.
    G_ref : torch.Tensor
        ``fft2`` of the initial reference image, ``(H, W)``; sets the radius.
    upsample_factor : float
        Subpixel registration upsample factor.
    """

    def __init__(
        self, peak_selection: str, G_ref: torch.Tensor, upsample_factor: float
    ) -> None:
        validate_peak_selection(peak_selection)
        self.guided = peak_selection == "guided"
        self.upsample_factor = upsample_factor
        period = lattice_period(G_ref) if self.guided else math.inf
        self.radius = RADIUS_FRACTION * period
        self.model = ShiftModel()

    def register(
        self, G_ref: torch.Tensor, G_moving: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Shifts ``(n, 2)`` of ``G_moving`` ``(n, H, W)`` at detector ``coords``."""
        if not self.guided:
            return relative_shifts(G_ref, G_moving, self.upsample_factor)
        prediction = self.model.predict(coords)
        shifts = guided_shifts(
            G_ref, G_moving, self.upsample_factor, prediction, self.radius
        )
        self.model.record(coords, shifts)
        return shifts

    def warn_if_ill_determined(self) -> None:
        """Warn when the smooth model misses the chosen peaks by a large fraction of r."""
        if not self.guided or math.isinf(self.radius):
            return
        rms = self.model.residual_rms()
        if rms > 0.5 * self.radius:
            warnings.warn(
                f"peak_selection='guided': the bright-field shifts depart from a "
                f"smooth model by {rms:.2f} px RMS, more than half the peak search "
                f"radius ({self.radius:.2f} px); the chosen cross-correlation peaks "
                "may still be off by a lattice vector.",
                UserWarning,
                stacklevel=3,
            )


def warn_if_multi_peaked(
    G_ref: torch.Tensor,
    measured_px: torch.Tensor,
    fitted_px: torch.Tensor,
    peak_selection: str,
) -> None:
    """Warn when argmax peak selection on a periodic image is likely wrong.

    The shifts are ambiguous modulo the lattice period ``p``. Either sign of a
    wrong pick is flagged: a fitted rim shift beyond ``p / 2``, or measured
    shifts that depart from the fitted smooth field by more than ``p / 10`` RMS
    (pixels jumping between lattice copies).

    Parameters
    ----------
    G_ref : torch.Tensor
        ``fft2`` of the reference bright-field image, ``(H, W)``.
    measured_px : torch.Tensor
        Measured shifts ``(N, 2)``, pixels.
    fitted_px : torch.Tensor
        Shifts of the fitted aberrations at the same pixels ``(N, 2)``, pixels.
    peak_selection : str
        The rule in use; only ``"argmax"`` is checked here (``"guided"`` checks
        itself, see :meth:`Registrar.warn_if_ill_determined`).
    """
    if peak_selection != "argmax":
        return
    period = lattice_period(G_ref)
    if math.isinf(period):
        return
    rim = float(fitted_px.abs().max())
    rms = float(torch.sqrt(((measured_px - fitted_px) ** 2).sum(1).mean()))
    if rim > 0.5 * period or rms > 0.1 * period:
        warnings.warn(
            f"The bright-field images are periodic (lattice period {period:.1f} px), "
            "so their cross-correlation is multi-peaked and the global maximum may "
            f"be the wrong copy (rim shift {rim:.1f} px, departure from a smooth "
            f"field {rms:.2f} px RMS). Pass peak_selection='guided' to choose each "
            "peak from a smooth prediction.",
            UserWarning,
            stacklevel=3,
        )
