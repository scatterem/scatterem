"""Shifts of the bright-field (BF) images of a defocused or aberrated probe.

Each BF detector pixel sees the specimen through a different beam tilt. Its
image is the specimen image, translated by a shift that scales with the slope
of the aberration function at that detector position (parallax: Lupini, Chi
and Jesse 2016; Varnavides et al. 2023; Kirkland 2020). This module measures
the shift of every BF image. A separate fit turns the shifts into aberrations.

Two estimators:

- ``shift_estimator="mean_template"``: register every image onto one common reference by
  upsampled cross-correlation (Guizar-Sicairos, Thurman and Fienup 2008);
  rebuild the reference as the aligned average and register again.
- ``shift_estimator="neighbour_mesh"``: register detector neighbours onto each other and fit
  all shifts to these relative measurements by least squares. No single
  starting reference is needed.

Fourier convention: ``torch.fft.fft2``, kernel ``exp(−2πi k·r)``, zero
frequency at index 0 (not fftshifted), ``k`` in cycles per scan pixel. A
positive shift ``u`` moves content to larger indices:
``F[f(r − u)] = F[f] · exp(−2πi k·u)``
(:func:`scatterem.physics.fourier.phase_ramp`).
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.fourier import phase_ramp
from scatterem.reconstruction.ptychography.direct import _bright_field_graph as graph
from scatterem.reconstruction.ptychography.direct._bright_field_peaks import (
    condition_spectra,
    match_to_template,
)
from scatterem.reconstruction.ptychography.direct._bright_field_sources import (
    ChunkedSpectrumSource,
    SpectrumSource,
    _as_source,
)

__all__ = ["ChunkedSpectrumSource", "SpectrumSource", "measure_bf_parallax"]

logger = logging.getLogger(__name__)

_SHIFT_ESTIMATORS = ("mean_template", "neighbour_mesh")


def _check_int(value, name: str, low: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < low
    ):
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _validate_options(
    shift_estimator,
    binning,
    peak_grid_divisor,
    lowpass_sigma_px,
    n_refinements,
    peak_tolerance,
    start_template,
) -> None:
    if shift_estimator not in _SHIFT_ESTIMATORS:
        raise ValueError(
            f"shift_estimator must be one of {_SHIFT_ESTIMATORS}, got {shift_estimator!r}"
        )
    _check_int(binning, "binning", 1)
    _check_int(peak_grid_divisor, "peak_grid_divisor", 1)
    _check_int(n_refinements, "n_refinements", 0)
    if lowpass_sigma_px is not None:
        s = float(lowpass_sigma_px)
        if not (math.isfinite(s) and s > 0):
            raise ValueError(
                f"lowpass_sigma_px must be None or finite > 0, got {lowpass_sigma_px!r}"
            )
    t = float(peak_tolerance)
    if not 0.0 <= t < 1.0:
        raise ValueError(f"peak_tolerance must be in [0, 1), got {peak_tolerance!r}")
    if shift_estimator == "neighbour_mesh" and start_template is not None:
        raise ValueError(
            "start_template is only valid with shift_estimator='mean_template'"
        )


def _detector_points(detector_coords_px, n: int) -> np.ndarray:
    q = detector_coords_px
    if isinstance(q, Tensor):
        if q.is_complex():
            raise ValueError("detector coordinates must be real")
        q = q.detach().cpu().numpy()
    q = np.asarray(q)
    if np.iscomplexobj(q) or q.dtype.kind not in "fiu":
        raise ValueError("detector coordinates must be real numbers")
    q = q.astype(np.float64)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError(f"detector coordinates must be (N, 2), got shape {q.shape}")
    if q.shape[0] != n:
        raise ValueError(f"{q.shape[0]} coordinates for {n} images")
    if not np.all(np.isfinite(q)):
        raise ValueError("detector coordinates must be finite")
    if n and np.unique(q, axis=0).shape[0] != n:
        raise ValueError("detector coordinates contain a duplicate row")
    return q


class _Nodes:
    """Pre-processed spectra of the estimator nodes (images or bins), by index."""

    def __init__(self, source, order, binning, pitch, q_sorted, sigma, block):
        self.source, self.order, self.sigma, self.block = source, order, sigma, block
        probe = self._read_raw(order[:1])
        self.dtype = (
            torch.complex128 if probe.dtype == torch.complex128 else torch.complex64
        )
        self.device = probe.device
        self.binned = None
        if binning == 1:
            self.coords, self.keys, self.member_bin = q_sorted, None, None
            return
        self.keys, self.member_bin, self.coords = graph.bin_members(
            q_sorted, binning, pitch
        )
        n0, n1 = source.scan_grid_shape
        acc = torch.zeros(
            (len(self.keys), n0, n1), dtype=torch.complex128, device=self.device
        )
        for start in range(0, len(order), block):
            x = self._read_raw(order[start : start + block]).to(torch.complex128)
            for k in range(x.shape[0]):
                acc[self.member_bin[start + k]] += x[k]
        counts = np.bincount(self.member_bin, minlength=len(self.keys)).astype(
            np.float64
        )
        acc /= torch.as_tensor(counts, device=self.device)[:, None, None]
        self.binned = condition_spectra(acc.to(self.dtype), sigma)

    def _read_raw(self, input_indices) -> Tensor:
        data = self.source.read([int(i) for i in input_indices])
        if not isinstance(data, Tensor):
            data = torch.as_tensor(np.asarray(data))
        if not data.is_complex():
            raise ValueError(
                "the spectrum source delivered real data; image_spectra must be complex"
            )
        if tuple(data.shape[1:]) != tuple(self.source.scan_grid_shape):
            raise ValueError(f"spectra shape {tuple(data.shape[1:])} != scan shape")
        if not bool(torch.isfinite(data).all()):
            raise ValueError("the spectra contain non-finite values")
        return data

    @property
    def n(self) -> int:
        return self.coords.shape[0]

    def get(self, ids: np.ndarray) -> Tensor:
        if self.binned is not None:
            return self.binned[torch.as_tensor(ids, device=self.device)]
        x = self._read_raw(self.order[ids]).to(device=self.device, dtype=self.dtype)
        return condition_spectra(x, self.sigma)


class _Progress:
    def __init__(self, total: int, enabled: bool):
        self.total, self.done, self.enabled = total, 0, enabled and total > 0

    def step(self, k: int) -> None:
        self.done += k
        if self.enabled:
            sys.stderr.write(f"\rbright-field shifts: {self.done}/{self.total}")
            sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")


def _template_estimate(
    nodes, reference, n_refinements, reg, block, progress
) -> np.ndarray:
    n = nodes.n
    n0, n1 = nodes.source.scan_grid_shape
    if reference is None:
        acc = torch.zeros((n0, n1), dtype=torch.complex128, device=nodes.device)
        for start in range(0, n, block):
            x = nodes.get(np.arange(start, min(n, start + block)))
            for k in range(x.shape[0]):
                acc += x[k].to(torch.complex128)
        ref = (acc / n).to(nodes.dtype)
    else:
        ref = reference
    shifts = np.zeros((n, 2))
    for it in range(n_refinements + 1):
        rebuild = it < n_refinements
        acc = torch.zeros((n0, n1), dtype=torch.complex128, device=nodes.device)
        for start in range(0, n, block):
            ids = np.arange(start, min(n, start + block))
            x = nodes.get(ids)
            s, _ = reg(x, ref)
            shifts[ids] = s
            if rebuild:
                ramp = phase_ramp(
                    (n0, n1),
                    torch.as_tensor(s, device=nodes.device),
                    dtype=x.dtype,
                    device=nodes.device,
                )
                aligned = x * ramp
                for k in range(x.shape[0]):
                    acc += aligned[k].to(torch.complex128)
            progress.step(len(ids))
        if rebuild:
            ref = (acc / n).to(nodes.dtype)
    return shifts


def _mesh_estimate(nodes, pitch, reg, block, progress, quiet) -> np.ndarray:
    edges = graph.neighbour_edges(nodes.coords, pitch, nodes.keys)
    d = np.zeros((len(edges), 2))
    keep = np.ones(len(edges), dtype=bool)
    for start in range(0, len(edges), block):
        e = edges[start : start + block]
        need, inv = np.unique(e, return_inverse=True)
        inv = inv.reshape(e.shape)
        x = nodes.get(need)
        s, empty = reg(
            x[torch.as_tensor(inv[:, 1], device=x.device)],
            x[torch.as_tensor(inv[:, 0], device=x.device)],
        )
        d[start : start + len(e)] = s
        keep[start : start + len(e)] = ~empty
        progress.step(len(e))
    shifts, n_comp = graph.solve_relative(nodes.n, edges[keep], d[keep])
    if n_comp > 1 and not quiet:
        logger.warning(
            "the detector-neighbour graph has %d connected components; each "
            "component has zero mean shift",
            n_comp,
        )
    return shifts


def _no_images(image_spectra, device) -> Tensor:
    """The ``(0, 2)`` result for no images, float64 for complex128 data."""
    dt = (
        torch.float64
        if getattr(image_spectra, "dtype", None) in (torch.complex128, np.complex128)
        else torch.float32
    )
    return torch.zeros((0, 2), dtype=dt, device=device)


def _start_reference(start_template, nodes: _Nodes, sigma) -> Tensor | None:
    """The conditioned starting reference, or None; checks shape and values."""
    if start_template is None:
        return None
    n0, n1 = nodes.source.scan_grid_shape
    r = start_template
    r = r if isinstance(r, Tensor) else torch.as_tensor(np.asarray(r))
    if not r.is_complex() or tuple(r.shape) != (n0, n1):
        raise ValueError(f"start_template must be complex ({n0}, {n1})")
    if not bool(torch.isfinite(r).all()):
        raise ValueError("start_template must be finite")
    return condition_spectra(r.to(device=nodes.device, dtype=nodes.dtype), sigma)


def measure_bf_parallax(
    image_spectra,
    detector_coords_px,
    *,
    shift_estimator: str = "mean_template",
    binning: int = 1,
    peak_grid_divisor: int = 20,
    lowpass_sigma_px: float | None = None,
    start_template=None,
    n_refinements: int = 2,
    peak_tolerance: float = 0.02,
    remove_image_mean: bool = True,
    status_line: bool = False,
    quiet: bool = False,
) -> Tensor:
    """Measure the translation of every BF image.

    Parameters
    ----------
    image_spectra : Tensor, numpy.ndarray or SpectrumSource
        The 2-D Fourier transforms of the N BF images (convention of the
        module docstring): a complex stack ``(N, n0, n1)``, or a source that
        delivers them in parts (:class:`SpectrumSource`,
        :class:`ChunkedSpectrumSource`).
    detector_coords_px : array_like
        Real ``(N, 2)`` detector coordinates in pixels, relative to the BF
        disc centre, in the order of the images. Finite, no duplicate rows.
    shift_estimator : {"mean_template", "neighbour_mesh"}, optional
        ``"mean_template"``: common reference; starting reference = the mean of
        all (binned) spectra, which is the incoherent BF image with the best
        signal-to-noise ratio; then ``n_refinements`` rebuilds as the aligned
        average. ``"neighbour_mesh"``: register detector neighbours
        (``|Δq| <= 1.5 p``, ``p`` the detector pitch; bins: indices within
        one) onto each other and solve the least-squares fit of
        ``s_j − s_i = d_ij``. Use it when a large aberration blurs the mean.
        It also recovers shifts larger than half the lattice repeat on a
        periodic specimen, with ``binning=1``, provided the shift difference
        between detector neighbours stays below half the lattice repeat and
        the neighbour graph is connected.
    binning : int, optional
        Detector binning factor (>= 1). Bins are centred on multiples of
        ``binning · p``; a bin spectrum is the mean of its members. Each BF
        pixel then gets a shift by a quadratic moving least-squares fit over
        the bins (Gaussian weights of width ``binning · p``), so every pixel
        gets a finite shift. 1 means no binning.
    peak_grid_divisor : int, optional
        The final peak grid is ``1/peak_grid_divisor`` scan pixel. Default 20:
        a grid error of at most 0.025 px before the parabolic step.
    lowpass_sigma_px : float, optional
        Gaussian low-pass ``exp(−2π² σ² |k|²)`` on every image; ``σ`` is the
        real-space standard deviation of the blur in scan pixels. A Gaussian
        has no ringing and zero phase, so it does not move the peak.
    start_template : array_like, optional
        Starting reference for ``"mean_template"``: complex ``(n0, n1)``, in the
        same Fourier convention as the input.
    n_refinements : int, optional
        Number of aligned-average rebuilds (>= 0). Default 2.
    peak_tolerance : float, optional
        In ``[0, 1)``. Peaks within this fraction of the highest peak count
        as equal; the one nearest to zero shift wins. This gives the true
        shift on a scan-periodic specimen when the shift is less than half
        the lattice repeat. Default 0.02.
    remove_image_mean : bool, optional
        Remove the mean shift over all images. Default True: the absolute
        offset depends on the reference and carries no information.
    status_line : bool, optional
        Show one progress line on ``sys.stderr``. Ignored when quiet.
    quiet : bool, optional
        No output: no log records above DEBUG, no warnings, no progress.

    Returns
    -------
    Tensor
        Real ``(N, 2)`` on the device of the image data, in scan pixels, row
        ``i`` for input row ``i``. dtype float64 for complex128 spectra, else
        float32: the precision follows the data. The result is the
        translation that registers each image onto the reference: an image
        equal to the reference translated by ``t`` gives ``−t`` (plus a common
        offset, zero after mean removal).

    Raises
    ------
    ValueError
        For an unknown shift estimator, an invalid option, spectra that are not a
        complex 3-D stack or source, bad coordinates, a count mismatch, a
        reference spectrum of the wrong shape or with ``"neighbour_mesh"``, or
        non-finite spectrum values.

    Notes
    -----
    All work runs in the canonical order of the coordinates (sorted by
    ``(q0, q1)``) in blocks that depend only on the scan shape. So a
    permutation of the input permutes the output exactly, and the chunk size
    of a source never changes the result.

    Degenerate inputs give, without an exception: ``N = 0``: an empty
    ``(0, 2)`` tensor; ``N = 1``: ``[[0, 0]]`` (with ``remove_image_mean=False``
    and ``"mean_template"``: the shift onto the reference); all images equal:
    equal shifts (zero after mean removal); a constant or zero image: shift 0
    relative to the reference, or dropped neighbour pairs (an isolated node
    gets 0); ``n0 = n1 = 1``: zeros; one bin: every pixel gets the bin
    shift; a disconnected neighbour graph: zero mean per component and a
    WARNING record.
    """
    _validate_options(
        shift_estimator,
        binning,
        peak_grid_divisor,
        lowpass_sigma_px,
        n_refinements,
        peak_tolerance,
        start_template,
    )
    source = _as_source(image_spectra)
    n0, n1 = source.scan_grid_shape
    q = _detector_points(detector_coords_px, source.image_count)
    n_img = q.shape[0]
    sigma = None if lowpass_sigma_px is None else float(lowpass_sigma_px)
    if n_img == 0:
        return _no_images(image_spectra, source.run_device)
    order = graph.canonical_order(q)
    q_sorted = q[order]
    block = int(min(64, max(1, (1 << 24) // (n0 * n1))))
    pitch = graph.min_detector_spacing(q_sorted)
    nodes = _Nodes(source, order, int(binning), pitch, q_sorted, sigma, block)
    ref = _start_reference(start_template, nodes, sigma)

    def reg(shifted_spectra, anchor_spectrum):
        return match_to_template(
            shifted_spectra,
            anchor_spectrum,
            peak_grid_divisor=int(peak_grid_divisor),
            peak_tolerance=float(peak_tolerance),
        )

    show = status_line and not quiet
    if shift_estimator == "mean_template":
        bar = _Progress(nodes.n * (int(n_refinements) + 1), show)
        node_shifts = _template_estimate(
            nodes, ref, int(n_refinements), reg, block, bar
        )
    else:
        n_edges = len(graph.neighbour_edges(nodes.coords, pitch, nodes.keys))
        bar = _Progress(n_edges, show)
        node_shifts = _mesh_estimate(nodes, pitch, reg, block, bar, quiet)
    bar.close()
    if nodes.binned is None:
        shifts = node_shifts
    else:
        shifts = graph.interpolate_shifts(
            q_sorted, nodes.coords, node_shifts, binning * pitch
        )
    if remove_image_mean:
        shifts = shifts - shifts.mean(axis=0)
    out = np.empty_like(shifts)
    out[order] = shifts
    logger.log(
        logging.DEBUG if quiet else logging.INFO,
        "bright-field shifts: %d images, %d nodes, estimator %s, binning %d",
        n_img,
        nodes.n,
        shift_estimator,
        binning,
    )
    dtype = torch.float64 if nodes.dtype == torch.complex128 else torch.float32
    return torch.as_tensor(out, dtype=dtype, device=nodes.device)
