"""The bright-field shifts of one bin level, by the chosen peak rule.

``peak_selection="argmax"`` and the neighbour-mesh estimator use
:func:`~scatterem.reconstruction.ptychography.direct._bf_parallax.measure_bf_parallax`.
``peak_selection="guided"`` with the mean-template estimator registers the
images in rings of increasing detector radius with the
:class:`~scatterem.reconstruction.ptychography.direct._peak_selection.Registrar`
(see :mod:`._peak_selection` for the rule).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from scatterem.utils.subpixel import _PAIRWISE_CHUNK_ELEMENTS

from . import _bright_field_graph as graph
from ._bf_parallax import measure_bf_parallax
from ._bright_field_peaks import lowpass_weight
from ._peak_selection import Registrar, lattice_period

__all__ = ["guided_parallax", "level_shifts"]


class _BinSpectra:
    """Mean spectra of the detector bins of :func:`_bright_field_graph.bin_members`.

    ``read(ids)`` gives the bin means ``(n, H, W)`` for bin indices ``ids``.
    The means are summed in complex128 and stored in the dtype of the data,
    the same as the binned nodes of ``measure_bf_parallax``.
    """

    def __init__(self, source, coords: np.ndarray, binning: int, device) -> None:
        self.pitch = graph.min_detector_spacing(coords)
        _, member_bin, self.coords = graph.bin_members(coords, binning, self.pitch)
        height, width = source.scan_grid_shape
        n_bins = self.coords.shape[0]
        acc = torch.zeros(
            (n_bins, height, width), dtype=torch.complex128, device=device
        )
        block = max(1, min(64, _PAIRWISE_CHUNK_ELEMENTS // (height * width)))
        dtype = torch.complex64
        for start in range(0, coords.shape[0], block):
            ids = list(range(start, min(coords.shape[0], start + block)))
            x = torch.as_tensor(source.read(ids)).to(device)
            dtype = torch.complex128 if x.dtype == torch.complex128 else dtype
            members = torch.as_tensor(member_bin[ids], device=device)
            acc.index_add_(0, members, x.to(torch.complex128))
        counts = np.bincount(member_bin, minlength=n_bins).astype(np.float64)
        acc /= torch.as_tensor(counts, device=device)[:, None, None]
        self.spectra = acc.to(dtype)

    def read(self, ids) -> torch.Tensor:
        return self.spectra[torch.as_tensor(list(ids), device=self.spectra.device)]


def guided_parallax(
    source,
    detector_coords_px: torch.Tensor,
    radial_order: torch.Tensor,
    G_ref: torch.Tensor,
    upsample_factor: float,
    *,
    binning: int = 1,
    lowpass_sigma_px: float | None = None,
) -> torch.Tensor:
    """Bright-field shifts with ``peak_selection="guided"``, in pixels.

    The guided counterpart of
    :func:`~scatterem.reconstruction.ptychography.direct._bf_parallax.measure_bf_parallax`
    with ``shift_estimator="mean_template"``: every image (or detector bin)
    is registered onto ``G_ref`` by a :class:`Registrar`, in batches of
    increasing detector radius, so each batch is predicted from the batches
    nearer the centre.

    Parameters
    ----------
    source : SpectrumSource
        Delivers the image spectra, ``read(indices) -> (n, H, W)`` complex,
        index ``i`` for row ``i`` of ``detector_coords_px``.
    detector_coords_px : torch.Tensor
        ``(N, 2)`` detector coordinates of the images, pixels, relative to
        the bright-field centre.
    radial_order : torch.Tensor
        ``(N,)`` image indices sorted by increasing detector radius.
    G_ref : torch.Tensor
        ``fft2`` of the reference image, ``(H, W)``; sets the search radius.
    upsample_factor : float
        Subpixel registration upsample factor.
    binning : int, optional
        Detector binning factor, as in ``measure_bf_parallax``: bins of
        ``binning`` detector pixels per axis are registered as their mean
        spectrum, in order of increasing bin radius, and every pixel then
        gets its shift by the moving least-squares fit over the bins
        (:func:`_bright_field_graph.interpolate_shifts`). 1 means no binning.
    lowpass_sigma_px : float, optional
        Gaussian low-pass (:func:`._bright_field_peaks.lowpass_weight`) on
        every image and on the reference, as in ``measure_bf_parallax``.
        The search radius comes from the unfiltered ``G_ref``.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` float32 shifts, row ``i`` for image ``i``, mean removed, in
        the sign convention of ``measure_bf_parallax`` (the translation that
        registers each image onto the reference).
    """
    registrar = Registrar("guided", G_ref, upsample_factor)
    coords = detector_coords_px.detach().to("cpu", torch.float64)
    height, width = G_ref.shape[-2:]
    weight = None
    if lowpass_sigma_px is not None:
        weight = lowpass_weight((height, width), float(lowpass_sigma_px), G_ref.device)
        G_ref = G_ref * weight.to(G_ref.real.dtype)
    bins = None
    if binning > 1:
        bins = _BinSpectra(source, coords.numpy(), int(binning), G_ref.device)
        node_coords = torch.as_tensor(bins.coords)
        order = torch.argsort((node_coords**2).sum(1), stable=True)
        read = bins.read
    else:
        node_coords = coords
        order = radial_order.to(device="cpu", dtype=torch.long)
        read = source.read
    batch = max(1, min(64, _PAIRWISE_CHUNK_ELEMENTS // (height * width)))
    shifts = torch.zeros((order.numel(), 2), dtype=torch.float64)
    for start in range(0, order.numel(), batch):
        ids = order[start : start + batch]
        moving = torch.as_tensor(read(ids.tolist())).to(G_ref.device)
        if weight is not None:
            moving = moving * weight.to(moving.real.dtype)
        found = registrar.register(G_ref, moving, node_coords[ids])
        shifts[ids] = found.detach().to("cpu", torch.float64)
    registrar.warn_if_ill_determined()
    if bins is not None:
        shifts = torch.as_tensor(
            graph.interpolate_shifts(
                coords.numpy(), bins.coords, shifts.numpy(), binning * bins.pitch
            )
        )
    shifts = shifts - shifts.mean(dim=0)
    return shifts.to(torch.float32)


def level_shifts(
    source,
    vbf,
    G_ref: torch.Tensor,
    *,
    peak_selection: str,
    shift_estimator: str,
    binning: int,
    peak_grid_divisor: int,
    lowpass_sigma_px: float | None,
    verbosity: int,
    measure=measure_bf_parallax,
) -> torch.Tensor:
    """The bright-field shifts of one bin level, by the chosen peak rule, pixels.

    ``"guided"`` with ``shift_estimator="mean_template"`` runs
    :func:`guided_parallax` (with the same ``binning`` and low-pass) when the
    reference image is periodic, and the ``"argmax"`` path when it is not.
    Everything else runs
    :func:`~scatterem.reconstruction.ptychography.direct._bf_parallax.measure_bf_parallax`,
    with ``peak_tolerance`` 0 for ``"argmax"`` and 0.02 for ``"guided"``.
    ``vbf`` is the virtual bright-field dataset: its
    ``bright_field_inds_centered`` are the detector coordinates and its
    ``bright_field_inds_radial_order`` the ring order. ``measure`` is that
    function; the driver passes its own module attribute,
    so a test can replace it there.
    """
    if peak_selection == "guided" and shift_estimator == "mean_template":
        if math.isinf(lattice_period(G_ref)):
            # No lattice: the guided pick is the global maximum (module
            # docstring of _peak_selection), so take the argmax path itself.
            peak_selection = "argmax"
        else:
            return guided_parallax(
                source,
                vbf.bright_field_inds_centered,
                vbf.bright_field_inds_radial_order,
                G_ref,
                float(peak_grid_divisor),
                binning=binning,
                lowpass_sigma_px=lowpass_sigma_px,
            )
    return measure(
        source,
        vbf.bright_field_inds_centered,
        shift_estimator=shift_estimator,
        binning=binning,
        peak_grid_divisor=peak_grid_divisor,
        lowpass_sigma_px=lowpass_sigma_px,
        start_template=G_ref if shift_estimator == "mean_template" else None,
        peak_tolerance=0.0 if peak_selection == "argmax" else 0.02,
        status_line=bool(verbosity),
        quiet=not verbosity,
    )
