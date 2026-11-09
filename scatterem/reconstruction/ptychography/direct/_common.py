"""Shared leaf helpers: phase-sign orientation, input resolution, chunking ranges,
the scan lattice."""

import logging
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from scatterem.data.datasets import (
    Dataset4DStem,
)
from scatterem.reconstruction._sampling import (
    _probe_aberrations as _probe_aberrations,
)
from scatterem.reconstruction._sampling import _require_meta as _require_meta
from scatterem.reconstruction._sampling import (
    _resolve_upsample_int as _resolve_upsample_int,
)

logger = logging.getLogger(__name__)

PhaseSignMode = Literal["preserve", "positive", "negative"]


def _phase_image_skewness(
    phase_image: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Return normalized third moment used to choose the physical phase branch."""
    phase = phase_image.detach().to(torch.float32)
    centered = phase - phase.mean()
    std = centered.square().mean().sqrt()
    # The two guards are resolved ON DEVICE with ``torch.where`` rather than by
    # ``if <0-dim cuda tensor>``.  Each python ``if`` on a device scalar is a
    # blocking ``cudaStreamSynchronize``, and this function sits at the TAIL of
    # the reconstruction, so the drains do not overlap with anything -- they
    # serialise as sync / launch a handful of tiny kernels / sync again.  The
    # returned values are unchanged in every case: the fallback is still exactly
    # ``0.0`` where ``std`` is non-finite or ``<= eps``, or where the ratio is
    # non-finite.  ``std.pow(3)`` cannot raise, so computing the ratio
    # unconditionally is safe -- a zero or non-finite ``std`` makes it
    # non-finite, which the same mask replaces.
    skewness = centered.pow(3).mean() / std.pow(3)
    finite = torch.isfinite(std) & (std > eps) & torch.isfinite(skewness)
    return torch.where(finite, skewness, 0.0)


def _orient_phase_image(
    phase_image: torch.Tensor,
    phase_sign: PhaseSignMode = "preserve",
    verbosity: int = 0,
) -> torch.Tensor:
    """Apply the phase sign convention to a direct-ptychography image.

    ``"preserve"`` (the default) returns the image unchanged. For a
    weak-phase object the forward model fixes the sign (positive phase for
    positive projected potential) once the aberrations and rotation are
    fixed. The one genuine ambiguity of the bright-field-shift fit, (C1,
    rotation) <-> (-C1, rotation + 180 deg), is resolved there with the
    rotation prior. The sign then does not depend on the image content.

    The sign is fixed only for weak-phase objects.

    ``"positive"``/``"negative"`` flip the image to positive/negative
    skewness. That heuristic suits atom columns but is data dependent: on
    low-contrast weak-phase specimens the skewness is near zero, a small change of
    the data flips it, and it can invert a correct image.
    """
    if phase_sign == "preserve":
        return phase_image
    if phase_sign not in ("positive", "negative"):
        raise ValueError(
            "phase_sign must be one of 'preserve', 'positive', or 'negative', "
            f"got {phase_sign!r}"
        )

    skewness = _phase_image_skewness(phase_image)
    target_sign = 1.0 if phase_sign == "positive" else -1.0
    if verbosity > 0:
        # Only the diagnostic path pays the D2H sync.
        should_flip = float(skewness.detach().cpu()) * target_sign < 0.0
        action = "flipping" if should_flip else "preserving"
        logger.info(
            "Direct ptycho phase skewness: %+.4f; %s phase sign for %s contrast",
            float(skewness.detach().cpu()),
            action,
            phase_sign,
        )
        return -phase_image if should_flip else phase_image
    # Apply the sign as a device-side multiply by exactly +-1.0, which is
    # bit-exact for every float and preserves NaN/inf, so the returned array is
    # identical to the branched form while the decision never reaches the host.
    sign = torch.where(skewness * target_sign < 0.0, -1.0, 1.0)
    return phase_image * sign




def _resolve_roi_center(roi_center, dataset):
    """The ROI centre as a ``(y, x)`` pair, or raise if it is none of the forms.

    ``"dark_field_center_of_mass"`` returns the geometric centre, exactly as
    ``"center"`` does -- the centre-of-mass variant was never implemented, and
    this preserves that rather than silently changing what callers get.
    """
    if roi_center == "center":
        return (dataset.shape[0] // 2, dataset.shape[1] // 2)
    if roi_center == "dark_field_center_of_mass":
        return (dataset.shape[0] // 2, dataset.shape[1] // 2)
    if isinstance(roi_center, tuple):
        return roi_center
    raise ValueError(f"Invalid roi_center: {roi_center}")


def _detector_dr(vbf) -> np.ndarray:
    """Real-space sampling of the detector, ``1 / (2 * k_max)``, from a vBF itself.

    A virtual bright field keeps its source cube's 4-entry sampling (scan, then
    detector pixel size) and its full-detector ``bright_field_mask``, so this is
    ``Dataset4DStem.dr`` evaluated on those, with the same arithmetic: equal, bit
    for bit, to the parent's ``dr`` at the time the vBF was built, and available
    when the vBF has no parent dataset.

    Parameters
    ----------
    vbf : DatasetVirtualBrightField4DStem
        Virtual bright field (eager or streaming).

    Returns
    -------
    numpy.ndarray
        Detector real-space sampling per detector axis, shape ``(2,)``, in Angstrom.

    Raises
    ------
    ValueError
        If ``vbf.sampling`` has no detector entries (fewer than 4) or
        ``vbf.bright_field_mask`` is not a 2-D detector mask.
    """
    sampling = np.atleast_1d(vbf.sampling)
    mask = vbf.bright_field_mask
    if sampling.shape[0] != 4 or mask.ndim != 2:
        raise ValueError(
            "the virtual bright field does not carry its detector sampling: it needs "
            "the source cube's 4-entry sampling (scan y, scan x, detector y, detector "
            f"x; got {sampling.tolist()}) and a 2-D bright_field_mask (got shape "
            f"{tuple(mask.shape)}). Build it with Dataset4DStem.get_vbf or pass both "
            "to DatasetVirtualBrightField4DStem."
        )
    k_max = sampling[-2:] * np.array(mask.shape) / 2
    return 1 / (2 * k_max)


def _resolve_vbf(dataset, bright_field_mask_threshold):
    """Datasets build/cache their own provider flavor (Dataset4DStem.get_vbf);
    anything else is already a vBF/provider and passes through."""
    if isinstance(dataset, Dataset4DStem):
        return dataset.get_vbf(bright_field_mask_threshold)
    return dataset


def _chunk_ranges(n_total: int, n_batches):
    """Yield ``(s, e)`` BF-pixel ranges: ceil-division batches with a clamped
    tail. Never yields an empty range (CUFFT_INVALID_SIZE guard) even when
    ``n_batches`` exceeds the number of non-empty batches — reachable on the
    empirical-SSNR path where half-pixel subsets are reconstructed."""
    n_b = int(n_batches) if n_batches is not None else 1
    size = max(1, int(math.ceil(n_total / max(1, n_b))))
    for s in range(0, n_total, size):
        yield s, min(s + size, n_total)


def zero_scan_lattice(spectrum: torch.Tensor, scan: Sequence[int]) -> torch.Tensor:
    """Set the scan-lattice rows of an acBF spectrum to 0, in place.

    A plane grid upsampled by ``(Uy, Ux)`` over a ``(Ny, Nx)`` scan reads the
    coarse scan spectrum tiled: its row ``Q`` reads ``G(K, Q mod N)``. On the
    ``Uy Ux`` scan-lattice rows ``Q = (m / dy, n / dx)``, ``[..., ::Ny, ::Nx]``
    of the corner-origin spectrum, that is ``G(K, 0)``: every bright-field
    pixel's scan DC, with the specimen's content at every lattice frequency
    folded onto it by the coarse scan. The scan DC is close to flat in ``K``
    (flat inside a pixel-sharp disc on a uniform detector; at a binned or
    blurred edge it varied by 12 to 23 % rms over the bright-field pixels of
    the planted truth below), so the acBF weight does not suppress it, about
    ``1 / phi`` larger than that content, and no plane's signal (the plane's
    own row ``Q = 0`` weighs 0). Kept, it put a grid with the period of the
    scan step into every plane. The folded content is not flat in ``K``: on
    the row ``Q`` it is the plane's own signal there plus the other lattice
    orders' alias. A model of the DC's profile across the disc could keep
    that signal, but only one accurate to well under its share (about 2 % of
    the DC per pixel on a planted truth). On a planted truth (a sweep of 10 seeds, C10 = -300 to -600 A, U = 2 and 5), subtracting a
    flat DC left the lattice rows 0.07 to 0.74 times the error of 0 for an
    ideal pixel-aligned aperture, but 1.26 to 34 times with the disc edge
    integrated by 2 x 2 binning, 3.4 to 66 times with it blurred by one
    detector pixel and up to 201 times with both (a blur of one binned
    pixel). In a separate check a 1 %
    per-pixel detector gain error made it up to 1.5 times. 0 loses exactly the
    specimen's content on those rows and needs no model of the DC, so the
    tiled scan spectrum carries no ``G(K, 0)`` by definition and the acBF
    estimate is 0 on every scan-lattice row, ``Q = 0`` included. The
    estimate is linear in ``G``, so this is the estimate of ``G`` with
    ``G(K, 0)`` dropped.

    At every U: at U = 1 the one such row is ``Q = 0``, already 0 by
    definition (its weight is 0); writing it keeps it exactly 0 where the
    float32 phases of the terms above C12 would leave a residue (on CUDA).

    Parameters
    ----------
    spectrum : torch.Tensor
        ``(..., Ny Uy, Nx Ux)`` corner-origin spectrum, written in place.
    scan : sequence of int
        The scan shape ``(Ny, Nx)``.

    Returns
    -------
    torch.Tensor
        ``spectrum``.
    """
    ny, nx = scan
    spectrum[..., ::ny, ::nx] = 0
    return spectrum


def has_orders_above_c12(aberrations: torch.Tensor) -> bool:
    """Whether any of the nine coefficients above C12 (C21 .. C34b) is nonzero.

    One host sync. With such orders the CUDA correction kernels can leave a
    float32 residue of the double-overlap function at ``Q = 0``: they build
    ``A(K)`` and ``A(K -+ 0)`` in different kernels, so ``Gamma(0, K)`` is no
    longer exactly 0 and the scan DC gets a weight of arbitrary phase.
    """
    return bool(aberrations[3:].detach().ne(0).any())


def remove_scan_lattice(
    image: torch.Tensor, upsample, aberrations: torch.Tensor | None = None
) -> torch.Tensor:
    """Remove the scan-lattice frequencies of a real image, in place.

    The image-side form of ``zero_scan_lattice``, for the single-plane
    ``direct_ptychography`` image, whose chunk images still read the scan DC
    (the k-outer kernels read ``G`` modulo the scan, the other chunk paths
    tile it): the scan lattice is taken out of their sum. An image's
    scan-lattice frequencies depend only on its scan-periodic mean cell,
    ``image.view(..., Ny, Uy, Nx, Ux).mean((-4, -2))``, and subtracting that
    cell zeroes them (the same as zeroing ``fft2(image)[..., ::Ny, ::Nx]``)
    and changes no other frequency, without an FFT. The image is linear in
    ``G``, so the result is the image of ``G`` with ``G(K, 0)`` dropped, up
    to the float32 rounding of the chunk sum (about 1e-7 relL2 on the test
    fixture, 97-99 % of it off the lattice: it is spread over every
    frequency). The lattice rows keep only the rounding of the subtraction,
    up to about 100 ``eps32 * ||image||_2`` on the test fixture.

    Parameters
    ----------
    image : torch.Tensor
        ``(..., Ny Uy, Nx Ux)`` real image, contiguous, written in place.
    upsample : sequence of int
        The per-axis upsampling ``(Uy, Ux)``. With both 1 the one lattice
        frequency is ``Q = 0``, the image mean, which weighs 0 there too. It
        is 0 up to rounding, so the image is returned unchanged, unless
        `aberrations` set an order above C12 (``has_orders_above_c12``).
        Then the mean is subtracted: on CUDA it can hold a residue of the
        scan DC (a relL2 of 2e-2 on an H200).
    aberrations : torch.Tensor, optional
        The 12 coefficients the image was corrected with. Only read at an
        upsampling of 1.

    Returns
    -------
    torch.Tensor
        ``image``.
    """
    uy, ux = (int(u) for u in upsample)
    if uy == 1 and ux == 1:
        if aberrations is not None and has_orders_above_c12(aberrations):
            image -= image.mean((-2, -1), keepdim=True)
        return image
    *batch, height, width = image.shape
    cells = image.view(*batch, height // uy, uy, width // ux, ux)
    cells -= cells.mean((-4, -2), keepdim=True)
    return image


def zero_ssnr_scan_lattice(
    ssnr: torch.Tensor, scan: Sequence[int], upsample
) -> torch.Tensor:
    """Set an upsampled phase-only SSNR map to 0 on the scan lattice.

    In place. The image is 0 on those frequencies (``remove_scan_lattice``),
    so its SSNR is 0 there too, whichever estimate the SSNR comes from.

    Parameters
    ----------
    ssnr : torch.Tensor
        ``(Ny Uy, Nx Ux)`` corner-origin SSNR map, written in place.
    scan : sequence of int
        The scan shape ``(Ny, Nx)``.
    upsample : sequence of int
        The per-axis upsampling ``(Uy, Ux)``. With both 1 the map is returned
        unchanged, as ``remove_scan_lattice`` returns the image.

    Returns
    -------
    torch.Tensor
        ``ssnr``.
    """
    if max(int(u) for u in upsample) > 1:
        zero_scan_lattice(ssnr, scan)
    return ssnr
