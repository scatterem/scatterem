"""Amplitude-weighted SSB assembly: the opt-in ``gamma_weighting="wiener"`` path.

The shipped kernel (``_direct_ptychography_forward_precomputed_A_kouter``)
divides the double-overlap function ``Gamma(Q, K)`` by its own modulus
(floored at ``GAMMA_FLOOR = 2**-20``, ``nn.functional.warp.ptychography._funcs``)
and multiplies ``G`` by ``conj(Gamma / |Gamma|)`` alone, after which
:func:`~scatterem.reconstruction.ptychography.direct.direct_ptychography`
divides the sum by the scalar ``vBF.n_bright_field``. That estimator is

    sum_K G conj(Gamma / |Gamma|) / N_bf

which is a PHASE-ONLY matched filter: the double-overlap magnitude is computed
and then discarded, so every bright-field pixel contributes to every object
frequency with the same weight, gated only by the hard 0/1 aperture test.

This module implements the Wigner-distribution-deconvolution estimator instead,

    sum_K G conj(Gamma) / sum_K |Gamma(Q, K)|^2

keeping ``|Gamma|`` in the numerator and dividing by the Q-DEPENDENT
denominator. The practical consequence is on the upsampling replica. The
correction kernel reads ``G[ik, iqy % ny, iqx % nx]``, so an object frequency
past the scan Nyquist is served by the SAME measured word as the tile below it,
distinguished only by the correction factor. When that factor has unit modulus
the aliased copy enters at the same amplitude as the signal and is suppressed
only by phase incoherence across K (``~1 / sqrt(N_bf)``); weighting by
``|Gamma|`` down-weights the bright-field pixels that carry little double
overlap at that frequency, so the tiles separate by amplitude as well as phase.
Measured on a 256x256 scan crop of a 300 kV data set (alpha = 4 mrad,
C1 = -2.2 um, 29 A step): at upsample 4, where the replica lands at half the
output Nyquist, the scan-lattice spots fall from 124x to 11.8x the local
background; at upsample 2, where 1/dr coincides EXACTLY with the output
Nyquist, the fast-scan stripe falls from 208x to 7.4x and the slow-scan one
from 37x to 5.5x. These figures predate the scan-lattice rule.
At an upsampling of 2 or more
both images are now 0 on the scan-lattice frequencies, which read each
pixel's scan DC ``G(K, 0)``: the phase-only sum drops them
(``_common.remove_scan_lattice``) and this path drops ``G(K, 0)`` from its
numerator before the divide (``_common.zero_scan_lattice``).

This path is deliberately written in plain torch rather than as a second Warp
kernel: it is opt-in, the default route is untouched and stays bit-identical,
and the accumulation is over Q-space rather than image space (legitimate
because ``ifft2`` is linear, so ``sum_K ifft2(X_K) == ifft2(sum_K X_K)``), which
is what lets the Q-dependent divide happen before the single inverse transform.
It is correspondingly slower than the Warp path and allocates per chunk.
"""

import logging
import math
import time
from typing import Literal

import torch

from scatterem.data.datasets import DatasetVirtualBrightField4DStem
from scatterem.physics.optics import lens_wavefront

from ._chunking import _upsampled_grid
from ._common import (
    PhaseSignMode,
    _orient_phase_image,
    _probe_aberrations,
    _require_meta,
    zero_scan_lattice,
)

logger = logging.getLogger(__name__)

#: Default Tikhonov floor on ``sum_K |Gamma|^2``, as a fraction of its maximum.
DEFAULT_WIENER_ALPHA = 1e-3

#: How ``Gamma(Q, K)`` weights each bright-field pixel in the SSB assembly.
GammaWeighting = Literal["phase", "wiener"]


def _aperture(
    qy: torch.Tensor, qx: torch.Tensor, wavelength: float, semiangle: float
) -> torch.Tensor:
    """Hard 0/1 probe-forming aperture, the test of the Warp ``probe_cone_pass``.

    Parameters
    ----------
    qy, qx : torch.Tensor
        Reciprocal-space coordinates, in inverse angstrom.
    wavelength : float
        Electron wavelength, in angstrom.
    semiangle : float
        Probe-forming semi-convergence angle, in radians.

    Returns
    -------
    torch.Tensor
        ``1.0`` inside the aperture and ``0.0`` outside.
    """
    # ``asin(lambda |q|) < alpha`` is the same test as the Warp helper's
    # ``lambda |q| < sin(alpha)`` for ``alpha <= pi/2`` (the sine increases on
    # [0, pi/2]); only a point within float32 rounding of the rim can fall on
    # the other side. The clamp only guards the domain for frequencies beyond
    # 1/wavelength, which are outside any physical aperture and so map to 0.0
    # either way.
    theta = torch.asin((torch.sqrt(qx * qx + qy * qy) * wavelength).clamp(max=1.0))
    return (theta < semiangle).to(torch.float32)


def _gamma(
    k_y: torch.Tensor,
    k_x: torch.Tensor,
    qy_rot: torch.Tensor,
    qx_rot: torch.Tensor,
    wavelength: float,
    semiangle: float,
    aberrations: torch.Tensor,
) -> torch.Tensor:
    """Double-overlap function ``Gamma(Q, K)`` for one bright-field chunk.

    Parameters
    ----------
    k_y, k_x : torch.Tensor
        Bright-field pixel coordinates, shaped to broadcast against the Q grid.
    qy_rot, qx_rot : torch.Tensor
        Object-frequency grid, already rotated into detector coordinates.
    wavelength : float
        Electron wavelength, in angstrom.
    semiangle : float
        Probe-forming semi-convergence angle, in radians.
    aberrations : torch.Tensor
        The twelve aberration coefficients, in angstrom.

    Returns
    -------
    torch.Tensor
        ``conj(A) A_minus - A conj(A_plus)``, complex, one plane per pixel.
    """

    def _lens(qy: torch.Tensor, qx: torch.Tensor) -> torch.Tensor:
        chi = lens_wavefront(aberrations, qy, qx, lambda_a=wavelength)
        return _aperture(qy, qx, wavelength, semiangle) * torch.exp(-1j * chi)

    a_centre = _lens(k_y, k_x)
    a_plus = _lens(k_y + qy_rot, k_x + qx_rot)
    a_minus = _lens(k_y - qy_rot, k_x - qx_rot)
    return torch.conj(a_centre) * a_minus - a_centre * torch.conj(a_plus)


def direct_ptychography_wiener(
    vBF: DatasetVirtualBrightField4DStem,
    upsample_int,
    wiener_alpha: float = DEFAULT_WIENER_ALPHA,
    k_chunk: int = 16,
    phase_sign: PhaseSignMode = "preserve",
    verbosity: int = 0,
) -> torch.Tensor:
    """Assemble the SSB phase image weighting by the double-overlap amplitude.

    Parameters
    ----------
    vBF : DatasetVirtualBrightField4DStem
        Virtual bright field carrying ``G``, the bright-field coordinates
        ``k``, and the acquisition ``meta`` (aberrations, rotation, wavelength,
        semi-convergence angle).
    upsample_int : tuple of int
        Integer upsampling factors ``(uy, ux)`` of the reconstruction grid.
    wiener_alpha : float, optional
        Tikhonov floor added to ``sum_K |Gamma|^2`` as a fraction of that sum's
        maximum, regularising the divide. Default is ``1e-3``.
    k_chunk : int, optional
        Number of bright-field pixels processed per chunk. Trades peak memory
        against launch overhead. Default is 16.
    phase_sign : str, optional
        Sign convention resolved by
        :func:`~scatterem.reconstruction.ptychography.direct._common._orient_phase_image`.
    verbosity : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    torch.Tensor
        The reconstructed phase image, on ``vBF.device``.

    Raises
    ------
    ValueError
        If ``wiener_alpha`` is negative, or if `vBF` carries no acquisition
        metadata to build ``Gamma`` from.
    """
    image, _ = _wiener_image_and_gain(
        vBF, upsample_int, wiener_alpha, k_chunk, phase_sign, verbosity
    )
    return image


def _wiener_image_and_gain(
    vBF: DatasetVirtualBrightField4DStem,
    upsample_int,
    wiener_alpha: float,
    k_chunk: int,
    phase_sign: PhaseSignMode,
    verbosity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`direct_ptychography_wiener`, plus its SSNR gain over phase-only SSB.

    The gain is the per-Q ratio of the wiener estimator's power SNR to the
    phase-only estimator's, for independent equal-variance bright-field noise.
    The wiener SNR goes as ``sum_K |Gamma|^2``. The phase-only SNR goes as
    ``(sum_K |Gamma|)^2 / N_live``, where ``N_live`` counts the K with
    ``Gamma != 0`` (the phase-only kernel gives each of them unit weight). The
    Q-dependent Tikhonov divide scales signal and noise alike, so it cancels.
    By Cauchy-Schwarz the gain is at least 1. It is exactly 1 where ``|Gamma|``
    is the same for all live K, because there the two images agree up to a
    per-Q factor. The gain is 0 where no K is live.
    """
    if wiener_alpha < 0.0:
        raise ValueError(f"wiener_alpha must be non-negative, got {wiener_alpha!r}")
    meta = vBF.meta
    if meta is None or meta.aberrations is None:
        raise ValueError(
            "the virtual bright field carries no acquisition metadata; the "
            "weighted assembly needs aberrations/rotation/wavelength/alpha."
        )
    upsample_int = tuple(upsample_int)
    _t0 = time.perf_counter()
    sums = _wiener_sums(vBF, upsample_int, k_chunk)
    numerator, denominator, abs_sum, n_live = sums
    n_y, n_x = numerator.shape

    # ``Gamma`` is identically zero for an aberration-free in-focus probe
    # (``conj(A) A_minus`` and ``A conj(A_plus)`` cancel term by term), which is
    # correct physics -- SSB has no phase contrast there -- but it makes the
    # denominator all zeros. Keep the relative Tikhonov floor for the normal
    # case and add an absolute one so that degenerate case returns zeros rather
    # than NaN.
    floor = wiener_alpha * denominator.max()
    weighted = numerator / (denominator + floor).clamp(
        min=torch.finfo(torch.float32).tiny
    )
    scale = math.sqrt(upsample_int[0] * upsample_int[1]) / math.sqrt(n_y * n_x)
    image = torch.fft.ifft2(weighted, norm="forward").imag * scale

    if verbosity > 0:
        logger.info(
            "Time to reconstruct direct ptychography image (wiener): %.3f s",
            time.perf_counter() - _t0,
        )
    live = abs_sum > 0
    gain = torch.where(
        live,
        n_live
        * denominator
        / abs_sum.square().clamp(min=torch.finfo(torch.float32).tiny),
        torch.zeros_like(denominator),
    )
    image = _orient_phase_image(image, phase_sign=phase_sign, verbosity=verbosity)
    return image, gain


def rotated_q_grid(
    vBF: DatasetVirtualBrightField4DStem, upsample_int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The upsampled object-frequency grid, rotated into detector coordinates.

    The same rotation as the Warp correction kernel: ``meta.rotation`` in
    degrees.

    Parameters
    ----------
    vBF : DatasetVirtualBrightField4DStem
        Virtual bright field with acquisition metadata.
    upsample_int : tuple of int
        Integer upsampling factors ``(uy, ux)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(qy_rot, qx_rot)``, each ``(Ny Uy, Nx Ux)``, corner-origin order.
    """
    _, qy_1d, qx_1d = _upsampled_grid(vBF, upsample_int)
    device = vBF.device
    rotation = math.radians(float(_require_meta(vBF).rotation))
    cos_rot, sin_rot = math.cos(rotation), math.sin(rotation)
    qy_2d, qx_2d = torch.meshgrid(qy_1d.to(device), qx_1d.to(device), indexing="ij")
    qx_rot = qx_2d * cos_rot - qy_2d * sin_rot
    qy_rot = qx_2d * sin_rot + qy_2d * cos_rot
    return qy_rot, qx_rot


def _wiener_sums(
    vBF: DatasetVirtualBrightField4DStem, upsample_int, k_chunk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-Q sums over K of the wiener assembly, on the upsampled grid.

    Returns ``(sum G conj(Gamma), sum |Gamma|^2, sum |Gamma|, N_live)``, where
    ``N_live`` counts the K with ``Gamma != 0``.
    """
    meta = _require_meta(vBF)
    aberrations = _probe_aberrations(vBF)
    wavelength = float(meta.wavelength)
    semiangle = float(meta.semiconvergence_angle)
    device = vBF.device

    qy_rot, qx_rot = rotated_q_grid(vBF, upsample_int)
    n_y, n_x = qy_rot.shape

    kx_all = vBF.k[:, 1].contiguous()
    ky_all = vBF.k[:, 0].contiguous()
    n_bf = int(vBF.n_bright_field)

    # ``G`` is stored on the UN-upsampled Q grid; reading it modulo its own
    # extents is what the upsampling tile means (see the Warp kernel's note).
    g_y, g_x = int(vBF.array.shape[0]), int(vBF.array.shape[1])
    tile_y = torch.arange(n_y, device=device) % g_y
    tile_x = torch.arange(n_x, device=device) % g_x

    numerator = torch.zeros((n_y, n_x), dtype=torch.complex64, device=device)
    denominator = torch.zeros((n_y, n_x), dtype=torch.float32, device=device)
    abs_sum = torch.zeros((n_y, n_x), dtype=torch.float32, device=device)
    n_live = torch.zeros((n_y, n_x), dtype=torch.float32, device=device)

    for start in range(0, n_bf, k_chunk):
        stop = min(start + k_chunk, n_bf)
        chunk = vBF.get_G_chunk(start, stop)
        # ``get_G_chunk`` hands back [Nqy, Nqx, Nk]; the arithmetic below wants
        # the bright-field axis outermost.
        if chunk.shape[-1] == stop - start:
            chunk = chunk.permute(2, 0, 1)
        g_tiled = chunk[:, tile_y][:, :, tile_x]

        k_y = ky_all[start:stop].view(-1, 1, 1)
        k_x = kx_all[start:stop].view(-1, 1, 1)
        gamma = _gamma(k_y, k_x, qy_rot, qx_rot, wavelength, semiangle, aberrations)
        numerator += (g_tiled * torch.conj(gamma)).sum(0)
        gamma_abs = gamma.abs()
        denominator += gamma_abs.square().sum(0)
        abs_sum += gamma_abs.sum(0)
        n_live += (gamma_abs > 0).sum(0)
        del chunk, g_tiled, gamma, gamma_abs
    if (n_y, n_x) != (g_y, g_x):
        # Upsampled, the scan-lattice rows read G(K, 0), each pixel's scan DC
        # with the lattice content folded onto it: 0 by definition.
        zero_scan_lattice(numerator, (g_y, g_x))
    return numerator, denominator, abs_sum, n_live


def reconstruct_wiener_weighted(
    dataset,
    vBF: DatasetVirtualBrightField4DStem,
    upsample: float | str,
    upsample_int,
    wiener_alpha: float,
    phase_sign: PhaseSignMode,
    return_snr: bool,
    ssnr_method: str,
    verbosity: int,
    bright_field_mask_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The ``gamma_weighting="wiener"`` branch of :func:`direct_ptychography`.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Dataset the SSNR is derived from when `return_snr` is True.
    vBF : DatasetVirtualBrightField4DStem
        Virtual bright field the assembly reads.
    upsample : float or str
        Upsampling request, forwarded to the analytical SSNR.
    upsample_int : tuple of int
        Resolved integer upsampling factors.
    wiener_alpha : float
        Tikhonov floor, see :func:`direct_ptychography_wiener`.
    phase_sign : str
        Phase-sign convention.
    return_snr : bool
        Whether to also return the analytical SSNR of the wiener image: the
        phase-only analytical SSNR times the per-Q gain of the wiener weights.
    ssnr_method : str
        SSNR method requested by the caller.
    verbosity : int
        Verbosity level.
    bright_field_mask_threshold : float, optional
        Threshold of the bright-field mask that `vBF` was built with. The
        phase-only SSNR uses the same mask, so both factors of the SSNR sum
        over the same K. ``None`` takes the 0.5 default of
        :func:`~scatterem.reconstruction.ptychography.direct.ctf.direct_ptychography_ssnr`.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor or None)
        The phase image and, when requested, its analytical SSNR.

    Raises
    ------
    ValueError
        If ``ssnr_method="empirical"`` is requested alongside this weighting.
    """
    # The empirical SSNR reconstructs its own checkerboard half-split through
    # the Warp path, so it would silently report the SSNR of the phase-only
    # estimator for a wiener image. Refuse rather than mislead.
    if return_snr and ssnr_method == "empirical":
        raise ValueError(
            "gamma_weighting='wiener' does not support ssnr_method='empirical': "
            "the half-split reconstructions run the phase-only assembly, so the "
            "returned SSNR would not describe the returned image."
        )
    phase_image, gain = _wiener_image_and_gain(
        vBF,
        upsample_int,
        wiener_alpha,
        k_chunk=16,
        phase_sign=phase_sign,
        verbosity=verbosity,
    )
    if not return_snr:
        return phase_image, None
    from .ctf import direct_ptychography_ssnr

    # The phase-only analytical SSNR times the per-Q gain of the wiener weights
    # (see ``_wiener_image_and_gain``). The phase-only SSNR alone describes a
    # different image: on a filled aperture it is 5.4x lower at one Q.
    with torch.no_grad():
        snr = direct_ptychography_ssnr(
            dataset,
            upsample=upsample,
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
        )
        snr = snr * gain.to(snr.device)
    return phase_image, snr
