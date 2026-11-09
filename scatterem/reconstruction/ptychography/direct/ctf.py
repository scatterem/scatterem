"""Phase-contrast transfer function and the analytic SSNR derived from it."""

import numpy as np
import torch

from scatterem.data.datasets import (
    Dataset4DStem,
    DatasetVirtualBrightField4DStem,
)
from scatterem.nn.functional.ptychography import (
    phase_contrast_transfer_function as _phase_contrast_transfer_function,
)
from scatterem.nn.functional.ptychography._ctf_unnormalized import (
    normalize_pctf as _normalize_pctf,
)
from scatterem.nn.functional.ptychography._ctf_unnormalized import (
    pctf_unnormalized as _pctf_unnormalized,
)
from scatterem.utils.spectral_mesh import dual_mesh
from scatterem.utils.lazy_plotting import plt, vis
from scatterem.utils.transfer import (
    double_and_triple_pixel_counts_torch,
)

from ._common import (
    _probe_aberrations,
    _require_meta,
    _resolve_upsample_int,
    _resolve_vbf,
    zero_scan_lattice,
)
from ._ssnr_analytical_warp import ssnr_epilogue


def phase_contrast_transfer_function(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    upsample: float | str = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    defocus_offset: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the phase-contrast transfer function (PCTF) for direct ptychography.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Dataset (or virtual bright-field dataset derived from one) whose
        acquisition geometry and aberrations parameterize the PCTF.
    upsample : float or str, optional
        Upsampling factor for the diffraction pattern, or ``"nyquist"`` to
        resolve it from the acquisition geometry.
    bright_field_mask_threshold : float, optional
        Threshold used to resolve the virtual bright field when `dataset` is
        not already one.
    verbosity : int, optional
        Verbosity level.
    defocus_offset : float, optional
        Added to the fitted defocus (``aberrations[0]``, Angstrom) for this
        evaluation only; the metadata is not modified. Without an
        offset, the probe has the aberrations ``meta.aberrations`` with C10
        less ``meta.defocus_guess``, as in :func:`direct_ptychography`.
    **kwargs : Any
        Forwarded to the virtual-bright-field resolution helper (e.g.
        ``n_batches``, the number of batches used to build the vBF).

    Returns
    -------
    torch.Tensor
        The phase-contrast transfer function, corner-centered (non-shifted).
        It is 0 on the scan-lattice frequencies ``Q = (m / dy, n / dx)``
        (``Q = 0`` at an upsampling of 1), where the phase-only images are 0.
    """
    pctf = _phase_contrast_transfer_function(
        None,
        *_pctf_arguments(
            dataset, upsample, bright_field_mask_threshold, verbosity, defocus_offset
        ),
    )
    return zero_scan_lattice(pctf, tuple(dataset.shape[:2]))


def _pctf_arguments(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    upsample: float | str = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    defocus_offset: float = 0.0,
) -> tuple:
    """What :func:`phase_contrast_transfer_function` passes to the ``nn.functional``
    PCTF after its ``G``: ``(aberrations, rotation, semiconvergence_angle,
    wavelength, Qx, Qy, Kx, Ky)``."""
    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)
    ds_rotation = torch.tensor(_require_meta(vBF).rotation, device=vBF.device)
    semiconvergence_angle = _require_meta(vBF).semiconvergence_angle
    wavelength = _require_meta(vBF).wavelength

    # The PCTF kernel never reads G's VALUES -- gamma depends only on Q, K, the
    # aberrations, the aperture and the wavelength -- so the U^2-tiled copy of G
    # this used to build (``torch.tile(vBF.G, (Uy, Ux, 1))``) was a pure
    # allocate-and-copy of up to 7.7 GiB per call at the Fig1 config.  Only the
    # SHAPE was ever used, and that shape is (len(Qy), len(Qx), len(Kx)).
    # NOTE this is the PCTF only: the reconstruction's own q-tiling of G (see
    # ``_direct_ptychography_*``) is load-bearing -- it is how single-sideband
    # ptychography reaches past the scan Nyquist -- and must not be removed.
    # Shapes from the vBF stack, not vBF.G: the streaming provider may keep no
    # resident G.
    new_shape = (
        int(round(vBF.array.shape[0] * upsample_int[0])),
        int(round(vBF.array.shape[1] * upsample_int[1])),
    )
    Qy, Qx = vBF.get_q_1d(new_shape)
    aberrations = _probe_aberrations(vBF)
    if defocus_offset:
        aberrations = aberrations.detach().clone()
        aberrations[0] += defocus_offset
    Kx = vBF.k[:, 1]
    Ky = vBF.k[:, 0]
    return (
        aberrations,
        ds_rotation,
        semiconvergence_angle,
        wavelength,
        Qx,
        Qy,
        Kx,
        Ky,
    )


def direct_ptychography_ssnr(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    upsample: float | str = "nyquist",
    verbosity: int = 0,
    bright_field_mask_threshold: float | None = None,
) -> torch.Tensor:
    """
    Compute the analytical spectral signal-to-noise ratio (SSNR) for direct ptychography.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Dataset (or virtual bright-field dataset derived from one) to compute
        the analytical SSNR for.
    upsample : float or str, optional
        Upsampling factor for the diffraction pattern, or ``"nyquist"`` to
        resolve it from the acquisition geometry.
    verbosity : int, optional
        Verbosity level. If > 1, plots diagnostics.
    bright_field_mask_threshold : float, optional
        Bright-field mask threshold the PCTF is built from. Left ``None`` this
        takes :func:`phase_contrast_transfer_function`'s own default, which is
        what every caller of this function has always received.

    Returns
    -------
    torch.Tensor
        The analytical SSNR in corner-centered (non-shifted) layout. It is 0
        on the scan-lattice frequencies, as the PCTF is.
    """
    # The threshold is FORWARDED only when the caller names one. Defaulting it to
    # the dataset's own value would change the bright-field mask the PCTF is built
    # from, hence this SSNR, hence the published fused image -- the physics decision
    # ``Dataset4DStem.get_vbf``'s docstring deliberately leaves to the caller. This
    # parameter is how a caller takes it; omitting it changes nothing.
    pctf_threshold = (
        {}
        if bright_field_mask_threshold is None
        else {"bright_field_mask_threshold": bright_field_mask_threshold}
    )
    # ``phase_contrast_transfer_function(dataset, verbosity=1, ...)`` split at its
    # bright-field normalisation, so the fused epilogue can do that divide too.
    pctf_args = _pctf_arguments(dataset, upsample, verbosity=1, **pctf_threshold)
    pctf_raw = _pctf_unnormalized(None, *pctf_args)
    ny, nx = pctf_raw.shape
    shape = (int(ny), int(nx))
    # Analytical Eq.13 form is an amplitude, dose- and object-independent quantity.
    # The empirical channels are POWER ratios proportional to dose. Square to a power
    # ratio and scale by the real fluence so the fallback shares the channels'
    # reference (object power |Psi_s|^2 is still assumed flat — fallback limitation).
    # fluence_per_probe exists only on Dataset4DStem, not the vBF sibling; fall back
    # to 1.0 (per-unit-dose) when a vBF is passed directly.
    fluence = float(getattr(dataset, "fluence_per_probe", 1.0))
    if verbosity <= 1 and pctf_raw.is_cuda:
        # Everything from here to the return as two bit-exact Warp launches
        # instead of ~20 small ATen ops on a host-bound path;
        # ``None`` = the ``ssnr_epilogue`` backend row declined.
        snr = _fused_ssnr(dataset, pctf_raw, pctf_args, shape, fluence)
        if snr is not None:
            return zero_scan_lattice(snr, tuple(dataset.shape[:2]))
    _, _, sc_angle, wavelength, _, _, Kx, Ky = pctf_args
    pctf_corner_center = _normalize_pctf(pctf_raw, Kx, Ky, sc_angle, wavelength)
    # The noise grid is built on the SAME upsample-aware axis as the PCTF and stays
    # on the dataset's device -- no GPU->CPU->GPU round-trip.
    ptycho_noise_2d = _analytical_ptycho_noise_2d(dataset, shape)
    snr_ptycho_analytical = _ssnr_from_pctf(
        pctf_corner_center, ptycho_noise_2d, fluence
    )

    if verbosity > 1:
        pctf = torch.fft.fftshift(pctf_corner_center)
        N2, N3 = _overlap_pixel_counts(dataset, shape)
        nconf = vis.NormalizationConfig(interval_type="manual", vmin=0, vmax=1)
        vis.show_2d([pctf], cbar=True, title=["pctf"], norm=nconf)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(N2.reshape(shape).cpu().numpy())
        ax[1].imshow(N3.reshape(shape).cpu().numpy())
        ax[2].imshow(pctf_corner_center.cpu().numpy())
        ax[3].imshow((ptycho_noise_2d == 0).cpu().numpy())
        plt.show()
        plt.tight_layout()

        p2 = torch.fft.fftshift(snr_ptycho_analytical).cpu().numpy()
        p2s = p2.shape[0] // 4
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        imax = ax.imshow(p2[p2s:-p2s, p2s:-p2s], cmap="magma")
        plt.colorbar(imax)
        plt.title("snr_ptycho_analytical")
        plt.tight_layout()
        plt.show()

    # The scan lattice: Q = 0 has no transfer, and at an upsampling of 2 or
    # more the other rows read the scan DC, which the image drops.
    return zero_scan_lattice(snr_ptycho_analytical, tuple(dataset.shape[:2]))


def _overlap_pixel_counts(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N2, N3)``: double- and triple-overlap pixel counts on the upsampled Q grid
    ``shape``, flattened, on ``dataset.device``."""
    qn = _radial_q(dataset, shape)
    R = _require_meta(dataset).semiconvergence_angle / _require_meta(dataset).wavelength
    return double_and_triple_pixel_counts_torch(qn.view(-1), R, dataset.sampling[-1])


def _radial_q(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem, shape: tuple[int, int]
) -> torch.Tensor:
    """``|q|`` on the upsampled Q grid ``shape``, corner origin, on ``dataset.device``."""
    upsample = np.array(shape) / np.array(dataset.shape[:2])
    q = dual_mesh(
        shape,
        np.array(dataset.sampling[:2]) / upsample,
        staggered=False,
        device=dataset.device,
    )
    return torch.linalg.vector_norm(q, dim=0)


def _fused_ssnr(dataset, pctf_raw, pctf_args, shape, fluence) -> torch.Tensor | None:
    """:func:`direct_ptychography_ssnr`'s tail through ``_ssnr_analytical_warp``,
    from the same operands the torch tail reads, or ``None`` if declined."""
    meta = _require_meta(dataset)
    dalpha0 = dataset.sampling[-1] * meta.wavelength
    rBF = meta.semiconvergence_angle / dalpha0
    _, _, sc_angle, wavelength, _, _, Kx, Ky = pctf_args
    return ssnr_epilogue(
        pctf_raw,
        Kx,
        Ky,
        sc_angle,
        wavelength,
        _radial_q(dataset, shape),
        meta.semiconvergence_angle / meta.wavelength,
        dataset.sampling[-1],
        np.pi * rBF**2,
        fluence,
    )


def _analytical_ptycho_noise_2d(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem, shape: tuple[int, int]
) -> torch.Tensor:
    """``sqrt((N2 + N3) / N_alpha)`` on the upsampled Q grid ``shape``, float32.

    The DEFOCUS-INVARIANT denominator of the analytical ptychography SSNR: it reads
    only ``sampling``, ``shape[:2]``, ``semiconvergence_angle`` and ``wavelength``.
    That is why every bit of the analytical SSNR's defocus dependence lives in the
    PCTF, and why a depth sweep builds this ONCE while
    :func:`direct_ptychography_ssnr` builds it per call.
    """
    N2, N3 = _overlap_pixel_counts(dataset, shape)
    dalpha0 = dataset.sampling[-1] * _require_meta(dataset).wavelength
    rBF = _require_meta(dataset).semiconvergence_angle / dalpha0
    noise_squared = (N2 + N3).reshape(tuple(shape)) / (np.pi * rBF**2)
    return torch.sqrt(noise_squared).to(device=dataset.device, dtype=torch.float32)


def _ssnr_from_pctf(
    pctf: torch.Tensor, noise_2d: torch.Tensor, fluence: float
) -> torch.Tensor:
    """``(PCTF / noise)^2 * fluence``, zeroed where the noise vanishes (the ``inf``
    that ``noise == 0`` produces is created and then overwritten)."""
    ssnr = (pctf / noise_2d) ** 2 * fluence
    ssnr[noise_2d == 0] = 0
    return ssnr


def _pctf_on_grid(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    upsample: float | str,
    bright_field_mask_threshold: float,
    verbosity: int,
    shape: tuple[int, int],
) -> torch.Tensor:
    """``phase_contrast_transfer_function`` with the CALLER's bright-field threshold.

    Deliberately not :func:`direct_ptychography_ssnr`'s spelling, which omits the
    argument and so takes this module's own 0.5 default: on a dataset run at any
    other threshold that builds a SECOND vBF (``get_vbf`` is keyed on the threshold)
    from a different bright-field mask.
    """
    pctf = phase_contrast_transfer_function(
        dataset,
        upsample=upsample,
        bright_field_mask_threshold=bright_field_mask_threshold,
        verbosity=verbosity,
    )
    if tuple(pctf.shape) != tuple(shape):
        raise ValueError(
            f"phase_contrast_transfer_function returned {tuple(pctf.shape)}, "
            f"expected {tuple(shape)}; the SSNR and the reconstruction "
            f"must share one grid."
        )
    return pctf
