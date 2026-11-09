"""Single-side-band direct ptychography and its empirical (half-split) SSNR."""

import logging
import time
import warnings

import numpy as np
import torch

from scatterem.data.datasets import Dataset4DStem, DatasetVirtualBrightField4DStem

from . import _config
from ._chunking import _iter_chunk_images, _upsampled_grid
from ._common import (
    PhaseSignMode,
    _orient_phase_image,
    _probe_aberrations,
    _require_meta,
    _resolve_upsample_int,
    _resolve_vbf,
    remove_scan_lattice,
    zero_ssnr_scan_lattice,
)
from ._gamma_weighting import (
    DEFAULT_WIENER_ALPHA,
    GammaWeighting,
    reconstruct_wiener_weighted,
)
from .ctf import direct_ptychography_ssnr

logger = logging.getLogger(__name__)


def _checkerboard_parity(bright_field_inds: torch.Tensor) -> torch.Tensor:
    """Checkerboard parity of integer detector-grid indices.

    ``bright_field_inds`` are integer (row, col) detector-pixel coordinates
    (``vBF.bright_field_inds_ordered_by_radius``), NOT ``vBF.k`` (which is
    mean-centered floats scaled by the reciprocal sampling — rounding it would
    collapse to one subset). Absolute vs centered indices is immaterial: a
    half-pixel offset only flips which subset is labeled True/False, which does
    not matter for a symmetric half-split.

    Returns ``True`` for the odd colour class (``row + col`` odd), one entry per
    ROW, decided by the index VALUES — so a radius-ordered (neither raster-
    ordered nor contiguous) index list is coloured correctly, unlike anything
    keyed on position within the list.

    Non-integral floats are rejected instead of truncated: casting ``vBF.k``
    would fold distinct rows onto one parity and return a lopsided split, i.e.
    a plausible-looking but wrong SSNR rather than an error.

    No third-party attribution is owed here. The checkerboard two-colouring of
    the integer lattice *is* the parity of ``row + col``; that is the definition
    of a checkerboard, not an implementation of one, so any independent
    implementation necessarily coincides in its arithmetic and there is no
    expression to attribute to anybody.
    """
    if bright_field_inds.numel() == 0:
        # No BF pixels at all: hand back an empty mask so the caller trips its
        # own too-small guard (warn + analytical SSNR) instead of an IndexError.
        return torch.zeros(0, dtype=torch.bool, device=bright_field_inds.device)
    if bright_field_inds.ndim != 2 or bright_field_inds.shape[1] != 2:
        raise ValueError(
            "bright_field_inds must be an (N, 2) tensor of integer (row, col) "
            f"detector indices; got shape {tuple(bright_field_inds.shape)}"
        )
    idx = bright_field_inds
    if idx.is_floating_point():
        rounded = torch.round(idx)
        if not torch.equal(rounded, idx):
            raise ValueError(
                "bright_field_inds must hold whole-number detector-pixel "
                "indices, but got non-integral floats — this is what passing "
                "vBF.k (mean-centered and scaled by the reciprocal sampling) "
                "looks like. Pass vBF.bright_field_inds_ordered_by_radius."
            )
        idx = rounded
    idx = idx.long()
    # Two lattice cells share a colour exactly when their row and column
    # indices agree in parity, so the colour is the disagreement of the two low
    # bits. Two's complement makes `& 1` the parity of negative (centered)
    # indices too, and this equals (row + col) % 2 for every input.
    return torch.ne(idx[:, 0] & 1, idx[:, 1] & 1)


def _assemble_phase_only(
    vBF: DatasetVirtualBrightField4DStem,
    upsample_int,
    n_batches: int,
    phase_sign: PhaseSignMode,
    verbosity: int,
) -> torch.Tensor:
    """Sum the aberration-corrected chunk images, the phase-only assembly.

    This is the shipped estimator: the Warp kernel applies the UNIT-MODULUS
    ``conj(Gamma / |Gamma|)``, so the only normalisation left here is the
    bright-field pixel count. Upsampled, the chunk images read each pixel's
    scan DC on the scan-lattice frequencies, which their sum drops
    (``_common.remove_scan_lattice``). See :mod:`._gamma_weighting` for
    the weighted alternative.

    Parameters
    ----------
    vBF : DatasetVirtualBrightField4DStem
        Virtual bright field supplying ``G`` and the acquisition metadata.
    upsample_int : tuple of int
        Resolved integer upsampling factors.
    n_batches : int
        Number of bright-field chunks.
    phase_sign : str
        Phase-sign convention.
    verbosity : int
        Verbosity level.

    Returns
    -------
    torch.Tensor
        The reconstructed phase image.
    """
    device = vBF.device
    new_shape, Qy, Qx = _upsampled_grid(vBF, upsample_int)
    # Handed on as a PYTHON FLOAT, not wrapped here.  ``_sin_cos_rot`` builds the
    # tensor with exactly this ``torch.as_tensor(..., float32, device)`` call, but
    # its cache for a scalar rotation is keyed BY VALUE, whereas wrapping it here
    # produced a fresh object per call and routed it down the IDENTITY-keyed branch
    # -- which cannot hit.  That miss then cascaded: a new ``[sin, cos]`` tensor per
    # call also missed ``_sin_cos_floats``, whose miss reads the two scalars back
    # with ``.item()``, i.e. two blocking ``cudaStreamSynchronize`` per call.
    rotation_t = float(_require_meta(vBF).rotation)
    aberrations = _probe_aberrations(vBF)

    with torch.no_grad():
        _t0 = time.perf_counter()
        phase_sum = torch.zeros(new_shape, device=device)
        for _, _, img in _iter_chunk_images(
            vBF,
            aberrations,
            rotation_t,
            Qy,
            Qx,
            upsample_int,
            n_batches,
            "sum",
        ):
            phase_sum += img
        # Pixel-count weighting: one global divide. The old
        # `phase_image /= n_batches_eff` averaged per-batch MEANS equally,
        # overweighting a short tail batch (29 vs 34 px at n_batches=64 with
        # 2171 BF pixels -> 1.17x tail overweight).  The sum's scan-lattice
        # frequencies (they read G(K, 0)) are 0 by definition; at U = 1
        # that is Q = 0, which CUDA leaves at a residue with orders above C12.
        remove_scan_lattice(phase_sum, upsample_int, aberrations)
        phase_image = phase_sum / vBF.n_bright_field
        phase_image = _orient_phase_image(
            phase_image, phase_sign=phase_sign, verbosity=verbosity
        )
        if verbosity > 0:
            logger.info(
                "Time to reconstruct directptychography image: %.3f s",
                time.perf_counter() - _t0,
            )
    return phase_image


def direct_ptychography(
    dataset: Dataset4DStem | DatasetVirtualBrightField4DStem,
    upsample: float | str = "nyquist",
    bright_field_mask_threshold: float = 0.5,
    verbosity: int = 0,
    n_batches: int = 25,
    return_snr: bool = False,
    phase_sign: PhaseSignMode = "preserve",
    ssnr_method: str = "analytical",
    gamma_weighting: GammaWeighting = "phase",
    wiener_alpha: float = DEFAULT_WIENER_ALPHA,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reconstruct a phase image with single-side-band direct ptychography.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Dataset containing the diffraction pattern. The probe has the
        aberrations ``meta.aberrations`` with C10 less ``meta.defocus_guess``,
        as in iterative ptychography and WDD.
    upsample : float or str, optional
        Upsampling factor for the diffraction pattern, or "nyquist" to derive
        it from the data. Default is "nyquist". At 2 or more (on either axis)
        the image and its SSNR are 0 on the scan-lattice frequencies ``Q = (m
        / dy, n / dx)``, which read each bright-field pixel's scan DC. At 1
        the one such frequency is ``Q = 0``: with an aberration above C12 the
        image mean is taken out there.
    bright_field_mask_threshold : float, optional
        Threshold for the bright-field mask. Default is 0.5.
    verbosity : int, optional
        Verbosity level. Default is 0.
    n_batches : int, optional
        Number of batches for the virtual bright-field pass. Default is 25.
    return_snr : bool, optional
        Whether to compute the ptychography SSNR (method selected by
        `ssnr_method`). Default is False.
    phase_sign : {"preserve", "positive", "negative"}, optional
        "preserve" keeps the sign the forward model gives: positive phase for
        positive projected potential, fixed by the aberrations and rotation
        (the bright-field-shift fit resolves its (C1, rotation) <->
        (-C1, rotation + 180 deg) branch with the rotation prior), so it does
        not depend on the image. "positive"/"negative" flip the image to
        positive/negative skewness; that is data dependent and can invert a
        correct image (e.g. low-contrast weak-phase specimens). Default is
        "preserve".
    ssnr_method : str, optional
        How to compute the SSNR when `return_snr` is True. "analytical"
        (default) uses the phase-contrast-transfer-function noise model
        (CTF-aware: ptycho SSNR -> 0 at low q, no DC/low-q transfer);
        "empirical" reconstructs a checkerboard BF half-split and forms the
        raw half-difference power ratio (data-driven, better at high dose;
        at an upsampling of 2 or more it is 0 on the scan-lattice frequencies,
        as the image is). Below ``_config._MIN_HALFSPLIT_PIXELS`` bright-field
        pixels per half it warns and returns the analytical SSNR, also 0 there.
    gamma_weighting : {"phase", "wiener"}, optional
        How the double-overlap function ``Gamma(Q, K)`` weights each
        bright-field pixel. ``"phase"`` (default, and the only behaviour before
        this argument existed) normalises ``Gamma`` to unit modulus and applies
        its phase alone, dividing the sum by the bright-field pixel count.
        ``"wiener"`` keeps ``|Gamma|`` in the numerator and divides by the
        Q-dependent ``sum_K |Gamma|^2``, which down-weights the upsampling
        replica (the aliased copies of the scan spectrum) -- measured at the
        multiples of the scan sampling frequency ``1/dr`` on a test case,
        124x -> 11.8x the local background against the phase-only image of
        the time. Both images are now 0 on those frequencies, which read the
        scan DC. It runs a slower plain-torch assembly (see
        :mod:`._gamma_weighting`) and is incompatible with
        ``ssnr_method="empirical"``. Its analytical SSNR is the phase-only one
        times the per-Q gain of the wiener weights, both on the
        `bright_field_mask_threshold` mask, so it describes the wiener image.
    wiener_alpha : float, optional
        Tikhonov floor on ``sum_K |Gamma|^2``, as a fraction of its maximum.
        Only used when ``gamma_weighting="wiener"``. Default is ``1e-3``.
    **kwargs
        ``snr_blur_sigma`` is forwarded to the empirical SSNR path.

    Returns
    -------
    phase_image : torch.Tensor
        The reconstructed phase image (imaginary part).
    snr_ptycho : torch.Tensor or None
        The SNR of the reconstructed phase image if `return_snr` is True,
        otherwise None.

    Raises
    ------
    ValueError
        If the dataset is not a valid Dataset4DStem object, if `upsample`
        is invalid, if `gamma_weighting` is unknown, or if
        ``gamma_weighting="wiener"`` is combined with
        ``ssnr_method="empirical"``.

    Examples
    --------
    On a tiny bright-field-disk cube, ``upsample=2`` doubles the scan grid;
    ``return_snr=False`` leaves the second return ``None``; ``phase_sign``
    ``"positive"`` and ``"negative"`` are exact opposites:

    >>> import numpy as np, torch
    >>> import warp
    >>> warp.config.log_level = warp.LOG_WARNING  # module-load chatter depends on the cache
    >>> from scatterem.data import Aberrations
    >>> from scatterem.data.datasets import Dataset4DStem
    >>> from scatterem.reconstruction.ptychography.direct import (
    ...     direct_ptychography,
    ... )
    >>> n_scan, n_det = 8, 16
    >>> rng = np.random.default_rng(0)
    >>> ky, kx = np.meshgrid(
    ...     np.arange(n_det) - n_det // 2,
    ...     np.arange(n_det) - n_det // 2,
    ...     indexing="ij",
    ... )
    >>> disk = (np.hypot(ky, kx) <= 4).astype(np.float32)
    >>> ry, rx = np.meshgrid(
    ...     np.arange(n_scan), np.arange(n_scan), indexing="ij"
    ... )
    >>> mod = 1 + 0.15 * np.cos(2 * np.pi * ry / n_scan) * np.cos(
    ...     2 * np.pi * rx / n_scan
    ... )
    >>> cube = disk[None, None] * mod[:, :, None, None].astype(np.float32)
    >>> cube = np.clip(
    ...     cube + 0.01 * rng.standard_normal(cube.shape).astype(np.float32),
    ...     0,
    ...     None,
    ... )
    >>> ab = Aberrations(array=torch.zeros(12))
    >>> ab.array[0] = -2.2e4
    >>> ds = Dataset4DStem.from_array(
    ...     np.ascontiguousarray(cube),
    ...     energy=300e3,
    ...     semiconvergence_angle=4e-3,
    ...     scan_step=29.0,
    ...     reciprocal_step=8.3e-3,
    ...     transform_to_amplitudes=True,
    ...     aberrations=ab,
    ...     device=torch.device("cpu"),
    ... )
    >>> phase, snr = direct_ptychography(
    ...     ds, upsample=2, n_batches=4, return_snr=False, phase_sign="preserve"
    ... )
    >>> tuple(phase.shape), snr is None
    ((16, 16), True)
    >>> _, snr = direct_ptychography(
    ...     ds, upsample=2, n_batches=4, return_snr=True, phase_sign="preserve"
    ... )
    >>> tuple(snr.shape)
    (16, 16)
    >>> pos, _ = direct_ptychography(
    ...     ds, upsample=2, n_batches=4, phase_sign="positive"
    ... )
    >>> neg, _ = direct_ptychography(
    ...     ds, upsample=2, n_batches=4, phase_sign="negative"
    ... )
    >>> bool(torch.allclose(pos, -neg))
    True
    """
    if gamma_weighting not in ("phase", "wiener"):
        raise ValueError(
            f"gamma_weighting must be 'phase' or 'wiener', got {gamma_weighting!r}"
        )
    upsample_int = _resolve_upsample_int(dataset, upsample, verbosity=verbosity)
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)

    if gamma_weighting == "wiener":
        return reconstruct_wiener_weighted(
            dataset,
            vBF,
            upsample=upsample,
            upsample_int=upsample_int,
            wiener_alpha=wiener_alpha,
            phase_sign=phase_sign,
            return_snr=return_snr,
            ssnr_method=ssnr_method,
            verbosity=verbosity,
            bright_field_mask_threshold=bright_field_mask_threshold,
        )

    # Empirical SSNR owns the reconstruction (checkerboard half-split). Passing the
    # already-built vBF avoids rebuilding it. The half reconstructions call back with
    # return_snr=False, so they fall through to the normal loop (no recursion).
    if return_snr and ssnr_method == "empirical":
        return direct_ptychography_empirical_ssnr(
            vBF,
            upsample=upsample,
            n_batches=n_batches,
            bright_field_mask_threshold=bright_field_mask_threshold,
            snr_blur_sigma=kwargs.get("snr_blur_sigma", 0.0),
            phase_sign=phase_sign,
            verbosity=verbosity,
        )

    phase_image = _assemble_phase_only(
        vBF, upsample_int, n_batches, phase_sign, verbosity
    )
    with torch.no_grad():
        if return_snr:  # ssnr_method == "empirical" already returned above
            if ssnr_method == "analytical":
                snr_ptycho = direct_ptychography_ssnr(
                    dataset, upsample=upsample, verbosity=verbosity
                )
            else:
                raise ValueError(f"Unknown ssnr_method: {ssnr_method!r}")
        else:
            snr_ptycho = None
    return phase_image, snr_ptycho


def direct_ptychography_empirical_ssnr(
    dataset,
    upsample: float | str = "nyquist",
    n_batches: int = 25,
    bright_field_mask_threshold: float = 0.5,
    snr_blur_sigma: float = 0.0,
    phase_sign: "PhaseSignMode" = "preserve",
    verbosity: int = 0,
):
    """Full phase image + empirical SSNR from a checkerboard BF half-split.

    Both halves are reconstructed with identical deterministic operations, so all
    aberration/upsample steps cancel in the half-difference. The SSNR is the raw
    half-split power ratio (NO /sqrt(fluence)); it is the SSNR of the full-dose
    reconstruction and is dose- and object-bearing, matching the tcDF channel.

    Parameters
    ----------
    dataset : Dataset4DStem or DatasetVirtualBrightField4DStem
        Dataset containing the diffraction pattern (or an already-built
        virtual bright field).
    upsample : float or str, optional
        Upsampling factor for the diffraction pattern, or "nyquist" to derive
        it from the data. Default is "nyquist".
    n_batches : int, optional
        Number of batches for the virtual bright-field pass. Default is 25.
    bright_field_mask_threshold : float, optional
        Threshold for the bright-field mask. Default is 0.5.
    snr_blur_sigma : float, optional
        Gaussian blur sigma applied when binning the empirical SSNR.
        Default is 0.0.
    phase_sign : {"positive", "negative", "preserve"}, optional
        Phase sign convention, as in `direct_ptychography`. Default is
        "preserve".
    verbosity : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    full : torch.Tensor
        The reconstructed phase image (imaginary part), combining both halves.
    ssnr_ptycho : torch.Tensor
        The empirical SSNR of the reconstructed phase image, same shape as
        `full`. If either checkerboard half has fewer than
        ``_config._MIN_HALFSPLIT_PIXELS`` bright-field pixels, it warns and
        returns the analytical SSNR of ``direct_ptychography`` instead. At an
        upsampling of 2 or more (on either axis) either SSNR is 0 on the
        scan-lattice frequencies, where the image is 0.
    """

    from scatterem.reconstruction._ssnr import compute_ssnr_from_halfset_images

    # The normal dispatch path passes the pre-built vBF; _resolve_vbf reuses
    # the dataset's cache when handed a raw Dataset4DStem.
    vBF = _resolve_vbf(dataset, bright_field_mask_threshold)

    parity = _checkerboard_parity(vBF.bright_field_inds_ordered_by_radius).to(
        vBF.device
    )
    n_a = int(parity.sum())
    n_b = int((~parity).sum())
    if min(n_a, n_b) < _config._MIN_HALFSPLIT_PIXELS:
        warnings.warn(
            f"checkerboard BF split too small (n_a={n_a}, n_b={n_b}); "
            "falling back to analytical SSNR",
            stacklevel=2,
        )
        return direct_ptychography(
            dataset,
            upsample=upsample,
            n_batches=n_batches,
            return_snr=True,
            phase_sign=phase_sign,
            ssnr_method="analytical",
            verbosity=verbosity,
        )

    # ONE parity-routed pass over shared G chunks: each chunk's per-pixel
    # images are summed into the A or B accumulator by checkerboard parity.
    # No G_A/G_B copies (the old boolean-index splitter duplicated the full
    # ~17 GiB G while the original stayed referenced). Sign is preserved
    # per half so the half-difference is clean (an independent skew-based flip
    # per half would corrupt N=(F_A-F_B)/2); the per-half pixel-count division
    # makes each half identical to a standalone weighted reconstruction.
    upsample_int = _resolve_upsample_int(vBF, upsample, verbosity=0)
    new_shape, Qy, Qx = _upsampled_grid(vBF, upsample_int)
    device = vBF.device
    rotation_t = torch.as_tensor(
        float(_require_meta(vBF).rotation), dtype=torch.float32, device=device
    )
    aberrations = _probe_aberrations(vBF)
    total_A = torch.zeros(new_shape, device=device)
    total_B = torch.zeros(new_shape, device=device)
    with torch.no_grad():
        for s, e, imag in _iter_chunk_images(
            vBF,
            aberrations,
            rotation_t,
            Qy,
            Qx,
            upsample_int,
            n_batches,
            "none",
        ):
            pm = parity[s:e]
            total_A += imag[..., pm].sum(-1)
            total_B += imag[..., ~pm].sum(-1)

    F_A = remove_scan_lattice(total_A, upsample_int, aberrations) / n_a  # no flip
    F_B = remove_scan_lattice(total_B, upsample_int, aberrations) / n_b

    full = (n_a * F_A + n_b * F_B) / (n_a + n_b)
    full = _orient_phase_image(full, phase_sign=phase_sign, verbosity=verbosity)

    # Reconstruction grid upsample factor (same axis math as compute_ssnr expects).
    upsample_grid = np.array(F_A.shape) / np.array(tuple(vBF.array.shape[:2]))
    sampling_up = tuple((np.array(vBF.sampling[:2]) / upsample_grid).tolist())

    ssnr_1d, _, bin_idx = compute_ssnr_from_halfset_images(
        F_A, F_B, sampling_up, gaussian_sigma=snr_blur_sigma, verbosity=verbosity
    )
    ssnr_ptycho = ssnr_1d[bin_idx].reshape(F_A.shape)  # NO /sqrt(fluence)
    return full, zero_ssnr_scan_lattice(ssnr_ptycho, vBF.array.shape[:2], upsample_int)
