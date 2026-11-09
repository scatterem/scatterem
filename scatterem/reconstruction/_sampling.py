"""Grid and metadata helpers shared by direct ptychography, the dark field and
their fusion."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "_fold_defocus_guess",
    "_probe_aberrations",
    "_require_meta",
    "_resolve_upsample_int",
]

# _require_meta and _resolve_upsample_int: moved verbatim from
# ptychography/direct/_common.py.


def _require_meta(dataset):
    """Return ``dataset.meta``, raising if it has not been loaded."""
    meta = dataset.meta
    if meta is None:
        raise ValueError("dataset.meta is not set; load metadata first")
    return meta


def _fold_defocus_guess(dataset, seed: torch.Tensor) -> torch.Tensor:
    """``seed`` with C10 = ``seed[0] - meta.defocus_guess``: the probe's C10.

    Every probe builder reads the metadata this way. With a zero guess
    ``seed`` itself comes back, so identity-keyed caches (the chunk kernels'
    ``_CHI_ZERO_FLAG``) still hit; otherwise a new tensor, and ``seed`` is not
    modified.
    """
    guess = float(_require_meta(dataset).defocus_guess or 0.0)
    if guess != 0.0:
        seed = seed.clone()
        seed[0] = seed[0] - guess
    return seed


def _probe_aberrations(dataset) -> torch.Tensor:
    """The aberrations the probe has: ``meta.aberrations`` with the defocus guess.

    A dataset that carries its defocus only as ``meta.defocus_guess`` (the hub
    datasets and the sparse-count reader) gets a defocused probe. A fit
    writes its result into ``meta.aberrations`` and resets the guess to 0, so
    the defocus counts once.
    """
    meta = _require_meta(dataset)
    if meta.aberrations is None:
        raise ValueError("dataset.meta.aberrations is not set; fit them first")
    return _fold_defocus_guess(dataset, meta.aberrations.array)


def _resolve_upsample_int(source, upsample, verbosity=0):
    """Resolve ``upsample`` ("nyquist" or a number) into the integer per-axis
    factors. ``source`` is any object exposing ``sampling`` and ``meta``
    (a Dataset4DStem or a vBF). Shared by the reconstruction, the empirical
    SSNR — they MUST agree on the grid."""
    if upsample == "nyquist":
        scan_sampling = np.array(source.sampling[:2])
        meta = _require_meta(source)
        nyquist_sampling = meta.wavelength / (4 * meta.semiconvergence_angle)
        upsample_float = scan_sampling / nyquist_sampling
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if verbosity > 0:
            logger.info("scan_sampling = %s", scan_sampling)
            logger.info("nyquist_sampling = %s", nyquist_sampling)
            logger.info("Upsampling to Nyquist, upsample factor: %s", upsample_int)
        return upsample_int
    if isinstance(upsample, str):
        raise ValueError(f"Invalid upsample: {upsample}")
    if isinstance(upsample, (int, float)):
        upsample_float = np.array([upsample, upsample])
        upsample_int = np.ceil(upsample_float).astype(np.uint32)
        if verbosity > 0:
            logger.info("Upsampling to Nyquist, upsample factor: %s", upsample_int)
        return upsample_int
    raise ValueError(f"Invalid upsample: {upsample}")
