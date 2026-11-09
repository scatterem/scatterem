"""``run`` for tilt-corrected dark field.

The config's fields and defaults are exactly the keyword arguments of the
function it wraps (``bright_field_mask_threshold`` is 0.3), and the result's
:meth:`as_tuple` is exactly that function's return value. tcDF reads and writes
no dataset cache, so it runs on ``data`` itself; a precomputed segment geometry
(built on that same dataset) may be passed as ``reuse={"segment_geometry": g}``.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from scatterem.data.datasets import Dataset4DStem
from scatterem.data.sampling import Sampling, reconstruction_sampling
from scatterem.reconstruction._runner import (
    Provenance,
    ReconstructionConfig,
    Result,
    register_runner,
)
from scatterem.reconstruction.darkfield import core as _tcdf

from ._scaling import _DEAD_FRAME_FRACTION

__all__ = [
    "TiltCorrectedDarkFieldConfig",
    "TiltCorrectedDarkFieldResult",
]


@dataclass(frozen=True, eq=False, kw_only=True)
class TiltCorrectedDarkFieldResult(Result):
    """Result of :class:`TiltCorrectedDarkFieldConfig`.

    Parameters
    ----------
    provenance : Provenance
        Where the result came from.
    diagnostics : Mapping, optional
        Secondary outputs (read-only; empty for this family).
    image : torch.Tensor
        The tilt-corrected dark-field (or, with ``region="bright_field"``,
        bright-field) image on the upsampled grid, on the dataset's device.
    ssnr : torch.Tensor or None
        The half-split SSNR map (same shape), or ``None`` unless
        ``return_snr`` was set.
    sampling : Sampling
        The pixel size ``(dy, dx)`` of ``image`` and its units, the scan step
        divided by the upsample factor (from
        :func:`~scatterem.data.sampling.reconstruction_sampling`). Pass it to
        ``vis.show_2d(..., sampling=)`` or ``diagnostics.metrics.frc(...,
        pixel_size=)``.
    """

    image: torch.Tensor
    ssnr: torch.Tensor | None
    sampling: Sampling

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(image, ssnr)``, what ``tilt_corrected_dark_field`` returns.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor or None)
            The image and the SSNR map (or ``None``).
        """
        return self.image, self.ssnr


@dataclass(frozen=True)
class TiltCorrectedDarkFieldConfig(ReconstructionConfig[TiltCorrectedDarkFieldResult]):
    """Tilt-corrected dark field, run through :func:`~scatterem.reconstruction.run`.

    Fields and defaults are those of
    :func:`scatterem.reconstruction.darkfield.core.tilt_corrected_dark_field`
    (its private ``_segment_geometry`` hook excepted); see it for their
    meaning. ``data`` is a ``Dataset4DStem``; ``reuse={"segment_geometry": g}``
    passes the private precomputed segment geometry (built on ``data`` with
    this config's parameters; a mismatch raises ``ValueError``).

    Parameters
    ----------
    n_dark_field_segments : int
        Number of azimuthal detector segments.
    verbosity : int
        Diagnostic-plot verbosity.
    bright_field_mask_threshold : float
        Bright-field mask threshold.
    upsample : float or str
        Upsampling factor, or ``"nyquist"``.
    return_snr : bool
        Whether to compute the half-split SSNR.
    snr_blur_sigma : float
        Blur sigma of the SSNR curve.
    shift_method : {"drizzle", "fourier"}
        How segments are shifted and accumulated.
    drizzle_pixfrac : float
        Drizzle drop size.
    drizzle_kde_sigma : float
        Drizzle smoothing sigma.
    dark_field_scaling : bool
        Rescale segments for scattering past the detector edge.
    dead_frame_fraction : float
        Minimum recorded fraction for `dark_field_scaling`.
    region : {"dark_field", "bright_field"}
        Which detector pixels feed the segments.
    """

    method: ClassVar[str] = "tilt_corrected_dark_field"

    n_dark_field_segments: int = 32
    verbosity: int = 0
    bright_field_mask_threshold: float = 0.3
    upsample: float | str = "nyquist"
    return_snr: bool = False
    snr_blur_sigma: float = 0.0
    shift_method: str = "drizzle"
    drizzle_pixfrac: float = 1.0
    drizzle_kde_sigma: float = 0.0
    dark_field_scaling: bool = False
    dead_frame_fraction: float = _DEAD_FRAME_FRACTION
    region: str = "dark_field"


@register_runner(TiltCorrectedDarkFieldConfig, reuse_keys=("segment_geometry",))
def _run_tilt_corrected_dark_field(
    data: Any, config: TiltCorrectedDarkFieldConfig, reuse
) -> TiltCorrectedDarkFieldResult:
    if not isinstance(data, Dataset4DStem):
        raise TypeError(
            f"tilt-corrected dark field needs a Dataset4DStem, got {type(data).__name__}"
        )
    # tcDF reads and writes no dataset cache, so it runs on ``data`` itself: a precomputed segment geometry
    # checks it was built on this very dataset object.
    image, ssnr = _tcdf.tilt_corrected_dark_field(
        data,
        n_dark_field_segments=config.n_dark_field_segments,
        verbosity=config.verbosity,
        bright_field_mask_threshold=config.bright_field_mask_threshold,
        upsample=config.upsample,
        return_snr=config.return_snr,
        snr_blur_sigma=config.snr_blur_sigma,
        shift_method=config.shift_method,
        drizzle_pixfrac=config.drizzle_pixfrac,
        drizzle_kde_sigma=config.drizzle_kde_sigma,
        dark_field_scaling=config.dark_field_scaling,
        dead_frame_fraction=config.dead_frame_fraction,
        region=config.region,
        _segment_geometry=reuse.get("segment_geometry"),
    )
    return TiltCorrectedDarkFieldResult(
        image=image,
        ssnr=ssnr,
        sampling=reconstruction_sampling(data, image),
        provenance=Provenance.capture(config, data),
    )
