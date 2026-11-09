"""``run`` for single-side-band direct ptychography.

The config's fields and defaults are exactly the keyword arguments of the
function it wraps (``bright_field_mask_threshold`` stays 0.5), and the result's
:meth:`as_tuple` is exactly that function's return value.

What differs from calling the function is only where state goes. Direct
ptychography builds its virtual bright field through ``dataset.get_vbf``,
which caches the provider on the dataset; the runner calls the same code on a
throwaway view instead (see :mod:`scatterem.reconstruction._views`), so
``data`` is left untouched. A provider built earlier is reused only when passed
as ``reuse={"vbf": vbf}`` (or a whole cache as ``reuse={"vbf_cache": ...}``).
"""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from scatterem.data.datasets import (
    Dataset4DStem,
    DatasetVirtualBrightField4DStem,
)
from scatterem.data.sampling import Sampling, reconstruction_sampling
from scatterem.reconstruction._runner import (
    Provenance,
    ReconstructionConfig,
    Result,
    register_runner,
)
from scatterem.reconstruction._views import (
    VBF_REUSE_KEYS,
    cache_diagnostics,
    isolated_view,
)
from scatterem.reconstruction.ptychography import direct as _dp

from ._common import PhaseSignMode
from ._gamma_weighting import DEFAULT_WIENER_ALPHA, GammaWeighting

#: ``reuse=`` keys of direct ptychography: a vBF.
_REUSE_KEYS = VBF_REUSE_KEYS

__all__ = [
    "DirectPtychographyConfig",
    "DirectPtychographyResult",
]


@dataclass(frozen=True, eq=False, kw_only=True)
class DirectPtychographyResult(Result):
    """Result of :class:`DirectPtychographyConfig`.

    Parameters
    ----------
    provenance : Provenance
        Where the result came from.
    diagnostics : Mapping, optional
        Secondary outputs (read-only).
    phase : torch.Tensor
        The reconstructed phase image, shape ``(Ny * uy, Nx * ux)`` for the
        resolved upsampling ``(uy, ux)``, on the dataset's device.
    ssnr : torch.Tensor or None
        The SSNR map of ``phase`` (same shape), or ``None`` unless
        ``return_snr`` was set.
    sampling : Sampling
        The pixel size ``(dy, dx)`` of ``phase`` and its units, the scan step
        divided by the upsample factor (from
        :func:`~scatterem.data.sampling.reconstruction_sampling`). Pass it to
        ``vis.show_2d(..., sampling=)`` or ``diagnostics.metrics.frc(...,
        pixel_size=)``.
    """

    __scatterem_name__ = "directptychography"

    def __scatterem_figure__(self, ctx):
        """Panels for :func:`scatterem.vis.show`."""
        from scatterem.reconstruction.ptychography.direct._show import direct_panels

        return direct_panels(self, ctx)

    phase: torch.Tensor
    ssnr: torch.Tensor | None
    sampling: Sampling

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(phase, ssnr)``, what ``direct_ptychography`` returns.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor or None)
            The phase image and the SSNR map (or ``None``).
        """
        return self.phase, self.ssnr


@dataclass(frozen=True)
class DirectPtychographyConfig(ReconstructionConfig[DirectPtychographyResult]):
    """Single-side-band direct ptychography, run through :func:`~scatterem.reconstruction.run`.

    Fields and defaults are those of
    :func:`scatterem.reconstruction.ptychography.direct.direct_ptychography`;
    see it for their meaning. ``snr_blur_sigma`` is the one keyword that
    function takes through ``**kwargs`` (used by the empirical SSNR).

    ``data`` is a ``Dataset4DStem`` or an already-built
    ``DatasetVirtualBrightField4DStem``. ``reuse={"vbf": provider}`` supplies
    a vBF built at this config's ``bright_field_mask_threshold`` from the same
    dataset, instead of building one; it is not checked against the threshold.

    Parameters
    ----------
    upsample : float or str
        Upsampling factor, or ``"nyquist"``.
    bright_field_mask_threshold : float
        Bright-field mask threshold.
    verbosity : int
        Verbosity level.
    n_batches : int
        Number of bright-field chunks.
    return_snr : bool
        Whether to compute the SSNR.
    phase_sign : {"preserve", "positive", "negative"}
        Global phase-sign convention.
    ssnr_method : str
        ``"analytical"`` or ``"empirical"``.
    gamma_weighting : {"phase", "wiener"}
        How ``Gamma(Q, K)`` weights each bright-field pixel.
    wiener_alpha : float
        Tikhonov floor for ``gamma_weighting="wiener"``.
    snr_blur_sigma : float
        Blur sigma of the empirical SSNR.
    rolloff_start : float or None
        When set, the phase spectrum is rolled off with a raised cosine from
        ``rolloff_start * 2 alpha / lambda`` to zero at ``2 alpha / lambda``,
        the transfer limit of direct ptychography, where the noise makes a
        bright rim in the spectrum (see
        :func:`~scatterem.reconstruction.ptychography.direct.rolloff_2alpha`).
        Typical value 0.9. The SSNR map is not changed. Default is None, no
        roll-off.
    """

    method: ClassVar[str] = "direct_ptychography"

    upsample: float | str = "nyquist"
    bright_field_mask_threshold: float = 0.5
    verbosity: int = 0
    n_batches: int = 25
    return_snr: bool = False
    phase_sign: PhaseSignMode = "preserve"
    ssnr_method: str = "analytical"
    gamma_weighting: GammaWeighting = "phase"
    wiener_alpha: float = DEFAULT_WIENER_ALPHA
    snr_blur_sigma: float = 0.0
    rolloff_start: float | None = None


def _direct_source(data: Any, config: DirectPtychographyConfig, reuse) -> Any:
    if isinstance(data, Dataset4DStem):
        return isolated_view(data, reuse, config.bright_field_mask_threshold)
    if isinstance(data, DatasetVirtualBrightField4DStem):
        if any(key in reuse for key in VBF_REUSE_KEYS):
            raise ValueError(
                "reuse={'vbf': ...} / {'vbf_cache': ...} needs a Dataset4DStem; "
                "the data is already a virtual bright field"
            )
        return data
    raise TypeError(
        "direct ptychography needs a Dataset4DStem or a "
        f"DatasetVirtualBrightField4DStem, got {type(data).__name__}"
    )


def _solver_phase_sign(config: DirectPtychographyConfig, reuse) -> PhaseSignMode:
    """The ``phase_sign`` the solver gets."""
    return config.phase_sign


def _check_rolloff_start(config: DirectPtychographyConfig) -> None:
    """Refuse a bad ``rolloff_start`` before the solve, not after it."""
    start = config.rolloff_start
    if start is None:
        return
    if not 0.0 <= start < 1.0:
        raise ValueError(
            f"rolloff_start must be in [0, 1) (a fraction of 2 alpha / lambda), "
            f"got {start!r}"
        )


def _rolled_off(source: Any, phase: torch.Tensor, start: float) -> torch.Tensor:
    """``phase`` with the 2-alpha roll-off, on the pixel grid of ``phase``."""
    meta = source.meta
    # reconstruction_sampling always sets the pixel size of a 2-D image.
    pixel = cast(tuple[float, float], reconstruction_sampling(source, phase).pixel_size)
    q_edge = 2.0 * float(meta.semiconvergence_angle) / float(meta.wavelength)
    return _dp.rolloff_2alpha(phase, pixel, q_edge, start=start)




@register_runner(DirectPtychographyConfig, reuse_keys=_REUSE_KEYS)
def _run_direct_ptychography(
    data: Any, config: DirectPtychographyConfig, reuse
) -> DirectPtychographyResult:
    _check_rolloff_start(config)
    source = _direct_source(data, config, reuse)
    phase_sign = _solver_phase_sign(config, reuse)
    # Looked up on the module at call time, so a test that patches the module
    # attribute reaches this path too.
    phase, ssnr = _dp.direct_ptychography(
        source,
        upsample=config.upsample,
        bright_field_mask_threshold=config.bright_field_mask_threshold,
        verbosity=config.verbosity,
        n_batches=config.n_batches,
        return_snr=config.return_snr,
        phase_sign=phase_sign,
        ssnr_method=config.ssnr_method,
        gamma_weighting=config.gamma_weighting,
        wiener_alpha=config.wiener_alpha,
        snr_blur_sigma=config.snr_blur_sigma,
    )
    if config.rolloff_start is not None:
        phase = _rolled_off(source, phase, config.rolloff_start)
    return DirectPtychographyResult(
        phase=phase,
        ssnr=ssnr,
        sampling=reconstruction_sampling(source, phase),
        provenance=Provenance.capture(config, data),
        diagnostics=cache_diagnostics(source, reuse),
    )


