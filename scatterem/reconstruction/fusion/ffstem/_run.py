"""``run`` for the FF-STEM fusion of the two direct channels.

:class:`FusedFullFieldConfig` carries the parameters of the fusion. The runner
reconstructs both channels through :func:`~scatterem.reconstruction.run` (so neither writes to the data)
unless they are handed in, and fuses them with
:func:`scatterem.reconstruction.fusion.ffstem.core.fused_full_field`.

``reuse={"ptycho": r1, "tcdf": r2}`` passes channel results computed earlier.
Each is checked against the channel config this fusion would have run: every
field but ``verbosity`` must agree, or the fusion refuses it rather than
fusing a stale channel.
"""

import dataclasses
import logging
import time
import warnings
from collections.abc import Mapping
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
    run,
)
from scatterem.reconstruction._views import (
    VBF_REUSE_KEYS,
)
from scatterem.reconstruction.darkfield._run import (
    TiltCorrectedDarkFieldConfig,
    TiltCorrectedDarkFieldResult,
)
from scatterem.reconstruction.fusion.ffstem import core as _fff
from scatterem.reconstruction.ptychography.direct._common import PhaseSignMode
from scatterem.reconstruction.ptychography.direct._run import (
    DirectPtychographyConfig,
    DirectPtychographyResult,
)

__all__ = [
    "FusedFullFieldConfig",
    "FusedFullFieldResult",
]

logger = logging.getLogger(__name__)

_PHASE_CHANNELS = ("ptychography", "bright_field")


def resolve_dark_field_segments(
    n_batches: int, n_dark_field_segments: int | None, *, stacklevel: int
) -> int:
    """``n_dark_field_segments``, falling back to ``n_batches`` with a warning.

    ``stacklevel`` counts frames from this function (1), so each caller can
    attribute the warning to its own caller.
    """
    if n_dark_field_segments is not None:
        return int(n_dark_field_segments)
    segments = int(n_batches)
    warnings.warn(
        f"n_dark_field_segments not given; falling back to n_batches={segments}. "
        "This is a SCIENTIFIC parameter (wedge membership, per-segment shifts, "
        "half-set parity), so changing n_batches for a memory reason silently "
        "changes the tilt-corrected dark field. Pass n_dark_field_segments "
        "explicitly.",
        stacklevel=stacklevel,
    )
    return segments


@dataclass(frozen=True, eq=False, kw_only=True)
class FusedFullFieldResult(Result):
    """Result of :class:`FusedFullFieldConfig`.

    Parameters
    ----------
    provenance : Provenance
        Where the result came from; its config has the threshold and the
        segment count resolved.
    diagnostics : Mapping, optional
        ``{"vbf_cache": ...}`` when a vBF cache was passed in, else empty.
    fused : torch.Tensor
        The fused image, on the upsampled grid, on the dataset's device.
    phase_weighted : torch.Tensor or None
        The low-frequency channel after its weight (``None`` unless the
        fusion's verbosity fills it, as in the function).
    tcdf_weighted : torch.Tensor or None
        The dark-field channel after its weight (likewise).
    phase_image : torch.Tensor
        The low-frequency image that was fused: the SSB phase, or the negated,
        mean-subtracted tilt-corrected bright field.
    ptycho : Result
        The low-frequency channel's own result (a
        :class:`~scatterem.reconstruction.ptychography.direct.DirectPtychographyResult`, or a
        :class:`~scatterem.reconstruction.darkfield.TiltCorrectedDarkFieldResult` with ``region="bright_field"``).
    tcdf : TiltCorrectedDarkFieldResult
        The dark-field channel's result.
    sampling : Sampling
        The pixel size ``(dy, dx)`` of ``fused`` and its units, the scan step
        divided by the upsample factor (from
        :func:`~scatterem.data.sampling.reconstruction_sampling`). Pass it to
        ``vis.show_2d(..., sampling=)`` or ``diagnostics.metrics.frc(...,
        pixel_size=)``.
    """

    fused: torch.Tensor
    phase_weighted: torch.Tensor | None
    tcdf_weighted: torch.Tensor | None
    phase_image: torch.Tensor
    ptycho: Result
    tcdf: TiltCorrectedDarkFieldResult
    sampling: Sampling

    def as_tuple(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return ``(fused, phase_weighted, tcdf_weighted)``.

        Returns
        -------
        tuple
            The three images of the result, in this order.
        """
        return self.fused, self.phase_weighted, self.tcdf_weighted


@dataclass(frozen=True)
class FusedFullFieldConfig(ReconstructionConfig[FusedFullFieldResult]):
    """SSNR-weighted FF-STEM fusion, run through :func:`~scatterem.reconstruction.run`.

    ``bright_field_mask_threshold=None`` resolves to
    ``data.bright_field_mask_threshold``, and ``n_dark_field_segments=None``
    to ``n_batches`` with a warning.

    Parameters
    ----------
    n_batches : int
        Bright-field chunks of the ptychography channel.
    upsample : float or str
        Upsampling factor, or ``"nyquist"``.
    verbosity : int
        Verbosity level.
    bright_field_mask_threshold : float or None
        Bright-field mask threshold of both channels.
    return_snr : bool
        Whether the channels compute their SSNR (the fusion needs it).
    snr_blur_sigma : float
        SSNR blur sigma of both channels.
    ssnr_method : str
        The ptychography SSNR, ``"analytical"`` or ``"empirical"``.
    ptycho_lowq_cutoff_frac : float
        Low-q band handed to the dark field.
    tcdf_shift_method : str
        The tcDF shift method.
    drizzle_pixfrac : float
        Drizzle drop size.
    drizzle_kde_sigma : float
        Drizzle smoothing sigma.
    n_dark_field_segments : int or None
        Dark-field segments.
    dark_field_scaling : bool
        The tcDF dark-field scaling.
    dead_frame_fraction : float
        The tcDF dead-frame fraction.
    phase_channel : {"ptychography", "bright_field"}
        What supplies the low-frequency half.
    n_bright_field_segments : int
        Segments of the bright-field phase channel.
    phase_sign : {"preserve", "positive", "negative"}
        Global sign convention of the ptychography channel, as in
        :class:`~scatterem.reconstruction.ptychography.direct.DirectPtychographyConfig`. The default ``"preserve"`` keeps
        the sign that the solver gives. ``"positive"`` is the convention of
        version 0.3 and of the FF-STEM paper scripts. The bright-field phase
        channel ignores it.
    """

    method: ClassVar[str] = "fused_full_field"

    n_batches: int = 32
    upsample: float | str = "nyquist"
    verbosity: int = 0
    bright_field_mask_threshold: float | None = None
    return_snr: bool = True
    snr_blur_sigma: float = 0.0
    ssnr_method: str = "analytical"
    ptycho_lowq_cutoff_frac: float = 0.0
    tcdf_shift_method: str = "drizzle"
    drizzle_pixfrac: float = 1.0
    drizzle_kde_sigma: float = 0.0
    n_dark_field_segments: int | None = None
    dark_field_scaling: bool = False
    dead_frame_fraction: float = 1e-3
    phase_channel: str = "ptychography"
    n_bright_field_segments: int = 8
    phase_sign: PhaseSignMode = "preserve"

    def channel_configs(
        self,
    ) -> tuple[
        DirectPtychographyConfig | TiltCorrectedDarkFieldConfig,
        TiltCorrectedDarkFieldConfig,
    ]:
        """Return the configs of the two channels this fusion runs.

        Needs the threshold and the segment count resolved (not ``None``).

        Returns
        -------
        tuple of ReconstructionConfig
            ``(low-frequency channel, dark-field channel)``.

        Raises
        ------
        ValueError
            On an unknown ``phase_channel``, or an unresolved threshold or
            segment count.
        """
        if self.phase_channel not in _PHASE_CHANNELS:
            raise ValueError(
                f"Invalid phase_channel: {self.phase_channel!r} "
                f"(expected 'ptychography' or 'bright_field')"
            )
        threshold = self.bright_field_mask_threshold
        if threshold is None or self.n_dark_field_segments is None:
            raise ValueError("resolve the threshold and the segment count first")
        shared: dict[str, Any] = dict(
            verbosity=self.verbosity,
            bright_field_mask_threshold=threshold,
            upsample=self.upsample,
            snr_blur_sigma=self.snr_blur_sigma,
        )
        drizzle: dict[str, Any] = dict(
            shift_method=self.tcdf_shift_method,
            drizzle_pixfrac=self.drizzle_pixfrac,
            drizzle_kde_sigma=self.drizzle_kde_sigma,
        )
        phase: DirectPtychographyConfig | TiltCorrectedDarkFieldConfig
        if self.phase_channel == "bright_field":
            phase = TiltCorrectedDarkFieldConfig(
                n_dark_field_segments=self.n_bright_field_segments,
                return_snr=True,
                region="bright_field",
                **shared,
                **drizzle,
            )
        else:
            phase = DirectPtychographyConfig(
                n_batches=self.n_batches,
                return_snr=self.return_snr,
                ssnr_method=self.ssnr_method,
                phase_sign=self.phase_sign,
                **shared,
            )
        tcdf = TiltCorrectedDarkFieldConfig(
            n_dark_field_segments=self.n_dark_field_segments,
            return_snr=self.return_snr,
            dark_field_scaling=self.dark_field_scaling,
            dead_frame_fraction=self.dead_frame_fraction,
            **shared,
            **drizzle,
        )
        return phase, tcdf


def _science(config_mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config_mapping.items() if k != "verbosity"}


def _channel(
    data: Any,
    config: ReconstructionConfig[Any],
    reused: Any,
    key: str,
    reuse: Mapping[str, Any],
) -> Result:
    """Run one channel, or check and return the result handed in for it."""
    if reused is None:
        return run(data, config, reuse=reuse)
    if not isinstance(reused, Result):
        raise TypeError(f"reuse[{key!r}] must be a Result, got {type(reused).__name__}")
    want = _science(config.resolved())
    got = _science(reused.provenance.config)
    if reused.provenance.method != config.method or got != want:
        diff = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError(
            f"reuse[{key!r}] was not computed with the settings this fusion "
            f"needs ({reused.provenance.method} vs {config.method}; differing: "
            f"{diff})"
        )
    return reused


def _low_frequency(ptycho: Result) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(ptycho, TiltCorrectedDarkFieldResult):
        # Bright field is dark where the specimen scatters -- the opposite
        # contrast sense to the dark field -- so it is negated before the sum.
        return -(ptycho.image - ptycho.image.mean()), ptycho.ssnr
    if isinstance(ptycho, DirectPtychographyResult):
        return ptycho.phase, ptycho.ssnr
    raise TypeError(f"unexpected phase-channel result {type(ptycho).__name__}")


@register_runner(FusedFullFieldConfig, reuse_keys=("ptycho", "tcdf", *VBF_REUSE_KEYS))
def _run_fused_full_field(
    data: Any, config: FusedFullFieldConfig, reuse
) -> FusedFullFieldResult:
    if not isinstance(data, Dataset4DStem):
        raise TypeError(f"the fusion needs a Dataset4DStem, got {type(data).__name__}")
    threshold = config.bright_field_mask_threshold
    if threshold is None:
        threshold = data.bright_field_mask_threshold
    # 4: this helper, the runner, ``run``, then ``run``'s caller.
    segments = resolve_dark_field_segments(
        config.n_batches, config.n_dark_field_segments, stacklevel=4
    )
    config = dataclasses.replace(
        config, bright_field_mask_threshold=threshold, n_dark_field_segments=segments
    )
    phase_config, tcdf_config = config.channel_configs()
    vbf_reuse = {k: reuse[k] for k in VBF_REUSE_KEYS if k in reuse}

    _t0 = time.perf_counter()
    reused_ptycho = reuse.get("ptycho")
    if isinstance(phase_config, TiltCorrectedDarkFieldConfig):
        ptycho = _channel(data, phase_config, reused_ptycho, "ptycho", {})
    else:
        ptycho = _channel(data, phase_config, reused_ptycho, "ptycho", vbf_reuse)
    tcdf = _channel(data, tcdf_config, reuse.get("tcdf"), "tcdf", {})
    if not isinstance(tcdf, TiltCorrectedDarkFieldResult):
        raise TypeError(
            "reuse['tcdf'] must be a TiltCorrectedDarkFieldResult, got "
            f"{type(tcdf).__name__}"
        )
    phase_image, ssnr_ptycho = _low_frequency(ptycho)
    if ssnr_ptycho is None or tcdf.ssnr is None:
        raise ValueError("fused_full_field requires SSNR estimates")

    _t_fuse = time.perf_counter()
    fused, phase_weighted, tcdf_weighted = _fff.fused_full_field(
        data,
        phase_image,
        tcdf.image,
        ssnr_ptycho,
        tcdf.ssnr,
        verbosity=config.verbosity,
        ptycho_lowq_cutoff_frac=config.ptycho_lowq_cutoff_frac,
    )
    if config.verbosity > 0:
        logger.info(
            "FF-STEM timing: total (recon+fuse) %.3f s, fusion-only %.3f s",
            time.perf_counter() - _t0,
            time.perf_counter() - _t_fuse,
        )
    diagnostics: dict[str, Any] = {}
    if "vbf_cache" in reuse:
        # The SSB channel is the only one that touches the cache; a reused or
        # bright-field channel leaves it as it came in.
        ran_ssb = reused_ptycho is None and isinstance(ptycho, DirectPtychographyResult)
        diagnostics["vbf_cache"] = (
            ptycho.diagnostics["vbf_cache"] if ran_ssb else reuse["vbf_cache"]
        )
    return FusedFullFieldResult(
        fused=fused,
        sampling=reconstruction_sampling(data, fused),
        phase_weighted=phase_weighted,
        tcdf_weighted=tcdf_weighted,
        phase_image=phase_image,
        ptycho=ptycho,
        tcdf=tcdf,
        provenance=Provenance.capture(config, data),
        diagnostics=diagnostics,
    )
