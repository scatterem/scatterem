"""``run`` for the aberration fit: the coefficients as a value, not a write.

:class:`AberrationFitConfig` carries the keyword arguments of
:func:`scatterem.reconstruction.ptychography.direct.determine_aberrations`
(``gradient_mask`` as a tuple of 12 booleans, so the config stays a plain
value). The runner calls
:func:`scatterem.reconstruction.ptychography.direct.determine_aberrations` on a
throwaway view of the data whose metadata is a detached copy, and whose cached
virtual bright fields (``reuse={"vbf": ...}`` / ``{"vbf_cache": ...}``) are
re-pointed at that copy. Whatever the fit writes on ``vBF.meta`` -- the fitted
aberrations, the rotation, the defocus guess -- therefore lands on the copy,
and the result reports it as :attr:`AberrationFitResult.meta_updates`.
:meth:`AberrationFitResult.applied_to` puts those onto a new copy of any
metadata, e.g. ``dataset.meta = result.applied_to(dataset.meta)``.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar, Literal

import numpy as np
import torch

from scatterem.data.aberrations import Aberrations
from scatterem.data.datasets import (
    Dataset4DStem,
    DatasetVirtualBrightField4DStem,
)
from scatterem.data.datasets._meta import copy_meta
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
from scatterem.reconstruction.ptychography.direct._peak_selection import (
    validate_peak_selection,
)

__all__ = ["AberrationFitConfig", "AberrationFitResult"]


@dataclass(frozen=True, eq=False, kw_only=True)
class AberrationFitResult(Result):
    """Result of :class:`AberrationFitConfig`.

    Parameters
    ----------
    provenance : Provenance
        Where the result came from.
    diagnostics : Mapping, optional
        ``{"vbf_cache": ...}`` when the run was given one to carry through,
        and ``{"defocus_sign_check": {...}}`` when ``check_defocus_sign`` was
        set: ``c10_fitted``, ``c10_kept``, ``energy_fitted``,
        ``energy_flipped`` and ``flipped``. Empty otherwise.
    aberrations : torch.Tensor
        The 12 fitted aberration coefficients.
    meta_updates : Mapping
        The metadata attributes the fit changed, by name, with their new
        values (read-only): what the older entry point wrote onto
        ``dataset.meta`` through the shared ``vBF.meta``. With
        ``update_dataset=True`` these are ``aberrations``, ``rotation`` and
        ``defocus_guess`` (reset to 0.0: the fitted defocus is in
        ``aberrations[0]``).
    rotation : float or None, optional
        The scan rotation (deg) the aberrations hold at: the fitted one with
        ``fit_rotation``, else the rotation the fit held. It is set also with
        ``update_dataset=False``, which writes no ``meta``. ``None`` when the
        fit did not report one.
    """

    __scatterem_name__ = "aberrationfit"

    def __scatterem_figure__(self, ctx):
        """Panels for :func:`scatterem.vis.show`."""
        from scatterem.reconstruction.ptychography.direct._show import aberration_panels

        return aberration_panels(self, ctx)

    aberrations: torch.Tensor
    meta_updates: Mapping[str, Any]
    rotation: float | None = None

    def as_tuple(self) -> tuple[torch.Tensor]:
        """Return ``(aberrations,)``: the fitted tensor wrapped in a 1-tuple.

        Returns
        -------
        tuple of torch.Tensor
            The fitted coefficients, wrapped in a 1-tuple.
        """
        return (self.aberrations,)

    def applied_to(self, meta: Any) -> Any:
        """Return a new copy of ``meta`` with this fit applied.

        The copy gets every entry of :attr:`meta_updates` and a new
        ``Aberrations`` holding :attr:`aberrations`, with ``defocus_guess``
        set to 0.0: the fitted C10 is in ``aberrations[0]`` alone, also when
        the fit ran with ``update_dataset=False``. When :attr:`meta_updates`
        has no ``rotation`` (``update_dataset=False``, or a held rotation), the
        copy gets :attr:`rotation`, so the aberrations keep the rotation they
        hold at. ``meta`` itself is not modified, and the copy shares no
        tensor with this result.

        Parameters
        ----------
        meta : object
            The metadata to start from, typically the fitted data's own.

        Returns
        -------
        object
            The updated copy.
        """
        new = copy_meta(meta)
        updates = copy_meta(SimpleNamespace(**self.meta_updates))
        for name, value in vars(updates).items():
            setattr(new, name, value)
        new.aberrations = Aberrations(array=self.aberrations.detach().clone())
        new.defocus_guess = 0.0
        if self.rotation is not None and "rotation" not in self.meta_updates:
            new.rotation = self.rotation
        return new


@dataclass(frozen=True)
class AberrationFitConfig(ReconstructionConfig[AberrationFitResult]):
    """The aberration fit on a 4D-STEM dataset, run through :func:`~scatterem.reconstruction.run`.

    Fields are the keyword arguments of
    :func:`scatterem.reconstruction.ptychography.direct.determine_aberrations`;
    see it for their meaning. ``update_dataset`` keeps its old effect on the fit
    itself (whether it records the rotation, the defocus guess and the
    coefficients on the metadata, here reported in
    :attr:`AberrationFitResult.meta_updates`); :func:`~scatterem.reconstruction.run` writes nothing.

    ``data`` is a ``Dataset4DStem`` or a ``DatasetVirtualBrightField4DStem``.

    Parameters
    ----------
    correction_method : str
        ``"combined"``, ``"bright-field-shifts"`` or ``"autofocus"``.
    fit_rotation : bool
        Whether to fit the scan rotation.
    target_percentage_nonzero_pixels : float
        Target percentage of nonzero pixels for the bright field.
    n_batches : int
        Number of bright-field batches of the SSB images of the defocus sign
        check and the autofocus (TV) step; not used by the bright-field
        shift fit.
    registration_upsample_factor : int
        Upsampling factor for the shift registration.
    lowpass_fwhm_bright_field : float or None
        FWHM of the lowpass filter applied to the bright field.
    bin_factors : tuple of int
        Bin factors for the bright field.
    verbosity : int
        Verbosity level.
    correct_order : int
        Order of the aberrations the autofocus stage corrects.
    gradient_mask : tuple of bool or None
        Which of the 12 coefficients the autofocus may adjust; ``None`` means
        all of them.
    num_iterations : int
        Number of autofocus iterations.
    lr : float
        Learning rate of the autofocus optimizer.
    bright_field_mask_threshold : float
        Bright-field mask threshold.
    roi_shape : tuple of int
        Shape of the region of interest.
    roi_center : str or tuple of int
        Center of the region of interest.
    upsample : int or str
        Upsampling factor, or ``"nyquist"``.
    update_dataset : bool
        Whether the fit records its results on the (copied) metadata.
    n_center_indices : int
        Number of center indices for the bright-field shift fit.
    phase_sign : {"preserve", "positive", "negative"}
        Sign convention for the fitted phase.
    sharpness_metric : str
        Autofocus objective, ``"sparsity"`` or ``"tv"``.
    reg_weight : float
        L2 regularization weight toward the seed in the autofocus.
    alignment_method : str
        Alignment method of the bright-field shift fit.
    peak_selection : {"argmax", "guided"}
        Which cross-correlation peak is each bright-field shift. Use
        ``"guided"`` on crystals: their correlation is multi-peaked, and the
        global maximum (``"argmax"``) can pick a lattice copy and flip the
        sign of C10.
    init_method : {"closed_form", "zeros"}
        Seed of the bright-field-shift fit.
    use_com_rotation_prior : {"auto", "off"}
        Whether an unset ``meta.rotation`` is seeded from the diffraction
        centre of mass.
    check_defocus_sign : bool
        Whether the fit compares the corrected image at the fitted
        coefficients and at their negative, and keeps the more coherent one. The result then carries the decision in
        ``diagnostics["defocus_sign_check"]``. Off by default: it costs two
        passes over the bright-field stack and can change the fit.
    """

    method: ClassVar[str] = "aberration_fit"

    correction_method: str = "combined"
    fit_rotation: bool = False
    target_percentage_nonzero_pixels: float = 0.75
    n_batches: int = 25
    registration_upsample_factor: int = 10
    lowpass_fwhm_bright_field: float | None = None
    bin_factors: tuple[int, ...] = (2, 1, 1)
    verbosity: int = 0
    correct_order: int = 1
    gradient_mask: tuple[bool, ...] | None = None
    num_iterations: int = 10
    lr: float = 20
    bright_field_mask_threshold: float = 0.5
    roi_shape: tuple[int, int] = (128, 128)
    roi_center: str | tuple[int, int] = "center"
    upsample: int | str = "nyquist"
    update_dataset: bool = True
    n_center_indices: int = 25
    phase_sign: Literal["preserve", "positive", "negative"] = "preserve"
    sharpness_metric: str = "sparsity"
    reg_weight: float = 0.05
    alignment_method: str = "reference"
    peak_selection: Literal["argmax", "guided"] = "argmax"
    init_method: Literal["closed_form", "zeros"] = "closed_form"
    use_com_rotation_prior: Literal["auto", "off"] = "auto"
    check_defocus_sign: bool = False

    def __post_init__(self) -> None:
        validate_peak_selection(self.peak_selection)
        for name, allowed in (
            ("init_method", ("closed_form", "zeros")),
            ("use_com_rotation_prior", ("auto", "off")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"{name} must be one of {allowed}, got {getattr(self, name)!r}"
                )
        mask = self.gradient_mask
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().reshape(-1).tolist()
            mask = tuple(bool(m) for m in mask)
            if len(mask) != 12:
                raise ValueError(
                    f"gradient_mask needs 12 entries, one per coefficient; got "
                    f"{len(mask)}"
                )
            object.__setattr__(self, "gradient_mask", mask)

    def gradient_mask_tensor(self) -> torch.Tensor:
        """Return ``gradient_mask`` as the boolean tensor the fit takes.

        Returns
        -------
        torch.Tensor
            Shape ``(12,)``, all ``True`` when ``gradient_mask`` is ``None``.
        """
        if self.gradient_mask is None:
            return torch.ones(12, dtype=torch.bool)
        return torch.tensor(self.gradient_mask, dtype=torch.bool)


def _detached(provider: Any, meta: Any) -> Any:
    """A shallow copy of a vBF provider reading and writing ``meta`` instead."""
    if provider is None:
        return None
    twin = copy.copy(provider)
    twin.meta = meta
    return twin


def _fit_source(data: Any, config: AberrationFitConfig, reuse) -> Any:
    if isinstance(data, Dataset4DStem):
        view = isolated_view(
            data, reuse, config.bright_field_mask_threshold, own_meta=True
        )
        # The seeded providers share the caller's metadata; the fit writes
        # through ``vBF.meta``, so each is re-pointed at the view's copy.
        view.vBF = _detached(view.vBF, view.meta)
        if view._vbf_spare is not None:
            threshold, spare = view._vbf_spare
            view._vbf_spare = (threshold, _detached(spare, view.meta))
        return view
    if isinstance(data, DatasetVirtualBrightField4DStem):
        if any(key in reuse for key in VBF_REUSE_KEYS):
            raise ValueError(
                "reuse={'vbf': ...} / {'vbf_cache': ...} needs a Dataset4DStem; "
                "the data is already a virtual bright field"
            )
        return _detached(data, copy_meta(data.meta))
    raise TypeError(
        "the aberration fit needs a Dataset4DStem or a "
        f"DatasetVirtualBrightField4DStem, got {type(data).__name__}"
    )


def _same(before: Any, after: Any) -> bool:
    """Value equality that sees tensor devices and recurses into plain objects."""
    if isinstance(before, torch.Tensor) or isinstance(after, torch.Tensor):
        return (
            isinstance(before, torch.Tensor)
            and isinstance(after, torch.Tensor)
            and before.shape == after.shape
            and before.dtype == after.dtype
            and before.device == after.device
            and torch.equal(before, after)
        )
    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        return type(before) is type(after) and np.array_equal(before, after)
    if type(before) is not type(after):
        return False
    if hasattr(before, "__dict__") and not isinstance(before, type):
        va, vb = vars(before), vars(after)
        return va.keys() == vb.keys() and all(_same(va[k], vb[k]) for k in va)
    try:
        return bool(before == after)
    except (TypeError, ValueError, RuntimeError):
        return before is after


def _meta_updates(before: Any, after: Any) -> dict[str, Any]:
    """The attributes of ``after`` added or changed relative to ``before``."""
    if after is None:
        return {}
    old = vars(before) if before is not None else {}
    return {
        name: value
        for name, value in vars(after).items()
        if name not in old or not _same(old[name], value)
    }


@register_runner(AberrationFitConfig, reuse_keys=VBF_REUSE_KEYS)
def _run_aberration_fit(
    data: Any, config: AberrationFitConfig, reuse
) -> AberrationFitResult:
    source = _fit_source(data, config, reuse)
    before = copy_meta(source.meta)
    # Looked up on the module at call time, so a test that patches the module
    # attribute reaches this path too.
    aberrations, vbf = _dp.determine_aberrations(
        dataset=source,
        correction_method=config.correction_method,
        fit_rotation=config.fit_rotation,
        registration_upsample_factor=config.registration_upsample_factor,
        lowpass_fwhm_bright_field=config.lowpass_fwhm_bright_field,
        bin_factors=config.bin_factors,
        upsample=config.upsample,
        n_batches=config.n_batches,
        roi_shape=config.roi_shape,
        roi_center=config.roi_center,
        num_iterations=config.num_iterations,
        lr=config.lr,
        bright_field_mask_threshold=config.bright_field_mask_threshold,
        target_percentage_nonzero_pixels=config.target_percentage_nonzero_pixels,
        correct_order=config.correct_order,
        gradient_mask=config.gradient_mask_tensor(),
        verbosity=config.verbosity,
        update_dataset=config.update_dataset,
        n_center_indices=config.n_center_indices,
        phase_sign=config.phase_sign,
        sharpness_metric=config.sharpness_metric,
        reg_weight=config.reg_weight,
        alignment_method=config.alignment_method,
        peak_selection=config.peak_selection,
        init_method=config.init_method,
        use_com_rotation_prior=config.use_com_rotation_prior,
        check_defocus_sign=config.check_defocus_sign,
    )
    diagnostics = cache_diagnostics(source, reuse)
    sign_check = getattr(vbf, "defocus_sign_check", None)
    if config.check_defocus_sign and sign_check is not None:
        diagnostics["defocus_sign_check"] = sign_check.as_dict()
    return AberrationFitResult(
        aberrations=aberrations,
        meta_updates=MappingProxyType(_meta_updates(before, source.meta)),
        rotation=getattr(vbf, "aberration_rotation", None),
        provenance=Provenance.capture(config, data),
        diagnostics=diagnostics,
    )
