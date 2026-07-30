"""Display normalization: pick a data interval, then apply a display stretch.

This module is an independent implementation written from the mathematical
definitions of interval selection and display stretching: no code, architecture,
docstrings or comments are taken from another project. The public names
(:class:`CustomNormalization`, :class:`NormalizationConfig` and the config field
names) are deliberately retained from the module this replaces, so that call sites
and published figure output do not change, and ``vmin``/``vmax``/``clip``/
``inverse``/``autoscale_None`` are ``matplotlib.colors.Normalize``'s own API.

Normalization happens in two stages, in this order:

1. **Interval.** Choose limits ``(vmin, vmax)`` from explicit settings or from the
   data, then map linearly onto the unit interval and clip::

       t = clip((x - vmin) / (vmax - vmin), 0, 1)

2. **Stretch.** Apply a monotone map ``s: [0, 1] -> [0, 1]`` with ``s(0) = 0`` and
   ``s(1) = 1``. A stretch redistributes contrast without moving the end points.

Because stage 1 clips, a stretch only ever sees ``t`` inside ``[0, 1]``, so every
formula below is evaluated strictly within its monotone domain -- there are no NaNs
from real powers of negative numbers or logarithms of non-positive arguments, and the
result is always inside ``[0, 1]``.

The public surface is :class:`NormalizationConfig` (what callers construct),
:class:`CustomNormalization` (a drop-in ``matplotlib.colors.Normalize``) and
:data:`NORMALIZATION_PRESETS` (named shorthands for common configs).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Optional, Tuple

import numpy as np
from matplotlib import colors

__all__ = [
    "NormalizationConfig",
    "CustomNormalization",
    "NORMALIZATION_PRESETS",
    "INTERVAL_TYPES",
    "STRETCH_TYPES",
]

INTERVAL_TYPES = ("quantile", "manual", "centered")
STRETCH_TYPES = ("linear", "power", "logarithmic", "asinh")

# Which config fields each stretch actually reads. Used to warn about a parameter
# that has been set but cannot take effect, which is otherwise silent.
_STRETCH_PARAMETERS = {
    "linear": (),
    "power": ("power",),
    "logarithmic": ("logarithmic_index",),
    "asinh": ("asinh_linear_range",),
}


@dataclass
class NormalizationConfig:
    """Settings for :class:`CustomNormalization`.

    Parameters
    ----------
    interval_type : {"quantile", "manual", "centered"}
        How ``(vmin, vmax)`` are chosen. ``"quantile"`` uses the
        ``lower_quantile``/``upper_quantile`` of the finite data, ``"manual"`` uses
        ``vmin``/``vmax`` (falling back to the finite data range for either one left
        as None), ``"centered"`` uses ``vcenter +/- half_range`` (``half_range``
        defaults to the largest finite deviation from ``vcenter``).
    stretch_type : {"linear", "power", "logarithmic", "asinh"}
        Which stretch to apply; see :func:`_stretch_pair` for the formulas.
    lower_quantile, upper_quantile : float
        Quantiles in ``[0, 1]`` with ``lower < upper``, used by ``"quantile"``.
    vmin, vmax : float, optional
        Explicit limits, used by ``"manual"``.
    vcenter, half_range : float, optional
        Center and half-width, used by ``"centered"``.
    power : float
        Exponent ``p > 0`` of the ``"power"`` stretch.
    logarithmic_index : float
        Curvature ``a > 0`` of the ``"logarithmic"`` stretch.
    asinh_linear_range : float
        Linear-range parameter ``a > 0`` of the ``"asinh"`` stretch.
    """

    interval_type: str = "quantile"
    stretch_type: str = "linear"
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    vcenter: float = 0.0
    half_range: Optional[float] = None
    power: float = 1.0
    logarithmic_index: float = 1000.0
    asinh_linear_range: float = 0.1


class _NeedsData(ValueError):
    """Raised internally when limits cannot be resolved without seeing the data."""


def _as_float_array(value: Any) -> np.ndarray:
    """Convert numpy arrays, masked arrays, torch tensors or sequences to float64.

    Conversion happens once, here, at the boundary. Masked entries become NaN so
    that the masked-array and NaN paths below coincide.
    """
    if hasattr(value, "detach"):  # torch.Tensor, duck-typed to avoid importing torch
        value = value.detach().cpu().double().numpy()
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    return np.asarray(value, dtype=np.float64)


def _finite_values(data: Any, reason: str) -> np.ndarray:
    """Return the finite entries of *data* as a flat array.

    Raises
    ------
    _NeedsData
        If *data* is None, so the caller can defer resolution until data arrives.
    ValueError
        If *data* contains no finite value at all, which no interval rule can
        summarize.
    """
    if data is None:
        raise _NeedsData(f"data are required to determine {reason}")
    flat = _as_float_array(data).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError(f"no finite values to normalize, so {reason} is undefined")
    return finite


def _interval_limits(config: NormalizationConfig, data: Any) -> Tuple[float, float]:
    """Return the ``(vmin, vmax)`` implied by *config* and, where needed, *data*.

    NaNs and infinities are excluded from every data-derived limit.
    """
    kind = config.interval_type
    if kind == "manual":
        if config.vmin is None or config.vmax is None:
            finite = _finite_values(data, "the manual interval limits")
            vmin = float(finite.min()) if config.vmin is None else float(config.vmin)
            vmax = float(finite.max()) if config.vmax is None else float(config.vmax)
        else:
            vmin, vmax = float(config.vmin), float(config.vmax)
    elif kind == "quantile":
        finite = _finite_values(data, "the quantile interval limits")
        lo, hi = np.quantile(finite, [config.lower_quantile, config.upper_quantile])
        vmin, vmax = float(lo), float(hi)
    else:  # "centered"
        center = float(config.vcenter)
        if config.half_range is None:
            finite = _finite_values(data, "the centered interval half-range")
            half = float(np.abs(finite - center).max())
        else:
            half = float(config.half_range)
        vmin, vmax = center - half, center + half

    if vmin > vmax:
        raise ValueError(
            f"inverted interval: vmin={vmin!r} exceeds vmax={vmax!r}; swap them "
            "explicitly rather than relying on an implicit flip"
        )
    return vmin, vmax


def _stretch_pair(
    config: NormalizationConfig,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``(forward, inverse)`` for the configured stretch.

    Every forward map ``s`` satisfies ``s(0) = 0``, ``s(1) = 1`` and is monotonically
    non-decreasing on ``[0, 1]``, so it maps ``[0, 1]`` onto ``[0, 1]``:

    ``"linear"``
        ``s(t) = t``. The identity; monotone on all of R. Its own inverse.
    ``"power"``
        ``s(t) = t ** p`` with ``p = power > 0``. Strictly increasing on
        ``[0, inf)``; ``p > 1`` darkens mid-tones, ``p < 1`` brightens them.
        Inverse ``y ** (1 / p)``.
    ``"logarithmic"``
        ``s(t) = log(1 + a * t) / log(1 + a)`` with ``a = logarithmic_index > 0``.
        Strictly increasing wherever ``1 + a * t > 0``, i.e. for ``t > -1 / a``, hence
        on all of ``[0, 1]``. Larger ``a`` compresses the bright end harder and so
        lifts faint detail. Inverse ``((1 + a) ** y - 1) / a``.
    ``"asinh"``
        ``s(t) = asinh(t / a) / asinh(1 / a)`` with ``a = asinh_linear_range > 0``.
        Strictly increasing on all of R, since ``asinh`` is; approximately linear for
        ``t << a`` and logarithmic for ``t >> a``. Inverse
        ``a * sinh(y * asinh(1 / a))``. This is the standard concave asinh stretch:
        steepest at ``t = 0``, so it lifts faint detail. It is deliberately *not*
        symmetric about the mid-tone -- a mid-tone-enhancing S-curve would be
        ``[1 + asinh((2t - 1) / a) / asinh(1 / a)] / 2``, a different curve that this
        module does not provide.
    """
    kind = config.stretch_type
    if kind == "linear":
        return (lambda t: t), (lambda y: y)
    if kind == "power":
        p = float(config.power)
        if p <= 0:
            raise ValueError(f"power must be > 0 to be monotone, got {p!r}")
        return (lambda t: np.power(t, p)), (lambda y: np.power(y, 1.0 / p))
    if kind == "logarithmic":
        a = float(config.logarithmic_index)
        if a <= 0:
            raise ValueError(f"logarithmic_index must be > 0, got {a!r}")
        denominator = np.log1p(a)
        return (
            lambda t: np.log1p(a * t) / denominator,
            lambda y: np.expm1(y * denominator) / a,
        )
    if kind == "asinh":
        a = float(config.asinh_linear_range)
        if a <= 0:
            raise ValueError(f"asinh_linear_range must be > 0, got {a!r}")
        denominator = np.arcsinh(1.0 / a)
        return (
            lambda t: np.arcsinh(t / a) / denominator,
            lambda y: a * np.sinh(y * denominator),
        )
    raise ValueError(f"unknown stretch_type {kind!r}; expected one of {STRETCH_TYPES}")


def _validate(config: NormalizationConfig) -> None:
    """Check the settings that do not depend on data, at construction time."""
    if config.interval_type not in INTERVAL_TYPES:
        raise ValueError(
            f"unknown interval_type {config.interval_type!r}; "
            f"expected one of {INTERVAL_TYPES}"
        )
    if config.stretch_type not in STRETCH_TYPES:
        raise ValueError(
            f"unknown stretch_type {config.stretch_type!r}; "
            f"expected one of {STRETCH_TYPES}"
        )
    lo, hi = float(config.lower_quantile), float(config.upper_quantile)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"need 0 <= lower_quantile < upper_quantile <= 1, got {lo!r} and {hi!r}"
        )
    if config.half_range is not None and float(config.half_range) < 0.0:
        raise ValueError(f"half_range must be >= 0, got {config.half_range!r}")


def _warn_inert_parameters(config: NormalizationConfig) -> None:
    """Warn about a stretch parameter that was set but cannot take effect.

    ``stretch_type`` alone decides which stretch is applied, so e.g. ``power=2.0``
    combined with ``stretch_type="linear"`` silently does nothing. Only values that
    differ from their default are reported, so forwarding a whole config never warns.
    """
    defaults = NormalizationConfig()
    used = _STRETCH_PARAMETERS[config.stretch_type]
    inert = [
        name
        for parameters in _STRETCH_PARAMETERS.values()
        for name in parameters
        if name not in used and getattr(config, name) != getattr(defaults, name)
    ]
    for name in sorted(set(inert)):
        warnings.warn(
            f"{name}={getattr(config, name)!r} has no effect when "
            f"stretch_type={config.stretch_type!r}; stretch_type alone selects the "
            "stretch, so set it to the stretch this parameter belongs to.",
            UserWarning,
            stacklevel=3,
        )


class CustomNormalization(colors.Normalize):
    """Map data into ``[0, 1]`` for display: interval selection, then a stretch.

    Parameters
    ----------
    config : NormalizationConfig, optional
        Settings; defaults to ``NormalizationConfig()``.
    data : array-like, optional
        Data used to resolve the limits immediately, so that ``vmin``/``vmax`` are
        populated before the first call (a colorbar needs them). Otherwise the limits
        are resolved on the first call, from the array passed there.
    clip : bool
        Accepted for ``matplotlib.colors.Normalize`` compatibility. The result is
        confined to ``[0, 1]`` in any case, because the normalized value is clipped
        before the stretch is applied.
    **overrides
        Any :class:`NormalizationConfig` field, passed directly instead of *config*.

    Notes
    -----
    The limits are resolved **once** and then frozen, whether by ``data=`` here or by
    the first call. One instance therefore describes one interval, which is what makes
    it safe to share across several images: they are all mapped through the same
    limits, and a colorbar drawn from those limits describes all of them. Pass a fresh
    instance per image to normalize each by its own limits.
    """

    def __init__(
        self,
        config: Optional[NormalizationConfig] = None,
        *,
        data: Any = None,
        clip: bool = False,
        **overrides: Any,
    ) -> None:
        base = NormalizationConfig() if config is None else config
        allowed = {field.name for field in fields(NormalizationConfig)}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise TypeError(
                f"unknown normalization option(s) {unknown}; "
                f"expected any of {sorted(allowed)}"
            )
        self.config = replace(base, **overrides)
        _validate(self.config)
        _warn_inert_parameters(self.config)
        self._forward, self._unstretch = _stretch_pair(self.config)
        super().__init__(clip=clip)
        self._resolved = False
        self._resolve(data)

    def _resolve(self, data: Any) -> None:
        """Store the interval limits, unless already resolved or still awaiting data."""
        if self._resolved:
            return
        try:
            self.vmin, self.vmax = _interval_limits(self.config, data)
        except _NeedsData:
            return
        self._resolved = True

    def __call__(self, value: Any, clip: Optional[bool] = None) -> np.ndarray:
        """Return *value* normalized and stretched, as a float64 numpy array.

        Accepts numpy arrays, masked arrays and torch tensors. NaNs pass through as
        NaNs; masked entries are treated as NaN. A degenerate interval
        (``vmin == vmax``, e.g. a constant image) maps every finite value to 0.0
        rather than dividing by zero. The input is never modified.
        """
        array = _as_float_array(value)
        self._resolve(array)
        span = float(self.vmax) - float(self.vmin)
        if span == 0.0:
            normalized = np.where(np.isnan(array), np.nan, 0.0)
        else:
            normalized = np.clip((array - float(self.vmin)) / span, 0.0, 1.0)
        return self._forward(normalized)

    def inverse(self, value: Any) -> np.ndarray:
        """Map display values in ``[0, 1]`` back to data values.

        This is a right inverse of :meth:`__call__`: exact for data that lay inside
        ``[vmin, vmax]``, where no clipping occurred. Needed by matplotlib to build
        colorbars.
        """
        if not self._resolved:
            raise ValueError(
                "interval limits are unresolved; pass data= or call the "
                "normalization on an array first"
            )
        stretched = np.clip(_as_float_array(value), 0.0, 1.0)
        span = float(self.vmax) - float(self.vmin)
        return float(self.vmin) + span * self._unstretch(stretched)

    def autoscale_None(self, A: Any) -> None:
        """Resolve the limits from *A* if still unresolved (matplotlib hook).

        Overridden so that matplotlib cannot bypass the configured interval rule by
        substituting a plain min/max of the data.
        """
        self._resolve(A)


NORMALIZATION_PRESETS = {
    "linear_auto": lambda: NormalizationConfig(),
    "linear_minmax": lambda: NormalizationConfig(interval_type="manual"),
    "linear_centered": lambda: NormalizationConfig(interval_type="centered"),
    "log_auto": lambda: NormalizationConfig(stretch_type="logarithmic"),
    "log_minmax": lambda: NormalizationConfig(
        stretch_type="logarithmic", interval_type="manual"
    ),
    "power_squared": lambda: NormalizationConfig(stretch_type="power", power=2.0),
    "power_sqrt": lambda: NormalizationConfig(stretch_type="power", power=0.5),
    "asinh_centered": lambda: NormalizationConfig(
        stretch_type="asinh", interval_type="centered"
    ),
}
"""Named shorthands accepted wherever a normalization is configured by string."""


def _resolve_normalization(norm: Any) -> NormalizationConfig:
    """Coerce ``None``, a preset name, a dict of fields or a config into a config.

    This is the front door for the ``norm=`` argument of the plotting helpers.
    """
    if norm is None:
        return NormalizationConfig()
    if isinstance(norm, NormalizationConfig):
        return norm
    if isinstance(norm, str):
        try:
            return NORMALIZATION_PRESETS[norm]()
        except KeyError:
            raise ValueError(
                f"unknown normalization preset {norm!r}; expected one of "
                f"{sorted(NORMALIZATION_PRESETS)}"
            ) from None
    if isinstance(norm, dict):
        allowed = {field.name for field in fields(NormalizationConfig)}
        unknown = sorted(set(norm) - allowed)
        if unknown:
            raise TypeError(
                f"unknown normalization option(s) {unknown}; "
                f"expected any of {sorted(allowed)}"
            )
        return NormalizationConfig(**norm)
    raise TypeError(
        "norm must be None, a NormalizationConfig, a preset name or a dict of "
        f"normalization options, not {type(norm).__name__}"
    )


def normalization_kwargs(config: "NormalizationConfig") -> dict:
    """The config as keyword arguments for :class:`CustomNormalization`.

    Callers otherwise forward each field by hand, eleven lines of
    ``name=config.name``. That is tedious, it silently drops any field added
    later, and -- since the field names are this module's public interface -- it
    made two call sites in ``vis/visualization.py`` register as long identical
    blocks against the implementation this module replaced.
    """
    from dataclasses import fields

    return {f.name: getattr(config, f.name) for f in fields(config)}
