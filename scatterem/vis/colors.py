"""Scientific colour maps, chosen from what the data is.

This module puts Fig. 6 of Crameri, Shephard & Heron (2020), "The misuse of
colour in science communication", Nat. Commun. 11, 5444
(doi:10.1038/s41467-020-19160-7) into code. A caller describes the data with
:class:`DataTraits` (the answers to the Fig. 6 questions) or names a role from
:data:`ROLES`. :func:`choose_colormap` then picks a map from the Scientific
Colour Maps 8.0.1 (doi:10.5281/zenodo.8409685, through ``cmcrameri``). It
records each question and its answer in :attr:`ColourChoice.trail`.

The user guide page ``user_guide/colour`` shows the decision tree and a gallery.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any

import cmcrameri.cm as cmc
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, Normalize
from numpy.typing import ArrayLike

from scatterem.vis._colormap_policy import is_allowed, suggestion_for
from scatterem.vis._colour_metrics import min_delta_e_to
from scatterem.vis._colour_metrics import simulate_cvd as simulate_cvd
from scatterem.vis._colour_metrics import srgb_to_lab as srgb_to_lab

__all__ = [
    "ColourChoice",
    "DataTraits",
    "UnscientificColormapWarning",
    "choose_colormap",
    "colormap_for_role",
    "register_colormaps",
    "resolve_display",
    "simulate_cvd",
    "srgb_to_lab",
    "warn_if_unscientific",
]

#: RGB of the figure background that each ``background`` value stands for.
_BACKGROUND_RGB: dict[str, tuple[float, float, float]] = {
    "light": (1.0, 1.0, 1.0),
    "dark": (0.0, 0.0, 0.0),
}
_OPPOSITE = {"light": "dark", "dark": "light"}
_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

#: Smallest CIELAB Delta E 1976 between every map colour and the colour NaN is
#: drawn in. Measured on 2026-10-14 with 256 samples. To white: batlow 29.8,
#: lipari 14.1, oleron 9.2, vik 8.4, batlowW 0.5, grayC 0.0. To black: batlow
#: 34.8, berlin 5.2, batlowK 2.6. Ten is about four just-noticeable differences.
BACKGROUND_MIN_DE = 10.0

#: Candidate maps per Fig. 6 class and background, in order of preference.
_CANDIDATES: dict[tuple[str, str], tuple[str, ...]] = {
    # No intuitive colours: a light figure takes the dark part for high values,
    # a dark figure the light part.
    ("sequential", "light"): ("batlow_r",),
    ("sequential", "dark"): ("batlow",),
    # A centric value: dark centre on a light figure, light centre on a dark one.
    # The paper states this on the intuitive-colours branch; scatterem applies it
    # to all diverging data, because a light centre on white makes NaN read as
    # zero (spec section 3.1).
    ("diverging", "light"): ("berlin",),
    ("diverging", "dark"): ("vik",),
    ("multi-sequential", "light"): ("oleron", "bukavu"),
    ("multi-sequential", "dark"): ("oleron", "bukavu"),
    ("cyclic", "light"): ("romaO",),
    ("cyclic", "dark"): ("romaO",),
}


class UnscientificColormapWarning(UserWarning):
    """A plot was asked to use a colour map outside the scientific allow-list."""


@dataclass(frozen=True)
class DataTraits:
    """The answers to the questions of Fig. 6 of Crameri et al. (2020).

    Parameters
    ----------
    ordered : bool, default=True
        Has the data an order? False gives a categorical map.
    centre : float, optional
        The central value of the data, if it has one (for example 0).
    ordered_about_centre : bool, default=True
        With a centre: True gives a diverging map, False a multi-sequential one.
    period : float, optional
        The period of periodic data (for example ``2 * pi``); gives a cyclic map.
        ``show_2d`` wraps the values into ``[-period/2, period/2)``, so data on
        any interval of one period shows correctly.
    small_variations : bool, default=True
        Must small variations show? True gives a continuous map, False discrete.
    levels : int, optional
        Number of colours of a discrete map. Default 10.
    intuitive : str, optional
        A map with colours that suit the parameter, if there is one. A map
        outside the scientific allow-list warns.
    """

    ordered: bool = True
    centre: float | None = None
    ordered_about_centre: bool = True
    period: float | None = None
    small_variations: bool = True
    levels: int | None = None
    intuitive: str | None = None


@dataclass(frozen=True)
class ColourChoice:
    """A colour map chosen by :func:`choose_colormap`, with its reasons.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The map, with its NaN ("bad") colour set.
    cls : str
        Fig. 6 class: sequential, diverging, multi-sequential or cyclic.
    kind : str
        Fig. 6 type: continuous, discrete or categorical.
    trail : tuple of (str, object)
        Each Fig. 6 question and its answer, in order.
    centre : float, optional
        Central value of diverging and multi-sequential data.
    period : float, optional
        Period of cyclic data.
    """

    cmap: Colormap
    cls: str
    kind: str
    trail: tuple[tuple[str, object], ...]
    centre: float | None = None
    period: float | None = None

    @property
    def name(self) -> str:
        """The name of the chosen map."""
        return self.cmap.name

    def limits(self, data: ArrayLike | None = None) -> tuple[float, float] | None:
        """Display limits that the class needs, or None when the data decides.

        Parameters
        ----------
        data : array_like, optional
            The values to show. Needed for a centred class.

        Returns
        -------
        tuple of float or None
            One period for a cyclic class. For a centred class, limits symmetric
            about the centre, so the map is truncated and never stretched on one
            side. None otherwise.
        """
        if self.cls == "cyclic" and self.period is not None:
            return (-self.period / 2.0, self.period / 2.0)
        if self.centre is None or data is None:
            return None
        values = np.abs(np.asarray(data, dtype=float) - self.centre)
        finite = values[np.isfinite(values)]
        half = float(finite.max()) if finite.size else 0.0
        half = half if half > 0.0 else 1.0
        return (self.centre - half, self.centre + half)

    def wrap(self, data: ArrayLike) -> np.ndarray:
        """``data`` wrapped into the period that a cyclic class shows.

        The limits of a cyclic class are ``[-period/2, period/2]``. A value out
        of them is wrapped by whole periods, not clipped, so ``[0, 2 pi)``
        phases keep all their colours.

        Parameters
        ----------
        data : array_like
            The values to show.

        Returns
        -------
        numpy.ndarray
            ``mod(data + period/2, period) - period/2`` for a cyclic class;
            ``data`` as an array otherwise. NaN stays NaN.
        """
        values = np.asarray(data)
        if self.cls != "cyclic" or self.period is None:
            return values
        half = self.period / 2.0
        return np.mod(values + half, self.period) - half

    def norm(self, data: ArrayLike | None = None) -> Normalize | None:
        """A matplotlib ``Normalize`` for :meth:`limits`, or None.

        Parameters
        ----------
        data : array_like, optional
            The values to show.

        Returns
        -------
        matplotlib.colors.Normalize or None
            Ready for ``imshow(norm=...)``.
        """
        lim = self.limits(data)
        return None if lim is None else Normalize(*lim)

    def display_norm(self) -> dict[str, Any] | None:
        """The ``NormalizationConfig`` fields that ``show_2d`` needs, or None.

        Returns
        -------
        dict or None
            ``"centered"`` about the centre for a centred class, ``"manual"`` over
            one period for a cyclic class, ``"manual"`` over the full range for a
            categorical class, None to keep the default.
        """
        if self.cls == "cyclic" and self.period is not None:
            half = self.period / 2.0
            return {"interval_type": "manual", "vmin": -half, "vmax": half}
        if self.centre is not None:
            return {"interval_type": "centered", "vcenter": float(self.centre)}
        if self.kind == "categorical":
            # Full finite range, so every label gets its own colour.
            return {"interval_type": "manual"}
        return None


#: Named answers to Fig. 6 for the data scatterem shows.
ROLES: dict[str, DataTraits] = {
    "intensity": DataTraits(intuitive="batlow"),
    "diffraction": DataTraits(intuitive="batlowK"),
    "phase": DataTraits(intuitive="batlow"),
    "signed": DataTraits(centre=0.0),
    "angle": DataTraits(period=2.0 * math.pi),
    "labels": DataTraits(ordered=False),
    "uncertainty": DataTraits(),
    "species_a": DataTraits(intuitive="devon"),
    "species_b": DataTraits(intuitive="lajolla"),
}


@functools.cache
def register_colormaps() -> None:
    """Register every Crameri map with matplotlib under its bare name.

    cmcrameri registers ``"cmc.batlow"``. This adds ``"batlow"`` and
    ``"batlow_r"`` too. Names matplotlib already has (berlin, managua, vanimo:
    the same tables) are left alone. Nothing else in matplotlib changes. The
    function runs once per process.
    """
    for name, cmap in cmc.cmaps.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)


def _classify(traits: DataTraits) -> tuple[str, str, list[tuple[str, object]]]:
    """Stage 1 of Fig. 6: class and type."""
    trail: list[tuple[str, object]] = [("ordered", traits.ordered)]
    if not traits.ordered:
        return "sequential", "categorical", trail
    trail.append(("centric value", traits.centre is not None))
    if traits.centre is not None:
        trail.append(("ordered about centre", traits.ordered_about_centre))
        cls = "diverging" if traits.ordered_about_centre else "multi-sequential"
    else:
        trail.append(("periodic", traits.period is not None))
        cls = "cyclic" if traits.period is not None else "sequential"
    trail.append(("important small variations", traits.small_variations))
    kind = "continuous" if traits.small_variations else "discrete"
    return cls, kind, trail


def _candidates(
    cls: str, traits: DataTraits, background: str, trail: list[tuple[str, object]]
) -> tuple[str, ...]:
    """Stage 2 of Fig. 6: candidate maps for the colour combination."""
    trail.append(("intuitive colours", traits.intuitive))
    trail.append(("background", background))
    default = _CANDIDATES[(cls, background)]
    if traits.intuitive is None:
        return default
    warn_if_unscientific(traits.intuitive, stacklevel=4)  # not exempt
    # An intuitive sequential map means "high is light". Its NaN fallback keeps
    # that direction, so it is batlow and not batlow_r.
    fallback = ("batlow",) if cls == "sequential" else default
    return (traits.intuitive, *fallback)


@functools.cache
def _distance(name: str, background: str) -> float:
    """Smallest Delta E between the map ``name`` and a background colour."""
    colours = mpl.colormaps[name](np.linspace(0.0, 1.0, 256))
    return min_delta_e_to(colours, _BACKGROUND_RGB[background])


def _with_nan_colour(
    names: tuple[str, ...], background: str, has_nan: bool
) -> tuple[Colormap, str]:
    """Fig. 6 "empty values": NaN must not look like any colour of the map."""
    if not has_nan:
        return mpl.colormaps[names[0]].with_extremes(bad=_TRANSPARENT), "transparent"
    for name in names:
        if _distance(name, background) >= BACKGROUND_MIN_DE:
            return mpl.colormaps[name].with_extremes(bad=_TRANSPARENT), "transparent"
    other = _OPPOSITE[background]
    if _distance(names[0], other) >= BACKGROUND_MIN_DE:
        bad = (*_BACKGROUND_RGB[other], 1.0)
        return mpl.colormaps[names[0]].with_extremes(bad=bad), other
    raise ValueError(f"no NaN colour keeps {names} apart from a {background} figure")


def choose_colormap(
    traits: DataTraits,
    *,
    data: ArrayLike | None = None,
    background: str = "light",
) -> ColourChoice:
    """Choose a colour map by Fig. 6 of Crameri, Shephard & Heron (2020).

    Parameters
    ----------
    traits : DataTraits
        The answers to the Fig. 6 questions.
    data : array_like, optional
        The values to show. Used only to detect NaN.
    background : {"light", "dark"}, default="light"
        The figure background.

    Returns
    -------
    ColourChoice
        The map, its class and type, and the trail of answers.

    Raises
    ------
    ValueError
        If ``background`` is not "light" or "dark".
    """
    if background not in _BACKGROUND_RGB:
        raise ValueError(f"background must be 'light' or 'dark', not {background!r}")
    register_colormaps()
    cls, kind, trail = _classify(traits)
    if kind == "categorical":
        return ColourChoice(mpl.colormaps["batlowS"], cls, kind, tuple(trail))
    names = _candidates(cls, traits, background, trail)
    has_nan = data is not None and bool(np.isnan(np.asarray(data)).any())
    trail.append(("empty values", has_nan))
    cmap, nan_colour = _with_nan_colour(names, background, has_nan)
    trail.append(("nan colour", nan_colour))
    if kind == "discrete":
        cmap = cmap.resampled(traits.levels or 10)
    trail.append(("map", cmap.name))
    return ColourChoice(
        cmap, cls, kind, tuple(trail), centre=traits.centre, period=traits.period
    )


def colormap_for_role(
    role: str, *, data: ArrayLike | None = None, background: str = "light"
) -> ColourChoice:
    """Choose the colour map for one of the named :data:`ROLES`.

    Parameters
    ----------
    role : str
        A key of :data:`ROLES`, for example ``"intensity"`` or ``"signed"``.
    data : array_like, optional
        The values to show. Used only to detect NaN.
    background : {"light", "dark"}, default="light"
        The figure background.

    Returns
    -------
    ColourChoice
        As :func:`choose_colormap`.

    Raises
    ------
    KeyError
        If ``role`` is unknown. The message lists the roles.
    """
    try:
        traits = ROLES[role]
    except KeyError:
        raise KeyError(f"unknown role {role!r}; roles are {sorted(ROLES)}") from None
    return choose_colormap(traits, data=data, background=background)


def warn_if_unscientific(cmap: str | Colormap, stacklevel: int = 3) -> None:
    """Warn when ``cmap`` is not on the scientific allow-list.

    The ``warnings`` filters decide how often the warning shows, so
    ``simplefilter("error")`` catches every call. ``show_2d`` warns once per
    call, not once per panel. For one warning per process, use
    ``warnings.filterwarnings("once", category=UnscientificColormapWarning)``.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        The map a caller asked for. An object is judged by its ``name``.
    stacklevel : int, default=3
        Passed to :func:`warnings.warn`, so the warning points at the caller.
    """
    name = cmap if isinstance(cmap, str) else cmap.name
    if is_allowed(name):
        return
    warnings.warn(
        f"colour map {name!r} is not scientifically derived (Crameri et al. 2020, "
        f"Nat. Commun. 11, 5444); use role={suggestion_for(name)!r} or a map from "
        "the Scientific Colour Maps",
        UnscientificColormapWarning,
        stacklevel=stacklevel,
    )


def resolve_display(
    cmap: str | Colormap | None,
    role: str,
    norm: Any,
    *,
    data: ArrayLike | None = None,
    background: str = "light",
) -> tuple[ColourChoice, Any]:
    """The colour choice and normalisation for one displayed image.

    Parameters
    ----------
    cmap : str or Colormap or None
        An explicit map. It wins over ``role``, with a warning if unscientific.
    role : str
        A key of :data:`ROLES`. It sets the class and the limits.
    norm : Any
        The caller's normalisation. None takes the role's limits.
    data : array_like, optional
        The values to show. Used to detect NaN and, for a centred class, to set
        the half-range of the limits.
    background : {"light", "dark"}, default="light"
        The figure background.

    Returns
    -------
    tuple
        ``(ColourChoice, norm)``, where ``norm`` is the caller's or the role's.
    """
    choice = colormap_for_role(role, data=data, background=background)
    if cmap is not None:
        warn_if_unscientific(cmap)
        explicit = mpl.colormaps.get_cmap(cmap)
        trail = (*choice.trail, ("explicit cmap", explicit.name))
        choice = dataclasses.replace(choice, cmap=explicit, trail=trail)
    if norm is not None:
        return choice, norm
    display = choice.display_norm()
    if (
        display is not None
        and display["interval_type"] == "centered"
        and data is not None
    ):
        # limits() keeps a zero half-range away from vmin == vmax, so data equal
        # to the centre everywhere shows the centre colour, not the negative end.
        lo, hi = choice.limits(data) or (0.0, 0.0)
        display = {**display, "half_range": (hi - lo) / 2.0}
    return choice, display
