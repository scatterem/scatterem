"""Choice between the Fourier-space and the real-space routes of the potential.

Literature: the real-space route adds truncated atomic potentials, and the
Fourier-space route sums the Fourier series of the periodic cell (Kirkland
2020, *Advanced Computing in Electron Microscopy*, 3rd ed., ch. 5 and 6).
The phase at the truncation radius uses the interaction parameter of
Kirkland (2020, ch. 5). Written for scatterem from this literature.

Units: ``volume_cutoff`` [Volt], ``projected_cutoff`` [Volt Angstrom].
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass

import numpy as np

from .potential_rolloff import DEFAULT_ROLLOFF, check_rolloff

__all__ = [
    "DEFAULT_PROJECTED_CUTOFF",
    "ROUTE_NAMES",
    "SpatialRouteSettings",
    "check_cutoff",
    "check_route",
    "choose_route",
    "route_rolloff",
]

_log = logging.getLogger(__name__)

#: Route spellings: the Fourier-space route and the real-space route.
ROUTE_NAMES = ("spectral", "spatial")

_FREE = "either"
_AXIAL_MODES = ("projected", "sectioned")

#: Projected potential at the truncation radius [Volt Angstrom]. Reason,
#: measured on 2026-11-08 against a cut at 1e-6 V Angstrom (3 x 3 cells,
#: 0.05 Angstrom pixels, one slice per cell): at 300 kV (``sigma = 6.526e-4``
#: rad/(V Angstrom)) the largest phase error per 10 nm of specimen is
#: 3.9e-5 rad on the Si, Au, Cu and MoS2 test CIFs (MoS2 the largest), half
#: the 7.8e-5 rad of a cut at 1e-3. The radius is then 3.2 (H), 3.8 (C),
#: 4.9 (Si), 5.2 (Cu) and 4.9 (Au) Angstrom, about 6 % more than at 1e-3;
#: the build time of these cells grew by 5 to 25 %.
DEFAULT_PROJECTED_CUTOFF = 5e-4

#: Rolloff that the free choice gives to each route.
_FREE_ROLLOFF = {"spectral": DEFAULT_ROLLOFF, "spatial": None}


def check_route(value) -> str:
    """Return a route name after a check.

    Parameters
    ----------
    value : str
        One of :data:`ROUTE_NAMES`.

    Returns
    -------
    str
        The checked route name.

    Raises
    ------
    TypeError
        If ``value`` is not a ``str``.
    ValueError
        If ``value`` is another string (also ``"either"``).
    """
    if not isinstance(value, str):
        raise TypeError(f"route: must be a str; got {value!r}")
    if value not in ROUTE_NAMES:
        raise ValueError(f"route: must be one of {ROUTE_NAMES}; got {value!r}")
    return value


def choose_route(*, multi_device, preference=_FREE) -> str:
    """Resolve the route choice of a simulator.

    Parameters
    ----------
    multi_device : bool
        True if the run is distributed over several devices.
    preference : str, optional
        ``"either"`` (default) or a name of :data:`ROUTE_NAMES`.

    Returns
    -------
    str
        For ``"either"``: ``"spatial"`` when ``multi_device``, else
        ``"spectral"``. A route name gives itself.

    Raises
    ------
    TypeError
        If ``multi_device`` is not a bool or ``preference`` not a str.
    ValueError
        For an unknown ``preference``.
    """
    if not isinstance(multi_device, (bool, np.bool_)):
        raise TypeError(f"multi_device: must be a bool; got {multi_device!r}")
    if preference != _FREE:
        return check_route(preference)
    return "spatial" if multi_device else "spectral"


def route_rolloff(*, route, preference: float | str | None = _FREE) -> float | None:
    """Resolve the potential rolloff for a route.

    Parameters
    ----------
    route : str
        A name of :data:`ROUTE_NAMES`.
    preference : str or float or None, optional
        ``"either"`` (default), ``None`` (no rolloff) or a rolloff depth.

    Returns
    -------
    float or None
        ``"spectral"``: ``DEFAULT_ROLLOFF`` for ``"either"``, ``None`` for
        ``None``, ``check_rolloff(w)`` for a number. ``"spatial"``: ``None``
        for ``"either"`` and ``None``.

    Raises
    ------
    ValueError
        For a rolloff depth with the ``"spatial"`` route (low-pass filter the
        transmission function instead), or a string other than ``"either"``.
    """
    target = check_route(route)
    if not isinstance(preference, str):
        depth = check_rolloff(preference)
        if depth is not None and target == "spatial":
            raise ValueError(
                "preference: the spatial route takes no potential rolloff; "
                "low-pass filter the transmission function with "
                "thin_slice_transmission(..., bandlimit=...) instead"
            )
        return depth
    if preference == _FREE:
        return _FREE_ROLLOFF[target]
    raise ValueError(
        "preference: must be 'either', None or a rolloff depth in (0, 1); "
        f"got {preference!r}"
    )


def check_cutoff(cutoff_level, *, field_label: str) -> float:
    """Return a positive, finite threshold as a Python float.

    Parameters
    ----------
    cutoff_level : float
        Threshold [Volt or Volt Angstrom].
    field_label : str
        Name of the checked field, for the message.

    Returns
    -------
    float
        The checked threshold.

    Raises
    ------
    TypeError
        For ``bool``, ``str``, complex or any non-real value.
    ValueError
        For a value that is not finite or ``<= 0``.
    """
    if isinstance(cutoff_level, (bool, np.bool_)) or not isinstance(
        cutoff_level, numbers.Real
    ):
        raise TypeError(f"{field_label}: must be a real number; got {cutoff_level!r}")
    level = float(cutoff_level)
    if not math.isfinite(level) or level <= 0.0:
        raise ValueError(f"{field_label}: must be finite and > 0; got {cutoff_level!r}")
    return level


@dataclass(frozen=True)
class SpatialRouteSettings:
    """Settings of the real-space route (immutable, hashable).

    Parameters
    ----------
    axial_mode : str, optional
        ``"projected"`` (default): each atom is integrated through all depth
        and put wholly into the slice that holds it; this matches the
        default ``"binned"`` spread of the Fourier route and is cheaper.
        ``"sectioned"``: the integral of the 3-D potential over each slice
        that the atom reaches.
    volume_cutoff : float or None, optional
        3-D potential at the truncation radius [Volt], ``"sectioned"`` only.
        ``None`` (default) means the route's own value.
    projected_cutoff : float, optional
        Projected potential at the truncation radius [Volt Angstrom].
        Default :data:`DEFAULT_PROJECTED_CUTOFF`.

    Raises
    ------
    TypeError
        For a non-str mode or a non-real threshold.
    ValueError
        For an unknown mode, a threshold that is not finite or ``<= 0``, or a
        ``volume_cutoff`` with ``"projected"``.
    """

    axial_mode: str = "projected"
    volume_cutoff: float | None = None
    projected_cutoff: float = DEFAULT_PROJECTED_CUTOFF

    def __post_init__(self) -> None:
        if not isinstance(self.axial_mode, str):
            raise TypeError(f"axial_mode: must be a str; got {self.axial_mode!r}")
        if self.axial_mode not in _AXIAL_MODES:
            raise ValueError(
                f"axial_mode: must be one of {_AXIAL_MODES}; got {self.axial_mode!r}"
            )
        if self.volume_cutoff is not None:
            vol = check_cutoff(self.volume_cutoff, field_label="volume_cutoff")
            if self.axial_mode == "projected":
                raise ValueError(
                    "volume_cutoff: applies only to axial_mode='sectioned'"
                )
            object.__setattr__(self, "volume_cutoff", vol)
        proj = check_cutoff(self.projected_cutoff, field_label="projected_cutoff")
        object.__setattr__(self, "projected_cutoff", proj)
