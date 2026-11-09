"""Angular-momentum algebra for inelastic transition potentials.

Independent implementation of the Racah single-sum formula for the Wigner 3-j
symbol, plus thin wrappers of the scipy spherical harmonic and spherical Bessel
function, for the ionization transition potentials (Brown *et al.*, Phys. Rev.
Research **1**, 033186 (2019); Dwyer, Ultramicroscopy **104**, 141 (2005)).
The scatterem authors wrote this module in 2026-05, in the same commit as the
former EELS port. It is not part of the 2026-11 clean-room rewrite of
``radial.py`` and ``transition_potentials.py``.

Only integer angular momenta occur for the orbital transitions we model, so the
Wigner 3-j symbol is evaluated directly from the Racah single-sum formula rather
than pulling in ``sympy``.  Results are cached because the same handful of
symbols is reused across every reciprocal-space grid point.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial, sqrt

from scipy.special import sph_harm_y, spherical_jn

__all__ = ["wigner_3j", "spherical_harmonic", "spherical_bessel"]


def _triangle_ok(j1: int, j2: int, j3: int) -> bool:
    return abs(j1 - j2) <= j3 <= (j1 + j2)


@lru_cache(maxsize=4096)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3-j symbol for integer arguments via the Racah formula.

    Returns ``0.0`` whenever a selection rule is violated.  Matches the
    convention used by ``sympy.physics.wigner.wigner_3j``.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if not _triangle_ok(j1, j2, j3):
        return 0.0
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if abs(m) > j:
            return 0.0

    # Prefactor (square root of a ratio of factorials).
    pref_num = (
        factorial(j1 + j2 - j3)
        * factorial(j1 - j2 + j3)
        * factorial(-j1 + j2 + j3)
        * factorial(j1 - m1)
        * factorial(j1 + m1)
        * factorial(j2 - m2)
        * factorial(j2 + m2)
        * factorial(j3 - m3)
        * factorial(j3 + m3)
    )
    pref_den = factorial(j1 + j2 + j3 + 1)
    prefactor = sqrt(pref_num / pref_den)

    # Summation over the integer t for which every factorial argument is >= 0.
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    summation = 0.0
    for t in range(t_min, t_max + 1):
        denom = (
            factorial(t)
            * factorial(j3 - j2 + m1 + t)
            * factorial(j3 - j1 - m2 + t)
            * factorial(j1 + j2 - j3 - t)
            * factorial(j1 - m1 - t)
            * factorial(j2 + m2 - t)
        )
        summation += (-1) ** t / denom

    return (-1) ** (j1 - j2 - m3) * prefactor * summation


def spherical_harmonic(m: int, l: int, azimuth, polar):
    """Complex spherical harmonic ``Y_l^m`` with ``azimuth`` (φ) and ``polar`` (θ).

    A wrapper of ``scipy.special.sph_harm_y(l, m, polar, azimuth)`` (scipy 1.15
    or newer, the dependency floor).
    """
    return sph_harm_y(l, m, polar, azimuth)


def spherical_bessel(l: int, x):
    return spherical_jn(l, x)
