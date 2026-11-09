"""The ``wavefunctions`` choice of the EELS radial states: ``"auto"`` and its fallback.

``"auto"`` (the default everywhere) resolves to GPAW when it is importable and to
the analytic hydrogenic states otherwise.  GPAW is an optional GPL extra
(``scatterem[eels]``), so the fallback is allowed, but it warns: the numbers then
differ from the GPAW results.  GPAW is never imported here; availability is
:func:`scatterem.simulation.eels.physics.radial.gpaw_available` (``find_spec``).
"""

from __future__ import annotations

from scatterem._warn import warn_outside_package

__all__ = ["HydrogenicFallbackWarning", "resolve_wavefunctions"]


class HydrogenicFallbackWarning(UserWarning):
    """``wavefunctions="auto"`` fell back to the analytic hydrogenic family.

    GPAW is not importable, so the bound and continuum states are the analytic
    approximations.  Cross-sections and EELS images then differ from the GPAW
    results and are not quantitative.
    """


_WAVEFUNCTIONS = ("auto", "gpaw", "hydrogenic")


def resolve_wavefunctions(wavefunctions: str) -> str:
    """Resolve a ``wavefunctions`` choice to ``"gpaw"`` or ``"hydrogenic"``.

    Parameters
    ----------
    wavefunctions : str
        ``"auto"``, ``"gpaw"`` or ``"hydrogenic"``.  ``"auto"`` resolves to
        ``"gpaw"`` when GPAW is importable, else to ``"hydrogenic"`` with a
        :class:`HydrogenicFallbackWarning`.

    Returns
    -------
    str
        ``"gpaw"`` or ``"hydrogenic"``.

    Raises
    ------
    ValueError
        If ``wavefunctions`` is not one of the three choices.
    """
    if wavefunctions not in _WAVEFUNCTIONS:
        raise ValueError(f"unknown wavefunctions {wavefunctions!r}")
    if wavefunctions != "auto":
        return wavefunctions
    # Late import of radial: tests patch radial.gpaw_available.
    from . import radial

    if radial.gpaw_available():
        return "gpaw"
    warn_outside_package(
        'wavefunctions="auto": GPAW is not installed, so the analytic '
        "hydrogenic bound and free continuum states are used. These EELS "
        "numbers are not quantitative and differ from the GPAW results. "
        "Install the eels extra to use GPAW. v0.4 uses a new absolute "
        "first-Born scale: the transition potentials in Angstrom^2/eV, the "
        "maps in probability per incident electron per eV. After a best "
        "scalar fit, the v0.4 map shapes differ from v0.3 by 18-61 % with these "
        "hydrogenic states. With GPAW they differ by 0.03-0.6 % for the K, L23 "
        "and M45 edges tested and by 1.4 % for Fe L1. "
        'Pass wavefunctions="hydrogenic" to choose the analytic states '
        "explicitly.",
        HydrogenicFallbackWarning,
    )
    return "hydrogenic"
