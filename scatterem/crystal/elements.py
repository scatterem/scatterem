"""The element table: chemical symbols and atomic numbers.

- ``SYMBOLS`` is the chemical symbol of each element, indexed by atomic
  number, through Z = 118. Index 0 is the empty string, so ``SYMBOLS[z]`` is
  the symbol of Z. It is a plain tuple; like the other module constants of
  :mod:`scatterem.crystal`, it is not in ``__all__`` (a tuple carries no
  docstring for the numpydoc gate), but importing it by name is supported.
- :func:`atomic_number` reads a symbol (any case) or a numeric string.
- :func:`symbol` returns the symbol of an atomic number.
- ``STANDARD_ATOMIC_WEIGHTS`` is the abridged standard atomic weight of each
  element, indexed by atomic number like ``SYMBOLS``. ``atomic_weight``
  reads it by symbol or number. The table is plain data, so every layer above
  ``crystal`` (``physics`` included) can use it without ``ase``.
"""

from __future__ import annotations

__all__ = ["atomic_number", "atomic_weight", "symbol"]

#: Chemical symbol of each element, indexed by atomic number (index 0 is
#: empty), through Z = 118 -- IUPAC names as of 2016. Scattering factors
#: (:mod:`scatterem.physics.atom_amplitudes`) exist for Z = 1..103 only, the range
#: of the Lobato & Van Dyck table.
SYMBOLS: tuple[str, ...] = (
    "",
    *"H He Li Be B C N O F Ne".split(),
    *"Na Mg Al Si P S Cl Ar K Ca".split(),
    *"Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(),
    *"Ga Ge As Se Br Kr Rb Sr Y Zr".split(),
    *"Nb Mo Tc Ru Rh Pd Ag Cd In Sn".split(),
    *"Sb Te I Xe Cs Ba La Ce Pr Nd".split(),
    *"Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split(),
    *"Lu Hf Ta W Re Os Ir Pt Au Hg".split(),
    *"Tl Pb Bi Po At Rn Fr Ra Ac Th".split(),
    *"Pa U Np Pu Am Cm Bk Cf Es Fm".split(),
    *"Md No Lr Rf Db Sg Bh Hs Mt Ds".split(),
    *"Rg Cn Nh Fl Mc Lv Ts Og".split(),
)

#: Abridged standard atomic weight of each element, indexed by atomic number
#: (index 0 is ``None``). Source: CIAAW, "Abridged Standard Atomic Weights
#: 2024", https://www.ciaaw.org/abridged-atomic-weights.htm (read 2026-10-22).
#: The values are dimensionless relative masses (A_r); numerically they are
#: the mean atomic mass in u (Da) of the element with a normal isotopic
#: composition. An element with no stable isotope has no standard weight
#: (Tc, Pm, Po-Ac, Np and above): its entry is ``None``. Pass the mass of the
#: isotope instead, where a calculation needs one.
# fmt: off
STANDARD_ATOMIC_WEIGHTS: tuple[float | None, ...] = (
    None,
    1.0080, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999,  # Z = 1-8
    18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06,  # Z = 9-16
    35.45, 39.95, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996,  # Z = 17-24
    54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.630,  # Z = 25-32
    74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.222,  # Z = 33-40
    92.906, 95.95, None, 101.07, 102.91, 106.42, 107.87, 112.41,  # Z = 41-48
    114.82, 118.71, 121.76, 127.60, 126.90, 131.29, 132.91, 137.33,  # Z = 49-56
    138.91, 140.12, 140.91, 144.24, None, 150.36, 151.96, 157.25,  # Z = 57-64
    158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,  # Z = 65-72
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,  # Z = 73-80
    204.38, 207.2, 208.98, None, None, None, None, None,  # Z = 81-88
    None, 232.04, 231.04, 238.03, None, None, None, None,  # Z = 89-96
    None, None, None, None, None, None, None, None,  # Z = 97-104
    None, None, None, None, None, None, None, None,  # Z = 105-112
    None, None, None, None, None, None,  # Z = 113-118
)
# fmt: on

_Z_BY_SYMBOL = {symbol: z for z, symbol in enumerate(SYMBOLS) if symbol}


def atomic_number(symbol: str) -> int:
    """Return the atomic number of an element symbol.

    Parameters
    ----------
    symbol : str
        Element symbol such as ``"Si"``. Case is normalised (``"SI"`` and
        ``"si"`` are accepted); a numeric string is read as the atomic number.

    Returns
    -------
    int
        The atomic number.

    Raises
    ------
    ValueError
        If `symbol` is not an element symbol.
    """
    token = symbol.strip()
    if token.isdigit():
        return int(token)
    z = _Z_BY_SYMBOL.get(token[:1].upper() + token[1:].lower())
    if z is None:
        raise ValueError(f"Unrecognized element symbol {symbol!r}")
    return z


def symbol(z: int) -> str:
    """Return the chemical symbol of an atomic number.

    Parameters
    ----------
    z : int
        Atomic number, 1 to 118.

    Returns
    -------
    str
        The symbol, for example ``"Si"`` for 14.

    Raises
    ------
    ValueError
        If `z` is not between 1 and 118.
    """
    index = int(z)
    if not 0 < index < len(SYMBOLS):
        raise ValueError(f"atomic number must be 1..{len(SYMBOLS) - 1}, got {z!r}")
    return SYMBOLS[index]


def atomic_weight(element: str | int) -> float:
    """Return the abridged standard atomic weight of an element.

    Parameters
    ----------
    element : str or int
        Element symbol (any case, as :func:`atomic_number` reads it) or atomic
        number.

    Returns
    -------
    float
        The CIAAW 2024 abridged standard atomic weight. Numerically it is the
        mean atomic mass in u (Da), for example 12.011 for carbon.

    Raises
    ------
    ValueError
        If `element` is not an element, or if the element has no standard
        atomic weight (no stable isotope, for example Tc or Pm).
    """
    z = atomic_number(element) if isinstance(element, str) else int(element)
    weight = STANDARD_ATOMIC_WEIGHTS[z] if 0 < z < len(STANDARD_ATOMIC_WEIGHTS) else 0
    if weight == 0:
        raise ValueError(f"atomic number must be 1..118, got {element!r}")
    if weight is None:
        raise ValueError(
            f"{symbol(z)} has no standard atomic weight (no stable isotope); "
            "pass the isotope mass instead"
        )
    return weight
