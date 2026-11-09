"""Born amplitudes of neutral atoms for fast electrons, and phase scales.

Literature: the scattering amplitude follows Lobato and Van Dyck (2014), Acta
Cryst. A70, 636-649, eq. 13 with the coefficients of their Table 1, and
eq. 2 for the relation to the Fourier transform of the atomic potential. The
interaction parameter follows Kirkland (2010), *Advanced Computing in
Electron Microscopy*, eq. 5.6 and ch. 5. The summation error analysis
follows Higham (2002), ch. 3-4. Constants are CODATA 2018 (Tiesinga et al.
2021) from :mod:`scatterem.physics.constants`. Written for scatterem from
this literature.

Units: spatial frequency ``g = 2 sin(theta) / lambda`` [1/Angstrom],
scattering amplitude ``f_e`` [Angstrom], potential transform
[Volt Angstrom^3], energy [eV].
"""

from __future__ import annotations

import logging
import math
import operator
from collections.abc import Sequence

import numpy as np
import torch

from ._lobato2014_table import COEFFICIENTS
from .constants import (
    CODATA18_ELECTRON_MASS,
    EXACT_E_CHARGE,
    EXACT_PLANCK,
    debroglie_at,
    gamma_at,
)
from .periodic_sampling import PeriodicGrid, dft_sample_frequencies

__all__ = [
    "POTENTIAL_PER_AMPLITUDE",
    "born_amplitude",
    "born_amplitude_array",
    "born_amplitudes",
    "gamma_wavelength",
    "potential_spectra",
    "validated_atomic_number",
]

_log = logging.getLogger(__name__)

#: ``h^2 / (2 pi m0 e)`` [Volt Angstrom^2]: the Fourier transform of the atomic
#: potential [Volt Angstrom^3] is this factor times ``f_e`` [Angstrom]
#: (Lobato and Van Dyck 2014, eq. 2). Rest mass ``m0``; about 47.8776.
POTENTIAL_PER_AMPLITUDE = (
    EXACT_PLANCK**2 / (2.0 * math.pi * CODATA18_ELECTRON_MASS * EXACT_E_CHARGE) * 1e20
)


def validated_atomic_number(atomic_number) -> int:
    """Return the atomic number as ``int`` after a check against Table 1.

    Parameters
    ----------
    atomic_number : int
        Atomic number ``Z``. Python and numpy integers are accepted.

    Returns
    -------
    int
        ``Z``, in ``1..103``.

    Raises
    ------
    ValueError
        If ``Z`` is not an integer (``bool`` and ``float`` are rejected), or
        if it is outside ``1..103``. The message names the number and the
        range.
    """
    top = len(COEFFICIENTS)
    if isinstance(atomic_number, bool):
        z = None
    else:
        try:
            z = operator.index(atomic_number)
        except TypeError:
            z = None
    if z is None:
        raise ValueError(
            f"atomic number {atomic_number!r} is not an integer; "
            f"the table covers 1..{top}"
        )
    if not 1 <= z <= top:
        raise ValueError(
            f"atomic number {z} is outside the table range 1..{top} "
            "(Lobato and Van Dyck 2014)"
        )
    return z


def _amplitude_of_square(x, z: int):
    """Return eq. 13 of Lobato and Van Dyck (2014) at ``x = g^2``.

    ``x`` is a float64 numpy array or tensor [1/Angstrom^2]. Each term uses
    ``a_i t_i (1 + t_i)`` with ``t_i = 1 / (1 + b_i x)``, which is equal to
    ``a_i (2 + b_i x) / (1 + b_i x)^2``. The sum has a fixed order.
    """
    a, b = COEFFICIENTS[z - 1]
    terms = []
    for a_i, b_i in zip(a, b, strict=True):
        t = 1.0 / (1.0 + b_i * x)
        terms.append(a_i * (t * (1.0 + t)))
    return ((terms[0] + terms[1]) + (terms[2] + terms[3])) + terms[4]


def born_amplitude(frequency: torch.Tensor, atomic_number: int) -> torch.Tensor:
    """Return the electron scattering amplitude ``f_e(g)`` of a neutral atom.

    The formula is eq. 13 of Lobato and Van Dyck (2014)::

        f_e(g) = sum_i a_i (2 + b_i g^2) / (1 + b_i g^2)^2

    The code evaluates it in float64 and rounds the result once to the input
    dtype. The error is at most about 1e-13 of ``f_e(0)`` before that
    rounding, also for rows with strong cancellation (Z = 2, 6). Autograd
    gives ``d f_e / d g`` through the same float64 graph.

    Parameters
    ----------
    frequency : torch.Tensor
        Spatial frequency ``g`` [1/Angstrom], real floating tensor of any
        shape. ``f_e`` is even in ``g``, so a negative value gives
        ``f_e(|g|)``.
    atomic_number : int
        Atomic number ``Z`` in ``1..103``.

    Returns
    -------
    torch.Tensor
        ``f_e`` [Angstrom], with the shape, dtype and device of
        ``frequency``.

    Raises
    ------
    TypeError
        If ``frequency`` is not a real floating tensor.
    ValueError
        If the atomic number is not valid.
    """
    z = validated_atomic_number(atomic_number)
    if not isinstance(frequency, torch.Tensor) or not frequency.is_floating_point():
        raise TypeError("frequency must be a real floating torch.Tensor")
    work = frequency
    if frequency.device.type == "mps":  # no float64 on this device
        work = frequency.cpu()
    g = work.to(torch.float64)
    f = _amplitude_of_square(g * g, z)
    return f.to(device=frequency.device, dtype=frequency.dtype)


def born_amplitude_array(frequency, atomic_number: int) -> np.ndarray:
    """Return ``f_e(g)`` [Angstrom] as a numpy float64 array.

    The formula and the evaluation are those of :func:`born_amplitude`.

    Parameters
    ----------
    frequency : array_like
        Spatial frequency ``g`` [1/Angstrom].
    atomic_number : int
        Atomic number ``Z`` in ``1..103``.

    Returns
    -------
    numpy.ndarray
        ``f_e`` [Angstrom], float64, shape ``np.shape(frequency)``.

    Raises
    ------
    ValueError
        If the atomic number is not valid.
    """
    z = validated_atomic_number(atomic_number)
    g = np.asarray(frequency, dtype=np.float64)
    return np.asarray(_amplitude_of_square(g * g, z), dtype=np.float64)


def born_amplitudes(frequency, atomic_numbers: Sequence[int]):
    """Return ``f_e(g)`` [Angstrom] for a list of atomic numbers.

    Parameters
    ----------
    frequency : torch.Tensor or array_like
        Spatial frequency ``g`` [1/Angstrom]. A tensor selects the torch form
        (:func:`born_amplitude`); other input selects the numpy form
        (:func:`born_amplitude_array`).
    atomic_numbers : sequence of int
        ``S`` atomic numbers, each in ``1..103``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``(S, *shape)``. Slice ``k`` is equal bit for bit to the one-element
        form for the ``k``-th number.

    Raises
    ------
    ValueError
        If the list is empty or holds a number that is not valid.
    """
    numbers = [validated_atomic_number(z) for z in atomic_numbers]
    if not numbers:
        raise ValueError("atomic_numbers must hold at least one number")
    if isinstance(frequency, torch.Tensor):
        return torch.stack([born_amplitude(frequency, z) for z in numbers])
    return np.stack([born_amplitude_array(frequency, z) for z in numbers])


def potential_spectra(grid: PeriodicGrid, atomic_numbers: Sequence[int]) -> np.ndarray:
    """Return the Fourier transforms of atomic potentials on a 2-D grid.

    For each atom, the value at frequency ``g`` is ``V(g) =
    POTENTIAL_PER_AMPLITUDE * f_e(|g|)`` [Volt Angstrom^3] (Lobato and Van
    Dyck 2014, eq. 2), with ``|g|^2 = k_y^2 + k_x^2`` from
    :func:`~scatterem.physics.periodic_sampling.dft_sample_frequencies`. On a 2-D grid
    this is the 3-D transform at ``k_z = 0``, which equals the 2-D transform
    of the projected potential (projection-slice theorem). It is the
    continuous transform, not a DFT of samples: to get the projected
    potential [Volt Angstrom] on a cell of area ``A`` [Angstrom^2], divide by
    ``A`` and apply the inverse FFT (``torch.fft.ifft2`` includes the factor
    ``1 / (Ny Nx)``, so multiply its result by ``Ny Nx``).

    Parameters
    ----------
    grid : PeriodicGrid
        A 2-D grid ``(y, x)``.
    atomic_numbers : sequence of int
        ``S`` atomic numbers, each in ``1..103``.

    Returns
    -------
    numpy.ndarray
        Float64 array ``(S, Ny, Nx)`` in FFT order [Volt Angstrom^3]. The
        code computes in float64. numpy has no device, so the caller moves
        the result once with ``torch.from_numpy(out).to(device, dtype)``.

    Raises
    ------
    ValueError
        If the grid is not 2-D, the list is empty or a number is not valid.
    """
    if grid.ndim != 2:
        raise ValueError(f"potential_spectra needs a 2-D grid; got {grid.ndim} axes")
    numbers = [validated_atomic_number(z) for z in atomic_numbers]
    if not numbers:
        raise ValueError("atomic_numbers must hold at least one number")
    ky, kx = dft_sample_frequencies(grid)
    x = ky**2 + kx**2
    return np.stack(
        [_amplitude_of_square(x, z) * POTENTIAL_PER_AMPLITUDE for z in numbers]
    )


def gamma_wavelength(kinetic_ev: float) -> float:
    """Return the product of the Lorentz factor and the wavelength.

    The phase of a slice is ``sigma * v`` with the projected potential ``v``
    [Volt Angstrom]. When ``v`` comes from ``f_e`` [Angstrom] through
    :data:`POTENTIAL_PER_AMPLITUDE`, the product of the two factors is::

        sigma * h^2 / (2 pi m0 e) = (2 pi gamma m0 e lambda / h^2)
                                    * (h^2 / (2 pi m0 e)) = gamma lambda

    So ``gamma lambda`` times a sum of ``f_e`` terms, divided by the cell area,
    is the phase [rad].

    Parameters
    ----------
    kinetic_ev : float
        Kinetic energy [eV]. Must be positive.

    Returns
    -------
    float
        ``gamma * lambda`` [Angstrom].

    Raises
    ------
    ValueError
        If ``kinetic_ev`` is not positive.
    """
    wavelength = debroglie_at(kinetic_ev)  # raises for kinetic_ev <= 0
    return gamma_at(kinetic_ev) * wavelength
