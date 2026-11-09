"""Absolute scale of the core-loss EELS outputs: per incident electron.

Two checks.

1. A first-Born oracle. One carbon atom (hydrogenic 1s, free continuum
   ``l' = 1``) in a periodic cell of area ``A`` is lit by a plane wave of unit
   total intensity, ``psi = 1/sqrt(A)``. Without elastic scattering, the
   probability per eV that the electron ionizes the atom and arrives inside a
   detector of semi-angle ``beta`` is

       P = (1/A^2) sum_q (2 pi)^2 d^2 sigma / (dE d^2 q),

   summed over the periodic beams ``q`` inside the detector, with the
   Bethe/Inokuti cross-section (Egerton, *EELS in the Electron Microscope*,
   3rd ed., Eq. 3.27, ``dOmega = d^2 q / k_n^2``, two spin states)

       d^2 sigma / (dE d^2 q) = 2 * 4 gamma^2 |eps(Q)|^2 / (a_0^2 Q^4 k_0 k_n).

   For ``1s -> eps p``, ``sum_m' |eps|^2 = 3 I(Q)^2`` with
   ``I = int u_c u_b j_1(Q r) dr``, the continuum normalised per Rydberg and
   converted to per eV. The oracle uses only numpy and scipy and these
   constants; it does not call the library's normalisation code. The map,
   the cube summed over the detector and the multislice pattern summed over
   the detector must equal ``P`` at every grid size.
2. Grid independence. The ``PrismEelsSimulator`` map of a fixed SrTiO3 cell
   must not depend on the number of pixels, nor on the PRISM interpolation
   factor, nor on the partitioning, beyond the approximation error. Before
   the fix the map grew as ``Ny Nx / (fy fx)``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from scipy.special import spherical_jn

from scatterem.simulation.eels.forward.multislice_eels import (
    multislice_transition_potential,
)
from scatterem.simulation.eels.forward.prism_eels import prism_transition_potential
from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.radial import (
    free_continuum_wavefunction,
    hydrogenic_bound_wavefunction,
)
from scatterem.simulation.eels.physics.transition_potentials import (
    Transition,
    build_transition_potentials,
)

#: CODATA 2018: Hartree [eV], Bohr radius [Angstrom], m_e c^2 [eV], h c [eV A].
_HARTREE = 27.211386245988
_BOHR = 0.529177210903
_RYDBERG = _HARTREE / 2.0
_ME_C2 = 510998.95
_HC = 12398.419843320026

_Z, _EPS, _E0, _L, _BETA = 6, 10.0, 100e3, 8.0, 40.0  # C K; eV; eV; A; mrad

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _wavelength(energy: float) -> float:
    """Relativistic electron wavelength [Angstrom]."""
    return _HC / math.sqrt(energy * (2.0 * _ME_C2 + energy))


def _first_born_probability(n: int) -> float:
    """``P`` of the module docstring on an ``n x n`` grid (numpy only)."""
    lam = _wavelength(_E0)
    binding = 0.5 * _Z**2 * _HARTREE
    k0 = 2.0 * math.pi / lam
    kn = 2.0 * math.pi / _wavelength(_E0 - binding - _EPS)
    gamma = 1.0 + _E0 / _ME_C2
    f = np.fft.fftfreq(n, d=_L / n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    inside = fy**2 + fx**2 <= (_BETA * 1e-3 / lam) ** 2
    Q = np.sqrt(
        (2.0 * math.pi) ** 2 * (fy[inside] ** 2 + fx[inside] ** 2) + (k0 - kn) ** 2
    )
    r = np.linspace(1e-6, 20.0, 100_001)  # Bohr; the 1s state is e^-120 at the end
    dr = r[1] - r[0]
    k = math.sqrt(_EPS / _RYDBERG)
    u_c = k * r * spherical_jn(1, k * r) / math.sqrt(math.pi * k)  # per Rydberg
    u_b = 2.0 * _Z**1.5 * r * np.exp(-_Z * r)
    radial = np.array(
        [np.sum(u_c * u_b * spherical_jn(1, q * _BOHR * r)) * dr for q in Q]
    )
    eps2 = 3.0 * radial**2 / _RYDBERG  # sum over m', per eV
    d2s = 2.0 * 4.0 * gamma**2 * eps2 / (_BOHR**2 * Q**4 * k0 * kn)
    area = _L * _L
    return float(np.sum(d2s) * (2.0 * math.pi) ** 2 / area**2)


@pytest.mark.parametrize("n", [32, 48, 64])
def test_kernels_match_the_first_born_probability(n):
    """Map, cube and multislice equal the closed form at every grid size."""
    tp = build_transition_potentials(
        [
            Transition(
                hydrogenic_bound_wavefunction(Z=_Z, n=1, l=0),
                free_continuum_wavefunction(l=1, epsilon=_EPS),
                0,
                m,
            )
            for m in (-1, 0, 1)
        ],
        _Z,
        (n, n),
        (_L / n, _L / n),
        eV=_E0,
        device=_DEVICE,
        dtype=torch.complex128,
    )
    # A plane wave of unit total intensity: ifft2 of sqrt(n^2) at q = 0.
    probe_q = torch.zeros((n, n), dtype=torch.complex128, device=_DEVICE)
    probe_q[0, 0] = float(n)
    T = torch.ones((1, n, n), dtype=torch.complex128, device=_DEVICE)
    site = np.array([[0.5, 0.5, 0.5]])
    scan = torch.tensor([[n / 2, n / 2]], dtype=torch.float64, device=_DEVICE)
    geo = dict(wavelength=_wavelength(_E0), gridsize=(_L, _L), slice_distance=0.0)
    f = torch.fft.fftfreq(n, d=_L / n, device=_DEVICE, dtype=torch.float64)
    det = f[:, None] ** 2 + f[None, :] ** 2 <= (_BETA * 1e-3 / geo["wavelength"]) ** 2

    expected = _first_born_probability(n)
    image = float(
        prism_eels_image(probe_q, T, tp, site, scan, detector_mrad=_BETA, **geo)[0]
    )
    cube = float(
        prism_transition_potential(probe_q, T, tp, site, scan, **geo)[0][det].sum()
    )
    psi = torch.fft.ifft2(probe_q)[None]
    msl = float(multislice_transition_potential(psi, T, tp, site, **geo)[0][det].sum())
    for name, got in (("map", image), ("cube", cube), ("multislice", msl)):
        assert got == pytest.approx(expected, rel=1e-4), (
            f"{name} at {n}^2 px: {got:.6e}, first Born {expected:.6e}"
        )


def _sto_map(pixels: int, f: int, partition: bool) -> float:
    from scatterem.physics.atom_set import AtomSet
    from scatterem.simulation.eels import PrismEelsSimulator

    a = 3.905
    sto = AtomSet(
        np.array([a, a, a]),
        np.array(
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            dtype=float,
        ),
        np.array([38, 22, 8, 8, 8]),
        msd=None,
    ).replicated((2, 2, 1))
    sim = PrismEelsSimulator(
        sto,
        eV=100e3,
        semiconvergence_angle=25.0,
        pixels=(pixels, pixels),
        scan=(4, 4),
        num_slices=2,
        interpolation_factor=f,
        device=_DEVICE,
    )
    opts = {}
    if partition:
        opts = dict(partition=dict(n_radial=4), partition_s2=dict(n_radial=4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sim.eels_map(
            (22, 2, 1),
            epsilon=10.0,
            detector_mrad=25.0,
            transitions=dict(wavefunctions="hydrogenic"),
            **opts,
        )
    return float(m.data.double().mean())


@pytest.mark.parametrize(
    ("f", "partition"), [(1, False), (2, False), (2, True)], ids=["f1", "f2", "f2-bip"]
)
def test_map_does_not_depend_on_the_grid(f, partition):
    """Fixed cell, 48 / 64 / 96 px: the map mean agrees to 0.5 % (it was x4)."""
    means = [_sto_map(px, f, partition) for px in (48, 64, 96)]
    spread = (max(means) - min(means)) / max(means)
    assert spread < 5e-3, f"map means {means} vary by {spread:.2%}"
    # A probability per eV of a few unit cells of Ti: far below 1.
    assert 1e-10 < means[-1] < 1e-6


def test_map_does_not_depend_on_the_interpolation_factor():
    """f = 2 against f = 1 at the same grid: only the PRISM error remains.

    The 3.9 Angstrom PRISM period is small for this probe, so the error is a
    few per cent. Before the fix, f = 2 read 1/4 of f = 1.
    """
    exact = _sto_map(64, 1, False)
    for f, partition in ((2, False), (2, True)):
        got = _sto_map(64, f, partition)
        assert got == pytest.approx(exact, rel=0.06), (f, partition, got, exact)
