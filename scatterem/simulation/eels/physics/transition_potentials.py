"""Inelastic transition potentials for core-loss EELS.

Written by the scatterem authors from the first Born approximation and the
multislice EELS formalism of Brown, Ciston and Ophus, Phys. Rev. Research 1,
033186 (2019) (``psi_n = H_n0 psi_0``), with the Rayleigh plane-wave expansion
and the Gaunt integral of three spherical harmonics.

The conventions are as follows.

* Form factor ``epsilon_n0(K) = <phi_n| exp(i K.r) |phi_0>`` for the bound
  orbital ``R_nl Y_lm`` and the continuum partial wave ``R_(eps l') Y_(l'm')``:
  ``epsilon = 4 pi sum_lambda i^lambda I_lambda(K) Y*_(lambda mu)(K_hat)
  G(l'm'; lambda mu; l m)`` with ``mu = m' - m``, the Gaunt integral ``G`` and
  the radial integral ``I_lambda = int u_c j_lambda(K r) u_b dr``. For
  ``l' = l`` the continuum is first orthogonalised to the bound orbital
  (Pauli), because the model states of the two families are not exact
  eigenstates of one Hamiltonian. For ``lambda = 0`` the integrand uses
  ``j_0(K r) - 1``, which is the same for orthogonal states and avoids
  cancellation at small ``K``.
* Frame: x along array axis -2, y along array axis -1, z along the beam. The
  FFT coefficient at frequency ``f`` (cycles/Angstrom) has the momentum
  transfer ``K = (-2 pi f_a, -2 pi f_b, q_z)`` with ``q_z = k_0 - k_n`` from the
  relativistic wavelengths. The kernel is ``1/K^2`` without retardation.
* Phase: ``H_n0`` is ``sigma`` times the projected Coulomb matrix element,
  without the constant factor ``-i`` of the first-order solution (each
  channel is a different final state, so its global phase is not observable).
* Units: the continuum is normalised per Rydberg; ``epsilon`` is converted to
  per eV. ``fft2(array) = H~(Q) / (dy dx)`` with
  ``H~ = sigma(E0) sqrt(2 k_0/k_n) 4 pi (E_h a_0) epsilon_eV / K^2`` in
  Angstrom^2 eV^(-1/2), so the real-space ``H`` is in eV^(-1/2). For a unit
  plane wave, ``int |H psi|^2 d^2r`` summed over the channels of one
  ``(n, l, epsilon)`` is the first-Born ``d sigma / dE`` in Angstrom^2/eV of
  the closed subshell: the factor 2 counts the two spin states of each
  orbital, and ``k_0/k_n`` converts the density of the outgoing wave to the
  detected flux. Divide ``|H|^2`` by 2 for one electron.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.constants import debroglie_at, phase_per_potential_at
from scatterem.physics.periodic_sampling import (
    PeriodicGrid,
    dft_sample_frequencies,
    ellipse_mask,
)

from ._angular import spherical_harmonic, wigner_3j
from ._radial_tables import (
    _J0M1_SERIES,
    _hermite_np,
    _hermite_torch,
    _j0m1_series,
    _k_table,
    _Nodes,
    _overlaps_np,
    _radial_nodes,
    _slopes_np,
)
from ._special_torch import assoc_legendre_torch as _assoc_legendre_torch
from ._special_torch import spherical_bessel_torch as _spherical_bessel_torch
from ._special_torch import ylm_norm as _ylm_norm
from .radial import (
    _BOHR_ANG,
    _HARTREE_EV,
    _RYDBERG_EV,
    AtomicRadialWavefunction,
    bound_wavefunction,
    continuum_wavefunction,
    resolve_wavefunctions,
)

__all__ = [
    "Transition",
    "TransitionPotentials",
    "build_transition_potentials",
    "subshell_transitions",
]

#: e^2 / (4 pi eps_0) = E_h a_0, in eV Angstrom.
_E2_EV_ANG = _HARTREE_EV * _BOHR_ANG


@dataclass
class Transition:
    """One ionisation channel: bound orbital ``(l, ml)`` to continuum ``(l', ml')``.

    Parameters
    ----------
    bound : AtomicRadialWavefunction
        Bound-state radial wavefunction.
    excited : AtomicRadialWavefunction
        Continuum-state radial wavefunction.
    ml : int
        Magnetic quantum number of the bound orbital.
    mlprime : int
        Magnetic quantum number of the continuum orbital.
    """

    bound: AtomicRadialWavefunction
    excited: AtomicRadialWavefunction
    ml: int
    mlprime: int

    @property
    def energy_loss(self) -> float:
        """Energy loss of the channel.

        Returns
        -------
        float
            ``excited.energy - bound.energy`` in eV. It exceeds the continuum
            energy by the binding energy.
        """
        return float(self.excited.energy - self.bound.energy)


def subshell_transitions(
    Z: int,
    n: int,
    l: int,
    epsilon: float,
    lprimes: Sequence[int] | None = None,
    order: int = 1,
    wavefunctions: str = "auto",
    xc: str = "PBE",
    potential_scale: float = 1.0,
) -> list[Transition]:
    """All channels of one ionised subshell at one continuum energy.

    Parameters
    ----------
    Z, n, l : int
        Element and the ionized bound orbital (e.g. Fe ``L`` edge: Z=26, n=2,
        l=1).
    epsilon : float
        Energy of the ejected electron above the ionization threshold [eV].
        Selecting different ``epsilon`` selects different points along the
        energy-loss edge.
    lprimes : list of int, optional
        Continuum angular momenta to include. Default: every ``l' >= 0`` with
        ``|l - l'| <= order``.
    order : int
        Maximum change in angular momentum ``|l - l'|`` retained when
        ``lprimes`` is not given explicitly. The leading small-K term of a
        channel has the multipole order ``|l - l'|``, so ``order = 1`` keeps
        the dipole channels and the ``l' = l`` channel.
    wavefunctions : str
        ``"auto"`` (default: GPAW if importable, else hydrogenic with a
        ``HydrogenicFallbackWarning``), ``"gpaw"`` or ``"hydrogenic"``.
        ``"auto"`` resolves once per call, so the fallback warns once.
    xc : str
        Exchange-correlation functional for the GPAW wavefunctions.
    potential_scale : float
        Multiplier on the atomic potential of the GPAW continuum states.

    Returns
    -------
    list of Transition
        One entry per ``(l', ml, mlprime)`` combination, ordered by ``l'``,
        then ``ml``, then ``mlprime``.
    """
    family = resolve_wavefunctions(wavefunctions)
    if lprimes is None:
        lprimes = list(range(max(0, l - order), l + order + 1))
    bound = bound_wavefunction(Z, n, l, xc=xc, wavefunctions=family)
    transitions = []
    for lp in lprimes:
        excited = continuum_wavefunction(
            Z,
            lp,
            epsilon,
            xc=xc,
            wavefunctions=family,
            potential_scale=potential_scale,
        )
        for ml in range(-l, l + 1):
            for mlp in range(-lp, lp + 1):
                transitions.append(Transition(bound, excited, ml, mlp))
    return transitions


# --------------------------------------------------------------------------- #
# Angular algebra
# --------------------------------------------------------------------------- #
def _multipoles(l: int, lp: int, ml: int, mlp: int) -> list[int]:
    """The ``lambda`` that the integer selection rules allow (exact, finite)."""
    mu = mlp - ml
    return [
        lam
        for lam in range(abs(l - lp), l + lp + 1)
        if (l + lp + lam) % 2 == 0 and abs(mu) <= lam
    ]


def _gaunt(lp: int, mlp: int, lam: int, mu: int, l: int, ml: int) -> float:
    """Gaunt integral ``int Y*_(l'm') Y_(lambda mu) Y_(lm) dOmega``."""
    return (
        (-1.0) ** mlp
        * sqrt((2 * lp + 1) * (2 * lam + 1) * (2 * l + 1) / (4.0 * pi))
        * wigner_3j(lp, lam, l, 0, 0, 0)
        * wigner_3j(lp, lam, l, -mlp, mu, ml)
    )


# --------------------------------------------------------------------------- #
# Radial integrals
# --------------------------------------------------------------------------- #
def _radial_overlap(
    lprimeprime: int,
    bound: AtomicRadialWavefunction,
    excited: AtomicRadialWavefunction,
    q1d: np.ndarray,
) -> np.ndarray:
    """Radial integral ``I_lambda(q)``.

    For ``l' = l`` the continuum is orthogonalised to the bound orbital first.

    Parameters
    ----------
    lprimeprime : int
        Multipole order ``lambda``.
    bound, excited : AtomicRadialWavefunction
        The two radial states.
    q1d : ndarray
        Momentum transfers in 1/Angstrom.

    Returns
    -------
    ndarray
        ``int u_c j_lambda(q a_0 r) u_b dr`` (``j_0 - 1`` for ``lambda = 0``),
        float64, with the shape of ``q1d``.
    """
    q = np.asarray(q1d, dtype=np.float64)
    kb = q.ravel() * _BOHR_ANG
    nodes = _radial_nodes(bound, excited, float(kb.max(initial=0.0)))
    return _overlaps_np([int(lprimeprime)], nodes, kb)[0].reshape(q.shape)


def _form_factor(
    transition: Transition, qabs: np.ndarray, qphi: np.ndarray, qtheta: np.ndarray
) -> np.ndarray:
    """Form factor ``epsilon_n0(K)`` with the numpy and scipy functions.

    ``I_lambda`` is integrated directly at every ``|K|`` (no table), so this is
    the most accurate reference. It is ``4 pi`` times the Gaunt integral form:
    ``4 pi sum_lambda i^lambda I_lambda(K) Y*_(lambda mu)(K_hat) G``.

    Parameters
    ----------
    transition : Transition
        The channel.
    qabs : ndarray
        ``|K|`` in 1/Angstrom.
    qphi : ndarray
        Azimuth of ``K`` (from x on array axis -2 towards y on axis -1).
    qtheta : ndarray
        Polar angle of ``K`` from the beam direction.

    Returns
    -------
    ndarray
        Complex ``epsilon_n0`` in Ry^(-1/2), with the shape of ``qabs``.
    """
    qabs = np.asarray(qabs, dtype=np.float64)
    kb = qabs * _BOHR_ANG
    tr = transition
    lams = _multipoles(tr.bound.l, tr.excited.l, tr.ml, tr.mlprime)
    if not lams:
        return np.zeros(qabs.shape, dtype=np.complex128)
    nodes = _radial_nodes(tr.bound, tr.excited, float(kb.max()))
    radial = _overlaps_np(lams, nodes, kb).reshape(len(lams), *qabs.shape)
    return _form_factor_np(tr, lams, radial, qphi, qtheta)


def _form_factor_np(tr, lams, radial, phi, theta) -> np.ndarray:
    """``epsilon_n0`` from ``I_lambda`` already on the points (numpy reference)."""
    l, lp, ml, mlp = tr.bound.l, tr.excited.l, tr.ml, tr.mlprime
    mu = mlp - ml
    eps = np.complex128(0.0)
    for i, lam in enumerate(lams):
        coef = 4.0 * pi * (1j**lam) * _gaunt(lp, mlp, lam, mu, l, ml)
        eps = eps + coef * radial[i] * np.conj(spherical_harmonic(mu, lam, phi, theta))
    return np.asarray(eps)


def _overlaps_torch(
    lpps: Sequence[int], nodes: _Nodes, kb: Tensor, ladder: dict | None
) -> Tensor:
    """``I_lambda(K)`` for ``K = kb`` (1/Bohr) with torch."""
    r = torch.as_tensor(nodes.r, dtype=torch.float64, device=kb.device)
    p = torch.as_tensor(nodes.p, dtype=torch.float64, device=kb.device)
    x = kb.reshape(-1, 1) * r.reshape(1, -1)
    if ladder is None:
        jall = _spherical_bessel_torch(max(lpps), x)
    else:
        if "jall" not in ladder:
            ladder["jall"] = _spherical_bessel_torch(
                int(ladder.get("lmax", max(lpps))), x
            )
        jall = ladder["jall"]
    rows = []
    for lam in lpps:
        j = jall[lam]
        if lam == 0:
            j = torch.where(x < _J0M1_SERIES, _j0m1_series(x * x), j - 1.0)
        rows.append((j * p).sum(dim=-1))
    return torch.stack(rows).reshape(len(lpps), *kb.shape)


def _table_torch(lams: Sequence[int], nodes: _Nodes, table: Tensor) -> tuple:
    """Values and slopes ``(I_lambda, dI_lambda/dK)`` on the K table, with torch.

    The slope uses ``(2 lambda + 1) j_lambda' = lambda j_(lambda-1) -
    (lambda + 1) j_(lambda+1)``, which has no ``1/x`` term.
    """
    r = torch.as_tensor(nodes.r, dtype=torch.float64, device=table.device)
    p = torch.as_tensor(nodes.p, dtype=torch.float64, device=table.device)
    x = table.reshape(-1, 1) * r.reshape(1, -1)
    jall = _spherical_bessel_torch(max(lams) + 1, x)
    pr = p * r
    vals, slopes = [], []
    for lam in lams:
        j = jall[lam]
        if lam == 0:
            j = torch.where(x < _J0M1_SERIES, _j0m1_series(x * x), j - 1.0)
        vals.append((j * p).sum(dim=-1))
        dj = -(lam + 1) * jall[lam + 1]
        if lam > 0:
            dj = dj + lam * jall[lam - 1]
        slopes.append((dj * pr).sum(dim=-1) / (2 * lam + 1))
    return torch.stack(vals), torch.stack(slopes)


def _radial_overlaps_torch(
    lpps: Sequence[int],
    bound: AtomicRadialWavefunction,
    excited: AtomicRadialWavefunction,
    q1d: Tensor,
    ladder: dict | None = None,
) -> Tensor:
    """Stacked radial integrals ``I_lambda(q)``, the torch twin of ``_radial_overlap``.

    Parameters
    ----------
    lpps : sequence of int
        Multipole orders ``lambda``, any order and gaps.
    bound, excited : AtomicRadialWavefunction
        The two radial states.
    q1d : Tensor
        Momentum transfers in 1/Angstrom.
    ladder : dict, optional
        Memo of the Bessel table. ``ladder["lmax"]`` sets the table depth; the
        table is stored as ``ladder["jall"]`` and reused by later calls with
        the same ``q1d`` and states.

    Returns
    -------
    Tensor
        Shape ``(len(lpps), *q1d.shape)``, float64.
    """
    kb = q1d.to(torch.float64) * _BOHR_ANG
    nodes = _radial_nodes(bound, excited, float(kb.max()) if kb.numel() else 0.0)
    return _overlaps_torch(list(lpps), nodes, kb, ladder)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _wavenumbers(eV: float, loss: float) -> tuple:
    """``(k_0, k_n)`` in 1/Angstrom from the relativistic wavelengths."""
    return 2.0 * pi / debroglie_at(eV), 2.0 * pi / debroglie_at(eV - loss)


def _scale(eV: float, loss: float) -> float:
    """Factor ``c`` of ``H~ = c epsilon_Ry / K^2``.

    ``c = sigma(E0) sqrt(2 k_0/k_n) 4 pi E_h a_0 / sqrt(Ry)``. The factor 2 is
    the spin degeneracy of the closed orbital, and ``k_0/k_n`` makes the
    detected flux equal the first-Born cross-section.
    """
    k0, kn = _wavenumbers(eV, loss)
    return (
        phase_per_potential_at(eV)
        * sqrt(2.0 * k0 / kn)
        * 4.0
        * pi
        * _E2_EV_ANG
        / sqrt(_RYDBERG_EV)
    )


def _q_z(eV: float, loss: float) -> float:
    """``k_0 - k_n`` in 1/Angstrom from the relativistic wavelengths."""
    k0, kn = _wavenumbers(eV, loss)
    return k0 - kn


def _pairs(transitions: Sequence[Transition]) -> dict:
    """Group transition indices by their (bound, excited) pair."""
    groups: dict = {}
    for i, tr in enumerate(transitions):
        groups.setdefault((id(tr.bound), id(tr.excited)), []).append(i)
    return groups


def _pair_lambdas(transitions, idx) -> list[int]:
    lams = set()
    for i in idx:
        tr = transitions[i]
        lams.update(_multipoles(tr.bound.l, tr.excited.l, tr.ml, tr.mlprime))
    return sorted(lams)


def _build_cpu(transitions, fa, fb, eV, sampling) -> np.ndarray:
    """``H~ / (dy dx)`` with numpy and scipy (the reference backend)."""
    qa, qb = 2.0 * pi * fa, 2.0 * pi * fb
    qperp = np.hypot(qa, qb)
    has_perp = qperp > 0.0
    kx = np.where(has_perp, -qa, 1.0)
    ky = np.where(has_perp, -qb, 0.0)
    phi = np.arctan2(ky, kx)
    out = np.zeros((len(transitions), *fa.shape), dtype=np.complex128)
    for idx in _pairs(transitions).values():
        first = transitions[idx[0]]
        qz = _q_z(eV, first.energy_loss)
        scale = _scale(eV, first.energy_loss) / (sampling[0] * sampling[1])
        kabs = np.sqrt(qperp**2 + qz**2)
        theta = np.arctan2(qperp, qz)
        kb = kabs * _BOHR_ANG
        lams = _pair_lambdas(transitions, idx)
        nodes = _radial_nodes(first.bound, first.excited, float(kb.max()))
        table = _k_table(float(kb.max()), nodes.r_cut)
        dk = float(table[1])
        radial = _overlaps_np(lams, nodes, table) if lams else np.zeros(0)
        slope = _slopes_np(lams, nodes, table) if lams else np.zeros(0)
        for i in idx:
            tr = transitions[i]
            mine = _multipoles(tr.bound.l, tr.excited.l, tr.ml, tr.mlprime)
            on_grid = [
                _hermite_np(kb, dk, radial[lams.index(lam)], slope[lams.index(lam)])
                for lam in mine
            ]
            eps = _form_factor_np(tr, mine, on_grid, phi, theta)
            out[i] = scale * eps / kabs**2
    return out


def _build_transition_potentials_gpu(
    transitions: Sequence[Transition],
    fa: Tensor,
    fb: Tensor,
    eV: float,
    sampling: tuple,
    device,
) -> tuple:
    """Vectorised torch build of ``H~ / (dy dx)`` before the band limit.

    Parameters
    ----------
    transitions : sequence of Transition
        The channels.
    fa, fb : Tensor
        Meshed FFT frequencies (cycles/Angstrom) along array axes -2 and -1.
    eV : float
        Probe energy [eV].
    sampling : tuple
        ``(dy, dx)`` in Angstrom.
    device : torch.device
        Device of the build.

    Returns
    -------
    tuple
        ``(recip, energy_losses)``: complex128 ``(n, Ny, Nx)`` on ``device``
        and the float64 energy losses in eV.
    """
    fa = fa.to(device=device, dtype=torch.float64)
    fb = fb.to(device=device, dtype=torch.float64)
    qa, qb = 2.0 * pi * fa, 2.0 * pi * fb
    qperp = torch.hypot(qa, qb)
    has_perp = qperp > 0.0
    safe = torch.where(has_perp, qperp, torch.ones_like(qperp))
    eiphi = torch.where(
        has_perp,
        torch.complex(-qa / safe, -qb / safe),
        torch.ones_like(qperp, dtype=torch.complex128),
    )
    out = torch.zeros(
        (len(transitions), *fa.shape), dtype=torch.complex128, device=device
    )
    powers = {0: torch.ones_like(eiphi)}
    for idx in _pairs(transitions).values():
        first = transitions[idx[0]]
        qz = _q_z(eV, first.energy_loss)
        scale = _scale(eV, first.energy_loss) / (sampling[0] * sampling[1])
        kabs = torch.sqrt(qperp**2 + qz**2)
        ct, st = qz / kabs, qperp / kabs
        kb = kabs * _BOHR_ANG
        kmax = float(kb.max())
        lams = _pair_lambdas(transitions, idx)
        if not lams:
            continue
        nodes = _radial_nodes(first.bound, first.excited, kmax)
        table_np = _k_table(kmax, nodes.r_cut)
        dk = float(table_np[1])
        table = torch.as_tensor(table_np, dtype=torch.float64, device=device)
        radial, slope = _table_torch(lams, nodes, table)
        on_grid = {
            lam: _hermite_torch(kb, dk, radial[i], slope[i])
            for i, lam in enumerate(lams)
        }
        conj_y: dict = {}
        for i in idx:
            tr = transitions[i]
            l, lp, ml, mlp = tr.bound.l, tr.excited.l, tr.ml, tr.mlprime
            mu = mlp - ml
            am = abs(mu)
            eps = torch.zeros_like(eiphi)
            for lam in _multipoles(l, lp, ml, mlp):
                if (lam, mu) not in conj_y:
                    while am not in powers:
                        top = max(powers)
                        powers[top + 1] = powers[top] * eiphi
                    base = _ylm_norm(lam, am) * _assoc_legendre_torch(lam, am, ct, st)
                    if mu >= 0:
                        conj_y[(lam, mu)] = base * torch.conj(powers[am])
                    else:
                        conj_y[(lam, mu)] = (-1.0) ** am * base * powers[am]
                coef = 4.0 * pi * (1j**lam) * _gaunt(lp, mlp, lam, mu, l, ml)
                eps = eps + coef * on_grid[lam] * conj_y[(lam, mu)]
            out[i] = scale * eps / kabs**2
    losses = np.array([tr.energy_loss for tr in transitions], dtype=np.float64)
    return out, losses


@dataclass
class TransitionPotentials:
    """Real-space transition potentials of a set of channels on one grid.

    Parameters
    ----------
    array : Tensor
        Complex tensor ``(n_transitions, Ny, Nx)`` of real-space transition
        potentials :math:`H_{n0}(\\mathbf r_\\perp)`.
    Z : int
        Atomic number of the ionized element (used to locate sites).
    energy : float
        Probe energy [eV].
    sampling : tuple[float, float]
        Real-space pixel size ``(dy, dx)`` [Å].
    energy_losses : ndarray
        Per-transition energy loss [eV].
    """

    array: Tensor
    Z: int
    energy: float
    sampling: tuple
    energy_losses: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def gpts(self) -> tuple:
        """Grid size of the potentials.

        Returns
        -------
        tuple
            `(Ny, Nx)`, the trailing shape of `array`.
        """
        return tuple(self.array.shape[-2:])

    def __len__(self) -> int:
        """Number of channels.

        Returns
        -------
        int
            The number of transitions.
        """
        return int(self.array.shape[0])

    def to(self, device=None, dtype=None) -> TransitionPotentials:
        """Move or cast ``array`` in place.

        Parameters
        ----------
        device : torch.device, optional
            Target device. Default is None, which leaves the device unchanged.
        dtype : torch.dtype, optional
            Target dtype. Default is None, which leaves the dtype unchanged.

        Returns
        -------
        TransitionPotentials
            This instance, with `array` moved/cast.
        """
        self.array = self.array.to(device=device, dtype=dtype)
        return self


def _resolve_build_device(method: str, device) -> tuple:
    """Choose the build device and backend.

    Parameters
    ----------
    method : str
        ``"auto"``, ``"gpu"`` or ``"cpu"``.
    device : torch.device or str or None
        Requested output device.

    Returns
    -------
    tuple
        ``(build_device, vectorised)``. ``"cpu"`` gives the numpy reference
        on the CPU. ``"auto"`` with an explicit ``device`` builds there. Else a
        CUDA output device, else a present CUDA device, is used with the
        vectorised builder; ``"auto"`` without CUDA builds on the CPU.

    Raises
    ------
    RuntimeError
        If ``method="gpu"`` and no CUDA device is requested or available.
    ValueError
        If ``method`` is unknown.
    """
    if method == "cpu":
        return torch.device("cpu"), False
    if method not in ("auto", "gpu"):
        raise ValueError(f"unknown method {method!r}; use 'auto', 'gpu' or 'cpu'")
    if device is not None and (method == "auto" or torch.device(device).type == "cuda"):
        return torch.device(device), True
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if method == "gpu":
        raise RuntimeError("method='gpu' requested but CUDA is not available")
    return torch.device("cpu"), True


def build_transition_potentials(
    transitions: Sequence[Transition],
    Z: int,
    gpts: Sequence[int],
    sampling: Sequence[float],
    eV: float,
    bandlimit: float | None = 2.0 / 3.0,
    device=None,
    dtype: torch.dtype = torch.complex64,
    method: str = "auto",
) -> TransitionPotentials:
    """Real-space transition potentials ``H_n0`` on a periodic grid.

    ``fft2(array) = H~(2 pi f) / (dy dx)`` inside the band limit, with
    ``H~ = sigma(E0) sqrt(2 k_0/k_n) 4 pi (E_h a_0) epsilon_eV(K) / |K|^2``
    (see the module docstring for the frame, phase, spin factor and units). The origin is the corner
    pixel. The forward models use ``psi_n = H_n0 psi_0``.

    Parameters
    ----------
    transitions : list of Transition
        Channels from :func:`subshell_transitions`.
    Z : int
        Atomic number of the ionized element.
    gpts : int
        Shape ``(2,)``. Grid size ``(Ny, Nx)``.
    sampling : float
        Shape ``(2,)``. Real-space pixel size ``(dy, dx)`` [Å].
    eV : float
        Probe energy [eV].
    bandlimit : float or None
        Radius of the hard elliptical band limit in reciprocal space, as a
        fraction of the Nyquist frequency of each axis. Default 2/3. None
        applies no band limit.
    device : torch.device, optional
        Device of the returned ``array``. Default None: the CPU.
    dtype : torch.dtype
        Complex dtype of the returned ``array``.
    method : str
        One of ``"auto"``, ``"gpu"`` or ``"cpu"``. Backend for the partial-
        wave / radial-overlap / form-factor build. ``"auto"`` (default) runs
        the vectorised path on ``device``, or with ``device=None`` on CUDA when
        available, else on the CPU; ``"gpu"`` forces CUDA; ``"cpu"`` numpy.

    Returns
    -------
    TransitionPotentials
        Real-space transition potentials on the requested grid, in eV^(-1/2).
    """
    if isinstance(transitions, Transition):
        transitions = [transitions]
    transitions = list(transitions)
    gpts = (int(gpts[0]), int(gpts[1]))
    sampling = (float(sampling[0]), float(sampling[1]))
    eV = float(eV)
    build_device, vectorised = _resolve_build_device(method, device)
    fa, fb = np.broadcast_arrays(
        *dft_sample_frequencies(
            PeriodicGrid(tuple(gpts), (gpts[0] * sampling[0], gpts[1] * sampling[1]))
        )
    )
    if vectorised:
        recip, losses = _build_transition_potentials_gpu(
            transitions,
            torch.as_tensor(np.ascontiguousarray(fa)),
            torch.as_tensor(np.ascontiguousarray(fb)),
            eV,
            sampling,
            build_device,
        )
    else:
        recip = torch.from_numpy(_build_cpu(transitions, fa, fb, eV, sampling))
        losses = np.array([tr.energy_loss for tr in transitions], dtype=np.float64)
    if bandlimit is not None:
        recip = recip * ellipse_mask(
            (int(recip.shape[-2]), int(recip.shape[-1])),
            bandlimit,
            half_last_axis=False,
            device=recip.device,
            dtype=recip.real.dtype,
        )
    array = torch.fft.ifft2(recip)
    out_device = torch.device("cpu") if device is None else torch.device(device)
    return TransitionPotentials(
        array=array.to(device=out_device, dtype=dtype),
        Z=Z,
        energy=eV,
        sampling=sampling,
        energy_losses=losses,
    )
