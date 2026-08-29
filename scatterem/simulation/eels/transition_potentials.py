"""Inelastic ionization transition potentials for STEM-EELS.

Implementation of the core-loss transition potential
:math:`H_{n0}(\\mathbf{r}_\\perp)` (Brown *et al.*, Phys. Rev. Research **1**,
033186 (2019), Eqs. (8)-(9); Dwyer, Ultramicroscopy **104**, 141 (2005)).

The transition potential for an ionization event taking a bound electron in
state :math:`(n, \\ell, m_\\ell)` to a continuum state :math:`(\\ell', m_{\\ell'})`
is expanded in partial waves :math:`\\ell''`:

.. math::

    H_{n0}(\\mathbf{q}) = \\frac{\\gamma}{2\\pi^2 k_n q^2}
        \\sum_{\\ell''} (-i)^{\\ell''} \\sqrt{(2\\ell'+1)(2\\ell''+1)(2\\ell+1)}\\;
        4\\pi\\, (-1)^{m_{\\ell'}+m_{\\ell''}}
        \\begin{pmatrix}\\ell' & \\ell'' & \\ell\\\\ 0 & 0 & 0\\end{pmatrix}
        \\begin{pmatrix}\\ell' & \\ell'' & \\ell\\\\ -m_{\\ell'} & -m_{\\ell''} & m_\\ell\\end{pmatrix}
        j_{\\ell''}(q)\\, Y_{\\ell''}^{m_{\\ell''}}(\\hat{\\mathbf q})

where :math:`j_{\\ell''}(q) = \\int u_{n\\ell}(r)\\, j_{\\ell''}(qr)\\, u_{\\ell'}(r)\\,dr`
is the radial overlap of the bound and continuum wavefunctions.

The potential is evaluated on the scatterem reciprocal-space grid (corner
origin, ``q_space_array`` convention) and returned in **real space** so it can be
multiplied directly onto the (real-space) electron wave during multislice.

The output is independent of how the radial wavefunctions were obtained, so the
GPAW backend and the analytic test backend share this code exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, pi, sqrt
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from scatterem.simulation._electron_optics import (
    energy2sigma,
    energy2wavelength,
    relativistic_mass_correction,
)

from .._grid import bandwidth_limit_array_torch, q_space_array
from . import trapezoid_fused as _tf
from ._angular import spherical_bessel, spherical_harmonic, wigner_3j
from .radial import (
    RadialWavefunction,
    bound_wavefunction,
    continuum_wavefunction,
)

__all__ = [
    "Transition",
    "subshell_transitions",
    "TransitionPotentials",
    "build_transition_potentials",
]

_BOHR_ANG = 0.52917721090380
_RYDBERG_EV = 13.605693122994


@dataclass
class Transition:
    """A single bound -> continuum ionization channel."""

    bound: RadialWavefunction
    excited: RadialWavefunction
    ml: int
    mlprime: int

    @property
    def energy_loss(self) -> float:
        """Energy loss of the fast electron for this transition [eV]."""
        return self.excited.energy - self.bound.energy


def subshell_transitions(
    Z: int,
    n: int,
    l: int,
    epsilon: float,
    lprimes: Optional[Sequence[int]] = None,
    order: int = 1,
    backend: str = "auto",
    xc: str = "PBE",
    potential_scale: float = 1.0,
) -> List[Transition]:
    """Enumerate the ionization channels for a subshell.

    Parameters
    ----------
    Z, n, l : int
        Element and the ionized bound orbital (e.g. Fe ``L`` edge: Z=26, n=2,
        l=1).
    epsilon : float
        Energy of the ejected electron above the ionization threshold [eV].
        Selecting different ``epsilon`` selects different points along the
        energy-loss edge.
    lprimes : sequence of int, optional
        Continuum angular momenta to include.  Defaults to every ``l'`` with
        ``|l - l'| <= order``, i.e. ``range(max(l - order, 0), l + order + 1)``.
        This includes the monopole ``l' = l`` channel as well as the dipole
        ``l' = l +/- 1`` channels; the published multislice-EELS treatments sum
        over this full set (``order = 1`` by default).  The partial-wave / Wigner-3j
        parity rule in :func:`_form_factor` zeros any channels that do not
        actually contribute.
    order : int
        Maximum change in angular momentum ``|l - l'|`` retained when
        ``lprimes`` is not given explicitly.
    backend : str
        ``"gpaw"``, ``"hydrogenic"`` or ``"auto"`` (see :mod:`.radial`).
    xc : str
        Exchange-correlation functional for the GPAW backend.
    potential_scale : float
        Empirical multiplier on the GPAW continuum effective potential (default
        1.0 = the published Brown 2019 physics; abTEM uses 1.02).  Affects mainly the
        l'=l monopole s-wave; see :func:`scatterem...radial.continuum_wavefunction`.
    """
    if lprimes is None:
        lprimes = [lp for lp in range(max(l - order, 0), l + order + 1)]

    bound = bound_wavefunction(Z, n, l, xc=xc, backend=backend)
    excited = {
        lp: continuum_wavefunction(
            Z, lp, epsilon, xc=xc, backend=backend, potential_scale=potential_scale
        )
        for lp in lprimes
    }

    transitions: List[Transition] = []
    for lp in lprimes:
        for ml in range(-l, l + 1):
            for mlprime in range(-lp, lp + 1):
                transitions.append(Transition(bound, excited[lp], ml, mlprime))
    return transitions


def _radial_overlap(
    lprimeprime: int,
    bound: RadialWavefunction,
    excited: RadialWavefunction,
    q1d: np.ndarray,
) -> np.ndarray:
    """Radial overlap integral :math:`\\int u_b\\, j_{\\ell''}(qr)\\, u_e\\,dr`.

    Evaluated on a 1-D ``q`` grid (1/Å) for later interpolation onto the 2-D
    reciprocal grid.  ``r`` is in Bohr.
    """
    rmax = min(bound.rmax, excited.rmax)
    r = np.linspace(0.0, rmax, 20000)  # Bohr
    ube = bound(r) * excited(r)  # u_b(r) u_e(r)
    # Spherical Bessel argument is dimensionless: q[1/Å] * (r * Bohr)[Å] * 2π.
    arg = 2.0 * pi * _BOHR_ANG * np.outer(q1d, r)
    jmat = spherical_bessel(lprimeprime, arg)  # (Nq, Nr)
    integrand = jmat * ube[None, :]
    integral = np.trapezoid(integrand, r, axis=1)
    # Unit normalisation matching the atomic-unit radial functions.
    return integral / (_BOHR_ANG * sqrt(_RYDBERG_EV))


def _form_factor(
    transition: Transition,
    qabs: np.ndarray,
    qphi: np.ndarray,
    qtheta: np.ndarray,
) -> np.ndarray:
    """Partial-wave sum giving the (unnormalised) form factor on the q-grid.

    Reference CPU (numpy/scipy) implementation.  The GPU-vectorised path in
    :func:`_build_transition_potentials_gpu` reproduces this exactly (validated
    to ~machine precision); this one is kept as the readable physics reference
    and as the ``method="cpu"`` fallback / oracle.
    """
    bound, excited = transition.bound, transition.excited
    l, lp = bound.l, excited.l
    ml, mlp = transition.ml, transition.mlprime

    q1d = np.linspace(0.0, float(qabs.max()) * 1.001 + 1e-6, 512)
    Hn0 = np.zeros_like(qabs, dtype=np.complex128)

    for lpp in range(abs(l - lp), l + lp + 1):
        # m-selection rule must admit at least one m'' for this l''.
        if all((ml - mlp - mpp) != 0 for mpp in range(-lpp, lpp + 1)):
            continue
        overlap_1d = _radial_overlap(lpp, bound, excited, q1d)
        jq = np.interp(qabs, q1d, overlap_1d)
        for mpp in range(-lpp, lpp + 1):
            if ml - mlp - mpp != 0:
                continue
            prefactor = (
                sqrt(4.0 * pi)
                * ((-1j) ** lpp)
                * sqrt((2 * lp + 1) * (2 * lpp + 1) * (2 * l + 1))
                * ((-1.0) ** (mlp + mpp))
                * wigner_3j(lp, lpp, l, 0, 0, 0)
                * wigner_3j(lp, lpp, l, -mlp, -mpp, ml)
            )
            if abs(prefactor) < 1e-12:
                continue
            ylm = spherical_harmonic(mpp, lpp, qphi, qtheta)
            Hn0 += prefactor * jq * ylm
    return Hn0


# --------------------------------------------------------------------------- #
# GPU-vectorised inner kernels
#
# The CPU path above spends ~all of its time in the per-transition radial
# overlap (a scipy ``spherical_jn`` matrix of shape ``(N_q1d, N_r)`` per channel,
# recomputed once for every transition that touches a given ``l''``).  The GPU
# path below batches that work onto CUDA tensors and deduplicates the radial
# overlap to the handful of unique ``(l', l'')`` channels, while reproducing the
# CPU numbers to ~machine precision:
#
#   * ``_spherical_bessel_torch`` matches ``scipy.special.spherical_jn`` to ~5e-16
#     (exact ``j_0, j_1`` + upward recurrence for ``x >= x_thr``; a 12-term power
#     series below ``x_thr`` where upward recurrence is unstable for ``l'' > x``).
#   * ``_spherical_harmonic_torch`` matches ``scipy.special.sph_harm_y`` to ~5e-16
#     (closed-form associated Legendre with the Condon-Shortley phase).
#   * ``_interp_torch`` reproduces ``numpy.interp`` (uniform grid, clamped ends).
# --------------------------------------------------------------------------- #
def _spherical_bessel_torch(
    lmax: int, x: Tensor, x_thr: float = 2.0, n_series: int = 12
) -> Tensor:
    """Spherical Bessel ``j_0..j_lmax(x)`` matching ``scipy.special.spherical_jn``.

    Returns a tensor of shape ``(lmax + 1, *x.shape)`` in double precision.  For
    ``x >= x_thr`` the exact ``j_0 = sin x / x`` and ``j_1`` seed a numerically
    stable upward recurrence (stable while ``l <= x``).  For ``x < x_thr`` the
    recurrence is replaced by the convergent power series, where ``x > l''`` may
    fail.  Validated against scipy to ~5e-16 over ``x in [0, ~2000]``.
    """
    x = x.to(torch.float64)
    shape = x.shape
    xf = x.reshape(-1)
    small = xf < x_thr
    # Both host round trips this function needs are taken HERE, before the
    # recurrence below is issued, and they are the only ones in it.  They depend
    # on nothing but the comparison above, so at this point the sync waits for one
    # cheap elementwise kernel instead of for the whole fp64 sin/cos/divide chain
    # -- and, more importantly, the recurrence and the series are then issued with
    # no drain between them or after them, so the caller's own work (the integrand,
    # the trapezoid, the interpolations, the harmonics) is enqueued while the card
    # is still on the ladder.  Measured on r7_eels/small/build_tp: the whole call's
    # cudaStreamSynchronize host time falls 6.7 -> 0.6 ms and the row 21.4 -> 20.5.
    anysmall = bool(small.any())
    if anysmall:
        # One bool -> index conversion, not lmax + 2 of them (``xf[small]`` plus
        # one ``out[ell, small]`` per level each run ``nonzero`` over the whole
        # argument matrix).
        sidx = small.nonzero(as_tuple=True)[0]
        xs = xf[sidx]
        x2 = xs * xs
    # Evaluate the recurrence on a safe argument everywhere; small-x entries are
    # overwritten by the series below (so they never feed an unstable recurrence).
    #
    # Every temporary on this path is the size of ``x`` -- the caller evaluates
    # this on a (Nq=512, Nr=20000) float64 argument matrix, i.e. 82 MB *per
    # level* -- so the spellings below are chosen to materialise each of them
    # once.  All of them are BITWISE the expressions they replace; the one
    # respelling that is NOT (``torch.div`` with a 0-dim tensor numerator in
    # place of a python scalar, which differs by 1 ULP on ~4 % of elements) is
    # deliberately avoided, so the scalar divisions below stay python scalars.
    xsafe = torch.where(small, torch.tensor(x_thr, dtype=xf.dtype, device=xf.device), xf)
    # Each level is written STRAIGHT INTO the (lmax+1, N) result, instead of being
    # accumulated in a python list and then copied into one by ``torch.stack``.
    # Every expression below is the one it replaces, on the same values in the same
    # order -- only the buffer each result lands in changes -- so this is bitwise
    # what it replaces, and ``benchmarks/lab/oracle.py``'s golden sha256 gates that
    # rather than taking it on trust.  ``out[k]`` is a row of a contiguous 2-D
    # tensor, hence contiguous itself, so no kernel here loses its vectorised path.
    #
    # What it removes is the stack, and that was the ONLY pass in this row carrying
    # no compulsory traffic: measured at 0.964 ms of an 18.9 ms
    # r7_eels/small/build_tp, reading and rewriting the whole ladder (4 x 82 MB on
    # the Ti L fixture) at 680 GB/s -- i.e. already at this card's DRAM roof --
    # purely to make levels that already existed adjacent in memory.  Every other
    # pass in the row was measured at 673-680 GB/s doing arithmetic the result
    # depends on.
    out = torch.empty((lmax + 1, xf.numel()), dtype=xf.dtype, device=xf.device)
    sinx = torch.sin(xsafe)  # shared by j0 and j1
    torch.div(sinx, xsafe, out=out[0])
    if lmax >= 1:
        torch.div(sinx, xsafe**2, out=out[1])
        # Released at its last use, not at function exit: holding it one statement
        # longer costs a whole extra argument-matrix buffer at peak (measured
        # +78 MiB on r7_eels/wide/build_tp, where the ladder IS the peak) and buys
        # nothing.
        del sinx
        out[1] -= torch.cos(xsafe) / xsafe
    else:
        del sinx
    for ell in range(1, lmax):
        col = (2 * ell + 1) / xsafe
        torch.mul(col, out[ell], out=out[ell + 1])
        del col
        out[ell + 1] -= out[ell - 1]

    if anysmall:
        # ``-x2 / 2.0`` is the series' ratio factor and it depends on NEITHER loop
        # variable, yet it was rebuilt once per (level, k) -- 44 times on the
        # registered Ti L fixture, i.e. 88 kernels over the small-x subvector, and
        # 176 of the row's 883 python-level dispatches at one line.  Hoisted, and
        # the body then runs in place: ``term`` is dead the moment the next one is
        # formed (``ssum`` has already consumed it), so the multiply and the divide
        # can overwrite it instead of allocating two fresh buffers per iteration.
        # Same operations on the same values in the same order, hence bitwise what
        # it replaces -- gated by the golden sha256 at every registered size.
        nhx2 = -x2 / 2.0
        dblfac = 1.0  # (2l+1)!!
        for ell in range(lmax + 1):
            if ell > 0:
                dblfac *= 2 * ell + 1
            # j_l(x) = x^l/(2l+1)!! * sum_k (-x^2/2)^k / (k! prod_i(2l+2i+1))
            term = torch.ones_like(xs)
            ssum = term.clone()
            for k in range(1, n_series):
                term = term.mul_(nhx2).div_(k * (2 * ell + 2 * k + 1))
                ssum = ssum.add_(term)
            out[ell, sidx] = xs**ell / dblfac * ssum
    return out.reshape(lmax + 1, *shape)


# Closed-form associated Legendre P_l^|m|(cos θ) with the Condon-Shortley phase,
# expressed in (cos θ, sin θ) so it stays exact at the poles.  Keyed by (l, |m|).
def _assoc_legendre_torch(ell: int, am: int, ct: Tensor, st: Tensor) -> Tensor:
    if ell == 0:
        return torch.ones_like(ct)
    if ell == 1:
        return ct if am == 0 else -st
    if ell == 2:
        if am == 0:
            return 0.5 * (3.0 * ct**2 - 1.0)
        if am == 1:
            return -3.0 * ct * st
        return 3.0 * st**2
    if ell == 3:
        if am == 0:
            return 0.5 * (5.0 * ct**3 - 3.0 * ct)
        if am == 1:
            return -1.5 * (5.0 * ct**2 - 1.0) * st
        if am == 2:
            return 15.0 * ct * st**2
        return -15.0 * st**3
    raise NotImplementedError(f"associated Legendre not tabulated for l={ell}")


def _spherical_harmonic_torch(
    m: int, ell: int, phi: Tensor, theta: Tensor, cache: Optional[dict] = None
) -> Tensor:
    """Complex ``Y_l^m`` matching ``scipy.special.sph_harm_y`` to ~5e-16.

    ``phi`` (azimuth) and ``theta`` (polar) are real double tensors; the result
    is complex128.

    ``cache`` is an optional mutable dict into which the three quantities that do
    *not* depend on ``(m, ell)`` are memoised: ``cos(theta)``, ``sin(theta)`` and
    ``exp(i |m| phi)`` (the last keyed by ``|m|``, of which there are at most
    ``l''_max + 1`` distinct values).  It exists because the caller evaluates this
    function once per ``(l'', m'')`` of a transition group -- 27 times per call on
    the registered Ti L fixture -- on ONE ``theta`` and ONE ``phi``, so those three
    tensors were being rebuilt 27 times each.  Every memoised value is the same
    kernel on the same input, hence bitwise what it replaces; passing ``None``
    reproduces the original per-call behaviour exactly.

    The dict is keyed by nothing, so it is only valid for a fixed
    ``(phi, theta)`` pair: the caller must create a fresh one whenever either
    changes (i.e. per transition group, since ``theta`` depends on the group's
    energy loss).
    """
    am = abs(m)
    if cache is None:
        ct = torch.cos(theta)
        st = torch.sin(theta)
        eiam = torch.exp(1j * am * phi.to(torch.complex128))
    else:
        if "ct" not in cache:
            cache["ct"] = torch.cos(theta)
            cache["st"] = torch.sin(theta)
        ct, st = cache["ct"], cache["st"]
        key = ("e", am)
        if key not in cache:
            if "phi_c" not in cache:
                cache["phi_c"] = phi.to(torch.complex128)
            cache[key] = torch.exp(1j * am * cache["phi_c"])
        eiam = cache[key]
    P = _assoc_legendre_torch(ell, am, ct, st)
    norm = sqrt((2 * ell + 1) / (4.0 * pi) * factorial(ell - am) / factorial(ell + am))
    y = norm * P * eiam
    if m < 0:
        y = ((-1.0) ** am) * torch.conj(y)
    return y


def _interp_torch(xq: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation matching ``numpy.interp`` for a *uniform* ``xp`` grid.

    ``xp`` is the 1-D query grid (uniform, ascending, ``xp[0] == 0``); ``xq`` is
    arbitrary-shaped.  Values outside ``[xp[0], xp[-1]]`` are clamped to the end
    samples (numpy's default), but the callers guarantee ``xq`` lies inside.
    """
    n = xp.numel()
    dx = (xp[-1] - xp[0]) / (n - 1)
    pos = (xq - xp[0]) / dx
    idx = torch.clamp(pos.floor().long(), 0, n - 2)
    frac = (pos - idx).clamp(0.0, 1.0)
    f0 = fp[idx]
    f1 = fp[idx + 1]
    out = f0 + (f1 - f0) * frac.to(fp.dtype)
    return out


def _radial_overlaps_torch(
    lpps: Sequence[int],
    bound: RadialWavefunction,
    excited: RadialWavefunction,
    q1d: Tensor,
    ladder: Optional[dict] = None,
) -> Tensor:
    """Batched radial overlaps ``∫ u_b j_{l''}(qr) u_e dr`` for several ``l''``.

    GPU/torch port of :func:`_radial_overlap` evaluated for every ``l''`` in
    ``lpps`` at once (shared Bessel argument matrix).  Returns ``(len(lpps), Nq)``
    in double precision.  Numerically identical to looping :func:`_radial_overlap`.

    ``ladder`` is an optional mutable cache, ``{"lmax": int}`` on the first call,
    into which the spherical-Bessel table is memoised for reuse by later calls that
    share the *identical* argument matrix.  The caller
    (:func:`_build_transition_potentials_gpu`) is responsible for only passing one
    when that is true -- see the guard there.  Passing ``None`` reproduces the
    original per-call behaviour exactly.
    """
    device = q1d.device
    rmax = min(bound.rmax, excited.rmax)
    r = torch.linspace(0.0, rmax, 20000, dtype=torch.float64, device=device)  # Bohr
    if ladder is not None and "jall" in ladder:
        jall = ladder["jall"]  # identical argument matrix; see the caller's guard
    else:
        arg = (2.0 * pi * _BOHR_ANG) * torch.outer(q1d, r)  # (Nq, Nr)
        # Build to the deepest level any sharing group needs, not just this one's.
        depth = max(lpps) if ladder is None else int(ladder["lmax"])
        jall = _spherical_bessel_torch(depth, arg)  # (depth+1, Nq, Nr)
        if ladder is not None:
            ladder["jall"] = jall
    # u_b(r) u_e(r): evaluate the cached numpy wavefunctions (np.interp) then move
    # to device.  These are cheap 1-D interpolations on the host -- and they are
    # issued AFTER the ladder above, not before it, because they need nothing from
    # it: the ~0.33 ms of host numpy work and the H2D then run while the card is on
    # the ladder instead of while it is idle.
    #
    # The host copy of ``r`` is RECOMPUTED here rather than fetched back from the
    # device.  ``r.cpu()`` looks free -- it is 160 kB, and the comment above says it
    # is deliberately issued behind the ladder -- but on the legacy default stream a
    # pageable D2H is cudaMemcpyAsync PLUS cudaStreamSynchronize, so "issued behind
    # the ladder" means "waits for the ladder": measured 5.730 ms of host wait over
    # three calls on a 21.7 ms r7_eels/medium/build_tp, the single largest host cost
    # in that row.  The host cannot run ahead and enqueue the transition loop until
    # it clears, so the drain costs more than the transfer.
    #
    # This is bitwise the tensor it replaces, not approximately: torch.linspace on
    # CPU and on CUDA agree to the last bit for these arguments (measured 0 of 20000
    # elements differing at all seven rmax values probed, including the fixture's
    # 40.0).  ``np.linspace`` does NOT -- it differs on 177-3726 of 20000 elements
    # at 1e-16..1e-14 -- so the obvious spelling is the wrong one, and the device
    # ``r`` above is deliberately left untouched so that every DEVICE consumer of it
    # (the outer product feeding the ladder, and the trapezoid below) is unchanged
    # by construction rather than by argument.
    r_np = torch.linspace(0.0, rmax, 20000, dtype=torch.float64).numpy()
    ube = torch.as_tensor(
        bound(r_np) * excited(r_np), dtype=torch.float64, device=device
    )
    # ``lpps`` is a contiguous ascending range for every caller in the library
    # (``range(|l_b - l'|, l_b + l' + 1)``), so the level selection is a slice --
    # a VIEW -- and not a gather that copies ``n_lpp * Nq * Nr`` float64s (82 MB
    # per level) out of the shared ladder only to multiply them once.  The
    # fallback keeps working for any other sequence.
    lp = list(lpps)
    if lp == list(range(lp[0], lp[-1] + 1)):
        sel = jall[lp[0] : lp[-1] + 1]
    else:
        sel = jall[lp]
    # The weighted trapezoid is THREE full-size intermediates on an 82 MB-per-level
    # argument matrix (``sel*ube``, then ``left+right``, then ``* dx``) to produce a
    # (n_lpp, Nq) answer, and every one of them was measured at 673-680 GB/s, i.e.
    # already at this card's DRAM roof -- so the defect is not the rate, it is that
    # only one of the three has to exist.  ``trapezoid_weighted_fused`` writes the
    # summand once and hands the SAME contiguous (n_lpp, Nq, Nr-1) tensor to torch's
    # reduce, so the reduction tree -- and the golden sha256 that gates this target
    # -- is unchanged by construction.  Bitwise identical, not approximately: see
    # that module's docstring for the four roundings and the FMA suppression.
    if _tf.trapezoid_weighted_supported(sel, ube, r):
        return _tf.trapezoid_weighted_fused(sel, ube, r) / (
            _BOHR_ANG * sqrt(_RYDBERG_EV)
        )
    integrand = sel * ube.view(1, 1, -1)  # (n_lpp, Nq, Nr)
    integral = torch.trapezoid(integrand, r, dim=-1)  # (n_lpp, Nq)
    return integral / (_BOHR_ANG * sqrt(_RYDBERG_EV))


def _build_transition_potentials_gpu(
    transitions: Sequence[Transition],
    qy: Tensor,
    qx: Tensor,
    qt: Tensor,
    eV: float,
    k0: float,
    gamma: float,
    sigma: float,
    sampling: tuple,
    device,
) -> tuple:
    """Vectorised reciprocal-space transition potentials for every transition.

    Returns ``(recip, losses)`` where ``recip`` is a complex128 tensor
    ``(n_trans, Ny, Nx)`` of the per-transition ``H_n0`` *after* the same
    per-transition relativistic / dynamical normalisation, orbital filling and
    pixel-area scaling the CPU path applies (i.e. exactly the array the CPU loop
    stores in ``recip`` before the bandwidth limit + ifft2), and ``losses`` is a
    1-D numpy array.  Reproduces the CPU path to ~machine precision.
    """
    n_trans = len(transitions)
    Ny, Nx = qt.shape
    recip = torch.zeros((n_trans, Ny, Nx), dtype=torch.complex128, device=device)
    losses = np.zeros(n_trans)

    qphi = torch.atan2(qx, qy)  # depends only on the grid

    # Group transitions by the excited orbital (l', energy): they share kz, qabs,
    # qtheta, q1d and the per-(l',l'') radial overlaps -- the expensive pieces.
    groups: dict = {}
    for i, tr in enumerate(transitions):
        losses[i] = tr.energy_loss
        key = (tr.excited.l, tr.excited.energy, id(tr.excited))
        groups.setdefault(key, []).append(i)

    # One spherical-Bessel ladder shared by every group, when they provably ask for
    # the identical one.  ``_radial_overlaps_torch`` evaluates the ladder on
    # ``(2*pi*a0) * outer(q1d, r)``, and neither factor depends on the detector grid:
    # ``r`` is ``linspace(0, min(bound.rmax, excited.rmax), 20000)`` and ``q1d`` is
    # ``linspace(0, f(energy_loss), 512)``.  So groups that share ``(energy_loss,
    # rmax)`` -- the normal case, since the groups differ only in the excited
    # orbital's *angular* momentum -- need bitwise the same table, and today each of
    # them rebuilds it: a Ti L edge evaluates 9 ladder levels where 4 distinct ones
    # exist, on three identical 82 MB argument matrices.
    #
    # Depth needed is ``max(l'') = l_bound + l_excited``, so the deepest is pure
    # integer arithmetic -- no extra device work to decide this.  Level ``l`` of the
    # ladder is bitwise independent of how deep the ladder was built (the upward
    # recurrence for ``l`` reads only ``l-1`` and ``l-2``; the small-x power series
    # is evaluated per level), so slicing a deeper table is exact rather than
    # approximate.  ``benchmarks/lab/oracle.py`` gates that with ``torch.equal``
    # instead of taking it on trust.
    #
    # The guard is deliberately conservative: anything that would make the argument
    # matrices differ falls back to the original per-group build.
    heads = [transitions[idxs[0]] for idxs in groups.values()]
    _rmaxes = {min(t.bound.rmax, t.excited.rmax) for t in heads}
    _losses = {t.energy_loss for t in heads}
    ladder: Optional[dict] = None
    if len(heads) > 1 and len(_rmaxes) == 1 and len(_losses) == 1:
        ladder = {"lmax": max(t.bound.l + t.excited.l for t in heads)}

    for idxs in groups.values():
        tr0 = transitions[idxs[0]]
        bound, excited = tr0.bound, tr0.excited
        lb, lp = bound.l, excited.l  # bound / excited orbital angular momenta
        loss = tr0.energy_loss
        kn = 1.0 / energy2wavelength(eV - loss)
        kz = k0 - kn
        qabs = torch.sqrt(qt**2 + kz**2)
        kz_t = torch.tensor(kz, dtype=qt.dtype, device=device)
        qtheta = pi - torch.atan2(qt, kz_t)

        # Shared q1d grid (same per excited orbital, matching the CPU path).
        q1dmax = float(qabs.max()) * 1.001 + 1e-6
        q1d = torch.linspace(0.0, q1dmax, 512, dtype=torch.float64, device=device)

        lpps = list(range(abs(lb - lp), lb + lp + 1))
        overlaps = _radial_overlaps_torch(
            lpps, bound, excited, q1d, ladder=ladder
        )  # (n_lpp, 512)
        # Interpolate each l'' overlap onto the 2-D |q| grid once per group.
        jq = {lpp: _interp_torch(qabs, q1d, overlaps[k]) for k, lpp in enumerate(lpps)}
        # Precompute Y_l''^m'' for every (l'', m'') touched by this group once.
        # ``ysh`` is this group's harmonic cache: ``qtheta`` and ``qphi`` are fixed
        # here, so cos/sin(qtheta) and exp(i|m''|qphi) are built once instead of
        # once per (l'', m'').  Created INSIDE the group loop -- qtheta is a
        # function of the group's energy loss, so sharing it across groups would be
        # wrong.
        ylm: dict = {}
        ysh: dict = {}
        for lpp in lpps:
            for mpp in range(-lpp, lpp + 1):
                ylm[(lpp, mpp)] = _spherical_harmonic_torch(
                    mpp, lpp, qphi, qtheta, cache=ysh
                )

        # Shared per-group normalisation (the CPU path's per-transition factors
        # depend only on (kn, qabs, lb), all constant within a group).
        norm = gamma / (2.0 * pi**2 * kn * qabs**2 * sigma)
        norm = norm * sqrt(4 * lb + 2) / (sampling[0] * sampling[1])

        for i in idxs:
            tr = transitions[i]
            ml, mlp = tr.ml, tr.mlprime
            # Seeded from the FIRST surviving term rather than from a zero buffer:
            # ``0 + t`` is bitwise ``t`` for every finite t except a negative zero,
            # whose sign it flips, and no component of ``t`` can be an exact zero
            # here (it is prefactor * jq * Y_l''^m'' on a grid where sin(qtheta) is
            # 1.2e-16 at the DC pixel and never 0).  Verified as bit-exact by the
            # golden sha256 at every registered size, not assumed.  Most transitions
            # contribute a single term, so this removes 27 zero-fills and 27 adds of
            # a 192x192 complex128 plane per call on the Ti L fixture.
            acc = None
            for lpp in lpps:
                mpp = ml - mlp  # the only m'' with a nonzero 3-j (m1+m2+m3=0)
                if mpp < -lpp or mpp > lpp:
                    continue
                prefactor = (
                    sqrt(4.0 * pi)
                    * ((-1j) ** lpp)
                    * sqrt((2 * lp + 1) * (2 * lpp + 1) * (2 * lb + 1))
                    * ((-1.0) ** (mlp + mpp))
                    * wigner_3j(lp, lpp, lb, 0, 0, 0)
                    * wigner_3j(lp, lpp, lb, -mlp, -mpp, ml)
                )
                if abs(prefactor) < 1e-12:
                    continue
                term = prefactor * jq[lpp] * ylm[(lpp, mpp)]
                acc = term if acc is None else acc + term
            if acc is None:
                acc = torch.zeros((Ny, Nx), dtype=torch.complex128, device=device)
            # ``recip[i] = acc * norm`` allocates the product and then COPIES it into
            # the row; the multiply can write the destination directly.
            torch.mul(acc, norm, out=recip[i])
    return recip, losses


@dataclass
class TransitionPotentials:
    """Stack of real-space transition potentials for one edge / energy point.

    Attributes
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
        return tuple(self.array.shape[-2:])

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def to(self, device=None, dtype=None) -> "TransitionPotentials":
        self.array = self.array.to(device=device, dtype=dtype)
        return self


def _resolve_build_device(method: str, device) -> tuple:
    """Pick the worker device for the partial-wave / form-factor build.

    Returns ``(build_device, use_gpu)``.  ``method`` is ``"auto"`` (use CUDA when
    available -- either the requested ``device`` is CUDA or a CUDA device exists),
    ``"gpu"`` (force CUDA; error if none) or ``"cpu"`` (force the numpy/scipy
    reference path).  ``build_device`` is where the heavy form-factor tensors
    live; the final ifft2 output is moved back to the caller's ``device``.
    """
    if method == "cpu":
        return torch.device("cpu"), False
    requested = torch.device(device) if device is not None else None
    if method == "gpu":
        if requested is not None and requested.type == "cuda":
            return requested, True
        if torch.cuda.is_available():
            return torch.device("cuda"), True
        raise RuntimeError("method='gpu' requested but CUDA is not available")
    # auto
    if requested is not None and requested.type == "cuda":
        return requested, True
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def build_transition_potentials(
    transitions: Sequence[Transition],
    Z: int,
    gpts: Sequence[int],
    sampling: Sequence[float],
    eV: float,
    bandwidth_limit: float = 2.0 / 3.0,
    device=None,
    dtype: torch.dtype = torch.complex64,
    method: str = "auto",
) -> TransitionPotentials:
    """Build real-space transition potentials on a scatterem grid.

    Parameters
    ----------
    transitions : sequence of Transition
        Channels from :func:`subshell_transitions`.
    Z : int
        Atomic number of the ionized element.
    gpts : (2,) int
        Grid size ``(Ny, Nx)``.
    sampling : (2,) float
        Real-space pixel size ``(dy, dx)`` [Å].
    eV : float
        Probe energy [eV].
    bandwidth_limit : float
        Hard bandwidth limit (fraction of Nyquist) applied in reciprocal space.
    device : torch device, optional
        Device of the returned ``array``.
    dtype : torch.dtype
        Output complex dtype (the heavy build is always done in complex128).
    method : {"auto", "gpu", "cpu"}
        Backend for the partial-wave / radial-overlap / form-factor build.
        ``"auto"`` (default) runs the GPU-vectorised path on CUDA when available
        and otherwise the numpy/scipy reference path.  ``"gpu"`` forces CUDA;
        ``"cpu"`` forces the reference path.  All paths are numerically identical
        to ~machine precision (the GPU path reproduces the reference to ~1e-12).
    """
    gpts = (int(gpts[0]), int(gpts[1]))
    sampling = (float(sampling[0]), float(sampling[1]))
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])

    k0 = 1.0 / energy2wavelength(eV)
    gamma = relativistic_mass_correction(eV)
    # Interaction parameter sigma(E).  The transition potential is applied to the
    # wave multiplicatively (psi_n = H * psi), exactly like the elastic phase
    # grating exp(i*sigma*V), so H must carry the same 1/sigma scaling for its
    # absolute amplitude to be physically meaningful.  Validated against abTEM
    # 1.0.9 (stem_eels/compare_abtem.py): including this divisor brings our
    # |H_n0| to within ~7% of abTEM's (residual = continuum-normalisation
    # convention); omitting it leaves the absolute cross-section ~1/sigma^2 too
    # small.  Relative maps / selection rules are unaffected (constant scalar).
    sigma = energy2sigma(eV)

    build_device, use_gpu = _resolve_build_device(method, device)

    if use_gpu:
        qy_np, qx_np = q_space_array(gpts, gridsize)
        qy = torch.tensor(
            np.ascontiguousarray(qy_np), dtype=torch.float64, device=build_device
        )
        qx = torch.tensor(
            np.ascontiguousarray(qx_np), dtype=torch.float64, device=build_device
        )
        qt = torch.sqrt(qy**2 + qx**2)
        arr, losses = _build_transition_potentials_gpu(
            transitions, qy, qx, qt, eV, k0, gamma, sigma, sampling, build_device
        )
    else:
        qy, qx = q_space_array(gpts, gridsize)  # meshed, 1/Å, corner-origin
        qt = np.sqrt(qy**2 + qx**2)

        recip = np.zeros((len(transitions), *gpts), dtype=np.complex128)
        losses = np.zeros(len(transitions))

        for i, tr in enumerate(transitions):
            loss = tr.energy_loss  # > 0
            losses[i] = loss
            kn = 1.0 / energy2wavelength(eV - loss)
            kz = k0 - kn  # momentum transfer along the beam
            qabs = np.sqrt(qt**2 + kz**2)
            # Polar angle of the scattering vector.  With complex spherical
            # harmonics the physical convention is theta measured from the +z
            # axis of the momentum-transfer vector q = (qt, -kz), i.e.
            # theta = pi - arctan(qt/kz); this is the convention the published
            # multislice-EELS results are computed in, and it reproduces them
            # numerically.  Validate any change against a published reference
            # before adopting it.
            qtheta = pi - np.arctan2(qt, kz)
            qphi = np.arctan2(qx, qy)

            Hn0 = _form_factor(tr, qabs, qphi, qtheta)
            # Relativistic / dynamical normalisation and orbital filling factor.
            Hn0 *= gamma / (2.0 * pi**2 * kn * qabs**2 * sigma)
            Hn0 *= sqrt(4 * tr.bound.l + 2)
            Hn0 /= sampling[0] * sampling[1]
            recip[i] = Hn0
        arr = torch.as_tensor(recip, dtype=torch.complex128, device=build_device)

    # Keep full double precision through the ill-conditioned ``1/qabs**2``
    # normalisation (kz = k0 - kn is a small difference of large numbers) and the
    # bandwidth-limit + inverse FFT; only cast to the requested output dtype at
    # the very end.
    arr = bandwidth_limit_array_torch(
        arr, limit=bandwidth_limit, qspace_in=True, qspace_out=True
    )
    real_space = torch.fft.ifft2(arr, dim=(-2, -1))
    # Preserve the original output contract: ``device=None`` returns a CPU tensor
    # (the build may have run on CUDA for speed, but downstream multislice mixes
    # it with CPU probes/transmissions when no device is given).
    out_device = torch.device("cpu") if device is None else device
    real_space = real_space.to(device=out_device, dtype=dtype)

    return TransitionPotentials(
        array=real_space,
        Z=Z,
        energy=float(eV),
        sampling=sampling,
        energy_losses=losses,
    )
