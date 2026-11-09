"""The vectorised transition-potential build, exercised on the CPU lane.

``build_transition_potentials`` has two implementations of the same physics: a
numpy/scipy reference (``method="cpu"``) and a torch-vectorised one
(``_build_transition_potentials_gpu``) that ``_resolve_build_device`` only
selects when CUDA is present.  The vectorised path is device-agnostic torch, so
everything below drives it on the CPU by handing it a CPU device directly --
which is what makes its "reproduces the reference to ~machine precision" claim
checkable without a GPU.

Every helper it is built from has an independent reference: ``scipy.special``
for the Bessel ladder and the harmonics, ``numpy.interp`` for the interpolator,
and the numpy ``_radial_overlap`` for the radial integral.
Those references are what is asserted against, not stored numbers.

The checks of the form factor and of the absolute scale use independent
oracles. They were written from textbook formulas and from the behavioural
specification of the EELS rewrite ("spec" below; see THIRD-PARTY-NOTICES.md):

* the angular factor is the Gaunt integral ``int Y*_(l'm') Y_(lambda mu)
  Y_(lm) dOmega``. The oracle integrates it exactly: the azimuthal integral
  gives ``2 pi`` times a Kronecker delta, and the polar integral is an exact
  rational integral of Legendre polynomials from Rodrigues' formula. A sphere
  quadrature of ``scipy.special.sph_harm_y`` checks this oracle;
* the radial integral is an adaptive ``scipy.integrate.quad`` over the
  textbook closed forms of the hydrogenic orbitals and the free continuum.
  The closed-form dipole limit of the spec (section 3.6) checks this oracle;
* the absolute scale comes from the first Born approximation (spec sections
  3.5, 7.6 and 7.7), with the beam kinematics from the textbook relativistic
  formulas.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
import torch
from scipy import integrate
from scipy.special import lpmv, sph_harm_y, spherical_jn

from scatterem.physics.periodic_sampling import PeriodicGrid, dft_sample_frequencies
from scatterem.simulation.eels.physics import transition_potentials as tpmod
from scatterem.simulation.eels.physics._angular import spherical_harmonic, wigner_3j
from scatterem.simulation.eels.physics._special_torch import (
    interp_torch as _interp_torch,
)
from scatterem.simulation.eels.physics._special_torch import (
    spherical_harmonic_torch as _spherical_harmonic_torch,
)
from scatterem.simulation.eels.physics.radial import (
    free_continuum_wavefunction,
    hydrogenic_bound_wavefunction,
)
from scatterem.simulation.eels.physics.transition_potentials import (
    Transition,
    TransitionPotentials,
    _assoc_legendre_torch,
    _build_transition_potentials_gpu,
    _form_factor,
    _gaunt,
    _multipoles,
    _radial_overlap,
    _radial_overlaps_torch,
    _resolve_build_device,
    _spherical_bessel_torch,
    build_transition_potentials,
    subshell_transitions,
)

_EV = 100_000.0
_CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Fixtures: analytic wavefunctions, so nothing touches GPAW or the disk cache
# --------------------------------------------------------------------------- #
def _bound(Z=8, n=1, ell=0):
    return hydrogenic_bound_wavefunction(Z, n, ell)


def _excited(ell=1, eps=1.0):
    return free_continuum_wavefunction(ell, eps)


def _transition(*, lb=0, lp=1, ml=0, mlprime=0, Z=8, n=1, eps=1.0) -> Transition:
    return Transition(
        bound=_bound(Z, n, lb), excited=_excited(lp, eps), ml=ml, mlprime=mlprime
    )


# --------------------------------------------------------------------------- #
# Transition
# --------------------------------------------------------------------------- #
def test_energy_loss_is_the_orbital_energy_difference():
    tr = _transition(eps=7.5)
    assert tr.energy_loss == pytest.approx(tr.excited.energy - tr.bound.energy)
    # A bound state has negative energy, so the loss exceeds the continuum energy
    # by the binding energy: this is the edge onset the simulator reports.
    assert tr.energy_loss > 7.5
    assert tr.energy_loss - 7.5 == pytest.approx(-tr.bound.energy)


# --------------------------------------------------------------------------- #
# _spherical_bessel_torch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lmax", [0, 1, 4])
def test_spherical_bessel_ladder_matches_scipy_on_both_branches(lmax):
    """Small-x power series and large-x recurrence, against scipy."""
    x = torch.tensor(
        [0.0, 1e-8, 0.25, 1.0, 1.999, 2.0, 3.7, 25.0, 400.0], dtype=torch.float64
    )
    got = _spherical_bessel_torch(lmax, x)
    assert got.shape == (lmax + 1, x.numel())
    assert got.dtype == torch.float64
    for ell in range(lmax + 1):
        np.testing.assert_allclose(
            got[ell].numpy(), spherical_jn(ell, x.numpy()), rtol=1e-11, atol=1e-14
        )


def test_spherical_bessel_ladder_keeps_the_input_shape():
    x = torch.rand(3, 4, dtype=torch.float64) * 6.0
    got = _spherical_bessel_torch(2, x)
    assert got.shape == (3, 3, 4)
    np.testing.assert_allclose(
        got[2].numpy(), spherical_jn(2, x.numpy()), rtol=1e-11, atol=1e-14
    )


def test_spherical_bessel_skips_the_series_when_no_argument_is_small():
    """The ``anysmall`` guard: an all-large argument never runs the series."""
    x = torch.linspace(2.0, 40.0, 17, dtype=torch.float64)
    got = _spherical_bessel_torch(3, x)
    for ell in range(4):
        np.testing.assert_allclose(
            got[ell].numpy(), spherical_jn(ell, x.numpy()), rtol=1e-11, atol=1e-14
        )


def test_spherical_bessel_small_x_limit_is_the_double_factorial_form():
    """j_l(x) -> x^l / (2l+1)!! as x -> 0, where the recurrence is unusable."""
    x = torch.tensor([1e-3], dtype=torch.float64)
    got = _spherical_bessel_torch(3, x)
    dblfac = 1.0
    for ell in range(4):
        if ell:
            dblfac *= 2 * ell + 1
        assert float(got[ell, 0]) == pytest.approx(1e-3**ell / dblfac, rel=1e-6)


def test_spherical_bessel_threshold_is_only_a_branch_not_a_value_change():
    """Moving ``x_thr`` moves which branch each element takes, not the answer."""
    x = torch.linspace(0.05, 6.0, 41, dtype=torch.float64)
    low = _spherical_bessel_torch(3, x, x_thr=0.1)
    high = _spherical_bessel_torch(3, x, x_thr=6.5, n_series=40)
    torch.testing.assert_close(low, high, rtol=1e-8, atol=1e-12)
    # ...and both are still scipy's answer, so this is agreement on the value
    # rather than two branches agreeing on the same mistake.
    for ell in range(4):
        np.testing.assert_allclose(
            high[ell].numpy(), spherical_jn(ell, x.numpy()), rtol=1e-8, atol=1e-12
        )


# --------------------------------------------------------------------------- #
# _assoc_legendre_torch  /  _spherical_harmonic_torch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ell,am",
    [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)],
)
def test_assoc_legendre_matches_scipy_with_the_condon_shortley_phase(ell, am):
    theta = torch.tensor(
        [0.0, 0.4, 1.2, math.pi / 2, 2.5, math.pi], dtype=torch.float64
    )
    ct, st = torch.cos(theta), torch.sin(theta)
    got = _assoc_legendre_torch(ell, am, ct, st)
    np.testing.assert_allclose(got.numpy(), lpmv(am, ell, ct.numpy()), atol=1e-12)


def test_assoc_legendre_is_exact_at_the_poles():
    """The (cosθ, sinθ) spelling must not lose the pole values to a sqrt(1-x^2)."""
    ct = torch.tensor([1.0, -1.0], dtype=torch.float64)
    st = torch.zeros(2, dtype=torch.float64)
    assert torch.equal(
        _assoc_legendre_torch(2, 1, ct, st), torch.zeros(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        _assoc_legendre_torch(3, 0, ct, st),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
    )


def test_assoc_legendre_recurrence_agrees_with_the_closed_forms():
    """The l > 3 recurrence must be the analytic continuation of the table.

    ``_assoc_legendre_torch`` keeps hand-written closed forms for l <= 3 and
    switches to the standard recurrences beyond. Running the recurrence *down
    into* the tabulated range and comparing against the closed forms is what
    shows the two halves are the same function -- if they used different
    conventions (Condon-Shortley phase, or sin vs sqrt(1-x^2)), a d-shell edge
    would return finite but wrong numbers instead of raising.
    """
    lpmv = pytest.importorskip("scipy.special").lpmv
    theta = torch.linspace(0.0, float(np.pi), 13, dtype=torch.float64)
    ct, st = torch.cos(theta), torch.sin(theta)
    for ell in range(4):
        for am in range(ell + 1):
            np.testing.assert_allclose(
                _assoc_legendre_torch(ell, am, ct, st).numpy(),
                lpmv(am, ell, ct.numpy()),
                rtol=0,
                atol=1e-12,
                err_msg=f"closed form l={ell} |m|={am} disagrees with scipy",
            )


@pytest.mark.parametrize(
    "m,ell", [(0, 0), (0, 1), (1, 1), (-1, 1), (2, 2), (-2, 2), (3, 3), (-3, 3)]
)
def test_spherical_harmonic_torch_matches_scipy(m, ell):
    phi = torch.tensor([0.0, 0.7, 2.1, 4.9], dtype=torch.float64)
    theta = torch.tensor([0.3, 1.1, 2.0, 2.9], dtype=torch.float64)
    got = _spherical_harmonic_torch(m, ell, phi, theta)
    assert got.dtype == torch.complex128
    np.testing.assert_allclose(
        got.numpy(),
        spherical_harmonic(m, ell, phi.numpy(), theta.numpy()),
        rtol=1e-12,
        atol=1e-14,
    )


def test_spherical_harmonic_cache_is_bit_identical_to_the_uncached_call():
    phi = torch.tensor([0.2, 1.3, 3.0], dtype=torch.float64)
    theta = torch.tensor([0.5, 1.9, 2.6], dtype=torch.float64)
    cache: dict = {}
    for m, ell in ((0, 1), (1, 1), (-1, 1), (1, 2), (2, 2)):
        cached = _spherical_harmonic_torch(m, ell, phi, theta, cache=cache)
        plain = _spherical_harmonic_torch(m, ell, phi, theta)
        assert torch.equal(cached, plain)
    # cos/sin(theta) built once, one exp(i|m|phi) per DISTINCT |m| (0, 1, 2 here).
    assert {"ct", "st", "phi_c"} <= set(cache)
    assert sorted(k[1] for k in cache if isinstance(k, tuple)) == [0, 1, 2]
    # A repeat of an already-cached (|m|) reuses the same tensor object.
    first = cache[("e", 1)]
    _spherical_harmonic_torch(-1, 2, phi, theta, cache=cache)
    assert cache[("e", 1)] is first


# --------------------------------------------------------------------------- #
# _interp_torch
# --------------------------------------------------------------------------- #
def test_interp_torch_reproduces_numpy_interp_including_the_clamped_ends():
    xp = torch.linspace(0.0, 4.0, 9, dtype=torch.float64)
    fp = torch.tensor(
        [0.0, 1.0, -2.0, 3.5, 0.25, -1.0, 2.0, 0.5, -0.75], dtype=torch.float64
    )
    xq = torch.tensor(
        [-5.0, 0.0, 0.125, 1.0, 2.3333, 3.99, 4.0, 9.0], dtype=torch.float64
    )
    got = _interp_torch(xq, xp, fp)
    np.testing.assert_allclose(
        got.numpy(), np.interp(xq.numpy(), xp.numpy(), fp.numpy()), rtol=0, atol=1e-14
    )


def test_interp_torch_is_exact_on_the_sample_points_and_keeps_the_query_shape():
    xp = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    fp = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float64)
    torch.testing.assert_close(_interp_torch(xp, xp, fp), fp)
    xq = torch.rand(3, 2, dtype=torch.float64)
    assert _interp_torch(xq, xp, fp).shape == (3, 2)


def test_interp_torch_carries_a_complex_sample_dtype():
    xp = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    fp = torch.tensor([0.0 + 0j, 1.0 + 2j, 2.0 - 1j], dtype=torch.complex128)
    got = _interp_torch(torch.tensor([0.25], dtype=torch.float64), xp, fp)
    assert got.dtype == torch.complex128
    torch.testing.assert_close(got, torch.tensor([0.5 + 1j], dtype=torch.complex128))


# --------------------------------------------------------------------------- #
# _radial_overlaps_torch  vs  the numpy _radial_overlap
# --------------------------------------------------------------------------- #
def test_batched_radial_overlaps_match_the_numpy_reference():
    tr = _transition(lb=1, lp=1, Z=24, n=2, eps=10.0)
    q1d_np = np.linspace(0.0, 2.0, 16)
    q1d = torch.as_tensor(q1d_np, dtype=torch.float64)
    lpps = [0, 1, 2]
    got = _radial_overlaps_torch(lpps, tr.bound, tr.excited, q1d)
    assert got.shape == (3, 16) and got.dtype == torch.float64
    for k, lpp in enumerate(lpps):
        ref = _radial_overlap(lpp, tr.bound, tr.excited, q1d_np)
        np.testing.assert_allclose(got[k].numpy(), ref, rtol=1e-10, atol=1e-14)


def test_batched_radial_overlaps_accept_a_non_contiguous_level_set():
    """A gapped ``lpps`` must gather rather than slice, and give the same rows."""
    tr = _transition(lb=1, lp=1, Z=24, n=2, eps=10.0)
    q1d = torch.linspace(0.0, 2.0, 12, dtype=torch.float64)
    dense = _radial_overlaps_torch([0, 1, 2], tr.bound, tr.excited, q1d)
    gapped = _radial_overlaps_torch([0, 2], tr.bound, tr.excited, q1d)
    assert gapped.shape == (2, 12)
    torch.testing.assert_close(gapped[0], dense[0])
    torch.testing.assert_close(gapped[1], dense[2])


def test_the_bessel_ladder_memo_is_built_once_and_reused(monkeypatch):
    tr = _transition(lb=1, lp=1, Z=24, n=2, eps=10.0)
    q1d = torch.linspace(0.0, 2.0, 12, dtype=torch.float64)
    calls = []
    real = tpmod._spherical_bessel_torch

    def counted(lmax, x, *a, **kw):
        calls.append(int(lmax))
        return real(lmax, x, *a, **kw)

    monkeypatch.setattr(tpmod, "_spherical_bessel_torch", counted)
    ladder = {"lmax": 3}
    first = _radial_overlaps_torch([0, 1, 2], tr.bound, tr.excited, q1d, ladder=ladder)
    assert calls == [3]  # built to the DEEPEST level any sharing group needs
    assert "jall" in ladder
    jall = ladder["jall"]
    second = _radial_overlaps_torch([0, 1, 2], tr.bound, tr.excited, q1d, ladder=ladder)
    assert calls == [3]  # the second call did not rebuild it
    assert ladder["jall"] is jall
    assert torch.equal(first, second)
    # A deeper ladder sliced back to level l is bitwise the shallow build.
    monkeypatch.setattr(tpmod, "_spherical_bessel_torch", real)
    shallow = _radial_overlaps_torch([0, 1, 2], tr.bound, tr.excited, q1d)
    assert torch.equal(shallow, first)


# --------------------------------------------------------------------------- #
# _resolve_build_device -- every dispatch arm
# --------------------------------------------------------------------------- #
def test_resolve_build_device_cpu_is_forced_regardless_of_the_output_device():
    assert _resolve_build_device("cpu", None) == (_CPU, False)
    assert _resolve_build_device("cpu", "cuda") == (_CPU, False)


def test_resolve_build_device_auto_vectorises_on_cpu_without_cuda(monkeypatch):
    """Without CUDA, ``auto`` still takes the vectorised builder, on the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_build_device("auto", None) == (_CPU, True)
    assert _resolve_build_device("auto", "cpu") == (_CPU, True)


def test_resolve_build_device_honours_a_cuda_output_device_without_probing(monkeypatch):
    """A CUDA output device selects the vectorised build directly."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_build_device("auto", "cuda") == (torch.device("cuda"), True)
    assert _resolve_build_device("gpu", "cuda:0") == (torch.device("cuda:0"), True)


def test_resolve_build_device_auto_and_gpu_take_a_present_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_build_device("auto", None) == (torch.device("cuda"), True)
    assert _resolve_build_device("gpu", None) == (torch.device("cuda"), True)
    assert _resolve_build_device("cpu", "cpu") == (_CPU, False)


def test_resolve_build_device_keeps_an_explicit_cpu_device_on_the_cpu(monkeypatch):
    """``device="cpu"`` builds on the CPU, also with CUDA present.

    ``method="auto"`` still takes the vectorised builder there; only
    ``device=None`` lets ``auto`` pick a present CUDA device.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_build_device("auto", "cpu") == (_CPU, True)
    assert _resolve_build_device("auto", torch.device("cpu")) == (_CPU, True)


def test_build_with_an_explicit_cpu_device_never_touches_cuda(monkeypatch):
    """The whole build runs on the CPU when CUDA looks present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = []
    real = tpmod._build_transition_potentials_gpu

    def spy(transitions, fa, fb, eV, sampling, device):
        seen.append(torch.device(device))
        return real(transitions, fa, fb, eV, sampling, device)

    monkeypatch.setattr(tpmod, "_build_transition_potentials_gpu", spy)
    transitions = subshell_transitions(8, 1, 0, 2.0, wavefunctions="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, (16, 16), (0.25, 0.25), _EV, device="cpu"
    )
    assert seen == [_CPU]
    assert tp.array.device == _CPU


def test_resolve_build_device_gpu_without_cuda_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="method='gpu' requested but CUDA"):
        _resolve_build_device("gpu", None)
    with pytest.raises(RuntimeError, match="method='gpu' requested but CUDA"):
        _resolve_build_device("gpu", "cpu")


# --------------------------------------------------------------------------- #
# _build_transition_potentials_gpu, driven on the CPU
# --------------------------------------------------------------------------- #
def _gpu_build(transitions, gpts=(16, 16), sampling=(0.25, 0.25), eV=_EV):
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    fa, fb = np.broadcast_arrays(
        *dft_sample_frequencies(PeriodicGrid(tuple(gpts), tuple(gridsize)))
    )
    return _build_transition_potentials_gpu(
        transitions, torch.as_tensor(fa), torch.as_tensor(fb), eV, sampling, _CPU
    )


def _cpu_build(transitions, gpts, sampling, eV=_EV):
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    fa, fb = np.broadcast_arrays(
        *dft_sample_frequencies(PeriodicGrid(tuple(gpts), tuple(gridsize)))
    )
    return tpmod._build_cpu(transitions, fa, fb, eV, sampling)


def test_vectorised_build_makes_one_bessel_table_per_radial_pair(monkeypatch):
    """One Bessel table per (bound, excited) pair, built to its deepest lambda + 1.

    The extra order gives the slope ``d I_lambda / dK`` of the Hermite lookup.
    """
    calls = []
    real = tpmod._spherical_bessel_torch

    def counted(lmax, x, *a, **kw):
        calls.append(int(lmax))
        return real(lmax, x, *a, **kw)

    monkeypatch.setattr(tpmod, "_spherical_bessel_torch", counted)
    shared_bound = _bound(8, 1, 0)
    e0, e1 = _excited(0, 1.0), _excited(1, 1.0)
    transitions = [
        Transition(bound=shared_bound, excited=e0, ml=0, mlprime=0),
        Transition(bound=shared_bound, excited=e1, ml=0, mlprime=-1),
        Transition(bound=shared_bound, excited=e1, ml=0, mlprime=0),
        Transition(bound=shared_bound, excited=e1, ml=0, mlprime=1),
    ]
    _gpu_build(transitions, (8, 8), (0.5, 0.5))
    assert calls == [1, 2]  # lambda = 0 for s -> s, lambda = 1 for s -> p, plus 1

    calls.clear()
    _gpu_build([transitions[1]], (8, 8), (0.5, 0.5))
    assert calls == [2]


def test_vectorised_build_handles_different_losses_per_pair():
    """Pairs at different energies have different q_z, so different potentials."""
    shared_bound = _bound(8, 1, 0)
    transitions = [
        Transition(bound=shared_bound, excited=_excited(1, 1.0), ml=0, mlprime=0),
        Transition(bound=shared_bound, excited=_excited(1, 9.0), ml=0, mlprime=0),
    ]
    recip, losses = _gpu_build(transitions, (8, 8), (0.5, 0.5))
    assert losses[0] != losses[1]
    assert torch.isfinite(recip).all()
    assert not torch.allclose(recip[0], recip[1])


def test_a_channel_with_no_surviving_partial_wave_is_exactly_zero():
    """No lambda satisfies the selection rules, so the row is exactly zero."""
    # l = l' = 0 admits only lambda = 0, whose single mu is 0; ml' - ml = -1 is
    # outside that, so no term survives.
    dead = Transition(bound=_bound(8, 1, 0), excited=_excited(0, 1.0), ml=1, mlprime=0)
    live = Transition(bound=_bound(8, 1, 0), excited=_excited(0, 1.0), ml=0, mlprime=0)
    recip, _ = _gpu_build([dead, live], (8, 8), (0.5, 0.5))
    assert torch.equal(recip[0], torch.zeros_like(recip[0]))
    assert float(recip[1].abs().max()) > 0.0
    cpu = _cpu_build([dead, live], (8, 8), (0.5, 0.5))
    assert np.array_equal(cpu[0], np.zeros_like(cpu[0]))


@pytest.mark.parametrize("ell,am", [(4, 0), (4, 2), (4, 4), (5, 1), (5, 5), (6, 3)])
def test_associated_legendre_continues_past_the_tabulated_range(ell, am):
    """The l > 3 recurrence must reproduce scipy's ``lpmv`` (Condon-Shortley).

    Checked in float64 across the full polar range including both poles, where
    the (cos, sin) parametrisation is what keeps the closed forms exact.
    """
    lpmv = pytest.importorskip("scipy.special").lpmv
    theta = torch.linspace(0.0, float(np.pi), 17, dtype=torch.float64)
    got = _assoc_legendre_torch(ell, am, torch.cos(theta), torch.sin(theta))
    expected = lpmv(am, ell, torch.cos(theta).numpy())
    np.testing.assert_allclose(got.numpy(), expected, rtol=0, atol=1e-11)


# --------------------------------------------------------------------------- #
# build_transition_potentials: the two backends, end to end
# --------------------------------------------------------------------------- #
def _force_vectorised(monkeypatch):
    monkeypatch.setattr(
        tpmod, "_resolve_build_device", lambda method, device: (_CPU, True)
    )


def test_the_two_backends_agree_end_to_end(monkeypatch):
    gpts, sampling = (16, 16), (0.25, 0.25)
    transitions = [_transition(lb=0, lp=1, ml=0, mlprime=m) for m in (-1, 0, 1)]
    reference = build_transition_potentials(
        transitions, 8, gpts, sampling, _EV, dtype=torch.complex128, method="cpu"
    )
    _force_vectorised(monkeypatch)
    vectorised = build_transition_potentials(
        transitions, 8, gpts, sampling, _EV, dtype=torch.complex128
    )
    assert vectorised.array.shape == reference.array.shape
    scale = float(reference.array.abs().max())
    assert scale > 0.0
    torch.testing.assert_close(
        vectorised.array, reference.array, rtol=1e-9, atol=1e-10 * scale
    )
    np.testing.assert_allclose(vectorised.energy_losses, reference.energy_losses)


def test_build_returns_a_cpu_tensor_when_no_device_is_requested():
    tp = build_transition_potentials(
        [_transition()], 8, (8, 8), (0.5, 0.5), _EV, method="cpu"
    )
    assert tp.array.device.type == "cpu"
    assert tp.array.dtype == torch.complex64  # the default output dtype
    assert tp.Z == 8 and tp.energy == pytest.approx(_EV)
    assert tp.sampling == (0.5, 0.5)
    assert tp.gpts == (8, 8)
    assert len(tp) == 1


def test_the_bandlimit_is_applied_in_reciprocal_space():
    """A tighter limit must strictly remove reciprocal-space content."""
    transitions = [_transition()]
    kw = dict(method="cpu", dtype=torch.complex128)
    wide = build_transition_potentials(
        transitions, 8, (16, 16), (0.25, 0.25), _EV, bandlimit=1.0, **kw
    )
    tight = build_transition_potentials(
        transitions, 8, (16, 16), (0.25, 0.25), _EV, bandlimit=1.0 / 3.0, **kw
    )
    wq = torch.fft.fft2(wide.array, dim=(-2, -1))
    tq = torch.fft.fft2(tight.array, dim=(-2, -1))
    peak = float(wq.abs().max())
    assert peak > 0.0
    assert float(tq.abs().sum()) < float(wq.abs().sum())
    # The tight mask really zeroes a substantial part of the band the wide one
    # keeps (a round-trip through ifft2/fft2 leaves ~1e-18 noise, not exact 0).
    killed = tq.abs() <= 1e-10 * peak
    assert float(wq[killed].abs().max()) > 1e-3 * peak
    # ...and everything it does keep is byte-for-byte what the wide limit kept.
    kept = tq.abs() > 1e-8 * peak
    torch.testing.assert_close(tq[kept], wq[kept], rtol=1e-7, atol=1e-9 * peak)


def test_transition_potentials_to_moves_device_and_dtype_in_place():
    tp = build_transition_potentials(
        [_transition()], 8, (8, 8), (0.5, 0.5), _EV, method="cpu"
    )
    same = tp.to(dtype=torch.complex128)
    assert same is tp  # ``to`` mutates and returns self
    assert tp.array.dtype == torch.complex128
    back = tp.to(device="cpu", dtype=torch.complex64)
    assert back.array.dtype == torch.complex64 and back.array.device.type == "cpu"


def test_transition_potentials_defaults_to_an_empty_loss_table():
    arr = torch.zeros(2, 4, 4, dtype=torch.complex64)
    tp = TransitionPotentials(array=arr, Z=8, energy=_EV, sampling=(0.1, 0.1))
    assert tp.energy_losses.shape == (0,)
    assert len(tp) == 2 and tp.gpts == (4, 4)


# --------------------------------------------------------------------------- #
# Independent oracles for the form factor and the absolute scale
# --------------------------------------------------------------------------- #
# CODATA 2018 values, written out here so that the oracle does not take them
# from the module under test.
_EH_EV = 27.211386245988  # Hartree energy [eV]
_A0_ANG = 0.529177210903  # Bohr radius [Angstrom]
_RY_EV = _EH_EV / 2.0  # Rydberg energy [eV]
_HC_EV_ANG = 12398.419843320026  # h c [eV Angstrom]
_MC2_EV = 510998.95000  # electron rest energy [eV]


def _poly_mul(a, b):
    """Product of two polynomials (ascending exact coefficients)."""
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
    return tuple(out)


@cache
def _legendre_derivative(ell, k):
    """Exact coefficients of ``d^k P_ell / dx^k`` from Rodrigues' formula.

    ``P_ell = (1 / (2^ell ell!)) d^ell/dx^ell (x^2 - 1)^ell``.
    """
    coeffs = [Fraction(0)] * (2 * ell + 1)
    for j in range(ell + 1):
        coeffs[2 * j] = Fraction(math.comb(ell, j) * (-1) ** (ell - j))
    for _ in range(ell + k):
        coeffs = [coeffs[i] * i for i in range(1, len(coeffs))] or [Fraction(0)]
    scale = Fraction(1, 2**ell * math.factorial(ell))
    return tuple(c * scale for c in coeffs)


def _one_minus_x2(power):
    """Exact coefficients of ``(1 - x^2)^power``."""
    coeffs = [Fraction(0)] * (2 * power + 1)
    for j in range(power + 1):
        coeffs[2 * j] = Fraction(math.comb(power, j) * (-1) ** j)
    return tuple(coeffs)


@cache
def _gaunt_exact(lp, mlp, lam, mu, ell, ml):
    """``int Y*_(l'm') Y_(lambda mu) Y_(l m) dOmega``, integrated exactly.

    ``Y_lm = N_lm P_l^m(cos theta) e^(i m phi)`` with the Condon-Shortley phase
    in ``P_l^m = (-1)^m (1 - x^2)^(m/2) d^m P_l / dx^m`` for ``m >= 0``, and
    ``Y_(l,-m) = (-1)^m conj(Y_lm)``. The azimuthal integral is ``2 pi`` if
    ``m' = mu + m`` and 0 otherwise. The polar integral is a polynomial in
    ``x = cos theta`` (the three ``|m|`` add to an even number), so this
    function integrates it exactly with rational arithmetic. A selection-rule
    zero is therefore an exact 0.0.
    """
    states = ((lp, mlp), (lam, mu), (ell, ml))
    if mlp != mu + ml or any(abs(m) > lv for lv, m in states):
        return 0.0
    poly = (Fraction(1),)
    norm2 = Fraction(1)
    sign = 1
    for lv, m in states:
        am = abs(m)
        poly = _poly_mul(poly, _legendre_derivative(lv, am))
        norm2 *= Fraction(
            (2 * lv + 1) * math.factorial(lv - am), math.factorial(lv + am)
        )
        if m > 0:  # Condon-Shortley phase; Y_(l,-m) = (-1)^m conj(Y_lm) undoes it
            sign *= (-1) ** m
    poly = _poly_mul(poly, _one_minus_x2(sum(abs(m) for _, m in states) // 2))
    polar = sum(Fraction(2, k + 1) * c for k, c in enumerate(poly) if k % 2 == 0)
    if polar == 0:
        return 0.0
    return (
        2.0 * math.pi * sign * math.sqrt(norm2) / (4.0 * math.pi) ** 1.5 * float(polar)
    )


def _oracle_partial_wave_sum(ell, lp, ml, mlp, orders, radial, kphi, ktheta):
    """``4 pi sum_lambda i^lambda I_lambda sum_mu conj(Y_lambda mu(K_hat)) G``.

    The sum runs over every ``mu`` of each order in ``orders``. Only the exact
    Gaunt integral decides which terms survive; no selection rule is coded
    here. ``radial`` maps each order to ``I_lambda`` on the points.
    """
    out = np.zeros(np.shape(kphi), dtype=np.complex128)
    for lam in orders:
        for mu in range(-lam, lam + 1):
            gaunt = _gaunt_exact(lp, mlp, lam, mu, ell, ml)
            ylm = sph_harm_y(lam, mu, ktheta, kphi)
            out = out + 4.0 * math.pi * (1j**lam) * gaunt * radial[lam] * np.conj(ylm)
    return out


def _u_bound_exact(Z, n, ell):
    """Textbook closed forms of the hydrogenic ``u = r R_nl`` (r in Bohr)."""
    if (n, ell) == (1, 0):
        return lambda r: 2.0 * Z**1.5 * r * np.exp(-Z * r)
    if (n, ell) == (2, 1):
        return lambda r: Z**2.5 / (2.0 * math.sqrt(6.0)) * r**2 * np.exp(-Z * r / 2.0)
    if (n, ell) == (3, 2):
        c = 4.0 / (81.0 * math.sqrt(30.0)) * Z**3.5
        return lambda r: c * r**3 * np.exp(-Z * r / 3.0)
    raise NotImplementedError((n, ell))


def _continuum_k(eps):
    """Continuum wavenumber in 1/Bohr for ``eps`` in eV (``k^2 = E`` in Ry)."""
    return math.sqrt(eps / _RY_EV)


def _u_free_exact(lp, eps):
    """Free continuum normalised per Rydberg: ``(pi k)^(-1/2) k r j_l'(k r)``."""
    k = _continuum_k(eps)
    amp = 1.0 / math.sqrt(math.pi * k)
    return lambda r: amp * k * r * spherical_jn(lp, k * r)


def _j_lambda_oracle(lam, x):
    """``j_lambda(x)``, or ``j_0(x) - 1`` for ``lambda = 0`` (spec section 2.4)."""
    if lam > 0:
        return spherical_jn(lam, x)
    if x < 1e-2:
        x2 = x * x
        return -x2 / 6.0 + x2 * x2 / 120.0 - x2**3 / 5040.0
    return float(spherical_jn(0, x)) - 1.0


def _radial_oracle(lam, Z, n, ell, lp, eps, kb):
    """``I_lambda(K)`` by adaptive quadrature, ``kb`` = |K| in 1/Bohr.

    For ``l' = l`` the continuum is first made orthogonal to the bound orbital
    (Gram-Schmidt, spec section 7.5).
    """
    ub = _u_bound_exact(Z, n, ell)
    uc = _u_free_exact(lp, eps)
    rmax = 80.0 * n / Z  # u_b ~ r^n exp(-Z r / n): far below 1e-20 here
    quad = dict(limit=800, epsabs=1e-16, epsrel=1e-11)
    if lp == ell:
        bc = integrate.quad(lambda r: ub(r) * uc(r), 0.0, rmax, **quad)[0]
        bb = integrate.quad(lambda r: ub(r) ** 2, 0.0, rmax, **quad)[0]

        def cont(r):
            return uc(r) - (bc / bb) * ub(r)
    else:
        cont = uc
    out = np.empty(np.size(kb))
    for i, k in enumerate(np.ravel(kb)):
        out[i] = integrate.quad(
            lambda r, k=k: cont(r) * _j_lambda_oracle(lam, k * r) * ub(r),
            0.0,
            rmax,
            **quad,
        )[0]
    return out.reshape(np.shape(kb))


def _beam_wavenumber(energy):
    """Relativistic ``k = 2 pi / lambda`` in 1/Angstrom, energy in eV."""
    return 2.0 * math.pi * math.sqrt(energy * (energy + 2.0 * _MC2_EV)) / _HC_EV_ANG


def _sigma_kirkland(energy):
    """Interaction parameter in rad/(V Angstrom) (Kirkland, Advanced Computing
    in Electron Microscopy): ``2 pi / (lambda E) (m c^2 + E) / (2 m c^2 + E)``."""
    wavelength = 2.0 * math.pi / _beam_wavenumber(energy)
    return (
        2.0
        * math.pi
        / (wavelength * energy)
        * (_MC2_EV + energy)
        / (2.0 * _MC2_EV + energy)
    )


def _oracle_recip(Z, n, ell, lp, ml, mlp, eps, gpts, sampling, energy=_EV):
    """Independent ``H~(Q) / (dy dx)`` on the FFT grid (spec 3.3, 3.5, 7.6, 7.7).

    ``H~ = sigma(E0) sqrt(2 k_0 / k_n) 4 pi (E_h a_0) epsilon_Ry(K)
    / (sqrt(Ry) |K|^2)``: ``e^2/(4 pi eps_0) = E_h a_0``, ``1/sqrt(Ry)``
    converts the per-Rydberg continuum to per eV, the 2 counts the two spin
    states of the closed orbital, and ``k_0/k_n`` is the flux factor.
    The momentum transfer of the FFT coefficient at frequency ``f`` is
    ``K = (-2 pi f_a, -2 pi f_b, k_0 - k_n)``.
    """
    loss = eps + Z**2 / (2.0 * n**2) * _EH_EV
    k0, kn = _beam_wavenumber(energy), _beam_wavenumber(energy - loss)
    qz = k0 - kn
    fa = np.fft.fftfreq(gpts[0], d=sampling[0])[:, None]
    fb = np.fft.fftfreq(gpts[1], d=sampling[1])[None, :]
    kx = -2.0 * math.pi * fa + 0.0 * fb
    ky = -2.0 * math.pi * fb + 0.0 * fa
    kperp = np.hypot(kx, ky)
    kabs = np.sqrt(kperp**2 + qz**2)
    ktheta = np.arctan2(kperp, qz)
    kphi = np.where(kperp > 0.0, np.arctan2(ky, kx), 0.0)
    orders = [
        lam
        for lam in range(abs(ell - lp), ell + lp + 1)
        if any(_gaunt_exact(lp, mlp, lam, mu, ell, ml) for mu in range(-lam, lam + 1))
    ]
    uniq, inverse = np.unique(kabs, return_inverse=True)
    radial = {
        lam: _radial_oracle(lam, Z, n, ell, lp, eps, uniq * _A0_ANG)[inverse].reshape(
            kabs.shape
        )
        for lam in orders
    }
    eps_ry = _oracle_partial_wave_sum(ell, lp, ml, mlp, orders, radial, kphi, ktheta)
    scale = (
        _sigma_kirkland(energy)
        * math.sqrt(2.0 * k0 / kn)
        * 4.0
        * math.pi
        * (_EH_EV * _A0_ANG)
        / math.sqrt(_RY_EV)
    )
    return scale * eps_ry / kabs**2 / (sampling[0] * sampling[1])


def _assert_close_to_oracle(got, ref, frac=1e-3):
    """``|got - ref| <= frac * max|ref|`` on every pixel."""
    got = got.numpy() if isinstance(got, torch.Tensor) else np.asarray(got)
    peak = float(np.abs(ref).max())
    assert peak > 0.0
    err = float(np.abs(got - ref).max())
    assert err <= frac * peak, f"max error {err:.3e} > {frac:g} x peak {peak:.3e}"


# --------------------------------------------------------------------------- #
# Checks of the oracles themselves (planted truths)
# --------------------------------------------------------------------------- #
def test_the_exact_gaunt_oracle_agrees_with_a_sphere_quadrature():
    """Exact Gaunt oracle and the module's ``_gaunt`` vs a quadrature, l <= 4.

    Gauss-Legendre in ``cos theta`` (8 nodes, exact to degree 15) times a
    uniform azimuth grid (16 nodes, exact for ``|m| < 16``) integrates every
    product of three harmonics up to l = 4 exactly (spec 5.3 item 6).
    """
    x, w = np.polynomial.legendre.leggauss(8)
    phi = 2.0 * math.pi * np.arange(16) / 16
    theta = np.arccos(x)[:, None] + 0.0 * phi[None, :]
    phig = phi[None, :] + 0.0 * x[:, None]
    weight = w[:, None] * (2.0 * math.pi / 16) + 0.0 * phig
    states = [(lv, m) for lv in range(5) for m in range(-lv, lv + 1)]
    ylm = {s: sph_harm_y(s[0], s[1], theta, phig) for s in states}
    assert _gaunt_exact(1, 0, 1, 0, 0, 0) == pytest.approx(1.0 / math.sqrt(4 * math.pi))
    for lp, mlp in states:
        for lam, mu in states:
            for ell, ml in states:
                quad = complex(
                    np.sum(
                        weight
                        * np.conj(ylm[(lp, mlp)])
                        * ylm[(lam, mu)]
                        * ylm[(ell, ml)]
                    )
                )
                exact = _gaunt_exact(lp, mlp, lam, mu, ell, ml)
                assert abs(quad - exact) < 1e-13, (lp, mlp, lam, mu, ell, ml)
                assert abs(_gaunt(lp, mlp, lam, mu, ell, ml) - exact) < 1e-13


def test_the_radial_oracle_reproduces_the_closed_form_dipole_limit():
    """1s -> eps p: ``I_1(K)/K -> D/3`` with ``D = 16 Z^(5/2) A k^2/(Z^2+k^2)^3``.

    ``A = (pi k)^(-1/2)`` (spec sections 2.5 and 3.6). This pins the bound and
    continuum closed forms and the quadrature of the oracle.
    """
    Z, eps, kb = 8, 1.0, 1e-3
    k = _continuum_k(eps)
    dipole = 16.0 * Z**2.5 * k**2 / math.sqrt(math.pi * k) / (Z**2 + k**2) ** 3
    got = float(_radial_oracle(1, Z, 1, 0, 1, eps, np.array([kb]))[0]) / kb
    assert got == pytest.approx(dipole / 3.0, rel=1e-5)


def test_the_kinematics_oracle_matches_the_tabulated_interaction_parameter():
    """Kirkland's sigma at 100 and 200 kV: 9.244e-4 and 7.288e-4 rad/(V A)."""
    assert _sigma_kirkland(100e3) == pytest.approx(9.244e-4, rel=2e-4)
    assert _sigma_kirkland(200e3) == pytest.approx(7.288e-4, rel=2e-4)


# --------------------------------------------------------------------------- #
# _form_factor: the m and parity selection rules
# --------------------------------------------------------------------------- #
def _q_points():
    """|K| in 1/Angstrom at one fixed direction (azimuth 0.7, polar 1.1)."""
    qabs = np.linspace(0.3, 8.0, 7)
    return qabs, np.full_like(qabs, 0.7), np.full_like(qabs, 1.1)


def test_form_factor_skips_channels_no_m_selection_rule_admits():
    """2p -> eps p, ml = 1, ml' = -1: only lambda = 2 can carry mu = -2."""
    tr = _transition(lb=1, lp=1, ml=1, mlprime=-1, Z=24, n=2, eps=10.0)
    qabs, qphi, qtheta = _q_points()
    full = _form_factor(tr, qabs, qphi, qtheta)
    assert np.all(np.isfinite(full))
    assert np.any(full != 0.0)

    triangle = [0, 1, 2]
    radial = {lam: _radial_overlap(lam, tr.bound, tr.excited, qabs) for lam in triangle}
    # The module's radial integral is itself checked here against the
    # quadrature oracle, so the angular comparison below rests on a checked
    # radial part.
    for lam in triangle:
        ref = _radial_oracle(lam, 24, 2, 1, 1, 10.0, qabs * _A0_ANG)
        np.testing.assert_allclose(
            radial[lam], ref, rtol=1e-5, atol=1e-6 * float(np.abs(ref).max())
        )

    top = _oracle_partial_wave_sum(1, 1, 1, -1, [2], radial, qphi, qtheta)
    peak = float(np.abs(full).max())
    np.testing.assert_allclose(full, top, rtol=1e-10, atol=1e-14 * peak)
    lower = _oracle_partial_wave_sum(1, 1, 1, -1, [0, 1], radial, qphi, qtheta)
    assert np.array_equal(lower, np.zeros_like(lower))


def test_form_factor_drops_parity_forbidden_partial_waves():
    """2p -> eps p, ml = ml' = 0: the odd order lambda = 1 has no parity."""
    tr = _transition(lb=1, lp=1, ml=0, mlprime=0, Z=24, n=2, eps=10.0)
    qabs, qphi, qtheta = _q_points()
    full = _form_factor(tr, qabs, qphi, qtheta)
    assert np.all(np.isfinite(full))
    assert np.any(full != 0.0)
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0

    radial = {
        lam: _radial_overlap(lam, tr.bound, tr.excited, qabs) for lam in (0, 1, 2)
    }
    even = _oracle_partial_wave_sum(1, 1, 0, 0, [0, 2], radial, qphi, qtheta)
    peak = float(np.abs(full).max())
    np.testing.assert_allclose(full, even, rtol=1e-10, atol=1e-14 * peak)
    odd = _oracle_partial_wave_sum(1, 1, 0, 0, [1], radial, qphi, qtheta)
    assert np.array_equal(odd, np.zeros_like(odd))


# --------------------------------------------------------------------------- #
# The two grid builders against the independent oracle
# --------------------------------------------------------------------------- #
def test_vectorised_build_reproduces_the_numpy_reference_form_factor():
    """O 1s -> eps s and eps p (ml' = -1, +1) at 1 eV, 12 x 12 at 1/3 A."""
    gpts, sampling = (12, 12), (1.0 / 3.0, 1.0 / 3.0)
    bound = _bound(8, 1, 0)
    cont_s, cont_p = _excited(0, 1.0), _excited(1, 1.0)
    transitions = [
        Transition(bound=bound, excited=cont_s, ml=0, mlprime=0),
        Transition(bound=bound, excited=cont_p, ml=0, mlprime=-1),
        Transition(bound=bound, excited=cont_p, ml=0, mlprime=1),
    ]
    recip, losses = _gpu_build(transitions, gpts, sampling)
    assert recip.shape == (3, 12, 12)
    assert recip.dtype == torch.complex128
    np.testing.assert_array_equal(losses, [tr.energy_loss for tr in transitions])

    cpu = _cpu_build(transitions, gpts, sampling)
    for i, (lp, mlp) in enumerate([(0, 0), (1, -1), (1, 1)]):
        peak = float(np.abs(cpu[i]).max())
        torch.testing.assert_close(
            recip[i], torch.from_numpy(cpu[i]), rtol=1e-9, atol=1e-11 * peak
        )
        ref = _oracle_recip(8, 1, 0, lp, 0, mlp, 1.0, gpts, sampling)
        _assert_close_to_oracle(recip[i], ref)
        _assert_close_to_oracle(cpu[i], ref)


def test_the_vectorised_build_drops_parity_forbidden_partial_waves():
    """2p -> eps p, ml = ml' = 0 at 10 eV, Z = 24, 8 x 8 at 0.5 A."""
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0
    gpts, sampling = (8, 8), (0.5, 0.5)
    tr = _transition(lb=1, lp=1, ml=0, mlprime=0, Z=24, n=2, eps=10.0)
    recip, _ = _gpu_build([tr], gpts, sampling)
    assert torch.isfinite(recip).all()
    assert float(recip.abs().max()) > 0.0
    assert _multipoles(1, 1, 0, 0) == [0, 2]

    cpu = _cpu_build([tr], gpts, sampling)
    peak = float(np.abs(cpu[0]).max())
    torch.testing.assert_close(
        recip[0], torch.from_numpy(cpu[0]), rtol=1e-9, atol=1e-11 * peak
    )
    ref = _oracle_recip(24, 2, 1, 1, 0, 0, 10.0, gpts, sampling)
    _assert_close_to_oracle(recip[0], ref)


def test_the_vectorised_build_does_a_d_shell_edge_and_matches_the_reference():
    """La 3d -> eps f at 1 eV: ml 0 -> 0 and ml 2 -> -2, 8 x 8 at 0.5 A."""
    gpts, sampling = (8, 8), (0.5, 0.5)
    bound, cont = _bound(57, 3, 2), _excited(3, 1.0)
    transitions = [
        Transition(bound=bound, excited=cont, ml=0, mlprime=0),
        Transition(bound=bound, excited=cont, ml=2, mlprime=-2),
    ]
    # mu = -4 admits only the top of the triangle [1, 5]. The harmonic then
    # needs P_5^4, above the closed forms (l <= 3) of the Legendre helper.
    assert _multipoles(2, 3, 2, -2) == [5]
    recip, _ = _gpu_build(transitions, gpts, sampling)
    assert torch.isfinite(recip).all()

    cpu = _cpu_build(transitions, gpts, sampling)
    for i, (ml, mlp) in enumerate([(0, 0), (2, -2)]):
        peak = float(np.abs(cpu[i]).max())
        torch.testing.assert_close(
            recip[i], torch.from_numpy(cpu[i]), rtol=1e-12, atol=1e-13 * peak
        )
        ref = _oracle_recip(57, 3, 2, 3, ml, mlp, 1.0, gpts, sampling)
        _assert_close_to_oracle(recip[i], ref)
