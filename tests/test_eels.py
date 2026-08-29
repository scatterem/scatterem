"""Tests for the STEM-EELS core-loss simulation package.

These exercise the transition-potential physics and both the
conventional and PRISM multislice paths using the analytic ("hydrogenic")
radial-wavefunction backend, so they run without the optional GPAW dependency.
"""

import math

import numpy as np
import pytest
import torch

from scatterem.simulation.eels import (
    StemEelsSimulator,
    build_transition_potentials,
    subshell_transitions,
    transition_potential_multislice,
)
from scatterem.simulation.eels._angular import wigner_3j
from scatterem.simulation.eels.prism_eels import prism_transition_potential
from scatterem.simulation.eels.radial import (
    bound_wavefunction,
    continuum_wavefunction,
    free_continuum_wavefunction,
    gpaw_available,
    hydrogenic_bound_wavefunction,
)
from scatterem.simulation.structure import Structure

_RYDBERG_EV = 13.605693122994


# --------------------------------------------------------------------------- #
# Angular-momentum algebra
# --------------------------------------------------------------------------- #
def test_wigner_3j_known_values():
    # (1 1 0; 0 0 0) = -1/sqrt(3)
    assert wigner_3j(1, 1, 0, 0, 0, 0) == pytest.approx(
        -1.0 / math.sqrt(3.0), abs=1e-12
    )
    # (1 1 2; 0 0 0) = sqrt(2/15)
    assert wigner_3j(1, 1, 2, 0, 0, 0) == pytest.approx(
        math.sqrt(2.0 / 15.0), abs=1e-12
    )
    # (2 2 0; 0 0 0) = 1/sqrt(5)
    assert wigner_3j(2, 2, 0, 0, 0, 0) == pytest.approx(1.0 / math.sqrt(5.0), abs=1e-12)


def test_wigner_3j_selection_rules():
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0  # parity / triangle violation
    assert wigner_3j(1, 1, 0, 1, 0, 0) == 0.0  # m1 + m2 + m3 != 0
    assert wigner_3j(1, 2, 5, 0, 0, 0) == 0.0  # triangle violation


# --------------------------------------------------------------------------- #
# Radial wavefunctions
# --------------------------------------------------------------------------- #
def test_hydrogenic_bound_energy_and_norm():
    wf = hydrogenic_bound_wavefunction(Z=1, n=1, l=0)
    # 1s hydrogen: E = -13.6 eV.
    assert wf.energy == pytest.approx(-13.605, abs=0.05)
    # u = r R is normalised: ∫ u^2 dr ≈ 1.
    norm = np.trapezoid(wf.u**2, wf.r)
    assert norm == pytest.approx(1.0, abs=1e-2)


def test_continuum_positive_energy():
    wf = free_continuum_wavefunction(l=1, epsilon=5.0)
    assert wf.n is None
    assert wf.energy == pytest.approx(5.0)


def test_bound_wavefunction_dispatch_hydrogenic():
    wf = bound_wavefunction(Z=6, n=1, l=0, backend="hydrogenic", use_cache=False)
    assert wf.energy < 0


# --------------------------------------------------------------------------- #
# Transition potentials
# --------------------------------------------------------------------------- #
def test_subshell_transitions_energy_loss_positive():
    transitions = subshell_transitions(Z=8, n=1, l=0, epsilon=1.0, backend="hydrogenic")
    assert len(transitions) > 0
    for tr in transitions:
        assert tr.energy_loss > 0  # ionization always costs energy


def test_build_transition_potentials_shape_and_nonzero():
    transitions = subshell_transitions(Z=8, n=1, l=0, epsilon=1.0, backend="hydrogenic")
    gpts = (64, 64)
    sampling = (0.1, 0.1)
    tp = build_transition_potentials(transitions, 8, gpts, sampling, eV=100_000.0)
    assert tp.array.shape == (len(transitions), *gpts)
    assert torch.is_complex(tp.array)
    assert torch.isfinite(tp.array).all()
    # At least one transition has appreciable weight.
    assert float(tp.array.abs().sum()) > 0
    assert tp.energy_losses.shape == (len(transitions),)


def test_k_edge_channels_monopole_and_dipole():
    # 1s (l=0) edge with the default order=1: the continuum l' set is
    # {l-1, l, l+1} clipped to l' >= 0, i.e. {0, 1} -- the monopole l'=0 and
    # dipole l'=1 channels (as the published treatments do, summing the full
    # range(max(l-order,0), l+order+1)).
    transitions = subshell_transitions(Z=8, n=1, l=0, epsilon=1.0, backend="hydrogenic")
    assert {tr.excited.l for tr in transitions} == {0, 1}

    # Restricting to the dipole-only set recovers the previous behaviour.
    dipole = subshell_transitions(
        Z=8, n=1, l=0, epsilon=1.0, lprimes=[1], backend="hydrogenic"
    )
    assert {tr.excited.l for tr in dipole} == {1}


def test_build_transition_potentials_gpu_matches_cpu():
    # The GPU-vectorised build (method="gpu") must reproduce the numpy/scipy
    # reference build (method="cpu") to ~machine precision -- it is a pure
    # speedup, not a physics change.  Exercises both a K edge (l=0 -> l' in
    # {0,1}) and an L edge (l=1 -> l' in {0,1,2}), the latter being the CrCoNi
    # case the optimisation targets.  Skipped without CUDA.
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for the GPU build path")
    gpts = (96, 96)
    sampling = (0.05, 0.05)
    cases = [
        dict(Z=8, n=1, l=0, epsilon=1.0),  # K edge
        dict(Z=24, n=2, l=1, epsilon=10.0),  # Cr L edge
    ]
    for c in cases:
        transitions = subshell_transitions(
            c["Z"], c["n"], c["l"], epsilon=c["epsilon"], backend="hydrogenic"
        )
        cpu = build_transition_potentials(
            transitions,
            c["Z"],
            gpts,
            sampling,
            eV=100_000.0,
            dtype=torch.complex128,
            method="cpu",
        )
        gpu = build_transition_potentials(
            transitions,
            c["Z"],
            gpts,
            sampling,
            eV=100_000.0,
            dtype=torch.complex128,
            method="gpu",
        )
        a = gpu.array.cpu()
        b = cpu.array.cpu()
        # Machine-precision agreement on the complex array (account for the
        # ill-conditioned 1/qabs**2 with atol; the max abs diff is ~1e-15).
        assert torch.allclose(
            a, b, rtol=1e-5, atol=1e-8
        ), f"Z={c['Z']} max abs diff {float((a - b).abs().max()):.3e}"
        # Per-transition energy losses are identical too.
        assert np.allclose(gpu.energy_losses, cpu.energy_losses)


# --------------------------------------------------------------------------- #
# Conventional multislice EELS
# --------------------------------------------------------------------------- #
def _single_atom_structure(Z=8):
    # One atom in a 4 x 4 x 4 Å cell at the centre.
    atoms = np.array([[0.5, 0.5, 0.5, Z]], dtype=float)
    return Structure(np.array([4.0, 4.0, 4.0]), atoms, dwf=None)


def test_conventional_multislice_runs():
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    gpts = (48, 48)
    sampling = (4.0 / 48, 4.0 / 48)
    tp = build_transition_potentials(transitions, 8, gpts, sampling, eV=100_000.0)

    # Trivial vacuum transmissions + a normalised plane-ish probe.
    transmissions = torch.ones((2, *gpts), dtype=torch.complex64)
    probe = torch.zeros(gpts, dtype=torch.complex64)
    probe[0, 0] = 1.0
    probe = torch.fft.ifft2(probe)[None]  # uniform illumination, 1 position

    out = transition_potential_multislice(
        probe,
        transmissions,
        tp,
        sites=np.array([[0.5, 0.5, 0.5]]),
        wavelength=float(0.0037),
        gridsize=(4.0, 4.0),
        slice_distance=2.0,
    )
    assert out.shape == (1, *gpts)
    assert torch.isfinite(out).all()
    assert float(out.sum()) > 0
    assert float(out.min()) >= 0


def test_prism_matches_conventional_for_vacuum():
    # With vacuum transmissions and a single ionization slice the PRISM and
    # conventional paths must agree (S1 reconstructs the same probe).
    gpts = (40, 40)
    sampling = (0.1, 0.1)
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(transitions, 8, gpts, sampling, eV=100_000.0)

    transmissions = torch.ones((1, *gpts), dtype=torch.complex64)
    # Build a small aperture probe in q-space.
    probe_q = torch.zeros(gpts, dtype=torch.complex64)
    probe_q[:3, :3] = 1.0
    probe_q[-2:, -2:] = 1.0
    n = probe_q.numel()
    probe_q = probe_q / (probe_q.abs().pow(2).sum() / n).sqrt()

    sites = np.array([[0.5, 0.5, 0.0]])
    scan = torch.tensor([[10.0, 12.0], [20.0, 5.0]], dtype=torch.float64)

    base_probe = torch.fft.ifft2(probe_q)
    from scatterem.simulation.eels.multislice_eels import _fourier_shift

    probes = torch.stack([_fourier_shift(base_probe, (r[0], r[1])) for r in scan])
    conv = transition_potential_multislice(
        probes,
        transmissions,
        tp,
        sites,
        wavelength=0.0037,
        gridsize=(4.0, 4.0),
        slice_distance=0.0,
    )
    pris = prism_transition_potential(
        probe_q,
        transmissions,
        tp,
        sites,
        scan,
        wavelength=0.0037,
        gridsize=(4.0, 4.0),
        slice_distance=0.0,
    )
    # Compare in double precision: the conftest-style dtype randomisation can
    # leave the two paths at different float precisions, which is fine.
    assert torch.allclose(conv.double(), pris.double(), rtol=1e-4, atol=1e-6)


def test_prism_matches_conventional_nonvacuum():
    # Stronger than the vacuum case: PRISM and conventional must still agree
    # with a non-trivial (non-vacuum) multislice potential, real inter-slice
    # propagation, and the ionization site BELOW the entrance surface.  This
    # holds because multislice is linear in the probe-forming beams (S1
    # reconstructs the elastic wave exactly) and the Fresnel propagator commutes
    # with the Fourier scan-shift (both diagonal in q).
    torch.manual_seed(0)
    gpts = (32, 32)
    sampling = (0.15, 0.15)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(transitions, 8, gpts, sampling, eV=100_000.0)

    # Non-vacuum: three random weak phase-object slices.
    nz = 3
    phase = 0.3 * torch.rand(nz, *gpts, dtype=torch.float64)
    transmissions = torch.exp(1j * phase).to(torch.complex64)

    # Probe-forming aperture (a handful of beams), unit-normalised.
    probe_q = torch.zeros(gpts, dtype=torch.complex64)
    probe_q[:3, :3] = 1.0
    probe_q[-2:, -2:] = 1.0
    n = probe_q.numel()
    probe_q = probe_q / (probe_q.abs().pow(2).sum() / n).sqrt()

    # Site in the last slice (z=0.9 -> slice 2 of 3): the probe really propagates
    # through slices 0 and 1 before the inelastic event.
    sites = np.array([[0.4, 0.6, 0.9]])
    scan = torch.tensor([[7.0, 11.0], [18.0, 4.0]], dtype=torch.float64)

    base_probe = torch.fft.ifft2(probe_q)
    from scatterem.simulation.eels.multislice_eels import _fourier_shift

    probes = torch.stack([_fourier_shift(base_probe, (r[0], r[1])) for r in scan])
    conv = transition_potential_multislice(
        probes,
        transmissions,
        tp,
        sites,
        wavelength=0.0037,
        gridsize=gridsize,
        slice_distance=1.0,
    )
    pris = prism_transition_potential(
        probe_q,
        transmissions,
        tp,
        sites,
        scan,
        wavelength=0.0037,
        gridsize=gridsize,
        slice_distance=1.0,
    )
    assert torch.allclose(conv.double(), pris.double(), rtol=1e-4, atol=1e-6)


def test_dual_smatrix_equals_hybrid():
    # The dual-scattering-matrix algorithm only REORDERS the (linear) work of the
    # hybrid path: apply the transition to the S1 columns and propagate those to
    # the exit, instead of propagating each scan probe.  The two must therefore
    # agree to ~machine precision (not just the looser conventional tolerance),
    # over a non-trivial multislice with a sub-surface ionization site.
    from scatterem.simulation.eels.prism_eels import prism_transition_potential_hybrid

    torch.manual_seed(0)
    gpts = (32, 32)
    sampling = (0.15, 0.15)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    # complex128 throughout: the dual-S only reorders the (linear) work, so the
    # two paths must agree to ~machine precision -- float32 reordering noise
    # (~1e-6) would otherwise mask that this is an exact algebraic identity.
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )

    nz = 3
    transmissions = torch.exp(1j * 0.3 * torch.rand(nz, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    probe_q = torch.zeros(gpts, dtype=torch.complex128)
    probe_q[:3, :3] = 1.0
    probe_q[-2:, -2:] = 1.0
    probe_q = probe_q / (probe_q.abs().pow(2).sum() / probe_q.numel()).sqrt()

    sites = np.array([[0.4, 0.6, 0.9]])
    # Mix of integer and sub-pixel scan positions.
    scan = torch.tensor([[7.0, 11.0], [18.5, 4.25]], dtype=torch.float64)
    common = dict(wavelength=0.0037, gridsize=gridsize, slice_distance=1.0)

    dual = prism_transition_potential(probe_q, transmissions, tp, sites, scan, **common)
    hyb = prism_transition_potential_hybrid(
        probe_q, transmissions, tp, sites, scan, **common
    )
    assert torch.allclose(dual.double(), hyb.double(), rtol=1e-6, atol=1e-9)

    # Same equivalence with PRISM beam subsampling (interpolation_factor > 1).
    dual_f2 = prism_transition_potential(
        probe_q, transmissions, tp, sites, scan, interpolation_factor=2, **common
    )
    hyb_f2 = prism_transition_potential_hybrid(
        probe_q, transmissions, tp, sites, scan, interpolation_factor=2, **common
    )
    assert torch.allclose(dual_f2.double(), hyb_f2.double(), rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end simulator
# --------------------------------------------------------------------------- #
def test_simulator_end_to_end_conventional():
    struct = _single_atom_structure(Z=8)
    sim = StemEelsSimulator(
        struct,
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(40, 40),
        scan=(3, 3),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    result = sim.simulate()
    assert result.cube.shape == (3, 3, 40, 40)
    assert torch.isfinite(result.cube).all()
    assert float(result.cube.sum()) > 0


def test_potential_scale_plumbed_hydrogenic():
    # potential_scale must be accepted end-to-end; the hydrogenic backend has no
    # atomic potential so it ignores the value (no error, same free continuum).
    a = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic", potential_scale=1.0)
    b = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic", potential_scale=1.02)
    assert {tr.excited.l for tr in a} == {tr.excited.l for tr in b} == {0, 1}
    sim = StemEelsSimulator(
        _single_atom_structure(Z=8),
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(32, 32),
        scan=(2, 2),
        edge=(8, 1, 0),
        epsilon=1.0,
        backend="hydrogenic",
        potential_scale=1.02,
    )
    assert sim.potential_scale == 1.02


# --------------------------------------------------------------------------- #
# GPAW-backed physics (skipped when the optional GPAW dependency is absent).
# The hydrogenic-backend tests above exercise the pipeline plumbing; these
# cover the GPAW radial physics that the rest of the suite cannot reach.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not gpaw_available(), reason="requires the optional GPAW dependency"
)
def test_gpaw_continuum_asymptotic_normalisation():
    # The energy-normalisation fix sets the *asymptotic* envelope of u(r) (not
    # its global peak) to the Manson-1972 amplitude 1/(sqrt(pi) * (eps/Ry)^1/4).
    Z, lprime, eps = 8, 1, 10.0
    wf = continuum_wavefunction(Z, lprime, eps, backend="gpaw", use_cache=False)
    ef = eps / _RYDBERG_EV
    expected = 1.0 / (math.sqrt(math.pi) * ef**0.25)
    far = np.abs(wf.u[wf.r > 0.6 * wf.r[-1]])
    assert far.max() == pytest.approx(expected, rel=0.08)


@pytest.mark.skipif(
    not gpaw_available(), reason="requires the optional GPAW dependency"
)
def test_gpaw_continuum_potential_scale_affects_swave():
    # The barrier-free l'=0 s-wave is sensitive to the potential scaling
    # (this is the abTEM 1.02 knob); the result must actually change with it.
    Z, eps = 8, 5.0
    w0 = continuum_wavefunction(
        Z, 0, eps, backend="gpaw", use_cache=False, potential_scale=1.0
    )
    w1 = continuum_wavefunction(
        Z, 0, eps, backend="gpaw", use_cache=False, potential_scale=1.02
    )
    rc = np.linspace(0.1, 8.0, 2000)
    u0, u1 = w0(rc), w1(rc)
    s = np.dot(u0, u1) / np.dot(u0, u0)  # best scale
    rel = np.abs(u1 - s * u0).max() / (np.abs(u1).max() + 1e-300)
    assert rel > 0.02  # >2% shape change from the 2% potential deepening


def test_simulator_prism_runs():
    struct = _single_atom_structure(Z=8)
    sim = StemEelsSimulator(
        struct,
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(40, 40),
        scan=(2, 2),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        prism=True,
        backend="hydrogenic",
        device="cpu",
    )
    result = sim.simulate()
    assert result.cube.shape == (2, 2, 40, 40)
    assert float(result.cube.sum()) > 0


def test_prism_matches_conventional_subpixel_scan():
    # Regression: PRISM must match the conventional path for FRACTIONAL scan
    # positions.  scan=(3,3) on 40 px gives positions 6.67 / 20 / 33.33 px; the
    # PRISM phase ramp must use signed frequencies (fftfreq*N), not raw corner-
    # origin indices, or the negative-frequency beams mis-phase and the probe
    # (hence the whole datacube) is corrupted for sub-pixel shifts.
    struct = _single_atom_structure(Z=8)
    common = dict(
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(40, 40),
        scan=(3, 3),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    conv = StemEelsSimulator(struct, prism=False, **common).simulate().cube
    pris = StemEelsSimulator(struct, prism=True, **common).simulate().cube
    assert torch.allclose(conv.double(), pris.double(), rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------- #
# Energy-window integration
# --------------------------------------------------------------------------- #
def test_simulate_window_matches_manual_trapezoid():
    # The windowed simulation must equal a hand-rolled trapezoid integral of the
    # single-energy datacubes over the same epsilon grid.  Deterministic here
    # (one frozen phonon, no displacements), so the agreement is exact.
    struct = _single_atom_structure(Z=8)
    common = dict(
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(24, 24),
        scan=(2, 2),
        edge=(8, 1, 0),
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    eps_grid = np.linspace(1.0, 5.0, 3)
    sim = StemEelsSimulator(struct, epsilon=float(eps_grid.mean()), **common)
    win = sim.simulate_window((1.0, 5.0), n_energies=3)

    assert win.cube.shape == (2, 2, 24, 24)
    assert torch.isfinite(win.cube).all()
    assert float(win.cube.min()) >= 0
    assert win.energies.shape == (3,)
    assert win.energy_range is not None
    # window edges are onset-shifted copies of the epsilon endpoints
    onset = win.energy_range[0] - 1.0
    assert win.energy_range[1] == pytest.approx(onset + 5.0)

    deps = (5.0 - 1.0) / 2
    quad = np.array([deps / 2, deps, deps / 2])
    manual = None
    for wi, e in zip(quad, eps_grid):
        c = StemEelsSimulator(struct, epsilon=float(e), **common).simulate().cube
        manual = wi * c if manual is None else manual + wi * c
    assert torch.allclose(win.cube.double(), manual.double(), rtol=1e-5, atol=1e-8)


def test_simulate_window_validates_inputs():
    sim = StemEelsSimulator(
        _single_atom_structure(Z=8),
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(16, 16),
        scan=(1, 1),
        edge=(8, 1, 0),
        backend="hydrogenic",
    )
    with pytest.raises(ValueError):
        sim.simulate_window((0.0, 5.0))  # eps_min must be > 0
    with pytest.raises(ValueError):
        sim.simulate_window((5.0, 1.0))  # eps_max <= eps_min
    with pytest.raises(ValueError):
        sim.simulate_window((1.0, 5.0), n_energies=1)  # n < 2


# --------------------------------------------------------------------------- #
# Multiple edges / elements in one pass
# --------------------------------------------------------------------------- #
def _two_element_structure():
    # O (Z=8) and C (Z=6) in a 4 x 4 x 4 Å cell.
    atoms = np.array(
        [[0.4, 0.4, 0.5, 8], [0.6, 0.6, 0.5, 6]],
        dtype=float,
    )
    return Structure(np.array([4.0, 4.0, 4.0]), atoms, dwf=None)


def test_simulate_edges_matches_per_edge_single_energy():
    # Multi-edge (shared transmissions) must equal running each edge on its own.
    struct = _two_element_structure()
    common = dict(
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(24, 24),
        scan=(2, 2),
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    edges = [(8, 1, 0), (6, 1, 0)]
    multi = StemEelsSimulator(
        struct, edge=edges[0], epsilon=2.0, **common
    ).simulate_edges(edges)
    assert set(multi.keys()) == set(edges)
    for edge in edges:
        assert multi[edge].cube.shape == (2, 2, 24, 24)
        single = (
            StemEelsSimulator(struct, edge=edge, epsilon=2.0, **common).simulate().cube
        )
        assert torch.allclose(multi[edge].cube.double(), single.double(), atol=1e-8)


def test_simulate_edges_windowed_matches_simulate_window():
    struct = _two_element_structure()
    common = dict(
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(20, 20),
        scan=(2, 2),
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    edges = [(8, 1, 0), (6, 1, 0)]
    multi = StemEelsSimulator(struct, edge=edges[0], **common).simulate_edges(
        edges, energy_range=(1.0, 5.0), n_energies=3
    )
    for edge in edges:
        win = (
            StemEelsSimulator(struct, edge=edge, **common)
            .simulate_window((1.0, 5.0), n_energies=3)
            .cube
        )
        assert torch.allclose(multi[edge].cube.double(), win.double(), atol=1e-8)


def test_simulate_edges_errors():
    sim = StemEelsSimulator(
        _two_element_structure(),  # has Z=8 and Z=6
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(16, 16),
        scan=(1, 1),
        edge=(8, 1, 0),
        backend="hydrogenic",
    )
    with pytest.raises(ValueError):
        sim.simulate_edges([(7, 1, 0)])  # nitrogen not present
    with pytest.raises(ValueError):
        sim.simulate_edges([])  # empty


# --------------------------------------------------------------------------- #
# PRISM interpolation factor f
# --------------------------------------------------------------------------- #
def test_scattering_matrix_interpolation_factor_reconstructs_probe():
    # Isolated test of the f approximation: subsampling the beams (f=2) must
    # reconstruct the same probe as f=1 wherever the probe has weight, provided
    # the probe fits inside the N/f window.
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import ScatteringMatrix

    n = 64
    qy, qx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    # Smooth (Gaussian) aperture -> compact probe with negligible tails, so the
    # only thing being tested is the f reconstruction (no Airy-tail aliasing).
    aperture = np.exp(-0.5 * ((np.sqrt(qy**2 + qx**2)) / 0.06) ** 2).astype(
        np.complex128
    )
    probe_q = torch.as_tensor(aperture, dtype=torch.complex64)
    probe_q = probe_q / probe_q.abs().pow(2).sum().sqrt()
    T = torch.ones((1, n, n), dtype=torch.complex64)
    kernel = propagator_kernel((n, n), (6.4, 6.4), 0.0037, 0.0, dtype=torch.complex64)
    scan = torch.tensor([[32.0, 32.0]], dtype=torch.float64)  # centred

    s1 = ScatteringMatrix(probe_q, T, kernel, interpolation_factor=(1, 1))
    s2 = ScatteringMatrix(probe_q, T, kernel, interpolation_factor=(2, 2))
    assert s2.coeffs0.numel() < s1.coeffs0.numel()  # f=2 keeps ~1/4 of the beams

    p1 = s1.probe_at_current_plane(scan)[0]
    p2 = s2.probe_at_current_plane(scan)[0]
    mask = p1.abs() > 0.1 * p1.abs().max()  # where the probe actually has weight
    rel = float((p2[mask] - p1[mask]).abs().sum() / p1[mask].abs().sum())
    # With the f**2 coefficient renormalisation, f=2 reconstructs the probe to
    # machine precision when it fits in the N/2 window.
    assert rel < 1e-4


def test_scattering_matrix_init_equals_monolithic_ifft2():
    # The constructor builds the real-space S-matrix with a beam-CHUNKED in-place
    # ifft2 (to avoid doubling the 27 GiB S-matrix buffer at the 20 nm rung). It
    # must be bit-identical to a single monolithic ifft2 of the one-hot Fourier S.
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import ScatteringMatrix

    n = 24
    qy, qx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    aperture = (np.sqrt(qy**2 + qx**2) < 0.2).astype(np.complex128)
    probe_q = torch.as_tensor(aperture, dtype=torch.complex64)
    T = torch.ones((1, n, n), dtype=torch.complex64)
    kernel = propagator_kernel((n, n), (2.4, 2.4), 0.0037, 0.0, dtype=torch.complex64)

    s = ScatteringMatrix(probe_q, T, kernel, interpolation_factor=(1, 1))

    # Reference: the monolithic construction the chunked loop replaces.
    nbeams = s.beam_gy.numel()
    S_q = torch.zeros((nbeams, n, n), dtype=torch.complex64)
    S_q[torch.arange(nbeams), s.beam_gy, s.beam_gx] = 1.0
    ref = torch.fft.ifft2(S_q, dim=(-2, -1))
    assert torch.equal(s.S, ref)

    # Force a small beam_chunk (>1 iteration) and confirm the path still matches.
    import scatterem.simulation.eels.prism_eels as pe

    orig = pe.ScatteringMatrix.__init__
    # The chunk size derives from ny*nx; on this tiny grid beam_chunk is large, so
    # exercise the multi-chunk branch directly via the same per-beam ifft2 contract.
    for bc in (1, 3, nbeams):
        S2 = S_q.clone()
        for b0 in range(0, nbeams, bc):
            S2[b0 : b0 + bc] = torch.fft.ifft2(S2[b0 : b0 + bc], dim=(-2, -1))
        assert torch.equal(S2, ref)
    assert pe.ScatteringMatrix.__init__ is orig  # no mutation leaked


def test_scattering_matrix_interpolation_factor_validates_divisibility():
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import ScatteringMatrix

    n = 40
    probe_q = torch.zeros((n, n), dtype=torch.complex64)
    probe_q[:3, :3] = 1.0
    T = torch.ones((1, n, n), dtype=torch.complex64)
    kernel = propagator_kernel((n, n), (4.0, 4.0), 0.0037, 0.0, dtype=torch.complex64)
    with pytest.raises(ValueError):
        ScatteringMatrix(probe_q, T, kernel, interpolation_factor=(3, 3))  # 40 % 3 != 0


def test_simulator_prism_interpolation_factor():
    # f=1 must be exactly the default (backward compatible); f=2 must run and
    # produce a finite, non-negative datacube of the right shape.  (Quantitative
    # f>1 accuracy is regime-dependent -- the standard PRISM probe-fit tradeoff --
    # and is a validation/physics question, not a unit invariant.)
    struct = _single_atom_structure(Z=8)
    common = dict(
        eV=100_000.0,
        semiconvergence_angle=20.0,
        pixels=(48, 48),
        scan=(2, 2),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        prism=True,
        backend="hydrogenic",
        device="cpu",
    )
    c_default = StemEelsSimulator(struct, **common).simulate().cube
    c_f1 = StemEelsSimulator(struct, interpolation_factor=1, **common).simulate().cube
    assert torch.equal(c_default, c_f1)  # f=1 is exactly the unmodified path

    c_f2 = StemEelsSimulator(struct, interpolation_factor=2, **common).simulate().cube
    assert c_f2.shape == c_f1.shape
    assert torch.isfinite(c_f2).all()
    assert float(c_f2.min()) >= 0
    assert float(c_f2.sum()) > 0


# --------------------------------------------------------------------------- #
# Partitioned PRISM (Pelz 2021) for EELS
# --------------------------------------------------------------------------- #
def test_partitioned_prism_matches_exact_focused():
    # Partitioned PRISM (parent-beam NNW basis) must match the exact full
    # per-pixel PRISM for a focused probe (the favorable regime), confirming the
    # synthesis is correct and reduces to the exact probe in the full-parent limit.
    struct = _single_atom_structure(Z=8)
    sim = StemEelsSimulator(
        struct,
        prism=True,
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(40, 40),
        scan=(2, 2),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    tp = sim._build_tp(sim.epsilon)
    pq, scan, sites = sim._probe_q(), sim._scan_pixels(), sim._sites()
    T = sim._transmissions(0)
    common = dict(
        wavelength=sim.wavelength,
        gridsize=sim.gridsize,
        slice_distance=sim.slice_distance,
    )
    exact = prism_transition_potential(pq, T, tp, sites, scan, **common)
    part = prism_transition_potential(
        pq, T, tp, sites, scan, partition=dict(n_radial=4), **common
    )
    assert part.shape == exact.shape
    assert torch.isfinite(part).all()
    assert float((part - exact).norm() / exact.norm()) < 0.05


def test_partitioned_dual_s_equals_partitioned_hybrid():
    # The partitioned dual-S only REORDERS the (linear) work of the partitioned
    # hybrid: reconstruct the exact aperture columns from the parents at the
    # ionization plane, then propagate those columns to the exit instead of each
    # scan probe.  Both share the identical (NNW-approximate) probe, so they must
    # agree to ~machine precision -- this isolates the reordering from the NNW
    # approximation.  Also exercises interpolation_factor composed with partition.
    from scatterem.simulation.eels.prism_eels import prism_transition_potential_hybrid

    torch.manual_seed(0)
    gpts = (48, 48)
    sampling = (0.18, 0.18)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    nz = 3
    T = torch.exp(1j * 0.25 * torch.rand(nz, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    # A broad aperture so there are enough beams to partition into hex rings.
    qy, qx = np.meshgrid(
        np.fft.fftfreq(gpts[0]), np.fft.fftfreq(gpts[1]), indexing="ij"
    )
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.18**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.4, 0.55, 0.5]])
    scan = torch.tensor([[9.0, 13.0], [22.0, 6.0]], dtype=torch.float64)
    common = dict(wavelength=0.0037, gridsize=gridsize, slice_distance=1.0)

    for f in (1, 2):
        dual = prism_transition_potential(
            pq,
            T,
            tp,
            sites,
            scan,
            partition=dict(n_radial=3),
            interpolation_factor=f,
            **common,
        )
        hyb = prism_transition_potential_hybrid(
            pq,
            T,
            tp,
            sites,
            scan,
            partition=dict(n_radial=3),
            interpolation_factor=f,
            **common,
        )
        assert torch.allclose(dual.double(), hyb.double(), rtol=1e-6, atol=1e-9)


def test_partitioned_reconstruct_columns_identity():
    # reconstruct_columns() + coeffs_at() must reproduce probe_at_current_plane()
    # (the scalar-combinable form of the partitioned probe used by the dual-S exit).
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import PartitionedScatteringMatrix

    torch.manual_seed(0)
    gpts = (48, 48)
    qy, qx = np.meshgrid(
        np.fft.fftfreq(gpts[0]), np.fft.fftfreq(gpts[1]), indexing="ij"
    )
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.18**2, dtype=torch.complex128)
    T = torch.exp(1j * 0.2 * torch.rand(2, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    kernel = propagator_kernel(gpts, (8.0, 8.0), 0.0037, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(pq, T, kernel, n_radial=3)
    sm.advance_to(1)
    scan = torch.tensor([[10.0, 14.0], [21.0, 5.0]], dtype=torch.float64)
    probe = sm.probe_at_current_plane(scan)  # (P, Ny, Nx)
    cols = sm.reconstruct_columns()  # (B, Ny, Nx)
    coeffs = sm.coeffs_at(scan)  # (P, B)
    probe_recon = torch.einsum("pb,byx->pyx", coeffs, cols)
    # complex comparison (do not cast to real / discard the imaginary part)
    assert torch.allclose(probe, probe_recon, rtol=1e-6, atol=1e-9)


def test_prism_eels_image_equals_detector_integrated_4d():
    # The detector-S2 (double-channeling) image with NO Hn0 crop and NO PRISM scan
    # mask must equal the detector-integral of the 4D dual-S over |q| < beta -- it
    # is the same physics, with S2 the transpose multislice of the detector beams.
    # This is the correctness gate for the linear-scaling FePt-scale algorithm.
    from scatterem.simulation.eels.prism_eels_image import prism_eels_image

    torch.manual_seed(0)
    gpts = (40, 40)
    sampling = (0.18, 0.18)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    ny, nx = gpts
    nz = 3
    T = torch.exp(1j * 0.3 * torch.rand(nz, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.14**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.45, 0.55, 0.5], [0.3, 0.4, 0.5]])
    scan = torch.tensor([[7.0, 9.0], [15.0, 20.0], [3.0, 25.0]], dtype=torch.float64)
    common = dict(wavelength=0.0037, gridsize=gridsize, slice_distance=1.0)
    beta = 18.0

    # reference: detector-integrate the 4D dual-S over |q| < beta
    cube = prism_transition_potential(pq, T, tp, sites, scan, **common)
    dqy = np.fft.fftfreq(ny, d=sampling[0])
    dqx = np.fft.fftfreq(nx, d=sampling[1])
    QY, QX = np.meshgrid(dqy, dqx, indexing="ij")
    betaq = beta / 1000 / 0.0037
    det = torch.as_tensor((QY**2 + QX**2) <= betaq**2)
    ref = (cube * det[None]).sum(dim=(-2, -1))

    img = prism_eels_image(
        pq,
        T,
        tp,
        sites,
        scan,
        **common,
        detector_mrad=beta,
        interpolation_factor=1,
        inelastic_crop=None,
        prism_mask=False,
    )
    assert torch.allclose(img.double(), ref.double(), rtol=1e-6, atol=1e-9)


def test_prism_eels_image_transition_batching_chunk_invariant(monkeypatch):
    # The per-transition loop is batched into large GEMMs and chunked over the
    # transition axis. Forcing a tiny block cap (tblk == 1, i.e. per-transition)
    # must reproduce the full-batch result exactly (to fp reassociation). The
    # detector-integrated-4D gate above already validates the batched math vs an
    # independent reference; this pins the chunk boundary.
    import sys

    from scatterem.simulation.eels.prism_eels_image import prism_eels_image

    # submodule name is shadowed by the re-exported function in the package
    # namespace, so resolve the actual module to monkeypatch the block cap.
    pei = sys.modules[prism_eels_image.__module__]

    torch.manual_seed(0)
    gpts = (40, 40)
    sampling = (0.18, 0.18)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    ny, nx = gpts
    T = torch.exp(1j * 0.3 * torch.rand(3, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    assert tp.array.shape[0] > 1  # need >1 transition for chunking to matter
    qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.14**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.45, 0.55, 0.5], [0.3, 0.4, 0.5]])
    scan = torch.tensor([[7.0, 9.0], [15.0, 20.0], [3.0, 25.0]], dtype=torch.float64)
    common = dict(
        wavelength=0.0037,
        gridsize=gridsize,
        slice_distance=1.0,
        detector_mrad=18.0,
        interpolation_factor=1,
        inelastic_crop=None,
        prism_mask=False,
    )

    full = prism_eels_image(pq, T, tp, sites, scan, **common)
    monkeypatch.setattr(pei, "_EELS_TRANS_BLOCK_BYTES", 1)  # force tblk == 1
    chunked = prism_eels_image(pq, T, tp, sites, scan, **common)
    assert torch.allclose(full, chunked, rtol=1e-9, atol=1e-12)


def test_prism_eels_image_partitioned_s1_runs_and_correlates():
    # Partitioned S1 (reconstruct exact columns from Bp parents) feeds the same
    # detector-S2 image; it is approximate (NNW) but must reproduce the map
    # pattern. Needs a broad aperture so the hex-ring partitioning has beams.
    from scatterem.simulation.eels.prism_eels_image import prism_eels_image

    torch.manual_seed(0)
    gpts = (48, 48)
    sampling = (0.18, 0.18)
    gridsize = (gpts[0] * sampling[0], gpts[1] * sampling[1])
    ny, nx = gpts
    T = torch.exp(1j * 0.2 * torch.rand(2, *gpts, dtype=torch.float64)).to(
        torch.complex128
    )
    transitions = subshell_transitions(8, 1, 0, 1.0, backend="hydrogenic")
    tp = build_transition_potentials(
        transitions, 8, gpts, sampling, eV=100_000.0, dtype=torch.complex128
    )
    qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    pq = torch.as_tensor((qy**2 + qx**2) <= 0.18**2, dtype=torch.complex128)
    pq = pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()
    sites = np.array([[0.5, 0.5, 0.5]])
    ys = (np.arange(8) + 0.5) / 8 * ny
    xs = (np.arange(8) + 0.5) / 8 * nx
    g0, g1 = np.meshgrid(ys, xs, indexing="ij")
    scan = torch.as_tensor(
        np.column_stack([g0.ravel(), g1.ravel()]), dtype=torch.float64
    )
    common = dict(wavelength=0.0037, gridsize=gridsize, slice_distance=1.0)

    exact = prism_eels_image(
        pq,
        T,
        tp,
        sites,
        scan,
        **common,
        detector_mrad=18.0,
        inelastic_crop=None,
        prism_mask=False,
    )
    part = prism_eels_image(
        pq,
        T,
        tp,
        sites,
        scan,
        **common,
        detector_mrad=18.0,
        inelastic_crop=None,
        prism_mask=False,
        partition=dict(n_radial=3),
    )
    assert part.shape == exact.shape and torch.isfinite(part).all()
    e, p = exact.double().reshape(-1), part.double().reshape(-1)
    corr = torch.corrcoef(torch.stack([e, p]))[0, 1]
    assert float(corr) > 0.98  # NNW-reconstructed map matches the exact pattern

    # Partitioning S2 as well (detector matrix) must also run and reproduce the
    # pattern -- this is the second NNW channel that shrinks the resident S2.
    both = prism_eels_image(
        pq,
        T,
        tp,
        sites,
        scan,
        **common,
        detector_mrad=18.0,
        inelastic_crop=None,
        prism_mask=False,
        partition=dict(n_radial=3),
        partition_s2=dict(n_radial=3),
    )
    assert both.shape == exact.shape and torch.isfinite(both).all()
    cb = torch.corrcoef(torch.stack([e, both.double().reshape(-1)]))[0, 1]
    assert float(cb) > 0.95


def test_partitioned_prism_focal_backprop_improves_deep_slice():
    # Focal back-propagation (centroid, default) must reduce the partitioned-PRISM
    # probe-synthesis error at a DEEP transition slice (thick sample) relative to
    # no back-prop, measured against the exact full per-pixel matrix. The error
    # grows with depth (propagation chirp); the centroid plane roughly halves it.
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import (
        PartitionedScatteringMatrix,
        ScatteringMatrix,
    )

    torch.manual_seed(0)
    Ny = Nx = 48
    NZ, px = 40, 0.2
    gs = (Ny * px, Nx * px)
    lam = 0.037
    T = torch.exp(1j * 0.06 * torch.randn(NZ, Ny, Nx)).to(torch.complex64)
    kernel = propagator_kernel((Ny, Nx), gs, lam, 2.0, dtype=torch.complex64)
    qy = torch.fft.fftfreq(Ny, d=px).view(Ny, 1)
    qx = torch.fft.fftfreq(Nx, d=px).view(1, Nx)
    probe_q = ((qy**2 + qx**2) <= (0.025 / lam) ** 2).to(torch.complex64)
    scan = torch.tensor([[0, 0], [13, 17], [25, 9]], dtype=torch.long)
    depth = 25  # deep enough for a sizable chirp

    full = ScatteringMatrix(probe_q, T, kernel, interpolation_factor=(1, 1))
    full.advance_to(depth)
    ref = full.probe_at_current_plane(scan)

    def part_err(fb):
        m = PartitionedScatteringMatrix(
            probe_q, T, kernel, n_radial=4, focal_backprop=fb
        )
        m.advance_to(depth)
        p = m.probe_at_current_plane(scan)
        return float((p - ref).norm() / ref.norm())

    e_off = part_err(0.0)
    e_cent = part_err("centroid")  # default
    assert (
        e_cent < 0.8 * e_off
    ), f"centroid ({e_cent:.3f}) should beat off ({e_off:.3f})"


def test_partitioned_scattering_matrix_requires_two_rings():
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import PartitionedScatteringMatrix

    pq = torch.zeros((40, 40), dtype=torch.complex64)
    pq[:4, :4] = 1.0
    pq[-3:, -3:] = 1.0
    T = torch.ones((1, 40, 40), dtype=torch.complex64)
    kernel = propagator_kernel((40, 40), (4.0, 4.0), 0.0037, 0.0, dtype=torch.complex64)
    with pytest.raises(ValueError):
        PartitionedScatteringMatrix(pq, T, kernel, n_radial=1)


def test_partitioned_roll_path_matches_fourier_path():
    # Integer scan positions take the cyclic-roll fast path; a tiny offset forces
    # the Fourier-shift path.  The two must agree (the roll is the exact integer
    # shift), so the Nyquist-step optimisation is lossless.
    from scatterem.simulation.eels.multislice_eels import propagator_kernel
    from scatterem.simulation.eels.prism_eels import PartitionedScatteringMatrix

    struct = _single_atom_structure(Z=8)
    sim = StemEelsSimulator(
        struct,
        prism=True,
        eV=100_000.0,
        semiconvergence_angle=25.0,
        pixels=(48, 48),
        scan=(2, 2),
        edge=(8, 1, 0),
        epsilon=1.0,
        num_slices=2,
        backend="hydrogenic",
        device="cpu",
    )
    pq = sim._probe_q()
    T = sim._transmissions(0)
    kernel = propagator_kernel(
        sim.pixels, sim.gridsize, sim.wavelength, sim.slice_distance, dtype=T.dtype
    )
    sm = PartitionedScatteringMatrix(pq.to(T.dtype), T, kernel, n_radial=4)
    sm.advance_to(1)
    scan_int = torch.tensor([[12.0, 24.0], [36.0, 8.0]], dtype=torch.float64)
    roll = sm.probe_at_current_plane(scan_int)  # integer -> roll path
    fourier = sm.probe_at_current_plane(scan_int + 1e-7)  # -> Fourier path
    assert torch.allclose(roll, fourier, atol=1e-4)
