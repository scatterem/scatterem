"""PrismEelsSimulator: a thin driver over the PRISM and PRISM-EELS kernels.

1. Each output equals the direct kernel call on inputs built here from the
   physics functions, bit for bit.
2. The elastic CBED equals an independent plane multislice of the same probe,
   cut to the PRISM window (Ophus 2017) and transformed.
3. Partitioned PRISM converges to unpartitioned PRISM at the same grid, slices,
   interpolation factor, crop and detector: in the EELS map with both matrices
   partitioned (BiP-PRISM), and in the elastic CBED.

Small CPU problems with hydrogenic wavefunctions (no GPAW).
"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
import torch

from scatterem.nn.modules.propagator import Propagator
from scatterem.nn.scan_exit_wave._prism_geometry import probe_disc
from scatterem.nn.scan_exit_wave.prism import PrismScanOp
from scatterem.nn.scan_exit_wave.prism_smeta import build_prism_smeta
from scatterem.physics.atom_set import AtomSet
from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.constants import debroglie_at
from scatterem.physics.potential_stack import potential_stack
from scatterem.physics.propagation import propagator_kernel
from scatterem.physics.thin_slice import thin_slice_transmission
from scatterem.simulation.eels import PrismEelsSimulator, PrismResult
from scatterem.simulation.eels.forward.prism_eels import prism_transition_potential
from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.transition_potentials import (
    build_transition_potentials,
    subshell_transitions,
)

EV = 100e3
ALPHA_MRAD = 20.0
EDGE = (22, 2, 1)
HYD = {"wavefunctions": "hydrogenic"}  # the analytic states, no GPAW
EPS = 5.0
DETECTOR_MRAD = 30.0


def _structure() -> AtomSet:
    """An asymmetric 8 A cell: no symmetry hides a misplaced window."""
    atoms = np.array(
        [
            [0.2, 0.3, 0.25, 29],
            [0.7, 0.55, 0.75, 14],
            [0.4, 0.9, 0.5, 8],
            [0.1, 0.6, 0.3, 22],
            [0.6, 0.15, 0.8, 22],
        ]
    )
    return AtomSet(np.array([8.0, 8.0, 8.0]), atoms[:, :3], atoms[:, 3])


def _sim(**kw) -> PrismEelsSimulator:
    args = dict(
        eV=EV,
        semiconvergence_angle=ALPHA_MRAD,
        pixels=(32, 32),
        scan=(3, 3),
        num_slices=4,
        device="cpu",
    )
    args.update(kw)
    return PrismEelsSimulator(_structure(), **args)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


# ----------------------------------------------------------------- 1. thin


def _direct_inputs(defocus: float, f: int, dtype=torch.complex64):
    """The kernels' inputs, built from the physics functions only."""
    st, pixels, n_slices = _structure(), (32, 32), 4
    wavelength = debroglie_at(EV)
    gridsize = (8.0, 8.0)
    sampling = (8.0 / 32, 8.0 / 32)
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    potential = potential_stack(
        BeamPartition.uniform(n_slices),
        st,
        pixels,
        device="cpu",
        dtype=real,
    )
    T = thin_slice_transmission(EV, potential, bandlimit=2 / 3).to(dtype)
    ap = probe_disc(
        pixels, (1 / 8.0, 1 / 8.0), wavelength, ALPHA_MRAD * 1e-3, "cpu",
        dtype=torch.float64,
    )  # fmt: skip
    q = torch.fft.fftfreq(32, d=sampling[0], dtype=torch.float64)
    chi = pi * wavelength * defocus * (q[:, None] ** 2 + q[None, :] ** 2)
    probe_q = (ap * torch.exp(-1j * chi)).to(dtype)
    probe_q = probe_q / (probe_q.abs().pow(2).sum() / probe_q.numel()).sqrt()
    tp = build_transition_potentials(
        subshell_transitions(*EDGE, EPS, wavefunctions="hydrogenic"),
        EDGE[0],
        pixels,
        sampling,
        EV,
        device="cpu",
        dtype=dtype,
    )
    sites = st.fractional[np.round(st.element_z).astype(int) == EDGE[0]]
    r = (np.arange(3) + 0.5) * 32 / 3
    scan = torch.as_tensor(
        np.stack(np.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2),
        dtype=torch.float64,
    )
    common = dict(
        wavelength=wavelength,
        gridsize=gridsize,
        slice_distance=8.0 / n_slices,
        interpolation_factor=f,
    )
    return (probe_q, T, tp, sites, scan), common


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_map_qeels_and_cube_are_the_direct_kernel_calls(dtype):
    sim = _sim(defocus=12.0, interpolation_factor=2, dtype=dtype)
    args, common = _direct_inputs(12.0, 2, dtype)
    part = dict(partition=dict(n_radial=2), partition_s2=dict(n_radial=3))
    opts = dict(detector_mrad=DETECTOR_MRAD, inelastic_crop=2, scan_shape=(3, 3))

    got = sim.eels_map(EDGE, transitions=HYD, epsilon=EPS, detector_mrad=DETECTOR_MRAD,
                       inelastic_crop=2, **part)  # fmt: skip
    want = prism_eels_image(*args, **common, **opts, **part)
    assert torch.equal(got.data, want)
    assert got.output == "map" and got.edge == EDGE and got.epsilon == EPS

    got = sim.qeels(
        EDGE,
        transitions=HYD,
        epsilon=EPS,
        detector_mrad=DETECTOR_MRAD,
        inelastic_crop=2,
    )
    want, q = prism_eels_image(*args, **common, **opts, qeels_axis="both")
    assert torch.equal(got.data, want) and torch.equal(got.q_pixels, q)

    got = sim.eels_cube(EDGE, transitions=HYD, epsilon=EPS, partition=dict(n_radial=2))
    want = prism_transition_potential(*args, **common, partition=dict(n_radial=2))
    assert torch.equal(got.data, want.reshape(3, 3, 32, 32))
    np.testing.assert_array_equal(got.energy_losses, args[2].energy_losses)


@pytest.mark.parametrize("n_radial", [None, 2])
def test_elastic_cbed_is_the_direct_scan_operator_call(n_radial):
    sim = _sim(interpolation_factor=2, defocus=-8.0)
    (_, T, _, _, scan), _ = _direct_inputs(-8.0, 2)
    wavelength = debroglie_at(EV)
    s_meta = build_prism_smeta(
        pixels=(16, 16),
        f=2,
        dx=np.array([0.25, 0.25], dtype=np.float32),
        wavelength=wavelength,
        semiconvergence_angle=ALPHA_MRAD * 1e-3,
        fov_shape=None,
        device="cpu",
        n_radial=n_radial,
    )
    op = PrismScanOp(s_meta, f=2)
    prop = Propagator(2.0, wavelength, (0.25, 0.25), (32, 32), "cpu")
    ap = probe_disc(
        (16, 16), (0.25, 0.25), wavelength, ALPHA_MRAD * 1e-3, "cpu",
        dtype=torch.float64,
    )  # fmt: skip
    q = torch.fft.fftfreq(16, d=0.25, dtype=torch.float64)
    chi = pi * wavelength * -8.0 * (q[:, None] ** 2 + q[None, :] ** 2)
    pq = (ap.double() * torch.exp(-1j * chi)).to(torch.complex64)
    pq = pq / pq.abs().pow(2).sum().sqrt()
    probe = torch.fft.fftshift(torch.fft.ifft2(pq, norm="ortho"))
    corner = scan - 8.0
    pos = torch.round(corner)
    with torch.no_grad():
        exit_waves = op(
            T,
            probe[None, None].expand(1, 9, 16, 16).contiguous(),
            pos.int(),
            (corner - pos).float(),
            prop,
        )[0]
    want = torch.fft.fft2(exit_waves, norm="ortho").abs().pow(2).reshape(3, 3, 16, 16)
    got = sim.elastic_cbed(n_radial=n_radial)
    assert torch.equal(got.data, want)
    assert got.output == "cbed" and got.edge is None
    assert got.detector_sampling == (0.25, 0.25)


def test_frozen_phonons_average_the_seeded_configurations():
    kw = dict(displacements=True, seed=7, scan=(2, 2))
    both = _sim(num_frozen_phonons=2, **kw).eels_map(
        EDGE, transitions=HYD, detector_mrad=30.0
    )
    one = _sim(**kw).eels_map(EDGE, transitions=HYD, detector_mrad=30.0)
    two = _sim(**dict(kw, seed=8)).eels_map(EDGE, transitions=HYD, detector_mrad=30.0)
    torch.testing.assert_close(both.data, (one.data + two.data) / 2)
    assert _rel(one.data, two.data) > 1e-4  # the configurations differ


def _spy_configurations(monkeypatch):
    """Record the ``jitter`` of each potential build and the ``sites`` of each
    EELS kernel call of the simulator."""
    import scatterem.simulation.eels.prism_eels_simulator as pes

    jitters, sites = [], []
    build, kernel = pes.potential_stack, pes.prism_eels_image

    def spy_build(*args, **kw):
        jitters.append(kw.get("jitter"))
        return build(*args, **kw)

    def spy_kernel(probe_q, T, tp, s, *args, **kw):
        sites.append(np.array(s, dtype=np.float64))
        return kernel(probe_q, T, tp, s, *args, **kw)

    monkeypatch.setattr(pes, "potential_stack", spy_build)
    monkeypatch.setattr(pes, "prism_eels_image", spy_kernel)
    return jitters, sites


@pytest.mark.parametrize("seed", [7, None])
def test_frozen_phonon_sites_follow_the_displaced_atoms(monkeypatch, seed):
    """Each configuration ionizes the atoms where its transmissions put
    them, also when ``seed=None`` draws a fresh seed per configuration."""
    from scatterem.physics.thermal_jitter import jitter_offsets

    jitters, sites = _spy_configurations(monkeypatch)
    sim = _sim(num_frozen_phonons=2, displacements=True, seed=seed, scan=(2, 2))
    sim.eels_map(EDGE, transitions=HYD, detector_mrad=DETECTOR_MRAD)
    assert len(jitters) == len(sites) == 2
    st = sim.structure
    pick = np.asarray(st.element_z) == EDGE[0]
    static = np.asarray(st.fractional, dtype=np.float64)[pick]
    for jitter, got in zip(jitters, sites, strict=True):
        assert jitter is not None
        offsets = jitter_offsets(st, (1, 1), jitter, device=sim.device)[0, 0]
        want = static + (offsets.cpu().numpy() / np.asarray(st.sides))[pick]
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
        assert np.abs(got - static).max() > 1e-4  # the sites did move
    assert not np.array_equal(sites[0], sites[1])


def _occupied(fill_ti: float) -> AtomSet:
    st = _structure()
    fill = np.where(np.asarray(st.element_z) == EDGE[0], fill_ti, 1.0)
    return AtomSet(st.sides, st.fractional, st.element_z, site_fraction=fill)


def test_zero_occupancy_sites_give_no_eels_signal():
    """A site with ``site_fraction=0`` holds no atom, so it does not ionize."""
    sim = PrismEelsSimulator(
        _occupied(0.0),
        eV=EV,
        semiconvergence_angle=ALPHA_MRAD,
        pixels=(32, 32),
        scan=(3, 3),
        num_slices=4,
        device="cpu",
    )
    opts = dict(transitions=HYD, epsilon=EPS)
    m = sim.eels_map(EDGE, detector_mrad=DETECTOR_MRAD, **opts)
    q = sim.qeels(EDGE, detector_mrad=DETECTOR_MRAD, **opts)
    c = sim.eels_cube(EDGE, **opts)
    for r in (m, q, c):
        assert torch.count_nonzero(r.data) == 0
    assert m.data.shape == (3, 3) and c.data.shape == (3, 3, 32, 32)


@pytest.mark.parametrize("output", ["map", "cube"])
def test_half_occupancy_gives_half_the_single_site_signal(output):
    """The inelastic signal of a site scales with its site fraction. The
    reference is the direct kernel call on the same (half-occupied)
    transmissions, with the sites at full weight."""
    sim = PrismEelsSimulator(
        _occupied(0.5),
        eV=EV,
        semiconvergence_angle=ALPHA_MRAD,
        pixels=(32, 32),
        scan=(3, 3),
        num_slices=4,
        device="cpu",
    )
    tp = sim.transition_potentials(EDGE, EPS, **HYD)
    common = dict(
        wavelength=sim.wavelength,
        gridsize=sim.gridsize,
        slice_distance=sim.slice_distance,
        interpolation_factor=1,
    )
    args = (sim.probe_q(), sim.transmissions(), tp, sim.sites(EDGE[0]))
    if output == "map":
        got = sim.eels_map(EDGE, transitions=HYD, epsilon=EPS,
                           detector_mrad=DETECTOR_MRAD).data  # fmt: skip
        full = prism_eels_image(*args, sim.scan_pixels(), **common,
                                detector_mrad=DETECTOR_MRAD, scan_shape=(3, 3))  # fmt: skip
    else:
        got = sim.eels_cube(EDGE, transitions=HYD, epsilon=EPS).data
        full = prism_transition_potential(*args, sim.scan_pixels(), **common)
        full = full.reshape(3, 3, 32, 32)
    assert float(full.abs().max()) > 0.0
    torch.testing.assert_close(got, 0.5 * full, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------- 2. smoke / physics


def test_eels_outputs_are_consistent():
    sim = _sim()
    m = sim.eels_map(EDGE, transitions=HYD, epsilon=EPS, detector_mrad=DETECTOR_MRAD)
    q = sim.qeels(EDGE, transitions=HYD, epsilon=EPS, detector_mrad=DETECTOR_MRAD)
    c = sim.eels_cube(EDGE, transitions=HYD, epsilon=EPS)
    assert isinstance(m, PrismResult)
    assert m.data.shape == (3, 3)
    assert q.data.shape[:2] == (3, 3) and q.q_pixels.shape == (q.data.shape[2], 2)
    assert c.data.shape == (3, 3, 32, 32)
    for r in (m, q, c):
        assert torch.isfinite(r.data).all() and float(r.data.min()) >= 0.0
        assert r.detector_sampling == (1 / 8.0, 1 / 8.0)
    assert float(m.data.max()) > 0.0
    torch.testing.assert_close(q.data.sum(-1), m.data, rtol=1e-10, atol=0.0)
    q0 = sim.qeels(
        EDGE, transitions=HYD, epsilon=EPS, detector_mrad=DETECTOR_MRAD, axis=0
    )
    torch.testing.assert_close(q0.data.sum(-1), m.data, rtol=1e-5, atol=0.0)


def _multislice_cbed(sim: PrismEelsSimulator) -> torch.Tensor:
    """Independent reference: plane multislice of the PRISM probe on the full
    grid, cut to the window centred on the probe, then transformed."""
    f = sim.interpolation_factor
    N, P = sim.pixels[0], sim.pixels[0] // f
    T = sim.transmissions().to(torch.complex128)
    kernel = propagator_kernel(
        (N, N), sim.gridsize, sim.wavelength, sim.slice_thickness,
        dtype=torch.complex128, device="cpu",
    )  # fmt: skip
    ap = probe_disc(
        (P, P), (f / 8.0, f / 8.0), sim.wavelength, sim.semiconvergence_angle * 1e-3,
        "cpu",
    ).to(torch.complex128)  # fmt: skip
    spectrum = torch.zeros((N, N), dtype=torch.complex128)
    spectrum[::f, ::f] = ap / ap.abs().pow(2).sum().sqrt()
    k = torch.fft.fftfreq(N, dtype=torch.float64)
    out = []
    for r in sim.scan_pixels():
        ramp = torch.exp(-2j * pi * (k[:, None] * r[0] + k[None, :] * r[1]))
        psi = torch.fft.ifft2(spectrum * ramp)
        for z in range(T.shape[0]):
            psi = psi * T[z]
            if z < T.shape[0] - 1:
                psi = torch.fft.ifft2(torch.fft.fft2(psi) * kernel)
        y0, x0 = (int(torch.round(v)) - P // 2 for v in r)
        cut = psi[(y0 + torch.arange(P)) % N][:, (x0 + torch.arange(P)) % N]
        out.append(torch.fft.fft2(cut).abs() ** 2)
    return torch.stack(out).reshape(*sim.scan, P, P)


@pytest.mark.parametrize(("f", "scan"), [(1, (3, 3)), (2, (3, 3)), (2, (5, 4))])
def test_elastic_cbed_matches_multislice_in_the_prism_window(f, scan):
    """(5, 4) puts the probe between pixels: the sub-pixel shift is exercised."""
    sim = _sim(interpolation_factor=f, scan=scan)
    got = sim.elastic_cbed().data.double()
    ref = _multislice_cbed(sim)
    assert got.shape == ref.shape == (*scan, 32 // f, 32 // f)
    assert _rel(got / got.sum(), ref / ref.sum()) < 1e-5
    # Unit probe intensity; the band-limited grating is not exactly unitary.
    total = got.sum((-2, -1))
    assert float(total.max()) < 1.001 and float(total.min()) > 0.95


def test_elastic_window_is_centred_on_the_probe():
    """Negative control: the reference cut a quarter window off does not match."""
    sim = _sim(interpolation_factor=2)
    got = sim.elastic_cbed().data.double()
    shifted = _sim(interpolation_factor=2, scan=(3, 3))
    shifted.scan_pixels = lambda: sim.scan_pixels() + 4.0  # type: ignore[method-assign]
    ref = _multislice_cbed(shifted)
    assert _rel(got / got.sum(), ref / ref.sum()) > 1e-3


# ------------------------------------------------------------ 3. convergence


def test_bip_prism_map_converges_to_the_unpartitioned_map():
    """Matched grid, slices, f, crop and detector; only the parents change."""
    sim = _sim(interpolation_factor=2)
    opts = dict(epsilon=EPS, detector_mrad=DETECTOR_MRAD, inelastic_crop=2)
    exact = sim.eels_map(EDGE, transitions=HYD, **opts).data
    errors = [
        _rel(
            sim.eels_map(EDGE, transitions=HYD, **opts,
                partition=dict(n_radial=n), partition_s2=dict(n_radial=n),
            ).data,
            exact,
        )
        for n in (2, 4, 10)
    ]  # fmt: skip
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[0] > 1e-5  # the coarse partition is an approximation
    assert errors[2] < 1e-5, errors


def test_partitioned_elastic_cbed_converges_to_prism():
    sim = _sim()
    exact = sim.elastic_cbed().data
    errors = [_rel(sim.elastic_cbed(n_radial=n).data, exact) for n in (2, 4, 10)]
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] < 1e-5, errors


# ------------------------------------------------------------------ 4. errors


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="dtype"):
        _sim(dtype=torch.float32)
    with pytest.raises(ValueError, match="num_slices"):
        _sim(num_slices=0)
    with pytest.raises(ValueError, match="Z=79"):
        _sim().eels_map((79, 4, 3), transitions=HYD, detector_mrad=30.0)
