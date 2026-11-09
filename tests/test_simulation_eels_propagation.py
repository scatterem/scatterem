"""Propagation primitives shared by the conventional and PRISM EELS paths.

These exercise ``multislice_eels`` and ``prism_eels`` at the level of their
numerical contracts -- Fresnel phase, Fourier-shift equivalences, the
bit-exactness claims made for the fast paths, and the scattering-matrix
synthesis identities -- using synthetic transition potentials so no radial /
GOS data is needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scatterem.physics import _ramp_fused as ramp_fused
from scatterem.physics.fourier import fourier_shift
from scatterem.physics.periodic_sampling import PeriodicGrid, dft_sample_frequencies
from scatterem.physics.propagation import (
    _fftfreq_f64,
    _scaled_propagator,
    fourier_shift_stack,
    propagate,
    propagator_kernel,
)
from scatterem.simulation._wave.exit_walk import (
    _slice_exit_wave,
    _unit_transmission_slices,
)
from scatterem.simulation._wave.partitioned_smatrix import PartitionedScatteringMatrix
from scatterem.simulation._wave.prism_beams import (
    _abs_complex,
    _factor_pair,
    _partition_weights,
    _select_parent_beams,
)
from scatterem.simulation._wave.smatrix import ScatteringMatrix
from scatterem.simulation.eels.forward.multislice_eels import (
    multislice_transition_potential,
)
from scatterem.simulation.eels.forward.prism_eels import (
    prism_transition_potential,
    prism_transition_potential_hybrid,
)
from scatterem.simulation.eels.physics.transition_potentials import TransitionPotentials

_WAVELENGTH = 0.0037  # A, ~100 kV


def _tp(gpts, n_trans=2, *, seed=0, dtype=torch.complex128) -> TransitionPotentials:
    """A synthetic transition-potential stack (the propagation code is agnostic)."""
    g = torch.Generator().manual_seed(seed)
    arr = torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64) + 1j * (
        torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64)
    )
    return TransitionPotentials(
        array=arr.to(dtype),
        Z=8,
        energy=100_000.0,
        sampling=(0.1, 0.1),
        energy_losses=np.full(n_trans, 532.0),
    )


def _aperture(gpts, radius=0.18, dtype=torch.complex128) -> torch.Tensor:
    qy, qx = np.meshgrid(
        np.fft.fftfreq(gpts[0]), np.fft.fftfreq(gpts[1]), indexing="ij"
    )
    pq = torch.as_tensor((qy**2 + qx**2) <= radius**2, dtype=dtype)
    return pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()


def _transmissions(nz, gpts, *, seed=0, strength=0.25, dtype=torch.complex128):
    g = torch.Generator().manual_seed(seed)
    phase = strength * torch.rand(nz, *gpts, generator=g, dtype=torch.float64)
    return torch.exp(1j * phase).to(dtype)


# --------------------------------------------------------------------------- #
# propagator_kernel
# --------------------------------------------------------------------------- #
def test_propagator_kernel_is_the_unit_modulus_fresnel_phase():
    gpts, gridsize, dz = (8, 6), (4.0, 3.0), 1.7
    k = propagator_kernel(gpts, gridsize, _WAVELENGTH, dz, dtype=torch.complex128)
    qy, qx = np.broadcast_arrays(
        *dft_sample_frequencies(PeriodicGrid(tuple(gpts), tuple(gridsize)))
    )
    expected = np.exp(-1j * math.pi * _WAVELENGTH * dz * (qy**2 + qx**2))
    assert k.shape == gpts
    np.testing.assert_allclose(k.numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(k.abs().numpy(), 1.0, atol=1e-12)
    assert k[0, 0] == pytest.approx(1.0 + 0j)  # DC beam is unretarded


def test_propagator_kernel_is_the_identity_at_zero_distance_and_honours_dtype():
    gpts = (6, 6)
    k = propagator_kernel(gpts, (3.0, 3.0), _WAVELENGTH, 0.0, dtype=torch.complex64)
    assert k.dtype == torch.complex64
    assert torch.equal(k, torch.ones(gpts, dtype=torch.complex64))


# --------------------------------------------------------------------------- #
# _scaled_propagator / propagate
# --------------------------------------------------------------------------- #
def test_scaled_propagator_memoises_on_the_kernel_and_keys_on_the_pixel_count():
    k = propagator_kernel((4, 4), (2.0, 2.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    a = _scaled_propagator(k, 16)
    b = _scaled_propagator(k, 16)
    assert a is b  # the memo, not a recompute
    torch.testing.assert_close(a, k / 16)
    c = _scaled_propagator(k, 8)  # a different pixel count invalidates it
    assert c is not a
    torch.testing.assert_close(c, k / 8)


def test_scaled_propagator_never_memoises_a_grad_tracking_kernel():
    k = propagator_kernel((4, 4), (2.0, 2.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    k = k.detach().requires_grad_(True)
    a = _scaled_propagator(k, 16)
    b = _scaled_propagator(k, 16)
    assert a is not b  # a cached non-leaf would be backpropagated through twice
    assert not hasattr(k, "_scatterem_prop_scaled")
    assert a.requires_grad


def test_propagate_is_the_normalised_fresnel_round_trip():
    torch.manual_seed(0)
    gpts = (8, 8)
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    w = torch.randn(3, *gpts, dtype=torch.complex128)
    expected = torch.fft.ifft2(torch.fft.fft2(w, dim=(-2, -1)) * k, dim=(-2, -1))
    torch.testing.assert_close(propagate(w.clone(), k), expected)
    # A 2-D wave takes the same (out-of-place) path.
    torch.testing.assert_close(
        propagate(w[0].clone(), k),
        torch.fft.ifft2(torch.fft.fft2(w[0], dim=(-2, -1)) * k, dim=(-2, -1)),
    )


def test_propagate_preserves_intensity_in_vacuum():
    torch.manual_seed(1)
    gpts = (16, 16)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 3.0, dtype=torch.complex128)
    w = torch.randn(2, *gpts, dtype=torch.complex128)
    out = propagate(w, k)
    # |kernel| == 1 and the FFT pair is unitary, so total intensity is conserved.
    torch.testing.assert_close(
        out.abs().pow(2).sum(dim=(-2, -1)), w.abs().pow(2).sum(dim=(-2, -1))
    )


# --------------------------------------------------------------------------- #
# Fourier shifts
# --------------------------------------------------------------------------- #
def test_fftfreq_memo_returns_the_same_object_and_the_right_values():
    a = _fftfreq_f64(12, torch.device("cpu"))
    b = _fftfreq_f64(12, torch.device("cpu"))
    assert a is b
    torch.testing.assert_close(a, torch.fft.fftfreq(12, dtype=torch.float64))


def test_fourier_shift_of_an_integer_offset_is_a_roll():
    torch.manual_seed(0)
    w = torch.randn(3, 8, 8, dtype=torch.complex128)
    shifted = fourier_shift(w, (2.0, -3.0))
    torch.testing.assert_close(
        shifted, torch.roll(w, shifts=(2, -3), dims=(-2, -1)), atol=1e-12, rtol=1e-12
    )


def test_two_half_pixel_shifts_compose_into_one_whole_pixel():
    torch.manual_seed(0)
    w = torch.randn(8, 8, dtype=torch.complex128)
    once = fourier_shift(fourier_shift(w, (0.5, 0.0)), (0.5, 0.0))
    torch.testing.assert_close(
        once, torch.roll(w, shifts=(1,), dims=(-2,)), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("gpts", [(8, 8), (6, 10)])  # power-of-two folds, 6x10 does not
def test_fourier_shift_stack_matches_the_per_position_loop(gpts):
    torch.manual_seed(0)
    w = torch.randn(*gpts, dtype=torch.complex128)
    shifts = torch.tensor([[1.0, 2.0], [-3.5, 0.25], [0.0, 0.0]], dtype=torch.float64)
    ref = torch.stack([fourier_shift(w, (float(s[0]), float(s[1]))) for s in shifts])
    torch.testing.assert_close(fourier_shift_stack(w, shifts), ref)


def test_fourier_shift_stack_chunks_over_positions_without_changing_the_result():
    torch.manual_seed(0)
    gpts = (8, 8)
    w = torch.randn(*gpts, dtype=torch.complex128)
    shifts = torch.tensor([[1.0, 2.0], [-3.5, 0.25], [4.0, -1.5]], dtype=torch.float64)
    whole = fourier_shift_stack(w, shifts)
    # phase_bytes just below one plane forces one chunk per scan position.
    chunked = fourier_shift_stack(w, shifts, phase_bytes=16 * gpts[0] * gpts[1])
    torch.testing.assert_close(chunked, whole)


def test_fourier_shift_stack_of_an_empty_scan_keeps_the_output_shape():
    w = torch.zeros(4, 4, dtype=torch.complex128)
    out = fourier_shift_stack(w, torch.zeros(0, 2, dtype=torch.float64))
    assert out.shape == (0, 4, 4)


# --------------------------------------------------------------------------- #
# Slice bookkeeping
# --------------------------------------------------------------------------- #
def test_unit_transmission_slices_flags_exactly_the_vacuum_slices():
    T = torch.ones(3, 4, 4, dtype=torch.complex64)
    T[1, 2, 2] = 1.0 + 1e-6j
    assert _unit_transmission_slices(T) == [True, False, True]


def test_skipping_unit_slices_is_bit_identical():
    torch.manual_seed(0)
    gpts = (8, 8)
    T = _transmissions(3, gpts, seed=3)
    T[0] = 1.0  # a genuinely empty slice
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    w = torch.randn(2, *gpts, dtype=torch.complex128)
    fast = _slice_exit_wave(w.clone(), T, k, 0, _unit_transmission_slices(T))
    slow = _slice_exit_wave(w.clone(), T, k, 0, None)
    assert torch.equal(fast, slow)


def test_slice_exit_wave_from_the_last_slice_only_transmits():
    torch.manual_seed(0)
    gpts = (6, 6)
    T = _transmissions(3, gpts, seed=1)
    k = propagator_kernel(gpts, (3.0, 3.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    w = torch.randn(2, *gpts, dtype=torch.complex128)
    torch.testing.assert_close(_slice_exit_wave(w, T, k, 2, None), w * T[2])


def test_slice_exit_wave_handles_a_promoting_dtype_pair():
    """complex64 waves against complex128 transmissions cannot multiply in place."""
    torch.manual_seed(0)
    gpts = (6, 6)
    T = _transmissions(2, gpts, seed=2, dtype=torch.complex128)
    k = propagator_kernel(gpts, (3.0, 3.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    w = torch.randn(2, *gpts, dtype=torch.complex64)
    out = _slice_exit_wave(w, T, k, 0, None)
    assert out.dtype == torch.complex128
    manual = propagate((w.to(torch.complex128) * T[0]), k) * T[1]
    torch.testing.assert_close(out, manual)
    assert w.dtype == torch.complex64  # the input was not written through


# --------------------------------------------------------------------------- #
# multislice_transition_potential
# --------------------------------------------------------------------------- #
def _vacuum_setup(gpts=(8, 8), n_trans=2):
    tp = _tp(gpts, n_trans)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    probe_q = torch.zeros(gpts, dtype=torch.complex128)
    probe_q[0, 0] = 1.0
    probes = torch.fft.ifft2(probe_q)[None]
    return tp, T, probes


def test_single_slice_vacuum_intensity_has_the_closed_form():
    gpts = (8, 8)
    tp, T, probes = _vacuum_setup(gpts)
    sites = np.array([[0.25, 0.5, 0.0]])
    out = multislice_transition_potential(
        probes,
        T,
        tp,
        sites,
        wavelength=_WAVELENGTH,
        gridsize=(4.0, 4.0),
        slice_distance=0.0,
    )
    h = fourier_shift(tp.array, (0.25 * gpts[0], 0.5 * gpts[1]))
    # Per incident electron: |fft2|^2 / (Ny Nx), so the pattern sums to the
    # real-space intensity (Parseval).
    expected = torch.fft.fft2(h * probes[0][None], dim=(-2, -1)).abs().pow(2).sum(dim=0)
    expected = expected / (gpts[0] * gpts[1])
    torch.testing.assert_close(out[0], expected.to(torch.float64))


def test_real_space_and_diffraction_outputs_obey_parseval():
    gpts = (8, 8)
    tp, T, probes = _vacuum_setup(gpts)
    sites = np.array([[0.3, 0.7, 0.0]])
    common = dict(wavelength=_WAVELENGTH, gridsize=(4.0, 4.0), slice_distance=0.0)
    diff = multislice_transition_potential(probes, T, tp, sites, **common)
    real = multislice_transition_potential(
        probes, T, tp, sites, return_real_space=True, **common
    )
    assert float(diff.sum()) == pytest.approx(float(real.sum()), rel=1e-10)
    assert float(real.min()) >= 0.0


def test_scan_batching_does_not_change_the_result():
    gpts = (8, 8)
    tp = _tp(gpts, 2)
    T = _transmissions(2, gpts, seed=4)
    probe_q = _aperture(gpts, 0.3)
    base = torch.fft.ifft2(probe_q)
    scan = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.5]], dtype=torch.float64)
    probes = fourier_shift_stack(base, scan)
    sites = np.array([[0.4, 0.6, 0.75]])
    common = dict(wavelength=_WAVELENGTH, gridsize=(4.0, 4.0), slice_distance=1.0)
    whole = multislice_transition_potential(probes, T, tp, sites, **common)
    batched = multislice_transition_potential(
        probes, T, tp, sites, batch_size=1, **common
    )
    torch.testing.assert_close(batched, whole)


def test_site_threshold_skips_weak_events_and_is_inert_when_small():
    gpts = (8, 8)
    tp, T, probes = _vacuum_setup(gpts)
    sites = np.array([[0.5, 0.5, 0.0]])
    common = dict(wavelength=_WAVELENGTH, gridsize=(4.0, 4.0), slice_distance=0.0)
    plain = multislice_transition_potential(probes, T, tp, sites, **common)
    kept = multislice_transition_potential(
        probes, T, tp, sites, site_threshold=1e-12, **common
    )
    torch.testing.assert_close(kept, plain)
    dropped = multislice_transition_potential(
        probes, T, tp, sites, site_threshold=1e12, **common
    )
    assert torch.equal(dropped, torch.zeros_like(dropped))


def test_site_z_is_wrapped_into_the_slice_stack():
    """A fractional z outside [0, 1) is folded, not clipped to the last slice."""
    gpts = (8, 8)
    tp = _tp(gpts, 1)
    T = _transmissions(4, gpts, seed=5)
    probes = torch.fft.ifft2(_aperture(gpts, 0.3))[None]
    common = dict(wavelength=_WAVELENGTH, gridsize=(4.0, 4.0), slice_distance=1.0)
    inside = multislice_transition_potential(
        probes, T, tp, np.array([[0.4, 0.6, 0.3]]), **common
    )
    wrapped = multislice_transition_potential(
        probes, T, tp, np.array([[0.4, 0.6, 2.3]]), **common
    )
    torch.testing.assert_close(wrapped, inside)


def test_a_single_site_row_is_accepted_without_a_leading_axis():
    gpts = (8, 8)
    tp, T, probes = _vacuum_setup(gpts)
    common = dict(wavelength=_WAVELENGTH, gridsize=(4.0, 4.0), slice_distance=0.0)
    flat = multislice_transition_potential(
        probes, T, tp, np.array([0.5, 0.5, 0.0]), **common
    )
    stacked = multislice_transition_potential(
        probes, T, tp, np.array([[0.5, 0.5, 0.0]]), **common
    )
    torch.testing.assert_close(flat, stacked)


# --------------------------------------------------------------------------- #
# prism_eels helpers
# --------------------------------------------------------------------------- #
def test_factor_pair_normalises_both_spellings():
    assert _factor_pair(3) == (3, 3)
    assert _factor_pair((2, 4)) == (2, 4)
    assert _factor_pair(np.array([2, 4])) == (2, 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_abs_complex_is_bit_identical_to_abs(device, dtype):
    """Enough samples that a 1-ulp disagreement cannot hide: CPU complex128
    ``hypot`` differs from ``abs`` on ~0.6 % of standard-normal inputs."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(100_000, dtype=dtype, generator=g).to(device)
    assert torch.equal(_abs_complex(z), z.abs())


def test_abs_complex_matches_abs_and_falls_back_under_autograd():
    z = torch.randn(4, 4, dtype=torch.complex128)
    r = torch.randn(4, 4, dtype=torch.float64)
    assert torch.equal(_abs_complex(r), r.abs())
    g = z.clone().requires_grad_(True)
    out = _abs_complex(g)
    assert out.grad_fn is not None and out.requires_grad
    torch.testing.assert_close(out, g.detach().abs())


# --------------------------------------------------------------------------- #
# ScatteringMatrix
# --------------------------------------------------------------------------- #
def test_scattering_matrix_rejects_invalid_interpolation_factors():
    gpts = (8, 8)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.3)
    with pytest.raises(ValueError, match="interpolation_factor must be >= 1"):
        ScatteringMatrix(pq, T, k, interpolation_factor=(0, 1))
    with pytest.raises(ValueError, match="must divide the grid"):
        ScatteringMatrix(pq, T, k, interpolation_factor=(3, 1))


def test_scattering_matrix_synthesises_the_fourier_shifted_probe():
    gpts = (16, 16)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.25)
    sm = ScatteringMatrix(pq, T, k)
    scan = torch.tensor([[3.0, 5.0], [1.5, -2.25]], dtype=torch.float64)
    got = sm.probe_at_current_plane(scan)
    base = torch.fft.ifft2(pq)
    ref = fourier_shift_stack(base, scan)
    torch.testing.assert_close(got, ref, atol=1e-12, rtol=1e-10)
    # The synthesis coefficients and the columns are the scalar-combinable form.
    torch.testing.assert_close(
        torch.einsum("pb,byx->pyx", sm.coeffs_at(scan), sm.S), got
    )


def test_scattering_matrix_advance_skips_vacuum_slices_bit_exactly():
    gpts = (8, 8)
    T = _transmissions(3, gpts, seed=6)
    T[1] = 1.0
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.3)
    sm = ScatteringMatrix(pq, T, k)
    sm.advance_to(3)
    manual = ScatteringMatrix(pq, torch.ones_like(T), k).S
    for j in range(3):
        manual = propagate(manual * T[j], k)
    assert torch.equal(sm.S, manual)
    assert sm._slice == 3


def test_scattering_matrix_interpolation_factor_keeps_a_beam_sublattice():
    gpts = (16, 16)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.3)
    full = ScatteringMatrix(pq, T, k)
    sub = ScatteringMatrix(pq, T, k, interpolation_factor=(2, 2))
    assert sub.S.shape[0] < full.S.shape[0]
    assert torch.all(sub.beam_gy % 2 == 0) and torch.all(sub.beam_gx % 2 == 0)
    # DFT decimation identity: keeping every f-th beam and restoring the 1/f^2
    # amplitude makes the synthesised probe the PERIODIC SUM of the full probe
    # over the (Ny/f, Nx/f) lattice -- exactly, not approximately.
    at_origin = torch.zeros(1, 2, dtype=torch.float64)
    probe_sub = sub.probe_at_current_plane(at_origin)[0]
    probe_full = full.probe_at_current_plane(at_origin)[0]
    f, (ny, nx) = 2, gpts
    periodic = sum(
        torch.roll(probe_full, shifts=(-m * ny // f, -n * nx // f), dims=(0, 1))
        for m in range(f)
        for n in range(f)
    )
    torch.testing.assert_close(probe_sub, periodic, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# PartitionedScatteringMatrix
# --------------------------------------------------------------------------- #
def test_partitioned_matrix_validates_its_geometry():
    gpts = (16, 16)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.3)
    with pytest.raises(ValueError, match="n_radial must be >= 2"):
        PartitionedScatteringMatrix(pq, T, k, n_radial=1)
    with pytest.raises(ValueError, match="interpolation_factor must be >= 1"):
        PartitionedScatteringMatrix(pq, T, k, n_radial=3, interpolation_factor=(1, 0))
    with pytest.raises(ValueError, match="must divide the grid"):
        PartitionedScatteringMatrix(pq, T, k, n_radial=3, interpolation_factor=(5, 1))


def test_select_parent_beams_covers_the_dc_beam_and_deduplicates():
    beams = np.array(
        [[y, x] for y in range(-3, 4) for x in range(-3, 4) if y * y + x * x <= 9],
        dtype=np.float64,
    )
    pidx = _select_parent_beams(beams, n_radial=2, n_angular=6)
    assert pidx.ndim == 1
    assert len(set(pidx.tolist())) == len(pidx)  # deduplicated
    assert 0.0 in {float(np.linalg.norm(beams[i])) for i in pidx}  # DC kept
    assert len(pidx) <= beams.shape[0]
    # More rings can only sample at least as many distinct parents.
    assert len(_select_parent_beams(beams, 3, 6)) >= len(pidx)


def test_partition_weights_are_cached_per_aperture_geometry():
    beams = np.array(
        [[y, x] for y in range(-3, 4) for x in range(-3, 4) if y * y + x * x <= 9],
        dtype=np.int64,
    )
    signed = beams.astype(np.float64)
    first = _partition_weights(beams, signed, 2, 6)
    second = _partition_weights(beams, signed, 2, 6)
    assert first[0] is second[0] and first[1] is second[1]
    # The NNW weights express every aperture beam as a partition of unity.
    np.testing.assert_allclose(first[1].sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("focal_backprop", [0.0, "0", "", None])
def test_focal_backprop_can_be_switched_off(focal_backprop):
    gpts = (24, 24)
    T = _transmissions(3, gpts, seed=7)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.2)
    sm = PartitionedScatteringMatrix(
        pq, T, k, n_radial=3, focal_backprop=focal_backprop
    )
    sm.advance_to(2)
    assert sm._backprop_n_half() == 0.0


def test_focal_backprop_is_zero_before_any_slice_is_traversed():
    gpts = (24, 24)
    T = _transmissions(3, gpts, seed=8)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(_aperture(gpts, 0.2), T, k, n_radial=3)
    assert sm._backprop_n_half() == 0.0


def test_focal_backprop_is_zero_in_the_full_parent_limit():
    """Every beam its own parent means there is no interpolation to correct."""
    gpts = (16, 16)
    T = _transmissions(2, gpts, seed=9)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.14)  # a small aperture, saturated by many rings
    sm = PartitionedScatteringMatrix(pq, T, k, n_radial=8, n_angular=12)
    sm.advance_to(1)
    assert sm.n_parents >= int(sm._by.shape[0])
    assert sm._backprop_n_half() == 0.0


def test_backprop_is_off_by_default_and_scales_for_a_float():
    gpts = (24, 24)
    T = torch.ones(3, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    pq = _aperture(gpts, 0.2)
    sm = PartitionedScatteringMatrix(pq, T, k, n_radial=3)
    sm.advance_to(2)
    assert sm._backprop_n_half() == 0.0  # focal_backprop defaults to None

    fixed = PartitionedScatteringMatrix(pq, T, k, n_radial=3, focal_backprop=0.5)
    fixed.advance_to(2)
    assert fixed._backprop_n_half() == pytest.approx(1.0)


@pytest.mark.parametrize(("depth", "expected"), [(1, 0), (2, 1), (3, 2), (4, 2)])
def test_centroid_backprop_is_half_the_traversed_depth(depth, expected):
    """``"centroid"`` is half the traversed depth, rounded to whole slices."""
    gpts = (24, 24)
    T = _transmissions(4, gpts, seed=11)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(
        _aperture(gpts, 0.2), T, k, n_radial=3, focal_backprop="centroid"
    )
    sm.advance_to(depth)
    assert sm._backprop_n_half() == expected


def test_focal_backprop_rejects_an_unknown_string():
    gpts = (16, 16)
    T = _transmissions(2, gpts, seed=11)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(
        _aperture(gpts, 0.2), T, k, n_radial=3, focal_backprop="median"
    )
    sm.advance_to(1)
    with pytest.raises(ValueError, match="centroid"):
        sm._backprop_n_half()


def test_fresnel_pow_inverts_itself_and_matches_the_per_slice_kernel():
    gpts = (16, 16)
    T = _transmissions(2, gpts, seed=12)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(_aperture(gpts, 0.2), T, k, n_radial=3)
    fwd = sm._fresnel_pow(1.0, 1.0)
    back = sm._fresnel_pow(1.0, -1.0)
    torch.testing.assert_close(fwd * back, torch.ones_like(fwd))
    # One forward slice reproduces the propagator itself (|kernel| == 1).
    torch.testing.assert_close(fwd, k, atol=1e-12, rtol=1e-10)


def test_reconstruct_columns_window_is_the_crop_of_the_full_reconstruction():
    gpts = (24, 24)
    T = _transmissions(2, gpts, seed=13)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(
        _aperture(gpts, 0.2), T, k, n_radial=3, focal_backprop=0.0
    )
    sm.advance_to(1)
    iy = torch.arange(4, 10)
    ix = torch.arange(11, 17)
    full = sm.reconstruct_columns()[:, iy][:, :, ix]
    torch.testing.assert_close(sm.reconstruct_columns_window(iy, ix), full)


def test_partitioned_beamlet_spectrum_carries_the_aperture_weights():
    gpts = (24, 24)
    T = torch.ones(1, *gpts, dtype=torch.complex128)
    k = propagator_kernel(gpts, (8.0, 8.0), _WAVELENGTH, 0.0, dtype=torch.complex128)
    sm = PartitionedScatteringMatrix(_aperture(gpts, 0.2), T, k, n_radial=3)
    Bq = sm._beamlet_spectrum()
    assert Bq.shape == (sm.n_parents, *gpts)
    torch.testing.assert_close(Bq[:, sm._by, sm._bx], sm._cw.t())
    # Nothing lands outside the aperture support.
    mask = torch.zeros(gpts, dtype=torch.bool)
    mask[sm._by, sm._bx] = True
    assert torch.equal(Bq[:, ~mask], torch.zeros_like(Bq[:, ~mask]))


# --------------------------------------------------------------------------- #
# Dual-S vs hybrid, and the option branches of the two entry points
# --------------------------------------------------------------------------- #
def _prism_case(gpts=(16, 16), nz=2):
    tp = _tp(gpts, 2, seed=21)
    T = _transmissions(nz, gpts, seed=22)
    pq = _aperture(gpts, 0.25)
    sites = np.array([[0.4, 0.6, 0.6]])
    scan = torch.tensor([[3.0, 5.0], [9.0, 2.0]], dtype=torch.float64)
    common = dict(
        wavelength=_WAVELENGTH,
        gridsize=(gpts[0] * 0.5, gpts[1] * 0.5),
        slice_distance=1.0,
    )
    return tp, T, pq, sites, scan, common


def test_dual_and_hybrid_agree_in_real_space_too():
    tp, T, pq, sites, scan, common = _prism_case()
    dual = prism_transition_potential(
        pq, T, tp, sites, scan, return_real_space=True, **common
    )
    hyb = prism_transition_potential_hybrid(
        pq, T, tp, sites, scan, return_real_space=True, **common
    )
    torch.testing.assert_close(dual, hyb, rtol=1e-8, atol=1e-12)


def test_dual_s_scan_chunking_does_not_change_the_result():
    tp, T, pq, sites, scan, common = _prism_case()
    whole = prism_transition_potential(pq, T, tp, sites, scan, **common)
    chunked = prism_transition_potential(pq, T, tp, sites, scan, scan_chunk=1, **common)
    torch.testing.assert_close(chunked, whole)


def test_slices_without_ionizable_atoms_are_skipped():
    """Only the ionization slice contributes; padding the stack changes nothing."""
    gpts = (16, 16)
    tp = _tp(gpts, 1, seed=23)
    pq = _aperture(gpts, 0.25)
    scan = torch.tensor([[3.0, 5.0]], dtype=torch.float64)
    common = dict(wavelength=_WAVELENGTH, gridsize=(8.0, 8.0), slice_distance=1.0)
    T = _transmissions(4, gpts, seed=24)
    # A single atom in slice 3 of 4: slices 0-2 hold no site and must be walked
    # through by ``advance_to`` alone.
    out = prism_transition_potential(
        pq, T, tp, np.array([[0.5, 0.5, 0.9]]), scan, **common
    )
    assert out.shape == (1, *gpts)
    assert float(out.sum()) > 0.0
    hyb = prism_transition_potential_hybrid(
        pq, T, tp, np.array([[0.5, 0.5, 0.9]]), scan, **common
    )
    torch.testing.assert_close(out, hyb, rtol=1e-8, atol=1e-12)


def test_partitioned_hybrid_ignores_the_interpolation_factor_argument_shape():
    """``partition`` composes with an int or a pair interpolation factor."""
    gpts = (24, 24)
    tp = _tp(gpts, 1, seed=25)
    T = _transmissions(2, gpts, seed=26)
    pq = _aperture(gpts, 0.2)
    sites = np.array([[0.45, 0.55, 0.6]])
    scan = torch.tensor([[4.0, 6.0]], dtype=torch.float64)
    common = dict(wavelength=_WAVELENGTH, gridsize=(12.0, 12.0), slice_distance=1.0)
    a = prism_transition_potential_hybrid(
        pq,
        T,
        tp,
        sites,
        scan,
        partition=dict(n_radial=3),
        interpolation_factor=1,
        **common,
    )
    b = prism_transition_potential_hybrid(
        pq,
        T,
        tp,
        sites,
        scan,
        partition=dict(n_radial=3),
        interpolation_factor=(1, 1),
        **common,
    )
    assert torch.equal(a, b)
    assert torch.isfinite(a).all() and float(a.sum()) > 0.0


# --------------------------------------------------------------------------- #
# Fused Warp kernels: the CPU lane can only assert that they decline
# --------------------------------------------------------------------------- #
def test_ramp_phase_fused_declines_on_cpu_and_for_the_wrong_dtype():
    w64 = torch.zeros(4, 4, dtype=torch.complex64)
    shifts = torch.zeros(2, 2, dtype=torch.float64)
    assert ramp_fused.ramp_phase_fused_supported(w64, shifts) is False


@pytest.mark.gpu
def test_ramp_phase_fused_matches_the_eager_ramp_on_cuda():
    device = torch.device("cuda")
    ny, nx, P = 8, 8, 3
    ky = torch.fft.fftfreq(ny, device=device, dtype=torch.float64)
    kx = torch.fft.fftfreq(nx, device=device, dtype=torch.float64)
    sy = torch.tensor([0.0, 3.5, -12.25], device=device, dtype=torch.float64)
    sx = torch.tensor([1.0, -2.0, 7.75], device=device, dtype=torch.float64)
    waves = torch.zeros(ny, nx, dtype=torch.complex64, device=device)
    assert ramp_fused.ramp_phase_fused_supported(waves, sy) is True
    got = ramp_fused.ramp_phase_fused(ky, kx, sy, sx, -2.0 * math.pi, (P, ny, nx))
    u = (-2.0 * math.pi) * (
        ky.view(1, ny, 1) * sy.view(P, 1, 1) + kx.view(1, 1, nx) * sx.view(P, 1, 1)
    )
    ref = torch.polar(torch.ones_like(u), u).to(torch.complex64)
    assert torch.equal(got, ref)
