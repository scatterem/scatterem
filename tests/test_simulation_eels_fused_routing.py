"""The perf-lab fast arms of the EELS simulators, as far as a CPU host can see them.

Three kinds of fast arm landed together, and each has a CPU-visible half:

* **Fused Warp tails** (``abs2_acc_fused``, ``abs2_sum_acc_fused``).  The kernels are spelled with CUDA ``__*_rn``
  intrinsics, so they cannot launch on a CPU device at all; what a CPU lane can
  check is that every gate *declines* here, and that each call site, when its gate
  accepts, hands the kernel the right operands in the right order and uses what
  comes back.  The latter is asserted by standing a reference implementation in
  for the kernel and requiring the result to equal the eager arm -- the stand-in
  replaces only the launch, and every other number comes from the library.  The
  bit-exactness of the kernels themselves is the GPU lane's job.
* **The blocked, spectrum-folding exit walk** in ``_slice_exit_wave``.  Blocking
  is refused off-CUDA because its argument is about L2, but the arithmetic is
  device-independent: forcing a block size on CPU must give exactly the dense
  walk, for every ``consume``/``spectrum`` combination, and ``consume`` must
  really write back into the caller's storage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scatterem.physics.propagation import propagator_kernel
from scatterem.simulation._wave import exit_walk
from scatterem.simulation._wave.exit_walk import (
    _slice_exit_wave,
    _unit_transmission_slices,
)
from scatterem.simulation._wave.prism_beams import _abs_complex
from scatterem.simulation.eels.forward import (
    abs2_acc_fused,
    abs2_sum_acc_fused,
    multislice_eels,
    prism_eels,
)
from scatterem.simulation.eels.forward.multislice_eels import (
    multislice_transition_potential,
)
from scatterem.simulation.eels.forward.prism_eels import (
    prism_transition_potential,
    prism_transition_potential_hybrid,
)
from scatterem.simulation.eels.physics.transition_potentials import (
    TransitionPotentials,
)

_WAVELENGTH = 0.0037  # A, ~100 kV


def _tp(gpts, n_trans=2, *, seed=0) -> TransitionPotentials:
    g = torch.Generator().manual_seed(seed)
    arr = torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64) + 1j * (
        torch.randn(n_trans, *gpts, generator=g, dtype=torch.float64)
    )
    return TransitionPotentials(
        array=arr.to(torch.complex128),
        Z=8,
        energy=100_000.0,
        sampling=(0.1, 0.1),
        energy_losses=np.full(n_trans, 532.0),
    )


def _transmissions(nz, gpts, *, seed=0, dtype=torch.complex128):
    g = torch.Generator().manual_seed(seed)
    phase = 0.25 * torch.rand(nz, *gpts, generator=g, dtype=torch.float64)
    return torch.exp(1j * phase).to(dtype)


def _aperture(gpts, radius=0.25):
    qy, qx = np.meshgrid(
        np.fft.fftfreq(gpts[0]), np.fft.fftfreq(gpts[1]), indexing="ij"
    )
    pq = torch.as_tensor((qy**2 + qx**2) <= radius**2, dtype=torch.complex128)
    return pq / (pq.abs().pow(2).sum() / pq.numel()).sqrt()


# --------------------------------------------------------------------------- #
# The gates: every one declines on a CPU host
# --------------------------------------------------------------------------- #
def test_abs2_accumulate_gate_declines_on_cpu():
    psi = torch.zeros(2, 4, 4, dtype=torch.complex64)
    acc = torch.zeros(2, 4, 4, dtype=torch.float64)
    assert abs2_acc_fused.abs2_accumulate_fused_supported(psi, acc) is False


def test_abs2_sum_accumulate_gate_declines_on_cpu():
    psi = torch.zeros(2, 3, 4, 4, dtype=torch.complex64)
    acc = torch.zeros(2, 4, 4, dtype=torch.float64)
    assert abs2_sum_acc_fused.abs2_sum_accumulate_fused_supported(psi, acc) is False




# --------------------------------------------------------------------------- #
# The call sites: when a gate accepts, the kernel's answer is the eager answer
# --------------------------------------------------------------------------- #
class _Recorder:
    """A reference stand-in for a fused launch that records its operands."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, acc, psi):
        self.calls += 1
        self.fn(acc, psi)


def _abs2_ref(acc, psi):
    acc += _abs_complex(psi).pow(2)


def _abs2_sum_ref(acc, psi):
    acc += psi.abs().pow(2).sum(dim=1)


def _prism_abs2_sum_ref(acc, psi):
    # The hybrid's eager arm spells ``abs`` as ``_abs_complex``.
    acc += _abs_complex(psi).pow(2).sum(dim=1)


@pytest.mark.parametrize("return_real_space", [False, True])
def test_multislice_routes_its_tail_through_the_fused_sum_when_accepted(
    monkeypatch, return_real_space
):
    gpts = (8, 8)
    tp = _tp(gpts, 2, seed=1)
    T = _transmissions(2, gpts, seed=2)
    probes = torch.fft.ifft2(_aperture(gpts))[None].repeat(2, 1, 1)
    sites = np.array([[0.3, 0.6, 0.25], [0.7, 0.2, 0.75]])
    kw = dict(
        wavelength=_WAVELENGTH,
        gridsize=(4.0, 4.0),
        slice_distance=1.0,
        return_real_space=return_real_space,
    )
    eager = multislice_transition_potential(probes, T, tp, sites, **kw)

    rec = _Recorder(_abs2_sum_ref)
    monkeypatch.setattr(
        abs2_sum_acc_fused, "abs2_sum_accumulate_fused_supported", lambda s, a: True
    )
    monkeypatch.setattr(multislice_eels, "abs2_sum_accumulate_fused", rec)
    fused = multislice_transition_potential(probes, T, tp, sites, **kw)
    assert rec.calls == 2  # one per site
    torch.testing.assert_close(fused, eager, rtol=0, atol=0)


def _prism_case(gpts=(16, 16)):
    tp = _tp(gpts, 2, seed=21)
    T = _transmissions(2, gpts, seed=22)
    sites = np.array([[0.4, 0.6, 0.6]])
    scan = torch.tensor([[3.0, 5.0], [9.0, 2.0]], dtype=torch.float64)
    common = dict(wavelength=_WAVELENGTH, gridsize=(8.0, 8.0), slice_distance=1.0)
    return _aperture(gpts), T, tp, sites, scan, common


def test_dual_s_synthesis_routes_through_the_fused_abs2_when_accepted(monkeypatch):
    pq, T, tp, sites, scan, common = _prism_case()
    # Same chunking on both arms: a batched FFT's last bits can depend on the
    # batch size on some CPUs, and this test is about the routing, not chunking.
    eager = prism_transition_potential(pq, T, tp, sites, scan, scan_chunk=1, **common)
    rec = _Recorder(_abs2_ref)
    monkeypatch.setattr(
        abs2_acc_fused, "abs2_accumulate_fused_supported", lambda e, d: True
    )
    monkeypatch.setattr(prism_eels, "abs2_accumulate_fused", rec)
    fused = prism_transition_potential(pq, T, tp, sites, scan, scan_chunk=1, **common)
    assert rec.calls == 2 * 2  # two transitions x two one-row scan chunks
    torch.testing.assert_close(fused, eager, rtol=0, atol=0)


def test_hybrid_routes_its_tail_through_the_fused_sum_when_accepted(monkeypatch):
    pq, T, tp, sites, scan, common = _prism_case()
    eager = prism_transition_potential_hybrid(pq, T, tp, sites, scan, **common)
    rec = _Recorder(_prism_abs2_sum_ref)
    monkeypatch.setattr(
        abs2_sum_acc_fused, "abs2_sum_accumulate_fused_supported", lambda s, a: True
    )
    monkeypatch.setattr(prism_eels, "abs2_sum_accumulate_fused", rec)
    fused = prism_transition_potential_hybrid(pq, T, tp, sites, scan, **common)
    assert rec.calls == 1
    torch.testing.assert_close(fused, eager, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# The blocked exit walk, forced on CPU
# --------------------------------------------------------------------------- #
def _walk_case(dtype=torch.complex128, t_dtype=torch.complex128, n=5):
    gpts = (8, 8)
    T = _transmissions(3, gpts, seed=5, dtype=t_dtype)
    T[1] = 1.0  # one unit slice, so the skip list is exercised too
    k = propagator_kernel(gpts, (4.0, 4.0), _WAVELENGTH, 1.0, dtype=torch.complex128)
    g = torch.Generator().manual_seed(6)
    w = torch.randn(n, *gpts, generator=g, dtype=torch.float64).to(dtype)
    return w, T, k, _unit_transmission_slices(T)


def _dense(w, T, k, unit, spectrum):
    out = _slice_exit_wave(w.clone(), T, k, 0, unit)
    return torch.fft.fft2(out, dim=(-2, -1)) if spectrum else out


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("spectrum", [False, True])
def test_blocked_walk_equals_the_dense_walk(monkeypatch, consume, spectrum):
    w, T, k, unit = _walk_case()
    ref = _dense(w, T, k, unit, spectrum)
    monkeypatch.setattr(exit_walk, "_exit_block_planes", lambda *a: 2)
    monkeypatch.setattr(exit_walk, "_CUFFT_BATCH_EXACT", {})
    src = w.clone()
    got = _slice_exit_wave(src, T, k, 0, unit, consume=consume, spectrum=spectrum)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    # ``consume`` hands the storage over: the result lands in it.  Without it the
    # caller's stack is left alone.
    assert (got.data_ptr() == src.data_ptr()) is consume
    if not consume:
        assert torch.equal(src, w)


def test_blocked_walk_keeps_a_promoting_dtype_out_of_the_callers_storage(monkeypatch):
    """complex64 waves against complex128 slices: the result cannot go back."""
    w, T, k, unit = _walk_case(dtype=torch.complex64)
    ref = _dense(w, T, k, unit, True)
    monkeypatch.setattr(exit_walk, "_exit_block_planes", lambda *a: 2)
    monkeypatch.setattr(exit_walk, "_CUFFT_BATCH_EXACT", {})
    src = w.clone()
    got = _slice_exit_wave(src, T, k, 0, unit, consume=True, spectrum=True)
    assert got.dtype == torch.complex128
    assert got.data_ptr() != src.data_ptr()
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("consume", [False, True])
def test_a_batch_variant_fft_falls_back_to_the_dense_walk(monkeypatch, consume):
    w, T, k, unit = _walk_case()
    ref = _dense(w, T, k, unit, True)
    monkeypatch.setattr(exit_walk, "_exit_block_planes", lambda *a: 2)
    monkeypatch.setattr(exit_walk, "_cufft_batch_bit_invariant", lambda flat, b: False)
    got = _slice_exit_wave(w.clone(), T, k, 0, unit, consume=consume, spectrum=True)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
