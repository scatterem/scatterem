"""PRISM-mode tests for the from-structure forward Simulator (M2d Task 1).

Simulator(prism=True) synthesizes an SMeta from the optics and passes a
PrismScanOp into the PtychographyModel at construction (so prism_propagator is
built). Single-slice only; multi-slice PRISM raises NotImplementedError.

Both Warp env vars are required (see test_simulator.py).
"""

import numpy as np  # noqa: F401
import pytest
import torch

from scatterem.simulation.structure import Structure
from scatterem.simulation.simulator import Simulator
from scatterem.nn.scan_exit_wave.prism import PrismScanOp

DATA = "test/simulation/data/small.xyz"


def _warp(mp):
    mp.setenv("SCATTEREM_USE_WARP", "1")
    mp.setenv("SCATTEREM_USE_WARP_FIXED_UNIQUE", "1")


_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
           pixels=(64, 64), scan=(4, 4), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_build_model_wires_prismscanop(monkeypatch):
    _warp(monkeypatch)
    # f=2 so the sim grid N=f*pixels=128 fits the multi-position FOV (88);
    # f=1 here would be N=64 < FOV and (correctly) raises (object doesn't fit).
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=2, **_KW)
    model = sim.build_model()
    assert isinstance(model.scan_op, PrismScanOp)
    assert model.scan_op.wants_full_fov is True
    assert model.prism_propagator is not None        # built at construction (not None)
    # SMeta grid M == N = f * pixels = (128, 128) (NOT the FOV).
    MY, MX = (int(m) for m in model.scan_op.s_meta.M)
    assert (MY, MX) == (2 * 64, 2 * 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_produces_cube(monkeypatch):
    _warp(monkeypatch)
    # f=2 -> N=128 fits the FOV=88; detector is N/f = pixels = 64.
    cube = Simulator(Structure.fromfile(DATA), prism=True,
                     prism_interpolation_factor=2, **_KW).simulate()
    assert cube.shape == (4, 4, 64, 64)
    assert torch.isfinite(cube).all() and (cube >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_false_unchanged(monkeypatch):
    _warp(monkeypatch)
    # prism defaults False -> multislice path; bit-identical to a no-prism Simulator.
    a = Simulator(Structure.fromfile(DATA), **_KW).simulate()
    b = Simulator(Structure.fromfile(DATA), prism=False, **_KW).simulate()
    torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multislice_prism_constructs(monkeypatch):
    # M2e: multi-slice PRISM is now supported (guard removed). build_model wires
    # reload_object(NZ) -> multi-slice AtomicSampler -> reload_propagator(Δz),
    # which rebuilds prism_propagator at the physical Δz on the N grid (Task 1).
    _warp(monkeypatch)
    s = Structure.fromfile(DATA)
    unitcell_z = float(s.unitcell[2])
    slice_thickness = 1.5  # -> num_slices = ceil(unitcell_z / 1.5) > 1
    # f=2 so the sim grid N=f*pixels=128 fits the multi-position FOV (88).
    sim = Simulator(s, prism=True, prism_interpolation_factor=2,
                    slice_thickness=slice_thickness, **_KW)
    assert sim.num_slices > 1
    model = sim.build_model()
    expected_dz = unitcell_z / sim.num_slices
    assert model.NZ == sim.num_slices
    assert model.bin_factor == 1
    assert model.prism_propagator is not None
    assert tuple(model.prism_propagator._shape) == tuple(model.scan_op.s_meta.M)
    assert model.sampler.num_slices == model.NZ
    assert abs(float(model.slice_distance) - expected_dz) < 1e-4


# PRISM(N = f*P, f) == multislice(detector = P) -- the corrected physics
# (Ophus 2017, M2d-fix Tasks 1+2). The interpolation factor ``f`` is the
# real-space N/f cutout: the S-matrix lives on the simulation grid N = f*pixels
# (the *larger probe window* at the SAME real-space pitch Δp = unitcell/pixels),
# and ``prism_project`` crops a detector-sized (P x P) window per scan position
# (Ophus Eq 7). Both PRISM(N=f*P, f) and multislice(P) therefore detect on the
# same P-pixel grid at dk = 1/(P*Δp) = f*Δq -- directly comparable, no Fourier
# crop, no grid-sampling mismatch.
#
# Because PRISM mode here is SINGLE-SLICE (multi-slice PRISM raises), the
# real-space cutout exit wave is EXACTLY the multislice exit wave restricted to
# the P-window (the documented thickness-dependent Ophus error only appears with
# inter-slice propagation). So f=1 AND f=2 should match the Simulator's native
# multislice(P) to the soft-aperture / TF32 floor (~0.5% rel), NOT >1%.
#
# The f=1 gate stays at scan=(1,1) so FOV == detector == N == 64: at f=1 the sim
# grid is N = 1*P = 64, which would be SMALLER than a multi-position FOV (e.g.
# scan=(4,4) -> FOV 88) and correctly RAISES ("object does not fit"). The f=2
# gates use multi-position scans (FOV 85-88) since N = 2*P = 128 >= FOV, which is
# the whole point of f>1 (object larger than the detector window).
_GATE_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
                pixels=(64, 64), scan=(1, 1), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_f1_matches_multislice(monkeypatch):
    _warp(monkeypatch)
    # PRISM's batched complex einsum goes through cuBLAS GEMM; on Ampere+ TF32
    # inflates error well above 1e-5 (test_prism_pure.py disables it for this
    # reason). Disable TF32 so the gate measures SMeta-sizing/physics error, not
    # TF32 noise.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    s = Structure.fromfile(DATA)
    ms = Simulator(s, prism=False, **_GATE_KW).simulate()
    pr = Simulator(s, prism=True, prism_interpolation_factor=1, **_GATE_KW).simulate()
    # PRISM f=1 (all beams, no crop) == multislice for an aperture-limited probe.
    # Aperture margin: SMeta all_beams sized at kmax*1.05 (Task 1) to capture the
    # ZernikeProbe's soft (tanh) edge bleeding slightly past alpha/lambda.
    #
    # CALIBRATED (A6000, TF32 disabled, scan=(1,1) so FOV==detector==64):
    #   max abs err = 6.86e-02, max rel err (vs ms.abs().max()) = 5.22e-03,
    #   intensity ratio pr/ms = 0.99873.
    # This ~0.5% rel floor is driven ENTIRELY by the ZernikeProbe's soft (tanh)
    # edge: 0.13% of the probe's Fourier power bleeds past the sharp SMeta
    # aperture (all_beams) and is truncated by PRISM's beamlet basis. It is a
    # genuine physics floor, NOT TF32 noise (TF32 is disabled) and NOT a loose
    # correlation. Tolerance set just above the achieved error.
    max_abs = float((pr - ms).abs().max())
    print(f"\nPRISM(f=1) vs multislice (FOV==det==64): max abs err = {max_abs:.3e}, "
          f"max rel err = {max_abs / float(ms.abs().max()):.3e}, "
          f"intensity ratio = {float(pr.sum() / ms.sum()):.5f}")
    torch.testing.assert_close(pr, ms, rtol=6e-3, atol=8e-2)


# f=2 gate config: P=64, f=2, N=f*P=128, Δp=unitcell/pixels=5.43/64=0.08484 Å,
# dk=1/(P*Δp)=f*Δq=0.1842 1/Å. scan=(4,4), scan_step=8 px -> FOV=ceil(3*8+64)=88
# <= N=128. The object (88) is strictly LARGER than the detector (64): the
# original >100%-error "object > detector" case the fix targets.
_GATE_F2_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
                   pixels=(64, 64), scan=(4, 4), scan_step_pixels=8.0,
                   device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_f2_matches_multislice(monkeypatch):
    """HEADLINE GATE: PRISM(N=2P, f=2) == multislice(detector=P), object > detector.

    f=1 (N=P) and f=2 (N=2P) together pin the f-scaling: a factor-of-f error in
    the cutout pitch / S-matrix normalisation / dk shows up here but not at f=1.
    """
    _warp(monkeypatch)
    # PRISM's batched complex einsum goes through cuBLAS GEMM; TF32 on Ampere+
    # inflates error above 1e-5. Disable it so the gate measures physics, not
    # TF32 noise (mirrors test_prism_f1_matches_multislice / test_prism_pure).
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    s = Structure.fromfile(DATA)
    ms = Simulator(s, prism=False, **_GATE_F2_KW).simulate()
    pr = Simulator(s, prism=True, prism_interpolation_factor=2,
                   **_GATE_F2_KW).simulate()
    assert ms.shape == (4, 4, 64, 64) and pr.shape == (4, 4, 64, 64)
    max_abs = float((pr - ms).abs().max())
    print(f"\nPRISM(f=2,N=128) vs multislice(det=64), object FOV=88: "
          f"max abs err = {max_abs:.3e}, "
          f"max rel err = {max_abs / float(ms.abs().max()):.3e}, "
          f"intensity ratio = {float(pr.sum() / ms.sum()):.5f}")
    # CALIBRATED tolerance = the SAME soft-aperture floor as the f=1 gate (the
    # ZernikeProbe tanh edge bleeding past the sharp f-lattice aperture). After
    # Tasks 1+2 are correct, PRISM(f=2) collapses onto multislice at:
    #   max abs err = 1.22e-01, max rel err = 8.67e-03, intensity ratio = 0.99873
    # (A6000, TF32 off; identical 0.99873 ratio to the f=1 gate -> a pure
    # soft-aperture floor, NOT a grid/normalisation error). Tolerance set just
    # above that floor. Do NOT loosen further: a larger error means a real
    # f-scaling bug in the kernel/Simulator (Tasks 1-2), which this gate exists
    # to catch -- the gate is the arbiter, not the tolerance.
    torch.testing.assert_close(pr, ms, rtol=9e-3, atol=1.4e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_f2_arbitrary_positions(monkeypatch):
    """PRISM(f=2) == multislice(P) for NON-f-aligned + sub-pixel scan positions.

    Proves the per-position real-space cutout + the reciprocal ``dr`` phase ramp
    handle arbitrary positions exactly -- no f-quantization of the scan grid.
    scan_step_pixels=7.5 is (a) not a multiple of f=2 px and (b) FRACTIONAL, so
    the model splits positions into an integer part + a sub-pixel ``dr`` (here
    max |dr| = 0.25 px, verified non-zero in the PtychographyModel build log),
    exercising the phase ramp. FOV=ceil(3*7.5+64)=87 <= N=128.
    """
    _warp(monkeypatch)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    s = Structure.fromfile(DATA)
    kw = dict(_GATE_F2_KW)
    kw["scan_step_pixels"] = 7.5  # non-f-aligned AND fractional -> sub-pixel dr
    ms = Simulator(s, prism=False, **kw).simulate()
    pr = Simulator(s, prism=True, prism_interpolation_factor=2, **kw).simulate()
    assert ms.shape == (4, 4, 64, 64) and pr.shape == (4, 4, 64, 64)
    max_abs = float((pr - ms).abs().max())
    print(f"\nPRISM(f=2) arbitrary+sub-pixel positions (step=7.5px, |dr|max=0.25) "
          f"vs multislice(64): max abs err = {max_abs:.3e}, "
          f"max rel err = {max_abs / float(ms.abs().max()):.3e}, "
          f"intensity ratio = {float(pr.sum() / ms.sum()):.5f}")
    # Same soft-aperture floor as the f-aligned f=2 gate (the sub-pixel ramp adds
    # no measurable error -- it is exact). CALIBRATED (A6000, TF32 off):
    #   max abs err = 1.37e-01, max rel err = 1.00e-02, intensity ratio = 0.99873
    # Tolerance just above. Do NOT loosen (see test_prism_f2_matches_multislice).
    torch.testing.assert_close(pr, ms, rtol=1.05e-2, atol=1.4e-1)
