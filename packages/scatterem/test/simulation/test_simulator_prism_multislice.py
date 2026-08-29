"""Multi-slice PRISM correctness gates for the from-structure Simulator (M2e).

G1: f=1 pure PRISM == multislice TIGHT for NZ in {1, 4, 16} (linearity).

f=1 pure PRISM is multislice-of-the-probe EXACTLY by linearity: the multislice
forward (transmit + Fresnel-propagate, slice by slice) is LINEAR in the incident
wave, and an f=1 pure-PRISM S-matrix spans the full BF-disk aperture with no
real-space cutout truncation. So PRISM(f=1, NZ) reconstructs the SAME exit wave
as multislice(NZ) for ANY NZ -- the only residual is the ~0.5% soft-aperture /
rounding floor (the ZernikeProbe tanh edge bleeding past the sharp f-lattice
aperture; identical floor to the M2d single-slice f=1 gate). If this FAILS, the
`prism_propagator` Δz or grid is wrong (or bin_factor != 1); DO NOT loosen the
tolerance -- diagnose the prism_propagator Δz/grid / reload_object ordering.

Note: f=1 requires scan=(1,1) since the FOV must fit N = f*pixels = pixels.

Both Warp env vars are required; TF32 is disabled in the numeric gate (PRISM's
batched complex einsum goes through cuBLAS GEMM, and TF32 on Ampere+ inflates
error well above the physics floor -- mirrors the M2d gates / test_prism_pure).
"""

import pytest
import torch

from scatterem.simulation.structure import Structure
from scatterem.simulation.simulator import Simulator

DATA = "test/simulation/data/small.xyz"


def _warp(mp):
    mp.setenv("SCATTEREM_USE_WARP", "1")
    mp.setenv("SCATTEREM_USE_WARP_FIXED_UNIQUE", "1")


# scan=(1,1) so FOV == detector == N == pixels = 64 at f=1 (object fits the sim
# grid). slice_thickness is set per-NZ below via the slice_thickness knob.
_G1_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
              pixels=(64, 64), scan=(1, 1), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_slices", [1, 4, 16])
def test_prism_f1_multislice_matches_multislice(monkeypatch, num_slices):
    """G1: PRISM(f=1, NZ) == multislice(NZ) TIGHT (soft-aperture floor) for NZ in {1,4,16}."""
    _warp(monkeypatch)
    # Disable TF32 so the gate measures physics (prism_propagator Δz/grid), not
    # TF32 GEMM noise -- mirrors test_prism_f1_matches_multislice.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    s = Structure.fromfile(DATA)
    unitcell_z = float(s.unitcell[2])
    # Drive the requested NZ exactly via the slice_thickness knob: choose Δz so
    # ceil(unitcell_z / Δz) == num_slices (a hair under unitcell_z/num_slices).
    if num_slices == 1:
        slice_thickness = None  # single-slice (no propagation)
    else:
        slice_thickness = unitcell_z / num_slices - 1e-6

    ms = Simulator(s, prism=False, slice_thickness=slice_thickness,
                   **_G1_KW).simulate()
    pr = Simulator(s, prism=True, prism_interpolation_factor=1,
                   slice_thickness=slice_thickness, **_G1_KW).simulate()

    assert ms.shape == (1, 1, 64, 64) and pr.shape == (1, 1, 64, 64)
    max_abs = float((pr - ms).abs().max())
    max_rel = max_abs / float(ms.abs().max())
    ratio = float(pr.sum() / ms.sum())
    print(f"\nG1 PRISM(f=1, NZ={num_slices}) vs multislice(NZ={num_slices}) "
          f"(FOV==det==N==64): max abs err = {max_abs:.3e}, "
          f"max rel err = {max_rel:.3e}, intensity ratio = {ratio:.5f}")

    # TIGHT: the same ~0.5% soft-aperture floor as the M2d single-slice f=1 gate,
    # for ANY NZ (linearity). Tolerance set just above that floor. Do NOT loosen
    # -- a larger error means a prism_propagator Δz/grid or bin_factor bug.
    torch.testing.assert_close(pr, ms, rtol=6e-3, atol=8e-2)


# G2 -- multi-slice f=2 (object > detector) vs multislice, CALIBRATED tolerance.
#
# Unlike f=1 (G1: exact-by-linearity, flat across NZ at the soft-aperture floor),
# f>1 PRISM carries the documented Ophus cutout-truncation error: the S-matrix
# lives on the N=f*pixels grid but ``prism_project`` crops a detector-sized (P x P)
# real-space window per scan position. With INTER-SLICE propagation the wave
# spreads, and at each slice the Fresnel propagator on the N grid wraps/truncates
# the part of the wave that has spread past the N/f cutout window -- a
# thickness-dependent error that GROWS with NZ (more propagation steps => more
# spreading past the window). This is the known, physical PRISM behavior, not a
# bug: thin specimens sit near the ~0.5% f=1 floor, thicker ones drift above it.
#
# Config: P=64, f=2, N=f*P=128, Δp=unitcell/pixels=5.43/64=0.08484 Å, scan=(4,4),
# scan_step=8 px -> FOV=ceil(3*8+64)=88 (object > 64-px detector, < N=128). The
# multislice reference detects on the same P=64 grid -- directly comparable.
_G2_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
              pixels=(64, 64), scan=(4, 4), scan_step_pixels=8.0, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_slices", [2, 8])
def test_prism_f2_multislice_matches_multislice(monkeypatch, num_slices):
    """G2: PRISM(f=2, NZ) vs multislice(NZ), object > detector; CALIBRATED tol.

    Reports how the cutout-truncation error grows with NZ. The tolerance is
    calibrated JUST above the achieved error at each NZ (documented below). A
    wild (>~30-50% rel) or non-physical error means a real f-scaling /
    prism_propagator bug -- STOP and diagnose, do NOT loosen.
    """
    _warp(monkeypatch)
    # Disable TF32 so the gate measures physics (cutout truncation + prism_propagator
    # Δz/grid), not TF32 GEMM noise -- mirrors the f=1 gates / test_prism_pure.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    s = Structure.fromfile(DATA)
    unitcell_z = float(s.unitcell[2])
    # Drive the requested NZ exactly via the slice_thickness knob.
    slice_thickness = unitcell_z / num_slices - 1e-6

    ms = Simulator(s, prism=False, slice_thickness=slice_thickness,
                   **_G2_KW).simulate()
    pr = Simulator(s, prism=True, prism_interpolation_factor=2,
                   slice_thickness=slice_thickness, **_G2_KW).simulate()

    assert ms.shape == (4, 4, 64, 64) and pr.shape == (4, 4, 64, 64)
    max_abs = float((pr - ms).abs().max())
    max_rel = max_abs / float(ms.abs().max())
    ratio = float(pr.sum() / ms.sum())
    print(f"\nG2 PRISM(f=2, NZ={num_slices}, N=128) vs multislice(NZ={num_slices}, det=64), "
          f"object FOV=88: max abs err = {max_abs:.3e}, "
          f"max rel err = {max_rel:.3e}, intensity ratio = {ratio:.5f}")

    # CALIBRATED (A6000, TF32 off; structure small.xyz, unitcell_z=5.43 Å, P=64,
    # f=2, N=128, scan=(4,4) step 8 px -> object FOV=88 > 64-px detector):
    #   NZ=2 (Δz=2.715 Å): max abs err = 1.217e-01, max rel err = 8.699e-03,
    #                       intensity ratio = 0.99874
    #   NZ=8 (Δz=0.679 Å): max abs err = 1.217e-01, max rel err = 8.698e-03,
    #                       intensity ratio = 0.99875
    #
    # ERROR GROWTH WITH NZ: essentially FLAT (8.699e-3 -> 8.698e-3 rel from NZ=2
    # to NZ=8) and pinned to the SAME ~0.87% floor + 0.99874 intensity ratio as
    # the M2d SINGLE-SLICE f=2 gate (test_prism_f2_matches_multislice: 8.67e-3
    # rel, 0.99873 ratio). This is the pure soft-aperture floor (the ZernikeProbe
    # tanh edge bleeding past the sharp f-lattice aperture), NOT a thickness-
    # dependent cutout-truncation error -- because small.xyz is only 5.43 Å thick,
    # the wave does not spread meaningfully past the N/f=64-px cutout window over
    # so little propagation, so the Ophus truncation error stays negligible at
    # every NZ. (Truncation error growing visibly with NZ requires a THICK
    # specimen; the thick-specimen behavior is demonstrated by the G4 crossover
    # benchmark, not this thin-specimen correctness gate.) The flatness + the
    # identical-to-single-slice floor is itself the sanity signal: a real f-scaling
    # / prism_propagator Δz/grid bug would inflate the error and break the ratio.
    #
    # Tolerance set just above the achieved error. Do NOT loosen further: a larger
    # error means a real bug (the gate is the arbiter, not the tolerance).
    torch.testing.assert_close(pr, ms, rtol=9e-3, atol=1.3e-1)
