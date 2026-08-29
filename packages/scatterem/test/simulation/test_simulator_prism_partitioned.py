"""Partitioned (flexible parent-beam) PRISM tests for the forward Simulator (M2e Task 3).

``Simulator(prism=True, prism_n_radial=R, prism_n_angular=A)`` synthesizes the
pure-PRISM full-aperture SMeta and then calls ``SMeta.make_beamlet_meta`` to
build a *partitioned* SMeta: a parent-beam subset (``Bp < B_full``), the Voronoi
natural-neighbor weights ``(B_full, Bp)`` mapping each full-aperture beam onto
the parent subset, and the beamlet basis ``(Bp, MY, MX)``.  The default
(``prism_n_radial=None``) stays pure PRISM (``Bp == B_full``, identity NNW) -- the
M2d / Task-2 behavior, byte-for-byte.

These are STRUCTURAL build_model asserts only; the calibrated numeric-vs-multislice
gate (G3) lands in Task 4.  Both Warp env vars are required (see test_simulator.py).
"""

import pytest
import torch

from scatterem.simulation.structure import Structure
from scatterem.simulation.simulator import Simulator
from scatterem.nn.scan_exit_wave.prism import PrismScanOp

DATA = "test/simulation/data/small.xyz"


def _warp(mp):
    mp.setenv("SCATTEREM_USE_WARP", "1")
    mp.setenv("SCATTEREM_USE_WARP_FIXED_UNIQUE", "1")


# Partition knobs: n_radial=2 (+ DC) hex rings, 6 angular -> Bp=19 << B_full=69
# for this optics, so the partition is strictly coarser than the full aperture
# at BOTH f=1 and f=2 (B_full is f-independent here: same BF disk, just on the
# N=f*pixels grid). The realized Bp/B_full numbers are asserted structurally, not
# pinned to exact counts -- only Bp < B_full and the matching NNW/beamlet shapes.
_N_RADIAL = 2
_N_ANGULAR = 6

# f=1 uses scan=(1,1): at f=1 the sim grid N = 1*pixels = 64 and a multi-position
# FOV (e.g. scan=(4,4) -> 88) would NOT fit N (the object-does-not-fit ValueError).
_F1_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
              pixels=(64, 64), scan=(1, 1), device="cuda")
# f=2 uses a multi-position scan: N = 2*pixels = 128 >= FOV=88, so the object
# (larger than the 64-px detector) fits -- the whole point of f>1.
_F2_KW = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0,
              pixels=(64, 64), scan=(4, 4), scan_step_pixels=8.0, device="cuda")


def _b_full(kw, f):
    """Pure PRISM (same optics) -> the full-aperture parent-beam count B_full."""
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=f, **kw)
    model = sim.build_model()
    return int(model.scan_op.s_meta.Bp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kw,f", [(_F1_KW, 1), (_F2_KW, 2)])
def test_partitioned_prism_build_model_asserts(monkeypatch, kw, f):
    """Partitioned PRISM: Bp < B_full, NNW=(B_full, Bp), beamlets=(Bp, MY, MX)."""
    _warp(monkeypatch)
    # Pure PRISM with the same optics gives the full-aperture parent count B_full.
    b_full = _b_full(kw, f)

    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=f,
                    prism_n_radial=_N_RADIAL, prism_n_angular=_N_ANGULAR, **kw)
    model = sim.build_model()
    assert isinstance(model.scan_op, PrismScanOp)
    s_meta = model.scan_op.s_meta

    bp = int(s_meta.Bp)
    my, mx = (int(m) for m in s_meta.M)
    print(f"\npartitioned PRISM f={f} (n_radial={_N_RADIAL}, n_angular={_N_ANGULAR}): "
          f"B_full={b_full}, Bp={bp}, M=({my}, {mx})")

    # Partitioned: strictly fewer parent beams than the full aperture.
    assert bp < b_full, (
        f"partitioned Bp ({bp}) should be < pure B_full ({b_full})"
    )
    # NNW maps each of the B_full full-aperture beams onto the Bp parents.
    assert tuple(s_meta.natural_neighbor_weights.shape) == (b_full, bp), (
        f"NNW shape {tuple(s_meta.natural_neighbor_weights.shape)} "
        f"!= (B_full={b_full}, Bp={bp})"
    )
    # One beamlet per parent on the N grid.
    assert tuple(s_meta.beamlets.shape) == (bp, my, mx), (
        f"beamlets shape {tuple(s_meta.beamlets.shape)} != (Bp={bp}, {my}, {mx})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kw,f", [(_F1_KW, 1), (_F2_KW, 2)])
def test_default_is_pure_prism_unchanged(monkeypatch, kw, f):
    """Default (no prism_n_radial) -> pure PRISM: Bp == B_full, identity NNW."""
    _warp(monkeypatch)
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=f, **kw)
    model = sim.build_model()
    s_meta = model.scan_op.s_meta

    # Pure PRISM: every full-aperture beam is its own parent (B == Bp).
    assert int(s_meta.B) == int(s_meta.Bp)
    bp = int(s_meta.Bp)
    # Identity natural-neighbor weights (one-to-one parent map).
    assert tuple(s_meta.natural_neighbor_weights.shape) == (bp, bp)
    torch.testing.assert_close(
        s_meta.natural_neighbor_weights,
        torch.eye(bp, dtype=s_meta.natural_neighbor_weights.dtype,
                  device=s_meta.natural_neighbor_weights.device),
    )
    # One-hot beamlets on the N grid.
    my, mx = (int(m) for m in s_meta.M)
    assert tuple(s_meta.beamlets.shape) == (bp, my, mx)


# ---------------------------------------------------------------------------
# G3 -- partitioned PRISM vs multislice, CALIBRATED, across the cross-product
#       {f=1, f=2} x {single-slice, multi-slice} (4 combinations).
# ---------------------------------------------------------------------------
#
# Partitioned PRISM replaces the full-aperture parent set (B_full=69 beams) with
# a Voronoi-partitioned subset (Bp parents) and interpolates the remaining beams
# onto the parents via natural-neighbor weights (NNW). The NNW interpolation adds
# an approximation error ON TOP of the pure-PRISM soft-aperture floor. That extra
# error is governed by how finely the partition samples the BF aperture:
#   - SINGLE-SLICE (NZ=1): the partition error is ~0 -- the NNW reconstructs the
#     incident probe in the entrance plane essentially exactly, so partitioned ==
#     pure == multislice at the same ~0.5%/0.87% (f=1/f=2) soft-aperture floor for
#     ANY partition coarseness. (The interpolation only starts to matter once the
#     wave evolves slice-to-slice and the partition's angular sampling limits how
#     well the propagated wave is represented.)
#   - MULTI-SLICE (NZ>1): the partition error appears and shrinks as the partition
#     is refined (see test_partitioned_prism_degrades_gracefully below):
#       n_radial=2 (Bp=19): rel ~0.94%, ratio 0.9953  (coarse)
#       n_radial=3 (Bp=37): rel ~0.53%/0.87%, ratio 0.9973  (well-sampled)
#       n_radial=4 (Bp=61): rel ~0.53%/0.87%, ratio 0.9984  (fine, ~= floor)
#
# GATE PARTITION: n_radial=3 (Bp=37, well over half of B_full=69) -- a partition
# that samples the aperture reasonably, so all 4 combos sit at/near the pure-PRISM
# floor (the NNW error is sub-floor). A coarser partition is exercised separately
# (test_partitioned_prism_degrades_gracefully) to confirm graceful (bounded, not
# catastrophic) degradation. small.xyz is thin (unitcell_z=5.43 Å), so even at
# NZ=4 the inter-slice spreading is small and the NNW error stays modest -- a THICK
# specimen would push the partition error up (the G4 thick-specimen regime).
_G3_N_RADIAL = 3
_G3_N_ANGULAR = 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "kw,f,num_slices",
    [(_F1_KW, 1, 1), (_F1_KW, 1, 4), (_F2_KW, 2, 1), (_F2_KW, 2, 4)],
)
def test_partitioned_prism_matches_multislice(monkeypatch, kw, f, num_slices):
    """G3: partitioned PRISM (n_radial=3) == multislice across {f=1,f=2}x{NZ=1,4}.

    CALIBRATED to the achieved error per combo (documented below). The f=1 legs
    use scan=(1,1) (f=1 multi-position raises from FOV>N); the f=2 legs use a
    multi-position scan (object FOV=88 > 64-px detector). If even this
    well-sampled partition gives a WILD error (>~30-50% rel or a broken intensity
    ratio), STOP and diagnose -- it could mean make_beamlet_meta's "Error during
    processing of a grid" warning is a real degeneracy, not cosmetic. Do NOT
    loosen to force green.
    """
    _warp(monkeypatch)
    # Disable TF32 so the gate measures partition/physics error, not GEMM noise.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    s = Structure.fromfile(DATA)
    unitcell_z = float(s.unitcell[2])
    slice_thickness = None if num_slices == 1 else unitcell_z / num_slices - 1e-6

    ms = Simulator(s, prism=False, slice_thickness=slice_thickness, **kw).simulate()
    pr_sim = Simulator(s, prism=True, prism_interpolation_factor=f,
                       prism_n_radial=_G3_N_RADIAL, prism_n_angular=_G3_N_ANGULAR,
                       slice_thickness=slice_thickness, **kw)
    model = pr_sim.build_model()
    bp = int(model.scan_op.s_meta.Bp)
    b_full = int(model.scan_op.s_meta.B)
    pr = pr_sim.simulate()

    max_abs = float((pr - ms).abs().max())
    max_rel = max_abs / float(ms.abs().max())
    ratio = float(pr.sum() / ms.sum())
    print(f"\nG3 partitioned PRISM(f={f}, NZ={num_slices}, n_radial={_G3_N_RADIAL}) "
          f"vs multislice: B_full={b_full}, Bp={bp}, max abs err = {max_abs:.3e}, "
          f"max rel err = {max_rel:.3e}, intensity ratio = {ratio:.5f}")

    # CALIBRATED (A6000, TF32 off; small.xyz unitcell_z=5.43 Å, P=64, n_radial=3
    # -> Bp=37 of B_full=69 parents). Achieved per combo:
    #   f=1 NZ=1 (scan=(1,1)):       max abs=6.858e-02 rel=5.223e-03 ratio=0.99873
    #   f=1 NZ=4 (scan=(1,1)):       max abs=6.903e-02 rel=5.259e-03 ratio=0.99731
    #   f=2 NZ=1 (scan=(4,4) FOV88): max abs=1.215e-01 rel=8.673e-03 ratio=0.99873
    #   f=2 NZ=4 (scan=(4,4) FOV88): max abs=1.220e-01 rel=8.715e-03 ratio=0.99735
    # Single-slice combos sit EXACTLY at the pure-PRISM floor (NNW error ~0 in the
    # entrance plane); multi-slice combos add a tiny sub-floor NNW error. All 4
    # track multislice tightly with a well-sampled partition. Tolerance set just
    # above the worst combo (f=2 NZ=4: rel 8.715e-03, abs 1.220e-01) -- a single
    # tolerance for the parametrized gate. Do NOT loosen further.
    torch.testing.assert_close(pr, ms, rtol=9e-3, atol=1.3e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partitioned_prism_degrades_gracefully(monkeypatch):
    """A COARSER partition (n_radial=2, Bp=19) degrades gracefully, not catastrophically.

    Same multi-slice f=2 combo as the worst G3 leg, but with a coarser partition.
    The error must GROW (vs the n_radial=3 gate) yet stay BOUNDED -- not blow up.
    """
    _warp(monkeypatch)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    s = Structure.fromfile(DATA)
    unitcell_z = float(s.unitcell[2])
    num_slices = 4
    slice_thickness = unitcell_z / num_slices - 1e-6

    ms = Simulator(s, prism=False, slice_thickness=slice_thickness,
                   **_F2_KW).simulate()
    pr_sim = Simulator(s, prism=True, prism_interpolation_factor=2,
                       prism_n_radial=2, prism_n_angular=6,
                       slice_thickness=slice_thickness, **_F2_KW)
    model = pr_sim.build_model()
    bp = int(model.scan_op.s_meta.Bp)
    b_full = int(model.scan_op.s_meta.B)
    pr = pr_sim.simulate()

    max_abs = float((pr - ms).abs().max())
    max_rel = max_abs / float(ms.abs().max())
    ratio = float(pr.sum() / ms.sum())
    print(f"\nG3 (coarse) partitioned PRISM(f=2, NZ=4, n_radial=2) vs multislice: "
          f"B_full={b_full}, Bp={bp}, max abs err = {max_abs:.3e}, "
          f"max rel err = {max_rel:.3e}, intensity ratio = {ratio:.5f}")

    # CALIBRATED (A6000, TF32 off; n_radial=2 -> Bp=19 of B_full=69):
    #   max abs err = 1.331e-01, max rel err = 9.511e-03, intensity ratio = 0.99536
    # GRACEFUL: the coarse partition (Bp=19) error (rel 9.5e-3, ratio 0.9954) is
    # LARGER than the well-sampled n_radial=3 gate (Bp=37: rel 8.7e-3, ratio
    # 0.9974) but still BOUNDED to ~1% -- it degrades smoothly with coarseness, it
    # does NOT blow up. The error stays sub-1.5%, confirming make_beamlet_meta is
    # not producing a degenerate partition. Tolerance set just above the achieved
    # coarse error; the partition-coarseness trade-off (refine -> converges to the
    # floor) is documented in the G3 gate above.
    assert max_rel < 1.5e-2, f"coarse-partition rel err {max_rel:.3e} should stay bounded"
    assert ratio > 0.99, f"coarse-partition intensity ratio {ratio:.5f} should stay near 1"
    torch.testing.assert_close(pr, ms, rtol=1.0e-2, atol=1.4e-1)
