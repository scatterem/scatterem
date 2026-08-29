"""PRISM grid-synthesis tests for the from-structure Simulator (M2d-fix Task 2).

These assert that ``Simulator(prism=True, prism_interpolation_factor=f)``
synthesizes the *correct* PRISM simulation grid:

* The S-matrix grid ``s_meta.M`` is ``N = f * pixels`` (the larger probe
  window at the SAME real-space pitch Δp), NOT the object FOV and NOT a
  ``prism_M``-padded FOV.
* The aperture is the **f-lattice** bright-field disk (Ophus 2017 Eq 6): only
  every f-th reciprocal-grid point inside ``|k| <= kmax*1.05``.  So the beam
  count ``s_meta.Bp`` is the dense-disk count / f² (and strictly LESS than the
  dense count for f >= 2).
* The detector window is ``pixels == N / f`` (dk = f·Δq).
* If the object FOV does not fit (``f*pixels < FOV``) the build RAISES a clear
  error telling the user to increase ``pixels``.

Both Warp env vars are required + CUDA (PrismScanOp construction needs them).
"""

import numpy as np
import pytest
import torch

from scatterem.simulation.structure import Structure
from scatterem.simulation.simulator import Simulator
from scatterem.utils.stem import fftfreq2

DATA = "test/simulation/data/small.xyz"


def _warp(mp):
    mp.setenv("SCATTEREM_USE_WARP", "1")
    mp.setenv("SCATTEREM_USE_WARP_FIXED_UNIQUE", "1")


# Shared optics. eV/aperture fixed; pixels/scan vary per test so that
# N = f*pixels >= FOV holds (FOV = detector + scan_step_pixels*(scan-1)).
_OPT = dict(eV=200e3, semiconvergence_angle=20e-3, defocus=0.0, device="cuda")


def _dense_disk_count(P, f, eV, alpha, dx0, dx1):
    """Reference: count of *all* reciprocal-grid points (dense disk) on the
    N=f*P grid inside |k| <= kmax*1.05, with the corner-origin (centered=False)
    convention SMeta/_build_smeta use."""
    from scatterem.utils.energy import energy2wavelength

    wavelength = float(energy2wavelength(eV))
    N0, N1 = f * P, f * P
    qf = fftfreq2((N0, N1), [dx0, dx1], centered=False, device="cuda")
    qmag = torch.linalg.norm(qf, dim=0)
    kmax = alpha / wavelength
    return int((qmag <= kmax * 1.05).sum())


def _flattice_count(P, f, eV, alpha, dx0, dx1):
    """Reference: count of f-lattice reciprocal points (every f-th corner-origin
    index) inside |k| <= kmax*1.05 on the N=f*P grid."""
    from scatterem.utils.energy import energy2wavelength

    wavelength = float(energy2wavelength(eV))
    N0, N1 = f * P, f * P
    qf = fftfreq2((N0, N1), [dx0, dx1], centered=False, device="cuda")
    qmag = torch.linalg.norm(qf, dim=0)
    kmax = alpha / wavelength
    iy = torch.arange(N0, device="cuda")[:, None]
    ix = torch.arange(N1, device="cuda")[None, :]
    lattice = ((iy % f) == 0) & ((ix % f) == 0)
    return int(((qmag <= kmax * 1.05) & lattice).sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "P,scan,ssp",
    [
        (64, (1, 1), 8.0),   # f=1: N=64, FOV=64 -> fits
        (64, (4, 4), 8.0),   # f=2 partner: N would be 128, FOV=88
    ],
)
def test_prism_smeta_grid_is_f1_n_equals_pixels(monkeypatch, P, scan, ssp):
    """f=1: s_meta.M == (P, P) (N = 1*P)."""
    _warp(monkeypatch)
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=1, pixels=(P, P),
                    scan=scan, scan_step_pixels=ssp, **_OPT)
    if scan == (4, 4):
        # f=1, P=64, scan=(4,4): N=64 < FOV=88 -> must raise (object doesn't fit)
        with pytest.raises(ValueError, match="(?i)pixels|fit|FOV"):
            sim.build_model()
        return
    model = sim.build_model()
    assert tuple(int(m) for m in model.scan_op.s_meta.M) == (P, P)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_smeta_grid_is_f2_n_equals_2pixels(monkeypatch):
    """f=2: s_meta.M == (2P, 2P) (N = f*P), object FOV fits inside N."""
    _warp(monkeypatch)
    P = 64
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=2, pixels=(P, P),
                    scan=(4, 4), scan_step_pixels=8.0, **_OPT)
    model = sim.build_model()
    assert tuple(int(m) for m in model.scan_op.s_meta.M) == (2 * P, 2 * P)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_aperture_is_flattice_f1(monkeypatch):
    """f=1: the f-lattice == the dense disk (every 1st point), so Bp equals
    the dense-disk count."""
    _warp(monkeypatch)
    P = 64
    s = Structure.fromfile(DATA)
    sim = Simulator(s, prism=True, prism_interpolation_factor=1, pixels=(P, P),
                    scan=(1, 1), scan_step_pixels=8.0, **_OPT)
    model = sim.build_model()
    dx = np.asarray(s.unitcell[:2], dtype=np.float64) / np.asarray([P, P])
    dense = _dense_disk_count(P, 1, _OPT["eV"], _OPT["semiconvergence_angle"],
                              float(dx[0]), float(dx[1]))
    flat = _flattice_count(P, 1, _OPT["eV"], _OPT["semiconvergence_angle"],
                           float(dx[0]), float(dx[1]))
    assert flat == dense  # f=1: lattice is the full grid
    assert int(model.scan_op.s_meta.Bp) == flat


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_aperture_is_flattice_f2(monkeypatch):
    """f=2: Bp equals the f-lattice count, which is STRICTLY LESS than the dense
    disk count (~dense/f²)."""
    _warp(monkeypatch)
    P = 64
    s = Structure.fromfile(DATA)
    sim = Simulator(s, prism=True, prism_interpolation_factor=2, pixels=(P, P),
                    scan=(4, 4), scan_step_pixels=8.0, **_OPT)
    model = sim.build_model()
    dx = np.asarray(s.unitcell[:2], dtype=np.float64) / np.asarray([P, P])
    dense = _dense_disk_count(P, 2, _OPT["eV"], _OPT["semiconvergence_angle"],
                              float(dx[0]), float(dx[1]))
    flat = _flattice_count(P, 2, _OPT["eV"], _OPT["semiconvergence_angle"],
                           float(dx[0]), float(dx[1]))
    Bp = int(model.scan_op.s_meta.Bp)
    assert Bp == flat
    assert Bp < dense          # f-lattice is sparser than the dense disk
    # ~dense / f² (allow a generous slack for the disk-boundary discretization)
    assert flat <= dense // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("f,P,scan", [(1, 64, (1, 1)), (2, 64, (4, 4))])
def test_prism_detector_equals_pixels(monkeypatch, f, P, scan):
    """Detector window == pixels == N/f (dk = f·Δq).  Read it off a tiny cube."""
    _warp(monkeypatch)
    cube = Simulator(Structure.fromfile(DATA), prism=True,
                     prism_interpolation_factor=f, pixels=(P, P), scan=scan,
                     scan_step_pixels=8.0, **_OPT).simulate()
    assert cube.shape[-2:] == (P, P)
    # N/f == P consistency
    model = Simulator(Structure.fromfile(DATA), prism=True,
                      prism_interpolation_factor=f, pixels=(P, P), scan=scan,
                      scan_step_pixels=8.0, **_OPT).build_model()
    MY, MX = (int(m) for m in model.scan_op.s_meta.M)
    assert (MY // f, MX // f) == (P, P)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prism_too_small_pixels_raises(monkeypatch):
    """f*pixels < FOV (object doesn't fit the sim grid) -> clear ValueError."""
    _warp(monkeypatch)
    # P=64, scan=(4,4), ssp=8 -> FOV=88; f=1 -> N=64 < 88.
    sim = Simulator(Structure.fromfile(DATA), prism=True,
                    prism_interpolation_factor=1, pixels=(64, 64),
                    scan=(4, 4), scan_step_pixels=8.0, **_OPT)
    with pytest.raises(ValueError, match="(?i)pixels"):
        sim.build_model()
