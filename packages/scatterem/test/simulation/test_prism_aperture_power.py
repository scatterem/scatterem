"""Power-based f-lattice-shell aperture sizing for faithful PRISM (A1, Task 2).

``build_prism_smeta_for_probe`` sizes the PRISM BF-disk aperture from the
*incoherent* multi-mode probe power spectrum ``P(k) = sum_m |FFT(probe[m])|^2``
instead of the fixed ``kmax*1.05`` margin of :func:`build_prism_smeta`.  It walks
the f-lattice shells (candidate radii ``{0, f*dk, 2f*dk, ...}``) and picks the
SMALLEST shell whose cumulative power reaches ``power_fraction`` of the total,
floored at ``kmax*1.2`` so the aperture is never tighter than 1.2x the
convergence angle.  This captures the soft (tanh) probe-edge tail the fixed
margin truncates, which is the too-tight-aperture footgun for f-lattice PRISM.

These are CPU-runnable structural / power asserts (no CUDA forward needed); the
SMeta math is device-agnostic, so the probe + SMeta are built on CPU.
"""

import numpy as np
import torch

from scatterem.simulation.simulator import (
    build_prism_smeta,
    build_prism_smeta_for_probe,
)
from scatterem.utils.stem import fftfreq2

# Representative optics (mirror the simulator PRISM tests: 200 keV, 20 mrad).
WAVELENGTH = 0.0251  # Aa, ~200 keV
SEMICONV = 20e-3  # rad
PIXELS = (64, 64)  # detector window
DX = np.array([0.08484, 0.08484], dtype=np.float32)  # Aa/px (small.xyz pitch)
DEVICE = "cpu"


def _kmax():
    return SEMICONV / WAVELENGTH


def _soft_edge_probe(f, n_modes=2, edge=2.0, radius_scale=1.0):
    """A representative soft-edge probe on the N=f*pixels grid (real space).

    Built as ``ifft2`` of a tanh-edged disk aperture in Fourier space at radius
    ``radius_scale * kmax/dk`` pixels.  The tanh edge bleeds Fourier power past
    ``kmax`` -- exactly the tail the fixed ``kmax*1.05`` margin truncates.
    Returns a complex ``(n_modes, MY, MX)`` tensor.
    """
    MY, MX = f * PIXELS[0], f * PIXELS[1]
    qf = fftfreq2((MY, MX), [float(DX[0]), float(DX[1])], centered=False, device=DEVICE)
    qmag = torch.linalg.norm(qf, dim=0)
    kmax = _kmax()
    dk = 1.0 / (MY * float(DX[0]))
    r_pix = radius_scale * kmax / dk
    rmag_pix = qmag / dk
    # Soft tanh edge (in pixels) -> bleeds past r_pix.
    soft = 0.5 * (1.0 - torch.tanh((rmag_pix - r_pix) / (edge / 2.0)))
    modes = []
    for m in range(n_modes):
        # Vary the phase ramp per mode so modes differ but share the support.
        ramp = torch.exp(2j * np.pi * (m + 1) * 0.01 * (qf[0] * MY + qf[1] * MX))
        ap = soft.to(torch.complex64) * ramp
        modes.append(torch.fft.ifft2(ap))
    return torch.stack(modes, dim=0)


def _bandlimited_probe(f, cutoff_frac=0.4):
    """A probe whose Fourier power is a HARD disk strictly inside ``kmax``.

    Support radius = ``cutoff_frac * kmax`` (well inside the convergence angle),
    so the 0.9999-power shell radius is tiny and the ``kmax*1.2`` floor must
    dominate.  Returns a complex ``(1, MY, MX)`` tensor.
    """
    MY, MX = f * PIXELS[0], f * PIXELS[1]
    qf = fftfreq2((MY, MX), [float(DX[0]), float(DX[1])], centered=False, device=DEVICE)
    qmag = torch.linalg.norm(qf, dim=0)
    kmax = _kmax()
    hard = (qmag <= cutoff_frac * kmax).to(torch.complex64)
    return torch.fft.ifft2(hard)[None, ...]


def _captured_power_fraction(probe, s_meta):
    """Measured fraction of incoherent probe power inside the SMeta aperture."""
    P = (torch.abs(torch.fft.fft2(probe, dim=(-2, -1))) ** 2).sum(dim=0)  # (MY, MX)
    total = float(P.sum())
    mask = s_meta.all_beams.to(P.device).bool()
    captured = float(P[mask].sum())
    return captured / total


def test_captured_power_meets_target():
    """(a) Returned aperture encloses >= power_fraction of incoherent probe power."""
    f = 1
    probe = _soft_edge_probe(f)
    s_meta = build_prism_smeta_for_probe(
        probe=probe,
        f=f,
        pixels=PIXELS,
        dx=DX,
        wavelength=WAVELENGTH,
        semiconvergence_angle=SEMICONV,
        fov_shape=PIXELS,
        device=DEVICE,
        power_fraction=0.9999,
    )
    frac = _captured_power_fraction(probe, s_meta)
    print(f"\n(a) captured power fraction = {frac:.6f} (target 0.9999)")
    assert frac >= 0.9999, f"captured power {frac:.6f} < target 0.9999"


def test_aperture_is_f_lattice_aligned_and_valid():
    """(b) Every selected beam index % f == 0; the SMeta is valid (Bp>0, beamlets/NNW set)."""
    f = 2
    probe = _soft_edge_probe(f)
    s_meta = build_prism_smeta_for_probe(
        probe=probe,
        f=f,
        pixels=PIXELS,
        dx=DX,
        wavelength=WAVELENGTH,
        semiconvergence_angle=SEMICONV,
        fov_shape=PIXELS,
        device=DEVICE,
    )
    MY, MX = int(s_meta.M[0]), int(s_meta.M[1])
    idx = s_meta.all_beams.cpu().numpy().nonzero()
    iy, ix = idx[0], idx[1]
    assert (iy % f == 0).all() and (ix % f == 0).all(), (
        "selected beams are not f-lattice-aligned"
    )
    assert s_meta.Bp > 0
    assert s_meta.beamlets is not None
    assert tuple(s_meta.beamlets.shape) == (s_meta.Bp, MY, MX)
    assert s_meta.natural_neighbor_weights is not None
    assert tuple(s_meta.natural_neighbor_weights.shape) == (s_meta.Bp, s_meta.Bp)


def test_power_aperture_wider_than_default_margin():
    """(c) For a soft-edge probe, the power aperture is WIDER than kmax*1.05."""
    f = 1
    probe = _soft_edge_probe(f)
    common = dict(
        f=f,
        pixels=PIXELS,
        dx=DX,
        wavelength=WAVELENGTH,
        semiconvergence_angle=SEMICONV,
        fov_shape=PIXELS,
        device=DEVICE,
    )
    s_power = build_prism_smeta_for_probe(probe=probe, power_fraction=0.9999, **common)
    s_default = build_prism_smeta(**common)
    print(
        f"\n(c) power: B={int(s_power.B)} R={s_power.numerical_aperture_radius_pixels} | "
        f"kmax*1.05: B={int(s_default.B)} R={s_default.numerical_aperture_radius_pixels}"
    )
    assert int(s_power.B) > int(s_default.B), (
        "power aperture should enclose more beams than kmax*1.05"
    )
    assert (
        s_power.numerical_aperture_radius_pixels
        > s_default.numerical_aperture_radius_pixels
    ), "power aperture radius should exceed kmax*1.05 radius"


def test_floor_respected_for_bandlimited_probe():
    """(d) A probe band-limited inside kmax still yields radius >= kmax*1.2."""
    f = 1
    probe = _bandlimited_probe(f, cutoff_frac=0.4)
    s_meta = build_prism_smeta_for_probe(
        probe=probe,
        f=f,
        pixels=PIXELS,
        dx=DX,
        wavelength=WAVELENGTH,
        semiconvergence_angle=SEMICONV,
        fov_shape=PIXELS,
        device=DEVICE,
        power_fraction=0.9999,
    )
    kmax = _kmax()
    dk0 = 1.0 / (int(s_meta.M[0]) * float(DX[0]))
    r_floor = int(round(kmax * 1.2 / dk0))
    print(
        f"\n(d) bandlimited: R={s_meta.numerical_aperture_radius_pixels} "
        f">= floor R={r_floor} (kmax*1.2)"
    )
    assert s_meta.numerical_aperture_radius_pixels >= r_floor, (
        f"radius {s_meta.numerical_aperture_radius_pixels} below kmax*1.2 floor {r_floor}"
    )


def test_partition_reduces_parent_count():
    """(e) Partition variant -> Bp < B_full (pure)."""
    f = 1
    probe = _soft_edge_probe(f)
    common = dict(
        probe=probe,
        f=f,
        pixels=PIXELS,
        dx=DX,
        wavelength=WAVELENGTH,
        semiconvergence_angle=SEMICONV,
        fov_shape=PIXELS,
        device=DEVICE,
        power_fraction=0.9999,
    )
    s_pure = build_prism_smeta_for_probe(**common)
    s_part = build_prism_smeta_for_probe(n_radial=2, n_angular=6, **common)
    print(f"\n(e) pure Bp={int(s_pure.Bp)} -> partitioned Bp={int(s_part.Bp)}")
    assert int(s_part.Bp) < int(s_pure.B), (
        f"partitioned Bp ({int(s_part.Bp)}) should be < full B ({int(s_pure.B)})"
    )
