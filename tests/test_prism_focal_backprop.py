"""Focal back-propagation (Pelz 2021) for partitioned PRISM.

The de-tilt removes only the LINEAR tilt carrier; a parent column propagated to
depth z also carries a QUADRATIC defocus chirp ~exp(-i pi lam z |Q|^2) that makes
the de-tilted columns hard to interpolate for thick samples. PrismScanOp back-
propagates the raw columns to the sample centre (focal_backprop=0.5) before the
NNW fusion, which should bring the partitioned CBED closer to pure PRISM.

These tests assert:
  1. pure PRISM (Bp == B) is BYTE-IDENTICAL regardless of focal_backprop
     (the correction is gated on partitioning).
  2. on a thick object, partitioned PRISM with focal_backprop=0.5 is closer to
     pure PRISM than with focal_backprop=0.0.
"""

import numpy as np
import pytest
import torch

from scatterem.nn.modules import Propagator
from scatterem.nn.scan_exit_wave import build_prism_smeta
from scatterem.nn.scan_exit_wave.prism import PrismScanOp

LAM = 0.02508
DX = np.array([0.18, 0.18], dtype=np.float32)


def _inputs(N, NZ, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    T = torch.exp(1j * 0.07 * torch.randn(NZ, N, N)).to(torch.complex64).to(dev)
    prop = Propagator(2.0, LAM, DX, (N, N), device=dev)
    # K-broadcast focused probe (aperture disk -> real space), (1, K, N, N).
    qy = torch.fft.fftfreq(N, d=float(DX[0]))
    qx = torch.fft.fftfreq(N, d=float(DX[1]))
    QY, QX = torch.meshgrid(qy, qx, indexing="ij")
    kmax = 20e-3 / LAM
    aperture = ((QY**2 + QX**2) <= kmax**2).to(torch.complex64)
    probe = torch.fft.ifft2(aperture, norm="ortho").to(dev)
    pos = torch.tensor([[0, 0], [N // 3, N // 4], [2 * N // 5, N // 2]], device=dev)
    K = pos.shape[0]
    probe = probe[None, None].expand(1, K, N, N).contiguous()
    dr = torch.zeros((K, 2), dtype=torch.float32, device=dev)
    return T, probe, pos, dr, prop


def _cbed(exit_wave):
    return torch.fft.fft2(exit_wave, norm="ortho").abs() ** 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pure_prism_bit_identical_under_focal_backprop():
    """focal_backprop must not touch pure PRISM (Bp == B)."""
    N, NZ = 64, 40
    T, probe, pos, dr, prop = _inputs(N, NZ)
    sm = build_prism_smeta(
        pixels=(N, N),
        f=1,
        dx=DX,
        wavelength=LAM,
        semiconvergence_angle=20e-3,
        fov_shape=(N, N),
        device="cuda",
        n_radial=None,
    )
    with torch.no_grad():
        e0 = PrismScanOp(sm, f=1, focal_backprop=0.0).forward(T, probe, pos, dr, prop)
        e5 = PrismScanOp(sm, f=1, focal_backprop=0.5).forward(T, probe, pos, dr, prop)
    assert torch.equal(e0, e5), (
        "focal_backprop changed pure PRISM (must be gated on Bp<B)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_focal_backprop_improves_partitioned_thick():
    """On a thick object, focal_backprop=0.5 < focal_backprop=0.0 error vs pure."""
    N, NZ = 64, 60  # ~120 A: enough propagation chirp to matter
    T, probe, pos, dr, prop = _inputs(N, NZ)
    sm_full = build_prism_smeta(
        pixels=(N, N),
        f=1,
        dx=DX,
        wavelength=LAM,
        semiconvergence_angle=20e-3,
        fov_shape=(N, N),
        device="cuda",
        n_radial=None,
    )
    sm_part = build_prism_smeta(
        pixels=(N, N),
        f=1,
        dx=DX,
        wavelength=LAM,
        semiconvergence_angle=20e-3,
        fov_shape=(N, N),
        device="cuda",
        n_radial=4,
    )
    assert sm_part.Bp < sm_full.B, "partitioned SMeta must subsample the aperture"
    with torch.no_grad():
        ref = _cbed(PrismScanOp(sm_full, f=1).forward(T, probe, pos, dr, prop))
        c0 = _cbed(
            PrismScanOp(sm_part, f=1, focal_backprop=0.0).forward(
                T, probe, pos, dr, prop
            )
        )
        c5 = _cbed(
            PrismScanOp(sm_part, f=1, focal_backprop=0.5).forward(
                T, probe, pos, dr, prop
            )
        )

    def rel(a):
        return (torch.linalg.norm(a - ref) / torch.linalg.norm(ref)).item()

    err0, err5 = rel(c0), rel(c5)
    print(
        f"\npartitioned CBED rel-err vs pure: no-backprop={err0:.3%} backprop={err5:.3%}"
    )
    assert err5 < err0, (
        f"focal_backprop did not improve accuracy ({err5:.3%} >= {err0:.3%})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_centroid_default_tracks_asymmetric_scattering():
    """The 'centroid' default must beat a fixed t/2 plane when the scattering is
    asymmetric (here confined to the top third -> centroid near the entrance)."""
    N, NZ = 64, 60
    dev = "cuda"
    torch.manual_seed(1)
    phase = torch.zeros(NZ, N, N)
    phase[: NZ // 3] = 0.07 * torch.randn(NZ // 3, N, N)  # scatter in top third only
    T = torch.exp(1j * phase).to(torch.complex64).to(dev)
    _, probe, pos, dr, prop = _inputs(N, NZ)  # probe/positions/propagator (reseeds)

    sm_full = build_prism_smeta(
        pixels=(N, N),
        f=1,
        dx=DX,
        wavelength=LAM,
        semiconvergence_angle=20e-3,
        fov_shape=(N, N),
        device=dev,
        n_radial=None,
    )
    sm_part = build_prism_smeta(
        pixels=(N, N),
        f=1,
        dx=DX,
        wavelength=LAM,
        semiconvergence_angle=20e-3,
        fov_shape=(N, N),
        device=dev,
        n_radial=4,
    )
    with torch.no_grad():
        ref = _cbed(PrismScanOp(sm_full, f=1).forward(T, probe, pos, dr, prop))
        c_half = _cbed(
            PrismScanOp(sm_part, f=1, focal_backprop=0.5).forward(
                T, probe, pos, dr, prop
            )
        )
        c_cent = _cbed(
            PrismScanOp(sm_part, f=1, focal_backprop="centroid").forward(
                T, probe, pos, dr, prop
            )
        )

    def rel(a):
        return (torch.linalg.norm(a - ref) / torch.linalg.norm(ref)).item()

    e_half, e_cent = rel(c_half), rel(c_cent)
    print(
        f"\nasymmetric (top-third) CBED rel-err: fixed-0.5={e_half:.3%} centroid={e_cent:.3%}"
    )
    assert e_cent < e_half, (
        f"centroid ({e_cent:.3%}) should beat fixed-t/2 ({e_half:.3%})"
    )
