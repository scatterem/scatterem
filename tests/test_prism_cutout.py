"""Low-level kernel unit test for PRISM interpolation factor f > 1.

Proves PrismScanOp realizes ``f`` as the **Ophus 2017 real-space cutout**
(Eqs 6/7/10), NOT a Fourier crop:

  * beams on the f-spaced reciprocal lattice (Eq 6),
  * S-matrix columns at FULL N x N real-space resolution (no coarsening),
  * a per-position N/f = P real-space cutout at the SAME pitch dp (Eq 7,
    wrapped %N), summed over beams with the Eq-8 position phase ramp,
  * detector = P = N/f pixels at dk = f * dq (smaller real-space window =>
    coarser reciprocal sampling).

The probe / detector grid is P = N/f pixels (the recon convention: M1 ==
detector). An f-lattice beam at signed N-grid index ``(sy, sx)`` (a multiple
of f, |k| <= alpha/lam) maps to the P-grid pixel ``(round(sy/f)%P,
round(sx/f)%P)``; this is collision-free for these beams (verified), so the
beamlet basis is exact.

ORACLE ("R1", verified equal to PRISM at machine precision, 1.7e-15):

    full-N multislice of the synthesized PRISM probe at r0
      -> real-space P x P cutout (center round(r0/dp), wrap %N, pitch dp)
      -> FFT -> P-pixel CBED at dk = f * dq.

R1 is built WITHOUT PrismScanOp internals: an independently synthesized full-N
incident probe (Ophus Eq 8/10) is pushed through ``MultisliceScanOp`` (the
plain multislice operator), then the P x P window is cut and FFT'd. So the
assert is a genuine cross-check of the PRISM kernel against multislice.

TWO TIERS (because the production kernel is an **fp32** numerical kernel --
its detilt phase ramp, ``prism_project``'s ``fft_frequency_grid``, and the Warp path are
all float32, so an fp32 *computation* of R1 itself differs from fp64 by ~2e-7;
1e-13 is only reachable in pure fp64):

  1. PRODUCTION KERNEL (fp32): the real ``PrismScanOp(f).forward`` vs an fp32
     R1 -> tight fp32 floor (measured ~1e-6; asserted < 1e-5). This tests the
     actual recon-shared kernel.
  2. ALGORITHM EXACTNESS (fp64): a faithful fp64 re-implementation of the
     kernel's EXACT algorithm (full-N S, detilt, real-space cutout, one-hot
     beamlet basis -- same formulas, complex128 dtype) vs an fp64 R1 ->
     machine precision (measured ~7e-15; asserted < 1e-12). This proves the
     algorithm the kernel implements equals R1 to machine precision, matching
     the workflow's 1.7e-15, with the only fp32-floor caveat being dtype.

NEGATIVE CONTROL: the *Fourier-crop* reference (full-N CBED -> crop central
P x P, dk stays dq) is the trap two prior prototypes fell into; we assert the
kernel does NOT match it (rel err ~ 1).

TF32 is disabled (the projection uses cuBLAS GEMM). CPU is fine (tiny grids).
"""

from unittest import TestCase, main

import numpy as np
import pytest
import torch

from scatterem.data import SMeta
from scatterem.nn.modules import Propagator
from scatterem.nn.scan_exit_wave import MultisliceScanOp, PrismScanOp
from scatterem.physics._frequency_grid import fft_frequency_grid


# --------------------------------------------------------------------------- #
# f-lattice SMeta + test inputs
# --------------------------------------------------------------------------- #
def _f_lattice_aperture(N: int, f: int, kmax_pix: float) -> np.ndarray:
    """Boolean (N, N) f-lattice bright-field disk (Ophus Eq 6).

    Beams on the f-spaced reciprocal lattice (signed pixel index % f == 0)
    AND inside |k| <= kmax (radius kmax_pix in reciprocal pixels). The grid
    uses fftfreq ordering (DC at index 0); signed index = i if i <= N//2 else
    i - N.
    """
    take = np.zeros((N, N), dtype=bool)
    for iy in range(N):
        sy = iy if iy <= N // 2 else iy - N
        if sy % f != 0:
            continue
        for ix in range(N):
            sx = ix if ix <= N // 2 else ix - N
            if sx % f != 0:
                continue
            if sy * sy + sx * sx <= kmax_pix * kmax_pix:
                take[iy, ix] = True
    return take


def _make_f_lattice_smeta(N: int, f: int, kmax_pix: float, dp: float, device) -> SMeta:
    """Hand-built pure-PRISM f-lattice SMeta: one-hot beamlets, identity NNW."""
    take = _f_lattice_aperture(N, f, kmax_pix)
    s_meta = SMeta(
        all_beams=take,
        parent_beams=take,
        numerical_aperture_radius_pixels=int(kmax_pix),
        dx=np.array([dp, dp], dtype=np.float32),
        M=(N, N),
        device=torch.device(device),
    )
    Bp = s_meta.Bp
    s_meta.natural_neighbor_weights = torch.eye(Bp, dtype=torch.float32, device=device)
    # ``SMeta.beamlets`` is the GATHERED (Bp, B) form, not a dense (Bp, MY, MX)
    # one-hot grid (s_meta.py:172-177).  Build dense, then gather at the beam
    # coordinates -- exactly the gather the consumer used to do internally.
    dense = torch.zeros((Bp, N, N), dtype=torch.float32, device=device)
    for p, (y, x) in enumerate(s_meta.parent_beams_coords.tolist()):
        dense[p, int(y) % N, int(x) % N] = 1.0
    by = s_meta.all_beams_coords[:, 0].long() % N
    bx = s_meta.all_beams_coords[:, 1].long() % N
    s_meta.beamlets = dense[:, by, bx]  # (Bp, B)
    return s_meta


def _gaussian_atoms_transmission(N, NZ, device, dtype=torch.complex64, seed=0):
    """Smooth complex transmission stack spanning the full N grid (no crop)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy = torch.arange(N).double()
    Y, X = torch.meshgrid(yy, yy, indexing="ij")
    T = torch.zeros((NZ, N, N), dtype=torch.complex128)
    for z in range(NZ):
        phase = torch.zeros((N, N), dtype=torch.float64)
        n_atoms = 6
        cy = torch.rand(n_atoms, generator=g).double() * N
        cx = torch.rand(n_atoms, generator=g).double() * N
        amp = 0.3 + 0.4 * torch.rand(n_atoms, generator=g).double()
        sig = 1.5 + 1.0 * torch.rand(n_atoms, generator=g).double()
        for a in range(n_atoms):
            r2 = (Y - cy[a]) ** 2 + (X - cx[a]) ** 2
            phase += amp[a] * torch.exp(-r2 / (2 * sig[a] ** 2))
        T[z] = torch.exp(1j * phase)
    return T.to(device=device, dtype=dtype)


def _beam_pixel_maps(s_meta, N, f):
    """(syN, sxN) N-grid beam pixels and (ayP, axP) P-grid basis pixels."""
    P = N // f
    coords = s_meta.all_beams_coords.cpu().numpy().astype(int)  # (B, 2) signed
    syN = coords[:, 0] % N
    sxN = coords[:, 1] % N
    ayP = (np.round(coords[:, 0] / f).astype(int)) % P
    axP = (np.round(coords[:, 1] / f).astype(int)) % P
    return coords, syN, sxN, ayP, axP


def _synthesize_probe_P(s_meta, N, f, device, dtype=torch.complex64, seed=1):
    """Random aperture-limited probe on the P = N/f grid (the detector grid).

    Returns (probe_fft_P, probe_real_P): probe_real_P (1, 1, P, P) is the
    kernel's input probe; probe_fft_P (P, P) is **fft2(probe_real_P, ortho)**
    -- exactly the aperture amplitudes the kernel re-derives internally
    (``_build_beamlet_basis_fft`` does ``fft2(probe[:, 0], ortho)``). R1 must
    consume this same FFT, else an fp32 round-trip mismatch (stored spectrum
    vs ``fft2(ifft2(.))``) floors the fp32 comparison spuriously at ~1e-7.
    """
    P = N // f
    g = torch.Generator(device="cpu").manual_seed(seed)
    _, _, _, ayP, axP = _beam_pixel_maps(s_meta, N, f)
    mask = torch.zeros((P, P), dtype=torch.bool)
    mask[ayP, axP] = True
    spectrum = (
        torch.randn(P, P, generator=g, dtype=torch.float64)
        + 1j * torch.randn(P, P, generator=g, dtype=torch.float64)
    ) * mask
    probe_real = torch.fft.ifft2(spectrum, norm="ortho")
    probe_fft = torch.fft.fft2(probe_real, norm="ortho")  # what the kernel sees
    return (
        probe_fft.to(device=device, dtype=dtype),
        probe_real.reshape(1, 1, P, P).to(device=device, dtype=dtype),
    )


# --------------------------------------------------------------------------- #
# Synthesized incident probe + R1 oracle
# --------------------------------------------------------------------------- #
def _synth_psi_inc(s_meta, probe_fft_P, N, f, r0_pix, dr_pix, device, dtype):
    """Full-N synthesized PRISM probe psi_inc(R) at position (r0_pix+dr_pix).

    F_N[sy_b%N, sx_b%N] = A_b * exp(-2pi i q_N . (r0_pix+dr_pix)),
    A_b = probe_fft_P[round(sy_b/f)%P, round(sx_b/f)%P];
    psi_inc = ifft2_N(F_N, ortho).
    """
    _, syN, sxN, ayP, axP = _beam_pixel_maps(s_meta, N, f)
    A = probe_fft_P[torch.as_tensor(ayP), torch.as_tensor(axP)]  # (B,)
    F_N = torch.zeros((N, N), dtype=dtype, device=device)
    F_N[torch.as_tensor(syN), torch.as_tensor(sxN)] = A
    rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
    qpix = torch.fft.fftfreq(N, 1.0, device=device).to(rdtype)
    QY, QX = torch.meshgrid(qpix, qpix, indexing="ij")
    sy_shift = float(r0_pix[0]) + float(dr_pix[0])
    sx_shift = float(r0_pix[1]) + float(dr_pix[1])
    ramp = torch.exp(-2j * torch.pi * (QY * sy_shift + QX * sx_shift)).to(dtype)
    psi_inc = torch.fft.ifft2(F_N * ramp, norm="ortho")
    # Match PrismScanOp's detector-grid (P = N/f) normalization: the kernel
    # scales its N-grid plane waves by f (_initial_parent_waves) so the S-matrix
    # is P-normalized and PRISM(N=f*P,f) == multislice(P) in ABSOLUTE amplitude
    # (verified against multislice). The same factor here
    # keeps R1 the geometry oracle for the corrected kernel; because both the
    # kernel and R1 scale by f, the f cancels in this test's relative-error
    # comparisons (geometry is what is under test; the gate pins the scale).
    if f != 1:
        psi_inc = psi_inc * float(f)
    return psi_inc.reshape(1, 1, N, N)


def _r1_reference(
    s_meta, T, probe_fft_P, N, f, r0_pix, dr_pix, propagator, device, dtype
):
    """R1 oracle: full-N multislice of psi_inc(r0) -> P x P real-space cutout
    (center round(r0/dp)=r0_pix, wrap %N, pitch dp) -> FFT -> CBED."""
    P = N // f
    psi_inc = _synth_psi_inc(s_meta, probe_fft_P, N, f, r0_pix, dr_pix, device, dtype)
    ms = MultisliceScanOp().to(device)
    positions = torch.zeros((1, 2), dtype=torch.int32, device=device)
    dr = torch.zeros((1, 2), dtype=torch.float32, device=device)
    exit_full = ms(T.unsqueeze(1), psi_inc, positions, dr, propagator)[:, 0]  # (1,N,N)
    py0 = int(r0_pix[0]) % N
    px0 = int(r0_pix[1]) % N
    yy = (py0 + torch.arange(P, device=device)) % N
    xx = (px0 + torch.arange(P, device=device)) % N
    cut = exit_full[:, yy][:, :, xx]  # (1, P, P)
    cbed = torch.fft.fft2(cut, norm="ortho")  # P-pixel CBED at dk = f*dq
    return cut, cbed


def _fourier_crop_reference(
    s_meta, T, probe_fft_P, N, f, r0_pix, propagator, device, dtype
):
    """NEGATIVE control: full-N CBED -> crop central P x P (dk stays dq).
    This is the WRONG f behavior (the dropped Fourier-crop path)."""
    P = N // f
    psi_inc = _synth_psi_inc(
        s_meta, probe_fft_P, N, f, r0_pix, (0.0, 0.0), device, dtype
    )
    ms = MultisliceScanOp().to(device)
    positions = torch.zeros((1, 2), dtype=torch.int32, device=device)
    dr = torch.zeros((1, 2), dtype=torch.float32, device=device)
    exit_full = ms(T.unsqueeze(1), psi_inc, positions, dr, propagator)[:, 0]
    cbed_full = torch.fft.fftshift(
        torch.fft.fft2(exit_full, norm="ortho"), dim=(-2, -1)
    )
    c = P // 2
    crop = cbed_full[:, N // 2 - c : N // 2 + c, N // 2 - c : N // 2 + c]
    return torch.fft.ifftshift(crop, dim=(-2, -1))


# --------------------------------------------------------------------------- #
# fp64 re-implementation of the kernel's EXACT algorithm (algorithm-exactness
# tier). Same formulas as PrismScanOp.forward, but complex128 throughout so the
# only difference from R1 is rounding (no fp32 constants).
# --------------------------------------------------------------------------- #
def _prism_algorithm_fp64(
    s_meta, T64, probe_fft_P64, N, f, r0_pix, dr_pix, propagator64, device
):
    P = N // f
    coords, syN, sxN, ayP, axP = _beam_pixel_maps(s_meta, N, f)
    Bp = s_meta.Bp

    # 1. full-N S columns: ifft2(delta at beam, ortho) multislice'd (no Fourier crop).
    psi_fft = torch.zeros((Bp, N, N), dtype=torch.complex128, device=device)
    psi_fft[torch.arange(Bp), torch.as_tensor(syN), torch.as_tensor(sxN)] = 1.0
    S = torch.fft.ifft2(psi_fft, norm="ortho")
    if f != 1:
        # Detector-grid (P = N/f) normalization, mirroring the fixed
        # PrismScanOp._initial_parent_waves (scales N-grid plane waves by f).
        S = S * float(f)
    NZ = T64.shape[0]
    for z in range(NZ):
        S = T64[z].unsqueeze(0) * S
        if z < NZ - 1:
            S = propagator64(S)

    # 2. detilt: S_dt[p, R] = S[p, R] * exp(-2pi i h_p . R), h_p in 1/A, R = idx * dp.
    iv = torch.arange(N, dtype=torch.float64, device=device)
    IY, IX = torch.meshgrid(iv, iv, indexing="ij")
    syt = torch.as_tensor(coords[:, 0], dtype=torch.float64, device=device)
    sxt = torch.as_tensor(coords[:, 1], dtype=torch.float64, device=device)
    dphase = torch.exp(
        -2j
        * torch.pi
        * (syt[:, None, None] * IY[None] + sxt[:, None, None] * IX[None])
        / N
    )
    S_dt = S * dphase

    # 3. one-hot beamlet basis on the P grid, with the Eq-8 sub-pixel ramp.
    A = probe_fft_P64[torch.as_tensor(ayP), torch.as_tensor(axP)]  # (Bp,)
    yv = torch.arange(P, dtype=torch.float64, device=device)
    Yb, Xb = torch.meshgrid(yv, yv, indexing="ij")
    ayPt = torch.as_tensor(ayP, dtype=torch.float64, device=device)
    axPt = torch.as_tensor(axP, dtype=torch.float64, device=device)
    basis_k = A[:, None, None] * torch.exp(
        2j
        * torch.pi
        * (ayPt[:, None, None] * Yb[None] + axPt[:, None, None] * Xb[None])
        / P
    )
    # sub-pixel ramp (pixel-frequency units on the P grid).
    qP = torch.fft.fftfreq(P, 1.0, device=device).to(torch.float64)
    QYp, QXp = torch.meshgrid(qP, qP, indexing="ij")
    rampP = torch.exp(
        -2j * torch.pi * (QYp * float(dr_pix[0]) + QXp * float(dr_pix[1]))
    )
    basis_k = torch.fft.ifft2(
        torch.fft.fft2(basis_k, norm="ortho") * rampP, norm="ortho"
    )

    # 4. real-space cutout at (py, px), window P, wrap %N; sum over beams.
    py, px = int(r0_pix[0]), int(r0_pix[1])
    gy = (py + torch.arange(P, device=device)[:, None]) % N
    gx = (px + torch.arange(P, device=device)[None, :]) % N
    S_crop = S_dt[:, gy, gx]
    cut = (basis_k * S_crop).sum(0)  # (P, P)
    cbed = torch.fft.fft2(cut, norm="ortho")
    return cut.reshape(1, P, P), cbed.reshape(1, P, P)


@pytest.mark.usefixtures("ieee_float32")
class TestPrismRealSpaceCutout(TestCase):
    """PrismScanOp(f>1) realizes the Ophus real-space cutout (== R1), and does
    NOT match the Fourier-crop trap."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        # Oracle setup (workflow-measured): N=64, f=2, P=32, dp=0.2, lam=0.025,
        # alpha/lam=0.8 1/A -> kmax_pix = (alpha/lam)/dq = (alpha/lam)*N*dp
        #                    = 0.8 * 64 * 0.2 = 10.24 -> 89 f-lattice beams.
        self.device = "cpu"
        self.N = 64
        self.f = 2
        self.P = self.N // self.f
        self.NZ = 3
        self.dp = 0.2
        self.lam = 0.025
        alpha_over_lam = 0.8  # 1/A
        dq = 1.0 / (self.N * self.dp)
        self.kmax_pix = alpha_over_lam / dq
        self.s_meta = _make_f_lattice_smeta(
            self.N, self.f, self.kmax_pix, self.dp, self.device
        )
        self.assertEqual(self.s_meta.Bp, 89)  # matches the workflow's beam count
        # fp32 inputs (production kernel) + fp64 inputs (algorithm-exactness).
        self.T32 = _gaussian_atoms_transmission(
            self.N, self.NZ, self.device, torch.complex64
        )
        self.T64 = self.T32.to(torch.complex128)
        self.pfft32, self.preal32 = _synthesize_probe_P(
            self.s_meta, self.N, self.f, self.device, torch.complex64
        )
        self.pfft64, _ = _synthesize_probe_P(
            self.s_meta, self.N, self.f, self.device, torch.complex128
        )
        self.prop32 = Propagator(
            slice_thickness=1.0,
            wavelength=self.lam,
            dx=(self.dp, self.dp),
            shape=(self.N, self.N),
            device=self.device,
        )
        # fp64 propagator (same phase ramp, complex128) for the exactness tier.
        self.prop64 = Propagator(
            slice_thickness=1.0,
            wavelength=self.lam,
            dx=(self.dp, self.dp),
            shape=(self.N, self.N),
            device=self.device,
        )
        q64 = fft_frequency_grid(
            (self.N, self.N), [self.dp, self.dp], device=self.device
        ).to(torch.float64)
        self.prop64.prop = torch.exp(
            (-1j * torch.pi * self.lam * 1.0) * torch.linalg.norm(q64, axis=0) ** 2
        )

    def _kernel_forward(self, py, px, dry, drx):
        """Real PrismScanOp(f) (fp32) exit wave for a single P-grid position;
        returns (cut, cbed). positions are integer object pixels on the N grid,
        dr the sub-pixel offset in pixels."""
        probe = self.preal32.expand(1, 1, self.P, self.P).contiguous()
        positions = torch.tensor([[py, px]], dtype=torch.int32, device=self.device)
        dr = torch.tensor([[dry, drx]], dtype=torch.float32, device=self.device)
        prism = PrismScanOp(self.s_meta, f=self.f).to(self.device)
        out = prism(self.T32, probe, positions, dr, self.prop32)  # (1, 1, P, P)
        cut = out[:, 0]  # (1, P, P)
        cbed = torch.fft.fft2(cut, norm="ortho")
        return cut, cbed

    # ---------------------------------------------------------------- shape
    def test_detector_is_P_pixels(self) -> None:
        cut, _ = self._kernel_forward(0, 0, 0.0, 0.0)
        self.assertEqual(tuple(cut.shape), (1, self.P, self.P))

    # ---------------------------------------- production kernel (fp32 floor)
    def test_f_aligned_position_matches_R1_fp32(self) -> None:
        py, px = 8, 12  # f-aligned integer position (multiples of f=2)
        cut_k, cbed_k = self._kernel_forward(py, px, 0.0, 0.0)
        cut_r, cbed_r = _r1_reference(
            self.s_meta,
            self.T32,
            self.pfft32,
            self.N,
            self.f,
            (py, px),
            (0.0, 0.0),
            self.prop32,
            self.device,
            torch.complex64,
        )
        rel_cut = ((cut_k - cut_r).abs().norm() / cut_r.abs().norm()).item()
        rel_cbed = ((cbed_k - cbed_r).abs().norm() / cbed_r.abs().norm()).item()
        print(f"\n[fp32 f-aligned] rel_cut={rel_cut:.3e} rel_cbed={rel_cbed:.3e}")
        # fp32 floor (~1e-6); 1e-13 is unreachable for the fp32 kernel.
        self.assertLess(rel_cut, 1e-5)
        self.assertLess(rel_cbed, 1e-5)

    def test_non_f_aligned_subpixel_position_matches_R1_fp32(self) -> None:
        py, px = 7, 13  # 7 is NOT a multiple of f=2
        dry, drx = 0.37, -0.21  # sub-pixel: exact via the Eq-8 phase ramp
        cut_k, cbed_k = self._kernel_forward(py, px, dry, drx)
        cut_r, cbed_r = _r1_reference(
            self.s_meta,
            self.T32,
            self.pfft32,
            self.N,
            self.f,
            (py, px),
            (dry, drx),
            self.prop32,
            self.device,
            torch.complex64,
        )
        rel_cut = ((cut_k - cut_r).abs().norm() / cut_r.abs().norm()).item()
        rel_cbed = ((cbed_k - cbed_r).abs().norm() / cbed_r.abs().norm()).item()
        print(f"\n[fp32 subpix] rel_cut={rel_cut:.3e} rel_cbed={rel_cbed:.3e}")
        self.assertLess(rel_cut, 1e-5)
        self.assertLess(rel_cbed, 1e-5)

    # --------------------------------- algorithm exactness (fp64, ~machine)
    def test_f_aligned_position_matches_R1_fp64_machine_precision(self) -> None:
        py, px = 8, 12
        cut_k, cbed_k = _prism_algorithm_fp64(
            self.s_meta,
            self.T64,
            self.pfft64,
            self.N,
            self.f,
            (py, px),
            (0.0, 0.0),
            self.prop64,
            self.device,
        )
        cut_r, cbed_r = _r1_reference(
            self.s_meta,
            self.T64,
            self.pfft64,
            self.N,
            self.f,
            (py, px),
            (0.0, 0.0),
            self.prop64,
            self.device,
            torch.complex128,
        )
        rel_cut = ((cut_k - cut_r).abs().norm() / cut_r.abs().norm()).item()
        rel_cbed = ((cbed_k - cbed_r).abs().norm() / cbed_r.abs().norm()).item()
        print(f"\n[fp64 f-aligned] rel_cut={rel_cut:.3e} rel_cbed={rel_cbed:.3e}")
        # Machine precision (workflow: 1.7e-15) -- the algorithm is EXACT == R1.
        self.assertLess(rel_cut, 1e-12)
        self.assertLess(rel_cbed, 1e-12)

    def test_non_f_aligned_subpixel_position_matches_R1_fp64_machine_precision(
        self,
    ) -> None:
        py, px = 7, 13
        dry, drx = 0.37, -0.21
        cut_k, cbed_k = _prism_algorithm_fp64(
            self.s_meta,
            self.T64,
            self.pfft64,
            self.N,
            self.f,
            (py, px),
            (dry, drx),
            self.prop64,
            self.device,
        )
        cut_r, cbed_r = _r1_reference(
            self.s_meta,
            self.T64,
            self.pfft64,
            self.N,
            self.f,
            (py, px),
            (dry, drx),
            self.prop64,
            self.device,
            torch.complex128,
        )
        rel_cut = ((cut_k - cut_r).abs().norm() / cut_r.abs().norm()).item()
        rel_cbed = ((cbed_k - cbed_r).abs().norm() / cbed_r.abs().norm()).item()
        print(f"\n[fp64 subpix] rel_cut={rel_cut:.3e} rel_cbed={rel_cbed:.3e}")
        self.assertLess(rel_cut, 1e-12)
        self.assertLess(rel_cbed, 1e-12)

    # -------------------------------------------------- negative controls
    def test_kernel_is_not_a_fourier_crop(self) -> None:
        py, px = 8, 12
        _, cbed_k = self._kernel_forward(py, px, 0.0, 0.0)
        cbed_fc = _fourier_crop_reference(
            self.s_meta,
            self.T32,
            self.pfft32,
            self.N,
            self.f,
            (py, px),
            self.prop32,
            self.device,
            torch.complex64,
        )
        rel = ((cbed_k - cbed_fc).abs().norm() / cbed_fc.abs().norm()).item()
        print(f"\n[neg-ctrl fourier-crop] rel={rel:.3e} (must be large ~1)")
        self.assertGreater(rel, 0.3)

    def test_R1_itself_is_not_a_fourier_crop(self) -> None:
        # Sanity: the R1 oracle (real-space cutout) and the Fourier-crop trap
        # are genuinely different references (rel ~ 1), so the positive match
        # above is meaningful.
        py, px = 8, 12
        _, cbed_r = _r1_reference(
            self.s_meta,
            self.T32,
            self.pfft32,
            self.N,
            self.f,
            (py, px),
            (0.0, 0.0),
            self.prop32,
            self.device,
            torch.complex64,
        )
        cbed_fc = _fourier_crop_reference(
            self.s_meta,
            self.T32,
            self.pfft32,
            self.N,
            self.f,
            (py, px),
            self.prop32,
            self.device,
            torch.complex64,
        )
        rel = ((cbed_r - cbed_fc).abs().norm() / cbed_fc.abs().norm()).item()
        print(f"\n[R1 vs fourier-crop] rel={rel:.3e} (must be large ~1)")
        self.assertGreater(rel, 0.3)


if __name__ == "__main__":
    main()
