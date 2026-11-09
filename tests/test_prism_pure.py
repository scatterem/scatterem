"""Pure-PRISM tests: parent_beams == all_beams, f == 1.

Algorithmic equivalence to multislice. Pelz 2021 explicitly notes
this in 'Calculation of Individual STEM Probes'."""

from unittest import TestCase, main

import numpy as np
import pytest
import torch

from scatterem.data import SMeta
from scatterem.nn.modules import Propagator
from scatterem.nn.scan_exit_wave import MultisliceScanOp, PrismScanOp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPrismScanOpSMatrix(TestCase):
    """Steps 1-3 of PrismScanOp.forward: produces an S-matrix of the
    expected shape (Bp, NY/f, NX/f) when given a full-FOV transmission."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"
        self.NY, self.NX = 32, 32
        self.NZ_bin = 4
        self.f = 1
        # Tiny aperture: 6x6 box -> B = 36 beams. With f=1 and Bp=B,
        # this is exactly pure-PRISM.
        take = np.zeros((self.NY, self.NX), dtype=bool)
        take[13:19, 13:19] = True
        self.s_meta = SMeta(
            all_beams=take,
            parent_beams=take,
            numerical_aperture_radius_pixels=3,
            dx=np.array([0.1, 0.1], dtype=np.float32),
            M=(self.NY, self.NX),
            device=torch.device(self.device),
        )
        self.T = torch.randn(
            (self.NZ_bin, self.NY, self.NX),
            dtype=torch.complex64,
            device=self.device,
        )
        self.propagator = Propagator(
            slice_thickness=1.0,
            wavelength=0.025,
            dx=(0.1, 0.1),
            shape=(self.NY, self.NX),
            device=self.device,
        )

    def test_wants_full_fov_is_true(self) -> None:
        op = PrismScanOp(self.s_meta, f=1).to(self.device)
        self.assertTrue(op.wants_full_fov)

    def test_build_s_matrix_shape(self) -> None:
        op = PrismScanOp(self.s_meta, f=1).to(self.device)
        S = op._build_s_matrix(self.T, self.propagator)
        self.assertEqual(
            tuple(S.shape),
            (self.s_meta.Bp, self.NY // self.f, self.NX // self.f),
        )
        self.assertEqual(S.dtype, torch.complex64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("ieee_float32")
class TestPureMatchesPrismMultislice(TestCase):
    """With Bp == B (parent_beams = all_beams) and f = 1, PrismScanOp
    must agree numerically with MultisliceScanOp on the same V.

    The PRISM equivalence holds exactly for aperture-limited probes (probes
    whose Fourier support is contained in the aperture). The probe is
    constructed by generating random FFT coefficients only at the aperture
    beam locations, then IFFTing to real space. This ensures the beamlet
    sum exactly reconstructs the probe up to float32 rounding.

    TF32 matmul is disabled for this test class because the batched complex
    einsum uses cuBLAS GEMM internally; TF32 reduces the mantissa to 10 bits
    and inflates the error above the 1e-5 threshold.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"
        self.NY, self.NX = 32, 32
        self.NZ_bin = 4
        self.K = 4  # small batch
        self.M = self.NY  # patch size == FOV (no per-position cropping needed)
        # 6x6 aperture, take all of it as parents.
        take = np.zeros((self.NY, self.NX), dtype=bool)
        take[13:19, 13:19] = True
        self.s_meta = SMeta(
            all_beams=take,
            parent_beams=take,
            numerical_aperture_radius_pixels=3,
            dx=np.array([0.1, 0.1], dtype=np.float32),
            M=(self.NY, self.NX),
            device=torch.device(self.device),
        )
        # Make natural_neighbor_weights = identity for pure-PRISM.
        s_meta_full = self.s_meta
        Bp = s_meta_full.Bp
        s_meta_full.natural_neighbor_weights = torch.eye(
            Bp,
            dtype=torch.float32,
            device=self.device,
        )
        # Beamlets: each parent's beamlet is a single-pixel mask at its own
        # aperture coordinate.  ``SMeta.beamlets`` is the GATHERED (Bp, B) form
        # -- the weight of each of the B beams in each parent -- not the dense
        # (Bp, MY, MX) one-hot grid (see s_meta.py:172-177: the consumer only
        # ever reads the weights at the beams, so the dense scatter was pure
        # waste).  Build the dense grid for readability, then gather it at the
        # beam coordinates, which is exactly what the consumer used to do.
        dense = torch.zeros(
            (Bp, self.NY, self.NX),
            dtype=torch.float32,
            device=self.device,
        )
        for p, (y, x) in enumerate(s_meta_full.parent_beams_coords.tolist()):
            dense[p, int(y) % self.NY, int(x) % self.NX] = 1.0
        by = s_meta_full.all_beams_coords[:, 0].long() % self.NY
        bx = s_meta_full.all_beams_coords[:, 1].long() % self.NX
        s_meta_full.beamlets = dense[:, by, bx]  # (Bp, B)

        # Synthetic V.
        self.T_full = torch.randn(
            (self.NZ_bin, self.NY, self.NX),
            dtype=torch.complex64,
            device=self.device,
        )
        # K-varying probes are not a supported PRISM input (the basis is
        # position-independent; only the sub-pixel ramp varies per position).
        # Build an aperture-limited probe with a single K-slice, then broadcast
        # it over K by expanding.  The PRISM beamlet sum reconstructs exactly
        # this probe (up to float32 rounding), so the equivalence test is meaningful.
        ay = s_meta_full.all_beams_coords[:, 0].long() % self.M
        ax = s_meta_full.all_beams_coords[:, 1].long() % self.M
        aperture_mask = torch.zeros(
            (self.M, self.M),
            dtype=torch.bool,
            device=self.device,
        )
        aperture_mask[ay, ax] = True
        probe_rand = torch.randn(
            (1, 1, self.M, self.M),
            dtype=torch.complex64,
            device=self.device,
        )
        probe_fft = torch.fft.fft2(probe_rand, norm="ortho")
        probe_single = torch.fft.ifft2(
            probe_fft * aperture_mask,
            norm="ortho",
        )  # (1, 1, M, M)
        self.probe = probe_single.expand(1, self.K, self.M, self.M).contiguous()
        # All scan positions at the origin so PRISM's per-position crop
        # is trivially the identity (M == NY).
        self.positions = torch.zeros(
            (self.K, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self.dr = torch.zeros(
            (self.K, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.propagator = Propagator(
            slice_thickness=1.0,
            wavelength=0.025,
            dx=(0.1, 0.1),
            shape=(self.M, self.M),
            device=self.device,
        )

    def test_pure_prism_matches_multislice(self) -> None:
        # Build T_patches such that each per-position patch == T_full
        # (positions are all at origin, M == NY -> the patch is the whole FOV).
        T_patches = self.T_full.unsqueeze(1).expand(self.NZ_bin, self.K, self.M, self.M)
        T_patches = T_patches.contiguous()

        ms = MultisliceScanOp().to(self.device)
        out_ms = ms(T_patches, self.probe, self.positions, self.dr, self.propagator)

        prism = PrismScanOp(self.s_meta, f=1).to(self.device)
        out_prism = prism(
            self.T_full, self.probe, self.positions, self.dr, self.propagator
        )

        torch.testing.assert_close(out_prism, out_ms, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    main()
