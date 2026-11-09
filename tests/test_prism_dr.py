"""PrismScanOp must consume sub-pixel scan offsets `dr` so that
PrismScanOp(probe, dr) matches MultisliceScanOp(self.shift(probe, dr))
for any dr in [-0.5, 0.5)."""

from unittest import TestCase, main

import numpy as np
import pytest
import torch

from scatterem.data import SMeta
from scatterem.nn.modules import Propagator, SubpixShift
from scatterem.nn.scan_exit_wave import MultisliceScanOp, PrismScanOp
from scatterem.physics.optics import smooth_disc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("ieee_float32")  # atol=1e-5, einsum-heavy: no TF32
class TestPrismScanOpAbsorbsDR(TestCase):
    """For pure PRISM (Bp == B, identity NNW + one-hot beamlets) and a
    M == NY full-FOV configuration, PrismScanOp(probe, dr) must equal
    MultisliceScanOp(probe_shifted_by_dr, dr=0) up to atol=rtol=1e-5."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"
        self.NY = self.NX = 32
        self.NZ_bin = 4
        self.K = 4
        self.M = self.NY  # full-FOV
        self.sampling = (0.1, 0.1)

        ap = smooth_disc((self.NY, self.NX), 4.0, rim_px=1.0, device=self.device)
        take = ap > ap.max() * 0.2
        self.s_meta = SMeta(
            all_beams=take.cpu().numpy(),
            parent_beams=take.cpu().numpy(),
            numerical_aperture_radius_pixels=int(take.sum().sqrt().item()),
            dx=np.array(self.sampling, dtype=np.float32),
            M=tuple(take.shape),
            device=torch.device(self.device),
        )
        # Pure-PRISM: identity NNW + one-hot beamlets at parent positions.
        Bp = self.s_meta.Bp
        self.s_meta.natural_neighbor_weights = torch.eye(Bp, device=self.device)
        # ``SMeta.beamlets`` is the GATHERED (Bp, B) form, not a dense
        # (Bp, MY, MX) one-hot grid (s_meta.py:172-177).  Build dense, then
        # gather at the beam coordinates -- exactly the gather the consumer
        # used to do internally.
        dense = torch.zeros((Bp, self.NY, self.NX), device=self.device)
        for p, (y, x) in enumerate(self.s_meta.parent_beams_coords.tolist()):
            dense[p, int(y) % self.NY, int(x) % self.NX] = 1.0
        by = self.s_meta.all_beams_coords[:, 0].long() % self.NY
        bx = self.s_meta.all_beams_coords[:, 1].long() % self.NX
        self.s_meta.beamlets = dense[:, by, bx]  # (Bp, B)

        # K-varying probes are not a supported PRISM input (the basis is
        # position-independent; only the sub-pixel ramp varies per position).
        # Build an aperture-limited probe with a single K-slice, then broadcast
        # it over K by expanding.
        probe_fft_single = torch.zeros(
            (1, 1, self.NY, self.NX), dtype=torch.complex64, device=self.device
        )
        ay = self.s_meta.all_beams_coords[:, 0].long() % self.NY
        ax = self.s_meta.all_beams_coords[:, 1].long() % self.NX
        probe_fft_single[..., ay, ax] = torch.randn(
            (1, 1, self.s_meta.B),
            dtype=torch.complex64,
            device=self.device,
        )
        probe_single = torch.fft.ifft2(probe_fft_single, norm="ortho")  # (1, 1, NY, NX)
        self.probe = probe_single.expand(1, self.K, self.NY, self.NX).contiguous()

        self.T_full = torch.randn(
            (self.NZ_bin, self.NY, self.NX),
            dtype=torch.complex64,
            device=self.device,
        )
        self.positions = torch.randint(
            0,
            self.NY,
            (self.K, 2),
            dtype=torch.int32,
            device=self.device,
        )
        # Non-trivial dr in (-0.5, 0.5).
        self.dr = (torch.rand((self.K, 2), device=self.device) - 0.5) * 0.8
        self.propagator = Propagator(
            slice_thickness=1.0,
            wavelength=0.025,
            dx=self.sampling,
            shape=(self.NY, self.NX),
            device=self.device,
        )

    def test_prism_with_dr_matches_multislice_with_shifted_probe(self) -> None:
        # Multislice reference: shift probe by dr in real space, then run on
        # per-position patches with dr=0.
        shifter = SubpixShift(
            (1, 1, self.NY, self.NX),
            self.device,
            subpix_shift=True,
        ).to(self.device)
        probe_shifted = shifter(self.probe, self.dr)
        py = self.positions[:, 0].long()
        px = self.positions[:, 1].long()
        my = torch.arange(self.M, device=self.device)
        mx = torch.arange(self.M, device=self.device)
        gy = (py[:, None, None] + my[None, :, None]) % self.NY
        gx = (px[:, None, None] + mx[None, None, :]) % self.NX
        T_patches = self.T_full[:, gy, gx]
        ms = MultisliceScanOp().to(self.device)
        out_ms = ms(
            T_patches,
            probe_shifted,
            self.positions,
            torch.zeros_like(self.dr),
            self.propagator,
        )

        # PrismScanOp: pass un-shifted probe + non-zero dr.
        prism = PrismScanOp(self.s_meta, f=1).to(self.device)
        out_prism = prism(
            self.T_full,
            self.probe,
            self.positions,
            self.dr,
            self.propagator,
        )

        torch.testing.assert_close(out_prism, out_ms, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    main()
