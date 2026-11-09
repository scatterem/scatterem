"""Partitioned-PRISM error envelope vs. multislice.

Pelz 2021 Fig. 4d publishes RMS errors of ~5%, ~2%, ~0.5% for
B_p = 19, 61, 217 parent beams at f=4. We accept up to 1.5x those
values to leave headroom for sample / aperture differences.

THE SPECIMEN IS A REAL POTENTIAL, DELIBERATELY.  This test used to run on
``T = torch.randn(complex)``, which is not a transmission function at all --
its modulus is not 1, so it is not unitary and it scatters like nothing
physical.  That mattered: on random T the focal back-propagation is a net
LOSS even when correctly compensated (0.117/0.041 against 0.067/0.023 for no
referencing at all), so the test asserted the opposite of the truth and would
have argued for deleting the referencing.  On a real SrTiO3 potential the
same comparison is 0.0034/0.0010 against 0.0115/0.0033 -- a 3.4x WIN.  A
specimen that is not a specimen cannot arbitrate physics.

Physics setup
-------------
PRISM builds its S-matrix on the full simulation grid (NY x NX) using a
full-FOV Fresnel propagator.  The multislice reference runs the same
propagator on per-position patches gathered from the same T_full.  With
M == NY (patch size equal to full-FOV size) and positions wrapping modulo
NY, every patch is a cyclic shift of T_full, which is exactly what the
PRISM crop step does.  Both operators therefore see identical physics;
the only approximation error is the NNW partitioning (Pelz 2021 Fig. 4d).

The probe is built aperture-limited on the (NY x NX) grid: random complex
coefficients placed at the aperture beam positions, then IFFTed to real
space.  PRISM reconstructs such a probe exactly (up to float32 rounding)
via the beamlet sum, so the test isolates the NNW error only.

Focal back-propagation
----------------------
``PrismScanOp(focal_backprop=...)`` has ``"centroid"`` as
the default.  It Fresnel-back-propagates the parent columns to the scattering
centroid before the NNW fusion and never undoes the shift, so the returned
wave is referenced to that plane rather than to the exit plane -- a second,
NON-partitioning error source that this envelope is not measuring.  The
envelope tests therefore pin ``focal_backprop=0.0`` so the residual is the
NNW approximation alone, exactly as the module docstring above claims.  The
plane shift itself is pinned separately by
``test_default_focal_backprop_returns_exit_plane_wave`` below.
"""

from unittest import TestCase, main

import numpy as np
import pytest
import torch

from scatterem.data import SMeta
from scatterem.nn.modules import Propagator
from scatterem.nn.scan_exit_wave import MultisliceScanOp, PrismScanOp
from scatterem.physics.atom_set import AtomSet
from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.optics import smooth_disc
from scatterem.physics.potential_stack import potential_stack
from scatterem.physics.thin_slice import thin_slice_transmission


def _srtio3_cell(a: float, nx: int, ny: int, sides: np.ndarray) -> AtomSet:
    """Cubic SrTiO3 (Pm-3m), ``nx`` x ``ny`` x 1 cells of side ``a`` [Å].

    Sr at the corner, Ti at the body centre and O at the face centres. The
    mean squared displacements [Å²] are Sr 0.0086, Ti 0.0064 and O 0.006.
    """
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(1), indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(float)
    positions = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    element_z = np.tile(np.array([38, 22, 8, 8, 8]), len(cells)).astype(np.int64)
    msd = np.array([{38: 0.0086, 22: 0.0064, 8: 0.006}[int(z)] for z in element_z])
    return AtomSet(
        np.asarray(sides, dtype=float),
        positions / sides[None, :],
        element_z,
        msd=msd,
        remark="SrTiO3",
    )


def _build_s_meta_partitioned(
    aperture_array: torch.Tensor,
    n_radial: int,
    n_angular: int,
    sampling: np.ndarray,
    device: torch.device,
) -> SMeta:
    """Construct an SMeta whose parent_beams are a hex-ring partition
    of all_beams, then run make_beamlet_meta to populate weights."""
    take = aperture_array > aperture_array.max() * 0.2
    s_meta = SMeta(
        all_beams=take.cpu().numpy(),
        parent_beams=take.cpu().numpy(),
        numerical_aperture_radius_pixels=int(take.sum().sqrt().item()),
        dx=sampling,
        M=tuple(take.shape),
        device=device,
    )
    return s_meta.make_beamlet_meta(
        n_radial_samples=n_radial, n_angular_samples=n_angular
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPartitionedErrorEnvelope(TestCase):
    """Same V, same probe, same positions; vary Bp via (n_radial, n_angular)
    and check that RMS amplitude error vs. multislice falls within the
    published envelope.

    Grid choice: M == NY so the full-FOV PRISM propagator is identical to
    the per-patch multislice propagator.  With M == NY every per-position
    patch is a cyclic shift of T_full (same physics as the PRISM crop step),
    giving a fair apples-to-apples comparison.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = "cuda"
        self.NY = self.NX = 64
        self.NZ_bin = 4
        self.K = 16
        # M == NY: probe covers the full simulation cell so PRISM (full-FOV)
        # and multislice use the same Fresnel propagator.
        self.M = self.NY
        self.sampling = np.array([0.1, 0.1], dtype=np.float32)

        ap = smooth_disc((self.NY, self.NX), 8.0, rim_px=1.0, device=self.device)
        self.aperture = ap

        # Real SrTiO3: unit-modulus transmission from real scattering
        # factors. The lattice constant is chosen to tile the existing
        # 6.4 A grid (64 px at 0.1 A) so nothing else in the fixture moves.
        a = self.NY * float(self.sampling[0]) / 2.0
        cell = np.array([2 * a, 2 * a, float(self.NZ_bin)])
        st = _srtio3_cell(a, 2, 2, cell)
        self.T_full = (
            thin_slice_transmission(
                200e3,
                potential_stack(
                    BeamPartition.uniform(self.NZ_bin),
                    st,
                    (self.NY, self.NX),
                    device=torch.device(self.device),
                    dtype=torch.float32,
                ),
                bandlimit=2 / 3,
            )
            .reshape(-1, self.NY, self.NX)
            .to(torch.complex64)
        )

        # Build an aperture-limited probe on the (NY x NX) grid.  PRISM
        # reconstructs such probes exactly via the beamlet sum, so the
        # RMS error isolates the NNW approximation only.
        take = ap > ap.max() * 0.2
        s_meta_ref = SMeta(
            all_beams=take.cpu().numpy(),
            parent_beams=take.cpu().numpy(),
            numerical_aperture_radius_pixels=int(take.sum().sqrt().item()),
            dx=self.sampling,
            M=(self.NY, self.NX),
            device=torch.device(self.device),
        )
        B = s_meta_ref.B
        ay = s_meta_ref.all_beams_coords[:, 0].long() % self.NY
        ax = s_meta_ref.all_beams_coords[:, 1].long() % self.NX
        probe_fft = torch.zeros(
            (1, self.K, self.M, self.M),
            dtype=torch.complex64,
            device=self.device,
        )
        # Same random coefficients for all K positions (position-independent
        # probe, as in standard STEM).
        rand_coeffs = torch.randn(
            (1, 1, B),
            dtype=torch.complex64,
            device=self.device,
        ).expand(1, self.K, -1)
        probe_fft[:, :, ay, ax] = rand_coeffs
        self.probe = torch.fft.ifft2(probe_fft, norm="ortho")  # (1, K, M, M)

        # Scan positions: wrap by NY (= M).
        self.positions = torch.randint(
            0,
            self.NY,
            (self.K, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self.dr = torch.zeros(
            (self.K, 2),
            dtype=torch.float32,
            device=self.device,
        )
        # Full-FOV propagator (shape = NY x NX = M x M).
        self.propagator = Propagator(
            slice_thickness=1.0,
            wavelength=0.025,
            dx=tuple(self.sampling.tolist()),
            shape=(self.M, self.M),
            device=self.device,
        )

    def _multislice_reference(self) -> torch.Tensor:
        # Gather per-position (M x M) patches from T_full (with M == NY,
        # this is a cyclic shift — same as PRISM's crop step).
        py = self.positions[:, 0].long()
        px = self.positions[:, 1].long()
        my = torch.arange(self.M, device=self.device)
        mx = torch.arange(self.M, device=self.device)
        gy = (py[:, None, None] + my[None, :, None]) % self.NY
        gx = (px[:, None, None] + mx[None, None, :]) % self.NX
        T_patches = self.T_full[:, gy, gx]  # (NZ_bin, K, M, M)
        ms = MultisliceScanOp().to(self.device)
        return ms(T_patches, self.probe, self.positions, self.dr, self.propagator)

    def _partitioned_and_reference(
        self,
        n_radial: int,
        n_angular: int,
        focal_backprop: "float | str",
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        s_meta = _build_s_meta_partitioned(
            self.aperture,
            n_radial,
            n_angular,
            self.sampling,
            torch.device(self.device),
        )
        prism = PrismScanOp(
            s_meta,
            f=1,
            focal_backprop=focal_backprop,
        ).to(self.device)
        # PrismScanOp de-tilts (and, with back-prop on, chirps) the S-matrix in
        # place; T_full itself is not written, but clone so repeated calls in one
        # test are order-independent.
        out_p = prism(
            self.T_full.clone(),
            self.probe,
            self.positions,
            self.dr,
            self.propagator,
        )
        return out_p, self._multislice_reference()

    @staticmethod
    def _rms_amplitude(out_p: torch.Tensor, out_ms: torch.Tensor) -> float:
        amp_p = out_p.abs()
        amp_ms = out_ms.abs()
        rms = (amp_p - amp_ms).pow(2).mean().sqrt()
        denom = amp_ms.pow(2).mean().sqrt()
        return (rms / denom).item()

    def _rms_amplitude_error(
        self,
        n_radial: int,
        n_angular: int,
        focal_backprop: "float | str" = 0.0,
    ) -> float:
        # focal_backprop=0.0 -> the operator returns the EXIT-plane wave, which
        # is the plane the multislice reference is at, so the residual measured
        # here is the NNW partitioning error alone (see module docstring).
        return self._rms_amplitude(
            *self._partitioned_and_reference(n_radial, n_angular, focal_backprop)
        )

    def test_few_parents_rough(self) -> None:
        # Bp = 19 (Pelz 2021): published ~5% RMS; allow 7.5%.
        err = self._rms_amplitude_error(n_radial=2, n_angular=6)
        self.assertLess(err, 0.075, f"Bp~19 rms_err={err:.4f}, expected < 0.075")

    def test_medium_parents(self) -> None:
        # Bp = 61: published ~2%; allow 3%.
        err = self._rms_amplitude_error(n_radial=4, n_angular=6)
        self.assertLess(err, 0.03, f"Bp~61 rms_err={err:.4f}, expected < 0.03")

    def test_many_parents_tight(self) -> None:
        # Bp = 217: published ~0.5%; allow 0.75%.
        err = self._rms_amplitude_error(n_radial=8, n_angular=12)
        self.assertLess(err, 0.0075, f"Bp~217 rms_err={err:.4f}, expected < 0.0075")

    def test_focal_backprop_beats_no_referencing(self) -> None:
        """The assertion that actually pins the forward-propagation fix.

        The envelope bounds alone are a weak discriminator at Bp=19 (the
        un-propagated result was 0.0779 against a 0.075 bound, a 4 % margin).
        This is the strong one: referencing to the scattering centroid must
        BEAT not referencing at all, and before the fix it lost to it by 7x.
        """
        for n_radial, n_angular in ((2, 6), (4, 6)):
            # Explicit, not relied-upon-default: _rms_amplitude_error defaults
            # to focal_backprop=0.0 (the envelope tests above pin it there so
            # they isolate NNW-only error), so "on" must ask for "centroid"
            # by name or this comparison silently degenerates to off vs off.
            on = self._rms_amplitude_error(
                n_radial, n_angular, focal_backprop="centroid"
            )
            off = self._rms_amplitude_error(n_radial, n_angular, focal_backprop=0.0)
            self.assertLess(
                on,
                off,
                f"Bp~{n_radial * n_angular}: focal_backprop on={on:.4f} "
                f"must beat off={off:.4f}",
            )

    def test_default_focal_backprop_returns_exit_plane_wave(self) -> None:
        """The returned wave must be at the exit plane: no extra Fresnel
        propagation may improve agreement with the multislice exit wave.

        Convention-free -- it scans the extra propagation distance instead of
        assuming the back-prop distance, so it stays valid whatever reference
        plane 'centroid' picks.

        Regression guard for a fixed bug. ``PrismScanOp`` used to Fresnel
        back-propagate the parent columns by ``n_half = bp_frac*NZ`` slices
        before the NNW fusion and never re-propagate forward, leaving the whole
        returned wave ``n_half`` slices upstream of the exit plane. With NZ=4
        and bp_frac=0.5015 the amplitude error against multislice was 0.763 at
        the exit plane and bottomed out at 0.041 two slices downstream; it is
        now 0.041 at the exit plane, where it belongs.

        The feature's own validation could not see this: it compares
        ``|FFT(exit)|**2``, and a propagation applied to a *final* wave is a
        pure phase in Fourier space (verified: 4e-07 relative change in
        intensity, 2.6e-01 in real-space amplitude). What made the omission
        observable in intensity anyway is that the back-propagation is applied
        *before* the per-column de-tilt, and a real-space phase ramp is a
        Fourier-space shift, which does not commute with a Fourier-space
        multiplication.
        """
        out_p, out_ms = self._partitioned_and_reference(
            n_radial=4,
            n_angular=6,
            focal_backprop="centroid",
        )
        # prop.prop = exp(-i pi lam dz |Q|^2); raising it to +n propagates n
        # further slices downstream (towards/past the exit plane).
        phase = torch.angle(self.propagator.prop)
        errs = {}
        for n in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            shifted = torch.fft.ifft2(torch.fft.fft2(out_p) * torch.exp(1j * n * phase))
            errs[n] = self._rms_amplitude(shifted, out_ms)
        best = min(errs, key=errs.get)
        self.assertEqual(
            best,
            0.0,
            "exit wave is not at the exit plane: agreement with multislice is "
            f"best after {best:+g} extra Fresnel slices "
            f"(err {errs[best]:.4f} vs {errs[0.0]:.4f} at the exit plane); "
            f"full scan { ({k: round(v, 4) for k, v in errs.items()}) }",
        )


if __name__ == "__main__":
    main()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFocalBackpropIsPhaseOnlyInReciprocalSpace(TestCase):
    """The forward propagation changes the exit WAVE but not the PATTERN.

    This is why the missing propagation survived so long: it is a phase-only
    multiply in reciprocal space, so every diffraction-plane observable --
    4D-STEM cubes and ADF/HAADF -- was already correct. Only consumers of the
    REAL-SPACE exit wave saw it, such as direct wave-function outputs.

    Keeping it as a test fixes the blast radius in place: if a future change
    to the reference-plane physics ever alters a detector signal, that is a
    different kind of change and should be noticed deliberately.
    """

    def test_diffraction_intensity_is_unchanged_by_the_forward_propagation(self):
        env = TestPartitionedErrorEnvelope("test_few_parents_rough")
        env.setUp()
        s_meta = _build_s_meta_partitioned(
            env.aperture, 2, 6, env.sampling, torch.device(env.device)
        )
        args = (env.T_full, env.probe, env.positions, env.dr, env.propagator)
        fixed = PrismScanOp(s_meta, f=1).to(env.device)(*args)
        orig = PrismScanOp._to_exit_plane
        PrismScanOp._to_exit_plane = lambda self, w, p: w
        try:
            bug = PrismScanOp(s_meta, f=1).to(env.device)(*args)
        finally:
            PrismScanOp._to_exit_plane = orig

        def rel(a, b):
            return float((a - b).abs().norm() / b.abs().norm())

        # the real-space wave really does move ...
        self.assertGreater(rel(fixed, bug), 0.05)
        # ... and the pattern really does not
        I_fixed = torch.fft.fft2(fixed).abs() ** 2
        I_bug = torch.fft.fft2(bug).abs() ** 2
        self.assertLess(rel(I_fixed, I_bug), 1e-5)
