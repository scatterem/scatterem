"""Contract for the opt-in ``gamma_weighting`` argument of direct ptychography.

The shipped assembly normalises the double-overlap function to unit modulus and
uses its phase alone; ``gamma_weighting="wiener"`` keeps ``|Gamma|`` and divides
by the Q-dependent ``sum_K |Gamma|^2``. These tests pin that the default route
is unchanged, that the new route is reachable and produces a finite image, and
that the combinations which would silently mislead are refused.
"""

import numpy as np
import pytest
import torch

from scatterem.data import Aberrations
from scatterem.data.datasets import Dataset4DStem
from scatterem.reconstruction import run
from scatterem.reconstruction.ptychography.direct import DirectPtychographyConfig


def _synthetic_dataset(n_scan: int = 16, n_det: int = 16) -> Dataset4DStem:
    """A small bright-field-disk dataset that the vBF machinery can threshold."""
    rng = np.random.default_rng(12345)
    ky, kx = np.meshgrid(
        np.arange(n_det) - n_det // 2, np.arange(n_det) - n_det // 2, indexing="ij"
    )
    disk = (np.hypot(ky, kx) <= 4.0).astype(np.float32)
    # A smooth scan-dependent modulation gives the reconstruction something to
    # transfer; without it every scan position is identical and G is a delta.
    ry, rx = np.meshgrid(np.arange(n_scan), np.arange(n_scan), indexing="ij")
    modulation = 1.0 + 0.15 * np.cos(2 * np.pi * ry / n_scan) * np.cos(
        2 * np.pi * rx / n_scan
    )
    cube = disk[None, None] * modulation[:, :, None, None].astype(np.float32)
    cube = cube + 0.01 * rng.standard_normal(cube.shape).astype(np.float32)
    # A defocus is REQUIRED, not decoration: with every aberration zero the
    # double-overlap function is identically zero (SSB has no contrast for an
    # aberration-free in-focus probe), so both estimators return zeros and the
    # weighting could not be told apart.
    aberrations = Aberrations(array=torch.zeros(12))
    aberrations.array[0] = -2.2e4
    return Dataset4DStem.from_array(
        np.ascontiguousarray(np.clip(cube, 0.0, None)),
        energy=300e3,
        semiconvergence_angle=4.0e-3,
        scan_step=29.0,
        reciprocal_step=8.3e-3,
        transform_to_amplitudes=True,
        aberrations=aberrations,
        device=torch.device("cpu"),
    )


def test_default_matches_explicit_phase_weighting():
    """The new argument is opt-in: the default route is bit-identical."""
    dataset = _synthetic_dataset()

    default_image, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()
    explicit_image, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting="phase",
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()

    torch.testing.assert_close(default_image, explicit_image, rtol=0, atol=0)


def test_wiener_weighting_returns_a_finite_image_of_the_same_shape():
    dataset = _synthetic_dataset()

    phase_image, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()
    wiener_image, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting="wiener",
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()

    assert wiener_image.shape == phase_image.shape
    assert torch.isfinite(wiener_image).all()
    # The whole point of the flag is that it changes the estimator.
    assert not torch.allclose(wiener_image, phase_image)


def test_wiener_alpha_changes_the_result():
    """``wiener_alpha`` regularises the Q-dependent divide, so it must bite."""
    dataset = _synthetic_dataset()

    small, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting="wiener",
            wiener_alpha=1e-4,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()
    large, _ = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting="wiener",
            wiener_alpha=1.0,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()

    assert not torch.allclose(small, large)


def test_unknown_gamma_weighting_is_rejected():
    dataset = _synthetic_dataset()

    with pytest.raises(ValueError, match="gamma_weighting"):
        run(
            dataset,
            DirectPtychographyConfig(
                upsample=2,
                n_batches=4,
                gamma_weighting="wienr",
                bright_field_mask_threshold=dataset.bright_field_mask_threshold,
            ),
        )


def test_wiener_refuses_the_empirical_ssnr():
    """The half-split runs the phase-only assembly, so its SSNR would not
    describe a wiener image. Refusing beats silently mismatching them."""
    dataset = _synthetic_dataset()

    with pytest.raises(ValueError, match="empirical"):
        run(
            dataset,
            DirectPtychographyConfig(
                upsample=2,
                n_batches=4,
                gamma_weighting="wiener",
                return_snr=True,
                ssnr_method="empirical",
                bright_field_mask_threshold=dataset.bright_field_mask_threshold,
            ),
        )


def _ssnr(dataset, gamma_weighting):
    _, ssnr = run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting=gamma_weighting,
            return_snr=True,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()
    assert ssnr is not None
    return ssnr


def test_wiener_analytical_ssnr_carries_the_wiener_gain():
    """Regression: the wiener mode returned the phase-only SSNR.

    This test checks the wiring, not the physics: the returned SSNR is the
    phase-only SSNR times the gain. With a filled aperture, ``|Gamma|`` varies
    across K. The review computed the two unnormalised power-SNR factors at Q
    index (0, 2), ``sum_K |Gamma|^2 = 2.120127`` and
    ``(sum_K |Gamma|)^2 / N_live = 0.390826`` (49 live K), so the SSNR ratio
    is 5.4247. The Monte Carlo test below checks that the gain is the measured
    SNR ratio of the two images.
    """
    dataset = _synthetic_dataset()
    dataset.meta.semiconvergence_angle = 0.0007

    phase = _ssnr(dataset, "phase")
    wiener = _ssnr(dataset, "wiener")

    assert not torch.equal(wiener, phase)
    ratio = float(wiener[0, 2] / phase[0, 2])
    assert ratio == pytest.approx(2.120127 / 0.390826, rel=1e-4)
    # Cauchy-Schwarz: the matched weights never lose SNR against unit weights.
    live = phase > 0
    assert torch.all(wiener[live] >= phase[live] * (1 - 1e-5))
    assert torch.all(wiener[~live] == 0)


def test_wiener_ssnr_equals_phase_ssnr_where_gamma_magnitude_is_flat():
    """Where every live K has the same ``|Gamma|``, the two images agree up to a
    per-Q factor, so their SSNRs must agree too."""
    dataset = _synthetic_dataset()

    torch.testing.assert_close(
        _ssnr(dataset, "wiener"), _ssnr(dataset, "phase"), rtol=1e-4, atol=0
    )


def test_wiener_ssnr_uses_the_callers_bright_field_mask(monkeypatch):
    """Regression: the gain sums over the caller's mask, so the
    phase-only factor must too. Before, it took the 0.5 default."""
    from scatterem.reconstruction.ptychography.direct import ctf

    seen = []
    real = ctf.direct_ptychography_ssnr

    def spy(*args, **kwargs):
        seen.append(kwargs.get("bright_field_mask_threshold"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ctf, "direct_ptychography_ssnr", spy)
    dataset = _synthetic_dataset()
    run(
        dataset,
        DirectPtychographyConfig(
            upsample=2,
            n_batches=4,
            gamma_weighting="wiener",
            return_snr=True,
            bright_field_mask_threshold=0.3,
        ),
    )
    assert seen == [0.3]


def _measured_snr_ratio(n_seeds: int = 200):
    """``(gain, measured SNR_wiener / SNR_phase)`` per Q, by Monte Carlo.

    The clean ``G(Q, K) = Gamma(Q, K) O(Q)`` is the weak-phase forward model
    with ``O = i FFT(real object)``. The noise is real, independent and of
    equal variance in every bright-field image. Both estimators are linear in
    ``G``, so the signal comes from the clean ``G`` and the noise power from
    the mean ``|FFT(image)|^2`` of noise-only runs on the same seeds.
    """
    from scatterem.reconstruction.ptychography.direct import _gamma_weighting as gw
    from scatterem.reconstruction.ptychography.direct._chunking import (
        _upsampled_grid,
    )
    from scatterem.reconstruction.ptychography.direct._common import _resolve_vbf
    from scatterem.reconstruction.ptychography.direct.core import (
        _assemble_phase_only,
    )

    dataset = _synthetic_dataset()
    dataset.meta.semiconvergence_angle = 0.0007
    vbf = _resolve_vbf(dataset, dataset.bright_field_mask_threshold)
    up = np.array([1, 1])
    _, gain = gw._wiener_image_and_gain(vbf, up, 1e-3, 16, "preserve", 0)
    ny, nx, nk = vbf.G.shape
    _, qy, qx = _upsampled_grid(vbf, up)
    q_y, q_x = torch.meshgrid(qy, qx, indexing="ij")
    meta = vbf.meta
    gamma = gw._gamma(
        vbf.k[:, 0].view(-1, 1, 1),
        vbf.k[:, 1].view(-1, 1, 1),
        q_y,
        q_x,
        float(meta.wavelength),
        float(meta.semiconvergence_angle),
        meta.aberrations.array,
    )
    rng = torch.Generator().manual_seed(0)
    obj = 1j * torch.fft.fft2(torch.randn(ny, nx, generator=rng), norm="ortho")
    clean = (gamma * obj[None]).permute(1, 2, 0).to(torch.complex64).contiguous()

    def spectra(g):
        vbf._G = g
        wiener = gw.direct_ptychography_wiener(vbf, up)
        phase = _assemble_phase_only(vbf, up, 4, "preserve", 0)
        return torch.fft.fft2(wiener), torch.fft.fft2(phase)

    signal_w, signal_p = spectra(clean)
    noise_w = torch.zeros(ny, nx)
    noise_p = torch.zeros(ny, nx)
    for seed in range(n_seeds):
        rng = torch.Generator().manual_seed(100 + seed)
        noise = torch.randn(ny, nx, nk, generator=rng)
        f_w, f_p = spectra(torch.fft.fft2(noise, dim=(0, 1), norm="ortho"))
        noise_w += f_w.abs().square()
        noise_p += f_p.abs().square()
    vbf.clear_memory()
    snr_w = signal_w.abs().square() / noise_w
    snr_p = signal_p.abs().square() / noise_p
    live = (gain > 0) & (signal_p.abs() > 1e-3 * signal_p.abs().max())
    return gain[live], (snr_w / snr_p)[live]


def test_wiener_gain_is_the_measured_snr_ratio_of_the_two_images():
    """Check the claim, not the formula.

    200 noise seeds take about 1.5 s on one CPU thread. At 200 seeds the
    per-Q ratio scatters by a few percent (up to about 20 % at single Q), so
    the bars are on the median over the live Q.
    """
    gain, measured = _measured_snr_ratio()
    assert gain.numel() > 100
    rel = (measured / gain - 1).abs()
    assert float(rel.median()) < 0.05
    varied = gain > 1.5
    assert int(varied.sum()) > 10
    assert float(rel[varied].median()) < 0.1
    # The phase-only SSNR (gain 1) would be wrong by far more there.
    assert float((measured[varied] - 1).abs().median()) > 0.4


def test_negative_wiener_alpha_is_rejected():
    from scatterem.reconstruction.ptychography.direct import (
        direct_ptychography_wiener,
    )
    from scatterem.reconstruction.ptychography.direct._common import _resolve_vbf

    dataset = _synthetic_dataset()
    vbf = _resolve_vbf(dataset, 0.5)

    with pytest.raises(ValueError, match="wiener_alpha"):
        direct_ptychography_wiener(vbf, (2, 2), wiener_alpha=-1.0)
