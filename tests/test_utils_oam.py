"""Analytic contracts for the virtual OAM sorter, ``diagnostics.metrics.oam``.

References used throughout: a vortex ``f(r) exp(i l theta)`` about the sorting
axis is an eigenstate of the OAM operator with eigenvalue ``l``, so its OAM
spectrum is a Kronecker delta at ``l``; an incoherent mixture of eigenstates
with powers ``p_l`` has spectrum ``p_l / sum(p)`` and a diagonal OAM density
matrix; a coherent superposition ``a f e^{i l1 theta} + b f e^{i l2 theta}``
with a *common* radial profile is a pure state in OAM space, so the degree of
coherence between its two channels is exactly 1.
"""

import numpy as np
import pytest
import torch

from scatterem.diagnostics.metrics.oam import (
    adjacent_coherence,
    mean_oam,
    oam_density_matrix,
    oam_spectrum,
    to_aperture_plane,
)
from scatterem.utils.interpolation import sample_image

N = 64


def _polar(n=N, center=None):
    c = n // 2 if center is None else center
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return np.hypot(y - c, x - c), np.arctan2(y - c, x - c)


def _vortex(ell, n=N, w=6.0, radial_power=None):
    """Laguerre-Gauss-like vortex r^|p| exp(-r^2/w^2) exp(i ell theta), unit power."""
    r, th = _polar(n)
    p = abs(ell) if radial_power is None else radial_power
    f = (r / w) ** p * np.exp(-(r**2) / w**2) * np.exp(1j * ell * th)
    return f / np.sqrt((np.abs(f) ** 2).sum())


@pytest.mark.parametrize("ell", [-3, -1, 0, 1, 2, 5])
def test_vortex_of_charge_l_gives_a_delta_spectrum(ell):
    ls, spec = oam_spectrum(_vortex(ell)[None], l_max=8, r_max=20.0)
    assert isinstance(spec, np.ndarray)
    assert ls.tolist() == list(range(-8, 9))
    assert spec.sum() == pytest.approx(1.0, abs=1e-12)
    assert spec[ls == ell].item() > 0.999
    assert mean_oam(ls, spec) == pytest.approx(ell, abs=5e-3)


def test_l_sign_flips_the_reported_charge():
    ls, spec = oam_spectrum(_vortex(3)[None], l_max=5, r_max=20.0, l_sign=-1)
    assert spec[ls == -3].item() > 0.999


def test_single_2d_field_is_one_mode():
    ls, a = oam_spectrum(_vortex(2), l_max=4, r_max=20.0)
    _, b = oam_spectrum(_vortex(2)[None], l_max=4, r_max=20.0)
    np.testing.assert_allclose(a, b)


def test_oam_is_preserved_by_the_aperture_plane_transform():
    """FT{f(r) e^{il theta}} = g(k) e^{il phi} (-i)^l: the charge is plane-invariant."""
    ap = to_aperture_plane(_vortex(-2, w=4.0)[None])
    assert (np.abs(ap) ** 2).sum() == pytest.approx(1.0, rel=1e-12)  # ortho FFT
    ls, spec = oam_spectrum(ap, l_max=6, r_max=24.0)
    assert spec[ls == -2].item() > 0.995


def test_incoherent_mixture_spectrum_is_the_power_ladder():
    p = np.array([0.7, 0.3])
    modes = np.stack([np.sqrt(p[0]) * _vortex(2), np.sqrt(p[1]) * _vortex(-2)])
    ls, spec = oam_spectrum(modes, l_max=4, r_max=20.0)
    assert spec[ls == 2].item() == pytest.approx(0.7, abs=1e-3)
    assert spec[ls == -2].item() == pytest.approx(0.3, abs=1e-3)
    assert mean_oam(ls, spec) == pytest.approx(2 * 0.7 - 2 * 0.3, abs=5e-3)


def test_density_matrix_diagonal_is_the_spectrum_and_it_is_a_state():
    rng = np.random.default_rng(0)
    modes = np.stack([_vortex(e) * rng.uniform(0.2, 1) for e in (-1, 0, 1, 3)])
    ls, rho = oam_density_matrix(modes, l_max=5, r_max=20.0)
    ls2, spec = oam_spectrum(modes, l_max=5, r_max=20.0)
    np.testing.assert_array_equal(ls, ls2)
    np.testing.assert_allclose(np.diag(rho).real, spec, atol=1e-12)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_adjacent_coherence_separates_mixture_from_superposition():
    a, b = _vortex(1, radial_power=1), _vortex(2, radial_power=1)
    mixture = np.stack([np.sqrt(0.6) * a, np.sqrt(0.4) * b])
    superposition = (np.sqrt(0.6) * a + np.sqrt(0.4) * b)[None]

    _, rho_mix = oam_density_matrix(mixture, l_max=4, r_max=20.0)
    g, i0, j = adjacent_coherence(rho_mix)
    assert g == pytest.approx(0.0, abs=1e-3)
    ls = np.arange(-4, 5)
    assert (ls[i0], ls[j]) == (1, 2)

    # A superposition's intensity is not rotationally symmetric, so its centroid
    # is off the vortex axis: sort about the known axis.
    axis = (N // 2, N // 2)
    _, rho_sup = oam_density_matrix(superposition, l_max=4, r_max=20.0, center=axis)
    g, i0, j = adjacent_coherence(rho_sup)
    # Exactly 1 analytically; bilinear ring sampling leaks ~2e-4 between channels.
    assert g == pytest.approx(1.0, abs=1e-3)
    assert (ls[i0], ls[j]) == (1, 2)


def test_adjacent_coherence_needs_two_channels():
    with pytest.raises(ValueError, match="at least two"):
        adjacent_coherence(np.ones((1, 1)))


def test_explicit_center_and_off_center_vortex():
    """The default axis is the intensity centroid; an explicit one is honoured."""
    r, th = _polar(center=30.0)
    f = (r / 6) * np.exp(-(r**2) / 36) * np.exp(1j * th)
    ls, spec = oam_spectrum(f[None], l_max=3, r_max=18.0)
    assert spec[ls == 1].item() > 0.999
    ls, spec = oam_spectrum(f[None], l_max=3, r_max=18.0, center=(30.0, 30.0))
    assert spec[ls == 1].item() > 0.999
    _, off = oam_spectrum(f[None], l_max=3, r_max=18.0, center=(34.0, 30.0))
    assert off[ls == 1].item() < 0.9


def test_torch_input_gives_torch_output_matching_numpy():
    modes = np.stack([_vortex(1), 0.5 * _vortex(-2)])
    ls_np, spec_np = oam_spectrum(modes, l_max=4, r_max=20.0)
    ls_t, spec_t = oam_spectrum(torch.as_tensor(modes), l_max=4, r_max=20.0)
    assert isinstance(spec_t, torch.Tensor) and isinstance(ls_t, torch.Tensor)
    np.testing.assert_array_equal(ls_t.numpy(), ls_np)
    np.testing.assert_allclose(spec_t.numpy(), spec_np, atol=1e-12)
    _, rho_t = oam_density_matrix(torch.as_tensor(modes), l_max=4, r_max=20.0)
    assert isinstance(rho_t, torch.Tensor)
    g, _, _ = adjacent_coherence(rho_t)
    assert isinstance(g, float)
    assert mean_oam(ls_t, spec_t) == pytest.approx(mean_oam(ls_np, spec_np))
    ap = to_aperture_plane(torch.as_tensor(modes))
    np.testing.assert_allclose(ap.numpy(), to_aperture_plane(modes), atol=1e-12)


def test_invalid_l_max_is_rejected():
    with pytest.raises(ValueError, match="l_max"):
        oam_spectrum(_vortex(0)[None], l_max=128, n_theta=256)


def test_invalid_l_sign_is_rejected():
    with pytest.raises(ValueError, match="l_sign"):
        oam_spectrum(_vortex(0)[None], l_max=4, l_sign=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_input_is_promoted_and_has_zero_charge(dtype):
    """A real, rotationally symmetric field is an l = 0 eigenstate."""
    gauss = torch.as_tensor(np.abs(_vortex(0)), dtype=dtype)
    ls, spec = oam_spectrum(gauss, l_max=3, r_max=20.0)
    assert spec.dtype == dtype
    assert spec[ls == 0].item() == pytest.approx(1.0, abs=1e-4)


def _frozen_bilinear(x, ys, xs):
    """Test oracle: ``oam._bilinear`` as it was before ``sample_image`` replaced it.

    Frozen verbatim copy of the removed code; never import or edit it to track
    ``sample_image``.
    """
    h, w = x.shape[-2:]
    y0, x0 = torch.floor(ys), torch.floor(xs)
    wy, wx = ys - y0, xs - x0
    out = torch.zeros(*x.shape[:-2], *ys.shape, dtype=x.dtype, device=x.device)
    for dy, ay in ((0, 1 - wy), (1, wy)):
        for dx, ax in ((0, 1 - wx), (1, wx)):
            yi, xi = (y0 + dy).long(), (x0 + dx).long()
            inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            val = x[..., yi.clamp(0, h - 1), xi.clamp(0, w - 1)]
            out = out + val * (ay * ax * inside)
    return out


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_ring_sampling_matches_the_replaced_bilinear(dtype):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(3, 12, 14, generator=g, dtype=dtype)
    real = x.real.dtype
    ys = torch.rand(5, 9, generator=g, dtype=torch.float64).to(real) * 16 - 2
    xs = torch.rand(5, 9, generator=g, dtype=torch.float64).to(real) * 18 - 2
    tol = 1e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        sample_image(x, torch.stack([ys, xs], dim=-1)),
        _frozen_bilinear(x, ys, xs),
        atol=tol,
        rtol=tol,
    )
