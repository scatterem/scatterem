"""Analytic contracts for the mixed-state probe metrics and gauge removal.

References: for mode stacks ``a_i``, ``b_j`` the density matrices are
``rho_a = sum_i |a_i><a_i|`` and ``Tr(rho_a rho_b) = sum_ij |<a_i|b_j>|^2``, so

* the normalised Hilbert-Schmidt fidelity of a state with itself is 1, and is
  unchanged by any unitary remixing of the modes (including per-mode phases);
* two states diagonal in one orthonormal basis with populations ``p``, ``q``
  have fidelity ``p.q / (|p| |q|)``; orthogonal pure states have fidelity 0;
* the purity of orthonormal modes with powers ``p_k`` is ``sum p^2 / (sum p)^2``.

The ptychographic gauges (common translation, linear phase ramp) are checked
against states whose centroid and ramp are known in closed form.
"""

import numpy as np
import pytest
import torch

from scatterem.diagnostics.metrics.probe_gauge import (
    remove_phase_ramp_gauge,
    remove_probe_gauges,
    remove_translation_gauge,
)

N = 48


def _random_modes(m, n=12, seed=0, dtype=torch.complex128):
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(m, n, n, generator=g, dtype=torch.float64)
    im = torch.randn(m, n, n, generator=g, dtype=torch.float64)
    return (re + 1j * im).to(dtype)


def _random_unitary(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(k, k, generator=g, dtype=torch.float64) + 1j * torch.randn(
        k, k, generator=g, dtype=torch.float64
    )
    return torch.linalg.qr(a)[0]


def _orthonormal_modes(m, n=12, seed=0):
    q = torch.linalg.qr(_random_modes(m, n, seed).reshape(m, -1).T)[0]
    return q.T.reshape(m, n, n)


def _gaussian(n=N, cy=None, cx=None, w=4.0):
    cy = n // 2 if cy is None else cy
    cx = n // 2 if cx is None else cx
    y, x = np.mgrid[0:n, 0:n].astype(float)
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * w**2)).astype(complex)


def _com(fields):
    inten = (np.abs(fields) ** 2).reshape(-1, *fields.shape[-2:]).sum(0)
    y, x = np.mgrid[0 : inten.shape[0], 0 : inten.shape[1]]
    return (y * inten).sum() / inten.sum(), (x * inten).sum() / inten.sum()


def _probe_state(n=N, seed=0):
    """Three smooth, localised modes with a power ladder (a realistic probe)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(float) - n // 2
    g = np.exp(-(y**2 + x**2) / (2 * 3.5**2))
    phase = np.exp(1j * (0.02 * (y**2 - x**2) + 0.05 * y))
    modes = np.stack([g * phase, y * g * phase / 3.5, x * g / 3.5]).astype(complex)
    modes *= np.sqrt(np.array([0.8, 0.15, 0.05]))[:, None, None]
    return modes * np.exp(1j * rng.uniform(0, 2 * np.pi))






















# --- gauge removal ------------------------------------------------------------


@pytest.mark.parametrize("n", [48, 47])
def test_translation_gauge_moves_the_centroid_to_the_fftshift_origin(n):
    modes = np.stack([_gaussian(n, 17.3, 29.6), 0.4 * _gaussian(n, 18.1, 28.2, w=3)])
    out = remove_translation_gauge(modes)
    cy, cx = _com(out)
    assert (cy, cx) == pytest.approx((n // 2, n // 2), abs=1e-6)
    np.testing.assert_allclose(
        (np.abs(out) ** 2).sum((1, 2)), (np.abs(modes) ** 2).sum((1, 2)), rtol=1e-12
    )
    back = remove_translation_gauge(out)
    np.testing.assert_allclose(back, out, atol=1e-10)


@pytest.mark.parametrize("q", [(3, -5), (-7, 2)])
def test_phase_ramp_gauge_removes_a_known_linear_ramp(q):
    base = _gaussian()
    y, x = np.mgrid[0:N, 0:N]
    ramped = base * np.exp(2j * np.pi * (q[0] * y + q[1] * x) / N)
    out = remove_phase_ramp_gauge(ramped[None])[0]
    np.testing.assert_allclose(np.abs(out), np.abs(ramped), atol=1e-12)
    overlap = np.abs(np.vdot(out, base)) / (np.linalg.norm(out) * np.linalg.norm(base))
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_remove_probe_gauges_accepts_the_solver_layout_and_drops_last_optionally():
    modes = _probe_state()
    solver_layout = torch.as_tensor(modes[:, None])  # (M, 1, H, W)
    kept = remove_probe_gauges(solver_layout)
    assert isinstance(kept, torch.Tensor) and kept.shape == (3, N, N)
    dropped = remove_probe_gauges(solver_layout, drop_last=True)
    assert dropped.shape == (2, N, N)
    none = remove_probe_gauges(modes, translation=False, phase_ramp=False)
    np.testing.assert_allclose(none, modes)
    single = remove_probe_gauges(modes[:1], drop_last=True)
    assert single.shape == (1, N, N)


def test_remove_probe_gauges_rejects_bad_shapes():
    with pytest.raises(ValueError, match="shape"):
        remove_probe_gauges(np.zeros((2, 3, 8, 8), complex))


# --- probe-state comparison ---------------------------------------------------


def _shift(fields, dy, dx):
    n = fields.shape[-1]
    k = np.fft.fftfreq(n)
    ramp = np.exp(-2j * np.pi * (dy * k[:, None] + dx * k[None, :]))
    return np.fft.ifft2(np.fft.fft2(fields) * ramp)








