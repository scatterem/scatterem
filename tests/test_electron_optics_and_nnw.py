"""Relativistic electron optics and natural-neighbour weights.

These assert the *defining properties* of each quantity -- the energy-momentum
relation, partition of unity, exactness at the parents, linear precision --
rather than agreement with any particular implementation, so a numerical
regression shows up as a physics violation rather than as a diff.
"""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, Delaunay

from scatterem.simulation._electron_optics import (
    ELECTRON_REST_ENERGY_EV,
    energy2sigma,
    energy2wavelength,
    relativistic_mass_correction,
)
from scatterem.simulation.eels._nnw import natural_neighbor_weights

ENERGIES = [1e3, 1e4, 6e4, 1e5, 2e5, 3e5, 4e5, 1e6]


# --------------------------------------------------------------------- physics


def test_rest_energy_is_511_kev():
    assert ELECTRON_REST_ENERGY_EV == pytest.approx(510_998.95, rel=1e-6)


@pytest.mark.parametrize("eV", ENERGIES)
def test_wavelength_satisfies_energy_momentum_relation(eV):
    """lambda = hc / sqrt(T^2 + 2 T m0c^2) is equivalent to E^2 = (pc)^2 + (m0c^2)^2."""
    lam = float(energy2wavelength(eV))
    hc = 12_398.419843320026  # eV*Angstrom, CODATA
    pc = hc / lam
    total = eV + ELECTRON_REST_ENERGY_EV
    assert pc**2 + ELECTRON_REST_ENERGY_EV**2 == pytest.approx(total**2, rel=1e-9)


def test_wavelength_at_300kv():
    """The value quoted throughout the BiP-PRISM paper.

    The paper and the experiment scripts carry LAM = 0.0196874, which is the
    true 0.01968749 truncated rather than rounded, so the tolerance is set by
    that truncation and not by any disagreement in the physics.
    """
    assert float(energy2wavelength(3e5)) == pytest.approx(0.0196875, abs=2e-7)


@pytest.mark.parametrize("eV", ENERGIES)
def test_gamma_matches_total_over_rest_energy(eV):
    assert float(relativistic_mass_correction(eV)) == pytest.approx(
        (eV + ELECTRON_REST_ENERGY_EV) / ELECTRON_REST_ENERGY_EV, rel=1e-12
    )


def test_sigma_tends_to_nonrelativistic_limit():
    """The relativistic factor (m0c^2+T)/(2m0c^2+T) -> 1/2 as T -> 0."""
    tiny = 1.0
    lam = float(energy2wavelength(tiny))
    assert float(energy2sigma(tiny)) == pytest.approx(
        2 * np.pi / (lam * tiny) * 0.5, rel=1e-5
    )


def test_quantities_are_monotonic_in_energy():
    E = np.array(ENERGIES)
    assert np.all(np.diff(energy2wavelength(E)) < 0)      # faster -> shorter
    assert np.all(np.diff(relativistic_mass_correction(E)) > 0)


def test_vectorised_matches_scalar():
    E = np.array(ENERGIES)
    assert np.allclose(energy2wavelength(E), [energy2wavelength(e) for e in E])
    assert np.allclose(energy2sigma(E), [energy2sigma(e) for e in E])


# ------------------------------------------------------------------------ nnw


def _hex_parents(n_radial=4, radius=10.78):
    """A disc aperture and a hex-ring parent subset, as the paper uses."""
    g = np.arange(-int(radius) - 1, int(radius) + 2)
    Y, X = np.meshgrid(g, g, indexing="ij")
    beams = np.c_[Y[Y**2 + X**2 <= radius**2], X[Y**2 + X**2 <= radius**2]].astype(float)
    ang, samples = 6, [[0.0, 0.0]]
    r_max = np.linalg.norm(beams, axis=1).max()
    for i, r in enumerate(np.linspace(0.0, r_max, n_radial + 1)[1:]):
        for a in np.linspace(-np.pi, np.pi, ang * (1 + i), endpoint=False):
            off = np.pi / ang * i
            samples.append([r * np.sin(a + off), r * np.cos(a + off)])
    d = np.linalg.norm(np.asarray(samples)[:, None, :] - beams[None, :, :], axis=2)
    return beams, beams[np.unique(np.argmin(d, axis=1))]


@pytest.mark.parametrize("method", ["linear", "sibson"])
def test_partition_of_unity(method):
    beams, parents = _hex_parents()
    w = natural_neighbor_weights(parents, beams, method=method)
    assert np.allclose(w.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ["linear", "sibson"])
def test_exact_at_the_parents(method):
    _, parents = _hex_parents()
    w = natural_neighbor_weights(parents, parents, method=method)
    assert np.allclose(w, np.eye(len(parents)))


@pytest.mark.parametrize("method", ["linear", "sibson"])
def test_weights_are_non_negative(method):
    beams, parents = _hex_parents()
    w = natural_neighbor_weights(parents, beams, method=method)
    assert (w >= 0).all()


@pytest.mark.parametrize("method", ["linear", "sibson"])
def test_reproduces_a_linear_field(method):
    """Linear precision: both schemes are exact on f(x) = a.x + c strictly inside
    the hull. Queries near the hull are excluded -- there the neighbourhood is
    one-sided and no natural-neighbour scheme reproduces a linear field."""
    beams, parents = _hex_parents()
    a, c = np.array([3.0, -2.0]), 5.0
    w = natural_neighbor_weights(parents, beams, minimum_weight_cutoff=0.0, method=method)
    hull = ConvexHull(parents)
    depth = np.min(np.abs(beams @ hull.equations[:, :2].T + hull.equations[:, 2]), axis=1)
    interior = (Delaunay(parents).find_simplex(beams) >= 0) & (depth > 1.5)
    assert interior.sum() > 100
    err = np.abs((w @ (parents @ a + c) - (beams @ a + c))[interior])
    assert err.max() < 1e-10


@pytest.mark.parametrize("method", ["linear", "sibson"])
def test_outside_the_hull_falls_back_to_nearest(method):
    _, parents = _hex_parents()
    far = np.array([[500.0, 500.0], [-400.0, 7.0]])
    w = natural_neighbor_weights(parents, far, method=method)
    assert np.allclose(w.sum(axis=1), 1.0)
    for row, q in zip(w, far):
        assert row.argmax() == np.argmin(np.linalg.norm(parents - q, axis=1))
        assert row.max() == pytest.approx(1.0)


def test_cutoff_sparsifies_but_preserves_unity():
    beams, parents = _hex_parents()
    dense = natural_neighbor_weights(parents, beams, minimum_weight_cutoff=0.0)
    sparse = natural_neighbor_weights(parents, beams, minimum_weight_cutoff=1e-2)
    assert (sparse > 0).sum() <= (dense > 0).sum()
    assert np.allclose(sparse.sum(axis=1), 1.0)


def test_rejects_unknown_method():
    _, parents = _hex_parents()
    with pytest.raises(ValueError, match="linear.*sibson"):
        natural_neighbor_weights(parents, parents, method="nonesuch")


def test_degenerate_parent_sets_fall_back_to_nearest():
    """Collinear parents have no triangulation; nearest-neighbour is the reading."""
    parents = np.c_[np.arange(5.0), np.zeros(5)]
    q = np.array([[1.4, 3.0], [3.9, -2.0]])
    w = natural_neighbor_weights(parents, q)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert w[0].argmax() == 1 and w[1].argmax() == 4
