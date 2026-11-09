"""Tests for natural_neighbor_weights nearest-known-point fallback.

Pre-fix: when a query point is far outside the convex hull of the known points,
all radius-subtraction retries in the while loop exhaust without finding a
natural-neighbor simplex, leaving weights_ind/indices_ind unbound and raising
UnboundLocalError.

Post-fix: the fallback assigns weight=1.0 to the nearest known point so the
function returns a valid weight row instead of crashing.
"""

import numpy as np
import pytest

from scatterem.utils.stem import natural_neighbor_weights

# --- helpers -----------------------------------------------------------------


def _known_points_triangle():
    """Three known points forming a small triangle near the origin."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=float)


# --- tests -------------------------------------------------------------------


class TestNNWNearestKnownPointFallback:
    """Fallback behaviour when the query lies far outside the convex hull."""

    def test_outside_hull_no_unbound_error(self):
        """Query far outside hull must not raise UnboundLocalError."""
        known = _known_points_triangle()
        # Radius ~14.14 — the loop subtracts at most 10.0, leaving r_new ~4.14
        # which is still well outside the hull, so the loop always exhausts.
        query = np.array([[10.0, 10.0]], dtype=float)
        # Must not raise
        natural_neighbor_weights(known, query)

    def test_outside_hull_weight_row_sums_to_one(self):
        """Fallback weight row must sum to 1.0 (valid interpolation weights)."""
        known = _known_points_triangle()
        query = np.array([[10.0, 10.0]], dtype=float)
        weights = natural_neighbor_weights(known, query)
        row_sum = weights[0].sum()
        assert abs(row_sum - 1.0) < 1e-9, (
            f"Expected weight row to sum to 1.0, got {row_sum}"
        )

    def test_outside_hull_nearest_point_gets_full_weight(self):
        """The nearest known point must receive weight 1.0 in the fallback row."""
        known = _known_points_triangle()
        query = np.array([[10.0, 10.0]], dtype=float)
        weights = natural_neighbor_weights(known, query)
        # Nearest point to (10,10) among [(0,0),(1,0),(0.5,1)] is (1,0) — index 1
        # (dist ~ 12.73) vs (0,0) dist=14.14 vs (0.5,1) dist=13.0
        # Actually (1,0): sqrt(81+100)=sqrt(181)~13.45, (0,0):sqrt(200)~14.14,
        # (0.5,1): sqrt(90.25+81)=sqrt(171.25)~13.09  → nearest is (0.5,1) idx=2
        dists = np.linalg.norm(known - query[0], axis=1)
        nearest_idx = int(np.argmin(dists))
        assert weights[0, nearest_idx] == pytest.approx(1.0, abs=1e-9), (
            f"Expected nearest point (index {nearest_idx}) to have weight 1.0, "
            f"got {weights[0, nearest_idx]}"
        )

    def test_inside_hull_unaffected(self):
        """A query inside the hull must still produce valid multi-point weights."""
        known = _known_points_triangle()
        # Centroid is well inside the triangle
        centroid = known.mean(axis=0, keepdims=True)
        weights = natural_neighbor_weights(known, centroid)
        row_sum = weights[0].sum()
        assert abs(row_sum - 1.0) < 1e-9, (
            f"In-hull query: expected weight row to sum to 1.0, got {row_sum}"
        )
        # At least two points must have non-zero weights (natural neighbours)
        nnz = np.count_nonzero(weights[0])
        assert nnz >= 2, f"In-hull query: expected >= 2 non-zero weights, got {nnz}"

    def test_multiple_outside_queries_all_valid(self):
        """Multiple outside-hull queries must all produce valid weight rows."""
        known = _known_points_triangle()
        queries = np.array(
            [
                [10.0, 10.0],
                [-5.0, -5.0],
                [20.0, -3.0],
            ],
            dtype=float,
        )
        weights = natural_neighbor_weights(known, queries)
        row_sums = weights.sum(axis=1)
        for i, s in enumerate(row_sums):
            assert abs(s - 1.0) < 1e-9, (
                f"Query {i}: expected weight row to sum to 1.0, got {s}"
            )


class TestLinearMethod:
    """The fast vectorised ``method='linear'`` (Delaunay barycentric) path."""

    def _hex(self, n_radial=4, n_angular=6, R=1.0):
        pts = [[0.0, 0.0]]
        for i, r in enumerate(np.linspace(0.0, R, n_radial + 1)[1:]):
            for a in np.linspace(-np.pi, np.pi, n_angular * (1 + i), endpoint=False):
                pts.append([r * np.sin(a), r * np.cos(a)])
        return np.asarray(pts)

    def test_exact_at_parents(self):
        # A query placed exactly at a known point gets weight 1 on itself.
        known = self._hex()
        w = natural_neighbor_weights(known, known, method="linear")
        assert np.allclose(np.diag(w), 1.0, atol=1e-9)
        assert np.allclose(w - np.eye(len(known)), 0.0, atol=1e-9)

    def _interior_queries(self, n=40, seed=0, R=0.6):
        # Strictly inside the parent hull (radius < parent max), not on any parent.
        rng = np.random.default_rng(seed)
        r = R * np.sqrt(rng.uniform(size=n))
        a = rng.uniform(-np.pi, np.pi, n)
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    def test_partition_of_unity_and_nonneg_inside(self):
        known = self._hex(n_radial=3, R=1.0)
        queries = self._interior_queries()
        w = natural_neighbor_weights(known, queries, method="linear")
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
        assert w.min() >= 0.0
        assert (np.count_nonzero(w, axis=1) >= 2).all()  # >= 2 neighbours inside

    def test_outside_hull_nearest_fallback(self):
        known = _known_points_triangle()
        query = np.array([[10.0, 10.0]], dtype=float)
        w = natural_neighbor_weights(known, query, method="linear")
        nearest = int(np.argmin(np.linalg.norm(known - query[0], axis=1)))
        assert w[0, nearest] == pytest.approx(1.0, abs=1e-9)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_reproduces_linear_field(self):
        # Barycentric weights reproduce ANY linear field exactly -- in particular
        # the identity, so w @ parents == query for interior points. This is the
        # core interpolation-correctness property (shared with Sibson).
        known = self._hex(n_radial=3, R=1.0)
        queries = self._interior_queries(seed=1)
        wl = natural_neighbor_weights(known, queries, method="linear")
        assert np.allclose(wl @ known, queries, atol=1e-9)

    def test_invalid_method_raises(self):
        known = _known_points_triangle()
        with pytest.raises(ValueError):
            natural_neighbor_weights(known, known, method="bogus")
