"""Natural-neighbour weights for reconstructing an aperture from parent beams.

Both schemes are written from their mathematical definitions.

Two schemes, one contract
-------------------------
Both return a row-stochastic matrix ``W`` of shape ``(n_interp, n_known)``
satisfying, for every query row:

* **partition of unity** -- the row sums to 1, so a constant field is
  reproduced exactly and the interpolation cannot change the total intensity;
* **exact at the parents** -- a query coinciding with a known point puts all
  its weight on that point;
* **nearest fallback** -- a query outside the convex hull of the known points
  takes the value of the nearest one, rather than extrapolating.

``method="linear"`` (default) is piecewise-linear (C0) barycentric
interpolation on the Delaunay triangulation. ``method="sibson"`` is Sibson's
natural-neighbour coordinate (C1 away from the parents), defined as the area
each existing Voronoi cell surrenders to the query point's cell when the query
is inserted into the diagram:

    w_i(q) = area(V_q & V_i) / area(V_q),

where ``V_i`` is the Voronoi cell of parent ``i`` in the original diagram and
``V_q`` is the cell the query would acquire. Sibson (1981), "A brief
description of natural neighbour interpolation", in *Interpreting Multivariate
Data*. It is computed here exactly, by clipping convex polygons against
perpendicular bisectors, rather than by sampling.

Sibson is the more faithful scheme and is markedly slower: it solves a small
polygon-clipping problem per query, where the linear scheme is a single
vectorised Delaunay lookup. The linear scheme is the default because it is what
the large sweeps use.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

__all__ = ["natural_neighbor_weights"]

# Candidate parents considered per query in the Sibson path. A Voronoi
# neighbour is necessarily among the nearest few; 32 is far beyond the ~6
# neighbours a locally hexagonal parent set produces, and the result is
# insensitive to it (checked by raising it).
_SIBSON_CANDIDATES = 32


def natural_neighbor_weights(
    known_points,
    interp_points,
    minimum_weight_cutoff: float = 1e-2,
    method: str = "linear",
) -> np.ndarray:
    """Weights reconstructing values at ``interp_points`` from ``known_points``.

    Parameters
    ----------
    known_points : (M, 2) array_like
        Coordinates of the parents.
    interp_points : (N, 2) array_like
        Coordinates at which the field is wanted.
    minimum_weight_cutoff : float, optional
        Weights below this are dropped and the row renormalised, which keeps
        the reconstruction sparse without breaking the partition of unity.
        Pass 0 to disable.
    method : {"linear", "sibson"}, optional
        See the module docstring.

    Returns
    -------
    (N, M) ndarray
        Row-stochastic weight matrix.
    """
    known = np.asarray(known_points, dtype=np.float64)
    interp = np.asarray(interp_points, dtype=np.float64)
    if known.ndim != 2 or known.shape[1] != 2:
        raise ValueError(f"known_points must be (M, 2), got {known.shape}")
    if interp.ndim != 2 or interp.shape[1] != 2:
        raise ValueError(f"interp_points must be (N, 2), got {interp.shape}")
    if len(known) == 0:
        raise ValueError("need at least one known point")

    if method == "linear":
        weights = _barycentric_weights(known, interp)
    elif method == "sibson":
        weights = _sibson_weights(known, interp)
    else:
        raise ValueError(f"method must be 'linear' or 'sibson', got {method!r}")

    if minimum_weight_cutoff:
        weights[weights < minimum_weight_cutoff] = 0.0
    return _normalise_rows(weights, known, interp)


# --------------------------------------------------------------------------
# shared


def _normalise_rows(weights: np.ndarray, known: np.ndarray, interp: np.ndarray) -> np.ndarray:
    """Force each row to sum to 1, falling back to the nearest parent."""
    total = weights.sum(axis=1)
    good = total > 0
    weights[good] /= total[good, None]
    if not good.all():
        # A row can empty out if the cutoff removed everything. Falling back to
        # the nearest parent keeps the partition of unity, which matters more
        # than the lost smoothness for the handful of rows involved.
        _, nearest = cKDTree(known).query(interp[~good])
        rows = np.nonzero(~good)[0]
        weights[rows] = 0.0
        weights[rows, np.atleast_1d(nearest)] = 1.0
    return weights


def _nearest_fallback(weights: np.ndarray, known: np.ndarray, interp: np.ndarray,
                      rows: np.ndarray) -> None:
    """Put unit weight on the nearest parent for the given rows, in place."""
    if rows.size == 0:
        return
    _, nearest = cKDTree(known).query(interp[rows])
    weights[rows, np.atleast_1d(nearest)] = 1.0


# --------------------------------------------------------------------------
# linear: barycentric coordinates on the Delaunay triangulation


def _barycentric_weights(known: np.ndarray, interp: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(interp), len(known)), dtype=np.float64)
    if len(known) < 3:
        _nearest_fallback(weights, known, interp, np.arange(len(interp)))
        return weights
    try:
        tri = Delaunay(known)
    except Exception:
        # Degenerate parent sets (all collinear, or duplicates) have no
        # triangulation; nearest-neighbour is the only sensible reading.
        _nearest_fallback(weights, known, interp, np.arange(len(interp)))
        return weights

    simplex = tri.find_simplex(interp)
    inside = simplex >= 0

    if inside.any():
        s = simplex[inside]
        # tri.transform[s, :2] maps a point into the simplex's barycentric
        # frame relative to its last vertex, tri.transform[s, 2].
        offset = interp[inside] - tri.transform[s, 2]
        first_two = np.einsum("kij,kj->ki", tri.transform[s, :2], offset)
        bary = np.column_stack([first_two, 1.0 - first_two.sum(axis=1)])
        # Round-off can make a coordinate slightly negative on an edge.
        np.clip(bary, 0.0, None, out=bary)
        vertices = tri.simplices[s]
        rows = np.repeat(np.nonzero(inside)[0], 3)
        np.add.at(weights, (rows, vertices.ravel()), bary.ravel())

    _nearest_fallback(weights, known, interp, np.nonzero(~inside)[0])
    return weights


# --------------------------------------------------------------------------
# sibson: exact Voronoi area stealing


def _sibson_weights(known: np.ndarray, interp: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(interp), len(known)), dtype=np.float64)
    if len(known) < 3:
        _nearest_fallback(weights, known, interp, np.arange(len(interp)))
        return weights

    tree = cKDTree(known)
    k = min(_SIBSON_CANDIDATES, len(known))
    # A box comfortably larger than the parent set bounds every Voronoi cell
    # that a query inside the hull can acquire, so clipping to it is exact.
    lo, hi = known.min(axis=0), known.max(axis=0)
    pad = 10.0 * max(np.max(hi - lo), 1.0)
    box = np.array([[lo[0] - pad, lo[1] - pad], [hi[0] + pad, lo[1] - pad],
                    [hi[0] + pad, hi[1] + pad], [lo[0] - pad, hi[1] + pad]])

    try:
        hull = Delaunay(known)
    except Exception:
        _nearest_fallback(weights, known, interp, np.arange(len(interp)))
        return weights
    outside = hull.find_simplex(interp) < 0

    fallback_rows = []
    for row, q in enumerate(interp):
        if outside[row]:
            fallback_rows.append(row)
            continue
        dist, idx = tree.query(q, k=k)
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        if dist[0] == 0.0:                      # query sits on a parent
            weights[row, idx[0]] = 1.0
            continue
        neigh = known[idx]

        # V_q: everything closer to q than to any candidate parent.
        cell = box
        for p in neigh:
            cell = _clip(cell, 2.0 * (p - q), float(p @ p - q @ q))
            if len(cell) < 3:
                break
        if len(cell) < 3:
            fallback_rows.append(row)
            continue

        # Split V_q by which parent owned each piece in the original diagram.
        stolen = np.zeros(len(idx))
        for i, p_i in enumerate(neigh):
            piece = cell
            for j, p_j in enumerate(neigh):
                if i == j:
                    continue
                piece = _clip(piece, 2.0 * (p_j - p_i), float(p_j @ p_j - p_i @ p_i))
                if len(piece) < 3:
                    break
            if len(piece) >= 3:
                stolen[i] = _area(piece)
        if stolen.sum() <= 0:
            fallback_rows.append(row)
            continue
        weights[row, idx] = stolen

    _nearest_fallback(weights, known, interp, np.asarray(fallback_rows, dtype=int))
    return weights


def _clip(poly: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Clip a convex polygon to the half-plane ``a . x <= b``.

    Sutherland--Hodgman: walk the edges, keeping inside vertices and inserting
    the crossing point wherever an edge changes side.
    """
    if len(poly) == 0:
        return poly
    side = poly @ a - b
    keep = side <= 0
    if keep.all():
        return poly
    if not keep.any():
        return poly[:0]
    out = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        if keep[i]:
            out.append(poly[i])
        if keep[i] != keep[j]:
            t = side[i] / (side[i] - side[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.asarray(out)


def _area(poly: np.ndarray) -> float:
    """Unsigned area of a simple polygon, by the shoelace formula."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
