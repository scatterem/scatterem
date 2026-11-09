"""Integer-lattice searches of the cell reduction (private).

Helpers of :mod:`scatterem.physics._orthobox`: the integer adjugate, the
diagonal of the row Hermite normal form, Lagrange-Gauss reduction of 2-D
bases, the enumeration of 2-D sublattices by index (Hart and Forcade 2008),
and the search for the beam edge.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "adjugate",
    "beam_search",
    "hnf_diagonal",
    "order_plane_pair",
    "plane_search",
]

#: Relative width of a tie in the length comparisons of the searches.
_TIE = 1e-12


def adjugate(p: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer adjugate and determinant of a 3 x 3 integer matrix."""
    m = [[int(x) for x in row] for row in p]
    adj = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            r = [x for x in range(3) if x != j]
            c = [x for x in range(3) if x != i]
            minor = m[r[0]][c[0]] * m[r[1]][c[1]] - m[r[0]][c[1]] * m[r[1]][c[0]]
            adj[i, j] = (-1) ** (i + j) * minor
    det = sum(m[0][k] * int(adj[k, 0]) for k in range(3))
    return adj, det


def hnf_diagonal(p: np.ndarray) -> list[int]:
    """Diagonal of the lower-triangular row Hermite normal form of `p`."""
    h = [[int(x) for x in row] for row in p]
    for j in (2, 1, 0):
        rows = list(range(j + 1))
        while True:
            nonzero = [r for r in rows if h[r][j] != 0]
            if len(nonzero) <= 1:
                break
            pivot = min(nonzero, key=lambda r: abs(h[r][j]))
            for r in nonzero:
                if r != pivot:
                    q = h[r][j] // h[pivot][j]
                    h[r] = [a - q * b for a, b in zip(h[r], h[pivot], strict=True)]
        pivot = next(r for r in rows if h[r][j] != 0)
        h[pivot], h[j] = h[j], h[pivot]
        if h[j][j] < 0:
            h[j] = [-a for a in h[j]]
    return [h[0][0], h[1][1], h[2][2]]


def quad(c: np.ndarray, gram: np.ndarray, d: np.ndarray | None = None):
    d = c if d is None else d
    return np.einsum("bi,ij,bj->b", c.astype(np.float64), gram, d.astype(np.float64))


def gauss(c0: np.ndarray, c1: np.ndarray, gram: np.ndarray):
    """Lagrange-Gauss reduction of integer coefficient pairs (batch).

    The coefficients stay exact integers; only the Gram matrix is float.
    """
    c0, c1 = c0.copy(), c1.copy()
    for _ in range(500):
        swap = quad(c1, gram) < quad(c0, gram)
        if swap.any():
            tmp = c0[swap].copy()
            c0[swap] = c1[swap]
            c1[swap] = tmp
        mu = np.rint(quad(c0, gram, c1) / quad(c0, gram)).astype(np.int64)
        if not mu.any():
            break
        c1 = c1 - mu[:, None] * c0
    return c0, c1


def hnf_2d(d_lo: int, d_hi: int) -> np.ndarray:
    """Rows (d, a, b, c) of the 2-D lower HNFs [[a, 0], [b, c]], a c = d."""
    out = [
        (d, a, b, d // a)
        for d in range(d_lo, d_hi)
        for a in range(1, d + 1)
        if d % a == 0
        for b in range(a)
    ]
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def plane_search(gram: np.ndarray, sine_cap: float, d_max: int):
    """Smallest-index sublattice of rows 0, 1 with a rectangular basis.

    Returns the index and the Gauss-reduced pair, or None up to `d_max`.
    """
    d_lo = 1
    while d_lo <= d_max:
        # Doubling batches while small (the usual case), then batches of about
        # 2e4 sublattices (the mean of sigma(d) is about 1.6 d).
        width = max(1, min(d_lo, 12000 // d_lo))
        d_hi = min(d_max + 1, d_lo + width)
        h = hnf_2d(d_lo, d_hi)
        c0 = np.stack([h[:, 1], np.zeros_like(h[:, 1])], axis=1)
        c1 = h[:, 2:4]
        r0, r1 = gauss(c0, c1, gram)
        n0, n1 = quad(r0, gram), quad(r1, gram)
        cos = np.abs(quad(r0, gram, r1)) / np.sqrt(n0 * n1)
        ok = np.flatnonzero(cos <= sine_cap)
        if ok.size:
            d = int(h[ok[0], 0])
            ok = ok[h[ok, 0] == d]
            best = ok[cos[ok] <= cos[ok].min() + _TIE]
            size = np.sqrt(n0[best]) + np.sqrt(n1[best])
            best = best[size <= size.min() * (1.0 + _TIE)]
            i = int(best[0])
            return d, r0[i], r1[i]
        d_lo = d_hi
    return None


def order_plane_pair(r0, r1, gram):
    """Choose b0 (closest in direction to a0) and b1 with (b0 x b1).n > 0."""
    cands = [r0, -r0, r1, -r1]
    lengths = [math.sqrt(float(quad(c[None], gram)[0])) for c in cands]
    cosines = [
        float(c @ gram[:, 0]) / (lengths[k] * math.sqrt(gram[0, 0]))
        for k, c in enumerate(cands)
    ]
    top = max(cosines)
    k = min(
        (k for k in range(4) if cosines[k] >= top - _TIE),
        key=lambda k: (lengths[k], k),
    )
    b0 = cands[k]
    b1 = r1 if k < 2 else r0
    if b0[0] * b1[1] - b0[1] * b1[0] < 0:
        b1 = -b1
    return b0, b1


def beam_search(A: np.ndarray, n: np.ndarray, sine_cap: float, k_max: int):
    """Smallest k >= 1 with a beam edge s k a2 + m0 a0 + m1 a1 within `sine_cap`.

    Returns ``(k, m0, m1, s)`` or None up to `k_max`.
    """
    h = float(A[2] @ n)
    s = 1 if h > 0 else -1
    t = A[2] - h * n
    u0 = A[0] / np.linalg.norm(A[0])
    u1 = np.cross(n, u0)
    plane = A[:2] @ np.stack([u0, u1]).T
    t2 = np.array([t @ u0, t @ u1])
    gram = A[:2] @ A[:2].T
    r0, r1 = gauss(np.array([[1, 0]]), np.array([[0, 1]]), gram)
    unimodular = np.stack([r0[0], r1[0]])
    reduced = unimodular.astype(np.float64) @ plane
    k = np.arange(1, k_max + 1, dtype=np.int64)
    target = -s * k[:, None] * t2
    base = np.rint(np.linalg.solve(reduced.T, target.T).T)
    best_rho = np.full(k.shape, np.inf)
    best_c = np.zeros((k.size, 2))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            c = base + np.array([di, dj])
            rho = np.linalg.norm(c @ reduced - target, axis=1)
            better = rho < best_rho
            best_rho[better] = rho[better]
            best_c[better] = c[better]
    lean = best_rho / np.sqrt(best_rho**2 + (k * h) ** 2)
    hits = np.flatnonzero(lean <= sine_cap)
    if hits.size == 0:
        return None
    i = int(hits[0])
    m = best_c[i].astype(np.int64) @ unimodular
    return int(k[i]), int(m[0]), int(m[1]), s
