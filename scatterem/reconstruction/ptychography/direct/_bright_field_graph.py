"""Host-side geometry for the BF shift estimators.

Canonical order of the images, detector pitch, binning, the neighbour graph,
the least-squares fit of relative shifts and the interpolation of bin shifts
to pixels. All work here is small and runs in NumPy float64.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bin_members",
    "canonical_order",
    "components",
    "interpolate_shifts",
    "min_detector_spacing",
    "neighbour_edges",
    "solve_relative",
]

_ROWS_PER_PASS = 512


def canonical_order(q: np.ndarray) -> np.ndarray:
    """Sort the images by ``(q0, q1)``; return the input index of each rank."""
    return np.lexsort((q[:, 1], q[:, 0]))


def min_detector_spacing(q: np.ndarray) -> float:
    """Smallest distance between two distinct coordinates; 1.0 for ``N < 2``."""
    n = q.shape[0]
    if n < 2:
        return 1.0
    best = np.inf
    for start in range(0, n, _ROWS_PER_PASS):
        rows = q[start : start + _ROWS_PER_PASS]
        d2 = ((rows[:, None, :] - q[None, :, :]) ** 2).sum(-1)
        d2[np.arange(rows.shape[0]), start + np.arange(rows.shape[0])] = np.inf
        best = min(best, float(d2.min()))
    return float(np.sqrt(best))


def bin_members(q: np.ndarray, binning: int, pitch: float):
    """Group canonically ordered coordinates into detector bins.

    Bin index ``m = floor(q / (binning · p) + ½)`` per axis: bins are centred
    on multiples of ``binning · p``, one of them on the disc centre.

    Returns
    -------
    keys : numpy.ndarray
        int ``(n_bins, 2)`` bin indices, sorted lexicographically.
    member_bin : numpy.ndarray
        int ``(N,)``: the bin of each (canonical) image.
    coords : numpy.ndarray
        float64 ``(n_bins, 2)``: mean member coordinate per bin.
    """
    m = np.floor(q / (binning * pitch) + 0.5).astype(np.int64)
    keys, member_bin = np.unique(m, axis=0, return_inverse=True)
    member_bin = member_bin.reshape(-1)
    coords = np.zeros((keys.shape[0], 2))
    counts = np.zeros(keys.shape[0])
    for i in range(q.shape[0]):
        coords[member_bin[i]] += q[i]
        counts[member_bin[i]] += 1
    return keys, member_bin, coords / counts[:, None]


def neighbour_edges(
    node_pos: np.ndarray, pitch: float, bin_ids: np.ndarray | None
) -> np.ndarray:
    """Edges ``(i, j)``, ``i < j``, of the detector-neighbour graph.

    Images are neighbours when ``|q_i − q_j| <= 1.5 p`` (the 8-neighbourhood
    of a square grid). Bins are neighbours when their indices differ by at
    most one per axis.

    Returns
    -------
    numpy.ndarray
        int64 ``(E, 2)`` sorted lexicographically.
    """
    n = node_pos.shape[0]
    pairs = []
    for start in range(0, n, _ROWS_PER_PASS):
        i = np.arange(start, min(n, start + _ROWS_PER_PASS))
        if bin_ids is None:
            d2 = ((node_pos[i, None, :] - node_pos[None, :, :]) ** 2).sum(-1)
            near = d2 <= (1.5 * pitch) ** 2 * (1.0 + 1e-9)
        else:
            near = (np.abs(bin_ids[i, None, :] - bin_ids[None, :, :]) <= 1).all(-1)
        near &= np.arange(n)[None, :] > i[:, None]
        ii, jj = np.nonzero(near)
        pairs.append(np.stack([i[ii], jj], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs).astype(np.int64)


def components(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label of each node, numbered by first node."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(a) for a in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(-1)


def solve_relative(n: int, edges: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, int]:
    """Least-squares shifts from relative measurements ``d_ij ≈ s_j − s_i``.

    Minimises ``Σ (s_j − s_i − d_ij)²`` with unit weights. The normal
    equations ``L s = b`` (graph Laplacian ``L``) are solved per axis by
    conjugate gradients in float64 with a Jacobi preconditioner. The
    matrix-vector product gathers over a padded neighbour table (no scatter).
    The iteration stops at a relative residual of 1e-12 or after ``10 N``
    steps. The mean of each connected component is then removed; an
    isolated node gets 0.

    Returns
    -------
    shifts : numpy.ndarray
        float64 ``(n, 2)``.
    n_components : int
    """
    labels = components(n, edges)
    n_comp = int(labels.max()) + 1 if n else 0
    nbr: list[list[int]] = [[] for _ in range(n)]
    sgn: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for e, (i, j) in enumerate(edges):
        nbr[i].append(j)
        nbr[j].append(i)
        sgn[j].append((e, 1.0))
        sgn[i].append((e, -1.0))
    width = max([len(v) for v in nbr] + [1])
    table = np.full((n, width), n, dtype=np.int64)
    etab = np.full((n, width), len(edges), dtype=np.int64)
    stab = np.zeros((n, width))
    for k in range(n):
        table[k, : len(nbr[k])] = nbr[k]
        for c, (e, s) in enumerate(sgn[k]):
            etab[k, c], stab[k, c] = e, s
    deg = np.array([len(v) for v in nbr], dtype=np.float64)
    d_pad = np.concatenate([d, np.zeros((1, 2))]) if n else d
    b = (stab[..., None] * d_pad[etab]).sum(axis=1)
    precond = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)

    def lap(x):
        xp = np.concatenate([x, [0.0]])
        return deg * x - xp[table].sum(axis=1)

    out = np.zeros((n, 2))
    for axis in range(2):
        rhs = b[:, axis]
        norm_b = float(np.sqrt(rhs @ rhs))
        if norm_b == 0.0:
            continue
        x = np.zeros(n)
        r = rhs.copy()
        z = precond * r
        p = z.copy()
        rho = float(r @ z)
        for _ in range(10 * n):
            ap = lap(p)
            pap = float(p @ ap)
            if not pap > 0:
                break
            alpha = rho / pap
            x += alpha * p
            r -= alpha * ap
            if float(np.sqrt(r @ r)) <= 1e-12 * norm_b:
                break
            z = precond * r
            rho_next = float(r @ z)
            p = z + (rho_next / rho) * p
            rho = rho_next
        out[:, axis] = x
    for c in range(n_comp):
        sel = labels == c
        out[sel] -= out[sel].mean(axis=0)
    return out, n_comp


def interpolate_shifts(
    q: np.ndarray, bin_coords: np.ndarray, bin_shifts: np.ndarray, width: float
) -> np.ndarray:
    """Shift of every BF pixel by a moving least-squares fit over the bins.

    Weights ``exp(−|Q_b − q|² / (2 w²))`` with ``w = binning · p``. Basis
    in ``Δq / w``: quadratic ``[1, Δ0, Δ1, Δ0², Δ0Δ1, Δ1²]``; the value at
    ``Δq = 0``. Fallbacks, in order: the affine basis ``[1, Δ0, Δ1]``, then
    the weighted mean. A basis is used only if at least as many bins as basis
    functions have a weight above 1e-6 and the weighted system has a
    condition number of at most 1e8. So every pixel gets a finite shift.

    Reason for the quadratic basis: the shift field is the gradient of the
    aberration function; defocus and astigmatism make it linear, coma
    quadratic. A quadratic fit reproduces both exactly. A design check with a
    coma-like term on a disc of radius 4 px and binning 3 gave a worst error
    of 0.13 px with the affine basis and 0.012 px with the quadratic one.
    """
    out = np.zeros((q.shape[0], 2))
    for start in range(0, q.shape[0], _ROWS_PER_PASS):
        qq = q[start : start + _ROWS_PER_PASS]
        delta = (bin_coords[None, :, :] - qq[:, None, :]) / width
        r2 = (delta**2).sum(-1)
        w = np.exp(-0.5 * r2)
        wsum = w.sum(1)
        nearest = np.argmin(r2, axis=1)
        res = np.where(
            wsum[:, None] > 0,
            (w[..., None] * bin_shifts[None]).sum(1)
            / np.where(wsum > 0, wsum, 1.0)[:, None],
            bin_shifts[nearest],
        )
        d0, d1 = delta[..., 0], delta[..., 1]
        one = np.ones_like(d0)
        bases = (
            np.stack([one, d0, d1, d0 * d0, d0 * d1, d1 * d1], axis=-1),
            np.stack([one, d0, d1], axis=-1),
        )
        done = np.zeros(len(qq), dtype=bool)
        n_heavy = (w > 1e-6).sum(1)
        for basis in bases:
            todo = np.flatnonzero(~done & (n_heavy >= basis.shape[-1]))
            if todo.size == 0:
                continue
            bw = basis[todo] * w[todo, :, None]
            a = np.einsum("pbi,pbj->pij", bw, basis[todo])
            good = np.linalg.cond(a) <= 1e8
            sel = todo[good]
            if sel.size:
                rhs = np.einsum("pbi,bk->pik", bw[good], bin_shifts)
                res[sel] = np.linalg.solve(a[good], rhs)[:, 0, :]
                done[sel] = True
        out[start : start + len(qq)] = res
    return out
