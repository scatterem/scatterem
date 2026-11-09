"""Sibson natural-neighbour interpolation weights.

WHAT IS COMPUTED
----------------
Insert a query point ``q`` into the Voronoi diagram of the sites. Its new cell
is carved out of its neighbours' cells; the area it takes from site ``p``,
divided by the area of the new cell, is ``p``'s Sibson weight. The weights are a
partition of unity, exact at the sites, reproduce linear functions exactly
(``sum_i w_i p_i = q``), are local (only the natural neighbours are non-zero),
and are C1 away from the sites -- what partitioned PRISM wants when it rebuilds
a beam's column from its parents'.

HOW, FROM FIRST PRINCIPLES
--------------------------
The natural neighbours of ``q`` are the vertices of the Delaunay triangles
whose circumcircle strictly contains ``q`` (the cavity). The area taken from
``p`` is ``R_p = V(p) intersected with {x : |x - q| < |x - p|}``: ``p``'s old cell cut
by the bisector of ``p`` and ``q``. Walking its boundary counter-clockwise, it
runs along ``p``'s old Voronoi edges through the circumcentres ``C_T`` of the
cavity triangles at ``p``, entering and leaving at the two points where the
bisector crosses ``p``'s cell -- which are the circumcentres ``g`` of ``q`` with
the two cavity-boundary edges at ``p``.

Take the shoelace sum about ``m = (p + q) / 2``. ``m`` lies on the bisector, so
the closing segment (along the bisector) contributes nothing, and every other
segment starts at one cavity triangle's ``C_T``:

* ``C_T -> C_T'`` across an interior cavity edge (``T'`` the neighbour), or
* ``C_T -> g`` across a cavity-boundary edge, plus ``g -> C_T`` on the way in.

So each (query, cavity triangle, vertex) contributes an independent term and
the whole thing is one vectorised pass -- no polygon is ever built or ordered.

WHY IT STAYS EXACT ON A LATTICE
-------------------------------
Beam coordinates are integer lattice points, which put queries exactly on
Delaunay edges and on circumcircles all the time. A circumcentre ``g(q, a, b)``
diverges when ``q`` is collinear with ``a`` and ``b``; here ``g`` is only ever
formed for a cavity-BOUNDARY edge, and ``q`` cannot lie on one (a point on an
edge segment is strictly inside the circumcircles of both triangles sharing it,
which makes that edge interior). Interior edges use the neighbour's ``C_T'``,
which does not depend on ``q``. No perturbation is needed, and linear precision
holds to ~1e-15 on the lattice. Two cases are handled explicitly:

* a query on a site gets weight 1 there;
* a query on the hull boundary has an unbounded cell, and the limit of Sibson's
  weights there is linear interpolation along that hull edge, which is used.

A query outside the hull has no bounded cell, so no Sibson weights exist
there; it gets weight 1 on the nearest site, as the linear path does.

BACKENDS AND COST
-----------------
Two backends compute the same terms; they agree to ~4e-15.

* NumPy (``device=None``, the default). A k-d tree finds each triangle's
  candidate queries, the terms are one vectorised pass, and only the non-zero
  weights are written into the dense ``(B, B_S)`` output -- which is allocated
  lazily, so the untouched 99.9 % of it costs nothing. 6 ms at 61 parents,
  12 ms at 817 (2361 beams); 82 ms at 3169 parents x 9393 beams. The previous
  per-query polygon implementation took 561-647 ms on the small sets.
* Warp (``device`` a CUDA device). One thread per query tests every triangle --
  58 M circumcircle tests at the largest size, which the GPU does in ~1 ms --
  and also finds the containing triangle, so the exact-node, inside-hull and
  hull-edge tests need no separate host pass. Each thread merges its weights
  per site into a fixed slot and normalises them, so only the ~0.1 % non-zero
  entries cross back to the host. 3.9 / 5.7 / 34 ms for the three sizes above,
  on the first call in a process whose GPU is already initialised.

The Warp path must NOT return the dense block from the GPU: at 9393 x 3169
that copy alone is 104 ms, which is why a first dense-output kernel LOST to
NumPy despite its ~1 ms of compute. It is also not the default, because
starting CUDA just for this costs ~110 ms and a CPU-only machine has no GPU:
pass ``device`` when the caller already works on the GPU. The kernel compiles
once per machine (~0.5 s) and is then loaded from Warp's disk cache.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.spatial import Delaunay, KDTree

from scatterem.physics import _warp_interop as wi
from scatterem.utils.warp.launch import ensure_warp_initialized

# Forward kernels only: nothing here is differentiated.
wp.set_module_options({"enable_backward": False})

__all__ = ["sibson_weights"]


#: Coincidence with a site, or with the hull boundary, as a fraction of the
#: site cloud's diameter.
_TOL = 1e-9


def _cc0(x, y):
    """Circumcentre of ``(0, x, y)`` for ``(..., 2)`` arrays."""
    xx = (x * x).sum(-1)
    yy = (y * y).sum(-1)
    d = 2.0 * (x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0])
    return np.stack(
        [(xx * y[..., 1] - yy * x[..., 1]) / d, (yy * x[..., 0] - xx * y[..., 0]) / d],
        axis=-1,
    )


def _cross(u, v):
    """z-component of ``u x v`` for ``(..., 2)`` arrays."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _triangulate(sites):
    """Delaunay triangles made CCW, with neighbours, circumcentres and radii^2."""
    tri = Delaunay(sites)
    s, nb = tri.simplices.copy(), tri.neighbors.copy()
    a, b, c = sites[s[:, 0]], sites[s[:, 1]], sites[s[:, 2]]
    flip = _cross(b - a, c - a) < 0  # make every triangle CCW
    s[flip, 1], s[flip, 2] = s[flip, 2], s[flip, 1].copy()
    nb[flip, 1], nb[flip, 2] = nb[flip, 2], nb[flip, 1].copy()
    a, b, c = sites[s[:, 0]], sites[s[:, 1]], sites[s[:, 2]]
    cc = a + _cc0(b - a, c - a)
    r2 = ((cc - a) ** 2).sum(-1)
    return tri, s, nb, cc, r2


def _hull_edge_rows(sites, q, hull, scale):
    """Linear weights along the hull edge a query sits on, or `on=False` if none."""
    p0, p1 = sites[hull[:, 0]], sites[hull[:, 1]]
    d = p1 - p0
    t = np.clip(((q[:, None] - p0) * d).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
    diff = q[:, None] - (p0 + t[..., None] * d)
    dist2 = (diff * diff).sum(-1)
    k = dist2.argmin(1)
    ar = np.arange(len(q))
    on = dist2[ar, k] <= (_TOL * scale) ** 2
    return on, k, t[ar, k]


def _sibson_numpy(known_points, interp_points) -> np.ndarray:
    """``(B, B_S)`` Sibson weights of ``interp_points`` over ``known_points``.

    Parameters
    ----------
    known_points : (B_S, 2) array_like
        Site coordinates.
    interp_points : (B, 2) array_like
        Query coordinates.

    Returns
    -------
    numpy.ndarray
        ``(B, B_S)`` float64 weights; every row sums to 1, and a query
        outside the convex hull of the sites has weight 1 on its nearest
        site.
    """
    sites = np.asarray(known_points, dtype=np.float64)
    q = np.asarray(interp_points, dtype=np.float64)
    nq, ns = len(q), len(sites)
    out = np.zeros((nq, ns))
    scale = float(np.ptp(sites, axis=0).max()) or 1.0

    dist, nearest = KDTree(sites).query(q)  # exact nodes
    nearest = np.asarray(nearest)  # 2-D query: arrays, not the scalar overload
    node = dist <= _TOL * scale
    out[np.nonzero(node)[0], nearest[node]] = 1.0

    tri, s, nb, cc, r2 = _triangulate(sites)
    simplex = tri.find_simplex(q)
    inside = (~node) & (simplex >= 0)
    outside = np.nonzero((~node) & (simplex < 0))[0]
    out[outside, nearest[outside]] = 1.0

    # Only a query in a triangle that touches the hull can sit on the hull.
    rim = np.zeros(nq, bool)
    rim[inside] = (nb[simplex[inside]] < 0).any(1)
    on = np.zeros(nq, bool)
    if rim.any():
        ri = np.nonzero(rim)[0]
        on_r, k, t = _hull_edge_rows(sites, q[ri], tri.convex_hull, scale)
        e = ri[on_r]
        on[e] = True
        hull = tri.convex_hull[k[on_r]]
        out[e, hull[:, 0]] += 1.0 - t[on_r]  # Sibson on the hull
        out[e, hull[:, 1]] += t[on_r]  # boundary is linear

    rows = np.nonzero(inside & ~on)[0]
    if rows.size == 0:
        return out
    Q = q[rows]
    lists = KDTree(Q).query_ball_point(
        cc, np.sqrt(r2) * (1 + 1e-12), return_sorted=False
    )
    lens = np.fromiter((len(hits) for hits in lists), np.int64, len(lists))
    ti = np.repeat(np.arange(len(s)), lens)
    qi = np.fromiter((j for hits in lists for j in hits), np.int64, int(lens.sum()))
    keep = ((cc[ti] - Q[qi]) ** 2).sum(-1) < r2[ti]  # the cavity, strictly
    ti, qi = ti[keep], qi[keep]
    qq, C = Q[qi], cc[ti]

    def in_cavity(n):
        ok = n >= 0
        nn = np.where(ok, n, 0)
        return ok & (((cc[nn] - qq) ** 2).sum(-1) < r2[nn])

    # vertex role: p, its CCW neighbours u and v, and the triangles across (p,v), (p,u)
    roles = ((0, 1, 2, 1, 2), (1, 2, 0, 2, 0), (2, 0, 1, 0, 1))
    vals, row_of, col_of = [], [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for jp, ju, jv, jnv, jnu in roles:
            p = sites[s[ti, jp]]
            m = 0.5 * (p + qq)  # on the bisector of p, q
            nv, nu = nb[ti, jnv], nb[ti, jnu]
            in_v, in_u = in_cavity(nv), in_cavity(nu)
            gv = qq + _cc0(p - qq, sites[s[ti, jv]] - qq)
            gu = qq + _cc0(p - qq, sites[s[ti, ju]] - qq)
            Xv = np.where(in_v[:, None], cc[np.where(nv >= 0, nv, 0)], gv)
            term = _cross(C - m, Xv - m)
            term = term + np.where(in_u, 0.0, _cross(gu - m, C - m))
            vals.append(term)
            row_of.append(qi)
            col_of.append(s[ti, jp])
    vals, row_of, col_of = (
        np.concatenate(vals),
        np.concatenate(row_of),
        np.concatenate(col_of),
    )
    # Normalise while still sparse: the dense (nq, ns) block is touched once.
    tot = np.bincount(row_of, weights=vals, minlength=rows.size)
    good = np.isfinite(tot) & (tot > 0)
    sel = good[row_of]
    np.add.at(out, (rows[row_of[sel]], col_of[sel]), vals[sel] / tot[row_of[sel]])
    return out


# --------------------------------------------------------------------- Warp ---
# The same terms as ``_sibson_numpy``, one thread per query. ``flag`` codes:
# 0 interior, 1 on a site, 2 outside the hull, 3 on a hull edge, 4 redo on host.


@wp.func
def _w_cc0(x: wp.vec2d, y: wp.vec2d) -> wp.vec2d:
    xx = wp.dot(x, x)  # pyright: ignore[reportCallIssue, reportArgumentType]
    yy = wp.dot(y, y)  # pyright: ignore[reportCallIssue, reportArgumentType]
    d = wp.float64(2.0) * (x[0] * y[1] - x[1] * y[0])  # pyright: ignore[reportIndexIssue]
    return wp.vec2d((xx * y[1] - yy * x[1]) / d, (yy * x[0] - xx * y[0]) / d)  # pyright: ignore[reportIndexIssue]


@wp.func
def _w_cr(u: wp.vec2d, v: wp.vec2d) -> wp.float64:
    return u[0] * v[1] - u[1] * v[0]  # pyright: ignore[reportIndexIssue]


@wp.func
def _w_incav(
    n: int,
    q: wp.vec2d,
    cc: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    r2: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
) -> int:
    if n < 0:
        return 0
    d = cc[n] - q
    if wp.dot(d, d) < r2[n]:
        return 1
    return 0


@wp.func
def _w_role(
    p: wp.vec2d,
    u: wp.vec2d,
    v: wp.vec2d,
    nv: int,
    nu: int,
    C: wp.vec2d,
    q: wp.vec2d,
    cc: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    r2: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
) -> wp.float64:
    """One (triangle, vertex) shoelace term about ``m = (p + q) / 2``."""
    m = (p + q) * wp.float64(0.5)  # pyright: ignore[reportOperatorIssue]
    Xv = q + _w_cc0(p - q, v - q)  # pyright: ignore[reportOperatorIssue]
    if _w_incav(nv, q, cc, r2) == 1:
        Xv = cc[nv]
    t = _w_cr(C - m, Xv - m)
    if _w_incav(nu, q, cc, r2) == 0:
        t = t + _w_cr(q + _w_cc0(p - q, u - q) - m, C - m)  # pyright: ignore[reportOperatorIssue]
    return t


@wp.func
def _w_push(
    i: int,
    j: int,
    v: wp.float64,
    cols: wp.array2d(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    vals: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    count: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    kmax: int,
) -> int:
    """Add ``v`` to site ``j`` in query ``i``'s slot row; 0 if the row is full."""
    n = count[i]
    for k in range(n):
        if cols[i, k] == j:
            vals[i, k] = vals[i, k] + v
            return 1
    if n < kmax:
        cols[i, n] = wp.int32(j)
        vals[i, n] = v
        count[i] = n + 1
        return 1
    return 0


@wp.func
def _w_contains(
    a: wp.vec2d, b: wp.vec2d, c: wp.vec2d, q: wp.vec2d, eps: wp.float64
) -> int:
    if _w_cr(b - a, q - a) >= -eps and _w_cr(c - b, q - b) >= -eps:  # pyright: ignore[reportOperatorIssue]
        if _w_cr(a - c, q - c) >= -eps:  # pyright: ignore[reportOperatorIssue]
            return 1
    return 0


@wp.func
def _w_finish(
    i: int,
    home: int,
    ok: int,
    q: wp.vec2d,
    sites: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    tris: wp.array(dtype=wp.vec3i),  # pyright: ignore[reportInvalidTypeForm]
    nbr: wp.array(dtype=wp.vec3i),  # pyright: ignore[reportInvalidTypeForm]
    tol2: wp.float64,
    cols: wp.array2d(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    vals: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    count: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
) -> int:
    """Classify query ``i`` from its containing triangle and normalise its row."""
    if home < 0:
        count[i] = 0
        return 2
    tr = tris[home]
    for k in range(3):
        d = sites[tr[k]] - q
        if wp.dot(d, d) <= tol2:
            cols[i, 0] = tr[k]
            vals[i, 0] = wp.float64(1.0)
            count[i] = 1
            return 1
    nb = nbr[home]
    for k in range(3):
        if nb[k] < 0:
            p0 = sites[tr[(k + 1) % 3]]
            e = sites[tr[(k + 2) % 3]] - p0
            f = wp.clamp(
                wp.dot(q - p0, e) / wp.dot(e, e), wp.float64(0.0), wp.float64(1.0)
            )
            r = q - (p0 + e * f)
            if wp.dot(r, r) <= tol2:
                cols[i, 0] = tr[(k + 1) % 3]
                vals[i, 0] = wp.float64(1.0) - f
                cols[i, 1] = tr[(k + 2) % 3]
                vals[i, 1] = f
                count[i] = 2
                return 3
    tot = wp.float64(0.0)
    for k in range(count[i]):
        tot = tot + vals[i, k]
    if ok == 0 or not (tot > wp.float64(0.0)) or not wp.isfinite(tot):
        return 4
    for k in range(count[i]):
        vals[i, k] = vals[i, k] / tot
    return 0


@wp.kernel
def _w_sib(
    sites: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    tris: wp.array(dtype=wp.vec3i),  # pyright: ignore[reportInvalidTypeForm]
    nbr: wp.array(dtype=wp.vec3i),  # pyright: ignore[reportInvalidTypeForm]
    cc: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    r2: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    q: wp.array(dtype=wp.vec2d),  # pyright: ignore[reportInvalidTypeForm]
    tol2: wp.float64,
    eps: wp.float64,
    kmax: int,
    cols: wp.array2d(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    vals: wp.array2d(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    count: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
    flag: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
):
    i = wp.tid()
    qi = q[i]
    count[i] = 0
    home = wp.int32(-1)
    ok = wp.int32(1)
    for t in range(tris.shape[0]):
        tr = tris[t]
        a = sites[tr[0]]
        b = sites[tr[1]]
        c = sites[tr[2]]
        if home < 0:
            if _w_contains(a, b, c, qi, eps) == 1:
                home = t
        d = cc[t] - qi
        if wp.dot(d, d) < r2[t]:
            nb = nbr[t]
            C = cc[t]
            va = _w_role(a, b, c, nb[1], nb[2], C, qi, cc, r2)
            vb = _w_role(b, c, a, nb[2], nb[0], C, qi, cc, r2)
            vc = _w_role(c, a, b, nb[0], nb[1], C, qi, cc, r2)
            ok = ok * _w_push(i, tr[0], va, cols, vals, count, kmax)  # pyright: ignore[reportArgumentType]
            ok = ok * _w_push(i, tr[1], vb, cols, vals, count, kmax)  # pyright: ignore[reportOperatorIssue, reportArgumentType]
            ok = ok * _w_push(i, tr[2], vc, cols, vals, count, kmax)  # pyright: ignore[reportArgumentType]
    flag[i] = _w_finish(i, home, ok, qi, sites, tris, nbr, tol2, cols, vals, count)  # pyright: ignore[reportArgumentType]


def _sibson_warp(sites, q, device, kmax: int = 64) -> np.ndarray:
    """The Warp backend: sparse output, dense block assembled on the host."""
    nq, ns = len(q), len(sites)
    scale = float(np.ptp(sites, axis=0).max()) or 1.0
    _, s, nb, cc, r2 = _triangulate(sites)
    cols = wp.empty((nq, kmax), dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
    vals = wp.empty((nq, kmax), dtype=wp.float64, device=device)  # pyright: ignore[reportArgumentType]
    count = wp.empty(nq, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
    flag = wp.empty(nq, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
    wp.launch(
        _w_sib,
        dim=nq,
        device=device,
        inputs=[
            wp.array(sites, dtype=wp.vec2d, device=device),
            wp.array(s.astype(np.int32), dtype=wp.vec3i, device=device),
            wp.array(nb.astype(np.int32), dtype=wp.vec3i, device=device),
            wp.array(cc, dtype=wp.vec2d, device=device),
            wp.array(r2, dtype=wp.float64, device=device),
            wp.array(q, dtype=wp.vec2d, device=device),
            (_TOL * scale) ** 2,
            1e-12 * scale * scale,
            int(kmax),
            cols,
            vals,
            count,
            flag,
        ],
    )
    c, v, n, f = cols.numpy(), vals.numpy(), count.numpy(), flag.numpy()
    out = np.zeros((nq, ns))
    live = np.arange(kmax)[None, :] < n[:, None]
    out[np.nonzero(live)[0], c[live].astype(np.intp)] = v[live]
    # More than `kmax` natural neighbours, or a degenerate total: redo on the host.
    redo = np.nonzero(f == 4)[0]
    if redo.size:
        out[redo] = _sibson_numpy(sites, q[redo])
    outside = np.nonzero(f == 2)[0]
    if outside.size:
        out[outside, KDTree(sites).query(q[outside])[1]] = 1.0
    return out


def sibson_weights(known_points, interp_points, device=None) -> np.ndarray:
    """``(B, B_S)`` Sibson weights of ``interp_points`` over ``known_points``.

    Parameters
    ----------
    known_points : (B_S, 2) array_like
        Site coordinates.
    interp_points : (B, 2) array_like
        Query coordinates.
    device : str or torch.device, optional
        A CUDA device runs the Warp kernel there; ``None`` or a CPU device (the
        default) runs NumPy. Pass it when the caller is already on the GPU --
        starting CUDA just for this costs more than the kernel saves.

    Returns
    -------
    numpy.ndarray
        ``(B, B_S)`` float64 weights, identical (to ~4e-15) on both backends.
        Every row sums to 1; a query outside the convex hull of the sites has
        weight 1 on its nearest site.
    """
    sites = np.asarray(known_points, dtype=np.float64)
    q = np.asarray(interp_points, dtype=np.float64)
    if device is None:
        return _sibson_numpy(sites, q)
    ensure_warp_initialized()  # wp.device_from_torch does not initialise Warp
    # ``str()`` first: warp resolves the string "cuda" to the current device,
    # where an index-less ``torch.device("cuda")`` object raises.
    wdev = wi.device_from_torch(str(device))
    if wdev.is_cuda:
        return _sibson_warp(sites, q, wdev)
    return _sibson_numpy(sites, q)
