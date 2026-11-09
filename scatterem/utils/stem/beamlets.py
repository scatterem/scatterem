"""PRISM beamlet sampling and the natural-neighbour weights it uses."""

import logging

import numpy as np
from numpy.fft import fftfreq
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


def beamlet_samples(
    beam_mask: ArrayLike,
    radius: float,
    n_angular_samples: int = 6,
    n_radial_samples: int = 3,
) -> np.ndarray:
    """Select PRISM beam coordinates closest to a polar sampling pattern.

    Builds a set of target points on concentric rings (plus the origin) and,
    for each target, picks the nearest actual beam coordinate present in
    `beam_mask`.

    Parameters
    ----------
    beam_mask : array_like
        Boolean (or 0/1) FFT-shifted mask of shape ``(M, M)`` selecting which
        Fourier-space beams are available.
    radius : float
        Outer radius, in pixel units, of the polar sampling pattern.
    n_angular_samples : int, optional
        Number of angular samples on the innermost ring; later rings use a
        proportionally larger count. Default is 6.
    n_radial_samples : int, optional
        Number of radial rings (including the origin). Default is 3.

    Returns
    -------
    numpy.ndarray
        Integer ``(y, x)`` beam coordinates closest to each polar sample
        point, with shape ``(n_samples, 2)``.
    """
    A = np.fft.fftshift(beam_mask)
    M = A.shape[0]

    my1 = np.tile(fftfreq(M, d=1 / M)[:, None], M).astype(np.int32)
    mx1 = np.repeat(fftfreq(M, d=1 / M)[:, None].T, M, axis=0).astype(np.int32)

    B = np.nonzero(A)
    beam_coords1 = np.array([my1[B], mx1[B]]).T
    beam_coords2 = beam_coords1 + 1e-2

    a_offset = np.pi / n_angular_samples
    my1 = np.tile(fftfreq(M, d=1 / M)[:, None], M).astype(np.int32)
    mx1 = np.repeat(fftfreq(M, d=1 / M)[:, None].T, M, axis=0).astype(np.int32)
    beam_coords1 = np.array([my1[B], mx1[B]]).T
    radial_samples = np.linspace(0, radius, n_radial_samples, endpoint=True)[1:]
    samples = [[0, 0]]
    for i, r in enumerate(radial_samples):
        angular_samples = np.linspace(
            -np.pi, np.pi, n_angular_samples * (1 + i), endpoint=False
        )
        # print(i, len(angular_samples))
        for a in angular_samples:
            # print(r, a)
            # print([r * np.sin(a + a_offset * i), r * np.cos(a + a_offset * i)])
            samples.append([r * np.sin(a + a_offset * i), r * np.cos(a + a_offset * i)])
    samples = np.array(samples)
    xy_diff = np.linalg.norm(samples[None, ...] - beam_coords2[:, None, :], axis=2)
    r_min_indices = np.argmin(xy_diff, axis=0)
    best_beam_coords = beam_coords1[r_min_indices]
    return best_beam_coords


def _linear_barycentric_weights(
    known_points, interp_points, minimum_weight_cutoff=1e-2
):
    r"""Vectorised piecewise-linear (Delaunay barycentric) interpolation weights.

    A fast drop-in alternative to the Sibson natural-neighbour weights: each query
    point's weights are the barycentric coordinates of the Delaunay triangle it
    falls in (3 parents), and queries outside the convex hull fall back to the
    nearest parent (weight 1). Like Sibson it is a partition of unity and exact at
    the known points, but it is **fully vectorised** -- one batched
    ``Delaunay.find_simplex`` + an einsum, no per-point Python loop -- so it is
    orders of magnitude faster (seconds -> milliseconds for thousands of points).
    Piecewise-linear (C0) rather than Sibson's C1, which is adequate for the smooth
    probe-reconstruction use in partitioned PRISM.
    """
    known = np.asarray(known_points, dtype=float)
    interp = np.asarray(interp_points, dtype=float)
    B, Bs = interp.shape[0], known.shape[0]
    weights = np.zeros((B, Bs))

    def _nearest_all():
        d = np.linalg.norm(known[None, :, :] - interp[:, None, :], axis=2)
        weights[np.arange(B), d.argmin(axis=1)] = 1.0
        return weights

    if Bs < 3:
        return _nearest_all()
    try:
        tri = Delaunay(known)
    except Exception:
        return _nearest_all()

    simplex = tri.find_simplex(interp)  # (B,), -1 outside the convex hull
    inside = simplex >= 0
    if inside.any():
        sx = simplex[inside]
        transform = tri.transform[sx]  # (Bin, 3, 2): [:, :2] affine, [:, 2] offset
        r = interp[inside] - transform[:, 2, :]
        bary2 = np.einsum("bij,bj->bi", transform[:, :2, :], r)  # (Bin, 2)
        bary = np.concatenate([bary2, 1.0 - bary2.sum(1, keepdims=True)], axis=1)
        verts = tri.simplices[sx]  # (Bin, 3) parent indices
        rows = np.repeat(np.nonzero(inside)[0], 3)
        weights[rows, verts.reshape(-1)] = bary.reshape(-1)
    out = np.nonzero(~inside)[0]
    if out.size:
        d = np.linalg.norm(known[None, :, :] - interp[out][:, None, :], axis=2)
        weights[out, d.argmin(axis=1)] = 1.0

    # Drop tiny / slightly-negative weights and renormalise so each row sums to 1
    # (matches the Sibson path's cutoff + renormalisation contract).
    weights[weights < minimum_weight_cutoff] = 0.0
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return weights / row_sum


def natural_neighbor_weights(
    known_points,
    interp_points,
    minimum_weight_cutoff=1e-2,
    method="linear",
    device=None,
):
    r"""Interpolation weights of ``interp_points`` on the sites ``known_points``.

    Every row is a partition of unity over the sites, exact at a site, and a
    point outside the sites' convex hull takes weight 1 on its nearest site.

    Parameters
    ----------
    known_points : (B_S, 2) ndarray
        X, y-coordinates of observations.
    interp_points : (B, 2) ndarray
        Coordinates of the grid points at which to calculate the
        interpolation.
    minimum_weight_cutoff : float, optional
        Delete weights smaller than this cutoff. Default is 1e-2.  Not applied
        by ``"sibson"``, which keeps every weight: dropping the small ones
        perturbs every other weight in the row and buys nothing.
    method : {"linear", "sibson"}, optional
        ``"linear"`` (default) is the vectorised Delaunay-barycentric interpolation
        (:func:`_linear_barycentric_weights`): orders of magnitude faster than the
        Sibson loop, with the same partition-of-unity / exact-at-parents /
        nearest-fallback contract (piecewise-linear C0 rather than Sibson's C1).
        ``"sibson"`` is natural-neighbour (area-stealing) interpolation,
        :mod:`scatterem.utils.stem._sibson` -- smooth (C1), exact at the
        parents, and the scheme partitioned PRISM is named after.
    device : str or torch.device, optional
        Only used by ``method="sibson"``: a CUDA device runs its Warp kernel
        there (2.5-4x faster, same weights to ~4e-15); ``None`` (default) or a
        CPU device runs NumPy. Pass it when the caller is already on the GPU.

    Returns
    -------
    (B, B_S) ndarray
        Weights for interpolating the grid locations from the B_S sampled points.
    """
    if method == "linear":
        return _linear_barycentric_weights(
            np.asarray(known_points, dtype=float),
            np.asarray(interp_points, dtype=float),
            minimum_weight_cutoff,
        )
    if method == "sibson":
        # Lazy, so the default "linear" path never loads warp.  Outside the
        # hull ``sibson_weights`` already puts weight 1 on the nearest site;
        # re-finding those rows here with ``w.sum(axis=1)`` was a full pass
        # over the dense block, faulting in every page the sparse scatter had
        # left untouched (+38 ms at 9393 x 3169, on both backends).
        from scatterem.utils.stem._sibson import sibson_weights

        return sibson_weights(known_points, interp_points, device=device)
    raise ValueError(f"unknown method {method!r} (use 'linear' or 'sibson')")
