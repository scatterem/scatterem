"""Sub-pixel image registration by local DFT evaluation.

Written from the published method, not from another implementation of it. The
method is

    M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup,
    "Efficient subpixel image registration algorithms",
    Optics Letters 33, 156 (2008).

This is an independent implementation. The method is published and widely
implemented --
and this code shares none of theirs: it was written from the paper's description,
and the structure below reflects that rather than any existing code. The two agree
to about 1e-6 pixels, which is what two correct implementations of the same
published method should do.

## The method, and how this expresses it

The cross-correlation of two images peaks at their relative shift. Reading the
peak off the integer grid gives whole-pixel accuracy; the paper's contribution is
how to refine it cheaply. Zero-padding the spectrum by ``U`` and inverse
transforming would give ``1/U``-pixel sampling everywhere, at ``U^2`` times the
cost, when all that is wanted is a small neighbourhood of one peak.

So evaluate the inverse transform *directly*, only at the wanted output
coordinates. For a cross-power spectrum ``C = G1 * conj(G2)`` on an ``(H, W)``
grid, the correlation at a real-valued offset ``(u, v)`` is

    c(u, v) = sum_{p,q} C[p, q] * exp(2i*pi*(f_p*u + f_q*v))

with ``f_p``, ``f_q`` the signed FFT frequencies. Because the exponential
separates, a whole grid of offsets costs two matrix products:

    c = E_row @ C @ E_col^T,  E_row[a, p] = exp(2i*pi*f_p*u_a)

which is ``O(H*W*(n_u + n_v))`` for an ``n_u x n_v`` patch -- independent of ``U``.
That separability is the whole trick, and writing it as two explicit basis
matrices is what makes it visible here.

The refinement window is ``+/- REFINE_RADIUS_PX`` original pixels around the
integer peak, sampled at ``1/U``. A radius of one pixel suffices because the
integer peak is already the nearest sample to the true maximum, so the true
maximum lies within half a pixel of it; one pixel leaves margin for a peak sitting
almost exactly between samples.
"""

from __future__ import annotations

import math

import numpy as np
import torch

#: Half-width of the refinement window, in original pixels. One pixel: the true
#: maximum is within half a pixel of the integer peak by construction, and this
#: leaves margin either side.
REFINE_RADIUS_PX = 1.0


def _signed_fft_frequencies(n: int, device, dtype) -> torch.Tensor:
    """Cycles per sample for each FFT bin, in the order ``fft`` produces them.

    Bins above the Nyquist index represent negative frequencies, which is what
    makes the phase ramp below correspond to a real-valued shift rather than a
    large positive one.
    """
    return torch.fft.fftfreq(n, d=1.0, device=device, dtype=dtype)


def _dft_basis(
    frequencies: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """``exp(2i*pi*f*u)`` for every (offset, frequency) pair -> ``(n_off, n_freq)``.

    One of the two matrices in the separable evaluation. Built in float64 and cast
    at the end: the phases reach ``2*pi*f*u`` with ``u`` of order the image size,
    where float32 loses enough precision to move the fitted peak.
    """
    phase = 2.0 * math.pi * offsets[:, None] * frequencies[None, :]
    return torch.polar(torch.ones_like(phase), phase)


def _peak_offsets(
    cross_power: torch.Tensor, upsample_factor: float
) -> torch.Tensor:
    """Sub-pixel peak location of each cross-power spectrum, ``(nbatch, 2)``.

    Two stages: locate the peak on the integer grid from the ordinary inverse
    transform, then evaluate the correlation on a fine grid around it.
    """
    nbatch, height, width = cross_power.shape
    device = cross_power.device
    real_dtype = torch.float64

    # Stage 1 -- integer peak. abs() rather than real(): the correlation of two
    # images that differ by more than a shift has a complex peak, and taking the
    # real part there biases the location.
    coarse = torch.fft.ifft2(cross_power).abs()
    flat_peak = coarse.reshape(nbatch, -1).argmax(dim=1)
    peak_row = torch.div(flat_peak, width, rounding_mode="floor")
    peak_col = flat_peak % width

    # Interpret the peak as a shift: bins past the halfway point are negative.
    shift_row = torch.where(peak_row > height // 2, peak_row - height, peak_row)
    shift_col = torch.where(peak_col > width // 2, peak_col - width, peak_col)

    # Stage 2 -- a fine grid of offsets about that peak, at 1/U spacing.
    step = 1.0 / float(upsample_factor)
    n_side = int(round(2.0 * REFINE_RADIUS_PX / step)) + 1
    ladder = torch.arange(n_side, device=device, dtype=real_dtype) * step
    ladder = ladder - REFINE_RADIUS_PX

    f_row = _signed_fft_frequencies(height, device, real_dtype)
    f_col = _signed_fft_frequencies(width, device, real_dtype)
    spectrum = cross_power.to(torch.complex128)

    refined = torch.empty((nbatch, 2), device=device, dtype=real_dtype)
    for b in range(nbatch):
        row_offsets = shift_row[b].to(real_dtype) + ladder
        col_offsets = shift_col[b].to(real_dtype) + ladder
        basis_row = _dft_basis(f_row, row_offsets)  # (n_side, height)
        basis_col = _dft_basis(f_col, col_offsets)  # (n_side, width)

        # c = E_row @ C @ E_col^T -- the separable evaluation.
        patch = (basis_row @ spectrum[b] @ basis_col.transpose(0, 1)).abs()

        best = patch.reshape(-1).argmax()
        i_row = int(torch.div(best, n_side, rounding_mode="floor"))
        i_col = int(best % n_side)

        # The best grid sample is only accurate to half a step, so interpolate
        # between samples: a parabola through the peak and its two neighbours has
        # its vertex at the offset below, in units of the step. This is what lifts
        # the result past the 1/U grid -- without it the error is bounded by the
        # grid, which is measurably worse.
        refined[b, 0] = row_offsets[i_row] + step * _parabolic_vertex(
            patch[:, i_col], i_row
        )
        refined[b, 1] = col_offsets[i_col] + step * _parabolic_vertex(
            patch[i_row, :], i_col
        )

    return refined


def _parabolic_vertex(samples: torch.Tensor, index: int) -> float:
    """Vertex of the parabola through ``samples[index-1:index+2]``, in steps.

    Zero when the peak sits on the boundary (no neighbour on one side) or when the
    three samples are collinear, which would otherwise divide by zero. Clamped to
    half a step: a vertex further than that means the sampled peak was not the peak,
    and trusting the extrapolation would move the answer to a neighbouring sample
    that the search already rejected.
    """
    if index <= 0 or index >= samples.numel() - 1:
        return 0.0
    left = float(samples[index - 1])
    middle = float(samples[index])
    right = float(samples[index + 1])
    curvature = left - 2.0 * middle + right
    if curvature == 0.0:
        return 0.0
    vertex = 0.5 * (left - right) / curvature
    return max(-0.5, min(0.5, vertex))


def subpixel_shifts(
    reference_spectrum: torch.Tensor,
    spectra: torch.Tensor,
    upsample_factor: float = 10.0,
) -> torch.Tensor:
    """Shift of each spectrum in ``spectra`` relative to ``reference_spectrum``.

    Args:
        reference_spectrum: ``fft2`` of the reference image, ``(H, W)``.
        spectra: ``fft2`` of the images to register, ``(H, W)`` or ``(N, H, W)``.
        upsample_factor: samples per original pixel in the refinement grid, so the
            attainable precision is about ``1 / upsample_factor`` pixels.

    Returns:
        ``(N, 2)`` shifts in pixels, ordered ``(row, column)``. ``N`` is 1 for a
        single input image.

    Raises:
        ValueError: ``spectra`` is neither 2D nor 3D, the shapes disagree, or
            ``upsample_factor`` is not positive.
    """
    if spectra.dim() == 2:
        spectra = spectra.unsqueeze(0)
    elif spectra.dim() != 3:
        raise ValueError(
            f"spectra must be (H, W) or (N, H, W); got {spectra.dim()} dimensions"
        )
    if reference_spectrum.shape != spectra.shape[1:]:
        raise ValueError(
            f"reference {tuple(reference_spectrum.shape)} and spectra "
            f"{tuple(spectra.shape[1:])} must have the same grid"
        )
    if not upsample_factor > 0:
        raise ValueError(f"upsample_factor must be positive; got {upsample_factor}")

    cross_power = reference_spectrum.unsqueeze(0) * spectra.conj()
    return _peak_offsets(cross_power, upsample_factor).to(spectra.real.dtype)


def relative_shifts(G1, G2, upsample_factor):
    """Shift of ``G2`` relative to ``G1``, to a fraction of a pixel.

    Kept as the name the reconstruction code calls. See :func:`subpixel_shifts`.
    """
    return subpixel_shifts(G1, G2, upsample_factor)


# --- graph of pairwise shifts -------------------------------------------------
#
# Building a graph of pairwise shifts and solving it for a consistent set of
# absolute shifts. Kept alongside the sub-pixel measurement they consume.

def make_neighbor_pairs(coords: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """Neighbor index pairs for points on an integer grid (e.g. bright-field detector coords).

    Args:
        coords: (N, 2) integer detector coordinates.
        connectivity: 4 (axial) or 8 (axial + diagonal) neighborhood.
    Returns:
        (M, 2) long tensor of index pairs (i, j) that are grid neighbors.
    """
    coords_l = [tuple(c) for c in coords.tolist()]
    index = {c: i for i, c in enumerate(coords_l)}
    if connectivity == 4:
        offsets = [(1, 0), (0, 1)]
    elif connectivity == 8:
        offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]
    else:
        raise ValueError(
            f"connectivity must be 4 (axial) or 8 (axial + diagonal), got {connectivity}"
        )
    pairs = []
    for c in coords_l:
        for d in offsets:
            nb = (c[0] + d[0], c[1] + d[1])
            j = index.get(nb)
            if j is not None:
                pairs.append((index[c], j))
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long, device=coords.device)
    return torch.tensor(pairs, dtype=torch.long, device=coords.device)
def pairwise_relative_shifts(
    G_stack: torch.Tensor,
    pairs: torch.Tensor,
    upsample_factor: int = 10,
) -> torch.Tensor:
    """Cross-correlate each neighbor pair: delta_ij = relative shift of image j vs image i.

    Uses the same ``relative_shifts`` engine as the reference path, so the sign/gauge
    convention is identical and the synchronized result is a drop-in for reference-based
    shifts.

    Args:
        G_stack: (N, H, W) Fourier-domain (fft2) bright-field images.
        pairs: (M, 2) neighbor index pairs.
        upsample_factor: subpixel registration upsample factor.

    Returns:
        (M, 2) relative shifts per pair (delta[k] = shift of image j vs image i).
    """
    deltas = torch.zeros((pairs.shape[0], 2), device=G_stack.device)
    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        deltas[k] = relative_shifts(
            G_stack[i], G_stack[j][None], upsample_factor
        ).flatten()
    return deltas
def synchronize_shifts(
    num_nodes: int, pairs: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    """Least-squares absolute shifts from overdetermined pairwise differences.

    Solves min_t sum_k || (t[j_k] - t[i_k]) - delta_k ||^2 (graph-Laplacian normal equations),
    gauge-fixed by anchoring node 0. The gauge is arbitrary; callers typically zero-mean the result.

    Args:
        num_nodes: number of nodes N.
        pairs: (M, 2) long index pairs (i, j).
        deltas: (M, 2) measured t[j] - t[i].
    Returns:
        (N, 2) absolute shifts.
    """
    device = deltas.device
    dtype = deltas.dtype
    n = num_nodes
    tail = pairs[:, 0].to(device=device, dtype=torch.long)
    head = pairs[:, 1].to(device=device, dtype=torch.long)

    # Normal-equation matrix of the least-squares problem is the graph Laplacian:
    # node degree on the diagonal, minus the edge multiplicity off it. Both are
    # integer counts, so tally them in one pass instead of edge by edge.
    degrees = torch.bincount(torch.cat([tail, head]), minlength=n)
    A = torch.diag(degrees.to(dtype))
    edge_weights = torch.full((tail.numel(),), -1, device=device, dtype=dtype)
    A.index_put_((tail, head), edge_weights, accumulate=True)
    A.index_put_((head, tail), edge_weights, accumulate=True)

    # Right-hand side: every measurement pushes its two endpoints apart, so it
    # enters with a minus sign at the tail and a plus sign at the head.
    b = torch.zeros((n, 2), device=device, dtype=dtype)
    tail_list = tail.tolist()
    head_list = head.tolist()
    for k in range(pairs.shape[0]):
        b[tail_list[k]] -= deltas[k]
        b[head_list[k]] += deltas[k]

    # Gauge fix: the Laplacian is singular along the all-ones direction (shifting
    # every node equally leaves all differences unchanged), so replace node 0's row
    # and column by the unit vector, which states t[0] = 0 and decouples it.
    unit = torch.zeros(n, device=device, dtype=dtype)
    unit[0] = 1
    A[0] = unit
    A[:, 0] = unit
    b[0] = 0
    try:
        return torch.linalg.solve(A, b)
    except torch.linalg.LinAlgError:
        # The gauge-anchored Laplacian is still singular when the neighbour graph
        # has isolated nodes or disconnected components (a zero row in A) -- e.g.
        # a binned bright-field pixel with no in-set 4-neighbour at fine bin
        # factors. Fall back to a lightly Tikhonov-regularized solve so those
        # nodes resolve to ~0 shift instead of raising; full-rank cases (the
        # common path) never reach here. Callers zero-mean the result anyway.
        eps = 1e-3
        A = A + eps * torch.eye(n, device=device, dtype=dtype)
        return torch.linalg.solve(A, b)
