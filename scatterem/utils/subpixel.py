"""Sub-pixel image registration by local DFT evaluation.

Written from the published method, not from another implementation of it. The
method is

    M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup,
    "Efficient subpixel image registration algorithms",
    Optics Letters 33, 156 (2008).

This is an independent implementation. The method is published and widely
implemented -- py4DSTEM's ``multicorr`` is one such implementation, under GPL-3.0 --
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

import torch

#: Half-width of the refinement window, in original pixels. One pixel: the true
#: maximum is within half a pixel of the integer peak by construction, and this
#: leaves margin either side.
REFINE_RADIUS_PX = 1.0


def _dft_basis(frequencies: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``exp(2i*pi*f*u)`` for every (offset, frequency) pair.

    One of the two matrices in the separable evaluation; ``offsets`` of shape
    ``(..., n_off)`` give ``(..., n_off, n_freq)``. The reference refinement
    passes float64: the phases reach ``2*pi*f*u`` with ``u`` of order the image
    size, where float32 loses enough precision to move the fitted peak.
    """
    phase = 2.0 * math.pi * offsets[..., None] * frequencies
    return torch.polar(torch.ones_like(phase), phase)


#: Refinement precision -> the real dtype its offsets and frequencies use.
_REFINE_REAL_DTYPE = {torch.complex128: torch.float64, torch.complex64: torch.float32}


def _peak_offsets(
    cross_power: torch.Tensor,
    upsample_factor: float,
    refine_dtype: torch.dtype = torch.complex128,
    integer_peak: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sub-pixel peak location of each cross-power spectrum, ``(nbatch, 2)``.

    Two stages: the integer peak (the global maximum of the ordinary inverse
    transform, or ``integer_peak``, row and col long tensors), then the correlation
    on a fine grid around it. Both run on the whole batch, with no host sync.

    ``refine_dtype`` is the fine-grid precision; the result is float64 for the
    default ``complex128`` (the reference), float32 for ``complex64``, which is
    faster but cannot resolve the curvature of a broad peak: on smooth test
    images it is off by 0.01-0.05 px, a fraction of the ``1/U`` step.
    """
    real_dtype = _REFINE_REAL_DTYPE.get(refine_dtype)
    if real_dtype is None:
        raise ValueError(
            f"refine_dtype must be torch.complex128 or torch.complex64; got {refine_dtype}"
        )
    nbatch, height, width = cross_power.shape
    device = cross_power.device

    # Stage 1 -- integer peak. abs() rather than real(): the correlation of two
    # images that differ by more than a shift has a complex peak, and taking the
    # real part there biases the location.
    if integer_peak is None:
        flat = torch.fft.ifft2(cross_power).abs().reshape(nbatch, -1).argmax(dim=1)
        peak_row, peak_col = flat // width, flat % width
        # Interpret the peak as a shift: bins past the halfway point are negative.
        shift_row = torch.where(peak_row > height // 2, peak_row - height, peak_row)
        shift_col = torch.where(peak_col > width // 2, peak_col - width, peak_col)
    else:
        shift_row, shift_col = integer_peak

    # Stage 2 -- a fine grid of offsets about that peak, at 1/U spacing.
    step = 1.0 / float(upsample_factor)
    n_side = round(2.0 * REFINE_RADIUS_PX / step) + 1
    ladder = torch.arange(n_side, device=device, dtype=real_dtype) * step
    ladder = ladder - REFINE_RADIUS_PX

    f_row = torch.fft.fftfreq(height, device=device, dtype=real_dtype)
    f_col = torch.fft.fftfreq(width, device=device, dtype=real_dtype)
    row_offsets = shift_row.to(real_dtype)[:, None] + ladder  # (nbatch, n_side)
    col_offsets = shift_col.to(real_dtype)[:, None] + ladder
    basis_row = _dft_basis(f_row, row_offsets)  # (nbatch, n_side, height)
    basis_col = _dft_basis(f_col, col_offsets)  # (nbatch, n_side, width)

    # c = E_row @ C @ E_col^T -- the separable evaluation, one per batch item.
    spectrum = cross_power.to(refine_dtype)
    patch = (basis_row @ spectrum @ basis_col.transpose(1, 2)).abs()
    if patch.is_cuda:  # one bit-exact Warp kernel for the tail below
        # Lazy, so Warp loads on first use. refine_tail returns None where its
        # kernel does not apply, and the torch ops below then run.
        from scatterem.utils._subpixel_tail_warp import refine_tail

        if (refined := refine_tail(patch, row_offsets, col_offsets, step)) is not None:
            return refined

    best = patch.reshape(nbatch, -1).argmax(dim=1)
    i_row = torch.div(best, n_side, rounding_mode="floor")
    i_col = best % n_side
    items = torch.arange(nbatch, device=device)

    # The best grid sample is only accurate to half a step, so interpolate: a
    # parabola through the peak and its two neighbours has its vertex at the
    # offset below, in steps. This lifts the result past the 1/U grid, which
    # alone bounds the error measurably worse.
    vertex_row = _parabolic_vertices(patch[items, :, i_col], i_row)
    vertex_col = _parabolic_vertices(patch[items, i_row, :], i_col)
    refined_row = row_offsets[items, i_row] + step * vertex_row
    refined_col = col_offsets[items, i_col] + step * vertex_col
    return torch.stack([refined_row, refined_col], dim=1)


def _parabolic_vertices(lines: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """:func:`_parabolic_vertex` for each row of ``lines`` at its ``index``.

    ``lines`` ``(nbatch, n)``, long ``index`` ``(nbatch,)`` -> ``(nbatch,)`` in
    the dtype of ``lines``: same arithmetic and rules, without host syncs.
    """
    n = lines.shape[1]
    interior = (index > 0) & (index < n - 1)
    rows = torch.arange(lines.shape[0], device=lines.device)
    left = lines[rows, (index - 1).clamp(0, n - 1)]
    middle = lines[rows, index]
    right = lines[rows, (index + 1).clamp(0, n - 1)]
    curvature = left - 2.0 * middle + right
    usable = interior & (curvature != 0.0)
    safe = torch.where(usable, curvature, torch.ones_like(curvature))
    vertex = (0.5 * (left - right) / safe).clamp(-0.5, 0.5)
    return torch.where(usable, vertex, torch.zeros_like(vertex))


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
    refine_dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Shift of each spectrum in ``spectra`` relative to ``reference_spectrum``.

    Parameters
    ----------
    reference_spectrum : torch.Tensor
        ``fft2`` of the reference image, ``(H, W)``.
    spectra : torch.Tensor
        ``fft2`` of the images to register, ``(H, W)`` or ``(N, H, W)``.
    upsample_factor : float, optional
        Samples per original pixel in the refinement grid, so the attainable
        precision is about ``1 / upsample_factor`` pixels.
    refine_dtype : torch.dtype, optional
        Precision of the fine-grid refinement, ``torch.complex128`` (default,
        the reference) or ``torch.complex64`` (faster, but off by a fraction of the ``1/U`` step;
        see :func:`_peak_offsets`).

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` shifts in pixels, ordered ``(row, column)``, in the real dtype
        of ``spectra``. ``N`` is 1 for a single input image.

    Raises
    ------
    ValueError
        If ``spectra`` is neither 2D nor 3D, the shapes disagree, or
        ``upsample_factor`` is not positive, or ``refine_dtype`` is not one of
        the two complex dtypes.
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
    return _peak_offsets(cross_power, upsample_factor, refine_dtype).to(
        spectra.real.dtype
    )


def relative_shifts(G1, G2, upsample_factor):
    """Shift of ``G2`` relative to ``G1``, to a fraction of a pixel.

    Kept as the name the reconstruction code calls. See :func:`subpixel_shifts`.

    Parameters
    ----------
    G1 : torch.Tensor
        ``fft2`` of the reference image, ``(H, W)``.
    G2 : torch.Tensor
        ``fft2`` of the image(s) to register, ``(H, W)`` or ``(N, H, W)``.
    upsample_factor : float
        Samples per original pixel in the refinement grid.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` shifts in pixels, ordered ``(row, column)``.
    """
    return subpixel_shifts(G1, G2, upsample_factor)


# --- graph of pairwise shifts -------------------------------------------------
#
# These three are scatterem's own -- measured against py4DSTEM at one identical
# line, which is `import numpy as np`. They moved here from utils/registration.py
# so that file, which is a port, need not ship at all.


def make_neighbor_pairs(coords: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """Neighbor index pairs for points on an integer grid (e.g. bright-field detector coords).

    Parameters
    ----------
    coords : torch.Tensor
        ``(N, 2)`` grid coordinates. Floating-point coordinates may be a
        translated integer grid (e.g. centred on a fractional centroid): they
        are re-anchored to the per-axis minimum and rounded, which is exact;
        truncating would merge the rows/columns straddling zero.
    connectivity : int, optional
        4 (axial) or 8 (axial + diagonal) neighborhood.

    Returns
    -------
    torch.Tensor
        ``(M, 2)`` long tensor of index pairs ``(i, j)`` that are grid
        neighbors.
    """
    if coords.is_floating_point() and len(coords) > 0:
        # Round, do not truncate: truncating centred coordinates can split one
        # bright-field disk into many components.
        coords = torch.round(coords - coords.min(0).values).long()
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


#: Default complex elements per chunk of :func:`pairwise_relative_shifts` (256 MiB
#: per buffer at complex128; the refinement holds a few at once).
_PAIRWISE_CHUNK_ELEMENTS = 2**24


def pairwise_relative_shifts(
    G_stack: torch.Tensor,
    pairs: torch.Tensor,
    upsample_factor: int = 10,
    chunk_size: int | None = None,
    refine_dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Cross-correlate each neighbor pair: delta_ij = relative shift of image j vs image i.

    Same refinement and sign/gauge convention as :func:`relative_shifts`, so the
    synchronized result is a drop-in for reference-based shifts. Pairs are
    processed in chunks, each one batched cross-power spectrum and refinement.

    Parameters
    ----------
    G_stack : torch.Tensor
        ``(N, H, W)`` Fourier-domain (``fft2``) bright-field images.
    pairs : torch.Tensor
        ``(M, 2)`` neighbor index pairs, on any device.
    upsample_factor : int, optional
        Subpixel registration upsample factor.
    chunk_size : int or None, optional
        Pairs per batch, positive (else ``ValueError``); ``None`` keeps
        ``chunk * H * W`` within 2**24.
        Results do not depend on it beyond floating-point summation order.
    refine_dtype : torch.dtype, optional
        Precision of the fine-grid refinement; see :func:`subpixel_shifts`.

    Returns
    -------
    torch.Tensor
        ``(M, 2)`` relative shifts per pair (``delta[k]`` = shift of image j
        vs image i), in the default floating dtype, on ``G_stack.device``.
    """
    num_pairs = pairs.shape[0]
    height, width = G_stack.shape[-2:]
    if chunk_size is None:
        chunk_size = max(1, _PAIRWISE_CHUNK_ELEMENTS // (height * width))
    elif chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    device = G_stack.device
    tail = pairs[:, 0].to(device=device, dtype=torch.long)
    head = pairs[:, 1].to(device=device, dtype=torch.long)
    deltas = torch.zeros((num_pairs, 2), device=device)
    for start in range(0, num_pairs, chunk_size):
        stop = min(num_pairs, start + chunk_size)
        cross_power = G_stack[tail[start:stop]] * G_stack[head[start:stop]].conj()
        # Round through the input's real dtype first, as the per-pair
        # relative_shifts call this replaced did.
        deltas[start:stop] = _peak_offsets(
            cross_power, upsample_factor, refine_dtype
        ).to(G_stack.real.dtype)
    return deltas


def synchronize_shifts(
    num_nodes: int,
    pairs: torch.Tensor,
    deltas: torch.Tensor,
    n_irls: int = 0,
    huber_c: float = 2.5,
) -> torch.Tensor:
    """Compute least-squares absolute shifts from overdetermined pairwise differences.

    Solves min_t sum_k w_k || (t[j_k] - t[i_k]) - delta_k ||^2 (graph-Laplacian
    normal equations), gauge-fixed by anchoring node 0. The gauge is arbitrary;
    callers typically zero-mean the result. By default every ``w_k`` is 1: plain
    least squares.

    A few pairwise measurements can fail grossly (wraparound-scale deltas on
    0.4-1% of edges of real bright-field data), and plain least squares smears
    them over the graph. ``n_irls > 0`` adds that many Huber-reweighted solves:
    an edge whose residual exceeds ``huber_c`` times the robust scale
    ``1.4826 * median(residual)`` gets weight ``huber_c * scale / residual``.
    It stops early once all residuals vanish. :func:`synchronize_shifts_weighted`
    also returns the final edge weights.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N.
    pairs : torch.Tensor
        ``(M, 2)`` long index pairs ``(i, j)``.
    deltas : torch.Tensor
        ``(M, 2)`` measured ``t[j] - t[i]``.
    n_irls : int, optional
        Huber reweighting iterations after the plain solve (0: plain LS).
    huber_c : float, optional
        Huber threshold in units of the robust residual scale.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` absolute shifts in the dtype of ``deltas``. ``ValueError`` if
        ``n_irls`` < 0 or ``huber_c`` is not positive.
    """
    return synchronize_shifts_weighted(num_nodes, pairs, deltas, n_irls, huber_c)[0]


def synchronize_shifts_weighted(
    num_nodes: int,
    pairs: torch.Tensor,
    deltas: torch.Tensor,
    n_irls: int = 0,
    huber_c: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`synchronize_shifts` plus the final ``(M,)`` edge weights in ``[0, 1]``
    (all ones without reweighting; a node whose edges all weigh ~0 was not measured)."""
    if n_irls < 0:
        raise ValueError(f"n_irls must be non-negative; got {n_irls}")
    if not huber_c > 0:
        raise ValueError(f"huber_c must be positive; got {huber_c}")
    device = deltas.device
    tail = pairs[:, 0].to(device=device, dtype=torch.long)
    head = pairs[:, 1].to(device=device, dtype=torch.long)
    weights = torch.ones(tail.numel(), device=device, dtype=deltas.dtype)
    shifts = _solve_weighted_laplacian(num_nodes, tail, head, deltas, weights)
    for _ in range(n_irls):
        residual = torch.linalg.vector_norm(shifts[head] - shifts[tail] - deltas, dim=1)
        if residual.numel() == 0:
            break
        scale = 1.4826 * residual.median()
        if not torch.isfinite(scale) or scale < 1e-12:
            break  # measurements already consistent, e.g. exact deltas
        weights = torch.clamp(huber_c * scale / residual.clamp_min(1e-12), max=1.0)
        shifts = _solve_weighted_laplacian(num_nodes, tail, head, deltas, weights)
    return shifts, weights


def _solve_weighted_laplacian(n: int, tail, head, deltas, weights) -> torch.Tensor:
    """Anchored solve of the weighted normal equations of :func:`synchronize_shifts`.

    ``tail``/``head`` are ``(M,)`` long, ``deltas`` ``(M, 2)``, ``weights`` ``(M,)``.
    """
    device, dtype = deltas.device, deltas.dtype

    # Normal-equation matrix of the least-squares problem is the weighted graph
    # Laplacian: weighted node degree on the diagonal, minus the edge weight off
    # it. With unit weights the entries are integer counts, exact in any dtype.
    degrees = torch.zeros(n, device=device, dtype=dtype)
    degrees.index_add_(0, torch.cat([tail, head]), torch.cat([weights, weights]))
    A = torch.diag(degrees)
    A.index_put_((tail, head), -weights, accumulate=True)
    A.index_put_((head, tail), -weights, accumulate=True)

    # Right-hand side: every measurement pushes its two endpoints apart, so it
    # enters with a minus sign at the tail and a plus sign at the head.
    weighted = weights[:, None] * deltas
    b = torch.zeros((n, 2), device=device, dtype=dtype)
    b.index_add_(0, tail, -weighted)
    b.index_add_(0, head, weighted)

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
