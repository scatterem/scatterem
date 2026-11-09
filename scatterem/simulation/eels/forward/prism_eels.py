"""Scattering-matrix (PRISM) STEM-EELS.

The PRISM algorithm of Brown *et al.*, Phys. Rev. Research **1**, 033186 (2019)
(Eqs. (10)-(11)) accelerates STEM-EELS by precomputing a scattering matrix
:math:`\\mathcal S_1` that propagates the probe-forming plane-wave components to
the plane of an inelastic transition.  Because that matrix is shared by *all*
scan positions, the cost of forming the probe at the ionization plane no longer
scales with the number of probe positions -- the key to the paper's near-linear
scaling.

For an **energy-filtered 4D** output the full exit diffraction pattern is
required, so the second scattering matrix :math:`\\mathcal S_2` cannot be reduced
to a small detector aperture; here the inelastically-scattered wave is propagated
to the exit surface by ordinary multislice.  The :math:`\\mathcal S_1` reuse
across scan positions is the part that makes this cheaper than the conventional
algorithm when many probe positions are scanned.

This module reuses the propagation primitives from
:mod:`scatterem.simulation.eels.forward.multislice_eels` and shares the
:class:`~scatterem.simulation.eels.physics.transition_potentials.TransitionPotentials`
data structure with the conventional path, so both produce identical physics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.fourier import fourier_shift
from scatterem.physics.propagation import propagator_kernel
from scatterem.simulation._backends import EELS_ABS2_ACC, EELS_ABS2_SUM, is_warp

from ..._wave.exit_walk import _slice_exit_wave, _unit_transmission_slices
from ..._wave.partitioned_smatrix import PartitionedScatteringMatrix
from ..._wave.prism_beams import _abs_complex, _factor_pair
from ..._wave.smatrix import ScatteringMatrix
from ..physics.transition_potentials import TransitionPotentials
from .abs2_acc_fused import abs2_accumulate_fused
from .abs2_sum_acc_fused import abs2_sum_accumulate_fused

__all__ = [
    "prism_transition_potential",
    "prism_transition_potential_hybrid",
]


#: Byte budget for the ``(scan_chunk, Ny, Nx)`` complex64 temporary that
#: :func:`prism_transition_potential`'s scan-synthesis GEMM writes, and the floor
#: below which the batch is never taken.
#:
#: The loop reads the WHOLE ``(Ncols, Ny, Nx)`` exit-column stack once per batch,
#: so its DRAM traffic is ``(P/scan_chunk) * |field| + P*Ny*Nx*8`` -- the batch
#: count multiplies the larger term.  At ``r7_eels/wide`` (Ncols = 473, 192 px)
#: ``field`` is 139.5 MB against an 18.9 MB temporary at the historical
#: ``scan_chunk = 64``, so the GEMM re-read 2.2 GB per (site, transition) pair to
#: produce 302 MB, and measured at **78 % of this card's copy roof** -- i.e. the
#: leg was DRAM-bound on a re-read, not on its own arithmetic.  Sizing the batch by
#: bytes instead of rows takes the leg 6.303 -> ~4.4 ms at that shape,
#: bit-exact at every batch size measured (``torch.equal`` True, max|d|
#: exactly 0.0, eight values from 64 to 1024 -- the GEMM's reduction is over
#: ``Ncols`` and does not cross the batch).
#:
#: The floor keeps this from ever *shrinking* an existing caller's batch: at 256 px
#: the budget alone would say 128 and at 512 px it would say 32, so large grids keep
#: the historical 64 rows and only the cheap-temporary shapes grow.
_SCAN_CHUNK_BUDGET_B = 64 << 20
_SCAN_CHUNK_MIN = 64


def _partitioned_columns(
    smatrix: ScatteringMatrix | PartitionedScatteringMatrix,
) -> Tensor:
    """Read reconstructed columns after the caller selects partitioned dispatch."""
    if not isinstance(smatrix, PartitionedScatteringMatrix):
        raise TypeError("partitioned dispatch requires a PartitionedScatteringMatrix")
    return smatrix.reconstruct_columns()


def _default_scan_chunk(ny: int, nx: int) -> int:
    """Scan positions per synthesis batch when the caller names none."""
    return max(_SCAN_CHUNK_MIN, _SCAN_CHUNK_BUDGET_B // (ny * nx * 8))


def prism_transition_potential(
    probe_q: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    scan_pixels: Tensor,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    return_real_space: bool = False,
    interpolation_factor=1,
    partition=None,
    scan_chunk: int | None = None,
) -> Tensor:
    """Dual-scattering-matrix PRISM STEM-EELS (Brown & Ophus).

    Implements the two-scattering-matrix algorithm of Brown, Ciston & Ophus,
    Phys. Rev. Research **1**, 033186 (2019), Eq. (11):

        S_{n,iz} = S2 . F . H_{n0} . F^{-1} . S1,

    where :math:`\\mathcal S_1` propagates the probe-forming plane waves from the
    entrance surface to the ionization slice ``iz`` and :math:`\\mathcal S_2`
    propagates the inelastically-scattered wave from ``iz`` to the exit surface.

    The key to the full PRISM benefit (vs. :func:`~scatterem.simulation.eels.forward.prism_eels.prism_transition_potential_hybrid`)
    is the *order of operations*. Because every operator after :math:`\\mathcal S_1`
    is linear, the exit wave for scan position ``p`` is::

        psi_exit(p) = S2[ H_n0 . (sum_b coeff_p[b] S1_col[b]) ]
                    = sum_b coeff_p[b] * S2[ H_n0 . S1_col[b] ],

    so the transition is applied to the ``Nbeams`` columns of :math:`\\mathcal S_1`
    and **those columns** are propagated to the exit (the action of
    :math:`\\mathcal S_2`), *then* the scan positions are synthesised by the same
    linear combination used to form the probe. The expensive exit-surface
    multislice therefore runs ``Nbeams`` times per (site, transition) -- **independent
    of the number of scan positions** ``P`` -- instead of ``P`` times. For
    ``P >> Nbeams`` (a dense scan through a small aperture, the PRISM regime) this
    is the source of the near-linear scaling.

    With the exact per-pixel scattering matrix (``partition=None``) the result is
    identical (to float rounding) to :func:`~scatterem.simulation.eels.forward.prism_eels.prism_transition_potential_hybrid`;
    this routine just reorders the work. Energy-filtered 4D output is preserved --
    ``S2`` is applied as on-the-fly multislice of the column basis (the full exit
    diffraction pattern is kept, so no detector reduction is made).
    Detector-integrating the returned cube over an EELS aperture reproduces the
    Brown & Ophus ``I(x, y)`` image.

    **Shrinking the column count (two compatible levers).**

    * ``interpolation_factor = f`` subsamples the aperture beams to ``Nbeams / f**2``
      columns -- exact within the PRISM aliasing window -- shrinking the exact
      columns of **both** legs (S1 build and exit) by ``f**2``.
    * ``partition`` (e.g. ``{"n_radial": 4}``) builds S1 on ``Bp << Nbeams`` parent
      beams (Pelz/Ophus partitioned PRISM), then reconstructs the exact aperture
      columns from the parents at the ionization plane and runs the same dual-S
      exit. This cheapens the **S1 build + memory** (``Bp`` parent propagations) and
      is *approximate* (natural-neighbor reconstruction error -- accurate in-focus /
      thin / small-aperture, degrading with thickness/aperture); the exit leg still
      propagates the reconstructed ``Nbeams / f**2`` columns, so the dual-S
      ``P``-independence is preserved. Note partitioning does **not** reduce the
      (usually dominant) exit leg -- only the S1 build; ``interpolation_factor`` is
      the lever for the exit. The two compose (``partition`` partitions the
      already-subsampled aperture). Unlike partitioning *S2*, this is correct: the
      reconstruction resolves the position-dependent NNW basis *before* ``H_{n0}``
      is applied (see ``PartitionedScatteringMatrix.reconstruct_columns``).

    Parameters
    ----------
    probe_q, transmissions, transition_potentials, sites, scan_pixels, wavelength, gridsize, slice_distance, return_real_space, interpolation_factor, partition
        As in :func:`~scatterem.simulation.eels.forward.prism_eels.prism_transition_potential_hybrid`. ``partition`` and
        ``interpolation_factor`` may be combined.
    scan_chunk : int, optional
        Number of scan positions synthesised per batch (caps the
        ``(scan_chunk, Ny, Nx)`` temporary in the linear-combination step).
        ``None`` (the default) sizes the batch by a **byte** budget instead of a
        row count -- see :data:`_SCAN_CHUNK_BUDGET_B`.  An explicit value is
        honoured exactly.

    Returns
    -------
    Tensor
        Intensity ``(P, Ny, Nx)`` summed over transitions and sites, per
        incident electron and per eV of energy loss: ``|fft2|^2 / (Ny Nx)``,
        or ``|psi|^2`` with ``return_real_space``.
    """
    device = transmissions.device
    ny, nx = transmissions.shape[-2:]
    nz = transmissions.shape[0]
    H = transition_potentials.array.to(device=device)  # (n_trans, Ny, Nx)
    n_trans = H.shape[0]
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = BeamPartition.uniform(nz).bin_index(sites[:, 2])  # potential_stack's

    # Slices whose transmission function is exactly ``1 + 0j``: their multiply in
    # the exit leg below is the identity and is skipped bit-identically.  The exit
    # leg is 85 % of a dual-S simulate and one transmit is a full pass over the
    # (Ncols, Ny, Nx) column stack, so this is worth the one batched reduction.
    unit_slices = _unit_transmission_slices(transmissions)

    P = scan_pixels.shape[0]
    out = torch.zeros((P, ny, nx), dtype=torch.float64, device=device)
    if scan_chunk is None:
        scan_chunk = _default_scan_chunk(ny, nx)

    # S1: built once, advanced lazily to each ionization plane, reused across all
    # scan positions (and all sites/transitions).  Both matrices expose the same
    # ``columns`` (scalar-combinable basis at the current slice) + ``coeffs_at``
    # interface, so the dual-S exit is identical for either.  With partitioning the
    # S1 build/memory uses only Bp parent columns; the exact aperture columns are
    # reconstructed (NNW) at the ionization plane for the exit -- so partitioning
    # cheapens the S1 leg (approximately, NNW error) while the exit stays
    # P-independent.  ``interpolation_factor`` subsamples the aperture beams (and
    # composes with partitioning), shrinking the exact-column count of BOTH legs.
    partitioned = partition is not None
    if partitioned:
        smatrix = PartitionedScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
            **partition,
        )
    else:
        smatrix = ScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
        )

    for i in range(nz):
        in_slice = np.nonzero(site_slice == i)[0]
        if in_slice.size == 0:
            continue
        smatrix.advance_to(i)
        # Scalar-combinable columns at slice i + the matching synthesis weights.
        cols = _partitioned_columns(smatrix) if partitioned else smatrix.S
        coeffs = smatrix.coeffs_at(scan_pixels)  # (P, Ncols)
        for s in in_slice:
            sy = sites[s, 0] % 1.0 * ny
            sx = sites[s, 1] % 1.0 * nx
            h_site = fourier_shift(H, (sy, sx))  # (n_trans, Ny, Nx)
            for t in range(n_trans):
                # Apply the transition to each column (H_n0 . S1), then propagate
                # the columns to the exit surface (the action of S2).
                post = h_site[t][None] * cols  # (Ncols, Ny, Nx)
                exit_cols = _slice_exit_wave(
                    post, transmissions, kernel, i, unit_slices
                )
                # Dead column stacks are released as soon as they are last read,
                # not when the next iteration rebinds them: otherwise the previous
                # slice's ``post``/``exit_cols``/``field`` and ``cols`` (an alias
                # of the OLD ``smatrix.S``) are all still alive inside the next
                # ``advance_to``, which is where the peak sits -- 2.03x less peak
                # on ``r7_eels/large/simulate``.  ``field`` IS
                # ``exit_cols`` when ``return_real_space``, so both are released
                # together.
                post = None
                field = (
                    exit_cols
                    if return_real_space
                    else torch.fft.fft2(exit_cols, dim=(-2, -1))
                )  # (Ncols, Ny, Nx)
                # Synthesise every scan position by the SAME linear combination
                # that forms the probe (chunked over P to cap memory).
                for c0 in range(0, P, scan_chunk):
                    c1 = min(c0 + scan_chunk, P)
                    ex = torch.einsum("cb,byx->cyx", coeffs[c0:c1], field)
                    # No ``.to(torch.float64)``: ``out`` is float64, so the in-place
                    # add already promotes the float32 operand inside TensorIterator
                    # -- casting first materialises a second full-size stack (read
                    # f32, write f64) that the add then reads again.  Bit-identical
                    # (the float32->float64 cast is exact either way), verified by
                    # ``torch.equal`` at this shape and by this target's ``cube:
                    # sha256`` contract.  Worth 14.7 ms of a 359 ms
                    # ``r7_eels/wide/simulate``;
                    #
                    # The three remaining passes (hypot -> pow -> add_) are 40
                    # B/element to consume 8; one Warp kernel does it in 24 and is
                    # BIT-EXACT, which is a contract on this path and not a
                    # courtesy -- see :mod:`.abs2_acc_fused`.  Worth a further
                    # ~14.4 ms of the same row.  ``dst`` is bound rather
                    # than re-subscripted so both arms address one view.
                    dst = out[c0:c1]
                    if is_warp(EELS_ABS2_ACC, ex, dst):
                        abs2_accumulate_fused(dst, ex)
                    else:
                        dst += _abs_complex(ex).pow(2)
                exit_cols = field = ex = dst = None
        cols = coeffs = h_site = None

    return out if return_real_space else out / (ny * nx)  # Parseval: per electron


def prism_transition_potential_hybrid(
    probe_q: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    scan_pixels: Tensor,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    return_real_space: bool = False,
    interpolation_factor=1,
    partition=None,
) -> Tensor:
    """Hybrid PRISM/multislice STEM-EELS (single scattering matrix).

    This is the original implementation: the **first** scattering matrix
    :math:`\\mathcal S_1` (reused across scan positions) forms the probe at the
    ionization plane, but the inelastically-scattered wave is then propagated to
    the exit by ordinary multislice **per scan position** -- so the exit-leg cost
    still scales with the number of probe positions.

    :func:`~scatterem.simulation.eels.prism_transition_potential` is the proper dual-scattering-matrix
    algorithm (Brown & Ophus) and is the preferred entry point; this function is
    kept for the partitioned-PRISM path and as the equivalence oracle. The two
    return bit-for-bit-equivalent results (up to float rounding).

    Parameters
    ----------
    probe_q : Tensor
        Probe-forming aperture in reciprocal space ``(Ny, Nx)`` (complex).
    transmissions : Tensor
        Real-space transmission functions ``(NZ, Ny, Nx)``.
    transition_potentials : TransitionPotentials
        Transition potentials for the edge.
    sites : ndarray
        Fractional ``(Nsite, 3)`` ``(y, x, z)`` coordinates of ionizable atoms.
    scan_pixels : Tensor
        Scan positions ``(P, 2)`` in pixels.
    wavelength, gridsize, slice_distance
        As in :func:`~scatterem.simulation.eels.multislice_transition_potential`.
    return_real_space : bool, optional
        If ``True``, return the real-space exit-wave intensity instead of the
        diffraction-space intensity.
    interpolation_factor : int or (int, int)
        PRISM interpolation factor ``f`` (or ``(fy, fx)``); see
        ``ScatteringMatrix``.  ``1`` (default) is exact; ``f > 1`` subsamples
        the beams to speed up the scattering matrix by ``f**2`` and is accurate
        while the probe fits in the ``1/f`` window.
    partition : dict or None
        If given (e.g. ``{"n_radial": 4}``), use a
        ``PartitionedScatteringMatrix`` (partitioned PRISM, Pelz 2021)
        instead; ``interpolation_factor`` is then ignored.  ``None`` (default)
        uses the exact per-pixel scattering matrix.

    Returns
    -------
    Tensor
        Intensity ``(P, Ny, Nx)`` summed over transitions and sites, per
        incident electron and per eV of energy loss: ``|fft2|^2 / (Ny Nx)``,
        or ``|psi|^2`` with ``return_real_space``.
    """
    device = transmissions.device
    ny, nx = transmissions.shape[-2:]
    nz = transmissions.shape[0]
    H = transition_potentials.array.to(device=device)
    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = BeamPartition.uniform(nz).bin_index(sites[:, 2])  # potential_stack's

    P = scan_pixels.shape[0]
    out = torch.zeros((P, ny, nx), dtype=torch.float64, device=device)

    # Slices whose transmission function is exactly ``1 + 0j`` -- their multiply is
    # the identity for every finite input and is skipped, bit-exactly.  Computed
    # once per call (one batched reduction, one device sync); ``transmissions`` is
    # only ever read from here on, never written in place.
    unit_slices = _unit_transmission_slices(transmissions)

    # S1: reused across all scan positions; advanced lazily to each plane.
    if partition is not None:
        smatrix = PartitionedScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
            **partition,
        )
    else:
        smatrix = ScatteringMatrix(
            probe_q.to(transmissions.dtype),
            transmissions,
            kernel,
            interpolation_factor=_factor_pair(interpolation_factor),
        )

    for i in range(nz):
        in_slice = np.nonzero(site_slice == i)[0]
        if in_slice.size == 0:
            continue
        smatrix.advance_to(i)
        probes = smatrix.probe_at_current_plane(scan_pixels)  # (P, Ny, Nx)
        for s in in_slice:
            sy = sites[s, 0] % 1.0 * ny
            sx = sites[s, 1] % 1.0 * nx
            h_site = fourier_shift(H, (sy, sx))  # (n_trans, Ny, Nx)
            # The (P, n_trans, Ny, Nx) product is handed over with no name kept on
            # it, so the walk can reuse its storage and return the spectrum in it:
            # one exit stack live instead of two (``_slice_exit_wave``, ``consume``).
            stack = _slice_exit_wave(
                h_site[None] * probes[:, None],
                transmissions,
                kernel,
                i,
                unit_slices,
                consume=True,
                spectrum=not return_real_space,
            )
            # The float32 sum is added straight into the float64 ``out``; casting
            # it first is a second full-size pass for no bits (same reason as the
            # dual-S synthesis above).  Where it is supported the whole tail is one
            # Warp kernel instead of four -- bit-exact, and bit-exact against
            # ATen's FOUR-accumulator reduction order rather than the obvious
            # sequential one; see ``abs2_sum_acc_fused``.
            if is_warp(EELS_ABS2_SUM, stack, out):
                abs2_sum_accumulate_fused(out, stack)
            else:
                out += _abs_complex(stack).pow(2).sum(dim=1)
            stack = None

    return out if return_real_space else out / (ny * nx)  # Parseval: per electron
