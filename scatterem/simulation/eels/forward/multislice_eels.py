"""Conventional transition-potential multislice for STEM-EELS.

Implements the standard (quadratically-scaling) multislice algorithm for
core-loss EELS described in Brown *et al.*, Phys. Rev. Research **1**, 033186
(2019), Sec. III and Dwyer, Ultramicroscopy **104**, 141 (2005):

For each probe position the elastic wave is propagated slice by slice.  At every
slice that contains an ionizable atom an inelastically-scattered wave
:math:`\\psi_n(\\mathbf r) = H_{n0}(\\mathbf r - \\mathbf r_{atom})\\,\\psi_0(\\mathbf r)`
is created for every transition channel, propagated independently to the exit
surface, and its diffraction intensity accumulated.  Summing the intensity over
all transitions and atom sites yields the energy-filtered diffraction pattern
for the chosen ionization edge -- i.e. one 4D-STEM dataset per edge/energy
window.

This module is backend-agnostic and runs on CPU or GPU; it has no GPAW
dependency (the transition potentials are precomputed elsewhere).

Notes
-----
The optional site threshold. The paper evaluates every transition at every site. ``site_threshold > 0``
skips sites that the probe does not reach, and the rule follows from
unitarity. After the ionization slice the walk multiplies by transmission
functions with ``|T| <= 1`` and by Fresnel propagators with ``|P| = 1``, so
the exit wave of one event carries at most the intensity it starts with. By
Parseval, the total diffraction intensity that the event ``(site, t)`` adds,
summed over all detector pixels, is therefore at most

    E_t = sum_r |H_t(r - r_site) psi_0(r)|^2.

Divide by ``N_t = sum_r |H_t(r)|^2``. The ratio ``E_t / N_t`` is the
``|H_t|^2``-weighted mean of the probe intensity ``|psi_0|^2`` over the
footprint of the transition potential: the probe intensity that the atom
sees. A unit-amplitude plane wave gives 1. A site is skipped when
``E_t < site_threshold * N_t`` for every probe of the batch and every
transition ``t``. The intensity lost from one probe is then less than
``site_threshold * sum N_t`` over the skipped ``(site, t)`` pairs. This is an
intensity form of the distance cut in Sec. IV D of the paper, which keeps
only "transitions within 4 Å of the probe". The default 0 skips nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.fourier import fourier_shift
from scatterem.physics.propagation import propagate, propagator_kernel
from scatterem.simulation._backends import EELS_ABS2_SUM, is_warp

from ..._wave.exit_walk import _slice_exit_wave, _unit_transmission_slices
from ..physics.transition_potentials import TransitionPotentials
from .abs2_sum_acc_fused import abs2_sum_accumulate_fused

__all__ = ["multislice_transition_potential"]


def multislice_transition_potential(
    probes: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    site_threshold: float = 0.0,
    batch_size: int | None = None,
    return_real_space: bool = False,
) -> Tensor:
    """Energy-filtered diffraction patterns for a batch of probe positions.

    Parameters
    ----------
    probes : Tensor
        Complex real-space probes ``(P, Ny, Nx)`` already placed at their scan
        positions.
    transmissions : Tensor
        Complex real-space transmission functions ``(NZ, Ny, Nx)`` -- one per
        depth slice.
    transition_potentials : TransitionPotentials
        Real-space transition potentials for the chosen edge.
    sites : ndarray
        Fractional coordinates ``(Nsite, 3)`` ``(y, x, z)`` in ``[0, 1)`` of the
        ionizable atoms (element ``transition_potentials.Z``) within the field
        of view.
    wavelength : float
        Probe wavelength [Å].
    gridsize : float
        Shape ``(2,)``.
        Field-of-view size ``(Ly, Lx)`` [Å].
    slice_distance : float
        Inter-slice propagation distance Δz [Å].
    site_threshold : float
        Skip a site when, for every probe of the batch and every transition,
        the probe intensity that the site sees (the ``|H|^2``-weighted mean of
        ``|psi|^2``) is below this value. The unit is the intensity of a
        unit-amplitude plane wave. The module docstring gives the derivation
        and the error bound. The default 0 skips nothing.
    batch_size : int, optional
        Probe-position batch size (memory control).  Defaults to all probes.
    return_real_space : bool
        If True, return the accumulated real-space exit intensity instead of the
        diffraction-space intensity.

    Returns
    -------
    Tensor
        Real, non-negative intensity ``(P, Ny, Nx)`` -- the inelastically
        scattered diffraction pattern (corner origin) summed over all
        transitions and sites.
    """
    device = probes.device
    P, ny, nx = probes.shape
    nz = transmissions.shape[0]
    H = transition_potentials.array.to(device=device)  # (n_trans, Ny, Nx)

    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    # Put each site in the slice that potential_stack gives its depth.
    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = BeamPartition.uniform(nz).bin_index(sites[:, 2])  # potential_stack's

    if batch_size is None:
        batch_size = P

    # Slices with no atoms transmit as exactly 1 + 0j; skipping those multiplies is
    # bit-exact and removes a full pass over the (Pb, n_trans, Ny, Nx) stack each.
    unit_slices = _unit_transmission_slices(transmissions)

    out = torch.zeros((P, ny, nx), dtype=torch.float64, device=device)

    for b0 in range(0, P, batch_size):
        b1 = min(b0 + batch_size, P)
        psi = probes[b0:b1].to(H.dtype)  # elastic wave (Pb, Ny, Nx)
        acc = torch.zeros((b1 - b0, ny, nx), dtype=torch.float64, device=device)

        for i in range(nz):
            in_slice = np.nonzero(site_slice == i)[0]
            for s in in_slice:
                sy = sites[s, 0] % 1.0 * ny
                sx = sites[s, 1] % 1.0 * nx
                # Transition potentials shifted to this atom: (n_trans, Ny, Nx).
                h_site = fourier_shift(H, (sy, sx))
                # Inelastically scattered wave for every probe x transition.
                psi_n = h_site[None] * psi[:, None]  # (Pb, n_trans, Ny, Nx)

                if site_threshold > 0.0:
                    strength = psi_n.abs().pow(2).sum(dim=(-2, -1))
                    norm = H.abs().pow(2).sum(dim=(-2, -1))[None]
                    keep = strength >= site_threshold * norm
                    if not bool(keep.any()):
                        continue

                # The product is handed over with no name kept on it, so the walk
                # can write its result -- and, unless ``return_real_space``, its
                # spectrum -- back into the product's own storage: one exit stack
                # live instead of three (the product, the walk's ``out`` and
                # ``fft2`` of it), as on the hybrid route; see ``_slice_exit_wave``'s
                # ``consume``.  ``psi_n`` is dropped BEFORE the call because the
                # dense walk only frees the product once no caller name holds it.
                handoff = [psi_n]
                psi_n = None
                stack = _slice_exit_wave(
                    handoff.pop(),
                    transmissions,
                    kernel,
                    i,
                    unit_slices,
                    consume=True,
                    spectrum=not return_real_space,
                )
                # ``.abs()`` already returns a fresh real stack of the same shape
                # as the complex one it came from -- half the bytes, but on this
                # row still 576 MiB -- so squaring it out of place doubled the
                # live set for one elementwise pass.  ``pow_`` is the same kernel
                # on the same values.  The float32 sum is added straight into the
                # float64 ``acc``: the in-place add already promotes inside
                # TensorIterator, so casting first is a second full-size pass
                # (read f32, write f64) that the add then reads again, for
                # bit-identical output -- the float32->float64 cast is exact
                # either way.
                #
                # The whole tail is one Warp kernel where it is supported: the
                # eager chain writes a full-size plane per stage for the next
                # stage to read back (24 B per stack element to move 8), and the
                # fused form keeps the transition walk in registers (8 B per stack
                # element).  Bit-exact -- it reproduces ATen's FOUR-accumulator
                # reduction order, not the obvious sequential one; see
                # ``abs2_sum_acc_fused``.  Measured 3.8-4.8x on the leg.
                if is_warp(EELS_ABS2_SUM, stack, acc):
                    abs2_sum_accumulate_fused(acc, stack)
                else:
                    acc += stack.abs().pow_(2).sum(dim=1)
                # The stack is dead here, but the NAME stays bound until the
                # next site rebinds it -- and the rebinding allocates its
                # replacement first, so a dead name costs a second full-size
                # stack for the whole of the next site's slice walk.  On
                # ``r7_eels/wide/conventional`` that is 1152 MiB each, and it is
                # the single largest term in the row's 6920 MiB peak.
                stack = None

            # Advance the elastic wave by one slice.
            if not unit_slices[i]:
                psi = psi * transmissions[i]
            if i < nz - 1:
                psi = propagate(psi, kernel)

        out[b0:b1] = acc

    return out
