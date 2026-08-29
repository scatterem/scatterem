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
"""

from __future__ import annotations

from math import pi
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .._grid import q_space_array
from .ramp_fused import ramp_phase_fused, ramp_phase_fused_supported
from .transition_potentials import TransitionPotentials

__all__ = ["propagator_kernel", "transition_potential_multislice"]


def propagator_kernel(
    gpts: Sequence[int],
    gridsize: Sequence[float],
    wavelength: float,
    dz: float,
    device=None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Fresnel free-space propagator ``exp(-i π λ Δz |q|²)`` (corner origin)."""
    qy, qx = q_space_array(gpts, gridsize)
    q2 = torch.as_tensor(qy**2 + qx**2, dtype=torch.float64, device=device)
    return torch.exp(-1j * pi * wavelength * dz * q2).to(dtype)


def _scaled_propagator(kernel: Tensor, n_pixels: int) -> Tensor:
    """``kernel * (1/n_pixels)``, memoised ON the ``kernel`` tensor itself.

    :func:`_propagate` folds the inverse transform's ``1/(MY*MX)`` into the
    propagator (see its docstring), but the propagator is a CONSTANT of the
    geometry while ``_propagate`` is called once per slice per position batch --
    270 times for a single ``r7f_tds_adf/fine/grad_forward`` image, each time
    recomputing the identical plane.  That row is host-bound (31.4 % of its wall
    is GPU idle), and this multiply is one full ATen dispatch of the ~10 the
    slice loop issues, measured at 2.51 ms of a 43.5 ms row plus 0.57 ms of
    device time.

    Bit-exactness is by construction: the memo returns the result of the *same*
    multiply on the *same* operands, so nothing about the arithmetic changes --
    it simply is not redone.

    The memo rides on the kernel tensor rather than in a module-level dict so
    that its lifetime is exactly the kernel's: no id-keyed entry can be aliased
    by a later allocation reusing the address, and no plane outlives the
    propagator it belongs to.  It is skipped entirely when ``kernel`` carries a
    graph -- a cached non-leaf would be backpropagated through more than once.
    """
    if kernel.requires_grad:
        return kernel * (1.0 / n_pixels)
    hit = getattr(kernel, "_scatterem_prop_scaled", None)
    if hit is not None and hit[0] == n_pixels:
        return hit[1]
    scaled = kernel * (1.0 / n_pixels)
    try:
        kernel._scatterem_prop_scaled = (n_pixels, scaled)
    except AttributeError:  # a Tensor subclass that refuses attributes
        pass
    return scaled


def _propagate(waves: Tensor, kernel: Tensor) -> Tensor:
    """One Fresnel propagation step (FFT · kernel · IFFT).

    For a large beam stack (the un-tiled 20 nm S-walk: (Bp, MY, MX) ~27 GB) the
    naive round-trip holds ``waves`` + the FFT result + the IFFT result (~3x27
    GB) -> OOM on 80 GB. Beam-chunk the round-trip IN PLACE so only one chunk's
    FFT transient is live. The 2-D FFT is independent per beam
    (``fft2(waves)[b] == fft2(waves[b])``), so the chunked in-place result is
    byte-identical to the single call. In-place is gated to the no-grad
    (forward-sim) path; recon keeps the out-of-place round-trip.

    The inverse transform runs UNNORMALISED (``norm="forward"``) with the
    ``1/(MY*MX)`` folded into the propagator instead: torch applies the default
    ``norm="backward"`` as a separate full-tensor rescale of the (B, MY, MX)
    result, whereas riding on the kernel costs one pass over a single (MY, MX)
    plane -- and, via :func:`_scaled_propagator`'s memo, only on the first call
    for a given propagator rather than on every slice.  This step is
    memory-bandwidth-bound (the broadcast multiply alone
    runs at 87.6 % of this card's peak), so that removed pass is ~16 % of it.
    Deliberately NOT folded into ``propagator_kernel``: that is public API, and
    ``prism_eels_image._conj_propagate`` normalises its own inverse transform
    with the same kernel.

    Do NOT try to fold this round trip onto ``waves`` with ``out=``.  Both
    ``torch.fft.fft2(w, out=w)`` and ``out=`` into a distinct buffer are
    bit-exact, but torch's CUDA out-variant transforms into a fresh allocation
    and *copies* into ``out``, so each one costs a full extra read+write of the
    stack: at the ``r7_eels/wide/conventional`` shape (2048 x 192 x 192
    complex128, 1152 MiB) the round trip goes 37.5 -> 41.3 ms for one ``out=``
    and 44.9 for two, and the row goes 710.6 -> 914.5 ms.  See
    ``docs/perf-lab/reports/2026-08-24-d284-*``."""
    MY, MX = int(waves.shape[-2]), int(waves.shape[-1])
    kernel = _scaled_propagator(kernel, MY * MX)
    inplace = waves.ndim == 3 and not (
        torch.is_grad_enabled() and waves.requires_grad
    )
    if inplace:
        chunk = max(1, int(6e8 // max(1, MY * MX)))
        if waves.shape[0] > chunk:
            for i in range(0, waves.shape[0], chunk):
                waves[i : i + chunk] = torch.fft.ifft2(
                    torch.fft.fft2(waves[i : i + chunk], dim=(-2, -1)) * kernel,
                    dim=(-2, -1),
                    norm="forward",
                )
            return waves
    return torch.fft.ifft2(
        torch.fft.fft2(waves, dim=(-2, -1)) * kernel, dim=(-2, -1), norm="forward"
    )


_FFTFREQ_F64: dict = {}


def _fftfreq_f64(n: int, device) -> Tensor:
    """``torch.fft.fftfreq(n, dtype=float64)``, memoised per ``(n, device)``.

    The sample frequencies are a pure function of the length and depend on
    nothing else, but ``_fourier_shift_stack`` is called once per scan batch --
    32 times for a single ``r7f_tds_adf/medium`` image -- and rebuilding them
    costs two kernel launches and a host ``arange`` every time, for two tensors
    of a few hundred float64s.  That is 1.9 % of the ``probe_build`` row.

    The cached tensor is only ever read (viewed, then multiplied), so handing
    the same object to every caller is safe; do not mutate it in place.
    """
    key = (int(n), str(device))
    v = _FFTFREQ_F64.get(key)
    if v is None:
        v = torch.fft.fftfreq(n, device=device, dtype=torch.float64)
        _FFTFREQ_F64[key] = v
    return v


def _fourier_shift(waves: Tensor, shift_yx: Tensor) -> Tensor:
    """Shift ``waves`` by ``shift_yx`` pixels via the Fourier shift theorem.

    ``shift_yx`` is ``(sy, sx)`` in fractional pixels.  Sub-pixel shifts are
    handled exactly.
    """
    ny, nx = waves.shape[-2:]
    # Build the ramp in float64 then cast to the wave's complex dtype, so the
    # result never depends on the global default dtype.
    ky = torch.fft.fftfreq(ny, device=waves.device, dtype=torch.float64).view(ny, 1)
    kx = torch.fft.fftfreq(nx, device=waves.device, dtype=torch.float64).view(1, nx)
    phase = torch.exp(
        -2j * pi * (ky * float(shift_yx[0]) + kx * float(shift_yx[1]))
    ).to(waves.dtype)
    return torch.fft.ifft2(torch.fft.fft2(waves, dim=(-2, -1)) * phase, dim=(-2, -1))


def _fourier_shift_stack(
    waves: Tensor, shifts: Tensor, phase_bytes: int = 1 << 27
) -> Tensor:
    """``_fourier_shift`` for many shifts at once -> ``(P, *waves.shape)``.

    Bitwise-identical to ``torch.stack([_fourier_shift(waves, s) for s in shifts])``
    and much cheaper, for two reasons that have nothing to do with vectorised
    arithmetic:

    * ``_fourier_shift`` reads ``float(shift_yx[0])`` / ``float(shift_yx[1])``, so a
      loop over a scan on the GPU pays **two device syncs per position** -- 2P
      pipeline stalls before any physics happens.  Here the shifts stay on the
      device and there are none.
    * the loop recomputes ``fft2(waves)`` once per position; it is the same tensor
      every time, so it is hoisted.

    Bitwise equality is why this is safe to drop in: the ramp is the same float64
    elementwise arithmetic (a broadcast multiply against ``(P,1,1)`` shifts gives the
    same products as a scalar multiply per position), and cuFFT's batched 2-D
    transform is bitwise independent per batch element -- both verified over the
    scan geometries this path uses, not assumed.

    The ramp is complex128 at ``(chunk, ny, nx)``, so it is chunked to ``phase_bytes``
    rather than built for all P at once; the output is preallocated instead of
    ``torch.stack``-ing a list, which halves the peak the loop form needed.  The
    transform then writes *into* that preallocation via ``out=``: assigning its
    result with ``out[a:b] = ifft2(...)`` allocates the whole chunk a second time
    and copies it, which is 6.9 % of this call at the shapes a scan uses.

    ``fftfreq`` comes from ``_fftfreq_f64``'s memo because it is a constant of the
    shape and this function is called once per scan batch (32 times per image at
    ``r7f_tds_adf/medium``), another 1.9 %.  ``wq = fft2(waves)`` is constant per
    *probe* across those same calls, but hoisting it needs a cache keyed on tensor
    identity and lifetime, and it is measured at only 1.2 % -- left alone
    deliberately.

    It is written as ``polar(1, u)`` rather than ``exp(-2j*pi*t)`` because the two
    are the same bits and the second is strictly more work: the exponent's real part
    is ``-0.0`` for every element, so a complex ``exp`` spends one of its three
    float64 transcendentals computing ``exp(-0.0) == 1.0`` and then multiplying by
    it, and the ``-2j*pi*t`` operand has to be materialised as a *complex128* plane
    (32 B/element) to say so.  ``polar`` takes the float64 angle directly and calls
    only ``cos``/``sin`` -- which is what ``exp`` reduces to here -- so the identity
    is exact, not approximate: verified ``torch.equal`` over the real scan, integer
    +-300 px, random fractional, zero and 1e-9 shifts, into complex64 and complex128
    (0 differing float32 words of 4.2 M each).  Do NOT go further and factor the ramp
    as ``exp(-2j*pi*ky*sy) * exp(-2j*pi*kx*sx)``: separability is *nearly* exact and
    moves 1.3e5 float32 words on this scan, which breaks the sha256 contracts the
    ``r7f_tds_adf`` and ``r7_eels`` targets are gated on.

    On CUDA at complex64 the whole ramp is one Warp kernel instead
    (``ramp_fused.ramp_phase_fused``), which is bit-exact and **1.47x on
    ``r7f_tds_adf/medium/probe_build``**.  The four eager passes above write 72
    bytes per element to produce 8, but that traffic is NOT what the kernel is
    buying: ``polar`` is bound by double-precision ``sin``/``cos``, not by memory
    (measured, ``cos(u)`` alone 0.1556 ms + ``sin(u)`` alone 0.1546 == ``polar``'s
    0.300), and a fused kernel that merely deletes the other three passes is worth
    only 1.029x.  What pays is that ``sincos()`` performs ONE Payne-Hanek argument
    reduction for the pair where two separate calls perform two -- and at the
    angles a scan uses (``|u|`` up to 804, the shifts being hundreds of pixels)
    that reduction is the expensive half.  See ``ramp_fused.py``, whose docstring
    names the three spellings that keep it bit-exact and the reason the launch
    needs an explicit ``stream=``.

    ``ifft2``'s own ``1/N`` normalisation is torch's, not cuFFT's -- the C2C plan is
    unnormalised, so ``norm="backward"`` (the default) is paid as a *separate
    full-size pass* over the ``(chunk, ny, nx)`` output.  On this row that pass is
    1.63 ms = **11.8 % of the whole device time** and it exists only to multiply by
    a constant.  When ``ny*nx`` is an exact power of two, ``1/N`` is an exponent
    decrement, so it commutes bit-for-bit with every multiply and add in the
    transform and can ride on ``wq`` -- ONE small ``(ny, nx)`` plane per call --
    leaving the inverse unnormalised.  Two things are load-bearing and neither
    works alone (both measured; 0.995x and 1.001x respectively):

    * the fold needs the ``out=`` to go, because torch *fuses* the normalisation
      into the out-variant's copy, so ``norm="forward"`` on its own merely turns a
      fused scalar-multiply into a plain ``copy_`` of the same 2 x 16 MiB;
    * dropping ``out=`` needs the fold, because without it the normalisation pass
      is still there, just writing into a fresh allocation instead of a slice.

    Together they are **1.124x on this row** (and drop its peak 561 -> 545 MiB, the
    preallocation no longer existing), re-measured over six alternating
    separately-launched arms; the ``small`` and ``fine`` rows are a sub-bar 1.7 %
    and 3.8 %, same sign, and carry no claim.
    Only the single-chunk case takes it -- which is every
    caller in this package at every registered size, ``chunk`` being 128 against a
    scan batch of 32 -- because a multi-chunk call still needs the preallocation
    and the fold buys nothing there.  The power-of-two test is an *exactness*
    condition, not a tuned one: at a non-power-of-two ``N`` the pre-scaling rounds
    where the post-scaling did not, which would move the sha256 contracts
    ``r7f_tds_adf`` and ``r7_eels`` are gated on.  Verified as bytes over the real
    scan: 0 of 134 217 728 int32 words differ.  (The one regime it would not be
    exact in is ``|fft2(waves)| < 2^-126 * N`` ~ 8e-34, where the scaled plane goes
    subnormal; a normalised wave function is 24 orders of magnitude clear of that --
    ``min|wq|`` is 6.3e-10 on this row -- and testing for it would cost a device
    sync per call, i.e. more than the pass being removed.)
    """
    ny, nx = waves.shape[-2:]
    ky1 = _fftfreq_f64(ny, waves.device)
    kx1 = _fftfreq_f64(nx, waves.device)
    ky = ky1.view(ny, 1)
    kx = kx1.view(1, nx)

    sh = shifts.to(device=waves.device, dtype=torch.float64).reshape(-1, 2)
    P = sh.shape[0]
    mid = (1,) * (waves.ndim - 2)  # keep leading wave dims broadcastable
    chunk = max(1, int(phase_bytes // (16 * ny * nx)))
    n_el = ny * nx
    # ``0 < P`` so an empty scan still returns the ``(0, *waves.shape)`` tensor
    # rather than falling out of the loop with nothing to return.
    fold = 0 < P <= chunk and (n_el & (n_el - 1)) == 0
    wq = torch.fft.fft2(waves, dim=(-2, -1), norm="forward" if fold else "backward")
    out = None
    if not fold:
        out = torch.empty(
            (P,) + tuple(waves.shape), dtype=waves.dtype, device=waves.device
        )
    fused = ramp_phase_fused_supported(waves, shifts)
    for a in range(0, P, chunk):
        b = min(a + chunk, P)
        if fused:
            phase = ramp_phase_fused(
                ky1, kx1, sh[a:b, 0], sh[a:b, 1], -2.0 * pi, (b - a, *mid, ny, nx)
            )
        else:
            sy = sh[a:b, 0].view(-1, *mid, 1, 1)
            sx = sh[a:b, 1].view(-1, *mid, 1, 1)
            u = (-2.0 * pi) * (ky * sy + kx * sx)
            one = torch.ones((), dtype=u.dtype, device=u.device).expand(u.shape)
            phase = torch.polar(one, u).to(waves.dtype)
        if fold:
            return torch.fft.ifft2(wq * phase, dim=(-2, -1), norm="forward")
        torch.fft.ifft2(wq * phase, dim=(-2, -1), out=out[a:b])
    return out


def _unit_transmission_slices(transmissions: Tensor) -> list:
    """Which slices have a transmission function that is *exactly* ``1 + 0j``.

    A slice holding no atoms has zero projected potential, so its transmission
    function is ``exp(0) == 1`` in every bit -- measured, not assumed: at all four
    registered ``r7_eels`` sizes ``max|T[j] - 1|`` is exactly ``0.0`` on 12 of 16
    slices (a 4-atom column in a 20 A cell).  Multiplying a wave by that plane is
    the identity for every finite input, so the multiply can be skipped and the
    result is **bit-identical**, not merely within tolerance.

    Worth skipping because it is not a cheap multiply: the inelastic stack is
    ``(P, n_trans, Ny, Nx)`` (1.2 GB at the ``wide`` config) and one pass over it
    runs at 88.5 % of this card's peak bandwidth, i.e. the multiply is *at the
    memory roof* and costs a full 3.55 ms whatever it multiplies by.

    Computed once per call as a single batched reduction so the whole test is one
    device sync (~0.1 ms) rather than one per slice, and it is only ever a
    saving: a dense crystal has no unit slice, and then this costs one pass over
    ``transmissions`` (4.7 MB) and nothing else.
    """
    flags = (transmissions == 1).flatten(1).all(dim=1)
    return [bool(v) for v in flags.tolist()]


def _slice_exit_wave(
    waves: Tensor,
    transmissions: Tensor,
    kernel: Tensor,
    start: int,
    unit_slices: Optional[Sequence[bool]] = None,
) -> Tensor:
    """Propagate ``waves`` through slices ``start..NZ-1`` (transmit + propagate).

    ``unit_slices[j]`` marks a slice whose transmission function is exactly
    ``1 + 0j`` (see :func:`_unit_transmission_slices`); its multiply is the
    identity and is skipped.  The propagation is *not* skipped -- composing two
    Fresnel kernels is not the same float as applying them either side of an
    FFT round trip, so that would not be bit-exact.

    The transmission multiply is a read-modify-write on a buffer this function
    already owns, from the second slice onward: ``_propagate`` hands back a
    freshly allocated stack, so ``out * transmissions[j]`` was allocating a
    second full-size copy of it and then dropping the first.  On
    ``r7_eels/wide/conventional`` that stack is 1152 MiB.  ``mul_`` is the same
    kernel with the same rounding, one pass cheaper, and it never touches
    ``waves`` itself -- ``owned`` starts False, so the first multiply is still
    the out-of-place one that allocates.

    The FFT round trip inside ``_propagate`` cannot be folded the same way; see
    its docstring for the measurement.
    """
    nz = transmissions.shape[0]
    out = waves
    owned = False
    # A promoting multiply cannot be done in place; decided once, not per slice.
    exact_inplace = torch.result_type(waves, transmissions) == waves.dtype
    for j in range(start, nz):
        if unit_slices is None or not unit_slices[j]:
            if owned and exact_inplace:
                out.mul_(transmissions[j])
            else:
                out = out * transmissions[j]
                owned = True
        if j < nz - 1:
            stepped = _propagate(out, kernel)
            # `_propagate` returns its own argument on the large-3-D in-place
            # branch, where the buffer is NOT ours; identity is the exact test.
            owned = owned or stepped is not out
            out = stepped
    return out


def transition_potential_multislice(
    probes: Tensor,
    transmissions: Tensor,
    transition_potentials: TransitionPotentials,
    sites: np.ndarray,
    *,
    wavelength: float,
    gridsize: Sequence[float],
    slice_distance: float,
    site_threshold: float = 0.0,
    batch_size: Optional[int] = None,
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
    gridsize : (2,) float
        Field-of-view size ``(Ly, Lx)`` [Å].
    slice_distance : float
        Inter-slice propagation distance Δz [Å].
    site_threshold : float
        Skip an inelastic event when the integrated scattered intensity is below
        this fraction of the transition-potential norm (cheap speed-up).
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
    n_trans = H.shape[0]

    kernel = propagator_kernel(
        (ny, nx), gridsize, wavelength, slice_distance, device=device, dtype=H.dtype
    )

    # Assign each site to a depth slice from its fractional z coordinate.
    sites = np.atleast_2d(np.asarray(sites, dtype=np.float64))
    site_slice = np.clip((sites[:, 2] % 1.0 * nz).astype(int), 0, nz - 1)

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
                h_site = _fourier_shift(H, (sy, sx))
                # Inelastically scattered wave for every probe x transition.
                psi_n = h_site[None] * psi[:, None]  # (Pb, n_trans, Ny, Nx)

                if site_threshold > 0.0:
                    strength = psi_n.abs().pow(2).sum(dim=(-2, -1))
                    norm = H.abs().pow(2).sum(dim=(-2, -1))[None]
                    keep = strength >= site_threshold * norm
                    if not bool(keep.any()):
                        continue

                psi_n = _slice_exit_wave(
                    psi_n, transmissions, kernel, i, unit_slices
                )
                # ``.abs()`` already returns a fresh real stack of the same shape
                # as the complex one it came from -- half the bytes, but on this
                # row still 576 MiB -- so squaring it out of place doubled the
                # live set for one elementwise pass.  ``pow_`` is the same kernel
                # on the same values.
                if return_real_space:
                    acc += psi_n.abs().pow_(2).sum(dim=1).to(torch.float64)
                else:
                    diff = torch.fft.fft2(psi_n, dim=(-2, -1))
                    acc += diff.abs().pow_(2).sum(dim=1).to(torch.float64)
                    diff = None
                # Both stacks are dead here, but the NAMES stay bound until the
                # next site rebinds them -- and the rebinding allocates its
                # replacement first, so a dead name costs a second full-size
                # stack for the whole of the next site's slice walk.  On
                # ``r7_eels/wide/conventional`` that is 1152 MiB each, and it is
                # the single largest term in the row's 6920 MiB peak.
                psi_n = None

            # Advance the elastic wave by one slice.
            if not unit_slices[i]:
                psi = psi * transmissions[i]
            if i < nz - 1:
                psi = _propagate(psi, kernel)

        out[b0:b1] = acc

    return out
