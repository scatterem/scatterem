"""Free-space propagation and Fourier shifts on a periodic grid.

The paraxial Fresnel propagator ``exp(-i pi lambda dz |q|^2)``, the exact
angular-spectrum form, one multislice propagation step, and the batched
Fourier-shift-theorem translation of a wave (the single-shift form lives in
:mod:`scatterem.physics.fourier`).  These are
the textbook multislice primitives (Kirkland, *Advanced Computing in Electron
Microscopy*, 2nd ed., ch. 6; Brown *et al.*, Phys. Rev. Research **1**, 033186
(2019), Sec. III), written for scatterem's EELS simulator and moved here
verbatim (2026-09-27) from ``simulation/eels/forward/multislice_eels.py`` so
that other simulators share them without reaching into a private EELS
module.  The reconstruction
:class:`~scatterem.nn.modules.propagator.Propagator` builds its kernel from the
same functions (physics-layer step 11).  The bit-exactness notes in the
docstrings (the ``sha256`` / ``torch.equal`` contracts of the EELS
benchmark targets) came with the code unchanged.

All grids are corner-origin (``fftfreq`` order).  Distances are in Angstrom.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import torch
from torch import Tensor

from scatterem.physics.periodic_sampling import PeriodicGrid, dft_sample_frequencies

from ._fresnel import angular_spectrum_kernel as angular_spectrum_kernel
from ._fresnel import propagator_phase as propagator_phase
from ._fresnel import tilt_phase as tilt_phase
from ._fresnel import transfer_phase as transfer_phase
from ._ramp_df import (
    ramp_mul_fused,
    ramp_phase_df,
    ramp_phase_df_supported,
    ramp_wq_fold_supported,
)
from ._ramp_fused import ramp_phase_fused, ramp_phase_fused_supported

__all__ = [
    "angular_spectrum_kernel",
    "fourier_shift_stack",
    "propagate",
    "propagator_kernel",
    "propagator_phase",
    "tilt_phase",
    "transfer_phase",
]


def propagator_kernel(
    gpts: Sequence[int],
    gridsize: Sequence[float],
    wavelength: float,
    dz: float,
    device=None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Fresnel free-space propagator ``exp(-i π λ Δz |q|²)`` (corner origin).

    Parameters
    ----------
    gpts : sequence of int
        Grid size `(Ny, Nx)`.
    gridsize : sequence of float
        Real-space grid extent `(Ly, Lx)`, in Angstrom.
    wavelength : float
        Electron wavelength, in Angstrom.
    dz : float
        Propagation distance, in Angstrom.
    device : torch.device, optional
        Device of the returned tensor. Default is None (current device).
    dtype : torch.dtype, optional
        Output complex dtype. Default is `torch.complex64`.

    Returns
    -------
    Tensor
        Fresnel propagator, shape `(Ny, Nx)`, corner-origin reciprocal grid.
    """
    # q^2 from the two 1-D axes on ``device``: a float64 square and one add,
    # each correctly rounded, so it is bit-equal to squaring the host mesh.
    # The host mesh (broadcast views) took ~2 ms at 512^2 on numpy's strided
    # path, plus the (Ny, Nx) float64 copy to the device.
    ay, ax = (
        torch.as_tensor(a, dtype=torch.float64, device=device)
        for a in dft_sample_frequencies(PeriodicGrid(tuple(gpts), tuple(gridsize)))
    )
    q2 = ay**2 + ax**2
    return torch.exp(-1j * pi * wavelength * dz * q2).to(dtype)


def _scaled_propagator(kernel: Tensor, n_pixels: int) -> Tensor:
    """``kernel * (1/n_pixels)``, memoised ON the ``kernel`` tensor itself.

    :func:`propagate` folds the inverse transform's ``1/(MY*MX)`` into the
    propagator (see its docstring), but the propagator is a CONSTANT of the
    geometry while ``propagate`` is called once per slice per position batch --
    270 times for a single benchmark image, each time
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


def _mul_kernel(spectrum: Tensor, kernel: Tensor) -> Tensor:
    """``spectrum * kernel``, done IN PLACE on ``spectrum`` when that is legal.

    ``spectrum`` is always the fresh output of :func:`torch.fft.fft2` -- torch
    owns it, nothing else aliases it, and it is dead the moment the inverse
    transform consumes it -- so writing the product back into it is the same
    arithmetic term for term (bit-exact) and halves the multiply's working set
    from input+output to input.

    That halving is a REGIME, not a spelling preference.  ``_exit_block_planes``
    blocks the ``r7_eels/large`` exit walk at B = 9 planes = 4.50 MiB against an
    A6000's 6.00 MiB L2 and marches each block through a 31-step chain: out of
    place the multiply's working set is 9 MiB and does not fit, in place it is
    4.5 MiB and does.  Measured at the live block shape, ``w * plane`` 23.552 us
    vs ``w.mul_(plane)`` 15.616 us = 1.5082x, while ``torch.mul(w, plane,
    out=scratch)`` into a distinct buffer recovers only 1.056x -- so it is the
    ALIASING, not the allocation, that pays.  An earlier measurement put these two
    spellings at 1.012x / 1.003x -- indistinguishable, both at roof -- on the
    ``wide/conventional`` 3.0 GB DRAM-resident stack, where neither fits in L2
    either way; that closure was right for its footprint, and
    ``_exit_block_planes`` shipped later and created a footprint it never
    covered.

    Falls back to the out-of-place product when ``kernel`` would broadcast the
    result UP past ``spectrum``'s shape (``mul_`` would raise) or when
    ``kernel`` carries a graph (``mul_`` would destroy the value autograd has
    to save in order to differentiate w.r.t. it)."""
    ks = kernel.shape
    if kernel.requires_grad or len(ks) > spectrum.ndim:
        return spectrum * kernel
    for have, want in zip(spectrum.shape[spectrum.ndim - len(ks) :], ks, strict=True):
        if want != 1 and want != have:
            return spectrum * kernel
    return spectrum.mul_(kernel)


def propagate(waves: Tensor, kernel: Tensor) -> Tensor:
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
    ``_detector_exit._conj_propagate`` normalises its own inverse transform
    with the same kernel.

    Do NOT try to fold this round trip onto ``waves`` with ``out=``.  Both
    ``torch.fft.fft2(w, out=w)`` and ``out=`` into a distinct buffer are
    bit-exact, but torch's CUDA out-variant transforms into a fresh allocation
    and *copies* into ``out``, so each one costs a full extra read+write of the
    stack: at the ``r7_eels/wide/conventional`` shape (2048 x 192 x 192
    complex128, 1152 MiB) the round trip goes 37.5 -> 41.3 ms for one ``out=``
    and 44.9 for two, and the row goes 710.6 -> 914.5 ms.

    Parameters
    ----------
    waves : Tensor
        Complex real-space waves ``(..., MY, MX)``.  A 3-D stack outside autograd
        is transformed IN PLACE and returned; otherwise a new tensor is.
    kernel : Tensor
        Fresnel propagator ``(MY, MX)`` for one step (see
        :func:`propagator_kernel`), corner origin.

    Returns
    -------
    Tensor
        The propagated waves, same shape and dtype as ``waves``.
    """
    MY, MX = int(waves.shape[-2]), int(waves.shape[-1])
    kernel = _scaled_propagator(kernel, MY * MX)
    inplace = waves.ndim == 3 and not (torch.is_grad_enabled() and waves.requires_grad)
    if inplace:
        chunk = max(1, int(6e8 // max(1, MY * MX)))
        if waves.shape[0] > chunk:
            for i in range(0, waves.shape[0], chunk):
                waves[i : i + chunk] = torch.fft.ifft2(
                    _mul_kernel(
                        torch.fft.fft2(waves[i : i + chunk], dim=(-2, -1)), kernel
                    ),
                    dim=(-2, -1),
                    norm="forward",
                )
            return waves
    return torch.fft.ifft2(
        _mul_kernel(torch.fft.fft2(waves, dim=(-2, -1)), kernel),
        dim=(-2, -1),
        norm="forward",
    )


_FFTFREQ_F64: dict = {}


def _fftfreq_f64(n: int, device) -> Tensor:
    """``torch.fft.fftfreq(n, dtype=float64)``, memoised per ``(n, device)``.

    The sample frequencies are a pure function of the length and depend on
    nothing else, but ``fourier_shift_stack`` is called once per scan batch --
    32 times for a single benchmark image -- and rebuilding them
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


# Kept beside physics.fourier.fourier_shift: fused Warp ramp + r7 sha256 contracts.
def fourier_shift_stack(
    waves: Tensor, shifts: Tensor, phase_bytes: int = 1 << 27
) -> Tensor:
    """``physics.fourier.fourier_shift`` for many shifts -> ``(P, *waves.shape)``.

    Bitwise-identical to ``torch.stack([fourier_shift(waves, s) for s in shifts])``
    and much cheaper, for two reasons that have nothing to do with vectorised
    arithmetic:

    * ``fourier_shift`` reads each entry of a host shift with ``float()``, so a
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
    shape and this function is called once per scan batch (32 times per image on
    the benchmark), another 1.9 %.  ``wq = fft2(waves)`` is constant per
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
    ``r7_eels`` target is gated on.

    On CUDA at complex64 the whole ramp is one Warp kernel instead
    (``_ramp_fused.ramp_phase_fused``), which is bit-exact and **1.47x on
    a probe-build benchmark**.  The four eager passes above write 72
    bytes per element to produce 8, but that traffic is NOT what the kernel is
    buying: ``polar`` is bound by double-precision ``sin``/``cos``, not by memory
    (measured, ``cos(u)`` alone 0.1556 ms + ``sin(u)`` alone 0.1546 == ``polar``'s
    0.300), and a fused kernel that merely deletes the other three passes is worth
    only 1.029x.  What pays is that ``sincos()`` performs ONE Payne-Hanek argument
    reduction for the pair where two separate calls perform two -- and at the
    angles a scan uses (``|u|`` up to 804, the shifts being hundreds of pixels)
    that reduction is the expensive half.  See ``_ramp_fused.py``, whose docstring
    names the three spellings that keep it bit-exact and the reason the launch
    needs an explicit ``stream=``.  The kernel that runs is the certified
    float-float one in ``_ramp_df.py``, with the same bits.  On
    the single-chunk power-of-two route with 2-D ``waves``, that kernel also
    applies ``wq *`` before its store, again with the same bits.

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
    ``r7_eels`` is gated on.  Verified as bytes over the real
    scan: 0 of 134 217 728 int32 words differ.  (The one regime it would not be
    exact in is ``|fft2(waves)| < 2^-126 * N`` ~ 8e-34, where the scaled plane goes
    subnormal; a normalised wave function is 24 orders of magnitude clear of that --
    ``min|wq|`` is 6.3e-10 on this row -- and testing for it would cost a device
    sync per call, i.e. more than the pass being removed.)

    Parameters
    ----------
    waves : Tensor
        Complex real-space waves ``(..., ny, nx)``.
    shifts : Tensor
        ``(P, 2)`` shifts ``(sy, sx)`` in pixels, on any device.
    phase_bytes : int, optional
        Byte budget of one complex128 ramp chunk (default ``1 << 27``).

    Returns
    -------
    Tensor
        ``(P, *waves.shape)``: ``waves`` shifted by each row of ``shifts``.
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
    ramp = ramp_phase_df if ramp_phase_df_supported(waves, shifts) else ramp_phase_fused
    if ramp_wq_fold_supported(waves, shifts, fold):
        # the DF ramp kernel applies ``wq *`` before its store
        # (same bits) and deletes the separate multiply pass.
        prod = ramp_mul_fused(ky1, kx1, sh[:, 0], sh[:, 1], -2.0 * pi, wq)
        return torch.fft.ifft2(prod, dim=(-2, -1), norm="forward")
    for a in range(0, P, chunk):
        b = min(a + chunk, P)
        if fused:
            phase = ramp(
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
