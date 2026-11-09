"""S-matrix column preparation of ``PrismScanOp``.

Private.  Between the multislice build and the per-probe reduction every
S-matrix column is *prepared*: propagated to the reference plane, de-tilted,
and stripped of the free-space phase it still carries from there to the exit
plane.  What is left is only what the specimen did to that beam, which is the
one quantity partitioned PRISM may interpolate across beams.  The
stripped phase is known in closed form for every beam, so the reduction puts it
back analytically per TARGET beam (:func:`vacuum_phase`) instead of
interpolating it.

THE INVARIANT THIS BUYS
-----------------------
In vacuum a prepared column is the same constant for every beam, on every grid
and for every reference plane, so any interpolation whose weights sum to one
reproduces the probe exactly.  Before this module PRISM kept the phase
``exp(-i pi lambda (z - z_ref) |h|^2)`` in its columns and interpolated it:
exact only with the reference plane at the entrance, 13 % wrong (r = 4,
30 mrad, 40 A) with it at mid-thickness -- and which of the two a vacuum run
got depended on whether the FFT size left 1e-14 of rounding noise in ``T - 1``.

One implementation, because there were several and they
disagreed: in place and beam-chunked (the transient is one ``(chunk, Y, X)``
buffer), with separable tilt ramps reduced modulo one period in float64.
"""

from __future__ import annotations

import torch
from torch import Tensor


def beam_chunk(max_bytes: int, shape: tuple[int, int]) -> int:
    """Number of ``(Y, X)`` complex64 planes that fit in ``max_bytes`` (>= 1)."""
    return max(1, int(max_bytes) // max(1, int(shape[0]) * int(shape[1]) * 8))


def reference_slices(rule: float | str, T: Tensor) -> float:
    """Slices from the EXIT plane back to the reference plane.

    Parameters
    ----------
    rule : float or "centroid"
        A float is the fraction of the propagated thickness (0 = exit plane,
        0.5 = mid-specimen, 1 = entrance).  ``"centroid"`` is the plane that
        minimises ``sum_z w_z (z_ref - z)^2`` with ``w_z = sum |T_z - 1|`` the
        scattering strength of slice ``z``: the residual a prepared column
        keeps is the beam walk from the scattering depth to the reference
        plane, and this is its least-squares minimiser.
    T : Tensor
        ``(NZ, Y, X)`` complex transmission.

    Returns
    -------
    float
        ``n_half`` in ``[0, NZ - 1]``.

    Notes
    -----
    The centroid is written relative to the mid-plane,
    ``z_ref = z_mid + sum w_z (z - z_mid) / sum w_z``.  With ``sum w_z == 0``
    (exact vacuum) every plane minimises the objective and the minimum-norm
    (Moore-Penrose) solution, offset 0, is taken.  That is a definition for an
    exactly singular problem, not a tolerance: a vacuum whose transmission
    carries rounding noise has the SAME weight in every slice and lands on the
    mid-plane as well, so the plane no longer depends on the FFT size.  No
    normalised centroid is continuous in ``T`` at ``T = 1``; the RESULT is,
    because a prepared column's dependence on the plane vanishes with the
    scattering strength.
    """
    span = int(T.shape[0]) - 1
    if rule != "centroid":
        return float(rule) * span
    # float64 and slice by slice, so |T - 1| never exists as a full (NZ, Y, X)
    # copy (a ~30 GB transient for a 10M-atom tip).
    w = torch.stack(
        [(T[z] - 1.0).abs().sum(dtype=torch.float64) for z in range(span + 1)]
    )
    depth = torch.arange(span + 1, device=w.device, dtype=torch.float64) - 0.5 * span
    total = float(w.sum())
    offset = float((depth * w).sum()) / total if total > 0.0 else 0.0
    return 0.5 * span - offset


def plane_transfer(phase: Tensor, n: float) -> Tensor:
    """``exp(1j * n * phase)``: the pure-phase transfer over ``n`` slices.

    Evaluated in float64 and rounded once, so the transfer there
    (``n = -n_half``) and back (``n = +n_half``) are exact conjugates and the
    round trip cancels to float32 rounding.  Never carries the antialias
    aperture: it moves a reference plane, it is not a physical propagation.

    :param phase: real per-slice phase
        (:meth:`~scatterem.nn.modules.Propagator.fresnel_phase`).
    :param n: slice count; fractional and negative values are fine because the
        phase is unwrapped.
    :returns: complex64, the shape of ``phase``.
    """
    return torch.exp(1j * float(n) * phase.to(torch.float64)).to(torch.complex64)


def separable_tilt_ramps(
    beams: Tensor, shape: tuple[int, int], device
) -> tuple[Tensor, Tensor]:
    """``exp(-2 pi i h_y y / Y)`` and ``exp(-2 pi i h_x x / X)`` per beam.

    Their outer product is the de-tilt ramp, without the ``(B, Y, X)`` array a
    direct evaluation materialises.  The phase is reduced modulo one period in
    float64 before the exponential: a float32 ``h y / Y`` reaches ~10^2 for the
    outer beams on a 528 grid and carries a ~1e-5 rad rounding error there,
    which would leave a residual tilt on the de-tilted column.

    :returns: ``(ramp_y, ramp_x)`` of shapes ``(B, Y)`` and ``(B, X)``, complex64.
    """
    Y, X = int(shape[0]), int(shape[1])

    def ramp(h: Tensor, n: int) -> Tensor:
        idx = torch.arange(n, device=device, dtype=torch.float64)
        frac = torch.remainder(h.to(torch.float64)[:, None] * idx[None, :], n) / n
        return torch.exp(-2j * torch.pi * frac).to(torch.complex64)

    return ramp(beams[:, 0], Y), ramp(beams[:, 1], X)


def vacuum_phase(beams: Tensor, phase: Tensor, n_half: float, n_slices: int) -> Tensor:
    """Free-space phase of each beam from the reference plane to the exit.

    ``exp(+i (NZ - 1 - n_half) phase[h_b])``: what :func:`prepare_columns_`
    strips from a column built on beam ``h_b``.  The reduction multiplies it
    back onto the probe coefficient of every TARGET beam.

    :param beams: ``(B, 2)`` signed integer beam coordinates.
    :param phase: ``(Y, X)`` per-slice propagator phase
        (:meth:`~scatterem.nn.modules.Propagator.fresnel_phase`).
    :returns: ``(B,)`` complex64.
    """
    Y, X = phase.shape
    at_beam = phase[beams[:, 0].long() % Y, beams[:, 1].long() % X].to(torch.float64)
    return torch.exp(1j * (n_slices - 1 - n_half) * at_beam).to(torch.complex64)


def prepare_columns_(
    S: Tensor,
    beams: Tensor,
    phase: Tensor,
    n_half: float,
    n_slices: int,
    max_bytes: int,
    ring: Tensor | None = None,
) -> float | None:
    """Prepare the columns in place, one beam chunk at a time.

    Per column ``c`` built on beam ``h_c`` this computes

        S_c <- IFFT(FFT(S_c) e^{-i n_half phase}) e^{-2 pi i h_c . r}
               e^{-i (NZ - 1 - n_half) phase[h_c]}

    i.e. reference plane, de-tilt, vacuum phase.  The FFT round trip writes
    back into the column's own storage and the de-tilt is two broadcast
    multiplies with separable ramps, so the transient is ONE ``(chunk, Y, X)``
    complex64 buffer.  A zero-distance propagation is the identity and is not
    transformed at all (unless ``ring`` needs the spectrum).

    :param S: ``(Bp, Y, X)`` complex64 or complex32, modified in place.  Must
        not require grad (``out=`` FFTs and in-place writes are invisible to
        autograd); use :func:`prepare_columns_autograd` there.  A complex32
        stack is processed through a complex64 copy of one chunk.
    :param beams: ``(Bp, 2)`` signed integer beam coordinates of the columns.
    :param phase: ``(Y, X)`` float per-slice propagator phase.
    :param n_half: slices from the exit plane back to the reference plane.
    :param n_slices: ``NZ`` of the build (the exit wave made ``NZ - 1`` steps).
    :param max_bytes: bound on the transient; sets the beam chunk.
    :param ring: optional boolean ``(Y, X)`` lab-frame mask; when given the
        share of the columns' spectral power inside it is returned (from the
        FFT this pass takes anyway).
    :returns: that share, or ``None`` without a ring.
    """
    Bp, Y, X = S.shape
    dev = S.device
    transform = bool(n_half) or ring is not None
    back = plane_transfer(phase, -n_half)
    strip = vacuum_phase(beams, phase, n_half, n_slices).conj()  # (Bp,)
    ramp_y, ramp_x = separable_tilt_ramps(beams, (Y, X), dev)
    cols = beam_chunk(max_bytes, (Y, X))
    edge = total = torch.zeros((), dtype=torch.float64, device=dev)
    for i in range(0, Bp, cols):
        j = min(i + cols, Bp)
        own = S.dtype == torch.complex64
        blk = S[i:j] if own else S[i:j].to(torch.complex64)
        if transform:
            tmp = torch.fft.fft2(blk)
            if ring is not None:
                edge = edge + torch.linalg.vector_norm(tmp[:, ring]).double() ** 2
                total = total + torch.linalg.vector_norm(tmp).double() ** 2
            if n_half:
                tmp.mul_(back)
                torch.fft.ifft2(tmp, out=blk)
            del tmp
        # de-tilt (separable) and vacuum phase, folded into the row ramp
        blk.mul_((ramp_y[i:j] * strip[i:j, None])[:, :, None])
        blk.mul_(ramp_x[i:j, None, :])
        if not own:
            S[i:j] = blk.to(S.dtype)
    return None if ring is None else float(edge / total.clamp(min=1e-300))


def prepare_columns_autograd(
    S: Tensor, beams: Tensor, phase: Tensor, n_half: float, n_slices: int
) -> Tensor:
    """Out-of-place twin of :func:`prepare_columns_` for a graph-tracked S.

    Same algebra, materialised: ``out=`` FFTs and in-place writes are not
    allowed on a tensor autograd is tracking.
    """
    Y, X = S.shape[-2:]
    Sr = S.to(torch.complex64)  # a complex32 build has no general FFT plan
    if n_half:
        Sr = torch.fft.ifft2(torch.fft.fft2(Sr) * plane_transfer(phase, -n_half))
    ramp_y, ramp_x = separable_tilt_ramps(beams, (Y, X), S.device)
    strip = vacuum_phase(beams, phase, n_half, n_slices).conj()
    out = Sr * (ramp_y * strip[:, None])[:, :, None] * ramp_x[:, None, :]
    return out.to(S.dtype)


__all__ = [
    "beam_chunk",
    "plane_transfer",
    "prepare_columns_",
    "prepare_columns_autograd",
    "reference_slices",
    "separable_tilt_ramps",
    "vacuum_phase",
]
