"""The C2R ``reduce="sum"`` body of ``_direct_ptychography_kouter``.

The shipped ``reduce="sum"`` branch reads ``ifft2(corrected).imag`` only, and for
ANY complex ``C`` ``Im(ifft2(C)) = irfft2(H)`` with
``H(q) = -i (C(q) - conj C(-q)) / 2`` Hermitian, so the correction kernel stores
``H`` on the half grid (``correct_aberrations_kouter_half``) and the transform
becomes a C2R.  No physics premise: the real part is unread.  Same per-k
transform, same k-sum order, same ``scale`` fold.  NOT bit-exact (~3 ulp, the
C2R rounds differently; 3.16e-07 against complex128 where the C2C route is
2.26e-07).  Measured at the ``r14_ffstem/gd2o3`` chunk ``(33, 1024, 1024)``: the
transform leg 63.573 -> 45.546 ms per 31 calls, 1.2621x on
``r14_ffstem/gd2o3/ptycho``.  At an odd ``W`` (245) it is 1.0448x and
``kouter_half_supported`` declines there.  The half buffer costs
more own allocation (277 -> 401.5 MiB at that row) for a lower row peak -- a
labelled trade.  Its kill switch ``SCATTEREM_KOUTER_HALF_C2R`` was deleted.

Where the half store declines, the two-for-one packed store
(:func:`_kouter_sum_packed`) packs two planes into one C2C instead: 1.4635x on
``r15_direct_ptychography/medium/direct_ptycho`` (W = 245), with the row's own
peak 82.9 -> 42.1 MiB.  Also NOT bit-exact.
"""

import math

import torch

from scatterem.nn._backends_kouter import KOUTER_HALF, KOUTER_PACKED
from scatterem.nn.functional.ptychography import (
    correct_aberrations_kouter_half,
    correct_aberrations_kouter_packed,
)
from scatterem.utils.env import resolve_backend


def _half_scratch(hoist, nk, hq, w, like) -> torch.Tensor:
    """The ``(nk, hq, w // 2 + 1)`` half-spectrum buffer: fresh, or the loop's.

    With a ``hoist`` dict the chunk loop keeps ONE buffer for the whole
    call (grown when a later chunk is wider; the tail takes a leading slice,
    which is contiguous).  It is consumed by the ``irfft2`` issued right after
    its correction, and the next chunk's correction is issued after that, on
    the same stream.
    """
    shape = (nk, hq, w // 2 + 1)
    if hoist is None:
        return torch.empty(shape, dtype=like.dtype, device=like.device)
    buf = hoist.get("half")
    if buf is None or int(buf.shape[0]) < nk:
        buf = hoist["half"] = torch.empty(shape, dtype=like.dtype, device=like.device)
    return buf if int(buf.shape[0]) == nk else buf[:nk]


def _kouter_sum_half(
    Gprime_k,
    aberrations,
    ds_rotation,
    semiconvergence_angle,
    wavelength,
    Qy,
    Qx,
    Kx,
    Ky,
    upsample,
    A,
    factor,
    hoist=None,
) -> torch.Tensor | None:
    """The ``reduce="sum"`` image through the half or packed store, or ``None``.

    The ``kouter_half`` row is asked first.  Where it chooses ``"full"`` (in
    practice an odd ``W``), the ``kouter_packed`` row is asked (see
    :func:`_kouter_sum_packed`).  ``None`` means both rows chose ``"full"``,
    and the caller must run the shipped full-grid correction +
    ``ifft2(...).imag``.

    ``hoist`` (1.0665x on ``r14_ffstem/carbon/ptycho``) is a
    dict the chunk loop creates once PER CALL: every chunk of one call shares
    G's dtype/device/Q extents, the upsample factor and the geometry, so the
    backend decision and ``scale`` are decided on the first chunk (only ``Nk``
    can differ, at the tail, and the predicate only asserts it equals
    ``Gprime_k.shape[0]``), and one scratch serves the loop
    (:func:`_half_scratch`).  BIT-EXACT: the same kernels on the same words,
    and ``scale`` is the same float64 expression.  ``None`` decides per chunk.
    """
    nk = int(Gprime_k.shape[0])
    hq = int(Gprime_k.shape[1]) * int(upsample[0])
    w = int(Gprime_k.shape[2]) * int(upsample[1])
    # The predicate declines under tracing; checking first keeps the registry's
    # host Python out of a traced region.
    if torch.compiler.is_compiling():
        return None
    plan = None if hoist is None else hoist.get("plan")
    if plan is None:
        args = (Gprime_k, (nk, hq, w), semiconvergence_angle, wavelength)
        ok = resolve_backend(KOUTER_HALF, None, *args) == "half"
        packed = not ok and resolve_backend(KOUTER_PACKED, None, *args) == "packed"
        plan = (ok, factor / math.sqrt(hq * w), packed)
        if hoist is not None:
            hoist["plan"] = plan
    ok, scale, packed = plan
    if not (ok or packed):
        return None
    if A is None:
        A = torch.empty((0, 2), dtype=torch.float32, device=Gprime_k.device)
    corr = (Gprime_k, aberrations, ds_rotation, semiconvergence_angle, wavelength)
    if packed:
        return _kouter_sum_packed(corr, (Qx, Qy, Kx, Ky, A), (nk, hq, w), scale)
    half = _half_scratch(hoist, nk, hq, w, Gprime_k)
    correct_aberrations_kouter_half(*corr, Qx, Qy, Kx, Ky, A, half)
    img = torch.fft.irfft2(half, s=(hq, w), dim=(-2, -1), norm="forward")
    return torch.sum(img, dim=0) * scale


def _kouter_sum_packed(
    corr: tuple, coords: tuple, full_shape: tuple, scale: float
) -> torch.Tensor:
    """The ``reduce="sum"`` image through the two-for-one packed store.

    ``corr`` is ``(Gprime_k, aberrations, ds_rotation, semiconvergence_angle,
    wavelength)`` and ``coords`` is ``(Qx, Qy, Kx, Ky, A)``.  The packed store
    holds ``Z_j = H_{2j} + i H_{2j+1}``, so ``s = sum_j ifft2(Z_j)`` gives
    ``sum_k Im(ifft2(C_k)) = Re(s) + Im(s)``, and the C2C runs over
    ``ceil(Nk/2)`` planes.  The body keeps the ``norm="forward"`` + ``scale``
    fold of the full-grid body.  It is NOT bit-exact against that body.  At the
    ``(90, 245, 245)`` chunk we measured the transform + k-sum leg at
    1.91x, with a peak-normalised error against complex128 of 2.8e-07 (the
    full-grid form: 3.2e-07).  The ``(ceil(Nk/2), Hq, W)`` store is half the
    size of the full-grid chunk.
    """
    nk, hq, w = full_shape
    G, ab, rot, sca, wl = corr
    Qx, Qy, Kx, Ky, A = coords
    packed = torch.empty(((nk + 1) // 2, hq, w), dtype=G.dtype, device=G.device)
    correct_aberrations_kouter_packed(G, ab, rot, sca, wl, Qx, Qy, Kx, Ky, A, packed)
    z = torch.fft.ifft2(packed, dim=(-2, -1), norm="forward")
    del packed
    s = torch.sum(z, dim=0)
    return (s.real + s.imag) * scale
