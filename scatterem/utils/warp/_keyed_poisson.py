"""Counter-based Poisson draws keyed by ``(key, row, pixel)`` (Warp, CPU and CUDA).

Pattern streaming's float16 tier draws the counts of a noise-free pattern on the fly. Every draw is a pure
function of ``(key, global row, pixel, draw index)``, so a pattern's noise
does not depend on the batch, rank, epoch, slot or thread that computes it.

- **Generator.** Philox-4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel
  random numbers: as easy as 1, 2, 3", SC'11), written from the paper: ten
  rounds of two 32x32 -> 64-bit multiplies with the multipliers
  ``0xD2511F53`` and ``0xCD9E8D57`` and the Weyl key bumps ``0x9E3779B9`` and
  ``0xBB67AE85``. The counter is ``(pixel, draw block, row low, row high)``
  and the key is the 64-bit ``key``; each Philox call gives four 32-bit
  uniforms.
- **Poisson.** Inversion by sequential search for ``lambda < 10``; above,
  Hörmann's transformed rejection with squeeze (PTRS; W. Hörmann, "The
  transformed rejection method for generating Poisson random variables",
  Insurance: Mathematics and Economics 12, 1993). Each rejection consumes the
  next two uniforms, so the count stays a pure function of the key.
- **Devices.** The integer stream is identical on CPU and CUDA. The Poisson
  transform uses ``log``/``exp`` in float64, which may differ in the last
  bit between the two, so counts are reproducible within a device type.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from scatterem.physics import _warp_interop as wi

__all__ = ["keyed_poisson", "philox4x32"]


_M0 = wp.constant(wp.uint32(0xD2511F53))
_M1 = wp.constant(wp.uint32(0xCD9E8D57))
_W0 = wp.constant(wp.uint32(0x9E3779B9))
_W1 = wp.constant(wp.uint32(0xBB67AE85))
_LOW = wp.constant(wp.uint64(0xFFFFFFFF))
_SHIFT = wp.constant(wp.uint64(32))
#: The inversion / PTRS switch.
_PTRS_FROM = 10.0


@wp.func
def _philox(c: wp.vec4ui, k0: wp.uint32, k1: wp.uint32) -> wp.vec4ui:
    x0, x1, x2, x3 = c[0], c[1], c[2], c[3]  # pyright: ignore[reportIndexIssue]
    a, b = k0, k1
    for _r in range(10):
        p0 = wp.uint64(_M0) * wp.uint64(x0)
        p1 = wp.uint64(_M1) * wp.uint64(x2)
        y0 = wp.uint32(p1 >> _SHIFT) ^ x1 ^ a  # pyright: ignore[reportOperatorIssue]
        y2 = wp.uint32(p0 >> _SHIFT) ^ x3 ^ b  # pyright: ignore[reportOperatorIssue]
        x1 = wp.uint32(p1 & _LOW)
        x3 = wp.uint32(p0 & _LOW)
        x0 = y0
        x2 = y2
        a = a + _W0
        b = b + _W1
    return wp.vec4ui(x0, x1, x2, x3)


@wp.func
def _uniform(
    pix: wp.uint32, j: wp.int32, rlo: wp.uint32, rhi: wp.uint32,
    k0: wp.uint32, k1: wp.uint32,
) -> wp.float64:  # fmt: skip
    """The ``j``-th uniform in (0, 1) of pixel ``pix`` of the row ``(rlo, rhi)``."""
    v = _philox(wp.vec4ui(pix, wp.uint32(j / 4), rlo, rhi), k0, k1)
    lane = j % 4
    u = v[0]  # pyright: ignore[reportIndexIssue]
    if lane == 1:
        u = v[1]  # pyright: ignore[reportIndexIssue]
    if lane == 2:
        u = v[2]  # pyright: ignore[reportIndexIssue]
    if lane == 3:
        u = v[3]  # pyright: ignore[reportIndexIssue]
    return (wp.float64(u) + wp.float64(0.5)) * wp.float64(2.3283064365386963e-10)


@wp.func
def _log_factorial(k: wp.float64) -> wp.float64:
    if k < wp.float64(10.0):
        s = wp.float64(0.0)
        i = wp.float64(2.0)
        while i <= k:
            s = s + wp.log(i)
            i = i + wp.float64(1.0)
        return s
    n = k + wp.float64(1.0)  # Stirling's series for log Gamma(n), n >= 11
    r = wp.float64(1.0) / n
    r2 = r * r  # pyright: ignore[reportOperatorIssue]
    tail = r * (
        wp.float64(1.0) / wp.float64(12.0)
        - r2 * (wp.float64(1.0) / wp.float64(360.0) - r2 / wp.float64(1260.0))
    )
    half_log_2pi = wp.float64(0.91893853320467274178)
    return (n - wp.float64(0.5)) * wp.log(n) - n + half_log_2pi + tail


@wp.func
def _inversion(
    lam: wp.float64, pix: wp.uint32, rlo: wp.uint32, rhi: wp.uint32,
    k0: wp.uint32, k1: wp.uint32,
) -> wp.float64:  # fmt: skip
    u = _uniform(pix, 0, rlo, rhi, k0, k1)  # pyright: ignore[reportArgumentType]
    k = wp.float64(0.0)
    p = wp.exp(-lam)
    f = p
    while u > f and k < wp.float64(1000.0):
        k = k + wp.float64(1.0)
        p = p * lam / k
        f = f + p
    return k


@wp.func
def _ptrs(
    lam: wp.float64, pix: wp.uint32, rlo: wp.uint32, rhi: wp.uint32,
    k0: wp.uint32, k1: wp.uint32,
) -> wp.float64:  # fmt: skip
    slam = wp.sqrt(lam)
    loglam = wp.log(lam)
    b = wp.float64(0.931) + wp.float64(2.53) * slam
    a = wp.float64(-0.059) + wp.float64(0.02483) * b
    log_inv_alpha = wp.log(
        wp.float64(1.1239) + wp.float64(1.1328) / (b - wp.float64(3.4))
    )
    vr = wp.float64(0.9277) - wp.float64(3.6224) / (b - wp.float64(2.0))
    result = wp.float64(-1.0)
    j = wp.int32(0)
    while result < wp.float64(0.0):
        uu = _uniform(pix, j, rlo, rhi, k0, k1) - wp.float64(0.5)
        v = _uniform(pix, j + 1, rlo, rhi, k0, k1)
        j = j + 2
        us = wp.float64(0.5) - wp.abs(uu)
        k = wp.floor((wp.float64(2.0) * a / us + b) * uu + lam + wp.float64(0.43))
        if us >= wp.float64(0.07) and v <= vr:
            result = k
        elif k >= wp.float64(0.0) and not (us < wp.float64(0.013) and v > us):
            lhs = wp.log(v) + log_inv_alpha - wp.log(a / (us * us) + b)
            if lhs <= -lam + k * loglam - _log_factorial(k):
                result = k
    return result


@wp.kernel
def _poisson_kernel(
    lam: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    rows: wp.array(dtype=wp.int64),  # pyright: ignore[reportInvalidTypeForm]
    k0: wp.uint32,
    k1: wp.uint32,
    out: wp.array2d(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
):
    i, p = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    row = wp.uint64(rows[i])
    rlo = wp.uint32(row & _LOW)
    rhi = wp.uint32(row >> _SHIFT)
    x = wp.float64(lam[i, p])
    k = wp.float64(0.0)
    if x > wp.float64(0.0):
        if x < wp.float64(_PTRS_FROM):
            k = _inversion(x, wp.uint32(p), rlo, rhi, k0, k1)
        else:
            k = _ptrs(x, wp.uint32(p), rlo, rhi, k0, k1)
    out[i, p] = wp.float32(k)  # pyright: ignore[reportArgumentType]


@wp.kernel
def _philox_kernel(
    ctr: wp.array(dtype=wp.vec4ui),  # pyright: ignore[reportInvalidTypeForm]
    k0: wp.uint32,
    k1: wp.uint32,
    out: wp.array(dtype=wp.vec4ui),  # pyright: ignore[reportInvalidTypeForm]
):
    i = wp.tid()
    out[i] = _philox(ctr[i], k0, k1)


def _split_key(key: int) -> tuple[int, int]:
    key = int(key)
    if not 0 <= key < 2**64:
        raise ValueError(f"key must be an unsigned 64-bit integer, got {key}")
    return key & 0xFFFFFFFF, key >> 32


def _wp_device(device: torch.device | str) -> str:
    dev = torch.device(device)
    if dev.type == "cuda":
        return f"cuda:{torch.cuda.current_device() if dev.index is None else dev.index}"
    return "cpu"


def philox4x32(
    counters: np.ndarray, key: tuple[int, int], *, device: str = "cpu"
) -> np.ndarray:
    """Philox-4x32-10 of each counter under ``key`` (for checks and tests).

    Parameters
    ----------
    counters : numpy.ndarray
        ``(n, 4)`` uint32 counters.
    key : tuple of int
        The two 32-bit key words ``(k0, k1)``.
    device : str, default="cpu"
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    numpy.ndarray
        ``(n, 4)`` uint32 outputs.
    """
    c = np.ascontiguousarray(counters, dtype=np.uint32).reshape(-1, 4)
    dev = _wp_device(device)
    wc = wp.array(c, dtype=wp.vec4ui, device=dev)
    out = wp.zeros(len(c), dtype=wp.vec4ui, device=dev)
    k0, k1 = (wp.uint32(int(k)) for k in key)
    wp.launch(_philox_kernel, dim=len(c), inputs=[wc, k0, k1, out], device=dev)
    return out.numpy().reshape(-1, 4)


def keyed_poisson(lam: torch.Tensor, rows: torch.Tensor, *, key: int) -> torch.Tensor:
    """Poisson counts with means ``lam``, keyed by ``(key, rows[i], pixel)``.

    Runs on ``lam``'s device, on the current torch stream there.

    Parameters
    ----------
    lam : Tensor
        ``(n, ...)`` float32 means, one row per pattern; finite and
        non-negative (checked for CPU tensors only).
    rows : Tensor
        ``(n,)`` int64 global rows of the patterns (any device).
    key : int
        Unsigned 64-bit key, e.g. ``keyed_seed(noise_seed, tilt)``.

    Returns
    -------
    Tensor
        Counts as float32, ``lam``'s shape and device.

    Raises
    ------
    ValueError
        ``rows`` does not give one row per pattern or holds a negative row,
        or a mean is negative or not finite.
    """
    if rows.ndim != 1 or rows.numel() != lam.shape[0]:
        raise ValueError(
            f"rows must give one global row per pattern: {tuple(rows.shape)} rows "
            f"for {lam.shape[0]} patterns"
        )
    if rows.device.type == "cpu" and rows.numel() and int(rows.min()) < 0:
        raise ValueError("global rows must be non-negative")
    flat = lam.to(torch.float32).reshape(lam.shape[0], -1).contiguous()
    # checked on the host only: on CUDA the check would synchronise the compute
    # stream every batch (the streamed means are checked at ingest)
    on_host = flat.device.type == "cpu"
    if (
        on_host
        and flat.numel()
        and not bool(((flat >= 0) & torch.isfinite(flat)).all())
    ):
        raise ValueError("Poisson means must be finite and non-negative")
    out = torch.empty_like(flat)
    if flat.numel() == 0:
        return out.reshape(lam.shape)
    k0, k1 = _split_key(key)
    r = rows.to(device=flat.device, dtype=torch.int64).contiguous()
    dev = _wp_device(flat.device)
    stream = None
    if flat.device.type == "cuda":
        stream = wi.stream_from_torch(torch.cuda.current_stream(flat.device))
    wp.launch(
        _poisson_kernel, dim=tuple(flat.shape), device=dev, stream=stream,
        inputs=[wi.from_torch(flat), wi.from_torch(r, dtype=wp.int64),
                wp.uint32(k0), wp.uint32(k1), wi.from_torch(out)],
    )  # fmt: skip
    return out.reshape(lam.shape)
