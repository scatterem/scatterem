"""Which FMA spelling ATen's complex64 multiply uses on a given CUDA device.

Why this exists
---------------
Many fused kernels in this package promise a result that is ``torch.equal`` to an
eager ATen expression with a complex64 product ``x * y``.  nvcc contracts
``c10::complex<float>::operator*`` into one FMA per half, and the kernels pin the
same FMAs with ``tl.math.fma`` or ``__fmaf_rn``.  The real half is
``fma(xr, yr, -(xi*yi))`` on every build we measured.  The imaginary half is NOT
the same on every device:

* sm_75 to sm_90 code of the torch 2.11 cu128 wheel: ``fma(xr, yi, xi*yr)``;
* sm_120 code of the same wheel (RTX PRO 6000 Blackwell, RTX 50xx):
  ``fma(xi, yr, xr*yi)``.

Measured on 2**20 random pairs (A6000 sm_86 and RTX PRO 6000 sm_120): each
spelling matches its device with 0 mismatches and the other device on no more
than 2/3 of the words.  The two spellings round a different product first, so
they differ by one ulp on about one word in three.

The order is a property of the torch build and the device, so this module asks
ATen once per device and does not guess from the compute capability.  It lives in
``physics``, the leaf layer, so that the Warp kernels in ``physics`` can use it
as well as the kernels in ``nn`` and ``simulation``; it imports only torch.

Use
---
:func:`aten_cmul_imag_swapped` returns ``False`` for ``fma(xr, yi, xi*yr)``,
``True`` for ``fma(xi, yr, xr*yi)``, and ``None`` when ATen uses neither (or the
probe cannot run now).  A kernel passes the bool as a constexpr (Triton) or an
int argument (Warp), and declines on ``None`` so that the caller keeps the eager
expression.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["aten_cmul_imag_swapped", "aten_cmul_swap_flag"]

_ORDER: dict[int, bool | None] = {}

# 64 pairs of float32 values with 20-bit significands in +-[1, 2).  Each product
# then has at most 40 bits, and a product plus a rounded product spans at most 42
# bits, so float64 holds every FMA input sum EXACTLY: one rounding to float32 gives
# the true FMA.  A fixed seed makes the probe the same on every call.
_N = 64


def _operands() -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(20261008)
    # ``device="cpu"`` pins the draws to the CPU generator's device: under a CUDA
    # default device (``torch.set_default_device``) they would otherwise land on
    # CUDA and raise.
    cpu = torch.device("cpu")
    sig = torch.randint(
        1 << 19, 1 << 20, (4, _N), generator=g, dtype=torch.int64, device=cpu
    )
    sign = torch.randint(0, 2, (4, _N), generator=g, dtype=torch.int64, device=cpu)
    sign = sign * 2 - 1
    v = (sign * sig).to(torch.float64) / float(1 << 19)
    return tuple(v[i] for i in range(4))


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)


def _classify(
    got: torch.Tensor,
    xr: torch.Tensor,
    xi: torch.Tensor,
    yr: torch.Tensor,
    yi: torch.Tensor,
) -> bool | None:
    """Name the imaginary FMA order of a measured product ``got`` of ``x * y``.

    ``got`` is the ``(n, 2)`` float32 real view of the product, on the CPU, of
    the complex64 ``x = xr + i xi`` and ``y = yr + i yi`` built from the float64
    operands of :func:`_operands`. Returns ``None`` if the real half is not
    ``fma(xr, yr, -(xi*yi))`` or if the imaginary half matches neither order.
    """
    d = torch.float64
    re = _f32(xr * yr - _f32(xi * yi).to(d))
    im_plain = _f32(xr * yi + _f32(xi * yr).to(d))
    im_swap = _f32(xi * yr + _f32(xr * yi).to(d))
    if not torch.equal(got[:, 0].view(torch.int32), re.view(torch.int32)):
        return None
    words = got[:, 1].view(torch.int32)
    plain = torch.equal(words, im_plain.view(torch.int32))
    swap = torch.equal(words, im_swap.view(torch.int32))
    if plain == swap:  # neither (or the operands could not tell them apart)
        return None
    return swap


def _probe(device: torch.device) -> bool | None:
    xr, xi, yr, yi = _operands()
    x = torch.complex(_f32(xr), _f32(xi)).to(device)
    y = torch.complex(_f32(yr), _f32(yi)).to(device)
    got = torch.view_as_real(x * y).cpu()
    return _classify(got, xr, xi, yr, yi)


def _cached_order(
    cache: dict[int, bool | None],
    device: torch.device | int | str,
    probe: Callable[[torch.device], bool | None],
) -> bool | None:
    """Run ``probe`` once per CUDA device and keep its answer in ``cache``.

    Returns ``None`` without a cache entry on a non-CUDA device, on a torch
    without CUDA, under CUDA graph capture or ``torch.compile`` tracing, and
    when ``probe`` raises a ``RuntimeError``.
    """
    device = torch.device(device)
    # ``is_available`` first: on a CPU-only torch build the other ``torch.cuda``
    # calls below are stubs that raise.
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    if index in cache:
        return cache[index]
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    try:
        with torch.no_grad():
            order = probe(torch.device("cuda", index))
    except RuntimeError:
        return None  # the probe cannot run now; ask again on the next call
    cache[index] = order
    return order


def aten_cmul_imag_swapped(device: torch.device | int | str) -> bool | None:
    """Return the imaginary-half FMA order of ATen's complex64 ``x * y`` on ``device``.

    ``False``: ``fma(xr, yi, xi*yr)``.  ``True``: ``fma(xi, yr, xr*yi)``.  ``None``:
    neither, a non-CUDA device or a torch without CUDA, or a first call that cannot run the probe now (under
    CUDA graph capture or ``torch.compile`` tracing, or the probe raises a
    ``RuntimeError``); a kernel must decline then, and nothing is cached.
    The result is cached per device after the first eager call.
    """
    return _cached_order(_ORDER, device, _probe)


def aten_cmul_swap_flag(device: torch.device | int | str) -> int:
    """The ``im_swap`` launch argument (0 or 1) of a Warp kernel on ``device``.

    0 on a non-CUDA device, where no ATen CUDA order applies (a CPU Warp launch).
    On a CUDA device the order is never guessed: when it is unknown, this raises
    instead of returning 0, because 0 is the sm_75 to sm_90 order and is wrong by
    one ulp on sm_120.  The kernels' predicates decline on an unknown order and
    every graph builder primes the cache before its capture, so a launch should
    never get here without an answer.

    Raises
    ------
    RuntimeError
        If ``device`` is CUDA and :func:`aten_cmul_imag_swapped` has no answer
        (e.g. its first call happens under CUDA graph capture).
    """
    device = torch.device(device)
    if device.type != "cuda":
        return 0
    order = aten_cmul_imag_swapped(device)
    if order is None:
        raise RuntimeError(
            f"ATen's complex64 FMA order on {device} is unknown (first asked under "
            "CUDA graph capture or torch.compile tracing, or ATen uses neither "
            "order); call aten_cmul_imag_swapped(device) eagerly before the capture"
        )
    return int(order)
