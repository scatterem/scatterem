"""Per-row float64 sums in the order of torch's own ``.sum()``.

Private.  :mod:`._clamp_cast_warp` and :mod:`._shift_clamp_warp`
use it for the dose sums of ``Dataset4DStem.__init__``.

Why this exists
---------------
The constructor turns the per-row float64 sums of the clamped cube into
``dose_per_probe_unnormalized``, a value with a bit-exact contract.  The slot
path of :mod:`._clamp_cast_warp` writes a float64 copy of each row (8 bytes
written per element) and runs torch's ``.sum()`` on it (8 bytes read): 52.7 % of
the row's device time on ``d29_pad_taper/large/fourierspace``.
The kernel here reads each float32 value once, clamps it, widens it in a
register and adds it.  It writes no float64 copy.

Numerics
--------
BIT-EXACT to ``row.to(torch.float64).sum()`` of an aligned contiguous row of
``n >= 2048`` elements.  The kernel adds the widened values in the order that
torch 2.11's ``reduce_kernel`` (``ATen/native/cuda/Reduce.cuh``) uses for that
sum:

* Block of 512 threads.  Thread ``t`` of block ``b`` starts at vector
  ``t + 512 b`` (4 elements per vector), steps ``512 * ctas`` vectors, and keeps
  one accumulator per vector lane.  Block 0 adds the ``n % 4`` tail elements to
  accumulator 0 (thread ``t`` takes element ``4 (n // 4) + t``).  Then
  ``((a0 + a1) + a2) + a3``.
* A pairwise tree over the 512 thread values (offsets 256 .. 1).
* ``ctas`` is torch's ``ctas_per_output``: 1 if ``ceil(n / 512) < 256``, else
  ``max(min(target, ceil(vpt / 16)), ceil(vpt / 256))`` with
  ``target = SMs * (max threads per SM // 512)``.  If ``ctas > 1``, a second
  launch runs the same 512-wide tree over the ``ctas`` block partials:
  thread ``t`` first adds partials ``t, t + 512, ...`` (0 if ``t >= ctas``).

Every step is a float64 add, so the compiler has no rounding choice left.  The
one exception is a NaN sum: its payload is that of one NaN input, chosen by the
operand order inside the add, which the compiler may swap.  If any row sum is
NaN, :func:`order_sums` sums the rows again with torch's ``.sum()``.

Gating
------
The ``clamp_sum_order`` row of :mod:`scatterem.data._backends`, whose predicate
is :func:`sum_order_supported`: a row of at least 2048 elements (torch's block
is 512 wide there) and torch 2.11 on CUDA, not ROCm.  The order is a property
of the torch build.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import warp as wp

from scatterem.physics import _warp_interop as wi
from scatterem.physics._warp_rn import fsqrt_rn as fsqrt

__all__ = [
    "CALLS",
    "ORDER_BLOCK",
    "block_tree_sum",
    "order_sums",
    "raw_order_sums",
    "row_sums_torch_order",
    "sum_order_supported",
]

#: ``[ran, NaN fallback]``.  Host-side counters only, so a probe can prove the
#: path was TAKEN rather than infer it from a timing.
CALLS = [0, 0]

#: Block width of torch's reduction for these rows, and the shortest row it is
#: replicated for (``n // 4 >= 512``, so torch's block is 512 wide).
ORDER_BLOCK = 512
_ORDER_MIN_ROW = 2048


@wp.func_native(
    """
__shared__ double _scatterem_order_tree[512];
int t = threadIdx.x;
_scatterem_order_tree[t] = v;
for (int off = 256; off > 0; off >>= 1) {
    __syncthreads();
    if (t < off) {
        _scatterem_order_tree[t] = _scatterem_order_tree[t] + _scatterem_order_tree[t + off];
    }
}
__syncthreads();
double r = _scatterem_order_tree[0];
__syncthreads();
return r;
"""
)
def block_tree_sum(v: wp.float64) -> wp.float64:
    """Pairwise sum of ``v`` over a 512-thread block; every thread gets the result.

    The kernels that call it launch whole blocks only (``dim`` is a multiple of
    the block), so every thread reaches each ``__syncthreads``.
    """
    ...


@wp.func
def _clamp_load(
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    sqrt: int,
) -> wp.float64:
    # The element code of `_clamp_cast_kernel` / `_clamp_sqrt_cast_kernel`.
    v = x[i]
    if v <= wp.float32(0.0):
        x[i] = wp.float32(0.0)
        return wp.float64(0.0)
    if sqrt != 0:
        r = fsqrt(v)
        x[i] = r
        return wp.float64(r)
    return wp.float64(v)


@wp.kernel
def _clamp_sum_order_kernel(
    x: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm]
    sqrt: int,
    row: int,
    ctas: int,
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # One 512-thread block per (row, b); see "Numerics".
    tid = wp.tid()
    t = tid % 512  # pyright: ignore[reportOperatorIssue]
    rb = tid // 512  # pyright: ignore[reportOperatorIssue]
    b = rb % ctas
    base = (rb // ctas) * row
    nvec = row // 4
    idx = t + b * 512
    stride = 512 * ctas
    a0 = wp.float64(0.0)
    a1 = wp.float64(0.0)
    a2 = wp.float64(0.0)
    a3 = wp.float64(0.0)
    while idx < nvec:
        e = base + idx * 4
        a0 = a0 + _clamp_load(x, e, sqrt)
        a1 = a1 + _clamp_load(x, e + 1, sqrt)
        a2 = a2 + _clamp_load(x, e + 2, sqrt)
        a3 = a3 + _clamp_load(x, e + 3, sqrt)
        idx += stride
    if b == 0:
        k = nvec * 4 + t
        if k < row:
            a0 = a0 + _clamp_load(x, base + k, sqrt)
    s = block_tree_sum(((a0 + a1) + a2) + a3)
    if t == 0:
        part[rb] = s


@wp.kernel
def _partials_order_kernel(
    part: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ctas: int,
    out: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
):
    # torch's last-block reduce: thread t holds 0 + partial t + partial t + 512
    # + ... (0 past ctas).  ``ctas`` can pass 512 (a row of more than 2**22
    # elements on a card with 528+ target blocks, or of more than 2**26 on any).
    tid = wp.tid()
    t = tid % 512  # pyright: ignore[reportOperatorIssue]
    r = tid // 512  # pyright: ignore[reportOperatorIssue]
    v = wp.float64(0.0)
    p = t
    while p < ctas:
        v = v + part[r * ctas + p]
        p += 512
    s = block_tree_sum(v)
    if t == 0:
        out[r] = s


#: ``device index -> SMs * (max threads per SM // 512)``, torch's target grid.
_TARGET_GRID: dict[int, int] = {}


def _ctas_per_row(row: int, device: torch.device) -> int:
    """torch's ``ctas_per_output`` for the ``.sum()`` of a float64 row of ``row``."""
    vpt = -(-row // ORDER_BLOCK)
    if vpt < 256:
        return 1
    key = device.index if device.index is not None else torch.cuda.current_device()
    target = _TARGET_GRID.get(key)
    if target is None:
        p = torch.cuda.get_device_properties(key)
        target = p.multi_processor_count * (
            p.max_threads_per_multi_processor // ORDER_BLOCK
        )
        _TARGET_GRID[key] = target
    return max(min(target, -(-vpt // 16)), -(-vpt // 256))


def sum_order_supported(row: int) -> bool:
    """The ``clamp_sum_order`` row's predicate: is the order known for ``row``?

    Parameters
    ----------
    row : int
        The number of elements in one row (one outer slice) of the cube.

    Returns
    -------
    bool
        Whether the row is long enough and the torch build is the one whose
        order the kernel copies (torch 2.11 on CUDA, not ROCm).
    """
    return (
        row >= _ORDER_MIN_ROW
        and torch.__version__.startswith("2.11.")
        and torch.version.hip is None
    )


def raw_order_sums(
    kernel: Callable[..., None],
    inputs: list,
    n_rows: int,
    row: int,
    on: torch.device,
) -> torch.Tensor:
    """Launch ``kernel(*inputs, row, ctas, part)`` and reduce the partials.

    Unlike :func:`order_sums`, a NaN sum is returned as the kernel made it.

    Parameters
    ----------
    kernel : warp.Kernel
        A kernel with the thread map of ``_clamp_sum_order_kernel``.
    inputs : list
        Its leading arguments, before ``row``, ``ctas`` and ``part``.
    n_rows : int
        The number of rows.
    row : int
        The number of elements in one row.
    on : torch.device
        The CUDA device of the operands.

    Returns
    -------
    torch.Tensor
        The ``(n_rows,)`` float64 row sums, on the device.
    """
    ctas = _ctas_per_row(row, on)
    part = torch.empty(n_rows * ctas, dtype=torch.float64, device=on)
    device = wi.device_from_torch(on)
    # Warp's default stream is not PyTorch's current stream; launch on the torch
    # stream so the kernel is ordered against the ops around it.
    stream = wi.stream_from_torch(torch.cuda.current_stream(on))
    f64 = {"dtype": wp.float64, "return_ctype": True}
    wp.launch(
        kernel,
        dim=n_rows * ctas * ORDER_BLOCK,
        inputs=[*inputs, row, ctas],
        outputs=[wi.from_torch(part, **f64)],
        device=device,
        stream=stream,
        block_dim=ORDER_BLOCK,
    )
    sums = part
    if ctas > 1:
        sums = torch.empty(n_rows, dtype=torch.float64, device=on)
        wp.launch(
            _partials_order_kernel,
            dim=n_rows * ORDER_BLOCK,
            inputs=[wi.from_torch(part, **f64), ctas],
            outputs=[wi.from_torch(sums, **f64)],
            device=device,
            stream=stream,
            block_dim=ORDER_BLOCK,
        )
    return sums


def order_sums(
    kernel: Callable[..., None], inputs: list, written: torch.Tensor, row: int
) -> torch.Tensor:
    """Launch ``kernel(*inputs, row, ctas, part)`` and finish the row sums.

    Parameters
    ----------
    kernel : warp.Kernel
        A kernel with the thread map of ``_clamp_sum_order_kernel``.
    inputs : list
        Its leading arguments, before ``row``, ``ctas`` and ``part``.
    written : torch.Tensor
        The clamped cube that the kernel writes, ``(n_rows, ...)``.  The NaN
        fallback sums its rows with torch's ``.sum()``.
    row : int
        The number of elements in one row of ``written``.

    Returns
    -------
    torch.Tensor
        The ``(n_rows,)`` float64 row sums, on the device.
    """
    n_rows = int(written.shape[0])
    sums = raw_order_sums(kernel, inputs, n_rows, row, written.device)
    CALLS[0] += 1
    # A NaN sum carries the payload of one of its NaN inputs, and which one wins
    # depends on the operand order inside each add, which a compiler may swap.
    # So only a NaN sum can differ from torch's bits: then sum the (already
    # clamped) rows with torch's `.sum()`.  One host read; the caller reads the
    # sums next.
    if bool(torch.isnan(sums).any()):
        CALLS[1] += 1
        return torch.stack([written[i].to(torch.float64).sum() for i in range(n_rows)])
    return sums


def row_sums_torch_order(array: torch.Tensor, sqrt: bool) -> torch.Tensor:
    """Clamp (and root) ``array`` IN PLACE and return its row sums in torch's order.

    Parameters
    ----------
    array : torch.Tensor
        A contiguous CUDA float32 cube that the ``clamp_cast`` row took.
    sqrt : bool
        Also take the square root in place.

    Returns
    -------
    torch.Tensor
        The ``(array.shape[0],)`` float64 row sums, on the device.
    """
    row = array.numel() // int(array.shape[0])
    # Named local: a raw ctype does not keep its tensor alive.
    flat = array.reshape(-1)
    x_ct = wi.from_torch(flat, dtype=wp.float32, return_ctype=True)
    return order_sums(_clamp_sum_order_kernel, [x_ct, int(sqrt)], array, row)
