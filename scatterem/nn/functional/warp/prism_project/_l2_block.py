"""The L2-blocked, graph-replayed no-grad forward of the Warp route.

The no-grad sub-pixel forward is a 4-pass chain over the K-expanded basis
``(Nmodes, K, Bp, M1, M2)`` -- 2 GiB at ``r2_prism_project/medium``: ramp
multiply (write), ``ifft2`` (two cuFFT passes, read + write each), project kernel
(read).  The stack is ~350x the 6 MiB L2, so every pass goes to DRAM.  Running
the chain in blocks of ``kb`` positions x ``pb`` parent beams keeps each block's
slab in L2 between the passes.

The beam sum is NOT reassociated: :func:`_prism_project_fwd_acc_kernel` resumes
its fp32 accumulator from ``out``, so every output element still adds
``p = 0, 1, ..., Bp - 1`` one at a time in one register, as
``_prism_project_fwd_kernel`` does (p-blocks run in order; k-blocks are
independent).  The ramp multiply is the same unrolled product as
``_make_prism_ramp_mul_kernel``, and blocking is only taken where cuFFT gives a
block batch the same bits as the dense batch (:func:`_ifft2_batch_exact`), so
the result is ``torch.equal`` to the dense chain.  The chain is ~3 launches per
block, so it is recorded once as a CUDA graph and replayed.

The ``prism_proj_l2_graph`` backend row decides (:func:`prism_proj_l2_supported`
is its predicate); the dense chain is the fallback.  Lab: 1.14x on
``r2_prism_project/medium/project`` (Warp), peak 5380 -> 1300 MiB.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from scatterem.nn._backends_graphs import PRISM_PROJ_L2_GRAPH
from scatterem.physics import _warp_interop as wi
from scatterem.physics._aten_cmul_order import (
    aten_cmul_imag_swapped,
    aten_cmul_swap_flag,
)
from scatterem.physics._cmul_warp import cmul_aten
from scatterem.utils.env import resolve_backend

from ._funcs import _fwd_prod
from .ramp import _subpixel_ramp

__all__ = ["PROJ_L2_GRAPH_STATS", "prism_proj_l2_blocked", "prism_proj_l2_supported"]

#: Positions per block; the block ramp kernel is instantiated at it.  Measured at
#: r2/medium (device time, contended card): (kb, pb) = (2, 4) 41.1 ms, (4, 2)
#: 45.5, (8, 1) 52.5, vs dense 50.3.
_PROJ_L2_BLOCK_K = 2
#: Byte budget for one block's ``(kb, pb, M1, M2)`` complex64 slab; ``pb`` is the
#: largest power of two that fits and divides ``Bp``.  4 MiB = (2, 4) at M = 256.
_PROJ_L2_BLOCK_BYTES = 4 * 1024 * 1024
#: Graphs held at once.  Never evicted (a dead graph strands its private memory
#: pool); past the cap the dense route runs.
_PROJ_L2_GRAPH_MAX = 4
#: Reach counters: ``calls`` = replays, ``captures`` = graphs recorded,
#: ``declined`` = calls sent to the dense route.
PROJ_L2_GRAPH_STATS = {"calls": 0, "captures": 0, "declined": 0}
_PROJ_L2_GRAPHS: dict[tuple, _ProjL2BlockGraph] = {}
_PROJ_L2_SEEN: set[tuple] = set()
_PROJ_IFFT_BATCH_EXACT: dict[tuple, bool] = {}


def _make_prism_ramp_mul_block_kernel(T: int):
    """``out[i, p] = basis[p] * ramp[i]`` for ``i < T``, one thread per (p, y, x).

    The block-sized twin of ``_make_prism_ramp_mul_kernel``: the same pinned
    product (``cmul_aten``, so the same bits), but the grid also spans the block's
    beams, because a block has too few of them for a per-pixel beam loop to fill
    the GPU.
    """
    RAMP_T = wp.types.vector(length=2 * T, dtype=wp.float32)

    @wp.kernel
    def _kernel(
        basis: wp.array3d(dtype=wp.vec2),  # (pb, M1, M2)  # pyright: ignore[reportInvalidTypeForm]
        ramp: wp.array3d(dtype=wp.vec2),  # (T, M1, M2)  # pyright: ignore
        out: wp.array4d(dtype=wp.vec2),  # (T, pb, M1, M2)  # pyright: ignore
        im_swap: int,  # ATen's imaginary FMA on this device (_aten_cmul_order)
    ):
        p, y, x = wp.tid()  # pyright: ignore
        rv = RAMP_T()
        for i in range(T):
            r = ramp[i, y, x]
            rv[2 * i] = r[0]
            rv[2 * i + 1] = r[1]
        b = basis[p, y, x]
        br = b[0]
        bi = b[1]
        for i in range(T):
            out[i, p, y, x] = cmul_aten(br, bi, rv[2 * i], rv[2 * i + 1], im_swap)

    return _kernel


_prism_ramp_mul_block_kernel = _make_prism_ramp_mul_block_kernel(_PROJ_L2_BLOCK_K)


@wp.kernel
def _prism_project_fwd_acc_kernel(
    basis_rs: wp.array(dtype=wp.vec2, ndim=4),  # (kb, pb, M1, M2) one block  # pyright: ignore[reportInvalidTypeForm]
    Sdt: wp.array(dtype=wp.vec2, ndim=3),  # (pb, SY, SX)  # pyright: ignore
    positions: wp.array(dtype=wp.int32, ndim=2),  # (kb, 2)  # pyright: ignore
    SY: int,
    SX: int,
    out: wp.array(dtype=wp.vec2, ndim=3),  # (kb, M1, M2) in/out  # pyright: ignore
) -> None:
    # ``_prism_project_fwd_kernel``'s periodic fp32 body, with the accumulator
    # loaded from ``out`` instead of zero and the beam loop over this block only.
    k, y, x = wp.tid()  # pyright: ignore
    py = positions[k, 0]
    px = positions[k, 1]
    acc = out[k, y, x]
    nb = basis_rs.shape[1]
    gy = (py + y) % SY
    gx = (px + x) % SX
    if gy < 0:
        gy += SY
    if gx < 0:
        gx += SX
    for p in range(nb):
        acc = acc + _fwd_prod(basis_rs[k, p, y, x], Sdt[p, gy, gx], 0)
    out[k, y, x] = acc


def _ifft2_batch_exact(n: int, block: int, M1: int, M2: int, device) -> bool:
    """Whether ``ifft2(norm="forward")`` at batch ``block`` matches batch ``n``.

    cuFFT picks its kernel partly from the batch count, so blocking is exact only
    if the block batch reproduces the dense batch slice for slice (it does not at
    128^2 on an A6000).  Asked once per shape, on random data, at the dense
    call's real batch count.
    """
    key = (device.index, n, block, M1, M2)
    verdict = _PROJ_IFFT_BATCH_EXACT.get(key)
    if verdict is None:
        g = torch.Generator(device=device).manual_seed(949)
        x = torch.randn((n, M1, M2), dtype=torch.complex64, device=device, generator=g)
        full = torch.fft.ifft2(x, norm="forward")
        part = torch.fft.ifft2(x[:block], norm="forward")
        verdict = bool(torch.equal(part, full[:block]))
        del x, full, part
        _PROJ_IFFT_BATCH_EXACT[key] = verdict
    return verdict


class _ProjL2BlockGraph:
    """The whole blocked forward for one ``(Sdt, basis_fft)`` pair, as ONE graph.

    ``Sdt`` and ``basis_fft`` are read in place (the cache key pins their
    address, shape and strides, so a replay reads whatever those tensors hold
    now); the per-call ramp and positions are copied into static buffers before
    the replay and the result is cloned out of a static one.
    """

    def __init__(self, Sdt, basis_fft, ramp, positions, kb, pb):
        _, Bp, M1, M2 = basis_fft.shape
        K = ramp.shape[0]
        dev = Sdt.device
        self.SY, self.SX = int(Sdt.shape[-2]), int(Sdt.shape[-1])
        self.ramp = ramp.clone()
        self.pos = positions.clone()
        self.out = torch.empty((K, M1, M2, 2), dtype=torch.float32, device=dev)
        self.shifted = torch.empty((kb, pb, M1, M2), dtype=torch.complex64, device=dev)
        self.blocks = [(p0, k0) for p0 in range(0, Bp, pb) for k0 in range(0, K, kb)]
        self.kb, self.pb = kb, pb
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):  # cuFFT plans + Warp module load, off-capture
            self._body(Sdt, basis_fft)
        torch.cuda.current_stream(dev).wait_stream(side)
        aten_cmul_imag_swapped(dev)  # prime the FMA order off-capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body(Sdt, basis_fft)

    def _body(self, Sdt, basis_fft):
        dev = wi.device_from_torch(Sdt.device)
        st = wi.stream_from_torch(torch.cuda.current_stream(Sdt.device))
        kb, pb = self.kb, self.pb
        plane = tuple(self.shifted.shape[-2:])
        sdt_r = torch.view_as_real(Sdt)
        b_r = torch.view_as_real(basis_fft[0])
        ramp_r = torch.view_as_real(self.ramp)
        sh_w = wi.from_torch(torch.view_as_real(self.shifted), dtype=wp.vec2)
        im_swap = aten_cmul_swap_flag(Sdt.device)
        self.out.zero_()
        for p0, k0 in self.blocks:
            wp.launch(
                _prism_ramp_mul_block_kernel,
                dim=(pb, *plane),
                inputs=[
                    wi.from_torch(b_r[p0 : p0 + pb], dtype=wp.vec2),
                    wi.from_torch(ramp_r[k0 : k0 + kb], dtype=wp.vec2),
                    sh_w,
                    im_swap,
                ],
                device=dev,
                stream=st,
            )
            # Out of place on purpose: ``out=`` computes into a temporary and
            # copies, a whole extra pass per block.
            rs = torch.fft.ifft2(self.shifted, norm="forward")
            wp.launch(
                _prism_project_fwd_acc_kernel,
                dim=(kb, *plane),
                inputs=[
                    wi.from_torch(torch.view_as_real(rs), dtype=wp.vec2),
                    wi.from_torch(sdt_r[p0 : p0 + pb], dtype=wp.vec2),
                    wi.from_torch(self.pos[k0 : k0 + kb], dtype=wp.int32),
                    self.SY,
                    self.SX,
                ],
                outputs=[wi.from_torch(self.out[k0 : k0 + kb], dtype=wp.vec2)],
                device=dev,
                stream=st,
            )

    def run(self, ramp: Tensor, positions: Tensor) -> Tensor:
        self.ramp.copy_(ramp)
        self.pos.copy_(positions)
        self.graph.replay()
        return self.out.clone()


def _proj_l2_block_p(Bp: int, M1: int, M2: int) -> int:
    """Beams per block under :data:`_PROJ_L2_BLOCK_BYTES` (0 = refuse)."""
    per = _PROJ_L2_BLOCK_K * M1 * M2 * 8
    pb = 1
    while pb * 2 * per <= _PROJ_L2_BLOCK_BYTES and Bp % (pb * 2) == 0:
        pb *= 2
    return pb if pb * per <= _PROJ_L2_BLOCK_BYTES and Bp % pb == 0 else 0


def prism_proj_l2_supported(Sdt: Tensor, basis_fft: Tensor, positions: Tensor) -> bool:
    """True if the blocked graph may replace the dense no-grad Warp forward.

    The caller has already established fp32 precision, a periodic crop, no live
    gradient and a non-identity ramp.  Here: one mode, contiguous CUDA
    complex64 operands with no lazy conj / neg bit, ``K`` a multiple of the
    position block, a beam block that fits its budget, a K-expanded stack larger
    than 3x L2 (smaller ones are already cache resident), and no capture or
    compile in progress.
    """
    if not (
        Sdt.is_cuda
        and basis_fft.is_cuda
        and basis_fft.ndim == 4
        and basis_fft.shape[0] == 1
        and Sdt.dtype == torch.complex64
        and basis_fft.dtype == torch.complex64
        and not (Sdt.is_conj() or Sdt.is_neg())
        and not (basis_fft.is_conj() or basis_fft.is_neg())
        and Sdt.is_contiguous()
        and basis_fft.is_contiguous()
        and int(positions.shape[0]) % _PROJ_L2_BLOCK_K == 0
        and not torch.cuda.is_current_stream_capturing()
        and not torch.compiler.is_compiling()
        and aten_cmul_imag_swapped(Sdt.device) is not None
    ):
        return False
    _, Bp, M1, M2 = basis_fft.shape
    K = int(positions.shape[0])
    l2 = torch.cuda.get_device_properties(Sdt.device).L2_cache_size
    return _proj_l2_block_p(Bp, M1, M2) != 0 and K * Bp * M1 * M2 * 8 * 4 > 3 * l2


def prism_proj_l2_blocked(
    Sdt: Tensor, basis_fft: Tensor, positions: Tensor, dr: Tensor
) -> Tensor | None:
    """The blocked forward under graph replay, or ``None`` to take the dense route.

    A graph is recorded on the SECOND sighting of a key, so a one-off call never
    pays a capture, and never more than :data:`_PROJ_L2_GRAPH_MAX` are held.
    """
    if resolve_backend(PRISM_PROJ_L2_GRAPH, None, Sdt, basis_fft, positions) != (
        "graph"
    ):
        PROJ_L2_GRAPH_STATS["declined"] += 1
        return None
    _, Bp, M1, M2 = basis_fft.shape
    K = int(positions.shape[0])
    kb, pb = _PROJ_L2_BLOCK_K, _proj_l2_block_p(Bp, M1, M2)
    key = (
        Sdt.device.index,
        Sdt.data_ptr(),
        tuple(Sdt.shape),
        basis_fft.data_ptr(),
        tuple(basis_fft.shape),
        K,
        kb,
        pb,
    )
    graph = _PROJ_L2_GRAPHS.get(key)
    if graph is None and (
        key not in _PROJ_L2_SEEN
        or len(_PROJ_L2_GRAPHS) >= _PROJ_L2_GRAPH_MAX
        or not _ifft2_batch_exact(K * Bp, kb * pb, M1, M2, Sdt.device)
    ):
        if len(_PROJ_L2_SEEN) < 64:
            _PROJ_L2_SEEN.add(key)
        PROJ_L2_GRAPH_STATS["declined"] += 1
        return None
    ramp = _subpixel_ramp(M1, M2, dr, Sdt.device)
    pos = positions.to(torch.int32).contiguous()
    if graph is None:
        graph = _ProjL2BlockGraph(Sdt, basis_fft, ramp, pos, kb, pb)
        _PROJ_L2_GRAPHS[key] = graph
        PROJ_L2_GRAPH_STATS["captures"] += 1
    PROJ_L2_GRAPH_STATS["calls"] += 1
    return torch.view_as_complex(graph.run(ramp, pos)).reshape(1, K, M1, M2)
