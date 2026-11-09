"""The multislice exit walk: carry a wave stack from one slice to the exit.

Generic (not EELS-specific) slice-walk helpers shared by the EELS forward
kernels (transition-potential multislice, dual-S PRISM, the detector-exit
matrix): skip exactly-unit transmission slices, walk the stack slice by slice,
and block that walk over the plane axis to stay L2-resident on CUDA when cuFFT
is verified bit-invariant under the blocking.  Moved verbatim (2026-09-27) from
``simulation/eels/forward/multislice_eels.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from scatterem.physics.propagation import propagate

from ._walk_graph import walk_graph


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


class _SliceAdvanceMixin:
    """``advance_to`` for a scattering matrix walked slice by slice.

    The host class sets ``S``, ``transmissions``, ``kernel``, ``_slice`` and
    ``_unit_slices`` (from :func:`_unit_transmission_slices`).
    """

    S: Tensor
    transmissions: Tensor
    kernel: Tensor
    _slice: int
    _unit_slices: list

    def advance_to(self, target_slice: int) -> None:
        """Propagate the scattering matrix forward to ``target_slice``.

        A slice whose transmission function is exactly ``1 + 0j`` (see
        :func:`_unit_transmission_slices`) has an identity multiply, so it is
        skipped -- bit-identically, and the multiply is a full pass over the
        ``(Nbeams, Ny, Nx)`` matrix.  The propagation is *not* skipped.

        Parameters
        ----------
        target_slice : int
            Index of the slice to advance the matrix to.
        """
        for j in range(self._slice, target_slice):
            if not self._unit_slices[j]:
                self.S = self.S * self.transmissions[j]
            self.S = propagate(self.S, self.kernel)
        self._slice = target_slice


def _slice_exit_wave(
    waves: Tensor,
    transmissions: Tensor,
    kernel: Tensor,
    start: int,
    unit_slices: Sequence[bool] | None = None,
    *,
    consume: bool = False,
    spectrum: bool = False,
) -> Tensor:
    """Propagate ``waves`` through slices ``start..NZ-1`` (transmit + propagate).

    ``spectrum=True`` returns ``fft2`` of the exit wave instead of the wave, and
    ``consume=True`` says the caller hands ``waves`` over and never reads it again.
    Together they bound the caller's live set to ONE stack: the blocked walk below
    consumes each block of ``waves`` before it writes that block's result, so the
    result can go back into ``waves``'s own storage, and the forward transform runs
    per block while the block is L2-resident rather than as a second full-size
    stack afterwards.  Both are bit-exact: the per-block transforms run at exactly
    the two batch counts :func:`_cufft_batch_bit_invariant` has just verified, and
    the write-back changes only the destination of a copy.  On
    ``r7_eels/wide/hybrid`` the input and the walk's ``out`` were two 1152 MiB
    blocks live at the row's peak, and ``fft2`` of the result was two more.  The
    dense walk ignores ``consume`` except to drop its own reference before the
    transform.

    ``unit_slices[j]`` marks a slice whose transmission function is exactly
    ``1 + 0j`` (see :func:`_unit_transmission_slices`); its multiply is the
    identity and is skipped.  The propagation is *not* skipped -- composing two
    Fresnel kernels is not the same float as applying them either side of an
    FFT round trip, so that would not be bit-exact.

    The transmission multiply is a read-modify-write on a buffer this function
    already owns, from the second slice onward: ``propagate`` hands back a
    freshly allocated stack, so ``out * transmissions[j]`` was allocating a
    second full-size copy of it and then dropping the first.  On
    ``r7_eels/wide/conventional`` that stack is 1152 MiB.  ``mul_`` is the same
    kernel with the same rounding, one pass cheaper, and it never touches
    ``waves`` itself -- ``owned`` starts False, so the first multiply is still
    the out-of-place one that allocates.

    The FFT round trip inside ``propagate`` cannot be folded the same way; see
    its docstring for the measurement.

    The slice walk is L2-BLOCKED over the plane axis when it pays (see
    :func:`_exit_block_planes`): planes are independent of one another all the way
    to the caller's incoherent sum, so running a block of them through the whole
    chain before starting the next block is the same arithmetic in a different
    order of *launches*, and it keeps every intermediate of every step inside this
    card's L2 instead of round-tripping it to DRAM.  The transmission multiply is
    elementwise and therefore exact under any blocking; **cuFFT is not**, and the
    difference is checked at run time rather than argued -- see
    :func:`_cufft_batch_bit_invariant`.
    """
    B = _exit_block_planes(waves, transmissions, kernel, start)
    if B:
        ny, nx = int(waves.shape[-2]), int(waves.shape[-1])
        flat = waves.reshape(-1, ny, nx)
        n = int(flat.shape[0])
        if _cufft_batch_bit_invariant(flat, B):
            # Full blocks replay one captured block walk (``_walk_graph``);
            # its gate guarantees the walk keeps ``flat``'s dtype.
            dense = _slice_exit_wave_dense
            walk = walk_graph(
                flat, transmissions, kernel, start, unit_slices, B, spectrum, dense
            )
            first = None
            if walk is None:
                first = _slice_exit_wave_dense(
                    flat[:B], transmissions, kernel, start, unit_slices
                )
            out_dtype = flat.dtype if first is None else first.dtype
            # Same dtype as ``waves``: the verdict above covers the per-block
            # transforms, and writing back into ``waves`` cannot round.
            same = out_dtype == flat.dtype
            fold = spectrum and same
            if consume and same and waves.is_contiguous():
                out = flat
            else:
                out = torch.empty((n, ny, nx), dtype=out_dtype, device=flat.device)
            i0 = B
            if first is not None:
                out[:B] = torch.fft.fft2(first, dim=(-2, -1)) if fold else first
                del first
            elif walk is not None:
                i0 = walk.run(flat, transmissions, kernel, out)
            for i in range(i0, n, B):
                blk = _slice_exit_wave_dense(
                    flat[i : i + B], transmissions, kernel, start, unit_slices
                )
                out[i : i + B] = torch.fft.fft2(blk, dim=(-2, -1)) if fold else blk
                del blk
            out = out.reshape(waves.shape[:-2] + (ny, nx))
            if spectrum and not fold:
                out = torch.fft.fft2(out, dim=(-2, -1))
            return out
        del flat
    out = _slice_exit_wave_dense(waves, transmissions, kernel, start, unit_slices)
    if not spectrum:
        return out
    if consume:
        del waves
    return torch.fft.fft2(out, dim=(-2, -1))


#: verdicts of :func:`_cufft_batch_bit_invariant`, keyed by
#: ``(device, dtype, ny, nx, block, n)``.  Values only; nothing here is a tensor.
_CUFFT_BATCH_EXACT: dict = {}


def _cufft_batch_bit_invariant(flat: Tensor, block: int) -> bool:
    """Does cuFFT return the *same bits* at batch ``block`` as at the full batch?

    It usually does and sometimes does not, and which is which is not predictable
    from the shape -- cuFFT selects a kernel partly from the batch count, so a
    blocked walk can silently land on a different algorithm.  Swept over grid
    edge at ``n = 1728`` planes with the block this module would choose
    (a one-off probe): edges 96 / 192 / 256 / 384 / 512 are **ndiff 0**,
    and edge **128 is not** -- 26 495 162 of 28 311 552 words differ on a single
    ``fft2``, 5.184e-07 of peak.  That is small, but ``r7_eels``'s contract on
    every one of its six variants is ``cube: sha256``, so small is still a
    failure; and 128 is exactly the ``small`` grid, i.e. the size the correctness
    gate runs at.  Hence a measured gate rather than a rule: the property is a
    property of *this* cuFFT on *this* device at *this* shape, so it is asked, not
    assumed, and the verdict is memoised.

    Asked of both transforms (the forward and inverse plans are chosen
    independently) and of both batch counts the walk actually uses -- ``block``
    and, when it is non-zero, the remainder ``n % block``.  One full-batch
    transform is paid per key, once; every later call reads the memo.
    """
    n = int(flat.shape[0])
    key = (
        flat.device.index,
        flat.dtype,
        int(flat.shape[-2]),
        int(flat.shape[-1]),
        int(block),
        n,
    )
    verdict = _CUFFT_BATCH_EXACT.get(key)
    if verdict is None:
        batches = [block] + ([n % block] if n % block else [])
        verdict = True
        for fn, kw in (
            (torch.fft.fft2, {}),
            (torch.fft.ifft2, {"norm": "forward"}),
        ):
            full = fn(flat, dim=(-2, -1), **kw)
            for b in batches:
                verdict = verdict and bool(
                    torch.equal(fn(flat[:b], dim=(-2, -1), **kw), full[:b])
                )
            del full
        _CUFFT_BATCH_EXACT[key] = verdict
    return verdict


def _exit_block_planes(
    waves: Tensor,
    transmissions: Tensor,
    kernel: Tensor,
    start: int,
) -> int:
    """Planes per L2 block for :func:`_slice_exit_wave`, or 0 to walk the whole stack.

    The exit-leg walk is five DRAM passes per slice (two cuFFT kernels, the
    propagator multiply, two more cuFFT kernels) over a stack that is 486 MiB at
    ``r7_eels/medium/hybrid`` -- 5095 MB of traffic per step for 1019 MB of
    compulsory traffic, every leg measured at 678-684 GB/s against a same-process
    678.4 GB/s copy roof, i.e. *at* the roof and reducible only by moving fewer
    bytes.  A block small enough to sit in L2 does not round-trip its
    intermediates at all, so the whole chain costs one read and one write.

    The threshold is a RESOURCE, not a fitted constant: three quarters of this
    device's L2, which is the largest working set that still leaves room for the
    two buffers a cuFFT pass has live at once.  Measured on an RTX A6000 (6 MiB
    L2, 294 912 B planes, so ``B`` = 16) over a 1728-plane stack, blocked vs
    whole-stack, 7-step chain: B = 4 / 6 / 8 / 12 / 16 / 24 / 32 / 48 / 64 reads
    0.49 / 0.74 / 0.98 / 1.31 / **1.33** / 1.04 / 0.88 / 0.90 / 0.91x.  The cliff
    is exactly where the block stops fitting (B = 24 is 6.75 MiB against a 6 MiB
    L2) and the fall-off below it is launch latency on blocks too small to fill
    the card -- so a device with a different L2 gets a different ``B`` from the
    same expression rather than inheriting this card's.

    Refusals: a stack that is one block or less (the write-back into the
    contiguous result would be a pure extra pass with nothing to repay it), a
    non-CUDA device (the argument is about L2 and ``L2_cache_size`` is a CUDA
    property), and grad (under autograd every intermediate is retained to be read
    again in the backward, so nothing is L2-resident when it matters, and the
    blocked walk would only multiply the graph).  The chain length is NOT a
    refusal: the blocked form wins at every length measured, 1.06x at one step
    rising to 1.33x at seven.
    """
    if not waves.is_cuda or waves.ndim < 3:
        return 0
    if torch.is_grad_enabled() and (
        waves.requires_grad or transmissions.requires_grad or kernel.requires_grad
    ):
        return 0
    if start >= int(transmissions.shape[0]):
        return 0
    ny, nx = int(waves.shape[-2]), int(waves.shape[-1])
    plane_bytes = ny * nx * waves.element_size()
    if plane_bytes <= 0:
        return 0
    l2 = int(torch.cuda.get_device_properties(waves.device).L2_cache_size)
    block = (3 * l2) // (4 * plane_bytes)
    if block < 2:
        return 0
    n = waves.numel() // (ny * nx)
    return int(block) if n > block else 0


def _slice_exit_wave_dense(
    waves: Tensor,
    transmissions: Tensor,
    kernel: Tensor,
    start: int,
    unit_slices: Sequence[bool] | None = None,
) -> Tensor:
    """The whole-stack slice walk; see :func:`_slice_exit_wave` for the contract."""
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
            stepped = propagate(out, kernel)
            # `propagate` returns its own argument on the large-3-D in-place
            # branch, where the buffer is NOT ours; identity is the exact test.
            owned = owned or stepped is not out
            out = stepped
    return out
