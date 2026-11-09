"""Slice-segment gradient checkpointing for the multislice torch chain.

The eager slice loop keeps every slice's activations for the backward, so its
per-position memory grows as ``O(N_sl)`` waves (about 1.4 complex
``(modes, P, P)`` waves per slice, spec 3 section 4.3). Here the loop runs in
segments of ``every`` slices under :func:`torch.utils.checkpoint.checkpoint`
(non-reentrant): only the wave entering each segment is kept, and a segment's
activations are recomputed when the backward reaches it. Peak activation memory
is then about ``N_sl / every`` boundary waves plus one segment's activations
(``O(N_sl / s + s)``, smallest near ``s = sqrt(N_sl)``), at the price of one
extra forward of the slice loop.

The recomputation runs the forward
again from a stored wave, so it is exact for absorptive (non-unitary)
transmissions too. The recomputed forward is the same sequence of kernels on the
same inputs, so the forward and the gradients equal the plain loop's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from torch import Tensor
from torch.utils.checkpoint import checkpoint

from scatterem.nn.modules import Propagator
from scatterem.nn.ptychography import multislice_transmit_slice

__all__ = ["checkpointed_slice_loop"]


def _segment(
    waves: Tensor,
    *T_slices: Tensor,
    propagator: Propagator | None,
    propagate_last: bool,
    inverse_index: Tensor | None,
    gather_index: Tensor | None,
) -> Tensor:
    """The plain slice loop over one segment (``propagate_last``: not the end)."""
    n = len(T_slices)
    for i, t in enumerate(T_slices):
        step = propagator if (i < n - 1 or propagate_last) else None
        waves = multislice_transmit_slice(
            t, waves, step, inverse_index=inverse_index, gather_index=gather_index
        )
    return waves


def checkpointed_slice_loop(
    T_slices: Sequence[Tensor],
    waves: Tensor,
    propagator: Propagator | None,
    every: int,
    inverse_index: Tensor | None = None,
    gather_index: Tensor | None = None,
) -> Tensor:
    """The multislice slice loop, checkpointed every ``every`` slices.

    Parameters
    ----------
    T_slices : sequence of Tensor
        Per-slice transmissions, as :class:`MultisliceScanOp` unbinds them.
    waves : Tensor
        ``(modes, K, M1, M2)`` complex entering waves.
    propagator : Propagator or None
        Applied after every slice but the last.
    every : int
        Slices per checkpointed segment (``>= 1``).
    inverse_index, gather_index : Tensor or None, optional
        Passed to :func:`multislice_transmit_slice` unchanged.

    Returns
    -------
    Tensor
        ``(modes, K, M1, M2)`` complex exit waves, equal to the plain loop's.

    Raises
    ------
    ValueError
        ``every < 1``.
    """
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    n = len(T_slices)
    for s0 in range(0, n, every):
        s1 = min(s0 + every, n)
        # No RNG runs inside a segment, so its state need not be stashed.
        # ``checkpoint`` is typed Optional; the segment always returns a Tensor.
        waves = cast(
            Tensor,
            checkpoint(
                _segment,
                waves,
                *T_slices[s0:s1],
                propagator=propagator,
                propagate_last=s1 < n,
                inverse_index=inverse_index,
                gather_index=gather_index,
                use_reentrant=False,
                preserve_rng_state=False,
            ),
        )
    return waves
