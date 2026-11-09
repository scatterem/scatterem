"""MultisliceScanOp: per-position multislice forward operator.

Pure refactor of an earlier multislice slice loop. No new physics.

The per-slice transmission is delegated to
``scatterem.nn.ptychography.multislice_transmit_slice`` so this operator and
other multislice forwards share the EXACT same routine (and
hence produce bit-for-bit identical output for both the Warp fused-transmit
path and the plain broadcast path)."""

import torch
from torch import Tensor

from scatterem.nn.modules import Propagator
from scatterem.nn.ptychography import multislice_transmit_slice
from scatterem.nn.scan_exit_wave._l2_block_graph import l2_blocked_slice_loop
from scatterem.nn.scan_exit_wave._slice_checkpoint import checkpointed_slice_loop
from scatterem.nn.scan_exit_wave.base import ScanExitWaveOperator


class MultisliceScanOp(ScanExitWaveOperator):
    """Standard per-position multislice.

    Two transmission input flavours, selected by ``inverse_index``:

    * ``inverse_index is None`` -- ``T`` is already-gathered per-position
      patches of shape ``(NZ_bin, K, M1, M2)``; each slice is broadcast-
      multiplied into the waves.

    * ``inverse_index`` given -- ``T`` is the *ungathered* per-unique-pixel
      (exponentiated) potential stack of shape ``(NZ_bin, U)``; the gather +
      complex multiply is fused per slice via the Warp ``fused_transmit``
      kernel.

    In both cases inter-slice Fresnel propagation is applied after every slice
    except the last (matching the model).

    ``exit_antialias_aperture`` (inherited; default ``None``, and ``None`` on
    every reconstruction path) optionally applies an antialiasing aperture once
    to the exit wave; see :meth:`forward`.

    ``checkpoint_every`` (default 0, off) checkpoints the slice loop in segments
    of that many slices when a gradient is recorded
    (``checkpointed_slice_loop`` in ``_slice_checkpoint``): ``O(N_sl / s + s)``
    activation memory for one extra forward, with the same output and gradients.
    """

    #: Slices per checkpointed segment under autograd; 0 runs the plain loop.
    checkpoint_every: int = 0

    @property
    def wants_full_fov(self) -> bool:
        """Whether the operator has to be handed the whole object field of view.

        Returns
        -------
        bool
            Always ``False``. Multislice propagates one probe window per scan
            position, so the caller may pass the cut-out patches.
        """
        return False

    def __init__(self) -> None:
        # Explicit no-argument constructor: the operator has no options, and
        # the documented signature stays empty (the base class takes *args).
        super().__init__()

    def forward(
        self,
        T: Tensor,  # (NZ_bin, K, M1, M2) gathered, or (NZ_bin, U) with an index
        probe: Tensor,  # (Nmodes, K, M1, M2) complex
        positions: Tensor,  # (K, 2) int — unused here; carried for ABC parity
        dr: Tensor,  # (K, 2) float — unused here; sub-pixel shift handled upstream
        propagator: Propagator,
        inverse_index: Tensor | None = None,  # (K, ph, pw) int — Warp fused path
        gather_index: Tensor | None = None,  # (K, ph, pw) int — torch route
    ) -> Tensor:
        """Multislice-propagate the probe through the transmission stack.

        Parameters
        ----------
        T : torch.Tensor
            Transmission, either already-gathered per-position patches
            ``(NZ_bin, K, M1, M2)`` (when ``inverse_index`` is None) or the
            ungathered per-unique-pixel stack ``(NZ_bin, U)`` (when
            ``inverse_index`` is given).
        probe : torch.Tensor
            ``(Nmodes, K, M1, M2)`` complex incident probe.
        positions : torch.Tensor
            ``(K, 2)`` int scan positions; unused here, carried for ABC parity.
        dr : torch.Tensor
            ``(K, 2)`` float sub-pixel offsets; unused here, handled upstream.
        propagator : Propagator
            Fresnel propagator applied between slices.
        inverse_index : torch.Tensor or None, optional
            ``(K, ph, pw)`` int gather indices for the Warp fused-transmit
            path; None selects the already-gathered flavour of ``T``.
        gather_index : torch.Tensor or None, optional
            The same indices for the torch route: ``T`` is then ungathered too and
            each slice is expanded as it is used, so the ``(NZ_bin, K, M1, M2)``
            copy of the stack never exists (2.75x the fused route's peak at 1024
            probes when it did).

        Returns
        -------
        torch.Tensor
            ``(Nmodes, K, M1, M2)`` complex exit waves.
        """
        del positions, dr  # used by other operators in the family
        # a stack larger than L2 runs blocks of positions
        # through every slice under one replayed CUDA graph, bit-identical
        # (``_l2_block_graph``; the ``multislice_l2_graph`` / ``gather_l2_graph``
        # backend rows).
        blocked = l2_blocked_slice_loop(
            self, T, probe, propagator, inverse_index, gather_index
        )
        if blocked is not None:
            return self._bandlimit_exit(blocked)
        waves = probe
        NZ_bin = T.shape[0]
        # Unbind ONCE here rather than indexing ``T[slice_id]`` inside the loop.
        # ``torch.unbind(T, 0)[i]`` IS the view ``T[i]`` -- same storage, strides and
        # values -- so the forward is bit-identical. The difference is in the BACKWARD:
        # ``T`` is a function of the object Parameter, and indexing a ``requires_grad``
        # stack per slice emits one select-backward PER SLICE, each of which allocates a
        # zero tensor of the WHOLE stack and then has the engine accumulate the whole
        # stack, to deliver one ``(K, M1, M2)`` plane of gradient. Measured on the
        # gathered flavour at ``NZ_bin=8, K=256, 128^2`` (a 256 MiB stack): 8 ``fill_``
        # + 8 slice ``copy_`` + 7 full-stack ``add_`` = 12.1 ms, 19 % of a 63.9 ms
        # training step. A single unbind-backward replaces all of it with one ``stack``
        # of the slice gradients (0.8 ms), measured 1.22x on the step. Every element
        # still receives exactly one contribution, so the gradient is unchanged -- the
        # removed adds were adding zeros. NOTE this is the flavour taken when
        # ``SCATTEREM_USE_WARP`` is unset, i.e. the default; the fused path's stack is
        # ``(NZ_bin, U)`` and small, where the same fix is worth 0.06 ms.
        T_slices = torch.unbind(T, 0)
        if self.checkpoint_every > 0 and torch.is_grad_enabled():
            waves = checkpointed_slice_loop(
                T_slices,
                waves,
                propagator,
                self.checkpoint_every,
                inverse_index=inverse_index,
                gather_index=gather_index,
            )
            return self._bandlimit_exit(waves)
        for slice_id in range(NZ_bin):
            # Skip inter-slice propagation on the last slice (matches the model).
            propagator_or_none = propagator if slice_id < NZ_bin - 1 else None
            waves = multislice_transmit_slice(
                T_slices[slice_id],
                waves,
                propagator_or_none,
                inverse_index=inverse_index,
                gather_index=gather_index,
            )
        # A band-limited multislice masks the wave after EVERY slice, the last
        # one included -- one more masked propagation than the loop above
        # performs.  Applying the aperture ALONE is equivalent for
        # ``|FFT(psi)|^2``: the two differ only by the unit-modulus Fresnel
        # factor, which cancels in the intensity, and this keeps the exit-wave
        # reference plane AT the last slice.  A separate channel from the
        # propagator's aperture because at ``NZ == 1`` there is no propagator
        # to carry one.  Identity (the same object) when no aperture is set.
        return self._bandlimit_exit(waves)


__all__ = [
    "MultisliceScanOp",
]
