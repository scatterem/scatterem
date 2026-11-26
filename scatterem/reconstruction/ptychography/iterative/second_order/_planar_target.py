"""The :class:`~.target.SolveTarget` methods of :class:`~.planar.PlanarPtychography`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from scatterem.optim.second_order.log import BHLog

from .target import SolveHooks, fixed_batch_list


class PlanarSolveTarget:
    """Mixin: the second-order solve interface of the 2D model.

    The host provides ``object``, ``probe``, ``offsets``, ``device`` and
    ``loss(indices)``.
    """

    object: Tensor
    offsets: Tensor

    @property
    def probe(self) -> Tensor:
        """The probe modes (provided by the host)."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The device of the state (provided by the host)."""
        raise NotImplementedError

    def loss(self, indices: Tensor) -> Tensor:
        """The loss of the positions ``indices`` (provided by the host)."""
        raise NotImplementedError

    def check_options(
        self,
        *,
        heldout: Any,
        object_prior: Any,
        optimize_object: bool,
        compile_forward: bool,
    ) -> None:
        """Refuse the options the 2D model does not support.

        Parameters
        ----------
        heldout, object_prior : object or None
            Held-out scoring and object prior; not supported.
        optimize_object : bool
            Unused.
        compile_forward : bool
            Not supported.

        Raises
        ------
        NotImplementedError
            If `heldout`, `object_prior` or `compile_forward` is given.
        """
        del optimize_object
        unsupported = {
            "heldout": heldout is not None,
            "object_prior": object_prior is not None,
            "compile_forward": compile_forward,
        }
        names = [name for name, given in unsupported.items() if given]
        if names:
            raise NotImplementedError(
                f"PlanarPtychography does not support {', '.join(names)}"
            )

    def solve_batches(self, batches: Iterable, compile_forward: bool) -> list:
        """The batches as int64 index tensors on the model's device.

        Parameters
        ----------
        batches : Iterable
            Index tensors or sequences, for example :meth:`batches`.
        compile_forward : bool
            Unused (refused by :meth:`check_options`).

        Returns
        -------
        list of Tensor
            The fixed batch list.

        Raises
        ------
        ValueError
            If `batches` is empty.
        """
        del compile_forward

        def convert(batch: Any) -> Tensor:
            return torch.as_tensor(batch, dtype=torch.int64).to(self.device)

        return fixed_batch_list(batches, convert)

    def object_parameters(self) -> list[Tensor]:
        """The object potential.

        Returns
        -------
        list of Tensor
            ``[object]``, the ``(H, W)`` complex potential.
        """
        return [self.object]

    def probe_parameters(self) -> list[Tensor]:
        """The probe modes.

        Returns
        -------
        list of Tensor
            ``[probe]``, the ``(M, 1, N, N)`` modes.
        """
        return [self.probe]

    def position_parameters(self) -> list[Tensor]:
        """The scan-position offsets.

        Returns
        -------
        list of Tensor
            ``[offsets]``, the ``(K, 2)`` offsets in pixels.
        """
        return [self.offsets]

    def grad_gates(self) -> dict[str, bool] | None:
        """The gradient gates of the model.

        Returns
        -------
        None
            The 2D model has none.
        """
        return None

    def solve_hooks(
        self,
        fixed: list,
        heldout: Any,
        object_prior: Any,
        callback: Callable[[int, BHLog], bool | None] | None,
    ) -> SolveHooks:
        """No score and no step hook; the callback unchanged.

        Parameters
        ----------
        fixed : list
            The fixed batch list (unused).
        heldout, object_prior : object or None
            Unused (refused by :meth:`check_options`).
        callback : callable or None
            The driver's pass callback.

        Returns
        -------
        tuple
            ``(None, None, callback)``.
        """
        del fixed, heldout, object_prior
        return None, None, callback

    def batch_loss(self, batch: Tensor) -> Tensor:
        """The amplitude loss of one batch (:meth:`loss`).

        Parameters
        ----------
        batch : Tensor
            1D int64 scan-position indices.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        return self.loss(batch)

    def finish_solve(self, heldout: Any, log: BHLog) -> None:
        """Nothing to tidy up.

        Parameters
        ----------
        heldout : object or None
            Unused.
        log : BHLog
            The solve log (unused).
        """
        del heldout, log
