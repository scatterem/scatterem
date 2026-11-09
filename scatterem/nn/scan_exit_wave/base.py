"""Abstract base for scan-time exit-wave operators."""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from scatterem.nn.modules import Propagator
from scatterem.physics.periodic_sampling import apply_pass_band


class ScanExitWaveOperator(nn.Module, ABC):
    """Map a transmission function, probe and scan positions to exit waves.

    Two flavours of ``T`` input, decided by ``wants_full_fov``:

    * ``wants_full_fov = False`` -- ``T`` is per-position patches of shape
      ``(NZ_bin, K, M1, M2)``; the operator runs an inner multislice-style
      loop over scan positions. This is the default multislice path.

    * ``wants_full_fov = True`` -- ``T`` is the full-FOV transmission stack
      of shape ``(NZ_bin, NY, NX)``; the operator builds an S-matrix from
      it (via plane-wave multislice) and projects to scan positions
      cheaply. This is the PRISM / partitioned-PRISM path.

    The probe is always given in real space ``(Nmodes, K, M1, M2)``;
    operators that need the Fourier-aperture form FFT internally.

    ``exit_antialias_aperture`` (default ``None``, and ``None`` on every
    reconstruction path) is the wave-side half of a band-limited forward that
    the Fresnel propagator cannot carry: the aperture applied once more after
    the last transmission, which the propagator only does BETWEEN slices.
    Each operator applies it through :meth:`_bandlimit_exit` at its own exit
    point -- the multislice loop on the exit wave, the PRISM family on the
    finished S-matrix columns -- on the grid it hands the field back on.

    Parameters
    ----------
    *args : Any
        Forwarded to ``torch.nn.Module.__init__``.
    **kwargs : Any
        Forwarded to ``torch.nn.Module.__init__``.

    Attributes
    ----------
    exit_antialias_aperture : torch.Tensor or None
        Real ``(Ny, Nx)`` antialiasing aperture in FFT layout on the operator's
        output grid, or ``None``.  A plain attribute, not a registered buffer,
        so it stays out of ``state_dict()`` and checkpoints written before it
        existed still load; being a plain attribute it is also NOT moved by
        ``.to(device)`` -- :meth:`_bandlimit_exit` moves it per call instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.exit_antialias_aperture: Tensor | None = None

    def _bandlimit_exit(self, waves: Tensor) -> Tensor:
        """Apply ``exit_antialias_aperture`` once, or hand ``waves`` back as is.

        Parameters
        ----------
        waves : torch.Tensor
            Complex ``(..., Ny, Nx)`` field on the operator's output grid.

        Returns
        -------
        torch.Tensor
            The same object when no aperture is attached (so the default path
            stays byte-identical), else the band-limited field.
        """
        if self.exit_antialias_aperture is None:
            return waves
        return apply_pass_band(waves, self.exit_antialias_aperture)

    @property
    @abstractmethod
    def wants_full_fov(self) -> bool:
        """Whether ``T`` must be the full-FOV transmission stack.

        Returns
        -------
        bool
            True for the PRISM / partitioned-PRISM path (``T`` is
            ``(NZ_bin, NY, NX)``); False for the default multislice path
            (``T`` is per-position patches ``(NZ_bin, K, M1, M2)``).
        """
        ...

    def needs_transmission(self) -> bool:
        """Whether the next :meth:`forward` will read ``T``.

        True by default.  An operator that keeps per-configuration state across
        the scan chunks of one specimen configuration (the PRISM family's
        forward-chunk cache) answers False once that state is built, and the
        caller then skips sampling the transmission altogether and passes
        ``None``: building an ``(NZ, N, N)`` stack per chunk only for the
        operator to discard it was 30-40 % of a fast PRISM run.

        Returns
        -------
        bool
            False only when ``T`` would be ignored.
        """
        return True

    @abstractmethod
    def forward(
        self,
        T: Tensor,
        probe: Tensor,
        positions: Tensor,
        dr: Tensor,
        propagator: Propagator,
        inverse_index: Tensor | None = None,
    ) -> Tensor:
        """Compute real-space exit waves for a batch of scan positions.

        Parameters
        ----------
        T : torch.Tensor
            Transmission tensor; shape depends on ``wants_full_fov``.
        probe : torch.Tensor
            Real-space probe, shape ``(Nmodes, K, M1, M2)``.
        positions : torch.Tensor
            Integer scan positions, shape ``(K, 2)``.
        dr : torch.Tensor
            Sub-pixel scan offsets, shape ``(K, 2)``.
        propagator : Propagator
            Fresnel propagator for the current slice thickness.
        inverse_index : torch.Tensor, optional
            Per-position -> unique-pixel inverse index, shape ``(K,)`` int64.
            When provided, the Warp ``fused_transmit`` path can use it to
            avoid a gather/broadcast round-trip (each unique pixel is only
            transmitted once). None means use the plain gather/broadcast
            path. Operators that do not support the Warp path (e.g.
            ``PrismScanOp``) ignore this argument.

        Returns
        -------
        torch.Tensor
            Complex, real-space exit waves of shape ``(Nmodes, K, M1, M2)``.
        """
        ...


__all__ = [
    "ScanExitWaveOperator",
]
