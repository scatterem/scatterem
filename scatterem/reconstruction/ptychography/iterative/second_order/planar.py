"""2D mixed-state ptychography for the second-order driver, with no trainer.

:class:`PlanarPtychography` holds the state of one 2D scan (mixed-state probe
modes, a single-slice object potential, the scan-position offsets and the
measured amplitudes) and its forward model. It implements
:class:`~.target.SolveTarget`, so ``solve_ptychography(model, model.batches(512),
solver="secant_gd")`` reconstructs it directly.

Forward model, per scan position ``k`` (the same physics as the unified
``Ptychography`` module for one scan, one slice and the lab frame, checked
in the tests to the bit):

1. each probe mode is shifted by the sub-pixel offset ``dr_k`` (a Fourier
   phase ramp);
2. it is multiplied by the transmission ``exp(i V)`` of the object window at
   the integer corner of position ``k``; ``Re V`` is the phase and ``Im V``
   the absorption;
3. the far field is ``fft2`` (orthonormal), centred by a ``(-1)^(y + x)``
   ramp on the window;
4. the amplitude is ``sqrt(sum_modes |far field|^2 + eps)``, cropped to the
   recorded detector when the model grid is larger (``model_grid``).

The loss of a batch is the normalised amplitude RMSE
(:func:`~scatterem.nn.losses.normalized_amplitude_rmse_loss`), over the pixels of the
detector mask when one is set (the needle band and the pixels without data
out).

A batch is a 1D tensor of scan-position indices: the 2D model has one scan,
so it needs no tilt index.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from torch import Tensor, nn

from scatterem.nn.losses import (
    masked_normalized_amplitude_rmse_loss,
    normalized_amplitude_rmse_loss,
)
from scatterem.nn.modules.mixed_probe_modes import make_mixed_probe

from ._planar_geometry import normalise_probe, pad_probe, scan_geometry
from ._planar_target import PlanarSolveTarget
from .planar_constraints import PlanarConstraints

logger = logging.getLogger(__name__)

__all__ = ["PlanarPtychography"]

_EPS = 1e-10


class PlanarPtychography(PlanarSolveTarget, nn.Module):
    """2D mixed-state ptychography model and second-order solve target.

    Parameters
    ----------
    amplitudes : Tensor
        ``(K, My, Mx)`` measured amplitudes (square roots of the counts),
        centred. They stay on their device; they are not in ``state_dict``.
    positions : Tensor
        ``(K, 2)`` integer top-left object pixels of the probe windows.
    offsets : Tensor
        ``(K, 2)`` sub-pixel offsets ``dr`` in pixels; refined with
        ``optimize_positions``.
    probe : Tensor
        ``(M, 1, N, N)`` complex real-space probe modes on the model grid,
        ``N >= My``.
    object_shape : tuple of int
        ``(H, W)`` of the object potential; it starts at zero (vacuum).
    norm : float
        The measurement norm, the power the probe was normalised to.
    detector_mask : Tensor or None, optional
        ``(My, Mx)`` bool, True for a pixel in the loss. Default None (all).
    constraints : PlanarConstraints or None, optional
        The constraints of :meth:`apply_pass_constraints`. Default none.
    """

    def __init__(
        self,
        amplitudes: Tensor,
        positions: Tensor,
        offsets: Tensor,
        probe: Tensor,
        object_shape: tuple[int, int],
        norm: float,
        detector_mask: Tensor | None = None,
        constraints: PlanarConstraints | None = None,
    ) -> None:
        super().__init__()
        device = amplitudes.device
        if probe.ndim == 3:
            probe = probe[:, None]
        n = int(probe.shape[-1])
        if n < amplitudes.shape[-1] or probe.shape[-2] != n:
            raise ValueError(
                f"probe grid {tuple(probe.shape[-2:])} smaller than the "
                f"detector {tuple(amplitudes.shape[-2:])} or not square"
            )
        self.amplitudes = amplitudes
        self.positions = positions.to(device=device, dtype=torch.int64)
        self.norm = float(norm)
        self.probe_model = nn.ParameterList(
            [nn.Parameter(probe.to(device=device, dtype=torch.complex64))]
        )
        self.object = nn.Parameter(
            torch.zeros(object_shape, dtype=torch.complex64, device=device)
        )
        self.offsets = nn.Parameter(offsets.to(device=device, dtype=torch.float32))
        self.constraints = constraints or PlanarConstraints()
        self.detector_mask: Tensor | None = None
        if detector_mask is not None:
            self.set_detector_mask(detector_mask)
        q = torch.fft.fftfreq(n, device=device)
        self._qy, self._qx = q[:, None], q[None, :]
        self._window = torch.arange(n, device=device)

    @classmethod
    def from_scan(
        cls,
        intensities: Tensor,
        scan_step: float,
        dk: float,
        probe: Tensor,
        *,
        rotation_deg: float = 0.0,
        model_grid: int | None = None,
        detector_mask: Tensor | None = None,
        constraints: PlanarConstraints | None = None,
        in_place: bool = False,
    ) -> PlanarPtychography:
        """Build the model from a 4D intensity stack, as ``Ptychography`` would.

        The measurement norm is the sum of the mean intensity pattern. The
        probe is padded to the model grid in the aperture plane and scaled to
        that norm. The geometry follows :func:`._planar_geometry.scan_geometry`.

        Parameters
        ----------
        intensities : Tensor
            ``(Ny, Nx, My, Mx)`` counts, centred, on the device of the
            model.
        scan_step : float
            Scan step in angstroms.
        dk : float
            Detector pixel in inverse angstroms.
        probe : Tensor
            ``(M, 1, n, n)`` or ``(M, n, n)`` complex real-space modes,
            ``n <= model_grid``.
        rotation_deg : float, optional
            Scan rotation in degrees. Default 0.
        model_grid : int or None, optional
            Side of the model's far-field grid (``detector_pad_to``); None
            uses the detector size.
        detector_mask, constraints : optional
            As for the constructor.
        in_place : bool, optional
            Take the square roots in ``intensities`` itself when it is
            float32 (no copy of a large stack; the caller's tensor then holds
            amplitudes). Default False.

        Returns
        -------
        PlanarPtychography
            The model with a vacuum object.
        """
        ny, nx, my, _mx = intensities.shape
        grid = int(model_grid or my)
        amplitudes = intensities.to(torch.float32)
        if amplitudes is intensities and not in_place:
            amplitudes = amplitudes.clone()
        norm = float(amplitudes.mean((0, 1)).sum())
        amplitudes = amplitudes.reshape(ny * nx, my, -1).sqrt_()
        device = amplitudes.device
        positions, offsets, shape = scan_geometry(
            (ny, nx), scan_step, dk, grid, rotation_deg, device
        )
        stack = probe if probe.ndim == 4 else probe[:, None]
        stack = normalise_probe(pad_probe(stack.to(device), grid), norm)
        return cls(
            amplitudes,
            positions,
            offsets,
            stack,
            shape,
            norm,
            detector_mask=detector_mask,
            constraints=constraints,
        )

    # ------------------------------------------------------------------ state
    @property
    def model(self) -> PlanarPtychography:
        """The model itself (``Ptychography`` keeps its forward in ``.model``)."""
        return self

    @property
    def probe(self) -> Tensor:
        """``(M, 1, N, N)`` probe modes (the parameter)."""
        return self.probe_model[0]

    @property
    def device(self) -> torch.device:
        """Device of the state and the data."""
        return self.object.device

    def set_detector_mask(self, mask: Tensor | None) -> None:
        """Set the pixels of the loss.

        Parameters
        ----------
        mask : Tensor or None
            ``(My, Mx)`` bool, True for a pixel in the loss; None for all.

        Raises
        ------
        ValueError
            If the shape is not the detector's, or no pixel is left.
        """
        if mask is None:
            self.detector_mask = None
            return
        mask = torch.as_tensor(mask, device=self.amplitudes.device).bool()
        if tuple(mask.shape) != tuple(self.amplitudes.shape[-2:]):
            raise ValueError(
                f"detector mask {tuple(mask.shape)} != detector "
                f"{tuple(self.amplitudes.shape[-2:])}"
            )
        if not bool(mask.any()):
            raise ValueError("the detector mask leaves no pixel in the loss")
        logger.info(
            "PlanarPtychography: %d of %d detector pixels in the loss",
            int(mask.sum()),
            mask.numel(),
        )
        self.detector_mask = mask

    def batches(self, batch_size: int, seed: int = 0) -> list[Tensor]:
        """A random partition of the scan positions into batches.

        Parameters
        ----------
        batch_size : int
            Positions per batch; the last batch may be smaller.
        seed : int, optional
            Seed of the permutation. Default 0.

        Returns
        -------
        list of Tensor
            The int64 index tensors; every position is in exactly one batch.
        """
        gen = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(self.amplitudes.shape[0], generator=gen)
        return list(order.split(int(batch_size)))

    # ---------------------------------------------------------------- forward
    def forward(self, indices: Tensor) -> Tensor:
        """Model amplitudes of the scan positions ``indices``.

        Parameters
        ----------
        indices : Tensor
            1D int64 scan-position indices.

        Returns
        -------
        Tensor
            ``(K, My, Mx)`` float32 amplitudes, centred.
        """
        probe = self.probe
        n = probe.shape[-1]
        dr = self.offsets[indices]
        theta = (
            -2
            * math.pi
            * (self._qx * dr[:, 1, None, None] + self._qy * dr[:, 0, None, None])
        )
        spectrum = torch.fft.fft2(probe, norm="ortho") * torch.exp(1j * theta)
        waves = torch.fft.ifft2(spectrum, norm="ortho")  # (M, K, N, N)
        corner = self.positions[indices]
        iy = corner[:, 0, None, None] + self._window[None, :, None]
        ix = corner[:, 1, None, None] + self._window[None, None, :]
        transmission = torch.exp(1j * self.object[iy, ix])
        exit_waves = (
            transmission * _centring_ramp(iy, ix, n, transmission.dtype) * waves
        )
        far = torch.fft.fft2(exit_waves, norm="ortho")
        amplitude = ((far.abs() ** 2).sum(0) + _EPS).sqrt()
        my, mx = self.amplitudes.shape[-2:]
        if (my, mx) != (n, n):
            oy, ox = n // 2 - my // 2, n // 2 - mx // 2
            amplitude = amplitude[:, oy : oy + my, ox : ox + mx]
        return amplitude

    def loss(self, indices: Tensor) -> Tensor:
        """The amplitude loss of the scan positions ``indices``.

        Parameters
        ----------
        indices : Tensor
            1D int64 scan-position indices.

        Returns
        -------
        Tensor
            Scalar amplitude RMSE loss, over the detector mask if set.
        """
        pred = self(indices)
        data = self.amplitudes[indices]
        if self.detector_mask is None:
            return normalized_amplitude_rmse_loss(pred, data)
        valid = self.detector_mask.expand_as(pred)
        return masked_normalized_amplitude_rmse_loss(pred, data, valid)

    # ------------------------------------------------------------ constraints
    def apply_pass_constraints(self) -> None:
        """Apply :attr:`constraints` in place (after an accepted pass)."""
        self.constraints.apply(self)

    def expand_modes(
        self, pmodes: int, mode0_frac: float
    ) -> tuple[PlanarPtychography, float]:
        """A ``pmodes``-mode copy of this one-mode state, object rescaled.

        Mode 0 is expanded into Hermite-like modes
        (:func:`~scatterem.nn.modules.mixed_probe_modes.make_mixed_probe`,
        ``mode0_frac`` of the power in mode 0) and scaled to the measurement
        norm. The data fix only the product of probe power and object
        transmission, so the object transmission takes the old power ratio:
        ``V <- V - 0.5 i log(c)``, ``c = sum |P|^2 / norm``. The new model
        shares the data and copies the offsets and the constraints.

        Parameters
        ----------
        pmodes : int
            Number of modes after the expansion, at least 2.
        mode0_frac : float
            Power fraction of mode 0.

        Returns
        -------
        tuple of (PlanarPtychography, float)
            The new model and the power ratio ``c`` of this one.
        """
        with torch.no_grad():
            old = self.probe.detach().cpu().numpy()
            ratio = float((np.abs(old) ** 2).sum()) / self.norm
            pows = [(1.0 - mode0_frac) / (pmodes - 1)] * (pmodes - 1)
            stack = make_mixed_probe(old[0, 0].astype(np.complex128), pmodes, pows)
            new = torch.as_tensor(np.asarray(stack)[:, None], dtype=torch.complex64)
            model = PlanarPtychography(
                self.amplitudes,
                self.positions,
                self.offsets.detach().clone(),
                normalise_probe(new.to(self.device), self.norm),
                (int(self.object.shape[0]), int(self.object.shape[1])),
                self.norm,
                detector_mask=self.detector_mask,
                constraints=self.constraints,
            )
            model.object.copy_(self.object.detach() - 0.5j * math.log(ratio))
        return model, ratio


def _axis_ramp(coords: Tensor, n: int) -> Tensor:
    """``exp(2 pi i (n // 2) c / n)`` per integer coordinate, in complex128."""
    phase = torch.remainder((n // 2) * coords, n).to(torch.float64) * (
        2.0 * math.pi / n
    )
    return torch.polar(torch.ones_like(phase), phase)


def _centring_ramp(iy: Tensor, ix: Tensor, n: int, dtype: torch.dtype) -> Tensor:
    """The ramp that centres the ``fft2`` of an ``n x n`` window on ``n // 2``.

    For even ``n`` it is ``(-1)^(y + x)``, exact in every dtype.
    """
    if n % 2 == 0:
        return (1 - 2 * torch.remainder(iy + ix, 2)).to(dtype)
    return (_axis_ramp(iy, n) * _axis_ramp(ix, n)).to(dtype)
