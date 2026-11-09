"""S-matrix beam/beamlet metadata for (partitioned) PRISM simulation."""

from dataclasses import dataclass

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch import device as torch_device

from scatterem.physics._frequency_grid import fft_frequency_grid
from scatterem.utils.stem.beamlets import natural_neighbor_weights
from scatterem.utils.h5 import h5_dataset, h5_group
from scatterem.utils.stem.fourier import fft_index_grid

__all__ = [
    "SMeta",
]


@dataclass
class SMeta:
    """Beam and beamlet metadata for a (partitioned) PRISM S-matrix.

    Precomputes the Fourier grid, per-beam coordinates/frequencies and
    (optionally) the natural-neighbor beamlet weights that
    :class:`~scatterem.nn.scan_exit_wave.prism.PrismScanOp` uses to build and
    project the S-matrix.

    Parameters
    ----------
    all_beams : numpy.typing.NDArray or torch.Tensor
        Boolean ``(MY, MX)`` mask of every beam inside the aperture.
    parent_beams : numpy.typing.NDArray or torch.Tensor
        Boolean ``(MY, MX)`` mask of the parent-beam subset (``all_beams``
        for pure PRISM).
    numerical_aperture_radius_pixels : int
        Aperture radius, in pixels, on the ``(MY, MX)`` grid.
    dx : numpy.typing.NDArray
        Real-space pixel size per axis.
    M : tuple of int
        Simulation grid shape ``(MY, MX)``.
    device : torch.device
        Device to build the tensors on.
    """

    M: NDArray[np.int32]
    dx: NDArray[np.float32]

    q: Tensor
    qf: Tensor
    q2: Tensor
    qf2: Tensor
    q_coords: Tensor
    r_indices: Tensor
    all_beams: Tensor
    parent_beams: Tensor
    beam_numbers: Tensor
    all_beams_q: Tensor
    all_beams_coords: Tensor
    parent_beams_q: Tensor
    parent_beams_coords: Tensor

    numerical_aperture_radius_pixels: int
    B: int
    Bp: int

    device: torch_device
    natural_neighbor_weights: Tensor | None = None
    beamlets: Tensor | None = None

    def __init__(
        self,
        all_beams: NDArray | Tensor,
        parent_beams: NDArray | Tensor,
        numerical_aperture_radius_pixels: int,
        dx: NDArray[np.float32],
        M: tuple[int, int],
        device: torch_device,
    ) -> None:
        self.dx = dx
        self.device = device
        self.all_beams = torch.as_tensor(all_beams).to(device).bool()
        self.parent_beams = torch.as_tensor(parent_beams).to(device).bool()
        self.numerical_aperture_radius_pixels = numerical_aperture_radius_pixels
        self.B = int(all_beams.sum())
        self.Bp = int(parent_beams.sum())

        MY, MX = M
        self.M = np.array(M)

        self.q = fft_frequency_grid([MY, MX], dx, half_pixel_shift=True, device=device)
        self.qf = fft_frequency_grid([MY, MX], dx, device=device)

        self.q2 = torch.linalg.vector_norm(self.q, dim=0) ** 2
        self.qf2 = torch.linalg.vector_norm(self.qf, dim=0) ** 2
        # Signed FFT index of every pixel: THE beam coordinate (DC at [0, 0]).
        self.q_coords = fft_index_grid((MY, MX), device=device)
        self.r_indices = torch.from_numpy(np.mgrid[:MY, :MX]).to(device)

        self.beam_numbers = (
            torch.ones_like(self.all_beams, dtype=torch.long, device=device) * -1
        )
        self.beam_numbers[self.all_beams] = torch.arange(0, self.B, device=device)

        all_beams_expanded = self.all_beams[None, ...].expand_as(self.q_coords)
        self.all_beams_q = self.qf[all_beams_expanded].reshape(2, self.B).T
        self.all_beams_coords = self.q_coords[all_beams_expanded].reshape(2, self.B).T

        parent_beams_expanded = self.parent_beams[None, ...].expand_as(self.q_coords)
        self.parent_beams_q = self.qf[parent_beams_expanded].reshape(2, self.Bp).T
        self.parent_beams_coords = (
            self.q_coords[parent_beams_expanded].reshape(2, self.Bp).T
        )

    def to_h5(self, file_path: str, key: str = "s_matrix_meta") -> None:
        """Write the constructor inputs to an HDF5 group.

        Parameters
        ----------
        file_path : str
            Path of the HDF5 file to write (opened in append mode).
        key : str, optional
            Group name to write under.
        """
        # Exactly the six inputs `__init__` takes, so `from_h5` can rebuild
        # everything else. `N` (the supersampled real-space grid) and the
        # derived `f = N // M` were commented out of `__init__` before this
        # class was first used and are not stored, so they are not written
        # either; `from_h5` still tolerates them in an older file.
        with h5py.File(file_path, "a") as f:
            g = f.create_group(key)
            g.create_dataset("M", data=self.M)
            g.create_dataset("all_beams", data=self.all_beams.cpu().numpy())
            g.create_dataset("parent_beams", data=self.parent_beams.cpu().numpy())
            g.create_dataset(
                "numerical_aperture_radius_pixels",
                data=self.numerical_aperture_radius_pixels,
            )
            g.create_dataset("dx", data=self.dx)
            g.create_dataset("device", data=str(self.device))

    @staticmethod
    def from_h5(file_path: str, key: str = "s_matrix_meta") -> "SMeta":
        """Rebuild an ``SMeta`` from a group written by :meth:`to_h5`.

        Parameters
        ----------
        file_path : str
            Path of the HDF5 file to read.
        key : str, optional
            Group name to read from.

        Returns
        -------
        SMeta
            The reconstructed metadata.
        """
        with h5py.File(file_path, "r") as f:
            g = h5_group(f, key)
            all_beams = h5_dataset(g, "all_beams")[...]
            parent_beams = h5_dataset(g, "parent_beams")[...]
            numerical_aperture_radius_pixels = int(
                h5_dataset(g, "numerical_aperture_radius_pixels")[()]
            )
            dx = h5_dataset(g, "dx")[...]
            M = h5_dataset(g, "M")[...]
            # h5py returns a scalar string dataset as `bytes`; `str()` on it
            # would hand `torch.device` the literal text "b'cpu'".
            stored_device = h5_dataset(g, "device")[()]
            if isinstance(stored_device, bytes):
                stored_device = stored_device.decode()
            device = torch.device(str(stored_device))
        # A file written before `N` left `__init__` still has an `N` dataset.
        # It is ignored: nothing in this class stores or derives from it.
        return SMeta(
            all_beams,
            parent_beams,
            numerical_aperture_radius_pixels,
            dx,
            (int(M[0]), int(M[1])),
            device,
        )

    def make_beamlet_meta(
        self,
        n_radial_samples: int,
        n_angular_samples: int = 6,
        radius: float | None = None,
    ) -> "SMeta":
        """Build a partitioned-PRISM ``SMeta`` with hex-sampled parent beams.

        Selects parent beams on ``n_radial_samples`` concentric hex rings
        (plus the DC beam) within the aperture, then computes the
        natural-neighbor beamlet weights of every aperture beam relative to
        those parent beams.

        Parameters
        ----------
        n_radial_samples : int
            Number of concentric hex rings of parent beams around the DC beam
            (``0``: the DC beam alone, a single uniform beamlet).  The weights
            are the natural-neighbor weights of the parents actually selected,
            for every value.
        n_angular_samples : int, optional
            Number of angular samples on the first ring; later rings scale
            proportionally.
        radius : float or None, optional
            Radius of the outermost ring, in reciprocal pixels of axis 0.
            ``None`` (the default): the largest beam radius.  A beam set that
            reaches past the aperture (the numerically weightless tail of its
            soft edge) passes the aperture's own extent, so the rings sample
            where the probe has weight.

        Returns
        -------
        SMeta
            New metadata with ``parent_beams``, ``natural_neighbor_weights``
            and ``beamlets`` set for the partitioned S-matrix.
        """
        # Sample target positions in q_coord space (same units as all_beams_coords).
        # Generate n_radial_samples concentric hex rings (plus the DC beam at the origin).
        # Use the actual max radius from all_beams_coords so the rings span the aperture
        # correctly regardless of how numerical_aperture_radius_pixels was computed.
        all_coords_np = self.all_beams_coords.cpu().numpy().astype(np.float32)  # (B, 2)
        # Rings and neighbours are laid out in RECIPROCAL LENGTH, not in pixel
        # index: on a rectangular grid the two reciprocal pixels differ
        # (dk = 1 / (M * dx) per axis) and a round aperture is an ellipse in
        # index space. Work in units of dk_y, i.e. scale the x index by
        # dk_x / dk_y -- exactly 1.0 on an isotropic grid, so nothing moves there.
        MY, MX = int(self.M[0]), int(self.M[1])
        dy_, dx_ = (float(v) for v in np.broadcast_to(np.asarray(self.dx, float), (2,)))
        metric = np.array([1.0, (MY * dy_) / (MX * dx_)], dtype=np.float32)
        all_coords_np = all_coords_np * metric
        if radius is None:
            radius = float(np.linalg.norm(all_coords_np, axis=1).max())
        a_offset = np.pi / n_angular_samples
        samples = [[0.0, 0.0]]  # DC beam always included
        radial_values = np.linspace(0, radius, n_radial_samples + 1, endpoint=True)[1:]
        for i, r in enumerate(radial_values):
            n_ang = n_angular_samples * (1 + i)
            angular_values = np.linspace(-np.pi, np.pi, n_ang, endpoint=False)
            for a in angular_values:
                samples.append(
                    [r * np.sin(a + a_offset * i), r * np.cos(a + a_offset * i)]
                )
        samples_arr = np.array(samples, dtype=np.float32)  # (n_targets, 2)

        # For each target, find the nearest beam in all_beams_coords (q_coord space).
        xy_diff = np.linalg.norm(
            samples_arr[:, None, :] - all_coords_np[None, :, :], axis=2
        )  # (n_targets, B)
        nearest_idx = np.argmin(xy_diff, axis=1)  # (n_targets,)
        unique_beam_indices = np.unique(nearest_idx)  # deduplicate

        # Parent mask on the (MY, MX) grid. ``all_beams_coords`` are signed FFT
        # indices (DC at [0, 0]), so the pixel of a beam is its index modulo M.
        parent_beams = torch.zeros((MY, MX), dtype=torch.bool)
        chosen = self.all_beams_coords.cpu().numpy()[unique_beam_indices]
        parent_beams[chosen[:, 0] % MY, chosen[:, 1] % MX] = True

        # The weights are ALWAYS computed for the parents actually selected, in
        # the order ``SMeta`` will list them (row-major over the mask), so the
        # (B, Bp) matrix and ``parent_beams_coords`` cannot disagree. Every
        # ``n_radial_samples`` takes this one path: DC alone (0 rings) is the
        # single uniform beamlet, one ring is DC + ``n_angular_samples``.
        parent_coords = self.q_coords.cpu()[:, parent_beams].T.numpy()  # (Bp, 2)
        nnw = natural_neighbor_weights(
            parent_coords * metric,
            all_coords_np,
            minimum_weight_cutoff=1e-2,
        )  # (B, Bp), every row a partition of unity

        # Beamlet weights AT the B beam positions, shape (Bp, B) = nnw.T. The
        # consumer (_build_beamlet_basis_fft) only reads the weights at the beams,
        # so the dense (Bp, MY, MX) one-hot scatter is pure waste (10s of GB on
        # large grids) -- store just the gathered (Bp, B) form.
        beamlets_array = np.ascontiguousarray(nnw.T)

        new_meta = SMeta(
            self.all_beams,
            parent_beams,
            self.numerical_aperture_radius_pixels,
            self.dx,
            tuple(self.M),
            self.device,
        )
        new_meta.natural_neighbor_weights = torch.as_tensor(nnw)
        new_meta.beamlets = torch.as_tensor(beamlets_array).to(new_meta.device)
        return new_meta
