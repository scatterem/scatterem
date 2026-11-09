"""A thin driver for PRISM-EELS (BiP-PRISM) and elastic PRISM simulations.

:class:`PrismEelsSimulator` builds the kernel inputs from an ``AtomSet`` and
calls the kernels; it adds no physics. The probe is ``probe_disc`` with a
defocus phase, the specimen ``potential_stack`` and
``thin_slice_transmission``, the ionization operators
``build_transition_potentials``. The EELS outputs come from
``prism_eels_image`` (map, qEELS) and ``prism_transition_potential`` (cube), the
elastic CBED from ``PrismScanOp`` on a ``build_prism_smeta`` beam set. Each
method returns a :class:`PrismResult`. For a call the driver does not cover,
pass the inputs from :meth:`~PrismEelsSimulator.probe_q`,
:meth:`~PrismEelsSimulator.scan_pixels`,
:meth:`~PrismEelsSimulator.transmissions`, :meth:`~PrismEelsSimulator.sites`
and :meth:`~PrismEelsSimulator.transition_potentials` to the kernels directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from scatterem.physics.atom_set import AtomSet
from scatterem.physics.beam_partition import BeamPartition
from scatterem.physics.constants import debroglie_at
from scatterem.physics.potential_stack import potential_stack
from scatterem.physics.thermal_jitter import ThermalJitter
from scatterem.physics.thin_slice import thin_slice_transmission
from scatterem.simulation.eels.forward.prism_eels import prism_transition_potential
from scatterem.simulation.eels.forward.prism_eels_image import prism_eels_image
from scatterem.simulation.eels.physics.transition_potentials import (
    TransitionPotentials,
    build_transition_potentials,
    subshell_transitions,
)
from scatterem.utils._device import resolve_device

from ._ionized_sites import configuration_jitter, ionized_sites, occupancy_sum
from ._prism_cbed import elastic_cbed_data
from ._prism_result import Edge, PrismResult

__all__ = ["PrismEelsSimulator", "PrismResult"]


class PrismEelsSimulator:
    """Core-loss EELS and elastic CBED of a periodic cell with PRISM kernels.

    The grid is the whole cell: ``pixels`` samples ``structure.sides[:2]``.
    The scan is a regular raster of ``scan`` positions over the cell, half a
    step from its edges. An ionized site contributes with the weight that its
    atom has in the elastic potential: its ``site_fraction`` when the
    structure has a partial occupancy, else 1. A site with fraction 0 does not
    ionize.

    Parameters
    ----------
    structure : AtomSet
        Rectangular box and atoms (fractional ``(y, x, depth)`` and ``Z``).
    eV : float
        Probe energy [eV].
    semiconvergence_angle : float
        Probe-forming aperture semi-angle [mrad].
    pixels : sequence of int
        Real-space grid ``(Ny, Nx)`` across the cell.
    scan : sequence of int
        Number of scan positions ``(Sy, Sx)``.
    num_slices : int, optional
        Number of equal slices of the cell along the beam. Default 1.
    defocus : float, optional
        Probe defocus [Angstrom]; the aperture carries
        ``exp(-i pi lambda defocus |q|^2)``. Default 0.
    interpolation_factor : int, optional
        PRISM interpolation factor ``f`` (Ophus 2017) of every output. Default 1
        keeps every aperture beam.
    num_frozen_phonons : int, optional
        Number of specimen configurations averaged. Default 1.
    displacements : bool, optional
        Draw thermal displacements for each configuration. Default False.
        In each EELS configuration the ionized atoms move with their atoms in
        the elastic potential (one draw for both).
    seed : int or None, optional
        Draw seed of configuration 0; configuration ``c`` uses
        ``ThermalJitter(draw_seed=seed + c)``. Default None: each configuration
        takes a fresh seed, which ``ThermalJitter`` logs at INFO.
    device : str or torch.device or None, optional
        Device of every tensor. ``None`` (default): the CPU, the device of the
        NumPy ``AtomSet`` (simulator convention). Pass ``"cuda"`` for the GPU.
    dtype : torch.dtype, optional
        ``torch.complex64`` (default) or ``torch.complex128`` for the EELS
        kernels. The elastic CBED runs in the precision of ``PrismScanOp``.

    Attributes
    ----------
    wavelength, gridsize, sampling : float, tuple, tuple
        Electron wavelength [Angstrom]; cell size ``sides[:2]`` and pixel
        size ``gridsize / pixels`` [Angstrom].
    slice_thickness, slice_distance : float
        ``sides[2] / num_slices``, and the propagation distance the EELS
        kernels take: ``slice_thickness``, but ``0.0`` for one slice.
    device : torch.device
        The resolved device. Every other argument is stored under its name.

    Raises
    ------
    ValueError
        If ``dtype`` is not complex64 or complex128, or a count is below 1.
    """

    def __init__(
        self,
        structure: AtomSet,
        *,
        eV: float,
        semiconvergence_angle: float,
        pixels: Sequence[int],
        scan: Sequence[int],
        num_slices: int = 1,
        defocus: float = 0.0,
        interpolation_factor: int = 1,
        num_frozen_phonons: int = 1,
        displacements: bool = False,
        seed: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"dtype must be complex64 or complex128, got {dtype}")
        counts = dict(
            num_slices=num_slices,
            interpolation_factor=interpolation_factor,
            num_frozen_phonons=num_frozen_phonons,
        )
        for name, value in counts.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        self.structure = structure
        self.eV = float(eV)
        self.wavelength = float(debroglie_at(self.eV))
        self.semiconvergence_angle = float(semiconvergence_angle)
        self.pixels = (int(pixels[0]), int(pixels[1]))
        self.scan = (int(scan[0]), int(scan[1]))
        self.num_slices = int(num_slices)
        self.defocus = float(defocus)
        self.interpolation_factor = int(interpolation_factor)
        self.num_frozen_phonons = int(num_frozen_phonons)
        self.displacements = bool(displacements)
        self.seed = seed
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype
        cell = np.asarray(structure.sides, dtype=np.float64)
        self.gridsize = (float(cell[0]), float(cell[1]))
        self.sampling = (
            self.gridsize[0] / self.pixels[0],
            self.gridsize[1] / self.pixels[1],
        )
        self.slice_thickness = float(cell[2]) / self.num_slices
        self.slice_distance = self.slice_thickness if self.num_slices > 1 else 0.0

    # ------------------------------------------------------------- inputs
    def _aperture(self, shape: tuple[int, int], dtype: torch.dtype) -> Tensor:
        """Defocused probe aperture on a ``shape`` window of the cell's pitch."""
        dk = (
            1.0 / (shape[0] * self.sampling[0]),
            1.0 / (shape[1] * self.sampling[1]),
        )
        from scatterem.nn.scan_exit_wave._prism_geometry import (  # noqa: PLC0415
            probe_disc,
        )

        alpha = self.semiconvergence_angle * 1e-3
        ap = probe_disc(
            shape, dk, self.wavelength, alpha, self.device, dtype=torch.float64
        )
        qy = torch.fft.fftfreq(shape[0], d=self.sampling[0], dtype=torch.float64)
        qx = torch.fft.fftfreq(shape[1], d=self.sampling[1], dtype=torch.float64)
        q2 = (qy[:, None] ** 2 + qx[None, :] ** 2).to(self.device)
        chi = pi * self.wavelength * self.defocus * q2
        return (ap * torch.exp(-1j * chi)).to(dtype)

    def probe_q(self) -> Tensor:
        """Probe-forming aperture on the full grid, as the EELS kernels take it.

        Returns
        -------
        torch.Tensor
            ``pixels``-shaped complex aperture with defocus, corner origin,
            scaled so that the real-space probe ``ifft2(probe_q)`` carries unit
            total intensity.
        """
        probe_q = self._aperture(self.pixels, self.dtype)
        power = (probe_q.abs().pow(2).sum() / probe_q.numel()).sqrt()
        return probe_q / power

    def scan_pixels(self) -> Tensor:
        """Scan positions in grid pixels.

        Returns
        -------
        torch.Tensor
            ``(Sy * Sx, 2)`` float64 ``(y, x)`` positions, row-major over the
            raster.
        """
        ry = (np.arange(self.scan[0]) + 0.5) * self.pixels[0] / self.scan[0]
        rx = (np.arange(self.scan[1]) + 0.5) * self.pixels[1] / self.scan[1]
        grid = np.stack(np.meshgrid(ry, rx, indexing="ij"), axis=-1).reshape(-1, 2)
        return torch.as_tensor(grid, dtype=torch.float64, device=self.device)

    def transmissions(self, config: int = 0) -> Tensor:
        """Real-space transmission functions of one specimen configuration.

        Parameters
        ----------
        config : int, optional
            Frozen-phonon configuration index; it offsets ``seed``. Default 0.

        Returns
        -------
        torch.Tensor
            ``(num_slices, Ny, Nx)`` transmissions of dtype ``dtype``.
        """
        jitter = configuration_jitter(self.displacements, self.seed, config)
        return self._transmissions(jitter)

    def _transmissions(self, jitter: ThermalJitter | None) -> Tensor:
        """Transmissions of the configuration that ``jitter`` draws."""
        real = torch.float64 if self.dtype == torch.complex128 else torch.float32
        potential = potential_stack(
            BeamPartition.uniform(self.num_slices),
            self.structure,
            self.pixels,
            jitter=jitter,
            device=self.device,
            dtype=real,
        )
        return thin_slice_transmission(self.eV, potential, bandlimit=2.0 / 3.0).to(
            self.dtype
        )

    def sites(self, Z: int) -> np.ndarray:
        """Equilibrium fractional positions of the atoms of one element.

        The EELS outputs do more than this: they move the sites with the
        thermal draw of each configuration and weight each site by its
        occupancy (see the class notes).

        Parameters
        ----------
        Z : int
            Atomic number.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)`` fractional ``(y, x, depth)`` positions.

        Raises
        ------
        ValueError
            If the structure has no atom of element ``Z``.
        """
        return ionized_sites(self.structure, Z, None, self.device)[0]

    def transition_potentials(
        self, edge: Sequence[int], epsilon: float, **transitions: Any
    ) -> TransitionPotentials:
        """Transition potentials of one subshell at one continuum energy.

        Parameters
        ----------
        edge : sequence of int
            ``(Z, n, l)`` of the ionized subshell, e.g. ``(22, 2, 1)`` for Ti L.
        epsilon : float
            Energy of the ejected electron above the threshold [eV].
        **transitions : Any
            Passed to ``subshell_transitions`` (``order``, ``wavefunctions``,
            ``xc``, ``potential_scale``, ``lprimes``).

        Returns
        -------
        TransitionPotentials
            The stack on this grid, in ``dtype``.
        """
        Z, n, ell = (int(v) for v in edge)
        return build_transition_potentials(
            subshell_transitions(Z, n, ell, float(epsilon), **transitions),
            Z,
            self.pixels,
            self.sampling,
            self.eV,
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------ outputs
    def _eels(
        self,
        output: Literal["map", "qeels", "cube"],
        edge: Sequence[int],
        epsilon: float,
        transitions: dict | None,
        options: dict,
    ) -> PrismResult:
        """Run one EELS kernel over the configurations and average."""
        kernel = prism_transition_potential if output == "cube" else prism_eels_image
        edge_t: Edge = (int(edge[0]), int(edge[1]), int(edge[2]))
        tp = self.transition_potentials(edge_t, epsilon, **(transitions or {}))
        probe_q, scan = self.probe_q(), self.scan_pixels()
        common = dict(
            wavelength=self.wavelength,
            gridsize=self.gridsize,
            slice_distance=self.slice_distance,
            interpolation_factor=self.interpolation_factor,
        )
        total, q_pixels = None, None
        for c in range(self.num_frozen_phonons):
            # One draw moves the elastic atoms and the ionized atoms.
            jitter = configuration_jitter(self.displacements, self.seed, c)
            T = self._transmissions(jitter)
            sites, weights = ionized_sites(
                self.structure, edge_t[0], jitter, self.device
            )

            def run(rows, T=T):
                return kernel(probe_q, T, tp, rows, scan, **common, **options)

            out, q_pixels = occupancy_sum(run, sites, weights)
            total = out if total is None else total + out
        assert total is not None  # num_frozen_phonons >= 1
        data = total / self.num_frozen_phonons
        if output == "cube":
            data = data.reshape(*self.scan, *self.pixels)
        return PrismResult(
            data=data,
            output=output,
            energy=self.eV,
            detector_sampling=(1.0 / self.gridsize[0], 1.0 / self.gridsize[1]),
            edge=edge_t,
            epsilon=float(epsilon),
            energy_losses=np.asarray(tp.energy_losses),
            q_pixels=q_pixels,
        )

    def eels_map(
        self,
        edge: Sequence[int],
        *,
        detector_mrad: float,
        epsilon: float = 1.0,
        transitions: dict | None = None,
        **options: Any,
    ) -> PrismResult:
        """Core-loss elemental map behind an EELS collection aperture.

        Parameters
        ----------
        edge : sequence of int
            ``(Z, n, l)`` of the ionized subshell. Every atom of element ``Z``
            ionizes.
        detector_mrad : float
            EELS collection semi-angle [mrad].
        epsilon : float, optional
            Energy of the ejected electron above the threshold [eV]. Default 1.
        transitions : dict or None, optional
            Keyword arguments for ``subshell_transitions``. Default None.
        **options : Any
            Passed to ``prism_eels_image``: ``partition``, ``partition_s2``
            (BiP-PRISM), ``inelastic_crop``, ``prism_mask``, ``mag_preserve``,
            ``focal_backprop``, ``contract``, ``scan_chunk``, ``reconstruction_workspace``.

        Returns
        -------
        PrismResult
            ``output="map"``, ``data`` of shape ``scan``.
        """
        options = dict(options, detector_mrad=detector_mrad, scan_shape=self.scan)
        return self._eels("map", edge, epsilon, transitions, options)

    def qeels(
        self,
        edge: Sequence[int],
        *,
        detector_mrad: float,
        epsilon: float = 1.0,
        axis: int | Literal["both"] = "both",
        transitions: dict | None = None,
        **options: Any,
    ) -> PrismResult:
        """Momentum-resolved core-loss output (qEELS; PACBED by scan average).

        Parameters
        ----------
        edge : sequence of int
            ``(Z, n, l)`` of the ionized subshell.
        detector_mrad : float
            EELS collection semi-angle [mrad].
        epsilon : float, optional
            Energy of the ejected electron above the threshold [eV]. Default 1.
        axis : {0, 1, "both"}, optional
            ``qeels_axis`` of ``prism_eels_image``: ``"both"`` (default) keeps
            each detector beam, 0 or 1 projects onto one momentum axis.
        transitions : dict or None, optional
            Keyword arguments for ``subshell_transitions``. Default None.
        **options : Any
            Passed to ``prism_eels_image`` (see :meth:`eels_map`).

        Returns
        -------
        PrismResult
            ``output="qeels"``, ``data`` of shape ``scan + (n_q,)`` and the
            matching ``q_pixels``. Summing ``data`` over its last axis gives
            the map of :meth:`eels_map` with the same settings.
        """
        options = dict(options, detector_mrad=detector_mrad, scan_shape=self.scan)
        options["qeels_axis"] = axis
        return self._eels("qeels", edge, epsilon, transitions, options)

    def eels_cube(
        self,
        edge: Sequence[int],
        *,
        epsilon: float = 1.0,
        transitions: dict | None = None,
        **options: Any,
    ) -> PrismResult:
        """Energy-filtered 4D-STEM cube from the dual scattering matrix.

        Parameters
        ----------
        edge : sequence of int
            ``(Z, n, l)`` of the ionized subshell.
        epsilon : float, optional
            Energy of the ejected electron above the threshold [eV]. Default 1.
        transitions : dict or None, optional
            Keyword arguments for ``subshell_transitions``. Default None.
        **options : Any
            Passed to ``prism_transition_potential`` (``partition``,
            ``scan_chunk``).

        Returns
        -------
        PrismResult
            ``output="cube"``, ``data`` of shape ``scan + pixels``.
        """
        return self._eels("cube", edge, epsilon, transitions, options)

    def elastic_cbed(
        self,
        *,
        n_radial: int | None = None,
        n_angular: int = 6,
        focal_backprop: float | str = "centroid",
    ) -> PrismResult:
        """Elastic 4D-STEM (CBED at each scan position) with ``PrismScanOp``.

        The probe uses the aperture of :meth:`probe_q` on the PRISM window
        ``pixels // interpolation_factor`` and carries unit total intensity.

        Parameters
        ----------
        n_radial : int or None, optional
            None (default) is PRISM on every aperture beam of the f-lattice. An
            integer selects partitioned PRISM with parent beams on this many
            rings (``build_prism_smeta``).
        n_angular : int, optional
            Parent beams on the first ring of a partitioned beam set. Default 6.
        focal_backprop : float or str, optional
            Reference plane of the partitioned interpolation, passed to
            ``PrismScanOp``. Default ``"centroid"``. Ignored without
            ``n_radial``.

        Returns
        -------
        PrismResult
            ``output="cbed"``, ``data`` of shape ``scan + window``.
        """
        data = elastic_cbed_data(self, n_radial, n_angular, focal_backprop)
        f = self.interpolation_factor
        return PrismResult(
            data=data,
            output="cbed",
            energy=self.eV,
            detector_sampling=(f / self.gridsize[0], f / self.gridsize[1]),
        )
