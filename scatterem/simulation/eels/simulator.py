"""High-level STEM-EELS driver producing energy-filtered 4D-STEM datasets.

:class:`StemEelsSimulator` ties together the atomic structure, the probe, the
transmission functions and the ionization transition potentials, and runs either
the conventional transition-potential multislice or the scattering-matrix (PRISM)
algorithm.  The result is one 4D-STEM dataset ``(scan_y, scan_x, det_y, det_x)``
for the requested ionization edge / energy window.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from scatterem.simulation._electron_optics import energy2wavelength

from .._grid import q_space_array
from ..structure import Structure
from ..transmission import make_transmission_functions
from .multislice_eels import _fourier_shift_stack, transition_potential_multislice
from .prism_eels import prism_transition_potential
from .transition_potentials import build_transition_potentials, subshell_transitions

__all__ = ["StemEelsSimulator", "StemEelsResult"]


@dataclass
class StemEelsResult:
    """Output of a STEM-EELS simulation.

    Attributes
    ----------
    cube : Tensor
        Energy-filtered 4D-STEM dataset ``(scan_y, scan_x, det_y, det_x)``
        (real, non-negative; diffraction patterns are corner-origin).
    edge : tuple[int, int, int]
        ``(Z, n, l)`` of the ionized subshell.
    epsilon : float
        Continuum-electron energy above threshold [eV].  For a windowed result
        (:meth:`StemEelsSimulator.simulate_window`) this is the mean of the
        integrated energy grid.
    energy_losses : ndarray
        For a single-energy result, the per-transition energy loss [eV] that was
        summed into ``cube``.  For a windowed result, the sampled energy-loss
        grid (length ``n_energies``) that was integrated.
    energy : float
        Probe energy [eV].
    sampling : tuple[float, float]
        Detector reciprocal-space sampling ``(dky, dkx)`` [1/Å].
    energy_range : tuple[float, float] | None
        For a windowed result, the integrated energy-loss window
        ``(onset + eps_min, onset + eps_max)`` [eV]; ``None`` for a single
        energy.
    energies : ndarray | None
        For a windowed result, the continuum energies above threshold (the
        ``epsilon`` grid) that were integrated; ``None`` for a single energy.
    """

    cube: Tensor
    edge: Tuple[int, int, int]
    epsilon: float
    energy_losses: np.ndarray
    energy: float
    sampling: Tuple[float, float]
    energy_range: Optional[Tuple[float, float]] = None
    energies: Optional[np.ndarray] = None


class StemEelsSimulator:
    """Simulate energy-filtered 4D-STEM-EELS datasets."""

    def __init__(
        self,
        structure: Structure,
        *,
        eV: float,
        semiconvergence_angle: float,
        pixels: Sequence[int],
        scan: Sequence[int],
        edge: Tuple[int, int, int],
        epsilon: float = 1.0,
        order: int = 1,
        defocus: float = 0.0,
        num_slices: int = 1,
        num_frozen_phonons: int = 1,
        displacements: bool = False,
        seed: Optional[int] = None,
        prism: bool = False,
        interpolation_factor: int = 1,
        backend: str = "auto",
        xc: str = "PBE",
        potential_scale: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        self.structure = structure
        self.eV = float(eV)
        self.wavelength = float(energy2wavelength(self.eV))
        self.semiconvergence_angle = float(semiconvergence_angle)
        self.pixels = (int(pixels[0]), int(pixels[1]))
        self.scan = (int(scan[0]), int(scan[1]))
        self.edge = (int(edge[0]), int(edge[1]), int(edge[2]))
        self.epsilon = float(epsilon)
        self.order = int(order)
        self.defocus = float(defocus)
        self.num_slices = int(num_slices)
        self.num_frozen_phonons = int(num_frozen_phonons)
        self.displacements = bool(displacements)
        self.seed = seed
        self.prism = bool(prism)
        self.interpolation_factor = interpolation_factor
        self.backend = backend
        self.xc = xc
        self.potential_scale = float(potential_scale)
        self.device = device
        self.dtype = dtype

        self.gridsize = (
            float(structure.unitcell[0]),
            float(structure.unitcell[1]),
        )
        self.sampling = (
            self.gridsize[0] / self.pixels[0],
            self.gridsize[1] / self.pixels[1],
        )
        self.slice_distance = (
            float(structure.unitcell[2]) / self.num_slices
            if self.num_slices > 1
            else 0.0
        )

    # ------------------------------------------------------------------ #
    def _probe_q(self) -> Tensor:
        """Probe-forming aperture (with defocus) in reciprocal space."""
        qy, qx = q_space_array(self.pixels, self.gridsize)
        q2 = qy**2 + qx**2
        k_ap = self.semiconvergence_angle / 1000.0 / self.wavelength  # 1/Å
        aperture = (np.sqrt(q2) <= k_ap).astype(np.float64)
        chi = pi * self.wavelength * self.defocus * q2  # defocus aberration
        probe_q = aperture * np.exp(-1j * chi)
        probe_q = torch.as_tensor(probe_q, dtype=self.dtype, device=self.device)
        # Normalise so the real-space probe carries unit intensity.
        n = probe_q.numel()
        power = (probe_q.abs().pow(2).sum() / n).sqrt()
        if float(power) > 0:
            probe_q = probe_q / power
        return probe_q

    def _scan_pixels(self) -> Tensor:
        sy, sx = self.scan
        ry = (np.arange(sy) + 0.5) * self.pixels[0] / sy
        rx = (np.arange(sx) + 0.5) * self.pixels[1] / sx
        grid = np.stack(np.meshgrid(ry, rx, indexing="ij"), axis=-1).reshape(-1, 2)
        return torch.as_tensor(grid, dtype=torch.float64, device=self.device)

    def _transmissions(self, fp_index: int) -> Tensor:
        subslices = [(i + 1) / self.num_slices for i in range(self.num_slices)]
        seed = None if self.seed is None else self.seed + fp_index
        T = make_transmission_functions(
            self.structure,
            self.pixels,
            self.eV,
            subslices=subslices,
            displacements=self.displacements,
            fftout=False,
            device=self.device,
            seed=seed,
        )
        return T.to(self.dtype)

    def _sites_for(self, Z: int) -> np.ndarray:
        """Fractional coordinates of every atom of element ``Z`` in the cell."""
        atoms = np.asarray(self.structure.atoms, dtype=np.float64)
        mask = np.round(atoms[:, 3]).astype(int) == int(Z)
        return np.atleast_2d(atoms[mask, :3])

    def _sites(self) -> np.ndarray:
        return self._sites_for(self.edge[0])

    @staticmethod
    def _energy_grid(
        energy_range: Tuple[float, float], n_energies: int
    ) -> Tuple[np.ndarray, list]:
        """Validate an energy window and return ``(eps_grid, trapezoid_weights)``.

        The weights integrate ``∫ I d(epsilon)`` on the uniform grid (trapezoid
        rule); they are plain Python floats.
        """
        eps_min, eps_max = float(energy_range[0]), float(energy_range[1])
        if eps_min <= 0.0:
            raise ValueError(f"eps_min must be > 0, got {eps_min}")
        if eps_max <= eps_min:
            raise ValueError(f"require eps_max > eps_min, got ({eps_min}, {eps_max})")
        n_energies = int(n_energies)
        if n_energies < 2:
            raise ValueError(f"n_energies must be >= 2, got {n_energies}")
        eps_grid = np.linspace(eps_min, eps_max, n_energies)
        deps = (eps_max - eps_min) / (n_energies - 1)
        weights = np.full(n_energies, deps)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return eps_grid, [float(w) for w in weights]

    # ------------------------------------------------------------------ #
    def _build_tp(self, epsilon: float, edge: Optional[Tuple[int, int, int]] = None):
        """Build the transition potentials for one continuum energy ``epsilon``
        of one ionization ``edge`` (defaults to ``self.edge``)."""
        Z, n, l = self.edge if edge is None else edge
        transitions = subshell_transitions(
            Z,
            n,
            l,
            epsilon,
            order=self.order,
            backend=self.backend,
            xc=self.xc,
            potential_scale=self.potential_scale,
        )
        return build_transition_potentials(
            transitions,
            Z,
            self.pixels,
            self.sampling,
            self.eV,
            device=self.device,
            dtype=self.dtype,
        )

    def _run_jobs(
        self,
        jobs: Sequence,
        probe_q: Tensor,
        scan_pixels: Tensor,
        batch_size: Optional[int],
    ) -> list:
        """Frozen-phonon-averaged detector intensities for several jobs at once.

        Each job is a ``(sites, tps, weights)`` triple and accumulates
        ``sum_i weights[i] * intensity_i`` over its own transition-potential
        stacks (energies).  The (expensive) transmission functions are computed
        **once per frozen phonon and reused across every job** -- since they are
        the elastic potential of the whole structure they do not depend on the
        ionized element/energy -- and the scan-shifted probes (conventional path)
        are built once.  Returns one accumulator tensor per job, in input order.
        """
        probes = None
        if not self.prism:
            base_probe = torch.fft.ifft2(probe_q)
            # Bitwise-identical to a per-position ``_fourier_shift`` loop, but
            # ``scan_pixels`` lives on the GPU, so that loop cost 2P device syncs
            # (one per ``float(shift)``) and rebuilt ``fft2(base_probe)`` P times.
            probes = _fourier_shift_stack(base_probe, scan_pixels).to(self.dtype)

        accumulators = [None] * len(jobs)
        for c in range(self.num_frozen_phonons):
            T = self._transmissions(c)
            for j, (sites, tps, weights) in enumerate(jobs):
                for w, tp in zip(weights, tps):
                    if self.prism:
                        intensity = prism_transition_potential(
                            probe_q,
                            T,
                            tp,
                            sites,
                            scan_pixels,
                            wavelength=self.wavelength,
                            gridsize=self.gridsize,
                            slice_distance=self.slice_distance,
                            interpolation_factor=self.interpolation_factor,
                        )
                    else:
                        intensity = transition_potential_multislice(
                            probes,
                            T,
                            tp,
                            sites,
                            wavelength=self.wavelength,
                            gridsize=self.gridsize,
                            slice_distance=self.slice_distance,
                            batch_size=batch_size,
                        )
                    term = intensity if w == 1.0 else w * intensity
                    accumulators[j] = (
                        term if accumulators[j] is None else accumulators[j] + term
                    )
        return accumulators

    def _run(
        self,
        tps: Sequence,
        weights: Sequence[float],
        sites: np.ndarray,
        probe_q: Tensor,
        scan_pixels: Tensor,
        batch_size: Optional[int],
    ) -> Tensor:
        """Single-job convenience wrapper around :meth:`_run_jobs`."""
        return self._run_jobs(
            [(sites, tps, weights)], probe_q, scan_pixels, batch_size
        )[0]

    def _det_sampling(self) -> Tuple[float, float]:
        return (1.0 / self.gridsize[0], 1.0 / self.gridsize[1])

    # ------------------------------------------------------------------ #
    def simulate(self, batch_size: Optional[int] = None) -> StemEelsResult:
        """Run the simulation at a single continuum energy ``self.epsilon`` and
        return the energy-filtered 4D-STEM dataset."""
        Z = self.edge[0]
        sites = self._sites()
        if sites.shape[0] == 0:
            raise ValueError(f"structure contains no atoms of element Z={Z}")

        tp = self._build_tp(self.epsilon)
        probe_q = self._probe_q()
        scan_pixels = self._scan_pixels()

        accumulated = self._run([tp], [1.0], sites, probe_q, scan_pixels, batch_size)
        cube = (accumulated / self.num_frozen_phonons).reshape(
            self.scan[0], self.scan[1], self.pixels[0], self.pixels[1]
        )
        return StemEelsResult(
            cube=cube,
            edge=self.edge,
            epsilon=self.epsilon,
            energy_losses=tp.energy_losses,
            energy=self.eV,
            sampling=self._det_sampling(),
        )

    # ------------------------------------------------------------------ #
    def simulate_window(
        self,
        energy_range: Tuple[float, float],
        n_energies: int = 16,
        batch_size: Optional[int] = None,
    ) -> StemEelsResult:
        """Energy-integrated 4D-STEM-EELS over an energy-loss window.

        Sweeps the continuum-electron energy ``epsilon`` across ``energy_range``
        (energies *above the ionization threshold*, eV) on a uniform grid of
        ``n_energies`` points and trapezoid-integrates the per-energy datacubes.

        Because the continuum states are energy-normalised (per unit energy),
        ``∫ I(epsilon) d(epsilon)`` is the signal collected in the corresponding
        energy-loss window ``[onset + eps_min, onset + eps_max]``, where the edge
        onset ``= -E_bound`` is the binding energy of the ionized subshell.  The
        mapping ``ΔE = epsilon + onset`` is linear, so integrating over
        ``epsilon`` equals integrating over energy loss.

        Parameters
        ----------
        energy_range : (float, float)
            ``(eps_min, eps_max)`` above threshold [eV]; ``eps_min`` must be
            ``> 0`` (the energy-normalised continuum diverges at threshold) and
            ``eps_max > eps_min``.
        n_energies : int
            Number of energies sampled across the window (``>= 2``).  More points
            reduce the trapezoid-integration error at proportional cost.
        batch_size : int, optional
            Forwarded to the conventional multislice scan-position batching.

        Notes
        -----
        Transition potentials are built once per energy (the GPAW continuum is
        additionally cached on disk), and the transmission functions are computed
        once per frozen phonon and reused across all energies.  Memory scales with
        ``n_energies`` (one transition-potential stack is held per sampled energy).
        """
        eps_grid, weights = self._energy_grid(energy_range, n_energies)

        Z = self.edge[0]
        sites = self._sites()
        if sites.shape[0] == 0:
            raise ValueError(f"structure contains no atoms of element Z={Z}")

        tps = [self._build_tp(float(e)) for e in eps_grid]
        # onset (binding energy) = energy_loss(epsilon) - epsilon, identical for
        # every transition at a given epsilon.
        onset = float(tps[0].energy_losses[0]) - float(eps_grid[0])

        probe_q = self._probe_q()
        scan_pixels = self._scan_pixels()
        accumulated = self._run(tps, weights, sites, probe_q, scan_pixels, batch_size)
        cube = (accumulated / self.num_frozen_phonons).reshape(
            self.scan[0], self.scan[1], self.pixels[0], self.pixels[1]
        )
        return StemEelsResult(
            cube=cube,
            edge=self.edge,
            epsilon=float(eps_grid.mean()),
            energy_losses=onset + eps_grid,
            energy=self.eV,
            sampling=self._det_sampling(),
            energy_range=(onset + float(eps_grid[0]), onset + float(eps_grid[-1])),
            energies=eps_grid,
        )

    # ------------------------------------------------------------------ #
    def simulate_edges(
        self,
        edges: Sequence[Tuple[int, int, int]],
        energy_range: Optional[Tuple[float, float]] = None,
        n_energies: int = 16,
        batch_size: Optional[int] = None,
    ) -> "dict[Tuple[int, int, int], StemEelsResult]":
        """Simulate several ionization edges (elements/subshells) in one pass.

        The elastic transmission functions and probes are shared across **all**
        edges -- they are the dynamical scattering of the whole structure and do
        not depend on which element is ionized -- so this is much cheaper than
        constructing a separate :class:`StemEelsSimulator` per element (which
        would recompute the transmissions, and the frozen-phonon passes, every
        time).

        Parameters
        ----------
        edges : sequence of (Z, n, l)
            The ionized subshells, e.g. ``[(8, 1, 0), (22, 2, 1)]`` for the O-K
            and Ti-L edges.  Every ``Z`` must be present in the structure.
        energy_range : (float, float), optional
            If given, each edge is energy-integrated over the **same** window of
            continuum energies above its own threshold (see
            :meth:`simulate_window`); the window therefore sits above each edge's
            own onset.  If ``None``, each edge is evaluated at the single energy
            ``self.epsilon``.
        n_energies : int
            Energies sampled per window when ``energy_range`` is given.
        batch_size : int, optional
            Forwarded to the conventional multislice scan-position batching.

        Returns
        -------
        dict
            Maps each ``(Z, n, l)`` edge to its :class:`StemEelsResult`.  (If the
            same edge is passed twice, only the last result is kept.)

        Notes
        -----
        All edges' transition potentials are held in memory simultaneously so the
        transmissions can be reused in a single frozen-phonon sweep; memory scales
        with the total number of (edge x energy) transition-potential stacks.
        """
        edges = [(int(e[0]), int(e[1]), int(e[2])) for e in edges]
        if len(edges) == 0:
            raise ValueError("edges must be non-empty")
        windowed = energy_range is not None
        if windowed:
            eps_grid, weights = self._energy_grid(energy_range, n_energies)
        else:
            eps_grid, weights = np.array([self.epsilon]), [1.0]

        # Build one job per edge: (sites, tps, weights).  Validate atoms exist.
        jobs = []
        tps_per_edge = []
        for edge in edges:
            Z = edge[0]
            sites = self._sites_for(Z)
            if sites.shape[0] == 0:
                raise ValueError(f"structure contains no atoms of element Z={Z}")
            tps = [self._build_tp(float(e), edge) for e in eps_grid]
            jobs.append((sites, tps, weights))
            tps_per_edge.append(tps)

        probe_q = self._probe_q()
        scan_pixels = self._scan_pixels()
        accumulators = self._run_jobs(jobs, probe_q, scan_pixels, batch_size)

        det_sampling = self._det_sampling()
        results: dict = {}
        for edge, tps, accumulated in zip(edges, tps_per_edge, accumulators):
            cube = (accumulated / self.num_frozen_phonons).reshape(
                self.scan[0], self.scan[1], self.pixels[0], self.pixels[1]
            )
            if windowed:
                onset = float(tps[0].energy_losses[0]) - float(eps_grid[0])
                results[edge] = StemEelsResult(
                    cube=cube,
                    edge=edge,
                    epsilon=float(eps_grid.mean()),
                    energy_losses=onset + eps_grid,
                    energy=self.eV,
                    sampling=det_sampling,
                    energy_range=(
                        onset + float(eps_grid[0]),
                        onset + float(eps_grid[-1]),
                    ),
                    energies=eps_grid,
                )
            else:
                results[edge] = StemEelsResult(
                    cube=cube,
                    edge=edge,
                    epsilon=self.epsilon,
                    energy_losses=tps[0].energy_losses,
                    energy=self.eV,
                    sampling=det_sampling,
                )
        return results
