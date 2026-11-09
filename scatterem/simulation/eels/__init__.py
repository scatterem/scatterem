"""STEM-EELS core-loss simulation with PRISM and beam-partitioned S-matrices.

Written for the public release. It re-exports the names of the BiP-PRISM paper
(Pelz, J. Phys. Mater. 2026, doi:10.1088/2515-7639/aebdd0).

Public API
----------
``PrismEelsSimulator``, ``PrismResult``
    A thin driver: elemental maps, qEELS and energy-filtered cubes
    (``prism_eels_image``, ``prism_transition_potential``) and elastic CBED
    (:class:`scatterem.nn.scan_exit_wave.PrismScanOp`) of a periodic cell.
``prism_eels_image``
    The double-channeling PRISM STEM-EELS image (Algorithm 5): S1 to the
    ionization plane and S2 to the detector, either of them beam-partitioned
    (BiP-PRISM).
``prism_transition_potential``, ``prism_transition_potential_hybrid``
    Energy-filtered 4D-STEM by dual-S PRISM (Brown & Ophus), and the earlier
    single-S variant (exit leg by per-probe multislice).
``multislice_transition_potential``
    Conventional (quadratic) transition-potential multislice.
``ScatteringMatrix``, ``PartitionedScatteringMatrix``, ``DetectorExitSMatrix``
    The plane-wave PRISM S-matrix, its beam-partitioned form and the
    detector-side exit S-matrix.
``subshell_transitions``, ``build_transition_potentials``, ``TransitionPotentials``
    Ionization transition potentials.
``AtomicRadialWavefunction``, ``bound_wavefunction``, ``continuum_wavefunction``
    Atomic radial wavefunctions. The default ``wavefunctions="auto"`` uses
    GPAW when it is installed, as in v0.3 (the ``eels`` extra:
    ``pip install "scatterem[eels] @ git+https://github.com/scatterem/scatterem"``).
    Without GPAW it falls back to analytic hydrogenic bound and free continuum
    states and emits a ``HydrogenicFallbackWarning``.

The optional ``gpaw`` dependency (the ``eels`` extra, see above) is needed
only for physically accurate transition potentials; everything else works
without it.
"""

from scatterem.simulation._wave.partitioned_smatrix import PartitionedScatteringMatrix
from scatterem.simulation._wave.smatrix import ScatteringMatrix

from .forward.multislice_eels import multislice_transition_potential
from .forward.prism_eels import (
    prism_transition_potential,
    prism_transition_potential_hybrid,
)
from .forward.prism_eels_image import DetectorExitSMatrix, prism_eels_image
from .physics.radial import (
    AtomicRadialWavefunction,
    HydrogenicFallbackWarning,
    bound_wavefunction,
    cache_dir,
    continuum_wavefunction,
    gpaw_available,
)
from .physics.transition_potentials import (
    Transition,
    TransitionPotentials,
    build_transition_potentials,
    subshell_transitions,
)
from .prism_eels_simulator import PrismEelsSimulator, PrismResult

__all__ = [
    "AtomicRadialWavefunction",
    "HydrogenicFallbackWarning",
    "DetectorExitSMatrix",
    "PartitionedScatteringMatrix",
    "PrismEelsSimulator",
    "PrismResult",
    "ScatteringMatrix",
    "Transition",
    "TransitionPotentials",
    "bound_wavefunction",
    "build_transition_potentials",
    "cache_dir",
    "continuum_wavefunction",
    "gpaw_available",
    "prism_eels_image",
    "prism_transition_potential",
    "prism_transition_potential_hybrid",
    "subshell_transitions",
    "multislice_transition_potential",
]
