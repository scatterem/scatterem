"""Forward simulation of core-loss STEM-EELS.

Exports the surface the BiP-PRISM paper describes: the multislice substrate
and the core-loss EELS algorithms built on it.
"""

from scatterem.simulation.scattering_factors import (
    calculate_scattering_factors,
    electron_scattering_factor,
    interaction_constant,
)
from scatterem.simulation.structure import Structure
from scatterem.simulation.potentials import make_potential
from scatterem.simulation.transmission import (
    make_transmission_functions,
    projected_potential_to_object,
)
from scatterem.simulation.eels import (
    StemEelsSimulator,
    StemEelsResult,
    subshell_transitions,
    build_transition_potentials,
    TransitionPotentials,
    transition_potential_multislice,
    prism_transition_potential,
    bound_wavefunction,
    continuum_wavefunction,
    gpaw_available,
)

__all__ = [
    # multislice substrate
    "Structure",
    "calculate_scattering_factors",
    "electron_scattering_factor",
    "interaction_constant",
    "make_potential",
    "make_transmission_functions",
    "projected_potential_to_object",
    # core-loss STEM-EELS
    "StemEelsSimulator",
    "StemEelsResult",
    "TransitionPotentials",
    "build_transition_potentials",
    "subshell_transitions",
    "transition_potential_multislice",
    "prism_transition_potential",
    "bound_wavefunction",
    "continuum_wavefunction",
    "gpaw_available",
]
