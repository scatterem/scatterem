"""Generic wave-optics building blocks of the forward simulators (private).

Not EELS-specific: the multislice exit walk (:mod:`.exit_walk`), the PRISM
scattering matrices (:mod:`.smatrix`, :mod:`.partitioned_smatrix`) and their
beam helpers (:mod:`.prism_beams`), used by the EELS paths.
The propagation primitives themselves live in
:mod:`scatterem.physics.propagation`.  Import from the submodules; this package
re-exports nothing.
"""
