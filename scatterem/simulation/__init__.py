"""Forward simulation: PRISM-EELS with beam-partitioned S-matrices, and elastic PRISM.

Written for the public release. The core-loss STEM-EELS simulation of
Pelz, J. Phys. Mater. (2026), doi:10.1088/2515-7639/aebdd0, is in
:mod:`scatterem.simulation.eels`; its driver is
:class:`~scatterem.simulation.eels.PrismEelsSimulator`. The specimen substrate
(structure, projected potential, transmission functions, propagation) is in
:mod:`scatterem.physics`. Elastic PRISM is
:class:`scatterem.nn.scan_exit_wave.PrismScanOp`.

This package re-exports nothing, so ``import scatterem.simulation`` is cheap.
"""
