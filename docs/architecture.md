# Architecture

The `scatterem` package is a stack of layers. Each layer is one or more
subpackages. A module imports only from its own subpackage and from lower
layers. It never imports from a higher layer. Thus the physics does not depend
on the solvers, and the solvers do not depend on plotting or on the paper
datasets.

The core has seven layers, from the forward physics (layer 0) to the
reconstruction methods (layer 6). Three small base layers sit below the core:
settings, rotations and crystals. The paper datasets and the root `scatterem`
namespace sit above it.

## The layers

| Layer | Subpackages | What they hold |
|---|---|---|
| 8 | `scatterem` | the root namespace: the FF-STEM entry points, `Dataset4DStem`, the paper datasets and `configure` |
| 7 | `scatterem.data.public` | the paper datasets, downloaded from Zenodo or Hugging Face on first use |
| **6** | `scatterem.reconstruction` | the reconstruction methods: direct ptychography, tilt-corrected dark field, their fusion, `run`, and iterative (second-order) ptychography |
| **5** | `scatterem.training` | internal: one check for a `torch.distributed` group of several ranks |
| **4** | `scatterem.simulation` | forward simulation: PRISM-EELS, BiP-PRISM and the `PrismEelsSimulator` driver |
| **3** | `scatterem.nn`, `scatterem.vis`, `scatterem.diagnostics` | `nn`: differentiable operators (ptychography kernels, PRISM and multislice scan operators), losses and regularizers. `vis`: plotting. `diagnostics`: probe gauges and orbital angular momentum metrics |
| **2** | `scatterem.data` | the 4D-STEM containers, metadata, acquisition fields, calibration and storage |
| **1** | `scatterem.utils`, `scatterem.optim` | `utils`: general helpers (interpolation, sub-pixel shifts, scan indexing, HDF5). `optim`: the generic second-order solvers |
| **0** | `scatterem.physics` | constants, electron optics, atom sets, potentials, transmission functions and propagation |
| −1 | `scatterem.crystal` | elements, phases and CIF reading |
| −2 | `scatterem.rotation` | rotations: quaternions and axis-angle |
| −3 | `scatterem._configure` | process-wide settings, read by `scatterem.configure()` |

The bold rows are the seven core layers. Two subpackages in the same layer do
not import each other. For example, `nn` does not import `vis`.

## The import rule

A module in layer *n* imports from layers below *n* only. There is one
exception: `scatterem.reconstruction` reads `scatterem.__version__` from the
root namespace, to write it into the run record.

The rule has these results for users:

1. Import from the subpackage that holds a name, for example
   `from scatterem.physics.optics import lens_wavefront`. The root namespace
   re-exports only the FF-STEM entry points, `Dataset4DStem`, the paper
   datasets and `configure`.
2. A lower layer does not call the higher layers. For example, a potential
   built with `scatterem.physics` runs no reconstruction code. (An import of
   any subpackage still runs the root `scatterem/__init__.py`, which imports
   `scatterem.reconstruction`.)
3. Plotting is optional. Only `vis` imports matplotlib at import time. A
   lower layer loads matplotlib only when one of its plotting functions runs.

## Where to look

| To do this | Start in |
|---|---|
| Load a paper dataset | `scatterem.data.public`, or `from scatterem import You2026Carbon` |
| Hold, calibrate or bin your own 4D-STEM data | `scatterem.data` |
| Build atoms, potentials and a probe | `scatterem.physics` |
| Simulate core-loss or elastic STEM data | `scatterem.simulation.eels`, `scatterem.nn.scan_exit_wave` |
| Reconstruct with FF-STEM | `scatterem.reconstruction.run`, or `scatterem.fused_full_field` |
| Reconstruct a mixed-state probe | `scatterem.reconstruction.ptychography.iterative.second_order` |
| Measure the probe modes | `scatterem.diagnostics.metrics` |
| Plot a result | `scatterem.vis` |

The [API reference](api.md) lists every public name by subpackage.
