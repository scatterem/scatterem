# scatterem

{.lead}
Fused full-field 4D-STEM, BiP-PRISM core-loss STEM-EELS, and elastic PRISM

A 4D-STEM dataset holds more information than any single reconstruction recovers.
Direct ptychography reads the phase gradient encoded in the bright-field disk and
recovers low and mid spatial frequencies with high efficiency, but its transfer
falls away past the disk. Tilt-corrected dark field reads the parallax shift of
scattering *outside* the disk and recovers high frequencies, but carries almost
nothing at low ones. Neither channel spans the band on its own, and used alone
each leaves a gap where the other is strong.

**Fused full-field STEM** combines them as an SSNR-weighted Wiener fusion, so each
spatial frequency is taken from whichever channel measures it better. The result
is one image with no missing band.

<a href="https://colab.research.google.com/github/scatterem/scatterem/blob/main/docs/colab/ffstem_you2026_demo.ipynb" class="colab-badge"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the demo notebook in Google Colab"></a>

Run it now on a free GPU: the notebook installs the package, fetches one dataset
from Zenodo and reconstructs it, with nothing to set up locally.

```python
from scatterem import Dataset4DStem
from scatterem.reconstruction import run
from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig
from scatterem.reconstruction.ptychography.direct import AberrationFitConfig

dataset = Dataset4DStem.from_array(
    array,                          # (scan_y, scan_x, det_y, det_x)
    energy=300e3,                   # eV
    semiconvergence_angle=19.7e-3,  # rad
    scan_step=0.25,                 # Angstrom
    device="cuda",
)
dataset.calibrate_reciprocal_from_bright_field()  # measure dk; from_array sets a placeholder

fit = run(
    dataset,
    AberrationFitConfig(correct_order=2, bright_field_mask_threshold=dataset.bright_field_mask_threshold),
)
dataset.meta = fit.applied_to(dataset.meta)

fused = run(dataset, FusedFullFieldConfig(n_dark_field_segments=32)).fused
```

That is the whole pipeline. Each channel is also available on its own, through
`run(dataset, DirectPtychographyConfig(...))` and
`run(dataset, TiltCorrectedDarkFieldConfig(...))`. All three share one
calibration, so they land on the same grid.

## Where to go next

| | |
|---|---|
| [Install](docs/install.md) | one `pip install`, plus what to know about GPUs |
| [Quickstart](docs/quickstart.md) | your own data, from array to fused image |
| [The method](docs/method.md) | what each channel measures, and why fusing them works |
| [Calibration](docs/calibration.md) | the parameters that decide whether a reconstruction is correct |
| [Figure scripts](docs/figures.md) | the four paper specimens, downloaded automatically |
| [Simulate STEM-EELS](docs/eels.md) | the forward simulation: maps, qEELS, cubes, and what they cost |
| [BiP-PRISM](docs/bip-prism.md) | how partitioning both scattering matrices makes that affordable |
| [Elastic PRISM](docs/prism.md) | elastic 4D-STEM from the plane-wave and partitioned scattering matrix |
| [Migrating from v0.3](docs/migrating.md) | where every v0.3 name went |
| [API](docs/api.md) | reference, generated from the source |

## Simulating core-loss STEM-EELS

The same package also solves the *forward* problem: dynamical simulation of
core-loss STEM-EELS by conventional transition-potential multislice, by PRISM,
and by **BiP-PRISM**. BiP-PRISM partitions both the probe-forming and the
detector-propagating scattering matrix, so a full-resolution elemental map fits
on one consumer GPU.

```python
from scatterem.simulation.eels import PrismEelsSimulator

sim = PrismEelsSimulator(
    structure,                     # scatterem.physics.atom_set.AtomSet
    eV=100e3,
    semiconvergence_angle=25.0,    # mrad
    pixels=(128, 128),
    scan=(16, 16),
    num_slices=16,
    device="cuda",
)

ti_map = sim.eels_map(
    (22, 2, 1),                    # Ti L: (Z, n, l)
    detector_mrad=25.0,
    partition=dict(n_radial=4),
    partition_s2=dict(n_radial=4),
)
ti_map.data                        # (16, 16) elemental map
cbed = sim.elastic_cbed()          # elastic 4D-STEM by PRISM
```

The EELS wavefunctions come from GPAW when it is installed. pip builds GPAW
from source, and the build needs a C++ compiler (see
[Installation](docs/install.md)):

```bash
CC=g++ pip install "scatterem[eels]" gpaw-data
```

Without GPAW, the example runs with analytic hydrogenic states and emits a
`HydrogenicFallbackWarning`: the numbers are then not quantitative. v0.4 uses a
new absolute first-Born scale. After a best scalar fit, the map shapes differ
from v0.3 by 18-61 % with the hydrogenic states. With GPAW, they differ by
0.03-0.6 % for the K, L23 and M45 edges we tested, and by 1.4 % for Fe L1. The exact published numbers of the BiP-PRISM paper came
from the paper's pre-release code, which uses the v0.3 transition potentials.
See
[The BiP-PRISM paper and v0.4](#bip-prism-paper-v04).

See [Simulate STEM-EELS](docs/eels.md) for the pipeline,
[BiP-PRISM](docs/bip-prism.md) for the algorithm and
[Elastic PRISM](docs/prism.md) for the elastic simulation.

## Citing this

Cite the paper for the part you used. The reconstruction methods are described
in:

> S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao, M. Wu, D. Stroppa,
> D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci, D. Krishnan, X. Ye,
> F. K. M. Schur, E. Spiecker, P. Pelz.
> *Gap-Free Information Transfer in 4D-STEM via Fusion of Complementary
> Scattering Channels.* **Advanced Science** 13 (2026) e76620.
> [doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)

The STEM-EELS simulation and the elastic PRISM code are described in:

> P. Pelz. *The bi-partitioned PRISM (BiP-PRISM) algorithm for fast and scalable
> core-loss STEM-EELS simulations.* **Journal of Physics: Materials** (2026).
> [doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)

`CITATION.cff` in the repository carries the same metadata in machine-readable
form. The datasets are on Zenodo under
[10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901), CC-BY-4.0 —
please cite the record as well as the paper if you use them.

## Scope

Three things live here, and each accompanies its own paper:

- the FF-STEM reconstruction pipeline: direct ptychography, tilt-corrected dark
  field, their fusion, and the calibration and I/O they need;
- the core-loss STEM-EELS forward simulation (BiP-PRISM);
- elastic PRISM simulation of 4D-STEM.

The wider `scatterem` codebase covers more. Those parts accompany their own
publications and are released with them.

```{toctree}
:hidden:
:caption: Get started

docs/install
docs/quickstart
```

```{toctree}
:hidden:
:caption: Background

docs/method
docs/calibration
```

```{toctree}
:hidden:
:caption: Simulation

docs/eels
docs/bip-prism
docs/prism
```

```{toctree}
:hidden:
:caption: Reference

docs/figures
docs/migrating
docs/api
```
