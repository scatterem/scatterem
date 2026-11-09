<p align="center">
  <img src="https://raw.githubusercontent.com/scatterem/scatterem/main/logo.png" alt="scatterem logo" width="160">
</p>

<h1 align="center">scatterem</h1>

<p align="center">
  <b>4D-STEM reconstruction and differentiable STEM simulations on the GPU, in PyTorch.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/scatterem/"><img src="https://img.shields.io/pypi/v/scatterem" alt="PyPI"></a>
  <a href="https://pypi.org/project/scatterem/"><img src="https://img.shields.io/pypi/pyversions/scatterem" alt="Python versions"></a>
  <a href="https://scatterem.github.io/scatterem"><img src="https://img.shields.io/badge/docs-scatterem.github.io-blue" alt="Documentation"></a>
  <a href="https://doi.org/10.5281/zenodo.22162829"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22162829.svg" alt="Zenodo DOI"></a>
  <a href="https://github.com/scatterem/scatterem/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-Apache--2.0-green" alt="Licence"></a>
</p>

scatterem holds the reference implementations of three published methods in
electron scattering. Each method comes with its paper, its data and the scripts
that make the paper's figures.

| Method | What it does | Paper |
|---|---|---|
| **FF-STEM** | Fuses direct (single-side-band) ptychography and tilt-corrected dark field with SSNR-weighted Wiener weights. The image keeps the low-frequency phase contrast and the high-frequency detail. | S. You *et al.*, *Adv. Sci.* 13 (2026) e76620, [doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620) |
| **BiP-PRISM** | Simulates core-loss STEM-EELS by transition-potential multislice, PRISM and the bi-partitioned PRISM algorithm. Full-resolution elemental maps fit on one consumer GPU. | P. Pelz, *J. Phys. Mater.* (2026), [doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0) |
| **Elastic PRISM** | Simulates 4D-STEM with the plane-wave and the beam-partitioned scattering matrix. | as used in the BiP-PRISM paper |

```bash
pip install scatterem                                # FF-STEM and elastic PRISM
CC=g++ pip install "scatterem[eels]" gpaw-data       # adds GPAW wavefunctions for EELS
```

- **Documentation:** [scatterem.github.io/scatterem](https://scatterem.github.io/scatterem)
- **Try it in the browser:** [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py) (one FF-STEM reconstruction on a free molab GPU)
- **FF-STEM data:** [doi:10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901) (CC-BY-4.0)
- **Upgrading from v0.3:** every module moved. Read [Migrating from v0.3](https://scatterem.github.io/scatterem/docs/migrating.html).

## Install

The install commands are at the top of this page. A CUDA GPU is strongly
recommended. The reconstructions are FFT- and kernel-bound, and the NVIDIA Warp
kernels need a GPU for anything beyond a small test. If the default `torch`
wheel does not match your driver, install a matching build first.

The `eels` extra adds GPAW, the all-electron backend for quantitative
STEM-EELS. pip builds GPAW from source, and the build needs a C++ compiler (see
[Installation](https://scatterem.github.io/scatterem/docs/install.html)).

The molab notebook installs the package, fetches one dataset from Zenodo and
runs the full FF-STEM reconstruction on a free molab GPU. It sizes the scan to
the GPU it gets, so it also works on smaller GPUs.

## Run the FF-STEM paper figures

These scripts run the FF-STEM pipeline on the data of the paper figures, with
the v0.4 code. Their outputs differ from the published figures and from v0.3.
The measured causes are a fix of the tcDF parallax, the new default sign of the
ptychography phase, changes to the aberration autofocus, the removal of the
scan DC from up-sampled images, the detector frame of the astigmatism in
tcDF and the new bright-field disk measurement, which sets `dk`. The paper figures were made with an earlier version of the code, which
this repository does not contain.
[Run the paper's figure scripts](https://scatterem.github.io/scatterem/docs/figures.html) gives the numbers.

The scripts are in this repository, not in the PyPI package. Clone the
repository at the release tag and install the same release from PyPI:

```bash
git clone --branch v0.4.0 https://github.com/scatterem/scatterem
cd scatterem
pip install scatterem==0.4.0
```

Each script downloads its dataset from Zenodo on the first run and checks it by
md5. You only set where the data goes:

```bash
export FFSTEM_DATA_ROOT=/path/with/space
python experiments/ff_stem/Fig2_carbon.py
```

| Script | Specimen | Download | Peak GPU memory |
| --- | --- | --- | --- |
| `Fig2_carbon.py` | amorphous carbon | 0.34 GB | 0.8 GiB |
| `Fig2_Co3O4.py` | Co₃O₄ | 0.56 GB | 1.2 GiB |
| `Fig4_diffraction.py` | Au, low dose | 0.5 GB | 5.5 GiB (0.4 GiB with `scan_edge_crop=224`) |
| `Fig1_Gd2O3.py` | Gd₂O₃ | 6.6 GB | 21.7 GiB |

The peaks were measured with v0.4 on an NVIDIA RTX A6000.
Each script also has a molab notebook in `notebooks/` that runs it in the
browser on a free GPU, for example
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_carbon.py)
for `Fig2_carbon.py`.

## Use FF-STEM on your own data

```python
from scatterem import Dataset4DStem
from scatterem.reconstruction import run
from scatterem.reconstruction.darkfield import TiltCorrectedDarkFieldConfig
from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig
from scatterem.reconstruction.ptychography.direct import (
    AberrationFitConfig,
    DirectPtychographyConfig,
)

dataset = Dataset4DStem.from_array(
    array,                          # (scan_y, scan_x, det_y, det_x)
    energy=300e3,                   # eV
    semiconvergence_angle=19.7e-3,  # rad
    scan_step=0.25,                 # Angstrom
    device="cuda",
)
# Measure dk from the bright-field disk. from_array sets a (1.0, 1.0)
# placeholder, and run() warns if it is still there.
dataset.calibrate_reciprocal_from_bright_field()
threshold = dataset.bright_field_mask_threshold

# The default "combined" fit takes its start from the bright-field shifts.
fit = run(dataset, AberrationFitConfig(correct_order=2, bright_field_mask_threshold=threshold))
dataset.meta = fit.applied_to(dataset.meta)

ptycho = run(dataset, DirectPtychographyConfig(upsample=2.0, bright_field_mask_threshold=threshold)).phase
dark_field = run(dataset, TiltCorrectedDarkFieldConfig(upsample=2.0, bright_field_mask_threshold=threshold)).image
fused = run(dataset, FusedFullFieldConfig(upsample=2.0, n_dark_field_segments=32)).fused
```

## Simulate core-loss STEM-EELS and elastic 4D-STEM

```python
import numpy as np
from scatterem.physics.atom_set import AtomSet
from scatterem.simulation.eels import PrismEelsSimulator

sto = AtomSet(
    np.array([3.905, 3.905, 3.905]),  # box sides (y, x, depth), Angstrom
    np.array([                        # fractional (y, x, depth)
        [0.0, 0.0, 0.0],              # Sr
        [0.5, 0.5, 0.5],              # Ti
        [0.5, 0.5, 0.0],              # O
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]),
    np.array([38, 22, 8, 8, 8]),      # atomic numbers
).replicated((4, 4, 8))

sim = PrismEelsSimulator(
    sto,
    eV=100e3,
    semiconvergence_angle=25.0,      # mrad -- note: NOT radians
    pixels=(128, 128),
    scan=(16, 16),
    num_slices=16,
    interpolation_factor=2,
    device="cuda",
)

ti_map = sim.eels_map(
    (22, 2, 1),                      # Ti L: (Z, n, l)
    epsilon=10.0,                    # eV above the ionization threshold
    detector_mrad=25.0,              # EELS collection semi-angle
    inelastic_crop=4,
    partition=dict(n_radial=4),      # partition S1 ...
    partition_s2=dict(n_radial=4),   # ... and S2: this is BiP-PRISM
)
cube = sim.eels_cube((22, 2, 1), epsilon=10.0)   # cube.data: (16, 16, 128, 128)
cbed = sim.elastic_cbed()                        # elastic PRISM, cbed.data: (16, 16, 64, 64)
```

Quantitative EELS needs the optional `gpaw` dependency (the `eels` extra). It
supplies all-electron bound and continuum radial wavefunctions. The default,
`wavefunctions="auto"`, uses GPAW when it is installed, as v0.3 did. Without
GPAW, it falls back to hydrogenic orbitals and emits a
`HydrogenicFallbackWarning`: the pipeline runs, but the numbers are not
quantitative. v0.4 puts the transition potentials on a new absolute first-Born
scale (Å²/eV). After a best scalar fit, the map shapes differ from v0.3 by
18-61 % with the hydrogenic states. With GPAW, they differ by 0.03-0.6 % for
the K, L23 and M45 edges we tested, and by 1.4 % for Fe L1. The hydrogenic studies of the BiP-PRISM paper give different numbers, but
their conclusions hold. They also hold when the v0.4 physics layer builds the
specimen (structure, potentials, transmissions and probe); the maps then change
by up to 12 % and the partitioning errors by up to 10 % of their value. The exact published numbers came from the paper's
pre-release code, which uses the v0.3 transition potentials. See the
[STEM-EELS documentation](https://scatterem.github.io/scatterem/docs/eels.html#bip-prism-paper-v04).

## Notes on this release

* **Aberration correction is the sharpness autofocus.** It fits aberrations by
  maximising image sharpness and holds the scan rotation fixed. So
  `meta.rotation` must be correct before the fit. The autofocus needs a
  start for the low orders C10, C12a and C12b. It raises `ValueError` if it
  has none: with `correction_method="autofocus"`, C10, C12a and C12b of
  `meta.aberrations` zero (a known Cs alone does not count),
  `meta.defocus_guess` 0 and a failed bright-field-shift fit. If only these
  slots are zero, the bright-field-shift fit gives the start.
  The default `"combined"` fit does not read `meta.aberrations`.
* **Sub-pixel dark-field shifts use drizzle by default**
  (`shift_method="drizzle"`, `tcdf_shift_method="drizzle"` in the fusion).
  The figure scripts and the notebooks use this default. The paper's
  figures used Fourier upsampling and a phase ramp (`shift_method="fourier"`).
  See [Method](https://scatterem.github.io/scatterem/docs/method.html).
* **The STEM-EELS code is the forward simulation only.** There is no EELS
  reconstruction here.
* This repository holds the published methods only. The wider `scatterem`
  codebase covers more; those parts accompany their own publications.

## Authors

The authors of this software are Shengbo You, Nikita Palatkin, Sihan Shao and
Philipp Pelz, as listed in
[`CITATION.cff`](https://github.com/scatterem/scatterem/blob/main/CITATION.cff).

The FF-STEM method is the work of the paper's authors: Shengbo You, Georgios
Varnavides, Sagar Khavnekar, Nikita Palatkin, Sihan Shao, Mingjian Wu, Daniel
Stroppa, Darya Chernikova, Baixu Zhu, Ricardo Egoavil, Stefano Vespucci, Dileep
Krishnan, Xingchen Ye, Florian K. M. Schur, Erdmann Spiecker and Philipp Pelz.
BiP-PRISM is the work of Philipp Pelz.

`CITATION.cff` carries the machine-readable metadata for the software and both
papers. Please cite the paper for the part you use.

## Licence

Apache-2.0. See
[`LICENSE`](https://github.com/scatterem/scatterem/blob/main/LICENSE).
Third-party notices and data sources are in
[`THIRD-PARTY-NOTICES.md`](https://github.com/scatterem/scatterem/blob/main/THIRD-PARTY-NOTICES.md):
the MIT notice of quantem, the BSD-3-Clause notice of torchvision, the names
kept from py_multislice, abTEM and py4DSTEM (with the Krivanek aberration
labels, which v0.4 shares with abTEM), and the scattering-factor table of
Lobato and Van Dyck (2014). Please cite that article when you use the table.
