# Notebooks

The folder `notebooks/` of the repository holds marimo notebooks. molab, the
free hosted notebook service of marimo, opens each one from GitHub and runs it
on a GPU. Click a badge, switch on the GPU in the notebook specs, and run all
cells. Each notebook installs `scatterem==0.4.1` from PyPI (the PEP 723 header
of the file), so you need no account, token or local install.

The FF-STEM figure notebooks download their dataset from Zenodo. The BiP-PRISM
notebooks build their specimens in the notebook; the FePt notebook downloads
the atomic model from the data page of its authors. Each notebook explains what
the figure shows, what to look at and how its numbers differ from the paper.
Each has controls (sliders, switches and menus) that change the calculation.

## FF-STEM

Paper: S. You *et al.*, *Advanced Science* **13**, e76620 (2026),
[doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620). Data: Zenodo record
[18008901](https://zenodo.org/records/18008901), CC-BY-4.0.

| Notebook | What it shows | Runtime | Peak GPU memory | Data |
|---|---|---|---|---|
| `ffstem_demo.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py) | Au nanoparticle at low dose: direct ptychography, tcDF and fusion, sized to the GPU | 18 s | 5.5 GiB | Zenodo 18008901 (0.50 GB) |
| `fig1_gd2o3.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig1_gd2o3.py) | Figure 1, Gd₂O₃ nanoparticles at 60 kV; with a phase-sign control | 8 s | 19.7 GiB | Zenodo 18008901 (6.6 GB) |
| `fig2_carbon.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_carbon.py) | Figure 2, amorphous carbon at 300 kV: the three images and the SSNR of each channel | 6 s | 0.8 GiB | Zenodo 18008901 (0.34 GB) |
| `fig2_co3o4.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_co3o4.py) | Figure 2, Co₃O₄ at 200 kV | 6 s | 1.2 GiB | Zenodo 18008901 (0.56 GB) |
| `fig4_diffraction.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig4_diffraction.py) | Figure 4, Au nanoparticle at low dose | 17 s | 5.5 GiB | Zenodo 18008901 (0.50 GB) |

## BiP-PRISM

Paper: P. Pelz, *J. Phys. Mater.* (2026),
[doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0).

| Notebook | What it shows | Runtime | Peak GPU memory | Data |
|---|---|---|---|---|
| `bip_prism_eels_map.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_prism_eels_map.py) | A small Ti L map of SrTiO₃, as on the page Simulate STEM-EELS | 5 s | 0.7 GiB | built in the notebook |
| `bip_fept_parents.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_fept_parents.py) | FePt nanoparticle (23 196 atoms): Fe L map and partitioning error against the number of parent rings, at the paper's 936² grid and 94² scan (f = 8), full float32 precision | 95 s | 15.6 GiB | FePt model of Yang *et al.*, *Nature* **542**, 75 (2017), downloaded from the authors' data page and checked by SHA-256; no explicit licence, cite the paper |
| `bip_validity_regime.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_validity_regime.py) | Validity regime: cube and map error against thickness for O, Ti, Fe and Cu columns, probe-side partitioning only | 47 s | 8.3 GiB | built in the notebook |
| `bip_lao_sto_interface.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_lao_sto_interface.py) | Stepped and intermixed LaAlO₃/SrTiO₃ interface: five edges, partitioned against the reference, at the paper's size | 77 s | 13.3 GiB | built in the notebook (recipe and seed of supplementary S1) |
| `bip_qeels_modes.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_qeels_modes.py) | qEELS outputs: map, spectrum image, field-summed spectrum, PACBED, energy-momentum stack, and the momentum-sum invariant | 9 s | 0.8 GiB | built in the notebook |
| `bip_map_scaling.py` [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_map_scaling.py) | Map scaling, measured live: time against thickness, memory against grid, time against scan, at reduced ranges | 92 s | 27.8 GiB | built in the notebook |

## About the numbers

We measured the runtime and the peak GPU memory on an NVIDIA RTX PRO 6000 (96 GB), the GPU of molab, with `scatterem` 0.4.1 (2026-11-10). The runtime is the wall time of the whole notebook, with the
default controls, from the Python start to the last cell. It includes the
imports and the first-use compilation of the GPU kernels. It does not include
the installation on molab or the dataset download (the data were already on
disk). The peak memory is `torch.cuda.max_memory_allocated` at the end of the
notebook; every notebook prints it in its last cell.

The FF-STEM figure notebooks use the calls and parameters of the figure scripts
(see [Run the paper's figure scripts](figures.md)). With the default controls,
their three images are the images that the scripts save. The release check
compares the default output of `fig2_carbon.py` with the release reference.

The BiP-PRISM notebooks run at reduced sizes, so that each one finishes in a few
minutes on one GPU. Each notebook gives its reduction. Since v0.4.0, a map is a
probability per incident electron per eV, and v0.4 builds the transition
potentials from a new first-Born model. The paper's numbers came from the code
before v0.3. So the notebooks show the relative quantities of the paper
(errors, ratios, scaling exponents) next to the paper's values, and the absolute
map values differ from the paper. Without GPAW, the notebooks use hydrogenic
states and show the `HydrogenicFallbackWarning`; see
[Simulate STEM-EELS](eels.md#transition-potentials-and-the-gpaw-question).
