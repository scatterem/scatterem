# Example gallery

The gallery has 13 cards. Twelve are marimo notebooks in the folder
`notebooks/` of the repository, and one is a script in `examples/`. molab, the
free hosted notebook service of marimo, opens each notebook from GitHub and
runs it on a GPU:

1. Click the molab badge of the card.
2. Switch on the GPU in the notebook specs.
3. Run all cells.

Each notebook installs `scatterem==0.5.0` from PyPI (the PEP 723 header of the
file), so you need no account, token or local install. Each notebook explains
what it shows, what to look at, and how its numbers differ from the paper. Its
controls (sliders, switches and menus) change the calculation. The image of a
notebook card is a figure of the notebook, made with the default controls.

The footer of a notebook card gives the file, the runtime, the peak GPU memory
(`GPU`) and the size of the data that the notebook downloads (`download`).
The script card has no image and no measured numbers; it says where the script
runs. See [About the numbers](#about-the-numbers) for how we measured them.

## FF-STEM

Paper: S. You *et al.*, *Advanced Science* **13**, e76620 (2026),
[doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620). Data: Zenodo
record [18008901](https://zenodo.org/records/18008901), CC-BY-4.0 (cite
[10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901)). The
notebooks download it from a byte-identical copy in the Hugging Face dataset
[philipp-pelz/public-4dstem](https://huggingface.co/datasets/philipp-pelz/public-4dstem),
with Zenodo as fallback, and check it by md5. The page [FF-STEM](ffstem.md)
explains the method. The page [Run the paper's figure scripts](figures.md)
gives the scripts behind the figure notebooks.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Au nanoparticle, low dose
:img-top: images/gallery/ffstem_demo.png
:img-alt: FF-STEM demo on the Au nanoparticle

Direct ptychography, tcDF and their fusion, sized to the GPU.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)
`ffstem_demo.py` · 18 s · GPU 5.5 GiB · download 0.50 GB
:::

:::{grid-item-card} Figure 1: Gd₂O₃ nanoparticles
:img-top: images/gallery/fig1_gd2o3.png
:img-alt: Figure 1 notebook, Gd2O3 nanoparticles

Gd₂O₃ nanoparticles at 60 kV, with a phase-sign control.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig1_gd2o3.py)
`fig1_gd2o3.py` · 8 s · GPU 19.7 GiB · download 6.6 GB
:::

:::{grid-item-card} Figure 2: amorphous carbon
:img-top: images/gallery/fig2_carbon.png
:img-alt: Figure 2 notebook, amorphous carbon

Amorphous carbon at 300 kV: the three images and the SSNR of each channel.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_carbon.py)
`fig2_carbon.py` · 6 s · GPU 0.8 GiB · download 0.34 GB
:::

:::{grid-item-card} Figure 2: Co₃O₄
:img-top: images/gallery/fig2_co3o4.png
:img-alt: Figure 2 notebook, Co3O4

Co₃O₄ at 200 kV.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_co3o4.py)
`fig2_co3o4.py` · 6 s · GPU 1.2 GiB · download 0.56 GB
:::

:::{grid-item-card} Figure 4: Au nanoparticle
:img-top: images/gallery/fig4_diffraction.png
:img-alt: Figure 4 notebook, Au nanoparticle at low dose

Au nanoparticle at low dose.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig4_diffraction.py)
`fig4_diffraction.py` · 17 s · GPU 5.5 GiB · download 0.50 GB
:::
::::

## Mixed-state ptychography

Paper: S. You *et al.*, *Transverse quantum-state characterization of
programmable electron optics*, arXiv:2608.05749 (2026),
[doi:10.48550/arXiv.2608.05749](https://doi.org/10.48550/arXiv.2608.05749).
Data: Zenodo record [24444109](https://doi.org/10.5281/zenodo.24444109),
CC-BY-4.0. The notebook downloads a lossless HDF5 copy of scan 1074 from the
Hugging Face dataset and checks it by sha256. The page
[Mixed-state ptychography](mixed-state.md) explains the method.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Measured vortex probe
:img-top: images/gallery/mems_vortex_scan1074.png
:img-alt: Nine-mode reconstruction of a measured vortex probe: mode 0 amplitude and phase, object phase, mode powers, OAM spectrum and mean OAM during the solve

A MEMS phase plate at +1.0 V, on a central crop of scan 1074:
preprocessing, a seed from bright-field shifts, a one-mode then a nine-mode
solve, the purity, the OAM spectrum and the OAM density matrix.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/mems_vortex_scan1074.py)
`mems_vortex_scan1074.py` · 12.4 min · GPU 19.5 GiB · download 4.7 GB
:::

:::{grid-item-card} Synthetic vortex state
:link: https://github.com/scatterem/scatterem/blob/v0.5.0/examples/vortex_secant_gd.py

A script, not a notebook: a two-mode vortex probe on a weak phase object,
reconstructed with `secant_gd`, one mode then three. It runs on a CPU in a
few minutes. Run `python examples/vortex_secant_gd.py` in a clone of the
repository.
+++
`examples/vortex_secant_gd.py` · CPU or GPU · no download
:::
::::

## BiP-PRISM

Paper: P. Pelz, *J. Phys. Mater.* (2026),
[doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0). The
notebooks build their specimens in the notebook. The FePt notebook downloads
the atomic model from the data page of its authors. The pages
[BiP-PRISM](bip-prism.md) and [Simulate STEM-EELS](eels.md) explain the method.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Ti L map of SrTiO₃
:img-top: images/gallery/bip_prism_eels_map.png
:img-alt: BiP-PRISM Ti L map of SrTiO3

A small Ti L map of SrTiO₃, as on the page Simulate STEM-EELS.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_prism_eels_map.py)
`bip_prism_eels_map.py` · 5 s · GPU 0.7 GiB · no download
:::

:::{grid-item-card} FePt nanoparticle
:img-top: images/gallery/bip_fept_parents.png
:img-alt: BiP-PRISM Fe L map of an FePt nanoparticle

An FePt nanoparticle (23 196 atoms): the Fe L map and the partitioning error
against the number of parent rings, at the paper's 936² grid and 94² scan
(f = 8), in full float32 precision.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_fept_parents.py)
`bip_fept_parents.py` · 95 s · GPU 15.6 GiB · download: FePt model of Yang *et al.*, *Nature* **542**, 75 (2017), checked by SHA-256; no explicit licence, cite the paper
:::

:::{grid-item-card} Validity regime
:img-top: images/gallery/bip_validity_regime.png
:img-alt: BiP-PRISM validity regime against thickness

The cube and map error against thickness for O, Ti, Fe and Cu columns, with
probe-side partitioning only.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_validity_regime.py)
`bip_validity_regime.py` · 47 s · GPU 8.3 GiB · no download
:::

:::{grid-item-card} LaAlO₃/SrTiO₃ interface
:img-top: images/gallery/bip_lao_sto_interface.png
:img-alt: BiP-PRISM maps of five edges at a LaAlO3/SrTiO3 interface, and each map minus the reference

A stepped and intermixed interface: five edges, partitioned against the
reference, at the paper's size (recipe and seed of supplementary S1).
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_lao_sto_interface.py)
`bip_lao_sto_interface.py` · 77 s · GPU 13.3 GiB · no download
:::

:::{grid-item-card} qEELS outputs
:img-top: images/gallery/bip_qeels_modes.png
:img-alt: BiP-PRISM qEELS outputs

The map, spectrum image, field-summed spectrum, PACBED, energy-momentum stack,
and the momentum-sum invariant.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_qeels_modes.py)
`bip_qeels_modes.py` · 9 s · GPU 0.8 GiB · no download
:::

:::{grid-item-card} Map scaling
:img-top: images/gallery/bip_map_scaling.png
:img-alt: BiP-PRISM map scaling measurements

Map scaling, measured live: time against thickness, memory against grid, and
time against scan, at reduced ranges.
+++
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/bip_map_scaling.py)
`bip_map_scaling.py` · 92 s · GPU 27.8 GiB · no download
:::
::::

## About the numbers

We measured the runtime and the peak GPU memory on an NVIDIA RTX PRO 6000
(96 GB), the GPU of molab, with `scatterem` 0.4.1 (2026-11-10). The runtime is
the wall time of the whole notebook, with the default controls, from the
Python start to the last cell (the mixed-state notebook: from its GPU check to
the last cell, with `scatterem` 0.5.0). It includes the imports and the
first-use compilation of the GPU kernels. It does not include the installation
on molab or the dataset download (the data were already on disk). The peak
memory is `torch.cuda.max_memory_allocated` at the end of the notebook; every
notebook prints it in its last cell.

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
