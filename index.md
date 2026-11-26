---
html_theme.sidebar_secondary.remove: true
---

<img src="logo.png" class="sem-hero-logo" alt="" width="88" height="88">

# scatterem

{.sem-tagline}
Differentiable scattering models and imaging algorithms for microscopy

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sem-hero-images

:::{grid-item}
:columns: 12 12 7 7
```{figure} docs/images/ffstem_carbon.png
:alt: Three images of the same area of amorphous carbon with Poisson noise at 2 × 10⁴ electrons per square angstrom. Direct ptychography shows the structure with low noise, tilt-corrected dark field is noisy, and the fused full-field image follows direct ptychography.

Amorphous carbon, 300 kV, with Poisson noise at 2 × 10{sup}`4` e/Å² added:
direct ptychography, tilt-corrected dark field and their fused full-field
image. The fusion weights each channel by its SSNR, so at this dose it keeps
mostly the direct ptychography. Data: You *et al.*, Zenodo,
[doi:10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901),
CC-BY-4.0.
```
:::

:::{grid-item}
:columns: 12 12 5 5
```{figure} docs/images/bip_prism_sto.png
:alt: Two simulated core-loss maps of SrTiO3 viewed along 001. The Ti L map shows a square lattice of bright Ti-O columns; the O K map shows the oxygen columns.

SrTiO3 [001], 100 kV: Ti L{sub}`2,3` and O K core-loss maps, simulated by
BiP-PRISM with analytic hydrogenic states (not quantitative).
```
:::
::::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} FF-STEM
Fuses direct ptychography and tilt-corrected dark field into one 4D-STEM image
with no missing band of spatial frequencies.

S. You *et al.*, *Adv. Sci.* **13** (2026) e76620,
[doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)
+++
[FF-STEM](docs/ffstem.md)
:::

:::{grid-item-card} BiP-PRISM
Simulates core-loss STEM-EELS. It partitions both scattering matrices, so a
full-resolution elemental map fits on one GPU.

P. Pelz, *J. Phys. Mater.* (2026),
[doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)
+++
[BiP-PRISM](docs/bip-prism.md)
:::

:::{grid-item-card} Elastic PRISM
Simulates elastic 4D-STEM data from the plane-wave and the partitioned
scattering matrix.

P. Pelz, *J. Phys. Mater.* (2026),
[doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)
+++
[Elastic PRISM](docs/prism.md)
:::

:::{grid-item-card} Mixed-state ptychography
Reconstructs a 2D object and a mixed-state probe with second-order solvers,
and measures the orbital angular momentum of the probe modes.

S. You *et al.*, arXiv:2608.05749 (2026),
[doi:10.48550/arXiv.2608.05749](https://doi.org/10.48550/arXiv.2608.05749)
+++
[API reference](docs/api.md)
:::
::::

```bash
pip install scatterem
```

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)
Run the FF-STEM demo on a free GPU, with nothing to install, or pick one of
the notebooks of the [example gallery](docs/gallery.md).

[Quickstart](docs/quickstart.md) · [API](docs/api.md) · [Cite](docs/cite.md)

```{toctree}
:hidden:
:caption: Get started

docs/install
docs/quickstart
```

```{toctree}
:hidden:
:caption: User guide

docs/gallery
docs/ffstem
docs/calibration
Figure scripts <docs/figures>
docs/mixed-state
docs/eels
docs/bip-prism
docs/prism
```

```{toctree}
:hidden:
:caption: Reference

API <docs/api>
docs/architecture
```

```{toctree}
:hidden:
:caption: Project

docs/cite
GitHub <https://github.com/scatterem/scatterem>
PyPI <https://pypi.org/project/scatterem/>
```
