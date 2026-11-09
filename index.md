---
html_theme.sidebar_secondary.remove: true
---

<img src="logo.png" class="sem-hero-logo" alt="" width="88" height="88">

# scatterem

{.sem-tagline}
4D-STEM reconstruction and differentiable STEM simulations on the GPU, in PyTorch.

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sem-hero-images

:::{grid-item}
:columns: 12 12 7 7
```{figure} docs/images/ffstem_carbon.png
:alt: Three images of the same area of amorphous carbon. Direct ptychography is strong at low spatial frequencies, tilt-corrected dark field shows the fine detail, and the fused full-field image has both.

Amorphous carbon, 300 kV: direct ptychography, tilt-corrected dark field and
their fused full-field image. Data: You *et al.*, Zenodo,
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

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} FF-STEM
Fuses direct ptychography and tilt-corrected dark field into one 4D-STEM image
with no missing band of spatial frequencies.

S. You *et al.*, *Adv. Sci.* **13** (2026) e76620,
[doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)
+++
[The method](docs/method.md)
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
::::

```bash
pip install scatterem
```

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/ffstem_demo.py)
Run the FF-STEM demo on a free GPU, with nothing to install.

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

docs/method
docs/calibration
Figure scripts <docs/figures>
docs/eels
docs/bip-prism
docs/prism
```

```{toctree}
:hidden:
:caption: Reference

API <docs/api>
docs/migrating
```

```{toctree}
:hidden:
:caption: Project

docs/cite
GitHub <https://github.com/scatterem/scatterem>
PyPI <https://pypi.org/project/scatterem/>
```
