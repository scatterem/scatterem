# scatterem — fused full-field STEM

Reference implementation of the fused full-field (FF-STEM) reconstruction:
direct (single-side-band) ptychography, tilt-corrected dark-field imaging, and
their SSNR-weighted Wiener fusion into one image that carries both the
low-frequency phase contrast of ptychography and the high-frequency detail of the
dark field.

S. You, G. Varnavides, S. Khavnekar, *et al.*, "Gap-Free Information Transfer in
4D-STEM via Fusion of Complementary Scattering Channels", *Advanced Science*
(2026) e76620. [doi.org/10.1002/advs.76620](https://doi.org/10.1002/advs.76620)

Data: [doi.org/10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901) (CC-BY-4.0)
Citation metadata: `CITATION.cff`

## Try it in the browser

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scatterem/scatterem/blob/main/docs/colab/ffstem_you2026_demo.ipynb)

The notebook installs the package, fetches one dataset from Zenodo and runs the
full reconstruction on a free Colab T4. Nothing to set up locally. It sizes the
scan to the GPU it lands on, so it also works on the smaller instances.

## Install

```bash
pip install git+https://github.com/scatterem/scatterem
```

A CUDA GPU is strongly recommended — the reconstructions are FFT- and
kernel-bound, and the NVIDIA Warp kernels this uses need one for anything beyond
a small test. Install a `torch` build matching your driver first if the default
wheel does not.

## Reproduce the paper figures

Each script downloads the dataset it needs from Zenodo on first run and verifies
it by md5, so the only thing to set is where the data should live:

```bash
export FFSTEM_DATA_ROOT=/path/with/space
python experiments/ff_stem/Fig2_carbon.py
```

| Script | Specimen | Download | Peak GPU memory |
| --- | --- | --- | --- |
| `Fig2_carbon.py` | amorphous carbon | 0.34 GB | ~1.2 GiB |
| `Fig2_Co3O4.py` | Co₃O₄ | 0.56 GB | — |
| `Fig4_diffraction.py` | Au, low dose | 0.5 GB | 2.5–17.6 GiB, set by `scan_edge_crop` |
| `Fig1_Gd2O3.py` | Gd₂O₃ | 6.6 GB | ~19.6 GiB |

`Fig1_Gd2O3.py` and the paper's `Fig4` configuration will not fit a free
Colab T4; the others will.

## Use it on your own data

```python
import torch
from scatterem import Dataset4dstem

dataset = Dataset4dstem.from_array(
    array,                          # (scan_y, scan_x, det_y, det_x)
    energy=300e3,                   # eV
    semiconvergence_angle=19.7e-3,  # rad
    scan_step=0.25,                 # Angstrom
    device="cuda",
)

# The autofocus is a local optimisation, so give it a starting defocus.
dataset.meta.aberrations.array[0] = -50.0   # Angstrom
dataset.determine_aberrations_(correct_order=2)

ptycho = dataset.direct_ptychography(upsample=2.0)
dark_field = dataset.tilt_corrected_dark_field(upsample=2.0)
fused = dataset.fused_full_field(upsample=2.0)
```

## Notes on this release

* **Aberration correction is the sharpness autofocus.** It fits aberrations by
  maximising image sharpness and holds the scan rotation fixed, so
  `meta.rotation` must already be right on the way in. It needs a non-zero
  starting guess and raises if `meta.aberrations` is all zero, rather than
  silently starting from nothing.
* **Sub-pixel dark-field shifts use Fourier upsampling and a phase ramp**
  (`shift_method="fourier"`), which is what the paper used.
* This is the FF-STEM pipeline only. The wider `scatterem` codebase covers
  iterative ptychography, tomography, and multislice simulation; those parts
  accompany their own publications and are not here.

## The `master` branch

`master` is the code published with the paper. It is kept so that it stays
available, and it receives no further development.

Its bright-field disk fit was replaced so that the branch carries no third-party
GPL-3.0 code and its Apache-2.0 licence is accurate. That shifts the reciprocal
sampling `dk` slightly — 0.07% to 1.37% depending on the dataset — so `master` no
longer reproduces the published numbers to the last digit, though the images are
unchanged. For a pixel-exact reproduction of the paper, use the tag
[`master-as-published`](../../tree/master-as-published).

This branch is an independent reimplementation of the same method: the same
algorithms written from the published descriptions rather than adapted from
`master`, so the two share no code. Use this one for anything new.

## Authors

This code was written by Shengbo You and Philipp Pelz.

The method is the work of the paper's authors: Shengbo You, Georgios Varnavides, Sagar Khavnekar, Nikita Palatkin, Sihan Shao, Mingjian Wu, Daniel Stroppa, Darya Chernikova, Baixu Zhu, Ricardo Egoavil, Stefano Vespucci, Dileep Krishnan, Xingchen Ye, Florian K. M. Schur, Erdmann Spiecker and Philipp Pelz.

`CITATION.cff` carries the machine-readable version for both the software and the
paper. If you use either, please cite the paper.

## Licence

Apache-2.0. See `LICENSE`.
