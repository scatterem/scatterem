# Gap-Free Information Transfer in 4D-STEM via Fusion of Complementary Scattering Channels

Scripts that run the reconstruction pipeline of the paper figures with the
current scatterem code, for:

> S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao, M. Wu, D. Stroppa,
> D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci, D. Krishnan, X. Ye,
> F. K. M. Schur, E. Spiecker, P. Pelz. *"Gap-Free Information Transfer in
> 4D-STEM via Fusion of Complementary Scattering Channels."* Advanced Science
> **13**, e76620 (2026). [doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)
> Keywords: electron microscopy, 4D-STEM, ptychography, dark-field STEM, data fusion.

The outputs differ from the published figures. Fixes made after the paper
change the tcDF parallax, the sign convention of the direct-ptychography phase,
the fitted aberrations (autofocus), the scan DC of up-sampled images (Au fused
image 0.018), the frame of the A1 astigmatism in tcDF (Gd₂O₃ tcDF 0.35) and
the bright-field disk measurement, which sets `dk` (carbon −0.36 %).
The scripts also use the drizzle tcDF shift, and the paper used the Fourier
shift. `docs/figures.qmd` gives the measured differences against v0.3 and
their causes.

## Method overview

Each script runs the same three-stage reconstruction pipeline on a 4D-STEM
dataset: **direct (single-side-band, SSB) ptychography** recovers the
low-to-mid spatial frequencies from the bright-field disk (phase-contrast,
gap-free at low q but band-limited to twice the bright-field aperture);
**tilt-corrected dark field (tcDF)** recovers complementary high-frequency
information from the dark-field scattering that direct ptychography cannot
reach; and **fused full-field (FFF)** combines the two channels by an
SSNR-weighted Wiener (Fourier) fusion, producing a single image with gap-free
information transfer across the whole accessible spatial-frequency range —
the central result of the paper. All three reconstruction stages
(`DirectPtychographyConfig`, `TiltCorrectedDarkFieldConfig`,
`FusedFullFieldConfig`, each passed to `scatterem.reconstruction.run`)
live in the `scatterem` core library; these scripts
configure and drive that library for each specimen, unmodified.

## Scripts

| Script | Specimen | kV / semiangle / dr | Data source[^1] | Peak GPU memory |
| --- | --- | --- | --- | --- |
| `Fig1_Gd2O3.py`       | Gd₂O₃ nanoparticles         | 60 / 30 mrad / 0.43 Å  | `You2026Gd2O3` | 21.7 GiB |
| `Fig2_carbon.py`      | amorphous carbon            | 300 / 19.68 mrad / 0.25 Å | `You2026Carbon` | 0.8 GiB |
| `Fig2_Co3O4.py`       | Co₃O₄                        | 200 / 21 mrad / 0.20 Å | `You2026Co3O4` | 1.2 GiB |
| `Fig4_diffraction.py` | Au nanoparticle (low dose)  | 200 / 30 mrad / 0.727 Å | `You2026AuLowDose` | 5.5 GiB |

[^1]: Each entry is a `scatterem.data.public` class that downloads its data
(`from scatterem.data.public import <name>`). The peak GPU memory is
`torch.cuda.max_memory_allocated` over the whole script, with v0.4 on an
NVIDIA RTX A6000.

All four scripts ran end to end on one 47 GB NVIDIA RTX A6000 against the
real Zenodo data (md5-checked) and produced finite direct-ptychography, tcDF
and fused outputs. Each script prints an `OK:` line, for example
`[Fig1] OK: dp(1024, 1024) tcdf(1024, 1024) fff(1024, 1024) dk=0.034589 finite=True`.

`Fig2_carbon.py`, `Fig2_Co3O4.py` and `Fig4_diffraction.py` also make a
comparison plot against a reference file (`carbon72_potential.npy`,
`sim1ppn.npy`, `au30mrad_lowdose_up4.npy`). These files are not in the Zenodo
record. When a file is absent, the script skips its comparison plot and prints
a message.

## Reconstruction outputs

Each script saves its three reconstruction images (direct ptychography,
tcDF, fused full-field) as `.npy` arrays and `.png` previews to `outputs/`.
The script makes the directory. `.gitignore` ignores it, so run the script
again to make the outputs.

## Data

The four datasets published with the paper are wrapped as named datasets in the
core library and download themselves on first use:

```python
from scatterem.data.public import You2026Gd2O3
dataset = You2026Gd2O3(root=DATA_ROOT, download=True, device="cuda")
```

Each class carries its specimen's acquisition constants (energy, semiconvergence
angle, scan step, scan rotation), applies any repair of the raw acquisition, and
calibrates the reciprocal-space pixel size from the measured bright-field disk.
Source: Zenodo record [18008901](https://zenodo.org/records/18008901)
(DOI 10.5281/zenodo.18008901), **CC-BY-4.0** — please cite it if you use the data.

Files are cached flat in `root`, so an existing local copy is reused (md5-verified)
rather than re-downloaded. A cold `You2026Gd2O3` fetch is 6.6 GB.

This release has no scripts for the other figures of the paper.

## Configuration

- `FFSTEM_DATA_ROOT` — directory the datasets are cached in. Defaults to
  `~/.cache/scatterem/you2026`.
- `FFSTEM_HEADLESS` — set to `1` to force the non-interactive Matplotlib
  `Agg` backend (also auto-enabled when no `DISPLAY` is set), so scripts run
  without a GUI, e.g. in CI or over SSH.

## Running

From this directory:

```bash
FFSTEM_HEADLESS=1 python Fig1_Gd2O3.py
```

Each script is also usable interactively (VS Code / Jupyter) via its
preserved `# %%` cell markers.

### GPU memory note

`Fig1_Gd2O3.py` (a 512×512 scan, 6.6 GB download) has the largest peak,
21.7 GiB on the RTX A6000: the fusion runs the tilt-corrected dark field
again, and its full-scan reduction allocates a large temporary array. On a GPU
with less free memory, or one that other jobs share, run Fig1 with the
expandable PyTorch allocator. This reduces the risk of a fragmentation OOM,
and no code change is necessary:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FFSTEM_HEADLESS=1 \
  python Fig1_Gd2O3.py
```

The `scatterem` library itself is not modified by this package — these
scripts exercise the already-merged FFF reconstruction code path
(`scatterem.reconstruction.run` with `DirectPtychographyConfig`,
`TiltCorrectedDarkFieldConfig`, `FusedFullFieldConfig` and
`AberrationFitConfig`, and `Dataset4DStem.crop_brightfield_`,
`.bright_field_radius_and_center_`) and the `scatterem.vis` visualization
helpers.
