# Gap-free Information Transfer in 4D-STEM via Fusion of Complementary Scattering Channels

Figure-reproduction scripts for:

> S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao, M. Wu, D. Stroppa,
> D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci, X. Ye, F. K. M. Schur,
> E. Spiecker, P. Pelz. *"Gap-free Information Transfer in 4D-STEM via Fusion
> of Complementary Scattering Channels."*
> Keywords: electron microscopy, 4D-STEM, ptychography, dark-field STEM, data fusion.

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
(`Dataset4dstem.direct_ptychography`, `.tilt_corrected_dark_field`,
`.fused_full_field`) live in the `scatterem` core library; these scripts
configure and drive that library for each specimen, unmodified.

## Scripts

| Script | Specimen | kV / semiangle / dr | Data source[^1] | Runs here? |
| --- | --- | --- | --- | --- |
| `Fig1_Gd2O3.py`       | Gd₂O₃ nanoparticles         | 60 / 30 mrad / 0.43 Å  | `You2026Gd2O3` | verified end-to-end on a remote NVIDIA H200 NVL, not locally (6.6 GB dataset; the local GPU was busy with concurrent jobs — see GPU memory note) |
| `Fig2_carbon.py`      | amorphous carbon            | 300 / 19.68 mrad / 0.25 Å | `You2026Carbon` | verified |
| `Fig2_Co3O4.py`       | Co₃O₄                        | 200 / 21 mrad / 0.20 Å | `You2026Co3O4` | verified |
| `Fig4_diffraction.py` | Au nanoparticle (low dose)  | 200 / 30 mrad / 0.727 Å | `You2026AuLowDose` | verified |
| `Fig4_nanotube.py`    | carbon nanotube             | 80 / 25 mrad / 0.316 Å | `nanotube171.npy`* | guarded (dense array absent locally; not covered by the named datasets — see Data section) |
| `Fig5_bio.py`         | biological (Müller)         | 300 / 7 mrad / 28.7 Å  | `42_transpose.npy`* | guarded (data absent locally) |
| `Fig6_vlp.py`         | virus-like particle         | 200 / 30.6 mrad / 11.0 Å | `fig6_vlp.npy`* | guarded (data absent locally) |

[^1]: Unstarred entries are self-downloading `scatterem.datasets` classes
(`from scatterem.datasets import <name>`). Starred (`*`) entries are bare
filenames you must place under `FFSTEM_DATA_ROOT` yourself — see the "not
covered" table in the Data section below.

"Verified" means the script was run headlessly end-to-end against the
monorepo's `scatterem` library and produced finite, correctly-shaped
direct-ptychography / tcDF / fused outputs. "Guarded" means the dataset is
not present on this machine; the script exits cleanly (`sys.exit(0)`) at a
data-existence check with an explanatory message instead of crashing.
`Fig2_carbon.py`, `Fig2_Co3O4.py` and `Fig4_diffraction.py` were verified on
the local 47 GB RTX A6000. `Fig1_Gd2O3.py` was verified end-to-end on a
remote **NVIDIA H200 NVL** against the real 6.6 GB `fig1_gd2o3.npy`
(md5-confirmed), producing
`[Fig1] OK: dp(1024, 1024) tcdf(1024, 1024) fff(1024, 1024) finite=True` — it
was not re-run locally, since its ~13 GB on-device footprint was judged too
large to run safely alongside other jobs already using the local GPU.

Each script also has a figure-specific comparison/composite plot (against a
parallax reconstruction or a ground-truth potential) that is itself guarded
behind the existence of its reference file — those reference files
(`sim1ppn.npy`, `au30mrad_lowdose_up4.npy`, `pptube171.npy`,
`carbon72_potential.npy`) are not present locally, so those comparison plots
are skipped with a message on this machine.

## Reconstruction outputs

Verified scripts save their three reconstruction images (direct ptychography,
tcDF, fused full-field) as both `.npy` arrays and `.png` previews to
`outputs/` (gitignored — regenerate by running the script; nothing under
`outputs/` is committed except the `.gitkeep` placeholder).

## Data

The four datasets published with the paper are wrapped as named datasets in the
core library and download themselves on first use:

```python
from scatterem.datasets import You2026Gd2O3
dataset = You2026Gd2O3(root=DATA_ROOT, download=True, device="cuda")
```

Each class carries its specimen's acquisition constants (energy, semiconvergence
angle, scan step, scan rotation), applies any repair of the raw acquisition, and
calibrates the reciprocal-space pixel size from the measured bright-field disk.
Source: Zenodo record [18008901](https://zenodo.org/records/18008901)
(DOI 10.5281/zenodo.18008901), **CC-BY-4.0** — please cite it if you use the data.

Files are cached flat in `root`, so an existing local copy is reused (md5-verified)
rather than re-downloaded. A cold `You2026Gd2O3` fetch is 6.6 GB.

Three scripts are **not** covered:

| Script | Why |
| --- | --- |
| `Fig4_nanotube.py` | the record's `171_new_alignment.h5` is a *sparse* `counts`/`indices` 4-defocus focal series, not the dense cube this script loads; densifying it needs a defocus choice and resolution of a scan-sampling discrepancy (0.334 vs 0.316 Å) |
| `Fig5_bio.py` | `42_transpose.npy` is not in the record |
| `Fig6_vlp.py` | `fig6_vlp.npy` is not in the record |

These three keep their `FFSTEM_DATA_ROOT` + local-file behavior and exit cleanly
when the data is absent.

## Configuration

- `FFSTEM_DATA_ROOT` — directory the datasets are cached in (and where the
  three unwrapped datasets above are looked up). Defaults to
  `/media/philipp/data_2/inr_datasets/fused_data_shengbo`.
- `FFSTEM_HEADLESS` — set to `1` to force the non-interactive Matplotlib
  `Agg` backend (also auto-enabled when no `DISPLAY` is set), so scripts run
  without a GUI, e.g. in CI or over SSH.

## Running

```bash
FFSTEM_HEADLESS=1 /home/philipp/projects/scatterem/.venv/bin/python \
  packages/scatterem_experiments/experiments/ff_stem/Fig1_Gd2O3.py
```

Each script is also usable interactively (VS Code / Jupyter) via its
preserved `# %%` cell markers.

### GPU memory note

The other three reconstructions were verified on a single 47 GB RTX A6000 and
are comfortable. The largest dataset (`Fig1_Gd2O3.py`, a 512×512 scan, ~6.6 GB
raw, ~13 GB on device) peaks near the memory ceiling — `fused_full_field()`
internally re-runs the tilt-corrected dark field, whose full-scan reduction
allocates a large temporary — so it was verified end-to-end on a remote
NVIDIA H200 NVL rather than re-run against the local RTX A6000, which had
other jobs on it (see the Scripts table above). On a memory-tight or shared
GPU, run Fig1 with PyTorch's expandable allocator to reduce the risk of a
fragmentation OOM (no code change needed):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FFSTEM_HEADLESS=1 \
  /home/philipp/projects/scatterem/.venv/bin/python \
  packages/scatterem_experiments/experiments/ff_stem/Fig1_Gd2O3.py
```

The `scatterem` library itself is not modified by this package — these
scripts exercise the already-merged FFF reconstruction code path
(`Dataset4dstem.direct_ptychography`, `.tilt_corrected_dark_field`,
`.fused_full_field`, `.determine_aberrations_`, `.crop_brightfield_`,
`.bright_field_radius_and_center_`) and the `scatterem.vis` visualization
helpers.
