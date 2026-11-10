# Run the paper's figure scripts

The scripts are in this repository, not in the PyPI package. Clone the
repository at the release tag and install the same release from PyPI:

```bash
git clone --branch v0.4.1 https://github.com/scatterem/scatterem
cd scatterem
pip install scatterem==0.4.1
```

Four scripts under `experiments/ff_stem/`, one per specimen. Each downloads its
dataset from Zenodo on first run and verifies it by md5, so the only thing to
decide is where the data should live:

```bash
export FFSTEM_DATA_ROOT=/path/with/space
python experiments/ff_stem/Fig2_carbon.py
```

Each writes `.png` and `.npy` for all three channels into
`experiments/ff_stem/outputs/`.

:::{admonition} These scripts do not give the published figures exactly
:class: important

The scripts run the paper's pipeline with the parameters of the paper, on the
v0.4 code. Several fixes since the paper change the output, and in
`Fig4_diffraction.py` they change the sign of the direct-ptychography image.
See
[How v0.4 differs from v0.3](#how-v04-differs-from-v03) for the measured
differences and [The published figures](#the-published-figures)
for the code that made the figures.
:::

## The specimens

| Script | Specimen | Download | Output | Peak GPU |
|---|---|---|---|---|
| `Fig2_carbon.py` | amorphous carbon | 0.34 GB | 96² | 0.8 GiB |
| `Fig2_Co3O4.py` | Co₃O₄ | 0.56 GB | 185² | 1.2 GiB |
| `Fig4_diffraction.py` | Au, low dose | 0.50 GB | 1024² | 5.5 GiB |
| `Fig1_Gd2O3.py` | Gd₂O₃ | 6.6 GB | 1024² | 21.7 GiB |

"Peak GPU" is `torch.cuda.max_memory_allocated` over the whole script, measured
with v0.4 on an NVIDIA RTX A6000.

Carbon is the one to run first: smallest download, about a second of
reconstruction, and it exercises the whole pipeline.

## Run them in the browser

Each script has an interactive marimo notebook in `notebooks/`. molab, the
free hosted notebook service of marimo, opens it from GitHub. Switch on the
GPU in the notebook specs. The notebook installs `scatterem` 0.4.1, downloads
the dataset from Zenodo and runs the calls of the script with the parameters
of the script. With the default controls, its three images are the images
that the script saves. The controls change the fusion: the number of
dark-field segments, each channel on or off, the low-q handover and the band
limit (and the phase sign for Gd₂O₃). The notebook also plots the SSNR of each
channel and the fusion weights against the spatial frequency.

| Script | Notebook |
|---|---|
| `Fig2_carbon.py` | [fig2_carbon.py](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_carbon.py) |
| `Fig2_Co3O4.py` | [fig2_co3o4.py](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig2_co3o4.py) |
| `Fig4_diffraction.py` | [fig4_diffraction.py](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig4_diffraction.py) |
| `Fig1_Gd2O3.py` | [fig1_gd2o3.py](https://molab.marimo.io/github/scatterem/scatterem/blob/main/notebooks/fig1_gd2o3.py) |

[Notebooks](notebooks.md) lists all notebooks, with their runtime and GPU
memory, including the BiP-PRISM figure notebooks.

:::{admonition} Fitting a smaller GPU
:class: note

The Au dataset (`You2026AuLowDose`) takes `scan_edge_crop`, which trims the scan
and so shrinks the output. The paper uses 128 (256² scan → 1024² output,
5.5 GiB); 192 gives 128² → 512² at 1.4 GiB; 224 gives 64² → 256² at 0.4 GiB.

Reducing the scan is the lever that works. `n_batches` changes the peak memory
only a little: 5.7, 5.5, 5.4 and 5.3 GiB for 15, 25 (the script value), 60 and
150 batches.
:::

## What each script does

The same five steps, with per-specimen parameters:

1. Construct the named dataset, which downloads, verifies, and calibrates `dk`
   from the bright-field disk.
2. Set the scan rotation, and a starting C10 for the autofocus.
3. `run(dataset, AberrationFitConfig(...))` — fit aberrations by sharpness,
   rotation held fixed.
4. Reconstruct all three channels.
5. Save each with a physical scale bar.

Reading one script is enough to see how to apply the pipeline to your own data;
[Quickstart](quickstart.md) is the same thing without the paper's parameters.

## Verifying your run

Each script prints a summary line and asserts the things that quietly go wrong:

```
[Fig2-carbon] OK: dp(96, 96) tcdf(96, 96) fff(96, 96) dk=0.041975 finite=True
```

`dk` must not be 1.0 — that is the uncalibrated placeholder, and it means the
bright-field fit fell back. `finite=True` means no NaNs, which otherwise indicate
a channel that produced nothing and a division by a zero SSNR.

## Comparison plots

Where a reference is available — a simulated potential for carbon, a phantom for
Co₃O₄ — the scripts also emit a side-by-side comparison. Those references are not
in the Zenodo record, so the scripts skip that step and say so rather than
failing:

```
[Fig2-carbon] reference carbon72_potential.npy not found; skipping comparison plot.
```

<span id="how-v0.4-differs-from-v0.3"></span>

## How v0.4 differs from v0.3

We ran each script from the v0.4 tree and from the v0.3 tree, with the same
data on the same GPU (NVIDIA RTX A6000). The v0.4 run is from the release
tree of 2026-11-08, with the bright-field disk measurement of v0.4. The table compares each v0.4 output
with the v0.3 output. "relL2" is $\lVert a - b\rVert / \lVert b\rVert$ with $b$
the v0.3 array. "Affine" is the residual after a least-squares fit
$a \approx \alpha b + \beta$, divided by the spread of the v0.4 array:
$\lVert a - (\alpha b + \beta)\rVert / \lVert a - \bar a\rVert$, with $\bar a$
the mean of $a$. A pure change of gain, offset or sign gives 0. The two
columns have different denominators. An image with a large mean has
$\lVert a - \bar a\rVert \ll \lVert b\rVert$, so its Affine value can be larger
than its relL2 value (carbon tcDF: 0.64 against 0.073).

| Script | Channel | relL2 | Pearson | Affine |
|---|---|---|---|---|
| `Fig2_carbon.py` | direct ptychography | 5.8e-3 | 0.99996 | 9.0e-3 |
| | tcDF | 0.073 | 0.766 | 0.64 |
| | fused | 0.32 | 0.811 | 0.58 |
| `Fig2_Co3O4.py` | direct ptychography | 4.4e-3 | 0.99999 | 4.6e-3 |
| | tcDF | 0.40 | 0.860 | 0.51 |
| | fused | 0.45 | 0.984 | 0.18 |
| `Fig4_diffraction.py` | direct ptychography | 2.0 | −0.999 | 0.051 |
| | tcDF | 0.099 | 0.937 | 0.35 |
| | fused | 0.57 | 0.264 | 0.96 |
| `Fig1_Gd2O3.py` | direct ptychography | 0.25 | 0.969 | 0.25 |
| | tcDF | 0.28 | 0.836 | 0.55 |
| | fused | 0.11 | 0.979 | 0.20 |

The calibrated `dk` changes with the bright-field disk measurement (item 6
below): against v0.3 it is −0.36 % (carbon, 0.042126 to 0.041975 Å⁻¹),
+0.01 % (Co₃O₄), −0.03 % (Au) and −0.003 % (Gd₂O₃). Two
v0.4 runs of the same script differ by at most 2e-4 (relL2, Gd2O3) and by at
most 8e-7 for the other three.

We found the cause of each change by switching single changes back to their
v0.3 behaviour:

1. **tcDF parallax.** tcDF now shifts each detector segment by its full
   parallax. v0.3 divided the shift by $2\pi$ a second time. All four datasets
   have a non-zero defocus, so the tcDF channel changes in all four scripts
   (relL2 0.07 to 0.43). The fused image changes through its tcDF channel: by
   0.35 (carbon), 0.32 (Co₃O₄), 0.03 (Au) and 0.02 (Gd₂O₃).
2. **Sign of the ptychography phase.** The library default is now
   `phase_sign="preserve"`: it keeps the sign of the phase. v0.3 flipped each
   image to positive skewness, which is `phase_sign="positive"`. With the
   default, the Au and Gd₂O₃ direct-ptychography images change sign (Pearson −1
   against the v0.3 rule). For Au, the preserved sign agrees with the tcDF
   image, and `Fig4_diffraction.py` keeps the default. Its fused image changes
   most of all (Pearson 0.26 against v0.3). For Gd₂O₃, the preserved sign is
   opposite to the tcDF image (Pearson −0.53 between the two channels), so the
   fused image would mix opposite contrasts. `Fig1_Gd2O3.py` therefore passes `phase_sign="positive"` to
   `DirectPtychographyConfig` and to `FusedFullFieldConfig`. The direct
   ptychography and tcDF images of Gd₂O₃ then correlate positively (Pearson
   0.53; v0.3: 0.54), and the table above shows that run. With the default
   sign, the Gd₂O₃ fused image has a Pearson of only 0.30 against the run with
   `phase_sign="positive"`.
3. **Aberration autofocus.** The autofocus now uses the exact derivative of its
   objective and ignores the scan DC of each bright-field pixel. The fitted
   aberrations change: C10 from −53.5 Å to −53.8 Å (carbon), from −43.1 Å to
   −43.6 Å (Co₃O₄) and from 31.5 Å to 19.6 Å (Gd₂O₃). Before item 6, this was
   the whole direct-ptychography change of carbon and Co₃O₄ (2.3e-3 and
   3.8e-3), and it is 0.25 (affine) of Gd₂O₃. The Au script holds its
   aberrations fixed.
4. **Scan DC of the up-sampled images.** Up-sampled direct-ptychography images
   no longer carry the scan DC on the scan lattice. This changes the Au image,
   which is up-sampled 4 times, by 0.050 (affine) and its fused image by 0.018.
5. **Astigmatism frame in tcDF.** tcDF reads the astigmatism A1 in the detector
   frame. This matters only with a scan rotation: it changes the Gd₂O₃ tcDF
   image (rotation 84°) by 0.35 at the v0.3 aberrations.
6. **Bright-field disk measurement.** v0.4 measures the disk radius at the
   level halfway between the inside and the outside intensity
   (`measure_pupil`), and the dataset methods use an edge band of 2 pixels.
   v0.3 took the median radius of a threshold sweep. `dk` moves by the
   values above. On carbon, this change alone moves the outputs by 4.8e-3
   (direct ptychography), 3.6e-4 (tcDF) and 2.8e-3 (fused), relL2 against
   the v0.4 run with the v0.3 disk fit.

Two other changes have no measurable effect on these scripts: the matmul
precision (v0.3 set `"medium"` at import) and the robust level of the
bright-field mask. Each changes the outputs by about the run-to-run difference
(at most 4e-7 relL2 on carbon, Co₃O₄ and Au, 2e-4 on Gd₂O₃). With changes 1 to 5
switched back, the v0.4 code before change 6 gave the v0.3 arrays to a relL2
of 3e-7 (carbon, Co₃O₄).

## The published figures

The figures in the paper were made with an earlier version of the code, before
the later fixes. This repository does not contain that version. v0.3 already
differs from it: its bright-field disk fit is an independent implementation, and
it gives a `dk` that differs from the published one by 0.15 % (carbon), 0.07 %
(Co₃O₄) and 1.37 % (Gd₂O₃). The images look the same, but a pixel-by-pixel
comparison does not match. The v0.4 differences in the table above add to these.

The paper applied the sub-pixel dark-field shifts with the Fourier path
(`tcdf_shift_method="fourier"`). The scripts use the drizzle default, as v0.3
did. On `Fig2_carbon.py`, the two paths give fused images that correlate to
0.993 (relative L2 difference 0.066). See [Method](method.md).

Scale-bar lengths are chosen on a 1-2-5 ladder, so a bar may be a different round
number than in the paper — 10 Å where the paper shows 5 Å on the same field, for
instance. The bar is correct; only which round number it picks differs.
