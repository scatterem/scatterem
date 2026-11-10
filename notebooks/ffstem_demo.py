# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.2",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "torch>=2.5.0",
# ]
# ///
"""FF-STEM fusion demo on the Au low-dose dataset, as a marimo notebook."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # FF-STEM fusion demo: Au nanoparticle, low dose

    This notebook reconstructs a real 4D-STEM acquisition with the FF-STEM
    fusion pipeline of `scatterem`. It runs three steps end to end on a GPU:

    - **Direct (SSB) ptychography** reconstructs the low and middle spatial
      frequencies from the phase gradient in the bright-field disk.
    - **Tilt-corrected dark field (tcDF)** recovers the higher spatial
      frequencies from the parallax shift of the dark-field scattering
      outside the bright-field disk.
    - **Fused full field (FFF)** combines the two channels with an
      SSNR-weighted Wiener fusion in Fourier space. The result is one image
      with no missing frequency band.

    **Paper:** S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao,
    M. Wu, D. Stroppa, D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci,
    D. Krishnan, X. Ye, F. K. M. Schur, E. Spiecker, P. Pelz, *"Gap-Free
    Information Transfer in 4D-STEM via Fusion of Complementary Scattering
    Channels"*, **Advanced Science** 13, e76620 (2026),
    doi:[10.1002/advs.76620](https://doi.org/10.1002/advs.76620).

    **Data:** Zenodo record [18008901](https://zenodo.org/records/18008901)
    (DOI 10.5281/zenodo.18008901), licence **CC-BY-4.0**. If you use the
    data, cite the record and the paper.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a
       GPU (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    The code and the data are public, so you need no token or account.
    The header of this file pins `scatterem==0.4.2`, and molab installs
    it. The dataset (about 500 MB) downloads on first use from the
    Hugging Face mirror of the Zenodo record (Zenodo is the fallback), and
    its md5 checksums are verified. Cite the Zenodo DOI. Set `FFSTEM_DATA_ROOT` to keep the data in a
    different directory (the default is `data`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Check the GPU
    """)
    return


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    _props = torch.cuda.get_device_properties(0)
    TOTAL_GIB = _props.total_memory / 2**30
    print(f"GPU: {_props.name} ({TOTAL_GIB:.1f} GiB total)")
    return TOTAL_GIB, os, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Size the scan to the GPU

    The configuration of the paper (`scan_edge_crop=128`, a 256x256 scan
    upsampled to a 1024x1024 image) needs about **5.5 GiB** of GPU memory.
    The table gives the measured peak memory of the cells below
    (`torch.cuda.max_memory_allocated`, scatterem 0.4 on an NVIDIA RTX
    A6000):

    | `scan_edge_crop` | scan | image | peak GPU memory |
    | --- | --- | --- | --- |
    | 128 (paper) | 256x256 | 1024x1024 | 5.5 GiB |
    | 192 | 128x128 | 512x512 | 1.4 GiB |
    | 224 | 64x64 | 256x256 | 0.4 GiB |

    On a small GPU, a larger `scan_edge_crop` makes the scan smaller.
    `n_batches` changes the peak only a little. The cell below selects the
    paper configuration when the GPU has 8 GiB or more. You can set
    `SCAN_EDGE_CROP` yourself.
    """)
    return


@app.cell
def _(TOTAL_GIB):
    if TOTAL_GIB >= 8:
        SCAN_EDGE_CROP = 128
        PROFILE = "paper (256x256 scan -> 1024x1024 image, about 5.5 GiB peak)"
    else:
        SCAN_EDGE_CROP = 192
        PROFILE = "reduced (128x128 scan -> 512x512 image, about 1.4 GiB peak)"

    print(f"Selected profile: {PROFILE}")
    print(f"scan_edge_crop = {SCAN_EDGE_CROP}")
    return (SCAN_EDGE_CROP,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Load the dataset

    `You2026AuLowDose` downloads three files of the Zenodo record (the
    Dectris master file and its two data blocks, about 500 MB) from the
    Hugging Face mirror, with Zenodo as fallback, and checks the md5 of each.
    If the files are already in the data directory, it uses them. The class
    also repairs the acquisition:

    1. It reshapes the raw frame stream into a square scan grid.
    2. It replaces a known bad 2x2 detector patch with the mean of the
       ring around it.
    3. It removes `scan_edge_crop` unreliable rows and columns from each
       scan edge.

    Then it calibrates the reciprocal-space pixel size `dk` from the
    measured bright-field disk radius.
    """)
    return


@app.cell
def _(SCAN_EDGE_CROP, os):
    from scatterem.data.public import You2026AuLowDose

    DATA_ROOT = os.environ.get("FFSTEM_DATA_ROOT", "data")
    dataset = You2026AuLowDose(
        root=DATA_ROOT,
        download=True,
        device="cuda",
        scan_edge_crop=SCAN_EDGE_CROP,
    )
    print(dataset)
    print(f"scan shape: {tuple(int(s) for s in dataset.array.shape[:2])}")
    print(f"detector shape: {tuple(int(s) for s in dataset.detector_shape)}")
    print(f"dk = {dataset.dk} 1/Angstrom")
    print(f"fluence = {dataset.fluence:.3g} e-/Angstrom^2")
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Fit the aberrations

    The parameters are those of the Figure 4 script of the paper. The fit
    is a sharpness ("autofocus") fit of the aberrations up to second order.
    It maximizes the total variation (`sharpness_metric="tv"`) in a 150x150
    region of interest and starts from a defocus guess.
    """)
    return


@app.cell
def _(dataset):
    from scatterem.reconstruction import run
    from scatterem.reconstruction.ptychography.direct import (
        AberrationFitConfig,
        DirectPtychographyConfig,
    )

    dataset.meta.aberrations.array[0] = -150.0
    dataset.meta = run(
        dataset,
        AberrationFitConfig(
            bright_field_mask_threshold=0.1,
            correction_method="autofocus",
            sharpness_metric="tv",
            bin_factors=(1, 1, 1),
            verbosity=1,
            correct_order=2,
            num_iterations=50,
            lr=1,
            roi_shape=(150, 150),
            upsample=1.0,
        ),
    ).applied_to(dataset.meta)
    aberrations_fitted = True
    return DirectPtychographyConfig, aberrations_fitted, run


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Reconstruct the three channels
    """)
    return


@app.cell
def _(DirectPtychographyConfig, aberrations_fitted, dataset, run):
    from scatterem.reconstruction.darkfield import TiltCorrectedDarkFieldConfig
    from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig

    assert aberrations_fitted
    direct_ptycho_image, ssnr_ptycho = run(
        dataset,
        DirectPtychographyConfig(
            upsample="nyquist",
            n_batches=25,
            return_snr=True,
            verbosity=1,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()

    tcDF, ssnr_tcdf = run(
        dataset,
        TiltCorrectedDarkFieldConfig(
            n_dark_field_segments=16,
            verbosity=0,
            bright_field_mask_threshold=0.1,
            upsample="nyquist",
            return_snr=True,
        ),
    ).as_tuple()

    # verbosity=1 is quieter than the verbosity=2 of Fig4_diffraction.py.
    # verbosity=2 also plots the intermediate images, which the next section
    # shows.
    fff, phase_weighted, tcdf_weighted = run(
        dataset,
        FusedFullFieldConfig(
            verbosity=1,
            bright_field_mask_threshold=0.3,
            n_dark_field_segments=32,
        ),
    ).as_tuple()

    print(f"direct ptychography: {tuple(direct_ptycho_image.shape)}")
    print(f"tilt-corrected dark field: {tuple(tcDF.shape)}")
    print(f"fused full field: {tuple(fff.shape)}")
    return direct_ptycho_image, fff, tcDF


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Look at the results

    The scale bars are physical, in Angstrom. The reconstructions are
    upsampled relative to the scan. Thus one image pixel is the scan step
    times (scan pixels / image pixels), not the scan step itself.
    """)
    return


@app.cell
def _(dataset, direct_ptycho_image, fff, mo, plt, tcDF):
    import scatterem.vis as vis
    from scatterem.data import Sampling

    def image_sampling(img):
        """Return the physical pixel size of an upsampled image."""
        ny, nx = img.shape[-2:]
        return Sampling(
            pixel_size=(
                float(dataset.sampling[0]) * int(dataset.array.shape[0]) / ny,
                float(dataset.sampling[1]) * int(dataset.array.shape[1]) / nx,
            ),
            units=("Å", "Å"),
        )

    _figures = []
    for _img, _title in (
        (direct_ptycho_image, "Direct ptychography (phase)"),
        (tcDF, "Tilt-corrected dark field"),
        (fff, "Fused full field"),
    ):
        _fig, _ax = plt.subplots(figsize=(6, 6))
        vis.show_2d(
            [_img],
            cbar=True,
            title=[_title],
            figax=(_fig, _ax),
            sampling=image_sampling(_img),
            clip_percentile=(2, 98),
        )
        _figures.append(_fig)
    mo.vstack(_figures)
    return


@app.cell
def _(fff, torch):
    assert fff is not None
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Other datasets

    The same Zenodo record (18008901) has three more named datasets in
    `scatterem.data.public`. Each class holds the acquisition constants of
    its specimen:

    - `You2026Carbon`: amorphous carbon (Figure 2 of the paper). It is the
      smallest of the four, about 0.8 GiB peak GPU memory.
    - `You2026Co3O4`: Co3O4 nanoparticles (Figure 2), about 1.2 GiB.
    - `You2026Gd2O3`: Gd2O3 nanoparticles (Figure 1). It is a 6.6 GB
      download and needs about 21.7 GiB peak GPU memory.

    The `notebooks/` folder of the repository has one notebook for each
    paper figure. The scripts that they run are in `experiments/ff_stem/`.
    """)
    return


if __name__ == "__main__":
    app.run()
