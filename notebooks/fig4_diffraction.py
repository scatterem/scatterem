# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.4.2",
#     "marimo==0.25.2",
#     "matplotlib>=3.10",
#     "numpy>=2.1.0",
#     "torch>=2.5.0",
# ]
# ///
"""Paper Figure 4: Au nanoparticle at low dose (FF-STEM), as an interactive marimo notebook."""

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
    # FF-STEM, Figure 4: Au nanoparticle at low dose

    This notebook reconstructs one experimental 4D-STEM dataset of the FF-STEM
    paper with three methods, and lets you change the fusion:

    - **Direct ptychography** (single side band, SSB) gets the low and middle
      spatial frequencies from the bright-field disk.
    - **Tilt-corrected dark field (tcDF)** gets complementary frequencies, out
      to the 2α limit, from the scattering outside the bright-field disk.
    - **Fused full field (FFF)** adds the two channels in Fourier space. The
      weight of each channel at each spatial frequency is its spectral
      signal-to-noise ratio (SSNR) divided by the sum of the two SSNRs.

    The specimen is an Au nanoparticle, recorded at 200 kV and at low dose on a
    counting detector with a 0.727 Å scan step. The dataset class
    repairs a bad detector patch, removes 128 unreliable rows and
    columns at each scan edge and applies the 180° scan rotation of
    this acquisition.

    The notebook uses the same code path and parameters as the figure script
    `experiments/ff_stem/Fig4_diffraction.py` of the `scatterem` repository. With the
    default controls, the three images are the images that the script saves.
    The output differs from the published figure, because the code changed
    after the paper; `docs/figures.md` in the repository gives the
    differences and their causes.

    **Paper:** S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao,
    M. Wu, D. Stroppa, D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci,
    D. Krishnan, X. Ye, F. K. M. Schur, E. Spiecker, P. Pelz, *"Gap-Free
    Information Transfer in 4D-STEM via Fusion of Complementary Scattering
    Channels"*, **Advanced Science** 13, e76620 (2026),
    doi:[10.1002/advs.76620](https://doi.org/10.1002/advs.76620).

    **Data:** `You2026AuLowDose` from Zenodo record
    [18008901](https://zenodo.org/records/18008901)
    (DOI 10.5281/zenodo.18008901), licence **CC-BY-4.0**. If you use the
    data, cite the record and the paper.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a
       GPU (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    The dataset (about 0.50 GB in three files) downloads on first use from the
    Hugging Face mirror of the Zenodo record (Zenodo is the fallback), and
    its md5 checksums are verified. Cite the Zenodo DOI. Set `FFSTEM_DATA_ROOT` to keep the data in a
    different directory (the default is `data`).

    ## What to look at

    - In the SSNR plot of section 7, the ptychography SSNR goes to zero
      at the lowest frequencies, because the method transfers no DC.
      The tcDF SSNR is high there.
    - The fusion takes the lowest frequencies from tcDF and the other
      frequencies, out to 2α/λ, from ptychography. Thus the summed
      SSNR has no gap from zero to 2α/λ.
    - In the images, the fused image keeps the lattice detail of the
      ptychography image and the flat background of the tcDF image.

    ## Differences from the paper

    The code changed after the paper, so the images differ from the
    published figure. The library now keeps the sign of the phase (`preserve`), so the
    direct ptychography image has the opposite sign of the published
    image. This sign agrees with the tcDF image. Against v0.3 the
    fused image moves by 0.57 (relative L2). The page `docs/figures.md` of the
    repository gives all measured differences and their causes.
    """)
    return


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On molab, select a GPU in the notebook specs "
            "and restart the notebook."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    return np, os, plt, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load the dataset

    The dataset class holds the acquisition constants (energy, convergence
    angle, scan step and, where needed, the scan rotation). It downloads the
    data and calibrates the reciprocal pixel size `dk` from the measured
    bright-field disk.
    """)
    return


@app.cell
def _(os, torch):
    from scatterem.data.public import You2026AuLowDose

    DATA_ROOT = os.environ.get("FFSTEM_DATA_ROOT", "data")
    dataset = You2026AuLowDose(
        root=DATA_ROOT, download=True, device=torch.device("cuda")
    )
    print(dataset)
    print(f"scan shape: {tuple(int(s) for s in dataset.array.shape[:2])}")
    print(f"dk = {float(dataset.sampling[2]):.5g} 1/Å")
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Fit the aberrations

    The fit maximizes the total-variation sharpness
    (`sharpness_metric="tv"`) up to second-order aberrations. The
    region of interest is 150x150, because the cropped scan is
    smaller than the scans of the other figures. The fit starts from C10 = -150.0 Å (defocus = −C10).
    """)
    return


@app.cell
def _(dataset):
    from scatterem.reconstruction import run
    from scatterem.reconstruction.ptychography.direct import AberrationFitConfig

    dataset.meta.aberrations.array[0] = -150.0  # initial C10 guess (Å); defocus == -C10
    dataset.meta = run(
        dataset,
        AberrationFitConfig(
            bright_field_mask_threshold=0.1,
            correction_method="autofocus",
            sharpness_metric="tv",
            bin_factors=(1, 1, 1),
            # verbosity=1 as in the figure script. On the Gd2O3 data the fit
            # result moves by about 1e-4 (relative) with verbosity=0.
            verbosity=1,
            correct_order=2,
            num_iterations=50,
            lr=1,
            roi_shape=(150, 150),
            upsample=1.0,
        ),
    ).applied_to(dataset.meta)
    print(dataset.meta.aberrations)
    return (run,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Controls

    The other controls change only the fusion:

    - **Dark-field segments** sets `n_dark_field_segments` of the fusion's
      tcDF channel (the paper uses 32). More segments give a finer
      tilt correction and less signal per segment. A change recomputes the
      tcDF channel of the fusion.
    - **Use ... in the fusion** switches one channel off: its SSNR is set to
      zero, so the other channel gets the full weight.
    - **Low-q handover** gives the ptychography weight below this fraction of
      the Nyquist frequency to the dark field, with a cosine ramp. It
      suppresses the low-frequency halo of the ptychography phase. The
      paper uses 0 (off). The handover moves weight to tcDF also when the
      tcDF switch is off, as `fused_full_field` does.
    - **Band limit** sets the hard cutoff of both weights, as a fraction of
      the 2α limit (the paper uses 1).
    """)
    return


@app.cell
def _(mo):
    segments = mo.ui.dropdown(
        options=["8", "16", "32", "64"], value="32", label="Dark-field segments"
    )
    use_ptycho = mo.ui.checkbox(value=True, label="Use ptychography in the fusion")
    use_tcdf = mo.ui.checkbox(value=True, label="Use tcDF in the fusion")
    lowq = mo.ui.slider(
        0.0, 0.3, step=0.01, value=0.0, label="Low-q handover (fraction of Nyquist)"
    )
    band = mo.ui.slider(
        0.5, 1.0, step=0.05, value=1.0, label="Band limit (fraction of 2α)"
    )
    mo.vstack([segments, use_ptycho, use_tcdf, lowq, band])
    return band, lowq, segments, use_ptycho, use_tcdf


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Reconstruct the three channels

    The three calls below are the calls of the figure script. The fusion
    runs its own direct ptychography and tcDF, with the fusion parameters
    (`bright_field_mask_threshold=0.3`, 32 dark-field segments), so its
    channels are not the two images on the left.
    """)
    return


@app.cell
def _(dataset, run, torch):
    from scatterem.reconstruction.ptychography.direct import (
        DirectPtychographyConfig,
    )

    direct_ptycho_image, _ssnr = run(
        dataset,
        DirectPtychographyConfig(
            upsample="nyquist",
            n_batches=25,
            return_snr=True,
            verbosity=0,
            bright_field_mask_threshold=dataset.bright_field_mask_threshold,
        ),
    ).as_tuple()
    return (direct_ptycho_image,)


@app.cell
def _(dataset, run):
    from scatterem.reconstruction.darkfield import TiltCorrectedDarkFieldConfig
    from scatterem.reconstruction.fusion.ffstem import FusedFullFieldConfig

    tcdf_image, _ssnr = run(
        dataset,
        TiltCorrectedDarkFieldConfig(
            n_dark_field_segments=16,
            verbosity=0,
            bright_field_mask_threshold=0.1,
            upsample="nyquist",
            return_snr=True,
            snr_blur_sigma=0.0,
        ),
    ).as_tuple()

    FUSION_DEFAULTS = dict(
        verbosity=0,
        bright_field_mask_threshold=0.3,
        n_dark_field_segments=32,
    )
    fusion_default = run(dataset, FusedFullFieldConfig(**FUSION_DEFAULTS))
    return FUSION_DEFAULTS, FusedFullFieldConfig, fusion_default, tcdf_image


@app.cell
def _(direct_ptycho_image, fusion_default, tcdf_image):
    # The three arrays that the figure script saves.
    panel_arrays = {
        "direct_ptychography": direct_ptycho_image.detach().cpu().numpy(),
        "tcdf": tcdf_image.detach().cpu().numpy(),
        "fused_full_field": fusion_default.fused.detach().cpu().numpy(),
    }
    print({_k: _v.shape for _k, _v in panel_arrays.items()})
    return (panel_arrays,)


@app.cell
def _(dataset):
    from scatterem.data import Sampling

    def image_sampling(img):
        """Return the pixel size of an upsampled image, in Å."""
        _ny, _nx = img.shape[-2:]
        return Sampling(
            pixel_size=(
                float(dataset.sampling[0]) * int(dataset.array.shape[0]) / _ny,
                float(dataset.sampling[1]) * int(dataset.array.shape[1]) / _nx,
            ),
            units=("Å", "Å"),
        )

    return (image_sampling,)


@app.cell
def _(image_sampling, plt):
    import scatterem.vis as vis

    def show_panels(images, titles, roles, units):
        """Show images side by side, each with its scale bar and colour bar."""
        _fig, _axes = plt.subplots(1, len(images), figsize=(4.6 * len(images), 4.6))
        for _ax, _img, _title, _role, _unit in zip(
            _axes, images, titles, roles, units, strict=True
        ):
            _n = len(_fig.axes)
            vis.show_2d(
                [_img],
                cbar=True,
                title=[_title],
                figax=(_fig, _ax),
                role=_role,
                sampling=image_sampling(_img),
                clip_percentile=(2, 98),
                tight_layout=False,
            )
            if len(_fig.axes) > _n:
                _fig.axes[-1].set_ylabel(_unit)
        _fig.tight_layout()
        return _fig

    return (show_panels,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The three images of the figure

    These are the outputs of the figure script, with the default controls.
    The display clips each image at its 2nd and 98th percentiles.
    """)
    return


@app.cell
def _(direct_ptycho_image, fusion_default, show_panels, tcdf_image):
    show_panels(
        [direct_ptycho_image, tcdf_image, fusion_default.fused],
        ["Direct ptychography (SSB)", "Tilt-corrected dark field", "Fused full field"],
        ["phase", "intensity", "phase"],
        ["phase (rad)", "intensity (arb. units)", "fused signal (rad scale)"],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Change the fusion

    The cell below re-runs the fusion only when you change the number of
    dark-field segments. It reuses the ptychography
    channel of the default fusion when it can. Then it fuses the two channels
    again with your switches, low-q handover and band limit.
    """)
    return


@app.cell
def _(
    FUSION_DEFAULTS,
    FusedFullFieldConfig,
    dataset,
    fusion_default,
    run,
    segments,
):
    _settings = dict(FUSION_DEFAULTS, n_dark_field_segments=int(segments.value))
    if _settings == FUSION_DEFAULTS:
        fusion = fusion_default
    else:
        _reuse = {}
        if _settings.get("phase_sign") == FUSION_DEFAULTS.get("phase_sign"):
            _reuse["ptycho"] = fusion_default.ptycho
        fusion = run(dataset, FusedFullFieldConfig(**_settings), reuse=_reuse)
    return (fusion,)


@app.cell
def _(band, dataset, fusion, lowq, np, torch, use_ptycho, use_tcdf):
    from scatterem.reconstruction.fusion.ffstem import fused_full_field

    _ssnr_p = torch.nan_to_num(fusion.ptycho.ssnr) * float(use_ptycho.value)
    _ssnr_t = torch.nan_to_num(fusion.tcdf.ssnr) * float(use_tcdf.value)
    if not (use_ptycho.value or use_tcdf.value):
        raise ValueError("Switch on at least one channel of the fusion.")
    fused_image, _, _ = fused_full_field(
        dataset,
        fusion.phase_image,
        fusion.tcdf.image,
        _ssnr_p,
        _ssnr_t,
        ptycho_lowq_cutoff_frac=float(lowq.value),
        q_hard_cutoff_frac=float(band.value),
    )

    # The fusion weights, computed as fused_full_field computes them, for the
    # plot below.
    _den = _ssnr_p + _ssnr_t + 1e-12
    _w_p = torch.where(_ssnr_p > 0, _ssnr_p / _den, torch.zeros_like(_den))
    _w_t = torch.where(_ssnr_t > 0, _ssnr_t / _den, torch.zeros_like(_den))
    _ny, _nx = _w_p.shape
    _d = float(dataset.sampling[0]) * int(dataset.array.shape[0]) / _ny
    _qy = torch.fft.fftfreq(_ny, d=_d, device=_w_p.device)
    _qx = torch.fft.fftfreq(_nx, d=_d, device=_w_p.device)
    q_norm = torch.sqrt(_qy[:, None] ** 2 + _qx[None, :] ** 2)
    if lowq.value > 0:
        _q_lp = float(lowq.value) * float(q_norm.max())
        _ramp = 0.5 * (1.0 - torch.cos(np.pi * (q_norm / _q_lp).clamp(0.0, 1.0)))
        _w_t = _w_t + _w_p * (1.0 - _ramp)
        _w_p = _w_p * _ramp
    two_alpha = 2.0 * dataset.meta.semiconvergence_angle / dataset.meta.wavelength
    _band = q_norm <= float(band.value) * two_alpha
    weights = (_w_p * _band, _w_t * _band)
    return fused_image, q_norm, two_alpha, weights


@app.cell
def _(fusion, fusion_default, fused_image, show_panels):
    _diff = float(
        (fused_image - fusion_default.fused).norm() / fusion_default.fused.norm()
    )
    print(f"relative L2 difference from the paper settings: {_diff:.3g}")
    print(
        f"fusion channels: ptychography {tuple(fusion.phase_image.shape)}, "
        f"tcDF {tuple(fusion.tcdf.image.shape)}"
    )
    show_panels(
        [fusion_default.fused, fused_image],
        ["Fused, paper settings", "Fused, your settings"],
        ["phase", "phase"],
        ["fused signal (rad scale)", "fused signal (rad scale)"],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Transfer of each channel

    The upper plot gives the azimuthal mean of the SSNR of each channel of
    the fusion, and their sum, against the spatial frequency |q|. The
    vertical lines mark α/λ (the radius of the bright-field disk) and 2α/λ
    (the limit of both channels). The lower plot gives the fusion weights
    with your settings. A gap in the transfer shows as a dip of the summed
    SSNR to zero.
    """)
    return


@app.cell
def _(dataset, fusion, plt, q_norm, two_alpha, use_ptycho, use_tcdf, weights):
    from scatterem.utils.spectral_mesh import shell_means

    _ny = fusion.phase_image.shape[-2]
    _d = float(dataset.sampling[0]) * int(dataset.array.shape[0]) / _ny
    _q, _s_p = shell_means(fusion.ptycho.ssnr.nan_to_num(), _d)
    _, _s_t = shell_means(fusion.tcdf.ssnr.nan_to_num(), _d)
    _, _w_p = shell_means(weights[0], _d)
    _, _w_t = shell_means(weights[1], _d)
    _on_p, _on_t = float(use_ptycho.value), float(use_tcdf.value)
    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    _a1.semilogy(
        _q,
        _on_p * _s_p + _on_t * _s_t,
        color="#b2182b",
        lw=4,
        alpha=0.6,
        label="fused (sum of the channels used)",
    )
    _a1.semilogy(
        _q,
        _s_p,
        color="#1b7837",
        ls="--",
        lw=2,
        label="direct ptychography" + ("" if _on_p else " (off)"),
    )
    _a1.semilogy(
        _q,
        _s_t,
        color="#5e3c99",
        ls="--",
        lw=2,
        label="tcDF" + ("" if _on_t else " (off)"),
    )
    _a1.set_ylabel("SSNR(|q|)")
    _a1.set_ylim(bottom=max(1e-4, float(min(_s_p.max(), _s_t.max())) * 1e-4))
    _a1.legend(frameon=False)
    _a2.plot(_q, _w_p, color="#1b7837", lw=2, label="weight of ptychography")
    _a2.plot(_q, _w_t, color="#5e3c99", lw=2, label="weight of tcDF")
    _a2.set_ylabel("fusion weight")
    _a2.set_xlabel("spatial frequency |q| (1/Å)")
    _a2.set_ylim(-0.05, 1.05)
    _a2.legend(frameon=False)
    for _ax in (_a1, _a2):
        for _x, _lab in ((two_alpha / 2, "α/λ"), (two_alpha, "2α/λ")):
            _ax.axvline(_x, color="0.4", lw=1, ls=":")
        _ax.grid(alpha=0.3)
    _a1.text(two_alpha / 2, _a1.get_ylim()[1], " α/λ", va="top", color="0.3")
    _a1.text(two_alpha, _a1.get_ylim()[1], " 2α/λ", va="top", color="0.3")
    _a2.set_xlim(0, min(float(q_norm.max()), 1.15 * two_alpha))
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(fused_image, panel_arrays, torch):
    import math

    assert all(math.isfinite(float(abs(_v).sum())) for _v in panel_arrays.values())
    assert bool(torch.isfinite(fused_image).all())
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
