# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scatterem==0.5.0",
#     "marimo==0.25.2",
#     "h5py>=3.10",
#     "hdf5plugin>=4.4",
#     "matplotlib>=3.10",
#     "numpy>=2.1.0",
#     "scipy>=1.13",
#     "torch>=2.5.0",
# ]
# ///
"""Mixed-state ptychography of a measured vortex probe (MEMS phase plate, scan 1074)."""

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
    # Mixed-state ptychography of a measured vortex probe

    This notebook reconstructs one measured 4D-STEM scan, from the raw counts
    to the orbital angular momentum (OAM) of the probe. A MEMS electrostatic
    spiral phase plate in the condenser aperture plane makes a vortex probe.
    The probe scans gold nanoparticles on amorphous carbon. Ptychography
    recovers the object and the probe as a mixed state: several incoherent
    modes, each with its own power. From the modes, the notebook computes:

    - the **purity** $\sum_m p_m^2$ of the probe state ($p_m$ is the power
      fraction of mode $m$);
    - the **OAM spectrum** $P_l$ and the **mean OAM** $\langle l \rangle$ in
      units of $\hbar$.

    All steps use the public `scatterem` API, numpy, scipy and torch:

    1. download and read the data;
    2. correct, centre and point-invert the patterns;
    3. measure the needle shadow of the device and build the masks;
    4. measure a first aperture phase from the bright-field image shifts
       (stage 1);
    5. refine one coherent probe mode, then nine modes (stage 2);
    6. read the purity and the OAM spectrum.

    **Paper:** S. You *et al.*,
    *"Transverse quantum-state characterization of programmable electron
    optics"*, arXiv:2608.05749 (2026),
    doi:[10.48550/arXiv.2608.05749](https://doi.org/10.48550/arXiv.2608.05749).

    **Data:** scan 1074 (specimen area 2, device at +1.0 V) of the paper's
    purity series. The raw files are in Zenodo record
    [24444109](https://doi.org/10.5281/zenodo.24444109), licence
    **CC-BY-4.0**. The notebook downloads a lossless HDF5 copy from the
    Hugging Face dataset `philipp-pelz/public-4dstem`; it decodes to the raw
    frames bit for bit. If you use the data, cite the record and the paper.

    ## Before you run this

    1. Switch on the GPU. On molab, open the notebook specs and select a
       GPU (molab gives an NVIDIA RTX PRO 6000 with 96 GB).
    2. Run all cells.

    The data file (4.7 GB) downloads on first use, and the notebook checks
    its sha256. To use a local copy, set `MEMS_VORTEX_FILE` to its path.
    With the default controls, the run took 12.4 minutes on an RTX PRO 6000
    (the GPU of molab) with the file already on disk, and used 19.5 GiB
    of GPU memory. The download adds its own time.

    ## What to look at

    - Section 4: the stage-1 aperture already carries most of the vortex
      winding. Its mean OAM is about +15 ħ, measured from the data alone.
    - Section 5: the one-mode refinement moves the mean OAM to about +20 ħ.
    - Section 7: the nine-mode state. The mean OAM is the robust
      measurement. The purity is an output of these settings (see below).
    - Section 8: the OAM density matrix, before and after the removal of
      the astigmatism that the device adds. Its off-diagonal elements are
      not a coherence measure (see there).

    ## Differences from the paper

    The paper's reconstruction used all 512 × 512 positions, 40 one-mode
    passes and then 150 nine-mode passes (end a), several hours on a
    GH200 GPU. Its result for this scan is purity 0.1535 and mean OAM
    +20.46 ħ (end a; end b: 0.1587 and +20.46 ħ; Fig. 3 of the paper).
    This notebook differs
    in four ways:

    1. **Crop.** It uses a central crop of the scan (default 128 × 128
       positions, 1/16 of the scan).
    2. **Nine-mode passes.** It uses 20 nine-mode passes, not 150.
    3. **One-mode passes.** It uses 120 one-mode passes, not 40. On the
       crop, the mean OAM of the one-mode probe still rises after 40
       passes (+19.0 ħ at 40, +20.0 ħ at 120, inner-ring sorter), and a
       one-mode pass costs about one tenth of a nine-mode pass.
    4. **Seed.** The paper's first aperture came from research code that
       is not part of scatterem. Section 4 replaces it with a short
       bright-field shift measurement written in this notebook. A
       flat-phase disk is a poor seed: after 120 one-mode passes its mean
       OAM was only +12.3 ħ (loss 0.249, against 0.222 from the stage-1
       seed).

    The paper's controls with simulated data showed that the mean OAM is
    a measurement (within 2 %). Purity recovery needs a strongly
    scattering object and a large scan (512 × 512 positions). On a smaller
    scan, the purity changes with the batch size, the start state and the
    number of passes. Thus expect a mean OAM near the paper's value and a
    purity that is much higher than 0.15. The
    purity falls slowly with the nine-mode passes, and a crop has fewer
    solver steps per pass than the full scan. Measured on the same data
    (RTX PRO 6000; the times are for the solve only):

    | Crop | One-mode passes | Nine-mode passes | Purity | Mean OAM, inner rings (ħ) | Solve time |
    |---|---|---|---|---|---|
    | 96 | 40 | 30 | 0.884 | +16.41 | 7 min |
    | 96 | 120 | 20 | 0.896 | +17.75 | 7 min |
    | 128 | 40 | 30 | 0.792 | +18.83 | 12 min |
    | 128 | 40 | 150 | 0.533 | +18.88 | 60 min |
    | **128** | **120** | **20** | **0.815** | **+19.68** | **11 min** |
    | 192 | 120 | 20 | 0.705 | +19.73 | 25 min |
    | paper: 512 | 40 | 150 | 0.1535 | +20.38 | several hours (GH200) |

    The column "inner rings" comes from an interim sorter: 4× interpolation,
    rings 4 to 39 px, no astigmatism removal. On the paper's probes it gives
    +20.38 ħ. Section 7 uses the sorter of the paper (Section V E), with
    rings over the whole aperture and the astigmatism removed. With it, the
    default run gave mean OAM +19.68 ħ and purity 0.815; two repeats
    with the same settings (one with a different batch partition) gave
    +19.74 and +19.78 ħ (purity 0.819 and 0.816). The paper's probes give
    +20.46 ħ with this sorter. A smaller crop gives a lower mean OAM; a
    larger crop gives a lower purity. More nine-mode passes lower the
    purity but do not move the mean OAM.

    An earlier sorter of the paper's analysis (no interpolation, rings 4 to
    39 px, $|l| \le 30$) gave +20.19 / +20.36 ħ for this scan. The paper
    (Section V E) shows that rings to 39 px only make the mean OAM depend
    on the translation gauge by up to 1.6 ħ, so it uses rings over the whole
    aperture. Section 7 prints the value of the earlier sorter too.
    """)
    return


@app.cell
def _():
    import hashlib
    import json
    import math
    import os
    import time

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
    T_START = time.perf_counter()
    return T_START, hashlib, json, math, np, os, plt, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Download the data

    The file holds `/data`, the counts as `(512, 512, 192, 192)` uint16
    (scan rows, scan columns, detector rows, detector columns), and the
    detector corrections: `/flag_mask` (pixels that the detector flags in
    every frame; their counts are 0), `/pixel_mask` and `/flatfield_gain`.
    The attribute `card_json` of `/metadata` holds the dataset card: the
    acquisition constants, the citation and notes. The notebook reads every
    acquisition constant from it.
    """)
    return


@app.cell
def _(hashlib, json, os):
    import h5py
    import hdf5plugin  # noqa: F401  (registers the Blosc2 filter of /data)

    REPO = "philipp-pelz/public-4dstem"
    PATH_IN_REPO = "data/You2026_MEMSvortex/You2026_MEMSvortex_scan1074.h5"
    SHA256 = "94324d7e3256c4040f754a13d8a1a661f053e0d073b54b3d79e3f34319da9d31"

    def sha256(path):
        """The sha256 of a file, read in 16 MB blocks."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        return h.hexdigest()

    data_file = os.environ.get("MEMS_VORTEX_FILE")
    if data_file is None:
        from scatterem.data.public.utils import download_url

        _root = os.path.expanduser("~/.cache/scatterem/public-4dstem")
        data_file = os.path.join(_root, os.path.basename(PATH_IN_REPO))
        # download_url skips the download if the file is already there.
        download_url(
            f"https://huggingface.co/datasets/{REPO}/resolve/main/{PATH_IN_REPO}",
            _root,
        )
        if sha256(data_file) != SHA256:
            raise RuntimeError(f"sha256 of {data_file} does not match; delete it")
    print(f"data file: {data_file}")
    with h5py.File(data_file, "r") as _f:
        card = json.loads(_f["metadata"].attrs["card_json"])
        print(f"/data {_f['data'].shape} {_f['data'].dtype}")
    acq = card["acquisition"]
    print(card["title"])
    for _k in (
        "energy_eV",
        "semiconvergence_rad",
        "dk_invA",
        "scan_step_A",
        "rotation_deg",
    ):
        print(f"  {_k}: {acq[_k]}")
    # The paper's result for this scan (the card's notes give it as text).
    PAPER = dict(purity=(0.1535, 0.1587), mean_oam=(20.46, 20.46))
    return PAPER, acq, data_file, h5py


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Controls

    - **Scan crop** sets the side of the central crop of the 512 × 512
      scan. The run time grows with the number of positions (the square
      of the side). The full scan (512) needs about 40 GB of GPU memory for
      the patterns alone. The smallest crop is 96; a smaller crop gives
      a lower mean OAM (see the table above).
    - **One-mode passes** sets the passes of the coherent (one-mode)
      refinement. The paper uses 40.
    - **Nine-mode passes** sets the passes of the mixed-state refinement.
      The paper uses 150.

    Press the button to start a run with new values.
    """)
    return


@app.cell
def _(mo):
    DEFAULTS = dict(crop=128, one_mode_passes=120, mode_passes=20)
    controls = (
        mo.md(
            """
        {crop}

        {one_mode_passes}

        {mode_passes}
        """
        )
        .batch(
            crop=mo.ui.slider(
                96,
                512,
                step=32,
                value=DEFAULTS["crop"],
                label="Scan crop (positions per side)",
            ),
            one_mode_passes=mo.ui.slider(
                0,
                200,
                step=10,
                value=DEFAULTS["one_mode_passes"],
                label="One-mode passes",
            ),
            mode_passes=mo.ui.slider(
                10,
                150,
                step=10,
                value=DEFAULTS["mode_passes"],
                label="Nine-mode passes",
            ),
        )
        .form(submit_button_label="Run with these values")
    )
    controls
    return DEFAULTS, controls


@app.cell
def _(DEFAULTS, controls):
    settings = dict(DEFAULTS, **(controls.value or {}))
    print(settings)
    return (settings,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Correct and centre the patterns

    The steps repeat the paper's preprocessing:

    1. Divide the counts by the flat-field gain. Fill the flagged and masked
       pixels with the local mean of their neighbours in the mean pattern
       (a later sub-pixel shift moves them, so a zero weight cannot follow
       them).
    2. Flip the detector rows. This is the frame of the paper's runs.
    3. Fit a circle to the edge of the bright-field disk in the mean
       pattern. A ray that ends in the needle shadow is an outlier, and the
       fit drops it.
    4. Compute the centre of mass (COM) of each pattern inside 1.3 disk
       radii, over all 512 × 512 positions. Fit a plane to the COM over the
       scan: this is the descan.
    5. Shift each pattern of the crop by minus the plane and by the
       sub-pixel offset of the disk centre (a Fourier shift). Put the disk
       centre on pixel 96.
    6. Point-invert the detector axes (k → −k), as the paper's runs do:
       `roll(flip(p, (-2, -1)), (1, 1), (-2, -1))`.

    The full pass over the file (step 4) reads all 4.7 GB once. It keeps
    only the crop on the GPU.
    """)
    return


@app.cell
def _(math, np):
    from scipy.ndimage import map_coordinates, uniform_filter

    def wavelength_A(energy_eV):
        """Relativistic electron wavelength in angstroms."""
        return 12.2642588 / math.sqrt(energy_eV * (1.0 + 0.978475e-6 * energy_eV))

    def fit_disk(mean, n_theta=720):
        """Centre ``(cy, cx)`` and radius of the bright-field disk, in pixels.

        The edge point on each ray from the centroid is the outermost
        crossing of half the disk level. A circle fit with outlier rejection
        drops the rays that end in the needle shadow.
        """
        m = np.asarray(mean, np.float64)
        vals = m[m > 0.3 * np.percentile(m, 99)]
        m = np.minimum(m, np.median(vals))  # clip hot pixels
        level = 0.5 * np.median(m[m > 0.3 * m.max()])
        yy, xx = np.indices(m.shape)
        cy, cx = yy[m > level].mean(), xx[m > level].mean()
        r = np.arange(0, 0.75 * min(m.shape), 0.25)
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        py = cy + r[None] * np.sin(th)[:, None]
        px = cx + r[None] * np.cos(th)[:, None]
        prof = map_coordinates(m, [py.ravel(), px.ravel()], order=1).reshape(
            n_theta, -1
        )
        above = prof >= level
        last = above.shape[1] - 1 - np.argmax(above[:, ::-1], 1)
        last = np.clip(last, 0, len(r) - 2)
        p0 = prof[np.arange(n_theta), last]
        p1 = prof[np.arange(n_theta), last + 1]
        edge = r[last] + np.clip((p0 - level) / np.maximum(p0 - p1, 1e-12), 0, 1) * (
            r[1] - r[0]
        )
        ey, ex = cy + edge * np.sin(th), cx + edge * np.cos(th)
        keep = np.ones(n_theta, bool)
        for _ in range(5):
            a = np.stack([2 * ey[keep], 2 * ex[keep], np.ones(keep.sum())], 1)
            (y0, x0, c), *_ = np.linalg.lstsq(
                a, ey[keep] ** 2 + ex[keep] ** 2, rcond=None
            )
            rad = math.sqrt(c + y0**2 + x0**2)
            res = np.hypot(ey - y0, ex - x0) - rad
            keep = np.abs(res) < max(1.0, 3 * np.std(res[keep]))
        return float(y0), float(x0), float(rad)

    return fit_disk, uniform_filter, wavelength_A


@app.cell
def _(
    acq,
    data_file,
    fit_disk,
    h5py,
    np,
    settings,
    time,
    torch,
    uniform_filter,
    wavelength_A,
):
    def load_scan(path, crop, device="cuda"):
        """Corrected, centred, point-inverted patterns of the central crop."""
        t0 = time.time()
        f = h5py.File(path, "r")
        data = f["data"]
        ns, nx, h, w = data.shape
        gain = f["flatfield_gain"][()].astype(np.float64)
        bad = (
            (f["pixel_mask"][()] != 0)
            | f["flag_mask"][()]
            | ~np.isfinite(gain)
            | (gain <= 0)
        )
        g = torch.as_tensor(
            np.where(bad, 1.0, gain), dtype=torch.float32, device=device
        )
        badt = torch.as_tensor(bad, device=device)

        def row(iy):
            c = torch.as_tensor(data[iy].astype(np.float32), device=device) / g
            c[:, badt] = 0
            return c

        # Step 1: mean pattern from every 8th scan row; fill values of bad pixels.
        acc = torch.zeros((h, w), dtype=torch.float64, device=device)
        for iy in range(0, ns, 8):
            acc += row(iy).sum(0, dtype=torch.float64)
        mean = (acc / (len(range(0, ns, 8)) * nx)).cpu().numpy()
        fill = uniform_filter(np.where(bad, 0, mean), 5) / np.maximum(
            uniform_filter((~bad).astype(float), 5), 1e-6
        )
        fillt = torch.as_tensor(fill[bad], dtype=torch.float32, device=device)
        # Steps 2 and 3: row flip, disk fit.
        cy, cx, radius = fit_disk(np.where(bad, fill, mean)[::-1])
        # Step 4: COM over all positions; keep the crop.
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        win = (torch.hypot(yy - cy, xx - cx) < 1.3 * radius).float()
        com = torch.empty(ns, nx, 2, device=device, dtype=torch.float64)
        c0 = (ns - crop) // 2
        cube = torch.empty((crop, crop, h, w), dtype=torch.float32, device=device)
        for iy in range(ns):
            c = row(iy)
            c[:, badt] = fillt
            c = c.flip(1)
            b = c * win
            t = b.sum((1, 2))
            com[iy, :, 0] = (b * yy).sum((1, 2)) / t
            com[iy, :, 1] = (b * xx).sum((1, 2)) / t
            if c0 <= iy < c0 + crop:
                cube[iy - c0] = c[c0 : c0 + crop]
        f.close()
        iy_, ix_ = np.meshgrid(np.arange(ns), np.arange(nx), indexing="ij")
        a = np.stack([iy_.ravel(), ix_.ravel(), np.ones(iy_.size)], 1)
        com_np = com.reshape(-1, 2).cpu().numpy()
        coef, *_ = np.linalg.lstsq(a, com_np, rcond=None)
        plane = (a @ coef).reshape(ns, nx, 2)
        plane -= plane.mean((0, 1))
        # Step 5: Fourier shift by minus the plane plus the sub-pixel offset,
        # then an integer window that puts the disk centre on pixel h // 2.
        y0, x0 = int(round(cy - h // 2)), int(round(cx - w // 2))
        pad = max(0, -y0, -x0, y0, x0)
        frac = np.array([y0 + h // 2 - cy, x0 + w // 2 - cx])
        shift = torch.as_tensor(
            -plane[c0 : c0 + crop, c0 : c0 + crop].reshape(-1, 2) + frac,
            dtype=torch.float32,
            device=device,
        )
        ky = torch.fft.fftfreq(h, device=device)[:, None]
        kx = torch.fft.fftfreq(w, device=device)[None, :]
        flat = cube.view(-1, h, w)
        for j in range(0, flat.shape[0], 2048):
            s = shift[j : j + 2048]
            ramp = torch.exp(
                -2j * torch.pi * (s[:, 0, None, None] * ky + s[:, 1, None, None] * kx)
            )
            c = torch.fft.ifft2(
                torch.fft.fft2(flat[j : j + 2048]) * ramp
            ).real.clamp_min(0)
            c = torch.nn.functional.pad(c, (pad, pad, pad, pad))
            c = c[:, y0 + pad : y0 + pad + h, x0 + pad : x0 + pad + w]
            # Step 6: point inversion k -> -k about pixel h // 2.
            flat[j : j + 2048] = torch.roll(torch.flip(c, dims=(1, 2)), (1, 1), (1, 2))
        print(
            f"disk centre ({cy:.2f}, {cx:.2f}) px, radius {radius:.2f} px; "
            f"descan peak-to-peak {np.ptp(plane[..., 0]):.2f} / {np.ptp(plane[..., 1]):.2f} px; "
            f"{time.time() - t0:.0f} s"
        )
        return cube, radius, plane

    intensities, disk_radius, descan_plane = load_scan(data_file, int(settings["crop"]))
    print(
        f"patterns {tuple(intensities.shape)}, "
        f"{intensities.numel() * 4 / 2**30:.1f} GiB on the GPU, "
        f"mean counts per pattern {float(intensities.sum((2, 3)).mean()):.0f}"
    )
    _dk_fit = acq["semiconvergence_rad"] / disk_radius / wavelength_A(acq["energy_eV"])
    print(
        f"dk from the disk fit {_dk_fit:.6f} 1/A; the card gives {acq['dk_invA'][0]} 1/A"
    )
    return descan_plane, disk_radius, intensities


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Masks and stage 1: the first aperture

    **Needle band.** The device holds the phase plate on a needle, and the
    needle casts a shadow across the bright-field disk. The shadow pixels
    are the pixels inside the disk below half the median; their mean
    direction from the centre is the needle azimuth. The band has a half
    width of 7 px × R / 35.65, as in the paper (14 px here). Its pixels
    stay in the data with zero weight (the **detector mask**). The **probe
    support** is the bounding box of the measured pixels without the band,
    widened by 2 px; a constraint sets the probe's aperture plane to zero
    outside it after each pass.

    **Stage 1.** Each detector pixel inside the bright-field disk gives a
    virtual bright-field image. For a weak object, the image of pixel
    $\mathbf{k}$ is the object shifted by $\nabla\chi(\mathbf{k}) / 2\pi$,
    with $\chi$ the aperture phase. The cell measures these shifts on a
    2 × 2-binned detector: it registers each pair of neighbour images
    (`pairwise_relative_shifts`) and solves for one shift per image
    (`synchronize_shifts`). A Gaussian low pass (5 scan steps) keeps the
    object frequencies for which the shift model holds near a vortex core.
    Then a least-squares integration gives $\chi$ on the disk without the
    needle band and without a 2-pixel core. That domain is simply
    connected, so a vortex winding shows up as a jump of $\chi$ across the
    band; no charge is assumed. The shift-to-gradient map
    $\nabla\chi = -2\pi\,\Delta r\,\Delta k\,b\,R(\theta)\,\mathbf{s}$
    (scan step $\Delta r$, detector pixel $\Delta k$, binning $b$, scan
    rotation $\theta$) was checked on data simulated with the same forward
    model: for a planted charge-20 vortex it returned a seed with a mean
    OAM of +15.4 ħ (truth +18.4 ħ; both with the inner-ring sorter of the
    introduction).
    """)
    return


@app.cell
def _(acq, disk_radius, intensities, np, torch):
    from scatterem.reconstruction.ptychography.iterative.regularization.aperture_support import (  # noqa: E501
        embed_centred,
        measured_bbox_support,
        ray_band_mask,
    )

    MODEL_GRID = 288  # far-field grid of the model; the 192 px detector in its centre
    DK = acq["dk_invA"][0]
    DR = acq["scan_step_A"][0]
    ROTATION = acq["rotation_deg"]
    band_half_width = 7.0 * disk_radius / 35.65

    mean_pattern = intensities.mean((0, 1)).double()
    _n = intensities.shape[-1]
    _yy, _xx = np.indices((_n, _n))
    _mp = mean_pattern.cpu().numpy()
    _cy = (_mp * _yy).sum() / _mp.sum()
    _cx = (_mp * _xx).sum() / _mp.sum()
    _r = np.hypot(_yy - _cy, _xx - _cx)
    _inside = _r < 0.9 * disk_radius
    _shadow = _inside & (_mp < 0.5 * np.median(_mp[_inside])) & (_r > 4)
    needle_deg = float(
        np.degrees(np.arctan2((_yy - _cy)[_shadow].mean(), (_xx - _cx)[_shadow].mean()))
    )
    _dev = intensities.device
    needle = ray_band_mask(
        (_n, _n), (_cy, _cx), needle_deg, band_half_width, device=_dev
    )
    measured = intensities.sum((0, 1)) > 0
    detector_mask = measured & ~needle
    chop = ray_band_mask(
        (_n, _n), (_cy, _cx), needle_deg, band_half_width + 2.0, device=_dev
    )
    probe_support = measured_bbox_support(measured, (MODEL_GRID, MODEL_GRID))
    probe_support[embed_centred(chop, (MODEL_GRID, MODEL_GRID))] = 0.0
    print(
        f"needle azimuth {needle_deg:.1f} deg ({int(_shadow.sum())} shadow px), "
        f"band half width {band_half_width:.2f} px, "
        f"{int(detector_mask.sum())} of {_n * _n} detector pixels in the loss"
    )
    del _shadow
    torch.cuda.empty_cache()
    return (
        DK,
        DR,
        MODEL_GRID,
        ROTATION,
        band_half_width,
        chop,
        detector_mask,
        mean_pattern,
        needle_deg,
        probe_support,
    )


@app.cell
def _(math, np, torch):
    import scipy.sparse as sp
    from scipy.ndimage import map_coordinates as _map_coordinates
    from scipy.sparse.linalg import lsqr

    from scatterem.utils.subpixel import (
        make_neighbor_pairs,
        pairwise_relative_shifts,
        synchronize_shifts,
    )

    def bf_shifts(arr, exclude, b=2, thr=0.3, lowpass=5.0):
        """Shift (scan px) of the virtual BF image of each binned BF pixel."""
        ny, nx, m, _ = arr.shape
        n = m // b
        mean = arr.mean((0, 1)).reshape(n, b, n, b).sum((1, 3))
        ex = exclude.reshape(n, b, n, b).any(3).any(1)
        inds = torch.argwhere((mean / mean.max() > thr) & ~ex)
        vbf = torch.empty((len(inds), ny, nx), device=arr.device)
        for i in range(ny):
            binned = arr[i].reshape(nx, n, b, n, b).sum((2, 4))
            vbf[:, i] = binned[:, inds[:, 0], inds[:, 1]].T
        vbf -= vbf.mean((1, 2), keepdim=True)
        g = torch.fft.fft2(vbf, norm="ortho")
        del vbf
        qy = torch.fft.fftfreq(ny, device=arr.device)[:, None]
        qx = torch.fft.fftfreq(nx, device=arr.device)[None, :]
        g *= torch.exp(-2 * (math.pi * lowpass) ** 2 * (qy**2 + qx**2))
        pairs = make_neighbor_pairs(inds.cpu(), 8).to(arr.device)
        d = pairwise_relative_shifts(g, pairs, 10)
        t = synchronize_shifts(len(inds), pairs, d.double(), n_irls=3)
        return inds.cpu().numpy(), (t - t.mean(0)).float().cpu().numpy()

    def integrate_phase(coords, grad, centre, cut_deg, cut_half_width, r_core=2.0):
        """Least-squares phase from its gradient on a cut, simply connected domain."""
        sn, cs = math.sin(math.radians(cut_deg)), math.cos(math.radians(cut_deg))
        dy, dx = coords[:, 0] - centre[0], coords[:, 1] - centre[1]
        in_cut = (np.abs(dy * cs - dx * sn) <= cut_half_width) & (
            dy * sn + dx * cs >= 0
        )
        keep = ~in_cut & (np.hypot(dy, dx) > r_core)
        index = {tuple(c): i for i, c in enumerate(coords.tolist())}
        rows, cols, vals, rhs = [], [], [], []
        for i, c in enumerate(coords.tolist()):
            if not keep[i]:
                continue
            for d in ((1, 0), (0, 1)):
                j = index.get((c[0] + d[0], c[1] + d[1]))
                if j is None or not keep[j]:
                    continue
                e = len(rhs)
                rows += [e, e]
                cols += [j, i]
                vals += [1.0, -1.0]
                rhs.append(0.5 * ((grad[i] + grad[j]) * d).sum())
        n = len(coords)
        a = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), n))
        gauge = sp.csr_matrix(
            (keep / keep.sum(), (np.zeros(n), np.arange(n))), shape=(1, n)
        )
        phi = lsqr(sp.vstack([a, gauge]), np.append(rhs, 0.0), atol=1e-10, btol=1e-10)[
            0
        ]
        return phi, keep

    def phasor_on_grid(coords, phi, keep, b, m):
        """exp(i phi) of the binned pixels, interpolated to the native m x m grid."""
        n = m // b
        z = np.zeros((n, n), np.complex128)
        w = np.zeros((n, n))
        z[coords[keep, 0], coords[keep, 1]] = np.exp(1j * phi[keep])
        w[coords[keep, 0], coords[keep, 1]] = 1.0
        u = (np.arange(m) + 0.5) / b - 0.5
        yy, xx = np.meshgrid(u, u, indexing="ij")
        zi = _map_coordinates(z.real, [yy, xx], order=1) + 1j * _map_coordinates(
            z.imag, [yy, xx], order=1
        )
        wi = _map_coordinates(w, [yy, xx], order=1)
        return np.exp(1j * np.angle(zi)), wi > 0.5

    return bf_shifts, integrate_phase, phasor_on_grid


@app.cell
def _(
    DK,
    DR,
    ROTATION,
    band_half_width,
    bf_shifts,
    chop,
    fit_disk,
    integrate_phase,
    intensities,
    math,
    mean_pattern,
    needle_deg,
    np,
    phasor_on_grid,
    time,
):
    _t0 = time.perf_counter()
    _b, _m = 2, intensities.shape[-1]
    bf_coords, bf_shift_px = bf_shifts(intensities, chop, b=_b)
    _c = 2 * math.pi * DR * DK * _b
    _t = math.radians(ROTATION)
    _A = -_c * np.array([[math.cos(_t), -math.sin(_t)], [math.sin(_t), math.cos(_t)]])
    _mp = mean_pattern.cpu().numpy()
    seed_cy, seed_cx, seed_r = fit_disk(_mp)
    _centre = ((seed_cy - (_b - 1) / 2) / _b, (seed_cx - (_b - 1) / 2) / _b)
    _phi, _keep = integrate_phase(
        bf_coords, bf_shift_px @ _A, _centre, needle_deg, band_half_width / _b + 1
    )
    _z, _ok = phasor_on_grid(bf_coords, _phi, _keep, _b, _m)
    _yy, _xx = np.indices(_mp.shape)
    _disk = (
        (np.hypot(_yy - seed_cy, _xx - seed_cx) < seed_r) & ~chop.cpu().numpy() & _ok
    )
    seed_aperture = (_disk * _z).astype(np.complex64)
    print(
        f"stage 1: {len(bf_coords)} binned BF pixels, rms shift "
        f"{np.sqrt((bf_shift_px**2).sum(1).mean()):.1f} scan steps, "
        f"{time.perf_counter() - _t0:.1f} s"
    )
    return (seed_aperture,)


@app.cell
def _(np):
    from scatterem.diagnostics.metrics import (
        adjacent_coherence,
        mean_oam,
        oam_density_matrix,
        oam_spectrum,
        remove_probe_gauges,
        remove_translation_gauge,
        to_aperture_plane,
    )

    def physical_modes(probe):
        """Orthogonal modes without the free (last) mode, sorted by power."""
        p = np.asarray(probe)
        p = p[:, 0] if p.ndim == 4 else p
        n = p.shape[0] - 1 if p.shape[0] > 1 else 1
        f = p[:n].reshape(n, -1).astype(np.complex128)
        lam, u = np.linalg.eigh(f @ f.conj().T)
        lam, u = lam[::-1], u[:, ::-1]
        return (u.conj().T @ f).reshape(n, *p.shape[-2:]), lam / lam.sum()

    def fit_common_phase(a, w):
        """Fit c1 r^2 + a (x^2 - y^2) + 2 b x y + tilt, common to all modes.

        ``a``: aperture-plane modes; ``w``: mode powers. The fit uses the
        phase differences of neighbouring pixels, weighted by the amplitudes
        and the mode powers, and repeats on the corrected modes. Returns the
        coefficients ``(c1, a, b, tx, ty)`` in rad per px^2 (px) and the
        centred grids ``(x, y)``.
        """
        n = a.shape[-1]
        inten = (np.abs(a) ** 2 * w[:, None, None]).sum(0)
        yy, xx = np.mgrid[:n, :n].astype(float)
        y = yy - (yy * inten).sum() / inten.sum()
        x = xx - (xx * inten).sum() / inten.sum()
        coef = np.zeros(5)
        b = a.copy()
        for _ in range(8):
            rows, rhs, wts = [], [], []
            amp = np.abs(b)
            thr = 0.05 * amp[0].max()
            for axis in (2, 1):  # x (columns), then y (rows)
                hi, lo = [slice(None)] * 3, [slice(None)] * 3
                hi[axis], lo[axis] = slice(1, None), slice(None, -1)
                hi, lo = tuple(hi), tuple(lo)
                g = np.angle(b[hi] * np.conj(b[lo]))
                ww = amp[hi] * amp[lo]
                xm = 0.5 * (x[hi[1:]] + x[lo[1:]])
                ym = 0.5 * (y[hi[1:]] + y[lo[1:]])
                ok = (amp[hi] > thr) & (amp[lo] > thr) & (np.hypot(xm, ym) > 10)
                for m in range(b.shape[0]):
                    k = ok[m]
                    if not k.any():
                        continue
                    X, Y = xm[k], ym[k]
                    one, zero = np.ones(k.sum()), np.zeros(k.sum())
                    if axis == 2:  # d chi / dx = 2 c1 x + 2 a x + 2 b y + tx
                        rows.append(np.c_[2 * X, 2 * X, 2 * Y, one, zero])
                    else:  # d chi / dy = 2 c1 y - 2 a y + 2 b x + ty
                        rows.append(np.c_[2 * Y, -2 * Y, 2 * X, zero, one])
                    rhs.append(g[m][k])
                    wts.append(ww[m][k] * w[m])
            d = np.concatenate(rows)
            sw = np.sqrt(np.concatenate(wts))
            step = np.linalg.lstsq(
                d * sw[:, None], np.concatenate(rhs) * sw, rcond=None
            )[0]
            coef += step
            chi = (
                coef[0] * (x**2 + y**2) + coef[1] * (x**2 - y**2) + 2 * coef[2] * x * y
            )
            b = a * np.exp(-1j * (chi + coef[3] * x + coef[4] * y))
            if np.abs(step[:3]).max() * (n / 4) ** 2 < 0.05:
                break
        return coef, (x, y)

    def probe_readout(probe, astig=True, rho=False, up=4, l_max=120, r_max=74.0):
        """Purity and OAM of a probe stack with the sorter of the paper (Section V E).

        1. Leave out the free (last) mode; take the orthogonal modes.
        2. With ``astig``, fit one defocus, two-fold astigmatism and tilt
           phase common to all modes, and remove the astigmatism only.
        3. Fix the translation gauge (real-space intensity centroid to the
           grid centre).
        4. Zero-pad the real-space modes by ``up``, so the aperture plane is
           sampled ``up`` times finer.
        5. Sort on rings 1 px apart from 4 px to ``r_max`` px of the
           reconstruction grid (74 px: the whole aperture), about the
           intensity centroid, for ``|l| <= l_max``.
        """
        p = np.asarray(remove_probe_gauges(np.asarray(probe), drop_last=True))
        power = (np.abs(p) ** 2).sum((1, 2))
        frac = power / power.sum()
        q, w = physical_modes(probe)
        coef, display0 = None, None
        if astig:
            a = to_aperture_plane(q)
            coef, (x, y) = fit_common_phase(a, w)
            chi = coef[0] * (x**2 + y**2) + coef[3] * x + coef[4] * y
            a = a * np.exp(-1j * (coef[1] * (x**2 - y**2) + 2 * coef[2] * x * y))
            display0 = a[0] * np.exp(-1j * chi)  # mode 0, whole common phase removed
            q = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(a, axes=(-2, -1)), norm="ortho"),
                axes=(-2, -1),
            )
        q = np.asarray(remove_translation_gauge(q))
        n = q.shape[-1]
        pad = (n * up - n) // 2
        ap = to_aperture_plane(np.pad(q, ((0, 0), (pad, pad), (pad, pad))))
        rings = dict(
            l_max=l_max,
            r_min=4.0 * up,
            r_max=r_max * up,
            n_r=int(round(r_max - 4.0)) + 1,
            n_theta=256,
        )
        ls, spectrum = oam_spectrum(ap, **rings)
        out = dict(
            purity=float((frac**2).sum()),
            mode_powers=frac,
            ls=np.asarray(ls),
            spectrum=np.asarray(spectrum),
            mean_oam=float(mean_oam(ls, spectrum)),
            coef=coef,
            display0=display0,
        )
        if rho:
            out["rho"] = np.asarray(oam_density_matrix(ap, **rings)[1])
        return out

    def inner_ring_readout(probe):
        """The earlier inner-ring sorter: no interpolation, rings 4 to 39 px,
        ``|l| <= 30``, both gauges removed."""
        p = np.asarray(remove_probe_gauges(np.asarray(probe), drop_last=True))
        p = p / np.sqrt((np.abs(p) ** 2).sum())
        ls, spectrum = oam_spectrum(
            to_aperture_plane(p), l_max=30, r_min=4.0, r_max=39.0, n_r=36, n_theta=256
        )
        return float(mean_oam(ls, spectrum))

    return adjacent_coherence, probe_readout, inner_ring_readout, to_aperture_plane


@app.cell
def _(mo):
    mo.md(r"""
    The figure shows the mean pattern after the corrections (log scale),
    the detector mask, and the stage-1 aperture: its amplitude is the disk
    without the needle band, its phase is the integrated $\chi$ (wrapped).
    """)
    return


@app.cell
def _(detector_mask, mean_pattern, np, plt, probe_readout, seed_aperture):
    from scatterem.nn.modules.mixed_probe_modes import probe_stack_from_aperture

    seed_readout = probe_readout(
        probe_stack_from_aperture(
            seed_aperture, seed_aperture.shape[0], 1, keep_position=True
        )
    )
    print(f"stage-1 aperture: mean OAM {seed_readout['mean_oam']:+.2f} hbar")
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 4.2))
    _ax[0].imshow(np.log1p(mean_pattern.cpu().numpy()), cmap="magma")
    _ax[0].set_title("mean pattern (log), point-inverted")
    _ax[1].imshow(detector_mask.cpu().numpy(), cmap="gray")
    _ax[1].set_title("detector mask (white: in the loss)")
    _im = _ax[2].imshow(
        np.where(np.abs(seed_aperture) > 0, np.angle(seed_aperture), np.nan),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    _ax[2].set_title(f"stage-1 phase, <l> = {seed_readout['mean_oam']:+.1f}")
    _fig.colorbar(_im, ax=_ax[2], label="phase (rad)")
    for _a in _ax:
        _a.set_xticks([])
        _a.set_yticks([])
    _fig.tight_layout()
    _fig
    return (probe_stack_from_aperture,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Stage 2a: one coherent mode

    `PlanarPtychography.from_scan` builds the model: one probe mode from the
    stage-1 aperture (kept at its position on the data disk), a vacuum
    object, and the scan positions of a raster with a 3° rotation. The
    loss is the normalised amplitude error over the detector mask.
    `PlanarConstraints` applies after each accepted pass: mode
    orthogonalisation, a soft real-space radial support, the aperture-plane
    probe support and the absorption bound of 2.5. The solver is
    `secant_gd` (minibatch gradient descent with one Newton step size per
    parameter group); it refines the object, the probe and the scan
    offsets (0.5 px trust bound per pass). The batch size is 128, as in
    the paper. With a single mode, the solver must correct the phase of
    the probe coherently; a mixed state would put the phase error into
    its weak modes.
    """)
    return


@app.cell
def _(probe_readout, solve_ptychography, time):
    from scatterem.reconstruction.ptychography.iterative.second_order import (
        PassStopCallback,
        PassStopRule,
    )

    def run_solve(model, batches, n_passes, rule, trace, label, t0, every=5):
        """One solve with the paper's settings; keeps the last accepted pass."""
        callback = PassStopCallback(model, rule, return_last=True)

        def on_pass(k, log):
            stop = callback(k, log)
            # Safety stop: 3 undone passes in a row mean that the solve does
            # not descend from this state, so the remaining passes would only
            # cost time.
            undone = [h["reverted"] for h in callback.history[-3:]]
            if len(undone) == 3 and all(undone):
                print(f"{label}: 3 passes in a row undone; stop after pass {k}")
                stop = True
            if k % every == 0 or k == n_passes - 1 or stop:
                r = probe_readout(model.probe.detach().cpu().numpy())
                h = callback.history[-1]
                trace.append(
                    dict(
                        stage=label,
                        k=k,
                        loss=float(h["loss"]),
                        purity=r["purity"],
                        mean_oam=r["mean_oam"],
                        t=time.perf_counter() - t0,
                    )
                )
                print(
                    f"{label} pass {k:3d}: loss {h['loss']:.5f}, purity {r['purity']:.4f}, "
                    f"<l> {r['mean_oam']:+.2f}, {time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
            return stop

        log = solve_ptychography(
            model,
            batches,
            n_iter=n_passes,
            solver="secant_gd",
            optimize_probe=True,
            optimize_positions=True,
            position_step_px=0.5,
            per_group_alpha=True,
            probe_mode_groups=True,
            callback=on_pass,
        )
        callback.restore_last()
        return list(log.losses)

    return PassStopRule, run_solve


@app.cell
def _():
    from scatterem.reconstruction.ptychography.iterative.second_order import (
        PlanarConstraints,
        PlanarPtychography,
        solve_ptychography,
    )
    from scatterem.reconstruction.ptychography.iterative.second_order.planar_constraints import (  # noqa: E501
        radial_support,
    )

    return PlanarConstraints, PlanarPtychography, radial_support, solve_ptychography


@app.cell
def _(
    DK,
    DR,
    MODEL_GRID,
    PassStopRule,
    PlanarConstraints,
    PlanarPtychography,
    ROTATION,
    detector_mask,
    intensities,
    probe_readout,
    probe_stack_from_aperture,
    probe_support,
    run_solve,
    seed_aperture,
    settings,
    time,
    torch,
):
    N_MODES, MODE0_FRAC = 9, 0.96  # 8 physical modes and 1 free mode
    constraints = PlanarConstraints(
        orthogonalize=True,
        radial_support=(1.2, 8.0),
        aperture_support=probe_support,
        absorption_width=2.5,
        absorption_margin=64,
    )
    _seed = probe_stack_from_aperture(
        seed_aperture, intensities.shape[-1], 1, keep_position=True
    )
    torch.manual_seed(0)
    # in_place=True turns the intensities into amplitudes, with no copy.
    model_one = PlanarPtychography.from_scan(
        intensities,
        DR,
        DK,
        _seed,
        rotation_deg=ROTATION,
        model_grid=MODEL_GRID,
        detector_mask=detector_mask,
        constraints=constraints,
        in_place=True,
    )
    trace = []
    t_solve = time.perf_counter()
    _n1 = int(settings["one_mode_passes"])
    if _n1 > 0:
        run_solve(
            model_one,
            model_one.batches(128, 0),
            _n1,
            PassStopRule(min_iters=_n1),
            trace,
            "one mode",
            t_solve,
        )
    one_mode_readout = probe_readout(model_one.probe.detach().cpu().numpy())
    print(f"one-mode probe: mean OAM {one_mode_readout['mean_oam']:+.2f} hbar")
    return MODE0_FRAC, N_MODES, model_one, t_solve, trace


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Stage 2b: nine modes

    `expand_modes` turns the coherent probe into nine orthogonal modes, with
    96 % of the power in mode 0. It rescales the object, because the data
    fix only the product of probe power and object transmission. The solve
    continues at batch 512, as in the paper. The callback keeps the state
    of the last accepted pass. After the solve, the radial support applies
    once more. The last mode is free: it takes what the model cannot
    describe (noise, background), and no metric uses it.
    """)
    return


@app.cell
def _(
    MODE0_FRAC,
    N_MODES,
    PassStopRule,
    model_one,
    radial_support,
    run_solve,
    settings,
    t_solve,
    torch,
    trace,
):
    model, _ratio = model_one.expand_modes(N_MODES, MODE0_FRAC)
    _n2 = int(settings["mode_passes"])
    _rule = PassStopRule(
        probe_tol=0.005, loss_tol=2e-3, min_iters=_n2, passes_after_move=1000
    )
    run_solve(model, model.batches(512, 0), _n2, _rule, trace, "nine modes", t_solve)
    with torch.no_grad():
        model.probe.copy_(radial_support(model.probe.detach(), 1.2, 8.0))
    probe = model.probe.detach().cpu().numpy()
    obj = model.object.detach().cpu().numpy()
    return obj, probe


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Purity and OAM

    The purity uses the powers of the physical modes (the free mode left
    out). The OAM read-out is the sorter of the paper (Section V E):

    1. Take the orthogonal physical modes.
    2. Fit one phase $c_1 |\mathbf{k}|^2 + a(k_x^2 - k_y^2) + 2 b k_x k_y$
       plus a tilt, common to all modes, to the phase differences of
       neighbouring aperture pixels, weighted by the amplitudes. Remove
       the two-fold astigmatism ($a$, $b$) only. The device acts as a weak
       electrostatic lens and adds 0.75–1.25 µm of astigmatism; its
       $\cos 2\phi$ phase couples $l$ to $l \pm 2$. The defocus term is
       fitted but kept, because a radial phase does not change $P_l$. A
       phase common to all modes does not change the mode powers.
    3. Move the real-space intensity centroid of the modes to the grid
       centre (translation gauge).
    4. Zero-pad the real-space modes by a factor of 4, so that the
       aperture plane is sampled 4 times finer.
    5. Sort on rings 1 px apart from 4 px to 74 px (the whole aperture,
       rim at 71.3 px), about the intensity centroid, for
       $|l| \le 120$.

    The cell also prints the value without the astigmatism removal, and the
    value of the earlier inner-ring sorter (no interpolation, rings 4 to
    39 px, $|l| \le 30$), which gives +20.19 / +20.36 ħ on the paper's
    probes.
    On the paper's own probes of this scan, the sorter of steps 1–5 gives
    +20.46 ħ (both ends), and +20.79 / +20.80 ħ without step 2.
    """)
    return


@app.cell
def _(
    DK, PAPER, acq, np, probe, probe_readout, inner_ring_readout, settings, wavelength_A
):
    final = probe_readout(probe, rho=True)
    raw = probe_readout(probe, astig=False, rho=True)
    _scale = np.pi * wavelength_A(acq["energy_eV"]) * DK**2  # rad per (px^2 A)
    _c1, _a, _b = final["coef"][:3]
    print(f"settings: {settings}")
    print(
        f"purity:                          {final['purity']:.4f}   (paper, full scan: "
        f"{PAPER['purity'][0]} end a, {PAPER['purity'][1]} end b)"
    )
    print(
        f"mean OAM, astigmatism removed:   {final['mean_oam']:+.2f} hbar  "
        f"(paper, Fig. 3: +{PAPER['mean_oam'][0]} both ends)"
    )
    print(
        f"mean OAM, astigmatism kept:      {raw['mean_oam']:+.2f} hbar  "
        "(paper probes: +20.79 / +20.80)"
    )
    print(
        f"mean OAM, inner-ring sorter:     {inner_ring_readout(probe):+.2f} hbar  "
        "(paper probes: +20.19 / +20.36)"
    )
    print(
        f"common phase: defocus term {abs(_c1) / _scale / 1e4:.2f} um, "
        f"two-fold astigmatism {np.hypot(_a, _b) / _scale / 1e4:.2f} um"
    )
    print("mode powers:", " ".join(f"{p:.3f}" for p in final["mode_powers"]))
    return final, raw


@app.cell
def _(final, np, obj, plt, trace):
    _fig, _ax = plt.subplots(2, 3, figsize=(14, 8.5))
    _a0 = final["display0"]
    _c = _a0.shape[0] // 2
    _a0 = _a0[_c - 96 : _c + 96, _c - 96 : _c + 96]
    _ax[0, 0].imshow(np.abs(_a0), cmap="gray")
    _ax[0, 0].set_title("mode 0, aperture-plane amplitude")
    _ax[0, 1].imshow(
        np.where(np.abs(_a0) > 0.1 * np.abs(_a0).max(), np.angle(_a0), np.nan),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    _ax[0, 1].set_title("mode 0 phase, common defocus and astigmatism removed")
    _ph = obj.real[64:-64, 64:-64]
    _lo, _hi = np.percentile(_ph, (1, 99))
    _ax[0, 2].imshow(_ph, cmap="gray", vmin=_lo, vmax=_hi)
    _ax[0, 2].set_title("object phase (rad), border cut")
    for _a in _ax[0]:
        _a.set_xticks([])
        _a.set_yticks([])
    _ax[1, 0].bar(
        np.arange(len(final["mode_powers"])), final["mode_powers"], color="#3b6ea5"
    )
    _ax[1, 0].set_xlabel("mode")
    _ax[1, 0].set_ylabel("power fraction")
    _ax[1, 0].set_title(f"purity {final['purity']:.3f}")
    _sel = np.abs(final["ls"] - final["mean_oam"]) <= 40
    _ax[1, 1].bar(final["ls"][_sel], final["spectrum"][_sel], color="#3b6ea5")
    _ax[1, 1].axvline(final["mean_oam"], color="#b2182b", lw=1.5)
    _ax[1, 1].set_xlabel("OAM l (hbar)")
    _ax[1, 1].set_ylabel("P_l")
    _ax[1, 1].set_title(f"OAM spectrum, <l> = {final['mean_oam']:+.2f}")
    for _stage, _col in (("one mode", "#999999"), ("nine modes", "#3b6ea5")):
        _rows = [r for r in trace if r["stage"] == _stage]
        if _rows:
            _ax[1, 2].plot(
                [r["t"] for r in _rows],
                [r["mean_oam"] for r in _rows],
                "o-",
                color=_col,
                label=_stage,
            )
    _ax[1, 2].set_xlabel("time (s)")
    _ax[1, 2].set_ylabel("<l> (hbar)")
    _ax[1, 2].legend(frameon=False)
    _ax[1, 2].set_title("mean OAM during the solve")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. OAM density matrix

    The projection of the modes onto the OAM basis also gives the density
    matrix $\rho_{ll'}$ (`oam_density_matrix`); its diagonal is the
    spectrum $P_l$. The plots show $\rho_{ll'}$ before and after the
    astigmatism removal of section 7: brightness gives $|\rho_{ll'}|$,
    scaled to its maximum, and hue gives its phase. The phase of an
    off-diagonal element also contains a term $(l - l')\theta_0$ from the
    azimuthal reference angle $\theta_0$ of the sorter. Before the removal,
    the astigmatism gives a checkerboard: $|\rho_{l,l\pm2}|$ is as large as
    $|\rho_{l,l\pm1}|$. The cell prints the ratio of the two, near the
    dominant channel. On the paper's probes of this scan, the removal
    lowers it from 0.91 / 1.04 to 0.64 (ends a / b).

    **Do not read the off-diagonal elements as the coherence of the
    source.** `adjacent_coherence` computes the coherence of two adjacent
    OAM channels from $\rho_{ll'}$, but here two properties of the
    delivered aperture couple the channels in every mode, with no relation
    to the source: the needle shadow, a hard azimuthal mask, couples $l$ to
    $l \pm 1$, and the remaining astigmatism couples $l$ to $l \pm 2$. In
    the paper, the needle-shadow mask alone raised the adjacent-channel
    coherence of the probe with the device retracted from 0.36–0.57 to
    0.49–0.76. The paper therefore derives no coherence measure from
    $\rho_{ll'}$ and uses the mode powers (the purity) as the measure of
    mixedness. The cell prints `adjacent_coherence` for completeness only.
    """)
    return


@app.cell
def _(adjacent_coherence, final, np, plt, raw):
    from matplotlib.colors import hsv_to_rgb

    def checkerboard(r):
        """Mean |rho_{l,l+-2}| / mean |rho_{l,l+-1}| within 8 channels of the peak."""
        m = np.abs(r["rho"])
        i0 = int(np.argmax(r["spectrum"]))
        idx = range(max(i0 - 8, 2), min(i0 + 9, len(m) - 2))
        d1 = np.mean([m[i, i + s] for i in idx for s in (-1, 1)])
        d2 = np.mean([m[i, i + s] for i in idx for s in (-2, 2)])
        return d2 / d1

    def rho_image(r, ls, sel):
        rho = r["rho"][np.ix_(sel, sel)]
        v = np.abs(rho) / np.abs(rho).max()
        h = (np.angle(rho) + np.pi) / (2 * np.pi)
        return hsv_to_rgb(np.stack([h, np.ones_like(v), v], -1))

    _ls = final["ls"]
    _l0 = int(round(final["mean_oam"]))
    _sel = np.flatnonzero(np.abs(_ls - _l0) <= 20)
    _ext = (
        _ls[_sel[0]] - 0.5,
        _ls[_sel[-1]] + 0.5,
        _ls[_sel[-1]] + 0.5,
        _ls[_sel[0]] - 0.5,
    )
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for _a, _r, _t in (
        (_ax[0], raw, "astigmatism kept"),
        (_ax[1], final, "astigmatism removed"),
    ):
        _a.imshow(rho_image(_r, _ls, _sel), extent=_ext)
        _a.set_title(f"rho_ll', {_t}; checkerboard {checkerboard(_r):.2f}")
        _a.set_xlabel("l'")
        _a.set_ylabel("l")
    _fig.tight_layout()
    for _name, _r in (("kept", raw), ("removed", final)):
        _g, _i, _j = adjacent_coherence(_r["rho"])
        print(
            f"astigmatism {_name}: |rho_l,l+-2| / |rho_l,l+-1| = {checkerboard(_r):.2f}; "
            f"adjacent_coherence {_g:.2f} (l = {int(_ls[_i])}, {int(_ls[_j])}; not a "
            "coherence measure here)"
        )
    _fig
    return


@app.cell
def _(T_START, final, math, time, torch):
    assert math.isfinite(final["purity"]) and math.isfinite(final["mean_oam"])
    print(
        f"wall time since the GPU check: {(time.perf_counter() - T_START) / 60:.1f} min"
    )
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return


if __name__ == "__main__":
    app.run()
