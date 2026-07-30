"""Fig. 1 -- Gd2O3 nanoparticles (60 kV): gap-free fused full-field reconstruction.

Reproduces Figure 1 of "Gap-free Information Transfer in 4D-STEM via Fusion
of Complementary Scattering Channels" (You, Varnavides, Khavnekar, Palatkin,
Shao, Wu, Stroppa, Chernikova, Zhu, Egoavil, Vespucci, Ye, Schur, Spiecker,
Pelz).

Specimen: Gd2O3 nanoparticles, 60 kV, 30 mrad semiconvergence angle.

What this demonstrates -- the same 4D-STEM dataset is reconstructed through
three complementary channels:
  1. direct (single-side-band, SSB) ptychography -- a phase-contrast image
     built purely from the bright-field disk; recovers spatial frequencies
     gap-free up to ~2x the bright-field aperture (2*alpha/lambda) but no
     further;
  2. tilt-corrected dark field (tcDF) -- recovers complementary
     *higher*-frequency information from dark-field scattering (outside the
     bright-field disk) that direct ptychography structurally cannot reach;
  3. fused full-field (FFF) -- an SSNR-weighted Wiener (Fourier-domain)
     combination of channels (1) and (2), transferring information gap-free
     across the *entire* accessible spatial-frequency range. This is the
     paper's central result.

Maps to: paper Figure 1.

Data: ``scatterem.datasets.You2026Gd2O3`` downloads ``fig1_gd2o3.npy`` from
Zenodo record 18008901 (CC-BY-4.0) into ``FFSTEM_DATA_ROOT`` on first use.
"""

# %%
import os

import matplotlib

if os.environ.get("FFSTEM_HEADLESS") or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # non-interactive: enables headless verification

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import scatterem.vis as vis
from scatterem.datasets import You2026Gd2O3
from scatterem.utils.data import Sampling
from scatterem.vis.visualization import show_2d_array  # noqa: F401 (kept for parity)
from scatterem.vis.visualization_utils import add_scalebar_to_ax  # noqa: F401

# %%
# --- Config ---------------------------------------------------------------
DATA_ROOT = Path(
    os.environ.get(
        "FFSTEM_DATA_ROOT", "/media/philipp/data_2/inr_datasets/fused_data_shengbo"
    )
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGTAG = "fig1_gd2o3"

device = torch.device("cuda")

# The named dataset carries this specimen's acquisition constants (60 kV,
# 30 mrad, 0.43 A scan step) -- including the 84 deg scan rotation that the
# aberration fit below depends on (see the note there) -- and downloads
# itself from Zenodo record 18008901 (CC-BY-4.0).
try:
    dataset = You2026Gd2O3(root=DATA_ROOT, download=True, device=device)
except torch.cuda.OutOfMemoryError:
    # A GPU OOM is a resource failure, not a missing dataset -- it must crash
    # loudly rather than be misreported below as "could not obtain the data".
    raise
except (RuntimeError, OSError) as exc:
    print(
        f"[Fig1] could not obtain the data for You2026Gd2O3 "
        f"(resolved FFSTEM_DATA_ROOT={DATA_ROOT}): {exc}\n"
        f"Place fig1_gd2o3.npy there yourself, or ensure network access to "
        f"Zenodo (record 18008901) for the automatic download. Skipping."
    )
    sys.exit(0)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

# %%
# Aberration determination. IMPORTANT: this dataset needs the correct scan
# rotation (~84 deg, carried by `You2026Gd2O3`). The bright-field-shift parallax
# auto-fit is unreliable on this low-dose 60 kV data and drifts to a spurious
# ~133 deg; leaving rotation at 0 forces the autofocus to absorb the error into
# a huge (~85 A) astigmatism. With rotation fixed at 84 deg, the autofocus
# (sparsity/L4 metric, correct_order=1) recovers the Gd2O3 particle and lattice
# fringes with a physical astigmatism (~20 A) -- matching the published panel.
# (The paper used a total-variation objective; here the monorepo's tv metric
# washes out the low-SNR contrast, so sparsity is used.)
bright_field_mask_threshold = 0.1
correction_method = "autofocus"  # see note above
bin_factors = (1, 1, 1)  # R5: bin=1 -> bin_factors=(1, 1, 1)
verbosity = 1
correct_order = 1  # low order (defocus + 2-fold astigmatism); rotation is fixed above
num_iterations = 50
lr = 1

roi_shape = (450, 450)
upsample = 1.0
dataset.meta.aberrations.array[0] = -50  # initial defocus guess (Angstrom)
dataset.determine_aberrations_(
    bright_field_mask_threshold=bright_field_mask_threshold,
    correction_method=correction_method,
    sharpness_metric="sparsity",
    bin_factors=bin_factors,
    verbosity=verbosity,
    correct_order=correct_order,
    num_iterations=num_iterations,
    lr=lr,
    roi_shape=roi_shape,
    upsample=upsample,
)

# %%
# Direct (single-side-band) ptychography: phase-contrast reconstruction built
# from the bright-field disk alone.
upsample = 2.0

direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample=upsample, n_batches=15, return_snr=True, verbosity=2
)

fig, ax = plt.subplots(figsize=(6, 6))
fig_direct_ptycho, ax_direct_ptycho = vis.show_2d(
    [direct_ptycho_image],
    cbar=True,
    title=["Phase, BF reconstruction"],
    figax=(fig, ax),
    clip_percentile=(2, 98),
)

# %%
# Tilt-corrected dark field: the complementary high-frequency channel,
# recovered from dark-field (outside bright-field disk) scattering.
tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=1,
    bright_field_mask_threshold=bright_field_mask_threshold,
    upsample=upsample,
    return_snr=True,
    snr_blur_sigma=0.0,
)
fig, ax = plt.subplots(figsize=(6, 6))
fig_tcDF, ax_tcDF = vis.show_2d(
    [
        tcDF,
    ],
    cbar=True,
    title=["tcDF"],
    figax=(fig, ax),
)
vis.show_2d(torch.fft.fftshift(ssnr_tcdf), cbar=True, title="ssnr_tcdf")

# %%
# Fused full-field: SSNR-weighted Wiener (Fourier-domain) fusion of the
# direct-ptychography and tcDF channels -- the paper's gap-free result.
#
# IMPORTANT (see plan R6): fused_full_field() internally re-runs its own
# direct ptychography and tcDF (at the library's own default parameters) and
# reassigns dataset.direct_ptychography_phase_image /
# dataset.tilt_corrected_dark_field_image. `direct_ptycho_image` and `tcDF`
# above are the explicit-call LOCALS captured before this call runs -- those
# (not the dataset attributes) are what we save below.
fused, phase_weighted, tcdf_weighted = dataset.fused_full_field(
    verbosity=2, bright_field_mask_threshold=0.3, n_dark_field_segments=32
)


# %%
# --- Output persistence (R6) ------------------------------------------------
def _save(name, tensor):
    arr = tensor.detach().cpu().numpy()
    np.save(OUTPUT_DIR / f"{FIGTAG}_{name}.npy", arr)
    # Physical scale bar in Angstrom. The reconstructions are UPSAMPLED
    # relative to the scan, so one image pixel is the scan step scaled by
    # (scan pixels / image pixels) -- passing dataset.sampling straight through
    # would overstate the bar by the upsample factor.
    ny, nx = arr.shape[-2:]
    sampling = Sampling(
        pixel_size=(
            float(dataset.sampling[0]) * int(dataset.array.shape[0]) / ny,
            float(dataset.sampling[1]) * int(dataset.array.shape[1]) / nx,
        ),
        units=("Å", "Å"),
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    vis.show_2d([tensor], cbar=True, title=[name], figax=(fig, ax), sampling=sampling)
    fig.savefig(OUTPUT_DIR / f"{FIGTAG}_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return arr


# `direct_ptycho_image` and `tcDF` are the LOCALS from the explicit calls
# above, captured before `fused_full_field()`; `fused` is fused's 1st return.
dp_arr = _save("direct_ptychography", direct_ptycho_image)
tc_arr = _save("tcdf", tcDF)
fff_arr = _save("fused_full_field", fused)

# %%
# --- Success line (R7) ------------------------------------------------------
# `finite=True` alone can't tell a properly calibrated run from one that fell
# back to the uncalibrated (1.0, 1.0) dk placeholder (finite is still true).
# Assert the placeholder is gone and print the calibrated dk itself, so the
# line proves calibration succeeded instead of merely that arrays are finite.
dk = float(dataset.sampling[2])
assert dk != 1.0, (
    f"dataset.sampling[2] is still the uncalibrated (1.0, 1.0) placeholder "
    f"(dk={dk!r}) -- calibrate_reciprocal_from_bright_field() must have "
    "fallen back silently; check the warnings printed above."
)
ok = all(np.isfinite(a).all() for a in (dp_arr, tc_arr, fff_arr))
print(
    f"[Fig1] OK: dp{dp_arr.shape} tcdf{tc_arr.shape} fff{fff_arr.shape} "
    f"dk={dk:.5g} finite={ok}"
)
# %%
