"""Fig. 4 (diffraction) -- Au nanoparticle, low dose (200 kV): gap-free FFF reconstruction.

Reproduces the low-dose diffraction panel of Figure 4 of "Gap-free
Information Transfer in 4D-STEM via Fusion of Complementary Scattering
Channels" (You, Varnavides, Khavnekar, Palatkin, Shao, Wu, Stroppa,
Chernikova, Zhu, Egoavil, Vespucci, Ye, Schur, Spiecker, Pelz).

Specimen: Au nanoparticle, 200 kV, 30 mrad semiconvergence angle, acquired at
low electron dose on a Dectris detector -- this figure demonstrates that the
fused full-field reconstruction remains gap-free even under photon/electron
starvation, where naive direct ptychography or dark field alone are noisy.

What this demonstrates -- as in Fig1_Gd2O3.py: direct (SSB) ptychography
recovers gap-free low/mid spatial frequencies from the bright-field disk;
tilt-corrected dark field (tcDF) recovers complementary high frequencies from
dark-field scattering; fused full-field (FFF) is their SSNR-weighted Wiener
fusion. Here the low-dose data requires extra preprocessing (raw Dectris
master-file concatenation, a hot-pixel/bad-patch repair, and a scan-edge
crop) before the standard pipeline runs; the fusion is also compared against
an independent parallax reconstruction and its power spectrum, when
available.

Maps to: paper Figure 4 (Au low-dose diffraction panel).

Data: ``scatterem.datasets.You2026AuLowDose`` downloads the Dectris master
plus its two data blocks from Zenodo record 18008901 (CC-BY-4.0) into
``FFSTEM_DATA_ROOT``; the frame-stream reshape, hot-pixel repair and
scan-edge crop mentioned above happen inside that class, not this script.
Optional comparison reference: ``au30mrad_lowdose_up4.npy`` (parallax
reconstruction, used only for the guarded composite/power-spectrum plots).
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
from scatterem.datasets import You2026AuLowDose
from scatterem.utils.data import Sampling
from scatterem.vis.visualization import show_2d_array
from scatterem.vis.visualization_utils import add_scalebar_to_ax

# %%
# --- Config ---------------------------------------------------------------
DATA_ROOT = Path(
    os.environ.get(
        "FFSTEM_DATA_ROOT", "/media/philipp/data_2/inr_datasets/fused_data_shengbo"
    )
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGTAG = "fig4_diffraction"

device = torch.device("cuda")
DR = 0.727  # real-space scan step (Angstrom)

# The named dataset owns this acquisition: it downloads the Dectris master plus
# its two data blocks from Zenodo record 18008901 (CC-BY-4.0), concatenates the
# frame stream into a square scan grid, repairs the known-bad 2x2 detector patch
# with its 4x4 ring mean, drops the outer 128 unreliable scan rows/columns, and
# applies the 180 deg scan/detector rotation offset of this acquisition.
try:
    dataset = You2026AuLowDose(root=DATA_ROOT, download=True, device=device)
except torch.cuda.OutOfMemoryError:
    # A GPU OOM is a resource failure, not a missing dataset -- it must crash
    # loudly rather than be misreported below as "could not obtain the data".
    raise
except (RuntimeError, OSError) as exc:
    print(
        f"[Fig4-diffraction] could not obtain the data for You2026AuLowDose "
        f"(resolved FFSTEM_DATA_ROOT={DATA_ROOT}): {exc}\n"
        f"Place Au30mrad-lowdose_0002_master.h5 and its two data-block "
        f"siblings there yourself, or ensure network access to Zenodo "
        f"(record 18008901) for the automatic download. Skipping."
    )
    sys.exit(0)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

print(f"fluence = {dataset.fluence} e-/A^2")

# %%
# Aberration autofocus (paper parameters): fit up to 2nd-order aberrations by
# the sharpness-based optimizer. The paper maximized a TOTAL-VARIATION sharpness metric, which in the
# monorepo is correction_method="autofocus" with sharpness_metric="tv".
# The bare "total-variation" string is only a deprecated alias that
# silently uses the DEFAULT sparsity (L4) metric. The optimal metric is
# DATASET-DEPENDENT:
# Au lattice fringes match the published figure best with the TV metric
# (verified locally vs sparsity/bright-field-shifts). A smaller ROI is used here (150x150) because the
# scan-cropped low-dose dataset is smaller than the other figures'.
bright_field_mask_threshold = 0.1
correction_method = "autofocus"  # see note above
n_batches = 25
bin_factors = (1, 1, 1)  # R5: bin=1 -> bin_factors=(1, 1, 1)
verbosity = 1
correct_order = 2
num_iterations = 50
lr = 1

roi_shape = (150, 150)
upsample = 1.0
dataset.meta.aberrations.array[0] = -150.0  # initial defocus guess (Angstrom)
dataset.determine_aberrations_(
    bright_field_mask_threshold=bright_field_mask_threshold,
    correction_method=correction_method,
    sharpness_metric="tv",
    bin_factors=bin_factors,
    verbosity=verbosity,
    correct_order=correct_order,
    num_iterations=num_iterations,
    lr=lr,
    roi_shape=roi_shape,
    upsample=upsample,
)
upsample = "nyquist"

# %%
# Direct (single-side-band) ptychography: phase-contrast reconstruction built
# from the bright-field disk alone, upsampled to the Nyquist limit of the
# calibrated dk.
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample="nyquist", verbosity=1, return_snr=True, n_batches=n_batches
)

direct_ptycho_image3 = direct_ptycho_image.clone()
p02, p98 = torch.quantile(direct_ptycho_image3.cpu(), torch.tensor([0.02, 0.98]))
direct_ptycho_image3 -= p02

fig, ax = plt.subplots(figsize=(6, 6))
fig_direct_ptycho, ax_direct_ptycho = vis.show_2d(
    [direct_ptycho_image3],
    cbar=True,
    title=["Phase, BF reconstruction"],
    figax=(fig, ax),
)

# %%
# Tilt-corrected dark field: the complementary high-frequency channel,
# recovered from dark-field (outside bright-field disk) scattering, also
# upsampled to Nyquist because of the low-dose data's sparser sampling.
tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=0,
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

# %%
# Fused full-field: SSNR-weighted Wiener (Fourier-domain) fusion of the
# direct-ptychography and tcDF channels -- the paper's gap-free result, here
# recovered despite the low electron dose.
#
# IMPORTANT (see plan R6): fused_full_field() internally re-runs its own
# direct ptychography and tcDF (at the library's own default parameters) and
# reassigns dataset.direct_ptychography_phase_image /
# dataset.tilt_corrected_dark_field_image. `direct_ptycho_image` and `tcDF`
# above are the explicit-call LOCALS captured before this call runs -- those
# (not the dataset attributes) are what we save below.
fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(
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
# above, captured before `fused_full_field()`; `fff` is fused's 1st return.
dp_arr = _save("direct_ptychography", direct_ptycho_image)
tc_arr = _save("tcdf", tcDF)
fff_arr = _save("fused_full_field", fff)

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
    f"[Fig4-diffraction] OK: dp{dp_arr.shape} tcdf{tc_arr.shape} "
    f"fff{fff_arr.shape} dk={dk:.5g} finite={ok}"
)

# %%
# --- Comparison plot 1 (guarded, R8): log power spectra vs. parallax -------
REF_FILE = DATA_ROOT / "au30mrad_lowdose_up4.npy"
if REF_FILE.exists():

    def hann2d(ny_, nx_):
        wy = np.hanning(ny_)[:, None]
        wx = np.hanning(nx_)[None, :]
        return wy * wx

    def power_spectrum(img, sampling_A_per_px):
        """Return power spectrum P and the frequency pixel size dk (A^-1/px)."""
        img = np.asarray(img, np.float32)
        ny_, nx_ = img.shape
        win = hann2d(ny_, nx_).astype(np.float32)
        x = (img - np.median(img)) * win
        F = np.fft.fftshift(np.fft.fft2(x))
        P = (np.abs(F) ** 2) / (win**2).sum()
        delta_k = 1.0 / (nx_ * sampling_A_per_px)
        return P, delta_k

    pp_series = np.load(REF_FILE)
    direct_img = direct_ptycho_image3.cpu().numpy()
    tcdf_img = tcDF.cpu().numpy()
    fff_img = fff.cpu().numpy()

    P_dp, dk_dp = power_spectrum(direct_img, DR)
    P_tc, dk_tc = power_spectrum(tcdf_img, DR)
    P_ff, dk_ff = power_spectrum(fff_img, DR)
    P_pp, dk_pp = power_spectrum(pp_series, DR)

    L_dp = np.log1p(P_dp)
    L_tc = np.log1p(P_tc)
    L_ff = np.log1p(P_ff)
    L_pp = np.log1p(P_pp)

    fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=100)
    show_2d_array(L_dp, figax=(fig, ax[0]))
    ax[0].set_title("Direct Ptychography", fontsize=25)
    ax[0].axis("off")

    show_2d_array(L_tc, figax=(fig, ax[1]))
    ax[1].set_title("tcDF", fontsize=25)
    ax[1].axis("off")

    show_2d_array(L_ff, figax=(fig, ax[2]))
    ax[2].set_title("Fused full-field", fontsize=25)
    ax[2].axis("off")

    show_2d_array(L_pp, figax=(fig, ax[3]))
    ax[3].set_title("Parallax", fontsize=25)
    ax[3].axis("off")

    scalebar_length = 0.2
    for i in range(4):
        add_scalebar_to_ax(
            ax=ax[i],
            array_size=20,
            sampling=dk_dp,
            length_units=scalebar_length,
            units="Å",
            width_px=10,
            pad_px=1,
            color="white",
            loc="lower right",
        )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{FIGTAG}_power_spectrum_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # %%
    # --- Comparison plot 2: real-space composite vs. parallax --------------
    def normalize01(a):
        a = np.asarray(a)
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        return (
            np.zeros_like(a, np.float32)
            if amax == amin
            else ((a - amin) / (amax - amin)).astype(np.float32)
        )

    dp_img = direct_ptycho_image3.cpu().numpy()
    tcdf_img = tcDF.cpu().numpy()
    fff_img = fff.cpu().numpy()

    dptop_n = normalize01(dp_img)
    tcdftop_n = normalize01(tcdf_img)
    ffftop_n = normalize01(fff_img)

    pp_series = np.load(REF_FILE)
    pp_series = 2 - pp_series
    pp_series = normalize01(pp_series)

    fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=200)

    show_2d_array(dptop_n, figax=(fig, ax[0]))
    ax[0].set_title("Direct Ptychography", fontsize=15)
    ax[0].axis("off")

    show_2d_array(tcdftop_n, figax=(fig, ax[1]))
    ax[1].set_title("Tilt-Corrected Dark Field", fontsize=15)
    ax[1].axis("off")

    show_2d_array(ffftop_n, figax=(fig, ax[2]))
    ax[2].set_title("Fused Full Field", fontsize=15)
    ax[2].axis("off")

    show_2d_array(pp_series, figax=(fig, ax[3]))
    ax[3].set_title("Parallax Reconstruction", fontsize=15)
    ax[3].axis("off")

    sampling = DR / 4
    scalebar_length = 20  # Å
    width_px = 8
    for i in range(4):
        add_scalebar_to_ax(
            ax=ax[i],
            array_size=20,
            sampling=sampling,
            length_units=scalebar_length,
            units="Å",
            width_px=width_px,
            pad_px=1,
            color="white",
            loc="lower right",
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    fig.savefig(
        OUTPUT_DIR / f"{FIGTAG}_realspace_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
else:
    print(
        f"[Fig4-diffraction] reference {REF_FILE.name} not found; "
        f"skipping comparison plots."
    )
# %%
