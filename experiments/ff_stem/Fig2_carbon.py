"""Fig. 2 (carbon) -- amorphous carbon (300 kV): gap-free fused full-field reconstruction.

Reproduces the amorphous-carbon panel of Figure 2 of "Gap-free Information
Transfer in 4D-STEM via Fusion of Complementary Scattering Channels" (You,
Varnavides, Khavnekar, Palatkin, Shao, Wu, Stroppa, Chernikova, Zhu, Egoavil,
Vespucci, Ye, Schur, Spiecker, Pelz).

Specimen: amorphous carbon, 300 kV, 19.68 mrad semiconvergence angle. This is
a high-resolution, weak-phase-object test case (light, low-Z, disordered
sample) used to validate against a simulated ground-truth potential.

What this demonstrates -- as in Fig1_Gd2O3.py: direct (SSB) ptychography
recovers gap-free low/mid spatial frequencies from the bright-field disk;
tilt-corrected dark field (tcDF) recovers complementary high frequencies from
dark-field scattering; fused full-field (FFF) is their SSNR-weighted Wiener
fusion. Here the fusion result is additionally compared (SSIM, power
spectrum) against a simulated ground-truth potential, when available.

Maps to: paper Figure 2 (carbon panel).

Data: ``scatterem.datasets.You2026Carbon`` downloads ``fig2_carbon.npy`` from
Zenodo record 18008901 (CC-BY-4.0) into ``FFSTEM_DATA_ROOT`` on first use.
Optional comparison reference: ``carbon72_potential.npy`` (ground-truth
potential, used only for the guarded SSIM/power-spectrum comparison plot).
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
from scatterem.datasets import You2026Carbon
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
FIGTAG = "fig2_carbon"

device = torch.device("cuda")
DR = 0.25  # real-space scan step (Angstrom)

# The named dataset carries this specimen's acquisition constants
# (300 kV, 19.68 mrad, 0.25 A scan step) and downloads itself from Zenodo
# (record 18008901, CC-BY-4.0) into DATA_ROOT if not already there. `dk` is
# calibrated from the measured bright-field disk during construction.
try:
    dataset = You2026Carbon(root=DATA_ROOT, download=True, device=device)
except torch.cuda.OutOfMemoryError:
    # A GPU OOM is a resource failure, not a missing dataset -- it must crash
    # loudly rather than be misreported below as "could not obtain the data".
    raise
except (RuntimeError, OSError) as exc:
    print(
        f"[Fig2-carbon] could not obtain the data for You2026Carbon "
        f"(resolved FFSTEM_DATA_ROOT={DATA_ROOT}): {exc}\n"
        f"Place fig2_carbon.npy there yourself, or ensure network access to "
        f"Zenodo (record 18008901) for the automatic download. Skipping."
    )
    sys.exit(0)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

print(f"fluence = {dataset.fluence} e-/A^2")

# %%
# Aberration autofocus (paper parameters): fit up to 2nd-order aberrations by
# the sharpness-based optimizer. The paper maximized a TOTAL-VARIATION sharpness metric, which in the
# monorepo is correction_method="autofocus" with sharpness_metric="tv"
# (set below). The bare "total-variation" string is only a deprecated alias
# for "autofocus" and silently uses the DEFAULT sparsity (L4) metric -- a
# different objective that mis-fits astigmatism on some datasets -- so the
# TV metric is selected explicitly to reproduce the paper.
bright_field_mask_threshold = 0.1
correction_method = "autofocus"  # paper TV objective (see note above)
n_batches = 25
bin_factors = (1, 1, 1)  # R5: bin=1 -> bin_factors=(1, 1, 1)
verbosity = 1
correct_order = 2
num_iterations = 50
lr = 1

roi_shape = (450, 450)
upsample = 1.0
dataset.meta.aberrations.array[0] = -50  # initial defocus guess (Angstrom)
dataset.determine_aberrations_(
    bright_field_mask_threshold=bright_field_mask_threshold,
    correction_method=correction_method,
    sharpness_metric="tv",  # paper used the total-variation sharpness metric
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
# from the bright-field disk alone, upsampled to the Nyquist limit of the
# calibrated dk.
direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample="nyquist", verbosity=1, return_snr=True, n_batches=n_batches
)

direct_ptycho_image2 = dataset.direct_ptychography_phase_image
p02, p98 = torch.quantile(direct_ptycho_image2.cpu(), torch.tensor([0.02, 0.98]))
direct_ptycho_image2 -= p02

fig_bf_analytic2, ax_bf_analytic2 = vis.show_2d(
    [direct_ptycho_image2],
    cbar=True,
    title=["Phase, BF reconstruction"],
)

# %%
# Tilt-corrected dark field: the complementary high-frequency channel,
# recovered from dark-field (outside bright-field disk) scattering. Note
# `upsample` here is still the 1.0 set in the aberration-fit block above
# (direct_ptychography above used its own explicit "nyquist" literal). This
# DP-at-Nyquist / tcDF-at-native-grid split is INTENTIONAL and faithful to
# the paper's original pipeline: direct ptychography is reconstructed at
# Nyquist for maximum resolution, while tcDF is kept on the native sampling
# grid -- not an oversight.
tcDF, ssnr_tcdf = dataset.tilt_corrected_dark_field(
    n_dark_field_segments=16,
    verbosity=0,
    bright_field_mask_threshold=0.85,
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
# direct-ptychography and tcDF channels -- the paper's gap-free result.
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
    f"[Fig2-carbon] OK: dp{dp_arr.shape} tcdf{tc_arr.shape} "
    f"fff{fff_arr.shape} dk={dk:.5g} finite={ok}"
)

# %%
# --- Comparison plot (guarded, R8): SSIM + power spectrum vs ground truth ---
REF_FILE = DATA_ROOT / "carbon72_potential.npy"
if REF_FILE.exists():
    pot = np.load(REF_FILE)
    # %%
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from skimage.metrics import structural_similarity as ssim

    def power_spectrum(img, eps=1e-12):
        F = np.fft.fftshift(np.fft.fft2(img))
        P = np.abs(F) ** 2
        return P

    def normalize01(a):
        a = np.asarray(a)
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        return (
            np.zeros_like(a, np.float32)
            if amax == amin
            else ((a - amin) / (amax - amin)).astype(np.float32)
        )

    pp_np = np.asarray(pot)
    fff_np = fff.detach().cpu().numpy()
    directptycho = direct_ptycho_image2.cpu().numpy()
    tcdf_np = tcDF.cpu().numpy()
    dpw_np = phase_weighted.cpu().numpy()
    tcdfw_np = tcdf_weighted.cpu().numpy()

    pp_n = normalize01(pp_np)
    fff_n = normalize01(fff_np)
    directptycho_n = normalize01(directptycho)
    tcdf_n = normalize01(tcdf_np)
    dpw_n = normalize01(dpw_np)
    tcdfw_n = normalize01(tcdfw_np)

    fdp = directptycho_n.copy()
    ftcdf = tcdf_n.copy()

    offset_down_fff = -30
    band_height_fff = 30
    sigma = 0.1
    scale_factor = 1

    fig, ax = plt.subplots(1, 4, figsize=(16, 8))
    show_2d_array(fdp, figax=(fig, ax[0]))
    img1 = pp_n
    img2 = fdp
    ssim_dp = ssim(img1, img2, data_range=img2.max() - img2.min())
    print(ssim_dp)
    psdp = power_spectrum(fdp)
    axins = inset_axes(
        ax[0], width="30%", height="30%", loc="upper right", borderpad=0.8
    )
    show_2d_array(psdp[10:-10, 10:-10], figax=[fig, axins])
    axins.set_xticks([])
    axins.set_yticks([])
    delta_k = 1.0 / (ssnr_ptycho.shape[0] * DR / upsample)

    scalebar_length = 1

    add_scalebar_to_ax(
        ax=axins,
        array_size=20,
        sampling=delta_k,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    ax[0].set_title(f"Direct Ptychography, SSIM:{ssim_dp:.3f}")
    ax[0].axis("off")

    show_2d_array(ftcdf, figax=(fig, ax[1]))
    img2 = ftcdf
    ssim_tcdf = ssim(img1, img2, data_range=img2.max() - img2.min())
    print(ssim_tcdf)
    ax[1].set_title(f"tcDF, SSIM:{ssim_tcdf:.3f}")

    pstcdf = power_spectrum(ftcdf)
    axins = inset_axes(
        ax[1], width="30%", height="30%", loc="upper right", borderpad=0.8
    )
    show_2d_array(pstcdf[10:-10, 10:-10], figax=[fig, axins])
    axins.set_xticks([])
    axins.set_yticks([])

    add_scalebar_to_ax(
        ax=axins,
        array_size=20,
        sampling=delta_k,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    ax[1].axis("off")

    show_2d_array(pp_n, figax=(fig, ax[3]))
    ax[3].set_title("Ground truth potential")
    pspp = power_spectrum(pp_n)
    axins = inset_axes(
        ax[3], width="30%", height="30%", loc="upper right", borderpad=0.8
    )
    show_2d_array(pspp[10:-10, 10:-10], figax=[fig, axins])
    axins.set_xticks([])
    axins.set_yticks([])

    add_scalebar_to_ax(
        ax=axins,
        array_size=20,
        sampling=delta_k,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )
    ax[3].axis("off")

    show_2d_array(fff_n, figax=(fig, ax[2]))
    img2 = fff_n
    ssim_fff = ssim(img1, img2, data_range=img2.max() - img2.min())
    print(ssim_fff)
    ax[2].set_title(f"Fused Full Field, SSIM:{ssim_fff:.3f}")

    psff = power_spectrum(fff_n)
    axins = inset_axes(
        ax[2], width="30%", height="30%", loc="upper right", borderpad=0.8
    )
    show_2d_array(psff[10:-10, 10:-10], figax=[fig, axins])
    axins.set_xticks([])
    axins.set_yticks([])

    add_scalebar_to_ax(
        ax=axins,
        array_size=20,
        sampling=delta_k,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    ax[2].axis("off")

    sampling = DR
    scalebar_length = 5  # Å

    add_scalebar_to_ax(
        ax=ax[3],
        array_size=20,
        sampling=sampling,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    add_scalebar_to_ax(
        ax=ax[2],
        array_size=20,
        sampling=sampling,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    add_scalebar_to_ax(
        ax=ax[0],
        array_size=20,
        sampling=sampling,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )
    add_scalebar_to_ax(
        ax=ax[1],
        array_size=20,
        sampling=sampling,
        length_units=scalebar_length,
        units="Å",
        width_px=1,
        pad_px=1,
        color="white",
        loc="lower right",
    )

    for a in ax:
        a.set_xlabel("x (pixels)")
        a.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{FIGTAG}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    print(
        f"[Fig2-carbon] reference {REF_FILE.name} not found; "
        f"skipping comparison plot."
    )
# %%
