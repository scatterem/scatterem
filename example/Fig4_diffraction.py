"""
Au low-dose 4D-STEM (HDF5): hot-pixel repair, reconstructions, and parallax comparison.

Reads a **master HDF5** (two internal datasets concatenated along scan), reshapes to
4D, fixes a small bad patch by neighborhood filling, crops the scan, then runs BF
calibration, aberration fit, direct ptychography, tcDF, and fused full field. Later
cells compare **log power spectra** and normalized images to a parallax reconstruction
loaded from ``.npy``.

**Dependencies**: ``h5py``, ``hdf5plugin`` (for compressed HDF5), PyTorch, SciPy stack.

Run cell-by-cell using ``# %%`` markers.

**Before you run**

- **GPU**: CUDA by default.
- **Paths**: Set ``H5_DIR`` / ``H5_FILENAME`` and ``PARALLAX_PATH``. Parallax file is
  loaded twice (power spectrum cell and RGB-style panel); keep paths consistent.
- **HDF5 layout** below matches ``entry/data/data_000001`` and ``data_000002``; change
  keys if your file differs.
"""

import matplotlib

matplotlib.use("inline")  # Notebook-friendly backend

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import torch

from scatterem2.vis.visualization import show_2d_array
from scatterem2.vis.visualization_utils import add_scalebar_to_ax

import scatterem2.vis as vis

from scatterem2.utils.data.aberrations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.utils.stem import energy2wavelength

# %% 1) Load HDF5, repair bad pixels, crop scan to (ny, nx, qy, qx)

device = torch.device("cuda")
E = 200e3  # eV
semiangle_cutoff = 30e-3  # rad
dr = 0.727  # Å

H5_DIR = "/home/shengbo/data/vlp/Au30mrad-lowdose/"
H5_FILENAME = "Au30mrad-lowdose_0002_master.h5"
with h5py.File(H5_DIR + H5_FILENAME, mode="r") as df:
    data1 = df['entry']['data']['data_000001'][:, :, :].astype(np.float16)
    data2 = df['entry']['data']['data_000002'][:, :, :].astype(np.float16)
    data_raw = np.concatenate((data1, data2), axis=0)
ds = int(np.sqrt(data_raw.shape[0]))
data = data_raw.reshape((ds, ds, data_raw.shape[1], data_raw.shape[2]))

qy0, qy1 = 42, 44
qx0, qx1 = 26, 28
win = data[:, :, qy0-1:qy1+1, qx0-1:qx1+1].copy()   # 4x4 around the bad 2x2
win[:, :, 1:-1, 1:-1] = np.nan                      # mask out the inner 2x2
data[:, :, qy0:qy1, qx0:qx1] = np.nanmean(win, axis=(-1, -2), keepdims=True)
ci = 128
data = data[ci:-ci, ci:-ci, :, :]

# %% 2) Metadata + dataset

meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=180,
    defocus_guess=float(0),
    sample_thickness_guess=0,
    vacuum_probe=None,
    sampling=(dr, dr, 1, 1),
    units=["A", "A", "A^-1", "A^-1"],
    shape=np.array(data.shape, dtype=np.int32),
    aberrations=Aberrations(array=torch.zeros((12,), device=device)),
)
ny, nx, M, M = data.shape

dataset = Dataset4dstem.from_array(
    array=data,
    origin=np.array((ny / 2, nx / 2, M / 2, M / 2), dtype=np.float32),
    name="dataset_full",
    signal_units="arb. units",
    meta=meta,
    transform_to_amplitudes=False,
    device=device,
    normalize=True,
)

total_intensity = dataset.array.sum()
print(f"Total intensity: {total_intensity}")
print(dataset)

# %% 3) Bright-field crop

data_bf = dataset.crop_brightfield_(
    thresh_lower=0.1,
    thresh_upper=0.99,
)
vis.show_2d(data_bf.array[:25, :25].sum((0, 1)), cbar=True, title="data")

print(f"fluence = {dataset.fluence} e-/A^2")

# %% 4) k-space calibration

rBF, c = data_bf.bright_field_radius_and_center_(
    thresh_lower=0.1, thresh_upper=0.99, plot_rbf=True, method="area"
)

dalpha = semiangle_cutoff / rBF
dk = dalpha / data_bf.meta.wavelength

data_bf.meta.sampling = (dr, dr, dk, dk)
data_bf.sampling = (dr, dr, dk, dk)
print(f"dk = {dk}")
print(f"dalpha = {dalpha}")
print(f"wavelength = {data_bf.meta.wavelength}")
print(data_bf)
print(data_bf.meta)
dataset.meta = data_bf.meta
dataset.sampling = data_bf.sampling
k_max = rBF * dk
nyquist = 1 / (2 * k_max)
print(f"nyquist = {nyquist}")
print(f"k_max = {k_max}")

# %% 5) Aberration fit; then set upsample to Nyquist for following cells

bright_field_mask_threshold = 0.1
correction_method = "total-variation"  
fit_rotation = False
target_percentage_nonzero_pixels = 0.75
n_batches = 25
registration_upsample_factor = 10
lowpass_fwhm_bright_field = None
bin = 1
arrow_scale = 25e-2
verbosity = 1
correct_order = 2
gradient_mask = torch.ones(12, dtype=torch.bool)
num_iterations = 50
lr = 1

roi_shape = (150, 150)
roi_center = "center"
upsample = 1.0
n_center_indices = 25
dataset.meta.aberrations.array[0] = -150.0
dataset.determine_aberrations_(
    bright_field_mask_threshold=bright_field_mask_threshold,
    correction_method=correction_method,
    fit_rotation=fit_rotation,
    target_percentage_nonzero_pixels=target_percentage_nonzero_pixels,
    n_batches=n_batches,
    registration_upsample_factor=registration_upsample_factor,
    lowpass_fwhm_bright_field=lowpass_fwhm_bright_field,
    bin=bin,
    arrow_scale=arrow_scale,
    verbosity=verbosity,
    correct_order=correct_order,
    gradient_mask=gradient_mask,
    num_iterations=num_iterations,
    lr=lr,
    roi_shape=roi_shape,
    roi_center=roi_center,
    upsample=upsample,
    n_center_indices=n_center_indices,
)
upsample = "nyquist"

# %% 6) Direct ptychography

direct_ptycho_image, ssnr_ptycho = dataset.direct_ptychography(
    upsample="nyquist", verbosity=1, return_snr=True, n_batches=n_batches)


direct_ptycho_image3 = direct_ptycho_image.clone()
p02, p98 = torch.quantile(direct_ptycho_image3.cpu(), torch.tensor([0.02, 0.98]))
direct_ptycho_image3 -= p02  # display stretch

fig, ax = plt.subplots(figsize=(6, 6))
fig_direct_ptycho, ax_direct_ptycho = vis.show_2d(
    [direct_ptycho_image3],
    cbar=True,
    title=["Phase, BF reconstruction"],
    figax=(fig, ax),
)

# %% 7) Tilt-corrected dark field (tcDF)

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

# %% 8) Fused full field

fff, phase_weighted, tcdf_weighted = dataset.fused_full_field(verbosity=2)

# %% 9) Log power spectra vs. parallax (4-panel, reciprocal-space scale bars)

PARALLAX_PATH = "/home/shengbo/data/vlp/au30mrad_lowdose_up4.npy"


def hann2d(ny, nx):
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    return wy * wx

def power_spectrum(img, sampling_A_per_px):
    """Return power spectrum P and the frequency pixel size Δk (Å^-1 / px)."""
    img = np.asarray(img, np.float32)
    ny, nx = img.shape
    win = hann2d(ny, nx).astype(np.float32)
    x = (img - np.median(img)) * win
    F = np.fft.fftshift(np.fft.fft2(x))
    P = (np.abs(F) ** 2) / (win**2).sum()

    # frequency pixel size (Å^-1 per pixel)
    delta_k = 1.0 / (nx * sampling_A_per_px)  # assuming square pixels (nx=ny)
    return P, delta_k


pp_series = np.load(PARALLAX_PATH)
pp_series = pp_series
direct_img = direct_ptycho_image3.cpu().numpy()
tcdf_img = tcDF.cpu().numpy()
fff_img = fff.cpu().numpy()

P_dp,  dk_dp  = power_spectrum(direct_img, dr)
P_tc,  dk_tc  = power_spectrum(tcdf_img, dr)
P_ff,  dk_ff  = power_spectrum(fff_img, dr)
P_pp,  dk_pp  = power_spectrum(pp_series, dr)

# Shared robust display range for the log power spectra
L_dp = np.log1p(P_dp)
L_tc = np.log1p(P_tc)
L_ff = np.log1p(P_ff)
L_pp = np.log1p(P_pp)

fs0 = fff.shape[0]//2
fs1 = fff.shape[1]//2
ps0 = pp_series.shape[0]//2
ps1 = pp_series.shape[1]//2


fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=100)
show_2d_array(L_dp, figax=(fig, ax[0]))
ax[0].set_title("Direct Ptychography", fontsize=25)
ax[0].axis('off')

show_2d_array(L_tc, figax=(fig, ax[1]))
ax[1].set_title("tcDF", fontsize=25)
ax[1].axis('off')

show_2d_array(L_ff, figax=(fig, ax[2]))
ax[2].set_title("Fused full-field", fontsize=25)
ax[2].axis('off')

show_2d_array(L_pp, figax=(fig, ax[3]))
ax[3].set_title("Parallax", fontsize=25)
ax[3].axis('off')

scalebar_length = 0.2
for i in range (4):
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

# %% 10) Normalized real-space panels + parallax (inverted) + scale bars

def normalize01(a):
    a = np.asarray(a)
    amin = np.nanmin(a); amax = np.nanmax(a)
    return np.zeros_like(a, np.float32) if amax == amin else ((a - amin) / (amax - amin)).astype(np.float32)

cx = 100
cy = 10
ci = 128

dp_img = direct_ptycho_image3.cpu().numpy()
tcdf_img = tcDF.cpu().numpy()
fff_img = fff.cpu().numpy()

dptop_n = normalize01(dp_img)
tcdftop_n = normalize01(tcdf_img)
ffftop_n = normalize01(fff_img)

pp_series = np.load(PARALLAX_PATH)
pp_series = 2 - pp_series

pp_series = normalize01(pp_series)
fig, ax = plt.subplots(1, 4, figsize=(16, 8), dpi=200)

show_2d_array(dptop_n, figax=(fig, ax[0]))
ax[0].set_title("Direct Ptychography", fontsize=15)

ax[0].axis('off')
show_2d_array(tcdftop_n, figax=(fig, ax[1]))
ax[1].set_title("Tilt-Corrected Dark Field", fontsize=15)
ax[1].axis('off')
show_2d_array(ffftop_n, figax=(fig, ax[2]))
ax[2].set_title("Fused Full Field", fontsize=15)
ax[2].axis('off')
show_2d_array(pp_series, figax=(fig, ax[3]))
ax[3].set_title("Parallax Reconstruction", fontsize=15)
ax[3].axis('off')

sampling = dr/4
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
plt.show()
