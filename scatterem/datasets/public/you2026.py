"""4D-STEM datasets from the FF-STEM fusion paper.

S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao, M. Wu, D. Stroppa,
D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci, X. Ye, F. K. M. Schur,
E. Spiecker, P. Pelz. "Gap-Free Information Transfer in 4D-STEM via Fusion of
Complementary Scattering Channels." Advanced Science (2026).
https://doi.org/10.1002/advs.76620

Data: Zenodo record 18008901 (DOI 10.5281/zenodo.18008901), CC-BY-4.0.

Each class carries the acquisition constants of one specimen -- beam energy,
probe semiconvergence angle, scan step and scan rotation -- plus any repair of
the raw acquisition. Reconstruction choices (aberration fitting, upsampling,
batching) stay with the caller.

The record also contains ``171_new_alignment.h5``, a sparse ``counts``/
``indices`` 4-defocus focal series of the carbon-nanotube specimen. It is not
wrapped here: it is not the dense cube the nanotube figure script consumes, and
densifying it requires choosing a defocus member and resolving a scan-sampling
discrepancy (file ``dxy`` = 0.334 A vs the script's 0.316 A).
"""

import numpy as np

from scatterem.datasets.public.base import PublicDataset4dstem

_REFERENCE = (
    "You et al., 'Gap-Free Information Transfer in 4D-STEM via Fusion of "
    "Complementary Scattering Channels', Adv. Sci. (2026), "
    "doi:10.1002/advs.76620"
)
_RECORD = "18008901"


class _DenseNpyDataset(PublicDataset4dstem):
    """Shared loader for the specimens published as a dense ``(ny, nx, M, M)`` .npy."""

    zenodo_record_id = _RECORD
    reference = _REFERENCE

    def _load_array(self) -> np.ndarray:
        return np.load(self.raw_folder / self.resources[0][0])


class You2026Gd2O3(_DenseNpyDataset):
    """Gd2O3 nanoparticles, 60 kV (paper Figure 1).

    ``(512, 512, 112, 112)`` uint16, ~6.5 GB. The 84 deg scan rotation is
    load-bearing: the bright-field-shift parallax auto-fit is unreliable on this
    low-dose data, and without the correct rotation the aberration autofocus
    absorbs the error into a spurious ~85 A astigmatism.
    """

    resources = [("fig1_gd2o3.npy", "a2c80d295889e55f15596e341ee84870")]
    energy = 60e3
    semiconvergence_angle = 30e-3
    scan_step = 0.43
    rotation = 84.0


class You2026Carbon(_DenseNpyDataset):
    """Amorphous carbon, 300 kV (paper Figure 2)."""

    resources = [("fig2_carbon.npy", "f12852e01036aa4778589e812f6e985e")]
    energy = 300e3
    semiconvergence_angle = 19.68e-3
    scan_step = 0.25
    host_dtype = np.float32


class You2026Co3O4(_DenseNpyDataset):
    """Co3O4, 200 kV (paper Figure 2)."""

    resources = [("fig2_co3o4.npy", "33853d9a4dc33d39735fbbe29a387ef0")]
    energy = 200e3
    semiconvergence_angle = 21e-3
    scan_step = 0.20
    host_dtype = np.float32


def prepare_au_lowdose(
    master_path,
    repair_bad_pixels: bool = True,
    scan_edge_crop: int = 128,
) -> np.ndarray:
    """Read the Au low-dose Dectris master file into a dense scan grid.

    The raw acquisition is a flat stream of diffraction patterns split across
    two data blocks, with a known-bad 2x2 detector patch and unreliable outer
    scan rows/columns. Repairing those is a property of the acquisition, not a
    reconstruction choice, so it happens here.

    Args:
        master_path: Path to ``..._master.h5``. Its ``entry/data`` group links
            to the two sibling ``_data_00000N.h5`` files, which must sit
            alongside it.
        repair_bad_pixels: Replace the bad 2x2 patch at ``[42:44, 26:28]`` with
            the nanmean of the surrounding 4x4 ring.
        scan_edge_crop: Number of scan rows/columns to drop from each edge.

    Returns:
        ``(ny, nx, M, M)`` float16 array.
    """
    # hdf5plugin registers the HDF5 compression filter(s) the real master
    # file's data blocks are compressed with; without this import h5py raises
    # OSError reading the actual Zenodo files (confirmed locally).
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(master_path, mode="r") as df:
        data1 = df["entry"]["data"]["data_000001"][:, :, :].astype(np.float16)
        data2 = df["entry"]["data"]["data_000002"][:, :, :].astype(np.float16)
        data_raw = np.concatenate((data1, data2), axis=0)

    # The stream is a square raster, scan-ordered.
    n_frames = data_raw.shape[0]
    ds = int(np.sqrt(n_frames))
    if ds * ds != n_frames:
        raise ValueError(
            f"prepare_au_lowdose: {n_frames} frames is not a perfect square "
            f"(nearest square {ds}**2 = {ds * ds}); a non-square frame count "
            "can't be reshaped into a square scan raster without silently "
            f"dropping {n_frames - ds * ds} frame(s) and misaligning the rest."
        )
    data = data_raw.reshape((ds, ds, data_raw.shape[1], data_raw.shape[2]))

    if repair_bad_pixels:
        qy0, qy1 = 42, 44
        qx0, qx1 = 26, 28
        win = data[:, :, qy0 - 1 : qy1 + 1, qx0 - 1 : qx1 + 1].copy()
        win[:, :, 1:-1, 1:-1] = np.nan  # mask the inner 2x2
        data[:, :, qy0:qy1, qx0:qx1] = np.nanmean(win, axis=(-1, -2), keepdims=True)

    if scan_edge_crop:
        c = scan_edge_crop
        if c >= ds // 2:
            raise ValueError(
                f"prepare_au_lowdose: scan_edge_crop={c} is too large for a "
                f"{ds}x{ds} scan (half-extent {ds // 2}); cropping this much "
                "from each edge would yield an empty or reversed array."
            )
        data = data[c:-c, c:-c, :, :]

    return data


class You2026AuLowDose(PublicDataset4dstem):
    """Au nanoparticle, 200 kV, low dose (paper Figure 4).

    Three files: the Dectris master plus its two external data blocks. The
    180 deg rotation is this acquisition's known scan/detector offset.

    Args:
        repair_bad_pixels: Replace the known-bad 2x2 detector patch with the
            nanmean of its surrounding ring. Default ``True``.
        scan_edge_crop: Number of scan rows/columns to drop from each edge.
            Default ``128``.
    """

    zenodo_record_id = _RECORD
    reference = _REFERENCE
    resources = [
        ("Au30mrad-lowdose_0002_master.h5", "14b112d2315362885fe037d8c53befb2"),
        (
            "Au30mrad-lowdose_0002_data_000001.h5",
            "3ce42ab9a1438d60f6e001188535ef36",
        ),
        (
            "Au30mrad-lowdose_0002_data_000002.h5",
            "15e83c44260a2910d5c10af6989f912b",
        ),
    ]
    energy = 200e3
    semiconvergence_angle = 30e-3
    scan_step = 0.727
    rotation = 180.0
    #: Scan rows/columns dropped from each edge (beam settling / scan artifacts).
    scan_edge_crop = 128
    #: Repair the known-bad 2x2 detector patch.
    repair_bad_pixels = True

    def __init__(self, *args, repair_bad_pixels=None, scan_edge_crop=None, **kwargs):
        if repair_bad_pixels is not None:
            self.repair_bad_pixels = repair_bad_pixels
        if scan_edge_crop is not None:
            self.scan_edge_crop = scan_edge_crop
        super().__init__(*args, **kwargs)

    def _load_array(self) -> np.ndarray:
        return prepare_au_lowdose(
            self.raw_folder / self.resources[0][0],
            repair_bad_pixels=self.repair_bad_pixels,
            scan_edge_crop=self.scan_edge_crop,
        )
