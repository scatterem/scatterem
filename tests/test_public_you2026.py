"""Tests for the Zenodo-backed FF-STEM paper datasets (You et al., Adv. Sci. 2026).

No test here touches the network except the one marked ``slow``: the download
path is exercised by monkeypatching ``download_url`` and by writing tiny
synthetic files into a ``tmp_path`` cache directory.
"""

import json
import os
import stat
import warnings

import numpy as np
import pytest

from scatterem.datasets import You2026Carbon, You2026Co3O4, You2026Gd2O3
from scatterem.datasets.public import base as base_module
from scatterem.datasets.public.base import PublicDataset4dstem, load_npy
from scatterem.datasets.public.you2026 import You2026AuLowDose, prepare_au_lowdose
from scatterem.datasets.utils import calculate_md5, zenodo_file_url
from scatterem.utils.data.datasets import Dataset4dstem

ZENODO_RECORD = "18008901"


def test_zenodo_file_url_builds_content_url():
    assert zenodo_file_url(ZENODO_RECORD, "fig2_carbon.npy") == (
        "https://zenodo.org/api/records/18008901/files/fig2_carbon.npy/content"
    )


def test_zenodo_file_url_accepts_int_record_id():
    assert zenodo_file_url(18008901, "master.h5") == (
        "https://zenodo.org/api/records/18008901/files/master.h5/content"
    )


def _write_npy(path, shape=(4, 4, 8, 8)):
    """Tiny synthetic 4D cube with a bright center disk (so it looks like data)."""
    arr = np.zeros(shape, dtype=np.float32)
    cy, cx = shape[2] // 2, shape[3] // 2
    arr[:, :, cy - 1 : cy + 2, cx - 1 : cx + 2] = 100.0
    np.save(path, arr)
    return arr


def _write_edge_disk_npy(path, shape=(4, 4, 32, 32), center=(3, 16), radius=6.0):
    """A disk placed close enough to the detector edge that the bright-field
    crop box clips to an empty slice (negative slice start wraps in Python
    slicing rather than clamping), reproducing the
    ``calibrate_reciprocal_from_bright_field`` edge-of-detector failure."""
    arr = np.zeros(shape, dtype=np.float32)
    cy, cx = center
    ys, xs = np.meshgrid(np.arange(shape[2]), np.arange(shape[3]), indexing="ij")
    disk = (np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) <= radius).astype(np.float32)
    arr[:, :, :, :] = disk * 100.0
    np.save(path, arr)
    return arr


@pytest.fixture
def synthetic_class(tmp_path):
    """A PublicDataset4dstem subclass pointed at a real tiny file in tmp_path.

    ``resources`` carries the file's REAL md5 so the production
    ``_missing_resources``/``check_integrity`` path runs unmodified.
    """
    fname = "synthetic_cube.npy"
    _write_npy(tmp_path / fname)

    class _Synthetic(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = [(fname, calculate_md5(tmp_path / fname))]
        energy = 60e3
        semiconvergence_angle = 30e-3
        scan_step = 0.43
        rotation = 84.0
        reference = "Synthetic test dataset"

        def _load_array(self):
            return np.load(self.raw_folder / self.resources[0][0])

    return _Synthetic


def test_base_uses_cached_file_without_downloading(
    tmp_path, synthetic_class, monkeypatch
):
    def _boom(*args, **kwargs):
        raise AssertionError("download_url must not be called when the cache is valid")

    monkeypatch.setattr("scatterem.datasets.public.base.download_url", _boom)
    ds = synthetic_class(root=tmp_path, download=True, calibrate=False)
    assert isinstance(ds, Dataset4dstem)


def test_base_builds_metadata_from_class_attributes(tmp_path, synthetic_class):
    ds = synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert ds.meta.energy == 60e3
    assert ds.meta.semiconvergence_angle == 30e-3
    assert ds.meta.rotation == 84.0
    assert tuple(float(s) for s in ds.sampling)[:2] == (0.43, 0.43)
    assert list(ds.units) == ["A", "A", "A^-1", "A^-1"]
    assert ds.name == "_Synthetic"
    assert tuple(ds.array.shape) == (4, 4, 8, 8)


def test_base_raises_when_data_absent_and_download_false(tmp_path, synthetic_class):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="synthetic_cube.npy"):
        synthetic_class(root=empty, download=False)


def test_base_rejects_corrupt_cache(tmp_path, synthetic_class):
    """A present file whose md5 does not match must not satisfy _missing_resources."""
    _write_npy(tmp_path / "synthetic_cube.npy", shape=(4, 4, 8, 8))
    ds_cls = synthetic_class
    ds_cls.resources = [("synthetic_cube.npy", "0" * 32)]  # wrong md5
    with pytest.raises(RuntimeError, match="synthetic_cube.npy"):
        ds_cls(root=tmp_path, download=False)


def test_base_download_fetches_every_declared_resource(
    tmp_path, synthetic_class, monkeypatch
):
    """download() hits every declared resource with its Zenodo content URL.

    Built from an already-valid cached instance (no exception dance): the
    ``_token``-guard construction contract also raises ``RuntimeError``, so a
    test that provokes a *constructor* failure to observe ``download()``'s
    side effects can't tell the two apart. Calling ``ds.download()`` directly
    tests it in isolation.
    """
    ds = synthetic_class(root=tmp_path, download=False, calibrate=False)

    calls = []

    def _fake_download(url, root, filename, md5):
        calls.append((url, str(root), filename, md5))

    monkeypatch.setattr("scatterem.datasets.public.base.download_url", _fake_download)

    ds.download()

    assert len(calls) == 1
    assert calls[0][0] == (
        "https://zenodo.org/api/records/18008901/files/synthetic_cube.npy/content"
    )
    assert calls[0][2] == "synthetic_cube.npy"


def test_base_init_downloads_only_the_missing_subset(tmp_path, monkeypatch):
    """__init__ passes download() a narrowed list so it doesn't re-hash (via
    download_url's own integrity check) files it already found valid."""
    present_name = "present.npy"
    missing_name = "missing.npy"
    _write_npy(tmp_path / present_name)
    present_md5 = calculate_md5(tmp_path / present_name)

    class _Partial(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = [(present_name, present_md5), (missing_name, "0" * 32)]
        energy = 60e3
        semiconvergence_angle = 30e-3
        scan_step = 0.43

        def _load_array(self):
            return np.load(self.raw_folder / present_name)

    calls = []

    def _fake_download(url, root, filename, md5):
        calls.append(filename)
        _write_npy(tmp_path / filename)

    monkeypatch.setattr("scatterem.datasets.public.base.download_url", _fake_download)

    _Partial(root=tmp_path, download=True, calibrate=False)

    assert calls == [missing_name]


def test_base_calibrate_true_moves_dk_off_placeholder(tmp_path, synthetic_class):
    """calibrate=True (the default) measures the fixture's centre hot-spot and
    replaces the (1.0, 1.0) dk placeholder."""
    ds = synthetic_class(root=tmp_path, download=False, calibrate=True)
    dk = float(ds.sampling[2])
    assert dk != pytest.approx(1.0)
    assert dk == pytest.approx(float(ds.meta.sampling[2]))
    assert tuple(float(s) for s in ds.sampling)[2:] == pytest.approx(
        tuple(float(s) for s in ds.meta.sampling)[2:]
    )


def test_base_calibrate_true_warns_and_keeps_placeholder_on_edge_disk(tmp_path):
    """A bright-field disk near the detector edge makes the crop box clip to
    an empty slice inside calibrate_reciprocal_from_bright_field (RuntimeError
    from torch's max() on a zero-element tensor). The dataset must still be
    constructed, with a warning instead of a propagated exception, and dk left
    at its placeholder."""
    fname = "edge_cube.npy"
    _write_edge_disk_npy(tmp_path / fname)

    class _EdgeDisk(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = [(fname, calculate_md5(tmp_path / fname))]
        energy = 60e3
        semiconvergence_angle = 30e-3
        scan_step = 0.43

        def _load_array(self):
            return np.load(self.raw_folder / fname)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _EdgeDisk(root=tmp_path, download=False, calibrate=True)

    assert isinstance(ds, Dataset4dstem)
    assert any("calibration failed" in str(w.message) for w in caught)
    assert tuple(float(s) for s in ds.sampling)[2:] == (1.0, 1.0)


def _write_uniform_npy(path, shape=(4, 4, 32, 32), value=5.0):
    """A uniform non-zero detector: no real bright-field edge anywhere, so the
    area-method threshold sweep treats the whole illuminated square as the
    "disk" -- a physically impossible radius (silently, no crash) that only
    the implausible-radius guard catches."""
    arr = np.full(shape, value, dtype=np.float32)
    np.save(path, arr)
    return arr


def test_base_calibrate_true_warns_and_keeps_placeholder_on_implausible_radius(
    tmp_path,
):
    """A uniform non-zero detector measures a finite, positive, but physically
    impossible bright-field radius (no real disk edge, so the equivalent-area
    radius approaches the size of the whole detector rather than a small
    fraction of it) -- the one silent failure mode in this path, since every
    other guarded cause raises loudly. Must still warn-and-construct rather
    than propagate a plausible-looking but wrong dk."""
    fname = "uniform_cube.npy"
    _write_uniform_npy(tmp_path / fname)

    class _Uniform(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = [(fname, calculate_md5(tmp_path / fname))]
        energy = 60e3
        semiconvergence_angle = 30e-3
        scan_step = 0.43

        def _load_array(self):
            return np.load(self.raw_folder / fname)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _Uniform(root=tmp_path, download=False, calibrate=True)

    assert isinstance(ds, Dataset4dstem)
    assert any("calibration failed" in str(w.message) for w in caught)
    # Which layer refuses it is an implementation detail: the disk fit may reject
    # the measurement itself, or hand it back for this guard to reject. Both
    # refuse to produce a wrong dk, and both must name the radius as the reason.
    assert any(
        "radius" in str(w.message) for w in caught
    ), f"no warning named the radius: {[str(w.message) for w in caught]}"
    assert tuple(float(s) for s in ds.sampling)[2:] == (1.0, 1.0)
    assert tuple(float(s) for s in ds.meta.sampling)[2:] == (1.0, 1.0)


def test_repr_credits_the_data_source(tmp_path, synthetic_class):
    """The CC-BY-4.0 attribution and paper reference must be discoverable from
    repr() -- extra_repr() was dead code (nothing in Dataset4dstem's MRO calls
    it), so this exercises _summary_rows() instead."""
    ds = synthetic_class(root=tmp_path, download=False, calibrate=False)
    text = repr(ds)
    assert "https://zenodo.org/records/18008901" in text
    assert "CC-BY-4.0" in text
    assert "Synthetic test dataset" in text


def test_subclass_missing_required_constant_raises_at_construction(tmp_path):
    """A subclass that forgets e.g. `energy` must fail fast and clearly at
    construction time, not later with a confusing TypeError deep inside
    calibrate_reciprocal_from_bright_field."""
    fname = "synthetic_cube.npy"
    _write_npy(tmp_path / fname)

    class _MissingEnergy(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = [(fname, calculate_md5(tmp_path / fname))]
        # energy left unset (None)
        semiconvergence_angle = 30e-3
        scan_step = 0.43

        def _load_array(self):
            return np.load(self.raw_folder / fname)

    with pytest.raises(TypeError, match="energy"):
        _MissingEnergy(root=tmp_path, download=False, calibrate=False)


def test_subclass_empty_resources_raises_at_construction(tmp_path):
    """An empty `resources` must not silently pass every integrity check
    (``all([]) is True``) -- it must fail fast and name the class."""

    class _NoResources(PublicDataset4dstem):
        zenodo_record_id = ZENODO_RECORD
        resources = []
        energy = 60e3
        semiconvergence_angle = 30e-3
        scan_step = 0.43

        def _load_array(self):
            raise AssertionError("_load_array should not be reached")

    with pytest.raises(TypeError, match="resources"):
        _NoResources(root=tmp_path, download=False, calibrate=False)


# The acquisition constants each class must carry, transcribed from the paper's
# figure scripts. A drift in either direction fails here.
EXPECTED_CONSTANTS = {
    You2026Gd2O3: dict(
        filename="fig1_gd2o3.npy",
        energy=60e3,
        semiconvergence_angle=30e-3,
        scan_step=0.43,
        rotation=84.0,
    ),
    You2026Carbon: dict(
        filename="fig2_carbon.npy",
        energy=300e3,
        semiconvergence_angle=19.68e-3,
        scan_step=0.25,
        rotation=0.0,
    ),
    You2026Co3O4: dict(
        filename="fig2_co3o4.npy",
        energy=200e3,
        semiconvergence_angle=21e-3,
        scan_step=0.20,
        rotation=0.0,
    ),
}


@pytest.mark.parametrize("cls", list(EXPECTED_CONSTANTS))
def test_declared_constants_match_the_paper(cls):
    want = EXPECTED_CONSTANTS[cls]
    assert cls.zenodo_record_id == ZENODO_RECORD
    assert [fn for fn, _ in cls.resources] == [want["filename"]]
    assert all(len(md5) == 32 for _, md5 in cls.resources)
    assert cls.energy == want["energy"]
    assert cls.semiconvergence_angle == want["semiconvergence_angle"]
    assert cls.scan_step == want["scan_step"]
    assert cls.rotation == want["rotation"]


@pytest.mark.parametrize("cls", list(EXPECTED_CONSTANTS))
def test_dense_classes_build_from_cached_file(cls, tmp_path, monkeypatch):
    """Construct each class from a tiny synthetic stand-in for its real file."""
    from scatterem.datasets.utils import calculate_md5

    want = EXPECTED_CONSTANTS[cls]
    _write_npy(tmp_path / want["filename"])
    monkeypatch.setattr(
        cls,
        "resources",
        [(want["filename"], calculate_md5(tmp_path / want["filename"]))],
    )

    ds = cls(root=tmp_path, download=False, calibrate=False)
    assert isinstance(ds, Dataset4dstem)
    assert ds.name == cls.__name__
    assert ds.meta.energy == want["energy"]
    assert ds.meta.semiconvergence_angle == want["semiconvergence_angle"]
    assert ds.meta.rotation == want["rotation"]
    assert tuple(float(s) for s in ds.sampling)[:2] == (
        want["scan_step"],
        want["scan_step"],
    )
    assert tuple(ds.array.shape) == (4, 4, 8, 8)


def test_you2026_repr_credits_the_data_source(tmp_path, monkeypatch):
    """Distinct name from the base-class ``test_repr_credits_the_data_source``
    above -- a module-level name collision would silently shadow that test
    (only the later definition survives in the module namespace)."""
    from scatterem.datasets.utils import calculate_md5

    _write_npy(tmp_path / "fig2_carbon.npy")
    monkeypatch.setattr(
        You2026Carbon,
        "resources",
        [("fig2_carbon.npy", calculate_md5(tmp_path / "fig2_carbon.npy"))],
    )
    text = repr(You2026Carbon(root=tmp_path, download=False, calibrate=False))
    assert "zenodo.org/records/18008901" in text
    assert "CC-BY-4.0" in text


AU_MASTER = "Au30mrad-lowdose_0002_master.h5"


def _write_au_master(path, n_frames=400, m=64):
    """Synthetic stand-in for the Dectris master file.

    The real master links externally to two sibling data files; the same access
    path (``entry/data/data_00000N``) works with plain in-file datasets here.
    """
    import h5py

    rng = np.random.default_rng(0)
    stream = (rng.random((n_frames, m, m), dtype=np.float32) + 1.0).astype(np.float32)
    half = n_frames // 2
    with h5py.File(path, "w") as f:
        g = f.create_group("entry/data")
        g.create_dataset("data_000001", data=stream[:half])
        g.create_dataset("data_000002", data=stream[half:])
    return stream


def test_prepare_au_raises_on_non_square_frame_count(tmp_path):
    """A frame count that isn't a perfect square can't be reshaped into a
    square scan raster without silently dropping frames and misaligning the
    rest -- must raise ValueError naming the actual frame count, not truncate."""
    _write_au_master(tmp_path / AU_MASTER, n_frames=401, m=64)
    with pytest.raises(ValueError, match="401"):
        prepare_au_lowdose(
            tmp_path / AU_MASTER, repair_bad_pixels=False, scan_edge_crop=0
        )


def test_prepare_au_raises_on_scan_edge_crop_too_large(tmp_path):
    """scan_edge_crop >= ds // 2 would yield an empty or reversed array via
    Python's `[c:-c]` slicing with no diagnostic -- must raise ValueError
    naming the actual crop and scan size instead."""
    _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)  # ds = 20
    with pytest.raises(ValueError, match="scan_edge_crop=10"):
        prepare_au_lowdose(
            tmp_path / AU_MASTER, repair_bad_pixels=False, scan_edge_crop=10
        )


def test_prepare_au_reshapes_stream_to_square_scan(tmp_path):
    _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)
    out = prepare_au_lowdose(
        tmp_path / AU_MASTER, repair_bad_pixels=False, scan_edge_crop=0
    )
    assert out.shape == (20, 20, 64, 64)


def test_prepare_au_crops_scan_edges(tmp_path):
    _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)
    out = prepare_au_lowdose(
        tmp_path / AU_MASTER, repair_bad_pixels=False, scan_edge_crop=4
    )
    assert out.shape == (12, 12, 64, 64)


def test_prepare_au_repairs_the_bad_2x2_patch(tmp_path):
    stream = _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)
    dense = stream.astype(np.float16).reshape(20, 20, 64, 64)

    # Independent expectation: nanmean of the 4x4 ring with the inner 2x2 masked.
    win = dense[:, :, 41:45, 25:29].astype(np.float32).copy()
    win[:, :, 1:-1, 1:-1] = np.nan
    expected = np.nanmean(win, axis=(-1, -2))

    out = prepare_au_lowdose(
        tmp_path / AU_MASTER, repair_bad_pixels=True, scan_edge_crop=0
    )
    patch = np.asarray(out[:, :, 42:44, 26:28], dtype=np.float32)
    # every pixel of the 2x2 holds the same ring mean
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                patch[:, :, i, j], expected, rtol=1e-2, atol=1e-2
            )


def test_prepare_au_leaves_patch_alone_when_repair_disabled(tmp_path):
    stream = _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)
    dense = stream.astype(np.float16).reshape(20, 20, 64, 64)
    out = prepare_au_lowdose(
        tmp_path / AU_MASTER, repair_bad_pixels=False, scan_edge_crop=0
    )
    np.testing.assert_array_equal(
        np.asarray(out[:, :, 42:44, 26:28]),
        np.asarray(dense[:, :, 42:44, 26:28]),
    )


def test_au_class_constants_match_the_paper():
    assert You2026AuLowDose.zenodo_record_id == ZENODO_RECORD
    assert [fn for fn, _ in You2026AuLowDose.resources] == [
        "Au30mrad-lowdose_0002_master.h5",
        "Au30mrad-lowdose_0002_data_000001.h5",
        "Au30mrad-lowdose_0002_data_000002.h5",
    ]
    assert You2026AuLowDose.energy == 200e3
    assert You2026AuLowDose.semiconvergence_angle == 30e-3
    assert You2026AuLowDose.scan_step == 0.727
    assert You2026AuLowDose.rotation == 180.0
    assert You2026AuLowDose.scan_edge_crop == 128


def test_au_class_builds_from_cached_files(tmp_path, monkeypatch):
    from scatterem.datasets.utils import calculate_md5

    _write_au_master(tmp_path / AU_MASTER, n_frames=400, m=64)
    monkeypatch.setattr(
        You2026AuLowDose,
        "resources",
        [(AU_MASTER, calculate_md5(tmp_path / AU_MASTER))],
    )
    ds = You2026AuLowDose(
        root=tmp_path, download=False, calibrate=False, scan_edge_crop=4
    )
    assert tuple(ds.array.shape) == (12, 12, 64, 64)
    assert ds.meta.rotation == 180.0
    assert tuple(float(s) for s in ds.sampling)[:2] == (0.727, 0.727)


@pytest.mark.slow
def test_real_zenodo_download_of_smallest_resource(tmp_path):
    """End-to-end: the URL shape and md5 are right for the real record.

    Fetches only the 0.2 MB Au master file. Requires network; deselected by
    default with -m 'not slow'.
    """
    import urllib.error

    from scatterem.datasets.utils import check_integrity, download_url

    filename, md5 = You2026AuLowDose.resources[0]
    try:
        download_url(
            zenodo_file_url(ZENODO_RECORD, filename),
            root=tmp_path,
            filename=filename,
            md5=md5,
        )
    except urllib.error.HTTPError:
        # A real HTTP status error (404/403/410/...) means the record moved,
        # the filename changed, or the URL form we build is wrong -- exactly
        # the breakage this test exists to catch, so it must FAIL, not skip.
        # HTTPError is a subclass of URLError (itself a subclass of
        # OSError), so this except clause MUST come before the broader
        # connectivity one below -- a single `except (URLError, OSError)`
        # would silently swallow a 404 as a "skip". Do not collapse these.
        raise
    except (urllib.error.URLError, OSError) as exc:
        # No network / DNS failure / connection refused / timeout: an
        # offline developer should see "skipped", not "failed".
        pytest.skip(f"test needs network access to Zenodo: {exc}")
    assert check_integrity(tmp_path / filename, md5)


# --- md5 verification stamp -------------------------------------------------
# The stamp lets an already-verified resource skip its md5 pass (9.0 s for the
# 6.1 GiB Fig. 1 cube, 62 % of that dataset's construction). It is only sound if
# it fails CLOSED, so these tests pin the fail-closed behaviour, not the saving.


def _stamp_path(tmp_path, fname="synthetic_cube.npy"):
    from scatterem.datasets.public.base import _md5_stamp_path

    return _md5_stamp_path(tmp_path / fname)


def _count_hashes(monkeypatch):
    """Wrap the md5 helper `base` imported, returning a mutable call counter."""
    from scatterem.datasets.public import base as base_mod

    real = base_mod.check_md5
    calls = []

    def _counting(fpath, md5, **kwargs):
        calls.append(str(fpath))
        return real(fpath, md5, **kwargs)

    monkeypatch.setattr(base_mod, "check_md5", _counting)
    return calls


def test_md5_stamp_lets_a_second_construction_skip_the_hash(
    tmp_path, synthetic_class, monkeypatch
):
    calls = _count_hashes(monkeypatch)

    synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert len(calls) == 1, "the first construction must hash"
    assert _stamp_path(tmp_path).is_file()

    synthetic_class(root=tmp_path, download=False, calibrate=False)
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert len(calls) == 1, "a stamped, untouched file must not be re-hashed"


def test_md5_stamp_records_the_files_real_size_and_mtime(tmp_path, synthetic_class):
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    st = (tmp_path / "synthetic_cube.npy").stat()
    stamp = json.loads(_stamp_path(tmp_path).read_text())
    assert stamp["md5"] == calculate_md5(tmp_path / "synthetic_cube.npy")
    assert stamp["size"] == st.st_size
    assert stamp["mtime_ns"] == st.st_mtime_ns


def test_md5_stamp_is_not_believed_after_the_file_changes(
    tmp_path, synthetic_class, monkeypatch
):
    """The stamp must fail CLOSED: an edited file is re-hashed and rejected."""
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert _stamp_path(tmp_path).is_file()

    calls = _count_hashes(monkeypatch)
    with open(tmp_path / "synthetic_cube.npy", "r+b") as fh:
        fh.seek(200)
        fh.write(b"\xff" * 8)

    with pytest.raises(RuntimeError, match="corrupt"):
        synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert len(calls) == 1, "a changed file must be re-hashed, not trusted"
    assert not _stamp_path(tmp_path).exists(), "a stale stamp must be removed"


def test_md5_stamp_is_not_believed_after_truncation(tmp_path, synthetic_class):
    """Size alone is enough: the mtime is restored so only the size differs."""
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    fpath = tmp_path / "synthetic_cube.npy"
    mtime_ns = fpath.stat().st_mtime_ns
    with open(fpath, "r+b") as fh:
        fh.truncate(fpath.stat().st_size - 64)
    os.utime(fpath, ns=(mtime_ns, mtime_ns))
    assert fpath.stat().st_mtime_ns == mtime_ns

    with pytest.raises(RuntimeError, match="corrupt"):
        synthetic_class(root=tmp_path, download=False, calibrate=False)


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{not json",
        "[1, 2, 3]",
        '"nope"',
        '{"size": 1, "mtime_ns": 2}',
        '{"md5": "00000000000000000000000000000000", "size": 1, "mtime_ns": 2}',
    ],
    ids=["empty", "notjson", "list", "string", "nomd5", "wrongmd5"],
)
def test_degenerate_md5_stamp_falls_back_to_hashing(
    tmp_path, synthetic_class, monkeypatch, payload
):
    """An unusable stamp must cost a hash, never manufacture a pass."""
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    _stamp_path(tmp_path).write_text(payload)

    calls = _count_hashes(monkeypatch)
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert len(calls) == 1


def test_md5_stamp_is_keyed_on_the_declared_md5(tmp_path, synthetic_class):
    """A stamp from a class declaring a different md5 must not be reused."""
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    synthetic_class.resources = [("synthetic_cube.npy", "0" * 32)]
    with pytest.raises(RuntimeError, match="corrupt"):
        synthetic_class(root=tmp_path, download=False, calibrate=False)


def test_md5_stamp_can_be_disabled_by_environment(
    tmp_path, synthetic_class, monkeypatch
):
    """``SCATTEREM_MD5_STAMP=0`` restores the unconditional md5 pass."""
    monkeypatch.setenv("SCATTEREM_MD5_STAMP", "0")
    calls = _count_hashes(monkeypatch)
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    synthetic_class(root=tmp_path, download=False, calibrate=False)
    assert len(calls) == 2
    assert not _stamp_path(tmp_path).exists(), "disabled means no stamp is written"


def test_md5_stamp_write_failure_does_not_break_construction(tmp_path, synthetic_class):
    """A read-only cache directory must not turn a good verification into a
    construction failure -- the stamp write is best effort."""
    mode = tmp_path.stat().st_mode
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        ds = synthetic_class(root=tmp_path, download=False, calibrate=False)
        synthetic_class(root=tmp_path, download=False, calibrate=False)
    finally:
        os.chmod(tmp_path, mode)
    assert tuple(ds.array.shape) == (4, 4, 8, 8)
    assert not _stamp_path(tmp_path).exists()
    assert not list(tmp_path.glob("*.tmp"))


# --- load_npy: the parallel-prefault .npy reader ----------------------------
#
# `np.load` on a multi-GiB cube is bound by the page faults on its own fresh
# destination buffer, not by the disk (6.1 GiB Fig. 1 cube, warm page cache:
# 1.9 GiB/s for np.load vs 4.8 GiB/s for the same sequential read into an
# already-faulted buffer). `load_npy` takes those faults on several threads
# first. The contract is "indistinguishable from np.load(path)", so these are
# equivalence tests plus one that the fast path actually runs.


def _count_prefaults(monkeypatch):
    """Record every ``_prefault`` call, so "the fast path ran" is measured
    rather than assumed -- a gate that silently tests the fallback proves
    nothing."""
    calls = []
    orig = base_module._prefault

    def counting(flat, nthreads):
        calls.append(nthreads)
        return orig(flat, nthreads)

    monkeypatch.setattr(base_module, "_prefault", counting)
    return calls


def _above_gate_uint16(extra=4096):
    """A uint16 array comfortably above ``_NPY_PREFAULT_MIN_BYTES``."""
    return np.arange(base_module._NPY_PREFAULT_MIN_BYTES // 2 + extra, dtype=np.uint16)


def test_load_npy_matches_np_load_above_the_gate(tmp_path, monkeypatch):
    path = tmp_path / "big.npy"
    np.save(path, _above_gate_uint16())
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    ref = np.load(path)
    assert calls, "a file above the gate must take the prefault path"
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert got.tobytes() == ref.tobytes()


def test_load_npy_below_the_gate_delegates_to_np_load(tmp_path, monkeypatch):
    """Below 32 MiB (glibc's ``DEFAULT_MMAP_THRESHOLD_MAX``) the destination
    comes from the recycled heap with already-warm pages, and prefaulting it is
    measured to LOSE -- 0.15x at 4 MiB, 0.65x at 16."""
    path = tmp_path / "small.npy"
    np.save(path, np.arange(1024, dtype=np.uint16))
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    assert calls == [], "a small file must not pay for threads"
    assert np.array_equal(got, np.load(path))


@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "int32", "float32", "float64", "complex64", ">u2"]
)
def test_load_npy_is_byte_exact_across_dtypes(tmp_path, dtype, monkeypatch):
    """Includes a big-endian dtype: the payload is copied raw and the byte order
    travels in the dtype, exactly as ``np.load`` does it."""
    dt = np.dtype(dtype)
    count = base_module._NPY_PREFAULT_MIN_BYTES // dt.itemsize + 17
    path = tmp_path / f"dt_{dt.str[1:]}.npy"
    np.save(path, (np.arange(count) % 251).astype(dt))
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    ref = np.load(path)
    assert calls, "would be vacuous if the fast path were skipped"
    assert got.dtype == ref.dtype and got.dtype.byteorder == ref.dtype.byteorder
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("order", ["C", "F"])
def test_load_npy_preserves_memory_order(tmp_path, order, monkeypatch):
    """A Fortran-ordered ``.npy`` must come back Fortran-ordered with np.load's
    strides -- the reshape mirrors numpy's own in ``read_array``."""
    side = 64
    rows = base_module._NPY_PREFAULT_MIN_BYTES // 2 // side + 32
    arr = np.arange(rows * side, dtype=np.uint16).reshape(rows, side)
    arr = np.asfortranarray(arr) if order == "F" else np.ascontiguousarray(arr)
    path = tmp_path / f"ord_{order}.npy"
    np.save(path, arr)
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    ref = np.load(path)
    assert calls
    assert got.shape == ref.shape and got.strides == ref.strides
    assert got.flags.c_contiguous == ref.flags.c_contiguous
    assert got.flags.f_contiguous == ref.flags.f_contiguous
    assert np.array_equal(got, ref)


def test_load_npy_can_be_disabled_by_environment(tmp_path, monkeypatch):
    """``SCATTEREM_NPY_PREFAULT=0`` restores plain ``np.load`` -- the escape
    hatch, and the control arm the speedup was measured against."""
    path = tmp_path / "env.npy"
    np.save(path, _above_gate_uint16())
    monkeypatch.setenv("SCATTEREM_NPY_PREFAULT", "0")
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    assert calls == []
    assert np.array_equal(got, np.load(path))


def test_load_npy_falls_back_for_a_zip_archive(tmp_path, monkeypatch):
    """An ``.npz`` cannot be memory-mapped, so it must fall through to
    ``np.load`` and still come back as an ``NpzFile`` rather than raising."""
    path = tmp_path / "arch.npz"
    np.savez(path, x=_above_gate_uint16())
    calls = _count_prefaults(monkeypatch)
    got = load_npy(path)
    assert calls == []
    assert "x" in got.files


def test_load_npy_agrees_with_np_load_on_object_arrays(tmp_path, monkeypatch):
    """An object array cannot be memory-mapped, so ``load_npy`` delegates -- and
    ``np.load`` without ``allow_pickle`` raises. Agreeing on the error IS the
    contract, since the contract is "indistinguishable from np.load(path)"."""
    path = tmp_path / "obj.npy"
    np.save(path, np.array([{"a": 1}, [2, 3]], dtype=object), allow_pickle=True)
    calls = _count_prefaults(monkeypatch)
    with pytest.raises(ValueError) as got:
        load_npy(path)
    with pytest.raises(ValueError) as ref:
        np.load(path)
    assert calls == []
    assert str(got.value) == str(ref.value)


def test_load_npy_rejects_a_truncated_file_exactly_as_np_load_does(tmp_path):
    """A truncated file must fail loudly -- silently returning the prefault's
    zeros for the missing tail would be the worst possible outcome.

    numpy's own header/length validation in ``open_memmap`` catches this before
    the fast path is entered, so ``load_npy`` falls back and raises numpy's
    error. Assert the two agree rather than asserting a message of our own,
    which for a file that is short *on disk* is unreachable."""
    path = tmp_path / "full.npy"
    np.save(path, _above_gate_uint16())
    truncated = tmp_path / "trunc.npy"
    truncated.write_bytes(path.read_bytes()[:-4096])
    with pytest.raises(ValueError) as got:
        load_npy(truncated)
    with pytest.raises(ValueError) as ref:
        np.load(truncated)
    assert str(got.value) == str(ref.value)


def test_load_npy_raises_if_the_file_shrinks_mid_read(tmp_path, monkeypatch):
    """The short-read guard. Length is validated by the ``mmap_mode`` probe, so
    the only way to reach a short ``readinto`` is for the file to shrink between
    the probe and the read -- truncate it from inside ``_prefault``, which runs
    in exactly that window. Without the guard this would return an array whose
    tail is the prefault's zeros."""
    path = tmp_path / "shrink.npy"
    np.save(path, _above_gate_uint16())
    full = path.read_bytes()
    orig = base_module._prefault

    def shrink_then_prefault(flat, nthreads):
        path.write_bytes(full[: len(full) // 2])
        return orig(flat, nthreads)

    monkeypatch.setattr(base_module, "_prefault", shrink_then_prefault)
    with pytest.raises(OSError, match="truncated"):
        load_npy(path)


def test_load_npy_raises_file_not_found_like_np_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_npy(tmp_path / "nope.npy")


def test_dense_npy_dataset_reads_through_load_npy(tmp_path, monkeypatch):
    """The wiring: ``_DenseNpyDataset._load_array`` must go through
    ``load_npy``, or every measurement behind it describes an unused
    function."""
    seen = []

    def spy(path):
        seen.append(path)
        return np.load(path)

    monkeypatch.setattr("scatterem.datasets.public.you2026.load_npy", spy)
    cube = tmp_path / "synthetic_cube.npy"
    _write_npy(cube, shape=(4, 4, 8, 8))
    cls = type(
        "SyntheticLoadNpy",
        (You2026Carbon,),
        {"resources": [("synthetic_cube.npy", calculate_md5(cube))]},
    )
    cls(root=tmp_path, download=False, calibrate=False)
    assert len(seen) == 1
