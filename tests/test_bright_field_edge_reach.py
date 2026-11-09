"""The edge band of the dataset disk measurement.

``Dataset4DStem.bright_field_radius_and_center``,
``calibrate_reciprocal_from_bright_field`` and ``measure_pupil`` use a
2-pixel edge band for ``"integral"`` and a 10-pixel band for ``"rayfit"``
by default. On a
disk whose top falls towards the edge (the
Gd2O3 FF-STEM data: from 1.0 to 0.84), a 10-pixel band reads the radius about
4 % small. The planted disk below has a known radius of 18 px and the same
slope.
"""

from __future__ import annotations

import numpy as np
import pytest

from scatterem.data.datasets import Dataset4DStem
from scatterem.data.pupil_gauge import measure_pupil

RADIUS_PX = 18.0
ALPHA = 30e-3


def _sloped_disk(n: int = 96, edge_px: float = 1.0) -> np.ndarray:
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - 47.3, xx - 48.6)
    top = 1.0 - 0.16 * (r / RADIUS_PX) ** 2
    return top * 0.5 * (1.0 - np.tanh((r - RADIUS_PX) / edge_px)) + 0.02


@pytest.fixture
def dataset() -> Dataset4DStem:
    cube = np.broadcast_to(_sloped_disk(), (4, 4, 96, 96)).astype(np.float32)
    return Dataset4DStem.from_array(
        cube.copy(),
        energy=60e3,
        semiconvergence_angle=ALPHA,
        scan_step=0.43,
        normalize=False,
        clip_neg_values=False,
    )


def test_the_dataset_default_band_reads_the_planted_radius(dataset):
    wide = measure_pupil(_sloped_disk(), edge_reach=10.0).rim_radius
    assert wide < 0.97 * RADIUS_PX  # the bias that the 2-pixel band removes
    radius = dataset.bright_field_radius_and_center().rim_radius
    assert radius == pytest.approx(RADIUS_PX, rel=0.015)
    assert radius == pytest.approx(
        measure_pupil(_sloped_disk(), edge_reach=2.0).rim_radius, rel=1e-6
    )


def test_the_calibration_passes_the_band_through(dataset):
    wavelength = dataset.meta.wavelength
    for reach in (2.0, 10.0):
        expected = ALPHA / (measure_pupil(_sloped_disk(), edge_reach=reach)[0])
        dk = dataset.copy().calibrate_reciprocal_from_bright_field(edge_reach=reach)
        assert dk == pytest.approx(expected / wavelength, rel=1e-5)
    assert dataset.calibrate_reciprocal_from_bright_field() == pytest.approx(
        ALPHA / (measure_pupil(_sloped_disk(), edge_reach=2.0)[0]) / wavelength,
        rel=1e-5,
    )


def test_the_scratch_crop_uses_the_callers_band(dataset):
    # The scratch crop of the calibration measures the disk again; it must
    # use the band of the caller, not the default.
    seen = []
    original = Dataset4DStem.bright_field_radius_and_center

    def spy(self, estimator="integral", *, edge_reach=None, arc_floor=None):
        seen.append(edge_reach)
        return original(self, estimator, edge_reach=edge_reach, arc_floor=arc_floor)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Dataset4DStem, "bright_field_radius_and_center", spy)
        dataset.calibrate_reciprocal_from_bright_field(edge_reach=7.0)
    assert seen == [7.0, 7.0]


def test_rayfit_keeps_the_wide_band_on_a_notched_disk():
    # The default band depends on the estimator. With 2 px, rayfit reads a
    # disk with a 90-degree notch about 11 % small.
    yy, xx = np.indices((96, 96))
    center = np.array([47.3, 48.6])
    r = np.hypot(yy - center[0], xx - center[1])
    disk = 0.5 * (1.0 - np.tanh(r - 24.0)) + 0.02
    phi = np.arctan2(yy - center[0], xx - center[1])
    disk[(phi > 0.2) & (phi < 0.2 + np.pi / 2) & (r < 30)] = 0.02
    default = measure_pupil(disk, "rayfit")
    assert default.rim_radius == pytest.approx(24.0, rel=1e-2)
    assert default.rim_radius == measure_pupil(disk, "rayfit", edge_reach=10.0)[0]
    narrow = measure_pupil(disk, "rayfit", edge_reach=2.0).rim_radius
    assert narrow < 0.95 * 24.0  # control: the narrow band misses it
