"""Clean-room bright-field disk fitting.

Every assertion below is stated against a synthetic pattern whose disk geometry
is known by construction, not against a previous implementation's output. That
matters because the implementation being replaced
(``utils/data/bright_field.py::_area_method``) is a port of a GPL-3.0 upstream
and cannot ship in an Apache-2.0 release.

The hot-pixel test encodes a failure this lab has actually been bitten by: a
single stuck pixel dominates the max used to normalise the threshold sweep, so
every threshold selects ~1 pixel, the radius curve is perfectly flat, and a
plateau-detecting fit reports that flatness as a confident sub-pixel radius.
"""

import math

import numpy as np
import pytest
import torch

from scatterem.utils.data.disk_fit import fit_bright_field_disk


def _disk(shape=(128, 128), radius=20.0, center=None, amplitude=100.0, blur=1.0):
    """A soft-edged disk, the shape a real bright-field probe actually takes."""
    ny, nx = shape
    cy, cx = (ny / 2 - 0.5, nx / 2 - 0.5) if center is None else center
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    # smooth roll-off over ~`blur` pixels: a real disk has no step edge
    profile = 0.5 * (1.0 - np.tanh((r - radius) / max(blur, 1e-6)))
    return torch.from_numpy((amplitude * profile).astype(np.float32))


@pytest.mark.parametrize("radius", [8.0, 15.0, 20.0, 31.5])
def test_recovers_a_known_radius(radius):
    r, _ = fit_bright_field_disk(_disk(radius=radius))
    assert r == pytest.approx(radius, rel=0.02)


@pytest.mark.parametrize("center", [(64.0, 64.0), (50.0, 77.0), (70.5, 60.25)])
def test_recovers_a_known_center(center):
    _, found = fit_bright_field_disk(_disk(radius=18.0, center=center))
    assert found == pytest.approx(np.array(center), abs=0.5)


def test_center_is_returned_as_y_x():
    """Guards an axis swap, which is silent on a centred disk."""
    _, found = fit_bright_field_disk(_disk(radius=12.0, center=(40.0, 90.0)))
    assert found[0] == pytest.approx(40.0, abs=0.5)
    assert found[1] == pytest.approx(90.0, abs=0.5)


def test_survives_a_single_hot_pixel():
    """The regression this module exists to prevent.

    One stuck pixel at 1000x the disk intensity must not capture the fit. The
    replaced implementation normalises by the raw max and reports r ~= 0.7 px
    here with no error.
    """
    dp = _disk(radius=20.0)
    dp[5, 110] = float(dp.max()) * 1000.0
    r, center = fit_bright_field_disk(dp)
    assert r == pytest.approx(20.0, rel=0.05)
    assert center == pytest.approx(np.array([63.5, 63.5]), abs=1.0)


def test_survives_several_hot_pixels():
    dp = _disk(radius=16.0)
    for yy, xx in ((3, 3), (120, 7), (64, 127), (100, 100)):
        dp[yy, xx] = float(dp.max()) * 500.0
    r, _ = fit_bright_field_disk(dp)
    assert r == pytest.approx(16.0, rel=0.05)


def test_raises_when_no_disk_is_present():
    """Pure noise has no plateau; returning a confident radius would be a lie."""
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="bright-field disk"):
        fit_bright_field_disk(torch.rand(128, 128) * 0.01)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        fit_bright_field_disk(torch.zeros(4, 128, 128))


def test_accepts_numpy_as_well_as_torch():
    a = fit_bright_field_disk(_disk(radius=14.0))
    b = fit_bright_field_disk(_disk(radius=14.0).numpy())
    assert a[0] == pytest.approx(b[0], rel=1e-9)
    assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_is_device_and_dtype_stable():
    dp = _disk(radius=17.0)
    r64, c64 = fit_bright_field_disk(dp.double())
    r32, c32 = fit_bright_field_disk(dp.float())
    assert r64 == pytest.approx(r32, rel=1e-4)
    assert c64 == pytest.approx(c32, abs=1e-3)


def test_returns_plain_python_and_numpy_types():
    """Callers put these straight into dk arithmetic; a 0-d tensor leaks device."""
    r, center = fit_bright_field_disk(_disk(radius=10.0))
    assert isinstance(r, float)
    assert isinstance(center, np.ndarray) and center.shape == (2,)


def test_agrees_with_the_previous_implementation_on_a_clean_disk():
    """No calibration shift on well-behaved data, which is the common case.

    Only asserted for a clean pattern: on a hot-pixel pattern the two are
    *supposed* to disagree, and the previous one is the wrong answer.
    """
    legacy = pytest.importorskip("scatterem.utils.data.bright_field")
    dp = _disk(radius=19.0)
    r_new, c_new = fit_bright_field_disk(dp)
    r_old, c_old = legacy.radius_and_center(dp, method="area")
    assert r_new == pytest.approx(float(r_old), rel=0.03)
    assert c_new == pytest.approx(np.asarray(c_old, dtype=float), abs=0.5)


@pytest.mark.parametrize("blur", [0.5, 1.0, 1.75, 2.5])
def test_radius_is_independent_of_edge_softness(blur):
    """The property that makes this fit trustworthy for dk calibration.

    A probe's edge softness varies with defocus and aberrations but its radius
    does not, so the estimate must not move with `blur`. This is also the
    regression guard for the bug this module was written around: selecting the
    "flattest window" of the radius curve ties on pixel-count quantisation, and
    any fixed tie-break drifts the threshold off the half-maximum. The drift
    grows with edge softness, which is exactly what this parametrisation sees.
    """
    r, _ = fit_bright_field_disk(_disk(radius=20.0, blur=blur))
    assert r == pytest.approx(20.0, rel=0.01)


def test_edge_softness_bias_is_better_than_the_implementation_it_replaces():
    """Documents *why* the replacement is not merely a licence workaround.

    Not a competitive benchmark for its own sake: the replaced code is what the
    published figures were calibrated with, so the size and direction of the
    change is a result the maintainer needs recorded.
    """
    legacy = pytest.importorskip("scatterem.utils.data.bright_field")
    dp = _disk(radius=20.0, blur=2.5)
    r_new, _ = fit_bright_field_disk(dp)
    r_old, _ = legacy.radius_and_center(dp, method="area")
    assert abs(r_new - 20.0) < abs(float(r_old) - 20.0)
    assert r_new == pytest.approx(20.0, rel=0.01)


def test_area_scales_as_radius_squared():
    """A physical invariant: doubling the radius must quadruple the disk area."""
    r_small, _ = fit_bright_field_disk(_disk(radius=10.0, shape=(256, 256)))
    r_big, _ = fit_bright_field_disk(_disk(radius=20.0, shape=(256, 256)))
    assert (math.pi * r_big**2) / (math.pi * r_small**2) == pytest.approx(4.0, rel=0.06)
