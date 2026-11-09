"""Tests of the bright-field disk measurement (``measure_pupil``).

Oracles: the geometry of synthetic disks made from the contract formula,
symmetries (transpose, row flip), invariances (offset, scale), the same
pattern without defects, and statistical bounds for counting noise.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from scatterem.data.pupil_gauge import ESTIMATORS, PupilEstimate, measure_pupil


def synth_disk(shape, center, radius, width, edge="tanh", i_in=1.0, i_out=0.0):
    """Disk ``i_out + (i_in - i_out) s(r - radius)`` at the pixel points."""
    y = np.arange(shape[0], dtype=float)[:, None]
    x = np.arange(shape[1], dtype=float)[None, :]
    u = (np.hypot(y - center[0], x - center[1]) - radius) / width
    if edge == "tanh":
        s = 0.5 * (1.0 - np.tanh(u))
    else:
        s = 0.5 * (1.0 - torch.special.erf(torch.from_numpy(u)).numpy())
    return i_out + (i_in - i_out) * s


def notch(pattern, center, angle, width, level):
    """Set a sector of angular width ``width`` [rad] about ``angle`` to ``level``."""
    y, x = np.indices(pattern.shape)
    theta = np.arctan2(y - center[0], x - center[1])
    delta = np.angle(np.exp(1j * (theta - angle)))
    out = pattern.copy()
    out[np.abs(delta) < width / 2] = level
    return out


def _radius_bound(radius):
    return 1e-2 if radius < 20 else 3e-3


def _random_center(rng, shape, radius):
    margin = radius + 8
    return np.array(
        [
            rng.uniform(margin, shape[0] - 1 - margin),
            rng.uniform(margin, shape[1] - 1 - margin),
        ]
    )


def _geometry_cases(seed=7, shape=(128, 128)):
    rng = np.random.default_rng(seed)
    for radius in (8, 14, 20, 0.35 * min(shape)):
        for width in (0.3, 1.0, min(4.0, radius / 5)):
            for edge in ("tanh", "erf"):
                for _ in range(2):
                    yield radius, width, edge, _random_center(rng, shape, radius)


def _errors(result, radius, center):
    return abs(result.rim_radius / radius - 1), float(
        np.hypot(*(result.rim_center - center))
    )


@pytest.mark.parametrize("shape", [(128, 128), (96, 160)])
@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_known_geometry(offset, shape):
    for radius, width, edge, center in _geometry_cases(shape=shape):
        pattern = synth_disk(shape, center, radius, width, edge) + offset
        # A soft edge (width > 1 px) needs the wide band that the docstring
        # names. The default band (2 px for "integral") reads a tanh edge of
        # width 4 px at R = 20 px 0.36 % large, above the 3e-3 bound.
        reach = 10.0 if width > 1.0 else None
        result = measure_pupil(pattern, edge_reach=reach)
        rel, dist = _errors(result, radius, center)
        assert rel <= _radius_bound(radius), (radius, width, edge, rel)
        assert dist <= 0.1, (radius, width, edge, dist)


def test_known_geometry_check_detects_wrong_radius():
    """Control: a disk 3 % larger fails the radius bound."""
    center = np.array([60.4, 66.2])
    result = measure_pupil(synth_disk((128, 128), center, 1.03 * 14, 1.0))
    assert _errors(result, 14, center)[0] > 1e-2


def test_result_form():
    result = measure_pupil(synth_disk((64, 64), (30.2, 31.7), 12, 1.0))
    assert isinstance(result, PupilEstimate)
    radius, center = result
    assert type(radius) is float
    assert isinstance(center, np.ndarray)
    assert center.dtype == np.float64 and center.shape == (2,)
    assert ESTIMATORS == ("integral", "rayfit")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_axis_order_transpose_and_flip(estimator):
    shape, center = (96, 160), np.array([30.3, 101.7])
    pattern = synth_disk(shape, center, 18, 1.0)
    result = measure_pupil(pattern, estimator)
    assert np.hypot(*(result.rim_center - center)) <= 0.1
    assert np.hypot(*(result.rim_center[::-1] - center)) > 0.1  # control
    trans = measure_pupil(np.ascontiguousarray(pattern.T), estimator)
    assert np.allclose(trans.rim_center, result.rim_center[::-1], rtol=0, atol=1e-9)
    assert math.isclose(trans.rim_radius, result.rim_radius, rel_tol=1e-12)
    flip = measure_pupil(pattern[::-1], estimator)
    assert np.allclose(
        flip.rim_center,
        [95 - result.rim_center[0], result.rim_center[1]],
        rtol=0,
        atol=1e-9,
    )


def _add_outliers(rng, pattern, center, radius, where, count):
    out = pattern.copy()
    level = 1e3 * pattern.max()
    placed = 0
    while placed < count:
        y, x = (
            rng.integers(1, pattern.shape[0] - 2),
            rng.integers(1, pattern.shape[1] - 2),
        )
        dist = math.hypot(y - center[0], x - center[1])
        if (where == "inside" and dist > radius - 3) or (
            where == "outside" and dist < radius + 3
        ):
            continue
        out[y, x] = level
        placed += 1
        if placed < count and rng.random() < 0.5:
            dy, dx = (0, 1) if rng.random() < 0.5 else (1, 0)
            out[y + dy, x + dx] = level
            placed += 1
    return out


def test_bright_outlier_pixels():
    rng = np.random.default_rng(21)
    shape = (128, 128)
    moved_centroid = 0.0
    for case in range(12):
        radius = rng.uniform(10, 40)
        width = rng.uniform(0.3, min(4.0, radius / 5))
        center = _random_center(rng, shape, radius)
        clean = synth_disk(shape, center, radius, width, i_in=100.0, i_out=5.0)
        where = ("inside", "outside", "anywhere")[case % 3]
        dirty = _add_outliers(rng, clean, center, radius, where, 8)
        ref = measure_pupil(clean)
        got = measure_pupil(dirty)
        assert abs(got.rim_radius / ref.rim_radius - 1) <= 5e-3
        assert np.hypot(*(got.rim_center - ref.rim_center)) <= 0.05
        y, x = np.indices(shape)
        moved_centroid = max(
            moved_centroid,
            math.hypot(
                (dirty * y).sum() / dirty.sum() - (clean * y).sum() / clean.sum(),
                (dirty * x).sum() / dirty.sum() - (clean * x).sum() / clean.sum(),
            ),
        )
    # control: the outliers are strong enough to move a plain centroid
    assert moved_centroid > 0.05


@pytest.mark.parametrize(
    ("counts", "radius_goal", "center_goal"), [(100, 1.5e-2, 0.15), (10, 6e-2, 0.3)]
)
def test_counting_noise(counts, radius_goal, center_goal):
    """RMS errors of 20 Poisson patterns.

    The design RMS is at most 0.25 of the goal. The mean square of 20 draws
    has a relative standard deviation of about sqrt(2 / 20) = 0.32, so the
    bound (16 times the design mean square) is more than 9 standard
    deviations away. The false-alarm rate is below 1e-6.
    """
    rng = np.random.default_rng(2026)
    rel, dist, rel_wrong = [], [], []
    for _ in range(20):
        center = _random_center(rng, (128, 128), 20)
        pattern = rng.poisson(counts * synth_disk((128, 128), center, 20, 1.0))
        result = measure_pupil(pattern)
        rel.append(result.rim_radius / 20 - 1)
        rel_wrong.append(result.rim_radius / 21 - 1)
        dist.append(np.hypot(*(result.rim_center - center)))
    assert math.sqrt(np.mean(np.square(rel))) <= radius_goal
    assert math.sqrt(np.mean(np.square(dist))) <= center_goal
    if counts == 100:  # control: a reference radius 5 % too large fails
        assert math.sqrt(np.mean(np.square(rel_wrong))) > radius_goal


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_scale_invariance(estimator):
    pattern = synth_disk((100, 110), (47.3, 52.9), 16, 1.5, i_in=3.0, i_out=0.4)
    ref = measure_pupil(pattern, estimator)
    for factor in (1e-3, 7.5, 1e4):
        got = measure_pupil(pattern * factor, estimator)
        assert math.isclose(got.rim_radius, ref.rim_radius, rel_tol=1e-9)
        assert np.allclose(got.rim_center, ref.rim_center, rtol=0, atol=1e-9)


def _noise_cases():
    rng = np.random.default_rng(3)
    return [
        np.zeros((64, 64)),
        np.full((64, 64), 5.0),
        rng.poisson(10.0, (64, 64)),
        100.0 + rng.standard_normal((64, 64)),
        synth_disk((64, 64), (31.5, 31.5), 0.6 * 64, 1.0),
        synth_disk((64, 64), (31.5, 31.5), 14, 1.0, i_in=1.0, i_out=0.8),
    ]


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_refuses_patterns_without_a_clear_disk(index, estimator):
    with pytest.raises(ValueError):
        measure_pupil(_noise_cases()[index], estimator)


def test_refusal_controls():
    """Control: the same settings with a clear disk pass."""
    measure_pupil(synth_disk((64, 64), (31.5, 31.5), 14, 1.0))
    low = _noise_cases()[5]
    result = measure_pupil(low, contrast_floor=1.1)
    assert abs(result.rim_radius / 14 - 1) <= 1e-2


def test_refuses_bad_inputs():
    with pytest.raises(ValueError, match="2-D"):
        measure_pupil(np.ones((3, 32, 32)))
    with pytest.raises(ValueError, match="empty"):
        measure_pupil(np.ones((0, 5)))
    with pytest.raises(TypeError):
        measure_pupil(np.ones((32, 32), complex))
    with pytest.raises(TypeError):
        measure_pupil(np.ones((32, 32), bool))
    with pytest.raises(TypeError):
        measure_pupil([[1.0] * 32] * 32)


def test_radius_cap_setting():
    center = np.array([31.5, 31.5])
    pattern = synth_disk((64, 64), center, 30, 1.0)
    with pytest.raises(ValueError, match="radius_cap"):
        measure_pupil(pattern)
    result = measure_pupil(pattern, radius_cap=1.0)
    rel, dist = _errors(result, 30, center)
    assert rel <= 3e-3 and dist <= 0.1
    with pytest.raises(ValueError, match="radius_cap"):
        measure_pupil(pattern, radius_cap=0.3)
    for bad in (0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="radius_cap"):
            measure_pupil(pattern, radius_cap=bad)
    for name in ("contrast_floor", "edge_reach"):
        with pytest.raises(ValueError, match=name):
            measure_pupil(pattern, **{name: float("inf")})
        with pytest.raises(TypeError, match=name):
            measure_pupil(pattern, **{name: "1"})


@pytest.mark.parametrize("degrees", [30, 60, 90])
def test_rayfit_on_notched_disks(degrees):
    rng = np.random.default_rng(degrees)
    for _ in range(2):
        center = _random_center(rng, (128, 128), 24)
        pattern = synth_disk((128, 128), center, 24, 1.0)
        pattern = notch(
            pattern, center, rng.uniform(-np.pi, np.pi), math.radians(degrees), 0.0
        )
        rel, dist = _errors(measure_pupil(pattern, "rayfit"), 24, center)
        assert rel <= 1e-2 and dist <= 0.2


def test_integral_misses_notched_disk():
    """Control: the default estimator is biased by a 90-degree notch."""
    center = np.array([63.2, 64.7])
    pattern = notch(
        synth_disk((128, 128), center, 24, 1.0), center, 0.3, math.pi / 2, 0.0
    )
    try:
        result = measure_pupil(pattern)
    except ValueError:
        return
    assert np.hypot(*(result.rim_center - center)) > 0.2


@pytest.mark.parametrize("side", range(4))
def test_rayfit_on_disk_near_a_border(side):
    rng = np.random.default_rng(40 + side)
    center = np.array([rng.uniform(40, 88), rng.uniform(40, 88)])
    axis, low = divmod(side, 2)
    center[axis] = 12.3 if low == 0 else 127 - 12.3
    pattern = synth_disk((128, 128), center, 24, 1.0)
    rel, dist = _errors(measure_pupil(pattern, "rayfit"), 24, center)
    assert rel <= 1e-2 and dist <= 0.2


def test_rayfit_on_complete_disks():
    for radius, width, edge, center in _geometry_cases(seed=12):
        pattern = synth_disk((128, 128), center, radius, width, edge) + 0.2
        rel, dist = _errors(measure_pupil(pattern, "rayfit"), radius, center)
        assert rel <= _radius_bound(radius), (radius, width, edge, rel)
        assert dist <= 0.1, (radius, width, edge, dist)


def test_coverage_setting():
    center = np.array([12.4, 70.3])
    pattern = synth_disk((128, 128), center, 24, 1.0)
    measure_pupil(pattern, "rayfit")
    with pytest.raises(ValueError, match="coverage"):
        measure_pupil(pattern, "rayfit", arc_floor=0.8)
    with pytest.raises(ValueError, match="arc_floor"):
        measure_pupil(pattern, arc_floor=0.5)
    for bad in (0, 1.2):
        with pytest.raises(ValueError, match="arc_floor"):
            measure_pupil(pattern, "rayfit", arc_floor=bad)


def test_estimator_choice():
    pattern = synth_disk((128, 128), (60.2, 0.4), 24, 1.0)
    with pytest.raises(ValueError, match="estimator"):
        measure_pupil(pattern, "fast")
    with pytest.raises(TypeError, match="estimator"):
        measure_pupil(pattern, 3)
    half_out = synth_disk((128, 128), (60.2, 2.4), 24, 1.0)
    with pytest.raises(ValueError):
        measure_pupil(half_out, "integral")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_reproducible_and_no_global_random_state(estimator):
    pattern = np.random.default_rng(4).poisson(
        50 * synth_disk((96, 96), (45.3, 49.8), 20, 1.0)
    )
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    first = measure_pupil(pattern, estimator)
    second = measure_pupil(pattern, estimator)
    assert first.rim_radius == second.rim_radius
    assert np.array_equal(first.rim_center, second.rim_center)
    after = np.random.get_state()
    assert all(
        np.array_equal(a, b) for a, b in zip(np_state[1:3], after[1:3], strict=True)
    )
    assert torch.equal(torch_state, torch.get_rng_state())


def _same(a, b):
    return math.isclose(a.rim_radius, b.rim_radius, rel_tol=1e-4) and np.allclose(
        a.rim_center, b.rim_center, rtol=0, atol=1e-3
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_input_dtypes_and_kinds_agree(estimator):
    counts = np.random.default_rng(6).poisson(
        30 * synth_disk((96, 96), (45.3, 49.8), 20, 1.0)
    )
    ref = measure_pupil(counts.astype(np.float64), estimator)
    for variant in (
        counts.astype(np.int32),
        counts.astype(np.float32),
        torch.from_numpy(counts.astype(np.int64)),
    ):
        assert _same(measure_pupil(variant, estimator), ref)
    grad = torch.tensor(counts, dtype=torch.float64, requires_grad=True)
    assert _same(measure_pupil(grad, estimator), ref)
    assert grad.grad is None and torch.equal(
        grad.detach(), torch.from_numpy(counts).double()
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_cuda_agrees_with_cpu(require_cuda, estimator):
    counts = (
        np.random.default_rng(6)
        .poisson(30 * synth_disk((96, 96), (45.3, 49.8), 20, 1.0))
        .astype(np.float32)
    )
    ref = measure_pupil(counts, estimator)
    assert _same(
        measure_pupil(torch.from_numpy(counts).to(require_cuda), estimator), ref
    )


@pytest.mark.slow
@pytest.mark.parametrize(("estimator", "limit"), [("integral", 0.1), ("rayfit", 1.0)])
def test_cost_of_a_512_pattern(estimator, limit):
    pattern = synth_disk((512, 512), (250.3, 260.7), 60, 1.0).astype(np.float32)
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        measure_pupil(pattern, estimator)
        best = min(best, time.perf_counter() - start)
    assert best < limit
