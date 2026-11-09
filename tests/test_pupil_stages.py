"""Tests of the internal stages of the disk measurement.

Oracles: ``numpy.median`` of shifted copies, a two-level image with known
levels, and points on a known circle.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from scatterem.data._pupil_stages import intermeans, median3
from scatterem.data._ring_regression import robust_circle


def test_median_filter_matches_numpy():
    image = np.random.default_rng(2).random((17, 23))
    padded = np.pad(image, 1, mode="edge")
    copies = [padded[i : i + 17, j : j + 23] for i in range(3) for j in range(3)]
    ref = np.median(np.stack(copies), axis=0)
    got = median3(torch.from_numpy(image)).numpy()
    assert np.array_equal(got, ref)
    assert not np.array_equal(image, ref)  # control: the filter changes data


def test_intermeans_on_two_levels():
    image = np.full((20, 30), 2.0)
    image[5:12, 4:20] = 9.0
    thr, low, high = intermeans(torch.from_numpy(image))
    assert (low, high) == (2.0, 9.0)
    assert thr == 5.5


def _circle_points(rng, n=120, outliers=0.3):
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = np.stack((10.3 + 25.0 * np.sin(theta), -4.2 + 25.0 * np.cos(theta)), 1)
    bad = rng.random(n) < outliers
    pts[bad] *= rng.uniform(0.3, 0.7, (int(bad.sum()), 1))
    return pts


def test_robust_circle_rejects_gross_outliers():
    rng = np.random.default_rng(13)
    pts = _circle_points(rng)
    fit = robust_circle(
        torch.from_numpy(pts),
        torch.ones(len(pts), dtype=torch.bool),
        (9.0, -3.0, 22.0),
    )
    assert abs(fit.radius - 25.0) < 1e-8
    assert np.allclose(fit.center, (10.3, -4.2), atol=1e-8)
    # control: an algebraic least-squares (Kasa) fit is pulled off
    a = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    cy, cx = sol[0] / 2, sol[1] / 2
    radius = math.sqrt(sol[2] + cy**2 + cx**2)
    assert abs(radius - 25.0) > 0.5 or math.hypot(cy - 10.3, cx + 4.2) > 0.5
