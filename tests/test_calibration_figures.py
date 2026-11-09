"""TESTS E: figures of the rotation search and of scan maps."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.image import AxesImage

from scatterem.data.calibration._rotation_solver import RotationSearch
from scatterem.vis.rotation_figures import angle_search_figure, plot_scan_maps

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _shut_figures(monkeypatch):
    def no_show(*args, **kwargs):
        raise AssertionError("plt.show was called")

    monkeypatch.setattr(plt, "show", no_show)
    yield
    plt.close("all")


def record(curves, curve_kind="curl", rotated_180=False, angles=None):
    angles = np.linspace(-90, 89, 180) if angles is None else angles
    return RotationSearch(
        curve_kind=curve_kind,
        angle_grid_deg=angles,
        hypothesis_curves=curves,
        search_angle_deg=12.0,
        scan_turn_deg=-168.0 if rotated_180 else 12.0,
        axes_exchanged=False,
        rotated_180=rotated_180,
        sign_test="flipped" if rotated_180 else "kept",
        potential_skewness=1.0,
        evaluator="tensor_ops",
    )


def data_lines(ax):
    return [ln for ln in ax.get_lines() if len(np.unique(ln.get_xdata())) > 1]


def vertical_lines(ax):
    return [ln for ln in ax.get_lines() if len(np.unique(ln.get_xdata())) == 1]


# E.1 -------------------------------------------------------------------------


@pytest.mark.parametrize("curve_kind", ["curl", "divergence"])
def test_objective_figure_content(curve_kind):
    a = np.linspace(-90, 89, 180)
    curves = {
        "axes_kept": 1 + np.cos(np.radians(2 * (a - 12))),
        "axes_exchanged": 1.5 + 0.2 * np.cos(np.radians(2 * a)),
    }
    # Unsorted angles: the figure sorts them for drawing, the record stays.
    perm = np.random.default_rng(0).permutation(a.size)
    rec = record({k: v[perm] for k, v in curves.items()}, curve_kind, angles=a[perm])
    fig = angle_search_figure(rec, figure_inches=(4.0, 3.0))
    assert fig.number in plt.get_fignums()
    assert np.allclose(fig.get_size_inches(), (4.0, 3.0))
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(data_lines(ax)) == 2
    for ln in data_lines(ax):
        assert np.all(np.diff(ln.get_xdata()) > 0)
    vl = vertical_lines(ax)
    assert len(vl) == 1 and np.all(np.asarray(vl[0].get_xdata()) == 12.0)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("as measured" in t for t in texts)
    assert any("swapped" in t for t in texts)
    assert curve_kind in ax.get_ylabel()
    assert "°" in ax.get_xlabel()
    np.testing.assert_array_equal(rec.angle_grid_deg, a[perm])


def test_half_turn_label():
    a = np.linspace(-90, 89, 180)
    fig = angle_search_figure(
        record({"axes_kept": np.cos(np.radians(a)) ** 2}, rotated_180=True)
    )
    texts = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "180" in texts and "-168.0" in texts


@pytest.mark.parametrize(
    "curves",
    [
        {"axes_kept": np.zeros(180), "axes_exchanged": np.zeros(180)},
        {"axes_kept": np.full(180, np.nan), "axes_exchanged": np.full(180, np.nan)},
        {"axes_kept": np.full(180, 0.3), "axes_exchanged": np.linspace(0, 1, 180)},
        {"axes_exchanged": np.where(np.arange(180) % 7 == 0, np.inf, 1.0)},
    ],
)
def test_degenerate_curves_do_not_raise(curves):
    fig = angle_search_figure(record(curves))
    assert (
        len(data_lines(fig.axes[0]))
        + len(
            [ln for ln in fig.axes[0].get_lines() if np.all(np.isnan(ln.get_ydata()))]
        )
        >= 1
    )


def test_one_hypothesis_gives_one_line():
    fig = angle_search_figure(record({"axes_exchanged": np.linspace(0, 1, 180)}))
    assert len(data_lines(fig.axes[0])) == 1
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert not any(t.startswith("as measured") for t in texts)


def test_objective_figure_rejects_other_input():
    with pytest.raises(ValueError):
        angle_search_figure({"hypothesis_curves": {}})


# E.2 -------------------------------------------------------------------------


def test_scan_map_figure():
    rng = np.random.default_rng(1)
    signed = rng.standard_normal((12, 20))
    positive = rng.uniform(1, 2, (16, 10))
    third = torch.tensor(rng.uniform(0, 1, (9, 9)), device="cuda" if CUDA else "cpu")
    step = (0.5, 2.0)
    fig = plot_scan_maps(
        [signed, positive, third],
        scan_spacing=step,
        distance_unit=("nm", "A"),
        map_captions=["a", "b", "c"],
    )
    images = [im for ax in fig.axes for im in ax.get_images()]
    assert len(images) == 3
    for im, m in zip(images, [signed, positive, third.cpu().numpy()], strict=True):
        np.testing.assert_array_equal(np.asarray(im.get_array()), m)
        left, right, bottom, top = im.get_extent()
        assert np.isclose(right - left, m.shape[1] * step[1])
        assert np.isclose(bottom - top, m.shape[0] * step[0])
    assert images[0].norm(0.0) == pytest.approx(0.5)
    assert fig.axes[0].get_title() == "a"
    assert "Å" in fig.axes[0].get_xlabel() and "nm" in fig.axes[0].get_ylabel()
    # Negative control: the positive map is not centred on zero.
    assert images[1].norm.vmin >= 1.0


def test_scan_map_overrides():
    m = np.random.default_rng(2).standard_normal((8, 8))
    fig = plot_scan_maps(
        [m],
        scan_spacing=1.0,
        distance_unit="nm",
        limits=(-1, 3),
        colour_role="intensity",
        colour_key=False,
        length_bar=False,
        figure_inches=(3, 3),
    )
    im = fig.axes[0].get_images()[0]
    assert (im.norm.vmin, im.norm.vmax) == (-1, 3)
    assert len(fig.axes) == 1
    assert np.allclose(fig.get_size_inches(), (3, 3))
    auto = (
        plot_scan_maps([m], scan_spacing=1.0, distance_unit="nm")
        .axes[0]
        .get_images()[0]
    )
    assert im.cmap.name != auto.cmap.name


def test_scan_map_errors():
    maps = [np.zeros((4, 4))] * 3
    with pytest.raises(ValueError):
        plot_scan_maps(maps, scan_spacing=1.0, distance_unit="nm", panel_grid=(1, 2))
    with pytest.raises(ValueError):
        plot_scan_maps(maps, scan_spacing=1.0, distance_unit="nm", map_captions=["a"])
    with pytest.raises(ValueError):
        plot_scan_maps([], scan_spacing=1.0, distance_unit="nm")
    with pytest.raises(ValueError):
        plot_scan_maps([np.zeros(4)], scan_spacing=1.0, distance_unit="nm")
    with pytest.raises(ValueError):
        plot_scan_maps(
            [np.zeros((4, 4), complex)], scan_spacing=1.0, distance_unit="nm"
        )
    with pytest.raises(ValueError):
        plot_scan_maps(maps, scan_spacing=-1.0, distance_unit="nm")


def test_panel_grid_removes_unused_panels():
    fig = plot_scan_maps(
        [np.ones((4, 4))] * 3,
        scan_spacing=1.0,
        distance_unit="nm",
        panel_grid=(2, 2),
        colour_key=False,
    )
    assert len([ax for ax in fig.axes if ax.get_images()]) == 3
    assert len(fig.axes) == 3
    assert all(isinstance(ax.get_images()[0], AxesImage) for ax in fig.axes)
