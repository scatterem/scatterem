"""Oklab domain colouring, replacing the quantem-derived JCh renderer.

Assertions are against colour-space properties rather than against the
implementation being replaced.
"""

import numpy as np
import pytest

from scatterem.vis.complex_color import (
    complex_to_rgba,
    oklch_to_srgb,
    phase_wheel,
    tile_to_rgba,
)


class TestOklch:
    def test_achromatic_endpoints_are_black_and_white(self):
        z, o = np.array([0.0]), np.array([1.0])
        np.testing.assert_allclose(oklch_to_srgb(z, z, z), 0.0, atol=1e-9)
        np.testing.assert_allclose(oklch_to_srgb(o, z, z), 1.0, atol=1e-6)

    def test_zero_chroma_is_grey_at_every_lightness(self):
        L = np.linspace(0, 1, 32)
        rgb = oklch_to_srgb(L, np.zeros_like(L), np.zeros_like(L))
        np.testing.assert_allclose(rgb[:, 0], rgb[:, 1], atol=1e-6)
        np.testing.assert_allclose(rgb[:, 1], rgb[:, 2], atol=1e-6)

    def test_lightness_is_monotonic_in_output_luminance(self):
        """The property that makes amplitude readable: more L, more light."""
        L = np.linspace(0.05, 0.95, 32)
        rgb = oklch_to_srgb(L, np.full_like(L, 0.1), np.full_like(L, 1.0))
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        assert np.all(np.diff(lum) > 0)

    def test_hue_is_periodic(self):
        L, C = np.array([0.7]), np.array([0.1])
        a = oklch_to_srgb(L, C, np.array([0.3]))
        b = oklch_to_srgb(L, C, np.array([0.3 + 2 * np.pi]))
        np.testing.assert_allclose(a, b, atol=1e-9)


class TestComplexToRgba:
    def test_result_is_always_in_gamut(self):
        """Chroma is reduced to fit rather than clipped, which would shift hue."""
        amp = np.linspace(0, 1, 64)[:, None] * np.ones(64)
        ph = np.linspace(-np.pi, np.pi, 64)[None, :] * np.ones((64, 1))
        rgba = complex_to_rgba(amp, ph, chroma=0.4)  # deliberately out of gamut
        assert (rgba >= 0).all() and (rgba <= 1).all()

    def test_equal_phase_steps_give_distinguishable_hues(self):
        ph = np.linspace(-np.pi, np.pi, 9)[:-1]
        rgba = complex_to_rgba(np.full(8, 0.7), ph)
        # every adjacent pair differs; no two phases collapse to one colour
        assert np.abs(np.diff(rgba[:, :3], axis=0)).max(axis=1).min() > 0.01

    def test_amplitude_drives_lightness_not_saturation(self):
        """Putting amplitude in chroma would grey out AND wash out weak signal."""
        ph = np.full(2, 1.0)
        rgba = complex_to_rgba(np.array([0.2, 0.9]), ph)
        lum = rgba[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
        assert lum[1] > lum[0]

    def test_real_input_is_colormapped(self):
        out = complex_to_rgba(np.linspace(0, 1, 16))
        assert out.shape == (16, 4)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-6)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="must match"):
            complex_to_rgba(np.zeros((4, 4)), np.zeros((4, 5)))

    def test_alpha_is_opaque(self):
        rgba = complex_to_rgba(np.full((3, 3), 0.5), np.zeros((3, 3)))
        np.testing.assert_allclose(rgba[..., 3], 1.0)


class TestPhaseWheel:
    def test_covers_a_full_turn_and_keeps_chroma(self):
        rgb, angles = phase_wheel(n=64)
        assert rgb.shape == (64, 3)
        np.testing.assert_allclose(angles[0], -np.pi)
        np.testing.assert_allclose(angles[-1], np.pi)
        # no hue desaturates to grey
        spread = rgb.max(axis=1) - rgb.min(axis=1)
        assert spread.min() > 0.02


class TestTile:
    def test_grid_is_tiled_in_reading_order(self):
        """Panels land row-major.

        Tested with an explicit pass-through norm, because the default per-panel
        scaling deliberately erases brightness differences *between* panels --
        which would make a dark panel and a light panel identical here and the
        ordering untestable.
        """
        a = np.zeros((4, 4))
        b = np.ones((4, 4))
        out = tile_to_rgba([[a, b], [b, a]], norm=lambda x: x)
        assert out.shape == (8, 8, 4)
        assert out[:4, :4, 0].mean() < out[:4, 4:, 0].mean()  # top-left dark
        assert out[4:, 4:, 0].mean() < out[4:, :4, 0].mean()  # bottom-right dark

    def test_each_panel_is_scaled_to_its_own_range(self):
        """A faint panel is stretched to full contrast rather than left near-black.

        One normalisation across the whole grid leaves everything but the
        brightest panel unreadable, which is the failure this avoids.
        """
        faint = np.linspace(0, 0.1, 16).reshape(4, 4)
        bright = np.linspace(0, 100.0, 16).reshape(4, 4)
        out = tile_to_rgba([[faint, bright]])
        left, right = out[:, :4, 0], out[:, 4:, 0]
        assert left.max() > 0.99, "the faint panel should reach full lightness"
        assert np.allclose(
            left, right, atol=1e-12
        ), "two panels differing only in scale must render identically"

    def test_values_outside_zero_one_are_not_clipped_to_white(self):
        """complex_to_rgba requires [0, 1] and CLIPS; tiling must normalise first.

        Without it a 0..1000 image rendered 99.9% pure white -- every pixel above
        1.0 clamped to the top of the colormap.
        """
        out = tile_to_rgba([[np.linspace(0, 1000, 64 * 64).reshape(64, 64)]])
        assert len(np.unique(out[..., 0])) > 100, "gradient collapsed to a few levels"
        assert float((out[..., 0] >= 0.999).mean()) < 0.05

    def test_a_constant_panel_does_not_divide_by_zero(self):
        out = tile_to_rgba([[np.full((4, 4), 7.0)]])
        assert np.isfinite(out).all()

    def test_non_finite_values_do_not_poison_the_range(self):
        panel = np.linspace(0, 1, 16).reshape(4, 4).copy()
        panel[0, 0] = np.nan
        out = tile_to_rgba([[panel]])
        assert np.isfinite(out[1:, :]).all()

    def test_mismatched_panels_raise(self):
        with pytest.raises(ValueError, match="share a shape"):
            tile_to_rgba([[np.zeros((4, 4)), np.zeros((5, 5))]])
