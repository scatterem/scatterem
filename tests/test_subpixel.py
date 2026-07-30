"""Sub-pixel registration, tested against known shifts rather than against
another implementation.

The point of this module is that it is independent, so testing it by comparing to
the implementation it replaced would defeat the purpose. Everything here asserts a
property of the answer instead: an injected shift comes back, to a stated accuracy.
"""


import numpy as np
import pytest
import torch

from scatterem.utils.subpixel import (
    REFINE_RADIUS_PX,
    _parabolic_vertex,
    _signed_fft_frequencies,
    make_neighbor_pairs,
    pairwise_relative_shifts,
    relative_shifts,
    subpixel_shifts,
    synchronize_shifts,
)

N = 64


def _gaussian(dy, dx, n=N, sigma=3.0, dtype=torch.float64):
    y, x = torch.meshgrid(
        torch.arange(n, dtype=dtype), torch.arange(n, dtype=dtype), indexing="ij"
    )
    return torch.exp(-(((y - n / 2 - dy) ** 2 + (x - n / 2 - dx) ** 2) / (2 * sigma**2)))


def _spectra(offsets, **kw):
    return torch.fft.fft2(torch.stack([_gaussian(dy, dx, **kw) for dy, dx in offsets]))


class TestKnownShifts:
    """The shift returned is the NEGATIVE of the offset applied to the second
    image, because it reports the shift of that image relative to the reference."""

    @pytest.mark.parametrize("upsample", [4, 8, 16, 20])
    def test_recovers_injected_offsets(self, upsample):
        offsets = [(0.0, 0.0), (2.0, -3.0), (2.25, -1.5), (-0.4, 0.7), (5.125, 4.875)]
        got = subpixel_shifts(
            torch.fft.fft2(_gaussian(0, 0)), _spectra(offsets), upsample
        )
        expected = -torch.tensor(offsets, dtype=torch.float64)
        # Parabolic interpolation between grid samples beats the 1/U grid itself,
        # so the tolerance is well under one step.
        np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-4)

    def test_integer_shift_is_exact(self):
        got = subpixel_shifts(
            torch.fft.fft2(_gaussian(0, 0)), _spectra([(3.0, -5.0)]), 8
        )
        np.testing.assert_allclose(got.numpy(), [[-3.0, 5.0]], atol=1e-6)

    def test_zero_shift_is_zero(self):
        got = subpixel_shifts(torch.fft.fft2(_gaussian(0, 0)), _spectra([(0.0, 0.0)]), 8)
        np.testing.assert_allclose(got.numpy(), [[0.0, 0.0]], atol=1e-9)

    def test_survives_noise(self):
        torch.manual_seed(0)
        ref = _gaussian(0, 0)
        moved = _gaussian(1.3, -0.8) + 0.05 * torch.rand(N, N, dtype=torch.float64)
        got = subpixel_shifts(torch.fft.fft2(ref), torch.fft.fft2(moved), 16)
        np.testing.assert_allclose(got.numpy(), [[-1.3, 0.8]], atol=0.05)

    def test_non_square_grid(self):
        ref = _gaussian(0, 0, n=48)
        moved = _gaussian(1.5, -2.5, n=48)
        got = subpixel_shifts(torch.fft.fft2(ref), torch.fft.fft2(moved), 8)
        np.testing.assert_allclose(got.numpy(), [[-1.5, 2.5]], atol=1e-4)

    def test_single_image_returns_one_row(self):
        got = subpixel_shifts(
            torch.fft.fft2(_gaussian(0, 0)), torch.fft.fft2(_gaussian(1.0, 1.0)), 8
        )
        assert got.shape == (1, 2)

    def test_relative_shifts_is_the_same_function(self):
        ref, moved = torch.fft.fft2(_gaussian(0, 0)), torch.fft.fft2(_gaussian(1.25, 0.5))
        assert torch.equal(relative_shifts(ref, moved, 8), subpixel_shifts(ref, moved, 8))


class TestValidation:
    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(ValueError, match="must be"):
            subpixel_shifts(torch.zeros(8, 8, dtype=torch.complex128), torch.zeros(8))

    def test_rejects_mismatched_grids(self):
        with pytest.raises(ValueError, match="same grid"):
            subpixel_shifts(
                torch.zeros(8, 8, dtype=torch.complex128),
                torch.zeros(1, 8, 16, dtype=torch.complex128),
            )

    def test_rejects_non_positive_upsample(self):
        z = torch.zeros(8, 8, dtype=torch.complex128)
        with pytest.raises(ValueError, match="must be positive"):
            subpixel_shifts(z, z, 0)


class TestPieces:
    @pytest.mark.parametrize("n", [1, 2, 7, 8, 15, 16, 64])
    def test_signed_frequencies_match_fft_layout(self, n):
        got = _signed_fft_frequencies(n, "cpu", torch.float64)
        np.testing.assert_allclose(got.numpy(), np.fft.fftfreq(n), atol=0, rtol=0)

    def test_parabolic_vertex_finds_a_known_maximum(self):
        # A parabola peaking a quarter-step right of the middle sample.
        peak = 0.25
        samples = torch.tensor([-((i - 1 - peak) ** 2) for i in range(3)])
        assert _parabolic_vertex(samples, 1) == pytest.approx(peak, abs=1e-12)

    def test_parabolic_vertex_is_zero_on_the_boundary(self):
        s = torch.tensor([3.0, 2.0, 1.0])
        assert _parabolic_vertex(s, 0) == 0.0
        assert _parabolic_vertex(s, 2) == 0.0

    def test_parabolic_vertex_is_zero_for_collinear_samples(self):
        assert _parabolic_vertex(torch.tensor([1.0, 2.0, 3.0]), 1) == 0.0

    def test_parabolic_vertex_is_clamped(self):
        """An extrapolation past half a step would move the answer to a sample the
        search already rejected."""
        s = torch.tensor([0.0, 1.0, 100.0])
        assert abs(_parabolic_vertex(s, 1)) <= 0.5

    def test_refine_radius_is_at_least_half_a_pixel(self):
        """The true maximum is within half a pixel of the integer peak."""
        assert REFINE_RADIUS_PX >= 0.5


class TestShiftGraph:
    def _grid(self):
        coords = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]], dtype=torch.long)
        offsets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        return coords, torch.fft.fft2(torch.stack([_gaussian(a, b, n=32) for a, b in offsets]))

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_neighbor_pairs_are_adjacent(self, connectivity):
        coords, _ = self._grid()
        pairs = make_neighbor_pairs(coords, connectivity)
        for i, j in pairs.tolist():
            step = (coords[j] - coords[i]).abs().max().item()
            assert step == 1

    def test_rejects_other_connectivity(self):
        coords, _ = self._grid()
        with pytest.raises(ValueError, match="connectivity"):
            make_neighbor_pairs(coords, 6)

    def test_synchronised_shifts_recover_the_layout(self):
        """Absolute positions from pairwise differences, up to the arbitrary gauge."""
        coords, stack = self._grid()
        pairs = make_neighbor_pairs(coords, 8)
        deltas = pairwise_relative_shifts(stack, pairs, 8)
        absolute = synchronize_shifts(len(coords), pairs, deltas)
        centred = absolute - absolute.mean(0)
        truth = -torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=centred.dtype,
        )
        truth = truth - truth.mean(0)
        np.testing.assert_allclose(centred.numpy(), truth.numpy(), atol=0.05)
