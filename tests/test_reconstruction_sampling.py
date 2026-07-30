"""The scale-bar helper. Its whole reason to exist is the upsample factor."""

import numpy as np
import pytest
import torch

from scatterem.vis.reconstruction_sampling import reconstruction_sampling


class _FakeDataset:
    def __init__(self, scan=(96, 96), step=(0.25, 0.25)):
        self.array = torch.zeros(scan + (8, 8))
        self.sampling = np.array([step[0], step[1], 1.0, 1.0])


def test_unupsampled_reconstruction_keeps_the_scan_step():
    s = reconstruction_sampling(_FakeDataset(), torch.zeros(96, 96))
    np.testing.assert_allclose(s.pixel_size, (0.25, 0.25))


def test_fourfold_upsample_quarters_the_pixel():
    """Fig. 4's case: passing dataset.sampling here would be 4x too long."""
    s = reconstruction_sampling(_FakeDataset(scan=(256, 256)), torch.zeros(1024, 1024))
    np.testing.assert_allclose(s.pixel_size, (0.25 / 4, 0.25 / 4))


def test_anisotropic_scan_and_output():
    ds = _FakeDataset(scan=(64, 128), step=(0.2, 0.4))
    s = reconstruction_sampling(ds, torch.zeros(128, 128))
    np.testing.assert_allclose(s.pixel_size, (0.2 * 64 / 128, 0.4 * 128 / 128))


def test_extra_leading_dimensions_are_ignored():
    s = reconstruction_sampling(_FakeDataset(), torch.zeros(3, 96, 96))
    np.testing.assert_allclose(s.pixel_size, (0.25, 0.25))


def test_shape_may_be_given_instead_of_an_image():
    s = reconstruction_sampling(_FakeDataset(), shape=(192, 192))
    np.testing.assert_allclose(s.pixel_size, (0.125, 0.125))


def test_units_reach_the_result():
    s = reconstruction_sampling(_FakeDataset(), torch.zeros(96, 96))
    assert tuple(s.units) == ("Å", "Å")


def test_needs_an_image_or_a_shape():
    with pytest.raises(ValueError, match="either the reconstructed image or its shape"):
        reconstruction_sampling(_FakeDataset())


def test_rejects_a_non_2d_shape():
    with pytest.raises(ValueError, match="2D output shape"):
        reconstruction_sampling(_FakeDataset(), shape=(4, 4, 4))


def test_agrees_with_the_inline_calculation_the_scripts_used():
    """Pinned against the expression the four figure scripts each carried."""
    ds = _FakeDataset(scan=(128, 128), step=(0.727, 0.727))
    img = torch.zeros(512, 512)
    expected = (
        float(ds.sampling[0]) * int(ds.array.shape[0]) / img.shape[-2],
        float(ds.sampling[1]) * int(ds.array.shape[1]) / img.shape[-1],
    )
    np.testing.assert_allclose(reconstruction_sampling(ds, img).pixel_size, expected)
