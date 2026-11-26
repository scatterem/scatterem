"""Contracts for ``scatterem.utils.interpolation``: sampling and its adjoint."""

import pytest
import torch

from scatterem.utils.interpolation import insert_into_image, sample_image

PADDINGS = ["zeros", "periodic", "border"]


def _random(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=dtype)


def _coords(query, spatial, seed, margin=2.0):
    """Coordinates that reach ``margin`` pixels beyond every edge."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(*query, len(spatial), generator=g, dtype=torch.float64)
    n = torch.tensor(spatial, dtype=torch.float64)
    return u * (n - 1 + 2 * margin) - margin


@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("spatial", [(7,), (5, 6), (4, 5, 6)])
def test_insert_is_the_exact_adjoint_of_sample(spatial, dtype, padding):
    x = _random((2, *spatial), dtype, 0)
    c = _coords((3, 11), spatial, 1)
    y = _random((2, 3, 11), dtype, 2)
    lhs = torch.vdot(sample_image(x, c, padding=padding).flatten(), y.flatten())
    image, _ = insert_into_image(y, c, spatial, padding=padding)
    rhs = torch.vdot(x.flatten(), image.flatten())
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("spatial", [(7,), (5, 6), (4, 5, 6)])
def test_sampling_at_integer_coordinates_returns_the_pixels(spatial):
    x = _random(spatial, torch.float64, 3)
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(n, dtype=torch.float64) for n in spatial], indexing="ij"
        ),
        -1,
    )
    torch.testing.assert_close(sample_image(x, grid), x)


def test_sampling_is_linear_between_pixels():
    x = torch.tensor([[0.0, 10.0], [20.0, 30.0]], dtype=torch.float64)
    c = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    # bilinear: (1-.25)*(0+10)/2 + .25*(20+30)/2 = 3.75 + 6.25
    assert sample_image(x, c).item() == pytest.approx(10.0)


def test_paddings_differ_only_outside_the_grid():
    # x[0] = 1 (not 0), so zeros and border differ on the low edge too: c = -0.5
    # is half x[0] under zeros (0.5) and all of it under border (1.0).
    x = torch.arange(1.0, 6.0, dtype=torch.float64)
    c = torch.tensor([[-1.0], [-0.5], [2.5], [4.0], [5.0]], dtype=torch.float64)
    assert sample_image(x, c, padding="zeros").tolist() == [0.0, 0.5, 3.5, 5.0, 0.0]
    assert sample_image(x, c, padding="periodic").tolist() == [5.0, 3.0, 3.5, 5.0, 1.0]
    assert sample_image(x, c, padding="border").tolist() == [1.0, 1.0, 3.5, 5.0, 5.0]


def test_zeros_padding_reads_a_nan_edge_pixel_into_out_of_grid_samples():
    # Documented, not desired: an out-of-grid corner is read from the clamped
    # edge pixel and weighted by 0, and 0 * nan is nan.
    x = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float64)
    c = torch.tensor([[-3.0], [1.5]], dtype=torch.float64)
    out = sample_image(x, c, padding="zeros")
    assert torch.isnan(out[0]) and out[1].item() == 1.5
    image, _ = insert_into_image(
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.tensor([[-3.0]], dtype=torch.float64),
        (3,),
    )
    assert torch.isnan(image[0]) and not torch.isnan(image[1:]).any()


def test_batch_axes_share_the_coordinates():
    x = _random((3, 2, 5, 6), torch.complex128, 4)
    c = _coords((7,), (5, 6), 5)
    out = sample_image(x, c)
    assert out.shape == (3, 2, 7)
    torch.testing.assert_close(out[1, 0], sample_image(x[1, 0], c))


def test_insert_weights_are_the_insertion_of_ones():
    c = _coords((40,), (6, 7), 6, margin=0.0)
    ones = torch.ones(40, dtype=torch.float64)
    image, weights = insert_into_image(ones, c, (6, 7))
    torch.testing.assert_close(image, weights)
    # Bilinear weights of an in-grid point sum to one.
    assert weights.sum().item() == pytest.approx(40.0)


def test_sample_is_differentiable_in_image_and_coordinates():
    x = _random((4, 5), torch.float64, 7).requires_grad_()
    # keep coordinates away from integers, where the derivative jumps
    c = (
        torch.tensor([[1.3, 2.6], [0.4, 3.2], [2.7, 0.8]], dtype=torch.float64)
    ).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: sample_image(a, b), (x, c))


def test_insert_is_differentiable_in_values():
    v = _random((2, 6), torch.complex128, 8).requires_grad_()
    c = _coords((6,), (4, 5), 9)
    assert torch.autograd.gradcheck(lambda a: insert_into_image(a, c, (4, 5))[0], (v,))


def test_float32_image_with_float64_coordinates_keeps_the_image_dtype():
    x = torch.zeros(4, 4, dtype=torch.float32)
    c = torch.zeros(3, 2, dtype=torch.float64)
    assert sample_image(x, c).dtype == torch.float32
    assert insert_into_image(torch.zeros(3), c, (4, 4))[0].dtype == torch.float32


def test_arguments_are_validated():
    x = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="padding"):
        sample_image(x, torch.zeros(2, 2), padding="reflect")
    with pytest.raises(ValueError, match="D <= min"):
        sample_image(torch.zeros(4), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape"):
        insert_into_image(torch.zeros(2), torch.zeros(2, 2), (4, 4, 4))
    with pytest.raises(ValueError, match="does not end"):
        insert_into_image(torch.zeros(3), torch.zeros(2, 2), (4, 4))
    with pytest.raises(ValueError, match="D <= min"):
        sample_image(torch.zeros(2, 2, 2, 2), torch.zeros(3, 4))
    with pytest.raises(ValueError, match="D <= min"):
        sample_image(torch.zeros(4, 4), torch.zeros(3, 0))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8, torch.bool])
def test_integer_inputs_are_refused_not_floor_sampled(dtype):
    c = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    with pytest.raises(TypeError, match=rf"x must be .*{dtype}.*x\.float\(\)"):
        sample_image(torch.ones(4, 4, dtype=dtype), c)
    with pytest.raises(TypeError, match=rf"values must be .*{dtype}.*values\.float"):
        insert_into_image(torch.ones(1, dtype=dtype), c, (4, 4))
