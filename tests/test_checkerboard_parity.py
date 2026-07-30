"""Behaviour of the checkerboard bright-field parity used by the empirical SSNR.

The half-split feeding ``direct_ptychography_empirical_ssnr`` is only a valid
noise estimate if the two detector-pixel subsets are a genuine checkerboard:
equal-sized, complementary, interleaved, and decided by the index VALUES rather
than by position in the (radius-ordered) index list.
"""

import pytest
import torch

from scatterem.reconstruction.direct_ptychography import _checkerboard_parity


def _grid(n_rows, n_cols, row0=0, col0=0):
    return torch.tensor(
        [[row0 + r, col0 + c] for r in range(n_rows) for c in range(n_cols)],
        dtype=torch.long,
    )


def test_4x4_grid_splits_into_two_equal_interleaved_sets():
    inds = _grid(4, 4)
    parity = _checkerboard_parity(inds)

    assert parity.dtype == torch.bool
    assert parity.shape == (16,)
    assert int(parity.sum()) == 8
    assert int((~parity).sum()) == 8

    # Interleaved: every 4-neighbour of a pixel lands in the other subset.
    colour = {(int(r), int(c)): bool(p) for (r, c), p in zip(inds, parity)}
    for (r, c), value in colour.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = colour.get((r + dr, c + dc))
            if neighbour is not None:
                assert neighbour is not value, f"({r},{c}) matches ({r+dr},{c+dc})"


def test_subsets_are_complementary_and_disjoint():
    inds = _grid(5, 7)  # odd count: the split is as even as a checkerboard allows
    parity = _checkerboard_parity(inds)

    rows = [(int(r), int(c)) for r, c in inds]
    set_a = {rc for rc, p in zip(rows, parity) if bool(p)}
    set_b = {rc for rc, p in zip(rows, parity) if not bool(p)}

    assert set_a.isdisjoint(set_b)
    assert set_a | set_b == set(rows)
    assert len(set_a) + len(set_b) == len(rows)
    assert abs(len(set_a) - len(set_b)) <= 1


def test_index_origin_offset_only_swaps_which_subset_is_true():
    """The docstring's claim: absolute vs centered indices is immaterial.

    An integer translation of the index grid maps the checkerboard onto itself.
    An ODD translation (a shift of one row OR one column) swaps the labels; an
    EVEN one (both, or two rows) leaves them untouched. Either way the PARTITION
    is the same, which is all a symmetric half-split needs.
    """
    base = _grid(4, 4)
    parity = _checkerboard_parity(base)

    for offset in ((1, 0), (0, 1), (3, 0), (0, -1), (-5, 2)):
        shifted = _checkerboard_parity(base + torch.tensor(offset))
        assert torch.equal(shifted, ~parity), f"odd shift {offset} must invert"

    for offset in ((1, 1), (2, 0), (0, 2), (-1, 1), (-4, -6)):
        shifted = _checkerboard_parity(base + torch.tensor(offset))
        assert torch.equal(shifted, parity), f"even shift {offset} must preserve"


def test_unordered_noncontiguous_indices_use_index_value_not_position():
    """The real input is ``bright_field_inds_ordered_by_radius``: a ragged disk
    of indices sorted by radius, so neither raster-ordered nor contiguous. A
    naive implementation keyed on array position gets this wrong."""
    # A radius-ordered ragged disk with gaps, deliberately scrambled.
    inds = torch.tensor(
        [
            [4, 4],
            [3, 9],
            [10, 2],
            [4, 5],
            [7, 7],
            [3, 8],
            [11, 2],
            [0, 0],
            [6, 13],
            [9, 4],
            [2, 3],
        ],
        dtype=torch.long,
    )
    parity = _checkerboard_parity(inds)

    expected = torch.tensor([(int(r) + int(c)) % 2 == 1 for r, c in inds])
    assert torch.equal(parity, expected)

    # Position-based colouring would alternate down the list; the value-based
    # answer must NOT coincide with it, or the test proves nothing.
    positional = (torch.arange(inds.shape[0]) % 2).bool()
    assert not torch.equal(parity, positional)

    # Colouring is a property of the pixel, not of the list order.
    perm = torch.randperm(inds.shape[0])
    assert torch.equal(_checkerboard_parity(inds[perm]), parity[perm])


def test_matches_row_plus_col_parity_including_negative_indices():
    """Equivalence to the definition (row + col) % 2 over centered indices."""
    inds = _grid(9, 9, row0=-4, col0=-4)
    expected = ((inds[:, 0] + inds[:, 1]) % 2).bool()
    assert torch.equal(_checkerboard_parity(inds), expected)


def test_whole_number_floats_accepted_but_non_integral_rejected():
    inds = _grid(4, 4)
    assert torch.equal(_checkerboard_parity(inds.double()), _checkerboard_parity(inds))

    # vBF.k-like input: mean-centered, scaled by the reciprocal sampling.
    k_like = (inds.float() - inds.float().mean(dim=0)) * 0.0123
    with pytest.raises(ValueError, match="vBF.k"):
        _checkerboard_parity(k_like)


def test_malformed_and_empty_inputs():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _checkerboard_parity(torch.zeros((4, 3), dtype=torch.long))
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        _checkerboard_parity(torch.arange(6))

    # A vBF with no bright-field pixels yields an empty mask, letting the caller
    # fall back to the analytical SSNR rather than raising IndexError.
    empty = _checkerboard_parity(torch.empty(0, dtype=torch.long))
    assert empty.dtype == torch.bool and empty.numel() == 0
    assert _checkerboard_parity(torch.empty((0, 2), dtype=torch.long)).numel() == 0
