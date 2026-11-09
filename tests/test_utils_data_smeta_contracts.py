"""Contracts for ``scatterem.data.s_meta.SMeta`` (scattering-matrix metadata).

The load-bearing contract here is the *storage form* of ``beamlets``: they are
kept pre-gathered at the ``B`` beam positions with shape ``(Bp, B)``, NOT as a
dense ``(Bp, MY, MX)`` one-hot scatter. The dense form costs tens of GB on real
grids and the only consumer (``_build_beamlet_basis_fft``) reads the weights at
the beams anyway. A silent revert to the dense form is the shape regression that
has broken the PRISM path before, so it is asserted explicitly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from scatterem.data import SMeta
from scatterem.data.s_meta import SMeta as SMetaDirect

M = (32, 32)
DX = np.array([0.1, 0.1], dtype=np.float32)


def _disk_mask(radius: float, shape=M) -> np.ndarray:
    """Aperture mask in the fftfreq (corner-origin) convention the class assumes.

    ``all_beams`` indexes the same grid as ``qf = fft_frequency_grid(...)``,
    so an on-axis aperture must have the DC pixel at ``[0, 0]``.
    """
    my, mx = shape
    yy, xx = np.mgrid[:my, :mx]
    centred = ((yy - my // 2) ** 2 + (xx - mx // 2) ** 2) <= radius**2
    return np.fft.ifftshift(centred)


def _smeta(all_beams=None, parent_beams=None, radius_px=4) -> SMeta:
    if all_beams is None:
        all_beams = _disk_mask(radius_px)
    if parent_beams is None:
        parent_beams = all_beams
    return SMeta(
        all_beams,
        parent_beams,
        numerical_aperture_radius_pixels=radius_px,
        dx=DX,
        M=M,
        device=torch.device("cpu"),
    )


def test_package_namespace_exports_the_s_meta_implementation():
    # The live SMeta is the s_meta one. A dead duplicate in the since-deleted
    # ``data_classes`` used to be shadowed by it (it read ``self.q`` before
    # assigning it).
    assert SMeta is SMetaDirect


class TestSMetaConstruction:
    def test_beam_counts_match_the_masks(self):
        all_beams = _disk_mask(4)
        parent_beams = _disk_mask(2)
        smeta = _smeta(all_beams, parent_beams)
        assert smeta.B == int(all_beams.sum())
        assert smeta.Bp == int(parent_beams.sum())

    def test_q_is_qf_shifted_by_half_a_reciprocal_pixel(self):
        smeta = _smeta()
        assert tuple(smeta.q.shape) == (2, *M)
        assert tuple(smeta.qf.shape) == (2, *M)
        # qf is plain fftfreq (DC at index 0); q is the pixel-CENTRE variant,
        # i.e. offset by half a reciprocal pixel on each axis -- not fftshifted.
        torch.testing.assert_close(smeta.qf[:, 0, 0], torch.zeros(2))
        half_pixel = torch.tensor(
            [0.5 / (M[0] * DX[0]), 0.5 / (M[1] * DX[1])], dtype=smeta.q.dtype
        )
        torch.testing.assert_close(
            smeta.q, smeta.qf + half_pixel[:, None, None], atol=1e-6, rtol=1e-5
        )

    def test_q_dft_alias_is_gone(self):
        # v0.4 has no compatibility aliases: use ``qf``.
        assert not hasattr(_smeta(), "q_dft")

    def test_squared_magnitudes_agree_with_their_grids(self):
        smeta = _smeta()
        torch.testing.assert_close(smeta.q2, (smeta.q**2).sum(0))
        torch.testing.assert_close(smeta.qf2, (smeta.qf**2).sum(0))

    def test_beam_numbers_enumerate_the_beams_and_mark_the_rest(self):
        smeta = _smeta()
        numbers = smeta.beam_numbers
        assert numbers.shape == smeta.all_beams.shape
        assert int(numbers[~smeta.all_beams].max()) == -1
        on_beams = numbers[smeta.all_beams]
        torch.testing.assert_close(on_beams, torch.arange(smeta.B))

    def test_beam_coordinate_tables_have_one_row_per_beam(self):
        smeta = _smeta()
        assert tuple(smeta.all_beams_coords.shape) == (smeta.B, 2)
        assert tuple(smeta.all_beams_q.shape) == (smeta.B, 2)
        assert tuple(smeta.parent_beams_coords.shape) == (smeta.Bp, 2)
        assert tuple(smeta.parent_beams_q.shape) == (smeta.Bp, 2)

    def test_r_indices_is_a_plain_pixel_index_grid(self):
        smeta = _smeta()
        assert tuple(smeta.r_indices.shape) == (2, *M)
        torch.testing.assert_close(
            smeta.r_indices[0, :, 0], torch.arange(M[0], dtype=smeta.r_indices.dtype)
        )
        torch.testing.assert_close(
            smeta.r_indices[1, 0, :], torch.arange(M[1], dtype=smeta.r_indices.dtype)
        )

    def test_masks_are_stored_as_bool_on_the_requested_device(self):
        smeta = _smeta()
        assert smeta.all_beams.dtype is torch.bool
        assert smeta.parent_beams.dtype is torch.bool
        assert smeta.all_beams.device == torch.device("cpu")

    def test_M_is_recorded_as_an_array(self):
        smeta = _smeta()
        np.testing.assert_array_equal(smeta.M, np.array(M))


class TestMakeBeamletMeta:
    @pytest.mark.parametrize("n_radial", [0, 1, 2, 3])
    def test_weights_are_built_for_the_parents_actually_selected(self, n_radial):
        """No ring count has its own path: (B, Bp) weights for the Bp parents in
        the mask, every row a partition of unity. ``n_radial_samples == 1`` used
        to return ``ones((B, 1))`` beside 7 parents -- weight 1 on each of them,
        49x the probe intensity."""
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=n_radial)
        assert beamlet.Bp == int(beamlet.parent_beams.sum())
        assert beamlet.Bp == 1 if n_radial == 0 else beamlet.Bp > 1
        assert tuple(beamlet.natural_neighbor_weights.shape) == (smeta.B, beamlet.Bp)
        assert tuple(beamlet.beamlets.shape) == (beamlet.Bp, smeta.B)
        torch.testing.assert_close(
            beamlet.beamlets.sum(0),
            torch.ones(smeta.B, dtype=beamlet.beamlets.dtype),
        )
        assert float(beamlet.beamlets.min()) >= 0.0

    def test_beamlets_are_stored_gathered_as_bp_by_b_not_as_a_dense_scatter(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        assert beamlet.beamlets.ndim == 2, (
            "beamlets must stay the gathered (Bp, B) table; a 3-D shape means "
            "the dense (Bp, MY, MX) one-hot scatter is back"
        )
        assert tuple(beamlet.beamlets.shape) == (beamlet.Bp, smeta.B)
        assert beamlet.beamlets.shape[1] != M[0], (
            "second axis must be the beam count B, not a detector dimension"
        )

    def test_beamlets_are_the_transpose_of_the_natural_neighbour_weights(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        torch.testing.assert_close(
            beamlet.beamlets.to(torch.float64),
            beamlet.natural_neighbor_weights.T.to(torch.float64),
        )

    def test_natural_neighbour_weights_are_a_partition_of_unity_over_parents(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        nnw = beamlet.natural_neighbor_weights
        assert tuple(nnw.shape) == (smeta.B, beamlet.Bp)
        torch.testing.assert_close(
            nnw.sum(dim=1).to(torch.float64),
            torch.ones(smeta.B, dtype=torch.float64),
            atol=1e-9,
            rtol=0,
        )
        assert bool((nnw >= 0).all())

    def test_more_radial_rings_give_at_least_as_many_parent_beams(self):
        smeta = _smeta()
        one = smeta.make_beamlet_meta(n_radial_samples=1)
        two = smeta.make_beamlet_meta(n_radial_samples=2)
        three = smeta.make_beamlet_meta(n_radial_samples=3)
        assert one.Bp <= two.Bp <= three.Bp

    def test_the_dc_beam_is_always_a_parent(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        assert bool(beamlet.parent_beams[0, 0]), "fftfreq DC pixel must be selected"

    def test_parent_beams_coords_are_consistent_with_bp(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        assert tuple(beamlet.parent_beams_coords.shape) == (beamlet.Bp, 2)
        assert int(beamlet.parent_beams.sum()) == beamlet.Bp

    def test_the_child_inherits_the_aperture_and_the_all_beams_mask(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        assert beamlet.B == smeta.B
        torch.testing.assert_close(beamlet.all_beams, smeta.all_beams)
        assert (
            beamlet.numerical_aperture_radius_pixels
            == smeta.numerical_aperture_radius_pixels
        )
        np.testing.assert_array_equal(beamlet.M, smeta.M)
        assert beamlet.device == smeta.device

    def test_parent_beams_are_a_subset_of_all_beams(self):
        smeta = _smeta()
        beamlet = smeta.make_beamlet_meta(n_radial_samples=2)
        assert bool((beamlet.parent_beams & ~smeta.all_beams).sum() == 0)

    def test_angular_sample_count_is_honoured(self):
        smeta = _smeta(radius_px=6)
        few = smeta.make_beamlet_meta(n_radial_samples=2, n_angular_samples=3)
        many = smeta.make_beamlet_meta(n_radial_samples=2, n_angular_samples=12)
        assert few.Bp < many.Bp

    def test_beamlets_are_deterministic(self):
        smeta = _smeta()
        first = smeta.make_beamlet_meta(n_radial_samples=2)
        second = smeta.make_beamlet_meta(n_radial_samples=2)
        torch.testing.assert_close(first.beamlets, second.beamlets)
        assert first.Bp == second.Bp


class TestSMetaHdf5RoundTrip:
    """``to_h5`` then ``from_h5`` must give back the same ``SMeta``.

    The pair was broken in three independent ways at once, all on the same
    never-exercised path: ``to_h5`` wrote ``self.N``, which ``__init__`` has not
    assigned since the ``N``/``f`` supersampling parameters were commented out
    of the signature; ``from_h5`` passed seven positional arguments to a
    six-parameter ``__init__``; and it stringified the h5 bytes scalar for
    ``device`` as ``"b\'cpu\'"``. Reproduced here as the round trip a caller
    would actually write.
    """

    def test_round_trip_restores_every_constructor_input(self, tmp_path):
        smeta = _smeta(radius_px=5)
        path = tmp_path / "s.h5"
        smeta.to_h5(str(path))

        restored = SMeta.from_h5(str(path))

        np.testing.assert_array_equal(restored.dx, smeta.dx)
        np.testing.assert_array_equal(restored.M, smeta.M)
        assert restored.device == smeta.device
        assert (
            restored.numerical_aperture_radius_pixels
            == smeta.numerical_aperture_radius_pixels
        )
        assert restored.B == smeta.B
        assert restored.Bp == smeta.Bp
        torch.testing.assert_close(restored.all_beams, smeta.all_beams)
        torch.testing.assert_close(restored.parent_beams, smeta.parent_beams)

    def test_round_trip_restores_every_derived_grid(self, tmp_path):
        smeta = _smeta(radius_px=5)
        path = tmp_path / "s.h5"
        smeta.to_h5(str(path))

        restored = SMeta.from_h5(str(path))

        for field in (
            "q",
            "qf",
            "q2",
            "qf2",
            "q_coords",
            "r_indices",
            "beam_numbers",
            "all_beams_q",
            "all_beams_coords",
            "parent_beams_q",
            "parent_beams_coords",
        ):
            torch.testing.assert_close(
                getattr(restored, field), getattr(smeta, field), msg=field
            )

    def test_round_trip_honours_the_key_argument(self, tmp_path):
        smeta = _smeta()
        path = tmp_path / "s.h5"
        smeta.to_h5(str(path), key="prism")

        with pytest.raises(KeyError):
            SMeta.from_h5(str(path))
        assert SMeta.from_h5(str(path), key="prism").B == smeta.B

    def test_from_h5_ignores_the_legacy_N_dataset(self, tmp_path):
        """Files written before ``N`` left ``__init__`` still carry it.

        ``N`` was the supersampled real-space grid and ``f = N // M`` the
        interpolation factor; both were commented out of ``__init__`` long ago
        and nothing reads them, so a stored ``N`` is now ignored rather than
        fed back in as a seventh positional argument.
        """
        import h5py

        smeta = _smeta()
        path = tmp_path / "s.h5"
        smeta.to_h5(str(path))
        with h5py.File(path, "a") as f:
            f["s_matrix_meta"].create_dataset("N", data=np.array([64, 64]))

        restored = SMeta.from_h5(str(path))

        np.testing.assert_array_equal(restored.M, smeta.M)
        assert not hasattr(restored, "N")


class TestSMetaDataclassSurface:
    """The declared fields must be the ones ``__init__`` assigns.

    ``f`` and ``N`` stayed declared after the supersampling parameters left
    ``__init__`` (the same root cause as the h5 round trip), so the generated
    ``__repr__`` read ``self.f`` first and raised ``AttributeError`` -- every
    ``repr``/``str`` of an ``SMeta``, including pytest's own failure output,
    was broken. Nothing in the package reads either name.
    """

    def test_every_declared_field_is_assigned(self):
        smeta = _smeta()
        for field in dataclasses.fields(smeta):
            assert hasattr(smeta, field.name), field.name

    def test_the_supersampling_fields_are_gone(self):
        names = {field.name for field in dataclasses.fields(SMeta)}
        assert "f" not in names
        assert "N" not in names

    def test_repr_and_str_do_not_raise(self):
        smeta = _smeta()
        text = repr(smeta)
        assert text.startswith("SMeta(")
        assert f"B={smeta.B}" in text
        assert str(smeta) == text
