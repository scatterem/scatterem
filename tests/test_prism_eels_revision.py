"""Small CPU regressions for detector phases and resolved PRISM-EELS output.

Synthetic localized transition channels avoid optional atomic-data backends.
"""

import numpy as np
import pytest
import torch

from scatterem.simulation.eels.multislice_eels import propagator_kernel
from scatterem.simulation.eels.prism_eels import prism_transition_potential
from scatterem.simulation.eels.prism_eels_image import (
    DetectorExitSMatrix,
    prism_eels_image,
)
from scatterem.simulation.eels.transition_potentials import TransitionPotentials


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _detector(vacuum=True, partition=None, **kwargs):
    n = 16
    g = torch.fft.fftfreq(n, dtype=torch.float64) * n
    det = torch.nonzero(g[:, None] ** 2 + g[None, :] ** 2 <= 25)
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    phase = torch.stack(
        [0.3 * torch.sin((y + j) * 0.7) * torch.cos(x * 0.9) for j in range(5)]
    )
    transmissions = torch.ones((5, n, n), dtype=torch.complex128)
    if not vacuum:
        transmissions = torch.exp(1j * phase.double())
    kernel = propagator_kernel(
        (n, n), (8.0, 8.0), 0.037, 7.0, dtype=torch.complex128, device="cpu"
    )
    return DetectorExitSMatrix(
        det, transmissions, kernel, partition=partition, **kwargs
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("mag_preserve", [False, True])
def test_partitioned_detector_vacuum_phase_at_each_depth(wrapped, mag_preserve):
    """Interpolation must reproduce the analytic conjugate beam INCLUDING phase."""
    sm = _detector(partition={"n_radial": 2}, mag_preserve=mag_preserve)
    assert sm.S.shape[0] < sm.ndet  # actually exercise interpolation
    iy = torch.tensor([14, 15, 0, 1]) if wrapped else torch.arange(4, 10)
    ix = torch.tensor([15, 0, 1, 2, 3]) if wrapped else torch.arange(3, 8)
    signed = (sm.det_idx + 8) % 16 - 8
    plane = torch.exp(
        -2j
        * torch.pi
        * (
            signed[:, 0, None, None] * iy.double()[None, :, None] / 16
            + signed[:, 1, None, None] * ix.double()[None, None, :] / 16
        )
    )
    for start in (0, 1, 3, 4):
        sm.peel_to(start)
        phase = sm.kernel[sm.det_idx[:, 0], sm.det_idx[:, 1]] ** (4 - start)
        torch.testing.assert_close(
            sm.columns_window(iy, ix),
            plane * phase[:, None, None],
            rtol=2e-11,
            atol=2e-11,
        )


def test_detector_dechirp_ablation_changes_vacuum_phase():
    corrected = _detector(partition={"n_radial": 2})
    ablated = _detector(partition={"n_radial": 2, "dechirp": False})
    idx = torch.arange(16)
    assert (
        corrected.columns_window(idx, idx) - ablated.columns_window(idx, idx)
    ).abs().max() > 0.01


def test_detector_full_parent_limit_in_specimen():
    exact = _detector(vacuum=False)
    full = _detector(vacuum=False, partition={"n_radial": 12, "n_angular": 12})
    assert full.S.shape[0] == exact.ndet
    iy, ix = torch.tensor([15, 0, 1]), torch.tensor([14, 15, 0, 1])
    for start in (0, 2, 4):
        exact.peel_to(start)
        full.peel_to(start)
        torch.testing.assert_close(
            full.columns_window(iy, ix),
            exact.columns_window(iy, ix),
            rtol=2e-11,
            atol=2e-11,
        )


@pytest.fixture(scope="module")
def specimen():
    n = 12
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    signed_y, signed_x = (y + n // 2) % n - n // 2, (x + n // 2) % n - n // 2
    h = torch.exp(-(signed_y**2 + signed_x**2) / 2)
    channels = torch.stack([h, h * (signed_y + 1j * signed_x) / 2]).to(torch.complex128)
    tp = TransitionPotentials(
        channels, 8, 100_000.0, (0.5, 0.5), np.array([530.0, 531.0])
    )
    transmissions = torch.exp(
        1j
        * torch.stack(
            [
                0.4 * torch.sin((y + j) * 0.7) * torch.cos((x - j) * 0.5)
                for j in range(3)
            ]
        )
    )
    pq = (signed_y**2 + signed_x**2 <= 9).to(torch.complex128)
    pq /= (pq.abs().square().sum() / pq.numel()).sqrt()
    sites = np.array([[0.02, 0.98, 0.01], [0.4, 0.6, 0.45], [0.7, 0.2, 0.9]])
    scan = torch.tensor([[0.0, 0.0], [11.0, 1.0], [4.25, 7.5]], dtype=torch.float64)
    return (pq, transmissions, tp, sites, scan), dict(
        wavelength=0.037,
        gridsize=(6.0, 6.0),
        slice_distance=2.0,
        detector_mrad=24.0,
        prism_mask=False,
    )


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("crop", [None, 2])
def test_resolved_projections_and_contractions(specimen, partitioned, crop):
    args, common = specimen
    options = dict(common, inelastic_crop=crop)
    if partitioned:
        options.update(partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    integrated = prism_eels_image(*args, **options)
    assert torch.isfinite(integrated).all() and integrated.min() > 0
    both, q_both = prism_eels_image(*args, **options, qeels_axis="both")
    torch.testing.assert_close(both.sum(-1), integrated, rtol=1e-11, atol=1e-12)
    probe, probe_q = prism_eels_image(
        *args, **options, qeels_axis="both", contract="probe", scan_chunk=1
    )
    torch.testing.assert_close(probe_q, q_both)
    torch.testing.assert_close(probe, both, rtol=1e-11, atol=1e-12)
    for axis in (0, 1):
        projected, q = prism_eels_image(*args, **options, qeels_axis=axis)
        assert torch.all(q[1:] > q[:-1])
        reference = torch.stack(
            [both[:, q_both[:, axis] == value].sum(-1) for value in q], dim=-1
        )
        torch.testing.assert_close(projected, reference, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(
            projected.sum(-1), integrated, rtol=1e-11, atol=1e-12
        )


def test_resolved_beams_match_independent_4d_path(specimen):
    args, common = specimen
    physics = {key: common[key] for key in ("wavelength", "gridsize", "slice_distance")}
    cube = prism_transition_potential(*args, **physics)
    both, q = prism_eels_image(*args, **common, qeels_axis="both")
    torch.testing.assert_close(
        both, cube[:, q[:, 0] % 12, q[:, 1] % 12], rtol=1e-10, atol=1e-12
    )


def test_full_parent_and_focal_limits(specimen):
    args, common = specimen
    exact = prism_eels_image(*args, **common)
    parents = {"n_radial": 12, "n_angular": 12}
    for focal in (None, 0.0, "centroid", 0.5):
        full = prism_eels_image(
            *args,
            **common,
            partition=parents,
            partition_s2=parents,
            focal_backprop=focal,
        )
        torch.testing.assert_close(full, exact, rtol=1e-10, atol=1e-12)
    sparse = dict(partition={"n_radial": 2}, partition_s2={"n_radial": 2})
    default = prism_eels_image(*args, **common, **sparse)
    zero = prism_eels_image(
        *args, **common, **sparse, focal_backprop=0.0, mag_preserve=True
    )
    torch.testing.assert_close(default, zero, rtol=1e-12, atol=1e-12)
    centroid = prism_eels_image(*args, **common, **sparse, focal_backprop="centroid")
    half = prism_eels_image(*args, **common, **sparse, focal_backprop=0.5)
    torch.testing.assert_close(centroid, half, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def cropped_focal_specimen():
    """Large enough that the four-pixel Fresnel margin cannot fill the grid."""
    n = 64
    generator = torch.Generator().manual_seed(731)
    phase = 0.4 * torch.randn((5, n, n), dtype=torch.float64, generator=generator)
    transmissions = torch.exp(1j * phase)
    freq = torch.fft.fftfreq(n, dtype=torch.float64) * n
    probe = (freq[:, None] ** 2 + freq[None, :] ** 2 <= 9).to(torch.complex128)
    physics = dict(wavelength=0.037, slice_distance=2.0, gridsize=(32.0, 32.0))
    kernel = propagator_kernel(
        (n, n),
        physics["gridsize"],
        physics["wavelength"],
        physics["slice_distance"],
        device="cpu",
        dtype=torch.complex128,
    )
    return probe, transmissions, kernel, physics


@pytest.mark.parametrize("focal", ["centroid", 0.5])
@pytest.mark.parametrize("leg", ["s1", "s2"])
def test_full_parent_focal_small_crop_is_exact(cropped_focal_specimen, focal, leg):
    from scatterem.simulation.eels.prism_eels import (
        PartitionedScatteringMatrix,
        ScatteringMatrix,
    )

    probe, transmissions, kernel, physics = cropped_focal_specimen
    iy, ix = torch.tensor([62, 63, 0, 1]), torch.arange(17, 21)
    if leg == "s1":
        exact = ScatteringMatrix(probe, transmissions, kernel)
        full = PartitionedScatteringMatrix(
            probe,
            transmissions,
            kernel,
            n_radial=12,
            n_angular=12,
            focal_backprop=focal,
            **physics,
        )
        assert full.n_parents == exact.S.shape[0] == 29
        exact.advance_to(4)
        full.advance_to(4)
        result = full.reconstruct_columns_window(iy, ix)
    else:
        det = torch.nonzero(probe)
        exact = DetectorExitSMatrix(det, transmissions, kernel)
        full = DetectorExitSMatrix(
            det,
            transmissions,
            kernel,
            partition={"n_radial": 12, "n_angular": 12},
            focal_backprop=focal,
            **physics,
        )
        assert full.S.shape[0] == exact.ndet == 29
        exact.peel_to(1)
        full.peel_to(1)
        result = full.columns_window(iy, ix)
    torch.testing.assert_close(result, exact.S[:, iy][:, :, ix], rtol=2e-11, atol=2e-11)


def test_partitioned_s1_positional_focal_backward_compatibility(cropped_focal_specimen):
    from scatterem.simulation.eels.prism_eels import PartitionedScatteringMatrix

    probe, transmissions, kernel, physics = cropped_focal_specimen
    # Before v0.3 the seventh positional argument was focal_backprop.
    positional = PartitionedScatteringMatrix(
        probe, transmissions, kernel, 2, 6, (1, 1), 0.5, **physics
    )
    keyword = PartitionedScatteringMatrix(
        probe, transmissions, kernel, 2, 6, (1, 1), focal_backprop=0.5, **physics
    )
    positional.advance_to(4)
    keyword.advance_to(4)
    iy, ix = torch.arange(17, 21), torch.arange(33, 37)
    torch.testing.assert_close(
        positional.reconstruct_columns_window(iy, ix),
        keyword.reconstruct_columns_window(iy, ix),
        rtol=2e-11,
        atol=2e-11,
    )
