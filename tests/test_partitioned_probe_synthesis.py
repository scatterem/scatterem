"""Pin the probe-synthesis contract the partitioned EELS forms rely on.

(a) PartitionedScatteringMatrix.probe_at_current_plane == coeffs_at() @
    reconstruct_columns() -- including with focal back-propagation active
    (mid-stack plane, nontrivial transmissions). This is the documented
    contract; the memory-bounded inelastic forms synthesise the per-slice
    centroid probe directly from the Bp parents and rely on it being the SAME
    field the reconstructed-column contraction forms.
(b) ScatteringMatrix.probe_at_current_plane == einsum(coeffs_at, S) -- the
    inline synthesis _compute_prism_eels used before delegating to the method.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_DEV = "cuda"


def _aperture_probe_q(n, radius_px, device):
    """Corner-origin circular aperture (Ny, Nx) complex64, unit amplitude."""
    fy = torch.fft.fftfreq(n, device=device) * n
    q2 = fy[:, None] ** 2 + fy[None, :] ** 2
    return (q2 <= radius_px**2).to(torch.complex64)


def _random_transmissions(nz, n, device, seed=0):
    """Weak-phase random transmissions exp(i*phi), reproducible."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.rand((nz, n, n), generator=g).to(device) * 0.3
    return torch.exp(1j * phi).to(torch.complex64)


def _kernel(n, device):
    """A plausible Fresnel kernel: quadratic phase in q, unit modulus."""
    fy = torch.fft.fftfreq(n, device=device) * n
    q2 = (fy[:, None] ** 2 + fy[None, :] ** 2) / float(n * n)
    return torch.exp(-1j * torch.pi * 0.5 * q2).to(torch.complex64)


def _advance(sm, T, kernel, n_slices):
    from scatterem.physics.propagation import propagate

    for ncur in range(n_slices):
        sm.S = sm.S * T[ncur]
        sm._slice = ncur
        if ncur < n_slices - 1:
            sm.S = propagate(sm.S, kernel)


def _rel(a, b):
    return float((a - b).abs().pow(2).mean().sqrt() / b.abs().pow(2).mean().sqrt())


@pytest.mark.parametrize("focal_backprop", ["centroid", 0.0])
def test_partitioned_probe_matches_reconstructed_contraction(focal_backprop):
    from scatterem.simulation._wave.partitioned_smatrix import (
        PartitionedScatteringMatrix,
    )

    n, nz = 64, 8
    probe_q = _aperture_probe_q(n, radius_px=9.0, device=_DEV)
    T = _random_transmissions(nz, n, _DEV)
    kernel = _kernel(n, _DEV)
    sm = PartitionedScatteringMatrix(
        probe_q, T, kernel, n_radial=2, n_angular=6, focal_backprop=focal_backprop
    )
    assert sm.n_parents < int(sm._by.shape[0]), "fixture must actually partition"
    _advance(sm, T, kernel, nz)  # mid-stack state with _slice == nz-1

    rpos = torch.tensor([[17.0, 41.0], [3.0, 60.0]], dtype=torch.float64, device=_DEV)
    probes = sm.probe_at_current_plane(rpos)  # (2, n, n)

    cols = sm.reconstruct_columns()  # (B, n, n)
    coeffs = sm.coeffs_at(rpos)  # (2, B)
    ref = torch.einsum("pb,byx->pyx", coeffs.to(cols.dtype), cols)

    # ~3e-4 in fp32: the integer-roll synthesis and the einsum/FFT reconstruct
    # path are algebraically equal but accumulate differently in complex64.
    # Both focal_backprop variants miss by the same amount (no back-prop bug),
    # and 3e-4 in the probe is ~6e-4 in |probe|^2 -- far below the 2e-2
    # downstream EELS equivalence tolerances.
    assert _rel(probes, ref) < 5e-4


def test_pure_probe_matches_inline_einsum():
    from scatterem.simulation._wave.smatrix import ScatteringMatrix

    n, nz = 64, 8
    probe_q = _aperture_probe_q(n, radius_px=9.0, device=_DEV)
    T = _random_transmissions(nz, n, _DEV, seed=1)
    kernel = _kernel(n, _DEV)
    sm = ScatteringMatrix(probe_q, T, kernel)
    _advance(sm, T, kernel, nz)

    rpos = torch.tensor([[17.0, 41.0]], dtype=torch.float64, device=_DEV)
    probe = sm.probe_at_current_plane(rpos)[0]
    ref = torch.einsum("b,byx->yx", sm.coeffs_at(rpos)[0].to(sm.S.dtype), sm.S)
    assert torch.allclose(probe, ref, atol=1e-6, rtol=1e-5)


def test_integer_path_parent_chunking_matches_unchunked(monkeypatch):
    """Force a tiny parent chunk; the integer-scan synthesis must be unchanged.

    Guards the memory-bounding rewrite: chunked accumulation only reorders the
    parent sum, so the probe matches the one-shot computation to fp tolerance.
    """
    from scatterem.simulation._wave import partitioned_smatrix
    from scatterem.simulation._wave.partitioned_smatrix import (
        PartitionedScatteringMatrix,
    )

    n, nz = 64, 8
    probe_q = _aperture_probe_q(n, radius_px=9.0, device=_DEV)
    T = _random_transmissions(nz, n, _DEV, seed=2)
    kernel = _kernel(n, _DEV)

    def _make():
        sm = PartitionedScatteringMatrix(
            probe_q, T, kernel, n_radial=2, n_angular=6, focal_backprop="centroid"
        )
        _advance(sm, T, kernel, nz)
        return sm

    rpos = torch.tensor([[17.0, 41.0], [3.0, 60.0]], dtype=torch.float64, device=_DEV)

    monkeypatch.setattr(
        partitioned_smatrix, "_PROBE_PARENT_CHUNK_ELEMS", 1
    )  # 1 parent/chunk
    chunked = _make().probe_at_current_plane(rpos)
    monkeypatch.setattr(partitioned_smatrix, "_PROBE_PARENT_CHUNK_ELEMS", int(6e8))
    full = _make().probe_at_current_plane(rpos)

    assert _rel(chunked, full) < 1e-5


def _full_memory_reference(sm, rpos):
    """The removed sub-pixel branch: full ``(Bp, Ny, Nx)`` tensors, one iFFT
    per (position, parent) with the whole shift as a Fourier ramp."""
    n_half = sm._backprop_n_half()
    S_src = sm.S
    if n_half != 0.0:
        S_src = torch.fft.ifft2(
            torch.fft.fft2(sm.S, dim=(-2, -1)) * sm._fresnel_pow(n_half, -1.0)
        )
    S_dt = sm._detilted_S(S_src)[None]
    Bq = sm._beamlet_spectrum()
    ry = rpos[:, 0].view(-1, 1, 1)
    rx = rpos[:, 1].view(-1, 1, 1)
    ramp = torch.exp(
        -2j * torch.pi * (sm._fy[None] * ry / sm.ny + sm._fx[None] * rx / sm.nx)
    ).to(sm._dtype)
    basis = torch.fft.ifft2(Bq[None] * ramp[:, None]) * float(sm.ny * sm.nx)
    out = (S_dt * basis).sum(1)
    if n_half != 0.0:
        out = torch.fft.ifft2(torch.fft.fft2(out) * sm._fresnel_pow(n_half, 1.0))
    return out


def _peak(fn):
    """Peak CUDA bytes that ``fn()`` allocates above the current level."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def test_half_pixel_positions_take_the_chunked_path(monkeypatch):
    """A half-pixel map read (at u = 2) keeps the parent-chunk
    memory bound and matches the full-memory Fourier-shift form.
    The old route built 4-5 full ``(Bp, Ny, Nx)`` tensors.

    Peaks in units of one ``(Bp, Ny, Nx)`` complex64 tensor, measured on this
    fixture with four positions: the old route 11.4; this route 1.0 with
    one-parent chunks and 6.4 at the default budget (one chunk of all parents,
    where the de-tilt before the position loop sets the peak).
    """
    from scatterem.simulation._wave import partitioned_smatrix
    from scatterem.simulation._wave.partitioned_smatrix import (
        PartitionedScatteringMatrix,
    )

    n, nz = 64, 8
    probe_q = _aperture_probe_q(n, radius_px=9.0, device=_DEV)
    T = _random_transmissions(nz, n, _DEV, seed=3)
    kernel = _kernel(n, _DEV)
    sm = PartitionedScatteringMatrix(
        probe_q, T, kernel, n_radial=2, n_angular=6, focal_backprop="centroid"
    )
    _advance(sm, T, kernel, nz)
    rpos = torch.tensor(
        [[17.5, 41.0], [3.0, 60.5], [-2.5, 0.25], [8.0, 9.0]],
        dtype=torch.float64,
        device=_DEV,
    )
    ref = _full_memory_reference(sm, rpos)
    old_peak = _peak(lambda: _full_memory_reference(sm, rpos))
    full = sm.n_parents * n * n * 8  # one (Bp, Ny, Nx) complex64 tensor

    default_peak = _peak(lambda: sm.probe_at_current_plane(rpos))
    assert default_peak < 0.75 * old_peak, (default_peak / full, old_peak / full)

    monkeypatch.setattr(partitioned_smatrix, "_PROBE_PARENT_CHUNK_ELEMS", n * n)
    got = sm.probe_at_current_plane(rpos)
    peak = _peak(lambda: sm.probe_at_current_plane(rpos))
    assert _rel(got, ref) < 1e-5
    assert peak < 2 * full, f"peak {peak} B is not bounded by the chunk ({full} B)"
