"""Half-set spectral SNR, shared by direct ptychography and the dark field.

The radial SSNR of two independent half-set images
(:func:`compute_ssnr_from_halfset_images`), its memoised radial binning, and the
1D Gaussian smoothing applied to the radial curve. Used by the tilt-corrected
dark-field channel and by the empirical direct-ptychography SSNR; neither
imports the other.
"""

import torch

from scatterem.utils.lazy_plotting import plt


def gaussian_blur1d(x, sigma):
    """Apply 1D Gaussian blur to a 1D torch tensor.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor to blur.
    sigma : float
        Gaussian standard deviation, in samples. Values ``<= 0`` return `x`
        unchanged.

    Returns
    -------
    torch.Tensor
        The blurred 1D tensor, reflect-padded at the edges.
    """
    if sigma <= 0:
        return x
    import math

    # 3 sigma on each side + 1 center
    k = int(math.ceil(3 * sigma))
    ksize = 2 * k + 1
    device = x.device
    dtype = x.dtype
    t = torch.arange(ksize, device=device, dtype=dtype) - k
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # reflect padding currently requires at least 3D tensors, so pad after adding batch/channel dims
    x_3d = x.view(1, 1, -1)
    x_pad = torch.nn.functional.pad(x_3d, (k, k), mode="reflect")
    x_blur = torch.nn.functional.conv1d(x_pad, kernel[None, None, :], padding=0)[0, 0]
    return x_blur


#: Memo for :func:`_ssnr_radial_bins`.  Small (a handful of ``(radial_bins,)``
#: vectors plus one ``(ny*nx,)`` int64 index per distinct grid) and bounded below.
_SSNR_BIN_CACHE = {}

#: Maximum number of distinct grids kept in :data:`_SSNR_BIN_CACHE`.  A caller that
#: sweeps something OTHER than the grid (the depth section sweeps the aberrations)
#: hits one key forever; this only bounds a caller that walks upsample factors.
_SSNR_BIN_CACHE_MAX = 8


def _ssnr_radial_bins(ny, nx, sampling, radial_bins, dtype, device):
    """The radial binning geometry of :func:`compute_ssnr_from_halfset_images`.

    ``q_mag`` is built from ``fftfreq(ny, sampling[0])`` and ``fftfreq(nx,
    sampling[1])`` alone, so the bin index, the per-bin COUNT and the per-bin mean
    ``|q|`` depend only on ``(ny, nx, sampling, radial_bins)`` -- never on the two
    half-set images.  Every caller of ``compute_ssnr_from_halfset_images`` re-derives
    them on every call: ``tilt_corrected_dark_field`` once per reconstruction and
    ``tilt_corrected_dark_field_depth_section`` once per DEPTH PLANE, and the direct
    ptychography channel the same way, all on a path that is ~100 % host-bound at the
    small scan grids where the tcDF figures are made.

    Two further things fall out of hoisting them, and they are the reason this is one
    change and not three:

    * ``den_N`` and ``den_S`` were the SAME vector computed twice (an ``index_add_``
      of ``ones_like`` over the same ``bin_idx``), so one of the two full-size
      ``ones_like`` allocations was pure waste;
    * a bin COUNT does not need a weight array at all -- ``torch.bincount`` produces
      it exactly, with no ``(ny*nx,)`` temporary.  Summing 1.0 in any order is exact
      below 2**24, so this is bit-identical to the ``index_add_`` form it replaces.

    The returned tensors are CACHED and shared between calls: they are read-only by
    contract.  Both library call sites (this module and
    ``direct_ptychography._fused_full_field``) only index with ``bin_idx`` and
    compare against ``q_rad``.
    """
    key = (
        int(ny),
        int(nx),
        tuple(float(s) for s in sampling),
        int(radial_bins),
        dtype,
        str(device),
    )
    hit = _SSNR_BIN_CACHE.get(key)
    if hit is not None:
        return hit

    qy = torch.fft.fftfreq(ny, d=sampling[0]).to(device)
    qx = torch.fft.fftfreq(nx, d=sampling[1]).to(device)
    QY, QX = torch.meshgrid(qy, qx, indexing="ij")
    q_mag = torch.sqrt(QX**2 + QY**2)

    flat_q = q_mag.flatten()
    q_min, q_max = flat_q.min(), flat_q.max()
    bin_edges = torch.linspace(
        q_min, q_max, steps=radial_bins + 1, device=flat_q.device
    )
    bin_idx = torch.bucketize(flat_q, bin_edges) - 1
    bin_idx = bin_idx.clamp(min=0, max=radial_bins - 1)

    den_safe = torch.bincount(bin_idx, minlength=radial_bins).to(dtype).clamp_min(1.0)

    qnum = torch.zeros(radial_bins, device=flat_q.device, dtype=flat_q.dtype)
    qnum.index_add_(0, bin_idx, flat_q)
    q_rad = qnum / den_safe

    if len(_SSNR_BIN_CACHE) >= _SSNR_BIN_CACHE_MAX:
        _SSNR_BIN_CACHE.clear()
    out = (bin_idx, den_safe, q_rad)
    _SSNR_BIN_CACHE[key] = out
    return out


def compute_ssnr_from_halfset_images(
    tcDF1, tcDF2, sampling, gaussian_sigma=0.0, verbosity=0
):
    """Estimate the radial spectral SNR from two independent half-set images.

    Treats ``tcDF1``/``tcDF2`` as two independent noise realizations of the
    same signal (``N = (FA - FB) / 2`` estimates the noise spectrum,
    ``S = (FA + FB) / 2`` the signal-plus-noise spectrum), radially bins their
    power in Fourier space, and forms ``SSNR(q) = (S_power - N_power) / N_power``.

    Parameters
    ----------
    tcDF1 : torch.Tensor
        First half-set real-space image, shape ``(H, W)``.
    tcDF2 : torch.Tensor
        Second half-set real-space image, same shape as `tcDF1`.
    sampling : tuple of float
        Real-space pixel sampling ``(dy, dx)``, used to build the Fourier
        frequency grid.
    gaussian_sigma : float, optional
        If greater than 0, 1D-Gaussian-smooth the radial SSNR curve with this
        sigma, in radial bins.
    verbosity : int, optional
        If greater than 1, plot the radial noise power, signal power and SSNR
        curves.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        The radial SSNR curve, the per-bin mean spatial frequency ``|q|``, and
        the flattened per-pixel radial bin index (for scattering the radial
        curve back onto the 2D grid).
    """
    FA = torch.fft.fftn(tcDF1, dim=(-2, -1), norm="ortho")
    FB = torch.fft.fftn(tcDF2, dim=(-2, -1), norm="ortho")

    N = 0.5 * (FA - FB)
    S = 0.5 * (FA + FB)

    power_N = N.abs() ** 2  # |N|^2
    power_S = S.abs() ** 2  # |S|^2

    ny, nx = N.shape[-2:]
    flat_N = power_N.flatten()
    flat_S = power_S.flatten()

    radial_bins = tcDF1.shape[0] // 2
    bin_idx, den_safe, q_rad = _ssnr_radial_bins(
        ny, nx, sampling, radial_bins, flat_N.dtype, flat_N.device
    )
    den_safe_N = den_safe

    # Noise power <|N|^2>
    num_N = torch.zeros(radial_bins, device=flat_N.device, dtype=flat_N.dtype)
    num_N.index_add_(0, bin_idx, flat_N)
    VarN_radial = num_N / den_safe_N

    # Total power <|S|^2>.  ``den_S`` was the identical bin count -- same ``bin_idx``,
    # same all-ones weights, same length -- so it is ``den_safe_N``.
    num_S = torch.zeros(radial_bins, device=flat_S.device, dtype=flat_S.dtype)
    num_S.index_add_(0, bin_idx, flat_S)
    S_power_radial = num_S / den_safe_N

    # SSNR(q) ≈ (signal power) / (noise power)
    signal_power_radial = (S_power_radial - VarN_radial).clamp_min(0.0)
    SSNRq = signal_power_radial / VarN_radial.clamp_min(1e-20)
    if gaussian_sigma > 0:
        SSNRq = gaussian_blur1d(SSNRq, gaussian_sigma)
    if verbosity > 1:
        plt.figure()
        plt.semilogy(
            q_rad.cpu().numpy(), VarN_radial.cpu().numpy(), label="|N| radial avg"
        )
        plt.semilogy(
            q_rad.cpu().numpy(),
            signal_power_radial.cpu().numpy(),
            label="|S| radial avg",
        )
        plt.semilogy(q_rad.cpu().numpy(), SSNRq.cpu().numpy(), label="SSNRq radial avg")
        plt.xlabel("Spatial frequency |q|")
        plt.ylabel("Radial mean (|N|, |S|)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    return SSNRq, q_rad, bin_idx
