from typing import List, Tuple

import numpy as np
import torch
from ase import units
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay
from torch import Tensor
from numpy.fft import fftfreq
from metpy.interpolate import geometry
from metpy.interpolate.points import natural_neighbor_point
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import euclidean
from tqdm import trange

def regularization_kernel_fourierspace(
    kernel_shape: ArrayLike,
    d: List[float],
    beta: float,
    alpha: float,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate multi-slice regularization kernel in Fourier space.

    Parameters
    ----------
    kernel_shape : array-like
        Shape of the kernel volume (z, y, x).
    d : list
        Sampling intervals in each dimension [dz, dy, dx].
    beta : float
        Regularization parameter controlling anisotropy strength along z.
    alpha : float
        Regularization parameter controlling smoothing strength along xy.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Fourier space regularization kernel.
    """
    q = fftfreq3(kernel_shape, d, device=device)
    W = 1 - torch.atan(
        (beta * torch.abs(q[0]) / torch.sqrt(q[1] ** 2 + q[2] ** 2 + 1e-3)) ** 2
    ) / (torch.pi / 2)
    Wa = W * torch.exp(-alpha * (q[1] ** 2 + q[2] ** 2))
    return Wa


def regularization_kernel_realspace(
    volume_shape: ArrayLike,
    d: List[float] = [1.0, 1.0, 1.0],
    beta: float = 1,
    alpha: float = 1,
    thresh: float = 0.03,
    device: str = "cuda:0",
) -> Tensor:
    """
    Calculate multi-slice regularization kernel in real space.

    Parameters
    ----------
    volume_shape : array-like
        Shape of the volume (z, y, x).
    d : list, optional
        Sampling intervals in each dimension [dz, dy, dx]. Default is [1.0, 1.0, 1.0].
    beta : float, optional
        Regularization parameter controlling anisotropy strength along z. Default is 1.
    alpha : float, optional
        Regularization parameter controlling smoothing strength along xy. Default is 1.
    thresh : float, optional
        Threshold for determining kernel size as fraction of maximum value. Default is 0.03.
    device : str, optional
        PyTorch device to use. Default is 'cuda:0'.

    Returns
    -------
    torch.Tensor
        Real space regularization kernel with odd dimensions, normalized to sum to 1.
    """
    volume_shape = np.array([v + (1 - v % 2) for v in volume_shape])
    kernel_fourierspace = regularization_kernel_fourierspace(
        volume_shape, d, beta, alpha, device
    )
    psf = torch.zeros(tuple(volume_shape), device=device)
    psf[0, 0, 0] = 1
    kernel_vol = torch.fft.ifftn(
        kernel_fourierspace * torch.fft.fftn(psf, norm="ortho")
    )

    X = torch.fft.fftshift(kernel_vol.real)

    center = np.fix(volume_shape / 2).astype(int)

    # Create projections by summing along each axis
    proj_xy = torch.sum(X, dim=0).cpu()  # Sum along x axis
    # proj_xz = torch.sum(X, dim=1).cpu()  # Sum along y axis
    proj_yz = torch.sum(X, dim=2).cpu()  # Sum along z axis

    mask_yz = proj_yz > thresh * proj_yz.max()
    # mask_xz = proj_xz > thresh * proj_xz.max()
    mask_xy = proj_xy > thresh * proj_xy.max()
    z_len_proj = mask_yz.to(torch.int32).sum(1)
    y_len_proj = mask_xy.to(torch.int32).sum(1)
    x_len_proj = mask_xy.to(torch.int32).sum(0)

    def get_nonzero_length(proj: np.ndarray) -> int:
        inds = np.arange(len(proj))
        nonzero_inds = inds[proj > 0]
        length = np.max(nonzero_inds) - np.min(nonzero_inds)
        return length + 1

    z_length = get_nonzero_length(z_len_proj)
    y_length = get_nonzero_length(y_len_proj)
    x_length = get_nonzero_length(x_len_proj)

    # Force z_length to be odd by adding 1 if even
    z_length = z_length + (1 - z_length % 2)
    y_length = y_length + (1 - y_length % 2)
    x_length = x_length + (1 - x_length % 2)

    kernel = X[
        center[0] - z_length // 2 : center[0] + z_length // 2 + 1,
        center[1] - y_length // 2 : center[1] + y_length // 2 + 1,
        center[2] - x_length // 2 : center[2] + x_length // 2 + 1,
    ].clone()
    kernel /= kernel.sum()
    return kernel


def circular_aperture(
    r: float, shape: Tuple[int, int], device: torch.device = torch.device("cuda")
) -> Tensor:
    """Create a circular aperture with anti-aliased edge.

    Parameters
    ----------
    r : float
        Radius of aperture in pixels
    shape : tuple
        Shape of output array (height, width)
    device : str
        Device to place tensor on ("cuda" or "cpu")

    Returns
    -------
    torch.Tensor
        2D tensor containing circular aperture with smooth edges
    """
    y = torch.arange(-shape[0] // 2, shape[0] // 2, device=device)
    x = torch.arange(-shape[1] // 2, shape[1] // 2, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    dist = torch.sqrt(X * X + Y * Y)

    # Anti-aliasing width (in pixels)
    aa_width = 1.0

    # Smooth transition using tanh
    mask = 0.5 * (1 - torch.tanh((dist - r) / (aa_width / 2)))
    mask = torch.fft.fftshift(mask)
    return mask


def beamlet_samples(A1, radius, n_angular_samples, n_radial_samples):
    """
    Determines which beams of the scattering matrix should be sampled, given an illumination aperture A,
    an aperture radius, the number of angular samples, and the number of radial samples.
    Assumes aperture is centered at (0,0).

    :param A:
    :param radius:
    :param n_angular_samples:
    :param n_radial_samples:
    :return: beamlet samples, (Bs,2)
    """
    A = np.fft.fftshift(A1)
    M = A.shape[0]

    my1 = np.tile(fftfreq(M, d=1 / M)[:, None], M).astype(np.int32)
    mx1 = np.repeat(fftfreq(M, d=1 / M)[:, None].T, M, axis=0).astype(np.int32)

    B = np.nonzero(A)
    beam_coords1 = np.array([my1[B], mx1[B]]).T
    beam_coords2 = beam_coords1 + 1e-2

    a_offset = np.pi / n_angular_samples
    my1 = np.tile(fftfreq(M, d=1 / M)[:, None], M).astype(np.int32)
    mx1 = np.repeat(fftfreq(M, d=1 / M)[:, None].T, M, axis=0).astype(np.int32)
    beam_coords1 = np.array([my1[B], mx1[B]]).T
    radial_samples = np.linspace(0, radius, n_radial_samples, endpoint=True)[1:]
    samples = [[0, 0]]
    for i, r in enumerate(radial_samples):
        angular_samples = np.linspace(-np.pi, np.pi, n_angular_samples * (1 + i), endpoint=False)
        # print(i, len(angular_samples))
        for a in angular_samples:
            # print(r, a)
            # print([r * np.sin(a + a_offset * i), r * np.cos(a + a_offset * i)])
            samples.append([r * np.sin(a + a_offset * i), r * np.cos(a + a_offset * i)])
    samples = np.array(samples)
    xy_diff = np.linalg.norm(samples[None, ...] - beam_coords2[:, None, :], axis=2)
    r_min_indices = np.argmin(xy_diff, axis=0)
    best_beam_coords = beam_coords1[r_min_indices]
    return best_beam_coords


def natural_neighbor_weights_internal(xp, yp, grid_loc, tri, neighbors, circumcenters, minimum_weight_cutoff=1e-2):
    r"""Generate a natural neighbor interpolation of the observations to the given point.

    This uses the Liang and Hale approach [Liang2010]_. The interpolation will fail if
    the grid point has no natural neighbors.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_loc: (float, float)
        Coordinates of the grid point at which to calculate the
        interpolation.
    tri: `scipy.spatial.Delaunay`
        Delaunay triangulation of the observations.
    neighbors: (N, ) ndarray
        Simplex codes of the grid point's natural neighbors. The codes
        will correspond to codes in the triangulation.
    circumcenters: list
        Pre-calculated triangle circumcenters for quick look ups. Requires
        indices for the list to match the simplices from the Delaunay triangulation.
    minimum_weight_cutoff:
        delete weights smaller than minimum_weight_cutoff
    Returns
    -------
    weights: (W,) ndarray
       weights for interpolating the grid location
    point_indices  (W,) ndarray
        integer indices of the points used for interpolation
    """
    edges = geometry.find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in geometry.order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    c1 = geometry.circumcenter(grid_loc, tri.points[p1], tri.points[p2])
    polygon = [c1]

    area_list = []
    pt_list = []
    total_area = 0.0
    indices = np.arange(xp.shape[0])
    for i in range(num_vertices):

        p3 = edge_vertices[(i + 2) % num_vertices]

        try:
            c2 = geometry.circumcenter(grid_loc, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(circumcenters[check_tri])
            # print('polygon',polygon)
            pts1 = [polygon[i] for i in ConvexHull(polygon).vertices]
            pt_mask = (tri.points[p2][0] == xp) & (tri.points[p2][1] == yp)
            pt_list.append(indices[pt_mask][0])
            cur_area = geometry.area(pts1)

            total_area += cur_area

            area_list.append(cur_area)

        except (ZeroDivisionError) as e:
            message = ('Error during processing of a grid. '
                       'Interpolation will continue but be mindful '
                       f'of errors in output. {e}')

            print(message)
            return np.nan

        polygon = [c2]

        p2 = p3

    weights = np.array([x / total_area for x in area_list])
    pt_list = np.array(pt_list)

    weights1 = weights[weights > minimum_weight_cutoff]
    pt_list1 = pt_list[weights > minimum_weight_cutoff]

    weights1 *= 1 / np.sum(weights1)

    return weights1, pt_list1


def natural_neighbor_weights(known_points, interp_points, minimum_weight_cutoff=1e-2):
    r"""Generate a natural neighbor interpolation of the observations to the given points.

    This uses the Liang and Hale approach [Liang2010]_. The interpolation will fail if
    the grid point has no natural neighbors.

    Parameters
    ----------
    known_points: (B_S, 2) ndarray
        x,y-coordinates of observations
    interp_points: (B, 2) ndarray
        Coordinates of the grid point at which to calculate the
        interpolation.
    minimum_weight_cutoff:
        delete weights smaller than minimum_weight_cutoff, default 1e-2
    Returns
    -------
    weights: (B,B_S) ndarray
       weights for interpolating the grid location from the B_S sampled points
    """
    interp_points = interp_points.copy() - 1e-3
    weights = np.zeros((interp_points.shape[0], known_points.shape[0]))

    tri = Delaunay(known_points)

    # members is dict with length B

    # print('interp_points', interp_points)
    for b in trange(interp_points.shape[0]):
        interp_point = interp_points[b]
        interp_point2 = interp_points[[b]]
        radius_subtracted = 0
        while radius_subtracted < 10:
            # print()
            # print('radius_subtracted',radius_subtracted)
            r = np.linalg.norm(interp_point)
            phi = np.arctan2(interp_point[0], interp_point[1])
            r_new = r - radius_subtracted
            interp_point_new = np.array([r_new * np.sin(phi), r_new * np.cos(phi)])
            interp_point2[:] = interp_point_new
            try:
                # print('interp_point_new', interp_point_new)
                # print('interp_point', interp_point)
                members, circumcenters = geometry.find_natural_neighbors(tri, interp_point2)
                neighbors = members[0]
                # print('members', members)
                # print('b', b)
                # print('neighbors', neighbors)
                # print('known_points[:, 0]', known_points[:, 0])
                # print('known_points[:, 1]', known_points[:, 1])
                #
                # print('circumcenters', circumcenters)
                weights_ind, indices_ind = natural_neighbor_weights_internal(known_points[:, 0], known_points[:, 1],
                                                                             interp_point_new, tri, neighbors,
                                                                             circumcenters,
                                                                             minimum_weight_cutoff=minimum_weight_cutoff)
                break
            except:
                radius_subtracted += 0.1

        weights[b, indices_ind] = weights_ind
    return weights


def fftfreq2(
    N: ArrayLike,
    dx: List[float] = [1.0, 1.0],
    centered: bool = False,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate 2D Fourier frequencies for a given grid size and sampling intervals.

    Parameters
    ----------
    N : array-like
        Grid dimensions (y, x).
    dx : list, optional
        Sampling intervals in each dimension [dy, dx]. Default is [1.0, 1.0].
    centered : bool, optional
        If True, shift frequencies by half a pixel. Default is False.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    Tensor
        Stacked frequency coordinates with shape (2, Ny, Nx).
        First dimension contains qy and qx components.
    """
    qxx = torch.fft.fftfreq(N[1], dx[1], device=device)
    qyy = torch.fft.fftfreq(N[0], dx[0], device=device)
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qy, qx = torch.meshgrid(qyy, qxx, indexing="ij")
    qy, qx = torch.meshgrid(qyy, qxx, indexing="ij")
    q = torch.stack([qy, qx], dim=0)
    return q


def fftfreq3(
    N: ArrayLike,
    dx: List[float] = [1.0, 1.0, 1.0],
    centered: bool = False,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate 3D Fourier frequencies for a given grid size and sampling intervals.

    Parameters
    ----------
    N : array-like
        Grid dimensions (z, y, x).
    dx : list, optional
        Sampling intervals in each dimension [dz, dy, dx]. Default is [1.0, 1.0, 1.0].
    centered : bool, optional
        If True, shift frequencies by half a pixel. Default is False.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    Tensor
        Stacked frequency coordinates with shape (3, Nz, Ny, Nx).
        First dimension contains qz, qy and qx components.
    """
    qxx = torch.fft.fftfreq(N[1], dx[2], device=device)
    qyy = torch.fft.fftfreq(N[1], dx[1], device=device)
    qzz = torch.fft.fftfreq(N[0], dx[0], device=device)
    if centered:
        qxx += 0.5 / N[2] / dx[2]
        qyy += 0.5 / N[1] / dx[1]
        qzz += 0.5 / N[0] / dx[0]
    qz, qy, qx = torch.meshgrid(qzz, qyy, qxx, indexing="ij")
    q = torch.stack([qz, qy, qx], dim=0)
    return q


def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]̄
    """

    return relativistic_mass_correction(energy) * units._me


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """

    return (
        units._hplanck
        * units._c
        / np.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [ev].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """

    return (
        2
        * np.pi
        * energy2mass(energy)
        * units.kg
        * units._e
        * units.C
        * energy2wavelength(energy)
        / (units._hplanck * units.s * units.J) ** 2
    )


def fftshift_checkerboard(w: int, h: int) -> np.ndarray:
    re = np.r_[w // 2 * [-1, 1]]  # even-numbered rows
    ro = np.r_[w // 2 * [1, -1]]  # odd-numbered rows
    return np.row_stack(h // 2 * (re, ro))


def probe_radius_and_center(
    DP: Tensor, thresh_lower: float = 0.01, thresh_upper: float = 0.99, N: int = 100
) -> Tuple[float, NDArray]:
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=DP.device)
    r_vals = torch.zeros(N, device=DP.device)

    DPmax = torch.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = torch.sqrt(torch.sum(mask) / torch.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = torch.gradient(r_vals, dim=0)[0]
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * torch.median(dr_dtheta))
    r = torch.mean(r_vals[mask])

    # Get origin
    thresh = torch.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    ar = DP * mask
    nx, ny = ar.shape
    ry, rx = torch.meshgrid(
        torch.arange(ny, device=DP.device), torch.arange(nx, device=DP.device)
    )
    tot_intens = torch.sum(ar)
    x0 = torch.sum(rx * ar) / tot_intens
    y0 = torch.sum(ry * ar) / tot_intens

    return float(r), np.array([y0.item(), x0.item()])


def get_probe_size(
    DP: ArrayLike, thresh_lower: float = 0.01, thresh_upper: float = 0.99, N: int = 100
) -> Tuple[float, float, float]:
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    ar = DP * mask
    nx, ny = np.shape(ar)
    ry, rx = np.meshgrid(np.arange(ny), np.arange(nx))
    tot_intens = np.sum(ar)
    x0 = np.sum(rx * ar) / tot_intens
    y0 = np.sum(ry * ar) / tot_intens

    return r, x0, y0


def advanced_raster_scan(
    ny: int = 10,
    nx: int = 10,
    fast_axis: int = 1,
    mirror: List[int] = [1, 1],
    theta: float = 0,
    dy: float = 1,
    dx: float = 1,
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> Tensor:
    """
    Generates a raster scan.

    Parameters
    ----------
    ny, nx : int
        Number of steps in *y* (vertical) and *x* (horizontal) direction
    fast_axis : int
        Which axis is the fast axis (1 for x, 0 for y)
    mirror : List[int]
        Mirror factors for x and y axes
    theta : float
        Rotation angle in degrees
    dy, dx : float
        Step size (grid spacing) in *y* and *x*
    device : torch.device
        Device for the output tensor

    Returns
    -------
    pos : Tensor
        A (N,2)-tensor of positions.
    """
    iiy, iix = np.indices((ny, nx), dtype=np.float32)

    if fast_axis != 1:
        iix, iiy = iiy, iix

    # Create positions directly without list comprehension
    positions = np.stack([dy * iiy.ravel(), dx * iix.ravel()], axis=1)

    # Center the positions
    center = positions.mean(axis=0)
    positions -= center

    # Apply mirroring
    positions[:, 0] *= mirror[0]
    positions[:, 1] *= mirror[1]

    # Apply rotation
    if theta != 0:
        theta_rad = np.radians(theta)
        R = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)],
            ]
        )
        positions = positions @ R

    # Translate to start from origin
    positions -= positions.min(axis=0)

    return torch.as_tensor(positions, device=device, dtype=dtype)


def batch_unique_with_inverse(
    positions: Tensor, patch_shape: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
    """
    :param positions: Each row is ``(y, x)`` for the top-left corner of a patch.
    :type positions: Tensor of shape ``(N, 2)``, dtype=torch.long
    :param patch_shape: The patch dimensions ``(height, width)``.
    :type patch_shape: tuple(int, int)

    :returns:
        - **unique_pts** (*Tensor, shape ``(M, 2)``, dtype=torch.long*) –
          All unique ``(y, x)`` points from all patches.
        - **inverse_indices** (*Tensor, shape ``(N, height, width)``, dtype=torch.long*) –
          Maps each patch pixel to the row index in ``unique_pts``.
    """
    # Unpack the patch shape
    ph, pw = patch_shape

    # If positions is not long, convert it
    if positions.dtype != torch.int32:
        positions = positions.int()

    # 1) Construct a grid of size (patch_height, patch_width) for the patch
    ys = torch.arange(
        ph, dtype=torch.int32, device=positions.device
    )  # [0, 1, ..., ph-1]
    xs = torch.arange(
        pw, dtype=torch.int32, device=positions.device
    )  # [0, 1, ..., pw-1]
    # meshgrid -> shape (ph, pw)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # stack to get shape (ph, pw, 2)
    patch_grid = torch.stack((grid_y, grid_x), dim=-1)

    # 2) Expand to (N, ph, pw, 2) by adding top-left offsets
    # positions: shape (N,2) -> (N,1,1,2)
    # patch_grid: shape (ph,pw,2) -> (1,ph,pw,2)
    # broadcast sum -> (N, ph, pw, 2)
    all_points = patch_grid.unsqueeze(0) + positions.unsqueeze(1).unsqueeze(1)

    # 3) Flatten to shape (N*ph*pw, 2)
    flat_points = all_points.view(-1, 2)

    # 4) Use torch.unique to get unique points (M,2) and inverse indices
    unique_pts, inverse = torch.unique(flat_points, return_inverse=True, dim=0)

    # 5) Reshape inverse to (N, ph, pw)
    inverse_indices = inverse.view(positions.shape[0], ph, pw)

    return unique_pts, inverse_indices


def _cartesian_aberrations(qy: Tensor, qx: Tensor, lam: Tensor, C: Tensor) -> Tensor:
    """
    Zernike polynomials in the cartesian coordinate system
    """

    u = qx * lam
    v = qy * lam
    u2 = u**2
    u3 = u**3
    u4 = u**4
    # u5 = u ** 5

    v2 = v**2
    v3 = v**3
    v4 = v**4
    # v5 = v ** 5

    C1 = C[0]
    C12a = C[1]
    C12b = C[2]
    C21a = C[3]
    C21b = C[4]
    C23a = C[5]
    C23b = C[6]
    C3 = C[7]
    C32a = C[8]
    C32b = C[9]
    C34a = C[10]
    C34b = C[11]

    chi = 0

    # r-2 = x-2 +y-2.
    chi += 1 / 2 * C1 * (u2 + v2)  # r^2
    # r-2 cos(2*phi) = x"2 -y-2.
    # r-2 sin(2*phi) = 2*x*y.
    chi += (
        1 / 2 * (C12a * (u2 - v2) + 2 * C12b * u * v)
    )  # r^2 cos(2 phi) + r^2 sin(2 phi)
    # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
    chi += (
        1 / 3 * (C23a * (u3 - 3 * u * v2) + C23b * (3 * u2 * v - v3))
    )  # r^3 cos(3phi) + r^3 sin(3 phi)
    # r-3 cos(phi) = x-3 +x*y-2.
    # r-3 sin(phi) = y*x-2 +y-3.
    chi += (
        1 / 3 * (C21a * (u3 + u * v2) + C21b * (v3 + u2 * v))
    )  # r^3 cos(phi) + r^3 sin(phi)
    # r-4 = x-4 +2*x-2*y-2 +y-4.
    chi += 1 / 4 * C3 * (u4 + v4 + 2 * u2 * v2)  # r^4
    # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
    chi += 1 / 4 * C34a * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
    # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
    chi += 1 / 4 * C34b * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
    # r-4 cos(2*phi) = x-4 -y-4.
    chi += 1 / 4 * C32a * (u4 - v4)
    # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
    chi += 1 / 4 * C32b * (2 * u3 * v + 2 * u * v3)
    # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
    # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
    # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
    # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
    # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
    # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

    chi *= 2 * torch.pi / lam

    return chi