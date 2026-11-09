"""PrismScanOp: scan-time exit-wave operator using a (partitioned) S-matrix.

Covers both pure PRISM (Ophus 2017) and partitioned PRISM (Pelz 2021).
The two cases are distinguished by the SMeta passed at construction:

* Pure PRISM:           SMeta with parent_beams == all_beams (Bp == B);
                        beamlet weights become the identity.
* Partitioned PRISM:    SMeta with parent_beams subset of all_beams
                        (Bp < B); SMeta carries natural-neighbor
                        weights and beamlets.

The propagator passed to ``forward`` MUST be sized for the full simulation
grid ``s_meta.M`` (i.e. ``propagator.shape == s_meta.M``). Callers that
hold a propagator built for a smaller detector window must construct a
new one (e.g. via ``Propagator(slice_thickness=..., wavelength=...,
dx=..., shape=s_meta.M, device=...)``) before calling PRISM.
"""

import os

import torch
from torch import Tensor

from scatterem.data import SMeta
from scatterem.nn.functional.warp.prism_project import prism_project
from scatterem.nn.functional.warp.prism_project._basis import BeamletBasis
from scatterem.nn.modules import Propagator, _bcast_cmul_
from scatterem.nn.scan_exit_wave._columns import (
    plane_transfer,
    prepare_columns_,
    prepare_columns_autograd,
    reference_slices,
    vacuum_phase,
)
from scatterem.nn.scan_exit_wave._prism_parent_waves import (
    check_propagator_grid,
    initial_parent_waves,
)
from scatterem.nn.scan_exit_wave._prism_s_build_graph import (
    s_build_l2_blocked,
    s_build_l2_generated,
)
from scatterem.nn.scan_exit_wave.base import ScanExitWaveOperator
from scatterem.physics.precision import resolve_precision

_PREPARE_BYTES = 1 << 30  #: bound on the in-place column preparation's transient


class PrismScanOp(ScanExitWaveOperator):
    """Scan-time exit-wave operator that projects a partitioned S-matrix.

    Constructs an S-matrix on a set of parent beams (a subset of the aperture
    beams) and projects to each scan position via beamlet-weighted
    natural-neighbor sums.

    Parameters
    ----------
    s_meta : SMeta
        Beam/beamlet metadata (parent beams, weights, simulation grid).
    f : int or tuple, optional
        Anisotropic interpolation factor ``(f_y, f_x)``; a scalar broadcasts
        to both axes.
    freeze_s_matrix : bool, optional
        If True, detach the transmission before building the S-matrix.
    scan_chunk : int or None, optional
        Chunk size for ``prism_project`` over scan positions.
    periodic : bool, optional
        Whether ``prism_project`` treats the S-matrix as periodic.
    precision_config : dict or str or None, optional
        Precision preset resolved by :func:`resolve_precision`.
    focal_backprop : float or str, optional
        Fraction of the thickness (0 = exit, 1 = entrance) to back-propagate
        parent columns to before NNW fusion, or ``"centroid"`` to
        auto-select the scattering centroid.
    use_warp : bool or None, optional
        Whether to use the Warp kernel for scan synthesis; None follows the
        ``SCATTEREM_USE_WARP`` environment variable.
    build_cols : int or None, optional
        Parent beams propagated at once in the S-matrix build (>= 1), capping
        its ``(cols, MY, MX)`` transients; None is one pass when the stack
        fits ~6e8 complex elements. ``SCATTEREM_PRISM_BUILD_COLS`` overrides.
    """

    @property
    def wants_full_fov(self) -> bool:
        """Whether ``T`` must be the full ``(NZ_bin, NY, NX)`` stack; always True."""
        return True

    def __init__(
        self,
        s_meta: SMeta,
        f: "int | tuple" = 1,
        freeze_s_matrix: bool = False,
        scan_chunk: int | None = None,
        periodic: bool = True,
        precision_config: "dict | str | None" = None,
        focal_backprop: "float | str" = "centroid",
        use_warp: "bool | None" = None,
        build_cols: "int | None" = None,
    ) -> None:
        super().__init__()
        self.s_meta = s_meta
        # Beam chunk of the S-matrix multislice in ``_build_s_matrix``. ``None``
        # (the default) keeps the historical rule -- single pass whenever the
        # whole (Bp, MY, MX) stack fits ~6e8 complex elements -- so PRISM stays
        # byte-identical; a positive int caps the beams in flight and thereby
        # the build's (cols, MY, MX) transients (2-3 of them).  The chunked
        # branch does the same per-plane arithmetic in the same order, so the
        # columns match the single pass to float32 round-off.  The
        # ``SCATTEREM_PRISM_BUILD_COLS`` env override (a test hook) takes
        # precedence over both.
        self.build_cols = None if build_cols is None else int(build_cols)
        if self.build_cols is not None and self.build_cols < 1:
            raise ValueError(f"build_cols must be >= 1, got {build_cols}")
        # Warp vs torch for the prism_project scan synthesis. None -> the
        # SCATTEREM_USE_WARP env (historical default); True/False overrides. The
        # Warp path is fp32-exact vs the torch oracle (~4e-5) and ~1.3x faster.
        self.use_warp = use_warp
        # Anisotropic interpolation factor (f_y, f_x); a scalar broadcasts. Long-
        # narrow specimens use f_y != f_x so each axis gets N_axis = f_axis*pixels.
        self.fy, self.fx = (
            (int(f), int(f)) if isinstance(f, int) else (int(f[0]), int(f[1]))
        )
        # Focal back-propagation (Pelz 2021): the parent columns are propagated
        # to a reference plane before the partitioned-PRISM NNW fusion, and the
        # synthesized wave forward again after it (``_to_exit_plane``). The
        # column preparation (``_columns.prepare_columns_``) also strips each
        # column's free-space phase from that plane to
        # the exit and the basis restores it per aperture beam, so what the
        # fusion interpolates is the specimen's action alone: exact in vacuum
        # for EVERY plane, and the plane only trades off the beam walk
        # ``2 h_p . g (z_ref - z_scatter)`` of the scattered part.
        #
        # A float fixes the fraction of the thickness (0.0 = exit, 0.5 = centre,
        # 1.0 = entrance), applied as ``bp_frac * (NZ_bin - 1)`` slices, which
        # is the distance the exit wave has actually propagated. ``"centroid"``
        # (the default) is the least-squares minimiser of that beam walk, the
        # SCATTERING CENTROID of the potential's z-distribution; see
        # ``_columns.reference_slices`` for its definition in vacuum.
        #
        # Measured with the vacuum phase stripped (fcc Au, 200 keV, 20 mrad,
        # 0.255 A pixels, probe on a column, rel. L2 of the CBED against
        # multislice):
        #
        #     t /A   r    frac=0    0.25     0.5      0.75     1.0    centroid
        #      8     4    0.0288   0.0288   0.0287   0.0287   0.0287   0.0287
        #     16     4    0.0790   0.0779   0.0772   0.0766   0.0763   0.0772
        #     32     4    0.0905   0.0743   0.0699   0.0762   0.0942   0.0699
        #     32     8    0.0466   0.0439   0.0430   0.0436   0.0460   0.0430
        #
        # Flat for a thin specimen, a bowl centred on the centroid once it is
        # thick enough to matter. An older table here put the optimum at the
        # entrance / 0.75 and the centroid 1.3-5x behind it: that was measured
        # with the vacuum phase still IN the columns, where moving the plane to
        # the entrance shrinks the phase the interpolation has to carry and so
        # wins for a reason that has nothing to do with the specimen.
        #
        # Only takes effect when partitioning (Bp < B); pure PRISM interpolates
        # nothing and keeps the exit plane. See ``forward``.
        self.focal_backprop = focal_backprop
        if self.fy < 1 or self.fx < 1:
            raise ValueError(f"f must be >= 1 per axis, got {f}")
        MY, MX = int(s_meta.M[0]), int(s_meta.M[1])
        if MY % (2 * self.fy) != 0 or MX % (2 * self.fx) != 0:
            raise ValueError(
                f"f-cropping requires 2*(fy,fx) to divide (MY, MX); got "
                f"M=({MY}, {MX}) and (fy,fx)=({self.fy}, {self.fx}), so "
                f"2*(fy,fx)=({2 * self.fy}, {2 * self.fx}). Adjust f or the grid so "
                f"that 2*fy | MY and 2*fx | MX."
            )
        self.freeze_s_matrix = freeze_s_matrix
        self.scan_chunk = scan_chunk
        self.periodic = bool(periodic)
        self.precision_config = resolve_precision(precision_config)
        # ---- Forward-chunk S-matrix cache (optional, default OFF) ------------
        # The forward-sim driver chunks the scan over positions and calls
        # ``forward`` once per chunk; the S-matrix (S_dt) and beamlet basis_fft
        # are POSITION-INDEPENDENT, so with a fixed potential (displacements=False
        # within one config) every chunk rebuilds bit-identical tensors. When a
        # caller wraps a config's chunk loop with ``enable_forward_cache`` the
        # first chunk builds them and the rest REUSE them -- collapsing
        # ceil(n_pos / fwd_chunk) redundant multislice S-builds to ONE. Gated to
        # the forward-sim path (nothing enables it on the autograd /
        # reconstruction path), so that path stays byte-identical.
        self._fwd_cache_enabled: bool = False
        self._fwd_cache: tuple[Tensor, BeamletBasis] | None = None

    # ---------------------------------------------------------------- helpers
    def _build_dtype(self) -> torch.dtype:
        """Complex dtype for the multislice S-matrix build.

        ``build='fp16'`` runs the whole multislice wave propagation (parent
        waves + per-slice transmit + Fresnel propagate) in ``complex32`` (two
        ``float16`` per element).  Every other build (``'fp32'``, the default)
        stays ``complex64``, so the fp32 / reconstruction path is byte-identical.
        """
        return (
            torch.complex32
            if self.precision_config["build"] == "fp16"
            else torch.complex64
        )



    # --------------------------------------------- forward-chunk S-matrix cache
    def enable_forward_cache(self) -> None:
        """Enable reuse of the position-independent S-matrix across chunks.

        A caller calls this AFTER any per-config potential change and
        BEFORE the chunk loop, then :meth:`disable_forward_cache` after it --
        so each frozen-phonon config gets a fresh build and the cache never
        crosses a potential change. Starts empty; the first ``forward``
        builds and stores ``(S_dt, basis_fft)``, the rest reuse.
        """
        self._fwd_cache_enabled = True
        self._fwd_cache = None

    def disable_forward_cache(self) -> None:
        """Turn the forward-chunk cache off and free the cached tensors.

        """
        self._fwd_cache_enabled = False
        self._fwd_cache = None

    def needs_transmission(self) -> bool:
        """Whether the next :meth:`forward` will read ``T``.

        Returns
        -------
        bool
            False once the forward-chunk cache holds this configuration's
            state; True otherwise.
        """
        return not (self._fwd_cache_enabled and self._fwd_cache is not None)

    def _project_route(self) -> str:
        """Map the resolved (store, project) precision config onto the
        ``prism_project`` precision route. Pure function of ``precision_config``;
        identical on the cache-hit and cache-miss paths."""
        project = self.precision_config["project"]
        store = self.precision_config["store"]
        if project == "fp16":
            return "fp16"
        if store == "complex32":
            return "mixed"
        return "fp32"

    def _initial_parent_waves(self, lo: int = 0, hi: int | None = None) -> Tensor:
        """IFFT of a delta at each parent-beam location.

        Returns a (Bp, MY, MX) complex tensor (or ``(hi-lo, MY, MX)`` for the
        beam slice ``[lo, hi)`` -- used to generate the parent waves one
        beam-chunk at a time so the full ``(Bp, MY, MX)`` psi_fft + its IFFT
        output never co-exist, which OOMs the un-tiled single-S build at 20 nm).
        Each row is a real-space plane wave whose Fourier-space support is a
        single pixel at the corresponding parent-beam coordinate.

        Amplitude normalization (Ophus Eq 7 cutout): the ortho IFFT on the
        N=(MY,MX) simulation grid gives a plane-wave amplitude of
        ``1/sqrt(MY*MX)`` (= 1/N for a square grid). But the probe beamlet
        basis and the per-position real-space cutout both live on the
        DETECTOR grid ``P = N/f`` (ortho there -> ``1/P``). Without correction
        the S-matrix columns are ``1/f`` too small in amplitude (``1/f^2`` in
        intensity) relative to plain multislice on the P grid. Scaling by
        ``f = N/P`` makes the plane-wave amplitude ``1/P``, INDEPENDENT of the
        simulation grid N, so PRISM(N=f*P, f) matches multislice(detector=P).
        At ``f == 1`` (the reconstruction path, N==P) this is a literal no-op
        (no multiply), so recon stays bit-identical."""
        return initial_parent_waves(self, lo, hi)

    def _build_s_matrix(self, T: Tensor, propagator: Propagator) -> Tensor:
        """Multislice the parent-beam initial waves through the full-FOV
        transmission stack. The S-matrix columns are kept at FULL N x N
        real-space resolution for ALL ``f`` (Ophus 2017: ``f`` is realized
        downstream as a real-space N/f cutout per probe position in
        ``prism_project``, NOT as a Fourier crop here).

        With an ``exit_antialias_aperture`` attached, the finished columns go
        through ``_bandlimit_exit`` -- the last of a band-limited multislice's
        N masked propagations, which this loop (propagating only BETWEEN
        slices) does not perform -- on the RAW columns, before the de-tilt
        ramp and the cutout, neither of which commutes with a k-space
        multiply; per beam-chunk on the chunked path so its transient stays
        one chunk wide.  Identity (the same object) when no aperture is set.

        ``_prism_s_build_graph``: a stack larger than L2
        is built in L2-sized blocks of beams under one replayed CUDA graph --
        first with each block's parent waves generated inside the graph, else
        over the parent waves of either branch -- bit-identical to the loops
        below (backend rows ``prism_build_gen_graph`` / ``prism_build_l2_graph``).

        Args:
            T:           (NZ_bin, MY, MX) complex transmission.
            propagator:  Fresnel propagator matched to (MY, MX).

        Returns:
            S of shape (Bp, MY, MX), complex64.
        """
        if self.freeze_s_matrix:
            T = T.detach()
        MY, MX = int(self.s_meta.M[0]), int(self.s_meta.M[1])
        check_propagator_grid(propagator, MY, MX)
        # fp16 build: run the per-slice transmit in complex32 too. Cast the
        # transmission to the build dtype so ``T[z] * waves`` does NOT silently
        # upcast the half-precision waves back to complex64. The Fresnel
        # ``propagator`` already preserves complex32 (it runs the FFT round-trip
        # with norm='ortho' to bound the fp16 intermediate and returns the input
        # dtype). The multislice loop is a chain of elementwise multiplies and
        # FFT round-trips with NO reduction across slices/beams, so no separate
        # fp32 accumulator is needed here; the only reduction (the parent-beam
        # sum) happens later inside the Warp projection kernel, which already
        # accumulates in fp32 (Task 8). fp32 build: ``T`` is already complex64,
        # so this cast is a no-op and the path stays byte-identical.
        build_dtype = self._build_dtype()
        if build_dtype != torch.complex64 and T.dtype != build_dtype:
            T = T.to(build_dtype)
        NZ_bin = T.shape[0]
        Bp = int(self.s_meta.Bp)
        device = self.s_meta.device
        # Per-chunk beam count: bound the build transient to ~6e8 complex
        # elements (override via SCATTEREM_PRISM_BUILD_COLS, e.g. for tests that
        # force the chunked path on a small grid).
        _cols_env = os.environ.get("SCATTEREM_PRISM_BUILD_COLS", "")
        if _cols_env:
            cols = int(_cols_env)
        elif self.build_cols is not None:
            cols = self.build_cols
        else:
            cols = max(1, int(6e8 // max(1, int(MY) * int(MX))))
        if (generated := s_build_l2_generated(self, T, propagator, cols)) is not None:
            return generated
        if cols >= Bp:
            # Single pass (recon / small grids): byte-identical to before.
            waves = self._initial_parent_waves()
            # On a non-power-of-two grid the propagator cannot fold the inverse
            # transform's 1/N onto its phase ramp (not bit-exact there), so torch
            # runs a SEPARATE full-field scaling pass over the whole beam stack --
            # one extra read+write of the S-matrix per slice.  Let the transmit
            # below absorb it: the propagator hands back unnormalized waves plus
            # the outstanding `inv_n`, and the plane-resident kernel applies
            # `T[z] * (w * inv_n)` in the pass it was already making.  Same two
            # roundings in the same order as before, so the result is bit-exact
            # (proven by torch.equal at the build shape, not assumed).  Deferral is
            # asked for only when autograd is not watching: the in-place kernel is
            # invisible to autograd, and with a live graph the extra scaling node
            # would move the ADJOINT's rounding even though the value is identical.
            defer = not (
                torch.is_grad_enabled() and (T.requires_grad or waves.requires_grad)
            )
            if s_build_l2_blocked(self, T, waves, propagator) is not None:
                return waves  # built in place by the block graph
            inv_n = 1.0
            for z in range(NZ_bin):
                if inv_n == 1.0:
                    waves = T[z].unsqueeze(0) * waves
                elif not _bcast_cmul_(waves, T[z], True, scale=inv_n):
                    # Kernel declined (see its gates): the same arithmetic in two
                    # torch passes, i.e. exactly what shipped before.
                    waves = T[z].unsqueeze(0) * (waves * inv_n)
                if z < NZ_bin - 1:
                    if defer:
                        waves, inv_n = propagator(waves, defer_norm=True)
                    else:
                        waves = propagator(waves)
            # Full-N S-matrix for ALL f (Ophus Eq 7): f is realized as the per-
            # position real-space N/f cutout in prism_project, NOT a Fourier crop.
            return self._bandlimit_exit(waves)
        # Large-grid (un-tiled single-S) path: GENERATE + multislice the parent
        # waves ONE BEAM-CHUNK at a time, writing straight into the output S. The
        # full (Bp, MY, MX) initial waves -- and its psi_fft, and a second output
        # buffer -- never co-exist; the old build needed ~3x(Bp, MY, MX) complex64
        # (psi_fft + ifft + out, ~3x27 GB at 20 nm) and OOM'd an 80 GB GPU. The
        # total FFT work is unchanged (Bp*NZ). Forward-only path (no autograd):
        # the recon path takes the single-pass branch above.
        S = torch.empty((Bp, MY, MX), dtype=build_dtype, device=device)
        # Same deferred-norm fold as the single-pass branch above, chunk-local:
        # on a non-power-of-two grid the propagator cannot ride the inverse
        # transform's 1/N in on its phase ramp, so torch runs a SEPARATE
        # full-field scaling pass over the beam chunk, once per slice.  Let the
        # next transmit absorb it -- same two roundings in the same order, so the
        # S-matrix is bit-exact.  ``inv_n`` is consumed by the transmit of the
        # following slice and the loop ends on a transmit, so nothing is
        # outstanding when ``w`` is written into ``S``.  Grad-gated exactly as
        # above (an in-place kernel write is invisible to autograd); this branch
        # is forward-only in production, so the gate is belt and braces.
        defer = not (torch.is_grad_enabled() and T.requires_grad)
        for i in range(0, Bp, cols):
            j = min(i + cols, Bp)
            w = self._initial_parent_waves(i, j)  # (j-i, MY, MX), build dtype
            defer_i = defer and not w.requires_grad
            if s_build_l2_blocked(self, T, w, propagator) is not None:
                S[i:j] = w  # the chunk, built in place by the same graph
                continue
            inv_n = 1.0
            for z in range(NZ_bin):
                # Plane-resident transmit: bit-identical to the torch expression
                # below (same operand order, explicit FMA) but the (MY,MX) plane is
                # loaded once per plane-tile instead of once per element of ``w``,
                # which is 1.31x at the production chunk shape -- a 2 MiB plane does
                # not survive in L2 against a 4.8 GB stream.  Declines (-> torch)
                # unless the stack is large, CUDA, complex64 and grad-free; ``w`` is
                # freshly built by ``_initial_parent_waves`` / ``propagator``, so the
                # in-place write aliases nothing.
                if inv_n == 1.0:
                    if not _bcast_cmul_(w, T[z], True):
                        w = T[z].unsqueeze(0) * w
                elif not _bcast_cmul_(w, T[z], True, scale=inv_n):
                    # Kernel declined (see its gates): the same arithmetic in two
                    # torch passes, i.e. exactly what shipped before.
                    w = T[z].unsqueeze(0) * (w * inv_n)
                if z < NZ_bin - 1:
                    if defer_i:
                        w, inv_n = propagator(w, defer_norm=True)
                    else:
                        w = propagator(w)
            S[i:j] = self._bandlimit_exit(w)
        return S



    # ------------------------------------------------------------ basis builder
    def _beamlet_basis(
        self,
        probe_fft_single: Tensor,
        s_meta: "SMeta",
        beam_scale: Tensor | None = None,
    ) -> BeamletBasis:
        """Build the position-independent Fourier-space beamlet basis.

        Factors out the K-independent part of step 6 in ``forward``.  The
        result lives in Fourier space; callers apply the sub-pixel ramp and
        the IFFT themselves (e.g. ``_prism_project_torch`` does
        ``ifft2(basis_fft * ramp) * sqrt(M1*M2)``).

        Args:
            probe_fft_single: ``(Nmodes, M1, M2)`` complex — FFT of a single
                K-slice of the probe (the ortho-normalised ``fft2``; the caller
                is responsible for computing it).  For a K-broadcast probe every
                K-slice is identical, so any one slice works.
            s_meta:  :class:`~scatterem.data.SMeta` carrying
                ``all_beams_coords``, ``beamlets``, and grid sizes ``M``.
            beam_scale: optional ``(B,)`` complex factor per APERTURE beam,
                multiplied onto its probe amplitude -- the free-space phase
                the column preparation stripped
                (:func:`~scatterem.nn.scan_exit_wave._columns.vacuum_phase`).

        Returns:
            ``basis_fft`` of shape ``(Nmodes, Bp, M1, M2)`` complex.
            ``basis_fft[m, p, qy, qx]`` holds the weighted probe amplitude
            ``Σ_b w_{p,b} Ψ_m(h_b)`` scattered onto the ``(M1, M2)`` beam
            grid — **before** the IFFT and the ``sqrt(M1*M2)`` renormalisation.

        Math (step 6 of ``forward``, K axis dropped)::

            Psi_aperture[m, b] = probe_fft_single[m, ay[b], ax[b]]
            weighted[m, p, b]  = beamlets_at_beams[p, b] * Psi_aperture[m, b]
            basis_fft[m, p, ay[b], ax[b]] += weighted[m, p, b]   ∀ b

        Beam-coordinate mapping (mirrors ``forward`` step 6)::

            scale_y = M1 / MY;  scale_x = M2 / MX
            ay = round(all_beams_coords[:,0] * scale_y) % M1
            ax = round(all_beams_coords[:,1] * scale_x) % M2
        """
        _, M1, M2 = probe_fft_single.shape
        MY, MX = int(s_meta.M[0]), int(s_meta.M[1])
        Bp = s_meta.Bp
        dtype = probe_fft_single.dtype

        # Map s_meta beam coordinates (MY, MX grid) to (M1, M2) FFT indices.
        scale_y = M1 / MY
        scale_x = M2 / MX
        ay = (s_meta.all_beams_coords[:, 0].float() * scale_y).round().long() % M1
        ax = (s_meta.all_beams_coords[:, 1].float() * scale_x).round().long() % M2

        # Gather probe-FFT amplitudes at each beam location -> (Nmodes, B)
        Psi_aperture = probe_fft_single[:, ay, ax]  # (Nmodes, B)
        if beam_scale is not None:
            Psi_aperture = Psi_aperture * beam_scale.to(dtype)[None, :]

        # NNW weights at the B beam locations, stored directly as (Bp, B)
        # (no dense (Bp, MY, MX) beamlet tensor to gather from). One row per
        # parent beam, always: ``SMeta.make_beamlet_meta`` computes the weights
        # for the parents it selected, whatever their number.
        beamlets_at_beams = s_meta.beamlets.to(dtype)  # (Bp, B)
        if tuple(beamlets_at_beams.shape) != (Bp, int(s_meta.B)):
            raise ValueError(
                f"s_meta.beamlets has shape {tuple(beamlets_at_beams.shape)}, "
                f"expected (Bp, B) = ({Bp}, {int(s_meta.B)}): the weights were "
                f"not built for this parent set."
            )

        # The FACTORS of the (Nmodes, Bp, M1, M2) basis: weighted[m, p, b] =
        # w_{pb} * Psi_m(h_b), scattered onto pixel ay * M2 + ax -- ACCUMULATED:
        # at f > 1 DISTINCT beams round onto the SAME detector pixel (37 beams
        # onto 13 pixels at f=2) and an assignment with duplicate indices races
        # on CUDA (0.10-0.46 relative run-to-run). Dense, the basis is as large as
        # the S-matrix at f = 1; ``BeamletBasis`` scatters it a block at a time.
        return BeamletBasis(Psi_aperture, beamlets_at_beams, ay * M2 + ax, (M1, M2))

    def _build_beamlet_basis_fft(
        self,
        probe_fft_single: Tensor,
        s_meta: "SMeta",
        beam_scale: Tensor | None = None,
    ) -> Tensor:
        """:meth:`_beamlet_basis` as ONE dense ``(Nmodes, Bp, M1, M2)`` tensor."""
        return self._beamlet_basis(probe_fft_single, s_meta, beam_scale).dense()

    # -------------------------------------------------------------- abstract
    def forward(
        self,
        T: Tensor,  # (NZ_bin, MY, MX) complex
        probe: Tensor,  # (Nmodes, K, M1, M2) complex (real-space)
        positions: Tensor,  # (K, 2) int
        dr: Tensor,  # (K, 2) float
        propagator: Propagator,
        inverse_index: Tensor | None = None,  # unused — PRISM crops S-matrix, no gather
    ) -> Tensor:
        """Build (or reuse) the S-matrix and project exit waves at scan positions.

        Parameters
        ----------
        T : torch.Tensor
            ``(NZ_bin, MY, MX)`` complex transmission function.
        probe : torch.Tensor
            ``(Nmodes, K, M1, M2)`` complex real-space probe (K-broadcast).
        positions : torch.Tensor
            ``(K, 2)`` int scan positions on the simulation grid.
        dr : torch.Tensor
            ``(K, 2)`` float sub-pixel offsets, in pixels.
        propagator : Propagator
            Fresnel propagator matched to ``s_meta.M``.
        inverse_index : torch.Tensor or None, optional
            Unused -- PRISM crops the S-matrix per position instead of
            gathering.

        Returns
        -------
        torch.Tensor
            ``(Nmodes, K, M1, M2)`` complex real-space exit waves.
        """
        del inverse_index  # PRISM does not gather; it crops the S-matrix per position
        # Forward-chunk cache hit: S_dt + basis_fft are position-independent, so
        # skip the multislice S-build / de-tilt / basis and run ONLY the per-
        # position projection. Enabled by the caller within a single config's
        # chunk loop (forward-sim only); the cache holds the FINAL S_dt (post
        # focal-backprop, de-tilt and store-downcast), so reuse is exact.
        if self._fwd_cache_enabled and self._fwd_cache is not None:
            return self._project(*self._fwd_cache, positions, dr, propagator)
        S = self._build_s_matrix(T, propagator)  # (Bp, SY, SX)

        # ---- Column preparation ---------------------------------------------
        #
        # ``_columns.prepare_columns_``: propagate every parent column to the
        # reference plane (Pelz 2021's focal back-propagation, the full per-Q
        # Fresnel transfer), de-tilt it, and strip the free-space phase it still
        # carries from that plane to the exit,
        #
        #   S_dt[p] = IFFT(FFT(S[p]) e^{-i n_half phi}) e^{-2 pi i h_p . r}
        #             e^{-i (NZ - 1 - n_half) phi[h_p]}.
        #
        # What is left is only what the SPECIMEN did to beam p -- in vacuum the
        # same constant for every beam, on every grid and for every reference
        # plane -- and that is the one thing the natural-neighbour fusion may
        # interpolate. The stripped phase is known in closed form, so it goes
        # back per APERTURE beam on the probe coefficients (``beam_scale`` of
        # the basis below) instead of being interpolated across the parents,
        # which is what made partitioned PRISM inexact in vacuum whenever the
        # reference plane was not the entrance.
        #
        # The reference plane exists for the interpolation, so it is resolved
        # only when there is one (Bp < B): the ``focal_backprop`` argument, through
        # ``_columns.reference_slices``. Pure PRISM keeps the exit plane.
        NZ_bin = int(T.shape[0])
        _fb = self.focal_backprop
        partitioned = self.s_meta.Bp < self.s_meta.B
        n_half = reference_slices(_fb or 0.0, T) if partitioned else 0.0
        # Remembered: the projected patches are propagated FORWARD by the same
        # amount so this operator still returns the EXIT-plane wave.
        self._bp_n_half = float(n_half)
        phase = propagator.fresnel_phase()  # (SY, SX) unwrapped, per slice
        parent = self.s_meta.parent_beams_coords.long()
        if S.requires_grad:
            S_dt = prepare_columns_autograd(S, parent, phase, n_half, NZ_bin)
        else:
            prepare_columns_(S, parent, phase, n_half, NZ_bin, _PREPARE_BYTES)
            S_dt = S  # prepared in place
        del S

        # ---- Optional store downcast (mixed / fp16 presets) -----------------
        #
        # Narrow S_dt from complex64 to complex32 (two float16 per element,
        # half the bytes) before projection.  The multislice build accumulation
        # stays in the build dtype (fp32 for mixed/fp32; fp16 for fp16 preset);
        # only the stored/projected S_dt is narrowed here.  Projection
        # consumption of the narrowed tensor is handled in Task 9.
        if self.precision_config["store"] == "complex32":
            S_dt = S_dt.to(torch.complex32)

        # ---- Steps 5–7: fused projection via prism_project -----------------
        #
        # KEY CONSTRAINT: the probe must be K-broadcast (all K slices identical);
        # the basis is position-independent and only the sub-pixel ramp varies
        # per position.  Use a single K-slice to build the basis.
        #
        # dr is passed in PIXELS (SubpixShift ramp convention); prism_project
        # expects pixel units directly — do NOT multiply by dx.
        probe_fft_single = torch.fft.fft2(probe[:, 0], norm="ortho")  # (Nmodes, M1, M2)
        basis_fft = self._beamlet_basis(
            probe_fft_single,
            self.s_meta,
            beam_scale=vacuum_phase(
                self.s_meta.all_beams_coords, phase, n_half, NZ_bin
            ),
        )  # (Nmodes, Bp, M1, M2), factored
        # ---- Project axis: forward the resolved precision to prism_project ---
        #
        # Map the (store, project) config onto prism_project's ``precision`` route:
        #   * project=fp16                 -> "fp16"  (true half-precision multiply)
        #   * store=complex32, project=fp32 -> "mixed" (complex32 ``S_dt`` kept as a
        #     float16 view; the Warp kernel casts each operand to fp32 PER ELEMENT
        #     before the multiply, so the arithmetic is fp32 but neither the
        #     S-matrix nor the basis is ever materialised in complex64 — the
        #     complex32 storage's projection-time memory win is preserved).
        #   * otherwise (store=complex64, project=fp32) -> "fp32" (byte-identical).
        # Store for reuse by the remaining forward chunks of this config.
        if self._fwd_cache_enabled:
            self._fwd_cache = (S_dt, basis_fft)
        return self._project(S_dt, basis_fft, positions, dr, propagator)

    def _project(
        self,
        S_dt: Tensor,
        basis_fft: BeamletBasis,
        positions: Tensor,
        dr: Tensor,
        propagator: Propagator,
    ) -> Tensor:
        """Per-position half of :meth:`forward`: ``(Nmodes, K, M1, M2)`` exit waves;
        ONE call, for the chunk that builds the state and the chunks that reuse it."""
        projected = prism_project(
            S_dt,
            basis_fft,
            positions,
            dr,
            chunk=self.scan_chunk,
            precision=self._project_route(),
            periodic=self.periodic,
            use_warp=self.use_warp,
        )
        return self._to_exit_plane(projected, propagator)

    def _to_exit_plane(self, out: Tensor, propagator: Propagator) -> Tensor:
        """Undo the focal back-propagation on the projected patches.

        ``ScanExitWaveOperator`` is contracted to return the real-space EXIT
        wave (``base.py``), which ``MultisliceScanOp`` does. Focal
        back-propagation moves the S-matrix columns to the interpolation-optimal
        plane so the NNW fusion sees a slowly-varying function of beam index --
        a genuine accuracy lever -- but the *fused result* then sits ``n_half``
        slices upstream and has to be brought back.

        It is not enough to note that Fresnel is a pure phase in Fourier space
        and therefore invisible to ``|FFT|**2``. That argument holds only for a
        propagation applied to the final wave. Here the back-propagation is
        applied to the S-matrix *before* the per-column de-tilt
        ``exp(-2j*pi*h_p.r)``, and a real-space phase ramp is a Fourier-space
        *shift*, which does not commute with a Fourier-space multiplication. So
        the omission is observable in intensity: measured against multislice, the
        default ``focal_backprop="centroid"`` was 1.7x WORSE than no back-prop at
        Bp~19 (0.176 vs 0.101 far-field, 0.746 vs 0.067 real-space).

        Propagating the patch rather than the full grid is exact for the same
        reason the crop is: the patch is a crop, so it keeps the real-space
        pixel size and only the grid shape changes.
        """
        n_half = getattr(self, "_bp_n_half", 0.0)
        if not n_half:
            return out
        kernel = plane_transfer(
            propagator.fresnel_phase(shape=out.shape[-2:], device=out.device), n_half
        )
        return torch.fft.ifft2(torch.fft.fft2(out.to(torch.complex64)) * kernel).to(
            out.dtype
        )


__all__ = [
    "PrismScanOp",
]
