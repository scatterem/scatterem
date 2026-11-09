# API reference

Sphinx builds this page from the docstrings of the installed package, so it
cannot drift from the code. Everything below is public; anything not listed
here is internal and may change without notice. The documentation build
fails when a public module exports a name that this page does not show.

## Package settings

```{eval-rst}
.. currentmodule:: scatterem

.. autosummary::
   :nosignatures:

   configure

.. raw:: html

   <span id="configure"></span>

.. autofunction:: configure

```

## Container

```{eval-rst}
.. currentmodule:: scatterem.data.datasets

.. autosummary::
   :nosignatures:

   Dataset
   Dataset4DStem
   DatasetVirtualBrightField4DStem
   with_meta

.. raw:: html

   <span id="dataset"></span>

.. autoclass:: Dataset
   :no-members:

.. raw:: html

   <span id="dataset4dstem"></span>
   <span id="dataset4dstem-from_array"></span>
   <span id="dataset4dstem-calibrate_reciprocal_from_bright_field"></span>
   <span id="dataset4dstem-bright_field_radius_and_center"></span>
   <span id="serializable-save"></span>
   <span id="serializable-load"></span>

.. autoclass:: Dataset4DStem
   :members: from_array, calibrate_reciprocal_from_bright_field, bright_field_radius_and_center, save, load

.. raw:: html

   <span id="datasetvirtualbrightfield4dstem"></span>

.. autoclass:: DatasetVirtualBrightField4DStem
   :no-members:

.. raw:: html

   <span id="with_meta"></span>

.. autofunction:: with_meta

```

## Running a reconstruction

```{eval-rst}
.. currentmodule:: scatterem.reconstruction

.. autosummary::
   :nosignatures:

   run
   Result
   ReconstructionConfig
   Provenance
   register_runner

.. raw:: html

   <span id="run"></span>

.. autofunction:: run

.. raw:: html

   <span id="result"></span>

.. autoclass:: Result
   :no-members:

.. raw:: html

   <span id="reconstructionconfig"></span>

.. autoclass:: ReconstructionConfig
   :no-members:

.. raw:: html

   <span id="provenance"></span>

.. autoclass:: Provenance
   :no-members:

.. raw:: html

   <span id="register_runner"></span>

.. autofunction:: register_runner

```

## Direct ptychography and the aberration fit

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.ptychography.direct

.. autosummary::
   :nosignatures:

   AberrationFitConfig
   AberrationFitResult
   DirectPtychographyConfig
   DirectPtychographyResult

.. raw:: html

   <span id="aberrationfitconfig"></span>

.. autoclass:: AberrationFitConfig
   :no-members:

.. raw:: html

   <span id="aberrationfitresult"></span>
   <span id="aberrationfitresult-applied_to"></span>
   <span id="aberrationfitresult-as_tuple"></span>

.. autoclass:: AberrationFitResult
   :members: applied_to, as_tuple

.. raw:: html

   <span id="directptychographyconfig"></span>

.. autoclass:: DirectPtychographyConfig
   :no-members:

.. raw:: html

   <span id="directptychographyresult"></span>
   <span id="directptychographyresult-as_tuple"></span>

.. autoclass:: DirectPtychographyResult
   :members: as_tuple

```

## Bright-field parallax

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.ptychography.direct

.. autosummary::
   :nosignatures:

   measure_bf_parallax
   fit_aberrations_and_rotation_to_bright_field_shifts
   aberrations_and_rotation_from_bright_field_shifts

.. raw:: html

   <span id="measure_bf_parallax"></span>

.. autofunction:: measure_bf_parallax

.. raw:: html

   <span id="fit_aberrations_and_rotation_to_bright_field_shifts"></span>

.. autofunction:: fit_aberrations_and_rotation_to_bright_field_shifts

.. raw:: html

   <span id="aberrations_and_rotation_from_bright_field_shifts"></span>

.. autofunction:: aberrations_and_rotation_from_bright_field_shifts

```

## Scan rotation and centre-of-mass calibration

```{eval-rst}
.. currentmodule:: scatterem.data.calibration

.. autosummary::
   :nosignatures:

   find_scan_rotation
   ScanRotation
   RotationSearch
   fit_origin
   fit_2D
   plane
   parabola
   bezier_two

.. raw:: html

   <span id="find_scan_rotation"></span>

.. autofunction:: find_scan_rotation

.. raw:: html

   <span id="scanrotation"></span>

.. autoclass:: ScanRotation
   :no-members:

.. raw:: html

   <span id="rotationsearch"></span>

.. autoclass:: RotationSearch
   :no-members:

.. raw:: html

   <span id="fit_origin"></span>

.. autofunction:: fit_origin

.. raw:: html

   <span id="fit_2d"></span>

.. autofunction:: fit_2D

.. raw:: html

   <span id="plane"></span>

.. autofunction:: plane

.. raw:: html

   <span id="parabola"></span>

.. autofunction:: parabola

.. raw:: html

   <span id="bezier_two"></span>

.. autofunction:: bezier_two

```

## Scan-rotation figures

```{eval-rst}
.. currentmodule:: scatterem.vis.rotation_figures

.. autosummary::
   :nosignatures:

   angle_search_figure
   plot_scan_maps

.. raw:: html

   <span id="angle_search_figure"></span>

.. autofunction:: angle_search_figure

.. raw:: html

   <span id="plot_scan_maps"></span>

.. autofunction:: plot_scan_maps

```

## Tilt-corrected dark field

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.darkfield

.. autosummary::
   :nosignatures:

   TiltCorrectedDarkFieldConfig
   TiltCorrectedDarkFieldResult
   tilt_corrected_dark_field_depth_section

.. raw:: html

   <span id="tiltcorrecteddarkfieldconfig"></span>

.. autoclass:: TiltCorrectedDarkFieldConfig
   :no-members:

.. raw:: html

   <span id="tiltcorrecteddarkfieldresult"></span>
   <span id="tiltcorrecteddarkfieldresult-as_tuple"></span>

.. autoclass:: TiltCorrectedDarkFieldResult
   :members: as_tuple

.. raw:: html

   <span id="tilt_corrected_dark_field_depth_section"></span>

.. autofunction:: tilt_corrected_dark_field_depth_section

```

## Fused full field

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.fusion.ffstem

.. autosummary::
   :nosignatures:

   FusedFullFieldConfig
   FusedFullFieldResult

.. raw:: html

   <span id="fusedfullfieldconfig"></span>

.. autoclass:: FusedFullFieldConfig
   :no-members:

.. raw:: html

   <span id="fusedfullfieldresult"></span>
   <span id="fusedfullfieldresult-as_tuple"></span>

.. autoclass:: FusedFullFieldResult
   :members: as_tuple

```

## Reconstruction functions

```{eval-rst}
.. currentmodule:: scatterem.reconstruction

.. autosummary::
   :nosignatures:

   direct_ptychography
   tilt_corrected_dark_field
   fused_full_field

.. raw:: html

   <span id="direct_ptychography"></span>

.. autofunction:: direct_ptychography

.. raw:: html

   <span id="tilt_corrected_dark_field"></span>

.. autofunction:: tilt_corrected_dark_field

.. raw:: html

   <span id="fused_full_field"></span>

.. autofunction:: fused_full_field

```

## Named datasets

```{eval-rst}
.. currentmodule:: scatterem.data.public

.. autosummary::
   :nosignatures:

   You2026Carbon
   You2026Co3O4
   You2026AuLowDose
   You2026Gd2O3

.. raw:: html

   <span id="you2026carbon"></span>

.. autoclass:: You2026Carbon
   :no-members:

.. raw:: html

   <span id="you2026co3o4"></span>

.. autoclass:: You2026Co3O4
   :no-members:

.. raw:: html

   <span id="you2026aulowdose"></span>

.. autoclass:: You2026AuLowDose
   :no-members:

.. raw:: html

   <span id="you2026gd2o3"></span>

.. autoclass:: You2026Gd2O3
   :no-members:

```

## Metadata

```{eval-rst}
.. currentmodule:: scatterem.data.acquisition

.. autosummary::
   :nosignatures:

   AcquisitionMetadata

.. raw:: html

   <span id="acquisitionmetadata"></span>

.. autoclass:: AcquisitionMetadata
   :no-members:

```

## Aberration coefficients and scan metadata

```{eval-rst}
.. currentmodule:: scatterem.data

.. autosummary::
   :nosignatures:

   Aberrations
   SMeta

.. raw:: html

   <span id="aberrations"></span>

.. autoclass:: Aberrations
   :no-members:

.. raw:: html

   <span id="smeta"></span>

.. autoclass:: SMeta
   :no-members:

```

## Aberrations and the probe

```{eval-rst}
.. currentmodule:: scatterem.physics.optics

.. autosummary::
   :nosignatures:

   LENS_LAYOUT
   LENS_TERMS
   AMPLITUDE_LAYOUT
   pack_lens
   unpack_lens
   lens_wavefront
   lens_wavefront_slope
   lens_to_amplitude
   lens_from_amplitude
   RIM_WIDTH_PX
   smooth_disc
   smooth_annulus
   illumination_disc

.. raw:: html

   <span id="lens_layout"></span>

.. autodata:: LENS_LAYOUT

.. raw:: html

   <span id="lens_terms"></span>

.. autodata:: LENS_TERMS

.. raw:: html

   <span id="amplitude_layout"></span>

.. autodata:: AMPLITUDE_LAYOUT

.. raw:: html

   <span id="pack_lens"></span>

.. autofunction:: pack_lens

.. raw:: html

   <span id="unpack_lens"></span>

.. autofunction:: unpack_lens

.. raw:: html

   <span id="lens_wavefront"></span>

.. autofunction:: lens_wavefront

.. raw:: html

   <span id="lens_wavefront_slope"></span>

.. autofunction:: lens_wavefront_slope

.. raw:: html

   <span id="lens_to_amplitude"></span>

.. autofunction:: lens_to_amplitude

.. raw:: html

   <span id="lens_from_amplitude"></span>

.. autofunction:: lens_from_amplitude

.. raw:: html

   <span id="rim_width_px"></span>

.. autodata:: RIM_WIDTH_PX

.. raw:: html

   <span id="smooth_disc"></span>

.. autofunction:: smooth_disc

.. raw:: html

   <span id="smooth_annulus"></span>

.. autofunction:: smooth_annulus

.. raw:: html

   <span id="illumination_disc"></span>

.. autofunction:: illumination_disc

```

## Calibration helpers

```{eval-rst}
.. currentmodule:: scatterem.data.pupil_gauge

.. autosummary::
   :nosignatures:

   measure_pupil
   PupilEstimate
   ESTIMATORS

.. raw:: html

   <span id="measure_pupil"></span>

.. autofunction:: measure_pupil

.. raw:: html

   <span id="pupilestimate"></span>

.. autoclass:: PupilEstimate
   :no-members:

.. raw:: html

   <span id="estimators"></span>

.. autodata:: ESTIMATORS

```

## Physics

```{eval-rst}
.. currentmodule:: scatterem.physics.constants

.. autosummary::
   :nosignatures:

   debroglie_at
   gamma_at
   moving_mass_at
   phase_per_potential_at
   bend_per_tesla_at
   EXACT_PLANCK
   EXACT_E_CHARGE
   EXACT_LIGHT_SPEED
   EXACT_AVOGADRO
   EXACT_BOLTZMANN
   CODATA18_ELECTRON_MASS
   CODATA18_DALTON
   CODATA18_FINE_STRUCTURE
   CODATA18_VACUUM_MU
   CODATA18_ELECTRON_RADIUS
   DERIVED_PLANCK_BAR
   DERIVED_ELECTRON_REST_EV
   DERIVED_AB_COUPLING

.. raw:: html

   <span id="debroglie_at"></span>

.. autofunction:: debroglie_at

.. raw:: html

   <span id="gamma_at"></span>

.. autofunction:: gamma_at

.. raw:: html

   <span id="moving_mass_at"></span>

.. autofunction:: moving_mass_at

.. raw:: html

   <span id="phase_per_potential_at"></span>

.. autofunction:: phase_per_potential_at

.. raw:: html

   <span id="bend_per_tesla_at"></span>

.. autofunction:: bend_per_tesla_at

.. raw:: html

   <span id="exact_planck"></span>

.. autodata:: EXACT_PLANCK

.. raw:: html

   <span id="exact_e_charge"></span>

.. autodata:: EXACT_E_CHARGE

.. raw:: html

   <span id="exact_light_speed"></span>

.. autodata:: EXACT_LIGHT_SPEED

.. raw:: html

   <span id="exact_avogadro"></span>

.. autodata:: EXACT_AVOGADRO

.. raw:: html

   <span id="exact_boltzmann"></span>

.. autodata:: EXACT_BOLTZMANN

.. raw:: html

   <span id="codata18_electron_mass"></span>

.. autodata:: CODATA18_ELECTRON_MASS

.. raw:: html

   <span id="codata18_dalton"></span>

.. autodata:: CODATA18_DALTON

.. raw:: html

   <span id="codata18_fine_structure"></span>

.. autodata:: CODATA18_FINE_STRUCTURE

.. raw:: html

   <span id="codata18_vacuum_mu"></span>

.. autodata:: CODATA18_VACUUM_MU

.. raw:: html

   <span id="codata18_electron_radius"></span>

.. autodata:: CODATA18_ELECTRON_RADIUS

.. raw:: html

   <span id="derived_planck_bar"></span>

.. autodata:: DERIVED_PLANCK_BAR

.. raw:: html

   <span id="derived_electron_rest_ev"></span>

.. autodata:: DERIVED_ELECTRON_REST_EV

.. raw:: html

   <span id="derived_ab_coupling"></span>

.. autodata:: DERIVED_AB_COUPLING

```

## Structures

```{eval-rst}
.. currentmodule:: scatterem.physics.atom_set

.. autosummary::
   :nosignatures:

   AtomSet
   read_atom_set
   has_partial_occupancy
   DEFAULT_TILT_SINE
   FULL_OCCUPANCY_TOLERANCE
   MAX_SUPERCELL_FACTOR

.. raw:: html

   <span id="atomset"></span>
   <span id="atomset-with_fractional"></span>
   <span id="atomset-with_rotation"></span>
   <span id="atomset-replicated"></span>

.. autoclass:: AtomSet
   :members: with_fractional, with_rotation, replicated

.. raw:: html

   <span id="read_atom_set"></span>

.. autofunction:: read_atom_set

.. raw:: html

   <span id="has_partial_occupancy"></span>

.. autofunction:: has_partial_occupancy

.. raw:: html

   <span id="default_tilt_sine"></span>

.. autodata:: DEFAULT_TILT_SINE

.. raw:: html

   <span id="full_occupancy_tolerance"></span>

.. autodata:: FULL_OCCUPANCY_TOLERANCE

.. raw:: html

   <span id="max_supercell_factor"></span>

.. autodata:: MAX_SUPERCELL_FACTOR

```

## Slicing along the beam

```{eval-rst}
.. currentmodule:: scatterem.physics.beam_partition

.. autosummary::
   :nosignatures:

   BeamPartition
   species_order
   edge_snap_margin
   EDGE_SNAP_ULPS

.. raw:: html

   <span id="beampartition"></span>
   <span id="beampartition-uniform"></span>
   <span id="beampartition-from_thicknesses"></span>
   <span id="beampartition-edge_values"></span>
   <span id="beampartition-thicknesses"></span>
   <span id="beampartition-subrange"></span>
   <span id="beampartition-bin_index"></span>
   <span id="beampartition-bin_index_clamped"></span>

.. autoclass:: BeamPartition
   :members: uniform, from_thicknesses, edge_values, thicknesses, subrange, bin_index, bin_index_clamped

.. raw:: html

   <span id="species_order"></span>

.. autofunction:: species_order

.. raw:: html

   <span id="edge_snap_margin"></span>

.. autofunction:: edge_snap_margin

.. raw:: html

   <span id="edge_snap_ulps"></span>

.. autodata:: EDGE_SNAP_ULPS

```

## Projected potentials

```{eval-rst}
.. currentmodule:: scatterem.physics.potential_stack

.. autosummary::
   :nosignatures:

   potential_stack
   check_stack_request
   StackRequest
   resolve_stack_device

.. raw:: html

   <span id="potential_stack"></span>

.. autofunction:: potential_stack

.. raw:: html

   <span id="check_stack_request"></span>

.. autofunction:: check_stack_request

.. raw:: html

   <span id="stackrequest"></span>

.. autoclass:: StackRequest
   :no-members:

.. raw:: html

   <span id="resolve_stack_device"></span>

.. autofunction:: resolve_stack_device

```

## Per-row replacements

```{eval-rst}
.. currentmodule:: scatterem.physics.row_replacements

.. autosummary::
   :nosignatures:

   RowReplacements

.. raw:: html

   <span id="rowreplacements"></span>

.. autoclass:: RowReplacements
   :no-members:

```

## Thermal displacements

```{eval-rst}
.. currentmodule:: scatterem.physics.thermal_jitter

.. autosummary::
   :nosignatures:

   ThermalJitter
   jitter_offsets
   colocated_rows
   COLOCATION_TOLERANCE

.. raw:: html

   <span id="thermaljitter"></span>

.. autoclass:: ThermalJitter
   :no-members:

.. raw:: html

   <span id="jitter_offsets"></span>

.. autofunction:: jitter_offsets

.. raw:: html

   <span id="colocated_rows"></span>

.. autofunction:: colocated_rows

.. raw:: html

   <span id="colocation_tolerance"></span>

.. autodata:: COLOCATION_TOLERANCE

```

## Depth shares

```{eval-rst}
.. currentmodule:: scatterem.physics.axial_shares

.. autosummary::
   :nosignatures:

   axial_shares
   AxialShares
   BEAM_SPREADS
   GAUSS_IMAGE_REACH

.. raw:: html

   <span id="axial_shares"></span>

.. autofunction:: axial_shares

.. raw:: html

   <span id="axialshares"></span>

.. autoclass:: AxialShares
   :no-members:

.. raw:: html

   <span id="beam_spreads"></span>

.. autodata:: BEAM_SPREADS

.. raw:: html

   <span id="gauss_image_reach"></span>

.. autodata:: GAUSS_IMAGE_REACH

```

## Potential rolloff

```{eval-rst}
.. currentmodule:: scatterem.physics.potential_rolloff

.. autosummary::
   :nosignatures:

   rolloff_mask
   check_rolloff
   DEFAULT_ROLLOFF

.. raw:: html

   <span id="rolloff_mask"></span>

.. autofunction:: rolloff_mask

.. raw:: html

   <span id="check_rolloff"></span>

.. autofunction:: check_rolloff

.. raw:: html

   <span id="default_rolloff"></span>

.. autodata:: DEFAULT_ROLLOFF

```

## Spectrum reuse

```{eval-rst}
.. currentmodule:: scatterem.physics.spectrum_sweep

.. autosummary::
   :nosignatures:

   spectrum_sweep
   SweepLedger

.. raw:: html

   <span id="spectrum_sweep"></span>

.. autofunction:: spectrum_sweep

.. raw:: html

   <span id="sweepledger"></span>

.. autoclass:: SweepLedger
   :no-members:

```

## Potential routes

```{eval-rst}
.. currentmodule:: scatterem.physics.route_settings

.. autosummary::
   :nosignatures:

   SpatialRouteSettings
   choose_route
   route_rolloff
   check_route
   check_cutoff
   ROUTE_NAMES
   DEFAULT_PROJECTED_CUTOFF

.. raw:: html

   <span id="spatialroutesettings"></span>

.. autoclass:: SpatialRouteSettings
   :no-members:

.. raw:: html

   <span id="choose_route"></span>

.. autofunction:: choose_route

.. raw:: html

   <span id="route_rolloff"></span>

.. autofunction:: route_rolloff

.. raw:: html

   <span id="check_route"></span>

.. autofunction:: check_route

.. raw:: html

   <span id="check_cutoff"></span>

.. autofunction:: check_cutoff

.. raw:: html

   <span id="route_names"></span>

.. autodata:: ROUTE_NAMES

.. raw:: html

   <span id="default_projected_cutoff"></span>

.. autodata:: DEFAULT_PROJECTED_CUTOFF

```

## Transmission functions

```{eval-rst}
.. currentmodule:: scatterem.physics.thin_slice

.. autosummary::
   :nosignatures:

   thin_slice_transmission
   phase_from_projection

.. raw:: html

   <span id="thin_slice_transmission"></span>

.. autofunction:: thin_slice_transmission

.. raw:: html

   <span id="phase_from_projection"></span>

.. autofunction:: phase_from_projection

```

## Scattering amplitudes

```{eval-rst}
.. currentmodule:: scatterem.physics.atom_amplitudes

.. autosummary::
   :nosignatures:

   born_amplitude
   born_amplitude_array
   born_amplitudes
   potential_spectra
   gamma_wavelength
   validated_atomic_number
   POTENTIAL_PER_AMPLITUDE

.. raw:: html

   <span id="born_amplitude"></span>

.. autofunction:: born_amplitude

.. raw:: html

   <span id="born_amplitude_array"></span>

.. autofunction:: born_amplitude_array

.. raw:: html

   <span id="born_amplitudes"></span>

.. autofunction:: born_amplitudes

.. raw:: html

   <span id="potential_spectra"></span>

.. autofunction:: potential_spectra

.. raw:: html

   <span id="gamma_wavelength"></span>

.. autofunction:: gamma_wavelength

.. raw:: html

   <span id="validated_atomic_number"></span>

.. autofunction:: validated_atomic_number

.. raw:: html

   <span id="potential_per_amplitude"></span>

.. autodata:: POTENTIAL_PER_AMPLITUDE

```

## Sampling grids and pass bands

```{eval-rst}
.. currentmodule:: scatterem.physics.periodic_sampling

.. autosummary::
   :nosignatures:

   PeriodicGrid
   dft_sample_frequencies
   CircularPassBand
   apply_pass_band
   ellipse_mask
   lowpass_ellipse
   MaskCacheStats
   mask_cache_stats
   clear_mask_cache

.. raw:: html

   <span id="periodicgrid"></span>
   <span id="periodicgrid-from_pixel_sizes"></span>

.. autoclass:: PeriodicGrid
   :members: from_pixel_sizes

.. raw:: html

   <span id="dft_sample_frequencies"></span>

.. autofunction:: dft_sample_frequencies

.. raw:: html

   <span id="circularpassband"></span>
   <span id="circularpassband-mask"></span>
   <span id="circularpassband-min_crop_shape"></span>

.. autoclass:: CircularPassBand
   :members: mask, min_crop_shape

.. raw:: html

   <span id="apply_pass_band"></span>

.. autofunction:: apply_pass_band

.. raw:: html

   <span id="ellipse_mask"></span>

.. autofunction:: ellipse_mask

.. raw:: html

   <span id="lowpass_ellipse"></span>

.. autofunction:: lowpass_ellipse

.. raw:: html

   <span id="maskcachestats"></span>

.. autoclass:: MaskCacheStats
   :no-members:

.. raw:: html

   <span id="mask_cache_stats"></span>

.. autofunction:: mask_cache_stats

.. raw:: html

   <span id="clear_mask_cache"></span>

.. autofunction:: clear_mask_cache

```

## STEM-EELS and PRISM driver

```{eval-rst}
.. currentmodule:: scatterem.simulation.eels

.. autosummary::
   :nosignatures:

   PrismEelsSimulator
   PrismResult

.. raw:: html

   <span id="prismeelssimulator"></span>
   <span id="prismeelssimulator-eels_map"></span>
   <span id="prismeelssimulator-qeels"></span>
   <span id="prismeelssimulator-eels_cube"></span>
   <span id="prismeelssimulator-elastic_cbed"></span>
   <span id="prismeelssimulator-probe_q"></span>
   <span id="prismeelssimulator-scan_pixels"></span>
   <span id="prismeelssimulator-transmissions"></span>
   <span id="prismeelssimulator-sites"></span>
   <span id="prismeelssimulator-transition_potentials"></span>

.. autoclass:: PrismEelsSimulator
   :members: eels_map, qeels, eels_cube, elastic_cbed, probe_q, scan_pixels, transmissions, sites, transition_potentials

.. raw:: html

   <span id="prismresult"></span>

.. autoclass:: PrismResult
   :no-members:

```

## Transition potentials

```{eval-rst}
.. currentmodule:: scatterem.simulation.eels

.. autosummary::
   :nosignatures:

   subshell_transitions
   build_transition_potentials
   Transition
   TransitionPotentials
   AtomicRadialWavefunction
   bound_wavefunction
   continuum_wavefunction
   gpaw_available
   cache_dir
   HydrogenicFallbackWarning

.. raw:: html

   <span id="subshell_transitions"></span>

.. autofunction:: subshell_transitions

.. raw:: html

   <span id="build_transition_potentials"></span>

.. autofunction:: build_transition_potentials

.. raw:: html

   <span id="transition"></span>

.. autoclass:: Transition
   :no-members:

.. raw:: html

   <span id="transitionpotentials"></span>

.. autoclass:: TransitionPotentials
   :no-members:

.. raw:: html

   <span id="atomicradialwavefunction"></span>

.. autoclass:: AtomicRadialWavefunction
   :no-members:

.. raw:: html

   <span id="bound_wavefunction"></span>

.. autofunction:: bound_wavefunction

.. raw:: html

   <span id="continuum_wavefunction"></span>

.. autofunction:: continuum_wavefunction

.. raw:: html

   <span id="gpaw_available"></span>

.. autofunction:: gpaw_available

.. raw:: html

   <span id="cache_dir"></span>

.. autofunction:: cache_dir

.. raw:: html

   <span id="hydrogenicfallbackwarning"></span>

.. autoclass:: HydrogenicFallbackWarning
   :no-members:

```

## EELS algorithms

```{eval-rst}
.. currentmodule:: scatterem.simulation.eels

.. autosummary::
   :nosignatures:

   multislice_transition_potential
   prism_transition_potential
   prism_transition_potential_hybrid
   prism_eels_image
   ScatteringMatrix
   PartitionedScatteringMatrix
   DetectorExitSMatrix

.. raw:: html

   <span id="multislice_transition_potential"></span>

.. autofunction:: multislice_transition_potential

.. raw:: html

   <span id="prism_transition_potential"></span>

.. autofunction:: prism_transition_potential

.. raw:: html

   <span id="prism_transition_potential_hybrid"></span>

.. autofunction:: prism_transition_potential_hybrid

.. raw:: html

   <span id="prism_eels_image"></span>

.. autofunction:: prism_eels_image

.. raw:: html

   <span id="scatteringmatrix"></span>

.. autoclass:: ScatteringMatrix
   :no-members:

.. raw:: html

   <span id="partitionedscatteringmatrix"></span>

.. autoclass:: PartitionedScatteringMatrix
   :no-members:

.. raw:: html

   <span id="detectorexitsmatrix"></span>

.. autoclass:: DetectorExitSMatrix
   :no-members:

```

## Elastic PRISM

```{eval-rst}
.. currentmodule:: scatterem.nn.scan_exit_wave

.. autosummary::
   :nosignatures:

   ScanExitWaveOperator
   PrismScanOp
   MultisliceScanOp
   build_prism_smeta
   build_prism_smeta_for_probe

.. raw:: html

   <span id="scanexitwaveoperator"></span>

.. autoclass:: ScanExitWaveOperator
   :no-members:

.. raw:: html

   <span id="prismscanop"></span>

.. autoclass:: PrismScanOp
   :no-members:

.. raw:: html

   <span id="multislicescanop"></span>

.. autoclass:: MultisliceScanOp
   :no-members:

.. raw:: html

   <span id="build_prism_smeta"></span>

.. autofunction:: build_prism_smeta

.. raw:: html

   <span id="build_prism_smeta_for_probe"></span>

.. autofunction:: build_prism_smeta_for_probe

```

## Propagation

```{eval-rst}
.. currentmodule:: scatterem.nn.modules

.. autosummary::
   :nosignatures:

   Propagator
   SubpixShift

.. raw:: html

   <span id="propagator"></span>

.. autoclass:: Propagator
   :no-members:

.. raw:: html

   <span id="subpixshift"></span>

.. autoclass:: SubpixShift
   :no-members:

```

## Slice transmission in ptychography

```{eval-rst}
.. currentmodule:: scatterem.nn.ptychography

.. autosummary::
   :nosignatures:

   multislice_transmit_slice

.. raw:: html

   <span id="multislice_transmit_slice"></span>

.. autofunction:: multislice_transmit_slice

```

## Beamlet sampling

```{eval-rst}
.. currentmodule:: scatterem.utils.stem

.. autosummary::
   :nosignatures:

   beamlet_samples
   natural_neighbor_weights
   fftshift_checkerboard

.. raw:: html

   <span id="beamlet_samples"></span>

.. autofunction:: beamlet_samples

.. raw:: html

   <span id="natural_neighbor_weights"></span>

.. autofunction:: natural_neighbor_weights

.. raw:: html

   <span id="fftshift_checkerboard"></span>

.. autofunction:: fftshift_checkerboard

```

## Free-space propagation and probe shifts

```{eval-rst}
.. currentmodule:: scatterem.physics.propagation

.. autosummary::
   :nosignatures:

   fourier_shift_stack
   propagate
   propagator_kernel
   propagator_phase
   angular_spectrum_kernel
   transfer_phase
   tilt_phase

.. raw:: html

   <span id="fourier_shift_stack"></span>

.. autofunction:: fourier_shift_stack

.. raw:: html

   <span id="propagate"></span>

.. autofunction:: propagate

.. raw:: html

   <span id="propagator_kernel"></span>

.. autofunction:: propagator_kernel

.. raw:: html

   <span id="propagator_phase"></span>

.. autofunction:: propagator_phase

.. raw:: html

   <span id="angular_spectrum_kernel"></span>

.. autofunction:: angular_spectrum_kernel

.. raw:: html

   <span id="transfer_phase"></span>

.. autofunction:: transfer_phase

.. raw:: html

   <span id="tilt_phase"></span>

.. autofunction:: tilt_phase

```

## Grids

```{eval-rst}
.. currentmodule:: scatterem.utils.spectral_mesh

.. autosummary::
   :nosignatures:

   dual_mesh
   shell_means
   ShellMeans

.. raw:: html

   <span id="dual_mesh"></span>

.. autofunction:: dual_mesh

.. raw:: html

   <span id="shell_means"></span>

.. autofunction:: shell_means

.. raw:: html

   <span id="shellmeans"></span>

.. autoclass:: ShellMeans
   :no-members:

```

## Visualisation

```{eval-rst}
.. currentmodule:: scatterem.vis.visualization

.. autosummary::
   :nosignatures:

   show_2d
   show_2d_array

.. raw:: html

   <span id="show_2d"></span>

.. autofunction:: show_2d

.. raw:: html

   <span id="show_2d_array"></span>

.. autofunction:: show_2d_array

```

## EELS radial states

```{eval-rst}
.. currentmodule:: scatterem.simulation.eels.physics.radial

.. autosummary::
   :nosignatures:

   resolve_wavefunctions
   hydrogenic_bound_wavefunction
   free_continuum_wavefunction

.. raw:: html

   <span id="resolve_wavefunctions"></span>

.. autofunction:: resolve_wavefunctions

.. raw:: html

   <span id="hydrogenic_bound_wavefunction"></span>

.. autofunction:: hydrogenic_bound_wavefunction

.. raw:: html

   <span id="free_continuum_wavefunction"></span>

.. autofunction:: free_continuum_wavefunction

```

## Crystal phases

```{eval-rst}
.. currentmodule:: scatterem.crystal.phase

.. autosummary::
   :nosignatures:

   Phase

.. raw:: html

   <span id="phase"></span>

.. autoclass:: Phase
   :no-members:

```

## Elements

```{eval-rst}
.. currentmodule:: scatterem.crystal.elements

.. autosummary::
   :nosignatures:

   atomic_number
   atomic_weight
   symbol

.. raw:: html

   <span id="atomic_number"></span>

.. autofunction:: atomic_number

.. raw:: html

   <span id="atomic_weight"></span>

.. autofunction:: atomic_weight

.. raw:: html

   <span id="symbol"></span>

.. autofunction:: symbol

```

## Fourier helpers

```{eval-rst}
.. currentmodule:: scatterem.physics.fourier

.. autosummary::
   :nosignatures:

   phase_ramp
   fourier_shift
   fourier_rescale
   dft_center
   fftshift_last2
   fftshift_rfft
   ifftshift_rfft
   rfft_shape

.. raw:: html

   <span id="phase_ramp"></span>

.. autofunction:: phase_ramp

.. raw:: html

   <span id="fourier_shift"></span>

.. autofunction:: fourier_shift

.. raw:: html

   <span id="fourier_rescale"></span>

.. autofunction:: fourier_rescale

.. raw:: html

   <span id="dft_center"></span>

.. autofunction:: dft_center

.. raw:: html

   <span id="fftshift_last2"></span>

.. autofunction:: fftshift_last2

.. raw:: html

   <span id="fftshift_rfft"></span>

.. autofunction:: fftshift_rfft

.. raw:: html

   <span id="ifftshift_rfft"></span>

.. autofunction:: ifftshift_rfft

.. raw:: html

   <span id="rfft_shape"></span>

.. autofunction:: rfft_shape

```

## Axis-angle rotations

```{eval-rst}
.. currentmodule:: scatterem.rotation.axis_angle

.. autosummary::
   :nosignatures:

   to_matrix

.. raw:: html

   <span id="to_matrix"></span>

.. autofunction:: to_matrix

```

## Quaternions

```{eval-rst}
.. currentmodule:: scatterem.rotation.quaternion

.. autosummary::
   :nosignatures:

   from_axis_angle
   from_matrix
   from_rotation_vector
   multiply
   to_matrix

.. raw:: html

   <span id="from_axis_angle"></span>

.. autofunction:: from_axis_angle

.. raw:: html

   <span id="from_matrix"></span>

.. autofunction:: from_matrix

.. raw:: html

   <span id="from_rotation_vector"></span>

.. autofunction:: from_rotation_vector

.. raw:: html

   <span id="multiply"></span>

.. autofunction:: multiply

.. raw:: html

   <span id="scatterem-rotation-quaternion-to_matrix"></span>

.. autofunction:: to_matrix

```

## Display and animation

```{eval-rst}
.. currentmodule:: scatterem.vis

.. autosummary::
   :nosignatures:

   show
   show_animated
   save_animated

.. raw:: html

   <span id="show"></span>

.. autofunction:: show

.. raw:: html

   <span id="show_animated"></span>

.. autofunction:: show_animated

.. raw:: html

   <span id="save_animated"></span>

.. autofunction:: save_animated

```

## Colour maps and normalisation

```{eval-rst}
.. currentmodule:: scatterem.vis

.. autosummary::
   :nosignatures:

   choose_colormap
   colormap_for_role
   register_colormaps
   ColourChoice
   DataTraits
   NormalizationConfig
   CustomNormalization

.. raw:: html

   <span id="choose_colormap"></span>

.. autofunction:: choose_colormap

.. raw:: html

   <span id="colormap_for_role"></span>

.. autofunction:: colormap_for_role

.. raw:: html

   <span id="register_colormaps"></span>

.. autofunction:: register_colormaps

.. raw:: html

   <span id="colourchoice"></span>

.. autoclass:: ColourChoice
   :no-members:

.. raw:: html

   <span id="datatraits"></span>

.. autoclass:: DataTraits
   :no-members:

.. raw:: html

   <span id="normalizationconfig"></span>

.. autoclass:: NormalizationConfig
   :no-members:

.. raw:: html

   <span id="customnormalization"></span>

.. autoclass:: CustomNormalization
   :no-members:

```

## Figure style and scale bars

```{eval-rst}
.. currentmodule:: scatterem.vis

.. autosummary::
   :nosignatures:

   use_style
   figure_params
   draw_scale_bar
   ScalebarConfig
   UncalibratedDataWarning

.. raw:: html

   <span id="use_style"></span>

.. autofunction:: use_style

.. raw:: html

   <span id="figure_params"></span>

.. autofunction:: figure_params

.. raw:: html

   <span id="draw_scale_bar"></span>

.. autofunction:: draw_scale_bar

.. raw:: html

   <span id="scalebarconfig"></span>

.. autoclass:: ScalebarConfig
   :no-members:

.. raw:: html

   <span id="uncalibrateddatawarning"></span>

.. autoclass:: UncalibratedDataWarning
   :no-members:

```

## Colour

```{eval-rst}
.. currentmodule:: scatterem.vis.complex_color

.. autosummary::
   :nosignatures:

   complex_to_rgba
   phase_wheel
   tile_to_rgba

.. raw:: html

   <span id="complex_to_rgba"></span>

.. autofunction:: complex_to_rgba

.. raw:: html

   <span id="phase_wheel"></span>

.. autofunction:: phase_wheel

.. raw:: html

   <span id="tile_to_rgba"></span>

.. autofunction:: tile_to_rgba

```

## Scale bars

```{eval-rst}
.. currentmodule:: scatterem.data.sampling

.. autosummary::
   :nosignatures:

   Sampling
   reconstruction_sampling

.. raw:: html

   <span id="sampling"></span>

.. autoclass:: Sampling
   :no-members:

.. raw:: html

   <span id="reconstruction_sampling"></span>

.. autofunction:: reconstruction_sampling

```

## Saving and loading

```{eval-rst}
.. currentmodule:: scatterem.utils.store

.. autosummary::
   :nosignatures:

   Serializable
   load_serializable
   read_schema
   SerializationError
   SCHEMA_VERSION

.. raw:: html

   <span id="serializable"></span>

.. autoclass:: Serializable
   :no-members:

.. raw:: html

   <span id="load_serializable"></span>

.. autofunction:: load_serializable

.. raw:: html

   <span id="read_schema"></span>

.. autofunction:: read_schema

.. raw:: html

   <span id="serializationerror"></span>

.. autoclass:: SerializationError
   :no-members:

.. raw:: html

   <span id="schema_version"></span>

.. autodata:: SCHEMA_VERSION

```

## Supporting modules

<span id="scatterem.data.detector_warp"></span>

### `scatterem.data.detector_warp`

```{eval-rst}
.. currentmodule:: scatterem.data.detector_warp

.. autosummary::
   :nosignatures:

   as_detector_warp
   bright_field_k
   detector_warp_matrix
   polynomial_warp
   warp_table

.. raw:: html

   <span id="as_detector_warp"></span>

.. autofunction:: as_detector_warp

.. raw:: html

   <span id="bright_field_k"></span>

.. autofunction:: bright_field_k

.. raw:: html

   <span id="detector_warp_matrix"></span>

.. autofunction:: detector_warp_matrix

.. raw:: html

   <span id="polynomial_warp"></span>

.. autofunction:: polynomial_warp

.. raw:: html

   <span id="warp_table"></span>

.. autofunction:: warp_table

```

<span id="scatterem.data.geometry"></span>

### `scatterem.data.geometry`

```{eval-rst}
.. currentmodule:: scatterem.data.geometry

.. autosummary::
   :nosignatures:

   Geometry

.. raw:: html

   <span id="geometry"></span>

.. autoclass:: Geometry
   :no-members:

```

<span id="scatterem.nn.regularizers.tv"></span>

### `scatterem.nn.regularizers.tv`

```{eval-rst}
.. currentmodule:: scatterem.nn.regularizers.tv

.. autosummary::
   :nosignatures:

   anisotropic_tv_subgradient
   isotropic_tv_subgradient
   mean_abs_forward_diff
   total_variation_2d
   total_variation_3d

.. raw:: html

   <span id="anisotropic_tv_subgradient"></span>

.. autofunction:: anisotropic_tv_subgradient

.. raw:: html

   <span id="isotropic_tv_subgradient"></span>

.. autofunction:: isotropic_tv_subgradient

.. raw:: html

   <span id="mean_abs_forward_diff"></span>

.. autofunction:: mean_abs_forward_diff

.. raw:: html

   <span id="total_variation_2d"></span>

.. autofunction:: total_variation_2d

.. raw:: html

   <span id="total_variation_3d"></span>

.. autofunction:: total_variation_3d

```

<span id="scatterem.physics.kernels.lens_wavefront_wp"></span>

### `scatterem.physics.kernels.lens_wavefront_wp`

```{eval-rst}
.. currentmodule:: scatterem.physics.kernels.lens_wavefront_wp

.. autosummary::
   :nosignatures:

   lens_to_warp
   lens_wavefront_f32
   lens_wavefront_focus_f32

.. raw:: html

   <span id="lens_to_warp"></span>

.. autofunction:: lens_to_warp

.. raw:: html

   <span id="lens_wavefront_f32"></span>

.. autofunction:: lens_wavefront_f32

.. raw:: html

   <span id="lens_wavefront_focus_f32"></span>

.. autofunction:: lens_wavefront_focus_f32

```

<span id="scatterem.reconstruction.darkfield.core"></span>

### `scatterem.reconstruction.darkfield.core`

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.darkfield.core

.. autosummary::
   :nosignatures:

   compute_ssnr_from_halfset_images
   gaussian_blur1d

.. raw:: html

   <span id="compute_ssnr_from_halfset_images"></span>

.. autofunction:: compute_ssnr_from_halfset_images

.. raw:: html

   <span id="gaussian_blur1d"></span>

.. autofunction:: gaussian_blur1d

```

<span id="scatterem.reconstruction.darkfield.drizzle"></span>

### `scatterem.reconstruction.darkfield.drizzle`

```{eval-rst}
.. currentmodule:: scatterem.reconstruction.darkfield.drizzle

.. autosummary::
   :nosignatures:

   drizzle_resample

.. raw:: html

   <span id="drizzle_resample"></span>

.. autofunction:: drizzle_resample

```

<span id="scatterem.vis.colors"></span>

### `scatterem.vis.colors`

```{eval-rst}
.. currentmodule:: scatterem.vis.colors

.. autosummary::
   :nosignatures:

   UnscientificColormapWarning
   resolve_display
   simulate_cvd
   srgb_to_lab
   warn_if_unscientific

.. raw:: html

   <span id="unscientificcolormapwarning"></span>

.. autoclass:: UnscientificColormapWarning
   :no-members:

.. raw:: html

   <span id="resolve_display"></span>

.. autofunction:: resolve_display

.. raw:: html

   <span id="simulate_cvd"></span>

.. autofunction:: simulate_cvd

.. raw:: html

   <span id="srgb_to_lab"></span>

.. autofunction:: srgb_to_lab

.. raw:: html

   <span id="warn_if_unscientific"></span>

.. autofunction:: warn_if_unscientific

```

<span id="scatterem.vis.normalization"></span>

### `scatterem.vis.normalization`

```{eval-rst}
.. currentmodule:: scatterem.vis.normalization

.. autosummary::
   :nosignatures:

   NORMALIZATION_PRESETS
   INTERVAL_TYPES
   STRETCH_TYPES

.. raw:: html

   <span id="normalization_presets"></span>

.. autodata:: NORMALIZATION_PRESETS
   :no-value:

.. raw:: html

   <span id="interval_types"></span>

.. autodata:: INTERVAL_TYPES

.. raw:: html

   <span id="stretch_types"></span>

.. autodata:: STRETCH_TYPES

```

<span id="scatterem.vis.scale_bar"></span>

### `scatterem.vis.scale_bar`

```{eval-rst}
.. currentmodule:: scatterem.vis.scale_bar

.. autosummary::
   :nosignatures:

   nice_bar_length

.. raw:: html

   <span id="nice_bar_length"></span>

.. autofunction:: nice_bar_length

```

<span id="scatterem.vis.show"></span>

### `scatterem.vis.show`

```{eval-rst}
.. currentmodule:: scatterem.vis.show

.. autosummary::
   :nosignatures:

   Animated
   FigureContext
   FrameSource
   Panel
   ShownFigure
   register

.. raw:: html

   <span id="animated"></span>

.. autoclass:: Animated
   :no-members:

.. raw:: html

   <span id="figurecontext"></span>

.. autoclass:: FigureContext
   :no-members:

.. raw:: html

   <span id="framesource"></span>

.. autoclass:: FrameSource
   :no-members:

.. raw:: html

   <span id="panel"></span>

.. autoclass:: Panel
   :no-members:

.. raw:: html

   <span id="shownfigure"></span>

.. autoclass:: ShownFigure
   :no-members:

.. raw:: html

   <span id="register"></span>

.. autofunction:: register

```

<span id="scatterem.vis.style"></span>

### `scatterem.vis.style`

```{eval-rst}
.. currentmodule:: scatterem.vis.style

.. autosummary::
   :nosignatures:

   line_colours
   style_params

.. raw:: html

   <span id="line_colours"></span>

.. autofunction:: line_colours

.. raw:: html

   <span id="style_params"></span>

.. autofunction:: style_params

```

<span id="scatterem.vis.visualization_utils"></span>

### `scatterem.vis.visualization_utils`

```{eval-rst}
.. currentmodule:: scatterem.vis.visualization_utils

.. autosummary::
   :nosignatures:

   add_cbar_to_ax
   add_colorbars
   special_ticks
   add_scalebar_to_ax
   estimate_scalebar_length

.. raw:: html

   <span id="add_cbar_to_ax"></span>

.. autofunction:: add_cbar_to_ax

.. raw:: html

   <span id="add_colorbars"></span>

.. autofunction:: add_colorbars

.. raw:: html

   <span id="special_ticks"></span>

.. autofunction:: special_ticks

.. raw:: html

   <span id="add_scalebar_to_ax"></span>

.. autofunction:: add_scalebar_to_ax

.. raw:: html

   <span id="estimate_scalebar_length"></span>

.. autofunction:: estimate_scalebar_length

```
