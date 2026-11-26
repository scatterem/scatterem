# Cite

Cite the paper for the part you used. Each method of the package comes with
its own paper.

## The papers

The reconstruction methods (FF-STEM: direct ptychography, tilt-corrected dark
field and their fusion) are described in:

> S. You, G. Varnavides, S. Khavnekar, N. Palatkin, S. Shao, M. Wu, D. Stroppa,
> D. Chernikova, B. Zhu, R. Egoavil, S. Vespucci, D. Krishnan, X. Ye,
> F. K. M. Schur, E. Spiecker, P. Pelz.
> *Gap-Free Information Transfer in 4D-STEM via Fusion of Complementary
> Scattering Channels.* **Advanced Science** 13 (2026) e76620.
> [doi:10.1002/advs.76620](https://doi.org/10.1002/advs.76620)

The STEM-EELS simulation (BiP-PRISM) and the elastic PRISM code are described
in:

> P. Pelz. *The bi-partitioned PRISM (BiP-PRISM) algorithm for fast and scalable
> core-loss STEM-EELS simulations.* **Journal of Physics: Materials** (2026).
> [doi:10.1088/2515-7639/aebdd0](https://doi.org/10.1088/2515-7639/aebdd0)

The journal has not yet given the BiP-PRISM paper a volume or an article
number.

The mixed-state ptychography (the second-order solvers, the 2D mixed-state
model, and the probe gauges and orbital angular momentum diagnostics) is
described in:

> S. You, P. Rosi, E. Rotunno, A. Roncaglia, L. Belsito, A. H. Tavabi,
> R. E. Dunin-Borkowski, V. Grillo, P. M. Pelz.
> *Transverse quantum-state characterization of programmable electron
> optics.* arXiv:2608.05749 (2026).
> [doi:10.48550/arXiv.2608.05749](https://doi.org/10.48550/arXiv.2608.05749)

This paper is a preprint. Cite the journal version when it is published.

## BibTeX

The entries below hold the metadata of `CITATION.cff` in the repository.

```bibtex
@article{you2026ffstem,
  author  = {You, Shengbo and Varnavides, Georgios and Khavnekar, Sagar and
             Palatkin, Nikita and Shao, Sihan and Wu, Mingjian and
             Stroppa, Daniel and Chernikova, Darya and Zhu, Baixu and
             Egoavil, Ricardo and Vespucci, Stefano and Krishnan, Dileep and
             Ye, Xingchen and Schur, Florian K. M. and Spiecker, Erdmann and
             Pelz, Philipp},
  title   = {Gap-Free Information Transfer in {4D-STEM} via Fusion of
             Complementary Scattering Channels},
  journal = {Advanced Science},
  year    = {2026},
  volume  = {13},
  number  = {56},
  pages   = {e76620},
  doi     = {10.1002/advs.76620}
}

@article{pelz2026bipprism,
  author  = {Pelz, Philipp},
  title   = {The bi-partitioned {PRISM} ({BiP-PRISM}) algorithm for fast and
             scalable core-loss {STEM-EELS} simulations},
  journal = {Journal of Physics: Materials},
  year    = {2026},
  doi     = {10.1088/2515-7639/aebdd0}
}

@article{you2026transverse,
  author  = {You, Shengbo and Rosi, Paolo and Rotunno, Enzo and
             Roncaglia, Alberto and Belsito, Luca and Tavabi, Amir H. and
             Dunin-Borkowski, Rafal E. and Grillo, Vincenzo and
             Pelz, Philipp M.},
  title   = {Transverse quantum-state characterization of programmable
             electron optics},
  journal = {arXiv preprint arXiv:2608.05749},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.05749}
}
```

## The software

Zenodo archives every release of scatterem. The concept DOI
[10.5281/zenodo.22162829](https://doi.org/10.5281/zenodo.22162829) always
resolves to the latest release. Each release also gets its own version DOI;
the Zenodo record lists it. Cite the version DOI when you report numbers that
depend on the release.

`CITATION.cff` in the repository holds the same metadata in machine-readable
form. GitHub shows it as "Cite this repository".

## The data

The FF-STEM datasets (the four specimens of the paper) are on Zenodo under
[10.5281/zenodo.18008901](https://doi.org/10.5281/zenodo.18008901), with the
licence CC-BY-4.0. If you use them, cite the record as well as the paper.

## Scope

Four things live here, and each comes with its own paper:

- the FF-STEM reconstruction pipeline: direct ptychography, tilt-corrected dark
  field, their fusion, and the calibration and I/O they need;
- the core-loss STEM-EELS forward simulation (BiP-PRISM);
- elastic PRISM simulation of 4D-STEM;
- mixed-state ptychography with second-order solvers, and the probe and
  orbital angular momentum diagnostics.

The wider `scatterem` codebase covers more. Those parts come with their own
publications and are released with them.
