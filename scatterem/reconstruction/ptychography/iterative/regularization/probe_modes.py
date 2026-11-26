"""Probe-mode access and orthogonalisation, with no dependency on the model.

:func:`orthogonalize_probe` is the ``regularization:`` probe constraint of the
unified ``Ptychography`` module (re-exported by its constraint module). It reads the probe through
``model.probe_model``, so any object with that attribute works; the 2D
second-order model
(:class:`~scatterem.reconstruction.ptychography.iterative.second_order.planar.PlanarPtychography`)
uses it too.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["orthogonalize_probe"]


def _get_probe_parameter(model: Any, probe_index: int) -> torch.Tensor:
    """Return a probe tensor from either a ParameterList or ParameterDict."""
    probe_model = model.probe_model
    if isinstance(probe_model, torch.nn.ParameterList):
        return probe_model[probe_index]

    if isinstance(probe_model, torch.nn.ParameterDict):
        key = str((probe_index, 0))
        if key not in probe_model:
            key = str((0, 0))
        return probe_model[key]

    return probe_model[probe_index]


def orthogonalize_probe(
    model: Any, regul_dict: dict[str, Any], probe_index: int = 0
) -> torch.Tensor:
    """Orthogonalize the selected probe modes by SVD, dominant mode at index 0.

    The highest-intensity mode is placed at index 0 (the coherent-mode
    convention). ``torch.linalg.eigh`` returns ascending eigenvalues, which alone
    would pin the WEAKEST mode at index 0 and make the dominant mode hop slots
    each call; a stable dominant-first ordering also keeps the per-element
    optimizer (Adam) state aligned with mode content across iterations.

    Generic over the K (probe-broadcast) axis: modes are orthogonalized as full
    ``(K, M1, M2)`` vectors, so the returned tensor keeps the probe-parameter
    shape ``(Nmodes, K, M1, M2)`` for any K (the previous ``squeeze(1)`` path
    only handled K == 1 and broke the write-back for PRISM probes).
    """
    del regul_dict

    modes = _get_probe_parameter(model, probe_index)
    orig_modes_dtype = modes.dtype
    if orig_modes_dtype != torch.complex64:
        modes = torch.complex(modes, torch.zeros_like(modes))

    input_shape = modes.shape
    modes_reshaped = modes.reshape(input_shape[0], -1)

    A = torch.matmul(modes_reshaped, modes_reshaped.conj().t())
    evals, evecs = torch.linalg.eigh(A)

    ortho_modes = torch.matmul(evecs.conj().t(), modes_reshaped).reshape(input_shape)

    # eigh is ascending -> sort ALL modes by descending integrated intensity so
    # the dominant coherent mode is index 0 and the ordering stays consistent.
    mode_int = ortho_modes.abs().pow(2).flatten(1).sum(1)
    order = torch.argsort(mode_int, descending=True)
    ortho_modes = ortho_modes[order]

    probe_pow = (mode_int[order] / mode_int.sum()).detach().cpu().numpy().round(3)
    logger.info(
        "Apply ortho pmode constraint for probe %s, "
        "relative pmode power = %s, probe int sum = %.4f",
        probe_index,
        probe_pow,
        mode_int.sum(),
    )

    return ortho_modes.to(orig_modes_dtype)
