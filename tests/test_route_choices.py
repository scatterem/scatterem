"""Shared input checks, device rule, route choice and real-space settings."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scatterem.physics.atom_set import AtomSet
from scatterem.physics.potential_rolloff import DEFAULT_ROLLOFF
from scatterem.physics.potential_stack import check_stack_request, resolve_stack_device
from scatterem.physics.route_settings import (
    DEFAULT_PROJECTED_CUTOFF,
    SpatialRouteSettings,
    check_cutoff,
    check_route,
    choose_route,
    route_rolloff,
)


def atoms():
    return AtomSet([3.0, 4.5, 2.0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [6, 8])


def test_request_values():
    req = check_stack_request((np.int64(30), 40), atoms(), replicas=(2, 3))
    assert req.replicas == (2, 3) and req.points == (30, 40)
    assert req.lengths == (6.0, 13.5) and all(type(v) is int for v in req.points)
    assert req.real_dtype == torch.get_default_dtype()
    # Default replicas=None means one copy per axis.
    one = check_stack_request((4, 4), atoms())
    assert one.replicas == (1, 1) and one.lengths == (3.0, 4.5)
    assert check_stack_request((4, 4), atoms(), replicas=None).replicas == (1, 1)
    assert (
        check_stack_request((4, 4), atoms(), dtype=torch.complex128).real_dtype
        == torch.float64
    )
    assert (
        check_stack_request((4, 4), atoms(), dtype=torch.complex64).real_dtype
        == torch.float32
    )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"atom_set": None}, TypeError),
        ({"points": (4, 4, 4)}, ValueError),
        ({"points": (4, 0)}, ValueError),
        ({"points": (4, False)}, TypeError),
        ({"points": (4, 4.0)}, TypeError),
        ({"points": 4}, TypeError),
        ({"replicas": (0, 1)}, ValueError),
        ({"replicas": ("1", 1)}, TypeError),
        ({"atom_scale": [1j, 1]}, TypeError),
        ({"atom_scale": ["x", "y"]}, TypeError),
        ({"atom_scale": [1.0]}, ValueError),
        ({"atom_scale": [1.0, np.inf]}, ValueError),
        ({"scale_squared": "yes"}, TypeError),
        ({"dtype": torch.float16}, TypeError),
    ],
)
def test_request_refusals(kwargs, error):
    args = {"atom_set": atoms(), "points": (4, 4)}
    args.update(kwargs)
    with pytest.raises(error):
        check_stack_request(args.pop("points"), args.pop("atom_set"), **args)


def test_device_rule():
    assert resolve_stack_device(None) == torch.device("cpu")
    assert resolve_stack_device("meta").type == "meta"
    assert resolve_stack_device(None, torch.zeros(2, device="meta")).type == "meta"
    assert resolve_stack_device(None, np.zeros(2), torch.zeros(2)).type == "cpu"


def test_device_rule_refuses_two_devices(require_cuda):
    with pytest.raises(ValueError):
        resolve_stack_device(None, torch.zeros(2), torch.zeros(2, device=require_cuda))
    assert (
        resolve_stack_device("cpu", torch.zeros(2, device=require_cuda)).type == "cpu"
    )


def test_route_choice():
    assert choose_route(multi_device=True) == "spatial"
    assert choose_route(multi_device=False, preference="either") == "spectral"
    assert choose_route(multi_device=False, preference="spatial") == "spatial"
    assert choose_route(multi_device=True, preference="spectral") == "spectral"
    for bad, error in (
        ("fft", ValueError),
        ("auto", ValueError),
        (3, TypeError),
        (None, TypeError),
    ):
        with pytest.raises(error):
            choose_route(multi_device=False, preference=bad)
    with pytest.raises(ValueError):
        check_route("either")
    with pytest.raises(TypeError):
        choose_route(multi_device=1, preference="either")
    with pytest.raises(TypeError):
        choose_route("either", multi_device=False)


def test_route_rolloff_table():
    assert route_rolloff(route="spectral") == DEFAULT_ROLLOFF
    assert route_rolloff(route="spectral", preference="either") == DEFAULT_ROLLOFF
    assert route_rolloff(route="spectral", preference=None) is None
    assert route_rolloff(route="spectral", preference=0.2) == 0.2
    assert route_rolloff(route="spatial") is None
    assert route_rolloff(route="spatial", preference=None) is None
    with pytest.raises(ValueError, match="thin_slice_transmission"):
        route_rolloff(route="spatial", preference=0.2)
    for bad in ("fine", "auto"):
        with pytest.raises(ValueError):
            route_rolloff(route="spectral", preference=bad)
    with pytest.raises(ValueError):
        route_rolloff(route="spectral", preference=1.5)
    with pytest.raises(ValueError):
        route_rolloff(route="fft")
    with pytest.raises(TypeError):
        route_rolloff("spectral")


def test_spatial_settings():
    a = SpatialRouteSettings(projected_cutoff=0.01)
    b = SpatialRouteSettings(projected_cutoff=np.float64(0.01))
    assert a == b and hash(a) == hash(b)
    assert SpatialRouteSettings().projected_cutoff == DEFAULT_PROJECTED_CUTOFF
    s = SpatialRouteSettings("sectioned", volume_cutoff=np.float32(0.5))
    assert type(s.volume_cutoff) is float and hash(s) == hash(
        SpatialRouteSettings("sectioned", volume_cutoff=0.5)
    )
    for bad in (True, "0.1", 1j):
        with pytest.raises(TypeError):
            SpatialRouteSettings(projected_cutoff=bad)
        with pytest.raises(TypeError):
            check_cutoff(bad, field_label="x")
    for bad in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            SpatialRouteSettings(projected_cutoff=bad)
    with pytest.raises(ValueError):
        SpatialRouteSettings("stacked")
    with pytest.raises(TypeError):
        SpatialRouteSettings(3)
    with pytest.raises(ValueError):
        SpatialRouteSettings("projected", volume_cutoff=0.5)
