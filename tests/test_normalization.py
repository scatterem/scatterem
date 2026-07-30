"""Tests for scatterem.vis.normalization.

Three groups:

* textbook checks -- hand-computed stretch values, and the structural properties every
  stretch must have ([0, 1] -> [0, 1], monotonically non-decreasing);
* degenerate-input checks -- constant images, all-NaN input, NaN mixed with finite
  values, and inverted limits;
* a regression guard -- agreement with an independent closed-form oracle, plus frozen
  numeric anchors, for the configurations the published figure scripts actually use, so
  that those figures cannot move.

On the regression guard: it originally compared against the module this one replaced.
That module has since been deleted, so comparing against it is no longer possible and a
skipped guard would be no guard at all. The comparison is instead against
:func:`_oracle` below -- limits, clip and stretch written out directly from the
documented formulas, sharing no code with the implementation under test -- and against
literal values in :data:`FROZEN`. Both were checked against the deleted module while it
still existed: the oracle reproduced it to a maximum absolute difference of exactly 0.0
over all thirteen configurations exercised here, and the frozen anchors are its output.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from matplotlib import ticker

from scatterem.vis.normalization import (
    NORMALIZATION_PRESETS,
    STRETCH_TYPES,
    CustomNormalization,
    NormalizationConfig,
    _resolve_normalization,
)

# A fixed array, deliberately wider than [0, 1] so that clipping is exercised.
RNG = np.random.default_rng(20260729)
FIXED_ARRAY = RNG.normal(size=(32, 32)) * 3.0 + 1.0


def _oracle_limits(
    data,
    interval_type="quantile",
    lower_quantile=0.02,
    upper_quantile=0.98,
    vmin=None,
    vmax=None,
    vcenter=0.0,
    half_range=None,
    **_,
):
    """The interval limits, written out from the definitions, independent of the
    implementation under test."""
    flat = np.asarray(data, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if interval_type == "manual":
        lo = finite.min() if vmin is None else vmin
        hi = finite.max() if vmax is None else vmax
    elif interval_type == "quantile":
        lo, hi = np.quantile(finite, [lower_quantile, upper_quantile])
    else:
        half = np.abs(finite - vcenter).max() if half_range is None else half_range
        lo, hi = vcenter - half, vcenter + half
    return float(lo), float(hi)


def _oracle(
    data,
    stretch_type="linear",
    power=1.0,
    logarithmic_index=1000.0,
    asinh_linear_range=0.1,
    **interval_kwargs,
):
    """Normalize *data* from first principles: limits, then clip, then stretch."""
    array = np.asarray(data, dtype=np.float64)
    lo, hi = _oracle_limits(array, **interval_kwargs)
    span = hi - lo
    if span == 0.0:
        t = np.where(np.isnan(array), np.nan, 0.0)
    else:
        t = np.clip((array - lo) / span, 0.0, 1.0)
    if stretch_type == "linear":
        return t
    if stretch_type == "power":
        return t**power
    if stretch_type == "logarithmic":
        a = logarithmic_index
        return np.log(1.0 + a * t) / np.log(1.0 + a)
    a = asinh_linear_range
    return np.arcsinh(t / a) / np.arcsinh(1.0 / a)


def _unit_stretch(stretch_type, **kwargs):
    """A normalization whose interval is exactly [0, 1], isolating the stretch."""
    return CustomNormalization(
        interval_type="manual", vmin=0.0, vmax=1.0, stretch_type=stretch_type, **kwargs
    )


# --------------------------------------------------------------------------------------
# textbook stretch values
# --------------------------------------------------------------------------------------


def test_linear_stretch_is_the_identity():
    values = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(_unit_stretch("linear")(values), values, rtol=0, atol=0)


def test_power_stretch_hand_values():
    norm = _unit_stretch("power", power=2.0)
    # t**2: 0 -> 0, 0.5 -> 0.25, 1 -> 1
    np.testing.assert_allclose(
        norm(np.array([0.0, 0.5, 1.0])), np.array([0.0, 0.25, 1.0]), rtol=0, atol=0
    )
    # p = 1/2 is the square root: 0.25 -> 0.5
    np.testing.assert_allclose(
        _unit_stretch("power", power=0.5)(np.array([0.25])),
        np.array([0.5]),
        rtol=0,
        atol=0,
    )


def test_power_stretch_with_unit_exponent_is_the_identity():
    values = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(_unit_stretch("power", power=1.0)(values), values)


def test_logarithmic_stretch_hand_values():
    # s(t) = log(1 + a t) / log(1 + a), a = 1000: s(0) = 0, s(1) = 1 exactly,
    # and s(1/2) = log(501) / log(1001).
    norm = _unit_stretch("logarithmic", logarithmic_index=1000.0)
    out = norm(np.array([0.0, 0.5, 1.0]))
    assert out[0] == 0.0
    assert out[2] == 1.0
    np.testing.assert_allclose(out[1], np.log(501.0) / np.log(1001.0), rtol=1e-15)


def test_asinh_stretch_hand_values_and_monotone_on_unit_interval():
    # s(t) = asinh(t / a) / asinh(1 / a), a = 0.1: s(0) = 0, s(1) = 1,
    # and s(1/2) = asinh(5) / asinh(10).
    norm = _unit_stretch("asinh", asinh_linear_range=0.1)
    out = norm(np.array([0.0, 0.5, 1.0]))
    assert out[0] == 0.0
    np.testing.assert_allclose(out[2], 1.0, rtol=1e-15)
    np.testing.assert_allclose(out[1], np.arcsinh(5.0) / np.arcsinh(10.0), rtol=1e-15)
    dense = norm(np.linspace(0.0, 1.0, 1001))
    assert np.all(np.diff(dense) > 0.0)


@pytest.mark.parametrize(
    "stretch_type,kwargs",
    [
        ("linear", {}),
        ("power", {"power": 2.0}),
        ("power", {"power": 0.5}),
        ("power", {"power": 3.7}),
        ("logarithmic", {"logarithmic_index": 1000.0}),
        ("logarithmic", {"logarithmic_index": 0.5}),
        ("asinh", {"asinh_linear_range": 0.1}),
        ("asinh", {"asinh_linear_range": 2.0}),
    ],
)
def test_every_stretch_maps_unit_interval_onto_itself_monotonically(
    stretch_type, kwargs
):
    t = np.linspace(0.0, 1.0, 2001)
    out = _unit_stretch(stretch_type, **kwargs)(t)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(out[-1], 1.0, rtol=1e-14)
    assert np.all(np.diff(out) >= 0.0), "stretch must be monotonically non-decreasing"


@pytest.mark.parametrize("stretch_type", STRETCH_TYPES)
def test_inverse_round_trips_in_range_data(stretch_type):
    data = np.linspace(-3.0, 7.0, 51)  # entirely inside the interval below
    norm = CustomNormalization(
        interval_type="manual", vmin=-3.0, vmax=7.0, stretch_type=stretch_type
    )
    np.testing.assert_allclose(norm.inverse(norm(data)), data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stretch_type": "power", "power": 0.0},
        {"stretch_type": "power", "power": -1.0},
        {"stretch_type": "logarithmic", "logarithmic_index": 0.0},
        {"stretch_type": "asinh", "asinh_linear_range": 0.0},
        {"stretch_type": "nonsense"},
        {"interval_type": "nonsense"},
        {"lower_quantile": 0.9, "upper_quantile": 0.1},
        {"half_range": -1.0},
    ],
)
def test_invalid_settings_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        CustomNormalization(**kwargs)


def test_unknown_option_is_a_clear_type_error():
    with pytest.raises(TypeError, match="unknown normalization option"):
        CustomNormalization(vmim=0.0)  # typo for vmin


# --------------------------------------------------------------------------------------
# interval types
# --------------------------------------------------------------------------------------


def test_manual_interval_on_known_array():
    data = np.arange(101, dtype=float)  # 0 .. 100
    norm = CustomNormalization(interval_type="manual", vmin=20.0, vmax=70.0)
    out = norm(data)
    assert (norm.vmin, norm.vmax) == (20.0, 70.0)
    assert out[0] == 0.0 and out[20] == 0.0  # everything at or below vmin clips to 0
    assert out[100] == 1.0 and out[70] == 1.0  # everything at or above vmax clips to 1
    np.testing.assert_allclose(out[45], 0.5)  # midpoint of [20, 70]


def test_manual_interval_falls_back_to_data_range():
    data = np.array([-4.0, 0.0, 6.0])
    norm = CustomNormalization(interval_type="manual", data=data)
    assert (norm.vmin, norm.vmax) == (-4.0, 6.0)
    # A single specified bound is kept; only the missing one comes from the data.
    half = CustomNormalization(interval_type="manual", vmin=-10.0, data=data)
    assert (half.vmin, half.vmax) == (-10.0, 6.0)


def test_quantile_interval_defaults_on_known_array():
    # For x = 0..100 the linearly interpolated 2% and 98% quantiles are exactly 2 and 98
    data = np.arange(101, dtype=float)
    norm = CustomNormalization()  # quantile is the default interval
    out = norm(data)
    assert norm.config.interval_type == "quantile"
    assert (norm.config.lower_quantile, norm.config.upper_quantile) == (0.02, 0.98)
    assert (norm.vmin, norm.vmax) == (2.0, 98.0)
    np.testing.assert_allclose(out[50], (50.0 - 2.0) / 96.0)
    assert out[0] == 0.0 and out[-1] == 1.0  # the 2% tails clip


def test_centered_interval():
    data = np.array([-3.0, 1.0, 8.0])
    auto = CustomNormalization(interval_type="centered", data=data)
    assert (auto.vmin, auto.vmax) == (-8.0, 8.0)  # widest deviation from vcenter = 0
    shifted = CustomNormalization(interval_type="centered", vcenter=50.0, data=data)
    assert (shifted.vmin, shifted.vmax) == (-3.0, 103.0)  # |−3 − 50| = 53
    explicit = CustomNormalization(
        interval_type="centered", vcenter=50.0, half_range=10.0
    )
    assert (explicit.vmin, explicit.vmax) == (40.0, 60.0)
    np.testing.assert_allclose(explicit(np.array([50.0])), np.array([0.5]))


def test_limits_resolve_lazily_on_first_call():
    norm = CustomNormalization()
    assert norm.vmin is None and norm.vmax is None
    norm(np.linspace(0.0, 1.0, 100))
    assert norm.vmin is not None and norm.vmax is not None


def test_config_object_and_keyword_overrides_agree():
    config = NormalizationConfig(interval_type="manual", vmin=1.0, vmax=3.0)
    from_config = CustomNormalization(config)
    from_kwargs = CustomNormalization(interval_type="manual", vmin=1.0, vmax=3.0)
    assert (from_config.vmin, from_config.vmax) == (from_kwargs.vmin, from_kwargs.vmax)
    # Overrides are applied on top of a supplied config.
    overridden = CustomNormalization(config, vmax=9.0)
    assert (overridden.vmin, overridden.vmax) == (1.0, 9.0)


# --------------------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,constant",
    [
        # Quantile and manual limits both collapse onto the constant itself.
        ({"interval_type": "quantile"}, 3.0),
        ({"interval_type": "manual"}, 3.0),
        # A centered interval is only degenerate when the constant *is* the center;
        # for a constant of 3.0 about 0.0 the interval is a legitimate [-3, 3].
        ({"interval_type": "centered", "vcenter": 3.0}, 3.0),
        ({"interval_type": "centered"}, 0.0),
    ],
)
def test_constant_image_does_not_divide_by_zero(kwargs, constant):
    data = np.full((4, 4), constant)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a divide-by-zero warning fails the test
        with np.errstate(all="raise"):
            norm = CustomNormalization(data=data, **kwargs)
            out = norm(data)
    assert norm.vmin == norm.vmax  # the interval really is degenerate
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)


def test_centered_interval_of_a_nonzero_constant_is_not_degenerate():
    # Guards the boundary of the case above: max|3 - 0| = 3, so the interval is [-3, 3]
    # and the data sit at its top end.
    data = np.full((4, 4), 3.0)
    norm = CustomNormalization(interval_type="centered", data=data)
    assert (norm.vmin, norm.vmax) == (-3.0, 3.0)
    assert np.all(norm(data) == 1.0)


def test_explicitly_zero_width_manual_interval_is_finite():
    norm = CustomNormalization(interval_type="manual", vmin=2.0, vmax=2.0)
    out = norm(np.array([1.0, 2.0, 3.0]))
    assert np.all(out == 0.0)
    # The inverse of a degenerate interval is the single value it collapsed to.
    np.testing.assert_allclose(norm.inverse(np.array([0.0, 1.0])), np.array([2.0, 2.0]))


@pytest.mark.parametrize("interval_type", ["quantile", "manual", "centered"])
def test_all_nan_input_raises_clearly(interval_type):
    norm = CustomNormalization(interval_type=interval_type)
    with pytest.raises(ValueError, match="no finite values"):
        norm(np.full((3, 3), np.nan))


def test_all_non_finite_input_raises_clearly():
    norm = CustomNormalization(interval_type="manual")
    with pytest.raises(ValueError, match="no finite values"):
        norm(np.array([np.nan, np.inf, -np.inf]))


def test_nans_are_ignored_by_limits_and_preserved_in_output():
    data = np.arange(101, dtype=float)
    with_nan = np.concatenate([data, [np.nan, np.nan]])
    norm = CustomNormalization()  # quantile
    out = norm(with_nan)
    # Limits match those of the finite subset, i.e. the NaNs did not shift them.
    assert (norm.vmin, norm.vmax) == (2.0, 98.0)
    np.testing.assert_allclose(norm.vmin, np.nanquantile(with_nan, 0.02))
    np.testing.assert_allclose(norm.vmax, np.nanquantile(with_nan, 0.98))
    assert np.isnan(out[-1]) and np.isnan(out[-2])
    assert np.all(np.isfinite(out[:-2]))


def test_infinities_are_ignored_by_limits():
    data = np.array([0.0, 1.0, 2.0, np.inf, -np.inf])
    norm = CustomNormalization(interval_type="manual", data=data)
    assert (norm.vmin, norm.vmax) == (0.0, 2.0)


def test_inverted_manual_limits_raise_instead_of_silently_flipping():
    with pytest.raises(ValueError, match="inverted interval"):
        CustomNormalization(interval_type="manual", vmin=1.0, vmax=0.0)


def test_masked_entries_are_treated_as_missing():
    data = np.ma.masked_array([0.0, 50.0, 1e6], mask=[False, False, True])
    norm = CustomNormalization(interval_type="manual", data=data)
    assert (norm.vmin, norm.vmax) == (0.0, 50.0)  # the masked outlier is excluded


def test_input_array_is_not_modified():
    data = FIXED_ARRAY.copy()
    reference = data.copy()
    CustomNormalization(interval_type="manual", vmin=0.0, vmax=1.0)(data)
    np.testing.assert_array_equal(data, reference)


def test_inverse_before_resolution_raises():
    with pytest.raises(ValueError, match="unresolved"):
        CustomNormalization().inverse(np.array([0.5]))


# --------------------------------------------------------------------------------------
# torch support
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("interval_type", ["quantile", "manual", "centered"])
def test_torch_input_gives_the_same_result_as_numpy(interval_type):
    tensor = torch.as_tensor(FIXED_ARRAY)
    from_numpy = CustomNormalization(interval_type=interval_type)(FIXED_ARRAY)
    from_torch = CustomNormalization(interval_type=interval_type)(tensor)
    assert isinstance(from_torch, np.ndarray)
    np.testing.assert_array_equal(from_torch, from_numpy)


def test_torch_low_precision_and_data_kwarg():
    half = torch.as_tensor(FIXED_ARRAY, dtype=torch.float16)
    norm = CustomNormalization(interval_type="manual", data=half)
    assert norm.vmin is not None and norm.vmax is not None
    assert np.all(np.isfinite(norm(half)))


# --------------------------------------------------------------------------------------
# regression guard: an independent oracle, and frozen anchors
# --------------------------------------------------------------------------------------

# Tolerance rationale: the oracle and the implementation evaluate the same float64
# affine map (x - vmin) / (vmax - vmin) and take their quantiles from the same numpy
# function, so on float64 input they agree bit-for-bit -- the measured maximum absolute
# difference for every configuration below is exactly 0.0, and was also exactly 0.0
# against the deleted module. The tolerance is therefore not fitted to an observed
# error; it is a few-ULP allowance so the guard keeps working if a future numpy changes
# the rounding of np.quantile's interpolation. Note the scope: bit-exactness holds for
# float64 input. On float32 input the implementation promotes at the boundary while the
# deleted module stayed in float32, which showed up as a ~1e-7 difference for
# data-derived limits; the shipped configurations were exact in both dtypes, and
# test_float32_input_matches_the_oracle covers that case explicitly.
ORACLE_RTOL = 1e-13
ORACLE_ATOL = 1e-15

# Exactly what the published figure scripts construct: a manual interval with explicit
# limits, and the linear stretch left at its default.
SHIPPED_CONFIGS = [
    {"interval_type": "manual", "vmin": 0, "vmax": 1},
    {"interval_type": "manual", "vmin": 0.0, "vmax": 9.685019254061988},
    {"interval_type": "manual", "vmin": -2.0, "vmax": 5.0},
    {"interval_type": "manual", "vmin": 0.0, "vmax": 0.5},
    # Manual with data-derived bounds, and the quantile defaults.
    {"interval_type": "manual"},
    {"interval_type": "quantile"},
]

OTHER_CONFIGS = [
    {"interval_type": "quantile", "lower_quantile": 0.1, "upper_quantile": 0.9},
    {"interval_type": "centered"},
    {"interval_type": "centered", "vcenter": 1.0},
    {
        "interval_type": "manual",
        "vmin": 0.0,
        "vmax": 1.0,
        "stretch_type": "power",
        "power": 2.0,
    },
    {"interval_type": "quantile", "stretch_type": "power", "power": 0.5},
    {"interval_type": "quantile", "stretch_type": "logarithmic"},
    {
        "interval_type": "quantile",
        "stretch_type": "logarithmic",
        "logarithmic_index": 10.0,
    },
]

# Literal output of the deleted implementation on FIXED_ARRAY, captured before it was
# removed: six sampled flat indices and the sum over all 1024 entries. These pin the
# absolute numbers, so an edit that changed the implementation and the oracle in the
# same direction would still be caught.
FROZEN_INDICES = [0, 1, 7, 100, 511, 1023]
FROZEN = {
    "manual_unit": (
        {"interval_type": "manual", "vmin": 0, "vmax": 1},
        [0.18514867821841596, 1.0, 0.0, 1.0, 1.0, 0.0],
        605.1930471771885,
    ),
    "quantile_defaults": (
        {"interval_type": "quantile"},
        [
            0.40397813611510613,
            0.4893336503800939,
            0.24949730660522498,
            1.0,
            0.7363870899402887,
            0.33750229212929816,
        ],
        497.99966185835103,
    ),
}

# Limits of the deleted implementation on FIXED_ARRAY, likewise captured before removal.
FROZEN_LIMITS = {
    "quantile": (-4.621221527913321, 7.276378308435867),
    "manual": (-9.685019254061988, 9.220068422490623),
    "centered": (-9.685019254061988, 9.685019254061988),
}


@pytest.mark.parametrize("kwargs", SHIPPED_CONFIGS)
def test_matches_the_oracle_for_shipped_configurations(kwargs):
    expected = _oracle(FIXED_ARRAY, **kwargs)
    actual = CustomNormalization(**kwargs)(FIXED_ARRAY.copy())
    np.testing.assert_allclose(actual, expected, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


@pytest.mark.parametrize("kwargs", OTHER_CONFIGS)
def test_matches_the_oracle_beyond_the_shipped_configurations(kwargs):
    expected = _oracle(FIXED_ARRAY, **kwargs)
    actual = CustomNormalization(**kwargs)(FIXED_ARRAY.copy())
    np.testing.assert_allclose(actual, expected, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen_anchors_from_the_replaced_implementation(name):
    kwargs, values, total = FROZEN[name]
    out = CustomNormalization(**kwargs)(FIXED_ARRAY.copy()).ravel()
    np.testing.assert_allclose(
        out[FROZEN_INDICES], values, rtol=ORACLE_RTOL, atol=ORACLE_ATOL
    )
    np.testing.assert_allclose(float(out.sum()), total, rtol=ORACLE_RTOL)


@pytest.mark.parametrize("interval_type", ["quantile", "manual", "centered"])
def test_frozen_limits_for_every_interval_type(interval_type):
    norm = CustomNormalization(interval_type=interval_type, data=FIXED_ARRAY)
    np.testing.assert_allclose(
        [norm.vmin, norm.vmax],
        FROZEN_LIMITS[interval_type],
        rtol=ORACLE_RTOL,
        atol=ORACLE_ATOL,
    )
    # The oracle agrees with the frozen values too, so the two guards corroborate.
    np.testing.assert_allclose(
        _oracle_limits(FIXED_ARRAY, interval_type=interval_type),
        FROZEN_LIMITS[interval_type],
        rtol=ORACLE_RTOL,
        atol=ORACLE_ATOL,
    )


@pytest.mark.parametrize("kwargs", SHIPPED_CONFIGS)
def test_float32_input_matches_the_oracle(kwargs):
    """float32 input is promoted at the boundary, so the float64 oracle still applies.

    This is where the implementation and the module it replaced could differ: the old
    one kept float32 through the affine map, giving a ~1e-7 difference when the limits
    came from the data. Promoting is the better choice -- the interval arithmetic is
    exact in float64 -- and the two configurations the figure scripts construct agreed
    in both dtypes anyway.
    """
    data32 = FIXED_ARRAY.astype(np.float32)
    expected = _oracle(data32.astype(np.float64), **kwargs)
    actual = CustomNormalization(**kwargs)(data32.copy())
    np.testing.assert_allclose(actual, expected, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


def test_asinh_is_the_standard_concave_stretch_not_the_symmetric_variant():
    """Records the one deliberate numerical divergence from the replaced module.

    This module implements the standard display stretch
    ``s(t) = asinh(t / a) / asinh(1 / a)``, which is steepest at ``t = 0`` and so lifts
    faint detail. The replaced module's "asinh" was instead a variant centred on the
    middle of the interval, ``[1 + asinh((2t - 1) / a) / asinh(1 / a)] / 2``: also
    monotone, but a mid-tone-enhancing S-curve, and a genuinely different mapping (the
    two differ by up to 0.46). No published figure moves, because every shipped
    construction leaves ``stretch_type`` at its ``"linear"`` default; the divergence is
    nonetheless reachable through the ``"asinh_centered"`` preset, which is why it is
    recorded here and in the module docstring rather than only in a commit message.
    """
    t = np.linspace(0.0, 1.0, 9)
    a = 0.1
    new = _unit_stretch("asinh", asinh_linear_range=a)(t)
    np.testing.assert_allclose(new, np.arcsinh(t / a) / np.arcsinh(1.0 / a), rtol=1e-14)
    symmetric = 0.5 * (1.0 + np.arcsinh((2.0 * t - 1.0) / a) / np.arcsinh(1.0 / a))
    assert np.max(np.abs(new - symmetric)) > 0.1  # genuinely different curves


# --------------------------------------------------------------------------------------
# matplotlib integration
# --------------------------------------------------------------------------------------


def test_works_as_a_matplotlib_norm_for_a_colorbar():
    """The shipped colorbar helper reads vmin/vmax and matplotlib calls inverse()."""
    import matplotlib.pyplot as plt

    from scatterem.vis.visualization_utils import add_cbar_to_ax

    norm = CustomNormalization(
        interval_type="manual", vmin=0.0, vmax=1.0, data=FIXED_ARRAY
    )
    fig, (ax, cax) = plt.subplots(1, 2)
    try:
        ax.imshow(np.asarray(norm(FIXED_ARRAY)), cmap="magma")
        colorbar = add_cbar_to_ax(fig, cax, norm, plt.get_cmap("magma"))
        assert colorbar is not None
        fig.canvas.draw()  # forces matplotlib through norm.inverse()
    finally:
        plt.close(fig)


def test_lazy_limits_are_usable_by_a_tick_locator():
    """A colorbar needs numeric limits; None would make AutoLocator raise."""
    norm = CustomNormalization()  # quantile interval, no data yet
    norm(np.linspace(0.0, 1.0, 100))
    assert norm.vmin is not None and norm.vmax is not None
    ticker.AutoLocator().tick_values(norm.vmin, norm.vmax)  # must not raise


def test_matplotlib_cannot_bypass_the_configured_interval():
    """imshow calls autoscale_None; the quantile rule must survive that."""
    import matplotlib.pyplot as plt

    data = np.arange(101, dtype=float).reshape(101, 1)
    norm = CustomNormalization()  # quantile defaults -> 2.0 / 98.0, not 0 / 100
    fig, ax = plt.subplots()
    try:
        ax.imshow(data, norm=norm)
        fig.canvas.draw()
        assert (norm.vmin, norm.vmax) == (2.0, 98.0)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------
# shared use across several images
# --------------------------------------------------------------------------------------


def test_limits_are_frozen_after_the_first_resolution():
    """One instance describes one interval, by design.

    ``show_2d``'s multi-array path builds a single normalization without ``data=`` and
    applies it to every array in the list, so that they share a scale and the one
    colorbar drawn from ``vmin``/``vmax`` describes all of them. That requires the
    limits to be resolved once and then held, which is what this asserts. Callers who
    want each image scaled to its own limits must pass a fresh instance per image.
    """
    first = np.linspace(0.0, 1.0, 101)
    second = first * 100.0
    norm = CustomNormalization()  # quantile, unresolved
    norm(first)
    limits_after_first = (norm.vmin, norm.vmax)
    out = norm(second)
    assert (norm.vmin, norm.vmax) == limits_after_first
    # The second, much brighter array saturates against the first array's limits.
    assert np.all(out[1:] == 1.0)


def test_a_fresh_instance_rescales_to_its_own_data():
    second = np.linspace(0.0, 1.0, 101) * 100.0
    norm = CustomNormalization(data=second)
    np.testing.assert_allclose([norm.vmin, norm.vmax], [2.0, 98.0])


# --------------------------------------------------------------------------------------
# inert parameters, presets and the norm= resolver
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,inert",
    [
        ({"stretch_type": "linear", "power": 2.0}, "power"),
        ({"stretch_type": "logarithmic", "power": 2.0}, "power"),
        ({"stretch_type": "asinh", "power": 2.0}, "power"),
        ({"stretch_type": "linear", "logarithmic_index": 10.0}, "logarithmic_index"),
        ({"stretch_type": "power", "asinh_linear_range": 1.0}, "asinh_linear_range"),
    ],
)
def test_a_parameter_that_cannot_take_effect_warns(kwargs, inert):
    with pytest.warns(UserWarning, match=f"{inert}=.*has no effect"):
        CustomNormalization(**kwargs)


def test_stretch_type_alone_selects_the_stretch():
    """power= does not override stretch_type; it is ignored, with a warning."""
    with pytest.warns(UserWarning):
        norm = CustomNormalization(
            interval_type="manual", vmin=0.0, vmax=1.0, stretch_type="asinh", power=2.0
        )
    t = np.array([0.5])
    np.testing.assert_allclose(norm(t), np.arcsinh(5.0) / np.arcsinh(10.0), rtol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # every parameter at its default
        {"stretch_type": "power", "power": 2.0},
        {"stretch_type": "logarithmic", "logarithmic_index": 10.0},
        {"stretch_type": "asinh", "asinh_linear_range": 1.0},
        {"interval_type": "manual", "vmin": 0.0, "vmax": 1.0},
    ],
)
def test_no_warning_for_parameters_that_do_take_effect(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CustomNormalization(**kwargs)


@pytest.mark.parametrize("preset", sorted(NORMALIZATION_PRESETS))
def test_every_preset_resolves_and_renders(preset):
    config = _resolve_normalization(preset)
    assert isinstance(config, NormalizationConfig)
    out = CustomNormalization(config)(FIXED_ARRAY)
    assert out.shape == FIXED_ARRAY.shape
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out.max() <= 1.0


def test_resolver_accepts_none_dict_and_config():
    assert _resolve_normalization(None) == NormalizationConfig()
    assert _resolve_normalization({"vmin": 1.0}) == NormalizationConfig(vmin=1.0)
    config = NormalizationConfig(interval_type="manual")
    assert _resolve_normalization(config) is config


@pytest.mark.parametrize(
    "bad,exception",
    [
        ("not_a_preset", ValueError),
        ({"vmim": 0.0}, TypeError),  # typo in a field name
        (3.5, TypeError),
    ],
)
def test_resolver_rejects_bad_input(bad, exception):
    with pytest.raises(exception):
        _resolve_normalization(bad)
