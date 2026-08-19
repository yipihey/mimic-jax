"""A deliberately minimal, separately labelled reduction of SAGE16's baryon cycle.

This module does not replace or modify the faithful :mod:`mimic_jax.sage16`
implementation.  It supplies a small teacher--student model whose coefficients
can be calibrated against SAGE16 and then tested on disjoint merger trees.

The reduced state combines SAGE's hot and ejected phases into one
``CircumgalacticGas`` reservoir and retains cold gas, long-lived stars, and one
black-hole proxy carrying regulation memory.  Halo infall and mergers remain
explicit forcing/event maps around the local reservoir evolution.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp

Array = Any


class ReducedState(NamedTuple):
    """Four-reservoir state used by the reduced SAGE16 experiment."""

    CircumgalacticGas: Array
    ColdGas: Array
    StellarMass: Array
    BlackHoleProxy: Array


class ReducedForcing(NamedTuple):
    """Halo quantities required by one reduced-model interval."""

    HaloMass: Array
    SpinMagnitude: Array
    OnePlusRedshift: Array


class ReducedParameters(NamedTuple):
    """Nine physically interpretable coefficients of the reduced model.

    Times are in Gyr and masses use SAGE's internal ``1e10 Msun/h`` basis.
    ``FeedbackMassLoadingAtPivot`` is defined at ``HaloMassPivot``.  The pivot
    is fixed rather than fitted because changing a unit convention must not add
    a scientific degree of freedom.
    """

    StarFormationTimescaleGyr: Array
    CoolingTimescaleGyr: Array
    FeedbackMassLoadingAtPivot: Array
    FeedbackHaloMassSlope: Array
    QuenchingHaloMass: Array
    QuenchingSlope: Array
    ColdGasThresholdPerSpin: Array
    CoolingRedshiftExponent: Array
    BlackHoleQuenchingMass: Array


class StaticEfficiencyParameters(NamedTuple):
    """Four-coefficient zero-reservoir baseline used in reduction studies."""

    CharacteristicHaloMass: Array
    PeakEfficiency: Array
    LowMassSlope: Array
    HighMassSlope: Array


class ReducedIntervalDiagnostics(NamedTuple):
    """Integrated transfers and the terminal instantaneous SFR."""

    CooledMass: Array
    LockedStellarMass: Array
    ReheatedMass: Array
    StarFormationRate: Array


HALO_MASS_PIVOT = 100.0
RECYCLE_FRACTION = 0.43
BLACK_HOLE_GROWTH_EFFICIENCY = 0.015


def _positive_power(base, exponent, xp):
    """Evaluate a positive power without optimizer-induced overflow."""

    safe_base = xp.maximum(base, xp.finfo(xp.float64).tiny)
    logarithm = xp.clip(xp.log(safe_base) * exponent, -700.0, 700.0)
    return xp.exp(logarithm)


def initial_reduced_state(**overrides) -> ReducedState:
    """Construct an empty non-negative reduced state."""

    unknown = set(overrides) - set(ReducedState._fields)
    if unknown:
        raise TypeError(f"Unknown reduced state fields: {sorted(unknown)}")
    return ReducedState(
        **{
            name: jnp.asarray(overrides.get(name, 0.0), dtype=jnp.float64)
            for name in ReducedState._fields
        }
    )


def reduced_baryonic_mass(state: ReducedState):
    """Return the structurally conserved mass across the four reservoirs."""

    return state.CircumgalacticGas + state.ColdGas + state.StellarMass + state.BlackHoleProxy


def static_stellar_mass(peak_halo_mass, parameters: StaticEfficiencyParameters):
    """Map peak halo mass directly to stellar mass with a double power law."""

    mass = jnp.maximum(jnp.asarray(peak_halo_mass, dtype=jnp.float64), 0.0)
    safe_characteristic_mass = jnp.maximum(
        parameters.CharacteristicHaloMass, jnp.finfo(jnp.float64).tiny
    )
    ratio = mass / safe_characteristic_mass
    safe_ratio = jnp.maximum(ratio, jnp.finfo(jnp.float64).tiny)
    denominator = _positive_power(safe_ratio, -parameters.LowMassSlope, jnp)
    denominator += _positive_power(safe_ratio, parameters.HighMassSlope, jnp)
    return 2.0 * parameters.PeakEfficiency * mass / denominator


def add_cosmological_infall(state: ReducedState, infalling_mass) -> ReducedState:
    """Add an explicit non-negative external source to circumgalactic gas."""

    mass = jnp.maximum(jnp.asarray(infalling_mass, dtype=jnp.float64), 0.0)
    return state._replace(CircumgalacticGas=state.CircumgalacticGas + mass)


def _apply_reduced_merger_event_impl(
    state,
    merger_mass_ratio,
    xp,
):
    ratio = xp.maximum(xp.asarray(merger_mass_ratio, dtype=xp.float64), 0.0)
    growth = xp.minimum(
        BLACK_HOLE_GROWTH_EFFICIENCY * ratio * state.ColdGas,
        state.ColdGas,
    )
    return (
        state._replace(
            ColdGas=state.ColdGas - growth,
            BlackHoleProxy=state.BlackHoleProxy + growth,
        ),
        growth,
    )


def apply_reduced_merger_event(
    state: ReducedState,
    merger_mass_ratio,
):
    """Grow the black-hole proxy in a conservative merger event.

    The event retains SAGE's fiducial black-hole growth coefficient but omits
    its detailed quasar-mode denominator and starburst chain.  The event is
    therefore an explicitly documented reduction, not an upstream-equivalent
    prescription.
    """

    return _apply_reduced_merger_event_impl(
        state,
        merger_mass_ratio,
        jnp,
    )


def _apply_reduced_merger_event_numpy(
    state: ReducedState,
    merger_mass_ratio,
):
    """NumPy backend for the host-side tree experiment."""

    import numpy as np

    return _apply_reduced_merger_event_impl(
        state,
        merger_mass_ratio,
        np,
    )


def merge_reduced_states(*states: ReducedState) -> ReducedState:
    """Apply the reduced model's additive merger event map."""

    if not states:
        raise ValueError("at least one reduced state is required")
    return ReducedState(
        **{name: sum(getattr(state, name) for state in states) for name in ReducedState._fields}
    )


def cooling_efficiency(
    forcing: ReducedForcing,
    parameters: ReducedParameters,
    black_hole_proxy=0.0,
):
    """Return the dimensionless cooling factor including effective quenching."""

    safe_mass = jnp.maximum(forcing.HaloMass, jnp.finfo(jnp.float64).tiny)
    mass_ratio = safe_mass / parameters.QuenchingHaloMass
    redshift_factor = _positive_power(
        forcing.OnePlusRedshift, parameters.CoolingRedshiftExponent, jnp
    )
    black_hole_factor = 1.0 + jnp.maximum(black_hole_proxy, 0.0) / (
        parameters.BlackHoleQuenchingMass
    )
    return redshift_factor / (
        1.0 + _positive_power(mass_ratio, parameters.QuenchingSlope, jnp) * black_hole_factor
    )


def feedback_mass_loading(forcing: ReducedForcing, parameters: ReducedParameters):
    """Return the effective cold-to-circumgalactic mass loading."""

    safe_mass = jnp.maximum(forcing.HaloMass, jnp.finfo(jnp.float64).tiny)
    loading = parameters.FeedbackMassLoadingAtPivot * _positive_power(
        safe_mass / HALO_MASS_PIVOT,
        -parameters.FeedbackHaloMassSlope,
        jnp,
    )
    return jnp.minimum(loading, 100.0)


def cold_gas_threshold(forcing: ReducedForcing, parameters: ReducedParameters):
    """Return the spin-dependent star-forming gas threshold.

    SAGE's disk-radius and critical-surface-density expressions reduce locally
    to a threshold proportional to the magnitude of the halo spin vector.  The
    proportionality is fitted here because the reduced model has discarded the
    explicit disk-radius state.
    """

    return parameters.ColdGasThresholdPerSpin * jnp.maximum(forcing.SpinMagnitude, 0.0)


def _reduced_interval_impl(
    state: ReducedState,
    forcing: ReducedForcing,
    parameters: ReducedParameters,
    dt_gyr,
    *,
    substeps: int,
    xp,
):
    if substeps < 1:
        raise ValueError("substeps must be at least one")

    duration = xp.maximum(xp.asarray(dt_gyr, dtype=xp.float64), 0.0)
    substep_duration = duration / substeps
    circumgalactic = xp.asarray(state.CircumgalacticGas, dtype=xp.float64)
    cold = xp.asarray(state.ColdGas, dtype=xp.float64)
    stars = xp.asarray(state.StellarMass, dtype=xp.float64)
    black_hole = xp.asarray(state.BlackHoleProxy, dtype=xp.float64)

    safe_mass = xp.maximum(forcing.HaloMass, xp.finfo(xp.float64).tiny)
    cooling_factor = _positive_power(
        forcing.OnePlusRedshift,
        parameters.CoolingRedshiftExponent,
        xp,
    )
    black_hole_factor = 1.0 + xp.maximum(black_hole, 0.0) / (parameters.BlackHoleQuenchingMass)
    cooling_factor /= (
        1.0
        + _positive_power(
            safe_mass / parameters.QuenchingHaloMass,
            parameters.QuenchingSlope,
            xp,
        )
        * black_hole_factor
    )
    cooling_rate = cooling_factor / parameters.CoolingTimescaleGyr
    loading = parameters.FeedbackMassLoadingAtPivot * _positive_power(
        safe_mass / HALO_MASS_PIVOT,
        -parameters.FeedbackHaloMassSlope,
        xp,
    )
    loading = xp.minimum(loading, 100.0)
    threshold = parameters.ColdGasThresholdPerSpin * xp.maximum(forcing.SpinMagnitude, 0.0)
    consumption_factor = 1.0 - RECYCLE_FRACTION + loading

    cooled_total = xp.zeros_like(circumgalactic)
    locked_total = xp.zeros_like(circumgalactic)
    reheated_total = xp.zeros_like(circumgalactic)
    for _ in range(substeps):
        cooled = circumgalactic * (-xp.expm1(-cooling_rate * substep_duration))
        circumgalactic = circumgalactic - cooled
        cold = cold + cooled

        available = xp.maximum(cold - threshold, 0.0)
        processed = available * (
            -xp.expm1(-consumption_factor * substep_duration / parameters.StarFormationTimescaleGyr)
        )
        locked = processed * (1.0 - RECYCLE_FRACTION) / consumption_factor
        reheated = processed * loading / consumption_factor
        cold = cold - processed
        stars = stars + locked
        circumgalactic = circumgalactic + reheated
        cooled_total = cooled_total + cooled
        locked_total = locked_total + locked
        reheated_total = reheated_total + reheated

    terminal_available = xp.maximum(cold - threshold, 0.0)
    terminal_sfr = terminal_available / parameters.StarFormationTimescaleGyr
    return (
        ReducedState(circumgalactic, cold, stars, black_hole),
        ReducedIntervalDiagnostics(
            cooled_total,
            locked_total,
            reheated_total,
            terminal_sfr,
        ),
    )


def evolve_reduced_interval(
    state: ReducedState,
    forcing: ReducedForcing,
    parameters: ReducedParameters,
    dt_gyr,
    *,
    substeps: int = 2,
):
    """Evolve the conservative local reservoir system over one forced interval.

    Cooling is followed by the star-formation/recycling/feedback transfer in
    each substep.  This is an explicit operator split, not the upstream SAGE16
    sequence and not an equivalence mode.
    """

    return _reduced_interval_impl(
        state,
        forcing,
        parameters,
        dt_gyr,
        substeps=substeps,
        xp=jnp,
    )


def _evolve_reduced_interval_numpy(
    state: ReducedState,
    forcing: ReducedForcing,
    parameters: ReducedParameters,
    dt_gyr,
    *,
    substeps: int = 2,
):
    """NumPy backend for host-side tree fitting, sharing the canonical formula."""

    import numpy as np

    return _reduced_interval_impl(
        state,
        forcing,
        parameters,
        dt_gyr,
        substeps=substeps,
        xp=np,
    )
