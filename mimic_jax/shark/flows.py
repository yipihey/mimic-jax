"""Exact stoichiometric assembly of SHARK's 19-variable baryon-cycle ODE."""

from typing import NamedTuple

import jax.numpy as jnp

from mimic_jax.shark.types import (
    SharkAugmentedFlowRates,
    SharkContinuousState,
    SharkFlowParameters,
    SharkFlowRates,
    SharkRhsResult,
    SharkState,
)


class SharkAugmentedRhsResult(NamedTuple):
    """Augmented derivative and the BH angular momentum removed from hot gas."""

    derivative: SharkContinuousState
    black_hole_angular_momentum_sink: object


def _add_derivatives(left: SharkState, right: SharkState) -> SharkState:
    return SharkState(*(a + b for a, b in zip(left, right)))


def direct_cooling_flow_derivative(
    state: SharkState, cooling_rate, cooling_specific_angular_momentum
) -> SharkState:
    """Route continuous cooling directly from hot halo to cold ISM.

    Upstream first performs a finite hot-to-``cold_halo_gas`` preparation map,
    then feeds the corresponding constant rate to its 19-variable ODE, which
    drains ``cold_halo_gas`` into the ISM.  That two-stage ordering remains in
    :func:`shark_rhs_from_rates`.  The explicit continuous formulation removes
    the numerical staging delay and represents the physical hot-to-cold
    transport as one conservative flow.
    """

    rate = jnp.asarray(cooling_rate)
    specific_angular_momentum = jnp.asarray(cooling_specific_angular_momentum)
    active = state.hot_halo_gas > 0.0
    safe_hot_mass = jnp.where(active, state.hot_halo_gas, 1.0)
    source_metallicity = jnp.where(active, state.hot_halo_gas_metals / safe_hot_mass, 0.0)
    metal_rate = rate * source_metallicity
    angular_momentum_rate = rate * specific_angular_momentum
    zero = jnp.zeros_like(rate)
    return SharkState(
        stellar_mass=zero,
        cold_gas=rate,
        cold_halo_gas=zero,
        hot_halo_gas=-rate,
        ejected_gas=zero,
        lost_gas=zero,
        stellar_metals=zero,
        cold_gas_metals=metal_rate,
        cold_halo_gas_metals=zero,
        hot_halo_gas_metals=-metal_rate,
        ejected_gas_metals=zero,
        lost_gas_metals=zero,
        formed_stellar_mass=zero,
        formed_stellar_metals=zero,
        stellar_angular_momentum=zero,
        cold_gas_angular_momentum=angular_momentum_rate,
        cold_halo_angular_momentum=zero,
        hot_halo_angular_momentum=-angular_momentum_rate,
        ejected_angular_momentum=zero,
    )


def reincorporation_flow_derivative(state: SharkState, reincorporation_rate) -> SharkState:
    """Route a continuous ejected-to-hot transfer conservatively.

    The exact upstream finite update moves mass and metals before resetting the
    hot-halo specific angular momentum from halo forcing.  The continuous mode
    instead transports the source gas's angular momentum and exposes any later
    halo-spin reset as a separate projection.  This keeps the flow itself
    conservative without conflating it with the reference projection order.
    """

    rate = jnp.asarray(reincorporation_rate)
    active = state.ejected_gas > 0.0
    source_mass = jnp.where(active, state.ejected_gas, 1.0)
    source_metallicity = jnp.where(active, state.ejected_gas_metals / source_mass, 0.0)
    source_specific_angular_momentum = jnp.where(
        active, state.ejected_angular_momentum / source_mass, 0.0
    )
    metal_rate = rate * source_metallicity
    angular_momentum_rate = rate * source_specific_angular_momentum
    zero = jnp.zeros_like(rate)
    return SharkState(
        stellar_mass=zero,
        cold_gas=zero,
        cold_halo_gas=zero,
        hot_halo_gas=rate,
        ejected_gas=-rate,
        lost_gas=zero,
        stellar_metals=zero,
        cold_gas_metals=zero,
        cold_halo_gas_metals=zero,
        hot_halo_gas_metals=metal_rate,
        ejected_gas_metals=-metal_rate,
        lost_gas_metals=zero,
        formed_stellar_mass=zero,
        formed_stellar_metals=zero,
        stellar_angular_momentum=zero,
        cold_gas_angular_momentum=zero,
        cold_halo_angular_momentum=zero,
        hot_halo_angular_momentum=angular_momentum_rate,
        ejected_angular_momentum=-angular_momentum_rate,
    )


def hot_halo_black_hole_accretion_derivative(state: SharkContinuousState, accretion_rate):
    """Move hot gas and its metals continuously into the black hole.

    SHARK does not store black-hole angular momentum in the same units as its
    baryonic component-AM ledger.  The removed hot-gas AM is therefore
    returned explicitly as a named sink rather than silently discarded.
    """

    rate = jnp.asarray(accretion_rate)
    reservoirs = state.reservoirs
    active = reservoirs.hot_halo_gas > 0.0
    safe_hot_mass = jnp.where(active, reservoirs.hot_halo_gas, 1.0)
    metallicity = jnp.where(active, reservoirs.hot_halo_gas_metals / safe_hot_mass, 0.0)
    specific_angular_momentum = jnp.where(
        active, reservoirs.hot_halo_angular_momentum / safe_hot_mass, 0.0
    )
    metal_rate = rate * metallicity
    angular_momentum_sink = rate * specific_angular_momentum
    zero = jnp.zeros_like(rate)
    reservoir_derivative = SharkState(
        stellar_mass=zero,
        cold_gas=zero,
        cold_halo_gas=zero,
        hot_halo_gas=-rate,
        ejected_gas=zero,
        lost_gas=zero,
        stellar_metals=zero,
        cold_gas_metals=zero,
        cold_halo_gas_metals=zero,
        hot_halo_gas_metals=-metal_rate,
        ejected_gas_metals=zero,
        lost_gas_metals=zero,
        formed_stellar_mass=zero,
        formed_stellar_metals=zero,
        stellar_angular_momentum=zero,
        cold_gas_angular_momentum=zero,
        cold_halo_angular_momentum=zero,
        hot_halo_angular_momentum=-angular_momentum_sink,
        ejected_angular_momentum=zero,
    )
    derivative = SharkContinuousState(
        reservoirs=reservoir_derivative,
        black_hole_mass=rate,
        black_hole_metals=metal_rate,
        black_hole_spin=zero,
        heating_radius=zero,
        excess_jet_power=zero,
    )
    return derivative, angular_momentum_sink


def shark_continuous_rhs_from_rates(
    time,
    state: SharkState,
    rates: SharkFlowRates,
    parameters: SharkFlowParameters,
    *,
    reincorporation_rate=0.0,
) -> SharkState:
    """Assemble implemented continuous flows, separate from reference order.

    ``shark_rhs_from_rates`` remains the exact upstream 19-equation evaluator.
    This explicitly named counterpart adds flows which upstream realizes in a
    pre-ODE finite map.  Further transports will enter here only after their
    reference and continuous semantics have independent tests.
    """

    # The reference RHS consumes a pre-filled cold-halo staging reservoir.
    # Continuous cooling is instead routed directly from the hot halo, so the
    # reference cooling term must be removed before the direct flow is added.
    base_rates = rates._replace(cooling=jnp.zeros_like(rates.cooling))
    base = shark_rhs_from_rates(time, state, base_rates, parameters).derivative
    cooling = direct_cooling_flow_derivative(
        state, rates.cooling, rates.cooling_specific_angular_momentum
    )
    reincorporation = reincorporation_flow_derivative(state, reincorporation_rate)
    return _add_derivatives(_add_derivatives(base, cooling), reincorporation)


def cold_gas_metallicity(state: SharkState, parameters: SharkFlowParameters):
    """Return upstream's pre-enrichment-floor cold-gas metallicity."""

    measured = jnp.where(state.cold_gas > 0.0, state.cold_gas_metals / state.cold_gas, 0.0)
    return jnp.where(
        (state.cold_gas > 0.0) & (state.cold_gas_metals > 0.0),
        measured,
        parameters.pre_enrichment_metallicity,
    )


def effective_stellar_yield(cold_metallicity, parameters: SharkFlowParameters):
    """Return SHARK's fixed or Robotham et al. evolving yield."""

    evolving = parameters.yield_mass_fraction - 0.25 * cold_metallicity
    return jnp.where(parameters.evolving_yield, evolving, parameters.yield_mass_fraction)


def shark_rhs_from_rates(
    time,
    state: SharkState,
    rates: SharkFlowRates,
    parameters: SharkFlowParameters,
) -> SharkRhsResult:
    """Assemble the exact upstream ``basic_physicalmodel_evaluator`` RHS.

    The prescription layer computes cooling, star formation, stellar-feedback,
    QSO-feedback, and angular-momentum rates.  This function only encodes their
    conservative routing among reservoirs.  ``time`` is accepted for the
    standard non-autonomous JAX integrator interface.
    """

    del time
    zcold = cold_gas_metallicity(state, parameters)
    zcool = jnp.maximum(rates.cooling_metallicity, parameters.pre_enrichment_metallicity)
    yield_eff = effective_stellar_yield(zcold, parameters)
    retained = 1.0 - parameters.recycle_fraction

    sfr = rates.star_formation
    j_sfr = rates.star_formation_angular_momentum
    beta_reheat = rates.stellar_reheating_loading
    beta_eject = rates.stellar_ejection_loading
    beta_j_reheat = rates.angular_momentum_reheating_loading
    beta_j_eject = rates.angular_momentum_ejection_loading
    beta_qso_reheat = rates.qso_reheating_loading
    beta_qso_eject = rates.qso_ejection_loading
    cooling = rates.cooling
    cooling_j = rates.cooling_specific_angular_momentum

    total_reheating = beta_reheat + beta_qso_reheat
    total_ejection = beta_eject + beta_qso_eject

    derivative = SharkState(
        stellar_mass=retained * sfr,
        cold_gas=cooling - (retained + total_reheating) * sfr,
        cold_halo_gas=-cooling,
        hot_halo_gas=(total_reheating - total_ejection) * sfr,
        ejected_gas=beta_eject * sfr,
        lost_gas=beta_qso_eject * sfr,
        stellar_metals=retained * zcold * sfr,
        cold_gas_metals=cooling * zcool + sfr * (yield_eff - (retained + total_reheating) * zcold),
        cold_halo_gas_metals=-cooling * zcool,
        hot_halo_gas_metals=(total_reheating - total_ejection) * zcold * sfr,
        ejected_gas_metals=beta_eject * zcold * sfr,
        lost_gas_metals=beta_qso_eject * zcold * sfr,
        formed_stellar_mass=sfr,
        formed_stellar_metals=zcold * sfr,
        stellar_angular_momentum=retained * j_sfr,
        cold_gas_angular_momentum=cooling * cooling_j - (retained + beta_j_reheat) * j_sfr,
        cold_halo_angular_momentum=-cooling * cooling_j,
        hot_halo_angular_momentum=(beta_j_reheat - beta_j_eject) * j_sfr,
        ejected_angular_momentum=beta_j_eject * j_sfr,
    )
    return SharkRhsResult(
        derivative=derivative,
        rates=rates,
        cold_gas_metallicity=zcold,
        effective_yield=yield_eff,
    )


def shark_rhs(time, state, rate_law, parameters):
    """Evaluate a state-dependent prescription layer and assemble its RHS."""

    rates = rate_law(time, state)
    return shark_rhs_from_rates(time, state, rates, parameters).derivative


def shark_continuous_rhs(
    time,
    state,
    rate_law,
    parameters,
    reincorporation_rate_law=None,
):
    """Evaluate the explicit continuous routing under state-dependent rates."""

    rates = rate_law(time, state)
    reincorporation_rate = (
        0.0 if reincorporation_rate_law is None else reincorporation_rate_law(time, state)
    )
    return shark_continuous_rhs_from_rates(
        time,
        state,
        rates,
        parameters,
        reincorporation_rate=reincorporation_rate,
    )


def shark_augmented_continuous_rhs_from_rates(
    time,
    state: SharkContinuousState,
    rates: SharkAugmentedFlowRates,
    parameters: SharkFlowParameters,
) -> SharkAugmentedRhsResult:
    """Assemble the augmented continuous reservoir plus BH flow state."""

    reservoir_derivative = shark_continuous_rhs_from_rates(
        time,
        state.reservoirs,
        rates.reservoirs,
        parameters,
        reincorporation_rate=rates.reincorporation,
    )
    black_hole_derivative, angular_momentum_sink = hot_halo_black_hole_accretion_derivative(
        state, rates.hot_halo_black_hole_accretion
    )
    derivative = black_hole_derivative._replace(
        reservoirs=_add_derivatives(reservoir_derivative, black_hole_derivative.reservoirs)
    )
    return SharkAugmentedRhsResult(derivative, angular_momentum_sink)


def shark_augmented_continuous_rhs(time, state, rate_law, parameters):
    """Evaluate augmented state-dependent rates and return their derivative."""

    rates = rate_law(time, state)
    return shark_augmented_continuous_rhs_from_rates(time, state, rates, parameters).derivative
