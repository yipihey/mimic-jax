"""Executable mass, metal, and angular-momentum ledgers for SHARK flows."""

from mimic_jax.shark.types import SharkContinuousState, SharkRhsResult, SharkState


def baryonic_mass(state: SharkState):
    """Return mass in the six physical ODE reservoirs (trackers excluded)."""

    return (
        state.stellar_mass
        + state.cold_gas
        + state.cold_halo_gas
        + state.hot_halo_gas
        + state.ejected_gas
        + state.lost_gas
    )


def metal_mass(state: SharkState):
    """Return metal mass in the six physical ODE reservoirs."""

    return (
        state.stellar_metals
        + state.cold_gas_metals
        + state.cold_halo_gas_metals
        + state.hot_halo_gas_metals
        + state.ejected_gas_metals
        + state.lost_gas_metals
    )


def angular_momentum(state: SharkState):
    """Return total angular momentum in the five tracked reservoirs."""

    return (
        state.stellar_angular_momentum
        + state.cold_gas_angular_momentum
        + state.cold_halo_angular_momentum
        + state.hot_halo_angular_momentum
        + state.ejected_angular_momentum
    )


def augmented_baryonic_mass(state: SharkContinuousState):
    """Return tracked reservoir mass including the central black hole."""

    return baryonic_mass(state.reservoirs) + state.black_hole_mass


def augmented_metal_mass(state: SharkContinuousState):
    """Return tracked metal mass including metals accreted by the black hole."""

    return metal_mass(state.reservoirs) + state.black_hole_metals


def flow_conservation_residuals(result: SharkRhsResult):
    """Return baryon, metal-source, and angular-momentum RHS residuals."""

    derivative = result.derivative
    expected_new_metals = result.effective_yield * result.rates.star_formation
    return (
        baryonic_mass(derivative),
        metal_mass(derivative) - expected_new_metals,
        angular_momentum(derivative),
    )
