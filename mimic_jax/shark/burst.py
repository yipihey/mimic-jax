"""Event-triggered Lagos23 bulge starbursts and black-hole growth.

Disk instabilities and mergers are genuine jump maps, but the starburst which
follows either event is an ordinary SHARK 19-variable flow.  This module keeps
that distinction explicit and reproduces the upstream ordering: compute and
remove the finite BH fuel, expose its accretion rate to the QSO prescription,
integrate the bulge burst, then add the BH mass after the flow solve.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mimic_jax.numerics import RK4, integrate_adaptive, integrate_fixed_step
from mimic_jax.shark.components import SharkSystemState
from mimic_jax.shark.flows import shark_rhs_from_rates
from mimic_jax.shark.hybrid import (
    _add_sized,
    burst_flow_state_from_system,
    project_burst_flow_state_to_system,
    proportional_transfer,
)
from mimic_jax.shark.prescriptions.agn import (
    griffin19_accretion_spin_upstream_rng,
    lagos23_qso_outflow_loadings,
)
from mimic_jax.shark.prescriptions.disk import (
    lagos23_disk_flow_rates,
    lagos23_disk_forcing,
)

Array = Any

_GRAVITATIONAL_CONSTANT = 6.67259e-11 * 1.9891e30 / 3.0856775807e22 / 1.0e6
_MPC_PER_KM_PER_SECOND_TO_GYR = 3.0856775807e22 / 3.15576e16 / 1.0e3


class Lagos23StarburstParameters(NamedTuple):
    """Fiducial event/burst parameters from ``sample_lagos23.cfg``."""

    black_hole_gas_fraction: Array
    black_hole_velocity_km_per_s: Array
    accretion_time_multiplier: Array
    minimum_bulge_gas_mass: Array


class SharkStarburstDiagnostics(NamedTuple):
    """Named products of one event-triggered burst episode."""

    active: Array
    black_hole_accretion_time: Array
    black_hole_transfer: Array
    black_hole_metal_transfer: Array
    mean_star_formation_rate: Array
    mean_formed_stellar_metallicity: Array
    rhs_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array


class SharkStarburstResult(NamedTuple):
    state: SharkSystemState
    diagnostics: SharkStarburstDiagnostics


def lagos23_starburst_parameters(
    *,
    black_hole_gas_fraction: float = 0.01,
    black_hole_velocity_km_per_s: float = 400.0,
    accretion_time_multiplier: float = 20.0,
    minimum_bulge_gas_mass: float = 1.0e5,
) -> Lagos23StarburstParameters:
    """Construct the pinned Lagos23 merger/instability burst parameters."""

    return Lagos23StarburstParameters(
        black_hole_gas_fraction=jnp.asarray(black_hole_gas_fraction, dtype=jnp.float64),
        black_hole_velocity_km_per_s=jnp.asarray(black_hole_velocity_km_per_s, dtype=jnp.float64),
        accretion_time_multiplier=jnp.asarray(accretion_time_multiplier, dtype=jnp.float64),
        minimum_bulge_gas_mass=jnp.asarray(minimum_bulge_gas_mass, dtype=jnp.float64),
    )


def black_hole_starburst_accretion_time(
    bulge_baryonic_mass,
    bulge_gas_radius_mpc_over_h,
    hubble_h,
    parameters: Lagos23StarburstParameters,
):
    """Return the upstream bulge dynamical time times ``tau_fold`` in Gyr."""

    mass = jnp.asarray(bulge_baryonic_mass)
    radius = jnp.asarray(bulge_gas_radius_mpc_over_h)
    active = (mass > 0.0) & (radius > 0.0)
    velocity = jnp.sqrt(
        _GRAVITATIONAL_CONSTANT * jnp.where(active, mass, 1.0) / jnp.where(active, radius, 1.0)
    )
    physical_radius = radius / jnp.asarray(hubble_h)
    time = (
        _MPC_PER_KM_PER_SECOND_TO_GYR
        * physical_radius
        / jnp.where(active, velocity, 1.0)
        * parameters.accretion_time_multiplier
    )
    return jnp.where(active, time, jnp.inf)


def black_hole_starburst_fuel(
    bulge_gas_mass,
    virial_velocity_km_per_s,
    black_hole_mass,
    seed_mass,
    parameters: Lagos23StarburstParameters,
):
    """Return the finite Lagos23 bulge-gas budget offered to the BH."""

    gas = jnp.asarray(bulge_gas_mass)
    velocity = jnp.asarray(virial_velocity_km_per_s)
    eligible = (
        (jnp.asarray(black_hole_mass) >= 0.99 * jnp.asarray(seed_mass))
        & (gas > 0.0)
        & (velocity > 0.0)
    )
    fuel = (
        parameters.black_hole_gas_fraction
        * gas
        / (1.0 + (parameters.black_hole_velocity_km_per_s / velocity) ** 2)
    )
    return jnp.where(eligible, fuel, 0.0)


def _burst_rates(time, state, forcing, black_hole, model_parameters):
    base = lagos23_disk_flow_rates(
        time,
        state,
        forcing,
        model_parameters.star_formation,
        model_parameters.stellar_feedback,
    )
    positive = state.cold_gas > 0.0
    metallicity = jnp.where(
        positive, state.cold_gas_metals / jnp.where(positive, state.cold_gas, 1.0), 0.0
    )
    qso = lagos23_qso_outflow_loadings(
        gas_mass=state.cold_gas,
        black_hole_mass_msun_over_h=black_hole.mass,
        hot_halo_accretion_rate_msun_over_h_per_gyr=(black_hole.hot_halo_accretion_rate),
        starburst_accretion_rate_msun_over_h_per_gyr=(black_hole.starburst_accretion_rate),
        spin=black_hole.spin,
        gas_metallicity=metallicity,
        circular_velocity_km_per_s=forcing.galaxy_velocity,
        star_formation_rate=base.star_formation,
        bulge_baryonic_mass=state.stellar_mass + state.cold_gas,
        bulge_radius_mpc=forcing.stellar_half_mass_radius,
        parameters=model_parameters.agn,
    )
    return base._replace(
        qso_reheating_loading=qso.reheating,
        qso_ejection_loading=qso.ejection,
    )


def evolve_shark_starburst(
    state: SharkSystemState,
    *,
    redshift,
    duration_gyr,
    virial_velocity,
    subhalo_velocity,
    galaxy_id,
    execution_seed,
    model_parameters,
    burst_parameters: Lagos23StarburstParameters = None,
    from_galaxy_merger=False,
    triggered=True,
    method: str = RK4,
    num_steps: int = 64,
    adaptive: bool = False,
    relative_tolerance: float = 0.05,
) -> SharkStarburstResult:
    """Evolve the complete upstream event-triggered bulge burst sequence.

    The inactive branch is returned unchanged unless an upstream merger or
    instability event explicitly triggered the episode and the bulge gas is
    above SHARK's ``mass_min``. Fixed-step integration is intended for
    convergence studies; adaptive Dormand--Prince uses the same process
    ordering but is not claimed to be bitwise GSL Cash--Karp equivalence.
    """

    burst_parameters = (
        lagos23_starburst_parameters() if burst_parameters is None else burst_parameters
    )
    duration = jnp.asarray(duration_gyr, dtype=jnp.float64)
    active = jnp.asarray(triggered) & (
        state.galaxy.bulge_gas.mass > burst_parameters.minimum_bulge_gas_mass
    )

    def evolve(active_state):
        galaxy = active_state.galaxy
        bulge_mass = galaxy.bulge_stars.mass + galaxy.bulge_gas.mass
        accretion_time = black_hole_starburst_accretion_time(
            bulge_mass,
            galaxy.bulge_gas.radius,
            model_parameters.cosmology.hubble_h,
            burst_parameters,
        )
        requested = black_hole_starburst_fuel(
            galaxy.bulge_gas.mass,
            virial_velocity,
            galaxy.black_hole.mass,
            model_parameters.black_hole_seed_mass,
            burst_parameters,
        )
        remaining_gas, transfer = proportional_transfer(galaxy.bulge_gas, requested)
        spin = griffin19_accretion_spin_upstream_rng(
            galaxy.black_hole.mass,
            galaxy.black_hole.spin,
            transfer.mass,
            accretion_time,
            galaxy_id,
            execution_seed,
            model_parameters.agn,
        ).astype(galaxy.black_hole.spin.dtype)
        rate = jnp.where(accretion_time > 0.0, transfer.mass / accretion_time, 0.0)
        black_hole_during_burst = galaxy.black_hole._replace(
            spin=spin,
            starburst_accretion_rate=galaxy.black_hole.starburst_accretion_rate + rate,
        )
        prepared = active_state._replace(
            galaxy=galaxy._replace(
                bulge_gas=remaining_gas,
                black_hole=black_hole_during_burst,
            )
        )
        initial_flow = burst_flow_state_from_system(prepared)
        forcing = lagos23_disk_forcing(
            gas_half_mass_radius=prepared.galaxy.bulge_gas.radius,
            stellar_half_mass_radius=prepared.galaxy.bulge_stars.radius,
            redshift=redshift,
            burst=True,
            galaxy_velocity=prepared.galaxy.maximum_circular_velocity,
            subhalo_velocity=subhalo_velocity,
        )

        def rhs(time, flow_state):
            rates = _burst_rates(
                time,
                flow_state,
                forcing,
                black_hole_during_burst,
                model_parameters,
            )
            return shark_rhs_from_rates(time, flow_state, rates, model_parameters.flow).derivative

        if adaptive:
            scale_floor = jax.tree_util.tree_map(
                lambda value: jnp.maximum(jnp.abs(value) * 1.0e-12, 1.0e-12),
                initial_flow,
            )
            solution = integrate_adaptive(
                rhs,
                initial_flow,
                duration=duration,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=scale_floor,
                initial_step=duration,
                max_steps=4096,
                max_attempts=16384,
            )
            final_flow = solution.final_state
            evaluations = solution.rhs_evaluations
            accepted = solution.accepted_steps
            rejected = solution.rejected_steps
        else:
            solution = integrate_fixed_step(
                rhs,
                initial_flow,
                duration=duration,
                num_steps=num_steps,
                method=method,
            )
            final_flow = solution.final_state
            evaluations = jnp.asarray(
                num_steps * {"forward_euler": 1, "heun_rk2": 2, "rk4": 4}[method],
                dtype=jnp.int32,
            )
            accepted = jnp.asarray(num_steps, dtype=jnp.int32)
            rejected = jnp.asarray(0, dtype=jnp.int32)
        projected, flow_diagnostics = project_burst_flow_state_to_system(
            prepared,
            final_flow,
            duration,
            from_galaxy_merger=from_galaxy_merger,
        )
        black_hole = projected.galaxy.black_hole._replace(
            mass=projected.galaxy.black_hole.mass + transfer.mass,
            metals=projected.galaxy.black_hole.metals + transfer.metals,
        )
        projected = projected._replace(galaxy=projected.galaxy._replace(black_hole=black_hole))
        residual = (projected.galaxy.bulge_gas.mass > 0.0) & (
            projected.galaxy.bulge_gas.mass < burst_parameters.minimum_bulge_gas_mass
        )

        def return_residual(value):
            value_galaxy = value.galaxy
            disk_was_empty = value_galaxy.disk_gas.mass <= 0.0
            combined = _add_sized(
                value_galaxy.disk_gas,
                value_galaxy.bulge_gas.mass,
                value_galaxy.bulge_gas.metals,
                value_galaxy.bulge_gas.angular_momentum,
            )
            combined = combined._replace(
                radius=jnp.where(
                    disk_was_empty,
                    value_galaxy.bulge_gas.radius,
                    value_galaxy.disk_gas.radius,
                )
            )
            empty = value_galaxy.bulge_gas._replace(
                mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0
            )
            return value._replace(galaxy=value_galaxy._replace(disk_gas=combined, bulge_gas=empty))

        projected = jax.lax.cond(residual, return_residual, lambda value: value, projected)
        diagnostics = SharkStarburstDiagnostics(
            active=jnp.asarray(True),
            black_hole_accretion_time=accretion_time,
            black_hole_transfer=transfer.mass,
            black_hole_metal_transfer=transfer.metals,
            mean_star_formation_rate=flow_diagnostics["mean_star_formation_rate"],
            mean_formed_stellar_metallicity=flow_diagnostics["mean_formed_stellar_metallicity"],
            rhs_evaluations=evaluations,
            accepted_steps=accepted,
            rejected_steps=rejected,
        )
        return SharkStarburstResult(projected, diagnostics)

    def inactive(inactive_state):
        zero = jnp.zeros_like(inactive_state.galaxy.bulge_gas.mass)
        diagnostics = SharkStarburstDiagnostics(
            active=jnp.asarray(False),
            black_hole_accretion_time=zero,
            black_hole_transfer=zero,
            black_hole_metal_transfer=zero,
            mean_star_formation_rate=zero,
            mean_formed_stellar_metallicity=zero,
            rhs_evaluations=jnp.asarray(0, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
        )
        return SharkStarburstResult(inactive_state, diagnostics)

    return jax.lax.cond(active, evolve, inactive, state)
