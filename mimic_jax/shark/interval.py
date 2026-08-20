"""One-snapshot SHARK reference and continuous/hybrid evolution.

The reference path composes the finite preparation maps, frozen-coefficient
19-variable solve, and post-solve projection in the order used by the pinned
upstream Lagos23 model.  The continuous path evaluates physical transport
rates from the evolving reservoirs and applies only genuine memory/active-set
projections between substeps.  They answer different numerical questions and
are never silently substituted for one another.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mimic_jax.numerics import RK4, fixed_step_update, integrate_adaptive, integrate_fixed_step
from mimic_jax.shark.components import (
    TYPE2_GALAXY,
    SharkSystemState,
)
from mimic_jax.shark.flows import shark_augmented_continuous_rhs_from_rates, shark_rhs_from_rates
from mimic_jax.shark.hybrid import (
    apply_black_hole_seed,
    apply_cooling_staging_transfer,
    apply_cosmological_infall,
    apply_hot_halo_black_hole_transfer,
    apply_reincorporation_transfer,
    enforce_baryon_fraction_limit,
    flow_state_from_system,
    project_flow_state_to_system,
)
from mimic_jax.shark.prescriptions.agn import (
    HeatingRadiusState,
    Lagos23AgnParameters,
    cooling_rate_after_heating_radius,
    griffin19_accretion_spin_upstream_rng,
    hot_halo_accretion_rate_for_saturated_heating,
    lagos23_agn_cooling_response,
    lagos23_agn_parameters,
    lagos23_hot_halo_accretion_rate,
    lagos23_qso_outflow_loadings,
    project_lagos23_heating_radius,
)
from mimic_jax.shark.prescriptions.cooling import (
    CoolingFunctionTable,
    Croton06CoolingParameters,
    cloudy_cie_cooling_table,
    cooling_luminosity_1e40_erg_per_s,
    croton06_unheated_cooling_from_table,
    lagos23_croton06_cooling_parameters,
    pseudo_cooling_luminosity,
)
from mimic_jax.shark.prescriptions.disk import lagos23_disk_flow_rates, lagos23_disk_forcing
from mimic_jax.shark.prescriptions.reincorporation import (
    ReincorporationParameters,
    continuous_reincorporation_rate,
    lagos23_reincorporation_parameters,
    reference_reincorporated_mass,
)
from mimic_jax.shark.prescriptions.reionisation import (
    Sobacchi13ReionisationParameters,
    sobacchi13_reionisation_parameters,
    sobacchi13_reionised_halo,
)
from mimic_jax.shark.prescriptions.star_formation import (
    Lagos23StarFormationParameters,
    lagos23_star_formation_parameters,
)
from mimic_jax.shark.prescriptions.stellar_feedback import (
    Lagos13FeedbackParameters,
    lagos13_feedback_parameters,
)
from mimic_jax.shark.prescriptions.structure import (
    SharkCosmology,
    number_density_200crit_per_cm3,
    quasi_hydrostatic_halo,
)
from mimic_jax.shark.types import (
    SharkAugmentedFlowRates,
    SharkContinuousState,
    SharkFlowParameters,
    shark_flow_parameters,
)

Array = Any


class Lagos23ModelParameters(NamedTuple):
    """All parameters required by a fiducial Lagos23 flow interval."""

    cosmology: SharkCosmology
    flow: SharkFlowParameters
    cooling: Croton06CoolingParameters
    star_formation: Lagos23StarFormationParameters
    stellar_feedback: Lagos13FeedbackParameters
    reincorporation: ReincorporationParameters
    reionisation: Sobacchi13ReionisationParameters
    agn: Lagos23AgnParameters
    black_hole_seed_mass: Array
    black_hole_seed_halo_mass: Array
    limit_baryon_fraction: Array


class SharkIntervalForcing(NamedTuple):
    """Tree/structure forcing held fixed over one reference interval.

    The group-level baryon cap inputs are explicit because a single-galaxy
    state cannot infer the inventory of sibling subhalos and galaxies.
    Radii use SHARK's stored ``Mpc/h`` convention; masses use ``Msun/h``.
    """

    redshift: Array
    duration_gyr: Array
    halo_mass: Array
    subhalo_mass: Array
    virial_velocity: Array
    subhalo_velocity: Array
    virial_radius: Array
    halo_dynamical_time: Array
    hot_specific_angular_momentum: Array
    cooling_specific_angular_momentum: Array
    accreted_mass: Array
    maximum_allowed_baryon_accretion: Array
    baryon_fraction_excess_after_infall: Array
    stripped_hot_halo_mass_for_density: Array
    galaxy_velocity: Array
    gas_half_mass_radius: Array
    stellar_half_mass_radius: Array
    is_central_subhalo: Array
    ignore_galaxy_formation: Array
    galaxy_id: Array
    execution_seed: Array


class SharkIntervalDiagnostics(NamedTuple):
    """Named physical and numerical products of one interval."""

    reincorporated_mass: Array
    infall_mass: Array
    baryon_cap_transfer: Array
    unsatisfied_baryon_cap: Array
    black_hole_seeded: Array
    hydrostatic: Array
    reionisation_suppressed: Array
    unheated_cooling_rate: Array
    cooling_rate: Array
    black_hole_accretion_rate: Array
    heating_ratio: Array
    heating_radius: Array
    cooling_transfer: Array
    black_hole_transfer: Array
    mean_star_formation_rate: Array
    mean_formed_stellar_metallicity: Array
    cooling_angular_momentum_projection: Array
    black_hole_angular_momentum_sink: Array
    rhs_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array


class SharkIntervalResult(NamedTuple):
    state: SharkSystemState
    diagnostics: SharkIntervalDiagnostics


class _ContinuousTrackedState(NamedTuple):
    """Physical state plus exact-in-method quadratures for report diagnostics."""

    physical: SharkContinuousState
    cumulative_cooling: Array
    cumulative_reincorporation: Array
    cumulative_black_hole_angular_momentum_sink: Array


def lagos23_model_parameters(
    *,
    cosmology: SharkCosmology = None,
    flow: SharkFlowParameters = None,
    cooling: Croton06CoolingParameters = None,
    star_formation: Lagos23StarFormationParameters = None,
    stellar_feedback: Lagos13FeedbackParameters = None,
    reincorporation: ReincorporationParameters = None,
    reionisation: Sobacchi13ReionisationParameters = None,
    agn: Lagos23AgnParameters = None,
    black_hole_seed_mass: float = 1.0e4,
    black_hole_seed_halo_mass: float = 1.0e10,
    limit_baryon_fraction: bool = False,
) -> Lagos23ModelParameters:
    """Construct the complete fiducial ``sample_lagos23.cfg`` parameter set."""

    # Imports are exposed constructors rather than a configuration parser so
    # the numerical kernel remains typed, immutable, and JAX-compatible.
    from mimic_jax.shark.prescriptions.structure import lagos23_cosmology

    return Lagos23ModelParameters(
        cosmology=lagos23_cosmology() if cosmology is None else cosmology,
        flow=shark_flow_parameters() if flow is None else flow,
        cooling=(lagos23_croton06_cooling_parameters() if cooling is None else cooling),
        star_formation=(
            lagos23_star_formation_parameters() if star_formation is None else star_formation
        ),
        stellar_feedback=(
            lagos13_feedback_parameters() if stellar_feedback is None else stellar_feedback
        ),
        reincorporation=(
            lagos23_reincorporation_parameters() if reincorporation is None else reincorporation
        ),
        reionisation=(
            sobacchi13_reionisation_parameters() if reionisation is None else reionisation
        ),
        agn=lagos23_agn_parameters() if agn is None else agn,
        black_hole_seed_mass=jnp.asarray(black_hole_seed_mass, dtype=jnp.float64),
        black_hole_seed_halo_mass=jnp.asarray(black_hole_seed_halo_mass, dtype=jnp.float64),
        limit_baryon_fraction=jnp.asarray(limit_baryon_fraction, dtype=jnp.bool_),
    )


def shark_interval_forcing(**values) -> SharkIntervalForcing:
    """Construct typed forcing and reject missing or unknown fields."""

    integer_fields = {"galaxy_id", "execution_seed"}
    boolean_fields = {"is_central_subhalo", "ignore_galaxy_formation"}
    unknown = set(values) - set(SharkIntervalForcing._fields)
    missing = set(SharkIntervalForcing._fields) - set(values)
    if unknown or missing:
        raise TypeError(
            f"SHARK interval forcing missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    converted = {}
    for name, value in values.items():
        if name in integer_fields:
            converted[name] = jnp.asarray(value, dtype=jnp.int64)
        elif name in boolean_fields:
            converted[name] = jnp.asarray(value, dtype=jnp.bool_)
        else:
            converted[name] = jnp.asarray(value, dtype=jnp.float64)
    return SharkIntervalForcing(**converted)


def _cooling_solution(state, forcing, parameters, table):
    hubble_h = parameters.cosmology.hubble_h
    hot_and_cold = state.subhalo.hot_halo_gas.mass + state.subhalo.cold_halo_gas.mass
    hot_and_cold_metals = state.subhalo.hot_halo_gas.metals + state.subhalo.cold_halo_gas.metals
    physical_hot = hot_and_cold / hubble_h
    physical_metals = hot_and_cold_metals / hubble_h
    # Preserve the pinned upstream mixed-unit addition for stripped satellite
    # gas; it is an audited reference quirk rather than a new convention.
    density_mass = physical_hot + jnp.where(
        forcing.is_central_subhalo,
        0.0,
        forcing.stripped_hot_halo_mass_for_density,
    )
    return croton06_unheated_cooling_from_table(
        physical_hot,
        physical_metals,
        density_mass,
        forcing.virial_radius / hubble_h,
        forcing.virial_velocity,
        forcing.halo_dynamical_time,
        parameters.cooling,
        table,
    )


def _hydrostatic(solution, forcing, parameters):
    density = number_density_200crit_per_cm3(forcing.redshift, parameters.cosmology)
    return quasi_hydrostatic_halo(
        forcing.halo_mass,
        solution.virial_temperature,
        jnp.power(10.0, solution.log10_cooling_function),
        density,
        forcing.redshift,
        parameters.agn.hot_halo_threshold,
        parameters.cosmology,
    )


def _disk_rates(time, state, disk_forcing, black_hole, parameters):
    base = lagos23_disk_flow_rates(
        time,
        state,
        disk_forcing,
        parameters.star_formation,
        parameters.stellar_feedback,
    )
    positive_gas = state.cold_gas > 0.0
    metallicity = jnp.where(
        positive_gas,
        state.cold_gas_metals / jnp.where(positive_gas, state.cold_gas, 1.0),
        0.0,
    )
    qso = lagos23_qso_outflow_loadings(
        gas_mass=state.cold_gas,
        black_hole_mass_msun_over_h=black_hole.mass,
        hot_halo_accretion_rate_msun_over_h_per_gyr=black_hole.hot_halo_accretion_rate,
        starburst_accretion_rate_msun_over_h_per_gyr=black_hole.starburst_accretion_rate,
        spin=black_hole.spin,
        gas_metallicity=metallicity,
        circular_velocity_km_per_s=disk_forcing.galaxy_velocity,
        star_formation_rate=base.star_formation,
        # These names follow the public function's physical intent; pinned
        # upstream passes the evolving disk baryonic mass and stellar radius.
        bulge_baryonic_mass=state.stellar_mass + state.cold_gas,
        bulge_radius_mpc=disk_forcing.stellar_half_mass_radius,
        parameters=parameters.agn,
    )
    return base._replace(
        qso_reheating_loading=qso.reheating,
        qso_ejection_loading=qso.ejection,
    )


def _prepare_reference_cooling(state, forcing, parameters, table):
    """Apply the ordered finite cooling preparation and return diagnostics."""

    duration = forcing.duration_gyr
    requested_reincorporation = reference_reincorporated_mass(
        state.subhalo.ejected_gas.mass,
        forcing.halo_mass,
        duration,
        ~forcing.is_central_subhalo,
        parameters.reincorporation,
    )
    state = apply_reincorporation_transfer(state, requested_reincorporation)
    state, infall = apply_cosmological_infall(
        state,
        forcing.accreted_mass,
        forcing.maximum_allowed_baryon_accretion,
        parameters.flow.pre_enrichment_metallicity,
    )
    cap_active = parameters.limit_baryon_fraction & forcing.is_central_subhalo
    cap_request = jnp.where(cap_active, forcing.baryon_fraction_excess_after_infall, 0.0)
    state, cap_transfer, unsatisfied = enforce_baryon_fraction_limit(state, cap_request)
    state, seeded = apply_black_hole_seed(
        state,
        forcing.halo_mass,
        parameters.black_hole_seed_halo_mass,
        parameters.black_hole_seed_mass,
    )
    hot = state.subhalo.hot_halo_gas._replace(
        angular_momentum=(state.subhalo.hot_halo_gas.mass * forcing.hot_specific_angular_momentum)
    )
    state = state._replace(subhalo=state.subhalo._replace(hot_halo_gas=hot))

    ineligible = (
        (state.galaxy.galaxy_type == TYPE2_GALAXY)
        | forcing.ignore_galaxy_formation
        | (state.subhalo.hot_halo_gas.mass <= 0.0)
    )
    reionised = sobacchi13_reionised_halo(
        forcing.virial_velocity, forcing.redshift, parameters.reionisation
    )
    solution = _cooling_solution(state, forcing, parameters, table)
    nominal_physical_bh_rate = lagos23_hot_halo_accretion_rate(
        pseudo_cooling_luminosity(solution.virial_temperature, solution.log10_cooling_function),
        state.galaxy.black_hole.mass,
        (state.subhalo.hot_halo_gas.mass + state.subhalo.cold_halo_gas.mass)
        / parameters.cosmology.hubble_h
        / forcing.halo_mass,
        forcing.virial_velocity,
        parameters.agn,
    )
    spin = griffin19_accretion_spin_upstream_rng(
        state.galaxy.black_hole.mass,
        state.galaxy.black_hole.spin,
        nominal_physical_bh_rate * duration,
        duration,
        forcing.galaxy_id,
        forcing.execution_seed,
        parameters.agn,
    )
    black_hole = state.galaxy.black_hole._replace(spin=spin)
    state = state._replace(galaxy=state.galaxy._replace(black_hole=black_hole))
    hydrostatic = _hydrostatic(solution, forcing, parameters)
    luminosity = cooling_luminosity_1e40_erg_per_s(
        solution.log10_cooling_function,
        solution.cooling_radius,
        forcing.virial_radius / parameters.cosmology.hubble_h,
        (state.subhalo.hot_halo_gas.mass + state.subhalo.cold_halo_gas.mass)
        / parameters.cosmology.hubble_h,
        parameters.cooling.core_radius_fraction,
    )
    response = lagos23_agn_cooling_response(
        pseudo_cooling_luminosity=pseudo_cooling_luminosity(
            solution.virial_temperature, solution.log10_cooling_function
        ),
        cooling_luminosity=luminosity,
        unheated_cooling_rate=solution.cooling_rate,
        cooling_radius_mpc=solution.cooling_radius,
        black_hole_mass_msun_over_h=state.galaxy.black_hole.mass,
        black_hole_starburst_accretion_rate_msun_over_h_per_gyr=(
            state.galaxy.black_hole.starburst_accretion_rate
        ),
        black_hole_spin=state.galaxy.black_hole.spin,
        hot_gas_fraction=(
            (state.subhalo.hot_halo_gas.mass + state.subhalo.cold_halo_gas.mass)
            / parameters.cosmology.hubble_h
            / forcing.halo_mass
        ),
        virial_velocity_km_per_s=forcing.virial_velocity,
        hydrostatic=hydrostatic,
        parameters=parameters.agn,
    )
    heating_state = project_lagos23_heating_radius(
        HeatingRadiusState(state.subhalo.heating_radius),
        response.candidate_heating_radius,
        forcing.redshift,
        parameters.agn,
    )
    physical_cooling, heating_ratio, saturated = cooling_rate_after_heating_radius(
        solution.cooling_rate,
        solution.cooling_radius,
        heating_state,
        parameters.agn,
    )
    saturated_bh_rate = hot_halo_accretion_rate_for_saturated_heating(
        solution.cooling_rate,
        forcing.virial_velocity,
        state.galaxy.black_hole.spin,
    )
    physical_bh_rate = jnp.where(saturated, saturated_bh_rate, response.black_hole_accretion_rate)
    eligible = ~(ineligible | reionised)
    stored_bh_rate = jnp.where(eligible, physical_bh_rate * parameters.cosmology.hubble_h, 0.0)
    stored_cooling_rate = jnp.where(eligible, physical_cooling * parameters.cosmology.hubble_h, 0.0)
    subhalo = state.subhalo._replace(
        heating_radius=jnp.where(
            eligible, heating_state.heating_radius_mpc, state.subhalo.heating_radius
        )
    )
    halo = state.halo._replace(hydrostatic=jnp.where(eligible, hydrostatic, state.halo.hydrostatic))
    state = state._replace(subhalo=subhalo, halo=halo)
    state, black_hole_transfer = apply_hot_halo_black_hole_transfer(state, stored_bh_rate, duration)
    state, cooling_transfer, realized_cooling_rate, cooling_j_residual = (
        apply_cooling_staging_transfer(
            state,
            stored_cooling_rate,
            duration,
            forcing.cooling_specific_angular_momentum,
        )
    )
    return (
        state,
        requested_reincorporation,
        infall,
        cap_transfer.mass,
        unsatisfied,
        seeded,
        hydrostatic,
        reionised,
        solution.cooling_rate * parameters.cosmology.hubble_h,
        realized_cooling_rate,
        state.galaxy.black_hole.hot_halo_accretion_rate,
        heating_ratio,
        cooling_transfer,
        black_hole_transfer,
        cooling_j_residual,
    )


def evolve_shark_reference_interval(
    state: SharkSystemState,
    forcing: SharkIntervalForcing,
    parameters: Lagos23ModelParameters,
    *,
    cooling_table: CoolingFunctionTable = None,
    method: str = RK4,
    num_steps: int = 64,
    adaptive: bool = False,
    relative_tolerance: float = 0.05,
) -> SharkIntervalResult:
    """Evolve one interval with upstream preparation/order semantics.

    ``adaptive=False`` uses a reproducible fixed-step solve for convergence
    studies.  ``adaptive=True`` uses mimic-jax Dormand--Prince with upstream's
    nominal 5% relative tolerance; it preserves the reference process order,
    but is deliberately not described as bitwise GSL Cash--Karp equivalence.
    """

    table = cloudy_cie_cooling_table() if cooling_table is None else cooling_table
    prepared = _prepare_reference_cooling(state, forcing, parameters, table)
    prepared_state = prepared[0]
    flow_initial = flow_state_from_system(prepared_state)
    cold_halo = prepared_state.subhalo.cold_halo_gas
    cooling_metallicity = jnp.where(cold_halo.mass > 0.0, cold_halo.metals / cold_halo.mass, 0.0)
    disk_forcing = lagos23_disk_forcing(
        gas_half_mass_radius=forcing.gas_half_mass_radius,
        stellar_half_mass_radius=forcing.stellar_half_mass_radius,
        redshift=forcing.redshift,
        galaxy_velocity=forcing.galaxy_velocity,
        subhalo_velocity=forcing.subhalo_velocity,
        cooling_rate=prepared[9],
        cooling_metallicity=cooling_metallicity,
        cooling_specific_angular_momentum=forcing.cooling_specific_angular_momentum,
    )
    frozen_black_hole = prepared_state.galaxy.black_hole

    def rhs(time, flow_state):
        rates = _disk_rates(time, flow_state, disk_forcing, frozen_black_hole, parameters)
        return shark_rhs_from_rates(time, flow_state, rates, parameters.flow).derivative

    if adaptive:
        scale_floor = jax.tree_util.tree_map(
            lambda value: jnp.maximum(jnp.abs(value) * 1.0e-12, 1.0e-12), flow_initial
        )
        solution = integrate_adaptive(
            rhs,
            flow_initial,
            duration=forcing.duration_gyr,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=scale_floor,
            initial_step=forcing.duration_gyr,
            max_steps=4096,
            max_attempts=16384,
        )
        flow_final = solution.final_state
        rhs_evaluations = solution.rhs_evaluations
        accepted_steps = solution.accepted_steps
        rejected_steps = solution.rejected_steps
    else:
        solution = integrate_fixed_step(
            rhs,
            flow_initial,
            start_time=0.0,
            duration=forcing.duration_gyr,
            num_steps=num_steps,
            method=method,
        )
        flow_final = solution.final_state
        evaluations_per_step = {"forward_euler": 1, "heun_rk2": 2, "rk4": 4}[method]
        rhs_evaluations = jnp.asarray(num_steps * evaluations_per_step, dtype=jnp.int32)
        accepted_steps = jnp.asarray(num_steps, dtype=jnp.int32)
        rejected_steps = jnp.asarray(0, dtype=jnp.int32)
    final, flow_diagnostics = project_flow_state_to_system(
        prepared_state, flow_final, forcing.duration_gyr
    )
    diagnostics = SharkIntervalDiagnostics(
        reincorporated_mass=prepared[1],
        infall_mass=prepared[2],
        baryon_cap_transfer=prepared[3],
        unsatisfied_baryon_cap=prepared[4],
        black_hole_seeded=prepared[5],
        hydrostatic=prepared[6],
        reionisation_suppressed=prepared[7],
        unheated_cooling_rate=prepared[8],
        cooling_rate=prepared[9],
        black_hole_accretion_rate=prepared[10],
        heating_ratio=prepared[11],
        heating_radius=final.subhalo.heating_radius,
        cooling_transfer=prepared[12].mass,
        black_hole_transfer=prepared[13].mass,
        mean_star_formation_rate=flow_diagnostics["mean_star_formation_rate"],
        mean_formed_stellar_metallicity=flow_diagnostics["mean_formed_stellar_metallicity"],
        cooling_angular_momentum_projection=prepared[14],
        black_hole_angular_momentum_sink=prepared[13].angular_momentum,
        rhs_evaluations=rhs_evaluations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
    )
    return SharkIntervalResult(final, diagnostics)


def _continuous_state_from_system(state):
    reservoirs = flow_state_from_system(state)
    return SharkContinuousState(
        reservoirs=reservoirs,
        black_hole_mass=state.galaxy.black_hole.mass,
        black_hole_metals=state.galaxy.black_hole.metals,
        black_hole_spin=state.galaxy.black_hole.spin,
        heating_radius=state.subhalo.heating_radius,
        excess_jet_power=state.halo.excess_jet_power,
    )


def _system_from_continuous(template, continuous, duration):
    projected, diagnostics = project_flow_state_to_system(template, continuous.reservoirs, duration)
    black_hole = projected.galaxy.black_hole._replace(
        mass=continuous.black_hole_mass,
        metals=continuous.black_hole_metals,
        spin=continuous.black_hole_spin,
    )
    return (
        projected._replace(
            galaxy=projected.galaxy._replace(black_hole=black_hole),
            subhalo=projected.subhalo._replace(heating_radius=continuous.heating_radius),
            halo=projected.halo._replace(excess_jet_power=continuous.excess_jet_power),
        ),
        diagnostics,
    )


def evolve_shark_continuous_interval(
    state: SharkSystemState,
    forcing: SharkIntervalForcing,
    parameters: Lagos23ModelParameters,
    *,
    cooling_table: CoolingFunctionTable = None,
    num_substeps: int = 64,
    method: str = RK4,
) -> SharkIntervalResult:
    """Evolve legitimate transports as rates with explicit hybrid projections.

    Infall is spread uniformly over the tree interval; reincorporation,
    cooling, star formation, SN/QSO feedback, BH growth, metals, and angular
    momentum are reevaluated from the evolving state.  The AGN heating-radius
    running maximum remains a projection after each baryonic substep.  Spin
    orientation is one explicit event after the interval, matching the fact
    that it is not a deterministic ODE state.
    """

    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    table = cloudy_cie_cooling_table() if cooling_table is None else cooling_table
    # Seeding and the group baryon ceiling are genuine threshold/projection
    # operations. Infall itself is left for the continuous source below.
    state, seeded = apply_black_hole_seed(
        state,
        forcing.halo_mass,
        parameters.black_hole_seed_halo_mass,
        parameters.black_hole_seed_mass,
    )
    cap_active = parameters.limit_baryon_fraction & forcing.is_central_subhalo
    cap_request = jnp.where(cap_active, forcing.baryon_fraction_excess_after_infall, 0.0)
    state, cap_transfer, unsatisfied = enforce_baryon_fraction_limit(state, cap_request)
    initial_physical = _continuous_state_from_system(state)
    duration = forcing.duration_gyr
    step_size = duration / num_substeps
    infall_rate = jnp.where(
        duration > 0.0,
        jnp.minimum(
            jnp.maximum(forcing.accreted_mass, 0.0),
            jnp.maximum(forcing.maximum_allowed_baryon_accretion, 0.0),
        )
        / duration,
        0.0,
    )
    disk_forcing_base = lagos23_disk_forcing(
        gas_half_mass_radius=forcing.gas_half_mass_radius,
        stellar_half_mass_radius=forcing.stellar_half_mass_radius,
        redshift=forcing.redshift,
        galaxy_velocity=forcing.galaxy_velocity,
        subhalo_velocity=forcing.subhalo_velocity,
        cooling_specific_angular_momentum=forcing.cooling_specific_angular_momentum,
    )

    def rates_and_candidate(time, current):
        del time
        reservoirs = current.reservoirs
        physical_hot = reservoirs.hot_halo_gas / parameters.cosmology.hubble_h
        physical_metals = reservoirs.hot_halo_gas_metals / parameters.cosmology.hubble_h
        cooling = croton06_unheated_cooling_from_table(
            physical_hot,
            physical_metals,
            physical_hot,
            forcing.virial_radius / parameters.cosmology.hubble_h,
            forcing.virial_velocity,
            forcing.halo_dynamical_time,
            parameters.cooling,
            table,
        )
        hydrostatic = _hydrostatic(cooling, forcing, parameters)
        luminosity = cooling_luminosity_1e40_erg_per_s(
            cooling.log10_cooling_function,
            cooling.cooling_radius,
            forcing.virial_radius / parameters.cosmology.hubble_h,
            physical_hot,
            parameters.cooling.core_radius_fraction,
        )
        response = lagos23_agn_cooling_response(
            pseudo_cooling_luminosity=pseudo_cooling_luminosity(
                cooling.virial_temperature, cooling.log10_cooling_function
            ),
            cooling_luminosity=luminosity,
            unheated_cooling_rate=cooling.cooling_rate,
            cooling_radius_mpc=cooling.cooling_radius,
            black_hole_mass_msun_over_h=current.black_hole_mass,
            black_hole_starburst_accretion_rate_msun_over_h_per_gyr=(
                state.galaxy.black_hole.starburst_accretion_rate
            ),
            black_hole_spin=current.black_hole_spin,
            hot_gas_fraction=physical_hot / forcing.halo_mass,
            virial_velocity_km_per_s=forcing.virial_velocity,
            hydrostatic=hydrostatic,
            parameters=parameters.agn,
        )
        regulated, ratio, saturated = cooling_rate_after_heating_radius(
            cooling.cooling_rate,
            cooling.cooling_radius,
            HeatingRadiusState(current.heating_radius),
            parameters.agn,
        )
        limited_bh = hot_halo_accretion_rate_for_saturated_heating(
            cooling.cooling_rate,
            forcing.virial_velocity,
            current.black_hole_spin,
        )
        physical_bh = jnp.where(saturated, limited_bh, response.black_hole_accretion_rate)
        suppressed = sobacchi13_reionised_halo(
            forcing.virial_velocity, forcing.redshift, parameters.reionisation
        )
        eligible = ~(
            suppressed
            | forcing.ignore_galaxy_formation
            | (state.galaxy.galaxy_type == TYPE2_GALAXY)
        )
        stored_cooling = jnp.where(eligible, regulated * parameters.cosmology.hubble_h, 0.0)
        stored_bh = jnp.where(eligible, physical_bh * parameters.cosmology.hubble_h, 0.0)
        black_hole = state.galaxy.black_hole._replace(
            mass=current.black_hole_mass,
            metals=current.black_hole_metals,
            spin=current.black_hole_spin,
            hot_halo_accretion_rate=stored_bh,
        )
        disk_forcing = disk_forcing_base._replace(
            cooling_rate=stored_cooling,
            cooling_metallicity=jnp.where(
                reservoirs.hot_halo_gas > 0.0,
                reservoirs.hot_halo_gas_metals / reservoirs.hot_halo_gas,
                0.0,
            ),
        )
        disk_rates = _disk_rates(0.0, reservoirs, disk_forcing, black_hole, parameters)
        reincorporation = continuous_reincorporation_rate(
            reservoirs.ejected_gas,
            forcing.halo_mass,
            ~forcing.is_central_subhalo,
            parameters.reincorporation,
        )
        augmented = SharkAugmentedFlowRates(
            reservoirs=disk_rates,
            hot_halo_black_hole_accretion=stored_bh,
            reincorporation=reincorporation,
        )
        return augmented, response.candidate_heating_radius, cooling, ratio, hydrostatic

    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def rhs(time, tracked):
        current = tracked.physical
        augmented, _, _, _, _ = rates_and_candidate(time, current)
        augmented_result = shark_augmented_continuous_rhs_from_rates(
            time, current, augmented, parameters.flow
        )
        derivative = augmented_result.derivative
        central_infall = jnp.where(forcing.is_central_subhalo, infall_rate, 0.0)
        reservoir_addition = derivative.reservoirs._replace(
            hot_halo_gas=derivative.reservoirs.hot_halo_gas + central_infall,
            hot_halo_gas_metals=(
                derivative.reservoirs.hot_halo_gas_metals
                + central_infall * parameters.flow.pre_enrichment_metallicity
            ),
            hot_halo_angular_momentum=(
                derivative.reservoirs.hot_halo_angular_momentum
                + central_infall * forcing.hot_specific_angular_momentum
            ),
        )
        physical_derivative = derivative._replace(reservoirs=reservoir_addition)
        return _ContinuousTrackedState(
            physical=physical_derivative,
            cumulative_cooling=augmented.reservoirs.cooling,
            cumulative_reincorporation=augmented.reincorporation,
            cumulative_black_hole_angular_momentum_sink=(
                augmented_result.black_hole_angular_momentum_sink
            ),
        )

    def substep(carry, index):
        del index
        time, current = carry
        advanced = fixed_step_update(rhs, time, current, step_size, method=method)
        _, candidate, _, _, _ = rates_and_candidate(time + step_size, advanced.physical)
        projected = project_lagos23_heating_radius(
            HeatingRadiusState(advanced.physical.heating_radius),
            candidate,
            forcing.redshift,
            parameters.agn,
        )
        advanced = advanced._replace(
            physical=advanced.physical._replace(heating_radius=projected.heating_radius_mpc)
        )
        return (time + step_size, advanced), advanced

    tracked_initial = _ContinuousTrackedState(
        physical=initial_physical,
        cumulative_cooling=zero,
        cumulative_reincorporation=zero,
        cumulative_black_hole_angular_momentum_sink=zero,
    )
    (_, final_tracked), _ = jax.lax.scan(
        substep,
        (zero, tracked_initial),
        jnp.arange(num_substeps),
    )
    final_continuous = final_tracked.physical
    accreted_black_hole_mass = jnp.maximum(
        final_continuous.black_hole_mass - initial_physical.black_hole_mass, 0.0
    )
    final_spin = griffin19_accretion_spin_upstream_rng(
        initial_physical.black_hole_mass,
        initial_physical.black_hole_spin,
        accreted_black_hole_mass,
        duration,
        forcing.galaxy_id,
        forcing.execution_seed,
        parameters.agn,
    )
    final_continuous = final_continuous._replace(black_hole_spin=final_spin)
    final, flow_diagnostics = _system_from_continuous(state, final_continuous, duration)
    final_rates, _, final_cooling, final_ratio, final_hydrostatic = rates_and_candidate(
        duration, final_continuous
    )
    evaluations_per_step = {"forward_euler": 1, "heun_rk2": 2, "rk4": 4}[method]
    diagnostics = SharkIntervalDiagnostics(
        reincorporated_mass=final_tracked.cumulative_reincorporation,
        infall_mass=infall_rate * duration,
        baryon_cap_transfer=cap_transfer.mass,
        unsatisfied_baryon_cap=unsatisfied,
        black_hole_seeded=seeded,
        hydrostatic=final_hydrostatic,
        reionisation_suppressed=sobacchi13_reionised_halo(
            forcing.virial_velocity, forcing.redshift, parameters.reionisation
        ),
        unheated_cooling_rate=(final_cooling.cooling_rate * parameters.cosmology.hubble_h),
        cooling_rate=final_rates.reservoirs.cooling,
        black_hole_accretion_rate=final_rates.hot_halo_black_hole_accretion,
        heating_ratio=final_ratio,
        heating_radius=final.subhalo.heating_radius,
        cooling_transfer=final_tracked.cumulative_cooling,
        black_hole_transfer=accreted_black_hole_mass,
        mean_star_formation_rate=flow_diagnostics["mean_star_formation_rate"],
        mean_formed_stellar_metallicity=flow_diagnostics["mean_formed_stellar_metallicity"],
        cooling_angular_momentum_projection=zero,
        black_hole_angular_momentum_sink=(
            final_tracked.cumulative_black_hole_angular_momentum_sink
        ),
        rhs_evaluations=jnp.asarray(num_substeps * evaluations_per_step, dtype=jnp.int32),
        accepted_steps=jnp.asarray(num_substeps, dtype=jnp.int32),
        rejected_steps=jnp.asarray(0, dtype=jnp.int32),
    )
    return SharkIntervalResult(final, diagnostics)
