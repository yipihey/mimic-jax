"""Faithful FoF-group orchestration for the fiducial SAGE16 module schedule."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.cooling_tables import CoolingTables
from mimic_jax.sage16.perturbations import log_fractionally_perturb, process_perturbations
from mimic_jax.sage16.precision import as_float64, require_x64
from mimic_jax.sage16.processes.cooling import apply_cooling
from mimic_jax.sage16.processes.cooling_budget import calculate_cooling_budget
from mimic_jax.sage16.processes.disk_instability import apply_disk_instability
from mimic_jax.sage16.processes.disk_radius import set_disk_scale_radius
from mimic_jax.sage16.processes.infall import apply_infall, prepare_infall_budget
from mimic_jax.sage16.processes.merger_clock import initialise_merger_clocks
from mimic_jax.sage16.processes.mergers import resolve_mergers_and_disruption
from mimic_jax.sage16.processes.quasar_mode import apply_quasar_mode
from mimic_jax.sage16.processes.radio_mode_heating import apply_radio_mode_heating
from mimic_jax.sage16.processes.reincorporation import apply_reincorporation
from mimic_jax.sage16.processes.reionization import apply_reionization
from mimic_jax.sage16.processes.satellite_stripping import apply_satellite_stripping
from mimic_jax.sage16.processes.star_formation import (
    apply_metal_enrichment,
    apply_star_formation_supernova,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
)
from mimic_jax.sage16.processes.starburst import apply_disk_instability_starburst
from mimic_jax.sage16.transfers import (
    CoolingBudget,
    CoolingTransfer,
    DiskInstabilityTransfer,
    MetalEnrichmentTransfer,
    QuasarModeTransfer,
    RadioModeHeatingTransfer,
    SatelliteStrippingTransfer,
    StarburstTransfer,
    StarFormationTransfer,
    UpstreamGroupFinalResult,
    UpstreamGroupGalaxyDiagnostics,
    UpstreamGroupHistoryResult,
    UpstreamGroupPreparationDiagnostics,
    UpstreamGroupPreparationResult,
    UpstreamGroupStepDiagnostics,
    UpstreamGroupStepResult,
)
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)


def _record_at(records, index):
    return jax.tree_util.tree_map(lambda value: value[index], records)


def _set_record(records, index, record):
    return jax.tree_util.tree_map(
        lambda values, value: values.at[index].set(value),
        records,
        record,
    )


def _set_galaxy_and_central(states, galaxy_index, central_index, galaxy, central):
    """Commit a process result without writing the central twice for Type 0."""

    def same_record(current):
        return _set_record(current, galaxy_index, galaxy)

    def distinct_records(current):
        current = _set_record(current, galaxy_index, galaxy)
        return _set_record(current, central_index, central)

    return jax.lax.cond(
        galaxy_index == central_index,
        same_record,
        distinct_records,
        states,
    )


def _validate_central_index(central_index, member_count):
    """Validate concrete indices while permitting scalar traced batch inputs."""

    if isinstance(central_index, int):
        if not 0 <= central_index < member_count:
            raise ValueError("central_index must select a member of the supplied FoF group")
    elif getattr(central_index, "ndim", None) != 0:
        raise ValueError("central_index must be a scalar")


def _zero_galaxy_diagnostics():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return UpstreamGroupGalaxyDiagnostics(
        active=jnp.asarray(False),
        satellite_stripping=SatelliteStrippingTransfer(zero, zero, zero, zero),
        cooling_budget=CoolingBudget(zero, zero, zero),
        radio_mode=RadioModeHeatingTransfer(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        ),
        cooling=CoolingTransfer(zero, zero),
        star_formation=StarFormationTransfer(zero, zero, zero, zero, zero, zero, zero),
        disk_instability=DiskInstabilityTransfer(zero, zero, zero, zero, zero, zero),
        quasar_mode=QuasarModeTransfer(zero, zero, zero, zero, zero, zero, zero, zero, zero),
        starburst=StarburstTransfer(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        ),
        enrichment=MetalEnrichmentTransfer(zero, zero, zero),
    )


def prepare_upstream_sequential_group(
    states: GalaxyState,
    halos: HaloForcing,
    context: StepContext,
    central_index: int,
    parameters: Sage16Parameters,
    units: Sage16Units,
) -> UpstreamGroupPreparationResult:
    """Run the four fiducial SAGE16 pre-timestep modules in configured order."""

    require_x64()
    member_count = states.HotGas.shape[0]
    _validate_central_index(central_index, member_count)

    reionized = jax.vmap(
        lambda state, halo: apply_reionization(state, halo, context, parameters, units)
    )(states, halos)
    infall_budget = prepare_infall_budget(
        reionized.state,
        halos,
        central_index,
        parameters,
    )
    disk_radius = jax.vmap(set_disk_scale_radius)(infall_budget.states, halos)
    merger_clock = initialise_merger_clocks(disk_radius.state, halos, units)
    diagnostics = UpstreamGroupPreparationDiagnostics(
        reionization_modifiers=reionized.modifier,
        infall_budget=infall_budget.transfer,
        disk_radius=disk_radius,
        merger_clock=merger_clock.diagnostics,
    )
    return UpstreamGroupPreparationResult(merger_clock.states, diagnostics)


def upstream_sequential_group_substep(
    states: GalaxyState,
    halos: HaloForcing,
    context: StepContext,
    central_index: int,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
) -> UpstreamGroupStepResult:
    """Execute one fiducial FoF substep with upstream live-state ordering.

    Full-halo infall and reincorporation run first on the FoF central.  The
    remaining physics is then galaxy-major: one live member completes every
    configured module before the next member begins.  The merger/disruption
    event phase runs last and dispatches its consumers in source order.
    """

    require_x64()
    member_count = states.HotGas.shape[0]
    _validate_central_index(central_index, member_count)
    if perturbations is None:
        perturbations = process_perturbations()

    central = _record_at(states, central_index)
    central_halo = _record_at(halos, central_index)
    infall = apply_infall(central, context, perturbations.infall)
    reincorporated = apply_reincorporation(
        infall.state,
        central_halo,
        context,
        parameters,
        perturbations.reincorporation,
    )
    states = _set_record(states, central_index, reincorporated.state)

    def scan_galaxy(current_states, galaxy_index):
        active = halos.Type[galaxy_index] != 3

        def process_live(live_states):
            galaxy = _record_at(live_states, galaxy_index)
            halo = _record_at(halos, galaxy_index)
            fof_central = _record_at(live_states, central_index)
            stripping = apply_satellite_stripping(
                galaxy,
                fof_central,
                halo,
                context,
                parameters,
                perturbations.satellite_stripping,
            )
            live_states = _set_galaxy_and_central(
                live_states,
                galaxy_index,
                central_index,
                stripping.satellite,
                stripping.central,
            )

            galaxy = _record_at(live_states, galaxy_index)
            cooling_budget = calculate_cooling_budget(
                galaxy,
                halo,
                context,
                units,
                cooling_tables,
            )
            galaxy = cooling_budget.state._replace(
                CoolingGas=log_fractionally_perturb(
                    cooling_budget.state.CoolingGas,
                    perturbations.cooling,
                )
            )
            radio_mode = apply_radio_mode_heating(
                galaxy,
                halo,
                context,
                parameters,
                units,
                perturbations.agn_heating,
            )
            cooling = apply_cooling(radio_mode.state, halo)

            star_formation_budget = calculate_star_formation_budget(
                cooling.state,
                halo,
                context,
                parameters,
                perturbations.star_formation,
            )
            fof_central_halo = _record_at(halos, central_index)
            supernova_budget = calculate_supernova_feedback_budget(
                cooling.state,
                fof_central_halo,
                parameters,
                units,
                star_formation_budget,
                perturbations.sn_reheating,
                perturbations.sn_ejection,
            )
            fof_central = _record_at(live_states, central_index)
            star_formation = apply_star_formation_supernova(
                cooling.state,
                fof_central,
                halo,
                parameters,
                supernova_budget,
            )
            live_states = _set_galaxy_and_central(
                live_states,
                galaxy_index,
                central_index,
                star_formation.galaxy,
                star_formation.central,
            )

            galaxy = _record_at(live_states, galaxy_index)
            instability = apply_disk_instability(
                galaxy,
                halo,
                parameters,
                units,
                perturbations.disk_instability,
            )
            quasar = apply_quasar_mode(
                instability.state,
                halo,
                parameters,
                units,
                None,
                perturbations.quasar_mode,
            )
            fof_central = _record_at(live_states, central_index)
            starburst = apply_disk_instability_starburst(
                quasar.state,
                fof_central,
                halo,
                fof_central_halo,
                parameters,
                units,
                perturbations,
            )
            live_states = _set_galaxy_and_central(
                live_states,
                galaxy_index,
                central_index,
                starburst.galaxy,
                starburst.central,
            )

            galaxy = _record_at(live_states, galaxy_index)
            fof_central = _record_at(live_states, central_index)
            enrichment = apply_metal_enrichment(
                galaxy,
                fof_central,
                fof_central_halo,
                halo.Type == 0,
                parameters,
            )
            live_states = _set_galaxy_and_central(
                live_states,
                galaxy_index,
                central_index,
                enrichment.galaxy,
                enrichment.central,
            )
            diagnostics = UpstreamGroupGalaxyDiagnostics(
                active=jnp.asarray(True),
                satellite_stripping=stripping.transfer,
                cooling_budget=cooling_budget.budget,
                radio_mode=radio_mode.transfer,
                cooling=cooling.transfer,
                star_formation=star_formation.transfer,
                disk_instability=instability.transfer,
                quasar_mode=quasar.transfer,
                starburst=starburst.transfer,
                enrichment=enrichment.transfer,
            )
            return live_states, diagnostics

        def skip_ejected(current):
            return current, _zero_galaxy_diagnostics()

        return jax.lax.cond(active, process_live, skip_ejected, current_states)

    states, galaxy_diagnostics = jax.lax.scan(
        scan_galaxy,
        states,
        jnp.arange(member_count, dtype=jnp.int32),
    )
    mergers = resolve_mergers_and_disruption(
        states,
        halos,
        context,
        parameters,
        units,
        perturbations,
    )
    diagnostics = UpstreamGroupStepDiagnostics(
        infall=infall.transfer,
        reincorporation=reincorporated.transfer,
        galaxies=galaxy_diagnostics,
        mergers=mergers.diagnostics,
    )
    return UpstreamGroupStepResult(
        mergers.states,
        mergers.halos,
        diagnostics,
        mergers.success,
    )


def evolve_upstream_sequential_group_interval(
    initial_states: GalaxyState,
    halos: HaloForcing,
    context: StepContext,
    central_index: int,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    num_substeps: int,
    perturbations=None,
) -> UpstreamGroupHistoryResult:
    """Run pre-timestep setup once and the exact fiducial schedule ``N`` times.

    The tree forcing is piecewise constant over this interval.  ``halo.dT`` is
    retained per member; rate-based kernels divide that object-local duration
    by ``num_substeps`` exactly as the upstream implementation does.
    """

    require_x64()
    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    if perturbations is None:
        perturbations = process_perturbations()

    def expand(value):
        if value.ndim != 0:
            raise ValueError("group-interval perturbations must be scalar")
        return jnp.broadcast_to(value, (num_substeps,) + value.shape)

    context = context._replace(num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32))
    prepared = prepare_upstream_sequential_group(
        initial_states,
        halos,
        context,
        central_index,
        parameters,
        units,
    )
    epoch_perturbations = jax.tree_util.tree_map(expand, perturbations)
    substep_numbers = jnp.arange(num_substeps, dtype=jnp.int32)
    interval_dt = context.time_interval / as_float64(num_substeps)

    def scan_substep(carry, inputs):
        current_states, current_halos = carry
        substep_number, substep_perturbations = inputs
        substep_context = context._replace(
            substep_number=substep_number,
            substep_dt=interval_dt,
            substep_time=(context.time + context.time_interval)
            - (as_float64(substep_number) + 0.5) * interval_dt,
        )
        result = upstream_sequential_group_substep(
            current_states,
            current_halos,
            substep_context,
            central_index,
            parameters,
            units,
            cooling_tables,
            substep_perturbations,
        )
        return (result.states, result.halos), (
            result.states,
            result.halos,
            result.diagnostics,
            result.success,
        )

    (final_states, final_halos), (states, live_halos, diagnostics, successes) = jax.lax.scan(
        scan_substep,
        (prepared.states, halos),
        (substep_numbers, epoch_perturbations),
    )
    return UpstreamGroupHistoryResult(
        final_states=final_states,
        final_halos=final_halos,
        prepared_states=prepared.states,
        preparation=prepared.diagnostics,
        states=states,
        halos=live_halos,
        diagnostics=diagnostics,
        success=jnp.all(successes),
    )


def evolve_upstream_sequential_group_final(
    initial_states: GalaxyState,
    halos: HaloForcing,
    context: StepContext,
    central_index: int,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    num_substeps: int,
    perturbations=None,
) -> UpstreamGroupFinalResult:
    """Run the exact group interval without retaining diagnostic histories.

    This catalogue-production path changes only materialization: it executes
    the same preparation and substep functions as the diagnostic history API.
    """

    require_x64()
    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    if perturbations is None:
        perturbations = process_perturbations()

    context = context._replace(num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32))
    prepared = prepare_upstream_sequential_group(
        initial_states,
        halos,
        context,
        central_index,
        parameters,
        units,
    )
    interval_dt = context.time_interval / as_float64(num_substeps)

    def substep(substep_number, carry):
        current_states, current_halos, success = carry
        substep_context = context._replace(
            substep_number=jnp.asarray(substep_number, dtype=jnp.int32),
            substep_dt=interval_dt,
            substep_time=(context.time + context.time_interval)
            - (as_float64(substep_number) + 0.5) * interval_dt,
        )
        result = upstream_sequential_group_substep(
            current_states,
            current_halos,
            substep_context,
            central_index,
            parameters,
            units,
            cooling_tables,
            perturbations,
        )
        return result.states, result.halos, success & result.success

    final_states, final_halos, success = jax.lax.fori_loop(
        0,
        num_substeps,
        substep,
        (prepared.states, halos, jnp.asarray(True)),
    )
    return UpstreamGroupFinalResult(final_states, final_halos, success)
