"""Discrete-time evolution of the currently implemented central-galaxy subset."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.cooling_tables import CoolingTables
from mimic_jax.sage16.perturbations import log_fractionally_perturb, process_perturbations
from mimic_jax.sage16.precision import as_float64
from mimic_jax.sage16.processes.cooling import apply_cooling
from mimic_jax.sage16.processes.cooling_budget import calculate_cooling_budget
from mimic_jax.sage16.processes.infall import apply_infall
from mimic_jax.sage16.processes.radio_mode_heating import apply_radio_mode_heating
from mimic_jax.sage16.processes.reincorporation import apply_reincorporation
from mimic_jax.sage16.processes.star_formation import quiescent_disk_step
from mimic_jax.sage16.transfers import (
    CentralHistoryResult,
    CentralStepDiagnostics,
    UpstreamCentralHistoryResult,
    UpstreamCentralStepDiagnostics,
)
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

UPSTREAM_SEQUENTIAL = "upstream_sequential"


def central_quiescent_step(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    cooling_gas,
    parameters: Sage16Parameters,
    units: Sage16Units,
    perturbations=None,
):
    """Apply reincorporation, a supplied cooling budget, and quiescent SF in C order.

    This is a faithful composition of the implemented modules for a Type-0
    central. Infall, cooling-budget calculation, AGN, disk instability, and
    merger events remain outside this deliberately scoped evolution function.
    """

    if perturbations is None:
        perturbations = process_perturbations()
    reincorporated = apply_reincorporation(
        state,
        halo,
        context,
        parameters,
        perturbations.reincorporation,
    )
    cooled = apply_cooling(
        reincorporated.state,
        halo,
        cooling_gas,
        perturbations.cooling,
    )
    quiescent = quiescent_disk_step(
        cooled.state,
        cooled.state,
        halo,
        halo,
        context,
        parameters,
        units,
        perturbations,
    )
    return quiescent.galaxy, CentralStepDiagnostics(
        cooling=cooled.transfer,
        reincorporation=reincorporated.transfer,
        star_formation=quiescent.transfer,
        enrichment=quiescent.enrichment,
    )


def evolve_central_history(
    initial_state: GalaxyState,
    halos: HaloForcing,
    contexts: StepContext,
    cooling_gas,
    parameters: Sage16Parameters,
    units: Sage16Units,
    perturbations=None,
) -> CentralHistoryResult:
    """Evolve independent finite epochs with ``jax.lax.scan``.

    Leading dimensions of ``halos``, ``contexts``, ``cooling_gas``, and every
    perturbation field are epochs ordered forward in cosmic time.
    """

    if perturbations is None:
        perturbations = process_perturbations(
            cooling=0.0 * cooling_gas,
            star_formation=0.0 * cooling_gas,
            sn_reheating=0.0 * cooling_gas,
            sn_ejection=0.0 * cooling_gas,
            reincorporation=0.0 * cooling_gas,
            agn_heating=0.0 * cooling_gas,
            infall=0.0 * cooling_gas,
            satellite_stripping=0.0 * cooling_gas,
        )

    def scan_step(state, inputs):
        halo, context, cooling, epoch_perturbations = inputs
        new_state, diagnostics = central_quiescent_step(
            state,
            halo,
            context,
            cooling,
            parameters,
            units,
            epoch_perturbations,
        )
        return new_state, (new_state, diagnostics)

    final_state, (states, diagnostics) = jax.lax.scan(
        scan_step,
        initial_state,
        (halos, contexts, cooling_gas, perturbations),
    )
    return CentralHistoryResult(final_state, states, diagnostics)


def upstream_sequential_central_step(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
):
    """Run the implemented central subset in exact Mini-Millennium module order.

    This reference update is an explicit sequence of finite process maps:
    infall application, reincorporation, cooling-budget calculation, radio-mode
    heating, cooling application, and the quiescent SF/SN/enrichment chain. It
    is not replaced by an ODE integrator. Snapshot-level infall preparation and
    the later instability/event modules remain outside this per-central map.
    """

    if perturbations is None:
        perturbations = process_perturbations()
    infall = apply_infall(state, context, perturbations.infall)
    reincorporated = apply_reincorporation(
        infall.state,
        halo,
        context,
        parameters,
        perturbations.reincorporation,
    )
    cooling_budget = calculate_cooling_budget(
        reincorporated.state,
        halo,
        context,
        units,
        cooling_tables,
    )
    perturbed_cooling = cooling_budget.state._replace(
        CoolingGas=log_fractionally_perturb(
            cooling_budget.state.CoolingGas,
            perturbations.cooling,
        )
    )
    radio_mode = apply_radio_mode_heating(
        perturbed_cooling,
        halo,
        context,
        parameters,
        units,
        perturbations.agn_heating,
    )
    cooled = apply_cooling(radio_mode.state, halo)
    quiescent = quiescent_disk_step(
        cooled.state,
        cooled.state,
        halo,
        halo,
        context,
        parameters,
        units,
        perturbations,
    )
    diagnostics = UpstreamCentralStepDiagnostics(
        infall=infall.transfer,
        cooling_budget=cooling_budget.budget,
        radio_mode=radio_mode.transfer,
        cooling=cooled.transfer,
        reincorporation=reincorporated.transfer,
        star_formation=quiescent.transfer,
        enrichment=quiescent.enrichment,
    )
    return quiescent.galaxy, diagnostics


def evolve_upstream_sequential_central_history(
    initial_state: GalaxyState,
    halos: HaloForcing,
    contexts: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
) -> UpstreamCentralHistoryResult:
    """Scan the faithful sequential update for the implemented central subset."""

    if perturbations is None:
        zero = 0.0 * contexts.redshift
        perturbations = process_perturbations(
            cooling=zero,
            star_formation=zero,
            sn_reheating=zero,
            sn_ejection=zero,
            reincorporation=zero,
            agn_heating=zero,
            infall=zero,
            satellite_stripping=zero,
        )

    def scan_step(state, inputs):
        halo, context, epoch_perturbations = inputs
        new_state, diagnostics = upstream_sequential_central_step(
            state,
            halo,
            context,
            parameters,
            units,
            cooling_tables,
            epoch_perturbations,
        )
        return new_state, (new_state, diagnostics)

    final_state, (states, diagnostics) = jax.lax.scan(
        scan_step,
        initial_state,
        (halos, contexts, perturbations),
    )
    return UpstreamCentralHistoryResult(final_state, states, diagnostics)


def subcycle_upstream_sequential_central(
    initial_state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    num_substeps: int,
    perturbations=None,
) -> UpstreamCentralHistoryResult:
    """Resolve one fixed-forcing tree interval into explicit SAGE substeps.

    Halo forcing is piecewise constant in this initial API. ``halo.dT`` remains
    the object's full tree-interval duration, while each rate-based module uses
    ``halo.dT / num_substeps`` exactly as upstream does.
    """

    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    if perturbations is None:
        zero = 0.0 * context.redshift
        perturbations = process_perturbations(
            cooling=zero,
            star_formation=zero,
            sn_reheating=zero,
            sn_ejection=zero,
            reincorporation=zero,
            agn_heating=zero,
            infall=zero,
            satellite_stripping=zero,
        )

    def expand(value):
        if value.ndim != 0:
            raise ValueError(
                "subcycle perturbations must be scalar over the tree interval; "
                "use evolve_upstream_sequential_central_history for per-epoch values"
            )
        return jnp.broadcast_to(value, (num_substeps,) + value.shape)

    epoch_perturbations = jax.tree_util.tree_map(expand, perturbations)
    substep_numbers = jnp.arange(num_substeps, dtype=jnp.int32)
    interval_dt = context.time_interval / num_substeps

    def scan_step(state, inputs):
        substep_number, epoch_perturbation = inputs
        substep_context = context._replace(
            substep_number=substep_number,
            num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32),
            substep_dt=interval_dt,
            substep_time=(context.time + context.time_interval)
            - (as_float64(substep_number) + 0.5) * interval_dt,
        )
        new_state, diagnostics = upstream_sequential_central_step(
            state,
            halo,
            substep_context,
            parameters,
            units,
            cooling_tables,
            epoch_perturbation,
        )
        return new_state, (new_state, diagnostics)

    final_state, (states, diagnostics) = jax.lax.scan(
        scan_step,
        initial_state,
        (substep_numbers, epoch_perturbations),
    )
    return UpstreamCentralHistoryResult(final_state, states, diagnostics)
