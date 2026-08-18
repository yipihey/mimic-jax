"""Discrete-time evolution of the currently implemented central-galaxy subset."""

import jax

from mimic_jax.sage16.perturbations import process_perturbations
from mimic_jax.sage16.processes.cooling import apply_cooling
from mimic_jax.sage16.processes.reincorporation import apply_reincorporation
from mimic_jax.sage16.processes.star_formation import quiescent_disk_step
from mimic_jax.sage16.transfers import CentralHistoryResult, CentralStepDiagnostics
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)


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
