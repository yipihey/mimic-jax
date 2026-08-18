"""Ordered fiducial SAGE16 merger/disruption event map."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import process_perturbations
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.disk_instability import apply_disk_instability
from mimic_jax.sage16.processes.quasar_mode import apply_quasar_mode
from mimic_jax.sage16.processes.starburst import apply_collisional_starburst
from mimic_jax.sage16.transfers import (
    DiskInstabilityTransfer,
    MergerOwnershipTransfer,
    MergerResolutionDiagnostics,
    MergerResolutionResult,
    QuasarModeTransfer,
    StarburstTransfer,
)
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

MERGER_ACTION_NONE = 0
MERGER_ACTION_DISRUPTION = 1
MERGER_ACTION_MERGER = 2

MERGER_ERROR_NONE = 0
MERGER_ERROR_INVALID_CONTEXT = 1
MERGER_ERROR_INVALID_DT = 2
MERGER_ERROR_UNSET_CLOCK = 3
MERGER_ERROR_INVALID_TARGET = 4

MERGER_STATUS_NONE = 0
MERGER_STATUS_INITIAL_BOUNDARY = 1
MERGER_STATUS_NOT_ELIGIBLE = 2
MERGER_STATUS_NONFINITE_CLOCK = 3

_MERGTIME_UNSET_THRESHOLD = 999.0


def _state_at(states: GalaxyState, index):
    return jax.tree_util.tree_map(lambda value: value[index], states)


def _set_state(states: GalaxyState, index, state: GalaxyState):
    return jax.tree_util.tree_map(lambda values, value: values.at[index].set(value), states, state)


def _halo_at(halos: HaloForcing, live_types, index):
    halo = jax.tree_util.tree_map(lambda value: value[index], halos)
    return halo._replace(Type=live_types[index])


def _zero_ownership():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return MergerOwnershipTransfer(*((zero,) * len(MergerOwnershipTransfer._fields)))


def _zero_quasar():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return QuasarModeTransfer(*((zero,) * len(QuasarModeTransfer._fields)))


def _zero_starburst():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return StarburstTransfer(*((zero,) * len(StarburstTransfer._fields)))


def _zero_instability():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return DiskInstabilityTransfer(*((zero,) * len(DiskInstabilityTransfer._fields)))


def _diagnostics(
    *,
    action=MERGER_ACTION_NONE,
    error=MERGER_ERROR_NONE,
    status=MERGER_STATUS_NONE,
    target_index=-1,
    eligible=False,
    current_mvir=0.0,
    virial_to_baryons=0.0,
    mass_ratio=0.0,
    source_dt=0.0,
    source_time=0.0,
    ownership=None,
    merger_quasar=None,
    merger_starburst=None,
    post_instability=None,
    post_quasar=None,
    post_starburst=None,
):
    integer = lambda value: jnp.asarray(value, dtype=jnp.int32)
    scalar = lambda value: jnp.asarray(value, dtype=jnp.float64)
    return MergerResolutionDiagnostics(
        action=integer(action),
        error=integer(error),
        status=integer(status),
        target_index=integer(target_index),
        eligible=jnp.asarray(eligible, dtype=jnp.bool_),
        current_mvir=scalar(current_mvir),
        virial_to_baryons=scalar(virial_to_baryons),
        mass_ratio=scalar(mass_ratio),
        source_dt=scalar(source_dt),
        source_time=scalar(source_time),
        ownership=_zero_ownership() if ownership is None else ownership,
        merger_quasar=_zero_quasar() if merger_quasar is None else merger_quasar,
        merger_starburst=_zero_starburst() if merger_starburst is None else merger_starburst,
        post_instability=_zero_instability() if post_instability is None else post_instability,
        post_quasar=_zero_quasar() if post_quasar is None else post_quasar,
        post_starburst=_zero_starburst() if post_starburst is None else post_starburst,
    )


def _merger_mass_ratio(source: GalaxyState, target: GalaxyState):
    source_mass = as_float64(source.StellarMass) + as_float64(source.ColdGas)
    target_mass = as_float64(target.StellarMass) + as_float64(target.ColdGas)
    smaller = jnp.minimum(source_mass, target_mass)
    larger = jnp.maximum(source_mass, target_mass)
    return jnp.where(larger > 0.0, smaller / larger, 1.0)


def _apply_merger_ownership(source: GalaxyState, target: GalaxyState):
    updated = target._replace(
        ColdGas=as_float32(target.ColdGas + source.ColdGas),
        MetalsColdGas=as_float32(target.MetalsColdGas + source.MetalsColdGas),
        StellarMass=as_float32(target.StellarMass + source.StellarMass),
        MetalsStellarMass=as_float32(target.MetalsStellarMass + source.MetalsStellarMass),
        HotGas=as_float32(target.HotGas + source.HotGas),
        MetalsHotGas=as_float32(target.MetalsHotGas + source.MetalsHotGas),
        EjectedGas=as_float32(target.EjectedGas + source.EjectedGas),
        MetalsEjectedGas=as_float32(target.MetalsEjectedGas + source.MetalsEjectedGas),
        ICS=as_float32(target.ICS + source.ICS),
        MetalsICS=as_float32(target.MetalsICS + source.MetalsICS),
        BlackHoleMass=as_float32(target.BlackHoleMass + source.BlackHoleMass),
        BulgeMass=as_float32(target.BulgeMass + source.StellarMass),
        MetalsBulgeMass=as_float32(target.MetalsBulgeMass + source.MetalsStellarMass),
    )
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    transfer = MergerOwnershipTransfer(
        cold_to_cold=as_float64(source.ColdGas),
        cold_to_hot=zero,
        hot_to_hot=as_float64(source.HotGas),
        ejected_to_ejected=as_float64(source.EjectedGas),
        stellar_to_stellar=as_float64(source.StellarMass),
        stellar_to_ics=zero,
        ics_to_ics=as_float64(source.ICS),
        black_hole_to_black_hole=as_float64(source.BlackHoleMass),
        black_hole_sink=zero,
        cold_metals_to_cold=as_float64(source.MetalsColdGas),
        cold_metals_to_hot=zero,
        hot_metals_to_hot=as_float64(source.MetalsHotGas),
        ejected_metals_to_ejected=as_float64(source.MetalsEjectedGas),
        stellar_metals_to_stellar=as_float64(source.MetalsStellarMass),
        stellar_metals_to_ics=zero,
        ics_metals_to_ics=as_float64(source.MetalsICS),
        stellar_to_bulge_component=as_float64(source.StellarMass),
        stellar_metals_to_bulge_component=as_float64(source.MetalsStellarMass),
    )
    return updated, transfer


def _apply_disruption_ownership(source: GalaxyState, target: GalaxyState):
    source_hot = as_float32(source.ColdGas + source.HotGas)
    source_hot_metals = as_float32(source.MetalsColdGas + source.MetalsHotGas)
    updated = target._replace(
        HotGas=as_float32(target.HotGas + source_hot),
        MetalsHotGas=as_float32(target.MetalsHotGas + source_hot_metals),
        EjectedGas=as_float32(target.EjectedGas + source.EjectedGas),
        MetalsEjectedGas=as_float32(target.MetalsEjectedGas + source.MetalsEjectedGas),
        ICS=as_float32(target.ICS + source.ICS),
        MetalsICS=as_float32(target.MetalsICS + source.MetalsICS),
    )
    # Upstream uses two separate += writes, so retain the intermediate float rounding.
    updated = updated._replace(
        ICS=as_float32(updated.ICS + source.StellarMass),
        MetalsICS=as_float32(updated.MetalsICS + source.MetalsStellarMass),
    )
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    transfer = MergerOwnershipTransfer(
        cold_to_cold=zero,
        cold_to_hot=as_float64(source.ColdGas),
        hot_to_hot=as_float64(source.HotGas),
        ejected_to_ejected=as_float64(source.EjectedGas),
        stellar_to_stellar=zero,
        stellar_to_ics=as_float64(source.StellarMass),
        ics_to_ics=as_float64(source.ICS),
        black_hole_to_black_hole=zero,
        black_hole_sink=as_float64(source.BlackHoleMass),
        cold_metals_to_cold=zero,
        cold_metals_to_hot=as_float64(source.MetalsColdGas),
        hot_metals_to_hot=as_float64(source.MetalsHotGas),
        ejected_metals_to_ejected=as_float64(source.MetalsEjectedGas),
        stellar_metals_to_stellar=zero,
        stellar_metals_to_ics=as_float64(source.MetalsStellarMass),
        ics_metals_to_ics=as_float64(source.MetalsICS),
        stellar_to_bulge_component=zero,
        stellar_metals_to_bulge_component=zero,
    )
    return updated, transfer


def apply_merger_ownership_event(source: GalaxyState, target: GalaxyState):
    """Apply the differentiable reservoir map for a known merger event.

    Event detection, target identity, and the major/minor branch are discrete.
    Once that event identity is fixed, this ownership map is an ordinary JAX
    function and can supply the two progenitor derivatives of the descendant.
    Immediate quasar/starburst consumers remain explicit downstream maps.
    """

    require_x64()
    return _apply_merger_ownership(source, target)


def apply_disruption_ownership_event(source: GalaxyState, target: GalaxyState):
    """Apply the fixed-identity SAGE disruption ownership map.

    The source black hole is an explicit sink in this event; the returned
    ``MergerOwnershipTransfer`` records it rather than hiding non-conservation.
    """

    require_x64()
    return _apply_disruption_ownership(source, target)


def _resolve_target(halos: HaloForcing, live_types, source_index, fof_central_index):
    count = live_types.shape[0]
    source_type = live_types[source_index]
    candidate = halos.CentralHalo[source_index]
    candidate_valid = (candidate >= 0) & (candidate < count) & (candidate != source_index)
    target = jnp.where((source_type == 2) & candidate_valid, candidate, fof_central_index)
    safe_target = jnp.clip(target, 0, count - 1)
    redirect = halos.CentralHalo[safe_target]
    redirect_valid = (redirect >= 0) & (redirect < count) & (redirect != source_index)
    target_was_consumed = live_types[safe_target] == 3
    target = jnp.where(target_was_consumed, redirect, target)
    valid = (
        (target >= 0)
        & (target < count)
        & (target != source_index)
        & (~target_was_consumed | redirect_valid)
    )
    return target, valid


def _set_target_and_central(states, target_index, central_index, target, central):
    states = _set_state(states, target_index, target)
    return jax.lax.cond(
        target_index == central_index,
        lambda current: current,
        lambda current: _set_state(current, central_index, central),
        states,
    )


def _apply_merger_consumers(
    states,
    halos,
    live_types,
    target_index,
    central_index,
    mass_ratio,
    parameters,
    units,
    perturbations,
):
    target = _state_at(states, target_index)
    target_halo = _halo_at(halos, live_types, target_index)
    central = _state_at(states, central_index)
    central_halo = _halo_at(halos, live_types, central_index)

    merger_quasar = apply_quasar_mode(
        target,
        target_halo,
        parameters,
        units,
        mass_ratio,
        perturbations.quasar_mode,
    )
    states = _set_state(states, target_index, merger_quasar.state)
    central = _state_at(states, central_index)
    merger_starburst = apply_collisional_starburst(
        merger_quasar.state,
        central,
        target_halo,
        central_halo,
        mass_ratio,
        0,
        target_halo.dT,
        parameters,
        units,
        perturbations.starburst,
        perturbations.sn_reheating,
        perturbations.sn_ejection,
    )
    states = _set_target_and_central(
        states,
        target_index,
        central_index,
        merger_starburst.galaxy,
        merger_starburst.central,
    )

    def apply_minor_followup(current_states):
        current_target = _state_at(current_states, target_index)
        instability = apply_disk_instability(
            current_target,
            target_halo,
            parameters,
            units,
            perturbations.disk_instability,
        )
        current_states = _set_state(current_states, target_index, instability.state)

        def apply_triggered(triggered_states):
            triggered_target = _state_at(triggered_states, target_index)
            quasar = apply_quasar_mode(
                triggered_target,
                target_halo,
                parameters,
                units,
                None,
                perturbations.quasar_mode,
            )
            triggered_states = _set_state(triggered_states, target_index, quasar.state)
            triggered_central = _state_at(triggered_states, central_index)
            burst = apply_collisional_starburst(
                quasar.state,
                triggered_central,
                target_halo,
                central_halo,
                quasar.state.UnstableDiskGasFraction,
                1,
                target_halo.dT,
                parameters,
                units,
                perturbations.starburst,
                perturbations.sn_reheating,
                perturbations.sn_ejection,
            )
            triggered_states = _set_target_and_central(
                triggered_states,
                target_index,
                central_index,
                burst.galaxy,
                burst.central,
            )
            return triggered_states, (quasar.transfer, burst.transfer)

        def skip_triggered(triggered_states):
            return triggered_states, (_zero_quasar(), _zero_starburst())

        current_states, (post_quasar, post_starburst) = jax.lax.cond(
            instability.state.UnstableDiskGasFraction > 0.0,
            apply_triggered,
            skip_triggered,
            current_states,
        )
        return current_states, (instability.transfer, post_quasar, post_starburst)

    def skip_minor_followup(current_states):
        return current_states, (_zero_instability(), _zero_quasar(), _zero_starburst())

    states, (post_instability, post_quasar, post_starburst) = jax.lax.cond(
        mass_ratio < parameters.ThresholdMajorMerger,
        apply_minor_followup,
        skip_minor_followup,
        states,
    )
    return (
        states,
        merger_quasar.transfer,
        merger_starburst.transfer,
        post_instability,
        post_quasar,
        post_starburst,
    )


def resolve_mergers_and_disruption(
    states: GalaxyState,
    halos: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    perturbations=None,
) -> MergerResolutionResult:
    """Apply the exact live-order fiducial merger/disruption phase.

    The leading dimension is the FoF workspace order.  Each merger mutates its
    target and runs the configured fiducial quasar/starburst consumers before
    the next source is inspected.  Consumed source records are retained but
    marked Type 3, matching upstream workspace ownership.
    """

    require_x64()
    if perturbations is None:
        perturbations = process_perturbations()

    count = halos.Type.shape[0]
    has_central = jnp.any(halos.Type == 0)
    fof_central_index = jnp.argmax(halos.Type == 0).astype(jnp.int32)
    invalid_context = context.num_substeps <= 0

    def scan_source(carry, source_index):
        current_states, live_types, halted = carry

        def skip_halted(_):
            return (current_states, live_types, halted), _diagnostics(
                error=jnp.where(
                    invalid_context,
                    MERGER_ERROR_INVALID_CONTEXT,
                    MERGER_ERROR_NONE,
                )
            )

        def inspect_source(_):
            source_type = live_types[source_index]
            is_satellite = (source_type == 1) | (source_type == 2)

            def skip_source(_):
                return (current_states, live_types, halted), _diagnostics()

            def process_source(_):
                source = _state_at(current_states, source_index)
                source_halo = _halo_at(halos, live_types, source_index)
                initial_boundary = (source_halo.dT <= 0.0) & (source_halo.SnapNum < 0)
                invalid_dt = (source_halo.dT <= 0.0) & ~initial_boundary

                def skip_initial(_):
                    return (current_states, live_types, halted), _diagnostics(
                        status=MERGER_STATUS_INITIAL_BOUNDARY
                    )

                def fail_dt(_):
                    return (current_states, live_types, jnp.asarray(True)), _diagnostics(
                        error=MERGER_ERROR_INVALID_DT
                    )

                def process_valid_dt(_):
                    source_dt = source_halo.dT / as_float64(context.num_substeps)

                    def fail_unset(_):
                        return (current_states, live_types, jnp.asarray(True)), _diagnostics(
                            error=MERGER_ERROR_UNSET_CLOCK,
                            source_dt=source_dt,
                        )

                    def process_clock(_):
                        merger_time = as_float32(as_float64(source.MergTime) - source_dt)
                        decremented_source = source._replace(MergTime=merger_time)
                        decremented_states = _set_state(
                            current_states,
                            source_index,
                            decremented_source,
                        )
                        fraction = (as_float64(context.substep_number) + 1.0) / as_float64(
                            context.num_substeps
                        )
                        current_mvir = source_halo.Mvir - source_halo.deltaMvir * (1.0 - fraction)
                        current_mvir = jnp.maximum(current_mvir, 0.0)
                        galaxy_baryons = as_float64(as_float32(source.StellarMass + source.ColdGas))
                        virial_to_baryons = jnp.where(
                            galaxy_baryons > 0.0,
                            current_mvir / galaxy_baryons,
                            -1.0,
                        )
                        eligible = (galaxy_baryons == 0.0) | (
                            (galaxy_baryons > 0.0)
                            & (virial_to_baryons <= parameters.ThresholdSatDisruption)
                        )

                        def skip_not_eligible(_):
                            return (
                                decremented_states,
                                live_types,
                                halted,
                            ), _diagnostics(
                                status=MERGER_STATUS_NOT_ELIGIBLE,
                                eligible=False,
                                current_mvir=current_mvir,
                                virial_to_baryons=virial_to_baryons,
                                source_dt=source_dt,
                            )

                        def inspect_eligible(_):
                            def skip_nonfinite(_):
                                return (
                                    decremented_states,
                                    live_types,
                                    halted,
                                ), _diagnostics(
                                    status=MERGER_STATUS_NONFINITE_CLOCK,
                                    eligible=True,
                                    current_mvir=current_mvir,
                                    virial_to_baryons=virial_to_baryons,
                                    source_dt=source_dt,
                                )

                            def resolve_event(_):
                                target_index, valid_target = _resolve_target(
                                    halos,
                                    live_types,
                                    source_index,
                                    fof_central_index,
                                )

                                def fail_target(_):
                                    return (
                                        decremented_states,
                                        live_types,
                                        jnp.asarray(True),
                                    ), _diagnostics(
                                        error=MERGER_ERROR_INVALID_TARGET,
                                        eligible=True,
                                        current_mvir=current_mvir,
                                        virial_to_baryons=virial_to_baryons,
                                        source_dt=source_dt,
                                    )

                                def apply_event(_):
                                    source_live = _state_at(decremented_states, source_index)
                                    target = _state_at(decremented_states, target_index)

                                    def disrupt(_):
                                        updated_target, ownership = _apply_disruption_ownership(
                                            source_live,
                                            target,
                                        )
                                        updated_states = _set_state(
                                            decremented_states,
                                            target_index,
                                            updated_target,
                                        )
                                        updated_types = live_types.at[source_index].set(3)
                                        return (
                                            updated_states,
                                            updated_types,
                                            halted,
                                        ), _diagnostics(
                                            action=MERGER_ACTION_DISRUPTION,
                                            target_index=target_index,
                                            eligible=True,
                                            current_mvir=current_mvir,
                                            virial_to_baryons=virial_to_baryons,
                                            source_dt=source_dt,
                                            ownership=ownership,
                                        )

                                    def merge(_):
                                        mass_ratio = _merger_mass_ratio(source_live, target)
                                        updated_target, ownership = _apply_merger_ownership(
                                            source_live,
                                            target,
                                        )
                                        updated_states = _set_state(
                                            decremented_states,
                                            target_index,
                                            updated_target,
                                        )
                                        (
                                            updated_states,
                                            merger_quasar,
                                            merger_starburst,
                                            post_instability,
                                            post_quasar,
                                            post_starburst,
                                        ) = _apply_merger_consumers(
                                            updated_states,
                                            halos,
                                            live_types,
                                            target_index,
                                            fof_central_index,
                                            mass_ratio,
                                            parameters,
                                            units,
                                            perturbations,
                                        )
                                        source_time = (context.time + source_halo.dT) - (
                                            as_float64(context.substep_number) + 0.5
                                        ) * source_dt
                                        final_target = _state_at(updated_states, target_index)
                                        final_target = final_target._replace(
                                            TimeOfLastMinorMerger=jnp.where(
                                                mass_ratio > 0.1,
                                                as_float32(source_time),
                                                final_target.TimeOfLastMinorMerger,
                                            ),
                                            BulgeMass=jnp.where(
                                                mass_ratio > parameters.ThresholdMajorMerger,
                                                final_target.StellarMass,
                                                final_target.BulgeMass,
                                            ),
                                            MetalsBulgeMass=jnp.where(
                                                mass_ratio > parameters.ThresholdMajorMerger,
                                                final_target.MetalsStellarMass,
                                                final_target.MetalsBulgeMass,
                                            ),
                                            TimeOfLastMajorMerger=jnp.where(
                                                mass_ratio > parameters.ThresholdMajorMerger,
                                                as_float32(source_time),
                                                final_target.TimeOfLastMajorMerger,
                                            ),
                                        )
                                        updated_states = _set_state(
                                            updated_states,
                                            target_index,
                                            final_target,
                                        )
                                        updated_types = live_types.at[source_index].set(3)
                                        return (
                                            updated_states,
                                            updated_types,
                                            halted,
                                        ), _diagnostics(
                                            action=MERGER_ACTION_MERGER,
                                            target_index=target_index,
                                            eligible=True,
                                            current_mvir=current_mvir,
                                            virial_to_baryons=virial_to_baryons,
                                            mass_ratio=mass_ratio,
                                            source_dt=source_dt,
                                            source_time=source_time,
                                            ownership=ownership,
                                            merger_quasar=merger_quasar,
                                            merger_starburst=merger_starburst,
                                            post_instability=post_instability,
                                            post_quasar=post_quasar,
                                            post_starburst=post_starburst,
                                        )

                                    return jax.lax.cond(
                                        merger_time > 0.0,
                                        disrupt,
                                        merge,
                                        operand=None,
                                    )

                                return jax.lax.cond(
                                    valid_target,
                                    apply_event,
                                    fail_target,
                                    operand=None,
                                )

                            return jax.lax.cond(
                                jnp.isfinite(merger_time),
                                resolve_event,
                                skip_nonfinite,
                                operand=None,
                            )

                        return jax.lax.cond(
                            eligible,
                            inspect_eligible,
                            skip_not_eligible,
                            operand=None,
                        )

                    return jax.lax.cond(
                        source.MergTime >= _MERGTIME_UNSET_THRESHOLD,
                        fail_unset,
                        process_clock,
                        operand=None,
                    )

                return jax.lax.cond(
                    initial_boundary,
                    skip_initial,
                    lambda operand: jax.lax.cond(
                        invalid_dt,
                        fail_dt,
                        process_valid_dt,
                        operand,
                    ),
                    operand=None,
                )

            return jax.lax.cond(is_satellite, process_source, skip_source, operand=None)

        enabled = has_central & ~invalid_context
        return jax.lax.cond(enabled & ~halted, inspect_source, skip_halted, operand=None)

    initial_halted = invalid_context
    (final_states, final_types, halted), diagnostics = jax.lax.scan(
        scan_source,
        (states, halos.Type, initial_halted),
        jnp.arange(count, dtype=jnp.int32),
    )
    result_halos = halos._replace(Type=final_types)
    success = ~halted
    return MergerResolutionResult(final_states, result_halos, diagnostics, success)
