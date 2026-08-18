"""Snapshot infall budgets and their finite SAGE16 substep application."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.transfers import (
    InfallBudgetResult,
    InfallBudgetTransfer,
    InfallResult,
    InfallTransfer,
)
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, StepContext


def _validated_mass_metals(mass, metals):
    mass = as_float32(mass)
    metals = as_float32(metals)
    valid_mass = mass >= 0.0
    mass = jnp.where(valid_mass, mass, as_float32(0.0))
    metals = jnp.where(valid_mass, metals, as_float32(0.0))
    metals = jnp.maximum(metals, as_float32(0.0))
    metals = jnp.minimum(metals, mass)
    return mass, metals


def prepare_infall_budget(
    states: GalaxyState,
    halos: HaloForcing,
    central_index: int,
    parameters: Sage16Parameters,
) -> InfallBudgetResult:
    """Consolidate satellite ejecta/ICS and calculate one FoF infall budget.

    Every state and halo leaf has a leading group-member axis. Type-3 members
    are excluded, matching upstream. Tree orchestration supplies the central
    index explicitly so this numerical kernel remains JIT-compatible.
    """

    require_x64()
    member_count = states.HotGas.shape[0]
    if isinstance(central_index, int):
        if not 0 <= central_index < member_count:
            raise ValueError("central_index must select a member of the supplied FoF group")
    elif getattr(central_index, "ndim", None) != 0:
        raise ValueError("central_index must be a scalar")
    active = halos.Type != 3
    member_indices = jnp.arange(member_count, dtype=jnp.int32)
    surrendered = active & (member_indices != central_index)

    # Upstream accumulates these eight double totals together in live member
    # order.  An explicit loop preserves that floating-point order and makes
    # trailing inactive padding an exact no-op; a parallel ``jnp.sum`` may use
    # a shape-dependent reduction tree.
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    initial_totals = (zero, zero, zero, zero, zero, zero, zero, zero)

    def accumulate(index, totals):
        values = (
            states.StellarMass[index],
            states.BlackHoleMass[index],
            states.ColdGas[index],
            states.HotGas[index],
            states.ICS[index],
            states.EjectedGas[index],
            states.MetalsICS[index],
            states.MetalsEjectedGas[index],
        )
        return tuple(
            jnp.where(active[index], total + as_float64(value), total)
            for total, value in zip(totals, values)
        )

    (
        total_stars,
        total_black_holes,
        total_cold,
        total_hot,
        total_ics,
        total_ejected,
        total_ics_metals,
        total_ejected_metals,
    ) = jax.lax.fori_loop(0, member_count, accumulate, initial_totals)

    central_ejected, central_ejected_metals = _validated_mass_metals(
        total_ejected, total_ejected_metals
    )
    central_ics, central_ics_metals = _validated_mass_metals(total_ics, total_ics_metals)
    updated_ejected = jnp.where(surrendered, as_float32(0.0), states.EjectedGas)
    updated_ejected_metals = jnp.where(surrendered, as_float32(0.0), states.MetalsEjectedGas)
    updated_ics = jnp.where(surrendered, as_float32(0.0), states.ICS)
    updated_ics_metals = jnp.where(surrendered, as_float32(0.0), states.MetalsICS)
    updated_ejected = updated_ejected.at[central_index].set(central_ejected)
    updated_ejected_metals = updated_ejected_metals.at[central_index].set(central_ejected_metals)
    updated_ics = updated_ics.at[central_index].set(central_ics)
    updated_ics_metals = updated_ics_metals.at[central_index].set(central_ics_metals)

    central_fraction = states.HaloBaryonFraction[central_index]
    central_fraction = jnp.where(
        central_fraction == -1.0,
        parameters.GlobalBaryonFraction,
        central_fraction,
    )
    group_baryons = (
        total_stars + total_cold + total_hot + total_ejected + total_black_holes + total_ics
    )
    target_baryons = central_fraction * halos.Mvir[central_index]
    infalling_gas = target_baryons - group_baryons
    updated_infalling = states.InfallingGas.at[central_index].set(infalling_gas)
    updated_fraction = states.HaloBaryonFraction.at[central_index].set(central_fraction)
    updated = states._replace(
        HaloBaryonFraction=updated_fraction,
        InfallingGas=updated_infalling,
        EjectedGas=updated_ejected,
        MetalsEjectedGas=updated_ejected_metals,
        ICS=updated_ics,
        MetalsICS=updated_ics_metals,
    )
    transfer = InfallBudgetTransfer(
        satellite_ejected_to_central=total_ejected - as_float64(states.EjectedGas[central_index]),
        satellite_ejected_metals_to_central=total_ejected_metals
        - as_float64(states.MetalsEjectedGas[central_index]),
        satellite_ics_to_central=total_ics - as_float64(states.ICS[central_index]),
        satellite_ics_metals_to_central=total_ics_metals
        - as_float64(states.MetalsICS[central_index]),
        target_baryons=target_baryons,
        group_baryons=group_baryons,
        infalling_gas=infalling_gas,
    )
    return InfallBudgetResult(updated, transfer)


def apply_infall(
    state: GalaxyState,
    context: StepContext,
    log_fractional_perturbation=0.0,
) -> InfallResult:
    """Apply one central's fixed snapshot infall budget for one substep."""

    require_x64()
    requested = log_fractionally_perturb(
        state.InfallingGas,
        log_fractional_perturbation,
    ) / as_float64(context.num_substeps)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def apply_nonnegative(_):
        updated = state._replace(
            HotGas=as_float32(as_float64(state.HotGas) + requested),
        )
        transfer = InfallTransfer(requested, requested, zero, zero, zero, zero, zero)
        return InfallResult(updated, transfer)

    def apply_negative(_):
        ejected_metallicity = as_float64(
            as_float32(metallicity(state.EjectedGas, state.MetalsEjectedGas))
        )
        has_ejected = state.EjectedGas > 0.0
        tentative_ejected = as_float32(as_float64(state.EjectedGas) + requested)
        tentative_ejected_metals = as_float32(
            as_float64(state.MetalsEjectedGas) + requested * ejected_metallicity
        )
        tentative_ejected_metals = jnp.maximum(tentative_ejected_metals, as_float32(0.0))
        depleted_ejected = has_ejected & (tentative_ejected < 0.0)
        remaining = jnp.where(
            has_ejected,
            jnp.where(depleted_ejected, as_float64(tentative_ejected), 0.0),
            requested,
        )
        new_ejected = jnp.where(
            has_ejected,
            jnp.where(depleted_ejected, as_float32(0.0), tentative_ejected),
            state.EjectedGas,
        )
        new_ejected_metals = jnp.where(
            has_ejected,
            jnp.where(depleted_ejected, as_float32(0.0), tentative_ejected_metals),
            state.MetalsEjectedGas,
        )

        hot_metallicity = as_float64(as_float32(metallicity(state.HotGas, state.MetalsHotGas)))
        change_hot_metals = (remaining < 0.0) & (state.MetalsHotGas > 0.0)
        new_hot_metals = jnp.where(
            change_hot_metals,
            as_float32(as_float64(state.MetalsHotGas) + remaining * hot_metallicity),
            state.MetalsHotGas,
        )
        new_hot_metals = jnp.maximum(new_hot_metals, as_float32(0.0))
        tentative_hot = as_float32(as_float64(state.HotGas) + remaining)
        depleted_hot = tentative_hot < 0.0
        new_hot = jnp.where(depleted_hot, as_float32(0.0), tentative_hot)
        new_hot_metals = jnp.where(depleted_hot, as_float32(0.0), new_hot_metals)
        updated = state._replace(
            EjectedGas=new_ejected,
            MetalsEjectedGas=new_ejected_metals,
            HotGas=new_hot,
            MetalsHotGas=new_hot_metals,
        )
        ejected_removed = as_float64(state.EjectedGas) - as_float64(new_ejected)
        hot_removed = as_float64(state.HotGas) - as_float64(new_hot)
        ejected_metals_removed = as_float64(state.MetalsEjectedGas) - as_float64(new_ejected_metals)
        hot_metals_removed = as_float64(state.MetalsHotGas) - as_float64(new_hot_metals)
        unfulfilled = jnp.maximum(-requested - ejected_removed - hot_removed, 0.0)
        transfer = InfallTransfer(
            requested,
            zero,
            ejected_removed,
            hot_removed,
            ejected_metals_removed,
            hot_metals_removed,
            unfulfilled,
        )
        return InfallResult(updated, transfer)

    return jax.lax.cond(
        requested >= 0.0,
        apply_nonnegative,
        apply_negative,
        operand=None,
    )
