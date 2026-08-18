"""Fiducial SAGE16 quasar-mode black-hole growth and threshold wind."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.transfers import QuasarModeResult, QuasarModeTransfer
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, Sage16Units

C_CGS = 2.9979e10


def _zero_transfer(trigger):
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return QuasarModeTransfer(trigger, zero, zero, zero, zero, zero, zero, zero, zero)


def apply_quasar_mode(
    state: GalaxyState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    efficiency_factor=None,
    log_fractional_perturbation=0.0,
) -> QuasarModeResult:
    """Apply one disk-instability or merger-triggered quasar event.

    ``efficiency_factor=None`` consumes the live disk-instability trigger.
    Supplying a factor exposes the same kernel for the later merger event map.
    """

    require_x64()
    trigger = (
        as_float64(state.UnstableDiskGasFraction)
        if efficiency_factor is None
        else as_float64(efficiency_factor)
    )
    active = (
        (trigger > 0.0)
        & (parameters.BlackHoleGrowthRate > 0.0)
        & (state.ColdGas > 0.0)
        & (halo.Vvir > 0.0)
    )

    def apply_active(_):
        requested = (
            parameters.BlackHoleGrowthRate
            * trigger
            / (1.0 + (280.0 / halo.Vvir) * (280.0 / halo.Vvir))
            * as_float64(state.ColdGas)
        )
        requested = log_fractionally_perturb(
            requested,
            log_fractional_perturbation,
        )
        accreted = jnp.minimum(requested, as_float64(state.ColdGas))
        cold_metals_accreted = metallicity(state.ColdGas, state.MetalsColdGas) * accreted
        grown = state._replace(
            BlackHoleMass=as_float32(as_float64(state.BlackHoleMass) + accreted),
            ColdGas=as_float32(as_float64(state.ColdGas) - accreted),
            MetalsColdGas=as_float32(as_float64(state.MetalsColdGas) - cold_metals_accreted),
            QuasarModeBHaccretionMass=as_float32(
                as_float64(state.QuasarModeBHaccretionMass) + accreted
            ),
        )
        quasar_energy = (
            parameters.QuasarModeEfficiency
            * 0.1
            * accreted
            * (C_CGS / units.UnitVelocity_in_cm_per_s) ** 2
        )
        cold_energy = 0.5 * as_float64(grown.ColdGas) * halo.Vvir**2
        hot_energy = 0.5 * as_float64(grown.HotGas) * halo.Vvir**2
        eject_cold = (parameters.QuasarModeEfficiency > 0.0) & (quasar_energy > cold_energy)
        cold_to_ejected = jnp.where(eject_cold, as_float64(grown.ColdGas), 0.0)
        cold_metals_to_ejected = jnp.where(
            eject_cold,
            as_float64(grown.MetalsColdGas),
            0.0,
        )
        after_cold = grown._replace(
            EjectedGas=jnp.where(
                eject_cold,
                as_float32(grown.EjectedGas + grown.ColdGas),
                grown.EjectedGas,
            ),
            MetalsEjectedGas=jnp.where(
                eject_cold,
                as_float32(grown.MetalsEjectedGas + grown.MetalsColdGas),
                grown.MetalsEjectedGas,
            ),
            ColdGas=jnp.where(eject_cold, as_float32(0.0), grown.ColdGas),
            MetalsColdGas=jnp.where(
                eject_cold,
                as_float32(0.0),
                grown.MetalsColdGas,
            ),
        )
        eject_hot = (parameters.QuasarModeEfficiency > 0.0) & (
            quasar_energy > cold_energy + hot_energy
        )
        hot_to_ejected = jnp.where(eject_hot, as_float64(after_cold.HotGas), 0.0)
        hot_metals_to_ejected = jnp.where(
            eject_hot,
            as_float64(after_cold.MetalsHotGas),
            0.0,
        )
        after_hot = after_cold._replace(
            EjectedGas=jnp.where(
                eject_hot,
                as_float32(after_cold.EjectedGas + after_cold.HotGas),
                after_cold.EjectedGas,
            ),
            MetalsEjectedGas=jnp.where(
                eject_hot,
                as_float32(after_cold.MetalsEjectedGas + after_cold.MetalsHotGas),
                after_cold.MetalsEjectedGas,
            ),
            HotGas=jnp.where(eject_hot, as_float32(0.0), after_cold.HotGas),
            MetalsHotGas=jnp.where(
                eject_hot,
                as_float32(0.0),
                after_cold.MetalsHotGas,
            ),
        )
        transfer = QuasarModeTransfer(
            trigger,
            requested,
            accreted,
            cold_metals_accreted,
            quasar_energy,
            cold_to_ejected,
            cold_metals_to_ejected,
            hot_to_ejected,
            hot_metals_to_ejected,
        )
        return QuasarModeResult(after_hot, transfer)

    def apply_zero(_):
        return QuasarModeResult(state, _zero_transfer(trigger))

    return jax.lax.cond(active, apply_active, apply_zero, operand=None)
