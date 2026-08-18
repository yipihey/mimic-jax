"""SAGE16 ejected-gas reincorporation as an explicit finite transfer."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity, object_substep_dt
from mimic_jax.sage16.transfers import ReincorporationResult, ReincorporationTransfer
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, StepContext


def apply_reincorporation(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    log_fractional_perturbation=0.0,
) -> ReincorporationResult:
    """Return ejected gas to the Type-0 hot halo using the fiducial SAGE16 prescription."""

    require_x64()
    vcrit = 445.48 * parameters.ReIncorporationFactor
    active = (halo.Type == 0) & (state.EjectedGas > 0.0) & (halo.Vvir > vcrit)

    def apply_active(_):
        dt = object_substep_dt(halo, context)
        amount = (
            (halo.Vvir / vcrit - 1.0) * as_float64(state.EjectedGas) / (halo.Rvir / halo.Vvir) * dt
        )
        amount = log_fractionally_perturb(amount, log_fractional_perturbation)
        amount = jnp.minimum(amount, as_float64(state.EjectedGas))
        metals = metallicity(state.EjectedGas, state.MetalsEjectedGas) * amount
        updated = state._replace(
            EjectedGas=as_float32(as_float64(state.EjectedGas) - amount),
            MetalsEjectedGas=as_float32(as_float64(state.MetalsEjectedGas) - metals),
            HotGas=as_float32(as_float64(state.HotGas) + amount),
            MetalsHotGas=as_float32(as_float64(state.MetalsHotGas) + metals),
        )
        return ReincorporationResult(updated, ReincorporationTransfer(amount, metals))

    def apply_inactive(_):
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return ReincorporationResult(state, ReincorporationTransfer(zero, zero))

    return jax.lax.cond(active, apply_active, apply_inactive, operand=None)
