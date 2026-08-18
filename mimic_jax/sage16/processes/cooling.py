"""Explicit application of the SAGE16 cooling budget to gas reservoirs."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.transfers import CoolingResult, CoolingTransfer
from mimic_jax.sage16.types import GalaxyState, HaloForcing


def apply_cooling(
    state: GalaxyState,
    halo: HaloForcing,
    cooling_gas=None,
    log_fractional_perturbation=0.0,
) -> CoolingResult:
    """Commit ``CoolingGas`` as a hot-to-cold finite transfer, matching upstream C writes."""

    require_x64()
    requested = log_fractionally_perturb(
        as_float64(state.CoolingGas if cooling_gas is None else cooling_gas),
        log_fractional_perturbation,
    )

    def apply_positive(_):
        hot = as_float64(state.HotGas)
        is_partial = requested < hot
        transferred = jnp.where(is_partial, requested, hot)
        transferred_metals = jnp.where(
            is_partial,
            metallicity(state.HotGas, state.MetalsHotGas) * requested,
            as_float64(state.MetalsHotGas),
        )
        updated = state._replace(
            ColdGas=as_float32(as_float64(state.ColdGas) + transferred),
            MetalsColdGas=as_float32(as_float64(state.MetalsColdGas) + transferred_metals),
            HotGas=as_float32(hot - transferred),
            MetalsHotGas=as_float32(as_float64(state.MetalsHotGas) - transferred_metals),
            # SAGE passes Vvir through a float helper argument before accumulating this diagnostic.
            Cooling=as_float64(state.Cooling)
            + jnp.where(
                halo.dT > 0.0,
                0.5 * requested * as_float64(as_float32(halo.Vvir)) ** 2 / halo.dT,
                0.0,
            ),
        )
        return CoolingResult(updated, CoolingTransfer(transferred, transferred_metals))

    def apply_zero(_):
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return CoolingResult(state, CoolingTransfer(zero, zero))

    return jax.lax.cond(requested > 0.0, apply_positive, apply_zero, operand=None)
