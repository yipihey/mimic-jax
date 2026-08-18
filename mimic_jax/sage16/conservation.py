"""Executable baryon and metal ledgers for the SAGE16 reservoirs."""

import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float64
from mimic_jax.sage16.types import GalaxyState


def baryonic_mass(state: GalaxyState):
    """Total modeled baryons without double-counting bulge stars as a separate reservoir."""

    return jnp.sum(
        jnp.stack(
            [
                as_float64(state.ColdGas),
                as_float64(state.HotGas),
                as_float64(state.EjectedGas),
                as_float64(state.StellarMass),
                as_float64(state.ICS),
                as_float64(state.BlackHoleMass),
            ]
        )
    )


def metal_mass(state: GalaxyState):
    """Total tracked metals without double-counting ``MetalsBulgeMass``."""

    return jnp.sum(
        jnp.stack(
            [
                as_float64(state.MetalsColdGas),
                as_float64(state.MetalsHotGas),
                as_float64(state.MetalsEjectedGas),
                as_float64(state.MetalsStellarMass),
                as_float64(state.MetalsICS),
            ]
        )
    )
