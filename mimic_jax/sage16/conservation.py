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


def active_group_baryonic_mass(states: GalaxyState, halo_types):
    """Total baryons owned by live (non-Type-3) records in a FoF workspace."""

    active = jnp.asarray(halo_types) != 3
    total = as_float64(states.ColdGas) + as_float64(states.HotGas)
    total = total + as_float64(states.EjectedGas)
    total = total + as_float64(states.StellarMass)
    total = total + as_float64(states.ICS)
    total = total + as_float64(states.BlackHoleMass)
    return jnp.sum(jnp.where(active, total, 0.0))


def active_group_metal_mass(states: GalaxyState, halo_types):
    """Tracked metals owned by live (non-Type-3) FoF workspace records."""

    active = jnp.asarray(halo_types) != 3
    total = as_float64(states.MetalsColdGas) + as_float64(states.MetalsHotGas)
    total = total + as_float64(states.MetalsEjectedGas)
    total = total + as_float64(states.MetalsStellarMass)
    total = total + as_float64(states.MetalsICS)
    return jnp.sum(jnp.where(active, total, 0.0))
