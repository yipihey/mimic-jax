"""State, forcing, parameter, and unit PyTree contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimic_jax.sage16 import (
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    metal_mass,
    sage16_units,
    step_context,
)


def test_complete_state_matches_metadata_precision_contract():
    state = initial_galaxy_state()

    assert len(state._fields) == 32
    assert state.ColdGas.dtype == jnp.float32
    assert state.MergTime.dtype == jnp.float32
    assert state.NewStellarMass.dtype == jnp.float64
    assert state.CoolingGas.dtype == jnp.float64
    assert float(state.HaloBaryonFraction) == -1.0
    assert np.isclose(float(state.MergTime), np.float32(999.9), rtol=0.0, atol=0.0)
    assert len(jax.tree_util.tree_leaves(state)) == 32


def test_initial_state_rejects_unknown_upstream_names():
    with pytest.raises(TypeError, match="Unknown SAGE16 galaxy fields"):
        initial_galaxy_state(NotAReservoir=1.0)


def test_forcing_parameters_context_and_units_are_pytrees():
    structures = [
        initial_halo_forcing(),
        fiducial_parameters(),
        sage16_units(),
        step_context(),
    ]

    for structure in structures:
        leaves = jax.tree_util.tree_leaves(structure)
        assert leaves
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    assert np.isclose(float(structures[2].G), 43.00707785642063, rtol=1.0e-14)


def test_conservation_ledgers_do_not_double_count_components_or_scratch():
    state = initial_galaxy_state(
        ColdGas=1.0,
        HotGas=2.0,
        EjectedGas=3.0,
        StellarMass=4.0,
        BulgeMass=1.5,
        ICS=5.0,
        BlackHoleMass=0.5,
        NewStellarMass=99.0,
        MetalsColdGas=0.01,
        MetalsHotGas=0.02,
        MetalsEjectedGas=0.03,
        MetalsStellarMass=0.04,
        MetalsBulgeMass=0.015,
        MetalsICS=0.05,
    )

    assert float(baryonic_mass(state)) == 15.5
    assert np.isclose(float(metal_mass(state)), 0.15, rtol=0.0, atol=5.0e-9)
    assert float(jax.jit(baryonic_mass)(state)) == 15.5
