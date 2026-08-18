"""JIT, VMAP, and automatic-derivative validation for the faithful smooth subset."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    quiescent_disk_step,
    sage16_units,
    step_context,
)


def _smooth_case():
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        StellarMass=2.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsStellarMass=0.04,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    return (
        state,
        halo,
        step_context(time_interval=1.0e-4),
        fiducial_parameters(),
        sage16_units(),
    )


def test_jit_matches_eager_exactly():
    state, halo, context, parameters, units = _smooth_case()
    eager = quiescent_disk_step(state, state, halo, halo, context, parameters, units)
    compiled = jax.jit(quiescent_disk_step)(state, state, halo, halo, context, parameters, units)

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(compiled)
    ):
        np.testing.assert_allclose(eager_leaf, compiled_leaf, rtol=0.0, atol=0.0)


def test_vmap_matches_independent_scalar_galaxies():
    state, halo, context, parameters, units = _smooth_case()
    cold_values = jnp.asarray([8.0, 10.0, 12.0], dtype=jnp.float32)
    states = jax.vmap(lambda cold: state._replace(ColdGas=cold))(cold_values)
    halos = jax.tree_util.tree_map(
        lambda value: jnp.repeat(value[jnp.newaxis, ...], 3, axis=0), halo
    )
    batched = jax.vmap(
        quiescent_disk_step,
        in_axes=(0, 0, 0, 0, None, None, None),
    )(states, states, halos, halos, context, parameters, units)

    for index in range(3):
        scalar = quiescent_disk_step(
            jax.tree_util.tree_map(lambda value: value[index], states),
            jax.tree_util.tree_map(lambda value: value[index], states),
            jax.tree_util.tree_map(lambda value: value[index], halos),
            jax.tree_util.tree_map(lambda value: value[index], halos),
            context,
            parameters,
            units,
        )
        assert float(batched.galaxy.StellarMass[index]) == float(scalar.galaxy.StellarMass)


def test_exact_parameter_derivative_matches_centered_finite_difference():
    state, halo, context, parameters, units = _smooth_case()

    def final_stellar_mass(sfr_efficiency):
        varied = parameters._replace(SfrEfficiency=sfr_efficiency)
        return quiescent_disk_step(
            state, state, halo, halo, context, varied, units
        ).galaxy.StellarMass

    automatic = jax.grad(final_stellar_mass)(parameters.SfrEfficiency)
    delta = jnp.asarray(1.0e-4, dtype=jnp.float64)
    finite_difference = (
        final_stellar_mass(parameters.SfrEfficiency + delta)
        - final_stellar_mass(parameters.SfrEfficiency - delta)
    ) / (2.0 * delta)

    assert float(automatic) != 0.0
    assert np.isclose(float(automatic), float(finite_difference), rtol=2.0e-3, atol=2.0e-4)
