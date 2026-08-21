"""Common configured-SAM protocol tests across SAGE16 and SHARK."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimic_jax import (
    ResponseCoordinate,
    Sage16ContinuousForcing,
    SharkContinuousForcing,
    available_models,
    characteristic_modes,
    load_model,
    replace_parameter_path,
    validate_parameter_response,
)
from mimic_jax.sage16 import initial_galaxy_state, initial_halo_forcing, ode_state_from_galaxy
from mimic_jax.shark import initial_shark_state, lagos23_disk_forcing

jax.config.update("jax_enable_x64", True)


def _sage_case():
    model = load_model("sage16")
    galaxy = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        DiskScaleRadius=0.01,
    )
    state = ode_state_from_galaxy(galaxy)
    forcing = Sage16ContinuousForcing(
        initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=150.0, dT=5.0e-4),
        galaxy.DiskScaleRadius,
    )
    return model, state, forcing


def _shark_case():
    model = load_model("shark")
    state = initial_shark_state(
        stellar_mass=5.0e9,
        cold_gas=3.0e9,
        cold_halo_gas=0.0,
        hot_halo_gas=2.0e10,
        ejected_gas=2.0e9,
        stellar_metals=8.0e7,
        cold_gas_metals=5.0e7,
        hot_halo_gas_metals=2.0e8,
        ejected_gas_metals=2.0e7,
        stellar_angular_momentum=4.0e11,
        cold_gas_angular_momentum=2.5e11,
        hot_halo_angular_momentum=1.0e12,
        ejected_angular_momentum=1.0e11,
    )
    forcing = SharkContinuousForcing(
        lagos23_disk_forcing(
            gas_half_mass_radius=0.008,
            stellar_half_mass_radius=0.006,
            redshift=0.5,
            galaxy_velocity=180.0,
            subhalo_velocity=220.0,
            cooling_rate=1.5e9,
            cooling_metallicity=0.01,
            cooling_specific_angular_momentum=80.0,
            qso_reheating_loading=0.05,
            qso_ejection_loading=0.02,
        ),
        jnp.asarray(2.0e8),
    )
    return model, state, forcing


def test_registry_loads_all_three_models_and_declares_limits():
    assert available_models() == ("sage16", "shark", "sapphire")
    sage = load_model("sage")
    shark = load_model("shark_lagos23")
    sapphire = load_model("saphire")

    assert sage.metadata.name == "sage16"
    assert shark.metadata.name == "shark"
    assert sage.metadata.capability("continuous_rhs").status == "supported"
    assert shark.metadata.capability("continuous_rhs").status == "partial"
    assert sage.metadata.capability("events").status == "model_specific"
    assert shark.metadata.capability("events").status == "model_specific"
    assert sage.metadata.capability("full_tree_physics_parity").status == "supported"
    assert shark.metadata.capability("full_tree_physics_parity").status == "supported"
    assert sage.metadata.capability("independent_topology_driver").status == "supported"
    assert shark.metadata.capability("independent_topology_driver").status == "unavailable"
    assert sapphire.metadata.name == "sapphire"
    assert sapphire.metadata.capability("continuous_rhs").status == "supported"
    assert sapphire.metadata.capability("events").status == "unavailable"
    assert sapphire.metadata.capability("full_tree_physics_parity").status == "not_applicable"
    assert sapphire.metadata.capability("population_observables").status == "partial"
    assert len(sapphire.metadata.state_variables) == 7
    assert "Eth_cgm" in {variable.name for variable in sapphire.metadata.state_variables}
    assert "M_cgm" in {variable.name for variable in sapphire.metadata.state_variables}
    assert "cgm_mass" in {variable.name for variable in sapphire.metadata.observable_variables}
    assert "merger" in {process.name for process in sage.metadata.processes}
    assert "merger" in {process.name for process in shark.metadata.processes}
    assert json.loads(json.dumps(shark.metadata.to_dict()))["name"] == "shark"


@pytest.mark.parametrize("case", (_sage_case, _shark_case))
def test_common_rhs_rate_and_conservation_interfaces(case):
    model, state, forcing = case()
    result = model.rhs_and_rates(0.0, state, forcing)
    derivative = model.rhs(0.0, state, forcing)
    np.testing.assert_allclose(
        jax.flatten_util.ravel_pytree(result.derivative)[0],
        jax.flatten_util.ravel_pytree(derivative)[0],
    )
    assert np.isfinite(np.asarray(model.rate_value(result, "cooling")))
    assert float(model.observable_value(state, result, "stellar_mass")) > 0.0
    assert float(model.observable_value(state, result, "star_formation_rate")) >= 0.0
    quantities = model.conserved_quantities(state)
    assert quantities[0].name == "baryons"
    assert float(quantities[0].value) > 0.0
    balances = model.conservation_balances(result)
    balance_names = {balance.name for balance in balances}
    assert {"baryons", "metals"}.issubset(balance_names)
    for balance in balances:
        np.testing.assert_allclose(balance.residual, 0.0, atol=2.0e-6, rtol=2.0e-13)


@pytest.mark.parametrize("case", (_sage_case, _shark_case))
def test_process_control_derivatives_preserve_baryon_conservation(case):
    model, state, forcing = case()
    controls = jnp.zeros(len(model.metadata.process_control_names), dtype=jnp.float64)

    def baryon_rate(current_controls):
        derivative = model.rhs(0.0, state, forcing, None, current_controls)
        flat = jax.flatten_util.ravel_pytree(derivative)[0]
        if model.metadata.name == "sage16":
            return jnp.sum(flat[:4])
        return jnp.sum(flat[:6])

    tangent = jax.jacfwd(baryon_rate)(controls)
    np.testing.assert_allclose(tangent, 0.0, atol=2.0e-6, rtol=2.0e-13)


def test_nested_shark_and_flat_sage_parameters_use_one_elasticity_api(tmp_path):
    sage, _, _ = _sage_case()
    sage_response = sage.parameter_response(
        lambda parameters: jnp.asarray([parameters.SfrEfficiency**2]),
        parameter_names=("SfrEfficiency",),
        observable_names=("squared_sfr_efficiency",),
    )
    np.testing.assert_allclose(sage_response.values, [[2.0]])
    assert sage_response.model == "MIMIC/SAGE16"

    shark, _, _ = _shark_case()
    parameter_name = "star_formation.efficiency_per_gyr"
    shark_response = shark.parameter_response(
        lambda parameters: jnp.asarray([parameters.star_formation.efficiency_per_gyr**2]),
        parameter_names=(parameter_name,),
        observable_names=("squared_sf_efficiency",),
    )
    np.testing.assert_allclose(shark_response.values, [[2.0]])
    assert shark_response.model == "SHARK Lagos23"
    archive = tmp_path / "shark-response.npz"
    shark_response.save(archive)
    with np.load(archive) as saved:
        assert saved["model"] == "SHARK Lagos23"
        assert saved["formulation"] == "controlled continuous Lagos23 disk subset"
    validation = validate_parameter_response(
        shark_response,
        lambda parameters: jnp.asarray([parameters.star_formation.efficiency_per_gyr**2]),
        shark.default_parameters,
        relative_steps=(1.0e-3,),
        parameter_replacer=replace_parameter_path,
    )
    np.testing.assert_allclose(validation.finite_difference[0], [[2.0]], rtol=1.0e-9)


@pytest.mark.parametrize(
    ("case", "parameter_name"),
    (
        (_sage_case, "SfrEfficiency"),
        (_shark_case, "star_formation.efficiency_per_gyr"),
    ),
)
def test_common_state_and_parameter_jacobians_are_named(case, parameter_name):
    model, state, forcing = case()
    jacobians = model.jacobians(
        time=0.0,
        state=state,
        forcing=forcing,
        parameter_names=(parameter_name,),
        redshift=0.5,
    )
    state_size = len(model.metadata.state_variables)
    assert jacobians.state_jacobian.shape == (state_size, state_size)
    assert jacobians.parameter_jacobian.shape == (state_size, 1)
    assert jacobians.parameter_coordinates[0].name == parameter_name
    assert jacobians.point.model == model.metadata.label
    assert np.all(np.isfinite(jacobians.parameter_jacobian))


@pytest.mark.parametrize("case", (_sage_case, _shark_case))
def test_local_response_is_physically_annotated_for_both_models(case):
    model, state, forcing = case()

    def output(current, current_forcing, parameters, controls):
        result = model.rhs_and_rates(0.0, current, current_forcing, parameters, controls)
        if model.metadata.name == "sage16":
            return jnp.asarray([current.ColdGas, result.rates.star_formation])
        return jnp.asarray([current.cold_gas, result.rates.star_formation])

    response = model.local_response(
        time=0.0,
        state=state,
        forcing=forcing,
        output=output,
        output_coordinates=(
            ResponseCoordinate("cold_gas", "cold gas", "native mass"),
            ResponseCoordinate("sfr", "star formation rate", "native mass/time"),
        ),
        redshift=0.5,
        halo_mass=1.0e12,
        halo_mass_unit="Msun",
    )
    assert response.point.model == model.metadata.label
    assert response.point.redshift == 0.5
    assert response.input_coordinates[0].unit == "fractional process change"
    assert response.state_jacobian.shape[0] == len(model.metadata.state_variables)
    assert response.input_jacobian.shape[1] == len(model.metadata.process_control_names)
    assert response.output_jacobian.shape[0] == 2
    modes = characteristic_modes(response)
    assert modes.time_unit == model.metadata.time_unit
    assert modes.right_eigenvectors.shape == response.state_jacobian.shape
    assert len(modes.state_coordinates) == len(model.metadata.state_variables)


def test_process_control_shape_is_checked_before_physics_execution():
    model, state, forcing = _sage_case()
    with pytest.raises(ValueError, match="expects 5 process controls"):
        model.rhs(0.0, state, forcing, log_process_perturbations=jnp.zeros(2))
