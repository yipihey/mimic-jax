"""Direct oracle checks for the first fiducial Lagos23 prescriptions."""

import json
from importlib import resources
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mimic_jax.shark import (
    SHARK_UPSTREAM_REVISION,
    HeatingRadiusState,
    angular_momentum,
    baryonic_mass,
    cloudy_cie_cooling_table,
    continuous_reincorporation_rate,
    cooling_luminosity_1e40_erg_per_s,
    cooling_rate_after_heating_radius,
    cooling_time_gyr,
    croton06_unheated_cooling,
    croton06_unheated_cooling_from_table,
    eddington_accretion_ratio,
    eddington_luminosity_1e40_erg_per_s,
    initial_shark_state,
    interpolate_log10_cooling_function,
    isothermal_shell_number_density,
    lagos13_feedback_loadings,
    lagos13_feedback_parameters,
    lagos23_agn_parameters,
    lagos23_bolometric_luminosity_1e40_erg_per_s,
    lagos23_br06_star_formation,
    lagos23_croton06_cooling_parameters,
    lagos23_disk_flow_rates,
    lagos23_disk_forcing,
    lagos23_hot_halo_accretion_rate,
    lagos23_mechanical_luminosity_1e40_erg_per_s,
    lagos23_qso_outflow_loadings,
    lagos23_reincorporation_parameters,
    lagos23_star_formation_parameters,
    metal_mass,
    project_lagos23_heating_radius,
    qso_critical_luminosity_1e40_erg_per_s,
    qso_outflow_velocity_km_per_s,
    reference_reincorporated_mass,
    reincorporation_flow_derivative,
    salpeter_timescale_gyr,
    sobacchi13_reionisation_parameters,
    sobacchi13_reionised_halo,
    thin_disk_efficiency_and_isco,
)
from mimic_jax.shark.prescriptions.agn import (
    griffin19_accretion_spin_upstream_rng,
    griffin19_merger_spin_upstream_rng,
    upstream_minstd_uniform_sequence,
)
from mimic_jax.shark.prescriptions.structure import (
    cosmic_age_gyr,
    duffy08_concentration,
    halo_dynamical_time_gyr,
    halo_virial_radius_mpc_over_h,
    halo_virial_velocity_km_per_s,
    hubble_parameter_km_s_mpc,
    lagos23_cosmology,
    nfw_enclosed_mass_fraction,
)

_FIXTURE = Path(__file__).parent / "fixtures/shark/lagos23_rate_oracle.json"


def _oracle():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _star_formation_case(case, parameters):
    return lagos23_br06_star_formation(
        case["cold_gas"],
        case["stars"],
        case["gas_radius"],
        case["stellar_radius"],
        case["gas_metallicity"],
        case["redshift"],
        case["burst"],
        case["galaxy_velocity"],
        case["gas_specific_angular_momentum"],
        parameters,
    )


def test_oracle_fixture_identifies_pinned_upstream_source():
    oracle = _oracle()
    assert oracle["provenance"]["upstream_revision"] == SHARK_UPSTREAM_REVISION
    assert oracle["provenance"]["config_name"] == "sample_lagos23.cfg"
    assert len(oracle["provenance"]["config_sha256"]) == 64


def test_halo_structure_and_cosmic_time_match_upstream_oracle():
    cosmology = lagos23_cosmology()
    for case in _oracle()["structure"]:
        arguments = case["mass"], case["redshift"], cosmology
        np.testing.assert_allclose(
            hubble_parameter_km_s_mpc(case["redshift"], cosmology),
            case["hubble_parameter"],
            rtol=2.0e-15,
        )
        np.testing.assert_allclose(
            cosmic_age_gyr(case["redshift"], cosmology), case["cosmic_age"], rtol=6.0e-8
        )
        np.testing.assert_allclose(
            halo_virial_velocity_km_per_s(*arguments), case["virial_velocity"], rtol=2.0e-15
        )
        np.testing.assert_allclose(
            halo_virial_radius_mpc_over_h(*arguments), case["virial_radius"], rtol=2.0e-15
        )
        np.testing.assert_allclose(
            halo_dynamical_time_gyr(*arguments), case["dynamical_time"], rtol=2.0e-15
        )
        concentration = duffy08_concentration(case["mass"], case["redshift"])
        np.testing.assert_allclose(concentration, case["concentration"], rtol=2.0e-15)
        np.testing.assert_allclose(
            nfw_enclosed_mass_fraction(case["normalized_radius"], concentration),
            case["nfw_enclosed_fraction"],
            rtol=1.0e-13,
        )


def test_pinned_upstream_random_engine_and_griffin19_spin_events_match_oracle():
    oracle = _oracle()
    for case in oracle["rng"]:
        np.testing.assert_allclose(
            upstream_minstd_uniform_sequence(case["seed"], 3),
            case["values"],
            rtol=0.0,
            atol=3.0e-16,
        )

    parameters = lagos23_agn_parameters()
    for case in oracle["spin_accretion"]:
        result = griffin19_accretion_spin_upstream_rng(
            case["black_hole_mass"],
            case["initial_spin"],
            case["accreted_mass"],
            case["accretion_time"],
            case["galaxy_id"],
            123456,
            parameters,
        )
        # The zero-spin branch repeatedly rounds through upstream float
        # storage during ten sub-chunks; all cases agree within 1e-4 relative.
        np.testing.assert_allclose(result, case["final_spin"], rtol=1.0e-4, atol=2.0e-7)

    for case in oracle["spin_merger"]:
        result = griffin19_merger_spin_upstream_rng(
            case["primary_mass"],
            case["secondary_mass"],
            case["primary_spin"],
            case["secondary_spin"],
            case["galaxy_id"],
            123456,
        )
        np.testing.assert_allclose(result, case["final_spin"], rtol=3.0e-6, atol=2.0e-7)

    jitted = jax.jit(griffin19_merger_spin_upstream_rng)
    assert jnp.isfinite(jitted(1.0e8, 2.0e7, 0.6, 0.2, 21, 123456))


def test_br06_radial_star_formation_matches_upstream_oracle():
    oracle = _oracle()
    parameters = lagos23_star_formation_parameters()

    for case in oracle["star_formation"]:
        result = _star_formation_case(case, parameters)
        # Upstream sample_lagos23 asks GSL for only 5% relative radial-quadrature
        # accuracy.  The deterministic 128-node JAX result agrees much more
        # closely; this explicit 5e-6 tolerance covers the observed oracle
        # difference without pretending the two quadrature algorithms are bitwise.
        np.testing.assert_allclose(result.mass, case["rate"], rtol=5.0e-6)
        np.testing.assert_allclose(
            result.angular_momentum,
            case["angular_momentum_rate"],
            rtol=5.0e-8,
            atol=1.0e-12,
        )


def test_lagos13_feedback_matches_upstream_oracle_exactly():
    oracle = _oracle()
    parameters = lagos13_feedback_parameters()

    for case in oracle["stellar_feedback"]:
        result = lagos13_feedback_loadings(
            1.0,
            case["subhalo_velocity"],
            case["galaxy_velocity"],
            case["redshift"],
            parameters,
        )
        expected = np.asarray(
            [
                case["reheating_loading"],
                case["ejection_loading"],
                case["angular_momentum_reheating_loading"],
                case["angular_momentum_ejection_loading"],
            ]
        )
        np.testing.assert_array_equal(np.asarray(result), expected)


def test_prescriptions_support_jit_vmap_and_exact_local_derivatives():
    case = _oracle()["star_formation"][0]
    parameters = lagos23_star_formation_parameters()
    eager = _star_formation_case(case, parameters)
    compiled = jax.jit(_star_formation_case)(case, parameters)
    np.testing.assert_allclose(compiled.mass, eager.mass, rtol=2.0e-15)
    np.testing.assert_allclose(compiled.angular_momentum, eager.angular_momentum, rtol=2.0e-15)

    efficiencies = jnp.asarray([0.8, 1.2, 1.6])

    def rate_for_efficiency(efficiency):
        varied = parameters._replace(efficiency_per_gyr=efficiency)
        return _star_formation_case(case, varied).mass

    rates = jax.vmap(rate_for_efficiency)(efficiencies)
    np.testing.assert_allclose(rates / efficiencies, rates[0] / efficiencies[0])
    derivative = jax.grad(rate_for_efficiency)(parameters.efficiency_per_gyr)
    np.testing.assert_allclose(
        derivative,
        eager.mass / parameters.efficiency_per_gyr,
        rtol=3.0e-14,
    )


def test_lagos23_disk_rate_layer_closes_the_oracled_prescriptions_and_jits():
    oracle = _oracle()
    star_case = oracle["star_formation"][0]
    feedback_case = oracle["stellar_feedback"][0]
    state = initial_shark_state(
        stellar_mass=star_case["stars"],
        cold_gas=star_case["cold_gas"],
        cold_gas_metals=star_case["gas_metallicity"] * star_case["cold_gas"],
    )
    forcing = lagos23_disk_forcing(
        gas_half_mass_radius=star_case["gas_radius"],
        stellar_half_mass_radius=star_case["stellar_radius"],
        redshift=star_case["redshift"],
        burst=star_case["burst"],
        galaxy_velocity=star_case["galaxy_velocity"],
        subhalo_velocity=feedback_case["subhalo_velocity"],
    )
    star_parameters = lagos23_star_formation_parameters()
    feedback_parameters = lagos13_feedback_parameters()

    def rate_layer(one_state):
        return lagos23_disk_flow_rates(
            0.0, one_state, forcing, star_parameters, feedback_parameters
        )

    eager = rate_layer(state)
    compiled = jax.jit(rate_layer)(state)
    np.testing.assert_allclose(eager.star_formation, star_case["rate"], rtol=5.0e-6)
    np.testing.assert_array_equal(
        eager.stellar_reheating_loading, feedback_case["reheating_loading"]
    )
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=2.0e-15)


def test_reference_lagos13_redshift_behavior_is_explicit_and_inactive_is_zero():
    parameters = lagos13_feedback_parameters()
    redshifts = jnp.asarray([0.0, 1.0, 4.0])
    loadings = jax.vmap(
        lambda redshift: lagos13_feedback_loadings(
            1.0, 100.0, 120.0, redshift, parameters
        ).reheating
    )(redshifts)
    np.testing.assert_array_equal(loadings, jnp.repeat(loadings[0], 3))
    assert lagos13_feedback_loadings(0.0, 100.0, 120.0, 0.0, parameters).reheating == 0.0


def test_reincorporation_reference_map_matches_upstream_and_continuous_rate_is_explicit():
    oracle = _oracle()
    parameters = lagos23_reincorporation_parameters()
    for case in oracle["reincorporation"]:
        realized = reference_reincorporated_mass(
            case["ejected_gas"],
            case["halo_mass"],
            case["interval_gyr"],
            case["satellite"],
            parameters,
        )
        np.testing.assert_array_equal(realized, case["realized_mass"])

    regular = oracle["reincorporation"][2]
    rate = continuous_reincorporation_rate(
        regular["ejected_gas"],
        regular["halo_mass"],
        regular["satellite"],
        parameters,
    )
    np.testing.assert_allclose(
        rate * regular["interval_gyr"],
        regular["requested_mass"],
        # The continuous state is float64; the reference map first rounds the
        # upstream halo/ejected masses to their stored C++ float fields.
        rtol=2.0e-8,
    )


def test_continuous_reincorporation_transfer_conserves_mass_metals_am_and_derivatives():
    state = initial_shark_state(
        hot_halo_gas=3.0,
        ejected_gas=8.0,
        hot_halo_gas_metals=0.03,
        ejected_gas_metals=0.16,
        hot_halo_angular_momentum=9.0,
        ejected_angular_momentum=40.0,
    )

    def ledgers(rate):
        derivative = reincorporation_flow_derivative(state, rate)
        return jnp.asarray(
            [
                baryonic_mass(derivative),
                metal_mass(derivative),
                angular_momentum(derivative),
            ]
        )

    np.testing.assert_allclose(ledgers(2.5), np.zeros(3), atol=1.0e-15)
    np.testing.assert_allclose(jax.jacfwd(ledgers)(2.5), np.zeros(3), atol=1.0e-15)


def test_sobacchi13_reionisation_gate_matches_upstream_oracle():
    oracle = _oracle()
    parameters = sobacchi13_reionisation_parameters()
    actual = [
        bool(sobacchi13_reionised_halo(case["virial_velocity"], case["redshift"], parameters))
        for case in oracle["reionisation"]
    ]
    assert actual == [case["reionised"] for case in oracle["reionisation"]]


def _cooling_case(case, parameters):
    return croton06_unheated_cooling(
        case["hot_mass"],
        case["density_mass"],
        case["virial_radius"],
        case["virial_velocity"],
        case["halo_dynamical_time"],
        case["log10_cooling_function_input"],
        parameters,
    )


def test_croton06_cooling_building_blocks_match_upstream_oracle():
    oracle = _oracle()
    parameters = lagos23_croton06_cooling_parameters()

    for case in oracle["cooling"]:
        result = _cooling_case(case, parameters)
        np.testing.assert_allclose(
            np.asarray(result),
            np.asarray(
                [
                    case["virial_temperature"],
                    case["log10_cooling_function"],
                    case["mean_number_density"],
                    case["halo_dynamical_time"],
                    case["halo_dynamical_time"] * 3.15576e16,
                    case["cooling_radius"],
                    case["cooling_rate"],
                ]
            ),
            rtol=3.0e-15,
        )
        shell_radius = min(0.75 * case["virial_radius"], case["cooling_radius"])
        shell_density = isothermal_shell_number_density(
            case["density_mass"], case["virial_radius"], shell_radius
        )
        np.testing.assert_allclose(shell_density, case["shell_number_density"], rtol=3.0e-15)
        shell_time = cooling_time_gyr(
            case["virial_temperature"],
            case["log10_cooling_function"],
            shell_density,
        )
        np.testing.assert_allclose(shell_time, case["shell_cooling_time"], rtol=3.0e-15)
        luminosity = cooling_luminosity_1e40_erg_per_s(
            case["log10_cooling_function"],
            case["cooling_radius"],
            case["virial_radius"],
            case["hot_mass"],
            parameters.core_radius_fraction,
        )
        np.testing.assert_allclose(luminosity, case["cooling_luminosity"], rtol=8.0e-15)


def test_croton06_cooling_is_jittable_vectorizable_and_differentiable():
    case = _oracle()["cooling"][2]
    parameters = lagos23_croton06_cooling_parameters()
    eager = _cooling_case(case, parameters)
    compiled = jax.jit(_cooling_case)(case, parameters)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=2.0e-15)

    masses = jnp.asarray([0.8, 1.0, 1.2]) * case["hot_mass"]

    def rate_for_mass(mass):
        varied = dict(case, hot_mass=mass, density_mass=mass)
        return _cooling_case(varied, parameters).cooling_rate

    rates = jax.vmap(rate_for_mass)(masses)
    assert np.all(np.asarray(rates) > 0.0)
    fiducial_mass = float(case["hot_mass"])
    derivative = jax.grad(rate_for_mass)(fiducial_mass)
    step = fiducial_mass * 1.0e-5
    finite_difference = (
        rate_for_mass(fiducial_mass + step) - rate_for_mass(fiducial_mass - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=3.0e-10)


def test_cloudy_cooling_table_interpolation_matches_upstream_gsl_oracle():
    payload = json.loads(
        resources.files("mimic_jax.shark")
        .joinpath("data/cloudy_cie.json")
        .read_text(encoding="utf-8")
    )
    assert payload["provenance"]["upstream_revision"] == SHARK_UPSTREAM_REVISION
    assert len(payload["provenance"]["source_sha256"]) == 9
    table = cloudy_cie_cooling_table()
    assert table.log10_temperature_k.shape == (227,)
    assert table.metallicity.shape == (8,)
    assert table.log10_cooling_function.shape == (8, 227)
    for case in _oracle()["cooling_function"]:
        actual = interpolate_log10_cooling_function(
            case["log10_temperature"], case["metallicity"], table
        )
        np.testing.assert_allclose(actual, case["log10_cooling_function"], atol=3.0e-15)

    cases = _oracle()["cooling_function"][2:6]
    temperatures = jnp.asarray([case["log10_temperature"] for case in cases])
    metallicities = jnp.asarray([case["metallicity"] for case in cases])
    vmapped = jax.vmap(
        lambda temperature, metallicity: interpolate_log10_cooling_function(
            temperature, metallicity, table
        )
    )(temperatures, metallicities)
    compiled = jax.jit(
        lambda temperature, metallicity: interpolate_log10_cooling_function(
            temperature, metallicity, table
        )
    )(temperatures[0], metallicities[0])
    np.testing.assert_allclose(compiled, vmapped[0], rtol=2.0e-15)
    assert np.isfinite(
        jax.grad(
            lambda temperature: interpolate_log10_cooling_function(
                temperature, metallicities[0], table
            )
        )(temperatures[0])
    )


def test_croton06_table_driven_rate_is_state_dependent():
    table = cloudy_cie_cooling_table()
    parameters = lagos23_croton06_cooling_parameters()

    def cooling_rate(hot_metal_mass):
        return croton06_unheated_cooling_from_table(
            1.0e11,
            hot_metal_mass,
            1.0e11,
            0.25,
            220.0,
            1.1,
            parameters,
            table,
        ).cooling_rate

    low = cooling_rate(1.0e8)
    high = cooling_rate(2.0e9)
    assert high > low
    assert np.isfinite(jax.grad(cooling_rate)(1.0e9))


def test_lagos23_deterministic_agn_functions_match_upstream_oracle():
    parameters = lagos23_agn_parameters()
    for case in _oracle()["agn"]:
        luminosity = eddington_luminosity_1e40_erg_per_s(case["stored_black_hole_mass"])
        np.testing.assert_allclose(luminosity, case["eddington_luminosity"], rtol=2.0e-15)
        accretion = lagos23_hot_halo_accretion_rate(
            case["pseudo_cooling_luminosity"],
            case["stored_black_hole_mass"],
            case["hot_gas_fraction"],
            case["virial_velocity"],
            parameters,
        )
        np.testing.assert_allclose(
            accretion, case["calculated_hot_halo_accretion_rate"], rtol=2.0e-15
        )
        mechanical = lagos23_mechanical_luminosity_1e40_erg_per_s(
            case["stored_black_hole_mass"],
            case["hot_accretion_rate"],
            case["starburst_accretion_rate"],
            case["stored_spin"],
            parameters,
        )
        np.testing.assert_allclose(mechanical, case["mechanical_luminosity"], rtol=2.0e-15)
        physical_rate = (
            case["hot_accretion_rate"] + case["starburst_accretion_rate"]
        ) / parameters.hubble_h
        physical_mass = case["stored_black_hole_mass"] / parameters.hubble_h
        ratio = eddington_accretion_ratio(
            physical_rate, physical_mass, parameters.radiative_efficiency
        )
        np.testing.assert_allclose(ratio, case["accretion_ratio"], rtol=2.0e-15)
        efficiency, isco = thin_disk_efficiency_and_isco(case["stored_spin"])
        np.testing.assert_array_equal(efficiency, case["radiative_efficiency"])
        np.testing.assert_array_equal(isco, case["isco_radius"])
        bolometric = lagos23_bolometric_luminosity_1e40_erg_per_s(
            case["stored_black_hole_mass"],
            case["hot_accretion_rate"],
            case["starburst_accretion_rate"],
            case["stored_spin"],
            parameters,
        )
        np.testing.assert_allclose(bolometric, case["bolometric_luminosity"], rtol=3.0e-15)
        critical = qso_critical_luminosity_1e40_erg_per_s(
            case["gas_mass"], case["bulge_baryonic_mass"], case["bulge_radius"]
        )
        np.testing.assert_allclose(critical, case["qso_critical_luminosity"], rtol=3.0e-15)
        salpeter = salpeter_timescale_gyr(bolometric, physical_mass)
        np.testing.assert_allclose(salpeter, case["salpeter_timescale"], rtol=3.0e-15)
        outflow_velocity = qso_outflow_velocity_km_per_s(
            bolometric,
            case["gas_metallicity"],
            case["gas_mass"],
            parameters,
        )
        np.testing.assert_allclose(outflow_velocity, case["qso_outflow_velocity"], rtol=3.0e-15)
        qso = lagos23_qso_outflow_loadings(
            gas_mass=case["gas_mass"],
            black_hole_mass_msun_over_h=case["stored_black_hole_mass"],
            hot_halo_accretion_rate_msun_over_h_per_gyr=case["hot_accretion_rate"],
            starburst_accretion_rate_msun_over_h_per_gyr=case["starburst_accretion_rate"],
            spin=case["stored_spin"],
            gas_metallicity=case["gas_metallicity"],
            circular_velocity_km_per_s=case["circular_velocity"],
            star_formation_rate=case["star_formation_rate"],
            bulge_baryonic_mass=case["bulge_baryonic_mass"],
            bulge_radius_mpc=case["bulge_radius"],
            parameters=parameters,
        )
        np.testing.assert_allclose(qso.reheating, case["qso_reheating_loading"], rtol=3e-15)
        np.testing.assert_allclose(qso.ejection, case["qso_ejection_loading"], rtol=3e-15)


def test_lagos23_heating_radius_is_an_explicit_markov_projection():
    parameters = lagos23_agn_parameters(memory_start_redshift=20.0)
    state = HeatingRadiusState(jnp.asarray(0.03))
    grown = project_lagos23_heating_radius(state, 0.05, 2.0, parameters)
    retained = project_lagos23_heating_radius(grown, 0.02, 1.0, parameters)
    before_memory = project_lagos23_heating_radius(state, 0.08, 25.0, parameters)
    np.testing.assert_allclose(grown.heating_radius_mpc, 0.05)
    np.testing.assert_allclose(retained.heating_radius_mpc, 0.05)
    np.testing.assert_allclose(before_memory.heating_radius_mpc, 0.03)

    regulated, ratio, saturated = cooling_rate_after_heating_radius(
        10.0, 0.10, retained, parameters
    )
    np.testing.assert_allclose(regulated, 5.0)
    np.testing.assert_allclose(ratio, 0.5)
    assert not bool(saturated)
    shut_off, ratio, saturated = cooling_rate_after_heating_radius(10.0, 0.06, retained, parameters)
    np.testing.assert_allclose(shut_off, 0.0)
    np.testing.assert_allclose(ratio, 1.0)
    assert bool(saturated)


def test_lagos23_agn_rate_has_exact_fractional_derivative_away_from_gates():
    parameters = lagos23_agn_parameters()

    def rate(log_black_hole_mass):
        black_hole_mass = jnp.exp(log_black_hole_mass)
        return jnp.log(
            lagos23_hot_halo_accretion_rate(1.0e6, black_hole_mass, 0.1, 220.0, parameters)
        )

    log_mass = jnp.log(1.0e8)
    np.testing.assert_allclose(jax.grad(rate)(log_mass), 1.0, rtol=2.0e-15)
    np.testing.assert_allclose(jax.jit(jax.grad(rate))(log_mass), 1.0, rtol=2.0e-15)
