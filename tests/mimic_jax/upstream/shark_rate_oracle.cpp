// Read-only oracle harness for public SHARK prescription functions.
//
// This file is compiled against a separately cloned, pinned upstream SHARK
// build. It does not link into mimic-jax and is not a replacement model.

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>

#include "cosmology.h"
#include "agn_feedback.h"
#include "dark_matter_halos.h"
#include "disk_instability.h"
#include "environment.h"
#include "execution.h"
#include "gas_cooling.h"
#include "galaxy.h"
#include "galaxy_mergers.h"
#include "halo.h"
#include "interpolator.h"
#include "options.h"
#include "physical_model.h"
#include "recycling.h"
#include "reincorporation.h"
#include "reionisation.h"
#include "simulation.h"
#include "star_formation.h"
#include "stellar_feedback.h"
#include "subhalo.h"

using namespace shark;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: shark_rate_oracle <upstream-config>\n";
    return 2;
  }

  Options options(argv[1]);
  CosmologicalParameters cosmological_parameters(options);
  auto cosmology = make_cosmology(cosmological_parameters);
  RecyclingParameters recycling(options);
  ExecutionParameters execution(options);
  DarkMatterHaloParameters dark_matter(options);
  SimulationParameters simulation(options);
  auto dark_matter_halos = make_dark_matter_halos(dark_matter, cosmology, simulation, execution);
  auto environment = make_environment(EnvironmentParameters(options), dark_matter_halos, cosmology,
                                      cosmological_parameters, simulation);
  StarFormation star_formation(StarFormationParameters(options), recycling, cosmology);
  StellarFeedback stellar_feedback(StellarFeedbackParameters(options), cosmology);
  auto reincorporation = make_reincorporation(ReincorporationParameters(options), nullptr);
  auto reionisation = make_reionisation(ReionisationParameters(options));
  auto agn_feedback = make_agn_feedback(AGNFeedbackParameters(options), cosmology, recycling,
                                        execution, dark_matter);
  GasCoolingParameters gas_cooling_parameters(options);
  Interpolator cooling_function(gas_cooling_parameters.cooling_table.get_temperatures(),
                                gas_cooling_parameters.cooling_table.get_metallicities(),
                                gas_cooling_parameters.cooling_table.get_lambda());
  GasCooling gas_cooling(gas_cooling_parameters, StarFormationParameters(options), execution,
                         reionisation, cosmology, agn_feedback, dark_matter, dark_matter_halos,
                         reincorporation, environment);
  BasicPhysicalModel physical_model(execution.ode_solver_precision, gas_cooling, stellar_feedback,
                                    star_formation, *agn_feedback, recycling,
                                    gas_cooling_parameters, AGNFeedbackParameters(options));

  struct StarFormationCase {
    double cold_gas;
    double stars;
    double gas_radius;
    double stellar_radius;
    double gas_metallicity;
    double redshift;
    bool burst;
    double galaxy_velocity;
    double gas_specific_angular_momentum;
  };
  const std::array<StarFormationCase, 4> star_formation_cases{{
      {1.0e10, 5.0e9, 0.010, 0.006, 0.010, 0.0, false, 200.0, 3.0},
      {3.0e9, 2.0e9, 0.006, 0.004, 0.004, 1.0, false, 120.0, 1.5},
      {8.0e10, 1.0e11, 0.025, 0.015, 0.020, 0.5, false, 300.0, 8.0},
      {2.0e10, 3.0e10, 0.003, 0.002, 0.015, 2.0, true, 250.0, 0.0},
  }};

  struct FeedbackCase {
    double subhalo_velocity;
    double galaxy_velocity;
    double redshift;
  };
  const std::array<FeedbackCase, 5> feedback_cases{{
      {60.0, 80.0, 0.0},
      {100.0, 120.0, 1.0},
      {150.0, 180.0, 2.0},
      {250.0, 260.0, 0.5},
      {400.0, 350.0, 0.0},
  }};

  struct ReincorporationCase {
    double ejected_gas;
    double halo_mass;
    double interval_gyr;
    bool satellite;
  };
  const std::array<ReincorporationCase, 5> reincorporation_cases{{
      {1.0e9, 1.0e10, 0.2, false},
      {1.0e9, 5.0e11, 0.2, false},
      {1.0e9, 1.0e12, 0.2, false},
      {1.0e9, 1.0e13, 0.2, false},
      {1.0e9, 1.0e12, 0.2, true},
  }};

  struct ReionisationCase {
    double virial_velocity;
    double redshift;
  };
  const std::array<ReionisationCase, 8> reionisation_cases{{
      {10.0, 0.0},
      {35.0, 0.0},
      {20.0, 5.0},
      {40.0, 5.0},
      {5.0, 9.0},
      {20.0, 9.0},
      {5.0, 10.0},
      {5.0, 12.0},
  }};

  struct CoolingCase {
    double hot_mass;
    double density_mass;
    double virial_radius;
    double virial_velocity;
    double halo_dynamical_time;
    double log10_cooling_function;
  };
  const std::array<CoolingCase, 5> cooling_cases{{
      {1.0e9, 1.0e9, 0.035, 70.0, 0.50, -23.5},
      {2.0e10, 2.0e10, 0.120, 140.0, 0.80, -22.8},
      {1.0e11, 1.0e11, 0.250, 220.0, 1.10, -24.0},
      {5.0e11, 6.0e11, 0.500, 350.0, 1.40, -23.2},
      {1.0e12, 1.4e12, 0.800, 500.0, 1.80, -24.5},
  }};

  struct CoolingFunctionCase {
    double log10_temperature;
    double metallicity;
  };
  const std::array<CoolingFunctionCase, 8> cooling_function_cases{{
      {3.5, -0.1},
      {4.0, 0.0},
      {4.17, 0.005},
      {5.31, 0.02},
      {6.27, 0.07},
      {7.43, 0.45},
      {8.5, 2.0},
      {9.5, 4.0},
  }};

  struct AgnCase {
    double black_hole_mass;
    double hot_gas_fraction;
    double virial_velocity;
    double pseudo_cooling_luminosity;
    double hot_accretion_rate;
    double starburst_accretion_rate;
    double spin;
    double gas_mass;
    double gas_metallicity;
    double circular_velocity;
    double star_formation_rate;
    double bulge_baryonic_mass;
    double bulge_radius;
  };
  const std::array<AgnCase, 5> agn_cases{{
      {1.0e6, 0.04, 100.0, 1.0e4, 1.0e3, 0.0, 0.1, 1.0e9, 0.002, 100.0, 1.0e8, 2.0e9, 0.004},
      {1.0e7, 0.08, 160.0, 1.0e5, 1.0e5, 2.0e4, 0.3, 2.0e9, 0.005, 150.0, 3.0e8, 5.0e9, 0.006},
      {1.0e8, 0.12, 220.0, 1.0e6, 1.0e7, 0.0, 0.6, 5.0e9, 0.01, 220.0, 1.0e9, 2.0e10, 0.010},
      {1.0e9, 0.15, 350.0, 1.0e7, 5.0e8, 1.0e8, 0.9, 1.0e10, 0.02, 300.0, 2.0e9, 1.0e11, 0.020},
      {5.0e9, 0.20, 500.0, 1.0e31, 1.0e9, 5.0e8, 0.998, 2.0e10, 0.03, 400.0, 4.0e9, 5.0e11, 0.040},
  }};

  struct StructureCase {
    double mass;
    double redshift;
    double normalized_radius;
  };
  const std::array<StructureCase, 5> structure_cases{{
      {1.0e10, 0.0, 0.01},
      {1.0e11, 0.5, 0.05},
      {1.0e12, 1.0, 0.1},
      {1.0e13, 2.0, 0.5},
      {5.0e14, 4.0, 1.0},
  }};

  struct SpinAccretionCase {
    long galaxy_id;
    double black_hole_mass;
    double initial_spin;
    double accreted_mass;
    double accretion_time;
  };
  const std::array<SpinAccretionCase, 5> spin_accretion_cases{{
      {10, 1.0e6, 0.0, 1.0e4, 0.05},
      {11, 1.0e7, 0.2, 2.0e5, 0.1},
      {12, 1.0e8, 0.6, 1.0e7, 0.2},
      {13, 1.0e9, -0.4, 5.0e7, 0.5},
      {14, 5.0e9, 0.9, 1.0e8, 1.0},
  }};

  struct SpinMergerCase {
    long galaxy_id;
    double primary_mass;
    double secondary_mass;
    double primary_spin;
    double secondary_spin;
  };
  const std::array<SpinMergerCase, 4> spin_merger_cases{{
      {20, 1.0e8, 1.0e8, 0.0, 0.0},
      {21, 1.0e8, 2.0e7, 0.6, 0.2},
      {22, 5.0e8, 1.0e8, -0.4, 0.8},
      {23, 2.0e7, 1.0e8, 0.3, -0.7},
  }};

  std::cout << "MIMIC_SHARK_ORACLE_BEGIN\n";
  std::cout << std::setprecision(17);
  std::cout << "{\n  \"cosmology\": {\"age_at_feedback_redshift_power_gyr\": "
            << cosmology->convert_redshift_to_age(0.12) << "},\n";
  std::cout << "  \"star_formation\": [\n";
  for (std::size_t index = 0; index < star_formation_cases.size(); ++index) {
    const auto &value = star_formation_cases[index];
    double angular_momentum_rate = 0.0;
    double rate = star_formation.star_formation_rate(
        value.cold_gas, value.stars, value.gas_radius, value.stellar_radius, value.gas_metallicity,
        value.redshift, value.burst, value.galaxy_velocity, angular_momentum_rate,
        value.gas_specific_angular_momentum);
    std::cout << "    {\"cold_gas\": " << value.cold_gas << ", \"stars\": " << value.stars
              << ", \"gas_radius\": " << value.gas_radius
              << ", \"stellar_radius\": " << value.stellar_radius
              << ", \"gas_metallicity\": " << value.gas_metallicity
              << ", \"redshift\": " << value.redshift
              << ", \"burst\": " << (value.burst ? "true" : "false")
              << ", \"galaxy_velocity\": " << value.galaxy_velocity
              << ", \"gas_specific_angular_momentum\": " << value.gas_specific_angular_momentum
              << ", \"rate\": " << rate << ", \"angular_momentum_rate\": " << angular_momentum_rate
              << "}";
    std::cout << (index + 1 == star_formation_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"stellar_feedback\": [\n";
  for (std::size_t index = 0; index < feedback_cases.size(); ++index) {
    const auto &value = feedback_cases[index];
    double beta_reheat = 0.0;
    double beta_eject = 0.0;
    double beta_j_reheat = 0.0;
    double beta_j_eject = 0.0;
    stellar_feedback.outflow_rate(1.0, value.subhalo_velocity, value.galaxy_velocity,
                                  value.redshift, beta_reheat, beta_eject, beta_j_reheat,
                                  beta_j_eject);
    std::cout << "    {\"subhalo_velocity\": " << value.subhalo_velocity
              << ", \"galaxy_velocity\": " << value.galaxy_velocity
              << ", \"redshift\": " << value.redshift << ", \"reheating_loading\": " << beta_reheat
              << ", \"ejection_loading\": " << beta_eject
              << ", \"angular_momentum_reheating_loading\": " << beta_j_reheat
              << ", \"angular_momentum_ejection_loading\": " << beta_j_eject << "}";
    std::cout << (index + 1 == feedback_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"reincorporation\": [\n";
  for (std::size_t index = 0; index < reincorporation_cases.size(); ++index) {
    const auto &value = reincorporation_cases[index];
    Halo halo(static_cast<Halo::id_t>(index), 0);
    halo.Mvir = value.halo_mass;
    Subhalo subhalo(static_cast<Subhalo::id_t>(index), 0);
    subhalo.ejected_galaxy_gas.mass = value.ejected_gas;
    subhalo.subhalo_type = value.satellite ? Subhalo::SATELLITE : Subhalo::CENTRAL;
    double requested = reincorporation->reincorporated_mass(halo, subhalo, 0.0, value.interval_gyr);
    double realized = requested;
    if (realized > subhalo.ejected_galaxy_gas.mass && realized > 0.0) {
      realized = subhalo.ejected_galaxy_gas.mass;
    }
    std::cout << "    {\"ejected_gas\": " << value.ejected_gas
              << ", \"halo_mass\": " << value.halo_mass
              << ", \"interval_gyr\": " << value.interval_gyr
              << ", \"satellite\": " << (value.satellite ? "true" : "false")
              << ", \"requested_mass\": " << requested << ", \"realized_mass\": " << realized
              << "}";
    std::cout << (index + 1 == reincorporation_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"reionisation\": [\n";
  for (std::size_t index = 0; index < reionisation_cases.size(); ++index) {
    const auto &value = reionisation_cases[index];
    bool reionised = reionisation->reionised_halo(value.virial_velocity, value.redshift);
    std::cout << "    {\"virial_velocity\": " << value.virial_velocity
              << ", \"redshift\": " << value.redshift
              << ", \"reionised\": " << (reionised ? "true" : "false") << "}";
    std::cout << (index + 1 == reionisation_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"cooling\": [\n";
  for (std::size_t index = 0; index < cooling_cases.size(); ++index) {
    const auto &value = cooling_cases[index];
    const double temperature = 35.9 * std::pow(value.virial_velocity, 2.0);
    const double log_lambda = std::min(value.log10_cooling_function, -23.0);
    const double mean_density = gas_cooling.mean_density(value.density_mass, value.virial_radius);
    const double characteristic_time = value.halo_dynamical_time * constants::GYR2S;
    const double raw_radius = gas_cooling.cooling_radius(
        value.hot_mass, value.virial_radius, characteristic_time, log_lambda, temperature);
    const double bounded_radius = std::min(raw_radius, value.virial_radius);
    const double rate = raw_radius < value.virial_radius
                            ? 0.5 * (raw_radius / value.virial_radius) *
                                  (value.hot_mass / value.halo_dynamical_time)
                            : value.hot_mass / value.halo_dynamical_time;
    const double shell_density =
        gas_cooling.density_shell(value.density_mass, value.virial_radius,
                                  std::min(0.75 * value.virial_radius, bounded_radius));
    const double local_cooling_time =
        gas_cooling.cooling_time(temperature, log_lambda, shell_density);
    const double luminosity = gas_cooling.cooling_luminosity(log_lambda, bounded_radius,
                                                             value.virial_radius, value.hot_mass);
    std::cout << "    {\"hot_mass\": " << value.hot_mass
              << ", \"density_mass\": " << value.density_mass
              << ", \"virial_radius\": " << value.virial_radius
              << ", \"virial_velocity\": " << value.virial_velocity
              << ", \"halo_dynamical_time\": " << value.halo_dynamical_time
              << ", \"log10_cooling_function_input\": " << value.log10_cooling_function
              << ", \"log10_cooling_function\": " << log_lambda
              << ", \"virial_temperature\": " << temperature
              << ", \"mean_number_density\": " << mean_density
              << ", \"cooling_radius\": " << bounded_radius << ", \"cooling_rate\": " << rate
              << ", \"shell_number_density\": " << shell_density
              << ", \"shell_cooling_time\": " << local_cooling_time
              << ", \"cooling_luminosity\": " << luminosity << "}";
    std::cout << (index + 1 == cooling_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"cooling_function\": [\n";
  for (std::size_t index = 0; index < cooling_function_cases.size(); ++index) {
    const auto &value = cooling_function_cases[index];
    const double interpolated = cooling_function.get(value.log10_temperature, value.metallicity);
    std::cout << "    {\"log10_temperature\": " << value.log10_temperature
              << ", \"metallicity\": " << value.metallicity
              << ", \"log10_cooling_function\": " << interpolated << "}";
    std::cout << (index + 1 == cooling_function_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"agn\": [\n";
  for (std::size_t index = 0; index < agn_cases.size(); ++index) {
    const auto &value = agn_cases[index];
    Galaxy galaxy(static_cast<Galaxy::id_t>(index));
    galaxy.smbh.mass = value.black_hole_mass;
    galaxy.smbh.spin = value.spin;
    galaxy.smbh.macc_hh = value.hot_accretion_rate;
    galaxy.smbh.macc_sb = value.starburst_accretion_rate;
    const double eddington = agn_feedback->eddington_luminosity(galaxy.smbh.mass);
    const double accretion = agn_feedback->accretion_rate_hothalo_smbh(
        value.pseudo_cooling_luminosity, 0.0, value.hot_gas_fraction, value.virial_velocity,
        galaxy);
    const double mechanical = agn_feedback->agn_mechanical_luminosity(galaxy.smbh);
    const double accretion_ratio = agn_feedback->accretion_rate_ratio(
        cosmology->comoving_to_physical_mass(value.hot_accretion_rate +
                                             value.starburst_accretion_rate),
        cosmology->comoving_to_physical_mass(galaxy.smbh.mass));
    const auto efficiency = agn_feedback->efficiency_luminosity_agn(galaxy.smbh.spin);
    const double bolometric = agn_feedback->agn_bolometric_luminosity(galaxy.smbh, true);
    const double critical = agn_feedback->qso_critical_luminosity(
        value.gas_mass, value.bulge_baryonic_mass, value.bulge_radius);
    const double salpeter = agn_feedback->salpeter_timescale(
        bolometric, cosmology->comoving_to_physical_mass(galaxy.smbh.mass));
    const double outflow_velocity = agn_feedback->qso_outflow_velocity(
        bolometric, cosmology->comoving_to_physical_mass(galaxy.smbh.mass), value.gas_metallicity,
        value.gas_mass, value.bulge_baryonic_mass, value.bulge_radius);
    double qso_reheating = 0.0;
    double qso_ejection = 0.0;
    agn_feedback->qso_outflow_rate(value.gas_mass, galaxy.smbh, value.gas_metallicity,
                                   value.circular_velocity, value.star_formation_rate,
                                   value.bulge_baryonic_mass, value.bulge_radius, qso_reheating,
                                   qso_ejection);
    std::cout << "    {\"black_hole_mass\": " << value.black_hole_mass
              << ", \"stored_black_hole_mass\": " << galaxy.smbh.mass
              << ", \"hot_gas_fraction\": " << value.hot_gas_fraction
              << ", \"virial_velocity\": " << value.virial_velocity
              << ", \"pseudo_cooling_luminosity\": " << value.pseudo_cooling_luminosity
              << ", \"hot_accretion_rate\": " << value.hot_accretion_rate
              << ", \"starburst_accretion_rate\": " << value.starburst_accretion_rate
              << ", \"spin\": " << value.spin << ", \"stored_spin\": " << galaxy.smbh.spin
              << ", \"gas_mass\": " << value.gas_mass
              << ", \"gas_metallicity\": " << value.gas_metallicity
              << ", \"circular_velocity\": " << value.circular_velocity
              << ", \"star_formation_rate\": " << value.star_formation_rate
              << ", \"bulge_baryonic_mass\": " << value.bulge_baryonic_mass
              << ", \"bulge_radius\": " << value.bulge_radius
              << ", \"eddington_luminosity\": " << eddington
              << ", \"calculated_hot_halo_accretion_rate\": " << accretion
              << ", \"mechanical_luminosity\": " << mechanical
              << ", \"accretion_ratio\": " << accretion_ratio
              << ", \"radiative_efficiency\": " << efficiency[0]
              << ", \"isco_radius\": " << efficiency[1]
              << ", \"bolometric_luminosity\": " << bolometric
              << ", \"qso_critical_luminosity\": " << critical
              << ", \"salpeter_timescale\": " << salpeter
              << ", \"qso_outflow_velocity\": " << outflow_velocity
              << ", \"qso_reheating_loading\": " << qso_reheating
              << ", \"qso_ejection_loading\": " << qso_ejection << "}";
    std::cout << (index + 1 == agn_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"structure\": [\n";
  for (std::size_t index = 0; index < structure_cases.size(); ++index) {
    const auto &value = structure_cases[index];
    const double velocity = dark_matter_halos->halo_virial_velocity(value.mass, value.redshift);
    const double radius = dark_matter_halos->halo_virial_radius(value.mass, value.redshift);
    const double concentration = dark_matter_halos->nfw_concentration(value.mass, value.redshift);
    const double enclosed =
        dark_matter_halos->enclosed_mass(value.normalized_radius, concentration);
    std::cout << "    {\"mass\": " << value.mass << ", \"redshift\": " << value.redshift
              << ", \"normalized_radius\": " << value.normalized_radius
              << ", \"hubble_parameter\": " << cosmology->hubble_parameter(value.redshift)
              << ", \"cosmic_age\": " << cosmology->convert_redshift_to_age(value.redshift)
              << ", \"virial_velocity\": " << velocity << ", \"virial_radius\": " << radius
              << ", \"dynamical_time\": "
              << constants::MPCKM2GYR *
                     cosmology->comoving_to_physical_size(radius, value.redshift) / velocity
              << ", \"concentration\": " << concentration
              << ", \"nfw_enclosed_fraction\": " << enclosed << "}";
    std::cout << (index + 1 == structure_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"rng\": [\n";
  for (std::size_t index = 0; index < 4; ++index) {
    const auto component_id = static_cast<long>(30 + index);
    std::default_random_engine generator(execution.seed + component_id);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::cout << "    {\"seed\": " << execution.seed + component_id << ", \"values\": ["
              << distribution(generator) << ", " << distribution(generator) << ", "
              << distribution(generator) << "]}";
    std::cout << (index + 1 == 4 ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"spin_accretion\": [\n";
  for (std::size_t index = 0; index < spin_accretion_cases.size(); ++index) {
    const auto &value = spin_accretion_cases[index];
    Galaxy galaxy(static_cast<Galaxy::id_t>(value.galaxy_id));
    galaxy.smbh.mass = value.black_hole_mass;
    galaxy.smbh.spin = value.initial_spin;
    agn_feedback->griffin19_spinup_accretion(value.accreted_mass, value.accretion_time, galaxy);
    std::cout << "    {\"galaxy_id\": " << value.galaxy_id
              << ", \"black_hole_mass\": " << value.black_hole_mass
              << ", \"initial_spin\": " << value.initial_spin
              << ", \"accreted_mass\": " << value.accreted_mass
              << ", \"accretion_time\": " << value.accretion_time
              << ", \"final_spin\": " << galaxy.smbh.spin << "}";
    std::cout << (index + 1 == spin_accretion_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"spin_merger\": [\n";
  for (std::size_t index = 0; index < spin_merger_cases.size(); ++index) {
    const auto &value = spin_merger_cases[index];
    Galaxy galaxy(static_cast<Galaxy::id_t>(value.galaxy_id));
    BlackHole primary;
    BlackHole secondary;
    primary.mass = value.primary_mass;
    primary.spin = value.primary_spin;
    secondary.mass = value.secondary_mass;
    secondary.spin = value.secondary_spin;
    agn_feedback->griffin19_spinup_mergers(primary, secondary, galaxy);
    std::cout << "    {\"galaxy_id\": " << value.galaxy_id
              << ", \"primary_mass\": " << value.primary_mass
              << ", \"secondary_mass\": " << value.secondary_mass
              << ", \"primary_spin\": " << value.primary_spin
              << ", \"secondary_spin\": " << value.secondary_spin
              << ", \"final_spin\": " << primary.spin << "}";
    std::cout << (index + 1 == spin_merger_cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"reference_interval\": ";
  {
    constexpr double redshift = 1.0;
    constexpr double interval = 0.2;
    auto halo = std::make_shared<Halo>(42, 100);
    auto subhalo = std::make_shared<Subhalo>(42, 100);
    halo->central_subhalo = subhalo;
    halo->Mvir = 8.0e11;
    halo->Vvir = 180.0;
    halo->lambda = 0.03;
    subhalo->host_halo = halo;
    subhalo->subhalo_type = Subhalo::CENTRAL;
    subhalo->Mvir = 8.0e11;
    subhalo->Vvir = 180.0;
    subhalo->lambda = 0.03;
    subhalo->L = {4.0e12, 0.0, 0.0};
    subhalo->hot_halo_gas.mass = 8.0e10;
    subhalo->hot_halo_gas.mass_metals = 8.0e8;
    subhalo->hot_halo_gas.sAM = 5.0;
    subhalo->ejected_galaxy_gas.mass = 1.0e10;
    subhalo->ejected_galaxy_gas.mass_metals = 1.0e8;
    subhalo->ejected_galaxy_gas.sAM = 5.0;
    subhalo->accreted_mass = 0.0;
    auto &galaxy = subhalo->emplace_galaxy(42);
    galaxy.galaxy_type = Galaxy::CENTRAL;
    galaxy.disk_stars.mass = 2.0e9;
    galaxy.disk_stars.mass_metals = 2.0e7;
    galaxy.disk_stars.rscale = 0.006;
    galaxy.disk_stars.sAM = 180.0 * 0.006 / constants::EAGLEJconv;
    galaxy.disk_gas.mass = 3.0e9;
    galaxy.disk_gas.mass_metals = 3.0e7;
    galaxy.disk_gas.rscale = 0.008;
    galaxy.disk_gas.sAM = 180.0 * 0.008 / constants::EAGLEJconv;
    galaxy.smbh.mass = 2.0e6;
    galaxy.smbh.spin = 0.3;
    galaxy.vmax = 180.0;
    physical_model.evolve_galaxy(*subhalo, galaxy, redshift, interval, true);
    std::cout << "{\"redshift\": " << redshift << ", \"duration_gyr\": " << interval
              << ", \"halo_mass\": " << halo->Mvir << ", \"virial_velocity\": " << halo->Vvir
              << ", \"virial_radius\": "
              << dark_matter_halos->halo_virial_radius(halo->Mvir, redshift)
              << ", \"halo_dynamical_time\": "
              << dark_matter_halos->subhalo_dynamical_time(*subhalo, redshift)
              << ", \"cooling_specific_angular_momentum\": " << subhalo->cold_halo_gas.sAM
              << ", \"stellar_mass\": " << galaxy.disk_stars.mass
              << ", \"cold_gas\": " << galaxy.disk_gas.mass
              << ", \"hot_halo_gas\": " << subhalo->hot_halo_gas.mass
              << ", \"cold_halo_gas\": " << subhalo->cold_halo_gas.mass
              << ", \"ejected_gas\": " << subhalo->ejected_galaxy_gas.mass
              << ", \"lost_gas\": " << subhalo->lost_galaxy_gas.mass
              << ", \"stellar_metals\": " << galaxy.disk_stars.mass_metals
              << ", \"cold_gas_metals\": " << galaxy.disk_gas.mass_metals
              << ", \"hot_halo_gas_metals\": " << subhalo->hot_halo_gas.mass_metals
              << ", \"ejected_gas_metals\": " << subhalo->ejected_galaxy_gas.mass_metals
              << ", \"black_hole_mass\": " << galaxy.smbh.mass
              << ", \"black_hole_spin\": " << galaxy.smbh.spin
              << ", \"black_hole_accretion_rate\": " << galaxy.smbh.macc_hh
              << ", \"cooling_rate\": " << subhalo->cooling_rate
              << ", \"star_formation_rate\": " << galaxy.sfr_disk
              << ", \"heating_radius\": " << subhalo->cooling_subhalo_tracking.rheat
              << ", \"heating_ratio\": " << galaxy.mheat_ratio
              << ", \"hydrostatic\": " << (halo->hydrostatic_eq ? "true" : "false")
              << ", \"rhs_evaluations\": " << physical_model.get_galaxy_ode_evaluations() << "}";
  }
  std::cout << ",\n  \"reference_starburst\": ";
  {
    constexpr double redshift = 1.0;
    constexpr double interval = 0.2;
    auto halo = std::make_shared<Halo>(84, 100);
    auto subhalo = std::make_shared<Subhalo>(84, 100);
    halo->central_subhalo = subhalo;
    halo->Mvir = 8.0e11;
    halo->Vvir = 180.0;
    subhalo->host_halo = halo;
    subhalo->subhalo_type = Subhalo::CENTRAL;
    subhalo->Mvir = 8.0e11;
    subhalo->Vvir = 180.0;
    subhalo->hot_halo_gas.mass = 8.0e10;
    subhalo->hot_halo_gas.mass_metals = 8.0e8;
    subhalo->ejected_galaxy_gas.mass = 1.0e10;
    subhalo->ejected_galaxy_gas.mass_metals = 1.0e8;
    auto &galaxy = subhalo->emplace_galaxy(84);
    galaxy.galaxy_type = Galaxy::CENTRAL;
    galaxy.bulge_stars.mass = 2.0e9;
    galaxy.bulge_stars.mass_metals = 2.0e7;
    galaxy.bulge_stars.rscale = 0.006;
    galaxy.bulge_stars.sAM = 1.0e9 / 2.0e9;
    galaxy.bulge_gas.mass = 3.0e9;
    galaxy.bulge_gas.mass_metals = 3.0e7;
    galaxy.bulge_gas.rscale = 0.008;
    galaxy.bulge_gas.sAM = 2.0e9 / 3.0e9;
    galaxy.smbh.mass = 2.0e6;
    galaxy.smbh.spin = 0.3;
    galaxy.vmax = 180.0;

    const double accretion_time = agn_feedback->smbh_accretion_timescale(galaxy, redshift);
    const double delta_mbh = agn_feedback->smbh_growth_starburst(
        galaxy.bulge_gas.mass, subhalo->Vvir, accretion_time, galaxy);
    const double delta_mzbh = delta_mbh / galaxy.bulge_gas.mass * galaxy.bulge_gas.mass_metals;
    galaxy.smbh.macc_sb += delta_mbh / accretion_time;
    galaxy.bulge_gas.mass -= delta_mbh;
    galaxy.bulge_gas.mass_metals -= delta_mzbh;
    physical_model.evolve_galaxy_starburst(*subhalo, galaxy, redshift, interval, true, false);
    galaxy.smbh.mass += delta_mbh;
    galaxy.smbh.mass_metals += delta_mzbh;

    std::cout << "{\"redshift\": " << redshift << ", \"duration_gyr\": " << interval
              << ", \"virial_velocity\": " << subhalo->Vvir
              << ", \"subhalo_velocity\": " << subhalo->Vvir << ", \"galaxy_id\": " << galaxy.id
              << ", \"black_hole_accretion_time\": " << accretion_time
              << ", \"black_hole_transfer\": " << delta_mbh
              << ", \"black_hole_metal_transfer\": " << delta_mzbh
              << ", \"black_hole_spin\": " << galaxy.smbh.spin
              << ", \"black_hole_mass\": " << galaxy.smbh.mass
              << ", \"black_hole_metals\": " << galaxy.smbh.mass_metals
              << ", \"bulge_stellar_mass\": " << galaxy.bulge_stars.mass
              << ", \"bulge_gas_mass\": " << galaxy.bulge_gas.mass
              << ", \"bulge_stellar_metals\": " << galaxy.bulge_stars.mass_metals
              << ", \"bulge_gas_metals\": " << galaxy.bulge_gas.mass_metals
              << ", \"hot_halo_gas\": " << subhalo->hot_halo_gas.mass
              << ", \"hot_halo_gas_metals\": " << subhalo->hot_halo_gas.mass_metals
              << ", \"ejected_gas\": " << subhalo->ejected_galaxy_gas.mass
              << ", \"ejected_gas_metals\": " << subhalo->ejected_galaxy_gas.mass_metals
              << ", \"lost_gas\": " << subhalo->lost_galaxy_gas.mass
              << ", \"lost_gas_metals\": " << subhalo->lost_galaxy_gas.mass_metals
              << ", \"star_formation_rate\": " << galaxy.sfr_bulge_diskins
              << ", \"rhs_evaluations\": " << physical_model.get_galaxy_ode_evaluations() << "}";
  }
  std::cout << "\n}\n";
  std::cout << "MIMIC_SHARK_ORACLE_END\n";
  return 0;
}
