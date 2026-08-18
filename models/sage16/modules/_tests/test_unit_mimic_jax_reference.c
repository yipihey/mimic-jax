/**
 * @file    test_unit_mimic_jax_reference.c
 * @brief   Executable C reference cases for the initial mimic-jax SAGE16 slice
 *
 * The Python equivalence checker parses the MIMIC_JAX_REFERENCE records emitted
 * here. Values come from the actual compiled SAGE16 modules, not duplicated
 * formulas. Keep these fixtures small, deterministic, and away from thresholds
 * unless a threshold is the behavior under test.
 */

#include "../../../../tests/framework/test_framework.h"
#include "../../../../tests/framework/test_phase_config.h"
#include "core/module_interface.h"
#include "core/module_registry.h"
#include "core/galaxy_pool.h"
#include "core/inheritance.h"
#include "include/globals.h"
#include "include/proto.h"
#include "include/types.h"
#include "module_system/generated/event_contracts.h"
#include "../sage_calculate_cooling_budget/cooling_tables.h"
#include "util/error.h"
#include "util/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "modules/_tests/sage_test_fixtures.h"

extern int sage_apply_cooling_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_apply_cooling_init(void);
extern int sage_apply_cooling_cleanup(void);
extern int sage_apply_infall_init(void);
extern int sage_apply_infall_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_apply_infall_cleanup(void);
extern int sage_calculate_cooling_budget_init(void);
extern int sage_calculate_cooling_budget_process(struct ModuleContext *ctx, struct Halo *halos,
                                                 int ngal);
extern int sage_calculate_cooling_budget_cleanup(void);
extern int sage_disk_instability_init(void);
extern int sage_disk_instability_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_disk_instability_cleanup(void);
extern int sage_set_disk_scale_radius_init(void);
extern int sage_set_disk_scale_radius_process(struct ModuleContext *ctx, struct Halo *halos,
                                              int ngal);
extern int sage_set_disk_scale_radius_cleanup(void);
extern int sage_initialise_merger_clock_init(void);
extern int sage_initialise_merger_clock_process(struct ModuleContext *ctx, struct Halo *halos,
                                                int ngal);
extern int sage_initialise_merger_clock_cleanup(void);
extern int sage_resolve_mergers_and_disruption_init(void);
extern int sage_resolve_mergers_and_disruption_process(struct ModuleContext *ctx,
                                                       struct Halo *halos, int ngal);
extern int sage_resolve_mergers_and_disruption_cleanup(void);
extern int sage_reincorporation_init(void);
extern int sage_reincorporation_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_reincorporation_cleanup(void);
extern int sage_satellite_stripping_init(void);
extern int sage_satellite_stripping_process(struct ModuleContext *ctx, struct Halo *halos,
                                            int ngal);
extern int sage_satellite_stripping_cleanup(void);
extern int sage_reionization_init(void);
extern int sage_reionization_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_reionization_cleanup(void);
extern int sage_prepare_infall_budget_init(void);
extern int sage_prepare_infall_budget_process(struct ModuleContext *ctx, struct Halo *halos,
                                              int ngal);
extern int sage_prepare_infall_budget_cleanup(void);
extern int sage_radio_mode_heating_init(void);
extern int sage_radio_mode_heating_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_radio_mode_heating_cleanup(void);
extern int sage_quasar_mode_init(void);
extern int sage_quasar_mode_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_quasar_mode_cleanup(void);
extern int sage_starburst_feedback_init(void);
extern int sage_starburst_feedback_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_starburst_feedback_cleanup(void);
extern int sage_calculate_star_formation_init(void);
extern int sage_calculate_star_formation_process(struct ModuleContext *ctx, struct Halo *halos,
                                                 int ngal);
extern int sage_calculate_star_formation_cleanup(void);
extern int sage_calculate_supernova_feedback_init(void);
extern int sage_calculate_supernova_feedback_process(struct ModuleContext *ctx, struct Halo *halos,
                                                     int ngal);
extern int sage_calculate_supernova_feedback_cleanup(void);
extern int sage_apply_star_formation_supernova_init(void);
extern int sage_apply_star_formation_supernova_process(struct ModuleContext *ctx,
                                                       struct Halo *halos, int ngal);
extern int sage_apply_star_formation_supernova_cleanup(void);
extern int sage_apply_metal_enrichment_init(void);
extern int sage_apply_metal_enrichment_process(struct ModuleContext *ctx, struct Halo *halos,
                                               int ngal);
extern int sage_apply_metal_enrichment_cleanup(void);

static void add_parameter(const char *name, const char *value) {
  const int index = MimicConfig.NumModelParams++;
  snprintf(MimicConfig.ModelParams[index].param_name, MAX_STRING_LEN, "%s", name);
  snprintf(MimicConfig.ModelParams[index].value, MAX_STRING_LEN, "%s", value);
}

static void configure_fiducial_slice(void) {
  MimicConfig.Omega = 0.25;
  MimicConfig.OmegaLambda = 0.75;
  MimicConfig.Hubble_h = 0.73;
  MimicConfig.SubSteps = 1;
  set_units();

  add_parameter("GlobalBaryonFraction", "0.17");
  add_parameter("SfrEfficiency", "0.05");
  add_parameter("StarFormingDiskFactor", "3.0");
  add_parameter("FeedbackReheatingEpsilon", "3.0");
  add_parameter("FeedbackEjectionEfficiency", "0.3");
  add_parameter("ReIncorporationFactor", "0.15");
  add_parameter("AGNrecipe", "2");
  add_parameter("RadioModeEfficiency", "0.08");
  add_parameter("BlackHoleGrowthRate", "0.015");
  add_parameter("QuasarModeEfficiency", "0.005");
  add_parameter("RecycleFraction", "0.43");
  add_parameter("Yield", "0.025");
  add_parameter("FracZleaveDisk", "0.0");
  add_parameter("ThresholdMajorMerger", "0.3");
  add_parameter("ThresholdSatDisruption", "1.0");

  test_pre_timestep_add("sage_reionization", PROCESSING_MODE_FULL_HALO);
  test_pre_timestep_add("sage_prepare_infall_budget", PROCESSING_MODE_FULL_HALO);
  test_pre_timestep_add("sage_set_disk_scale_radius", PROCESSING_MODE_FULL_HALO);
  test_pre_timestep_add("sage_initialise_merger_clock", PROCESSING_MODE_FULL_HALO);
  test_phase_add("galaxy_physics", "sage_apply_infall", PROCESSING_MODE_FULL_HALO);
  test_phase_add("galaxy_physics", "sage_reincorporation", PROCESSING_MODE_FULL_HALO);
  test_phase_add("galaxy_physics", "sage_satellite_stripping", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_calculate_cooling_budget", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_radio_mode_heating", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_cooling", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_calculate_star_formation", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_calculate_supernova_feedback", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_star_formation_supernova",
                 PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_disk_instability", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_quasar_mode", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_starburst_feedback", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_metal_enrichment", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("satellite_mergers", "sage_resolve_mergers_and_disruption",
                 PROCESSING_MODE_FULL_HALO);
  test_phase_add("satellite_mergers", "sage_quasar_mode", PROCESSING_MODE_PER_EVENT);
  test_phase_add("satellite_mergers", "sage_starburst_feedback", PROCESSING_MODE_PER_EVENT);
}

static void setup_halo(struct Halo *halo, struct GalaxyData *galaxy, double mvir, double rvir,
                       double vvir, double dt) {
  memset(halo, 0, sizeof(*halo));
  memset(galaxy, 0, sizeof(*galaxy));
  halo->Type = 0;
  halo->SnapNum = 63;
  halo->Mvir = mvir;
  halo->Rvir = rvir;
  halo->Vvir = vvir;
  halo->dT = dt;
  halo->galaxy = galaxy;
}

static void setup_context(struct ModuleContext *ctx, struct Halo *central, int num_substeps) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->central_galaxy = central;
  ctx->params = &MimicConfig;
  ctx->snapshot_number = 63;
  ctx->num_substeps = num_substeps;
  ctx->substep_number = 0;
  ctx->time = 13.8;
  ctx->substep_dt = central->dT / num_substeps;
}

static int test_emit_reference_cases(void) {
  struct Halo halo;
  struct GalaxyData galaxy;
  struct ModuleContext context;

  init_memory_system(0);
  reset_config();
  configure_fiducial_slice();

  TEST_ASSERT(sage_calculate_cooling_budget_init() == 0, "Cooling-budget init should succeed");
  TEST_ASSERT(sage_apply_cooling_init() == 0, "Cooling-application init should succeed");
  TEST_ASSERT(sage_reionization_init() == 0, "Reionization init should succeed");
  TEST_ASSERT(sage_prepare_infall_budget_init() == 0, "Infall-budget init should succeed");
  TEST_ASSERT(sage_apply_infall_init() == 0, "Infall-application init should succeed");
  TEST_ASSERT(sage_satellite_stripping_init() == 0, "Satellite-stripping init should succeed");
  TEST_ASSERT(sage_reincorporation_init() == 0, "Reincorporation init should succeed");
  TEST_ASSERT(sage_radio_mode_heating_init() == 0, "Radio-mode init should succeed");
  TEST_ASSERT(sage_calculate_star_formation_init() == 0, "SF init should succeed");
  TEST_ASSERT(sage_calculate_supernova_feedback_init() == 0, "SN init should succeed");
  TEST_ASSERT(sage_apply_star_formation_supernova_init() == 0, "SF/SN apply init should succeed");
  TEST_ASSERT(sage_disk_instability_init() == 0, "Disk-instability init should succeed");
  TEST_ASSERT(sage_set_disk_scale_radius_init() == 0, "Disk-radius init should succeed");
  TEST_ASSERT(sage_initialise_merger_clock_init() == 0, "Merger-clock init should succeed");
  TEST_ASSERT(sage_resolve_mergers_and_disruption_init() == 0,
              "Merger resolver init should succeed");
  TEST_ASSERT(sage_quasar_mode_init() == 0, "Quasar-mode init should succeed");
  TEST_ASSERT(sage_starburst_feedback_init() == 0, "Starburst init should succeed");
  TEST_ASSERT(sage_apply_metal_enrichment_init() == 0, "Metal enrichment init should succeed");

  const double log_z_sun = log10(0.02);
  printf("MIMIC_JAX_REFERENCE case=cooling_interpolation midpoint=%.17g low_temperature=%.17g "
         "high_temperature=%.17g primordial=%.17g\n",
         get_metaldependent_cooling_rate(5.025, log_z_sun - 0.75),
         get_metaldependent_cooling_rate(3.0, log_z_sun),
         get_metaldependent_cooling_rate(9.0, log_z_sun),
         get_metaldependent_cooling_rate(5.5, -10.0));

  setup_halo(&halo, &galaxy, 1.0, 0.1, 100.0, 0.01);
  setup_context(&context, &halo, 10);
  context.redshift = 2.0;
  TEST_ASSERT(sage_reionization_process(&context, &halo, 1) == 0,
              "Reionization reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=reionization HaloBaryonFraction=%.17g\n",
         galaxy.HaloBaryonFraction);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 1);
  halo.Spin[0] = 100.0f;
  halo.Spin[1] = 150.0f;
  halo.Spin[2] = 200.0f;
  galaxy.DiskScaleRadius = 0.123f;
  TEST_ASSERT(sage_set_disk_scale_radius_process(&context, &halo, 1) == 0,
              "Disk-radius reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=disk_radius DiskScaleRadius=%.17g\n",
         (double)galaxy.DiskScaleRadius);

  struct Halo clock_halos[4];
  struct GalaxyData clock_galaxies[4];
  setup_halo(&clock_halos[0], &clock_galaxies[0], 100.0, 0.5, 200.0, 0.01);
  setup_halo(&clock_halos[1], &clock_galaxies[1], 20.0, 0.2, 100.0, 0.01);
  setup_halo(&clock_halos[2], &clock_galaxies[2], 5.0, 0.1, 50.0, 0.01);
  setup_halo(&clock_halos[3], &clock_galaxies[3], 20.0, 0.2, 100.0, 0.01);
  clock_halos[0].Len = 1000;
  clock_halos[1].Type = 1;
  clock_halos[1].Len = 200;
  clock_halos[2].Type = 2;
  clock_halos[2].Len = 0;
  clock_halos[2].CentralHalo = 1;
  clock_halos[3].Type = 3;
  clock_halos[3].Len = 200;
  clock_galaxies[0].MergTime = 5.0f;
  clock_galaxies[1].MergTime = 999.9f;
  clock_galaxies[1].StellarMass = 5.0f;
  clock_galaxies[1].ColdGas = 2.0f;
  clock_galaxies[2].MergTime = 999.9f;
  clock_galaxies[2].StellarMass = 3.0f;
  clock_galaxies[2].ColdGas = 1.0f;
  clock_galaxies[3].MergTime = 999.9f;
  setup_context(&context, &clock_halos[0], 1);
  TEST_ASSERT(sage_initialise_merger_clock_process(&context, clock_halos, 4) == 0,
              "Merger-clock reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=merger_clock CentralMergTime=%.17g "
         "SatelliteMergTime=%.17g OrphanMergTime=%.17g Type3MergTime=%.17g\n",
         (double)clock_galaxies[0].MergTime, (double)clock_galaxies[1].MergTime,
         (double)clock_galaxies[2].MergTime, (double)clock_galaxies[3].MergTime);

  struct Halo event_halos[3];
  struct GalaxyData event_galaxies[3];
  setup_halo(&event_halos[0], &event_galaxies[0], 100.0, 0.2, 300.0, 0.1);
  setup_halo(&event_halos[1], &event_galaxies[1], 2.0, 0.1, 100.0, 0.1);
  setup_halo(&event_halos[2], &event_galaxies[2], 1.0, 0.1, 100.0, 0.1);
  event_halos[0].Len = 1000;
  event_halos[0].Vmax = 300.0f;
  event_halos[1].Type = 1;
  event_halos[1].CentralHalo = 0;
  event_halos[1].Len = 20;
  event_halos[1].Vmax = 100.0f;
  event_halos[2].Type = 1;
  event_halos[2].CentralHalo = 0;
  event_halos[2].Len = 20;
  event_halos[2].Vmax = 100.0f;

  event_galaxies[0].MergTime = 999.9f;
  event_galaxies[0].ColdGas = 10.0f;
  event_galaxies[0].HotGas = 5.0f;
  event_galaxies[0].EjectedGas = 1.0f;
  event_galaxies[0].StellarMass = 10.0f;
  event_galaxies[0].BulgeMass = 2.0f;
  event_galaxies[0].ICS = 0.5f;
  event_galaxies[0].BlackHoleMass = 0.01f;
  event_galaxies[0].MetalsColdGas = 0.2f;
  event_galaxies[0].MetalsHotGas = 0.1f;
  event_galaxies[0].MetalsEjectedGas = 0.02f;
  event_galaxies[0].MetalsStellarMass = 0.2f;
  event_galaxies[0].MetalsBulgeMass = 0.04f;
  event_galaxies[0].MetalsICS = 0.01f;
  event_galaxies[0].DiskScaleRadius = 0.0001f;
  event_galaxies[0].TimeOfLastMajorMerger = -1.0f;
  event_galaxies[0].TimeOfLastMinorMerger = -1.0f;

  event_galaxies[1].MergTime = 0.4f;
  event_galaxies[1].ColdGas = 1.0f;
  event_galaxies[1].HotGas = 1.0f;
  event_galaxies[1].EjectedGas = 0.2f;
  event_galaxies[1].StellarMass = 1.0f;
  event_galaxies[1].BulgeMass = 0.2f;
  event_galaxies[1].ICS = 0.1f;
  event_galaxies[1].BlackHoleMass = 0.03f;
  event_galaxies[1].MetalsColdGas = 0.02f;
  event_galaxies[1].MetalsHotGas = 0.02f;
  event_galaxies[1].MetalsEjectedGas = 0.004f;
  event_galaxies[1].MetalsStellarMass = 0.02f;
  event_galaxies[1].MetalsBulgeMass = 0.004f;
  event_galaxies[1].MetalsICS = 0.002f;

  event_galaxies[2].MergTime = -0.1f;
  event_galaxies[2].ColdGas = 1.0f;
  event_galaxies[2].HotGas = 0.5f;
  event_galaxies[2].EjectedGas = 0.1f;
  event_galaxies[2].StellarMass = 2.0f;
  event_galaxies[2].BulgeMass = 0.3f;
  event_galaxies[2].ICS = 0.05f;
  event_galaxies[2].BlackHoleMass = 0.02f;
  event_galaxies[2].MetalsColdGas = 0.02f;
  event_galaxies[2].MetalsHotGas = 0.01f;
  event_galaxies[2].MetalsEjectedGas = 0.002f;
  event_galaxies[2].MetalsStellarMass = 0.04f;
  event_galaxies[2].MetalsBulgeMass = 0.006f;
  event_galaxies[2].MetalsICS = 0.001f;

  setup_context(&context, &event_halos[0], 1);
  TEST_ASSERT(sage_resolve_mergers_and_disruption_process(&context, event_halos, 3) == 0,
              "Merger/disruption reference case should succeed");

  const struct ModuleEvent controlled_merger_event = {
      .producer_module_id = MODULE_ID_SAGE_RESOLVE_MERGERS_AND_DISRUPTION,
      .event_id = SAGE_RESOLVE_MERGERS_AND_DISRUPTION_EVENT_MERGER,
      .source_index = 2,
      .target_index = 0,
      .value0 = 0.15,
      .value1 = 0.1,
  };
  context.active_event = &controlled_merger_event;
  TEST_ASSERT(sage_quasar_mode_process(&context, &event_halos[0], 1) == 0,
              "Merger quasar consumer reference case should succeed");
  TEST_ASSERT(sage_starburst_feedback_process(&context, &event_halos[0], 1) == 0,
              "Merger starburst consumer reference case should succeed");
  context.active_event = NULL;

  printf("MIMIC_JAX_REFERENCE case=merger_event CentralColdGas=%.17g CentralHotGas=%.17g "
         "CentralEjectedGas=%.17g CentralStellarMass=%.17g CentralBulgeMass=%.17g "
         "CentralICS=%.17g CentralBlackHoleMass=%.17g CentralMetalsColdGas=%.17g "
         "CentralMetalsHotGas=%.17g CentralMetalsEjectedGas=%.17g "
         "CentralMetalsStellarMass=%.17g CentralMetalsBulgeMass=%.17g "
         "CentralMetalsICS=%.17g CentralQuasarAccretion=%.17g CentralSFR=%.17g "
         "CentralOutflowRate=%.17g CentralUnstableFraction=%.17g LastMinor=%.17g "
         "LastMajor=%.17g DisruptedType=%.17g MergedType=%.17g "
         "DisruptedClock=%.17g MergedClock=%.17g\n",
         (double)event_galaxies[0].ColdGas, (double)event_galaxies[0].HotGas,
         (double)event_galaxies[0].EjectedGas, (double)event_galaxies[0].StellarMass,
         (double)event_galaxies[0].BulgeMass, (double)event_galaxies[0].ICS,
         (double)event_galaxies[0].BlackHoleMass, (double)event_galaxies[0].MetalsColdGas,
         (double)event_galaxies[0].MetalsHotGas, (double)event_galaxies[0].MetalsEjectedGas,
         (double)event_galaxies[0].MetalsStellarMass, (double)event_galaxies[0].MetalsBulgeMass,
         (double)event_galaxies[0].MetalsICS, (double)event_galaxies[0].QuasarModeBHaccretionMass,
         (double)event_galaxies[0].StarFormationRate,
         (double)event_galaxies[0].SupernovaOutflowRate, event_galaxies[0].UnstableDiskGasFraction,
         (double)event_galaxies[0].TimeOfLastMinorMerger,
         (double)event_galaxies[0].TimeOfLastMajorMerger, (double)event_halos[1].Type,
         (double)event_halos[2].Type, (double)event_galaxies[1].MergTime,
         (double)event_galaxies[2].MergTime);

  struct Halo inheritance_source;
  struct GalaxyData inheritance_source_galaxy;
  struct Halo inheritance_workspace[1];
  setup_halo(&inheritance_source, &inheritance_source_galaxy, 100.0, 1.0, 200.0, 1.0);
  inheritance_source.SnapNum = 4;
  inheritance_source.Type = 0;
  inheritance_source.CentralHalo = -1;
  inheritance_source.HaloNr = 7;
  inheritance_source.UniqueGalaxyID = 4444;
  inheritance_source.UniqueCentralGalaxyID = 3333;
  inheritance_source.Len = 500;
  inheritance_source.deltaMvir = 4.0;
  inheritance_source.infallMvir = -1.0;
  inheritance_source.infallVvir = -1.0;
  inheritance_source.infallVmax = -1.0;
  inheritance_source.Pos[0] = 1.0f;
  inheritance_source.Pos[1] = 2.0f;
  inheritance_source.Pos[2] = 3.0f;
  inheritance_source.Vel[0] = 2.0f;
  inheritance_source.Vel[1] = 3.0f;
  inheritance_source.Vel[2] = 4.0f;
  inheritance_source.Spin[0] = 3.0f;
  inheritance_source.Spin[1] = 4.0f;
  inheritance_source.Spin[2] = 5.0f;
  inheritance_source.Vmax = 210.0f;
  inheritance_source.VelDisp = 90.0f;
  inheritance_source.MostBoundID = 100;
  inheritance_source_galaxy.ColdGas = 4.0f;
  inheritance_source_galaxy.StarFormationRate = 6.0f;
  inheritance_source_galaxy.Rheat = 0.2f;
  inheritance_source_galaxy.MergTime = 2.0f;
  inheritance_source_galaxy.InfallingGas = 9.0;

  struct InheritanceDescendant inheritance_descendant;
  memset(&inheritance_descendant, 0, sizeof(inheritance_descendant));
  inheritance_descendant.halo_nr = 42;
  inheritance_descendant.current_snap = 5;
  inheritance_descendant.current_time = 10.0;
  inheritance_descendant.new_halo_dt = 2.5;
  inheritance_descendant.virial_mass = 150.0;
  inheritance_descendant.virial_radius = 1.5;
  inheritance_descendant.virial_velocity = 250.0;
  inheritance_descendant.is_fof_central = 1;
  inheritance_descendant.unique_galaxy_id = 111000222;
  inheritance_descendant.halo_payload.SnapNum = 5;
  inheritance_descendant.halo_payload.Len = 1234;
  inheritance_descendant.halo_payload.Mvir = 150.0;
  inheritance_descendant.halo_payload.Rvir = 1.5;
  inheritance_descendant.halo_payload.Vvir = 250.0;
  inheritance_descendant.halo_payload.Pos[0] = 10.0f;
  inheritance_descendant.halo_payload.Pos[1] = 11.0f;
  inheritance_descendant.halo_payload.Pos[2] = 12.0f;
  inheritance_descendant.halo_payload.Vel[0] = 20.0f;
  inheritance_descendant.halo_payload.Vel[1] = 21.0f;
  inheritance_descendant.halo_payload.Vel[2] = 22.0f;
  inheritance_descendant.halo_payload.Spin[0] = 0.1f;
  inheritance_descendant.halo_payload.Spin[1] = 0.11f;
  inheritance_descendant.halo_payload.Spin[2] = 0.12f;
  inheritance_descendant.halo_payload.Vmax = 300.0f;
  inheritance_descendant.halo_payload.VelDisp = 120.0f;
  inheritance_descendant.halo_payload.MostBoundID = 987654321;

  const struct InheritanceProgenitorGalaxy inheritance_progenitor = {
      .source = &inheritance_source,
      .source_time = 14.0,
      .is_main_branch = 1,
  };
  memset(inheritance_workspace, 0, sizeof(inheritance_workspace));
  TEST_ASSERT(inherit_descendant_halos(inheritance_workspace, 0, 1, &inheritance_descendant,
                                       &inheritance_progenitor, 1) == 1,
              "Inheritance reference case should retain one main-branch galaxy");
  printf("MIMIC_JAX_REFERENCE case=inheritance ColdGas=%.17g StarFormationRate=%.17g "
         "Rheat=%.17g MergTime=%.17g InfallingGas=%.17g Type=%.17g SnapNum=%.17g "
         "HaloNr=%.17g CentralHalo=%.17g dT=%.17g Len=%.17g Mvir=%.17g "
         "deltaMvir=%.17g Rvir=%.17g Vvir=%.17g infallMvir=%.17g infallVvir=%.17g "
         "infallVmax=%.17g Vmax=%.17g Pos0=%.17g Spin2=%.17g MostBoundID=%.17g\n",
         (double)inheritance_workspace[0].galaxy->ColdGas,
         (double)inheritance_workspace[0].galaxy->StarFormationRate,
         (double)inheritance_workspace[0].galaxy->Rheat,
         (double)inheritance_workspace[0].galaxy->MergTime,
         inheritance_workspace[0].galaxy->InfallingGas, (double)inheritance_workspace[0].Type,
         (double)inheritance_workspace[0].SnapNum, (double)inheritance_workspace[0].HaloNr,
         (double)inheritance_workspace[0].CentralHalo, inheritance_workspace[0].dT,
         (double)inheritance_workspace[0].Len, inheritance_workspace[0].Mvir,
         inheritance_workspace[0].deltaMvir, inheritance_workspace[0].Rvir,
         inheritance_workspace[0].Vvir, inheritance_workspace[0].infallMvir,
         inheritance_workspace[0].infallVvir, inheritance_workspace[0].infallVmax,
         (double)inheritance_workspace[0].Vmax, (double)inheritance_workspace[0].Pos[0],
         (double)inheritance_workspace[0].Spin[2], (double)inheritance_workspace[0].MostBoundID);

  struct Halo group_halos[3];
  struct GalaxyData group_galaxies[3];
  setup_halo(&group_halos[0], &group_galaxies[0], 100.0, 0.2, 200.0, 0.001);
  setup_halo(&group_halos[1], &group_galaxies[1], 20.0, 0.1, 100.0, 0.002);
  setup_halo(&group_halos[2], &group_galaxies[2], 0.0, 0.0, 0.0, 0.001);
  group_halos[0].CentralHalo = 0;
  group_halos[0].Len = 1000;
  group_halos[0].Vmax = 200.0f;
  group_halos[0].Spin[0] = 1.0f;
  group_halos[0].Spin[1] = 2.0f;
  group_halos[0].Spin[2] = 3.0f;
  group_halos[1].Type = 1;
  group_halos[1].CentralHalo = 0;
  group_halos[1].Len = 200;
  group_halos[1].Vmax = 100.0f;
  group_halos[2].Type = 3;
  group_halos[2].CentralHalo = 0;
  group_halos[2].Len = 0;
  group_halos[2].Vmax = 0.0f;

  group_galaxies[0].HotGas = 8.0f;
  group_galaxies[0].ColdGas = 3.0f;
  group_galaxies[0].StellarMass = 5.0f;
  group_galaxies[0].EjectedGas = 1.0f;
  group_galaxies[0].ICS = 0.5f;
  group_galaxies[0].BlackHoleMass = 0.01f;
  group_galaxies[0].MetalsHotGas = 0.16f;
  group_galaxies[0].MetalsColdGas = 0.06f;
  group_galaxies[0].MetalsStellarMass = 0.1f;
  group_galaxies[0].MetalsEjectedGas = 0.02f;
  group_galaxies[0].MetalsICS = 0.01f;
  group_galaxies[0].MergTime = 999.9f;

  group_galaxies[1].HotGas = 3.0f;
  group_galaxies[1].ColdGas = 1.5f;
  group_galaxies[1].StellarMass = 1.0f;
  group_galaxies[1].EjectedGas = 0.2f;
  group_galaxies[1].ICS = 0.1f;
  group_galaxies[1].BlackHoleMass = 0.005f;
  group_galaxies[1].MetalsHotGas = 0.06f;
  group_galaxies[1].MetalsColdGas = 0.03f;
  group_galaxies[1].MetalsStellarMass = 0.02f;
  group_galaxies[1].MetalsEjectedGas = 0.004f;
  group_galaxies[1].MetalsICS = 0.002f;
  group_galaxies[1].DiskScaleRadius = 0.005f;
  group_galaxies[1].MergTime = 10.0f;

  group_galaxies[2].ColdGas = 7.0f;
  group_galaxies[2].HotGas = 11.0f;
  group_galaxies[2].MergTime = 4.0f;

  setup_context(&context, &group_halos[0], 2);
  context.redshift = 0.5;
  context.time_interval = 0.002;
  context.substep_dt = context.time_interval / context.num_substeps;
  TEST_ASSERT(sage_reionization_process(&context, group_halos, 3) == 0,
              "Group reionization should succeed");
  TEST_ASSERT(sage_prepare_infall_budget_process(&context, group_halos, 3) == 0,
              "Group infall-budget preparation should succeed");
  TEST_ASSERT(sage_set_disk_scale_radius_process(&context, group_halos, 3) == 0,
              "Group disk-radius setup should succeed");
  TEST_ASSERT(sage_initialise_merger_clock_process(&context, group_halos, 3) == 0,
              "Group merger-clock setup should succeed");

  for (int step = 0; step < context.num_substeps; step++) {
    context.substep_number = step;
    context.substep_time =
        (context.time + context.time_interval) - (step + 0.5) * context.substep_dt;
    TEST_ASSERT(sage_apply_infall_process(&context, group_halos, 3) == 0,
                "Group infall application should succeed");
    TEST_ASSERT(sage_reincorporation_process(&context, group_halos, 3) == 0,
                "Group reincorporation should succeed");

    for (int galaxy_index = 0; galaxy_index < 3; galaxy_index++) {
      if (group_halos[galaxy_index].Type == 3) {
        continue;
      }
      struct Halo *member = &group_halos[galaxy_index];
      TEST_ASSERT(sage_satellite_stripping_process(&context, member, 1) == 0,
                  "Group satellite stripping should succeed");
      TEST_ASSERT(sage_calculate_cooling_budget_process(&context, member, 1) == 0,
                  "Group cooling budget should succeed");
      TEST_ASSERT(sage_radio_mode_heating_process(&context, member, 1) == 0,
                  "Group radio mode should succeed");
      TEST_ASSERT(sage_apply_cooling_process(&context, member, 1) == 0,
                  "Group cooling application should succeed");
      TEST_ASSERT(sage_calculate_star_formation_process(&context, member, 1) == 0,
                  "Group star formation should succeed");
      TEST_ASSERT(sage_calculate_supernova_feedback_process(&context, member, 1) == 0,
                  "Group supernova budget should succeed");
      TEST_ASSERT(sage_apply_star_formation_supernova_process(&context, member, 1) == 0,
                  "Group star formation application should succeed");
      TEST_ASSERT(sage_disk_instability_process(&context, member, 1) == 0,
                  "Group disk instability should succeed");
      TEST_ASSERT(sage_quasar_mode_process(&context, member, 1) == 0,
                  "Group quasar mode should succeed");
      TEST_ASSERT(sage_starburst_feedback_process(&context, member, 1) == 0,
                  "Group starburst should succeed");
      TEST_ASSERT(sage_apply_metal_enrichment_process(&context, member, 1) == 0,
                  "Group metal enrichment should succeed");
    }
    TEST_ASSERT(sage_resolve_mergers_and_disruption_process(&context, group_halos, 3) == 0,
                "Group merger resolution should succeed");
    if (step == 0) {
      printf("MIMIC_JAX_REFERENCE case=group_interval_step0 CentralMetalsHotGas=%.17g "
             "SatelliteMetalsHotGas=%.17g\n",
             (double)group_galaxies[0].MetalsHotGas, (double)group_galaxies[1].MetalsHotGas);
    }
  }

  printf("MIMIC_JAX_REFERENCE case=group_interval "
         "CentralHaloBaryonFraction=%.17g CentralInfallingGas=%.17g "
         "CentralColdGas=%.17g CentralHotGas=%.17g CentralEjectedGas=%.17g "
         "CentralStellarMass=%.17g CentralBulgeMass=%.17g CentralICS=%.17g "
         "CentralBlackHoleMass=%.17g CentralMetalsColdGas=%.17g "
         "CentralMetalsHotGas=%.17g CentralMetalsEjectedGas=%.17g "
         "CentralMetalsStellarMass=%.17g CentralMetalsBulgeMass=%.17g "
         "CentralMetalsICS=%.17g CentralSFR=%.17g CentralOutflowRate=%.17g "
         "CentralDiskScaleRadius=%.17g CentralMergTime=%.17g "
         "SatelliteColdGas=%.17g SatelliteHotGas=%.17g SatelliteEjectedGas=%.17g "
         "SatelliteStellarMass=%.17g SatelliteBulgeMass=%.17g SatelliteICS=%.17g "
         "SatelliteBlackHoleMass=%.17g SatelliteMetalsColdGas=%.17g "
         "SatelliteMetalsHotGas=%.17g SatelliteMetalsEjectedGas=%.17g "
         "SatelliteMetalsStellarMass=%.17g SatelliteMetalsBulgeMass=%.17g "
         "SatelliteMetalsICS=%.17g SatelliteSFR=%.17g SatelliteOutflowRate=%.17g "
         "SatelliteDiskScaleRadius=%.17g SatelliteMergTime=%.17g "
         "CentralType=%.17g SatelliteType=%.17g ConsumedType=%.17g\n",
         group_galaxies[0].HaloBaryonFraction, group_galaxies[0].InfallingGas,
         (double)group_galaxies[0].ColdGas, (double)group_galaxies[0].HotGas,
         (double)group_galaxies[0].EjectedGas, (double)group_galaxies[0].StellarMass,
         (double)group_galaxies[0].BulgeMass, (double)group_galaxies[0].ICS,
         (double)group_galaxies[0].BlackHoleMass, (double)group_galaxies[0].MetalsColdGas,
         (double)group_galaxies[0].MetalsHotGas, (double)group_galaxies[0].MetalsEjectedGas,
         (double)group_galaxies[0].MetalsStellarMass, (double)group_galaxies[0].MetalsBulgeMass,
         (double)group_galaxies[0].MetalsICS, (double)group_galaxies[0].StarFormationRate,
         (double)group_galaxies[0].SupernovaOutflowRate, (double)group_galaxies[0].DiskScaleRadius,
         (double)group_galaxies[0].MergTime, (double)group_galaxies[1].ColdGas,
         (double)group_galaxies[1].HotGas, (double)group_galaxies[1].EjectedGas,
         (double)group_galaxies[1].StellarMass, (double)group_galaxies[1].BulgeMass,
         (double)group_galaxies[1].ICS, (double)group_galaxies[1].BlackHoleMass,
         (double)group_galaxies[1].MetalsColdGas, (double)group_galaxies[1].MetalsHotGas,
         (double)group_galaxies[1].MetalsEjectedGas, (double)group_galaxies[1].MetalsStellarMass,
         (double)group_galaxies[1].MetalsBulgeMass, (double)group_galaxies[1].MetalsICS,
         (double)group_galaxies[1].StarFormationRate,
         (double)group_galaxies[1].SupernovaOutflowRate, (double)group_galaxies[1].DiskScaleRadius,
         (double)group_galaxies[1].MergTime, (double)group_halos[0].Type,
         (double)group_halos[1].Type, (double)group_halos[2].Type);

  struct Halo infall_halos[2];
  struct GalaxyData infall_galaxies[2];
  setup_halo(&infall_halos[0], &infall_galaxies[0], 100.0, 0.2, 200.0, 0.01);
  setup_halo(&infall_halos[1], &infall_galaxies[1], 0.0, 0.1, 100.0, 0.01);
  infall_halos[1].Type = 2;
  setup_context(&context, &infall_halos[0], 10);
  infall_galaxies[0].HaloBaryonFraction = 0.17;
  infall_galaxies[0].StellarMass = 5.0f;
  infall_galaxies[0].ColdGas = 3.0f;
  infall_galaxies[0].HotGas = 8.0f;
  infall_galaxies[0].EjectedGas = 1.0f;
  infall_galaxies[0].ICS = 0.5f;
  infall_galaxies[0].BlackHoleMass = 0.1f;
  infall_galaxies[0].MetalsEjectedGas = 0.02f;
  infall_galaxies[0].MetalsICS = 0.01f;
  infall_galaxies[1].HaloBaryonFraction = 0.17;
  infall_galaxies[1].HotGas = 3.0f;
  infall_galaxies[1].EjectedGas = 2.0f;
  infall_galaxies[1].ICS = 1.5f;
  infall_galaxies[1].MetalsHotGas = 0.06f;
  infall_galaxies[1].MetalsEjectedGas = 0.04f;
  infall_galaxies[1].MetalsICS = 0.03f;
  TEST_ASSERT(sage_prepare_infall_budget_process(&context, infall_halos, 2) == 0,
              "Infall-budget reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=infall_budget InfallingGas=%.17g EjectedGas=%.17g "
         "MetalsEjectedGas=%.17g ICS=%.17g MetalsICS=%.17g SatelliteEjectedGas=%.17g "
         "SatelliteICS=%.17g SatelliteHotGas=%.17g\n",
         infall_galaxies[0].InfallingGas, (double)infall_galaxies[0].EjectedGas,
         (double)infall_galaxies[0].MetalsEjectedGas, (double)infall_galaxies[0].ICS,
         (double)infall_galaxies[0].MetalsICS, (double)infall_galaxies[1].EjectedGas,
         (double)infall_galaxies[1].ICS, (double)infall_galaxies[1].HotGas);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 4);
  galaxy.InfallingGas = 12.0;
  galaxy.HotGas = 5.0f;
  TEST_ASSERT(sage_apply_infall_process(&context, &halo, 1) == 0,
              "Positive-infall reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=infall_positive HotGas=%.17g\n", (double)galaxy.HotGas);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 1);
  galaxy.InfallingGas = -8.0;
  galaxy.EjectedGas = 3.0f;
  galaxy.MetalsEjectedGas = 0.06f;
  galaxy.HotGas = 10.0f;
  galaxy.MetalsHotGas = 0.2f;
  TEST_ASSERT(sage_apply_infall_process(&context, &halo, 1) == 0,
              "Negative-infall reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=infall_negative EjectedGas=%.17g MetalsEjectedGas=%.17g "
         "HotGas=%.17g MetalsHotGas=%.17g\n",
         (double)galaxy.EjectedGas, (double)galaxy.MetalsEjectedGas, (double)galaxy.HotGas,
         (double)galaxy.MetalsHotGas);

  struct Halo stripping_halos[2];
  struct GalaxyData stripping_galaxies[2];
  setup_halo(&stripping_halos[0], &stripping_galaxies[0], 100.0, 0.2, 200.0, 0.01);
  setup_halo(&stripping_halos[1], &stripping_galaxies[1], 10.0, 0.1, 100.0, 0.01);
  stripping_halos[1].Type = 1;
  setup_context(&context, &stripping_halos[0], 10);
  stripping_galaxies[0].HotGas = 100.0f;
  stripping_galaxies[0].MetalsHotGas = 2.0f;
  stripping_galaxies[1].HaloBaryonFraction = 0.17f;
  stripping_galaxies[1].StellarMass = 0.4f;
  stripping_galaxies[1].ColdGas = 0.3f;
  stripping_galaxies[1].HotGas = 5.0f;
  stripping_galaxies[1].EjectedGas = 0.2f;
  stripping_galaxies[1].BlackHoleMass = 0.05f;
  stripping_galaxies[1].ICS = 0.1f;
  stripping_galaxies[1].MetalsHotGas = 0.1f;
  TEST_ASSERT(sage_satellite_stripping_process(&context, &stripping_halos[1], 1) == 0,
              "Satellite-stripping reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=satellite_stripping SatelliteHotGas=%.17g "
         "SatelliteMetalsHotGas=%.17g CentralHotGas=%.17g CentralMetalsHotGas=%.17g\n",
         (double)stripping_galaxies[1].HotGas, (double)stripping_galaxies[1].MetalsHotGas,
         (double)stripping_galaxies[0].HotGas, (double)stripping_galaxies[0].MetalsHotGas);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 10);
  galaxy.HotGas = 8.0f;
  galaxy.MetalsHotGas = 0.16f;
  TEST_ASSERT(sage_calculate_cooling_budget_process(&context, &halo, 1) == 0,
              "Cooling-budget reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=cooling_budget CoolingGas=%.17g Rcool=%.17g "
         "CoolingLambda=%.17g\n",
         galaxy.CoolingGas, galaxy.Rcool, galaxy.CoolingLambda);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 10);
  galaxy.HotGas = 8.0f;
  galaxy.MetalsHotGas = 0.16f;
  galaxy.BlackHoleMass = 0.01f;
  galaxy.Rheat = 0.01f;
  TEST_ASSERT(sage_calculate_cooling_budget_process(&context, &halo, 1) == 0,
              "Radio-mode cooling-budget reference case should succeed");
  const double cooling_before_radio_mode = galaxy.CoolingGas;
  TEST_ASSERT(sage_radio_mode_heating_process(&context, &halo, 1) == 0,
              "Radio-mode reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=radio_mode CoolingGasBefore=%.17g CoolingGas=%.17g "
         "BlackHoleMass=%.17g HotGas=%.17g MetalsHotGas=%.17g Rheat=%.17g Heating=%.17g\n",
         cooling_before_radio_mode, galaxy.CoolingGas, (double)galaxy.BlackHoleMass,
         (double)galaxy.HotGas, (double)galaxy.MetalsHotGas, (double)galaxy.Rheat, galaxy.Heating);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.01);
  setup_context(&context, &halo, 1);
  galaxy.ColdGas = 2.0f;
  galaxy.HotGas = 8.0f;
  galaxy.MetalsColdGas = 0.04f;
  galaxy.MetalsHotGas = 0.16f;
  galaxy.CoolingGas = 1.5;
  TEST_ASSERT(sage_apply_cooling_process(&context, &halo, 1) == 0,
              "Cooling reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=cooling ColdGas=%.17g HotGas=%.17g "
         "MetalsColdGas=%.17g MetalsHotGas=%.17g Cooling=%.17g\n",
         (double)galaxy.ColdGas, (double)galaxy.HotGas, (double)galaxy.MetalsColdGas,
         (double)galaxy.MetalsHotGas, galaxy.Cooling);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 100.0, 0.01);
  setup_context(&context, &halo, 10);
  galaxy.HotGas = 2.0f;
  galaxy.EjectedGas = 4.0f;
  galaxy.MetalsHotGas = 0.04f;
  galaxy.MetalsEjectedGas = 0.08f;
  TEST_ASSERT(sage_reincorporation_process(&context, &halo, 1) == 0,
              "Reincorporation reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=reincorporation HotGas=%.17g EjectedGas=%.17g "
         "MetalsHotGas=%.17g MetalsEjectedGas=%.17g\n",
         (double)galaxy.HotGas, (double)galaxy.EjectedGas, (double)galaxy.MetalsHotGas,
         (double)galaxy.MetalsEjectedGas);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 200.0, 0.1);
  setup_context(&context, &halo, 10);
  halo.Vmax = 200.0f;
  galaxy.ColdGas = 5.0f;
  galaxy.StellarMass = 10.0f;
  galaxy.BulgeMass = 2.0f;
  galaxy.MetalsStellarMass = 0.2f;
  galaxy.MetalsBulgeMass = 0.04f;
  galaxy.DiskScaleRadius = 0.003f;
  TEST_ASSERT(sage_disk_instability_process(&context, &halo, 1) == 0,
              "Disk-instability reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=disk_instability BulgeMass=%.17g "
         "MetalsBulgeMass=%.17g UnstableDiskGasFraction=%.17g\n",
         (double)galaxy.BulgeMass, (double)galaxy.MetalsBulgeMass, galaxy.UnstableDiskGasFraction);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 300.0, 0.1);
  setup_context(&context, &halo, 10);
  galaxy.ColdGas = 10.0f;
  galaxy.HotGas = 5.0f;
  galaxy.EjectedGas = 1.0f;
  galaxy.MetalsColdGas = 0.2f;
  galaxy.MetalsHotGas = 0.1f;
  galaxy.MetalsEjectedGas = 0.02f;
  galaxy.BlackHoleMass = 0.01f;
  galaxy.UnstableDiskGasFraction = 0.5;
  TEST_ASSERT(sage_quasar_mode_process(&context, &halo, 1) == 0,
              "Quasar-mode reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=quasar_mode ColdGas=%.17g HotGas=%.17g "
         "EjectedGas=%.17g MetalsColdGas=%.17g MetalsHotGas=%.17g "
         "MetalsEjectedGas=%.17g BlackHoleMass=%.17g "
         "QuasarModeBHaccretionMass=%.17g UnstableDiskGasFraction=%.17g\n",
         (double)galaxy.ColdGas, (double)galaxy.HotGas, (double)galaxy.EjectedGas,
         (double)galaxy.MetalsColdGas, (double)galaxy.MetalsHotGas, (double)galaxy.MetalsEjectedGas,
         (double)galaxy.BlackHoleMass, (double)galaxy.QuasarModeBHaccretionMass,
         galaxy.UnstableDiskGasFraction);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 300.0, 0.1);
  setup_context(&context, &halo, 10);
  galaxy.ColdGas = 10.0f;
  galaxy.HotGas = 5.0f;
  galaxy.EjectedGas = 1.0f;
  galaxy.StellarMass = 5.0f;
  galaxy.BulgeMass = 1.0f;
  galaxy.MetalsColdGas = 0.2f;
  galaxy.MetalsHotGas = 0.1f;
  galaxy.MetalsEjectedGas = 0.02f;
  galaxy.MetalsStellarMass = 0.1f;
  galaxy.MetalsBulgeMass = 0.02f;
  galaxy.UnstableDiskGasFraction = 0.2;
  TEST_ASSERT(sage_starburst_feedback_process(&context, &halo, 1) == 0,
              "Starburst reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=starburst ColdGas=%.17g HotGas=%.17g EjectedGas=%.17g "
         "StellarMass=%.17g BulgeMass=%.17g MetalsColdGas=%.17g MetalsHotGas=%.17g "
         "MetalsEjectedGas=%.17g MetalsStellarMass=%.17g MetalsBulgeMass=%.17g "
         "StarFormationRate=%.17g SupernovaOutflowRate=%.17g "
         "UnstableDiskGasFraction=%.17g\n",
         (double)galaxy.ColdGas, (double)galaxy.HotGas, (double)galaxy.EjectedGas,
         (double)galaxy.StellarMass, (double)galaxy.BulgeMass, (double)galaxy.MetalsColdGas,
         (double)galaxy.MetalsHotGas, (double)galaxy.MetalsEjectedGas,
         (double)galaxy.MetalsStellarMass, (double)galaxy.MetalsBulgeMass,
         (double)galaxy.StarFormationRate, (double)galaxy.SupernovaOutflowRate,
         galaxy.UnstableDiskGasFraction);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 150.0, 0.0001);
  setup_context(&context, &halo, 1);
  galaxy.ColdGas = 10.0f;
  galaxy.HotGas = 5.0f;
  galaxy.EjectedGas = 1.0f;
  galaxy.StellarMass = 2.0f;
  galaxy.MetalsColdGas = 0.2f;
  galaxy.MetalsHotGas = 0.1f;
  galaxy.MetalsEjectedGas = 0.01f;
  galaxy.MetalsStellarMass = 0.04f;
  galaxy.DiskScaleRadius = 0.01f;
  TEST_ASSERT(sage_calculate_star_formation_process(&context, &halo, 1) == 0,
              "SF reference case should succeed");
  TEST_ASSERT(sage_calculate_supernova_feedback_process(&context, &halo, 1) == 0,
              "SN reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=star_formation_budget NewStellarMass=%.17g "
         "SupernovaReheatedMass=%.17g SupernovaEjectedMass=%.17g\n",
         galaxy.NewStellarMass, galaxy.SupernovaReheatedMass, galaxy.SupernovaEjectedMass);
  TEST_ASSERT(sage_apply_star_formation_supernova_process(&context, &halo, 1) == 0,
              "SF/SN apply reference case should succeed");
  TEST_ASSERT(sage_apply_metal_enrichment_process(&context, &halo, 1) == 0,
              "Metal enrichment reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=star_formation_final ColdGas=%.17g HotGas=%.17g "
         "EjectedGas=%.17g StellarMass=%.17g MetalsColdGas=%.17g MetalsHotGas=%.17g "
         "MetalsEjectedGas=%.17g MetalsStellarMass=%.17g StarFormationRate=%.17g "
         "SupernovaOutflowRate=%.17g NewStellarMass=%.17g\n",
         (double)galaxy.ColdGas, (double)galaxy.HotGas, (double)galaxy.EjectedGas,
         (double)galaxy.StellarMass, (double)galaxy.MetalsColdGas, (double)galaxy.MetalsHotGas,
         (double)galaxy.MetalsEjectedGas, (double)galaxy.MetalsStellarMass,
         (double)galaxy.StarFormationRate, (double)galaxy.SupernovaOutflowRate,
         galaxy.NewStellarMass);

  setup_halo(&halo, &galaxy, 100.0, 0.2, 150.0, 0.0001);
  setup_context(&context, &halo, 1);
  halo.Vmax = 150.0f;
  galaxy.ColdGas = 10.0f;
  galaxy.HotGas = 5.0f;
  galaxy.EjectedGas = 1.0f;
  galaxy.StellarMass = 2.0f;
  galaxy.BulgeMass = 0.5f;
  galaxy.MetalsColdGas = 0.2f;
  galaxy.MetalsHotGas = 0.1f;
  galaxy.MetalsEjectedGas = 0.01f;
  galaxy.MetalsStellarMass = 0.04f;
  galaxy.MetalsBulgeMass = 0.01f;
  galaxy.DiskScaleRadius = 0.0001f;
  TEST_ASSERT(sage_calculate_star_formation_process(&context, &halo, 1) == 0,
              "Composed SF reference case should succeed");
  TEST_ASSERT(sage_calculate_supernova_feedback_process(&context, &halo, 1) == 0,
              "Composed SN reference case should succeed");
  TEST_ASSERT(sage_apply_star_formation_supernova_process(&context, &halo, 1) == 0,
              "Composed SF/SN apply reference case should succeed");
  TEST_ASSERT(sage_disk_instability_process(&context, &halo, 1) == 0,
              "Composed disk-instability reference case should succeed");
  TEST_ASSERT(sage_quasar_mode_process(&context, &halo, 1) == 0,
              "Composed quasar-mode reference case should succeed");
  TEST_ASSERT(sage_starburst_feedback_process(&context, &halo, 1) == 0,
              "Composed starburst reference case should succeed");
  TEST_ASSERT(sage_apply_metal_enrichment_process(&context, &halo, 1) == 0,
              "Composed metal-enrichment reference case should succeed");
  printf("MIMIC_JAX_REFERENCE case=post_quiescent_chain ColdGas=%.17g HotGas=%.17g "
         "EjectedGas=%.17g StellarMass=%.17g BulgeMass=%.17g BlackHoleMass=%.17g "
         "MetalsColdGas=%.17g MetalsHotGas=%.17g MetalsEjectedGas=%.17g "
         "MetalsStellarMass=%.17g MetalsBulgeMass=%.17g NewStellarMass=%.17g "
         "UnstableDiskGasFraction=%.17g StarFormationRate=%.17g "
         "SupernovaOutflowRate=%.17g QuasarModeBHaccretionMass=%.17g\n",
         (double)galaxy.ColdGas, (double)galaxy.HotGas, (double)galaxy.EjectedGas,
         (double)galaxy.StellarMass, (double)galaxy.BulgeMass, (double)galaxy.BlackHoleMass,
         (double)galaxy.MetalsColdGas, (double)galaxy.MetalsHotGas, (double)galaxy.MetalsEjectedGas,
         (double)galaxy.MetalsStellarMass, (double)galaxy.MetalsBulgeMass, galaxy.NewStellarMass,
         galaxy.UnstableDiskGasFraction, (double)galaxy.StarFormationRate,
         (double)galaxy.SupernovaOutflowRate, (double)galaxy.QuasarModeBHaccretionMass);

  sage_apply_metal_enrichment_cleanup();
  sage_resolve_mergers_and_disruption_cleanup();
  sage_initialise_merger_clock_cleanup();
  sage_set_disk_scale_radius_cleanup();
  sage_starburst_feedback_cleanup();
  sage_quasar_mode_cleanup();
  sage_disk_instability_cleanup();
  sage_apply_star_formation_supernova_cleanup();
  sage_calculate_supernova_feedback_cleanup();
  sage_calculate_star_formation_cleanup();
  sage_reincorporation_cleanup();
  sage_satellite_stripping_cleanup();
  sage_apply_infall_cleanup();
  sage_prepare_infall_budget_cleanup();
  sage_reionization_cleanup();
  sage_radio_mode_heating_cleanup();
  sage_apply_cooling_cleanup();
  sage_calculate_cooling_budget_cleanup();
  test_free_substep_phases();
  test_free_pre_timestep();
  galaxy_pool_destroy();
  check_memory_leaks();
  return TEST_PASS;
}

int main(void) {
  initialize_error_handling(LOG_LEVEL_WARNING, NULL);
  TEST_RUN(test_emit_reference_cases);
  TEST_SUMMARY();
  return TEST_RESULT();
}
