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
#include "include/globals.h"
#include "include/proto.h"
#include "include/types.h"
#include "../sage_calculate_cooling_budget/cooling_tables.h"
#include "util/error.h"
#include "util/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "modules/_tests/sage_test_fixtures.h"

extern int sage_apply_cooling_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_calculate_cooling_budget_init(void);
extern int sage_calculate_cooling_budget_process(struct ModuleContext *ctx, struct Halo *halos,
                                                 int ngal);
extern int sage_calculate_cooling_budget_cleanup(void);
extern int sage_reincorporation_init(void);
extern int sage_reincorporation_process(struct ModuleContext *ctx, struct Halo *halos, int ngal);
extern int sage_reincorporation_cleanup(void);
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

  add_parameter("SfrEfficiency", "0.05");
  add_parameter("StarFormingDiskFactor", "3.0");
  add_parameter("FeedbackReheatingEpsilon", "3.0");
  add_parameter("FeedbackEjectionEfficiency", "0.3");
  add_parameter("ReIncorporationFactor", "0.15");
  add_parameter("RecycleFraction", "0.43");
  add_parameter("Yield", "0.025");
  add_parameter("FracZleaveDisk", "0.0");

  test_phase_add("galaxy_physics", "sage_calculate_cooling_budget", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_cooling", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_reincorporation", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_calculate_star_formation", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_calculate_supernova_feedback", PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_star_formation_supernova",
                 PROCESSING_MODE_BY_GALAXY);
  test_phase_add("galaxy_physics", "sage_apply_metal_enrichment", PROCESSING_MODE_BY_GALAXY);
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
  TEST_ASSERT(sage_reincorporation_init() == 0, "Reincorporation init should succeed");
  TEST_ASSERT(sage_calculate_star_formation_init() == 0, "SF init should succeed");
  TEST_ASSERT(sage_calculate_supernova_feedback_init() == 0, "SN init should succeed");
  TEST_ASSERT(sage_apply_star_formation_supernova_init() == 0, "SF/SN apply init should succeed");
  TEST_ASSERT(sage_apply_metal_enrichment_init() == 0, "Metal enrichment init should succeed");

  const double log_z_sun = log10(0.02);
  printf("MIMIC_JAX_REFERENCE case=cooling_interpolation midpoint=%.17g low_temperature=%.17g "
         "high_temperature=%.17g primordial=%.17g\n",
         get_metaldependent_cooling_rate(5.025, log_z_sun - 0.75),
         get_metaldependent_cooling_rate(3.0, log_z_sun),
         get_metaldependent_cooling_rate(9.0, log_z_sun),
         get_metaldependent_cooling_rate(5.5, -10.0));

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

  sage_apply_metal_enrichment_cleanup();
  sage_apply_star_formation_supernova_cleanup();
  sage_calculate_supernova_feedback_cleanup();
  sage_calculate_star_formation_cleanup();
  sage_reincorporation_cleanup();
  sage_calculate_cooling_budget_cleanup();
  test_free_substep_phases();
  check_memory_leaks();
  return TEST_PASS;
}

int main(void) {
  initialize_error_handling(LOG_LEVEL_WARNING, NULL);
  TEST_RUN(test_emit_reference_cases);
  TEST_SUMMARY();
  return TEST_RESULT();
}
