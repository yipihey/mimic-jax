#!/usr/bin/env python3
"""Compile SAGE16 C reference cases and compare their outputs with mimic-jax."""

import argparse
import math
import os
import subprocess
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mimic_jax.sage16 import (  # noqa: E402
    apply_cooling,
    apply_metal_enrichment,
    apply_reincorporation,
    apply_star_formation_supernova,
    calculate_cooling_budget,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    metal_dependent_cooling_rate,
    sage16_units,
    step_context,
)

REFERENCE_PREFIX = "MIMIC_JAX_REFERENCE "
REFERENCE_TEST = "models/sage16/modules/_tests/test_unit_mimic_jax_reference.c"
REFERENCE_LOG = "tests/unit/build/test_unit_mimic_jax_reference.run.log"
CASE_TOLERANCES = {
    "cooling_budget": (1.0e-13, 0.0),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-c-log",
        action="store_true",
        help="Reuse the last compiled C test log instead of rebuilding the reference executable",
    )
    return parser.parse_args()


def run_c_reference(repository: Path) -> None:
    environment = os.environ.copy()
    environment["TEST_SUMMARY"] = "1"
    subprocess.run(
        ["tests/unit/run_tests.sh", REFERENCE_TEST],
        cwd=repository,
        env=environment,
        check=True,
    )


def parse_c_reference(path: Path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = line.find(REFERENCE_PREFIX)
        if marker < 0:
            continue
        tokens = line[marker + len(REFERENCE_PREFIX) :].split()
        values = dict(token.split("=", 1) for token in tokens)
        case = values.pop("case")
        records[case] = {name: float(value) for name, value in values.items()}
    required = {
        "cooling",
        "cooling_budget",
        "cooling_interpolation",
        "reincorporation",
        "star_formation_budget",
        "star_formation_final",
    }
    missing = required - set(records)
    if missing:
        raise RuntimeError(f"Missing C reference records: {sorted(missing)}")
    return records


def select(record, names):
    return {name: float(getattr(record, name)) for name in names}


def calculate_jax_reference():
    parameters = fiducial_parameters()
    units = sage16_units()
    cooling_tables = load_cooling_tables()

    log_z_sun = math.log10(0.02)
    interpolation = {
        "midpoint": float(metal_dependent_cooling_rate(5.025, log_z_sun - 0.75, cooling_tables)),
        "low_temperature": float(metal_dependent_cooling_rate(3.0, log_z_sun, cooling_tables)),
        "high_temperature": float(metal_dependent_cooling_rate(9.0, log_z_sun, cooling_tables)),
        "primordial": float(metal_dependent_cooling_rate(5.5, -10.0, cooling_tables)),
    }

    cooling_budget_state = initial_galaxy_state(HotGas=8.0, MetalsHotGas=0.16)
    cooling_budget_halo = initial_halo_forcing(Rvir=0.2, Vvir=200.0, dT=0.01)
    cooling_budget = calculate_cooling_budget(
        cooling_budget_state,
        cooling_budget_halo,
        step_context(num_substeps=10, time_interval=0.01),
        units,
        cooling_tables,
    ).state

    cooling_state = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=8.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.16,
        CoolingGas=1.5,
    )
    cooling_halo = initial_halo_forcing(Vvir=200.0, dT=0.01)
    cooled = apply_cooling(cooling_state, cooling_halo).state

    reincorporation_state = initial_galaxy_state(
        HotGas=2.0,
        EjectedGas=4.0,
        MetalsHotGas=0.04,
        MetalsEjectedGas=0.08,
    )
    reincorporation_halo = initial_halo_forcing(
        Vvir=100.0,
        Rvir=0.2,
        dT=0.01,
    )
    reincorporated = apply_reincorporation(
        reincorporation_state,
        reincorporation_halo,
        step_context(num_substeps=10, time_interval=0.01),
        parameters,
    ).state

    star_formation_state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=2.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.01,
        MetalsStellarMass=0.04,
        DiskScaleRadius=0.01,
    )
    star_formation_halo = initial_halo_forcing(
        Mvir=100.0,
        Rvir=0.2,
        Vvir=150.0,
        dT=0.0001,
    )
    context = step_context(time_interval=0.0001)
    star_formation = calculate_star_formation_budget(
        star_formation_state,
        star_formation_halo,
        context,
        parameters,
    )
    budget = calculate_supernova_feedback_budget(
        star_formation_state,
        star_formation_halo,
        parameters,
        units,
        star_formation,
    )
    applied = apply_star_formation_supernova(
        star_formation_state,
        star_formation_state,
        star_formation_halo,
        parameters,
        budget,
    )
    enriched = apply_metal_enrichment(
        applied.galaxy,
        applied.central,
        star_formation_halo,
        True,
        parameters,
    ).galaxy

    return {
        "cooling_interpolation": interpolation,
        "cooling_budget": select(
            cooling_budget,
            ("CoolingGas", "Rcool", "CoolingLambda"),
        ),
        "cooling": select(
            cooled,
            ("ColdGas", "HotGas", "MetalsColdGas", "MetalsHotGas", "Cooling"),
        ),
        "reincorporation": select(
            reincorporated,
            ("HotGas", "EjectedGas", "MetalsHotGas", "MetalsEjectedGas"),
        ),
        "star_formation_budget": select(
            budget,
            ("NewStellarMass", "SupernovaReheatedMass", "SupernovaEjectedMass"),
        ),
        "star_formation_final": select(
            enriched,
            (
                "ColdGas",
                "HotGas",
                "EjectedGas",
                "StellarMass",
                "MetalsColdGas",
                "MetalsHotGas",
                "MetalsEjectedGas",
                "MetalsStellarMass",
                "StarFormationRate",
                "SupernovaOutflowRate",
                "NewStellarMass",
            ),
        ),
    }


def compare_records(c_reference, jax_reference) -> None:
    discrepancies = []
    compared = 0
    exact = 0
    for case, c_values in c_reference.items():
        jax_values = jax_reference[case]
        if set(c_values) != set(jax_values):
            discrepancies.append(
                f"{case}: field mismatch C={sorted(c_values)} JAX={sorted(jax_values)}"
            )
            continue
        for name, c_value in c_values.items():
            compared += 1
            jax_value = jax_values[name]
            relative_tolerance, absolute_tolerance = CASE_TOLERANCES.get(case, (0.0, 0.0))
            if jax_value == c_value:
                exact += 1
            elif not math.isclose(
                jax_value,
                c_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                discrepancies.append(
                    f"{case}.{name}: C={c_value:.17g}, JAX={jax_value:.17g}, "
                    f"abs_diff={abs(jax_value - c_value):.3e}, "
                    f"rtol={relative_tolerance:.1e}, atol={absolute_tolerance:.1e}"
                )
    if discrepancies:
        raise AssertionError("C/JAX equivalence failed:\n" + "\n".join(discrepancies))
    print(f"MIMIC-JAX equivalence: {compared} fields match compiled SAGE16 " f"({exact} exactly)")
    print("Tolerances: cooling budget rtol=1e-13; every other controlled field exact")


def main() -> int:
    arguments = parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    if not arguments.reuse_c_log:
        run_c_reference(repository)
    c_reference = parse_c_reference(repository / REFERENCE_LOG)
    compare_records(c_reference, calculate_jax_reference())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
