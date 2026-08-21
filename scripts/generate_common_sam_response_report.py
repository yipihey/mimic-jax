#!/usr/bin/env python3
"""Generate the first controlled common-response report for SAGE16 and SHARK."""

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mimic_jax import (
    ResponseCoordinate,
    Sage16ContinuousForcing,
    SharkContinuousForcing,
    characteristic_modes,
    frequency_response,
    load_model,
    scale_state_space,
    state_space_in_gyr,
    step_response,
)
from mimic_jax.numerics import integrate_fixed_step
from mimic_jax.reporting import (
    Artifact,
    ComparedRun,
    ComparisonMetric,
    ComparisonReport,
    Diagnostic,
    DiagnosticStatus,
    ReportLink,
    ReportSection,
    ScalarMetric,
    capture_provenance,
    write_report,
)
from mimic_jax.sage16 import initial_galaxy_state, initial_halo_forcing, ode_state_from_galaxy
from mimic_jax.shark import initial_shark_state, lagos23_disk_forcing

MASS_UNIT = 1.0e10
DISK_SCALE_RADIUS = 0.005
VIRIAL_VELOCITY = 220.0
HALF_MASS_FACTOR = 1.678346990
SECONDS_PER_GYR = 3.15576e16
MODEL_COLORS = {"sage16": "#3567a9", "shark": "#d05a3a"}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/sage16-shark-response-foundation"),
    )
    parser.add_argument("--validation-duration-gyr", type=float, default=0.05)
    parser.add_argument("--validation-steps", type=int, default=100)
    return parser.parse_args()


def _cases():
    sage = load_model("sage16")
    galaxy = initial_galaxy_state(
        ColdGas=1.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.02,
        MetalsHotGas=0.10,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        DiskScaleRadius=DISK_SCALE_RADIUS,
    )
    sage_state = ode_state_from_galaxy(galaxy)
    sage_forcing = Sage16ContinuousForcing(
        initial_halo_forcing(
            Mvir=100.0,
            Rvir=0.2,
            Vvir=VIRIAL_VELOCITY,
            dT=1.0e-3,
        ),
        galaxy.DiskScaleRadius,
    )
    sage_result = sage.rhs_and_rates(0.0, sage_state, sage_forcing)
    sage_time_gyr = sage.metadata.time_unit_in_gyr
    cooling_rate = float(sage_result.rates.cooling) * MASS_UNIT / sage_time_gyr
    reincorporation_rate = float(sage_result.rates.reincorporation) * MASS_UNIT / sage_time_gyr

    shark = load_model("shark")
    specific_angular_momentum = 2.0 * VIRIAL_VELOCITY * DISK_SCALE_RADIUS
    shark_state = initial_shark_state(
        stellar_mass=1.0 * MASS_UNIT,
        cold_gas=1.0 * MASS_UNIT,
        hot_halo_gas=5.0 * MASS_UNIT,
        ejected_gas=1.0 * MASS_UNIT,
        stellar_metals=0.02 * MASS_UNIT,
        cold_gas_metals=0.02 * MASS_UNIT,
        hot_halo_gas_metals=0.10 * MASS_UNIT,
        ejected_gas_metals=0.02 * MASS_UNIT,
        stellar_angular_momentum=1.0 * MASS_UNIT * specific_angular_momentum,
        cold_gas_angular_momentum=1.0 * MASS_UNIT * specific_angular_momentum,
        hot_halo_angular_momentum=5.0 * MASS_UNIT * specific_angular_momentum,
        ejected_angular_momentum=1.0 * MASS_UNIT * specific_angular_momentum,
    )
    shark_forcing = SharkContinuousForcing(
        lagos23_disk_forcing(
            gas_half_mass_radius=DISK_SCALE_RADIUS * HALF_MASS_FACTOR,
            stellar_half_mass_radius=DISK_SCALE_RADIUS * HALF_MASS_FACTOR,
            redshift=0.0,
            galaxy_velocity=VIRIAL_VELOCITY,
            subhalo_velocity=VIRIAL_VELOCITY,
            cooling_rate=cooling_rate,
            cooling_metallicity=0.02,
            cooling_specific_angular_momentum=specific_angular_momentum,
        ),
        jnp.asarray(reincorporation_rate),
    )
    return {
        "sage16": (sage, sage_state, sage_forcing),
        "shark": (shark, shark_state, shark_forcing),
    }


def _state_scales(model_name):
    if model_name == "sage16":
        return np.asarray([1.0, 5.0, 1.0, 1.0, 0.02, 0.10, 0.02, 0.02])
    specific_angular_momentum = 2.0 * VIRIAL_VELOCITY * DISK_SCALE_RADIUS
    return np.asarray(
        [MASS_UNIT] * 6
        + [0.02 * MASS_UNIT] * 6
        + [MASS_UNIT] * 2
        + [MASS_UNIT * specific_angular_momentum] * 5
    )


def _responses(cases):
    periods = np.geomspace(0.03, 100.0, 180)
    angular_frequencies = 2.0 * np.pi / periods
    responses = {}
    for name, (model, state, forcing) in cases.items():
        result = model.rhs_and_rates(0.0, state, forcing)
        baseline_sfr = result.rates.star_formation

        def fractional_sfr(
            current,
            current_forcing,
            parameters,
            controls,
            model=model,
            baseline_sfr=baseline_sfr,
        ):
            current_result = model.rhs_and_rates(
                0.0, current, current_forcing, parameters, controls
            )
            return jnp.asarray([current_result.rates.star_formation / baseline_sfr])

        response = model.local_response(
            time=0.0,
            state=state,
            forcing=forcing,
            output=fractional_sfr,
            output_coordinates=(
                ResponseCoordinate(
                    "fractional_sfr",
                    "fractional star-formation rate",
                    "dimensionless",
                    "SFR divided by its value at the matched operating point.",
                ),
            ),
            redshift=0.0,
            halo_mass=1.0e12,
            halo_mass_unit="Msun/h",
            qualification="Matched local disk experiment; not a population comparison.",
        )
        response = scale_state_space(state_space_in_gyr(response), _state_scales(name))
        transfer = np.asarray(frequency_response(response, angular_frequencies))[:, 0, :]
        modes = characteristic_modes(response)
        stable_times = np.sort(
            np.asarray(modes.response_times_gyr)[np.asarray(modes.stable, dtype=bool)]
        )
        responses[name] = {
            "model": model,
            "state": state,
            "forcing": forcing,
            "result": result,
            "baseline_sfr": float(baseline_sfr),
            "space": response,
            "transfer": transfer,
            "stable_times": stable_times,
        }
    return periods, responses


def _native_sfr_per_gyr(name, response):
    if name == "sage16":
        return response["baseline_sfr"] * MASS_UNIT / response["model"].metadata.time_unit_in_gyr
    return response["baseline_sfr"]


def _validate_linear_response(responses, duration_gyr, steps):
    epsilon_values = np.asarray([1.0e-3, 1.0e-2, 5.0e-2])
    validation = {}
    times_gyr = np.linspace(0.0, duration_gyr, steps + 1)
    for name, values in responses.items():
        model = values["model"]
        state = values["state"]
        forcing = values["forcing"]
        baseline_sfr = values["baseline_sfr"]
        cooling_index = model.metadata.process_control_names.index("cooling")
        linear_unit_step = np.asarray(step_response(values["space"], times_gyr))[
            :, 0, cooling_index
        ]
        duration_native = duration_gyr / model.metadata.time_unit_in_gyr
        baseline = integrate_fixed_step(
            lambda time, current, model=model, forcing=forcing: model.rhs(time, current, forcing),
            state,
            duration=duration_native,
            num_steps=steps,
            method="rk4",
        )

        def sfr_history(
            states,
            controls,
            model=model,
            forcing=forcing,
            baseline_sfr=baseline_sfr,
        ):
            return jax.vmap(
                lambda current: model.rhs_and_rates(
                    0.0, current, forcing, None, controls
                ).rates.star_formation
                / baseline_sfr
            )(states)

        baseline_history = np.asarray(
            sfr_history(
                baseline.states,
                jnp.zeros(len(model.metadata.process_control_names)),
            )
        )
        nonlinear = []
        linear = []
        errors = []
        for epsilon in epsilon_values:
            controls = np.zeros(len(model.metadata.process_control_names))
            controls[cooling_index] = epsilon
            fixed_controls = jnp.asarray(controls)
            perturbed = integrate_fixed_step(
                lambda time, current, model=model, forcing=forcing, controls=fixed_controls: model.rhs(
                    time, current, forcing, None, controls
                ),
                state,
                duration=duration_native,
                num_steps=steps,
                method="rk4",
            )
            nonlinear_history = (
                np.asarray(sfr_history(perturbed.states, fixed_controls)) - baseline_history
            )
            linear_history = epsilon * linear_unit_step
            denominator = max(abs(float(nonlinear_history[-1])), 1.0e-15)
            errors.append(abs(float(nonlinear_history[-1] - linear_history[-1])) / denominator)
            nonlinear.append(nonlinear_history)
            linear.append(linear_history)
        validation[name] = {
            "times_gyr": times_gyr,
            "epsilon": epsilon_values,
            "nonlinear": np.asarray(nonlinear),
            "linear": np.asarray(linear),
            "relative_final_error": np.asarray(errors),
        }
    return validation


def _plot_reservoirs(path, responses):
    labels = ("stars", "cold", "hot", "ejected")
    values = {
        "sage16": np.asarray([1.0, 1.0, 5.0, 1.0]),
        "shark": np.asarray([1.0, 1.0, 5.0, 1.0]),
    }
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for index, name in enumerate(("sage16", "shark")):
        axis.bar(
            x + (index - 0.5) * 0.34,
            values[name],
            width=0.34,
            color=MODEL_COLORS[name],
            label=responses[name]["model"].metadata.label,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"Reservoir mass [$10^{10}\,M_\odot/h$]")
    axis.set_title("Matched resolved baryon inventory")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_response(path, periods, responses, validation):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    for name, values in responses.items():
        cooling_index = values["model"].metadata.process_control_names.index("cooling")
        axes[0].loglog(
            periods,
            np.abs(values["transfer"][:, cooling_index]),
            color=MODEL_COLORS[name],
            linewidth=2.2,
            label=values["model"].metadata.label,
        )
        sample = validation[name]
        epsilon_index = 1
        axes[1].plot(
            sample["times_gyr"],
            100.0 * sample["nonlinear"][epsilon_index],
            color=MODEL_COLORS[name],
            linewidth=2.2,
            label=f"{values['model'].metadata.label}: nonlinear",
        )
        axes[1].plot(
            sample["times_gyr"],
            100.0 * sample["linear"][epsilon_index],
            color=MODEL_COLORS[name],
            linestyle="--",
            linewidth=1.6,
            label=f"{values['model'].metadata.label}: local response",
        )
    axes[0].set_xlabel("Period of cooling-supply variation [Gyr]")
    axes[0].set_ylabel("Fractional SFR response / fractional cooling change")
    axes[0].set_title("Which cooling variations reach the SFR?")
    axes[0].axhline(1.0, color="0.55", linewidth=0.8)
    axes[1].set_xlabel("Time after a sustained 1% cooling increase [Gyr]")
    axes[1].set_ylabel("Change in SFR [% of initial SFR]")
    axes[1].set_title("Local prediction versus nonlinear evolution")
    axes[1].axhline(0.0, color="0.55", linewidth=0.8)
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_modes(path, responses):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True, sharex=True)
    for axis, name in zip(axes, ("sage16", "shark")):
        times = responses[name]["stable_times"]
        axis.barh(np.arange(times.size), times, color=MODEL_COLORS[name])
        axis.set_xscale("log")
        axis.set_xlim(0.25, 40.0)
        axis.set_xlabel("Local damping time [Gyr]")
        axis.set_title("SAGE16" if name == "sage16" else "SHARK Lagos23")
        axis.set_yticks(np.arange(times.size), [f"mode {index + 1}" for index in range(times.size)])
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Stable coupled reservoir mode (fast to slow)")
    figure.suptitle("Several coupled baryon-cycle memory times emerge")
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_process_gains(path, responses, period_gyr=1.0):
    common = ("cooling", "star_formation", "sn_reheating", "sn_ejection", "reincorporation")
    figure, axis = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    x = np.arange(len(common))
    for index, name in enumerate(("sage16", "shark")):
        model = responses[name]["model"]
        transfer = np.asarray(
            frequency_response(responses[name]["space"], [2.0 * np.pi / period_gyr])
        )[0, 0]
        gains = [
            abs(transfer[model.metadata.process_control_names.index(process)]) for process in common
        ]
        axis.bar(
            x + (index - 0.5) * 0.34,
            gains,
            width=0.34,
            color=MODEL_COLORS[name],
            label="SAGE16" if name == "sage16" else "SHARK Lagos23",
        )
    axis.set_xticks(
        x,
        ("cooling", "star\nformation", "SN\nreheating", "SN\nejection", "reincorporation"),
    )
    axis.set_ylabel("Fractional SFR response at a 1 Gyr variation period")
    axis.set_title("SFR response to one-Gyr process variations")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _artifact(key, title, filename, description):
    return Artifact(key, title, f"assets/{filename}", "image/svg+xml", "figure", description)


def main():
    args = _arguments()
    repository = Path(__file__).resolve().parents[1]
    output = args.output
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    periods, responses = _responses(cases)
    validation = _validate_linear_response(
        responses, args.validation_duration_gyr, args.validation_steps
    )

    reservoir_figure = _artifact(
        "matched-reservoirs",
        "Matched local reservoir inventory",
        "matched-reservoirs.svg",
        "The same four principal mass reservoirs initialize both local calculations; model-specific trackers and angular momentum remain separate.",
    )
    response_figure = _artifact(
        "cooling-response",
        "Cooling-to-SFR dynamical response",
        "cooling-response.svg",
        "Frequency response uses variation period 2π/ω. Dashed curves are frozen-coefficient predictions; solid curves evolve each nonlinear RHS.",
    )
    modes_figure = _artifact(
        "response-times",
        "Stable local response times",
        "response-times.svg",
        "Finite stable eigenmode damping times after converting both models to Gyr and applying an input-output-invariant state scaling.",
    )
    process_figure = _artifact(
        "process-gains",
        "Matched process-response amplitudes",
        "process-gains.svg",
        "Absolute fractional SFR response to a fractional process perturbation varying with a one-Gyr period; signs and phases remain in the array product.",
    )
    _plot_reservoirs(assets / "matched-reservoirs.svg", responses)
    _plot_response(assets / "cooling-response.svg", periods, responses, validation)
    _plot_modes(assets / "response-times.svg", responses)
    _plot_process_gains(assets / "process-gains.svg", responses)

    np.savez_compressed(
        assets / "local-response-arrays.npz",
        variation_period_gyr=periods,
        sage_transfer=responses["sage16"]["transfer"],
        shark_transfer=responses["shark"]["transfer"],
        sage_process_names=np.asarray(responses["sage16"]["model"].metadata.process_control_names),
        shark_process_names=np.asarray(responses["shark"]["model"].metadata.process_control_names),
        sage_stable_response_times_gyr=responses["sage16"]["stable_times"],
        shark_stable_response_times_gyr=responses["shark"]["stable_times"],
        sage_validation_nonlinear=validation["sage16"]["nonlinear"],
        sage_validation_linear=validation["sage16"]["linear"],
        shark_validation_nonlinear=validation["shark"]["nonlinear"],
        shark_validation_linear=validation["shark"]["linear"],
        validation_epsilon=validation["sage16"]["epsilon"],
        validation_time_gyr=validation["sage16"]["times_gyr"],
    )
    balance_residuals = {
        name: max(
            abs(float(balance.residual))
            for balance in values["model"].conservation_balances(values["result"])
        )
        for name, values in responses.items()
    }
    maximum_linear_error = max(
        float(np.max(values["relative_final_error"])) for values in validation.values()
    )
    summary = {
        "experiment": {
            "halo_mass_msun_over_h": 1.0e12,
            "redshift": 0.0,
            "virial_velocity_km_per_s": VIRIAL_VELOCITY,
            "disk_scale_radius_mpc_over_h": DISK_SCALE_RADIUS,
            "qualification": "matched local continuous-flow experiment, not a population comparison",
        },
        "models": {
            name: {
                "label": values["model"].metadata.label,
                "formulation": values["model"].metadata.formulation,
                "baseline_sfr_msun_over_h_per_gyr": _native_sfr_per_gyr(name, values),
                "stable_response_times_gyr": values["stable_times"].tolist(),
                "linear_validation_relative_final_error": validation[name][
                    "relative_final_error"
                ].tolist(),
                "maximum_conservation_residual_native_rate": balance_residuals[name],
            }
            for name, values in responses.items()
        },
    }
    (assets / "local-response-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (assets / "model-protocols.json").write_text(
        json.dumps(
            {name: values["model"].metadata.to_dict() for name, values in responses.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    array_artifact = Artifact(
        "local-response-arrays",
        "Local response arrays",
        "assets/local-response-arrays.npz",
        "application/x-npz",
        "data",
        "Complex signed transfer arrays, modes, and nonlinear validation histories.",
    )
    summary_artifact = Artifact(
        "local-response-summary",
        "Machine-readable local response summary",
        "assets/local-response-summary.json",
        "application/json",
        "data",
        "Operating point, formulation qualifications, response times, and validation errors.",
    )
    protocol_artifact = Artifact(
        "model-protocols",
        "Machine-readable model protocols",
        "assets/model-protocols.json",
        "application/json",
        "metadata",
        "State, forcing, parameter, process, capability, formulation, and upstream-revision metadata for both configured models.",
    )
    period_index = int(np.argmin(abs(periods - 10.0)))
    cooling_gains = {}
    for name, values in responses.items():
        index = values["model"].metadata.process_control_names.index("cooling")
        cooling_gains[name] = abs(values["transfer"][period_index, index])
    conservation_status = (
        DiagnosticStatus.PASSED
        if max(balance_residuals.values()) <= 2.0e-6
        else DiagnosticStatus.FAILED
    )
    linear_status = (
        DiagnosticStatus.PASSED if maximum_linear_error <= 0.10 else DiagnosticStatus.WARNING
    )
    health = (
        Diagnostic(
            "common-protocol",
            "Common executable protocol",
            DiagnosticStatus.PASSED,
            "Both configured models expose state, forcing, parameters, named process controls, RHS/rates, conservation, Jacobians, and annotated local response through one API.",
            method="mimic_jax.sam.ConfiguredSamModel",
        ),
        Diagnostic(
            "conservation",
            "Local conservation",
            conservation_status,
            "Baryon and source-corrected metal rate residuals close at the matched operating point.",
            metrics=tuple(
                ScalarMetric(
                    f"{name}-maximum-residual",
                    f"{values['model'].metadata.label} maximum residual",
                    balance_residuals[name],
                    "native rate",
                    "Maximum absolute common-ledger residual.",
                )
                for name, values in responses.items()
            ),
            tolerance="maximum absolute residual <= 2e-6 native rate units",
        ),
        Diagnostic(
            "linear-validity",
            "Local response versus nonlinear evolution",
            linear_status,
            f"The frozen local response was tested against 0.1%, 1%, and 5% sustained cooling perturbations over {args.validation_duration_gyr:g} Gyr.",
            metrics=(
                ScalarMetric(
                    "maximum-relative-final-error",
                    "Largest final relative discrepancy",
                    maximum_linear_error,
                    "fraction",
                    "Across both models and all three perturbation sizes.",
                ),
            ),
            artifacts=(response_figure,),
            method="JAX local Jacobian versus nonlinear RK4 evolution",
            tolerance="maximum final relative discrepancy <= 0.10",
        ),
        Diagnostic(
            "population-isolation",
            "Population-level model isolation",
            DiagnosticStatus.NOT_EVALUATED,
            "The experiment matches one local baryon inventory and forcing boundary; it does not yet run both full models on the same merger-tree population.",
        ),
        Diagnostic(
            "agn-closed-loop",
            "Matched AGN regulation",
            DiagnosticStatus.NOT_EVALUATED,
            "SAGE radio-mode history and SHARK hot-halo AGN/heating memory are outside the two matched continuous subsets used here. No cross-model AGN conclusion is made.",
        ),
    )
    ecosystem_body = """\
| Package/model | What it already does well | mimic-jax decision |
|---|---|---|
| Sapphire | Purpose-built JAX regulator, Diffrax integration, optimization, Fisher/HMC and accelerator execution | Interoperate and reuse generic ecosystem tools; compare scientific response where states and forcing can be matched |
| Diffmah | Differentiable halo mass-assembly histories | Optional controlled forcing adapter; retain real merger trees as reference |
| Diffstar / DiffstarPop | Differentiable SFH and population distributions | Complementary emulator/comparison layer, not a SAM port |
| DSPS | Differentiable SFH/metallicity-to-SED/photometry mapping | Optional output adapter with SPS/filter provenance |
| Galacticus | Mature adaptive ODE plus tree-event architecture | Numerical/architectural reference and independent analytic tests; no copied implementation |
| SAGE16 / SHARK | Established production SAM physics and familiar observables | Preserve upstream equivalence, expose common flows/events/conservation/responses |
"""
    report = ComparisonReport(
        comparison_id="sage16-shark-response-foundation",
        title="Do SAGE16 and SHARK regulate the same matched disk in the same way?",
        summary=(
            "A deliberately controlled first response experiment. SAGE16 and SHARK start with "
            "the same principal baryon inventory, disk scale, velocity, cooling supply, and "
            "metal fractions. Their familiar instantaneous SFRs are close here, but their "
            "coupled responses need not be. This is the foundation for—not a substitute for—"
            "the same-tree population comparison."
        ),
        baseline=ComparedRun(
            "sage16",
            "MIMIC/SAGE16 continuous central subset",
            "sage16-common-local-response",
        ),
        candidate=ComparedRun(
            "shark",
            "SHARK Lagos23 controlled disk subset",
            "shark-common-local-response",
        ),
        metrics=(
            ComparisonMetric.from_values(
                key="instantaneous-sfr",
                label="Instantaneous SFR at the matched state",
                baseline=_native_sfr_per_gyr("sage16", responses["sage16"]),
                candidate=_native_sfr_per_gyr("shark", responses["shark"]),
                unit="Msun/h/Gyr",
                interpretation="The near agreement is a useful operating point, not a calibration or population-equivalence result.",
            ),
            ComparisonMetric.from_values(
                key="cooling-sfr-gain-10gyr",
                label="Cooling-to-SFR response at a 10 Gyr period",
                baseline=float(cooling_gains["sage16"]),
                candidate=float(cooling_gains["shark"]),
                unit="fraction/fraction",
                interpretation="This asks how strongly a slow fractional cooling variation appears in fractional SFR near the same local state.",
            ),
            ComparisonMetric.from_values(
                key="slowest-stable-mode",
                label="Slowest finite stable local mode",
                baseline=float(responses["sage16"]["stable_times"][-1]),
                candidate=float(responses["shark"]["stable_times"][-1]),
                unit="Gyr",
                interpretation="This is the longest damping time among locally stable modes; neutral tracker modes are excluded.",
            ),
        ),
        provenance=capture_provenance(
            repository=repository,
            command=(
                "scripts/generate_common_sam_response_report.py",
                "--output",
                output.as_posix(),
                "--validation-duration-gyr",
                str(args.validation_duration_gyr),
                "--validation-steps",
                str(args.validation_steps),
            ),
            configuration_paths=(repository / "docs/dev/MIMIC-JAX-COMMON-SAM-PROTOCOL-PLAN.md",),
        ),
        health=health,
        sections=(
            ReportSection(
                "matched-question",
                "What exactly is being compared?",
                "One controlled local experiment removes several avoidable convention differences before asking a dynamical question.",
                artifacts=(reservoir_figure,),
                body=(
                    r"Both calculations use $M_\star=M_\mathrm{cold}=M_\mathrm{eject}=10^{10}\,M_\odot/h$, "
                    r"$M_\mathrm{hot}=5\times10^{10}\,M_\odot/h$, a 5 kpc/$h$ disk scale, "
                    "$V=220$ km/s, and $z=0$. The SHARK half-mass radius is the matching exponential-disk conversion. "
                    "The SAGE cooling and reincorporation rates are converted from the MIMIC code time unit to Gyr and used as SHARK's prepared external supply. "
                    "This aligns the local question while retaining each model's own star-formation and feedback prescriptions."
                ),
            ),
            ReportSection(
                "response",
                "Does SFR remember the same cooling perturbation in the same way?",
                "No global linearity is assumed. JAX differentiates each nonlinear RHS at the matched state, and the resulting local prediction is checked by evolving the nonlinear model.",
                artifacts=(response_figure, process_figure, array_artifact),
                body=(
                    "The left panel treats cooling variability with period $T=2\\pi/\\omega$ and shows "
                    f"$|H(i\\omega)|$ for fractional SFR per fractional cooling change. The right panel gives the direct physical experiment: keep cooling 1% higher and watch the SFR over the next {args.validation_duration_gyr:g} Gyr. "
                    "Agreement with the nonlinear trajectories is the validity test; the comparison does not rely on the matrix calculation alone. "
                    "The process-gain panel must be read with the declared closure: SHARK cooling is prepared external forcing in this controlled subset, so reincorporated hot gas has no return path to SFR here. Its zero reincorporation gain is therefore a boundary limitation, not yet a statement about full SHARK. SAGE SN ejection is also zero at this operating point because that branch is inactive."
                ),
            ),
            ReportSection(
                "memory",
                "How many memory times does each baryon cycle contain?",
                "The coupled reservoirs generate several stable damping times rather than one recipe timescale.",
                artifacts=(modes_figure,),
                body=(
                    "These are eigenmode e-folding times of the local state Jacobian after converting both native time conventions to Gyr. "
                    "Exact neutral tracker/integrated-mass modes are excluded. Mode composition is not labelled yet because mixed mass, metal, and angular-momentum coordinates require a separately declared nondimensional participation convention."
                ),
            ),
            ReportSection(
                "limits",
                "What does this first comparison not establish?",
                "The experiment is deliberately narrower than the full scientific claim.",
                body=(
                    "It does not show that full SAGE and SHARK populations differ for model-physics reasons; that requires the same merger-tree forcing, aligned selections, and both hybrid event schedulers. "
                    "It also does not compare the closed AGN loops: SAGE's radio-mode heating history and SHARK's hot-halo BH/heating-radius state must be matched without reducing either to an externally prescribed loading. "
                    "Those remain explicit not-evaluated gates rather than blank panels or inferred conclusions."
                ),
            ),
            ReportSection(
                "ecosystem",
                "How does this fit the differentiable-galaxy ecosystem?",
                "The discovery of Sapphire narrows the engineering target: reuse generic differentiable tooling and focus mimic-jax on faithful established-model representation and comparison.",
                body=ecosystem_body,
                artifacts=(protocol_artifact,),
            ),
            ReportSection(
                "next-gate",
                "What is the next decisive experiment?",
                "Run the two complete hybrid models on common halo histories, then compare regulation rather than only outputs.",
                artifacts=(summary_artifact,),
                body=(
                    "The next science report should pair common population observables with baryon flows, parameter elasticities, historical process responses, cooling-to-SFR transfer, and AGN regulation. "
                    "A result will be called a model-physics difference only after tree semantics, cosmology, forcing interpolation, numerical error, and selection differences are controlled."
                ),
            ),
        ),
        links=(
            ReportLink(
                "SAGE–SHARK interoperability audit",
                "../sage16-shark-interoperability-audit/index.md",
            ),
            ReportLink("SAGE response report", "../sage16-linear-response/index.md"),
            ReportLink(
                "Common protocol plan", "../../docs/dev/MIMIC-JAX-COMMON-SAM-PROTOCOL-PLAN.md"
            ),
            ReportLink("Sapphire", "https://github.com/virajpandya/sapphire"),
        ),
    )
    write_report(report, output)


if __name__ == "__main__":
    main()
