"""Sapphire native-runtime boundary and artifact-contract tests."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from mimic_jax import REFERENCE_SCALE, characteristic_modes, load_model
from mimic_jax.sapphire import (
    SAPPHIRE_ARTIFACT_SCHEMA,
    SapphireBackendUnavailableError,
    SapphireNativeArtifact,
    fiducial_sapphire_case,
    write_sapphire_artifact,
)

jax.config.update("jax_enable_x64", True)


def _manifest():
    state = [
        {"name": name, "label": name, "unit": unit, "description": "test state"}
        for name, unit in zip(
            ("M_star", "M_ism", "M_cgm", "Eth_cgm", "MZ_star", "MZ_ism", "MZ_cgm"),
            ("Msun", "Msun", "Msun", "erg", "Msun", "Msun", "Msun"),
        )
    ]
    inputs = [
        {
            "name": "Mdot_in_dm",
            "label": "dark-matter accretion",
            "unit": "fractional forcing change",
            "description": "test input",
        }
    ]
    observables = [
        {
            "name": "stellar_mass",
            "label": "stellar mass",
            "unit": "Msun",
            "description": "test output",
        }
    ]
    return {
        "schema_version": SAPPHIRE_ARTIFACT_SCHEMA,
        "model": {
            "name": "sapphire",
            "label": "Sapphire Pandya23",
            "version": "0.130",
            "revision": "ee50e858e3427de50368c32205001248849b8be0",
            "repository": "https://github.com/virajpandya/sapphire",
            "formulation": "native test fixture",
        },
        "qualification": "Synthetic matrices test the adapter, not Sapphire physics.",
        "coordinates": {
            "state": state,
            "input": inputs,
            "observable": observables,
            "parameter": [
                {
                    "name": "A_M",
                    "label": "mass loading normalization",
                    "unit": "native Sapphire coordinate",
                    "description": "test parameter",
                }
            ],
            "rate": [
                {
                    "name": "Mdot_sfr",
                    "label": "star formation",
                    "unit": "Msun/yr",
                    "description": "test rate",
                }
            ],
        },
        "linearization_point": {
            "time_gyr": 8.0,
            "redshift": 0.6,
            "halo_mass_msun": 1.0e12,
        },
        "derivatives": {"method": "synthetic fixture"},
    }


def _arrays():
    state = np.asarray([1.0e10, 2.0e9, 8.0e10, 4.0e58, 1.0e8, 2.0e7, 1.6e8])
    state_jacobian = -np.diag(np.arange(1.0, 8.0))
    return {
        "linearization_state": state,
        "state_derivative": -state * 1.0e-2,
        "state_jacobian": state_jacobian,
        "input_jacobian": np.ones((7, 1)),
        "output_jacobian": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "direct_input_jacobian": np.zeros((1, 1)),
        "observable_values": np.asarray([state[0]]),
        "parameter_values": np.asarray([0.0]),
        "parameter_output_jacobian": np.asarray([[2.0]]),
        "rate_values": np.asarray([3.0]),
    }


def test_sapphire_case_is_complete_positive_and_serializable():
    case = fiducial_sapphire_case()
    encoded = json.loads(json.dumps(case.to_dict()))
    assert encoded["schema_version"] == "mimic-jax-sapphire-case/v1"
    assert encoded["initial_state"]["Eth_cgm"] > 0.0
    assert encoded["forcing"]["Mvir"] == 1.0e12
    assert encoded["parameters"]["A_SF"] == 0.8


def test_sapphire_artifact_round_trip_exposes_common_response(tmp_path):
    written = write_sapphire_artifact(tmp_path, _manifest(), _arrays())
    loaded = SapphireNativeArtifact.load(tmp_path)
    assert written.manifest == loaded.manifest
    assert loaded.observable("stellar_mass") == 1.0e10
    assert loaded.rate("Mdot_sfr") == 3.0
    np.testing.assert_allclose(loaded.parameter_output_jacobian(), [[2.0]])
    response = loaded.local_response()
    assert response.point.model == "Sapphire Pandya23"
    assert response.point.redshift == 0.6
    assert response.derivative_method == "synthetic fixture"
    assert response.state_jacobian.shape == (7, 7)
    assert response.input_jacobian.shape == (7, 1)
    modes = characteristic_modes(response)
    assert np.all(modes.stable)
    np.testing.assert_allclose(np.sort(modes.response_times), 1.0 / np.arange(7.0, 0.0, -1.0))


def test_sapphire_artifact_refuses_checksum_tampering(tmp_path):
    write_sapphire_artifact(tmp_path, _manifest(), _arrays())
    with (tmp_path / "arrays.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        SapphireNativeArtifact.load(tmp_path)


def test_sapphire_requires_explicit_native_backend(tmp_path):
    model = load_model("sapphire")
    with pytest.raises(SapphireBackendUnavailableError, match="isolated backend"):
        model.run_local_case(output_directory=tmp_path)


def test_pinned_native_sapphire_fixture_passes_derivative_convergence_and_budget_gates():
    fixture = Path(__file__).resolve().parents[1] / "data" / "sapphire" / "native-v0.130-controlled"
    artifact = SapphireNativeArtifact.load(fixture)
    assert artifact.manifest["model"]["revision"] == ("ee50e858e3427de50368c32205001248849b8be0")
    assert artifact.manifest["model"]["version"] == "0.130"
    assert artifact.manifest["qualification"].startswith("Native Sapphire")
    local_validation = tuple(
        value
        for name, value in artifact.derivative_validation.items()
        if not name.startswith("trajectory_")
    )
    assert max(local_validation) < 1.0e-7
    assert (
        artifact.derivative_validation["trajectory_parameter_output_jacobian_relative_l2_error"]
        < 5.0e-3
    )
    steps, errors = artifact.trajectory_parameter_validation
    np.testing.assert_allclose(steps, [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5])
    assert np.all(np.isfinite(errors))
    assert np.min(errors) < 2.0e-3
    assert np.max(np.abs(artifact.convergence_fraction)) < 1.0e-4

    model = load_model("sapphire")
    balances = model.conservation_balances(artifact)
    for balance in balances:
        throughput = max(float(balance.source_rate), float(balance.sink_rate), 1.0)
        assert abs(float(balance.residual)) / throughput < 1.0e-12

    response = model.local_response(artifact=artifact)
    assert response.state_jacobian.shape == (7, 7)
    assert response.input_jacobian.shape == (7, 5)
    assert artifact.parameter_output_jacobian().shape == (6, 16)
    assert artifact.trajectory_parameter_output_jacobian().shape == (6, 16)

    normalized = model.parameter_response(
        artifact,
        parameter_names=("A_M", "A_SF"),
        observable_names=("stellar_mass", "star_formation_rate"),
        normalization=REFERENCE_SCALE,
        observable_scales=artifact.arrays["observable_values"][[0, 3]],
        parameter_scales=np.ones(2),
    )
    assert normalized.values.shape == (2, 2)
    assert normalized.model == "Sapphire Pandya23"
    assert np.all(np.isfinite(normalized.values))
