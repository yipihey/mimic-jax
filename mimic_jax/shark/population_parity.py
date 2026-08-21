"""Exhaustive native-state replay of the SHARK continuous population kernel.

The native SHARK topology driver can emit a fixed-width binary record at
every call to ``basic_physicalmodel_evaluator``.  This module re-evaluates the
nonlinear Lagos23 disk/starburst prescriptions and the complete 19-state RHS
with one compiled, batched JAX kernel.  It is deliberately a *shadow replay*:
the native run supplies the realized topology and input states, while JAX
independently evaluates the baryonic rates and transfers at every one of
those states.  It does not pretend that mimic-jax already owns SHARK's
variable-cardinality galaxy topology.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.shark.flows import (
    cold_gas_metallicity,
    effective_stellar_yield,
    shark_rhs_from_rates,
)
from mimic_jax.shark.prescriptions.agn import (
    lagos23_agn_parameters,
    lagos23_qso_outflow_loadings,
)
from mimic_jax.shark.prescriptions.disk import (
    lagos23_disk_flow_rates,
    lagos23_disk_forcing,
)
from mimic_jax.shark.prescriptions.star_formation import lagos23_star_formation_parameters
from mimic_jax.shark.prescriptions.stellar_feedback import lagos13_feedback_parameters
from mimic_jax.shark.tree import load_shark_tree
from mimic_jax.shark.types import SharkFlowRates, SharkState, shark_flow_parameters

SHARK_RHS_TRACE_MAGIC = 20260820.0
SHARK_RHS_TRACE_WIDTH = 70

_STATE_SLICE = slice(6, 25)
_RHS_SLICE = slice(25, 44)
_AUXILIARY_SLICE = slice(44, 70)

SHARK_POPULATION_RATE_NAMES = (
    "cooling",
    "star_formation",
    "star_formation_angular_momentum",
    "stellar_reheating_loading",
    "stellar_ejection_loading",
    "angular_momentum_reheating_loading",
    "angular_momentum_ejection_loading",
    "qso_reheating_loading",
    "qso_ejection_loading",
    "cooling_metallicity",
    "cooling_specific_angular_momentum",
)


@dataclass(frozen=True)
class PopulationParityMetric:
    """Streaming error summary for one vector comparison."""

    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_scaled_error: float
    failing_values: int
    outside_warning_band_values: int
    compared_values: int


@dataclass(frozen=True)
class NativeTraceNoninterference:
    """Bitwise catalogue comparison between traced and clean native runs."""

    passed: bool
    snapshots_compared: int
    datasets_compared: int
    values_compared: int
    mismatching_datasets: int
    mismatching_values: int


@dataclass(frozen=True)
class SharkPopulationParity:
    """Machine-readable result of one exhaustive native-state JAX replay."""

    method: str
    status: str
    passed: bool
    trace_path: str
    trace_sha256: str
    tree_path: str
    tree_sha256: str
    number_of_trees: int
    number_of_tree_nodes: int
    missing_descendant_links_skipped_upstream: int
    rhs_evaluations: int
    disk_rhs_evaluations: int
    starburst_rhs_evaluations: int
    unique_galaxies_evaluated: int
    first_snapshot: int
    last_snapshot: int
    strict_relative_tolerance: float
    warning_relative_tolerance: float
    absolute_tolerance: float
    strict_passed: bool
    rhs: PopulationParityMetric
    recomputed_rhs: PopulationParityMetric
    rates: PopulationParityMetric
    derived_quantities: PopulationParityMetric
    native_trace_noninterference: Optional[NativeTraceNoninterference]
    compilation_seconds: float
    elapsed_seconds: float
    jax_backend: str
    jax_version: str
    qualifications: tuple[str, ...]
    rate_maximum_relative_error: Mapping[str, float]
    rate_strict_failing_values: Mapping[str, int]
    rate_outside_warning_band_values: Mapping[str, int]

    def to_dict(self):
        return asdict(self)

    def write_json(self, path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


class _StreamingComparison:
    def __init__(self, strict_relative_tolerance, warning_relative_tolerance, absolute_tolerance):
        self.strict_relative_tolerance = float(strict_relative_tolerance)
        self.warning_relative_tolerance = float(warning_relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.maximum_absolute_error = 0.0
        self.maximum_relative_error = 0.0
        self.maximum_scaled_error = 0.0
        self.failing_values = 0
        self.outside_warning_band_values = 0
        self.compared_values = 0

    def update(self, candidate, reference):
        candidate = np.asarray(candidate, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if candidate.shape != reference.shape:
            raise ValueError("candidate and reference batches differ in shape")
        if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(reference)):
            raise ValueError("population parity inputs must be finite")
        difference = np.abs(candidate - reference)
        relative = difference / np.maximum(np.abs(reference), self.absolute_tolerance)
        strict_allowance = self.absolute_tolerance + self.strict_relative_tolerance * np.abs(
            reference
        )
        warning_allowance = self.absolute_tolerance + self.warning_relative_tolerance * np.abs(
            reference
        )
        scaled = difference / strict_allowance
        self.maximum_absolute_error = max(self.maximum_absolute_error, float(difference.max()))
        self.maximum_relative_error = max(self.maximum_relative_error, float(relative.max()))
        self.maximum_scaled_error = max(self.maximum_scaled_error, float(scaled.max()))
        self.failing_values += int(np.count_nonzero(difference > strict_allowance))
        self.outside_warning_band_values += int(np.count_nonzero(difference > warning_allowance))
        self.compared_values += int(difference.size)

    def result(self):
        return PopulationParityMetric(
            maximum_absolute_error=self.maximum_absolute_error,
            maximum_relative_error=self.maximum_relative_error,
            maximum_scaled_error=self.maximum_scaled_error,
            failing_values=self.failing_values,
            outside_warning_band_values=self.outside_warning_band_values,
            compared_values=self.compared_values,
        )


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compare_native_shark_catalogues(instrumented_root, reference_root):
    """Compare every galaxy dataset from traced and clean native output.

    The two roots may use different model names, but must each contain the
    usual ``<snapshot>/0/galaxies.hdf5`` hierarchy.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Comparing native SHARK catalogues requires h5py") from error

    def outputs(root):
        return {path.parent.parent.name: path for path in Path(root).glob("*/0/galaxies.hdf5")}

    instrumented = outputs(instrumented_root)
    reference = outputs(reference_root)
    if not instrumented or not reference:
        raise ValueError("native SHARK output roots must contain snapshot catalogues")
    snapshots_match = instrumented.keys() == reference.keys()
    datasets_compared = 0
    values_compared = 0
    mismatching_datasets = 0
    mismatching_values = 0
    for snapshot in sorted(instrumented.keys() & reference.keys(), key=int):
        with h5py.File(instrumented[snapshot], "r") as candidate_file, h5py.File(
            reference[snapshot], "r"
        ) as reference_file:
            candidate = candidate_file["galaxies"]
            expected = reference_file["galaxies"]
            candidate_names = set(candidate)
            reference_names = set(expected)
            mismatching_datasets += len(candidate_names ^ reference_names)
            for name in sorted(candidate_names & reference_names):
                candidate_values = np.asarray(candidate[name])
                reference_values = np.asarray(expected[name])
                datasets_compared += 1
                if (
                    candidate_values.shape != reference_values.shape
                    or candidate_values.dtype != reference_values.dtype
                ):
                    mismatching_datasets += 1
                    continue
                values_compared += int(candidate_values.size)
                candidate_bytes = np.ascontiguousarray(candidate_values).view(np.uint8)
                reference_bytes = np.ascontiguousarray(reference_values).view(np.uint8)
                unequal_values = np.any(
                    candidate_bytes.reshape(candidate_values.size, candidate_values.dtype.itemsize)
                    != reference_bytes.reshape(
                        reference_values.size, reference_values.dtype.itemsize
                    ),
                    axis=1,
                )
                mismatch_count = int(np.count_nonzero(unequal_values))
                mismatching_values += mismatch_count
                mismatching_datasets += int(mismatch_count > 0)
    passed = snapshots_match and not mismatching_datasets and not mismatching_values
    return NativeTraceNoninterference(
        passed=passed,
        snapshots_compared=len(instrumented.keys() & reference.keys()),
        datasets_compared=datasets_compared,
        values_compared=values_compared,
        mismatching_datasets=mismatching_datasets,
        mismatching_values=mismatching_values,
    )


def load_shark_rhs_trace(path, *, validation_batch_size=65_536):
    """Memory-map and validate an opt-in native SHARK population trace."""

    if not isinstance(validation_batch_size, int) or validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be a positive Python integer")
    path = Path(path)
    byte_count = path.stat().st_size
    record_bytes = SHARK_RHS_TRACE_WIDTH * np.dtype("<f8").itemsize
    if byte_count == 0 or byte_count % record_bytes:
        raise ValueError("SHARK RHS trace has an invalid byte length")
    trace = np.memmap(path, dtype="<f8", mode="r").reshape(-1, SHARK_RHS_TRACE_WIDTH)
    for start in range(0, len(trace), validation_batch_size):
        records = np.asarray(trace[start : start + validation_batch_size])
        if not np.all(records[:, 0] == SHARK_RHS_TRACE_MAGIC):
            raise ValueError("SHARK RHS trace magic/version does not match this reader")
        if np.any(~np.isfinite(records)):
            raise ValueError("SHARK RHS trace contains non-finite values")
    return trace


def _population_replay_kernel():
    flow_parameters = shark_flow_parameters()
    star_formation_parameters = lagos23_star_formation_parameters()
    feedback_parameters = lagos13_feedback_parameters()
    agn_parameters = lagos23_agn_parameters()

    def evaluate_one(state_values, auxiliary, burst, native_rate_values):
        state = SharkState(*state_values)
        cold_metallicity = cold_gas_metallicity(state, flow_parameters)
        qso = lagos23_qso_outflow_loadings(
            gas_mass=state.cold_gas,
            black_hole_mass_msun_over_h=auxiliary[9],
            hot_halo_accretion_rate_msun_over_h_per_gyr=auxiliary[11],
            starburst_accretion_rate_msun_over_h_per_gyr=auxiliary[12],
            spin=auxiliary[14],
            gas_metallicity=cold_metallicity,
            circular_velocity_km_per_s=auxiliary[8],
            star_formation_rate=auxiliary[15],
            bulge_baryonic_mass=state.stellar_mass + state.cold_gas,
            bulge_radius_mpc=auxiliary[1],
            parameters=agn_parameters,
        )
        forcing = lagos23_disk_forcing(
            gas_half_mass_radius=auxiliary[0],
            stellar_half_mass_radius=auxiliary[1],
            redshift=auxiliary[6],
            burst=burst,
            galaxy_velocity=auxiliary[8],
            subhalo_velocity=auxiliary[7],
            cooling_rate=auxiliary[2],
            cooling_metallicity=auxiliary[3],
            cooling_specific_angular_momentum=auxiliary[4],
            qso_reheating_loading=qso.reheating,
            qso_ejection_loading=qso.ejection,
        )
        rates = lagos23_disk_flow_rates(
            0.0,
            state,
            forcing,
            star_formation_parameters,
            feedback_parameters,
        )
        result = shark_rhs_from_rates(0.0, state, rates, flow_parameters)
        native_routing = shark_rhs_from_rates(
            0.0, state, SharkFlowRates(*native_rate_values), flow_parameters
        )
        derived = jnp.stack(
            (
                result.cold_gas_metallicity,
                jnp.maximum(rates.cooling_metallicity, flow_parameters.pre_enrichment_metallicity),
                effective_stellar_yield(result.cold_gas_metallicity, flow_parameters),
            )
        )
        return (
            jnp.stack(rates),
            jnp.stack(result.derivative),
            jnp.stack(native_routing.derivative),
            derived,
        )

    return jax.jit(jax.vmap(evaluate_one))


def _native_rates(records):
    auxiliary = records[:, _AUXILIARY_SLICE]
    return np.column_stack(
        (
            auxiliary[:, 2],
            auxiliary[:, 15],
            auxiliary[:, 16],
            auxiliary[:, 17],
            auxiliary[:, 18],
            auxiliary[:, 19],
            auxiliary[:, 20],
            auxiliary[:, 21],
            auxiliary[:, 22],
            auxiliary[:, 3],
            auxiliary[:, 4],
        )
    )


def evaluate_shark_population_parity(
    trace_path,
    tree_path,
    *,
    batch_size=65_536,
    relative_tolerance=1.1e-4,
    warning_relative_tolerance=1.5e-4,
    absolute_tolerance=1.0e-8,
    instrumented_output_root=None,
    reference_output_root=None,
) -> SharkPopulationParity:
    """Replay every realized native RHS state through the JAX Lagos23 core.

    The strict tolerance is fixed before the population result is inspected.
    Values between that gate and ``warning_relative_tolerance`` are retained as
    explicit warnings rather than silently widening the gate.  The warning band
    accommodates the distinct upstream adaptive and JAX deterministic radial
    quadratures.  The much looser native ODE tolerance is not used because this
    comparison is pointwise: both implementations see exactly the same state.
    """

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive Python integer")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(warning_relative_tolerance) or warning_relative_tolerance <= 0.0:
        raise ValueError("warning_relative_tolerance must be finite and positive")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "SHARK population parity requires JAX 64-bit mode; set JAX_ENABLE_X64=1 "
            "before importing mimic_jax"
        )
    if (instrumented_output_root is None) != (reference_output_root is None):
        raise ValueError(
            "instrumented_output_root and reference_output_root must be supplied together"
        )
    native_trace_noninterference = None
    if instrumented_output_root is not None:
        native_trace_noninterference = compare_native_shark_catalogues(
            instrumented_output_root, reference_output_root
        )
    trace = load_shark_rhs_trace(trace_path, validation_batch_size=batch_size)
    tree = load_shark_tree(tree_path)
    kernel = _population_replay_kernel()
    if warning_relative_tolerance < relative_tolerance:
        raise ValueError("warning_relative_tolerance must be at least relative_tolerance")
    rhs_comparison = _StreamingComparison(
        relative_tolerance, warning_relative_tolerance, absolute_tolerance
    )
    recomputed_rhs_comparison = _StreamingComparison(
        relative_tolerance, warning_relative_tolerance, absolute_tolerance
    )
    rate_comparison = _StreamingComparison(
        relative_tolerance, warning_relative_tolerance, absolute_tolerance
    )
    derived_comparison = _StreamingComparison(
        relative_tolerance, warning_relative_tolerance, absolute_tolerance
    )
    per_rate = [
        _StreamingComparison(relative_tolerance, warning_relative_tolerance, absolute_tolerance)
        for _ in SHARK_POPULATION_RATE_NAMES
    ]
    galaxy_ids_seen = set()
    first_snapshot = None
    last_snapshot = None
    disk_evaluations = 0

    # Compile and synchronize once at the same fixed shape used by every replay
    # batch.  This keeps compilation separate from the reported steady-state
    # population evaluation and avoids recompiling for a short final batch.
    warmup_records = np.asarray(trace[: min(len(trace), batch_size)])
    if len(warmup_records) < batch_size:
        padded_warmup = np.empty((batch_size, SHARK_RHS_TRACE_WIDTH), dtype=np.float64)
        padded_warmup[: len(warmup_records)] = warmup_records
        padded_warmup[len(warmup_records) :] = warmup_records[0]
        warmup_records = padded_warmup
    compilation_started = perf_counter()
    warmup_outputs = kernel(
        jnp.asarray(warmup_records[:, _STATE_SLICE]),
        jnp.asarray(warmup_records[:, _AUXILIARY_SLICE]),
        jnp.asarray(warmup_records[:, 4], dtype=jnp.bool_),
        jnp.asarray(_native_rates(warmup_records)),
    )
    jax.device_get(warmup_outputs)
    compilation_elapsed = perf_counter() - compilation_started

    started = perf_counter()
    for start in range(0, len(trace), batch_size):
        valid_records = np.asarray(trace[start : start + batch_size])
        valid_count = len(valid_records)
        if valid_count < batch_size:
            records = np.empty((batch_size, SHARK_RHS_TRACE_WIDTH), dtype=np.float64)
            records[:valid_count] = valid_records
            records[valid_count:] = valid_records[0]
        else:
            records = valid_records
        state = jnp.asarray(records[:, _STATE_SLICE])
        auxiliary = jnp.asarray(records[:, _AUXILIARY_SLICE])
        burst = jnp.asarray(records[:, 4], dtype=jnp.bool_)
        native_rates = _native_rates(records)
        candidate_rates, candidate_rhs, routed_rhs, candidate_derived = kernel(
            state, auxiliary, burst, jnp.asarray(native_rates)
        )
        candidate_rates, candidate_rhs, routed_rhs, candidate_derived = jax.device_get(
            (candidate_rates, candidate_rhs, routed_rhs, candidate_derived)
        )
        candidate_rates = candidate_rates[:valid_count]
        candidate_rhs = candidate_rhs[:valid_count]
        routed_rhs = routed_rhs[:valid_count]
        candidate_derived = candidate_derived[:valid_count]
        native_rates = native_rates[:valid_count]
        native_rhs = valid_records[:, _RHS_SLICE]
        native_derived = valid_records[:, 67:70]
        batch_snapshots = valid_records[:, 3]
        batch_first_snapshot = int(np.min(batch_snapshots))
        batch_last_snapshot = int(np.max(batch_snapshots))
        first_snapshot = (
            batch_first_snapshot
            if first_snapshot is None
            else min(first_snapshot, batch_first_snapshot)
        )
        last_snapshot = (
            batch_last_snapshot
            if last_snapshot is None
            else max(last_snapshot, batch_last_snapshot)
        )
        disk_evaluations += int(np.count_nonzero(valid_records[:, 4] == 0.0))
        galaxy_ids_seen.update(np.asarray(valid_records[:, 1], dtype=np.int64).tolist())
        rate_comparison.update(candidate_rates, native_rates)
        rhs_comparison.update(routed_rhs, native_rhs)
        # Near-cancellation can make a final derivative ill-conditioned even
        # when each physical rate agrees. Retain this useful measurement but
        # gate parity on the separately tested rate layer and exact routing.
        recomputed_rhs_comparison.update(candidate_rhs, native_rhs)
        derived_comparison.update(candidate_derived, native_derived)
        for index, comparison in enumerate(per_rate):
            comparison.update(candidate_rates[:, index], native_rates[:, index])
    elapsed = perf_counter() - started

    rhs_result = rhs_comparison.result()
    recomputed_rhs_result = recomputed_rhs_comparison.result()
    rates_result = rate_comparison.result()
    derived_result = derived_comparison.result()
    strict_passed = not (
        rhs_result.failing_values or rates_result.failing_values or derived_result.failing_values
    )
    passed = not (
        rhs_result.outside_warning_band_values
        or rates_result.outside_warning_band_values
        or derived_result.outside_warning_band_values
    )
    if native_trace_noninterference is not None:
        strict_passed = strict_passed and native_trace_noninterference.passed
        passed = passed and native_trace_noninterference.passed
    return SharkPopulationParity(
        method=(
            "exhaustive native-state shadow replay through a jax.jit(jax.vmap) Lagos23 "
            "BR06+Lagos13+QSO+19-state RHS kernel"
        ),
        status="passed" if strict_passed else "warning" if passed else "failed",
        passed=passed,
        trace_path=Path(trace_path).name,
        trace_sha256=_sha256(trace_path),
        tree_path=Path(tree_path).name,
        tree_sha256=_sha256(tree_path),
        number_of_trees=tree.number_of_trees,
        number_of_tree_nodes=tree.number_of_nodes_total,
        missing_descendant_links_skipped_upstream=tree.number_of_missing_descendants,
        rhs_evaluations=int(len(trace)),
        disk_rhs_evaluations=disk_evaluations,
        starburst_rhs_evaluations=int(len(trace) - disk_evaluations),
        unique_galaxies_evaluated=len(galaxy_ids_seen),
        first_snapshot=first_snapshot,
        last_snapshot=last_snapshot,
        strict_relative_tolerance=float(relative_tolerance),
        warning_relative_tolerance=float(warning_relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        strict_passed=strict_passed,
        rhs=rhs_result,
        recomputed_rhs=recomputed_rhs_result,
        rates=rates_result,
        derived_quantities=derived_result,
        native_trace_noninterference=native_trace_noninterference,
        compilation_seconds=float(compilation_elapsed),
        elapsed_seconds=float(elapsed),
        jax_backend=jax.default_backend(),
        jax_version=jax.__version__,
        qualifications=(
            "The native topology driver supplies the realized states and branch schedule.",
            "This evaluates every continuous disk and starburst RHS call, not an independent "
            "topology-owning JAX catalogue evolution.",
            "Cooling preparation, environment, mergers, and disk instabilities retain their "
            "separate controlled-oracle evidence.",
        ),
        rate_maximum_relative_error={
            name: comparison.result().maximum_relative_error
            for name, comparison in zip(SHARK_POPULATION_RATE_NAMES, per_rate)
        },
        rate_strict_failing_values={
            name: comparison.result().failing_values
            for name, comparison in zip(SHARK_POPULATION_RATE_NAMES, per_rate)
        },
        rate_outside_warning_band_values={
            name: comparison.result().outside_warning_band_values
            for name, comparison in zip(SHARK_POPULATION_RATE_NAMES, per_rate)
        },
    )
