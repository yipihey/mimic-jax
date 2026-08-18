"""Fractional perturbations around faithful SAGE16 process transfers."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from mimic_jax.sage16.precision import require_x64

Array = Any

PROCESS_NAMES = (
    "cooling",
    "star_formation",
    "sn_reheating",
    "sn_ejection",
    "reincorporation",
    "agn_heating",
    "infall",
    "satellite_stripping",
    "disk_instability",
    "quasar_mode",
    "starburst",
)


class ProcessPerturbations(NamedTuple):
    """Logarithmic multipliers for implemented finite-time process transfers."""

    cooling: Array
    star_formation: Array
    sn_reheating: Array
    sn_ejection: Array
    reincorporation: Array
    agn_heating: Array
    infall: Array
    satellite_stripping: Array
    disk_instability: Array
    quasar_mode: Array
    starburst: Array


def process_perturbations(**overrides) -> ProcessPerturbations:
    """Construct scalar or epoch-vector perturbations, with zero as the faithful model."""

    require_x64()
    unknown = set(overrides) - set(PROCESS_NAMES)
    if unknown:
        raise TypeError(f"Unknown implemented SAGE16 processes: {sorted(unknown)}")
    return ProcessPerturbations(
        **{name: jnp.asarray(overrides.get(name, 0.0), dtype=jnp.float64) for name in PROCESS_NAMES}
    )


def perturbations_from_matrix(values) -> ProcessPerturbations:
    """Map a process-by-epoch epsilon matrix to the named SAGE16 perturbation PyTree."""

    require_x64()
    values = jnp.asarray(values, dtype=jnp.float64)
    if values.ndim != 2 or values.shape[0] != len(PROCESS_NAMES):
        raise ValueError(
            f"Expected process perturbations with shape ({len(PROCESS_NAMES)}, n_epoch)"
        )
    return ProcessPerturbations(*tuple(values[index] for index in range(len(PROCESS_NAMES))))


def log_fractionally_perturb(value, epsilon):
    """Apply ``value -> value * exp(epsilon)`` so epsilon is a fractional perturbation."""

    return value * jnp.exp(jnp.asarray(epsilon, dtype=jnp.float64))
