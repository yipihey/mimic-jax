"""Shared numerical helpers with direct counterparts in SAGE16 shared headers."""

import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float64
from mimic_jax.sage16.types import HaloForcing, StepContext


def metallicity(gas, metals):
    """Match ``mimic_get_metallicity``: zero unless both inputs are positive, capped at one."""

    gas64 = as_float64(gas)
    metals64 = as_float64(metals)
    denominator = jnp.where(gas64 > 0.0, gas64, 1.0)
    ratio = metals64 / denominator
    return jnp.where((gas64 > 0.0) & (metals64 > 0.0), jnp.minimum(ratio, 1.0), 0.0)


def object_substep_dt(halo: HaloForcing, context: StepContext):
    """Match SAGE's per-object ``dT / num_substeps`` and initial-boundary no-op."""

    valid_dt = halo.dT > 0.0
    initial_boundary = (halo.SnapNum < 0) & (halo.dT <= 0.0)
    invalid = jnp.asarray(jnp.nan, dtype=jnp.float64)
    return jnp.where(
        valid_dt,
        halo.dT / as_float64(context.num_substeps),
        jnp.where(initial_boundary, 0.0, invalid),
    )
