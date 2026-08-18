"""Fiducial SAGE16 merger-clock initialization for one FoF group."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.transfers import MergerClockDiagnostics, MergerClockResult
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Units

_MIN_NUM_PART_SAT_HALO = 10
_MERGTIME_UNSET = 999.9
_MERGTIME_UNSET_THRESHOLD = 999.0
_MERGTIME_CEILING = 998.0
_MERGTIME_IMMEDIATE = -1.0


def _target_indices(halos: HaloForcing, fof_central_index):
    """Apply the pre-event Type-2 target rules from ``central_link.h``."""

    count = halos.Type.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    candidate = halos.CentralHalo
    candidate_in_range = (candidate >= 0) & (candidate < count) & (candidate != indices)
    safe_candidate = jnp.clip(candidate, 0, jnp.maximum(count - 1, 0))
    candidate_type = halos.Type[safe_candidate]
    candidate_is_central = (candidate_type == 0) | (candidate_type == 1)
    valid_type2_target = candidate_in_range & candidate_is_central
    return jnp.where(
        (halos.Type == 2) & valid_type2_target,
        candidate,
        fof_central_index,
    )


def initialise_merger_clocks(
    states: GalaxyState,
    halos: HaloForcing,
    units: Sage16Units,
) -> MergerClockResult:
    """Initialize live satellite merger clocks for a batched FoF group.

    The leading dimension is the live group order used by upstream.  Type-0
    promotion resets the sentinel, an unset Type-2 orphan is forced to merge,
    and an unset Type-1 satellite receives the Binney--Tremaine dynamical-
    friction time.  Existing clocks are not recalculated.

    A group without a Type-0 central is returned unchanged, matching upstream.
    """

    require_x64()
    has_central = jnp.any(halos.Type == 0)
    fof_central_index = jnp.argmax(halos.Type == 0).astype(jnp.int32)
    targets = _target_indices(halos, fof_central_index)

    before = states.MergTime
    is_type0 = halos.Type == 0
    reset_central = is_type0 & (before < _MERGTIME_UNSET_THRESHOLD)
    force_immediate = (halos.Type == 2) & (before > _MERGTIME_UNSET_THRESHOLD) & has_central
    eligible = (
        ((halos.Type == 1) | (halos.Type == 2))
        & (before > _MERGTIME_UNSET_THRESHOLD)
        & ~force_immediate
        & has_central
    )

    central_len = halos.Len[targets]
    central_rvir = halos.Rvir[targets]
    central_vvir = halos.Vvir[targets]
    sat_len = halos.Len
    coulomb = jnp.where(
        sat_len > 0,
        jnp.log1p(as_float64(central_len) / as_float64(jnp.maximum(sat_len, 1))),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    satellite_mass = as_float64(halos.Mvir) + as_float64(states.StellarMass)
    satellite_mass = satellite_mass + as_float64(states.ColdGas)
    resolved = (satellite_mass > 0.0) & (coulomb > 0.0) & (sat_len >= _MIN_NUM_PART_SAT_HALO)
    numerator = 2.0 * 1.17 * central_rvir * central_rvir * central_vvir
    safe_denominator = jnp.where(
        resolved,
        coulomb * units.G * satellite_mass,
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    calculated = jnp.where(
        resolved,
        numerator / safe_denominator,
        jnp.asarray(_MERGTIME_IMMEDIATE, dtype=jnp.float64),
    )
    calculated = jnp.where(
        calculated >= _MERGTIME_UNSET_THRESHOLD,
        jnp.asarray(_MERGTIME_CEILING, dtype=jnp.float64),
        calculated,
    )

    after = jnp.where(reset_central, as_float32(_MERGTIME_UNSET), before)
    after = jnp.where(force_immediate, as_float32(0.0), after)
    after = jnp.where(eligible, as_float32(calculated), after)
    updated = states._replace(MergTime=after)
    diagnostics = MergerClockDiagnostics(
        before=before,
        after=after,
        target_indices=jnp.where(has_central, targets, -1),
        initialized=eligible,
        forced_immediate=force_immediate,
        reset_central=reset_central & has_central,
    )
    return MergerClockResult(updated, diagnostics)
