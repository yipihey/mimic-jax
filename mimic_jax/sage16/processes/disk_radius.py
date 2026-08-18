"""Fiducial SAGE16 disk-scale-radius initialization."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.transfers import DiskScaleRadiusResult
from mimic_jax.sage16.types import GalaxyState, HaloForcing

_EPSILON_SMALL = 1.0e-10


def set_disk_scale_radius(state: GalaxyState, halo: HaloForcing) -> DiskScaleRadiusResult:
    """Recompute the Mo--Mao--White disk radius for a Type-0 central.

    This is a forcing-derived pre-timestep state update, not a baryonic
    transfer.  Type-1/2 satellites retain the radius inherited from the last
    time they were a central, exactly as in upstream SAGE16.
    """

    require_x64()

    def update_central(_):
        # The upstream helper accepts float arguments even though Rvir and Vvir
        # are stored as doubles in Halo.  Preserve that narrowing here.
        vvir = as_float32(halo.Vvir)
        rvir = as_float32(halo.Rvir)

        def physical_radius(_):
            spin = as_float64(halo.Spin)
            spin_magnitude = jnp.sqrt(spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2])
            vvir64 = as_float64(vvir)
            rvir64 = as_float64(rvir)
            spin_parameter = spin_magnitude / (1.414 * vvir64 * rvir64)
            return as_float32((spin_parameter / 1.414) * rvir64)

        def fallback_radius(_):
            return as_float32(jnp.asarray(0.1, dtype=jnp.float32) * rvir)

        radius = jax.lax.cond(
            (vvir > _EPSILON_SMALL) & (rvir > _EPSILON_SMALL),
            physical_radius,
            fallback_radius,
            operand=None,
        )
        return DiskScaleRadiusResult(
            state._replace(DiskScaleRadius=radius),
            radius,
            jnp.asarray(True),
        )

    def preserve_satellite(_):
        return DiskScaleRadiusResult(
            state,
            state.DiskScaleRadius,
            jnp.asarray(False),
        )

    return jax.lax.cond(halo.Type == 0, update_central, preserve_satellite, operand=None)
