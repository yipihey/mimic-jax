"""Fiducial MIMIC/SAGE16 Type-1 satellite hot-gas stripping."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.transfers import (
    SatelliteStrippingResult,
    SatelliteStrippingTransfer,
)
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    StepContext,
)


def _float_baryon_sum(state):
    """Reproduce the all-float reservoir expression in the upstream C module."""

    total = as_float32(state.StellarMass + state.ColdGas)
    total = as_float32(total + state.HotGas)
    total = as_float32(total + state.EjectedGas)
    total = as_float32(total + state.BlackHoleMass)
    return as_float64(as_float32(total + state.ICS))


def apply_satellite_stripping(
    satellite: GalaxyState,
    central: GalaxyState,
    satellite_halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    log_fractional_perturbation=0.0,
) -> SatelliteStrippingResult:
    """Strip one Type-1 satellite immediately before that galaxy's cooling call.

    The excess is recomputed from the live satellite state and divided by the
    configured number of substeps. It is not a partition of a fixed snapshot
    budget, and it is deliberately independent of ``dt`` for upstream parity.
    """

    require_x64()
    active = (satellite_halo.Type == 1) & (satellite.HotGas > 0.0)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def apply_active(_):
        halo_baryon_fraction = jnp.where(
            satellite.HaloBaryonFraction > 0.0,
            satellite.HaloBaryonFraction,
            parameters.GlobalBaryonFraction,
        )
        satellite_baryons = _float_baryon_sum(satellite)
        allowed_baryons = halo_baryon_fraction * satellite_halo.Mvir
        requested = (satellite_baryons - allowed_baryons) / as_float64(context.num_substeps)
        requested = log_fractionally_perturb(
            requested,
            log_fractional_perturbation,
        )

        def strip_positive(_):
            requested_metals = requested * metallicity(satellite.HotGas, satellite.MetalsHotGas)
            stripped_gas = jnp.minimum(requested, as_float64(satellite.HotGas))
            stripped_metals = jnp.minimum(
                requested_metals,
                as_float64(satellite.MetalsHotGas),
            )
            updated_satellite = satellite._replace(
                HotGas=as_float32(as_float64(satellite.HotGas) - stripped_gas),
                MetalsHotGas=as_float32(as_float64(satellite.MetalsHotGas) - stripped_metals),
            )
            updated_central = central._replace(
                HotGas=as_float32(as_float64(central.HotGas) + stripped_gas),
                MetalsHotGas=as_float32(as_float64(central.MetalsHotGas) + stripped_metals),
            )
            transfer = SatelliteStrippingTransfer(
                stripped_gas,
                stripped_metals,
                allowed_baryons,
                satellite_baryons,
            )
            return SatelliteStrippingResult(
                updated_satellite,
                updated_central,
                transfer,
            )

        def strip_zero(_):
            transfer = SatelliteStrippingTransfer(
                zero,
                zero,
                allowed_baryons,
                satellite_baryons,
            )
            return SatelliteStrippingResult(satellite, central, transfer)

        return jax.lax.cond(
            requested > 0.0,
            strip_positive,
            strip_zero,
            operand=None,
        )

    def apply_inactive(_):
        return SatelliteStrippingResult(
            satellite,
            central,
            SatelliteStrippingTransfer(zero, zero, zero, zero),
        )

    return jax.lax.cond(active, apply_active, apply_inactive, operand=None)
