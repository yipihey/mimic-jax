"""Fiducial MIMIC/SAGE16 disk-instability structural response."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.transfers import DiskInstabilityResult, DiskInstabilityTransfer
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, Sage16Units


def _float_difference(left, right):
    return as_float32(left - right)


def apply_disk_instability(
    state: GalaxyState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    log_fractional_perturbation=0.0,
) -> DiskInstabilityResult:
    """Move unstable stellar-disk mass to the bulge and emit the gas trigger.

    This preserves the upstream finite structural map. The optional response
    perturbation scales the live unstable mass before the ordinary disk-mass
    cap; zero is exactly the compiled SAGE16 calculation.
    """

    require_x64()
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    disk_stars_float = _float_difference(state.StellarMass, state.BulgeMass)
    disk_mass_float = as_float32(state.ColdGas + disk_stars_float)
    disk_mass = as_float64(disk_mass_float)

    def apply_positive_disk(_):
        vmax_squared = as_float32(halo.Vmax * halo.Vmax)
        critical = (
            as_float64(vmax_squared)
            * (parameters.StarFormingDiskFactor * as_float64(state.DiskScaleRadius))
            / units.G
        )
        critical = jnp.minimum(critical, disk_mass)
        raw_unstable = disk_mass - critical
        unstable_total = jnp.minimum(
            log_fractionally_perturb(raw_unstable, log_fractional_perturbation),
            disk_mass,
        )
        gas_fraction = as_float64(state.ColdGas) / disk_mass
        unstable_gas = gas_fraction * unstable_total
        unstable_stars = (1.0 - gas_fraction) * unstable_total
        disk_stars = as_float64(disk_stars_float)
        disk_metals_float = _float_difference(
            state.MetalsStellarMass,
            state.MetalsBulgeMass,
        )
        disk_metals = as_float64(disk_metals_float)

        def transfer_stars(_):
            transferred_stars = jnp.minimum(unstable_stars, disk_stars)
            transferred_metals = metallicity(disk_stars, disk_metals) * transferred_stars
            transferred_metals = jnp.where(
                disk_metals <= 0.0,
                0.0,
                jnp.minimum(transferred_metals, disk_metals),
            )
            bulge = as_float32(as_float64(state.BulgeMass) + transferred_stars)
            bulge_metals = as_float32(as_float64(state.MetalsBulgeMass) + transferred_metals)
            updated = state._replace(
                BulgeMass=jnp.minimum(bulge, state.StellarMass),
                MetalsBulgeMass=jnp.minimum(bulge_metals, state.MetalsStellarMass),
            )
            return updated, transferred_stars, transferred_metals

        def transfer_no_stars(_):
            return state, zero, zero

        updated, transferred_stars, transferred_metals = jax.lax.cond(
            (unstable_stars > 0.0) & (disk_stars > 0.0),
            transfer_stars,
            transfer_no_stars,
            operand=None,
        )
        unstable_fraction = jnp.where(
            (unstable_gas > 0.0) & (state.ColdGas > 0.0),
            unstable_gas / as_float64(state.ColdGas),
            0.0,
        )
        updated = updated._replace(UnstableDiskGasFraction=unstable_fraction)
        transfer = DiskInstabilityTransfer(
            disk_mass,
            critical,
            unstable_gas,
            unstable_fraction,
            transferred_stars,
            transferred_metals,
        )
        return DiskInstabilityResult(updated, transfer)

    def apply_empty_disk(_):
        updated = state._replace(UnstableDiskGasFraction=zero)
        return DiskInstabilityResult(
            updated,
            DiskInstabilityTransfer(zero, zero, zero, zero, zero, zero),
        )

    return jax.lax.cond(
        disk_mass > 0.0,
        apply_positive_disk,
        apply_empty_disk,
        operand=None,
    )
