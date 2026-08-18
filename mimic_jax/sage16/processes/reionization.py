"""Fiducial SAGE16 reionization suppression of halo baryon accretion."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.transfers import ReionizationResult
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

EPSILON_SMALL = 1.0e-10
REIONIZATION_ALPHA = 6.0
IONIZED_VIRIAL_TEMPERATURE_K = 1.0e4


def reionization_modifier(
    halo: HaloForcing,
    context: StepContext,
    units: Sage16Units,
):
    """Return the Gnedin/Kravtsov modifier used by upstream SAGE16."""

    require_x64()
    redshift = context.redshift
    scale_factor = 1.0 / (1.0 + redshift)
    uv_on_scale_factor = 1.0 / 9.0
    reionized_scale_factor = 1.0 / 8.0
    relative_to_uv_on = scale_factor / uv_on_scale_factor
    relative_to_reionized = scale_factor / reionized_scale_factor

    before_uv = (
        3.0
        * scale_factor
        / ((2.0 + REIONIZATION_ALPHA) * (5.0 + 2.0 * REIONIZATION_ALPHA))
        * relative_to_uv_on**REIONIZATION_ALPHA
    )
    during_reionization = (
        (3.0 / scale_factor)
        * uv_on_scale_factor**2
        * (
            1.0 / (2.0 + REIONIZATION_ALPHA)
            - 2.0 / jnp.sqrt(relative_to_uv_on) / (5.0 + 2.0 * REIONIZATION_ALPHA)
        )
        + scale_factor**2 / 10.0
        - (uv_on_scale_factor**2 / 10.0) * (5.0 - 4.0 / jnp.sqrt(relative_to_uv_on))
    )
    after_reionization = (3.0 / scale_factor) * (
        uv_on_scale_factor**2
        * (
            1.0 / (2.0 + REIONIZATION_ALPHA)
            - 2.0 / jnp.sqrt(relative_to_uv_on) / (5.0 + 2.0 * REIONIZATION_ALPHA)
        )
        + (reionized_scale_factor**2 / 10.0) * (5.0 - 4.0 / jnp.sqrt(relative_to_reionized))
        - (uv_on_scale_factor**2 / 10.0) * (5.0 - 4.0 / jnp.sqrt(relative_to_uv_on))
        + scale_factor * reionized_scale_factor / 3.0
        - (reionized_scale_factor**2 / 3.0) * (3.0 - 2.0 / jnp.sqrt(relative_to_reionized))
    )
    filtering_function = jnp.where(
        scale_factor <= uv_on_scale_factor,
        before_uv,
        jnp.where(
            scale_factor < reionized_scale_factor,
            during_reionization,
            after_reionization,
        ),
    )
    jeans_mass = 25.0 / jnp.sqrt(units.Omega) * 2.21
    filtering_mass = jeans_mass * filtering_function**1.5
    characteristic_velocity = jnp.sqrt(IONIZED_VIRIAL_TEMPERATURE_K / 36.0)
    one_plus_redshift_cubed = (1.0 + redshift) ** 3
    omega_at_redshift = (
        units.Omega
        * one_plus_redshift_cubed
        / (units.Omega * one_plus_redshift_cubed + units.OmegaLambda)
    )
    x_at_redshift = omega_at_redshift - 1.0
    critical_overdensity = 18.0 * jnp.pi**2 + 82.0 * x_at_redshift - 39.0 * x_at_redshift**2
    hubble_at_redshift = units.Hubble * jnp.sqrt(
        units.Omega * one_plus_redshift_cubed + units.OmegaLambda
    )
    characteristic_mass = characteristic_velocity**3 / (
        units.G * hubble_at_redshift * jnp.sqrt(0.5 * critical_overdensity)
    )
    suppressing_mass = jnp.maximum(filtering_mass, characteristic_mass)
    # The upstream helper accepts float Mvir even though the halo field is double.
    halo_mass = as_float64(as_float32(halo.Mvir))
    factor = 1.0 + 0.26 * suppressing_mass / halo_mass
    return 1.0 / factor**3


def apply_reionization(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
) -> ReionizationResult:
    """Set ``HaloBaryonFraction`` for one non-ejected halo."""

    require_x64()

    def apply_non_ejected(_):
        modifier = jax.lax.cond(
            halo.Mvir > EPSILON_SMALL,
            lambda _: reionization_modifier(halo, context, units),
            lambda _: jnp.asarray(0.0, dtype=jnp.float64),
            operand=None,
        )
        return ReionizationResult(
            state._replace(HaloBaryonFraction=parameters.GlobalBaryonFraction * modifier),
            modifier,
        )

    return jax.lax.cond(
        halo.Type == 3,
        lambda _: ReionizationResult(state, jnp.asarray(1.0, dtype=jnp.float64)),
        apply_non_ejected,
        operand=None,
    )
