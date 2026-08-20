"""Finite maps, projections, and event maps around SHARK's ODE flows.

These functions preserve the mathematical distinction visible in upstream
SHARK.  Snapshot budgets and source caps are finite maps; the heating radius
and post-solve bounds are projections; disk instabilities and mergers are
events.  They are pure functions and are differentiable on each fixed branch.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mimic_jax.shark.components import (
    CENTRAL_GALAXY,
    TYPE2_GALAXY,
    BaryonComponent,
    RotatingBaryonComponent,
    SharkGalaxyState,
    SharkSubhaloState,
    SharkSystemState,
    SizedBaryonComponent,
    galaxy_baryonic_mass,
    galaxy_gas_mass,
    galaxy_stellar_mass,
)
from mimic_jax.shark.types import SharkState

Array = Any

_GRAVITATIONAL_CONSTANT = 6.67259e-11 * 1.9891e30 / 3.0856775807e22 / 1.0e6
_EAGLE_J_CONVERSION = 0.835
_DISK_HALF_MASS_SCALE = 1.678346990
_REFERENCE_TOLERANCE = 1.0e-10


class FiniteTransfer(NamedTuple):
    """Realized source-capped transfer and its carried extensive quantities."""

    mass: Array
    metals: Array
    angular_momentum: Array


class DiskInstabilityResult(NamedTuple):
    state: SharkSystemState
    triggered: Array
    transferred_mass: Array
    transferred_metals: Array
    angular_momentum_projection_residual: Array


class GalaxyMergerResult(NamedTuple):
    central: SharkGalaxyState
    mass_ratio: Array
    gas_to_stellar_ratio: Array
    major: Array
    burst_triggered: Array
    angular_momentum_projection_residual: Array


class StrippingResult(NamedTuple):
    satellite: SharkSystemState
    central_subhalo: SharkSubhaloState
    transferred_mass: Array
    transferred_metals: Array
    target_radius: Array


def _safe_ratio(numerator, denominator):
    denominator = jnp.asarray(denominator)
    return jnp.where(denominator > 0.0, jnp.asarray(numerator) / denominator, 0.0)


def _add_baryon(component: BaryonComponent, mass, metals):
    return component._replace(mass=component.mass + mass, metals=component.metals + metals)


def _add_rotating(component: RotatingBaryonComponent, mass, metals, angular_momentum=0.0):
    return component._replace(
        mass=component.mass + mass,
        metals=component.metals + metals,
        angular_momentum=component.angular_momentum + angular_momentum,
    )


def _add_sized(component: SizedBaryonComponent, mass, metals, angular_momentum=0.0):
    return component._replace(
        mass=component.mass + mass,
        metals=component.metals + metals,
        angular_momentum=component.angular_momentum + angular_momentum,
    )


def proportional_transfer(component, requested_mass) -> tuple[Any, FiniteTransfer]:
    """Remove a capped mass carrying source composition and total AM.

    The function supports all three baryonic component schemas.  Its returned
    transfer is suitable for explicitly routing to a destination reservoir.
    """

    realized = jnp.minimum(jnp.maximum(jnp.asarray(requested_mass), 0.0), component.mass)
    fraction = _safe_ratio(realized, component.mass)
    metals = fraction * component.metals
    angular_momentum = fraction * getattr(component, "angular_momentum", 0.0)
    updated = component._replace(mass=component.mass - realized, metals=component.metals - metals)
    if hasattr(component, "angular_momentum"):
        updated = updated._replace(angular_momentum=component.angular_momentum - angular_momentum)
    return updated, FiniteTransfer(realized, metals, angular_momentum)


def apply_reincorporation_transfer(state: SharkSystemState, requested_mass):
    """Apply upstream's finite ejected-to-hot reincorporation transfer."""

    ejected, transfer = proportional_transfer(state.subhalo.ejected_gas, requested_mass)
    hot = _add_rotating(
        state.subhalo.hot_halo_gas,
        transfer.mass,
        transfer.metals,
        transfer.angular_momentum,
    )
    return state._replace(subhalo=state.subhalo._replace(ejected_gas=ejected, hot_halo_gas=hot))


def apply_cosmological_infall(
    state: SharkSystemState,
    accreted_mass,
    maximum_allowed_baryon_accretion,
    pre_enrichment_metallicity,
    infall_specific_angular_momentum=0.0,
):
    """Apply the exact positive, capped central-halo infall budget."""

    realized = jnp.minimum(
        jnp.maximum(jnp.asarray(accreted_mass), 0.0),
        jnp.maximum(jnp.asarray(maximum_allowed_baryon_accretion), 0.0),
    )
    hot = _add_rotating(
        state.subhalo.hot_halo_gas,
        realized,
        realized * pre_enrichment_metallicity,
        realized * infall_specific_angular_momentum,
    )
    return state._replace(subhalo=state.subhalo._replace(hot_halo_gas=hot)), realized


def enforce_baryon_fraction_limit(state: SharkSystemState, excess_inside_halo_mass):
    """Conservatively move an inside-halo excess from hot to ejected gas.

    The upstream branch assumes the hot reservoir can supply the group-level
    excess.  We make the realized source cap explicit; any unsatisfied excess
    is returned and must be treated as a constraint failure by the caller.
    """

    hot, transfer = proportional_transfer(state.subhalo.hot_halo_gas, excess_inside_halo_mass)
    ejected = _add_rotating(
        state.subhalo.ejected_gas,
        transfer.mass,
        transfer.metals,
        transfer.angular_momentum,
    )
    updated = state._replace(subhalo=state.subhalo._replace(hot_halo_gas=hot, ejected_gas=ejected))
    unsatisfied = jnp.maximum(jnp.asarray(excess_inside_halo_mass) - transfer.mass, 0.0)
    return updated, transfer, unsatisfied


def apply_black_hole_seed(state: SharkSystemState, halo_mass, seed_halo_mass, seed_mass):
    """Apply SHARK's thresholded black-hole seeding event."""

    active = (jnp.asarray(halo_mass) > seed_halo_mass) & (state.galaxy.black_hole.mass == 0.0)
    black_hole = state.galaxy.black_hole._replace(
        mass=jnp.where(active, seed_mass, state.galaxy.black_hole.mass),
        spin=jnp.where(active, 0.0, state.galaxy.black_hole.spin),
    )
    return state._replace(galaxy=state.galaxy._replace(black_hole=black_hole)), active


def apply_hot_halo_black_hole_transfer(state: SharkSystemState, accretion_rate, interval_gyr):
    """Apply the upstream finite, source-capped hot-mode BH growth map."""

    hot, transfer = proportional_transfer(
        state.subhalo.hot_halo_gas, jnp.asarray(accretion_rate) * interval_gyr
    )
    black_hole = state.galaxy.black_hole._replace(
        mass=state.galaxy.black_hole.mass + transfer.mass,
        metals=state.galaxy.black_hole.metals + transfer.metals,
        hot_halo_accretion_rate=jnp.asarray(accretion_rate),
    )
    updated = state._replace(
        galaxy=state.galaxy._replace(black_hole=black_hole),
        subhalo=state.subhalo._replace(hot_halo_gas=hot),
    )
    # Upstream BH spin is dimensionless, so removed gas AM has no represented
    # receiving reservoir.  Return it as an explicit ledger boundary.
    return updated, transfer


def apply_cooling_staging_transfer(
    state: SharkSystemState,
    cooling_rate,
    interval_gyr,
    cooling_specific_angular_momentum,
):
    """Apply upstream's pre-ODE hot-to-cold-halo cooling map and AM projection."""

    hot, transfer = proportional_transfer(
        state.subhalo.hot_halo_gas, jnp.asarray(cooling_rate) * interval_gyr
    )
    cold_before = state.subhalo.cold_halo_gas
    cold = _add_rotating(cold_before, transfer.mass, transfer.metals, 0.0)
    projected_angular_momentum = cold.mass * cooling_specific_angular_momentum
    projection_residual = projected_angular_momentum - (
        cold_before.angular_momentum + transfer.angular_momentum
    )
    cold = cold._replace(angular_momentum=projected_angular_momentum)
    realized_rate = jnp.where(interval_gyr > 0.0, transfer.mass / interval_gyr, 0.0)
    return (
        state._replace(subhalo=state.subhalo._replace(hot_halo_gas=hot, cold_halo_gas=cold)),
        transfer,
        realized_rate,
        projection_residual,
    )


def flow_state_from_system(state: SharkSystemState) -> SharkState:
    """Project a complete disk-system state into upstream's 19 ODE entries."""

    galaxy = state.galaxy
    subhalo = state.subhalo
    zero = jnp.zeros_like(galaxy.disk_stars.mass)
    return SharkState(
        stellar_mass=galaxy.disk_stars.mass,
        cold_gas=galaxy.disk_gas.mass,
        cold_halo_gas=subhalo.cold_halo_gas.mass,
        hot_halo_gas=subhalo.hot_halo_gas.mass,
        ejected_gas=subhalo.ejected_gas.mass,
        lost_gas=subhalo.lost_gas.mass,
        stellar_metals=galaxy.disk_stars.metals,
        cold_gas_metals=galaxy.disk_gas.metals,
        cold_halo_gas_metals=subhalo.cold_halo_gas.metals,
        hot_halo_gas_metals=subhalo.hot_halo_gas.metals,
        ejected_gas_metals=subhalo.ejected_gas.metals,
        lost_gas_metals=subhalo.lost_gas.metals,
        formed_stellar_mass=zero,
        formed_stellar_metals=zero,
        stellar_angular_momentum=galaxy.disk_stars.angular_momentum,
        cold_gas_angular_momentum=galaxy.disk_gas.angular_momentum,
        cold_halo_angular_momentum=subhalo.cold_halo_gas.angular_momentum,
        hot_halo_angular_momentum=subhalo.hot_halo_gas.angular_momentum,
        ejected_angular_momentum=subhalo.ejected_gas.angular_momentum,
    )


def burst_flow_state_from_system(state: SharkSystemState) -> SharkState:
    """Project a complete system into upstream's 19-entry bulge-burst state.

    SHARK reuses the same ODE for quiescent disks and event-triggered bulge
    starbursts.  Cooling and every angular-momentum entry are deliberately
    zero for the burst solve; the persistent halo reservoirs still take part
    in stellar and QSO feedback.
    """

    galaxy = state.galaxy
    subhalo = state.subhalo
    zero = jnp.zeros_like(galaxy.bulge_stars.mass)
    return SharkState(
        stellar_mass=galaxy.bulge_stars.mass,
        cold_gas=galaxy.bulge_gas.mass,
        cold_halo_gas=zero,
        hot_halo_gas=subhalo.hot_halo_gas.mass,
        ejected_gas=subhalo.ejected_gas.mass,
        lost_gas=subhalo.lost_gas.mass,
        stellar_metals=galaxy.bulge_stars.metals,
        cold_gas_metals=galaxy.bulge_gas.metals,
        cold_halo_gas_metals=zero,
        hot_halo_gas_metals=subhalo.hot_halo_gas.metals,
        ejected_gas_metals=subhalo.ejected_gas.metals,
        lost_gas_metals=subhalo.lost_gas.metals,
        formed_stellar_mass=zero,
        formed_stellar_metals=zero,
        stellar_angular_momentum=zero,
        cold_gas_angular_momentum=zero,
        cold_halo_angular_momentum=zero,
        hot_halo_angular_momentum=zero,
        ejected_angular_momentum=zero,
    )


def _project_component(mass, metals, angular_momentum, radius, velocity):
    positive = mass >= _REFERENCE_TOLERANCE
    bounded_mass = jnp.where(positive, mass, 0.0)
    bounded_metals = jnp.where(positive, jnp.clip(metals, 0.0, bounded_mass), 0.0)
    bounded_angular_momentum = jnp.where(positive, jnp.maximum(angular_momentum, 0.0), 0.0)
    new_radius = jnp.where(
        bounded_angular_momentum > 0.0,
        _safe_ratio(bounded_angular_momentum, bounded_mass)
        / jnp.maximum(velocity, jnp.finfo(jnp.float64).tiny)
        * _EAGLE_J_CONVERSION,
        radius,
    )
    return SizedBaryonComponent(bounded_mass, bounded_metals, bounded_angular_momentum, new_radius)


def project_flow_state_to_system(
    state: SharkSystemState,
    flow_state: SharkState,
    interval_gyr,
):
    """Apply upstream's post-ODE bounds and reconstruct disk sizes/diagnostics."""

    velocity = state.galaxy.maximum_circular_velocity
    disk_stars = _project_component(
        flow_state.stellar_mass,
        flow_state.stellar_metals,
        flow_state.stellar_angular_momentum,
        state.galaxy.disk_stars.radius,
        velocity,
    )
    disk_gas = _project_component(
        flow_state.cold_gas,
        flow_state.cold_gas_metals,
        flow_state.cold_gas_angular_momentum,
        state.galaxy.disk_gas.radius,
        velocity,
    )

    def rotating(mass, metals, angular_momentum):
        active = mass >= _REFERENCE_TOLERANCE
        bounded_mass = jnp.where(active, mass, 0.0)
        return RotatingBaryonComponent(
            bounded_mass,
            jnp.where(active, jnp.clip(metals, 0.0, bounded_mass), 0.0),
            jnp.where(active, jnp.maximum(angular_momentum, 0.0), 0.0),
        )

    def baryon(mass, metals):
        active = mass >= _REFERENCE_TOLERANCE
        bounded_mass = jnp.where(active, mass, 0.0)
        return BaryonComponent(
            bounded_mass, jnp.where(active, jnp.clip(metals, 0.0, bounded_mass), 0.0)
        )

    galaxy = state.galaxy._replace(
        disk_stars=disk_stars,
        disk_gas=disk_gas,
        stellar_mass_ever_formed=(
            state.galaxy.stellar_mass_ever_formed + flow_state.formed_stellar_mass
        ),
    )
    subhalo = state.subhalo._replace(
        cold_halo_gas=rotating(
            flow_state.cold_halo_gas,
            flow_state.cold_halo_gas_metals,
            flow_state.cold_halo_angular_momentum,
        ),
        hot_halo_gas=rotating(
            flow_state.hot_halo_gas,
            flow_state.hot_halo_gas_metals,
            flow_state.hot_halo_angular_momentum,
        ),
        ejected_gas=rotating(
            flow_state.ejected_gas,
            flow_state.ejected_gas_metals,
            flow_state.ejected_angular_momentum,
        ),
        lost_gas=baryon(flow_state.lost_gas, flow_state.lost_gas_metals),
    )
    mean_sfr = jnp.where(interval_gyr > 0.0, flow_state.formed_stellar_mass / interval_gyr, 0.0)
    mean_formed_metallicity = jnp.where(
        flow_state.formed_stellar_mass > 0.0,
        flow_state.formed_stellar_metals / flow_state.formed_stellar_mass,
        0.0,
    )
    diagnostics = {
        "mean_star_formation_rate": mean_sfr,
        "mean_formed_stellar_metallicity": mean_formed_metallicity,
    }
    return state._replace(galaxy=galaxy, subhalo=subhalo), diagnostics


def project_burst_flow_state_to_system(
    state: SharkSystemState,
    flow_state: SharkState,
    interval_gyr,
    *,
    from_galaxy_merger,
):
    """Project a solved burst state with upstream's tracker semantics.

    Bulge angular momentum and size are event-map quantities in SHARK and are
    not evolved by the burst ODE.  This projection therefore changes only
    masses/metals, the shared halo reservoirs, and the appropriate burst
    diagnostic.  It is the exact counterpart of
    ``BasicPhysicalModel::to_galaxy_starburst``.
    """

    galaxy = state.galaxy
    formed_stars = flow_state.stellar_mass - galaxy.bulge_stars.mass
    formed_metals = flow_state.stellar_metals - galaxy.bulge_stars.metals
    burst_increment = BaryonComponent(formed_stars, formed_metals)
    merger_burst = BaryonComponent(
        galaxy.merger_burst_stars.mass + jnp.where(from_galaxy_merger, burst_increment.mass, 0.0),
        galaxy.merger_burst_stars.metals
        + jnp.where(from_galaxy_merger, burst_increment.metals, 0.0),
    )
    instability_burst = BaryonComponent(
        galaxy.instability_burst_stars.mass
        + jnp.where(from_galaxy_merger, 0.0, burst_increment.mass),
        galaxy.instability_burst_stars.metals
        + jnp.where(from_galaxy_merger, 0.0, burst_increment.metals),
    )
    stellar_mass = jnp.maximum(flow_state.stellar_mass, 0.0)
    gas_mass = jnp.maximum(flow_state.cold_gas, 0.0)
    bulge_stars = galaxy.bulge_stars._replace(
        mass=stellar_mass,
        metals=jnp.clip(flow_state.stellar_metals, 0.0, stellar_mass),
    )
    bulge_gas = galaxy.bulge_gas._replace(
        mass=gas_mass,
        metals=jnp.clip(flow_state.cold_gas_metals, 0.0, gas_mass),
    )
    updated_galaxy = galaxy._replace(
        bulge_stars=bulge_stars,
        bulge_gas=bulge_gas,
        merger_burst_stars=merger_burst,
        instability_burst_stars=instability_burst,
        stellar_mass_ever_formed=(galaxy.stellar_mass_ever_formed + flow_state.formed_stellar_mass),
    )
    updated_subhalo = state.subhalo._replace(
        hot_halo_gas=state.subhalo.hot_halo_gas._replace(
            mass=jnp.maximum(flow_state.hot_halo_gas, 0.0),
            metals=jnp.clip(flow_state.hot_halo_gas_metals, 0.0, flow_state.hot_halo_gas),
        ),
        ejected_gas=state.subhalo.ejected_gas._replace(
            mass=jnp.maximum(flow_state.ejected_gas, 0.0),
            metals=jnp.clip(flow_state.ejected_gas_metals, 0.0, flow_state.ejected_gas),
        ),
        lost_gas=state.subhalo.lost_gas._replace(
            mass=jnp.maximum(flow_state.lost_gas, 0.0),
            metals=jnp.clip(flow_state.lost_gas_metals, 0.0, flow_state.lost_gas),
        ),
    )
    diagnostics = {
        "mean_star_formation_rate": jnp.where(
            interval_gyr > 0.0, flow_state.formed_stellar_mass / interval_gyr, 0.0
        ),
        "mean_formed_stellar_metallicity": jnp.where(
            flow_state.formed_stellar_mass > 0.0,
            flow_state.formed_stellar_metals / flow_state.formed_stellar_mass,
            0.0,
        ),
    }
    return state._replace(galaxy=updated_galaxy, subhalo=updated_subhalo), diagnostics


def disk_size(galaxy: SharkGalaxyState):
    mass = galaxy.disk_stars.mass + galaxy.disk_gas.mass
    weighted = (
        galaxy.disk_stars.mass * galaxy.disk_stars.radius
        + galaxy.disk_gas.mass * galaxy.disk_gas.radius
    )
    return _safe_ratio(weighted, mass)


def bulge_size(galaxy: SharkGalaxyState):
    mass = galaxy.bulge_stars.mass + galaxy.bulge_gas.mass
    weighted = (
        galaxy.bulge_stars.mass * galaxy.bulge_stars.radius
        + galaxy.bulge_gas.mass * galaxy.bulge_gas.radius
    )
    return _safe_ratio(weighted, mass)


def toomre_stability_parameter(galaxy: SharkGalaxyState):
    """Return upstream's disk-instability diagnostic."""

    mass = galaxy.disk_stars.mass + galaxy.disk_gas.mass
    radius = disk_size(galaxy)
    valid = (mass > 0.0) & (radius > 0.0)
    denominator = 1.68 * _GRAVITATIONAL_CONSTANT * mass / jnp.where(valid, radius, 1.0)
    value = galaxy.maximum_circular_velocity / jnp.sqrt(denominator)
    return jnp.where(valid, value, 100.0)


def disk_instability_bulge_radius(galaxy: SharkGalaxyState, cgal=0.49, interaction=2.0):
    """Return the Cole-style remnant radius used by the instability map."""

    disk_mass_value = galaxy.disk_stars.mass + galaxy.disk_gas.mass
    bulge_mass_value = galaxy.bulge_stars.mass + galaxy.bulge_gas.mass
    disk_radius = disk_size(galaxy)
    bulge_radius = galaxy.bulge_gas.radius
    disk_energy = jnp.where(
        (disk_mass_value > 0.0) & (disk_radius > 0.0),
        cgal * disk_mass_value**2 / disk_radius,
        0.0,
    )
    bulge_energy = jnp.where(
        (bulge_mass_value > 0.0) & (bulge_radius > 0.0),
        cgal * bulge_mass_value**2 / bulge_radius,
        0.0,
    )
    interaction_energy = jnp.where(
        disk_radius + bulge_radius > 0.0,
        interaction * disk_mass_value * bulge_mass_value / (disk_radius + bulge_radius),
        0.0,
    )
    return (
        cgal
        * (disk_mass_value + bulge_mass_value) ** 2
        / (disk_energy + bulge_energy + interaction_energy)
    )


def apply_disk_instability_event(
    state: SharkSystemState,
    *,
    stability_threshold=1.0,
    cgal=0.49,
    interaction=2.0,
):
    """Apply the exact finite disk-to-bulge transfer on the active branch.

    The subsequent starburst remains a continuous burst episode and is not
    hidden inside this event map.
    """

    diagnostic = toomre_stability_parameter(state.galaxy)
    triggered = diagnostic < stability_threshold

    def event(active_state):
        galaxy = active_state.galaxy
        radius = disk_instability_bulge_radius(galaxy, cgal, interaction)
        transferred_mass = galaxy.disk_stars.mass + galaxy.disk_gas.mass
        transferred_metals = galaxy.disk_stars.metals + galaxy.disk_gas.metals
        before_angular_momentum = (
            galaxy.disk_stars.angular_momentum
            + galaxy.disk_gas.angular_momentum
            + galaxy.bulge_stars.angular_momentum
            + galaxy.bulge_gas.angular_momentum
        )
        bulge_stars = _add_sized(
            galaxy.bulge_stars,
            galaxy.disk_stars.mass,
            galaxy.disk_stars.metals,
            galaxy.disk_stars.angular_momentum,
        )._replace(radius=radius)
        bulge_gas = _add_sized(
            galaxy.bulge_gas,
            galaxy.disk_gas.mass,
            galaxy.disk_gas.metals,
            galaxy.disk_gas.angular_momentum,
        )._replace(radius=radius)
        bulge_mass_value = bulge_stars.mass + bulge_gas.mass
        pseudo_specific_angular_momentum = jnp.sqrt(
            _GRAVITATIONAL_CONSTANT * bulge_mass_value * radius
        )
        bulge_stars = bulge_stars._replace(
            angular_momentum=bulge_stars.mass * pseudo_specific_angular_momentum
        )
        bulge_gas = bulge_gas._replace(
            angular_momentum=bulge_gas.mass * pseudo_specific_angular_momentum
        )
        zero_stars = galaxy.disk_stars._replace(
            mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0
        )
        zero_gas = galaxy.disk_gas._replace(mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0)
        tracker = _add_baryon(
            galaxy.instability_assembly_stars,
            galaxy.disk_stars.mass,
            galaxy.disk_stars.metals,
        )
        after_angular_momentum = bulge_stars.angular_momentum + bulge_gas.angular_momentum
        updated_galaxy = galaxy._replace(
            disk_stars=zero_stars,
            disk_gas=zero_gas,
            bulge_stars=bulge_stars,
            bulge_gas=bulge_gas,
            instability_assembly_stars=tracker,
        )
        updated = active_state._replace(galaxy=updated_galaxy)
        return DiskInstabilityResult(
            updated,
            jnp.asarray(True),
            transferred_mass,
            transferred_metals,
            after_angular_momentum - before_angular_momentum,
        )

    def no_event(inactive_state):
        zero = jnp.zeros_like(inactive_state.galaxy.disk_stars.mass)
        return DiskInstabilityResult(inactive_state, jnp.asarray(False), zero, zero, zero)

    return jax.lax.cond(triggered, event, no_event, state)


def remnant_radius(
    central_mass, satellite_mass, central_radius, satellite_radius, cgal=0.49, orbit=1.0
):
    """Return the non-dissipative Cole et al. merger remnant radius."""

    central_energy = jnp.where(
        (central_mass > 0.0) & (central_radius > 0.0), central_mass**2 / central_radius, 0.0
    )
    satellite_energy = jnp.where(
        (satellite_mass > 0.0) & (satellite_radius > 0.0),
        satellite_mass**2 / satellite_radius,
        0.0,
    )
    orbital_energy = jnp.where(
        central_radius + satellite_radius > 0.0,
        orbit / cgal * central_mass * satellite_mass / (central_radius + satellite_radius),
        0.0,
    )
    denominator = central_energy + satellite_energy + orbital_energy
    return jnp.where(denominator > 0.0, (central_mass + satellite_mass) ** 2 / denominator, 0.0)


def _composite_radius(galaxy):
    mass = galaxy_baryonic_mass(galaxy) - galaxy.black_hole.mass
    disk_mass_value = galaxy.disk_stars.mass + galaxy.disk_gas.mass
    weighted = (
        disk_mass_value * disk_size(galaxy)
        + (galaxy.bulge_stars.mass + galaxy.bulge_gas.mass) * galaxy.bulge_stars.radius
    )
    return _safe_ratio(weighted, mass)


def merger_bulge_radius(
    central: SharkGalaxyState,
    satellite: SharkGalaxyState,
    mass_ratio,
    gas_to_stellar_ratio,
    *,
    major_merger_ratio=0.25,
    minor_burst_ratio=0.1,
    gas_burst_ratio=0.3,
    cgal=0.49,
    orbit=1.0,
    gas_dissipation=1.0,
    dissipation_mass_ratio=0.3,
    enclosed_dark_matter_mass=0.0,
):
    """Compute the upstream merger-remnant radius from explicit structure."""

    major = mass_ratio >= major_merger_ratio
    gas_rich_minor = (mass_ratio >= minor_burst_ratio) & (gas_to_stellar_ratio > gas_burst_ratio)
    central_bulge_mass = central.bulge_stars.mass + central.bulge_gas.mass
    central_disk_gas = central.disk_gas.mass
    minor_mass = jnp.where(
        gas_rich_minor, central_bulge_mass + central_disk_gas, central_bulge_mass
    )
    minor_radius = jnp.where(
        minor_mass > 0.0,
        (
            bulge_size(central) * central_bulge_mass
            + jnp.where(gas_rich_minor, central_disk_gas * central.disk_gas.radius, 0.0)
        )
        / minor_mass,
        0.0,
    )
    central_baryons = galaxy_baryonic_mass(central) - central.black_hole.mass
    central_mass = jnp.where(major, central_baryons + enclosed_dark_matter_mass, minor_mass)
    central_radius = jnp.where(major, _composite_radius(central), minor_radius)
    satellite_mass = galaxy_baryonic_mass(satellite) - satellite.black_hole.mass
    radius = remnant_radius(
        central_mass,
        satellite_mass,
        central_radius,
        _composite_radius(satellite),
        cgal,
        orbit,
    )
    stellar_mass = galaxy_stellar_mass(central) + galaxy_stellar_mass(satellite)
    gas_mass = galaxy_gas_mass(central) + galaxy_gas_mass(satellite)
    gas_to_stars = _safe_ratio(gas_mass, stellar_mass)
    shrink = jnp.minimum(1.0 + gas_to_stars / gas_dissipation, 3.0)
    dissipative = (gas_dissipation > 0.0) & (mass_ratio > dissipation_mass_ratio)
    shrink = jnp.where((stellar_mass == 0.0) & (gas_mass > 0.0), 3.0, shrink)
    return jnp.where(dissipative & (gas_mass > 0.0), radius / shrink, radius)


def apply_galaxy_merger_event(
    central: SharkGalaxyState,
    satellite: SharkGalaxyState,
    remnant_bulge_radius,
    merged_black_hole_spin,
    *,
    major_merger_ratio=0.25,
    minor_burst_ratio=0.1,
    gas_burst_ratio=0.3,
):
    """Apply upstream's major/minor baryonic merger map on a fixed branch."""

    central_baryons = galaxy_baryonic_mass(central) - central.black_hole.mass
    satellite_baryons = galaxy_baryonic_mass(satellite) - satellite.black_hole.mass
    raw_ratio = _safe_ratio(satellite_baryons, central_baryons)
    mass_ratio = jnp.where(raw_ratio > 1.0, 1.0 / raw_ratio, raw_ratio)
    total_stars = galaxy_stellar_mass(central) + galaxy_stellar_mass(satellite)
    total_gas = galaxy_gas_mass(central) + galaxy_gas_mass(satellite)
    gas_to_stellar_ratio = jnp.where(total_stars > 0.0, total_gas / total_stars, 1.0)
    major = mass_ratio >= major_merger_ratio
    minor_burst = (mass_ratio >= minor_burst_ratio) & (gas_to_stellar_ratio > gas_burst_ratio)
    burst_triggered = major | minor_burst

    before_angular_momentum = (
        central.disk_stars.angular_momentum
        + central.disk_gas.angular_momentum
        + central.bulge_stars.angular_momentum
        + central.bulge_gas.angular_momentum
        + satellite.disk_stars.angular_momentum
        + satellite.disk_gas.angular_momentum
        + satellite.bulge_stars.angular_momentum
        + satellite.bulge_gas.angular_momentum
    )

    satellite_stars_mass = galaxy_stellar_mass(satellite)
    satellite_stars_metals = satellite.disk_stars.metals + satellite.bulge_stars.metals
    satellite_gas_mass = galaxy_gas_mass(satellite)
    satellite_gas_metals = satellite.disk_gas.metals + satellite.bulge_gas.metals
    satellite_gas_angular_momentum = (
        satellite.disk_gas.angular_momentum + satellite.bulge_gas.angular_momentum
    )

    major_bulge_stars = _add_sized(
        central.bulge_stars,
        central.disk_stars.mass + satellite_stars_mass,
        central.disk_stars.metals + satellite_stars_metals,
        central.disk_stars.angular_momentum
        + satellite.disk_stars.angular_momentum
        + satellite.bulge_stars.angular_momentum,
    )
    major_bulge_gas = _add_sized(
        central.bulge_gas,
        central.disk_gas.mass + satellite_gas_mass,
        central.disk_gas.metals + satellite_gas_metals,
        central.disk_gas.angular_momentum + satellite_gas_angular_momentum,
    )
    minor_bulge_stars = _add_sized(
        central.bulge_stars,
        satellite_stars_mass,
        satellite_stars_metals,
        satellite.disk_stars.angular_momentum + satellite.bulge_stars.angular_momentum,
    )
    minor_disk_gas = _add_sized(
        central.disk_gas,
        satellite_gas_mass,
        satellite_gas_metals,
        satellite_gas_angular_momentum,
    )
    minor_bulge_gas = jax.tree_util.tree_map(
        lambda burst_value, quiet_value: jnp.where(minor_burst, burst_value, quiet_value),
        _add_sized(
            central.bulge_gas,
            minor_disk_gas.mass,
            minor_disk_gas.metals,
            minor_disk_gas.angular_momentum,
        ),
        central.bulge_gas,
    )
    zero_disk_stars = central.disk_stars._replace(
        mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0
    )
    zero_disk_gas = central.disk_gas._replace(
        mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0
    )
    minor_disk_gas = jax.tree_util.tree_map(
        lambda value: jnp.where(minor_burst, 0.0, value), minor_disk_gas
    )
    disk_stars = jax.tree_util.tree_map(
        lambda major_value, minor_value: jnp.where(major, major_value, minor_value),
        zero_disk_stars,
        central.disk_stars,
    )
    disk_gas = jax.tree_util.tree_map(
        lambda major_value, minor_value: jnp.where(major, major_value, minor_value),
        zero_disk_gas,
        minor_disk_gas,
    )
    bulge_stars = jax.tree_util.tree_map(
        lambda major_value, minor_value: jnp.where(major, major_value, minor_value),
        major_bulge_stars,
        minor_bulge_stars,
    )._replace(radius=remnant_bulge_radius)
    bulge_gas = jax.tree_util.tree_map(
        lambda major_value, minor_value: jnp.where(major, major_value, minor_value),
        major_bulge_gas,
        minor_bulge_gas,
    )._replace(radius=remnant_bulge_radius)

    bulge_mass_value = bulge_stars.mass + bulge_gas.mass
    pseudo_specific_angular_momentum = jnp.where(
        bulge_mass_value > 0.0,
        jnp.sqrt(_GRAVITATIONAL_CONSTANT * bulge_mass_value * remnant_bulge_radius),
        0.0,
    )
    bulge_stars = bulge_stars._replace(
        angular_momentum=bulge_stars.mass * pseudo_specific_angular_momentum
    )
    bulge_gas = bulge_gas._replace(
        angular_momentum=bulge_gas.mass * pseudo_specific_angular_momentum
    )

    black_hole = central.black_hole._replace(
        mass=central.black_hole.mass + satellite.black_hole.mass,
        metals=central.black_hole.metals + satellite.black_hole.metals,
        hot_halo_accretion_rate=(
            central.black_hole.hot_halo_accretion_rate
            + satellite.black_hole.hot_halo_accretion_rate
        ),
        starburst_accretion_rate=(
            central.black_hole.starburst_accretion_rate
            + satellite.black_hole.starburst_accretion_rate
        ),
        assembly_mass=central.black_hole.assembly_mass + satellite.black_hole.mass,
        spin=merged_black_hole_spin,
    )
    merger_tracker = _add_baryon(
        central.merger_assembly_stars,
        jnp.where(major, central.disk_stars.mass + satellite_stars_mass, satellite_stars_mass),
        jnp.where(
            major,
            central.disk_stars.metals + satellite_stars_metals,
            satellite_stars_metals,
        ),
    )
    stripped_tracker = _add_baryon(
        central.tidally_stripped_stars,
        satellite.tidally_stripped_stars.mass,
        satellite.tidally_stripped_stars.metals,
    )
    updated = central._replace(
        disk_stars=disk_stars,
        disk_gas=disk_gas,
        bulge_stars=bulge_stars,
        bulge_gas=bulge_gas,
        black_hole=black_hole,
        merger_assembly_stars=merger_tracker,
        tidally_stripped_stars=stripped_tracker,
        galaxy_type=jnp.asarray(CENTRAL_GALAXY, dtype=jnp.int32),
    )
    after_angular_momentum = (
        disk_stars.angular_momentum
        + disk_gas.angular_momentum
        + bulge_stars.angular_momentum
        + bulge_gas.angular_momentum
    )
    return GalaxyMergerResult(
        updated,
        mass_ratio,
        gas_to_stellar_ratio,
        major,
        burst_triggered,
        after_angular_momentum - before_angular_momentum,
    )


def enclosed_exponential_mass(radius, mass, half_mass_radius):
    scale = half_mass_radius / 1.67
    normalized = radius / jnp.where(scale > 0.0, scale, 1.0)
    enclosed = mass * (1.0 - (1.0 + normalized) * jnp.exp(-normalized))
    return jnp.where((mass > 0.0) & (scale > 0.0), enclosed, 0.0)


def apply_halo_ram_pressure_stripping(
    satellite: SharkSystemState,
    central_subhalo: SharkSubhaloState,
    target_radius,
    infall_virial_radius,
):
    """Apply upstream's cumulative gradual hot/cold-halo stripping map."""

    subhalo = satellite.subhalo
    total_profile_mass = (
        subhalo.hot_halo_gas.mass + subhalo.cold_halo_gas.mass + subhalo.stripped_hot_halo_gas.mass
    )
    cumulative_target = total_profile_mass * (
        1.0 - jnp.power(jnp.asarray(target_radius) / infall_virial_radius, 2.0)
    )
    requested = jnp.maximum(cumulative_target - subhalo.stripped_hot_halo_gas.mass, 0.0)
    available = subhalo.hot_halo_gas.mass + subhalo.cold_halo_gas.mass
    realized = jnp.minimum(requested, available)
    hot_fraction = _safe_ratio(subhalo.hot_halo_gas.mass, available)
    hot, hot_transfer = proportional_transfer(subhalo.hot_halo_gas, realized * hot_fraction)
    cold, cold_transfer = proportional_transfer(
        subhalo.cold_halo_gas, realized * (1.0 - hot_fraction)
    )
    transferred_metals = hot_transfer.metals + cold_transfer.metals
    stripped = _add_baryon(subhalo.stripped_hot_halo_gas, realized, transferred_metals)
    central_hot = _add_rotating(central_subhalo.hot_halo_gas, realized, transferred_metals, 0.0)
    satellite_subhalo = subhalo._replace(
        hot_halo_gas=hot,
        cold_halo_gas=cold,
        stripped_hot_halo_gas=stripped,
        halo_stripping_radius=jnp.asarray(target_radius),
    )
    return StrippingResult(
        satellite._replace(subhalo=satellite_subhalo),
        central_subhalo._replace(hot_halo_gas=central_hot),
        realized,
        transferred_metals,
        jnp.asarray(target_radius),
    )


def apply_ism_ram_pressure_stripping(
    satellite: SharkSystemState,
    central_subhalo: SharkSubhaloState,
    target_radius,
):
    """Apply upstream's disk+bulge gas stripping outside a target radius."""

    galaxy = satellite.galaxy
    disk_requested = galaxy.disk_gas.mass - enclosed_exponential_mass(
        target_radius, galaxy.disk_gas.mass, galaxy.disk_gas.radius
    )
    bulge_requested = galaxy.bulge_gas.mass - enclosed_exponential_mass(
        target_radius, galaxy.bulge_gas.mass, galaxy.bulge_gas.radius
    )
    disk, disk_transfer = proportional_transfer(galaxy.disk_gas, disk_requested)
    bulge, bulge_transfer = proportional_transfer(galaxy.bulge_gas, bulge_requested)
    transferred_mass = disk_transfer.mass + bulge_transfer.mass
    transferred_metals = disk_transfer.metals + bulge_transfer.metals
    central_hot = _add_rotating(
        central_subhalo.hot_halo_gas, transferred_mass, transferred_metals, 0.0
    )
    tracker = _add_baryon(galaxy.ram_pressure_stripped_gas, transferred_mass, transferred_metals)
    updated_galaxy = galaxy._replace(
        disk_gas=disk,
        bulge_gas=bulge,
        ram_pressure_stripped_gas=tracker,
        ism_stripping_radius=jnp.asarray(target_radius),
    )
    return StrippingResult(
        satellite._replace(galaxy=updated_galaxy),
        central_subhalo._replace(hot_halo_gas=central_hot),
        transferred_mass,
        transferred_metals,
        jnp.asarray(target_radius),
    )


def tidal_stellar_retention_fraction(dark_matter_mass_ratio):
    """Return the Errani et al. stellar retention fraction used by SHARK."""

    ratio = jnp.clip(jnp.asarray(dark_matter_mass_ratio), 0.0, 1.0)
    return jnp.clip(2.0**3.57 * ratio**2.06 / (1.0 + ratio) ** 3.57, 0.0, 1.0)


def apply_tidal_stripping_to_target(
    satellite: SharkSystemState,
    central_subhalo: SharkSubhaloState,
    target_cumulative_stellar_loss,
):
    """Apply SHARK's disk-first tidal stripping and associated cold-gas loss."""

    galaxy = satellite.galaxy
    requested = jnp.maximum(
        jnp.asarray(target_cumulative_stellar_loss) - galaxy.tidally_stripped_stars.mass,
        0.0,
    )
    total_stars = galaxy_stellar_mass(galaxy)
    realized = jnp.minimum(requested, total_stars)
    disk_loss = jnp.minimum(realized, galaxy.disk_stars.mass)
    bulge_loss = jnp.maximum(realized - disk_loss, 0.0)
    disk_stars, disk_stellar_transfer = proportional_transfer(galaxy.disk_stars, disk_loss)
    disk_fraction = _safe_ratio(disk_loss, galaxy.disk_stars.mass)
    disk_gas, disk_gas_transfer = proportional_transfer(
        galaxy.disk_gas, disk_fraction * galaxy.disk_gas.mass
    )
    bulge_stars, bulge_stellar_transfer = proportional_transfer(galaxy.bulge_stars, bulge_loss)
    bulge_fraction = _safe_ratio(bulge_loss, galaxy.bulge_stars.mass)
    bulge_gas, bulge_gas_transfer = proportional_transfer(
        galaxy.bulge_gas, bulge_fraction * galaxy.bulge_gas.mass
    )
    stellar_metals = disk_stellar_transfer.metals + bulge_stellar_transfer.metals
    gas_mass = disk_gas_transfer.mass + bulge_gas_transfer.mass
    gas_metals = disk_gas_transfer.metals + bulge_gas_transfer.metals
    central_hot = _add_rotating(central_subhalo.hot_halo_gas, gas_mass, gas_metals, 0.0)
    central_stellar_halo = _add_baryon(central_subhalo.stellar_halo, realized, stellar_metals)
    tracker = _add_baryon(galaxy.tidally_stripped_stars, realized, stellar_metals)
    updated_galaxy = galaxy._replace(
        disk_stars=disk_stars,
        disk_gas=disk_gas,
        bulge_stars=bulge_stars,
        bulge_gas=bulge_gas,
        tidally_stripped_stars=tracker,
    )
    updated_central = central_subhalo._replace(
        hot_halo_gas=central_hot,
        stellar_halo=central_stellar_halo,
        mean_stellar_halo_progenitor_mass_numerator=(
            central_subhalo.mean_stellar_halo_progenitor_mass_numerator + total_stars * realized
        ),
    )
    return StrippingResult(
        satellite._replace(galaxy=updated_galaxy),
        updated_central,
        realized + gas_mass,
        stellar_metals + gas_metals,
        jnp.asarray(0.0),
    )


def decrement_merger_clock(galaxy: SharkGalaxyState, interval_gyr):
    """Evolve the type-2 merger clock and return the terminal-event flag."""

    is_type2 = galaxy.galaxy_type == TYPE2_GALAXY
    event = is_type2 & (galaxy.merger_clock < interval_gyr)
    clock = jnp.where(is_type2 & ~event, galaxy.merger_clock - interval_gyr, galaxy.merger_clock)
    return galaxy._replace(merger_clock=clock), event
