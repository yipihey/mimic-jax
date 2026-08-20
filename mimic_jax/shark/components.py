"""Complete immutable component state used by SHARK's hybrid evolution.

The upstream model owns baryons at three levels: galaxy, subhalo, and halo.
The compact :class:`~mimic_jax.shark.types.SharkState` mirrors the 19 entries
passed to upstream's ODE solver; the types below retain everything needed by
the finite maps surrounding that ODE.  Topology itself remains ordinary
Python metadata because mergers change collection sizes.

All angular momenta stored here are *total* angular momenta.  Upstream stores
specific angular momenta, but total values make conservative transfers
structural and avoid dividing by a reservoir while it is empty.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp

Array = Any

CENTRAL_GALAXY = 0
TYPE1_GALAXY = 1
TYPE2_GALAXY = 2
FLYBY_GALAXY = 3

CENTRAL_SUBHALO = 0
SATELLITE_SUBHALO = 1
FLYBY_SUBHALO = 2


class BaryonComponent(NamedTuple):
    """A mass reservoir and the metal mass carried by it."""

    mass: Array
    metals: Array


class RotatingBaryonComponent(NamedTuple):
    """A mass/metal reservoir with total angular momentum."""

    mass: Array
    metals: Array
    angular_momentum: Array


class SizedBaryonComponent(NamedTuple):
    """A rotating galaxy component with an upstream half-mass scale radius."""

    mass: Array
    metals: Array
    angular_momentum: Array
    radius: Array


class BlackHoleComponent(NamedTuple):
    """The complete upstream black-hole state used by the Lagos23 model."""

    mass: Array
    metals: Array
    hot_halo_accretion_rate: Array
    starburst_accretion_rate: Array
    assembly_mass: Array
    spin: Array


class SharkGalaxyState(NamedTuple):
    """Persistent baryonic and diagnostic state of one SHARK galaxy."""

    disk_stars: SizedBaryonComponent
    disk_gas: SizedBaryonComponent
    bulge_stars: SizedBaryonComponent
    bulge_gas: SizedBaryonComponent
    black_hole: BlackHoleComponent
    merger_burst_stars: BaryonComponent
    merger_assembly_stars: BaryonComponent
    instability_burst_stars: BaryonComponent
    instability_assembly_stars: BaryonComponent
    tidally_stripped_stars: BaryonComponent
    ram_pressure_stripped_gas: BaryonComponent
    stellar_mass_ever_formed: Array
    mass_weighted_formation_time: Array
    merger_clock: Array
    maximum_circular_velocity: Array
    ism_stripping_radius: Array
    heat_to_cooling_ratio: Array
    galaxy_type: Array


class SharkSubhaloState(NamedTuple):
    """Persistent baryonic and memory state owned by one SHARK subhalo."""

    hot_halo_gas: RotatingBaryonComponent
    cold_halo_gas: RotatingBaryonComponent
    ejected_gas: RotatingBaryonComponent
    lost_gas: BaryonComponent
    stripped_hot_halo_gas: BaryonComponent
    stellar_halo: BaryonComponent
    mean_stellar_halo_progenitor_mass_numerator: Array
    heating_radius: Array
    halo_stripping_radius: Array
    cooling_history_integral: Array
    subhalo_type: Array


class SharkHaloState(NamedTuple):
    """Mutable group-level state not supplied directly by the merger tree."""

    excess_jet_power: Array
    hydrostatic: Array


class SharkSystemState(NamedTuple):
    """One galaxy, its owning subhalo reservoirs, and host-halo memory."""

    galaxy: SharkGalaxyState
    subhalo: SharkSubhaloState
    halo: SharkHaloState


def _f64(value=0.0):
    return jnp.asarray(value, dtype=jnp.float64)


def baryon_component(mass=0.0, metals=0.0) -> BaryonComponent:
    """Construct a scalar mass/metal component."""

    return BaryonComponent(_f64(mass), _f64(metals))


def rotating_component(mass=0.0, metals=0.0, angular_momentum=0.0):
    """Construct a scalar rotating component using total angular momentum."""

    return RotatingBaryonComponent(_f64(mass), _f64(metals), _f64(angular_momentum))


def sized_component(mass=0.0, metals=0.0, angular_momentum=0.0, radius=0.0):
    """Construct a scalar galaxy component."""

    return SizedBaryonComponent(_f64(mass), _f64(metals), _f64(angular_momentum), _f64(radius))


def black_hole_component(
    mass=0.0,
    metals=0.0,
    hot_halo_accretion_rate=0.0,
    starburst_accretion_rate=0.0,
    assembly_mass=0.0,
    spin=0.0,
):
    """Construct a scalar black-hole component."""

    return BlackHoleComponent(
        _f64(mass),
        _f64(metals),
        _f64(hot_halo_accretion_rate),
        _f64(starburst_accretion_rate),
        _f64(assembly_mass),
        _f64(spin),
    )


def initial_shark_galaxy_state(**overrides) -> SharkGalaxyState:
    """Return an empty galaxy with selected complete fields replaced."""

    values = dict(
        disk_stars=sized_component(),
        disk_gas=sized_component(),
        bulge_stars=sized_component(),
        bulge_gas=sized_component(),
        black_hole=black_hole_component(),
        merger_burst_stars=baryon_component(),
        merger_assembly_stars=baryon_component(),
        instability_burst_stars=baryon_component(),
        instability_assembly_stars=baryon_component(),
        tidally_stripped_stars=baryon_component(),
        ram_pressure_stripped_gas=baryon_component(),
        stellar_mass_ever_formed=_f64(),
        mass_weighted_formation_time=_f64(),
        merger_clock=_f64(),
        maximum_circular_velocity=_f64(),
        ism_stripping_radius=_f64(),
        heat_to_cooling_ratio=_f64(),
        galaxy_type=jnp.asarray(CENTRAL_GALAXY, dtype=jnp.int32),
    )
    unknown = set(overrides) - set(values)
    if unknown:
        raise TypeError(f"Unknown SHARK galaxy fields: {sorted(unknown)}")
    values.update(overrides)
    return SharkGalaxyState(**values)


def initial_shark_subhalo_state(**overrides) -> SharkSubhaloState:
    """Return an empty subhalo baryon/memory state."""

    values = dict(
        hot_halo_gas=rotating_component(),
        cold_halo_gas=rotating_component(),
        ejected_gas=rotating_component(),
        lost_gas=baryon_component(),
        stripped_hot_halo_gas=baryon_component(),
        stellar_halo=baryon_component(),
        mean_stellar_halo_progenitor_mass_numerator=_f64(),
        heating_radius=_f64(),
        halo_stripping_radius=_f64(),
        cooling_history_integral=_f64(),
        subhalo_type=jnp.asarray(CENTRAL_SUBHALO, dtype=jnp.int32),
    )
    unknown = set(overrides) - set(values)
    if unknown:
        raise TypeError(f"Unknown SHARK subhalo fields: {sorted(unknown)}")
    values.update(overrides)
    return SharkSubhaloState(**values)


def initial_shark_halo_state(**overrides) -> SharkHaloState:
    """Return an empty host-halo memory state."""

    values = dict(excess_jet_power=_f64(), hydrostatic=jnp.asarray(False))
    unknown = set(overrides) - set(values)
    if unknown:
        raise TypeError(f"Unknown SHARK halo fields: {sorted(unknown)}")
    values.update(overrides)
    return SharkHaloState(**values)


def initial_shark_system_state(**overrides) -> SharkSystemState:
    """Return an empty complete galaxy/subhalo/halo state."""

    values = dict(
        galaxy=initial_shark_galaxy_state(),
        subhalo=initial_shark_subhalo_state(),
        halo=initial_shark_halo_state(),
    )
    unknown = set(overrides) - set(values)
    if unknown:
        raise TypeError(f"Unknown SHARK system fields: {sorted(unknown)}")
    values.update(overrides)
    return SharkSystemState(**values)


def galaxy_stellar_mass(galaxy: SharkGalaxyState):
    return galaxy.disk_stars.mass + galaxy.bulge_stars.mass


def galaxy_gas_mass(galaxy: SharkGalaxyState):
    return galaxy.disk_gas.mass + galaxy.bulge_gas.mass


def galaxy_baryonic_mass(galaxy: SharkGalaxyState):
    return galaxy_stellar_mass(galaxy) + galaxy_gas_mass(galaxy) + galaxy.black_hole.mass


def galaxy_metal_mass(galaxy: SharkGalaxyState):
    """Metals in physical galaxy reservoirs, excluding diagnostic trackers."""

    return (
        galaxy.disk_stars.metals
        + galaxy.disk_gas.metals
        + galaxy.bulge_stars.metals
        + galaxy.bulge_gas.metals
        + galaxy.black_hole.metals
    )


def system_baryonic_mass(state: SharkSystemState):
    """Mass in all modeled reservoirs, including explicit escaped material."""

    subhalo = state.subhalo
    return (
        galaxy_baryonic_mass(state.galaxy)
        + subhalo.hot_halo_gas.mass
        + subhalo.cold_halo_gas.mass
        + subhalo.ejected_gas.mass
        + subhalo.lost_gas.mass
        + subhalo.stellar_halo.mass
    )


def system_metal_mass(state: SharkSystemState):
    """Metal mass in all modeled physical reservoirs."""

    subhalo = state.subhalo
    return (
        galaxy_metal_mass(state.galaxy)
        + subhalo.hot_halo_gas.metals
        + subhalo.cold_halo_gas.metals
        + subhalo.ejected_gas.metals
        + subhalo.lost_gas.metals
        + subhalo.stellar_halo.metals
    )


def system_angular_momentum(state: SharkSystemState):
    """Total represented baryonic angular momentum.

    Bulge remnant construction and black-hole accretion have explicitly
    documented angular-momentum boundaries, so callers compare this ledger
    only across processes that claim AM conservation.
    """

    galaxy = state.galaxy
    subhalo = state.subhalo
    return (
        galaxy.disk_stars.angular_momentum
        + galaxy.disk_gas.angular_momentum
        + galaxy.bulge_stars.angular_momentum
        + galaxy.bulge_gas.angular_momentum
        + subhalo.hot_halo_gas.angular_momentum
        + subhalo.cold_halo_gas.angular_momentum
        + subhalo.ejected_gas.angular_momentum
    )
