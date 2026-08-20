"""One-interval SHARK hybrid scheduler with upstream process ordering.

The topology layer is deliberately ordinary Python: mergers add/remove galaxy
objects and cannot be represented by a fixed-shape ODE state without padding
or silently changing the algorithm.  Once an event schedule is known, every
reservoir map and flow solve is a pure JAX function and is differentiable on
that fixed branch.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp

from mimic_jax.shark.burst import (
    SharkStarburstResult,
    evolve_shark_starburst,
    lagos23_starburst_parameters,
)
from mimic_jax.shark.components import (
    SharkGalaxyState,
    SharkSubhaloState,
    SharkSystemState,
    galaxy_baryonic_mass,
    galaxy_gas_mass,
    galaxy_stellar_mass,
)
from mimic_jax.shark.hybrid import (
    DiskInstabilityResult,
    apply_disk_instability_event,
    apply_galaxy_merger_event,
    apply_halo_ram_pressure_stripping,
    apply_ism_ram_pressure_stripping,
    apply_tidal_stripping_to_target,
    merger_bulge_radius,
)
from mimic_jax.shark.interval import (
    Lagos23ModelParameters,
    SharkIntervalForcing,
    SharkIntervalResult,
    evolve_shark_continuous_interval,
    evolve_shark_reference_interval,
)
from mimic_jax.shark.prescriptions.agn import griffin19_merger_spin_upstream_rng


@dataclass(frozen=True)
class Lagos23MergerParameters:
    """Pinned merger and disk-instability settings from ``sample_lagos23``."""

    major_merger_ratio: float = 0.25
    minor_burst_ratio: float = 0.1
    gas_burst_ratio: float = 0.3
    orbit: float = 1.0
    cgal: float = 0.49
    gas_dissipation: float = 1.0
    dissipation_mass_ratio: float = 0.3
    disk_instability_threshold: float = 1.0
    disk_instability_interaction: float = 2.0


@dataclass(frozen=True)
class SharkEnvironmentSchedule:
    """Precomputed active-set targets for one satellite environment update.

    The topology/forcing layer supplies the active stripping radii after
    applying upstream's halo-profile root solve. ``None`` means that branch
    is inactive. The conservative transfer maps remain pure JAX functions;
    the cumulative tidal target follows upstream's disk-first convention.
    """

    central_subhalo: SharkSubhaloState
    halo_stripping_radius: Optional[float] = None
    infall_virial_radius: Optional[float] = None
    ism_stripping_radius: Optional[float] = None
    cumulative_tidal_stellar_loss: Optional[float] = None


@dataclass(frozen=True)
class SharkHybridEventSchedule:
    """Topology-changing events known at the beginning of one interval."""

    merging_satellites: Tuple[SharkGalaxyState, ...] = ()
    enclosed_dark_matter_mass: float = 0.0
    environment: Optional[SharkEnvironmentSchedule] = None


class SharkHybridIntervalResult(NamedTuple):
    """Complete state plus ordered event/flow diagnostics."""

    state: SharkSystemState
    central_subhalo: SharkSubhaloState
    interval: SharkIntervalResult
    merger_count: int
    merger_burst: SharkStarburstResult
    disk_instability: DiskInstabilityResult
    instability_burst: SharkStarburstResult
    environmentally_transferred_mass: object
    environmentally_transferred_metals: object


def lagos23_merger_parameters() -> Lagos23MergerParameters:
    """Return the pinned Lagos23 event parameters."""

    return Lagos23MergerParameters()


def _ratio(numerator, denominator):
    return jnp.where(denominator > 0.0, numerator / denominator, 0.0)


def _merge_one(
    state,
    satellite,
    forcing,
    merger_parameters,
    enclosed_dark_matter_mass,
):
    central = state.galaxy
    central_mass = galaxy_baryonic_mass(central) - central.black_hole.mass
    satellite_mass = galaxy_baryonic_mass(satellite) - satellite.black_hole.mass
    raw_ratio = _ratio(satellite_mass, central_mass)
    mass_ratio = jnp.where(raw_ratio > 1.0, 1.0 / raw_ratio, raw_ratio)
    total_stars = galaxy_stellar_mass(central) + galaxy_stellar_mass(satellite)
    total_gas = galaxy_gas_mass(central) + galaxy_gas_mass(satellite)
    gas_ratio = jnp.where(total_stars > 0.0, total_gas / total_stars, 1.0)
    radius = merger_bulge_radius(
        central,
        satellite,
        mass_ratio,
        gas_ratio,
        major_merger_ratio=merger_parameters.major_merger_ratio,
        minor_burst_ratio=merger_parameters.minor_burst_ratio,
        gas_burst_ratio=merger_parameters.gas_burst_ratio,
        cgal=merger_parameters.cgal,
        orbit=merger_parameters.orbit,
        gas_dissipation=merger_parameters.gas_dissipation,
        dissipation_mass_ratio=merger_parameters.dissipation_mass_ratio,
        enclosed_dark_matter_mass=enclosed_dark_matter_mass,
    )
    spin = griffin19_merger_spin_upstream_rng(
        central.black_hole.mass,
        satellite.black_hole.mass,
        central.black_hole.spin,
        satellite.black_hole.spin,
        forcing.galaxy_id,
        forcing.execution_seed,
    ).astype(central.black_hole.spin.dtype)
    merger = apply_galaxy_merger_event(
        central,
        satellite,
        radius,
        spin,
        major_merger_ratio=merger_parameters.major_merger_ratio,
        minor_burst_ratio=merger_parameters.minor_burst_ratio,
        gas_burst_ratio=merger_parameters.gas_burst_ratio,
    )
    return state._replace(galaxy=merger.central)


def _run_burst(state, forcing, parameters, *, from_merger, triggered, method, num_steps):
    return evolve_shark_starburst(
        state,
        redshift=forcing.redshift,
        duration_gyr=forcing.duration_gyr,
        virial_velocity=forcing.virial_velocity,
        subhalo_velocity=forcing.subhalo_velocity,
        galaxy_id=forcing.galaxy_id,
        execution_seed=forcing.execution_seed,
        model_parameters=parameters,
        burst_parameters=lagos23_starburst_parameters(),
        from_galaxy_merger=from_merger,
        triggered=triggered,
        method=method,
        num_steps=num_steps,
    )


def evolve_shark_hybrid_interval(
    state: SharkSystemState,
    forcing: SharkIntervalForcing,
    parameters: Lagos23ModelParameters,
    *,
    events: Optional[SharkHybridEventSchedule] = None,
    merger_parameters: Optional[Lagos23MergerParameters] = None,
    formulation="reference",
    method="rk4",
    num_steps=64,
) -> SharkHybridIntervalResult:
    """Apply SHARK's merger → instability → environment → flow sequence.

    Upstream triggers all due galaxy mergers and their bulge bursts before
    testing disk stability.  Satellite environment processing occurs inside
    cooling preparation, before infall/reincorporation/cooling.  The final
    flow can be either the upstream-order reference interval or the explicitly
    continuous interval; the choice is never implicit.
    """

    if events is None:
        events = SharkHybridEventSchedule()
    if merger_parameters is None:
        merger_parameters = Lagos23MergerParameters()

    current = state
    for satellite in events.merging_satellites:
        current = _merge_one(
            current,
            satellite,
            forcing,
            merger_parameters,
            events.enclosed_dark_matter_mass,
        )
    merger_burst = _run_burst(
        current,
        forcing,
        parameters,
        from_merger=True,
        triggered=bool(events.merging_satellites),
        method=method,
        num_steps=num_steps,
    )
    current = merger_burst.state

    instability = apply_disk_instability_event(
        current,
        stability_threshold=merger_parameters.disk_instability_threshold,
        cgal=merger_parameters.cgal,
        interaction=merger_parameters.disk_instability_interaction,
    )
    instability_burst = _run_burst(
        instability.state,
        forcing,
        parameters,
        from_merger=False,
        triggered=instability.triggered,
        method=method,
        num_steps=num_steps,
    )
    current = instability_burst.state

    central_subhalo = (
        current.subhalo if events.environment is None else events.environment.central_subhalo
    )
    transferred_mass = jnp.asarray(0.0, dtype=jnp.float64)
    transferred_metals = jnp.asarray(0.0, dtype=jnp.float64)
    environment = events.environment
    if environment is not None:
        if environment.halo_stripping_radius is not None:
            if environment.infall_virial_radius is None:
                raise ValueError("infall_virial_radius is required with halo_stripping_radius")
            result = apply_halo_ram_pressure_stripping(
                current,
                central_subhalo,
                environment.halo_stripping_radius,
                environment.infall_virial_radius,
            )
            current = result.satellite
            central_subhalo = result.central_subhalo
            transferred_mass = transferred_mass + result.transferred_mass
            transferred_metals = transferred_metals + result.transferred_metals
        if environment.ism_stripping_radius is not None:
            result = apply_ism_ram_pressure_stripping(
                current, central_subhalo, environment.ism_stripping_radius
            )
            current = result.satellite
            central_subhalo = result.central_subhalo
            transferred_mass = transferred_mass + result.transferred_mass
            transferred_metals = transferred_metals + result.transferred_metals
        if environment.cumulative_tidal_stellar_loss is not None:
            result = apply_tidal_stripping_to_target(
                current,
                central_subhalo,
                environment.cumulative_tidal_stellar_loss,
            )
            current = result.satellite
            central_subhalo = result.central_subhalo
            transferred_mass = transferred_mass + result.transferred_mass
            transferred_metals = transferred_metals + result.transferred_metals

    if formulation == "reference":
        interval = evolve_shark_reference_interval(
            current, forcing, parameters, method=method, num_steps=num_steps
        )
    elif formulation == "continuous":
        interval = evolve_shark_continuous_interval(
            current,
            forcing,
            parameters,
            method=method,
            num_substeps=num_steps,
        )
    else:
        raise ValueError("formulation must be 'reference' or 'continuous'")
    return SharkHybridIntervalResult(
        state=interval.state,
        central_subhalo=central_subhalo,
        interval=interval,
        merger_count=len(events.merging_satellites),
        merger_burst=merger_burst,
        disk_instability=instability,
        instability_burst=instability_burst,
        environmentally_transferred_mass=transferred_mass,
        environmentally_transferred_metals=transferred_metals,
    )
