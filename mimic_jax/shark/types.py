"""Immutable state and rate metadata for the SHARK baryon-cycle ODE.

The field order is the exact order used by ``BasicPhysicalModel<19>`` in
upstream SHARK.  Keeping it explicit makes C++/JAX oracle fixtures readable
and prevents array indices from becoming an undocumented second schema.
"""

from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp

Array = Any

SHARK_UPSTREAM_REPOSITORY = "https://github.com/ICRAR/shark"
SHARK_UPSTREAM_REVISION = "5af50d8fa7a040883409b10171c645e1db4e5fb2"
SHARK_RELEASE_BASELINE = "v2.0.0"

SHARK_ODE_STATE_NAMES = (
    "stellar_mass",
    "cold_gas",
    "cold_halo_gas",
    "hot_halo_gas",
    "ejected_gas",
    "lost_gas",
    "stellar_metals",
    "cold_gas_metals",
    "cold_halo_gas_metals",
    "hot_halo_gas_metals",
    "ejected_gas_metals",
    "lost_gas_metals",
    "formed_stellar_mass",
    "formed_stellar_metals",
    "stellar_angular_momentum",
    "cold_gas_angular_momentum",
    "cold_halo_angular_momentum",
    "hot_halo_angular_momentum",
    "ejected_angular_momentum",
)

SHARK_ODE_STATE_DESCRIPTIONS = {
    "stellar_mass": "Stellar mass of the disk or burst component being evolved.",
    "cold_gas": "Cold interstellar gas of that disk or burst component.",
    "cold_halo_gas": "Halo gas already selected to cool onto the galaxy.",
    "hot_halo_gas": "Quasi-hydrostatic hot-halo gas.",
    "ejected_gas": "Gas ejected from the halo by stellar feedback.",
    "lost_gas": "Gas lost from the tracked halo by QSO feedback.",
    "stellar_metals": "Metals locked in the stellar component.",
    "cold_gas_metals": "Metals in the cold interstellar gas.",
    "cold_halo_gas_metals": "Metals in the cooling halo-gas reservoir.",
    "hot_halo_gas_metals": "Metals in the hot-halo gas.",
    "ejected_gas_metals": "Metals in ejected gas.",
    "lost_gas_metals": "Metals in QSO-lost gas.",
    "formed_stellar_mass": "Episode-integrated stellar mass formed before recycling.",
    "formed_stellar_metals": "Episode-integrated pre-existing metals in formed stars.",
    "stellar_angular_momentum": "Total angular momentum of the stellar component.",
    "cold_gas_angular_momentum": "Total angular momentum of cold interstellar gas.",
    "cold_halo_angular_momentum": "Total angular momentum of cooling halo gas.",
    "hot_halo_angular_momentum": "Total angular momentum of hot-halo gas.",
    "ejected_angular_momentum": "Total angular momentum of ejected gas.",
}


class SharkState(NamedTuple):
    """The complete upstream 19-variable disk/starburst ODE state.

    Masses are in SHARK's comoving ``Msun/h`` convention, time is Gyr,
    metal fields are metal masses, and angular-momentum fields are total
    (mass times specific) angular momenta.  The state is a JAX PyTree.
    """

    stellar_mass: Array
    cold_gas: Array
    cold_halo_gas: Array
    hot_halo_gas: Array
    ejected_gas: Array
    lost_gas: Array
    stellar_metals: Array
    cold_gas_metals: Array
    cold_halo_gas_metals: Array
    hot_halo_gas_metals: Array
    ejected_gas_metals: Array
    lost_gas_metals: Array
    formed_stellar_mass: Array
    formed_stellar_metals: Array
    stellar_angular_momentum: Array
    cold_gas_angular_momentum: Array
    cold_halo_angular_momentum: Array
    hot_halo_angular_momentum: Array
    ejected_angular_momentum: Array


class SharkFlowParameters(NamedTuple):
    """Parameters used directly by SHARK's 19-variable flow assembly."""

    recycle_fraction: Array
    yield_mass_fraction: Array
    evolving_yield: Array
    pre_enrichment_metallicity: Array


class SharkFlowRates(NamedTuple):
    """Instantaneous physical rates/loadings consumed by the flow assembly.

    ``stellar_*`` and ``qso_*`` fields are dimensionless mass-loading
    factors.  The angular-momentum loadings multiply
    ``star_formation_angular_momentum_rate``, exactly as in upstream SHARK.
    """

    cooling: Array
    star_formation: Array
    star_formation_angular_momentum: Array
    stellar_reheating_loading: Array
    stellar_ejection_loading: Array
    angular_momentum_reheating_loading: Array
    angular_momentum_ejection_loading: Array
    qso_reheating_loading: Array
    qso_ejection_loading: Array
    cooling_metallicity: Array
    cooling_specific_angular_momentum: Array


class SharkRhsResult(NamedTuple):
    """Derivative together with the named rates and derived local quantities."""

    derivative: SharkState
    rates: SharkFlowRates
    cold_gas_metallicity: Array
    effective_yield: Array


class SharkContinuousState(NamedTuple):
    """Augmented state for continuous flows and hybrid AGN memory.

    Upstream's 19-variable ODE omits the black hole and heating history because
    they are updated immediately before/after it.  Promoting them here makes
    hot-mode growth Markovian while retaining the heating radius as a projected
    hybrid state rather than inventing a smooth evolution law.
    """

    reservoirs: SharkState
    black_hole_mass: Array
    black_hole_metals: Array
    black_hole_spin: Array
    heating_radius: Array
    excess_jet_power: Array


class SharkAugmentedFlowRates(NamedTuple):
    """Rates consumed by the augmented continuous SHARK state."""

    reservoirs: SharkFlowRates
    hot_halo_black_hole_accretion: Array
    reincorporation: Array


def initial_shark_state(**overrides: float) -> SharkState:
    """Return an all-zero float64 ODE state with selected fields overridden."""

    unknown = set(overrides) - set(SHARK_ODE_STATE_NAMES)
    if unknown:
        raise TypeError(f"Unknown SHARK ODE state fields: {sorted(unknown)}")
    values: Dict[str, Array] = {
        name: jnp.asarray(overrides.get(name, 0.0), dtype=jnp.float64)
        for name in SHARK_ODE_STATE_NAMES
    }
    return SharkState(**values)


def initial_shark_continuous_state(
    *,
    reservoirs: SharkState = None,
    black_hole_mass: float = 0.0,
    black_hole_metals: float = 0.0,
    black_hole_spin: float = 0.0,
    heating_radius: float = 0.0,
    excess_jet_power: float = 0.0,
) -> SharkContinuousState:
    """Construct the augmented continuous/hybrid state in float64."""

    return SharkContinuousState(
        reservoirs=initial_shark_state() if reservoirs is None else reservoirs,
        black_hole_mass=jnp.asarray(black_hole_mass, dtype=jnp.float64),
        black_hole_metals=jnp.asarray(black_hole_metals, dtype=jnp.float64),
        black_hole_spin=jnp.asarray(black_hole_spin, dtype=jnp.float64),
        heating_radius=jnp.asarray(heating_radius, dtype=jnp.float64),
        excess_jet_power=jnp.asarray(excess_jet_power, dtype=jnp.float64),
    )


def shark_flow_parameters(
    *,
    recycle_fraction: float = 0.4588,
    yield_mass_fraction: float = 0.02908,
    evolving_yield: bool = True,
    pre_enrichment_metallicity: float = 1.0e-4,
) -> SharkFlowParameters:
    """Construct the Lagos et al. (2023) sample flow parameters."""

    return SharkFlowParameters(
        recycle_fraction=jnp.asarray(recycle_fraction, dtype=jnp.float64),
        yield_mass_fraction=jnp.asarray(yield_mass_fraction, dtype=jnp.float64),
        evolving_yield=jnp.asarray(evolving_yield, dtype=jnp.bool_),
        pre_enrichment_metallicity=jnp.asarray(pre_enrichment_metallicity, dtype=jnp.float64),
    )


def zero_flow_rates() -> SharkFlowRates:
    """Return zero rates in the exact flow-rate schema."""

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return SharkFlowRates(*(zero for _ in SharkFlowRates._fields))


def stack_shark_states(states) -> SharkState:
    """Stack scalar states into a batched state accepted by ``jax.vmap`` kernels."""

    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
