"""Pure state inheritance maps between SAGE16 merger-tree snapshots."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.precision import as_float64, require_x64
from mimic_jax.sage16.transfers import InheritanceResult, LocalCentralResult
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    InheritanceDescendant,
    initial_galaxy_state,
)


def reset_snapshot_accumulators(state: GalaxyState) -> GalaxyState:
    """Reset exactly the 13 ``init: repeat`` SAGE16 galaxy properties."""

    zero64 = jnp.asarray(0.0, dtype=jnp.float64)
    zero32 = jnp.asarray(0.0, dtype=jnp.float32)
    return state._replace(
        InfallingGas=zero64,
        CoolingGas=zero64,
        NewStellarMass=zero64,
        StarFormationRate=zero32,
        QuasarModeBHaccretionMass=zero32,
        SupernovaReheatedMass=zero64,
        SupernovaEjectedMass=zero64,
        Cooling=zero64,
        Heating=zero64,
        Rcool=zero64,
        CoolingLambda=zero64,
        SupernovaOutflowRate=zero32,
        UnstableDiskGasFraction=zero64,
    )


def _apply_descendant_properties(halo: HaloForcing, descendant: InheritanceDescendant):
    previous_mvir = halo.Mvir
    previous_vvir = halo.Vvir
    previous_vmax = halo.Vmax
    payload = descendant.payload
    grew = descendant.virial_mass > previous_mvir
    became_satellite = ~descendant.is_fof_central
    capture_infall = became_satellite & (halo.Type == 0)
    return halo._replace(
        Type=jnp.where(descendant.is_fof_central, 0, 1).astype(jnp.int32),
        Len=payload.Len,
        Mvir=descendant.virial_mass,
        deltaMvir=descendant.virial_mass - previous_mvir,
        Rvir=jnp.where(grew, descendant.virial_radius, halo.Rvir),
        Vvir=jnp.where(grew, descendant.virial_velocity, halo.Vvir),
        infallMvir=jnp.where(capture_infall, previous_mvir, halo.infallMvir),
        infallVvir=jnp.where(capture_infall, previous_vvir, halo.infallVvir),
        infallVmax=jnp.where(capture_infall, as_float64(previous_vmax), halo.infallVmax),
        Pos=payload.Pos,
        Vel=payload.Vel,
        VelDisp=payload.VelDisp,
        Vmax=payload.Vmax,
        Spin=payload.Spin,
        MostBoundID=payload.MostBoundID,
    )


def _make_orphan(halo: HaloForcing):
    was_central = halo.Type == 0
    return halo._replace(
        Type=jnp.asarray(2, dtype=jnp.int32),
        Len=jnp.asarray(0, dtype=jnp.int32),
        deltaMvir=-halo.Mvir,
        Mvir=jnp.asarray(0.0, dtype=jnp.float64),
        infallMvir=jnp.where(was_central, halo.Mvir, halo.infallMvir),
        infallVvir=jnp.where(was_central, halo.Vvir, halo.infallVvir),
        infallVmax=jnp.where(was_central, as_float64(halo.Vmax), halo.infallVmax),
    )


def inherit_progenitor(
    state: GalaxyState,
    halo: HaloForcing,
    descendant: InheritanceDescendant,
    source_time,
    is_main_branch,
) -> InheritanceResult:
    """Deep-copy semantics for one progenitor galaxy into a descendant slice."""

    require_x64()
    reset_state = reset_snapshot_accumulators(state)
    copied_halo = halo._replace(
        HaloNr=descendant.halo_nr,
        dT=as_float64(source_time) - descendant.current_time,
    )
    retained = copied_halo.Type != 3
    central_or_satellite = (copied_halo.Type == 0) | (copied_halo.Type == 1)

    def transition(_):
        return jax.lax.cond(
            jnp.asarray(is_main_branch, dtype=jnp.bool_),
            lambda _: _apply_descendant_properties(copied_halo, descendant),
            lambda _: _make_orphan(copied_halo),
            operand=None,
        )

    inherited_halo = jax.lax.cond(
        central_or_satellite,
        transition,
        lambda _: copied_halo,
        operand=None,
    )
    return InheritanceResult(reset_state, inherited_halo, retained, jnp.asarray(False))


def initialise_new_central(descendant: InheritanceDescendant) -> InheritanceResult:
    """Create the upstream default Type-0 galaxy when no central progenitor survives."""

    require_x64()
    payload = descendant.payload
    halo = payload._replace(
        SnapNum=descendant.current_snap - 1,
        Type=jnp.asarray(0, dtype=jnp.int32),
        HaloNr=descendant.halo_nr,
        UniqueGalaxyID=descendant.unique_galaxy_id,
        UniqueCentralGalaxyID=jnp.asarray(0, dtype=jnp.int64),
        dT=descendant.new_halo_dt,
        deltaMvir=jnp.asarray(0.0, dtype=jnp.float64),
        CentralMvir=jnp.asarray(0.0, dtype=jnp.float64),
        infallMvir=jnp.asarray(-1.0, dtype=jnp.float64),
        infallVvir=jnp.asarray(-1.0, dtype=jnp.float64),
        infallVmax=jnp.asarray(-1.0, dtype=jnp.float64),
    )
    return InheritanceResult(
        initial_galaxy_state(),
        halo,
        jnp.asarray(True),
        jnp.asarray(True),
    )


def set_local_central(halos: HaloForcing) -> LocalCentralResult:
    """Set one subhalo slice's ``CentralHalo`` links and report invalid topology."""

    candidates = (halos.Type == 0) | (halos.Type == 1)
    count = jnp.sum(candidates, dtype=jnp.int32)
    central_index = jnp.argmax(candidates).astype(jnp.int32)
    valid = count == 1
    central_links = jnp.where(
        valid,
        jnp.full_like(halos.CentralHalo, central_index),
        halos.CentralHalo,
    )
    return LocalCentralResult(
        halos._replace(CentralHalo=central_links),
        jnp.where(valid, central_index, -1),
        valid,
    )
