"""Explicit finite-time transfer records emitted by SAGE16 process kernels."""

from typing import Any, NamedTuple

from mimic_jax.sage16.types import GalaxyState

Array = Any


class CoolingTransfer(NamedTuple):
    """Hot-to-cold transfer committed by ``sage_apply_cooling``."""

    gas: Array
    metals: Array


class CoolingResult(NamedTuple):
    state: GalaxyState
    transfer: CoolingTransfer


class CoolingBudget(NamedTuple):
    """Finite cooling budget and diagnostics calculated before AGN heating."""

    gas: Array
    radius: Array
    cooling_lambda: Array


class CoolingBudgetResult(NamedTuple):
    state: GalaxyState
    budget: CoolingBudget


class ReincorporationTransfer(NamedTuple):
    """Ejected-to-hot transfer committed by ``sage_reincorporation``."""

    gas: Array
    metals: Array


class ReincorporationResult(NamedTuple):
    state: GalaxyState
    transfer: ReincorporationTransfer


class StarFormationBudget(NamedTuple):
    """Transport fields passed between the fiducial SF and SN calculation modules."""

    NewStellarMass: Array
    SupernovaReheatedMass: Array
    SupernovaEjectedMass: Array


class StarFormationTransfer(NamedTuple):
    """Committed reservoir transfers from quiescent star formation and SN feedback."""

    formed_stars: Array
    locked_stars: Array
    cold_to_hot: Array
    hot_to_ejected: Array
    cold_metals_to_stars: Array
    cold_metals_to_hot: Array
    hot_metals_to_ejected: Array


class StarFormationApplyResult(NamedTuple):
    galaxy: GalaxyState
    central: GalaxyState
    transfer: StarFormationTransfer


class MetalEnrichmentTransfer(NamedTuple):
    """New metal source split between cold disk gas and the central hot halo."""

    produced_metals: Array
    new_metals_to_cold: Array
    new_metals_to_hot: Array


class MetalEnrichmentResult(NamedTuple):
    galaxy: GalaxyState
    central: GalaxyState
    transfer: MetalEnrichmentTransfer


class QuiescentStepResult(NamedTuple):
    galaxy: GalaxyState
    central: GalaxyState
    budget: StarFormationBudget
    transfer: StarFormationTransfer
    enrichment: MetalEnrichmentTransfer


class CentralStepDiagnostics(NamedTuple):
    """Explicit transfers emitted by one implemented central-galaxy substep."""

    cooling: CoolingTransfer
    reincorporation: ReincorporationTransfer
    star_formation: StarFormationTransfer
    enrichment: MetalEnrichmentTransfer


class CentralHistoryResult(NamedTuple):
    """Final state, per-epoch states, and per-epoch physical transfers."""

    final_state: GalaxyState
    states: GalaxyState
    diagnostics: CentralStepDiagnostics
