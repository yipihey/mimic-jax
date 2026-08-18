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


class ReionizationResult(NamedTuple):
    state: GalaxyState
    modifier: Array


class InfallBudgetTransfer(NamedTuple):
    """Snapshot budget and ownership changes from ``sage_prepare_infall_budget``."""

    satellite_ejected_to_central: Array
    satellite_ejected_metals_to_central: Array
    satellite_ics_to_central: Array
    satellite_ics_metals_to_central: Array
    target_baryons: Array
    group_baryons: Array
    infalling_gas: Array


class InfallBudgetResult(NamedTuple):
    states: GalaxyState
    transfer: InfallBudgetTransfer


class InfallTransfer(NamedTuple):
    """External source/sink and reservoir removals from one infall substep."""

    requested: Array
    external_to_hot: Array
    ejected_to_external: Array
    hot_to_external: Array
    ejected_metals_to_external: Array
    hot_metals_to_external: Array
    unfulfilled_removal: Array


class InfallResult(NamedTuple):
    state: GalaxyState
    transfer: InfallTransfer


class SatelliteStrippingTransfer(NamedTuple):
    """Hot gas and metals stripped from one Type-1 satellite to its FoF central."""

    gas: Array
    metals: Array
    allowed_baryons: Array
    satellite_baryons_before: Array


class SatelliteStrippingResult(NamedTuple):
    satellite: GalaxyState
    central: GalaxyState
    transfer: SatelliteStrippingTransfer


class RadioModeHeatingTransfer(NamedTuple):
    """Coupled cooling suppression, BH growth, and heating from radio-mode AGN."""

    cooling_before: Array
    cooling_after_prior_heating: Array
    accretion_rate: Array
    black_hole_accreted: Array
    hot_metals_accreted: Array
    heating_mass: Array
    heating_radius_before: Array
    heating_radius_after: Array


class RadioModeHeatingResult(NamedTuple):
    state: GalaxyState
    transfer: RadioModeHeatingTransfer


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


class UpstreamCentralStepDiagnostics(NamedTuple):
    """Ordered diagnostics for the implemented upstream central-galaxy slice."""

    infall: InfallTransfer
    cooling_budget: CoolingBudget
    radio_mode: RadioModeHeatingTransfer
    cooling: CoolingTransfer
    reincorporation: ReincorporationTransfer
    star_formation: StarFormationTransfer
    enrichment: MetalEnrichmentTransfer


class UpstreamCentralHistoryResult(NamedTuple):
    """History emitted by the faithful sequential update of the implemented slice."""

    final_state: GalaxyState
    states: GalaxyState
    diagnostics: UpstreamCentralStepDiagnostics
