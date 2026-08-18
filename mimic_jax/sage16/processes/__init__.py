"""Small pure JAX functions corresponding to fiducial SAGE16 process modules."""

from mimic_jax.sage16.processes.cooling import apply_cooling
from mimic_jax.sage16.processes.cooling_budget import calculate_cooling_budget
from mimic_jax.sage16.processes.disk_instability import apply_disk_instability
from mimic_jax.sage16.processes.disk_radius import set_disk_scale_radius
from mimic_jax.sage16.processes.infall import apply_infall, prepare_infall_budget
from mimic_jax.sage16.processes.merger_clock import initialise_merger_clocks
from mimic_jax.sage16.processes.mergers import (
    apply_disruption_ownership_event,
    apply_merger_ownership_event,
    resolve_mergers_and_disruption,
)
from mimic_jax.sage16.processes.quasar_mode import apply_quasar_mode
from mimic_jax.sage16.processes.radio_mode_heating import apply_radio_mode_heating
from mimic_jax.sage16.processes.reincorporation import apply_reincorporation
from mimic_jax.sage16.processes.reionization import (
    apply_reionization,
    reionization_modifier,
)
from mimic_jax.sage16.processes.satellite_stripping import apply_satellite_stripping
from mimic_jax.sage16.processes.star_formation import (
    apply_metal_enrichment,
    apply_star_formation_supernova,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    quiescent_disk_step,
)
from mimic_jax.sage16.processes.starburst import (
    apply_collisional_starburst,
    apply_disk_instability_starburst,
)

__all__ = [
    "apply_cooling",
    "apply_collisional_starburst",
    "apply_disk_instability",
    "apply_disk_instability_starburst",
    "apply_infall",
    "apply_disruption_ownership_event",
    "apply_merger_ownership_event",
    "initialise_merger_clocks",
    "calculate_cooling_budget",
    "apply_metal_enrichment",
    "apply_reincorporation",
    "apply_reionization",
    "apply_quasar_mode",
    "apply_satellite_stripping",
    "apply_radio_mode_heating",
    "apply_star_formation_supernova",
    "calculate_star_formation_budget",
    "calculate_supernova_feedback_budget",
    "quiescent_disk_step",
    "prepare_infall_budget",
    "reionization_modifier",
    "resolve_mergers_and_disruption",
    "set_disk_scale_radius",
]
