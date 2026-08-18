"""Population summaries derived from ordinary SAGE16 catalogue fields.

These routines intentionally sit outside the differentiable physics kernels. They turn
already-evolved catalogues into familiar practitioner-facing observables without teaching the
reservoir model about binning, plotting, or report presentation.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StellarMassFunction:
    """A number density per logarithmic stellar-mass interval."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    number_density: np.ndarray


@dataclass(frozen=True)
class GroupBaryonInventory:
    """Baryonic reservoirs summed by FoF group and halo-mass bin."""

    halo_mass_bin_edges: np.ndarray
    halo_mass_bin_centres: np.ndarray
    group_counts: np.ndarray
    reservoir_names: Tuple[str, ...]
    reservoir_mass: np.ndarray
    universal_baryon_allotment: np.ndarray
    allotment_fractions: np.ndarray


def _finite_array(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _bin_edges(values, name):
    edges = _finite_array(values, name).astype(np.float64, copy=False)
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and contain at least two values")
    return edges


def stellar_mass_function(
    stellar_mass,
    *,
    volume_mpc_over_h_cubed: float,
    hubble_h: float,
    bin_edges: Sequence[float],
) -> StellarMassFunction:
    """Calculate ``phi(M*)`` using the same units as the MIMIC plotting registry.

    ``stellar_mass`` is supplied in SAGE's internal ``1e10 Msun/h`` units. Bin edges are
    ``log10(M*/Msun)``. The returned number density is in ``Mpc^-3 dex^-1`` and follows the
    existing MIMIC normalization ``counts / volume * h^3 / bin_width``.
    """

    mass = _finite_array(stellar_mass, "stellar_mass").astype(np.float64, copy=False)
    edges = _bin_edges(bin_edges, "bin_edges")
    if volume_mpc_over_h_cubed <= 0.0 or not np.isfinite(volume_mpc_over_h_cubed):
        raise ValueError("volume_mpc_over_h_cubed must be finite and positive")
    if hubble_h <= 0.0 or not np.isfinite(hubble_h):
        raise ValueError("hubble_h must be finite and positive")

    positive = mass > 0.0
    logarithmic_mass = np.log10(mass[positive] * 1.0e10 / hubble_h)
    counts, _ = np.histogram(logarithmic_mass, bins=edges)
    widths = np.diff(edges)
    density = counts / float(volume_mpc_over_h_cubed) * hubble_h**3 / widths
    return StellarMassFunction(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * widths,
        counts=counts.astype(np.int64),
        number_density=density,
    )


def safe_fractional_difference(candidate, reference):
    """Return ``(candidate-reference)/reference`` with invalid zero references masked."""

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    valid = np.isfinite(candidate) & np.isfinite(reference) & (reference != 0.0)
    result = np.full(candidate.shape, np.nan, dtype=np.float64)
    result[valid] = (candidate[valid] - reference[valid]) / reference[valid]
    return result, valid


def group_baryon_inventory(
    *,
    unique_galaxy_id,
    unique_central_galaxy_id,
    galaxy_type,
    central_halo_mass,
    reservoirs: Mapping[str, Sequence[float]],
    hubble_h: float,
    global_baryon_fraction: float,
    halo_mass_bin_edges: Sequence[float],
) -> GroupBaryonInventory:
    """Aggregate catalogue reservoirs into physical FoF-group baryon budgets.

    Every output galaxy is assigned to ``UniqueCentralGalaxyID``. Reservoirs are summed across
    the group, while the Type-0 record supplies the central halo mass. Fractions are normalized
    by ``GlobalBaryonFraction * Mvir`` in each halo-mass bin, so the stack answers where the
    group's universal baryon allotment resides rather than merely showing reservoir composition.
    """

    galaxy_id = _finite_array(unique_galaxy_id, "unique_galaxy_id").astype(np.int64, copy=False)
    central_id = _finite_array(unique_central_galaxy_id, "unique_central_galaxy_id").astype(
        np.int64, copy=False
    )
    galaxy_type = _finite_array(galaxy_type, "galaxy_type").astype(np.int64, copy=False)
    halo_mass = _finite_array(central_halo_mass, "central_halo_mass").astype(np.float64, copy=False)
    size = galaxy_id.size
    if any(array.size != size for array in (central_id, galaxy_type, halo_mass)):
        raise ValueError("group identity, type, and halo-mass arrays must have equal length")
    if hubble_h <= 0.0 or not np.isfinite(hubble_h):
        raise ValueError("hubble_h must be finite and positive")
    if global_baryon_fraction <= 0.0 or not np.isfinite(global_baryon_fraction):
        raise ValueError("global_baryon_fraction must be finite and positive")
    if not reservoirs:
        raise ValueError("at least one reservoir is required")

    reservoir_names = tuple(reservoirs)
    reservoir_values = []
    for name in reservoir_names:
        values = _finite_array(reservoirs[name], f"reservoir {name}").astype(np.float64, copy=False)
        if values.size != size:
            raise ValueError("all reservoir arrays must match the galaxy identity arrays")
        if np.any(values < 0.0):
            raise ValueError(f"reservoir {name} contains negative mass")
        reservoir_values.append(values)

    group_ids, group_index = np.unique(central_id, return_inverse=True)
    central_rows = np.flatnonzero(galaxy_type == 0)
    if central_rows.size != group_ids.size:
        raise ValueError("each output FoF group must contain exactly one Type-0 central")
    central_lookup = {int(galaxy_id[row]): int(row) for row in central_rows}
    if set(central_lookup) != set(int(identifier) for identifier in group_ids):
        raise ValueError("UniqueCentralGalaxyID values must identify the Type-0 records")

    central_rows_by_group = np.asarray(
        [central_lookup[int(identifier)] for identifier in group_ids], dtype=np.int64
    )
    group_halo_mass = halo_mass[central_rows_by_group]
    group_reservoir_mass = np.stack(
        [
            np.bincount(group_index, weights=values, minlength=group_ids.size)
            for values in reservoir_values
        ],
        axis=1,
    )

    edges = _bin_edges(halo_mass_bin_edges, "halo_mass_bin_edges")
    logarithmic_halo_mass = np.full(group_halo_mass.shape, np.nan, dtype=np.float64)
    positive_halo_mass = group_halo_mass > 0.0
    logarithmic_halo_mass[positive_halo_mass] = np.log10(
        group_halo_mass[positive_halo_mass] * 1.0e10 / hubble_h
    )
    bin_index = np.searchsorted(edges, logarithmic_halo_mass, side="right") - 1
    valid = positive_halo_mass & (bin_index >= 0) & (bin_index < edges.size - 1)
    selected_bins = bin_index[valid]
    number_of_bins = edges.size - 1
    group_counts = np.bincount(selected_bins, minlength=number_of_bins).astype(np.int64)
    mass_by_bin = np.stack(
        [
            np.bincount(
                selected_bins,
                weights=group_reservoir_mass[valid, reservoir],
                minlength=number_of_bins,
            )
            for reservoir in range(len(reservoir_names))
        ],
        axis=1,
    )
    allotment = np.bincount(
        selected_bins,
        weights=global_baryon_fraction * group_halo_mass[valid],
        minlength=number_of_bins,
    )
    fractions = np.full_like(mass_by_bin, np.nan, dtype=np.float64)
    populated = allotment > 0.0
    fractions[populated] = mass_by_bin[populated] / allotment[populated, None]
    return GroupBaryonInventory(
        halo_mass_bin_edges=edges,
        halo_mass_bin_centres=edges[:-1] + 0.5 * np.diff(edges),
        group_counts=group_counts,
        reservoir_names=reservoir_names,
        reservoir_mass=mass_by_bin,
        universal_baryon_allotment=allotment,
        allotment_fractions=fractions,
    )


def binned_fraction(values, selected, *, bin_edges):
    """Return counts and the selected fraction in each finite one-dimensional bin."""

    values = _finite_array(values, "values").astype(np.float64, copy=False)
    selected = np.asarray(selected, dtype=bool)
    if selected.ndim != 1 or selected.size != values.size:
        raise ValueError("selected must be a one-dimensional mask matching values")
    edges = _bin_edges(bin_edges, "bin_edges")
    total, _ = np.histogram(values, bins=edges)
    numerator, _ = np.histogram(values[selected], bins=edges)
    fraction = np.full(total.shape, np.nan, dtype=np.float64)
    populated = total > 0
    fraction[populated] = numerator[populated] / total[populated]
    return total.astype(np.int64), numerator.astype(np.int64), fraction


def binned_quantiles(values, measurements, *, bin_edges, quantiles=(0.16, 0.5, 0.84)):
    """Calculate finite measurement quantiles in bins without inventing empty-bin values."""

    values = _finite_array(values, "values").astype(np.float64, copy=False)
    measurements = np.asarray(measurements, dtype=np.float64)
    if measurements.ndim != 1 or measurements.size != values.size:
        raise ValueError("measurements must be one-dimensional and match values")
    edges = _bin_edges(bin_edges, "bin_edges")
    quantiles = _finite_array(quantiles, "quantiles").astype(np.float64, copy=False)
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie between zero and one")
    result = np.full((quantiles.size, edges.size - 1), np.nan, dtype=np.float64)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= lower) & (values < upper) & np.isfinite(measurements)
        counts[index] = np.count_nonzero(mask)
        if counts[index]:
            result[:, index] = np.quantile(measurements[mask], quantiles)
    return counts, result
