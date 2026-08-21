"""Model-neutral catalogue summaries shared by SAGE16 and SHARK.

These ordinary NumPy reductions sit outside differentiable physics kernels.
They make selections, bins, units, and zero handling identical across model
adapters so a comparison cannot accidentally change the observable definition.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from mimic_jax.catalogue import ComparisonCatalogue


@dataclass(frozen=True)
class MassFunction:
    """Number density per logarithmic physical-mass interval."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    number_density: np.ndarray


@dataclass(frozen=True)
class BinnedRelation:
    """Median and central percentile interval in bins of a predictor."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class BinnedFraction:
    """Selected fraction and integer counts in bins of a predictor."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    selected_counts: np.ndarray
    fraction: np.ndarray


def _validated_vectors(predictor, response=None):
    x = np.asarray(predictor, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("predictor must be one-dimensional")
    if response is None:
        return x
    y = np.asarray(response, dtype=np.float64)
    if y.ndim != 1 or y.shape != x.shape:
        raise ValueError("response must be one-dimensional and match predictor")
    return x, y


def _validated_edges(bin_edges):
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be a strictly increasing one-dimensional array")
    return edges


def binned_relation(
    predictor,
    response,
    *,
    bin_edges,
    percentiles=(16.0, 84.0),
) -> BinnedRelation:
    """Return a finite-value median relation with explicit empty bins."""

    x, y = _validated_vectors(predictor, response)
    edges = _validated_edges(bin_edges)
    lower_percentile, upper_percentile = percentiles
    if not 0.0 <= lower_percentile <= 50.0 <= upper_percentile <= 100.0:
        raise ValueError("percentiles must bracket the median within [0, 100]")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    indices = np.searchsorted(edges, x, side="right") - 1
    number_of_bins = edges.size - 1
    counts = np.zeros(number_of_bins, dtype=np.int64)
    median = np.full(number_of_bins, np.nan)
    lower = np.full(number_of_bins, np.nan)
    upper = np.full(number_of_bins, np.nan)
    for index in range(number_of_bins):
        selected = y[indices == index]
        counts[index] = selected.size
        if selected.size:
            lower[index], median[index], upper[index] = np.percentile(
                selected, (lower_percentile, 50.0, upper_percentile)
            )
    return BinnedRelation(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * np.diff(edges),
        counts=counts,
        median=median,
        lower=lower,
        upper=upper,
    )


def binned_selected_fraction(predictor, selected, *, bin_edges) -> BinnedFraction:
    """Return a boolean-selected fraction with explicit empty bins."""

    x = _validated_vectors(predictor)
    mask = np.asarray(selected, dtype=bool)
    if mask.ndim != 1 or mask.shape != x.shape:
        raise ValueError("selected must be one-dimensional and match predictor")
    edges = _validated_edges(bin_edges)
    finite = np.isfinite(x)
    counts, _ = np.histogram(x[finite], bins=edges)
    selected_counts, _ = np.histogram(x[finite & mask], bins=edges)
    fraction = np.full(counts.shape, np.nan, dtype=np.float64)
    populated = counts > 0
    fraction[populated] = selected_counts[populated] / counts[populated]
    return BinnedFraction(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * np.diff(edges),
        counts=counts.astype(np.int64),
        selected_counts=selected_counts.astype(np.int64),
        fraction=fraction,
    )


def mass_function(
    physical_mass,
    *,
    volume_mpc_over_h_cubed: float,
    hubble_h: float,
    bin_edges: Sequence[float],
) -> MassFunction:
    """Return ``dn/dlog10(M)`` in physical ``Mpc^-3 dex^-1``.

    Input masses are physical ``Msun`` while the volume retains the common
    simulation convention ``(Mpc/h)^3``.  The explicit ``h^3`` conversion is
    therefore applied exactly once here.
    """

    mass = _validated_vectors(physical_mass)
    if np.any(~np.isfinite(mass)):
        raise ValueError("physical_mass must contain only finite values")
    edges = _validated_edges(bin_edges)
    if not np.isfinite(volume_mpc_over_h_cubed) or volume_mpc_over_h_cubed <= 0.0:
        raise ValueError("volume_mpc_over_h_cubed must be finite and positive")
    if not np.isfinite(hubble_h) or hubble_h <= 0.0:
        raise ValueError("hubble_h must be finite and positive")
    positive = mass > 0.0
    counts, _ = np.histogram(np.log10(mass[positive]), bins=edges)
    widths = np.diff(edges)
    return MassFunction(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * widths,
        counts=counts.astype(np.int64),
        number_density=counts / volume_mpc_over_h_cubed * hubble_h**3 / widths,
    )


def catalogue_mass_function(
    catalogue: ComparisonCatalogue,
    field: str,
    *,
    bin_edges: Sequence[float],
) -> MassFunction:
    """Evaluate one mass function through the canonical catalogue contract."""

    quantity = catalogue.field(field)
    if quantity.unit != "Msun":
        raise ValueError(f"Mass-function field {field!r} must use physical Msun")
    return mass_function(
        quantity.values,
        volume_mpc_over_h_cubed=catalogue.effective_volume_mpc_over_h_cubed,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def catalogue_cosmic_sfr_density(
    catalogue: ComparisonCatalogue,
    *,
    minimum_stellar_mass_msun: Optional[float] = None,
    maximum_stellar_mass_msun: Optional[float] = None,
) -> float:
    """Return SFR density in physical ``Msun yr^-1 Mpc^-3``.

    The optional stellar-mass selection is explicit because legacy SAGE plots
    impose one while the native SHARK summary often does not.
    """

    stellar_mass = catalogue.values("stellar_mass")
    sfr = catalogue.values("star_formation_rate")
    selected = np.isfinite(stellar_mass) & np.isfinite(sfr)
    if minimum_stellar_mass_msun is not None:
        selected &= stellar_mass >= minimum_stellar_mass_msun
    if maximum_stellar_mass_msun is not None:
        selected &= stellar_mass <= maximum_stellar_mass_msun
    physical_volume = catalogue.effective_volume_mpc_over_h_cubed / catalogue.hubble_h**3
    return float(np.sum(sfr[selected]) / physical_volume)


def catalogue_mass_density(
    catalogue: ComparisonCatalogue,
    field: str,
    *,
    minimum_stellar_mass_msun: Optional[float] = None,
    maximum_stellar_mass_msun: Optional[float] = None,
) -> float:
    """Return one canonical mass density in physical ``Msun Mpc^-3``."""

    quantity = catalogue.field(field)
    if quantity.unit != "Msun":
        raise ValueError(f"Mass-density field {field!r} must use physical Msun")
    if quantity.values.ndim != 1:
        raise ValueError(f"Mass-density field {field!r} must be scalar per galaxy")
    stellar_mass = catalogue.values("stellar_mass")
    selected = np.isfinite(stellar_mass) & np.isfinite(quantity.values)
    if minimum_stellar_mass_msun is not None:
        selected &= stellar_mass >= minimum_stellar_mass_msun
    if maximum_stellar_mass_msun is not None:
        selected &= stellar_mass <= maximum_stellar_mass_msun
    physical_volume = catalogue.effective_volume_mpc_over_h_cubed / catalogue.hubble_h**3
    return float(np.sum(quantity.values[selected]) / physical_volume)


def _catalogue_selection(catalogue: ComparisonCatalogue, centrals_only: bool) -> np.ndarray:
    if not centrals_only:
        return np.ones(catalogue.galaxy_count, dtype=bool)
    return catalogue.values("galaxy_type") == 0


def catalogue_log_relation(
    catalogue: ComparisonCatalogue,
    *,
    predictor_field: str,
    response_field: str,
    bin_edges: Sequence[float],
    centrals_only: bool = False,
) -> BinnedRelation:
    """Return a median log--log relation for positive canonical quantities."""

    selected = _catalogue_selection(catalogue, centrals_only)
    predictor_values = catalogue.values(predictor_field)[selected]
    response_values = catalogue.values(response_field)[selected]
    if predictor_values.ndim != 1 or response_values.ndim != 1:
        raise ValueError("Log-relation fields must be scalar per galaxy")
    predictor = np.full(predictor_values.shape, np.nan)
    response = np.full(response_values.shape, np.nan)
    predictor[predictor_values > 0.0] = np.log10(predictor_values[predictor_values > 0.0])
    response[response_values > 0.0] = np.log10(response_values[response_values > 0.0])
    return binned_relation(predictor, response, bin_edges=bin_edges)


def catalogue_quenched_fraction(
    catalogue: ComparisonCatalogue,
    *,
    bin_edges: Sequence[float],
    specific_sfr_threshold_per_year: float = 1.0e-11,
    centrals_only: bool = False,
) -> BinnedFraction:
    """Return a common sSFR-selected quenched fraction."""

    selected = _catalogue_selection(catalogue, centrals_only)
    stellar = catalogue.values("stellar_mass")[selected]
    sfr = catalogue.values("star_formation_rate")[selected]
    log_stellar = np.full(stellar.shape, np.nan)
    positive = stellar > 0.0
    log_stellar[positive] = np.log10(stellar[positive])
    specific_sfr = np.full(stellar.shape, np.inf)
    specific_sfr[positive] = sfr[positive] / stellar[positive]
    return binned_selected_fraction(
        log_stellar,
        specific_sfr < specific_sfr_threshold_per_year,
        bin_edges=bin_edges,
    )


def catalogue_cold_gas_fraction_relation(
    catalogue: ComparisonCatalogue,
    *,
    bin_edges: Sequence[float],
    centrals_only: bool = False,
) -> BinnedRelation:
    """Return ``M_cold/(M_cold+M_star)`` versus total stellar mass."""

    selected = _catalogue_selection(catalogue, centrals_only)
    stellar = catalogue.values("stellar_mass")[selected]
    cold = catalogue.values("cold_gas_mass")[selected]
    predictor = np.full(stellar.shape, np.nan)
    predictor[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    denominator = stellar + cold
    fraction = np.full(stellar.shape, np.nan)
    fraction[denominator > 0.0] = cold[denominator > 0.0] / denominator[denominator > 0.0]
    return binned_relation(predictor, fraction, bin_edges=bin_edges)


def catalogue_metallicity_relation(
    catalogue: ComparisonCatalogue,
    *,
    mass_field: str,
    metal_field: str,
    bin_edges: Sequence[float],
    centrals_only: bool = False,
) -> BinnedRelation:
    """Return log total metal mass fraction versus physical stellar mass."""

    selected = _catalogue_selection(catalogue, centrals_only)
    stellar = catalogue.values("stellar_mass")[selected]
    mass = catalogue.values(mass_field)[selected]
    metals = catalogue.values(metal_field)[selected]
    predictor = np.full(stellar.shape, np.nan)
    predictor[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    metallicity = np.full(stellar.shape, np.nan)
    positive = (mass > 0.0) & (metals > 0.0)
    metallicity[positive] = np.log10(metals[positive] / mass[positive])
    return binned_relation(predictor, metallicity, bin_edges=bin_edges)


def catalogue_black_hole_bulge_relation(
    catalogue: ComparisonCatalogue,
    *,
    bin_edges: Sequence[float],
    centrals_only: bool = False,
) -> BinnedRelation:
    """Return log black-hole mass versus log bulge stellar mass."""

    selected = _catalogue_selection(catalogue, centrals_only)
    bulge = catalogue.values("bulge_stellar_mass")[selected]
    black_hole = catalogue.values("black_hole_mass")[selected]
    predictor = np.full(bulge.shape, np.nan)
    response = np.full(black_hole.shape, np.nan)
    predictor[bulge > 0.0] = np.log10(bulge[bulge > 0.0])
    response[black_hole > 0.0] = np.log10(black_hole[black_hole > 0.0])
    return binned_relation(predictor, response, bin_edges=bin_edges)


def catalogue_stellar_to_halo_relation(
    catalogue: ComparisonCatalogue,
    *,
    bin_edges: Sequence[float],
    centrals_only: bool = True,
) -> BinnedRelation:
    """Return log stellar mass versus log host-halo mass."""

    selected = _catalogue_selection(catalogue, centrals_only)
    halo = catalogue.values("host_halo_mass")[selected]
    stellar = catalogue.values("stellar_mass")[selected]
    predictor = np.full(halo.shape, np.nan)
    response = np.full(stellar.shape, np.nan)
    predictor[halo > 0.0] = np.log10(halo[halo > 0.0])
    response[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    return binned_relation(predictor, response, bin_edges=bin_edges)
