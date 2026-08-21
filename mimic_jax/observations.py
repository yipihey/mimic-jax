"""Model-neutral observational products with explicit comparison conventions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Array = Any

BALDRY2008_DOI = "10.1111/j.1365-2966.2008.13348.x"
BALDRY2008_COMPLETENESS_LOG_MASS = 8.5
BALDRY2008_CHABRIER_SHIFT_DEX = -0.26


@dataclass(frozen=True)
class BinnedObservation:
    """One observed binned relation with explicit uncertainties and metadata."""

    coordinate: Array
    values: Array
    lower_errors: Array
    upper_errors: Array
    coordinate_name: str
    coordinate_unit: str
    value_name: str
    value_unit: str
    source: str
    doi: str


def load_baldry2008_stellar_mass_function(
    path,
    *,
    hubble_h: float,
    chabrier_imf: bool = True,
) -> BinnedObservation:
    """Load the Baldry et al. table used by the upstream SAGE example.

    ``hubble_h`` is a declared target convention for the comparison, not the
    cosmology of one model.  Masses lose their published ``h`` dependence and
    receive the same Salpeter-to-Chabrier shift used by MIMIC, while number
    densities are multiplied by ``h**3``.
    """

    if hubble_h <= 0.0:
        raise ValueError("hubble_h must be positive")
    table = np.loadtxt(Path(path), delimiter=",", comments="#", dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("The Baldry table must contain mass, density, and uncertainty")
    logarithmic_mass = np.log10(10.0 ** table[:, 0] / hubble_h**2)
    if chabrier_imf:
        logarithmic_mass = logarithmic_mass + BALDRY2008_CHABRIER_SHIFT_DEX
    values = table[:, 1] * hubble_h**3
    errors = table[:, 2] * hubble_h**3
    return BinnedObservation(
        coordinate=logarithmic_mass,
        values=values,
        lower_errors=errors,
        upper_errors=errors,
        coordinate_name="log10 stellar mass",
        coordinate_unit="dex(Msun)",
        value_name="stellar mass function",
        value_unit="Mpc^-3 dex^-1",
        source="Baldry, Glazebrook & Driver (2008)",
        doi=BALDRY2008_DOI,
    )
