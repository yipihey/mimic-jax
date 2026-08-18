"""Ordinary-Python loading and pure-JAX interpolation of the SAGE cooling tables."""

import math
from pathlib import Path
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16.precision import require_x64

Array = Any

COOLING_FILE_NAMES = (
    "stripped_mzero.cie",
    "stripped_m-30.cie",
    "stripped_m-20.cie",
    "stripped_m-15.cie",
    "stripped_m-10.cie",
    "stripped_m-05.cie",
    "stripped_m-00.cie",
    "stripped_m+05.cie",
)
METALLICITIES_FEH = (-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5)
COOLING_TABLE_SIZE = 91
LOG_TEMP_MIN = 4.0
LOG_TEMP_STEP = 0.05
TEMP_INDEX_MAX = COOLING_TABLE_SIZE - 1
Z_SUN = 0.02


class CoolingTables(NamedTuple):
    """JAX arrays read from the eight Sutherland-Dopita table files."""

    log_metallicities: Array
    log_cooling_rates: Array


def default_cooling_table_directory() -> Path:
    """Locate the model-owned cooling tables in a source checkout."""

    repository = Path(__file__).resolve().parents[2]
    return (
        repository
        / "models"
        / "sage16"
        / "modules"
        / "sage_calculate_cooling_budget"
        / "CoolFunctions"
    )


def load_cooling_tables(directory: Optional[Path] = None) -> CoolingTables:
    """Read upstream table column six with the same float32 input rounding as C."""

    require_x64()
    directory = default_cooling_table_directory() if directory is None else Path(directory)
    tables = []
    for file_name in COOLING_FILE_NAMES:
        path = directory / file_name
        try:
            values = np.loadtxt(path, dtype=np.float32, usecols=(5,))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read SAGE16 cooling table {path}: {error}") from error
        if values.shape != (COOLING_TABLE_SIZE,):
            raise ValueError(
                f"SAGE16 cooling table {path} has shape {values.shape}; "
                f"expected ({COOLING_TABLE_SIZE},)"
            )
        tables.append(values.astype(np.float64))

    log10_z_sun = math.log10(Z_SUN)
    metallicities = np.asarray(
        [metallicity + log10_z_sun for metallicity in METALLICITIES_FEH],
        dtype=np.float64,
    )
    return CoolingTables(
        log_metallicities=jnp.asarray(metallicities, dtype=jnp.float64),
        log_cooling_rates=jnp.asarray(np.stack(tables), dtype=jnp.float64),
    )


def _temperature_interpolation(log_temperature, table_index, tables: CoolingTables):
    log_temperature = jnp.maximum(log_temperature, LOG_TEMP_MIN)
    index = ((log_temperature - LOG_TEMP_MIN) * (1.0 / LOG_TEMP_STEP)).astype(jnp.int32)
    index = jnp.minimum(index, TEMP_INDEX_MAX - 1)
    log_temperature_index = LOG_TEMP_MIN + LOG_TEMP_STEP * index
    rate1 = tables.log_cooling_rates[table_index, index]
    rate2 = tables.log_cooling_rates[table_index, index + 1]
    return rate1 + (rate2 - rate1) * (1.0 / LOG_TEMP_STEP) * (
        log_temperature - log_temperature_index
    )


def metal_dependent_cooling_rate(log_temperature, log_metallicity, tables: CoolingTables):
    """Reproduce SAGE's piecewise bilinear interpolation and return Lambda in CGS."""

    require_x64()
    log_metallicity = jnp.clip(
        log_metallicity,
        tables.log_metallicities[0],
        tables.log_metallicities[-1],
    )
    table_index = jnp.searchsorted(
        tables.log_metallicities[1:],
        log_metallicity,
        side="left",
    )
    rate1 = _temperature_interpolation(log_temperature, table_index, tables)
    rate2 = _temperature_interpolation(log_temperature, table_index + 1, tables)
    metallicity1 = tables.log_metallicities[table_index]
    metallicity2 = tables.log_metallicities[table_index + 1]
    log_rate = rate1 + (rate2 - rate1) / (metallicity2 - metallicity1) * (
        log_metallicity - metallicity1
    )
    return jnp.power(10.0, log_rate)
