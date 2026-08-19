#!/usr/bin/env python

"""
Mimic Stellar Mass Function Plot

This module generates a stellar mass function plot from Mimic galaxy data.
Requires: StellarMass property (from galaxy physics modules)
"""

from pathlib import Path

import numpy as np
from figures import AXIS_LABEL_SIZE, get_mass_function_labels, get_stellar_mass_label, setup_legend
from matplotlib.ticker import MultipleLocator
from output_utils import (
    calculate_mass_function,
    check_field_has_values,
    check_required_fields,
    get_profile_axes,
    save_and_close_figure,
    setup_figure,
    validate_filtered_data,
)

# Physical limits for stellar mass functions
STELLAR_MASS_MAX = 13.0  # log10(Msun) - maximum stellar mass
BINWIDTH_DEX = 0.1  # Standard bin width in dex
PLOT_XLIM = (8.0, 12.5)  # Plot x-axis limits
PLOT_YLIM = (1.0e-6, 1.0e-1)  # Plot y-axis limits
SSFR_CUT = -11.0  # log10(sSFR/yr^-1) cut between red and blue galaxies


def plot(
    galaxies,
    volume,
    metadata,
    params,
    output_dir="plots",
    output_format=".png",
    verbose=False,
):
    """
    Create a stellar mass function plot.

    Args:
        galaxies: Galaxy data as a numpy recarray
        volume: Simulation volume in (Mpc/h)^3
        metadata: Dictionary with additional metadata
        params: Dictionary with Mimic parameters
        output_dir: Output directory for the plot
        output_format: File format for the output
        verbose: Whether to print verbose output

    Returns:
        Tuple of (plot_path, skip_message):
            - plot_path (str or None): Path to saved plot file if successful
            - skip_message (str or None): Reason for skipping if validation failed
    """
    # Extract necessary metadata
    hubble_h = metadata["hubble_h"]

    mass_min, mass_max, y_min, y_max = get_profile_axes(
        params, "stellar_mass_function", PLOT_XLIM, PLOT_YLIM, log_y=True
    )

    # Get WhichIMF from the parameters if available
    whichimf = 1  # Default to Chabrier
    if params and "WhichIMF" in params:
        whichimf = int(params["WhichIMF"])

    # Check for required and optional fields
    success, optional, msg = check_required_fields(
        galaxies,
        required_fields=["StellarMass"],
        optional_fields=["StarFormationRate"],
        plot_name="Stellar Mass Function",
    )

    if not success:
        return None, f"Required fields missing: {msg}"

    # Field-level validation: Check if StellarMass has any non-zero values
    has_values, count, msg = check_field_has_values(
        galaxies.StellarMass, "StellarMass", threshold=0.0
    )
    if not has_values:
        return None, f"Field validation failed: {msg}"

    # Select all galaxies with valid stellar mass
    w = np.where(galaxies.StellarMass > 0.0)[0]

    # Filter-level validation: Check if filtering produced results
    is_valid, skip_msg = validate_filtered_data(w, "Stellar Mass Function", verbose)
    if not is_valid:
        return None, skip_msg

    # NOW create the figure (only if validation passed)
    fig, ax = setup_figure()

    mass = np.log10(galaxies.StellarMass[w] * 1.0e10 / hubble_h)

    # Check if we have SFR for red/blue separation
    has_sfr = optional.get("StarFormationRate", False)

    # Calculate specific SFR for red/blue division (if SFR properties available)
    if has_sfr:
        sfr = galaxies.StarFormationRate[w]
        stellar_mass = galaxies.StellarMass[w] * 1.0e10 / hubble_h
        ssfr = sfr / stellar_mass

    # Calculate mass function
    xaxis, smf = calculate_mass_function(
        mass, volume, hubble_h, BINWIDTH_DEX, mass_min, STELLAR_MASS_MAX
    )

    # Print debugging info
    if verbose:
        print(f"  mi={mass_min}, ma={STELLAR_MASS_MAX}")
        print(f"  min mass={min(mass)}, max mass={max(mass)}")
        print(f"  volume={volume}, hubble_h={hubble_h}")
        print(f"  whichimf={whichimf}")
        print(f"  has_sfr={has_sfr}")

    # Plot stellar mass function
    ax.plot(xaxis, smf, "k-", label="Model - All")

    # Add red/blue separation if SFR properties are available
    if has_sfr:
        # Red galaxies (passive)
        w_red = np.where(ssfr < 10.0**SSFR_CUT)[0]
        mass_red = mass[w_red]
        _, smf_red = calculate_mass_function(
            mass_red, volume, hubble_h, BINWIDTH_DEX, mass_min, STELLAR_MASS_MAX
        )

        # Blue galaxies (star-forming)
        w_blue = np.where(ssfr >= 10.0**SSFR_CUT)[0]
        mass_blue = mass[w_blue]
        _, smf_blue = calculate_mass_function(
            mass_blue, volume, hubble_h, BINWIDTH_DEX, mass_min, STELLAR_MASS_MAX
        )

        # Plot red and blue galaxy populations
        ax.plot(xaxis, smf_red, "r:", lw=2, label="Model - Red")
        ax.plot(xaxis, smf_blue, "b:", lw=2, label="Model - Blue")

    # Baldry, Glazebrook & Driver (2008), MNRAS 388, 945: z~0.1 field stellar
    # mass function. Columns: log10(M*) (Salpeter IMF; shifted -0.26 dex to
    # Chabrier below), phi, phi uncertainty — h-dependence removed by the
    # hubble_h conversions that follow.
    data_path = Path(__file__).resolve().parents[4] / (
        "data/observations/baldry2008_stellar_mass_function.csv"
    )
    baldry = np.loadtxt(data_path, delimiter=",", comments="#", dtype=np.float32)

    # Convert Baldry data to appropriate units and IMF
    baldry_xval = np.log10(10 ** baldry[:, 0] / hubble_h / hubble_h)
    if whichimf == 1:  # Chabrier IMF
        baldry_xval = baldry_xval - 0.26  # Convert from Salpeter to Chabrier

    baldry_yvalU = (baldry[:, 1] + baldry[:, 2]) * hubble_h * hubble_h * hubble_h
    baldry_yvalL = (baldry[:, 1] - baldry[:, 2]) * hubble_h * hubble_h * hubble_h

    # Plot observational data with uncertainty band
    ax.fill_between(baldry_xval, baldry_yvalU, baldry_yvalL, facecolor="purple", alpha=0.25)

    # Add a legend entry for Baldry data
    ax.plot([], [], color="purple", alpha=0.3, lw=8, label="Baldry et al. 2008 (z=0.1)")

    # Customize the plot
    ax.set_yscale("log")
    ax.set_xlim(mass_min, mass_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_minor_locator(MultipleLocator(BINWIDTH_DEX))

    # Set labels with larger font sizes
    ax.set_ylabel(get_mass_function_labels(), fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel(get_stellar_mass_label(), fontsize=AXIS_LABEL_SIZE)

    # Add consistently styled legend
    setup_legend(ax, loc="lower left")

    # Save and close the figure
    plot_path = save_and_close_figure(
        fig, output_dir, "StellarMassFunction", output_format, verbose
    )
    return plot_path, None
