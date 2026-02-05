#!/usr/bin/env python3
"""
Plotting Module for ZL+vMIT Hybrid EOS
======================================

Simple plotting utilities for use with EOSCollection from zlvmit_table_reader.

USAGE EXAMPLE
-------------
    from zlvmit_table_reader import EOSCollection
    from zlvmit_plot_results import (
        plot_betaeq, plot_fixedyc, plot_composition, plot_mixed_phase_boundaries,
        setup_matplotlib_style, ETA_STYLES, N_SAT
    )

    # Load EOS data
    eos = EOSCollection("output/zlvmit_eos", eq_mode="beta")

    # Create a simple plot
    fig, ax = plt.subplots()
    nB_values = np.linspace(0.1, 1.5, 100)
    plot_betaeq(ax, eos, 'chi', nB_values, T=0, eta_values=[0.0, 0.3, 1.0])
    plt.show()

AVAILABLE FUNCTIONS
-------------------
    plot_betaeq()              - Plot quantity vs nB for beta equilibrium
    plot_fixedyc()             - Plot quantity vs nB for fixed Y_C
    plot_composition()         - Plot particle fractions Y_i vs nB
    plot_mixed_phase_boundaries() - Plot phase diagram T(nB)
    setup_matplotlib_style()   - Configure matplotlib for publication-quality plots

AVAILABLE QUANTITIES (for plot_betaeq, plot_fixedyc)
----------------------------------------------------
    'chi'         - Quark volume fraction (0 = pure hadron, 1 = pure quark)
    'P_total'     - Total pressure [MeV/fm³]
    'e_total'     - Total energy density [MeV/fm³]
    'f_total'     - Total free energy density [MeV/fm³]
    's_total'     - Total entropy density [1/fm³]
    'F_specific'  - Free energy per baryon f/nB [MeV]
    'S_specific'  - Entropy per baryon s/nB
    'e_per_nB'    - Energy per baryon [MeV]

SPECIES GROUPS (for plot_composition)
-------------------------------------
    'basic'     - Y_p, Y_n, Y_u, Y_d, Y_s, Y_e (particle fractions)
    'hadrons'   - Y_p, Y_n only
    'quarks'    - Y_u, Y_d, Y_s only
    'charge'    - Y_C^H, Y_C^Q, Y_C^tot (charge fractions per phase)
    'electrons' - Y_e^(L,H), Y_e^(L,Q), Y_e^G (electron fractions)
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
#                              CONSTANTS
# ==============================================================================

N_SAT = 0.16  # Saturation density [fm^-3]


# ==============================================================================
#                              COLORS
# ==============================================================================

# Standard colors (RGB tuples matching Mathematica style)
STANDARD_RED    = (0.80, 0.25, 0.33)
STANDARD_GREEN  = (0.24, 0.60, 0.44)
STANDARD_BLUE   = (0.24, 0.60, 0.80)
STANDARD_GRAY   = (0.40, 0.40, 0.40)
STANDARD_ORANGE = (0.90, 0.40, 0.00)
STANDARD_CYAN   = (0.10, 0.60, 0.60)
STANDARD_PURPLE = (0.50, 0.35, 0.65)


# ==============================================================================
#                         LINE STYLES PER ETA
# ==============================================================================

# Use this dictionary to get consistent styling for different eta values.
# Example: style = ETA_STYLES[0.30]
#          ax.plot(x, y, **style)  # applies color, linestyle, linewidth, label

ETA_STYLES = {
    0.00: {'color': STANDARD_RED,    'linestyle': '-',  'linewidth': 4.0, 'label': r'$\eta = 0$'},
    0.10: {'color': STANDARD_ORANGE, 'linestyle': '--', 'linewidth': 4.0, 'label': r'$\eta = 0.1$'},
    0.30: {'color': STANDARD_GREEN,  'linestyle': '--', 'linewidth': 4.0, 'label': r'$\eta = 0.3$'},
    0.60: {'color': STANDARD_BLUE,   'linestyle': '--', 'linewidth': 4.0, 'label': r'$\eta = 0.6$'},
    1.00: {'color': STANDARD_GRAY,   'linestyle': '-',  'linewidth': 4.0, 'label': r'$\eta = 1$'},
}


# ==============================================================================
#                      QUANTITY LABELS AND UNITS
# ==============================================================================

# Maps quantity names to LaTeX labels and units for axis labeling.
# The plotting functions use this automatically.

QUANTITY_INFO = {
    'chi':        {'label': r'$\chi$',              'unit': ''},
    'P_total':    {'label': r'$P$',                 'unit': r'[MeV/fm$^3$]'},
    'e_total':    {'label': r'$\varepsilon$',       'unit': r'[MeV/fm$^3$]'},
    'f_total':    {'label': r'$f$',                 'unit': r'[MeV/fm$^3$]'},
    's_total':    {'label': r'$s$',                 'unit': r'[1/fm$^3$]'},
    'n_e_tot':    {'label': r'$n_e$',               'unit': r'[fm$^{-3}$]'},
    'F_specific': {'label': r'$F = f/n_B$',         'unit': r'[MeV]'},
    'S_specific': {'label': r'$S = s/n_B$',         'unit': ''},
    'e_per_nB':   {'label': r'$\varepsilon/n_B$',   'unit': r'[MeV]'},
}


# ==============================================================================
#                    SPECIES STYLES (for composition plots)
# ==============================================================================

# Style definitions for each particle species in composition plots.

SPECIES_STYLES = {
    # Total fractions (weighted average over phases)
    'Y_p_tot': {'color': STANDARD_RED,    'linestyle': '-',  'linewidth': 2.5, 'label': r'$Y_p$'},
    'Y_n_tot': {'color': STANDARD_BLUE,   'linestyle': '-',  'linewidth': 2.5, 'label': r'$Y_n$'},
    'Y_u_tot': {'color': STANDARD_CYAN,   'linestyle': '--', 'linewidth': 2.0, 'label': r'$Y_u$'},
    'Y_d_tot': {'color': STANDARD_ORANGE, 'linestyle': '--', 'linewidth': 2.0, 'label': r'$Y_d$'},
    'Y_s_tot': {'color': STANDARD_GREEN,  'linestyle': '--', 'linewidth': 2.0, 'label': r'$Y_s$'},
    'Y_e_tot': {'color': STANDARD_PURPLE, 'linestyle': ':',  'linewidth': 2.0, 'label': r'$Y_e$'},

    # Charge fractions per phase
    'Y_C_H':   {'color': STANDARD_RED,    'linestyle': '-',  'linewidth': 2.5, 'label': r'$Y_C^H$'},
    'Y_C_Q':   {'color': STANDARD_BLUE,   'linestyle': '-',  'linewidth': 2.5, 'label': r'$Y_C^Q$'},
    'Y_C_tot': {'color': STANDARD_GRAY,   'linestyle': '-',  'linewidth': 3.0, 'label': r'$Y_C^{tot}$'},

    # Electron fractions per phase (Y = n_e / n_B)
    'Y_eL_H':  {'color': STANDARD_RED,    'linestyle': ':',  'linewidth': 2.0, 'label': r'$Y_e^{L,H}$'},
    'Y_eL_Q':  {'color': STANDARD_BLUE,   'linestyle': ':',  'linewidth': 2.0, 'label': r'$Y_e^{L,Q}$'},
    'Y_eG':    {'color': STANDARD_GRAY,   'linestyle': ':',  'linewidth': 2.0, 'label': r'$Y_e^G$'},

    # Electron densities (n_e in fm^-3)
    'n_eL_H':  {'color': STANDARD_RED,    'linestyle': ':',  'linewidth': 2.0, 'label': r'$n_e^{L,H}$'},
    'n_eL_Q':  {'color': STANDARD_BLUE,   'linestyle': ':',  'linewidth': 2.0, 'label': r'$n_e^{L,Q}$'},
    'n_eG':    {'color': STANDARD_GRAY,   'linestyle': ':',  'linewidth': 2.0, 'label': r'$n_e^G$'},
}


# ==============================================================================
#                      PREDEFINED SPECIES GROUPS
# ==============================================================================

# Convenient groupings for plot_composition(species=...).
# Pass the group name as a string, e.g., plot_composition(..., species='basic')

SPECIES_GROUPS = {
    'basic':              ['Y_p_tot', 'Y_n_tot', 'Y_u_tot', 'Y_d_tot', 'Y_s_tot', 'Y_e_tot'],
    'hadrons':            ['Y_p_tot', 'Y_n_tot'],
    'quarks':             ['Y_u_tot', 'Y_d_tot', 'Y_s_tot'],
    'charge':             ['Y_C_H', 'Y_C_Q', 'Y_C_tot'],
    'electrons':          ['Y_eL_H', 'Y_eL_Q', 'Y_eG'],
    'electron_densities': ['n_eL_H', 'n_eL_Q', 'n_eG'],
}


# ==============================================================================
#                        MATPLOTLIB STYLE SETUP
# ==============================================================================

def setup_matplotlib_style():
    """
    Configure matplotlib for publication-quality plots with CMU Serif font.

    Call this once at the start of your notebook/script:
        from zlvmit_plot_results import setup_matplotlib_style
        setup_matplotlib_style()
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['axes.titlesize'] = 25


# ==============================================================================
#                          HELPER FUNCTIONS
# ==============================================================================

def _get_quantity_value(eos_collection, quantity, nB, T, eta, Y_C=None):
    """
    Get quantity value from EOSCollection, computing specific quantities on the fly.

    Handles 'F_specific', 'S_specific', 'e_per_nB' by dividing by nB.
    """
    if quantity == 'F_specific':
        f = eos_collection.get('f_total', nB, T=T, eta=eta, Y_C=Y_C)
        return f / nB if nB > 0 else np.nan
    elif quantity == 'S_specific':
        s = eos_collection.get('s_total', nB, T=T, eta=eta, Y_C=Y_C)
        return s / nB if nB > 0 else np.nan
    elif quantity == 'e_per_nB':
        e = eos_collection.get('e_total', nB, T=T, eta=eta, Y_C=Y_C)
        return e / nB if nB > 0 else np.nan
    else:
        return eos_collection.get(quantity, nB, T=T, eta=eta, Y_C=Y_C)


# ==============================================================================
#                          PLOTTING FUNCTIONS
# ==============================================================================

def plot_betaeq(ax, eos_collection, quantity, nB_values, T, eta_values,
                xlim=None, ylim=None):
    """
    Plot quantity vs nB for beta equilibrium mode.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    eos_collection : EOSCollection
        Loaded EOS data (must be eq_mode='beta').
    quantity : str
        What to plot: 'chi', 'P_total', 'e_total', 'f_total', 's_total',
        'F_specific', 'S_specific', 'e_per_nB'.
    nB_values : array-like
        Baryon densities [fm^-3].
    T : float
        Temperature [MeV].
    eta_values : list of float
        List of eta values to plot (e.g., [0.0, 0.3, 1.0]).
    xlim : tuple, optional
        X-axis limits (xmin, xmax).
    ylim : tuple, optional
        Y-axis limits (ymin, ymax).

    Example
    -------
        fig, ax = plt.subplots()
        nB = np.linspace(0.1, 1.5, 100)
        plot_betaeq(ax, eos, 'chi', nB, T=50, eta_values=[0.0, 0.3, 0.6, 1.0])
        plt.show()
    """
    nB_normalized = np.array(nB_values) / N_SAT

    for eta in eta_values:
        style = ETA_STYLES.get(eta, {'color': 'black', 'linestyle': '-', 'linewidth': 4.0})
        values = [_get_quantity_value(eos_collection, quantity, nB, T=T, eta=eta)
                  for nB in nB_values]
        ax.plot(nB_normalized, values,
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], label=style.get('label', f'η={eta}'))

    # Axis labels
    q_info = QUANTITY_INFO.get(quantity, {'label': quantity, 'unit': ''})
    ax.set_xlabel(r'$n_B/n_0$')
    ylabel = q_info['label'] + (f" {q_info['unit']}" if q_info['unit'] else "")
    ax.set_ylabel(ylabel)

    # Title and formatting
    ax.set_title(rf'$\beta$-eq., $T={T}$ MeV')
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_fixedyc(ax, eos_collection, quantity, nB_values, T, YC, eta_values,
                 xlim=None, ylim=None):
    """
    Plot quantity vs nB for fixed Y_C mode.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    eos_collection : EOSCollection
        Loaded EOS data (must be eq_mode='fixed_yc').
    quantity : str
        What to plot: 'chi', 'P_total', 'e_total', 'f_total', 's_total',
        'F_specific', 'S_specific', 'e_per_nB'.
    nB_values : array-like
        Baryon densities [fm^-3].
    T : float
        Temperature [MeV].
    YC : float
        Charge fraction Y_C.
    eta_values : list of float
        List of eta values to plot.
    xlim : tuple, optional
        X-axis limits.
    ylim : tuple, optional
        Y-axis limits.

    Example
    -------
        fig, ax = plt.subplots()
        nB = np.linspace(0.1, 1.5, 100)
        plot_fixedyc(ax, eos, 'P_total', nB, T=50, YC=0.5, eta_values=[0.0, 1.0])
        plt.show()
    """
    nB_normalized = np.array(nB_values) / N_SAT

    for eta in eta_values:
        style = ETA_STYLES.get(eta, {'color': 'black', 'linestyle': '-', 'linewidth': 4.0})
        values = [_get_quantity_value(eos_collection, quantity, nB, T=T, eta=eta, Y_C=YC)
                  for nB in nB_values]
        ax.plot(nB_normalized, values,
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], label=style.get('label', f'η={eta}'))

    # Axis labels
    q_info = QUANTITY_INFO.get(quantity, {'label': quantity, 'unit': ''})
    ax.set_xlabel(r'$n_B/n_0$')
    ylabel = q_info['label'] + (f" {q_info['unit']}" if q_info['unit'] else "")
    ax.set_ylabel(ylabel)

    # Title and formatting
    ax.set_title(rf'$Y_C={YC}$, $T={T}$ MeV')
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_composition(ax, eos_collection, nB_values, T, eta, species='basic',
                     Y_C=None, xlim=None, ylim=None, show_mixed_phase=True, title=None):
    """
    Plot particle fractions Y_i vs nB for a single (eta, T, Y_C) point.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    eos_collection : EOSCollection
        Loaded EOS data.
    nB_values : array-like
        Baryon densities [fm^-3].
    T : float
        Temperature [MeV].
    eta : float
        Single eta value.
    species : str or list
        Which species to plot:
        - 'basic': Y_p, Y_n, Y_u, Y_d, Y_s, Y_e
        - 'hadrons': Y_p, Y_n
        - 'quarks': Y_u, Y_d, Y_s
        - 'charge': Y_C^H, Y_C^Q, Y_C^tot
        - 'electrons': Y_e^(L,H), Y_e^(L,Q), Y_e^G
        - Or a list like ['Y_p_tot', 'Y_n_tot', 'Y_u_tot']
    Y_C : float, optional
        Charge fraction (for fixed_yc mode).
    xlim : tuple, optional
        X-axis limits.
    ylim : tuple, optional
        Y-axis limits.
    show_mixed_phase : bool
        If True, shade the mixed phase region (0 < chi < 1) in gray.
    title : str, optional
        Custom title (auto-generated if None).

    Example
    -------
        fig, ax = plt.subplots()
        nB = np.linspace(0.1, 1.5, 200)
        plot_composition(ax, eos, nB, T=50, eta=0.3, species='basic')
        plt.show()
    """
    # Resolve species group name to list of column names
    if isinstance(species, str):
        species_list = SPECIES_GROUPS.get(species, [species])
    else:
        species_list = species

    nB_normalized = np.array(nB_values) / N_SAT

    # Shade mixed phase region (where 0 < chi < 1)
    if show_mixed_phase:
        chi_values = []
        for nB in nB_values:
            try:
                chi = eos_collection.get('chi', nB, T=T, eta=eta, Y_C=Y_C)
                chi_values.append(chi)
            except:
                chi_values.append(np.nan)
        chi_values = np.array(chi_values)

        mixed_mask = (chi_values > 0) & (chi_values < 1)
        if np.any(mixed_mask):
            # Find contiguous mixed phase regions
            in_mixed = False
            start_idx = 0
            for i, is_mixed in enumerate(mixed_mask):
                if is_mixed and not in_mixed:
                    start_idx = i
                    in_mixed = True
                elif not is_mixed and in_mixed:
                    ax.axvspan(nB_normalized[start_idx], nB_normalized[i-1],
                               alpha=0.15, color='gray', zorder=0)
                    in_mixed = False
            if in_mixed:  # Handle case where mixed phase extends to end
                ax.axvspan(nB_normalized[start_idx], nB_normalized[-1],
                           alpha=0.15, color='gray', zorder=0)

    # Plot each species
    for sp in species_list:
        style = SPECIES_STYLES.get(sp, {'color': 'black', 'linestyle': '-',
                                         'linewidth': 2.0, 'label': sp})
        values = []
        for nB in nB_values:
            try:
                val = eos_collection.get(sp, nB, T=T, eta=eta, Y_C=Y_C)
                values.append(val)
            except:
                values.append(np.nan)
        values = np.array(values)

        valid = ~np.isnan(values)
        if np.any(valid):
            ax.plot(nB_normalized[valid], values[valid],
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=style['linewidth'], label=style['label'])

    # Labels and formatting
    ax.set_xlabel(r'$n_B/n_0$')
    ax.set_ylabel(r'$Y_i$')

    if title is not None:
        ax.set_title(title)
    else:
        if Y_C is not None:
            ax.set_title(rf'$\eta={eta}$, $Y_C={Y_C}$, $T={T}$ MeV')
        else:
            ax.set_title(rf'$\eta={eta}$, $\beta$-eq., $T={T}$ MeV')

    ax.legend(frameon=False, loc='best', ncol=2, fontsize=16)
    ax.grid(True, alpha=0.3, linestyle=':')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_mixed_phase_boundaries(ax, eos_collection, eta_values, Y_C=None,
                                normalize_nB=True, xlim=None, ylim=None,
                                fill_region=True, fill_alpha=0.15):
    """
    Plot T(nB) phase diagram showing mixed phase boundaries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    eos_collection : EOSCollection
        Loaded EOS data (with boundaries).
    eta_values : list of float
        List of eta values to plot.
    Y_C : float, optional
        Charge fraction (for fixed_yc mode, None for beta equilibrium).
    normalize_nB : bool
        If True, plot nB/n0; otherwise nB [fm^-3].
    xlim : tuple, optional
        X-axis limits.
    ylim : tuple, optional
        Y-axis limits.
    fill_region : bool
        If True, fill the mixed phase region with color.
    fill_alpha : float
        Transparency for filled region.

    Example
    -------
        fig, ax = plt.subplots()
        plot_mixed_phase_boundaries(ax, eos, eta_values=[0.0, 0.3, 0.6, 1.0])
        plt.show()
    """
    for eta in eta_values:
        # Check if boundary data exists for this eta (and Y_C for fixedyc mode)
        try:
            # Get boundary arrays - this will raise an error if not found
            T, nB_onset, nB_offset = eos_collection.get_boundary_arrays(eta, Y_C)
        except (ValueError, KeyError) as e:
            print(f"Warning: No phase boundaries for eta={eta}, Y_C={Y_C}: {e}")
            continue

        if len(T) == 0:
            print(f"Warning: No data for eta={eta}, Y_C={Y_C}")
            continue

        style = ETA_STYLES.get(eta, {'color': 'black', 'linestyle': '-', 'linewidth': 4.0})

        # Normalize density
        if normalize_nB:
            nB_onset = nB_onset / N_SAT
            nB_offset = nB_offset / N_SAT

        # Plot onset and offset boundaries
        ax.plot(nB_onset, T, color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], label=style.get('label', f'η={eta}'))
        ax.plot(nB_offset, T, color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'])

        # Fill mixed phase region
        if fill_region:
            ax.fill_betweenx(T, nB_onset, nB_offset, color=style['color'], alpha=fill_alpha)

    # Labels and formatting
    xlabel = r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$T$ [MeV]')
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
