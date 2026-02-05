"""
tov_solver.py
=============
TOV (Tolman-Oppenheimer-Volkoff) equation solver with:
- Crust merging (attach/interpolate modes)
- Baryonic mass computation
- Tidal deformability (Love number k2 and Λ)

Units: Natural units (MeV, fm, M_sun, km).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import os

# Import physical constants
from general_physics_constants import (
    M_sun_MeV, r_sun_fm, r_sun_km, rho_sol, G_natural, m_nucleon_MeV,
    n0_default, hc, MEV_FM3_TO_KM2_INV, MEV_FM3_TO_DYNE_CM2, MEV_FM3_TO_G_CM3
)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EOSTable_for_TOV:
    """EOS table with P, epsilon, nB columns."""
    P: np.ndarray       # Pressure [MeV/fm³]
    epsilon: np.ndarray # Energy density [MeV/fm³]
    nB: np.ndarray      # Baryon density [fm⁻³]
    
    def __post_init__(self):
        """Convert to numpy arrays."""
        self.P = np.asarray(self.P)
        self.epsilon = np.asarray(self.epsilon)
        self.nB = np.asarray(self.nB)
        
    @classmethod
    def from_file(cls, filepath: str, columns: Tuple[int, int, int] = (0, 1, 2),
                  skip_header: int = 0) -> 'EOSTable_for_TOV':
        """
        Load EOS from file.
        
        Args:
            filepath: Path to data file
            columns: Tuple of (P_col, epsilon_col, nB_col) indices
            skip_header: Number of header lines to skip
        """
        data = np.loadtxt(filepath, skiprows=skip_header)
        P_col, e_col, nB_col = columns
        return cls(
            P=data[:, P_col],
            epsilon=data[:, e_col],
            nB=data[:, nB_col]
        )


@dataclass
class TOVResult:
    """Result from a single TOV integration."""
    e_c: float          # Central energy density [MeV/fm³]
    n_c: float          # Central baryon density [fm⁻³]
    P_c: float          # Central pressure [MeV/fm³]
    R: float            # Radius [km]
    M: float            # Gravitational mass [M_sun]
    M_b: Optional[float] = None  # Baryonic mass [M_sun]
    k2: Optional[float] = None   # Love number
    Lambda: Optional[float] = None  # Tidal deformability


# =============================================================================
# CRUST HANDLING
# =============================================================================

# Default crust file paths
CRUST_PATHS = {
    'BPS': '/Users/mircoguerrini/Desktop/Research/Crust/BPST0.dat',
    'compose_sfho': '/Users/mircoguerrini/Desktop/Research/Compose/SFHO_Compose/eos.thermo.ns',
}


def load_crust_table(crust_name: str, custom_path: Optional[str] = None) -> EOSTable_for_TOV:
    """
    Load a crust EOS table.
    
    Args:
        crust_name: 'BPS', 'compose_sfho', or 'personalized'
        custom_path: Path to custom crust file (required if personalized)
        
    Returns:
        EOSTable_for_TOV with crust data
    """
    if crust_name == 'personalized':
        if custom_path is None:
            raise ValueError("Must provide custom_path for personalized crust")
        filepath = custom_path
        # Assume columns: P, epsilon, nB
        return EOSTable_for_TOV.from_file(filepath, columns=(0, 1, 2))
    
    elif crust_name == 'BPS':
        # BPS file format: P [km^-2], epsilon [km^-2], nB [fm⁻³]
        # Convert to MeV/fm³
        filepath = CRUST_PATHS['BPS']
        data = np.loadtxt(filepath)
        P_geo = data[:, 0]       # km^-2
        e_geo = data[:, 1]       # km^-2
        nB = data[:, 2]          # fm⁻³ 

        # Convert to MeV/fm³
        P_mev = P_geo / MEV_FM3_TO_KM2_INV      # km^-2 → MeV/fm³
        e_mev = e_geo / MEV_FM3_TO_KM2_INV         # km^-2 → MeV/fm³

        return EOSTable_for_TOV(P=P_mev, epsilon=e_mev, nB=nB)
    
    elif crust_name == 'compose_sfho':
        # Compose format
        filepath = CRUST_PATHS['compose_sfho']
        return _load_compose_crust(filepath)
    
    else:
        raise ValueError(f"Unknown crust: {crust_name}. Use 'BPS', 'compose_sfho', or 'personalized'")


def _load_compose_crust(filepath: str) -> EOSTable_for_TOV:
    """
    Load Compose format crust table.
    """
    ### TODO: Implement this function
    return EOSTable_for_TOV(P=0, epsilon=0, nB=0)


def add_crust(
    eos_table: EOSTable_for_TOV,
    crust_name: str = 'No',
    mode: str = 'attach',
    n_transition: Optional[float] = None,
    delta_n: float = 0.01,
    delta_P: float = 0.0, 
    custom_crust_path: Optional[str] = None,
    save_merged: bool = False,
    output_dir: Optional[str] = None,
    input_filename: Optional[str] = None,
    verbose: bool = False,
) -> EOSTable_for_TOV:
    """
    Add crust to EOS table.

    Args:
        eos_table: High-density EOS table
        crust_name: 'No', 'BPS', 'compose_sfho', or 'personalized'
        mode: 'attach', 'interpolate', or 'maxwell'
        n_transition: Transition density [fm⁻³] (if None, use crust max)
        delta_n: Width of interpolation region [fm⁻³] (for 'interpolate' mode)
        delta_P: Pressure smoothing width [MeV/fm³] (for 'maxwell' mode)
                 If delta_P=0, sharp Maxwell construction; if delta_P>0, smooth crossover
        custom_crust_path: Path to custom crust file
        save_merged: Whether to save merged table
        output_dir: Directory for output file
        input_filename: Base name for output file
        verbose: Print transition information

    Returns:
        Merged EOSTable_for_TOV
    """
    if crust_name == 'No':
        return eos_table

    # Load crust
    crust = load_crust_table(crust_name, custom_crust_path)


    if mode == 'attach':
        merged = _attach_crust(eos_table, crust, n_transition)
    elif mode == 'interpolate':
        merged = _interpolate_crust(eos_table, crust, n_transition, delta_n)
    elif mode == 'maxwell':
        merged = _interpolate_crust_maxwell(eos_table, crust, delta_P, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'attach', 'interpolate', or 'maxwell'")
    
    # Save if requested
    if save_merged and output_dir is not None:
        base = input_filename or "eos"
        base = os.path.splitext(os.path.basename(base))[0]
        outfile = os.path.join(output_dir, f"{base}_withcrust_{crust_name}_{mode}.dat")
        _save_eos_table(merged, outfile)
        print(f"Saved merged EOS to: {outfile}")
    
    return merged


def _attach_crust(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV, n_transition: float) -> EOSTable_for_TOV:
    """Simple attachment at transition density."""
    # Use crust below transition, EOS above
    crust_mask = crust.nB <= n_transition # generate a np list of bools 
    eos_mask = eos.nB > n_transition # generate a np list of bools 
    
    P = np.concatenate([crust.P[crust_mask], eos.P[eos_mask]])
    epsilon = np.concatenate([crust.epsilon[crust_mask], eos.epsilon[eos_mask]])
    nB = np.concatenate([crust.nB[crust_mask], eos.nB[eos_mask]])
    
    return EOSTable_for_TOV(P=P, epsilon=epsilon, nB=nB)


def _interpolate_crust(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV, n_transition: float,
                       delta_n: float) -> EOSTable_for_TOV:
    """
    Smooth tanh interpolation between crust and EOS using μB.

    Interpolates P and μB = (P + ε) / n_B, then computes ε = μB * n_B - P.
    This ensures thermodynamic consistency in the transition region.
    """
    n_low = n_transition - delta_n
    n_high = n_transition + delta_n

    # Compute baryon chemical potential μB = (P + ε) / n_B
    muB_crust = (crust.P + crust.epsilon) / crust.nB
    muB_eos = (eos.P + eos.epsilon) / eos.nB

    # Create interpolators for P and μB
    crust_P_interp = PchipInterpolator(crust.nB, crust.P, extrapolate=True)
    crust_muB_interp = PchipInterpolator(crust.nB, muB_crust, extrapolate=True)
    eos_P_interp = PchipInterpolator(eos.nB, eos.P, extrapolate=True)
    eos_muB_interp = PchipInterpolator(eos.nB, muB_eos, extrapolate=True)

    # Unified nB grid with dense sampling in transition region
    nB_crust_below = crust.nB[crust.nB < n_low]
    nB_eos_above = eos.nB[eos.nB > n_high]
    nB_transition = np.linspace(n_low, n_high, 50)
    unified_nB = np.concatenate([nB_crust_below, nB_transition, nB_eos_above])
    unified_nB = np.unique(unified_nB)  # Remove duplicates and sort

    # Blending function: f(n) = 0.5 * (1 + tanh((n - n_transition) / (delta_n/2)))
    def blend(n):
        return 0.5 * (1.0 + np.tanh((n - n_transition) / (delta_n / 2.0)))

    # Interpolate P and μB with blending
    P_merged = np.zeros_like(unified_nB)
    muB_merged = np.zeros_like(unified_nB)

    for i, n in enumerate(unified_nB):
        if n <= n_low:
            P_merged[i] = crust_P_interp(n)
            muB_merged[i] = crust_muB_interp(n)
        elif n >= n_high:
            P_merged[i] = eos_P_interp(n)
            muB_merged[i] = eos_muB_interp(n)
        else:
            f = blend(n)
            P_merged[i] = (1 - f) * crust_P_interp(n) + f * eos_P_interp(n)
            muB_merged[i] = (1 - f) * crust_muB_interp(n) + f * eos_muB_interp(n)

    # Compute ε from μB: ε = μB * n_B - P
    e_merged = muB_merged * unified_nB - P_merged

    return EOSTable_for_TOV(P=P_merged, epsilon=e_merged, nB=unified_nB)


def _interpolate_crust_maxwell(eos: EOSTable_for_TOV, crust: EOSTable_for_TOV,
                                delta_P: float = 0.0, verbose: bool = False) -> EOSTable_for_TOV:
    """
    Maxwell-style interpolation between crust and EOS in pressure space.

    Finds P_trans where μ_crust(P) = μ_eos(P), then interpolates:
    ε(P) = ½[1 - tanh((P - P_trans)/δP)] ε_crust(P) + ½[1 + tanh((P - P_trans)/δP)] ε_eos(P)

    Args:
        eos: High-density EOS table
        crust: Crust EOS table
        delta_P: Pressure smoothing width [MeV/fm³].
                 If 0, sharp Maxwell construction; if >0, smooth crossover.
        verbose: Print transition information

    Returns:
        Merged EOSTable_for_TOV with smooth (or sharp) transition
    """
    # Compute μB = (P + ε) / n_B for both phases
    muB_crust = (crust.P + crust.epsilon) / crust.nB
    muB_eos = (eos.P + eos.epsilon) / eos.nB

    # Create interpolators: μB(P) for both phases
    # Sort by pressure for interpolation
    idx_crust = np.argsort(crust.P)
    idx_eos = np.argsort(eos.P)

    P_crust_sorted = crust.P[idx_crust]
    muB_crust_sorted = muB_crust[idx_crust]
    e_crust_sorted = crust.epsilon[idx_crust]
    nB_crust_sorted = crust.nB[idx_crust]

    P_eos_sorted = eos.P[idx_eos]
    muB_eos_sorted = muB_eos[idx_eos]
    e_eos_sorted = eos.epsilon[idx_eos]
    nB_eos_sorted = eos.nB[idx_eos]

    # Interpolators for μB(P), ε(P), nB(P)
    muB_crust_of_P = PchipInterpolator(P_crust_sorted, muB_crust_sorted, extrapolate=True)
    muB_eos_of_P = PchipInterpolator(P_eos_sorted, muB_eos_sorted, extrapolate=True)
    e_crust_of_P = PchipInterpolator(P_crust_sorted, e_crust_sorted, extrapolate=True)
    e_eos_of_P = PchipInterpolator(P_eos_sorted, e_eos_sorted, extrapolate=True)
    nB_crust_of_P = PchipInterpolator(P_crust_sorted, nB_crust_sorted, extrapolate=True)
    nB_eos_of_P = PchipInterpolator(P_eos_sorted, nB_eos_sorted, extrapolate=True)

    # Find P_trans where μ_crust(P) = μ_eos(P)
    # Use root finder with initial guess at typical crust-core transition (~0.08 fm⁻³)
    from scipy.optimize import root

    def delta_mu(P):
        return float(muB_crust_of_P(P) - muB_eos_of_P(P))

    # Initial guess: P at n_B ~ 0.08 fm⁻³ (typical crust-core boundary)
    n_guess = 0.08  # fm⁻³
    # Find closest point in crust table to get P guess
    idx_guess = np.argmin(np.abs(nB_crust_sorted - n_guess))
    P_guess = P_crust_sorted[idx_guess]

    # Find root using scipy.optimize.root
    try:
        sol = root(delta_mu, P_guess)
        if sol.success:
            P_trans = float(sol.x[0])
        else:
            # Fallback: use the guess
            P_trans = P_guess
            if verbose:
                print(f"  Warning: Root finder did not converge, using P_guess = {P_trans:.4f} MeV/fm³")
    except Exception as e:
        P_trans = P_guess
        if verbose:
            print(f"  Warning: Could not find P_trans ({e}), using P_guess = {P_trans:.4f} MeV/fm³")

    if verbose:
        n1 = float(nB_crust_of_P(P_trans))
        n2 = float(nB_eos_of_P(P_trans))
        mu_trans = float(muB_crust_of_P(P_trans))
        print(f"  Maxwell transition: P_trans = {P_trans:.4f} MeV/fm³")
        print(f"    n_crust(P_trans) = {n1:.4f} fm⁻³")
        print(f"    n_eos(P_trans) = {n2:.4f} fm⁻³")
        print(f"    μB(P_trans) = {mu_trans:.2f} MeV")
        print(f"    δP = {delta_P:.4f} MeV/fm³")

    # Create unified pressure grid
    # Use crust below P_trans - 3*delta_P, EOS above P_trans + 3*delta_P
    P_low = P_trans - 3 * delta_P if delta_P > 0 else P_trans
    P_high = P_trans + 3 * delta_P if delta_P > 0 else P_trans

    P_crust_use = P_crust_sorted[P_crust_sorted <= P_high]
    P_eos_use = P_eos_sorted[P_eos_sorted >= P_low]

    # Unified pressure grid
    P_unified = np.unique(np.concatenate([P_crust_use, P_eos_use]))
    P_unified = np.sort(P_unified)

    # Blending function in pressure space
    def blend(P):
        if delta_P <= 0:
            # Sharp transition (Maxwell)
            return 0.0 if P < P_trans else 1.0
        else:
            # Smooth crossover
            return 0.5 * (1.0 + np.tanh((P - P_trans) / delta_P))

    # Compute ε(P) and nB(P) using blending
    e_merged = np.zeros_like(P_unified)
    nB_merged = np.zeros_like(P_unified)

    for i, P in enumerate(P_unified):
        f = blend(P)
        # ε(P) = (1-f) * ε_crust(P) + f * ε_eos(P)
        e_merged[i] = (1.0 - f) * float(e_crust_of_P(P)) + f * float(e_eos_of_P(P))
        # nB(P) = (1-f) * nB_crust(P) + f * nB_eos(P)
        nB_merged[i] = (1.0 - f) * float(nB_crust_of_P(P)) + f * float(nB_eos_of_P(P))

    return EOSTable_for_TOV(P=P_unified, epsilon=e_merged, nB=nB_merged)


def _save_eos_table(eos: EOSTable_for_TOV, filepath: str) -> None:
    """Save EOS table to file."""
    header = "# Merged EOS Table\n# Columns: P [MeV/fm³], epsilon [MeV/fm³], nB [fm⁻³]"
    np.savetxt(filepath, np.column_stack([eos.P, eos.epsilon, eos.nB]),
               header=header, fmt='%.10e')


# =============================================================================
# TOV SOLVER 
# =============================================================================


def _detect_maxwell_construction(eos: EOSTable_for_TOV, 
                                   P_tol: float = 1e-4) -> Optional[Tuple[float, float, float]]:
    """Detect Maxwell construction in EOS (constant-P region).
    
    For Maxwell construction, multiple n_B values share the same P.
    This function finds such regions and returns (P_trans, e_onset, e_offset).
    
    Args:
        eos: EOS table
        P_tol: Relative tolerance for detecting constant P
        
    Returns:
        None if no Maxwell construction detected, otherwise (P_trans, e_low, e_high)
        where e_low is the onset (lower density) and e_high is the offset (higher density)
    """
    # Sort by pressure
    idx_P = np.argsort(eos.P)
    P_sorted = eos.P[idx_P]
    eps_sorted = eos.epsilon[idx_P]
    nB_sorted = eos.nB[idx_P]
    
    # Find duplicate P values (constant P region)
    dP = np.diff(P_sorted)
    P_mean = 0.5 * (P_sorted[:-1] + P_sorted[1:])
    
    # Constant P where |dP/P| < tolerance
    is_constant = np.abs(dP) < P_tol * np.abs(P_mean)
    
    if not np.any(is_constant):
        return None
    
    # Find the largest constant-P region (the mixed phase plateau)
    # Group consecutive constant-P points
    constant_idx = np.where(is_constant)[0]
    if len(constant_idx) == 0:
        return None
    
    # Find contiguous groups
    groups = np.split(constant_idx, np.where(np.diff(constant_idx) != 1)[0] + 1)
    
    # Find the largest group (main phase transition)
    largest_group = max(groups, key=len)
    if len(largest_group) < 2:  # Need at least a few points to be a real Maxwell plateau
        return None
    
    # Get indices of the plateau region
    start_idx = largest_group[0]
    end_idx = largest_group[-1] + 1  # +1 because diff reduces length by 1
    
    P_trans = np.mean(P_sorted[start_idx:end_idx+1])
    e_low = eps_sorted[start_idx]      # Onset (lower ε, hadron side)
    e_high = eps_sorted[end_idx]        # Offset (higher ε, quark side)
    
    return (P_trans, e_low, e_high)


def _create_interpolators(eos: EOSTable_for_TOV) -> Tuple[PchipInterpolator, PchipInterpolator, PchipInterpolator]:
    """Create interpolators for P(e), e(P), n(P).

    Handles Maxwell construction by using only the stable branches:
    - Below P_trans: use the low-density (hadron) branch
    - Above P_trans: use the high-density (quark) branch
    
    For e(P) and n(P), when P has duplicates (constant-P plateau),
    we keep only one point per unique P to ensure strictly increasing x-values.
    """
    # P(epsilon) - epsilon is always unique, just sort
    idx_e = np.argsort(eos.epsilon)
    eps_sorted = eos.epsilon[idx_e]
    P_for_e = eos.P[idx_e]
    # Remove duplicate epsilon values if any (shouldn't happen but be safe)
    _, unique_idx_e = np.unique(eps_sorted, return_index=True)
    eps_unique = eps_sorted[unique_idx_e]
    P_for_e_unique = P_for_e[unique_idx_e]
    P_of_e = PchipInterpolator(eps_unique, P_for_e_unique, extrapolate=True)

    # e(P) and n(P) - need to handle constant P regions
    idx_P = np.argsort(eos.P)
    P_sorted = eos.P[idx_P]
    eps_for_P = eos.epsilon[idx_P]
    nB_for_P = eos.nB[idx_P]
    
    # Remove duplicate P values - keep LAST occurrence (high-density/quark side)
    # This ensures that during TOV integration (from high to low P), 
    # we use the correct branch at the transition
    _, unique_idx_P = np.unique(P_sorted[::-1], return_index=True)
    unique_idx_P = len(P_sorted) - 1 - unique_idx_P  # Convert back to forward indices
    unique_idx_P = np.sort(unique_idx_P)
    
    P_unique = P_sorted[unique_idx_P]
    eps_for_P_unique = eps_for_P[unique_idx_P]
    nB_for_P_unique = nB_for_P[unique_idx_P]
    
    e_of_P = PchipInterpolator(P_unique, eps_for_P_unique, extrapolate=True)
    n_of_P = PchipInterpolator(P_unique, nB_for_P_unique, extrapolate=True)

    return P_of_e, e_of_P, n_of_P


class _CombinedSolution:
    """
    Combines two ODE solutions for phase transition handling.

    When a Maxwell construction phase transition occurs:
    - sol1: integration from center to transition radius (high-density phase)
    - sol2: integration from transition to surface (low-density phase)
    """

    def __init__(self, sol1, sol2, r_transition):
        self.sol1 = sol1
        self.sol2 = sol2
        self.r_transition = r_transition
        self.t = np.concatenate([sol1.t, sol2.t[1:]])
        self.y = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
        self.t_events = sol2.t_events

    def sol(self, r):
        """Interpolate solution at radius r."""
        r = np.atleast_1d(r)
        result = np.where(
            r <= self.r_transition,
            self.sol1.sol(r),
            self.sol2.sol(r)
        )
        return result.squeeze()


def solve_tov_single(
    e_c: float,
    eos: EOSTable_for_TOV,
    P_of_e: PchipInterpolator,
    e_of_P: PchipInterpolator,
    n_of_P: PchipInterpolator,
    compute_baryonic: bool = True,
    compute_tidal: bool = True,
    r_min: float = 0.001,
    r_max: float = 15.0,
    p_surface: float = 1e-10,
    phase_transition: Optional[Tuple[float, float, float]] = None,
) -> TOVResult:
    """Solve TOV equations for a single central energy density.

    Args:
        e_c: Central energy density [MeV/fm³]
        eos: EOS table
        P_of_e, e_of_P, n_of_P: Interpolators
        compute_baryonic: Whether to compute baryonic mass
        compute_tidal: Whether to compute tidal deformability
        r_min: Starting radius in r_sun units
        r_max: Maximum radius in r_sun units
        p_surface: Surface pressure threshold (dimensionless)
        phase_transition: Optional (P_trans, e_low, e_high) for Maxwell construction.

    Returns:
        TOVResult with computed quantities
    """
    # Central values
    p0 = float(P_of_e(e_c))
    n_c = float(n_of_P(p0))
    if p0 <= 0:
        raise ValueError(f"Central pressure <= 0 for e_c = {e_c}")

    # Prepare interpolation arrays (sorted by P)
    idx_p = np.argsort(eos.P)
    p_grid = np.ascontiguousarray(eos.P[idx_p])
    e_interp_p = np.ascontiguousarray(eos.epsilon[idx_p])

    # Initial conditions
    m0 = e_c * 4.0 / 3.0 * np.pi * r_min**3 / M_sun_MeV
    y0 = np.array([m0, 1.0])

    # Surface detection event
    def surface_event(r, y):
        return y[1] - p_surface
    surface_event.terminal = True
    surface_event.direction = -1

    def rhs_tov(r, y):
        return _tov_rhs_scipy(r, y, p_grid, e_interp_p, p0, e_c, rho_sol)

    # Add phase transition event if specified
    events = [surface_event]
    if phase_transition is not None:
        P_trans = phase_transition[0]
        def transition_event(r, y):
            return y[1] - P_trans / p0
        transition_event.terminal = True
        transition_event.direction = -1
        events.append(transition_event)

    # First integration
    sol = solve_ivp(rhs_tov, [r_min, r_max], y0, method='DOP853',
                    events=events, dense_output=True, rtol=1e-10, atol=1e-12)

    # Handle phase transition if it occurred
    if (phase_transition is not None and 
        len(sol.t_events) > 1 and sol.t_events[1].size > 0):
        
        P_trans, e_low, e_high = phase_transition
        r_trans = sol.t_events[1][0]
        m_trans = sol.sol(r_trans)[0]

        # For post-transition: use low-density (hadron) branch of EOS
        # Select only points with epsilon <= e_low (hadron phase)
        hadron_mask = eos.epsilon <= e_low * 1.01  # Small tolerance
        if np.sum(hadron_mask) > 2:
            # Use hadron branch for interpolation
            idx_h = np.argsort(eos.P[hadron_mask])
            p_grid_post = np.ascontiguousarray(eos.P[hadron_mask][idx_h])
            e_interp_p_post = np.ascontiguousarray(eos.epsilon[hadron_mask][idx_h])
        else:
            # Fallback: modify full grid (original approach)
            p_grid_post = p_grid.copy()
            e_interp_p_post = e_interp_p.copy()
            e_interp_p_post[p_grid <= P_trans] = e_low

        def rhs_post(r, y):
            return _tov_rhs_scipy(r, y, p_grid_post, e_interp_p_post, p0, e_c, rho_sol)

        sol_post = solve_ivp(rhs_post, [r_trans, r_max], 
                             np.array([m_trans, P_trans / p0]),
                             method='DOP853', events=surface_event,
                             dense_output=True, rtol=1e-10, atol=1e-12)

        sol = _CombinedSolution(sol, sol_post, r_trans)

    # Extract results
    r_surface = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
    M_msun = sol.sol(r_surface)[0]
    R_km = r_surface * r_sun_km

    result = TOVResult(e_c=e_c, n_c=n_c, P_c=p0, R=R_km, M=M_msun)

    if compute_baryonic:
        result.M_b = _compute_baryonic_mass(sol, r_surface, p0, n_of_P, M_sun_MeV)

    if compute_tidal:
        result.k2, result.Lambda = _compute_tidal(
            sol, r_surface, M_msun, R_km, p0, e_of_P, n_of_P, eos)

    return result


def _tov_rhs_scipy(r: float, y: np.ndarray, 
                   p_grid: np.ndarray, e_interp_p: np.ndarray,
                   p0: float, e0: float, rho_s: float) -> np.ndarray:
    """TOV RHS for scipy solver."""
    m, p = y[0], y[1]
    
    if p <= 0.0 or r <= 0.0:
        return np.array([0.0, 0.0])
    
    # Get energy density from pressure
    P_physical = p * p0
    e = np.interp(P_physical, p_grid, e_interp_p)
    
    # dm/dr = 3 * (e / rho_s) * r^2
    dmdr = 3.0 * (e / rho_s) * r * r
    
    # dp/dr from TOV equation (negative sign for pressure decrease outward)
    if abs(r - m) < 1e-30 or r < 1e-30:
        dpdr = 0.0
    else:
        factor1 = 0.5 * (e / p0) * m
        factor2 = 1.0 + (p * p0) / e if e > 0 else 1.0
        factor3 = 1.0 + 3.0 * (p0 / rho_s) * (p * r**3) / m if m > 0 else 1.0
        dpdr = -factor1 * factor2 * factor3 / (r * (r - m))
    
    return np.array([dmdr, dpdr])



def _compute_baryonic_mass(sol, r_surface: float, p0: float,
                           n_of_P: CubicSpline, M_sun: float) -> float:
    """
    Compute baryonic mass.
    
    M_b = m_NB * ∫ 4π r² n_B / √(1 - 2Gm/r) dr
    
    In dimensionless units with r in r_sun and m in M_sun:
    2Gm/r = 2 * (G * M_sun / (r_sun * c²)) * (m/r) = m/r (since r_sun = 2GM_sun/c²)
    """
    # Sample solution
    r_grid = np.linspace(0.001, r_surface, 500)
    m_grid = sol.sol(r_grid)[0]
    p_grid = sol.sol(r_grid)[1]
    
    # Get n_B from pressure
    P_physical = p_grid * p0
    n_grid = np.array([float(n_of_P(P)) for P in P_physical])
    
    # Integrand: 4π r² n_B / √(1 - m/r) in dimensionless units
    # Need to convert r to fm: r_fm = r * r_sun_fm
    # Result should be in MeV, then divide by M_sun
    integrand = np.zeros_like(r_grid)
    for i, (r, m, n) in enumerate(zip(r_grid, m_grid, n_grid)):
        r_fm = r * r_sun_fm
        metric = 1.0 - m / r if r > m else 0.01  # Avoid singularity
        if metric > 0:
            integrand[i] = 4.0 * np.pi * r_fm**2 * n / np.sqrt(metric)
    
    # Integrate (dr in units of r_sun, convert to fm)
    dr_fm = (r_grid[1] - r_grid[0]) * r_sun_fm
    M_b_MeV = m_nucleon_MeV * np.trapz(integrand, dx=dr_fm)
    M_b_msun = M_b_MeV / M_sun
    
    return M_b_msun


def _compute_tidal(sol, r_surface: float, M_msun: float, R_km: float,
                   p0: float, e_of_P, n_of_P,
                   eos: EOSTable_for_TOV) -> Tuple[float, float]:
### TODO: implement tidal deformability
    return 0.0, 0.0


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def compute_tov_sequence(
    eos_file: str,
    e_c_vec: np.ndarray,
    add_crust_table: str = 'No',
    add_crust_mode: str = 'attach',
    n_transition: Optional[float] = None,
    delta_n: float = 0.01,
    custom_crust_path: Optional[str] = None,
    compute_baryonic_mass: bool = True,
    compute_tidal: bool = True,
    output_file: Optional[str] = None,
    eos_columns: Tuple[int, int, int] = (0, 1, 2),
    skip_header: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute TOV sequence for array of central energy densities.
    
    Args:
        eos_file: Path to EOS table file
        e_c_vec: Array of central energy densities [MeV/fm³]
        add_crust_table: 'No', 'BPS', 'compose_sfho', or 'personalized'
        add_crust_mode: 'attach' or 'interpolate'
        n_transition: Transition density for crust [fm⁻³]
        delta_n: Interpolation width [fm⁻³]
        custom_crust_path: Path to custom crust file
        compute_baryonic_mass: Whether to compute M_b
        compute_tidal: Whether to compute k2 and Lambda
        output_file: Path for output file (optional)
        eos_columns: Column indices (P, epsilon, nB)
        skip_header: Header lines to skip
        verbose: Print progress
        
    Returns:
        Structured array with columns:
        - e_c, n_c, P_c, R, M, [M_b], [k2], [Lambda]
    """
    # Load EOS
    eos = EOSTable_for_TOV.from_file(eos_file, columns=eos_columns, skip_header=skip_header)
    
    # Add crust
    eos = add_crust(
        eos, 
        crust_name=add_crust_table,
        mode=add_crust_mode,
        n_transition=n_transition,
        delta_n=delta_n,
        custom_crust_path=custom_crust_path,
    )
    
    # Create interpolators
    P_of_e, e_of_P, n_of_P = _create_interpolators(eos)
    
    # Detect Maxwell construction (constant-P region from phase transition)
    phase_transition = _detect_maxwell_construction(eos)
    if phase_transition is not None and verbose:
        P_trans, e_low, e_high = phase_transition
        print(f"  Maxwell construction detected: P_trans={P_trans:.4f} MeV/fm³, "
              f"e_low={e_low:.1f}, e_high={e_high:.1f} MeV/fm³")
    
    # Prepare output
    n_stars = len(e_c_vec)
    results = []
    
    if verbose:
        print(f"Computing TOV for {n_stars} central densities...")
    
    for i, e_c in enumerate(e_c_vec):
        try:
            result = solve_tov_single(
                e_c, eos, P_of_e, e_of_P, n_of_P,
                compute_baryonic=compute_baryonic_mass,
                compute_tidal=compute_tidal,
                phase_transition=phase_transition,
            )
            results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_stars}] e_c={e_c:.1f}, M={result.M:.3f} M_sun, R={result.R:.2f} km")
                
        except Exception as exc:
            if verbose:
                print(f"  [{i+1}/{n_stars}] e_c={e_c:.1f} FAILED: {exc}")
            continue
    
    # Convert to structured array
    output_data = _results_to_array(results, compute_baryonic_mass, compute_tidal)
    
    # Save if requested
    if output_file is not None:
        _save_tov_results(output_data, output_file, compute_baryonic_mass, compute_tidal)
        if verbose:
            print(f"Saved results to: {output_file}")
    
    return output_data



# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_ec_logspace(e_min: float, e_max: float, n_points: int) -> np.ndarray:
    """Generate logarithmically spaced central energy densities."""
    return 10.0 ** np.linspace(np.log10(e_min), np.log10(e_max), n_points)


def _results_to_array(results: list, include_Mb: bool, include_tidal: bool) -> np.ndarray:
    """Convert list of TOVResult to numpy array."""
    n = len(results)
    n_cols = 5  # e_c, n_c, P_c, R, M
    if include_Mb:
        n_cols += 1
    if include_tidal:
        n_cols += 2
    
    data = np.zeros((n, n_cols))
    for i, r in enumerate(results):
        col = 0
        data[i, col] = r.e_c; col += 1
        data[i, col] = r.n_c; col += 1
        data[i, col] = r.P_c; col += 1
        data[i, col] = r.R; col += 1
        data[i, col] = r.M; col += 1
        if include_Mb:
            data[i, col] = r.M_b if r.M_b is not None else np.nan; col += 1
        if include_tidal:
            data[i, col] = r.k2 if r.k2 is not None else np.nan; col += 1
            data[i, col] = r.Lambda if r.Lambda is not None else np.nan
    
    return data


def _save_tov_results(data: np.ndarray, filepath: str, 
                      include_Mb: bool, include_tidal: bool) -> None:
    """Save TOV results to file."""
    cols = ['e_c[MeV/fm3]', 'n_c[fm-3]', 'P_c[MeV/fm3]', 'R[km]', 'M[Msun]']
    if include_Mb:
        cols.append('M_b[Msun]')
    if include_tidal:
        cols.extend(['k2', 'Lambda'])
    
    header = "TOV Sequence Results\n" + "  ".join(cols)
    np.savetxt(filepath, data, header=header, fmt='%.10e')


