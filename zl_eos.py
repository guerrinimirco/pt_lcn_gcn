"""
zl_eos.py
=========
Single-point solvers for Zhao-Lattimer (ZL) nucleonic EOS.

This module provides solvers for different equilibrium conditions:
- Beta equilibrium (charge neutrality, beta eq)
- Fixed charge fraction (Y_C = n_p/n_B)
- Trapped neutrinos (fixed Y_L)

All thermodynamic functions are in zl_thermodynamics_nucleons.py.
All table generation is in zl_compute_tables.py.

Usage:
    from zl_eos import solve_zl_beta_eq, solve_zl_fixed_yc
    
    result = solve_zl_beta_eq(n_B=0.16, T=10.0)
    print(f"P = {result.P_total} MeV/fm³")
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.optimize import root

from zl_parameters import ZLParams, get_zl_default
from zl_thermodynamics_nucleons import (
    compute_nucleon_thermo, compute_nucleon_density, compute_zl_thermo_from_mu_n,
    compute_V_interaction, compute_P_interaction,
    compute_effective_mu_nucleons, compute_nucleon_densities_for_solver, G_NUCLEON
)
from general_thermodynamics_leptons import electron_thermo, neutrino_thermo, photon_thermo
from general_physics_constants import hc, hc3, PI2


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class ZLEOSResult:
    """Complete result from ZL EOS calculation at one point."""
    # Convergence info
    converged: bool = False
    error: float = 0.0
    
    # Inputs
    n_B: float = 0.0       # Baryon density (fm⁻³)
    T: float = 0.0         # Temperature (MeV)
    Y_C: float = 0.0       # Charge fraction
    Y_S: float = 0.0       # Strange fraction
    Y_L: float = 0.0       # Lepton fraction (for trapped neutrinos)
    
    # Chemical potentials
    mu_p: float = 0.0      # Proton chemical potential (MeV)
    mu_n: float = 0.0      # Neutron chemical potential (MeV)
    mu_e: float = 0.0      # Electron chemical potential (MeV)
    mu_nu: float = 0.0     # Neutrino chemical potential (MeV)
    mu_B: float = 0.0      # Baryon chemical potential (MeV)
    mu_C: float = 0.0      # Charge chemical potential (MeV)
    mu_S: float = 0.0      # Strange chemical potential (MeV)
    mu_L: float = 0.0      # Lepton chemical potential (MeV)
    
    # Densities
    n_p: float = 0.0       # Proton density (fm⁻³)
    n_n: float = 0.0       # Neutron density (fm⁻³)
    n_e: float = 0.0       # Electron density (fm⁻³)
    n_nu: float = 0.0      # Neutrino density (fm⁻³)
    
    # Thermodynamics
    P_total: float = 0.0   # Total pressure (MeV/fm³)
    e_total: float = 0.0   # Total energy density (MeV/fm³)
    s_total: float = 0.0   # Total entropy density (fm⁻³)
    
    # Fractions
    Y_p: float = 0.0       # Proton fraction
    Y_n: float = 0.0       # Neutron fraction
    Y_e: float = 0.0       # Electron fraction


# =============================================================================
# INITIAL GUESS GENERATION
# =============================================================================
def get_default_guess_beta_eq(n_B: float, T: float, params: ZLParams) -> np.ndarray:
    """
    Generate initial guess for beta equilibrium: [μ_p, μ_n, μ_e, n_p, n_n].
    """
    m_p, m_n = params.m_p, params.m_n
    n0 = params.n0
    
    # Estimate proton fraction (low at low T, increases with T)
    Y_p_est = 0.05 + 0.15 * (T / 50.0)
    Y_p_est = max(0.01, min(Y_p_est, 0.5))
    

    n_p_est = Y_p_est * n_B
    n_n_est = (1.0 - Y_p_est) * n_B
    
    # Fermi momentum estimate
    kF_p = hc * (6.0 * PI2 * n_p_est / 2.0)**(1.0/3.0) if n_p_est > 0 else 0.0
    kF_n = hc * (6.0 * PI2 * n_n_est / 2.0)**(1.0/3.0) if n_n_est > 0 else 0.0
    
    # Effective chemical potential ~ sqrt(kF² + m²)
    mu_p_eff_est = np.sqrt(kF_p**2 + m_p**2) if n_p_est > 0 else m_p
    mu_n_eff_est = np.sqrt(kF_n**2 + m_n**2) if n_n_est > 0 else m_n
    
    # Add mean-field contribution (rough estimate)
    V_est = 4.0 * n_p_est * n_n_est * (params.a0 + params.b0 * (n_B/n0)**(params.gamma - 1)) / n0
    
    mu_p_est = mu_p_eff_est + V_est * 0.5
    mu_n_est = mu_n_eff_est + V_est * 0.5
    mu_e_est = max(0.0, mu_n_est - mu_p_est)
    
    return np.array([mu_p_est, mu_n_est, mu_e_est, n_p_est, n_n_est])


def get_default_guess_fixed_yc(n_B: float, Y_C: float, T: float, 
                                params: ZLParams,
                                include_electrons: bool = True) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C.
    
    Returns [μ_p, μ_n] if include_electrons=False,
    Returns [μ_p, μ_n, μ_e] if include_electrons=True.
    """
    m_p, m_n = params.m_p, params.m_n
    
    n_p = Y_C * n_B
    n_n = (1.0 - Y_C) * n_B
    
    kF_p = hc * (6.0 * PI2 * n_p / 2.0)**(1.0/3.0) if n_p > 0 else 0.0
    kF_n = hc * (6.0 * PI2 * n_n / 2.0)**(1.0/3.0) if n_n > 0 else 0.0
    
    mu_p_est = np.sqrt(kF_p**2 + m_p**2) if n_p > 0 else m_p
    mu_n_est = np.sqrt(kF_n**2 + m_n**2) if n_n > 0 else m_n
    
    if include_electrons:
        # Estimate mu_e from n_e = n_p (charge neutrality)
        n_e = n_p
        kF_e = hc * (3 * PI2 * n_e)**(1.0/3.0) if n_e > 0 else 0.0
        m_e = 0.511  # MeV
        mu_e_est = np.sqrt(kF_e**2 + m_e**2) if n_e > 0 else m_e
        return np.array([mu_p_est, mu_n_est, mu_e_est])
    
    return np.array([mu_p_est, mu_n_est])


def get_default_guess_trapped_neutrinos(n_B: float, Y_L: float, T: float,
                                          params: ZLParams) -> np.ndarray:
    """Generate initial guess for trapped neutrinos: [μ_p, μ_n, μ_e, μ_ν, n_p, n_n]."""
    guess_beta = get_default_guess_beta_eq(n_B, T, params)
    mu_nu_est = 10.0  # Small initial neutrino chemical potential
    return np.array([guess_beta[0], guess_beta[1], guess_beta[2], 
                     mu_nu_est, guess_beta[3], guess_beta[4]])


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_zl_beta_eq(
    n_B: float, T: float, params: ZLParams = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> ZLEOSResult:
    """
    Solve ZL EOS in beta equilibrium.
    
    Solves 5 equations for 5 unknowns: [μ_p, μ_n, μ_e, n_p, n_n]
    
    Equations: (note, 1 and 2 take the place of mean field equations)
        1. n_p(μ_p, n_p, n_n, T) = n_p  (self-consistency)
        2. n_n(μ_n, n_p, n_n, T) = n_n  (self-consistency)
        3. n_p + n_n = n_B              (baryon number)
        4. μ_n - μ_p = μ_e              (beta equilibrium)
        5. n_p = n_e(μ_e, T)            (charge neutrality)
    
    Args:
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        params: ZL parameters (uses default if None)
        include_photons: Include photon contributions
        initial_guess: Initial guess [μ_p, μ_n, μ_e, n_p, n_n]
        
    Returns:
        ZLEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_zl_default()
    
    result = ZLEOSResult(n_B=n_B, T=T)
    
    m_p, m_n = params.m_p, params.m_n
    n0 = params.n0
    
    if initial_guess is None:
        x0 = get_default_guess_beta_eq(n_B, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        mu_p, mu_n, mu_e, n_p, n_n = x
        
        # Compute effective μ and densities
        nmd = compute_nucleon_densities_for_solver(mu_p, mu_n, n_p, n_n, T, params)
        n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
        
        eq1 = nmd.n_p_calc - n_p
        eq2 = nmd.n_n_calc - n_n
        eq3 = nmd.n_B - n_B
        eq4 = mu_n - mu_p - mu_e
        eq5 = nmd.n_C - n_e  # Charge neutrality: n_p = n_e
        
        return [eq1, eq2, eq3, eq4, eq5]
    
    # Solve
    sol = root(equations, x0, method='hybr')
    if not sol.success:
        sol = root(equations, x0, method='lm')
    
    mu_p, mu_n, mu_e, n_p, n_n = sol.x
    
    residuals = equations(sol.x)
    error = sum(r**2 for r in residuals)
    result.converged = (error < 0.01)
    result.error = error
    
    # Store results
    result.mu_p, result.mu_n, result.mu_e = mu_p, mu_n, mu_e
    result.n_p, result.n_n = n_p, n_n
    result.Y_p = n_p / n_B
    result.Y_n = n_n / n_B
    result.Y_C = result.Y_p
    
    # Compute hadronic thermodynamics using helper function
    h_thermo = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)
    
    # Add electron contribution
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    result.n_e = e_thermo.n
    result.Y_e = result.n_e / n_B
    
    result.P_total = h_thermo.P + e_thermo.P
    result.e_total = h_thermo.e + e_thermo.e
    result.s_total = h_thermo.s + e_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = h_thermo.mu_B
    result.mu_C = h_thermo.mu_C
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C
# =============================================================================
def solve_zl_fixed_yc(
    n_B: float, Y_C: float, T: float, params: ZLParams = None,
    include_photons: bool = True,
    include_electrons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> ZLEOSResult:
    """
    Solve ZL EOS with fixed charge fraction Y_C.
    
    If include_electrons=False: Solves 2 equations for [μ_p, μ_n]
    If include_electrons=True:  Solves 3 equations for [μ_p, μ_n, μ_e]
        with charge neutrality n_e(μ_e) = n_p
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction n_p/n_B
        T: Temperature (MeV)
        params: ZL parameters
        include_photons: Include photon contributions
        include_electrons: If True, solve for electrons with n_e = n_p (charge neutrality)
        initial_guess: Initial guess [μ_p, μ_n] or [μ_p, μ_n, μ_e]
        
    Returns:
        ZLEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_zl_default()
    
    result = ZLEOSResult(n_B=n_B, T=T, Y_C=Y_C)
    
    m_p, m_n = params.m_p, params.m_n
    
    n_p = Y_C * n_B
    n_n = (1.0 - Y_C) * n_B
    Y_p, Y_n = Y_C, 1.0 - Y_C
    
    if initial_guess is None:
        x0 = get_default_guess_fixed_yc(n_B, Y_C, T, params, include_electrons)
    else:
        x0 = initial_guess
    
    if include_electrons:
        # Solve 3 equations: n_p(μ_p) = n_p, n_n(μ_n) = n_n, n_e(μ_e) = n_p
        def equations(x):
            mu_p, mu_n, mu_e = x
            
            # Compute effective μ and densities
            nmd = compute_nucleon_densities_for_solver(mu_p, mu_n, n_p, n_n, T, params)
            n_e = electron_thermo(mu_e, T, include_antiparticles=True).n
            
            eq1 = nmd.n_p_calc - n_p
            eq2 = nmd.n_n_calc - n_n
            eq3 = n_e - nmd.n_C  # Charge neutrality: n_e = n_p
            
            return [eq1, eq2, eq3]
        
        sol = root(equations, x0, method='hybr')
        if not sol.success:
            sol = root(equations, x0, method='lm')
        
        mu_p, mu_n, mu_e = sol.x
        
        residuals = equations(sol.x)
        error = sum(r**2 for r in residuals)
        result.converged = (error < 0.01)
        result.error = error
        
        result.mu_p, result.mu_n, result.mu_e = mu_p, mu_n, mu_e
        result.n_p, result.n_n = n_p, n_n
        result.Y_p, result.Y_n = Y_p, Y_n
        
        # Compute electron quantities
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        result.n_e = e_thermo.n
        result.Y_e = result.n_e / n_B
        
    else:
        # Solve 2 equations: n_p(μ_p) = n_p, n_n(μ_n) = n_n
        def equations(x):
            mu_p, mu_n = x
            
            # Compute effective μ and densities
            nmd = compute_nucleon_densities_for_solver(mu_p, mu_n, n_p, n_n, T, params)
            
            eq1 = nmd.n_p_calc - n_p
            eq2 = nmd.n_n_calc - n_n
            
            return [eq1, eq2]
        
        sol = root(equations, x0, method='hybr')
        if not sol.success:
            sol = root(equations, x0, method='lm')
        
        mu_p, mu_n = sol.x
        
        residuals = equations(sol.x)
        error = sum(r**2 for r in residuals)
        result.converged = (error < 0.01)
        result.error = error
        
        result.mu_p, result.mu_n = mu_p, mu_n
        result.n_p, result.n_n = n_p, n_n
        result.Y_p, result.Y_n = Y_p, Y_n
    
    # Compute hadronic thermodynamics using helper function
    h_thermo = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)
    
    result.P_total = h_thermo.P
    result.e_total = h_thermo.e
    result.s_total = h_thermo.s
    
    # Add electron thermodynamics if included
    if include_electrons:
        e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
        result.P_total += e_thermo.P
        result.e_total += e_thermo.e
        result.s_total += e_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = h_thermo.mu_B
    result.mu_C = h_thermo.mu_C
    
    return result


# =============================================================================
# SOLVER: TRAPPED NEUTRINOS
# =============================================================================
def solve_zl_trapped_neutrinos(
    n_B: float, Y_L: float, T: float, params: ZLParams = None,
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> ZLEOSResult:
    """
    Solve ZL EOS with trapped neutrinos (fixed Y_L).
    
    Solves 6 equations for 6 unknowns: [μ_p, μ_n, μ_e, μ_ν, n_p, n_n]
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_L: Lepton fraction (n_e + n_ν - n_ν̄)/n_B
        T: Temperature (MeV)
        params: ZL parameters
        include_photons: Include photon contributions
        initial_guess: Initial guess [μ_p, μ_n, μ_e, μ_ν, n_p, n_n]
        
    Returns:
        ZLEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_zl_default()
    
    result = ZLEOSResult(n_B=n_B, T=T, Y_L=Y_L)
    
    m_p, m_n = params.m_p, params.m_n
    
    if initial_guess is None:
        x0 = get_default_guess_trapped_neutrinos(n_B, Y_L, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        mu_p, mu_n, mu_e, mu_nu, n_p, n_n = x
        
        # Compute effective μ and densities
        nmd = compute_nucleon_densities_for_solver(mu_p, mu_n, n_p, n_n, T, params)
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        n_e = e_thermo.n
        n_nu = nu_thermo.n
        
        eq1 = nmd.n_p_calc - n_p
        eq2 = nmd.n_n_calc - n_n
        eq3 = nmd.n_B - n_B
        eq4 = mu_n - mu_p - mu_e + mu_nu  # Beta eq with neutrinos
        eq5 = nmd.n_C - n_e                # Charge neutrality
        eq6 = (n_e + n_nu) / n_B - Y_L    # Lepton fraction
        
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    
    sol = root(equations, x0, method='hybr')
    if not sol.success:
        sol = root(equations, x0, method='lm')
    
    mu_p, mu_n, mu_e, mu_nu, n_p, n_n = sol.x
    
    residuals = equations(sol.x)
    error = sum(r**2 for r in residuals)
    result.converged = (error < 0.01)
    result.error = error
    
    result.mu_p, result.mu_n, result.mu_e, result.mu_nu = mu_p, mu_n, mu_e, mu_nu
    result.n_p, result.n_n = n_p, n_n
    result.Y_p = n_p / n_B
    result.Y_n = n_n / n_B      
    result.Y_C = result.Y_p
    
    # Compute hadronic thermodynamics using helper function
    h_thermo = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)
    
    # Add lepton contributions
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_e = result.n_e / n_B
    
    result.P_total = h_thermo.P + e_thermo.P + nu_thermo.P
    result.e_total = h_thermo.e + e_thermo.e + nu_thermo.e
    result.s_total = h_thermo.s + e_thermo.s + nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.mu_B = h_thermo.mu_B
    result.mu_C = h_thermo.mu_C
    
    return result


# =============================================================================
# RESULT TO GUESS CONVERSION
# =============================================================================
def result_to_guess(result: ZLEOSResult, eq_type: str, include_electrons: bool = True) -> np.ndarray:
    """Convert result to initial guess array for next n_B point.
    
    Args:
        result: Previous result to extract guess from
        eq_type: Equilibrium type ('beta_eq', 'fixed_yc', 'trapped_neutrinos')
        include_electrons: For fixed_yc mode, whether electrons are included
    """
    if eq_type == 'beta_eq':
        return np.array([result.mu_p, result.mu_n, result.mu_e, result.n_p, result.n_n])
    elif eq_type == 'fixed_yc':
        if include_electrons:
            return np.array([result.mu_p, result.mu_n, result.mu_e])
        else:
            return np.array([result.mu_p, result.mu_n])
    elif eq_type == 'trapped_neutrinos':
        return np.array([result.mu_p, result.mu_n, result.mu_e, result.mu_nu, result.n_p, result.n_n])
    else:
        return np.array([result.mu_p, result.mu_n])


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("ZL EOS Solvers Test")
    print("=" * 50)
    
    params = get_zl_default()
    n_B = 0.16
    T = 10.0
    
    print(f"\nTest at n_B={n_B} fm⁻³, T={T} MeV")
    
    # Beta equilibrium
    r = solve_zl_beta_eq(n_B, T, params)
    print(f"\nBeta equilibrium:")
    print(f"  converged={r.converged}, error={r.error:.2e}")
    print(f"  Y_p={r.Y_p:.4f}, P={r.P_total:.2f} MeV/fm³")
    
    # Fixed Y_C
    r = solve_zl_fixed_yc(n_B, Y_C=0.3, T=T, params=params)
    print(f"\nFixed Y_C=0.3:")
    print(f"  converged={r.converged}, error={r.error:.2e}")
    print(f"  P={r.P_total:.2f} MeV/fm³")
    
    # Trapped neutrinos
    r = solve_zl_trapped_neutrinos(n_B, 0.4, T, params)
    print(f"\nTrapped neutrinos Y_L=0.4:")
    print(f"  converged={r.converged}, error={r.error:.2e}")
    print(f"  Y_p={r.Y_p:.4f}, P={r.P_total:.2f} MeV/fm³")
    
    print("\nOK!")
