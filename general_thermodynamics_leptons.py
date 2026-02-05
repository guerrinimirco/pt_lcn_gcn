"""
general_thermodynamics_leptons.py
================================
Model-independent thermodynamics for leptons and photons.

This module provides thermodynamic quantities (n, P, ε, s) for:
- Photons (blackbody radiation) - analytic Stefan-Boltzmann
- Leptons (e, μ, ν) - via Fermi integrals (including m=0 UR limit for ν)

All quantities in natural units:
- Energies/masses: MeV
- Lengths: fm
- Number density: fm⁻³
- Pressure/energy density: MeV/fm³
- Entropy density: fm⁻³

References:
- Mathematica notebook SFHO_HypDelMes_new.nb
"""
import numpy as np
from dataclasses import dataclass

from general_particles import Particle, Electron, Muon, Neutrino
from general_fermi_integrals import solve_fermi_jel
from general_physics_constants import hc, hc3, PI, PI2


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
ZETA3 = 1.2020569031595943  # Riemann zeta(3), for photon number density


# =============================================================================
# DATA CLASS FOR THERMODYNAMIC RESULTS
# =============================================================================
@dataclass
class ThermoResult:
    """
    Container for thermodynamic quantities.
    
    Attributes:
        n: Number density (fm⁻³) - net particle number (particles - antiparticles)
        P: Pressure (MeV/fm³)
        e: Energy density (MeV/fm³)
        s: Entropy density (fm⁻³)
        mu: Chemical potential (MeV) - optional, set by from_density functions
    """
    n: float      # Net number density
    P: float      # Pressure
    e: float      # Energy density
    s: float      # Entropy density
    mu: float = 0.0  # Chemical potential (optional)
    
    def __repr__(self):
        return (f"ThermoResult(n={self.n:.4e}, P={self.P:.4e}, "
                f"e={self.e:.4e}, s={self.s:.4e})")


# =============================================================================
# PHOTONS (Blackbody Radiation)
# =============================================================================
def photon_thermo(T: float) -> ThermoResult:
    """
    Thermodynamic quantities for blackbody photon gas.
    
    Photons are massless bosons with g = 2 (two polarizations) and μ = 0.
    
    Formulas (Stefan-Boltzmann):
        P = (π²/45) × T⁴ / (ℏc)³
        ε = 3P  (radiation equation of state)
        s = (4π²/45) × T³ / (ℏc)³
        n = (2ζ(3)/π²) × T³ / (ℏc)³
    
    Args:
        T: Temperature in MeV
        
    Returns:
        ThermoResult with (n, P, e, s)
    """
    if T <= 0:
        return ThermoResult(n=0.0, P=0.0, e=0.0, s=0.0)
    
    T3 = T**3
    T4 = T * T3
    
    # Pressure: P = (π²/45) × T⁴ / (ℏc)³
    P = (PI2 / 45.0) * T4 / hc3
    
    # Energy density: ε = 3P (equation of state for radiation)
    e = 3.0 * P
    
    # Entropy density: s = (4π²/45) × T³ / (ℏc)³ = 4P/T
    s = (4.0 * PI2 / 45.0) * T3 / hc3
    
    # Number density: n = (2ζ(3)/π²) × T³ / (ℏc)³
    n = (2.0 * ZETA3 / PI2) * T3 / hc3
    
    return ThermoResult(n=n, P=P, e=e, s=s)


# =============================================================================
# LEPTONS (Electrons, Muons, Neutrinos)
# =============================================================================
def lepton_thermo(mu: float, T: float, particle: Particle,
                  include_antiparticles: bool = True) -> ThermoResult:
    """
    Thermodynamic quantities for leptons using Fermi integrals.
    
    Works for both massive (e, μ) and massless (ν) leptons.
    The JEL approximation correctly handles:
    - m → 0 ultra-relativistic limit
    - T → 0 degenerate limit
    - Antiparticle contributions
    
    When include_antiparticles=True:
    - n returns the NET number density (n_particle - n_antiparticle)
    - P, e, s include contributions from both particles and antiparticles
    
    Args:
        mu: Chemical potential in MeV
        T: Temperature in MeV
        particle: Particle object (Electron, Muon, or Neutrino)
        include_antiparticles: If True, includes both particles and antiparticles
        
    Returns:
        ThermoResult with (n, P, e, s)
    """
    # solve_fermi_jel returns (n, P, e, s, ns)
    n, P, e, s, _ = solve_fermi_jel(mu, T, particle.mass, particle.g_degen, 
                                     include_antiparticles)
    return ThermoResult(n=n, P=P, e=e, s=s)


def electron_thermo(mu_e: float, T: float, 
                    include_antiparticles: bool = True) -> ThermoResult:
    """
    Thermodynamic quantities for electrons and positrons.
    
    Args:
        mu_e: Electron chemical potential in MeV
        T: Temperature in MeV
        include_antiparticles: If True, includes positrons
        
    Returns:
        ThermoResult with electron/positron thermodynamics
    """
    return lepton_thermo(mu_e, T, Electron, include_antiparticles)


def electron_thermo_from_density(n_e: float, T: float, 
                                   mu_e_guess: float = None) -> ThermoResult:
    """
    Compute electron thermodynamics for a prescribed electron density.
    
    This function inverts the n(μ) relationship to find the chemical potential
    that gives the desired density, then computes thermodynamic quantities.
    
    Useful for FIXED_YC_NEUTRAL mode where n_e is set to n_Q_had for 
    charge neutrality rather than being determined by equilibrium.
    
    Args:
        n_e: Prescribed electron density (fm⁻³)
        T: Temperature (MeV)
        mu_e_guess: Optional initial guess for μ_e (MeV). If provided, uses this
                   as starting point for root finding. Useful when chaining
                   calculations across density points.
        
    Returns:
        ThermoResult with (n, P, e, s) for electrons at the given density
        Also stores mu_e in result.mu attribute
    """
    from scipy.optimize import root
    
    if n_e <= 0:
        return ThermoResult(n=0.0, P=0.0, e=0.0, s=0.0)
    
    # Find μ_e such that n(μ_e) = n_e
    def residual(mu):
        result = electron_thermo(mu[0], T, include_antiparticles=True)
        return result.n - n_e
    
    # Initial guess
    if mu_e_guess is not None:
        x0 = mu_e_guess
    else:
        # Estimate from ultra-relativistic limit: n ≈ μ³/(3π²ℏc³)
        x0 = np.sign(n_e) * (abs(n_e) * 3.0 * PI2 * hc3)**(1.0/3.0)
    
    # Solve using root
    sol = root(residual, [x0], method='hybr')
    
    mu_e = sol.x[0]
    
    result = electron_thermo(mu_e, T, include_antiparticles=True)
    result.mu = mu_e  # Store μ_e for later use
    return result


def muon_thermo(mu_mu: float, T: float,
                include_antiparticles: bool = True) -> ThermoResult:
    """
    Thermodynamic quantities for muons and antimuons.
    
    Muons appear when μ_μ > m_μ ≈ 105.66 MeV, which typically occurs
    at very high densities in neutron star cores.
    
    In beta equilibrium: μ_μ = μ_e (if muons are present)
    
    Args:
        mu_mu: Muon chemical potential in MeV
        T: Temperature in MeV
        include_antiparticles: If True, includes antimuons
        
    Returns:
        ThermoResult with muon/antimuon thermodynamics
    """
    return lepton_thermo(mu_mu, T, Muon, include_antiparticles)


def neutrino_thermo(mu_nu: float, T: float,
                    include_antiparticles: bool = True) -> ThermoResult:
    """
    Thermodynamic quantities for neutrinos and antineutrinos.
    
    Uses Fermi integrals with m=0, which correctly gives the
    ultra-relativistic limit (ε = 3P).
    
    Args:
        mu_nu: Neutrino chemical potential in MeV
        T: Temperature in MeV
        include_antiparticles: If True, includes antineutrinos
        
    Returns:
        ThermoResult with neutrino/antineutrino thermodynamics
    """
    return lepton_thermo(mu_nu, T, Neutrino, include_antiparticles)


# =============================================================================
# SELF-TEST AND VALIDATION
# =============================================================================
if __name__ == "__main__":
    print("Lepton and Photon Thermodynamics Module")
    print("=" * 70)
    
    T = 10.0  # MeV
    
    # --- Photons ---
    print(f"\n1. PHOTONS at T = {T} MeV")
    print("-" * 50)
    res_gamma = photon_thermo(T)
    print(f"   n = {res_gamma.n:.6e} fm⁻³")
    print(f"   P = {res_gamma.P:.6e} MeV/fm³")
    print(f"   ε = {res_gamma.e:.6e} MeV/fm³")
    print(f"   s = {res_gamma.s:.6e} fm⁻³")
    print(f"   Check ε/P = {res_gamma.e/res_gamma.P:.1f} (should be 3)")
    
    # --- Neutrinos (using Fermi integrals with m=0) ---
    mu_nu = 50.0
    print(f"\n2. NEUTRINOS (m=0) with μ = {mu_nu} MeV, T = {T} MeV")
    print("-" * 50)
    res_nu = neutrino_thermo(mu_nu, T)
    print(f"   n (net) = {res_nu.n:.6e} fm⁻³")
    print(f"   P       = {res_nu.P:.6e} MeV/fm³")
    print(f"   ε       = {res_nu.e:.6e} MeV/fm³")
    print(f"   s       = {res_nu.s:.6e} fm⁻³")
    print(f"   Check ε/P = {res_nu.e/res_nu.P:.2f} (should be 3 for UR)")
    
    # --- Thermal neutrinos (μ=0) ---
    print(f"\n3. THERMAL NEUTRINOS (μ=0) at T = {T} MeV")
    print("-" * 50)
    res_nu_th = neutrino_thermo(0.0, T)
    print(f"   n (net) = {res_nu_th.n:.6e} fm⁻³ (should be ~0)")
    print(f"   P       = {res_nu_th.P:.6e} MeV/fm³")
    print(f"   Ratio P_ν/P_γ = {res_nu_th.P/res_gamma.P:.6f} (should be 7/8 = 0.875)")
    
    # --- Electrons ---
    mu_e = 50.0
    print(f"\n4. ELECTRONS with μ = {mu_e} MeV, T = {T} MeV")
    print("-" * 50)
    res_e = electron_thermo(mu_e, T)
    print(f"   n (net) = {res_e.n:.6e} fm⁻³")
    print(f"   P       = {res_e.P:.6e} MeV/fm³")
    print(f"   ε       = {res_e.e:.6e} MeV/fm³")
    print(f"   s       = {res_e.s:.6e} fm⁻³")
    
    # --- Muons ---
    mu_mu = 150.0
    print(f"\n5. MUONS with μ = {mu_mu} MeV, T = {T} MeV")
    print("-" * 50)
    print(f"   (Muon mass = {Muon.mass:.2f} MeV)")
    res_mu = muon_thermo(mu_mu, T)
    print(f"   n (net) = {res_mu.n:.6e} fm⁻³")
    print(f"   P       = {res_mu.P:.6e} MeV/fm³")
    print(f"   ε       = {res_mu.e:.6e} MeV/fm³")
    print(f"   s       = {res_mu.s:.6e} fm⁻³")
    