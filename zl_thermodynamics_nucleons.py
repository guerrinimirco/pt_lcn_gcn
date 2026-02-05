"""
zl_thermodynamics_nucleons.py
=============================
Low-level thermodynamic functions for ZL hadronic matter.

Computes thermodynamic quantities at given (μ, n, T) for nucleons (p, n).
Uses Johns-Ellis-Lattimer (JEL) Fermi integral approximation.

Units:
- Energy/mass/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from general_physics_constants import hc, hc3, PI2
from zl_parameters import ZLParams, get_zl_default
from general_fermi_integrals import solve_fermi_jel, invert_fermi_density
import general_particles


# =============================================================================
# CONSTANTS
# =============================================================================
G_NUCLEON = general_particles.get_particle("p").g_degen  # Degeneracy: 2 (spin)


# =============================================================================
# SINGLE NUCLEON THERMODYNAMICS without MF potential (only effective Fermi integrals)
# =============================================================================

# Class for Nucleon thermodynamics
@dataclass
class NucleonThermo:
    """Thermodynamic result for a single nucleon species."""
    n: float = 0.0      # Number density (fm⁻³)
    P: float = 0.0      # Pressure (MeV/fm³)
    e: float = 0.0      # Energy density (MeV/fm³)
    s: float = 0.0      # Entropy density (fm⁻³)


# Compute single nucleon thermodynamics
def compute_nucleon_thermo(mu_eff: float, T: float, m: float,
                            include_antiparticles: bool = True) -> NucleonThermo:
    """
    Compute thermodynamic quantities for a single nucleon species.
    
    Args:
        mu_eff: Effective chemical potential (MeV)
        T: Temperature (MeV)
        m: Nucleon effective mass (MeV)
        include_antiparticles: Include antinucleon contribution
        
    Returns:
        NucleonThermo with n, P, e, s
    """
    result = solve_fermi_jel(mu_eff, T, m, G_NUCLEON,
                             include_antiparticles=include_antiparticles)
    
    return NucleonThermo(
        n=result[0],
        P=result[1],
        e=result[2],
        s=result[3]
    )


def compute_nucleon_density(mu_eff: float, T: float, m: float,
                             include_antiparticles: bool = True) -> float:
    """Compute nucleon number density for given effective μ."""
    result = solve_fermi_jel(mu_eff, T, m, G_NUCLEON,
                             include_antiparticles=include_antiparticles)
    return result[0]

# =============================================================================
# COMPLETE THERMODYNAMICS
# =============================================================================
@dataclass
class ZLThermo:
    """Full thermodynamic result for ZL nucleon matter."""
    n_p: float = 0.0       # Proton density (fm⁻³)
    n_n: float = 0.0       # Neutron density (fm⁻³)
    Y_p: float = 0.0       # Proton fraction
    Y_n: float = 0.0       # Neutron fraction
    T: float = 0.0         # Temperature (MeV)
    mu_p: float = 0.0      # Proton physical chemical potential (MeV)
    mu_n: float = 0.0      # Neutron physical chemical potential (MeV)
    P: float = 0.0         # Total pressure (MeV/fm³)
    e: float = 0.0         # Total energy density (MeV/fm³)
    s: float = 0.0         # Total entropy density (fm⁻³)
    f: float = 0.0         # Free energy density f = e - s*T (MeV/fm³)
    Y_C: float = 0.0       # Charge fraction
    Y_S: float = 0.0       # Strangeness fraction
    n_B: float = 0.0       # Baryon density (fm⁻³)
    n_C: float = 0.0       # Charge density (fm⁻³)
    n_S: float = 0.0       # Strangeness density (fm⁻³)
    mu_B: float = 0.0      # Baryon chemical potential (MeV)
    mu_C: float = 0.0      # Charge chemical potential (MeV)
    mu_S: float = 0.0      # Strangeness chemical potential (MeV)
    


def compute_zl_thermo_from_mu_n(mu_p: float, mu_n: float, n_p: float, n_n: float, 
                       T: float, params: ZLParams = None) -> ZLThermo:
    """
    Compute full ZL thermodynamics from (μ_p, μ_n, n_p, n_n, T).
    
    This is the main user-facing function that takes all inputs and returns
    complete thermodynamic results.
    Note: having both n_i and µ_i as inputs seems to be reduntant. However, here we treat n_i with the role of, say,
          "mean fields" (as e.g. sigma in RMF models). n_i(µ_i,T) is computed directly in solving functions. 
           In solvers we will require that n_i(µ_i,T) = n_i namely n_i_calc = n_i.
    
    Args:
        mu_p: Proton chemical podential (MeV)
        mu_n: Neutron chemical potential (MeV)
        n_p: Proton density (fm⁻³)
        n_n: Neutron density (fm⁻³)
        T: Temperature (MeV)
        params: ZL model parameters (uses default if None)
        
    Returns:
        ZLThermo with P, e, s, f, Y_C, mu_B, mu_C
    """
    if params is None:
        params = get_zl_default()
    
    m_p, m_n = params.m_p, params.m_n
    
    # Effective chemical potentials
    mu_p_eff, mu_n_eff = compute_effective_mu_nucleons(mu_p, mu_n, n_p, n_n, params)
    
    # -------------------------------------------------------------------------
    # Kinetic contributions from Fermi integrals
    # -------------------------------------------------------------------------
    thermo_p = compute_nucleon_thermo(mu_p_eff, T, m_p)
    thermo_n = compute_nucleon_thermo(mu_n_eff, T, m_n)
    
    n_p_calc = thermo_p.n
    n_n_calc = thermo_n.n

    P_kin = thermo_p.P + thermo_n.P
    e_kin = thermo_p.e + thermo_n.e
    s_kin = thermo_p.s + thermo_n.s
    
    # -------------------------------------------------------------------------
    # Total thermodynamics
    # -------------------------------------------------------------------------
    
    n_B = n_p_calc + n_n_calc
    n_C = n_p_calc
    n_S = 0.0

    Y_p = n_p_calc / n_B
    Y_n = n_n_calc / n_B
    Y_C = n_C / n_B
    Y_S = n_S / n_B

    mu_B = mu_n  # Baryon chemical potential
    mu_C = mu_p - mu_n  # Charge chemical potential

    # Interaction energy density
    V_int = compute_V_interaction(n_p_calc, n_n_calc, params)
    
    # Interaction pressure
    P_int = compute_P_interaction(n_p_calc, n_n_calc, params)

    P_total = P_kin + P_int
    e_total = e_kin + V_int
    s_total = s_kin  # Entropy is purely kinetic in this model
    
    # Free energy density
    f_total = e_total - s_total * T
    
    return ZLThermo(
        n_p=n_p_calc, n_n=n_n_calc, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_p=Y_p, Y_n=Y_n, Y_C=Y_C, Y_S=Y_S,
        T=T,
        mu_p=mu_p, mu_n=mu_n, mu_B=mu_B, mu_C=mu_C,
        P=P_total, e=e_total, s=s_total, f=f_total,
    )


def compute_zl_thermo_from_mu(
    mu_p: float, mu_n: float, T: float, params: ZLParams = None
) -> ZLThermo:
    """
    Compute full ZL thermodynamics from physical chemical potentials (μ_p, μ_n, T).
    
    This requires solving for the self-consistent densities since:
    μ_eff = μ - μHv(n_p, n_n) and n = n(μ_eff)
    
    Uses scipy.optimize.root for robust convergence.
    
    Args:
        mu_p, mu_n: Physical chemical potentials (MeV)
        T: Temperature (MeV)
        params: ZL model parameters (uses default if None)
        
    Returns:
        ZLThermo with n_p, n_n, P, e, s, f, Y_C, mu_B, mu_C
    """
    from scipy.optimize import root
    
    if params is None:
        params = get_zl_default()
    
    m_p, m_n = params.m_p, params.m_n
    
    def equations(x):
        n_p, n_n = x
        
        # Compute effective μ from physical μ and densities
        mu_p_eff, mu_n_eff = compute_effective_mu_nucleons(mu_p, mu_n, n_p, n_n, params)
        
        # Compute densities from effective μ
        thermo_p = compute_nucleon_thermo(mu_p_eff, T, m_p)
        thermo_n = compute_nucleon_thermo(mu_n_eff, T, m_n)
        
        return [thermo_p.n - n_p, thermo_n.n - n_n]
    
    # Initial guess: use free gas result (interaction = 0)
    thermo_p_guess = compute_nucleon_thermo(mu_p, T, m_p)
    thermo_n_guess = compute_nucleon_thermo(mu_n, T, m_n)
    x0 = [max(thermo_p_guess.n, 1e-6), max(thermo_n_guess.n, 1e-6)]
    
    sol = root(equations, x0, method='hybr')
    if not sol.success:
        sol = root(equations, x0, method='lm')
    
    n_p, n_n = sol.x
    
    # Compute full thermodynamics with converged densities
    return compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)


def compute_zl_thermo_from_n(
    n_B: float, Y_C: float, T: float, params: ZLParams = None
) -> ZLThermo:
    """
    Compute full ZL thermodynamics from densities (n_B, Y_C, T).
    
    This inverts the Fermi integrals to find effective μ, then adds the 
    interaction contributions to get physical μ.
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction
        T: Temperature (MeV)
        params: ZL model parameters (uses default if None)
        
    Returns:
        ZLThermo with mu_p, mu_n, P, e, s, Y_p, mu_B, mu_C
    """
    if params is None:
        params = get_zl_default()
    
    m_p, m_n = params.m_p, params.m_n
    n_p = Y_C * n_B
    n_n = (1 - Y_C) * n_B
    
    # Invert Fermi integrals to get effective μ
    mu_p_eff = invert_fermi_density(n_p, T, m_p, G_NUCLEON)
    mu_n_eff = invert_fermi_density(n_n, T, m_n, G_NUCLEON)
    
    # Add interaction contributions to get physical μ
    mu_pHv = compute_mu_pHv(n_p, n_n, params)
    mu_nHv = compute_mu_nHv(n_p, n_n, params)
    
    mu_p = mu_p_eff + mu_pHv
    mu_n = mu_n_eff + mu_nHv
    
    # Compute full thermodynamics
    return compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, params)


# =============================================================================
# ZL INTERACTION POTENTIAL
# =============================================================================
def compute_V_interaction(n_p: float, n_n: float, params: ZLParams) -> float:
    """
    Compute the ZL interaction energy density.
    
    V = 4*n_p*n_n * [a0/n0 + b0/n0 * (nB/n0)^(γ-1)]
      + (n_n-n_p)² * [a1/n0 + b1/n0 * (nB/n0)^(γ1-1)]
    
    Args:
        n_p: Proton density (fm⁻³)
        n_n: Neutron density (fm⁻³)
        params: ZL model parameters
        
    Returns:
        V: Interaction energy density (MeV/fm³)
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0
    
    n0 = params.n0
    x = n_B / n0  # Dimensionless density
    
    # Symmetric matter contribution (attractive at saturation)
    sym_term = 4.0 * n_p * n_n * (
        params.a0 / n0 + params.b0 / n0 * x**(params.gamma - 1)
    )
    
    # Symmetry energy contribution
    asym_term = (n_n - n_p)**2 * (
        params.a1 / n0 + params.b1 / n0 * x**(params.gamma1 - 1)
    )
    
    return sym_term + asym_term


def compute_P_interaction(n_p: float, n_n: float, params: ZLParams) -> float:
    """
    Compute the ZL interaction pressure contribution.
    
    P_int = 4*n_p*n_n * [a0/n0 + γ·b0/n0 * x^(γ-1)]
          + (n_n-n_p)² * [a1/n0 + γ1·b1/n0 * x^(γ1-1)]
    
    Note: The power-law terms have extra γ and γ1 factors compared to V.
    
    Args:
        n_p: Proton density (fm⁻³)
        n_n: Neutron density (fm⁻³)
        params: ZL parameters
        
    Returns:
        P_int: Interaction pressure (MeV/fm³)
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0
    
    n0 = params.n0
    x = n_B / n0
    
    gamma, gamma1 = params.gamma, params.gamma1
    a0, b0, a1, b1 = params.a0, params.b0, params.a1, params.b1
    
    # Symmetric pressure: 4*np*nn * [a0/n0 + γ*b0/n0 * x^(γ-1)]
    P_sym = 4.0 * n_p * n_n * (a0 / n0 + gamma * b0 / n0 * x**(gamma - 1))
    
    # Asymmetric pressure: (nn-np)² * [a1/n0 + γ1*b1/n0 * x^(γ1-1)]
    P_asym = (n_n - n_p)**2 * (a1 / n0 + gamma1 * b1 / n0 * x**(gamma1 - 1))
    
    return P_sym + P_asym

# =============================================================================
# INTERACTION CHEMICAL POTENTIALS
# =============================================================================
def compute_mu_pHv(n_p: float, n_n: float, params: ZLParams) -> float:
    """
    Compute μpHv = ∂fHv/∂n_p at fixed n_n.
    
    This is the interaction contribution to the proton chemical potential.
    
    Formula (from Mathematica):
        μpHv = 4*nn*(a0/n0 + b0/n0*x^(γ-1))
             - 2*(nn-np)*(a1/n0 + b1/n0*x^(γ1-1))
             + 4*b0*nn*np*x^(γ-2)*(γ-1)/n0²
             + b1*(nn-np)²*x^(γ1-2)*(γ1-1)/n0²
    
    where x = nB/n0 = (np+nn)/n0.
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0
    
    n0 = params.n0
    x = n_B / n0
    gamma, gamma1 = params.gamma, params.gamma1
    a0, b0, a1, b1 = params.a0, params.b0, params.a1, params.b1
    
    # Term 1: 4*nn*(a0/n0 + b0/n0*x^(γ-1))
    term1 = 4.0 * n_n * (a0/n0 + b0/n0 * x**(gamma - 1))
    
    # Term 2: -2*(nn-np)*(a1/n0 + b1/n0*x^(γ1-1))
    term2 = -2.0 * (n_n - n_p) * (a1/n0 + b1/n0 * x**(gamma1 - 1))
    
    # Term 3: 4*b0*nn*np*x^(γ-2)*(γ-1)/n0²
    term3 = 4.0 * b0 * n_n * n_p * x**(gamma - 2) * (gamma - 1) / n0**2
    
    # Term 4: b1*(nn-np)²*x^(γ1-2)*(γ1-1)/n0²
    term4 = b1 * (n_n - n_p)**2 * x**(gamma1 - 2) * (gamma1 - 1) / n0**2
    
    return term1 + term2 + term3 + term4


def compute_mu_nHv(n_p: float, n_n: float, params: ZLParams) -> float:
    """
    Compute μnHv = ∂fHv/∂n_n at fixed n_p.
    
    This is the interaction contribution to the neutron chemical potential.
    
    Formula (from Mathematica):
        μnHv = 4*np*(a0/n0 + b0/n0*x^(γ-1))
             + 2*(nn-np)*(a1/n0 + b1/n0*x^(γ1-1))
             + 4*b0*nn*np*x^(γ-2)*(γ-1)/n0²
             + b1*(nn-np)²*x^(γ1-2)*(γ1-1)/n0²
    """
    n_B = n_p + n_n
    if n_B < 1e-15:
        return 0.0
    
    n0 = params.n0
    x = n_B / n0
    gamma, gamma1 = params.gamma, params.gamma1
    a0, b0, a1, b1 = params.a0, params.b0, params.a1, params.b1
    
    # Term 1: 4*np*(a0/n0 + b0/n0*x^(γ-1))
    term1 = 4.0 * n_p * (a0/n0 + b0/n0 * x**(gamma - 1))
    
    # Term 2: +2*(nn-np)*(a1/n0 + b1/n0*x^(γ1-1))
    term2 = 2.0 * (n_n - n_p) * (a1/n0 + b1/n0 * x**(gamma1 - 1))
    
    # Term 3: 4*b0*nn*np*x^(γ-2)*(γ-1)/n0²
    term3 = 4.0 * b0 * n_n * n_p * x**(gamma - 2) * (gamma - 1) / n0**2
    
    # Term 4: b1*(nn-np)²*x^(γ1-2)*(γ1-1)/n0²
    term4 = b1 * (n_n - n_p)**2 * x**(gamma1 - 2) * (gamma1 - 1) / n0**2
    
    return term1 + term2 + term3 + term4


# =============================================================================
# EFFECTIVE CHEMICAL POTENTIALS
# =============================================================================
def compute_effective_mu_nucleons(mu_p: float, mu_n: float, n_p: float, n_n: float, 
                          params: ZLParams) -> Tuple[float, float]:
    """
    Compute effective (kinetic) chemical potentials for nucleons.
    
    The effective chemical potential is:
        μ*_p = μ_p - μpHv  (where μpHv = ∂fHv/∂n_p at fixed n_n)
        μ*_n = μ_n - μnHv  (where μnHv = ∂fHv/∂n_n at fixed n_p)
    
    Args:
        mu_p, mu_n: Physical chemical potentials (MeV)
        n_p, n_n: Nucleon densities (fm⁻³)
        params: ZL parameters
        
    Returns:
        (mu_p_eff, mu_n_eff): Effective chemical potentials (MeV)
    """
    mu_pHv = compute_mu_pHv(n_p, n_n, params)
    mu_nHv = compute_mu_nHv(n_p, n_n, params)
    
    mu_p_eff = mu_p - mu_pHv
    mu_n_eff = mu_n - mu_nHv
    
    return mu_p_eff, mu_n_eff


@dataclass
class NucleonMuDensity:
    """Result of computing effective μ and densities for nucleons."""
    # Physical chemical potentials (MeV)
    mu_p: float = 0.0
    mu_n: float = 0.0
    # Input number densities (fm⁻³)
    n_p: float = 0.0
    n_n: float = 0.0
    # Effective chemical potentials (MeV)
    mu_p_eff: float = 0.0
    mu_n_eff: float = 0.0
    # Computed densities (fm⁻³)
    n_p_calc: float = 0.0
    n_n_calc: float = 0.0
    
    # Derived properties (using input densities)
    @property
    def n_B(self) -> float:
        """Baryon density: n_B = n_p + n_n"""
        return self.n_p + self.n_n
    
    @property
    def n_C(self) -> float:
        """Charge density: n_C = n_p"""
        return self.n_p
    
    # Derived properties (using calculated densities)
    @property
    def n_B_calc(self) -> float:
        """Calculated baryon density: n_B = n_p_calc + n_n_calc"""
        return self.n_p_calc + self.n_n_calc
    
    @property
    def n_C_calc(self) -> float:
        """Calculated charge density: n_C = n_p_calc"""
        return self.n_p_calc
    
    @property
    def mu_B(self) -> float:
        """Baryon chemical potential: μ_B = μ_n"""
        return self.mu_n
    
    @property
    def mu_C(self) -> float:
        """Charge chemical potential: μ_C = μ_p - μ_n"""
        return self.mu_p - self.mu_n
    
    @property
    def mu_S(self) -> float:
        """Charge chemical potential: μ_C = μ_p - μ_n"""
        return 0


def compute_nucleon_densities_for_solver(
    mu_p: float, mu_n: float,
    n_p: float, n_n: float,
    T: float, params: ZLParams
) -> NucleonMuDensity:
    """
    Compute effective chemical potentials and resulting densities.
    Useful in solvers.
    
    This combines compute_effective_mu_nucleons and compute_nucleon_density
    into a single function call - a common pattern in EOS solvers.
    
    Args:
        mu_p, mu_n: Physical chemical potentials (MeV)
        n_p, n_n: Current nucleon densities for mean field (fm⁻³)
        T: Temperature (MeV)
        params: ZL parameters
        
    Returns:
        NucleonMuDensity with effective μ and calculated densities
    """
    m_p, m_n = params.m_p, params.m_n
    
    # Effective chemical potentials
    mu_p_eff, mu_n_eff = compute_effective_mu_nucleons(mu_p, mu_n, n_p, n_n, params)
    
    # Compute densities from effective μ
    n_p_calc = compute_nucleon_density(mu_p_eff, T, m_p)
    n_n_calc = compute_nucleon_density(mu_n_eff, T, m_n)
    
    return NucleonMuDensity(
        mu_p=mu_p, mu_n=mu_n,
        n_p=n_p, n_n=n_n,
        mu_p_eff=mu_p_eff, mu_n_eff=mu_n_eff,
        n_p_calc=n_p_calc, n_n_calc=n_n_calc
    )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("ZL Thermodynamics Test")
    print("=" * 50)
    
    # Test at n_B = n0
    n_p = 0.08
    n_n = 0.08
    T = 10.0
    m = 938
    
    print(f"\nTest at n_p={n_p}, n_n={n_n} fm⁻³, T={T} MeV, M*={m} MeV:")
    
    # Find μ_eff for proton using general invert_fermi_density
    mu_eff_p = invert_fermi_density(n_p, T, m, G_NUCLEON)
    thermo_p = compute_nucleon_thermo(mu_eff_p, T, m)
    
    print(f"  μ_eff(p) = {mu_eff_p:.2f} MeV")
    print(f"  n_p = {thermo_p.n:.4f} fm⁻³")
    print(f"  P_p = {thermo_p.P:.2f} MeV/fm³")
    
    print("\nOK!")
