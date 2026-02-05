"""
vmit_thermodynamics_quarks.py
=============================
Low-level thermodynamic functions for vMIT quark matter.

Computes thermodynamic quantities at given (μ, n, T) for u, d, s quarks.
Uses Johns-Ellis-Lattimer (JEL) Fermi integral approximation.

All functions operate at the individual quark level.
The vector field and bag constant effects are handled separately.

Units:
- Energy/mass/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from general_physics_constants import hc, hc3, PI2
from vmit_parameters import VMITParams, get_vmit_default
from general_fermi_integrals import solve_fermi_jel, invert_fermi_density
import general_particles


# =============================================================================
# CONSTANTS
# =============================================================================
G_QUARK = general_particles.get_particle("quark").g_degen  # Degeneracy: spin(2) × color(3) = 6


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class QuarkThermo:
    """Thermodynamic result for a single quark flavor."""
    n: float = 0.0      # Number density (fm⁻³)
    P: float = 0.0      # Pressure (MeV/fm³)
    e: float = 0.0      # Energy density (MeV/fm³)
    s: float = 0.0      # Entropy density (fm⁻³)
    f: float = 0.0      # Free energy density (MeV/fm³)


@dataclass
class VMITThermo:
    """Full thermodynamic result for vMIT quark matter (without leptons)."""
    # Inputs
    n_u: float = 0.0       # Up quark density (fm⁻³)
    n_d: float = 0.0       # Down quark density (fm⁻³)
    n_s: float = 0.0       # Strange quark density (fm⁻³)
    n_B: float = 0.0       # Baryon density (fm⁻³)
    n_C: float = 0.0       # Charge density (fm⁻³)
    n_S: float = 0.0       # Strangeness density (fm⁻³)
    T: float = 0.0         # Temperature (MeV)
    mu_u: float = 0.0      # Up quark physical chemical potential (MeV)
    mu_d: float = 0.0      # Down quark physical chemical potential (MeV)
    mu_s: float = 0.0      # Strange quark physical chemical potential (MeV)
    # Outputs
    P: float = 0.0         # Total pressure (MeV/fm³)
    e: float = 0.0         # Total energy density (MeV/fm³)
    s: float = 0.0         # Total entropy density (fm⁻³)
    f: float = 0.0         # Free energy density f = e - s*T (MeV/fm³)
    Y_C: float = 0.0       # Charge fraction
    Y_S: float = 0.0       # Strangeness fraction
    mu_B: float = 0.0      # Baryon chemical potential (MeV)
    mu_C: float = 0.0      # Charge chemical potential (MeV)
    mu_S: float = 0.0      # Strangeness chemical potential (MeV)

# =============================================================================
# SINGLE QUARK THERMODYNAMICS
# =============================================================================
def compute_quark_thermo(mu_eff: float, T: float, m: float, 
                          include_antiparticles: bool = True) -> QuarkThermo:
    """
    Compute thermodynamic quantities for a single quark flavor.
    
    Args:
        mu_eff: Effective (kinetic) chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        include_antiparticles: Include antiquark contribution
        
    Returns:
        QuarkThermo with n, P, e, s, dn_dmu
    """
    result = solve_fermi_jel(mu_eff, T, m, G_QUARK, 
                             include_antiparticles=include_antiparticles)
    
    return QuarkThermo(
        n=result[0],
        P=result[1],
        e=result[2],
        s=result[3]
    )


def compute_quark_density(mu_eff: float, T: float, m: float,
                           include_antiparticles: bool = True) -> float:
    """Compute quark number density for given effective μ."""
    result = solve_fermi_jel(mu_eff, T, m, G_QUARK, 
                             include_antiparticles=include_antiparticles)
    return result[0]


# =============================================================================
# VECTOR FIELD
# =============================================================================
def compute_vector_field(n_u: float, n_d: float, n_s: float, 
                          params: VMITParams) -> float:
    """
    Compute the vector mean field V.
    
    V = a * ℏc * (n_u + n_d + n_s)
    
    Args:
        n_u, n_d, n_s: Quark number densities (fm⁻³)
        params: vMIT model parameters
        
    Returns:
        V: Vector field strength (MeV)
    """
    n_total = n_u + n_d + n_s
    return params.a * hc * n_total


def compute_vector_pressure(n_u: float, n_d: float, n_s: float,
                             params: VMITParams) -> float:
    """
    Compute pressure contribution from vector field.
    
    P_V = (1/2) * a * ℏc * (n_u + n_d + n_s)²
    
    Returns:
        P_V: Vector pressure contribution (MeV/fm³)
    """
    n_total = n_u + n_d + n_s
    return 0.5 * params.a * hc * n_total**2


def compute_vector_energy(n_u: float, n_d: float, n_s: float,
                           params: VMITParams) -> float:
    """
    Compute energy density contribution from vector field.
    
    e_V = (1/2) * a * ℏc * (n_u + n_d + n_s)² (same as P_V)
    
    Returns:
        e_V: Vector energy density contribution (MeV/fm³)
    """
    return compute_vector_pressure(n_u, n_d, n_s, params)


# =============================================================================
# BAG CONSTANT
# =============================================================================
def compute_bag_pressure(params: VMITParams) -> float:
    """
    Compute pressure contribution from bag constant.
    
    P_B = -B / (ℏc)³
    
    Returns:
        P_B: Bag pressure contribution (MeV/fm³), negative
    """
    return -params.B / hc3


def compute_bag_energy(params: VMITParams) -> float:
    """
    Compute energy density contribution from bag constant.
    
    e_B = +B / (ℏc)³
    
    Returns:
        e_B: Bag energy density contribution (MeV/fm³), positive
    """
    return params.B / hc3


# =============================================================================
# EFFECTIVE CHEMICAL POTENTIAL
# =============================================================================
def compute_mu_effective(mu: float, n_u: float, n_d: float, n_s: float,
                          params: VMITParams) -> float:
    """
    Compute effective (kinetic) chemical potential from physical μ.
    
    μ_eff = μ - V = μ - a * ℏc * (n_u + n_d + n_s)
    
    The effective μ is what enters the Fermi integrals.
    """
    V = compute_vector_field(n_u, n_d, n_s, params)
    return mu - V


def compute_effective_mu_quarks(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    params: VMITParams
) -> Tuple[float, float, float]:
    """
    Compute effective (kinetic) chemical potentials for all three quark flavors.
    
    μ_eff = μ - V where V = a * ℏc * (n_u + n_d + n_s)
    
    This is the vMIT equivalent of ZL's compute_effective_mu_nucleons.
    
    Args:
        mu_u, mu_d, mu_s: Physical chemical potentials (MeV)
        n_u, n_d, n_s: Quark number densities (fm⁻³)
        params: vMIT parameters
        
    Returns:
        (mu_u_eff, mu_d_eff, mu_s_eff): Effective chemical potentials (MeV)
    """
    V = compute_vector_field(n_u, n_d, n_s, params)
    return mu_u - V, mu_d - V, mu_s - V


@dataclass
class QuarkMuDensity:
    """Result of computing effective μ and densities for all quark flavors."""
    # Physical chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    # Input number densities (fm⁻³)
    n_u: float = 0.0
    n_d: float = 0.0
    n_s: float = 0.0
    # Effective chemical potentials (MeV)
    mu_u_eff: float = 0.0
    mu_d_eff: float = 0.0
    mu_s_eff: float = 0.0
    # Computed densities (fm⁻³)
    n_u_calc: float = 0.0
    n_d_calc: float = 0.0
    n_s_calc: float = 0.0
    
    # Derived properties
    @property
    def n_B_calc(self) -> float:
        """Baryon density: n_B = (n_u + n_d + n_s) / 3"""
        return (self.n_u_calc + self.n_d_calc + self.n_s_calc) / 3.0
    
    @property
    def n_C_calc(self) -> float:
        """Charge density: n_C = (2/3)n_u - (1/3)n_d - (1/3)n_s"""
        return (2.0/3.0)*self.n_u_calc - (1.0/3.0)*self.n_d_calc - (1.0/3.0)*self.n_s_calc
    
    @property
    def n_S_calc(self) -> float:
        """Strangeness density: n_S = n_s"""
        return self.n_s_calc

    @property
    def n_B(self) -> float:
        """Baryon density: n_B = (n_u + n_d + n_s) / 3"""
        return (self.n_u + self.n_d + self.n_s) / 3.0
    
    @property
    def n_C(self) -> float:
        """Charge density: n_C = (2/3)n_u - (1/3)n_d - (1/3)n_s"""
        return (2.0/3.0)*self.n_u - (1.0/3.0)*self.n_d - (1.0/3.0)*self.n_s
    
    @property
    def n_S(self) -> float:
        """Strangeness density: n_S = n_s"""
        return self.n_s
    
    @property
    def mu_B(self) -> float:
        """Baryon chemical potential: μ_B = 2μ_d + μ_u"""
        return 2*self.mu_d + self.mu_u
    
    @property
    def mu_C(self) -> float:
        """Charge chemical potential: μ_C = μ_u - μ_d"""
        return self.mu_u - self.mu_d
    
    @property
    def mu_S(self) -> float:
        """Strangeness chemical potential: μ_S = μ_s - μ_d"""
        return self.mu_s - self.mu_d
    


def compute_quark_dentities_for_solver(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    T: float, params: VMITParams
) -> QuarkMuDensity:
    """
    Compute effective chemical potentials and resulting densities.
    Useful in solvers.
    
    This combines compute_effective_mu_quarks and compute_quark_density
    into a single function call - a common pattern in EOS solvers.
    
    Args:
        mu_u, mu_d, mu_s: Physical chemical potentials (MeV)
        n_u, n_d, n_s: Current quark densities for mean field (fm⁻³)
        T: Temperature (MeV)
        params: vMIT parameters
        
    Returns:
        QuarkMuDensity with effective μ and calculated densities
    """
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Effective chemical potentials
    mu_u_eff, mu_d_eff, mu_s_eff = compute_effective_mu_quarks(
        mu_u, mu_d, mu_s, n_u, n_d, n_s, params)
    
    # Compute densities from effective μ
    n_u_calc = compute_quark_density(mu_u_eff, T, m_u)
    n_d_calc = compute_quark_density(mu_d_eff, T, m_d)
    n_s_calc = compute_quark_density(mu_s_eff, T, m_s)
    
    return QuarkMuDensity(
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
        mu_u_eff=mu_u_eff, mu_d_eff=mu_d_eff, mu_s_eff=mu_s_eff,
        n_u_calc=n_u_calc, n_d_calc=n_d_calc, n_s_calc=n_s_calc,
        n_u=n_u, n_d=n_d, n_s=n_s
    )


def compute_mu_physical(mu_eff: float, n_u: float, n_d: float, n_s: float,
                         params: VMITParams) -> float:
    """
    Compute physical chemical potential from effective μ.
    
    μ = μ_eff + V = μ_eff + a * ℏc * (n_u + n_d + n_s)
    """
    V = compute_vector_field(n_u, n_d, n_s, params)
    return mu_eff + V


# =============================================================================
# FULL QUARK MATTER THERMODYNAMICS (without leptons)
# =============================================================================
def compute_vmit_thermo_from_mu_n(
    mu_u: float, mu_d: float, mu_s: float,
    n_u: float, n_d: float, n_s: float,
    T: float, params: VMITParams = None
) -> VMITThermo:
    """
    Compute full vMIT thermodynamics from (μ_u, μ_d, μ_s, n_u, n_d, n_s, T).
    
    This is the main user-facing function that takes all inputs and returns
    complete thermodynamic results for quark matter (without leptons).
    
    Note: having both n_i and μ_i as inputs seems redundant. However, here n_i
          plays the role of "mean fields" (as in ZL). n_i(μ_i,T) is computed 
          directly in the solving functions.
          In solvers we will require that n_i(µ_i,T) = n_i namely n_i_calc = n_i.
    
    Args:
        mu_u, mu_d, mu_s: Physical chemical potentials (MeV)
        n_u, n_d, n_s: Quark number densities (fm⁻³)
        T: Temperature (MeV)
        params: vMIT model parameters (uses default if None)
        
    Returns:
        VMITThermo with P, e, s, f, Y_C, Y_S, mu_B, mu_C, mu_S
    """
    if params is None:
        params = get_vmit_default()
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Compute effective chemical potentials
    mu_u_eff, mu_d_eff, mu_s_eff = compute_effective_mu_quarks(
        mu_u, mu_d, mu_s, n_u, n_d, n_s, params
    )
    
    # Kinetic contributions from Fermi integrals
    thermo_u = compute_quark_thermo(mu_u_eff, T, m_u)
    thermo_d = compute_quark_thermo(mu_d_eff, T, m_d)
    thermo_s = compute_quark_thermo(mu_s_eff, T, m_s)

    n_u_calc = thermo_u.n
    n_d_calc = thermo_d.n
    n_s_calc = thermo_s.n
    
    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s
    
    # Vector and bag contributions
    P_V = compute_vector_pressure(n_u_calc, n_d_calc, n_s_calc, params)
    P_B = compute_bag_pressure(params)
    e_B = compute_bag_energy(params)
    
    # Total thermodynamics (e_V = P_V for vector field)
    P_total = P_kin + P_V + P_B
    e_total = e_kin + P_V + e_B
    s_total = s_kin
    
    # Free energy density
    f_total = e_total - s_total * T
    
    n_B = (n_u_calc + n_d_calc + n_s_calc) / 3.0
    n_C = (2.0/3.0)*n_u_calc - (1.0/3.0)*n_d_calc - (1.0/3.0)*n_s_calc
    n_S = n_s_calc
    Y_C = n_C / n_B 
    Y_S = n_S / n_B 
    
    # Conserved charge chemical potentials
    mu_B = mu_u + 2*mu_d 
    mu_C = mu_u - mu_d
    mu_S = mu_s - mu_d
    
    return VMITThermo(
        n_u=n_u_calc, n_d=n_d_calc, n_s=n_s_calc, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_C=Y_C, Y_S=Y_S,
        T=T,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        P=P_total, e=e_total, s=s_total, f=f_total,
    )


def compute_quark_matter_thermo_from_n(
    n_u: float, n_d: float, n_s: float, T: float, 
    params: VMITParams = None
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute full quark matter thermodynamics from given densities.
    
    This inverts Fermi integrals to find μ_eff, then computes all quantities.
    
    Args:
        n_u, n_d, n_s: Quark number densities (fm⁻³)
        T: Temperature (MeV)
        params: vMIT parameters
        
    Returns:
        (mu_u, mu_d, mu_s, P_quarks, e_quarks, s_quarks, n_B)
        - mu_u, mu_d, mu_s: Physical chemical potentials (MeV)
        - P_quarks, e_quarks: Pressure and energy density (MeV/fm³)
        - s_quarks: Entropy density (fm⁻³)
        - n_B: Baryon density (fm⁻³)
    """
    if params is None:
        params = get_vmit_default()
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    n_total = n_u + n_d + n_s
    n_B = n_total / 3.0
    
    # Invert Fermi integrals to get effective μ
    mu_u_eff = invert_fermi_density(n_u, T, m_u, G_QUARK)
    mu_d_eff = invert_fermi_density(n_d, T, m_d, G_QUARK)
    mu_s_eff = invert_fermi_density(n_s, T, m_s, G_QUARK)
    
    # Physical chemical potentials
    mu_u = compute_mu_physical(mu_u_eff, n_u, n_d, n_s, params)
    mu_d = compute_mu_physical(mu_d_eff, n_u, n_d, n_s, params)
    mu_s = compute_mu_physical(mu_s_eff, n_u, n_d, n_s, params)
    
    # Compute thermodynamics
    thermo_u = compute_quark_thermo(mu_u_eff, T, m_u)
    thermo_d = compute_quark_thermo(mu_d_eff, T, m_d)
    thermo_s = compute_quark_thermo(mu_s_eff, T, m_s)
    
    # Total kinetic contributions
    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s
    
    # Add vector and bag contributions
    P_V = compute_vector_pressure(n_u, n_d, n_s, params)
    P_B = compute_bag_pressure(params)
    e_B = compute_bag_energy(params)
    
    P_quarks = P_kin + P_V + P_B
    e_quarks = e_kin + P_V + e_B  # Note: e_V = P_V for vector field
    s_quarks = s_kin
    
    return (mu_u, mu_d, mu_s, P_quarks, e_quarks, s_quarks, n_B)


def compute_quark_matter_thermo_from_mu(
    mu_u: float, mu_d: float, mu_s: float, T: float, 
    params: VMITParams = None
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute full quark matter thermodynamics from given physical chemical potentials.
    
    This requires solving for the self-consistent densities since:
    μ_eff = μ - V(n_u, n_d, n_s) and n = n(μ_eff)
    
    Uses scipy.optimize.root for robust convergence.
    
    Args:
        mu_u, mu_d, mu_s: Physical chemical potentials (MeV)
        T: Temperature (MeV)
        params: vMIT parameters
        
    Returns:
        (n_u, n_d, n_s, P_quarks, e_quarks, s_quarks, n_B)
        - n_u, n_d, n_s: Quark number densities (fm⁻³)
        - P_quarks, e_quarks: Pressure and energy density (MeV/fm³)
        - s_quarks: Entropy density (fm⁻³)
        - n_B: Baryon density (fm⁻³)
    """
    from scipy.optimize import root
    
    if params is None:
        params = get_vmit_default()
    
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    def equations(mu_eff_vec):
        mu_u_eff, mu_d_eff, mu_s_eff = mu_eff_vec
        n_u = compute_quark_thermo(mu_u_eff, T, m_u).n
        n_d = compute_quark_thermo(mu_d_eff, T, m_d).n
        n_s = compute_quark_thermo(mu_s_eff, T, m_s).n
        V = compute_vector_field(n_u, n_d, n_s, params)
        return [mu_u_eff + V - mu_u,
                mu_d_eff + V - mu_d,
                mu_s_eff + V - mu_s]
    
    # Initial guess: V = 0 (free gas limit)
    x0 = [mu_u, mu_d, mu_s]
    sol = root(equations, x0, method='hybr')
    
    mu_u_eff, mu_d_eff, mu_s_eff = sol.x
    
    # Final thermodynamics
    thermo_u = compute_quark_thermo(mu_u_eff, T, m_u)
    thermo_d = compute_quark_thermo(mu_d_eff, T, m_d)
    thermo_s = compute_quark_thermo(mu_s_eff, T, m_s)
    
    n_u, n_d, n_s = thermo_u.n, thermo_d.n, thermo_s.n
    n_B = (n_u + n_d + n_s) / 3.0
    
    # Total kinetic contributions
    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s
    
    # Add vector and bag contributions
    P_V = compute_vector_pressure(n_u, n_d, n_s, params)
    P_B = compute_bag_pressure(params)
    e_B = compute_bag_energy(params)
    
    P_quarks = P_kin + P_V + P_B
    e_quarks = e_kin + P_V + e_B
    s_quarks = s_kin
    
    return (n_u, n_d, n_s, P_quarks, e_quarks, s_quarks, n_B)



# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("vMIT Thermodynamics Test")
    print("=" * 50)
    
    params = get_vmit_default()
    print(f"Parameters: B^1/4={params.B4} MeV, a={params.a} fm²")
    
    # Test at n_B = 2*n0
    n_B = 0.32
    n_u = n_B  # Equal densities for test
    n_d = n_B
    n_s = n_B
    T = 50.0
    
    print(f"\nTest at n_B={n_B} fm⁻³, T={T} MeV:")
    
    mu_u, mu_d, mu_s, P, e, s, nB = compute_quark_matter_thermo_from_n(
        n_u, n_d, n_s, T, params
    )
    
    print(f"  μ_u = {mu_u:.2f} MeV")
    print(f"  μ_d = {mu_d:.2f} MeV")
    print(f"  μ_s = {mu_s:.2f} MeV")
    print(f"  P = {P:.2f} MeV/fm³")
    print(f"  ε = {e:.2f} MeV/fm³")
    print(f"  s = {s:.4f} fm⁻³")
    print(f"  n_B = {nB:.4f} fm⁻³")
    
    print("\nOK!")
