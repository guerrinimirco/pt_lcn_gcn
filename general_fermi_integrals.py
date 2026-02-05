"""
general_fermi_integrals.py
========================
Efficient evaluation of relativistic Fermi-Dirac integrals for EOS calculations.

Methods:
    1. JEL (Johns, Ellis, Lattimer 1996) - Fast rational approximation, ~10⁻⁴ accuracy
    2. Gauss-Laguerre quadrature - Higher accuracy alternative
    3. Analytic T=0 and m=0 limits

Reference:
    Johns, Ellis & Lattimer, ApJ 473, 1020 (1996)
    https://arxiv.org/abs/2311.03025v2

Units: All quantities in MeV (energy/mass) and fm (length)
Returns: n (fm⁻³), P (MeV/fm³), e (MeV/fm³), s (fm⁻³), ns (fm⁻³)
"""
import numpy as np
import scipy.integrate as integrate
from general_physics_constants import hc, hc3, PI2

# Numba JIT decorator - use identity if numba not available
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# =============================================================================
# JEL APPROXIMATION PARAMETERS (Table 9 of JEL 1996)
# =============================================================================
# Fermion parameters (M=3, N=3)
aJEL = 0.433
MJEL = 3
NJEL = 3

# Coefficient matrix p_{mn} from Table 9
pmn = np.array([
    [5.34689, 18.0517, 21.3422, 8.53240],
    [16.8441, 55.7051, 63.6901, 24.6213],
    [17.4708, 56.3902, 62.1319, 23.2602],
    [6.07364, 18.9992, 20.0285, 7.11153]
], dtype=np.float64)

# =============================================================================
# PRE-COMPUTED LOOKUP TABLES
# =============================================================================
_f_grid = np.logspace(-7, 8, 10000)
_sqrt_term = np.sqrt(1 + _f_grid / aJEL)
_psi_grid = 2 * _sqrt_term + np.log((_sqrt_term - 1) / (_sqrt_term + 1))
_PSI_TABLE_MIN = _psi_grid[0]

# Gauss-Laguerre nodes and weights
_GL_ORDER = 30
_GL_NODES, _GL_WEIGHTS = np.polynomial.laguerre.laggauss(_GL_ORDER)
_GL_NODES = _GL_NODES.astype(np.float64)
_GL_WEIGHTS = _GL_WEIGHTS.astype(np.float64)


# =============================================================================
# JEL CORE FUNCTIONS
# =============================================================================
@njit(fastmath=True, cache=True)
def _psi_of_f(f):
    """
    Compute ψ(f) = 2√(1+f/a) + ln[(√(1+f/a)-1)/(√(1+f/a)+1)]
    
    This is the inverse function needed to find f from (μ-m)/T.
    """
    if f < 1e-5:
        if f <= 0.0:
            return -10000.0
        # Taylor expansion for small f
        return 2.0 + np.log(f / (4.0 * aJEL))
    
    sqrt_val = np.sqrt(1.0 + f / aJEL)
    return 2.0 * sqrt_val + np.log((sqrt_val - 1.0) / (sqrt_val + 1.0))


@njit(fastmath=True, cache=True)
def _dpsi_df(f):
    """Derivative dψ/df for Newton-Raphson iteration."""
    if f < 1e-5:
        return 1.0 / f
    sqrt_val = np.sqrt(1.0 + f / aJEL)
    return (f + aJEL) / (f * aJEL * sqrt_val)


@njit(fastmath=True, cache=True)
def _find_f_jel(mu, T, m, psi_grid, f_grid):
    """
    Find the JEL parameter f such that ψ(f) = (μ-m)/T.
    
    Uses interpolation for initial guess, then Newton-Raphson refinement.
    
    Returns:
        f_val: The JEL parameter
        err: Convergence error (|ψ(f) - ψ_target|)
    """
    psi_target = (mu - m) / T
    
    # Initial guess based on regime
    if psi_target < -14.66737:  # Deep non-degenerate (vacuum)
        if psi_target < -700:
            f_guess = 1e-300
        else:
            f_guess = 4.0 * aJEL * np.exp(psi_target - 2.0)
    else:
        # Interpolate from pre-computed table
        f_guess = np.interp(psi_target, psi_grid, f_grid)
    
    # Newton-Raphson refinement
    f_curr = f_guess
    tol = 1e-10
    max_iter = 50
    err = 1.0
    
    for _ in range(max_iter):
        if f_curr < 1e-300:
            f_curr = 0.0
            break
        
        val = _psi_of_f(f_curr)
        diff = val - psi_target
        err = np.abs(diff)
        
        if err < tol:
            break
        
        deriv = _dpsi_df(f_curr)
        if deriv == 0:
            break
        
        f_next = f_curr - diff / deriv
        if f_next <= 0:
            f_next = f_curr * 0.1  # Damped step for stability
        f_curr = f_next
    
    return f_curr, err


@njit(fastmath=True, cache=True)
def _compute_thermo_single(f_val, T, m, g):
    """
    Compute (n, P, ε) for a single species at given f, T, m.
    
    Uses the JEL rational approximation (Eqs. 21-24 of JEL 1996).
    Only computes particles (not antiparticles).
    """
    if f_val <= 1e-300:
        return 0.0, 0.0, 0.0
        
    # Dimensionless variables
    g_val = np.sqrt(1.0 + f_val) * T / m
    f1 = 1.0 + f_val
    g1 = 1.0 + g_val
    
    # Pre-compute powers
    f_pow = np.ones(MJEL + 1)
    g_pow = np.ones(NJEL + 1)
    for i in range(1, MJEL + 1):
        f_pow[i] = f_pow[i-1] * f_val
    for j in range(1, NJEL + 1):
        g_pow[j] = g_pow[j-1] * g_val
    
    # Accumulate sums
    sum_n, sum_P, sum_e = 0.0, 0.0, 0.0
    term_fg_ratio = f_val / f1
    term_fg_prod_ratio = (f_val * g_val) / (f1 * g1)
    term_g_ratio = g_val / g1
    
    for i in range(MJEL + 1):
        for j in range(NJEL + 1):
            p_val = pmn[i, j]
            base_term = p_val * f_pow[i] * g_pow[j]
            
            sum_P += base_term
            
            coeff_n = (1.0 + i + (0.25 + 0.5*j - MJEL) * term_fg_ratio 
                      + (0.75 - 0.5*NJEL) * term_fg_prod_ratio)
            sum_n += base_term * coeff_n
            
            coeff_e = 1.5 + j + (1.5 - NJEL) * term_g_ratio
            sum_e += base_term * coeff_e
    
    # Common prefactors
    const_pre = g / (2.0 * PI2 * hc3)
    denom_common = f1**(MJEL + 1) * g1**NJEL
    denom_n = f1**(MJEL + 0.5) * g1**NJEL * np.sqrt(1.0 + f_val / aJEL)
    
    # Physical quantities
    n_res = const_pre * m**3 * f_val * g_val**1.5 * g1**1.5 / denom_n * sum_n
    P_res = const_pre * m**4 * f_val * g_val**2.5 * g1**1.5 / denom_common * sum_P
    e_kinetic = const_pre * m**4 * f_val * g_val**2.5 * g1**1.5 / denom_common * sum_e
    e_res = n_res * m + e_kinetic  # Total energy = rest mass + kinetic
    
    return n_res, P_res, e_res


# =============================================================================
# ANALYTIC LIMITING CASES
# =============================================================================
@njit(fastmath=True, cache=True)
def _compute_exact_T0(mu, m, g, include_antiparticles):
    """
    Exact analytical results for T=0 (degenerate) Fermi gas.
    
    Valid when T << |μ - m|.
    """
    mu_abs = np.abs(mu)
    
    # No particles if μ < m
    if mu_abs <= m:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # For antiparticles only with μ < 0, no contribution
    if (not include_antiparticles) and (mu < 0):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Fermi momentum
    kF = np.sqrt(mu_abs**2 - m**2)
    pi2 = PI2
    
    # Logarithmic term
    log_term = np.log((kF + mu_abs) / m) if m > 1e-12 else 0.0
    
    # Number density (net)
    n_val = np.sign(mu) * g * kF**3 / (6.0 * pi2 * hc3)
    
    # Scalar density
    ns_val = (m * g / (4.0 * pi2 * hc3)) * (kF * mu_abs - m**2 * log_term)
    
    # Pressure
    term_P_poly = (2.0 * kF**3 - 3.0 * m**2 * kF) * mu_abs
    P_val = (g / (48.0 * pi2 * hc3)) * (term_P_poly + 3.0 * m**4 * log_term)
    
    # Energy density
    term_e_poly = (2.0 * kF**3 + m**2 * kF) * mu_abs
    e_val = (g / (16.0 * pi2 * hc3)) * (term_e_poly - m**4 * log_term)
    
    return n_val, P_val, e_val, 0.0, ns_val


@njit(fastmath=True, cache=True)
def _compute_ur_limit(mu, T, g):
    """
    Exact analytical solution for massless (ultra-relativistic) Fermi gas.
    
    Assumes thermal equilibrium with particles + antiparticles.
    Valid when T >> m or when m → 0.
    """
    pi2 = PI2
    
    # Common prefactor
    pre = g / (6.0 * hc3)
    
    # Pressure: P = (g/6ℏ³c³)[7π²T⁴/60 + μ²T²/2 + μ⁴/(4π²)]
    term_P_T4 = (7.0 * pi2 / 60.0) * T**4
    term_P_T2 = 0.5 * T**2 * mu**2
    term_P_mu4 = mu**4 / (4.0 * pi2)
    P_val = pre * (term_P_T4 + term_P_T2 + term_P_mu4)
    
    # Energy density (ε = 3P for ultra-relativistic)
    e_val = 3.0 * P_val
    
    # Entropy density
    term_s_T3 = (7.0 * pi2 / 15.0) * T**3
    term_s_mu2 = T * mu**2
    s_val = pre * (term_s_T3 + term_s_mu2)
    
    # Number density (net)
    term_n_T2 = mu * T**2
    term_n_mu3 = mu**3 / pi2
    n_val = pre * (term_n_T2 + term_n_mu3)
    
    # Scalar density is 0 for massless
    ns_val = 0.0
    
    return n_val, P_val, e_val, s_val, ns_val


# =============================================================================
# MAIN JEL SOLVER
# =============================================================================
@njit(fastmath=True, cache=True)
def calculate_jel_fast(mu, T, m, g_deg, psi_grid, f_grid, 
                       include_antiparticles, return_error):
    """
    Main JEL calculator for Fermi integrals.
    
    Parameters:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Mass (MeV)
        g_deg: Degeneracy factor (spin × isospin × color)
        psi_grid, f_grid: Pre-computed lookup tables
        include_antiparticles: If True, include antiparticle contribution
        return_error: If True, append convergence error to output
    
    Returns:
        Array [n, P, e, s, ns] or [n, P, e, s, ns, err]
        Units: n (fm⁻³), P (MeV/fm³), e (MeV/fm³), s (fm⁻³), ns (fm⁻³)
    """
    # Massless limit
    if m < 1.0e-5:
        res_ur = _compute_ur_limit(mu, T, g_deg)
        if return_error:
            return np.array([res_ur[0], res_ur[1], res_ur[2], res_ur[3], res_ur[4], 0.0])
        return np.array(res_ur)
    
    # Zero temperature limit
    if T < 1.0e-4:
        res = _compute_exact_T0(mu, m, g_deg, include_antiparticles)
        if return_error:
            return np.array([res[0], res[1], res[2], res[3], res[4], 0.0])
        return np.array(res)
    
    # Finite T, finite m: Full JEL calculation
    f_part, err_part = _find_f_jel(mu, T, m, psi_grid, f_grid)
    n_p, P_p, e_p = _compute_thermo_single(f_part, T, m, g_deg)
    
    n_tot, P_tot, e_tot, max_err = n_p, P_p, e_p, err_part
    
    if include_antiparticles:
        f_anti, err_anti = _find_f_jel(-mu, T, m, psi_grid, f_grid)
        n_a, P_a, e_a = _compute_thermo_single(f_anti, T, m, g_deg)
        
        n_tot = n_p - n_a   # Net number density
        P_tot = P_p + P_a   # Total pressure
        e_tot = e_p + e_a   # Total energy
        max_err = max(err_part, err_anti)
    
    # Derived quantities
    s_tot = (P_tot + e_tot - mu * n_tot) / T  # Entropy density
    ns_tot = (e_tot - 3.0 * P_tot) / m if m > 1e-9 else 0.0  # Scalar density
    
    if return_error:
        return np.array([n_tot, P_tot, e_tot, s_tot, ns_tot, max_err])
    return np.array([n_tot, P_tot, e_tot, s_tot, ns_tot])


# =============================================================================
# GAUSS-LAGUERRE ALTERNATIVE
# =============================================================================
@njit(cache=True, fastmath=True)
def _fermi_gauss_laguerre_kernel(mu, T, m, g, nodes, weights, include_antiparticles):
    """
    Gauss-Laguerre quadrature for relativistic Fermi integrals.
    
    Uses change of variables x = (E - m)/T for exponential weighting.
    """
    inv_T = 1.0 / T
    
    # Prefactors
    const_n = g / (2.0 * PI2 * hc3)
    const_P = g / (6.0 * PI2 * hc3)
    const_e = g / (2.0 * PI2 * hc3)
    
    # Exponential factors
    A_p = (m - mu) * inv_T
    expA_p = np.exp(A_p)
    A_a = (m + mu) * inv_T
    expA_a = np.exp(A_a)
    
    n_sum, P_sum, e_sum = 0.0, 0.0, 0.0
    
    for i in range(nodes.shape[0]):
        x = nodes[i]
        w = weights[i]
        
        E = m + T * x
        k2 = E * E - m * m
        k = np.sqrt(k2) if k2 > 0.0 else 0.0
        
        exp_minus_x = np.exp(-x)
        
        # Distribution functions (effective, without e^{-x})
        denom_p = expA_p + exp_minus_x
        f_p_eff = 1.0 / denom_p
        
        if include_antiparticles:
            denom_a = expA_a + exp_minus_x
            f_a_eff = 1.0 / denom_a
        else:
            f_a_eff = 0.0
        
        diff = f_p_eff - f_a_eff
        sumfa = f_p_eff + f_a_eff
        
        # Integrand contributions
        Tk = T * k
        n_sum += w * Tk * E * diff
        P_sum += w * Tk * k * k * sumfa
        e_sum += w * Tk * E * E * sumfa
    
    # Physical quantities
    n_res = const_n * n_sum
    P_res = const_P * P_sum
    e_res = const_e * e_sum
    
    s_res = (P_res + e_res - mu * n_res) / T
    ns_res = (e_res - 3.0 * P_res) / m if m > 0.0 else 0.0
    
    return n_res, P_res, e_res, s_res, ns_res


# =============================================================================
# PUBLIC API
# =============================================================================
def solve_fermi_jel(mu, T, m, g, include_antiparticles=True, return_error=False):
    """
    Solve Fermi integrals using JEL approximation.
    
    Parameters:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Particle mass (MeV)
        g: Degeneracy factor
        include_antiparticles: Include antiparticle contribution (default True)
        return_error: Return convergence error (default False)
    
    Returns:
        tuple or array: (n, P, e, s, ns) [, err]
            n: Number density (fm⁻³)
            P: Pressure (MeV/fm³)
            e: Energy density (MeV/fm³)
            s: Entropy density (fm⁻³)
            ns: Scalar density (fm⁻³)
    """
    return calculate_jel_fast(
        float(mu), float(T), float(m), float(g),
        _psi_grid, _f_grid,
        include_antiparticles, return_error
    )


def solve_fermi_gl(mu, T, m, g, include_antiparticles=True):
    """
    Solve Fermi integrals using Gauss-Laguerre quadrature.
    
    Higher accuracy than JEL, but slower.
    Falls back to analytic limits for T→0 and m→0.
    """
    mu, T, m, g = float(mu), float(T), float(m), float(g)
    
    # Analytic limits
    if m < 1.0e-5:
        return _compute_ur_limit(mu, T, g)
    
    if T < 1.0e-4:
        return _compute_exact_T0(mu, m, g, include_antiparticles)
    
    return _fermi_gauss_laguerre_kernel(
        mu, T, m, g, _GL_NODES, _GL_WEIGHTS, include_antiparticles
    )


def invert_fermi_density(n_target: float, T: float, m: float, g: float,
                          include_antiparticles: bool = True,
                          tol: float = 1e-10) -> float:
    """
    Find the chemical potential μ that gives a target density n.
    
    This is the inverse of solve_fermi_jel: given n, find μ such that
    solve_fermi_jel(μ, T, m, g)[0] = n.
    
    Uses Brent's method for robust root-finding.
    
    Args:
        n_target: Target number density (fm⁻³)
        T: Temperature (MeV)
        m: Particle mass (MeV)
        g: Degeneracy factor
        include_antiparticles: Include antiparticle contribution
        tol: Convergence tolerance
        
    Returns:
        mu: Chemical potential (MeV) that gives the target density
    """
    from scipy.optimize import brentq

    
    def density_residual(mu):
        result = solve_fermi_jel(mu, T, m, g, include_antiparticles=include_antiparticles)
        return result[0] - n_target
    
    # Estimate bounds from T=0 Fermi momentum
    n_abs = abs(n_target)
    kF = hc * (6.0 * PI2 * n_abs / g)**(1.0/3.0)
    mu_estimate = np.sqrt(kF**2 + m**2)
    
    # Set search bounds
    if n_target >= 0:
        mu_lo = m * 0.5
        mu_hi = max(mu_estimate * 2.0, m + 500.0)
    else:
        # Negative density means we need negative μ
        mu_lo = -max(mu_estimate * 2.0, m + 500.0)
        mu_hi = -m * 0.5
    
    # Adjust bounds if needed
    n_hi = solve_fermi_jel(mu_hi, T, m, g, include_antiparticles=include_antiparticles)[0]
    while n_hi < n_target:
        mu_hi *= 1.5
        n_hi = solve_fermi_jel(mu_hi, T, m, g, include_antiparticles=include_antiparticles)[0]
    
    n_lo = solve_fermi_jel(mu_lo, T, m, g, include_antiparticles=include_antiparticles)[0]
    while n_lo > n_target:
        mu_lo *= 0.5 if mu_lo > 0 else 1.5
        n_lo = solve_fermi_jel(mu_lo, T, m, g, include_antiparticles=include_antiparticles)[0]
    
    try:
        mu = brentq(density_residual, mu_lo, mu_hi, xtol=tol)
    except ValueError:
        # Fallback: Newton-Raphson
        mu = mu_estimate if n_target >= 0 else -mu_estimate
        for _ in range(50):
            result = solve_fermi_jel(mu, T, m, g, include_antiparticles=include_antiparticles)
            n_calc = result[0]
            # Estimate dn/dμ from thermodynamic relation
            dn_dmu = max(result[4] if len(result) > 4 else abs(n_calc) / (abs(mu) + 1.0), 1e-15)
            delta = (n_calc - n_target) / dn_dmu
            mu -= 0.5 * delta
            if abs(delta) < tol:
                break
    
    return mu


def Fermi_Numerical(mu, T, m, g, include_antiparticles=True):
    """
    Direct numerical integration (scipy.quad) for validation.
    
    Slow but accurate reference implementation.
    """
    prefactor = g / (2.0 * PI2 * hc3)
    
    def distrib(E, chem_pot):
        arg = (E - chem_pot) / T
        if arg > 100:
            return 0.0
        if arg < -100:
            return 1.0
        return 1.0 / (np.exp(arg) + 1.0)
    
    def integrands(k):
        E = np.sqrt(k**2 + m**2)
        f_p = distrib(E, mu)
        f_a = distrib(E, -mu) if include_antiparticles else 0.0
        
        dn = (f_p - f_a) * k**2
        dP = (f_p + f_a) * k**4 / (3.0 * E)
        de = (f_p + f_a) * k**2 * E
        return dn, dP, de
    
    upper_limit = max(1000.0, abs(mu) + 20*T)
    
    n_res = prefactor * integrate.quad(lambda k: integrands(k)[0], 0, upper_limit)[0]
    P_res = prefactor * integrate.quad(lambda k: integrands(k)[1], 0, upper_limit)[0]
    e_res = prefactor * integrate.quad(lambda k: integrands(k)[2], 0, upper_limit)[0]
    
    s_res = (P_res + e_res - mu * n_res) / T
    ns_res = (e_res - 3*P_res) / m if m > 0 else 0
    
    return n_res, P_res, e_res, s_res, ns_res