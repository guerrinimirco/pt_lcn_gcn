"""
vmit_compute_tables.py
=======================
User-friendly script for generating vMIT quark EOS tables.

This module provides:
- VMITTableSettings: Configuration for table generation
- compute_vmit_table(): Main function to generate tables
- save_vmit_results(): Save results to file
- results_to_arrays(): Convert results for plotting

All solvers are in vmit_eos.py.
All thermodynamic functions are in vmit_thermodynamics_quarks.py.

Usage:
    from vmit_compute_tables import compute_vmit_table, VMITTableSettings
    
    settings = VMITTableSettings(
        equilibrium='beta_eq',
        T_values=[0.1, 10.0, 50.0],
        n_B_values=np.linspace(0.1, 10, 100) * 0.16
    )
    results = compute_vmit_table(settings)
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# Import from vmit_eos (solvers)
from vmit_eos import (
    VMITEOSResult,
    solve_vmit_beta_eq, solve_vmit_fixed_yc, solve_vmit_fixed_yc_ys, solve_vmit_trapped_neutrinos,
    result_to_guess,
    get_default_guess_beta_eq, get_default_guess_fixed_yc, get_default_guess_trapped_neutrinos
)
from vmit_parameters import VMITParams, get_vmit_default


# =============================================================================
# SETTINGS DATACLASS
# =============================================================================
@dataclass
class VMITTableSettings:
    """
    Configuration for vMIT EOS table generation.
    
    Equilibrium types:
    - 'beta_eq': Beta equilibrium with charge neutrality
    - 'fixed_yc': Fixed charge fraction Y_C
    - 'fixed_yc_ys': Fixed charge and strangeness fractions Y_C, Y_S
    - 'trapped_neutrinos': Beta eq with fixed lepton fraction Y_L
    """
    # Model parameters
    params: Optional[VMITParams] = None  # None = use default
    
    # Equilibrium type
    equilibrium: str = 'beta_eq'
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 10, 100) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    
    # Constraint parameters (depending on equilibrium mode)
    Y_C_values: List[float] = field(default_factory=lambda: [0.0])  # For fixed_yc
    Y_S_values: List[float] = field(default_factory=lambda: [0.0])  # For fixed_yc_ys
    Y_L_values: List[float] = field(default_factory=lambda: [0.4])  # For trapped_neutrinos
    
    # Options
    include_photons: bool = True
    include_leptons: bool = True  # Only for fixed_yc/fixed_yc_ys: include electrons
    
    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True
    
    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None  # Auto-generate if None


# =============================================================================
# TABLE GENERATOR
# =============================================================================
def compute_vmit_table(settings: VMITTableSettings) -> Dict[Tuple, List[VMITEOSResult]]:
    """
    Compute vMIT EOS table with optimized initial guess propagation.
    
    Uses previous solution as initial guess for next n_B point (like Mathematica).
    
    Args:
        settings: VMITTableSettings configuration
        
    Returns:
        Dictionary mapping (T, [Y_C/Y_L/...]) tuple to list of VMITEOSResult
    """
    params = settings.params if settings.params is not None else get_vmit_default()
    eq_type = settings.equilibrium.lower()
    
    n_B_arr = np.asarray(settings.n_B_values)
    T_list = list(settings.T_values)
    
    # Build parameter grid based on equilibrium type
    if eq_type == 'beta_eq':
        grid_params = [(T,) for T in T_list]
        param_names = ['T']
    elif eq_type == 'fixed_yc':
        Y_C_list = list(settings.Y_C_values)
        grid_params = [(T, Y_C) for T in T_list for Y_C in Y_C_list]
        param_names = ['T', 'Y_C']
    elif eq_type == 'fixed_yc_ys':
        Y_C_list = list(settings.Y_C_values)
        Y_S_list = list(settings.Y_S_values)
        grid_params = [(T, Y_C, Y_S) for T in T_list for Y_C in Y_C_list for Y_S in Y_S_list]
        param_names = ['T', 'Y_C', 'Y_S']
    elif eq_type == 'trapped_neutrinos':
        Y_L_list = list(settings.Y_L_values)
        grid_params = [(T, Y_L) for T in T_list for Y_L in Y_L_list]
        param_names = ['T', 'Y_L']
    else:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")
    
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print("=" * 70)
        print("vMIT EOS TABLE GENERATION")
        print("=" * 70)
        print(f"\nModel: {params.name}")
        print(f"Parameters: B^1/4={params.B4} MeV, a={params.a} fm²")
        print(f"Equilibrium: {eq_type}")
        print(f"Density grid: {n_points} points, n_B = [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm⁻³")
        print(f"Parameter grid: {n_tables} tables")
        print()
    
    all_results = {}
    total_start = time.time()
    
    for idx, grid_param in enumerate(grid_params):
        T = grid_param[0]
        Y_C = grid_param[1] if len(grid_param) > 1 and eq_type in ('fixed_yc', 'fixed_yc_ys') else None
        Y_S = grid_param[2] if len(grid_param) > 2 and eq_type == 'fixed_yc_ys' else None
        Y_L = grid_param[1] if len(grid_param) > 1 and eq_type == 'trapped_neutrinos' else None
        
        if settings.print_results:
            print("-" * 70)
            param_str = f"T={T}"
            if Y_C is not None:
                param_str += f", Y_C={Y_C}"
            if Y_S is not None:
                param_str += f", Y_S={Y_S}"
            if Y_L is not None:
                param_str += f", Y_L={Y_L}"
            print(f"[{idx+1}/{n_tables}] {param_str}")
        
        start_time = time.time()
        results = []
        guess = None

        # Try to use result from previous table as initial guess for the first point
        # This helps when moving from Y_S=0.0 -> Y_S=0.1 -> Y_S=0.5 etc.
        if idx > 0:
            prev_param = grid_params[idx-1]
            prev_T = prev_param[0]
            prev_results = all_results.get(prev_param)
            
            # Check if relevant parameters (T, Y_C) match
            should_use_prev = False
            if eq_type == 'fixed_yc_ys':
                prev_YC = prev_param[1]
                # If T and Y_C match, we are just changing Y_S
                if prev_T == T and prev_YC == Y_C:
                    should_use_prev = True
            elif eq_type == 'fixed_yc':
                # If T matches, we are just changing Y_C
                if prev_T == T:
                    should_use_prev = True
            elif eq_type == 'beta_eq':
                 # If T changes, maybe use previous T? (Optional, but less critical)
                 pass

            if should_use_prev and prev_results and len(prev_results) > 0 and prev_results[0].converged:
                guess = result_to_guess(prev_results[0], eq_type)
        
        for i, n_B in enumerate(n_B_arr):
            # Call appropriate solver
            if eq_type == 'beta_eq':
                r = solve_vmit_beta_eq(n_B, T, params, 
                                       include_photons=settings.include_photons, 
                                       initial_guess=guess)
            elif eq_type == 'fixed_yc':
                r = solve_vmit_fixed_yc(n_B, Y_C, T, params, 
                                        include_photons=settings.include_photons,
                                        include_electrons=settings.include_leptons,
                                        initial_guess=guess)
            elif eq_type == 'fixed_yc_ys':
                r = solve_vmit_fixed_yc_ys(n_B, Y_C, Y_S, T, params, 
                                           include_photons=settings.include_photons,
                                           include_electrons=settings.include_leptons,
                                           initial_guess=guess)
            elif eq_type == 'trapped_neutrinos':
                r = solve_vmit_trapped_neutrinos(n_B, Y_L, T, params, 
                                                  include_photons=settings.include_photons, 
                                                  initial_guess=guess)
            
            results.append(r)
            
            # Update guess for next point
            if r.converged:
                if eq_type in ('fixed_yc', 'fixed_yc_ys'):
                    guess = result_to_guess(r, eq_type, include_electrons=settings.include_leptons)
                else:
                    guess = result_to_guess(r, eq_type)
            
            # Print progress
            if settings.print_results:
                should_print = (i < settings.print_first_n or 
                               (settings.print_errors and not r.converged))
                if should_print:
                    status = "OK" if r.converged else "FAILED"
                    print(f"[{i:4d}] n_B={n_B:.4e} [{status}] P={r.P_total:.4f} Y_s={r.Y_s:.4f} err={r.error:.2e}")
        
        elapsed = time.time() - start_time
        all_results[grid_param] = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  {elapsed:.2f}s ({elapsed*1000/n_points:.1f}ms/pt), Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    total_elapsed = time.time() - total_start
    
    if settings.print_timing:
        print("\n" + "=" * 70)
        print(f"Total: {total_elapsed:.2f}s, Avg: {total_elapsed*1000/(n_points * n_tables):.1f}ms/point")
    
    if settings.save_to_file:
        save_vmit_results(all_results, settings, params, eq_type)
    
    return all_results


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_vmit_results(
    all_results: Dict[Tuple, List[VMITEOSResult]], 
    settings: VMITTableSettings,
    params: VMITParams,
    eq_type: str
):
    """Save results to file."""
    if settings.output_filename:
        filename = settings.output_filename
    else:
        filename = f"/Users/mircoguerrini/Desktop/Research/Python_codes/output/vmit_B{int(params.B4)}_a{params.a}_{eq_type}.dat"
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"# vMIT EOS Table: {params.name}\n")
        f.write(f"# Parameters: B^1/4={params.B4} MeV, a={params.a} fm²\n")
        f.write(f"# Equilibrium: {eq_type}\n")
        
        # Define columns based on equilibrium type
        # Format: inputs first, then mu, then Y, then thermodynamics
        if eq_type == 'fixed_yc':
            columns = ['n_B', 'Y_C', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        elif eq_type == 'fixed_yc_ys':
            columns = ['n_B', 'Y_C', 'Y_S', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        elif eq_type == 'trapped_neutrinos':
            columns = ['n_B', 'Y_L', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'mu_nu', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 'Y_nu',
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        else:  # beta_eq
            columns = ['n_B', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 
                       'P_total', 'e_total', 's_total', 'f_total', 'converged']
        
        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")
        
        for params_tuple, results in all_results.items():
            # Get Y_C, Y_S, Y_L from grid params depending on mode
            Y_C_val = params_tuple[1] if eq_type in ('fixed_yc', 'fixed_yc_ys') and len(params_tuple) > 1 else 0.0
            Y_S_val = params_tuple[2] if eq_type == 'fixed_yc_ys' and len(params_tuple) > 2 else 0.0
            Y_L_val = params_tuple[1] if eq_type == 'trapped_neutrinos' and len(params_tuple) > 1 else 0.0
            
            for r in results:
                if r.converged:
                    f_total = r.e_total - r.s_total * r.T
                    
                    # Build row based on equilibrium type
                    if eq_type == 'fixed_yc':
                        row = [r.n_B, Y_C_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e, 
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    elif eq_type == 'fixed_yc_ys':
                        row = [r.n_B, Y_C_val, Y_S_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e, 
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    elif eq_type == 'trapped_neutrinos':
                        Y_nu = r.n_nu / r.n_B if hasattr(r, 'n_nu') and r.n_B > 0 else 0.0
                        row = [r.n_B, Y_L_val, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e, r.mu_nu,
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, Y_nu, r.P_total, r.e_total, r.s_total, f_total, 1]
                    else:  # beta_eq
                        row = [r.n_B, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e, 
                               r.Y_u, r.Y_d, r.Y_s, r.Y_e, r.P_total, r.e_total, r.s_total, f_total, 1]
                    f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float) else f"{v:>14}" for v in row) + "\n")
    
    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[VMITEOSResult]) -> Dict[str, np.ndarray]:
    """Convert list of VMITEOSResult to dictionary of numpy arrays (converged only)."""
    # Input parameters
    input_attrs = ['n_B', 'T', 'Y_C', 'Y_S', 'Y_L']
    
    # Chemical potentials
    mu_attrs = ['mu_u', 'mu_d', 'mu_s', 'mu_e', 'mu_nu']
    
    # Fractions
    Y_attrs = ['Y_u', 'Y_d', 'Y_s', 'Y_e', 'Y_nu']
    
    # Thermodynamic quantities
    thermo_attrs = ['P_total', 'e_total', 's_total', 'f_total']
    
    # Other
    other_attrs = ['error']
    
    all_attrs = input_attrs + mu_attrs + Y_attrs + thermo_attrs + other_attrs
    
    arrays = {}
    for attr in all_attrs:
        try:
            arrays[attr] = np.array([getattr(r, attr) for r in results if r.converged])
        except AttributeError:
            # Skip attributes that don't exist (e.g., mu_nu for non-trapped mode)
            pass
    
    # Compute f_total = e_total - s_total * T if not present
    if 'f_total' not in arrays and 'e_total' in arrays and 's_total' in arrays:
        T_arr = np.array([r.T for r in results if r.converged])
        arrays['f_total'] = arrays['e_total'] - arrays['s_total'] * T_arr
    
    arrays['converged'] = np.array([r.converged for r in results])
    return arrays


# =============================================================================
# CONFIGURATION (EDIT THIS SECTION)
# =============================================================================
settings = VMITTableSettings(
    params=None,  # Use default
    equilibrium='fixed_yc_ys',  # 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos' 'beta_eq
    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=np.concatenate([[0.1],np.linspace(2.5, 120, 48)]),
    Y_C_values=[0.0, 0.1, 0.3, 0.5],
    Y_S_values=[0.0, 0.1,0.2,0.4,0.6,0.8,1],
    #Y_L_values=[0.4, 0.3],
    include_photons=True,
    include_leptons=True,  # Only for fixed_yc/fixed_yc_ys: include electrons
    print_results=True,
    print_first_n=1,
    print_errors=False,
    print_timing=True,
    save_to_file=True,
    output_filename=None, #
)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("vMIT EOS TABLE GENERATOR")
    print("=" * 70 + "\n")
    
    all_results = compute_vmit_table(settings)
    
    print("\nDONE!")
