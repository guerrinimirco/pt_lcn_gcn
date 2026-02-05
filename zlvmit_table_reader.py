"""
Simplified EOS Reader for ZLvMIT Hybrid Tables

Simple Python code to:
- Read EOS files (nB, YC, T, eta are inputs)
- Provide simple functions like P_tot(nB, YC, T, eta) for plotting
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from pathlib import Path
from typing import Optional, Union


class EOSReader:
    """Simple class to read and interpolate EOS data"""

    # Column names in the expected order (after input columns)
    COLUMN_NAMES = [
        'T', 'eta', 'converged', 'error', 'chi',
        'mu_p_H', 'mu_n_H', 'n_p_H', 'n_n_H',
        'mu_u_Q', 'mu_d_Q', 'mu_s_Q', 'n_u_Q', 'n_d_Q', 'n_s_Q',
        'mu_eL_H', 'mu_eL_Q', 'mu_eG', 'n_eL_H', 'n_eL_Q', 'n_eG',
        'P_total', 'e_total', 's_total', 'f_total', 'n_e_tot',
        'Y_p_H', 'Y_n_H', 'Y_u_Q', 'Y_d_Q', 'Y_s_Q',
        'Y_C_H', 'Y_C_Q', 'Y_S_Q',
        'Y_p_tot', 'Y_n_tot', 'Y_u_tot', 'Y_d_tot', 'Y_s_tot',
        'Y_e_tot', 'Y_B_tot', 'Y_C_tot', 'Y_S_tot'
    ]

    def __init__(self, filename: str, eq_mode: str = 'betaeq'):
        """
        Initialize EOS reader

        Parameters:
        -----------
        filename : str
            Path to EOS data file
        eq_mode : str
            Either 'betaeq' or 'fixedyc'
        """
        if eq_mode not in ['betaeq', 'fixedyc']:
            raise ValueError(f"eq_mode must be 'betaeq' or 'fixedyc', got '{eq_mode}'")

        self.filename = filename
        self.eq_mode = eq_mode
        self.data = {}
        self.interpolators = {}

        self._load_data()
        self._create_interpolators()

    def _load_data(self):
        """Load data from file and organize into dictionary"""
        # Read all data
        raw_data = np.loadtxt(self.filename)

        # Both modes have same input column structure: n_B, T, eta, [outputs...]
        self.data['n_B'] = raw_data[:, 0]  # Keep nB at full precision
        col_offset = 1

        # Load remaining columns dynamically
        for i, col_name in enumerate(self.COLUMN_NAMES):
            col_idx = col_offset + i
            if col_idx < raw_data.shape[1]:
                self.data[col_name] = raw_data[:, col_idx]

        # Round T to 3 decimals to reduce unique values (but keep nB precise)
        if 'T' in self.data:
            self.data['T'] = np.round(self.data['T'], 3)

        # For fixedyc mode, extract Y_C from the Y_C_tot output column
        if self.eq_mode == 'betaeq':
            self._input_dims = ['n_B', 'T', 'eta']
        else:  # fixedyc
            # Y_C_tot is in the output columns (should be column -2 or named 'Y_C_tot')
            # Round to 3 decimal places to reduce unique values while keeping precision
            if 'Y_C_tot' in self.data:
                self.data['Y_C'] = np.round(self.data['Y_C_tot'], 3)
            else:
                # Y_C_tot is second to last column in raw data
                raw_yc = np.loadtxt(self.filename, usecols=-2)
                self.data['Y_C'] = np.round(raw_yc, 3)
            self._input_dims = ['n_B', 'Y_C', 'T', 'eta']

    def _create_interpolators(self):
        """Create interpolators for all quantities using fast RegularGridInterpolator"""
        from scipy.interpolate import RegularGridInterpolator

        # For fixedyc mode, Y_C varies in the mixed phase creating a sparse grid
        # Solution: create separate 2D interpolators (n_B, T) for each unique Y_C
        if self.eq_mode == 'fixedyc':
            self._create_fixedyc_interpolators()
            return

        # Beta equilibrium mode: simple 2D grid (n_B, T)
        dim_names = ['n_B', 'T']

        # Get unique values for each dimension (grid axes)
        grid_axes = []
        self._interp_dims = []

        for dim_name in dim_names:
            unique_vals = np.unique(self.data[dim_name])
            self._interp_dims.append(dim_name)
            grid_axes.append(unique_vals)

        # Reshape data onto regular grid
        grid_shape = tuple(len(axis) for axis in grid_axes)

        # Create index mapping for grid positions
        dim_indices = {}
        for dim_name, axis in zip(self._interp_dims, grid_axes):
            dim_indices[dim_name] = {val: idx for idx, val in enumerate(axis)}

        # Create interpolators for each quantity
        for name, values in self.data.items():
            if name not in self._input_dims:
                grid_data = np.full(grid_shape, np.nan)
                for row_idx in range(len(values)):
                    grid_idx = tuple(
                        dim_indices[dim_name][self.data[dim_name][row_idx]]
                        for dim_name in self._interp_dims
                    )
                    grid_data[grid_idx] = values[row_idx]

                self.interpolators[name] = RegularGridInterpolator(
                    grid_axes,
                    grid_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

        self._grid_axes = grid_axes

    def _create_fixedyc_interpolators(self):
        """Create interpolators for fixedyc mode: 3D RegularGridInterpolator (n_B, Y_C, T)"""
        from scipy.interpolate import RegularGridInterpolator

        self._interp_dims = ['n_B', 'Y_C', 'T']

        # Get unique grid values (sorted)
        self._nB_unique = np.sort(np.unique(self.data['n_B']))
        self._T_unique = np.sort(np.unique(self.data['T']))
        self._YC_unique = np.sort(np.unique(self.data['Y_C']))

        n_nB = len(self._nB_unique)
        n_YC = len(self._YC_unique)
        n_T = len(self._T_unique)

        # Create index maps for fast lookup
        nB_to_idx = {v: i for i, v in enumerate(self._nB_unique)}
        YC_to_idx = {v: i for i, v in enumerate(self._YC_unique)}
        T_to_idx = {v: i for i, v in enumerate(self._T_unique)}

        # Build 3D grids for each quantity
        for name, values in self.data.items():
            if name not in self._input_dims:
                # Initialize 3D grid with NaN
                grid = np.full((n_nB, n_YC, n_T), np.nan)

                # Fill grid from flat data
                for i, (nB, YC, T, val) in enumerate(zip(
                    self.data['n_B'], self.data['Y_C'], self.data['T'], values
                )):
                    i_nB = nB_to_idx.get(nB)
                    i_YC = YC_to_idx.get(YC)
                    i_T = T_to_idx.get(T)
                    if i_nB is not None and i_YC is not None and i_T is not None:
                        grid[i_nB, i_YC, i_T] = val

                # Only create interpolator if we have enough valid data
                if np.sum(~np.isnan(grid)) > 10:
                    self.interpolators[name] = RegularGridInterpolator(
                        (self._nB_unique, self._YC_unique, self._T_unique),
                        grid,
                        method='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )

        # Store Y_C range for bounds checking
        self._YC_min = self._YC_unique.min()
        self._YC_max = self._YC_unique.max()

    def get(self, quantity: str, n_B: float, T: float = 0.0,
            eta: float = 0.0, Y_C: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Get any quantity at given point(s)

        Parameters:
        -----------
        quantity : str
            Name of quantity (e.g., 'P_total', 'e_total', 'chi')
        n_B : float or array
            Baryon density (fm^-3)
        T : float or array
            Temperature (MeV), default 0.0
        eta : float or array
            Mixing parameter, default 0.0
        Y_C : float or array, optional
            Charge fraction (required for fixedyc mode)

        Returns:
        --------
        Interpolated value(s)
        """
        # For fixedyc mode, use 3D linear interpolation (n_B, Y_C, T)
        if self.eq_mode == 'fixedyc':
            if Y_C is None:
                raise ValueError("Y_C must be provided for fixedyc mode")

            if quantity not in self.interpolators:
                available = sorted(self.interpolators.keys())
                raise ValueError(
                    f"Unknown quantity '{quantity}'. "
                    f"Available: {', '.join(available[:10])}..."
                )

            interpolator = self.interpolators[quantity]

            # Build query points (n_B, Y_C, T)
            is_scalar = np.isscalar(n_B) and np.isscalar(T) and np.isscalar(Y_C)

            if is_scalar:
                query_point = np.array([[n_B, Y_C, T]])
            else:
                n_B_arr, Y_C_arr, T_arr = np.broadcast_arrays(n_B, Y_C, T)
                query_point = np.column_stack([
                    n_B_arr.ravel(),
                    Y_C_arr.ravel(),
                    T_arr.ravel()
                ])

            result = interpolator(query_point)

            if is_scalar:
                return float(result[0])
            else:
                return result.reshape(n_B_arr.shape)

        # Beta equilibrium mode
        if quantity not in self.interpolators:
            available = sorted(self.interpolators.keys())
            raise ValueError(
                f"Unknown quantity '{quantity}'. "
                f"Available: {', '.join(available[:10])}..."
            )

        # Build interpolation points (n_B, T)
        is_scalar = np.isscalar(n_B) and np.isscalar(T)

        if is_scalar:
            query_point = np.array([[n_B, T]])
        else:
            n_B_arr, T_arr = np.broadcast_arrays(n_B, T)
            query_point = np.column_stack([n_B_arr.ravel(), T_arr.ravel()])

        result = self.interpolators[quantity](query_point)

        if is_scalar:
            return float(result[0])
        else:
            return result.reshape(n_B_arr.shape)

    def available_quantities(self):
        """Return list of all available quantities"""
        return sorted(self.interpolators.keys())

    # =========================================================================
    # Convenience methods for common quantities
    # =========================================================================

    def P_tot(self, n_B, T=0.0, eta=0.0, Y_C=None):
        """Total pressure (MeV/fm^3)"""
        return self.get('P_total', n_B, T, eta, Y_C)

    def e_tot(self, n_B, T=0.0, eta=0.0, Y_C=None):
        """Total energy density (MeV/fm^3)"""
        return self.get('e_total', n_B, T, eta, Y_C)

    def s_tot(self, n_B, T=0.0, eta=0.0, Y_C=None):
        """Total entropy density"""
        return self.get('s_total', n_B, T, eta, Y_C)

    def chi(self, n_B, T=0.0, eta=0.0, Y_C=None):
        """Quark volume fraction"""
        return self.get('chi', n_B, T, eta, Y_C)


class PhaseBoundarySingleEta:
    """Class to read and interpolate phase boundary data for a single eta value"""

    def __init__(self, filename: str, eq_mode: str = 'betaeq'):
        """
        Initialize phase boundaries reader for single eta

        Parameters:
        -----------
        filename : str
            Path to phase boundary file
        eq_mode : str
            'betaeq' or 'fixedyc' - determines column layout
        """
        self.filename = filename
        self.eq_mode = eq_mode
        self.eta_value = None

        # Read eta value from header
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('# eta'):
                    self.eta_value = float(line.split('=')[1].strip())
                    break

        # Load data
        data = np.loadtxt(filename)

        if eq_mode == 'betaeq':
            # Columns: T, n_B_onset, n_B_offset, ...
            self.T = data[:, 0]
            self.n_B_onset = data[:, 1]
            self.n_B_offset = data[:, 2]
            self.Y_C = None

            # Create 1D interpolators for betaeq
            self._onset_interp = interp1d(
                self.T, self.n_B_onset,
                kind='linear',
                fill_value='extrapolate'
            )
            self._offset_interp = interp1d(
                self.T, self.n_B_offset,
                kind='linear',
                fill_value='extrapolate'
            )
        else:
            # Columns: Y_C, T, n_B_onset, n_B_offset, ...
            self.Y_C = data[:, 0]
            self.T = data[:, 1]
            self.n_B_onset = data[:, 2]
            self.n_B_offset = data[:, 3]
            # No pre-built interpolators for fixedyc - need Y_C filtering first

    def nB_onset(self, T, Y_C=None):
        """Get phase transition onset density at temperature T"""
        if self.eq_mode == 'betaeq':
            return self._onset_interp(T)
        else:
            if Y_C is None:
                raise ValueError("Y_C must be provided for fixedyc mode")
            mask = np.abs(self.Y_C - Y_C) < 0.01
            if np.sum(mask) == 0:
                return np.nan
            interp = interp1d(self.T[mask], self.n_B_onset[mask],
                            kind='linear', fill_value='extrapolate')
            return interp(T)

    def nB_offset(self, T, Y_C=None):
        """Get phase transition offset density at temperature T"""
        if self.eq_mode == 'betaeq':
            return self._offset_interp(T)
        else:
            if Y_C is None:
                raise ValueError("Y_C must be provided for fixedyc mode")
            mask = np.abs(self.Y_C - Y_C) < 0.01
            if np.sum(mask) == 0:
                return np.nan
            interp = interp1d(self.T[mask], self.n_B_offset[mask],
                            kind='linear', fill_value='extrapolate')
            return interp(T)

    def get_boundary_arrays(self, Y_C=None):
        """
        Get raw T, nB_onset, nB_offset arrays.
        For fixedyc mode, filters by Y_C value.

        Returns:
        --------
        T, nB_onset, nB_offset : arrays
        """
        if self.eq_mode == 'betaeq':
            return self.T, self.n_B_onset, self.n_B_offset
        else:
            if Y_C is None:
                raise ValueError("Y_C must be provided for fixedyc mode")
            mask = np.abs(self.Y_C - Y_C) < 0.01
            if np.sum(mask) == 0:
                return np.array([]), np.array([]), np.array([])
            return self.T[mask], self.n_B_onset[mask], self.n_B_offset[mask]


class PhaseBoundaries:
    """Collection of phase boundary tables for multiple eta values"""

    def __init__(self, base_path: str, eq_mode: str = 'betaeq',
                 eta_values: list = None, Y_C_values: list = None, verbose: bool = True):
        """
        Initialize phase boundaries collection

        Parameters:
        -----------
        base_path : str
            Directory containing phase boundary files
        eq_mode : str
            Either 'betaeq' or 'fixedyc'
        eta_values : list, optional
            List of eta values to load (default: [0.0, 0.1, 0.3, 0.6, 1.0])
        Y_C_values : list, optional
            List of Y_C values to load for fixedyc mode (default: [0.5])
        verbose : bool
            Print loading messages
        """
        self.base_path = Path(base_path)
        self.eq_mode = eq_mode
        self.eta_values = eta_values if eta_values is not None else [0.0, 0.1, 0.3, 0.6, 1.0]
        self.Y_C_values = Y_C_values if Y_C_values is not None else [0.5]
        self.verbose = verbose

        self.boundaries = {}  # keyed by eta for betaeq, by (eta, Y_C) for fixedyc

        self._load_all()

    def _load_all(self):
        """Load all phase boundary tables"""
        for eta in self.eta_values:
            eta_str = f"eta{eta:.2f}"

            if self.eq_mode == 'betaeq':
                boundary_file = self.base_path / f"phase_boundaries_{eta_str}_betaeq_B180_a0.2.dat"
                if boundary_file.exists():
                    self.boundaries[eta] = PhaseBoundarySingleEta(str(boundary_file), self.eq_mode)
                    if self.verbose:
                        print(f"✓ Loaded phase boundaries for eta={eta}")
                elif self.verbose:
                    print(f"✗ Warning: {boundary_file.name} not found")
            else:  # fixedyc - single file per eta contains all Y_C values
                # New format: single file without Y_C in filename, all Y_C as column
                boundary_file = self.base_path / f"phase_boundaries_{eta_str}_fixedYC_B180_a0.2.dat"
                if boundary_file.exists():
                    # Load once, store for all Y_C values (filtering happens at query time)
                    table = PhaseBoundarySingleEta(str(boundary_file), self.eq_mode)
                    for Y_C in self.Y_C_values:
                        self.boundaries[(eta, Y_C)] = table
                    if self.verbose:
                        print(f"✓ Loaded phase boundaries for eta={eta} (all Y_C values)")
                elif self.verbose:
                    print(f"✗ Warning: {boundary_file.name} not found")

    def _get_boundary_key(self, eta, Y_C=None):
        """Get the correct key for the boundaries dict based on eq_mode."""
        if isinstance(eta, (list, np.ndarray)):
            eta = float(eta[0])
        else:
            eta = float(eta)

        if not self.boundaries:
            raise ValueError("No phase boundary tables loaded")

        tolerance = 1e-6

        if self.eq_mode == 'betaeq':
            # Key is just eta
            for key in self.boundaries.keys():
                if abs(eta - key) < tolerance:
                    return key
            raise ValueError(f"eta={eta} not available. Available: {sorted(self.boundaries.keys())}")
        else:
            # fixedyc: key is (eta, Y_C)
            if Y_C is None:
                raise ValueError("Y_C must be provided for fixedyc mode")
            for key in self.boundaries.keys():
                if isinstance(key, tuple) and len(key) == 2:
                    if abs(eta - key[0]) < tolerance and abs(Y_C - key[1]) < tolerance:
                        return key
            available = [(k[0], k[1]) for k in self.boundaries.keys() if isinstance(k, tuple)]
            raise ValueError(f"(eta={eta}, Y_C={Y_C}) not available. Available: {available}")

    def nB_onset(self, T, eta, Y_C=None):
        """
        Get phase transition onset density at temperature T

        Parameters:
        -----------
        T : float
            Temperature
        eta : float
            Must exactly match one of the loaded eta values
        Y_C : float, optional
            Charge fraction (required for fixedyc mode)
        """
        key = self._get_boundary_key(eta, Y_C)
        return self.boundaries[key].nB_onset(T, Y_C)

    def nB_offset(self, T, eta, Y_C=None):
        """
        Get phase transition offset density at temperature T

        Parameters:
        -----------
        T : float
            Temperature
        eta : float
            Must exactly match one of the loaded eta values
        Y_C : float, optional
            Charge fraction (required for fixedyc mode)
        """
        key = self._get_boundary_key(eta, Y_C)
        return self.boundaries[key].nB_offset(T, Y_C)

    def get_boundary_arrays(self, eta, Y_C=None):
        """
        Get raw T, nB_onset, nB_offset arrays for given eta (and Y_C for fixedyc).

        Returns:
        --------
        T, nB_onset, nB_offset : arrays
        """
        key = self._get_boundary_key(eta, Y_C)
        return self.boundaries[key].get_boundary_arrays(Y_C)


class EOSCollection:
    """Collection of EOS tables for multiple eta values"""

    def __init__(self, base_path: str, eq_mode: str = 'betaeq',
                 eta_values: list = None, Y_C_values: list = None, verbose: bool = True):
        """
        Initialize EOS collection

        Parameters:
        -----------
        base_path : str
            Directory containing EOS files
        eq_mode : str
            Either 'betaeq' or 'fixedyc'
        eta_values : list, optional
            List of eta values to load (default: [0.0, 0.1, 0.3, 0.6, 1.0])
        Y_C_values : list, optional
            List of Y_C values for fixedyc mode (default: [0.5])
        verbose : bool
            Print loading messages
        """

        self.base_path = Path(base_path)
        self.eq_mode = eq_mode
        self.eta_values = eta_values if eta_values is not None else [0.0, 0.1, 0.3, 0.6, 1.0]
        self.Y_C_values = Y_C_values if Y_C_values is not None else [0.5]
        self.verbose = verbose

        self.eos_tables = {}
        self.boundaries = None

        self._load_all()

    def _load_all(self):
        """Load all EOS tables and boundaries"""
        for eta in self.eta_values:
            # Format eta for filename
            eta_str = f"eta{eta:.2f}"

            # Determine filenames
            if self.eq_mode == 'betaeq':
                eos_file = self.base_path / f"table_hybrid_betaeq_{eta_str}_B180_a0.2_complete.dat"
            else:  # fixedyc
                eos_file = self.base_path / f"table_hybrid_fixedYC_{eta_str}_B180_a0.2_complete.dat"

            # Load EOS table
            if eos_file.exists():
                self.eos_tables[eta] = EOSReader(str(eos_file), self.eq_mode)
                if self.verbose:
                    print(f"✓ Loaded {self.eq_mode} EOS table for eta={eta}")
            elif self.verbose:
                print(f"✗ Warning: {eos_file.name} not found")

        # Load phase boundaries collection (handles all eta values)
        self.boundaries = PhaseBoundaries(
            base_path=str(self.base_path),
            eq_mode=self.eq_mode,
            eta_values=self.eta_values,
            Y_C_values=self.Y_C_values,
            verbose=self.verbose
        )

    def _validate_eta(self, eta):
        """Validate that eta is in the available eta_values (exact match only)"""
        if isinstance(eta, (list, np.ndarray)):
            eta = float(eta[0])
        else:
            eta = float(eta)

        if not self.eos_tables:
            raise ValueError("No EOS tables loaded")

        # Check if eta is in the available values (with small tolerance for floating point)
        available_etas = list(self.eos_tables.keys())
        tolerance = 1e-6

        for available_eta in available_etas:
            if abs(eta - available_eta) < tolerance:
                return available_eta

        # If not found, raise error
        raise ValueError(
            f"eta={eta} not available. "
            f"Available eta values: {sorted(available_etas)}"
        )

    def get(self, quantity: str, n_B, T=0.0, eta=0.0, Y_C=None):
        """
        Get quantity from the eta table

        Parameters:
        -----------
        eta : float
            Must exactly match one of the loaded eta values
        """
        valid_eta = self._validate_eta(eta)
        return self.eos_tables[valid_eta].get(quantity, n_B, T, 0.0, Y_C)

    def nB_onset(self, T, eta=0.0, Y_C=None):
        """Get phase transition onset density"""
        return self.boundaries.nB_onset(T, eta, Y_C)

    def nB_offset(self, T, eta=0.0, Y_C=None):
        """Get phase transition offset density"""
        return self.boundaries.nB_offset(T, eta, Y_C)

    def get_boundary_arrays(self, eta, Y_C=None):
        """
        Get raw T, nB_onset, nB_offset arrays for given eta (and Y_C for fixedyc).

        Returns:
        --------
        T, nB_onset, nB_offset : arrays
        """
        return self.boundaries.get_boundary_arrays(eta, Y_C)

    def to_tov_table(self, eta: float, T: float = 0.0, Y_C: float = None):
        """
        Extract (P, ε, n_B) arrays for TOV solver at fixed (eta, T, Y_C).
        
        Args:
            eta: Surface tension parameter (must match loaded eta values)
            T: Temperature in MeV (default 0.0)
            Y_C: Charge fraction (required for fixedyc mode)
            
        Returns:
            EOSTable_for_TOV ready for tov_solver.solve_tov_sequence()
        """
        from tov_solver import EOSTable_for_TOV
        
        valid_eta = self._validate_eta(eta)
        reader = self.eos_tables[valid_eta]
        
        # Get n_B grid from the reader's data
        nB_values = np.unique(reader.data['n_B'])
        nB_values = np.sort(nB_values)
        
        # Extract P and ε at each n_B point for the specified (T, Y_C)
        P_arr = np.array([reader.P_tot(nB, T, 0.0, Y_C) for nB in nB_values])
        e_arr = np.array([reader.e_tot(nB, T, 0.0, Y_C) for nB in nB_values])
        
        # Filter out NaN values (from extrapolation failures)
        valid = ~(np.isnan(P_arr) | np.isnan(e_arr))
        
        return EOSTable_for_TOV(
            P=P_arr[valid], 
            epsilon=e_arr[valid], 
            nB=nB_values[valid]
        )


# =============================================================================
# SIMPLE USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("="*70)
    print("Simple EOS Reader Example")
    print("="*70)

    # Example 1: Single file
    print("\n1. Reading single EOS file:")
    print("-" * 70)

    eos_file = "output/zlvmit_eos/table_hybrid_eta060_B180_a0.2_complete.dat"
    if Path(eos_file).exists():
        eos = EOSReader(eos_file, eq_mode='betaeq')

        # Simple usage - just like you requested!
        n_B = 0.3
        T = 0.0
        eta = 0.6

        P = eos.P_tot(n_B, T, eta)
        e = eos.e_tot(n_B, T, eta)
        chi = eos.chi(n_B, T, eta)

        print(f"At n_B={n_B}, T={T}, eta={eta}:")
        print(f"  P_total = {P:.3f} MeV/fm^3")
        print(f"  e_total = {e:.3f} MeV/fm^3")
        print(f"  chi     = {chi:.3f}")

        # Simple plot - as requested!
        print("\nCreating simple plot...")
        n_B_array = np.linspace(0.1, 0.8, 100)
        P_array = eos.P_tot(n_B_array, T=0.0, eta=0.6)

        plt.figure(figsize=(8, 5))
        plt.plot(n_B_array, P_array, 'b-', linewidth=2)
        plt.xlabel('n_B [fm$^{-3}$]')
        plt.ylabel('P [MeV/fm$^3$]')
        plt.title('Pressure vs Baryon Density')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('simple_pressure_plot.png', dpi=150)
        print("Saved: simple_pressure_plot.png")
    else:
        print(f"File not found: {eos_file}")

    # Example 2: Collection of files
    print("\n2. Loading collection of EOS tables:")
    print("-" * 70)

    base_path = "output/zlvmit_eos"
    if Path(base_path).exists():
        collection = EOSCollection(base_path, eq_mode='betaeq')

        print(f"\nLoaded {len(collection.eos_tables)} EOS tables")
        print(f"Available eta values: {list(collection.eos_tables.keys())}")
