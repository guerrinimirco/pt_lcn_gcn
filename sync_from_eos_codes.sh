#!/bin/bash
# ============================================================================
# Sync ZL+vMIT files from eos-codes into pt_lcn_gcn
#
# Usage:
#   ./sync_from_eos_codes.sh                  # uses default source path
#   ./sync_from_eos_codes.sh /path/to/eos-codes  # custom source path
#
# This copies the 17 files (16 .py + 1 notebook) needed to run
# notebook_ZLvMIT_hybrid.ipynb from the eos-codes repository.
# ============================================================================

set -e

# Source directory (default or user-provided)
SRC="${1:-/Users/mircoguerrini/Desktop/Research/Python_codes}"

# Destination directory (where this script lives)
DST="$(cd "$(dirname "$0")" && pwd)"

# Files needed by notebook_ZLvMIT_hybrid.ipynb
FILES=(
    # Notebook
    notebook_ZLvMIT_hybrid.ipynb

    # General utilities
    general_physics_constants.py
    general_particles.py
    general_fermi_integrals.py
    general_thermodynamics_leptons.py

    # ZL (hadronic phase)
    zl_parameters.py
    zl_eos.py
    zl_thermodynamics_nucleons.py
    zl_compute_tables.py

    # vMIT (quark phase)
    vmit_parameters.py
    vmit_eos.py
    vmit_thermodynamics_quarks.py
    vmit_compute_tables.py

    # ZL+vMIT hybrid / mixed phase
    zlvmit_mixed_phase_eos.py
    zlvmit_table_reader.py
    zlvmit_plot_results.py

    # TOV solver
    tov_solver.py
)

# Check source exists
if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory not found: $SRC"
    exit 1
fi

# Sync files
echo "Syncing from: $SRC"
echo "         to:  $DST"
echo ""

copied=0
missing=0
for f in "${FILES[@]}"; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$DST/$f"
        echo "  [OK] $f"
        ((copied++))
    else
        echo "  [MISSING] $f"
        ((missing++))
    fi
done

echo ""
echo "Done: $copied files copied, $missing missing."

if [ "$missing" -gt 0 ]; then
    echo "WARNING: Some files were not found in $SRC"
    exit 1
fi
