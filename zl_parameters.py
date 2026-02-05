"""
zl_parameters.py
=================
Parameter dataclasses for the Zhao-Lattimer (ZL) nucleonic EOS model.

The ZL model is a schematic density-functional parametrization with:
- Symmetric matter term: 4*nB²*Yp*Yn * [a0/n0 + b0/n0 * (nB/n0)^(γ-1)]
- Symmetry energy term: nB²*(Yn-Yp)² * [a1/n0 + b1/n0 * (nB/n0)^(γ1-1)]

References:
- Zhao & Lattimer, PRD 102, 023021 (2020)
- C. Constantinou et al. Phys. Rev. D 104.12 (2021)
- C. Constantinou et al. Phys. Rev. D 107.7 (2023)
- C. Constantinou et al. Phys. Rev. D 112.9 (2025)
- M. Guerrini PhD Thesis (2026)
"""
from dataclasses import dataclass


@dataclass
class ZLParams:
    """
    Parameters for the Zhao-Lattimer nucleonic EOS.
    
    Default values correspond to the parametrization from Constantinou et al.
    
    Attributes:
        name: Parameter set identifier
        m_p: Proton mass (MeV)
        m_n: Neutron mass (MeV)
        n0: Saturation density (fm⁻³)
        a0: Symmetric matter linear coefficient (MeV)
        b0: Symmetric matter power-law coefficient (MeV)
        gamma: Symmetric matter power-law exponent
        a1: Symmetry energy linear coefficient (MeV)
        b1: Symmetry energy power-law coefficient (MeV)
        gamma1: Symmetry energy power-law exponent
    """
    name: str = "ZL_default"
    m_p: float = 939.5     # MeV (proton mass)
    m_n: float = 939.5     # MeV (neutron mass)
    n0: float = 0.16       # fm^-3 (saturation density)
    a0: float = -96.64     # MeV (symmetric linear term)
    b0: float = 58.85      # MeV (symmetric power term)
    gamma: float = 1.40    # symmetric power-law exponent
    a1: float = -26.06     # MeV (symmetry energy linear term)
    b1: float = 7.34       # MeV (symmetry energy power term)
    gamma1: float = 2.45   # symmetry energy power-law exponent


def get_zl_default() -> ZLParams:
    """Get default ZL parameter set (Constantinou et al.)."""
    return ZLParams(name="ZL_Constantinou")


def get_zl_custom(
    a0: float = -96.64, b0: float = 58.85, gamma: float = 1.40,
    a1: float = -26.06, b1: float = 7.34, gamma1: float = 2.45,
    n0: float = 0.16, name: str = "ZL_custom"
) -> ZLParams:
    """
    Create custom ZL parameter set.
    
    Args:
        a0, b0, gamma: Symmetric matter parameters
        a1, b1, gamma1: Symmetry energy parameters
        n0: Saturation density (fm⁻³)
        name: Parameter set name
        
    Returns:
        ZLParams with specified values
    """
    return ZLParams(
        name=name,
        n0=n0,
        a0=a0, b0=b0, gamma=gamma,
        a1=a1, b1=b1, gamma1=gamma1
    )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("ZL Parameters Test")
    print("=" * 50)
    
    params = get_zl_default()
    print(f"\nDefault parameters: {params.name}")
    print(f"  n0    = {params.n0} fm⁻³")
    print(f"  a0    = {params.a0} MeV")
    print(f"  b0    = {params.b0} MeV")
    print(f"  γ     = {params.gamma}")
    print(f"  a1    = {params.a1} MeV")
    print(f"  b1    = {params.b1} MeV")
    print(f"  γ1    = {params.gamma1}")
    
    print("\n✓ All OK")
