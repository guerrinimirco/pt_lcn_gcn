"""
vmit_parameters.py
===================
Parameter dataclasses for the vMIT (vector-enhanced MIT bag) quark EOS model.

The vMIT model includes:
- Current quark masses (u, d, s)
- Bag constant B (confining pressure)
- Vector interaction parameter a = g_V²/m_V²

References:
- R. O. Gomes et al. Astrophys. J. 877.2 (2019)
- C. Constantinou et al. Phys. Rev. D 104.12 (2021)
- C. Constantinou et al. Phys. Rev. D 107.7 (2023)
- C. Constantinou et al. Phys. Rev. D 112.9 (2025)
- M. Guerrini PhD Thesis (2026)
"""
from dataclasses import dataclass, field


@dataclass
class VMITParams:
    """
    Parameters for the vMIT quark EOS.
    
    Attributes:
        name: Parameter set identifier
        m_u: Up quark mass (MeV)
        m_d: Down quark mass (MeV)
        m_s: Strange quark mass (MeV)
        a: Vector coupling parameter g_V²/m_V² (fm²)
        B4: Bag constant B^(1/4) (MeV)
        B: Bag constant B (MeV⁴) = B4⁴
    """
    name: str = "vMIT_default"
    m_u: float = 5.0       # MeV (up quark mass)
    m_d: float = 7.0       # MeV (down quark mass)
    m_s: float = 150.0     # MeV (strange quark mass)
    a: float = 0.2         # fm² (vector coupling = g_V²/m_V²)
    B4: float = 180.0      # MeV (bag constant B^1/4)
    
    @property
    def B(self) -> float:
        """Bag constant B = (B^1/4)^4 in MeV⁴."""
        return self.B4**4


def get_vmit_default() -> VMITParams:
    """Get default vMIT parameter set."""
    return VMITParams(name="vMIT_default")


def get_vmit_custom(
    m_u: float = 5.0, m_d: float = 7.0, m_s: float = 150.0,
    a: float = 0.2, B4: float = 180.0, name: str = "vMIT_custom"
) -> VMITParams:
    """
    Create custom vMIT parameter set.
    
    Args:
        m_u, m_d, m_s: Quark masses (MeV)
        a: Vector coupling g_V²/m_V² (fm²)
        B4: Bag constant B^(1/4) (MeV)
        name: Parameter set name
        
    Returns:
        VMITParams with specified values
    """
    return VMITParams(
        name=name,
        m_u=m_u, m_d=m_d, m_s=m_s,
        a=a, B4=B4
    )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    from general_physics_constants import hc3
    
    print("vMIT Parameters Test")
    print("=" * 50)
    
    params = get_vmit_default()
    print(f"\nDefault parameters: {params.name}")
    print(f"  m_u   = {params.m_u} MeV")
    print(f"  m_d   = {params.m_d} MeV")
    print(f"  m_s   = {params.m_s} MeV")
    print(f"  a     = {params.a} fm²")
    print(f"  B^1/4 = {params.B4} MeV")
    print(f"  B     = {params.B:.4e} MeV⁴")
    print(f"  B     = {params.B/hc3:.4f} MeV/fm³")
    
    print("\n✓ All OK")
