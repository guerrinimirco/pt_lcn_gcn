"""
general_particles.py
====================
Generic particle definitions for relativistic mean-field (RMF) EOS calculations.

Quantum Number Conventions:
---------------------------
- Baryon number (B): +1 for baryons, 0 for mesons/leptons
- Charge (Q): Electric charge in units of e
- Strangeness (S): s-quark carries S = +1, anti-s carries S = -1
  (This is OPPOSITE to the historical PDG convention where s has S = -1)
- Strong charge (C): C = Q for hadrons and quarks, C = 0 for leptons
  This is useful for charge conservation in strong interactions.
  
- Isospin 3rd component (I₃): defined consistently with our S convention
- Hypercharge: Y = B - S (with our convention)
- Gell-Mann–Nishijima: Q = I₃ + Y/2 = I₃ + (B - S)/2

Strangeness Examples (our convention):
    - Λ (uds): 1 s-quark → S = +1
    - Σ (uus/uds/dds): 1 s-quark → S = +1
    - Ξ (uss/dss): 2 s-quarks → S = +2
    - K⁺ (us̄): 1 anti-s → S = -1
    - K⁻ (ūs): 1 s-quark → S = +1

Reference masses from PDG 2022.
"""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass(frozen=True)
class Particle:
    """
    Immutable particle definition with all relevant quantum numbers.
    
    Attributes:
        name: Human-readable identifier (e.g., "p", "Lambda", "K+")
        spin: Spin quantum number J (in units of ℏ)
        g_degen: Spin degeneracy factor (2J + 1), times color if applicable
        baryon_no: Baryon number B (+1 for baryons, 0 for mesons/leptons)
        lepton_no: Lepton number L (+1 for leptons, 0 otherwise)
        mass: Rest mass in MeV (optional for generic particles)
        charge: Electric charge Q in units of e (optional)
        strangeness: S = (# of s-quarks) - (# of anti-s quarks) with s→+1 
        isospin_3: Third component of isospin I₃ (optional)
    """
    name: str
    spin: float                          # J
    g_degen: float                       # Degeneracy (2J+1) × color
    baryon_no: float = 0.0               # B
    lepton_no: float = 0.0               # L  
    mass: Optional[float] = None         # MeV (None for generic)
    charge: Optional[float] = None       # Q in units of e
    strangeness: Optional[float] = None  # S (s-quark = +1)
    isospin_3: Optional[float] = None    # I₃
    
    @property
    def is_boson(self) -> bool:
        """True if particle is a boson (integer spin)."""
        return (self.spin % 1) == 0
    
    @property
    def is_fermion(self) -> bool:
        """True if particle is a fermion (half-integer spin)."""
        return not self.is_boson
    
    @property
    def is_lepton(self) -> bool:
        """True if particle is a lepton."""
        return self.lepton_no != 0
    
    @property
    def is_hadron(self) -> bool:
        """True if particle is a hadron (baryon or meson)."""
        return (self.baryon_no != 0) or (self.strangeness != 0) or (self.name in _MESON_NAMES)
    
    @property
    def is_quark(self) -> bool:
        """True if particle is a quark (fractional baryon number)."""
        return 0 < abs(self.baryon_no) < 1
    
    @property
    def strong_charge(self) -> float:
        """
        Strong charge C: equals Q for hadrons and quarks, 0 for leptons.
        Conserved in strong interactions.
        """
        if self.is_lepton:
            return 0.0
        return self.charge
    
    # Alias for strong_charge
    C = strong_charge
    
    @property
    def is_strange(self) -> bool:
        """True if particle contains strange quarks."""
        return self.strangeness != 0
    
    @property
    def is_antiparticle(self) -> bool:
        """Heuristic: negative baryon number or specific naming."""
        return self.baryon_no < 0 or self.name.endswith("_bar")
    
    def __str__(self) -> str:
        return (f"{self.name} (m={self.mass:.2f} MeV, Q={self.charge:+.0f}, "
                f"B={self.baryon_no:.0f}, S={self.strangeness:+.0f}, C={self.strong_charge:+.0f})")


# Helper set for meson identification (mesons with S=0 and B=0)
_MESON_NAMES = {"pi+", "pi0", "pi-", "eta", "eta_prime", 
                "K+", "K0", "K-", "K0_bar"}


def create_antiparticle(p: Particle, name: Optional[str] = None) -> Particle:
    """
    Create antiparticle by flipping charge, baryon number, lepton number, 
    strangeness, and isospin. Mass and spin degeneracy remain the same.
    """
    if name is None:
        if p.name.endswith("+"):
            name = p.name[:-1] + "-"
        elif p.name.endswith("-"):
            name = p.name[:-1] + "+"
        elif p.name.endswith("_bar"):
            name = p.name[:-4]
        else:
            name = p.name + "_bar"
    
    return Particle(
        name=name,
        mass=p.mass,
        spin=p.spin,
        charge=-p.charge,
        baryon_no=-p.baryon_no,
        lepton_no=-p.lepton_no,
        strangeness=-p.strangeness,
        isospin_3=-p.isospin_3,
        g_degen=p.g_degen
    )


# =============================================================================
# LEPTONS (L = +1 for particles, C = 0 always)
# =============================================================================
Electron = Particle(
    name="e-", mass=0.510999, spin=0.5,
    charge=-1.0, baryon_no=0.0, lepton_no=1.0,
    strangeness=0.0, isospin_3=0.0, g_degen=2.0
)

Muon = Particle(
    name="mu-", mass=105.6584, spin=0.5,
    charge=-1.0, baryon_no=0.0, lepton_no=1.0,
    strangeness=0.0, isospin_3=0.0, g_degen=2.0
)

# Single generic neutrino (all flavors assumed massless)
Neutrino = Particle(
    name="nu", mass=0.0, spin=0.5,
    charge=0.0, baryon_no=0.0, lepton_no=1.0,
    strangeness=0.0, isospin_3=0.0, g_degen=1.0
)

# Antileptons
Positron = create_antiparticle(Electron, "e+")
AntiMuon = create_antiparticle(Muon, "mu+")
AntiNeutrino = create_antiparticle(Neutrino, "nu_bar")


# =============================================================================
# BARYONS - NUCLEONS (J = 1/2, S = 0)
# Isospin doublet: I = 1/2
# =============================================================================
Proton = Particle(
    name="p", mass=938.2721, spin=0.5,
    charge=+1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=+0.5, g_degen=2.0
)

Neutron = Particle(
    name="n", mass=939.5654, spin=0.5,
    charge=0.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=-0.5, g_degen=2.0
)


# =============================================================================
# BARYONS - HYPERONS (J = 1/2, S > 0 with our convention)
# =============================================================================

# --- Lambda: uds, I = 0, S = +1 ---
Lambda = Particle(
    name="Lambda", mass=1115.683, spin=0.5,
    charge=0.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=0.0, g_degen=2.0
)

# --- Sigma: I = 1 triplet, S = +1 ---
SigmaP = Particle(
    name="Sigma+", mass=1189.37, spin=0.5,
    charge=+1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=+1.0, g_degen=2.0
)

Sigma0 = Particle(
    name="Sigma0", mass=1192.642, spin=0.5,
    charge=0.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=0.0, g_degen=2.0
)

SigmaM = Particle(
    name="Sigma-", mass=1197.449, spin=0.5,
    charge=-1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=-1.0, g_degen=2.0
)

# --- Xi (Cascade): I = 1/2 doublet, S = +2 ---
Xi0 = Particle(
    name="Xi0", mass=1314.86, spin=0.5,
    charge=0.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+2.0, isospin_3=+0.5, g_degen=2.0
)

XiM = Particle(
    name="Xi-", mass=1321.71, spin=0.5,
    charge=-1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=+2.0, isospin_3=-0.5, g_degen=2.0
)


# =============================================================================
# BARYONS - DELTA RESONANCES (J = 3/2, S = 0)
# Isospin quartet: I = 3/2
# =============================================================================
DeltaPP = Particle(
    name="Delta++", mass=1232.0, spin=1.5,
    charge=+2.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=+1.5, g_degen=4.0
)

DeltaP = Particle(
    name="Delta+", mass=1232.0, spin=1.5,
    charge=+1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=+0.5, g_degen=4.0
)

Delta0 = Particle(
    name="Delta0", mass=1232.0, spin=1.5,
    charge=0.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=-0.5, g_degen=4.0
)

DeltaM = Particle(
    name="Delta-", mass=1232.0, spin=1.5,
    charge=-1.0, baryon_no=1.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=-1.5, g_degen=4.0
)


# =============================================================================
# MESONS - PSEUDOSCALAR (J = 0)
# =============================================================================

# --- Pions: I = 1 triplet, S = 0 ---
PiP = Particle(
    name="pi+", mass=139.570, spin=0.0,
    charge=+1.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=+1.0, g_degen=1.0
)

Pi0 = Particle(
    name="pi0", mass=134.977, spin=0.0,
    charge=0.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=0.0, g_degen=1.0
)

PiM = Particle(
    name="pi-", mass=139.570, spin=0.0,
    charge=-1.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=-1.0, g_degen=1.0
)

# --- Kaons: I = 1/2 doublets ---
# K⁺ = us̄ → S = -1 (anti-strange)
KaonP = Particle(
    name="K+", mass=493.677, spin=0.0,
    charge=+1.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=-1.0, isospin_3=+0.5, g_degen=1.0
)

# K⁰ = ds̄ → S = -1 (anti-strange)
Kaon0 = Particle(
    name="K0", mass=497.611, spin=0.0,
    charge=0.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=-1.0, isospin_3=-0.5, g_degen=1.0
)

# K⁻ = ūs → S = +1 (strange)
KaonM = Particle(
    name="K-", mass=493.677, spin=0.0,
    charge=-1.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=-0.5, g_degen=1.0
)

# K̄⁰ = d̄s → S = +1 (strange)
Kaon0Bar = Particle(
    name="K0_bar", mass=497.611, spin=0.0,
    charge=0.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=+1.0, isospin_3=+0.5, g_degen=1.0
)

# --- Eta mesons: I = 0, S = 0 ---
Eta = Particle(
    name="eta", mass=547.862, spin=0.0,
    charge=0.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=0.0, g_degen=1.0
)

EtaPrime = Particle(
    name="eta_prime", mass=957.78, spin=0.0,
    charge=0.0, baryon_no=0.0, lepton_no=0.0,
    strangeness=0.0, isospin_3=0.0, g_degen=1.0
)


# =============================================================================
# QUARKS (for reference, not typically used directly in hadronic EOS)
# g = 2(spin) × 3(color) = 6
# =============================================================================
Up = Particle(
    name="u", mass=2.16, spin=0.5,
    charge=+2/3, baryon_no=1/3, lepton_no=0.0,
    strangeness=0.0, isospin_3=+0.5, g_degen=6.0
)

Down = Particle(
    name="d", mass=4.67, spin=0.5,
    charge=-1/3, baryon_no=1/3, lepton_no=0.0,
    strangeness=0.0, isospin_3=-0.5, g_degen=6.0
)

Strange = Particle(
    name="s", mass=93.4, spin=0.5,
    charge=-1/3, baryon_no=1/3, lepton_no=0.0,
    strangeness=+1.0, isospin_3=0.0, g_degen=6.0
)

# Generic quark (for vMIT EOS where u,d,s share degeneracy)
Quark = Particle(
    name="quark", spin=0.5,
    baryon_no=1/3, lepton_no=0.0, 
    g_degen=6.0
)

# =============================================================================
# PARTICLE GROUPS (for convenient iteration)
# =============================================================================
LEPTONS = [Electron, Muon]
NUCLEONS = [Proton, Neutron]
HYPERONS_OCTET = [Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
DELTAS = [DeltaPP, DeltaP, Delta0, DeltaM]
BARYONS_OCTET = NUCLEONS + HYPERONS_OCTET
BARYONS_ALL = BARYONS_OCTET + DELTAS

PIONS = [PiP, Pi0, PiM]
KAONS = [KaonP, Kaon0, KaonM, Kaon0Bar]
ETAS = [Eta, EtaPrime]
MESONS_PSEUDOSCALAR = PIONS + KAONS + ETAS

QUARKS = [Up, Down, Strange, Quark]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_particle(name: str) -> Optional[Particle]:
    """Find a particle by its name from all defined particles."""
    all_particles = (LEPTONS + [Neutrino] + BARYONS_ALL + 
                    MESONS_PSEUDOSCALAR + QUARKS +
                    [Positron, AntiMuon, AntiNeutrino])
    for p in all_particles:
        if p.name == name:
            return p
    return None


def verify_gell_mann_nishijima(p: Particle, verbose: bool = False) -> bool:
    """
    Verify the Gell-Mann–Nishijima formula with OUR strangeness convention.
    
    Standard formula: Q = I₃ + (B + S)/2  where s-quark has S = -1
    Our convention:   Q = I₃ + (B - S)/2  where s-quark has S = +1
    
    Returns True if the relation holds (within floating point tolerance).
    """
    Q_calc = p.isospin_3 + (p.baryon_no - p.strangeness) / 2
    is_valid = np.isclose(Q_calc, p.charge, atol=1e-10)
    
    if verbose:
        status = "✓" if is_valid else "✗"
        print(f"{status} {p.name}: Q={p.charge:+.1f}, I₃={p.isospin_3:+.1f}, "
              f"B={p.baryon_no:.0f}, S={p.strangeness:+.0f}, C={p.strong_charge:+.1f} → Q_calc={Q_calc:+.1f}")
    
    return is_valid


def verify_all_particles(verbose: bool = True) -> bool:
    """Verify Gell-Mann–Nishijima Q = I₃ + (B-S)/2 for all hadrons."""
    all_hadrons = BARYONS_ALL + MESONS_PSEUDOSCALAR
    all_valid = True
    
    if verbose:
        print("Verifying Gell-Mann–Nishijima formula: Q = I₃ + (B - S)/2")
        print("(Using our convention: s-quark has S = +1)")
        print("=" * 70)
    
    for p in all_hadrons:
        valid = verify_gell_mann_nishijima(p, verbose)
        all_valid = all_valid and valid
    
    if verbose:
        print("=" * 70)
        print(f"All particles valid: {all_valid}")
    
    return all_valid


def verify_strong_charge(verbose: bool = True) -> bool:
    """Verify that C = Q for hadrons/quarks and C = 0 for leptons."""
    all_valid = True
    
    if verbose:
        print("\nVerifying strong charge C:")
        print("  C = Q for hadrons/quarks")
        print("  C = 0 for leptons")
        print("=" * 70)
    
    # Check leptons: C should be 0
    for p in LEPTONS + [Neutrino]:
        valid = (p.strong_charge == 0.0)
        all_valid = all_valid and valid
        if verbose:
            status = "✓" if valid else "✗"
            print(f"{status} {p.name} (lepton): Q={p.charge:+.1f}, C={p.strong_charge:+.1f}")
    
    # Check hadrons: C should equal Q
    for p in BARYONS_ALL + MESONS_PSEUDOSCALAR:
        valid = np.isclose(p.strong_charge, p.charge, atol=1e-10)
        all_valid = all_valid and valid
        if verbose:
            status = "✓" if valid else "✗"
            print(f"{status} {p.name} (hadron): Q={p.charge:+.1f}, C={p.strong_charge:+.1f}")
    
    if verbose:
        print("=" * 70)
        print(f"All strong charges valid: {all_valid}")
    
    return all_valid


def print_particle_table(particles: List[Particle], title: str = "Particles"):
    """Print a formatted table of particle properties."""
    print(f"\n{title}")
    print("=" * 90)
    print(f"{'Name':<12} {'Mass (MeV)':<12} {'J':<5} {'Q':<6} {'C':<6} {'B':<4} {'S':<5} {'I₃':<6} {'g':<4} {'Type':<8}")
    print("-" * 90)
    for p in particles:
        ptype = "Boson" if p.is_boson else "Fermion"
        print(f"{p.name:<12} {p.mass:<12.3f} {p.spin:<5.1f} {p.charge:<+6.1f} {p.strong_charge:<+6.1f} "
              f"{p.baryon_no:<4.0f} {p.strangeness:<+5.0f} {p.isospin_3:<+6.1f} {p.g_degen:<4.0f} {ptype:<8}")


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Particle Definitions Module")
    print("Strangeness convention: s-quark → S = +1, anti-s → S = -1")
    print("Strong charge: C = Q (hadrons/quarks), C = 0 (leptons)\n")
    
    # Verify quantum numbers
    verify_all_particles(verbose=True)
    
    # Verify strong charge
    verify_strong_charge(verbose=False)  # Brief check
    
    # Print tables
    print_particle_table(LEPTONS + [Neutrino], "LEPTONS")
    print_particle_table(NUCLEONS, "NUCLEONS")
    print_particle_table(HYPERONS_OCTET, "HYPERONS (Octet)")
    print_particle_table(DELTAS, "DELTA RESONANCES")
    print_particle_table(KAONS, "KAONS")
    
    # Show is_boson deduction works
    print("\n\nVerifying is_boson deduction from spin:")
    print("-" * 50)
    for p in [Proton, Electron, PiP, Delta0, Neutrino]:
        print(f"{p.name}: spin={p.spin}, is_boson={p.is_boson}, is_fermion={p.is_fermion}")