""" Physical constants and atomic units """
from ase import units as ase_units
from numpy import pi

from vibes.helpers import AttributeDict


# physical constants
AMU = ase_units._amu  # = 1.66053904e-27  # [kg]
LIGHT = ase_units._c  # = 299792458  # [m / s]
PLANCK_CONSTANT = ase_units._hplanck  # = 6.62607015e-34  # [J s]
BOLTZMANN = ase_units._k  # = 1.38064852e-23  # [J / K]
AVOGADRO = ase_units._Nav  # = 6.02214076e23  # [1]
ALPHA = ase_units.alpha  # = 1 / 137.035999046  # [1]

# Mathematical constants
PI = pi

# Conversion factors
AA = 1e-10  # [m]
PICO = 1e-12  # [s]
FEMTO = 1e-3 * PICO  # [s]
EV = ase_units._e  # = 1.60217733e-19  # [J]
THZ = 1 / PICO  # [1/s]

# Atomic units
ELECTRON_MASS = ase_units._me  # = 5.48579909070e-4 * AMU  # [kg]
HARTREE = ase_units.Hartree * EV  # = 27.21138602 * EV  # [J]
BOHR = ase_units.Bohr * AA  # = 0.52917721092 * AA  # [m]
HBAR = ase_units._hbar  # = PLANCK_CONSTANT / 2 / PI  # [J s]

atomic_units = AttributeDict(
    {
        "bohr": BOHR,
        "AA": BOHR / AA,
        "hartree": HARTREE,
        "eV": HARTREE / EV,
        "c": 1 / ALPHA,
        "kg": ELECTRON_MASS,
        "u": ELECTRON_MASS / AMU,
        "s": HBAR / HARTREE,
        "fs": HBAR / HARTREE / FEMTO,
        "kB": BOLTZMANN / HARTREE,
        "K": HARTREE / BOLTZMANN,
    }
)

# force constants
omega_to_THz = (EV / AA ** 2 / AMU) ** 0.5 / THZ / 2 / PI  # 15.633302 THz
THz_to_cm = THZ / LIGHT / 100  # 33.3564 [1/cm]
omega_to_cm = omega_to_THz * THz_to_cm

amu_AA_THz_to_eV = AMU * AA ** 2 * THZ ** 2 / EV

# old stuff
# Bohr_to_AA = atomic_units.AA
# Hartree_to_eV = atomic_units.eV
# au_to_kg = atomic_units.kg
# kg_to_au = 1 / au_to_kg
# u_in_kg = AMU
# u_in_au = u_in_kg * kg_to_au
# c_in_au = atomic_units.c
# au_to_s = atomic_units.s
# au_to_fs = atomic_units.fs
# au_to_cm = au_to_s * 2 * PI * 3e10
# au_to_K = atomic_units.K

kB = BOLTZMANN / EV  # [eV/K] 8.6173383e-05
EvTokJmol = EV / 1000 * AVOGADRO  # [kJ/mol] 96.4853910
kJmolToEv = 1 / EvTokJmol
THzToEv = PLANCK_CONSTANT * 1e12 / EV  # [eV]
