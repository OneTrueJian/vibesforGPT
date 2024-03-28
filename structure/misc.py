import numpy as np
from numpy import cos, sin


def get_sysname(atoms, spacegroup=None, empirical=True, metal=True):
    """Get name of the system

    Args:
      atoms(ase.atoms.Atoms): The structure to name
      spacegroup(int): space group of atoms to attach to the name (Default value = None)
      empirical(bool): return empirical name (remove duplicitiy) (Default value = True)
      metal(bool): use `mode='metal'` instead of `'hill'` to put metal elements first

    Returns:
      str: The name of atoms

    """

    if metal:
        mode = "metal"
    else:
        mode = "hill"

    chemical_formula = atoms.get_chemical_formula(mode=mode, empirical=empirical)

    if spacegroup is None and hasattr(atoms, "spacegroup"):
        spacegroup = atoms.spacegroup

    if spacegroup is None:
        return chemical_formula

    sg_number = spacegroup.number
    wyckoff_pos = spacegroup.wyckoffs
    sysname = f"{chemical_formula}_{sg_number}"
    wyck_uniq, wyck_mult = np.unique(wyckoff_pos, return_counts=1)
    for mult, wyck in zip(wyck_mult, wyck_uniq):
        sysname += f"_{mult}{wyck}"
    return sysname


def generate_lattice(a, b=None, c=None, alpha=90, beta=90, gamma=90, lattice_type=None):
    """Create a Lattice using unit cell lengths (Angstrom) and angles (in degrees).

    Args:
      a(float): *a* lattice parameter.
      b(float, optional): *b* lattice parameter. (Default value = None)
      c(float, optional): *c* lattice parameter. (Default value = None)
      alpha(float, optional): *alpha* angle in degrees. (Default value = 90)
      beta(float, optional): *beta* angle in degrees. (Default value = 90)
      gamma(float, optional): *gamma* angle in degrees. (Default value = 90)
      lattice_type (str): The lattice type
      lattice_type:  (Default value = None)

    Returns:
        np.ndarray: lattice

    """
    if lattice_type == "cubic":
        return np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    elif lattice_type == "tetragonal":
        if c is None:
            print("Error: Tetragonal lattice needs parameter `c`.")
            return None
        b = a
    elif lattice_type == "orthorhombic":
        if b is None or c is None:
            print("Error: Orthorhombic lattice needs parameters `b` and `c`.")
    elif lattice_type == "monoclinic":
        if b is None or c is None:
            print("Error: Monoclinic lattice needs parameters `b` and `c`.")
        if beta == 90:
            print("Warning: Have you set beta? It is 90 -> orthorhombic.")
    elif lattice_type == "hexagonal":
        if c is None:
            print("Error: Hexagonal lattice needs parameters `c`.")
        b = a
        gamma = 120
    elif lattice_type == "rhombohedral":
        b = a
        c = a
        beta = alpha
        gamma = alpha

    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)
    val = (cos(alpha_r) * cos(beta_r) - cos(gamma_r)) / (sin(alpha_r) * sin(beta_r))
    # Sometimes rounding errors result in values slightly > 1.
    val = max(min(val, 1), -1)
    gamma_star = np.arccos(val)
    vector_a = [a * sin(beta_r), 0.0, a * cos(beta_r)]
    vector_b = [
        -b * sin(alpha_r) * cos(gamma_star),
        b * sin(alpha_r) * sin(gamma_star),
        b * cos(alpha_r),
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])
