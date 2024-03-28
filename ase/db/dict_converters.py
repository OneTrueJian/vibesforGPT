"""Functions to convert ASE Objects to dicts and getting them from dicts"""
import numpy as np
from ase.db.row import AtomsRow
from ase.db.row import atoms2dict as ase_atoms2dict


def atoms2dict(atoms):
    """Converts a Atoms object into a dict

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The Atoms object to be converted into a dictionary

    Returns
    -------
    atoms_dict: dict
        The dictionary of atoms
    """
    if atoms is None:
        return None
    if isinstance(atoms, dict):
        return atoms
    atoms_dict = ase_atoms2dict(atoms)
    if "cell" in atoms_dict:
        try:
            atoms_dict["cell"] = atoms.cell.array
        except AttributeError:
            atoms_dict["cell"] = atoms.cell

    # add information that is missing after using ase.atoms2dict
    atoms_dict["info"] = atoms.info

    # attach calculator
    for key, val in calc2dict(atoms.calc).items():
        atoms_dict[key] = val
    return atoms_dict


def dict2atoms(atoms_dict):
    """Converts a dict into an Atoms object

    Parameters
    ----------
    atoms_dict: dict
        A dictionary representing the pAtoms object

    Returns
    -------
    atoms: ase.atoms.Atoms
        The corresponding Atoms object
    """
    try:
        atoms = AtomsRow(atoms_dict).toatoms(attach_calculator=True)
    except AttributeError:
        atoms = AtomsRow(atoms_dict).toatoms(attach_calculator=False)

    # Attach missing information
    if "info" in atoms_dict:
        atoms.info = atoms_dict["info"]
    if "command" in atoms_dict:
        atoms.calc.command = atoms_dict["command"]
    if "results" in atoms_dict:
        atoms.calc.results = atoms_dict["results"]

    # attach calculator
    if atoms.calc:
        for key, val in atoms.calc.results.items():
            if isinstance(val, list):
                atoms.calc.results[key] = np.array(val)
        if "use_pimd_wrapper" in atoms.calc.parameters:
            pimd = atoms.calc.parameters["use_pimd_wrapper"]
            if isinstance(pimd, int):
                atoms.calc.parameters["use_pimd_wrapper"] = ("localhost", pimd)
    return atoms


def calc2dict(calculator):
    """Converts an ASE Calculator into a dict

    Parameters
    ----------
    calculator: ase.calculators.calulator.Calculator
        The calculator to convert to a dictionary

    Returns
    -------
    calc_dict: dict
        The corresponding dictionary
    """
    if calculator is None:
        return {}
    if isinstance(calculator, dict):
        return calculator
    calc_dict = {}
    calc_dict["calculator"] = calculator.name.lower()
    calc_dict["calculator_parameters"] = calculator.todict()
    try:
        calc_dict["command"] = calculator.command
    except AttributeError:
        pass
    calc_dict["results"] = calculator.results
    return calc_dict
