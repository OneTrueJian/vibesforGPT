""" tools for converting atoms objects to json representations """


import json
from pathlib import Path, PosixPath
from typing import Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, get_calculator_class
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import dict2constraint, voigt_6_to_full_3x3_stress
from ase.db.row import atoms2dict as ase_atoms2dict
from ase.io.jsonio import MyEncoder

from vibes import keys
from vibes.helpers.lists import expand_list, list_dim, reduce_list
from vibes.konstanten import n_yaml_digits


key_symbols = "symbols"
key_masses = "masses"
key_constraints = "constraint"


class NumpyEncoder(json.JSONEncoder):
    """Decode numerical objects that json cannot parse by default"""

    def default(self, obj) -> str:
        """Default JSON encoding"""
        if hasattr(obj, "tolist") and callable(obj.tolist):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, complex):
            return (float(obj.real), float(obj.imag))
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, PosixPath):
            return str(obj)

        return super().default(obj)


def atoms2dict(
    atoms: Atoms, reduce: bool = True, add_constraints: bool = False
) -> dict:
    """Converts an Atoms object into a dict

    Args:
        atoms: The structure to be converted to a dict
        reduce: use reduced representation of `symbols` and `masses`

    Returns:
        atoms_dict: The dict representation of atoms
    """

    atoms_dict = {}

    # if periodic system, append lattice (before positions)
    if any(atoms.pbc):
        atoms_dict.update({"pbc": np.asarray(atoms.pbc)})
        atoms_dict.update({"cell": atoms.cell.tolist()})

    atoms_dict.update({"positions": atoms.positions.tolist()})

    if atoms.get_velocities() is not None:
        atoms_dict.update({"velocities": atoms.get_velocities().tolist()})

    # append symbols and masses
    atoms_dict.update(
        {
            key_symbols: reduce_list(atoms.get_chemical_symbols(), reduce=reduce),
            key_masses: reduce_list(atoms.get_masses(), reduce=reduce),
        }
    )

    atoms_dict.update({"info": atoms.info})

    if add_constraints:
        atoms_dict[key_constraints] = [constr.todict() for constr in atoms.constraints]

    return atoms_dict


def calc2dict(calculator: SinglePointCalculator) -> dict:
    """Converts an ase calculator calc into a dict

    Args:
        calculator: The calculator to be converted to a dict

    Returns:
        calculator_dict: The dict representation of calculator
    """

    if calculator is None:
        return {}
    if isinstance(calculator, dict):
        return calculator

    params = calculator.parameters
    for key, val in params.items():
        if isinstance(val, tuple):
            params[key] = list(val)

    calculator_dict = {
        "calculator": calculator.__class__.__name__,
        "calculator_parameters": params,
    }
    if hasattr(calculator_dict, "command"):
        calculator_dict.update({"command": calculator.command})

    return calculator_dict


def input2dict(
    atoms: Atoms,
    calculator: SinglePointCalculator = None,
    primitive: Atoms = None,
    supercell: Atoms = None,
    settings: dict = None,
) -> dict:
    """Convert metadata information to plain dict

    Args:
        atoms: The structure to be converted to a dict
        calculator: The calculator to be converted to a dict
        primitive: The primitive cell structure
        supercell: The supercell cell structure
        settings: The settings used to generate the inputs

    Returns:
    input_dict: The dictionary representation of the inputs with the following items

        calculator: The calc_dict of the inputs
        atoms: The atoms_dict of the inputs
        primitive: The dict representation of the primitive cell structure
        supercell: The dict representation of the supercell cell structure
        settings: The dict representation of the settings used to generate the inputs
    """
    # structure
    atoms_dict = atoms2dict(atoms)

    # calculator
    if calculator is None:
        calculator = atoms.calc

    calculator_dict = calc2dict(calculator)

    input_dict = {"calculator": calculator_dict, "atoms": atoms_dict}

    if primitive:
        input_dict.update({"primitive": atoms2dict(primitive)})

    if supercell:
        input_dict.update({"supercell": atoms2dict(supercell)})

    # save the configuration
    if settings:
        settings_dict = settings.to_dict()

        input_dict.update({"settings": settings_dict})

    return input_dict


def results2dict(atoms: Atoms, calculator: SinglePointCalculator = None) -> dict:
    """extract information from atoms and calculator and convert to plain dict

    Args:
        atoms: The structure to be converted to a dict
        calculator: The calculator to be converted to a dict

    Returns:
        dict: dictionary with items
            calculator: calculator_dict of the inputs
            atoms: atoms_dict of the inputs
    """
    atoms_dict = atoms2dict(atoms)

    # calculated values
    calculator_dict = {}

    if calculator is None:
        calculator = atoms.calc

    # convert stress to 3x3 if present
    if keys.stress in calculator.results:
        stress = calculator.results[keys.stress]
        if len(stress) == 6:
            calculator.results[keys.stress] = voigt_6_to_full_3x3_stress(stress)
    # convert stresses to Nx3x3 if present
    if keys.stresses in calculator.results:
        stresses = calculator.results[keys.stresses]
        if len(stresses[0]) == 6:
            calculator.results[keys.stresses] = voigt_6_to_full_3x3_stress(stresses)

    # convert numpy arrays into ordinary lists
    for key, val in calculator.results.items():
        if isinstance(val, np.ndarray):
            calculator_dict[key] = val.tolist()
        elif isinstance(val, (float, np.double)):
            calculator_dict[key] = float(val)
        else:
            calculator_dict[key] = val

    if not calculator_dict:
        raise RuntimeError("calculator_dict is empty, was the calculation successful?")

    return {"atoms": atoms_dict, "calculator": calculator_dict}


def dict2atoms(
    atoms_dict: dict, calculator_dict: dict = None, single_point_calculator: bool = True
) -> Atoms:
    """Convert dictionaries into atoms and calculator objects

    Args:
        atoms_dict: The dict representation of atoms
        calculator_dict: The dict representation of calc
        single_point_calculator: return calculator as SinglePointCalculator

    Returns:
        atoms: atoms represented by atoms_dict with calculator represented by calc_dict
    """
    if "pbc" not in atoms_dict:
        atoms_dict.update({"pbc": "cell" in atoms_dict})

    uuid = atoms_dict.pop("unique_id", None)

    try:
        velocities = atoms_dict.pop("velocities")
    except KeyError:
        velocities = None

    if key_masses in atoms_dict:
        atoms_dict[key_masses] = expand_list(atoms_dict[key_masses])
    if key_symbols in atoms_dict:
        atoms_dict[key_symbols] = expand_list(atoms_dict[key_symbols])

    if key_constraints in atoms_dict:
        atoms_dict[key_constraints] = [
            dict2constraint(d) for d in atoms_dict[key_constraints]
        ]

    atoms = Atoms(**atoms_dict)

    if velocities is not None:
        atoms.set_velocities(velocities)

    # Calculator
    if calculator_dict is not None:
        results = {}
        if "results" in calculator_dict:
            results = calculator_dict.pop("results")
        if single_point_calculator:
            calculator = results2singlepoint(
                atoms=atoms, results=results, calculator_dict=calculator_dict
            )

        else:
            calculator = get_calculator_class(calculator_dict["calculator"].lower())(
                **calculator_dict["calculator_parameters"]
            )
            calculator.results = results
        if "command" in calculator_dict:
            calculator.command = calculator_dict["command"]
    else:
        calculator = None

    atoms.calc = calculator
    if "info" in atoms_dict:
        atoms.info = atoms_dict["info"]
    else:
        atoms.info = {}
    atoms.info["unique_id"] = uuid
    return atoms


def results2singlepoint(results=None, atoms=None, calculator_dict=None):
    """Make SinglePointCalculator with non-ase properties"""

    if results is None:
        results = {}
    if calculator_dict is None:
        calculator_dict = {}

    # remove properties not allowed by ase so the SinglePointCalculator
    # doesn't complain
    additional_properties = [
        prop for prop in results.keys() if prop not in all_properties
    ]
    additional_results = {p: results.pop(p) for p in additional_properties}

    calculator = SinglePointCalculator(atoms, **results)
    if "calculator" in calculator_dict:
        calculator.name = calculator_dict["calculator"].lower()
    if "calculator_parameters" in calculator_dict:
        calculator.parameters.update(calculator_dict["calculator_parameters"])

    # restore properties that are not in ase's all_properties
    # (will be removed once ase allows it)
    for prop, result in additional_results.items():
        calculator.results[prop] = result

    return calculator


def dict2json(dct: dict, indent: int = 0, outer: bool = True) -> str:
    """convert python dictionary with scientific data to JSON

    Args:
        dct: Dictionary to convert to JSONAble format
        indent: indentation for the json string
        outer: if True add outer { } to the string

    Returns:
        json string of the dict
    """

    parts = []
    ind = indent * " "

    for key, val in dct.items():
        # convert np.ndarrays etc to lists
        if hasattr(val, "tolist") and callable(val.tolist):
            val = val.tolist()

        if isinstance(val, str):
            rep = f'"{val}"'
        elif isinstance(val, (float, np.double)):
            rep = "{1: .{0}e}".format(n_yaml_digits, val)
        elif isinstance(val, dict):
            # recursive formatting
            rep = f"{{\n{dict2json(val, 2*(1 + indent // 2), False)}}}"
        elif val == []:  # empty list
            rep = json.dumps(val, cls=NumpyEncoder)
        elif (
            isinstance(val, list)
            and len(list_dim(val)) == 2
            and list_dim(val)[1] == 3
            and isinstance(val[0][0], float)
        ):
            # this is most likely positions, velocities, forces, etc. -> format!
            rep = [
                " [{1: .{0}e}, {2: .{0}e}, {3: .{0}e}]".format(n_yaml_digits, *elem)
                for elem in val
            ]
            # join to have a comma separated list
            rep = f",\n{2*ind}".join(rep)
            # add leading [ and trailing ]
            rep = f"\n{2*ind}[{rep[1:]}"
            rep += "]"
        elif (
            isinstance(val, list)
            and len(list_dim(val)) == 3
            and list_dim(val)[1:3] == [3, 3]
            and isinstance(val[0][0][0], float)
        ):
            # this is most likely atomic stress -> format!
            rep = [
                "["
                + "[{1: .{0}e}, {2: .{0}e}, {3: .{0}e}]".format(n_yaml_digits, *elem[0])
                + f",\n{2*ind} "
                + "[{1: .{0}e}, {2: .{0}e}, {3: .{0}e}]".format(n_yaml_digits, *elem[1])
                + f",\n{2*ind} "
                + "[{1: .{0}e}, {2: .{0}e}, {3: .{0}e}]".format(n_yaml_digits, *elem[2])
                + "]"
                for elem in val
            ]
            # join to have a comma separated list
            rep = f",\n{2*ind}".join(rep)
            # add leading [ and trailing ]
            rep = f"\n{(2*ind)[:-1]}[{rep}"
            rep += "]"

        else:
            rep = json.dumps(val, cls=NumpyEncoder)

        parts.append(f'{ind}"{key}": {rep}')

    rep = ",\n".join(parts)

    if outer:
        rep = f"{{{rep}}}"

    # make sure only " are used to be understood by JSON
    return rep.replace("'", '"')


def get_json(obj) -> str:
    """Return json representation of obj

    Args:
        obj: Object to convert to json

    Returns:
        json string of obj
    """
    return json.dumps(obj, cls=MyEncoder, sort_keys=True)


def atoms2json(atoms, reduce=True):
    """return json representation of Atoms"""
    rep = dict2json(atoms2dict(atoms, reduce=reduce))
    return rep


def json2atoms(rep):
    """return Atoms from json string"""
    atoms = dict2atoms(json.loads(rep))
    return atoms


def atoms_calc2json(
    atoms: Atoms,
    ignore_results: bool = False,
    ignore_keys: Sequence[str] = ["unique_id"],
    ignore_calc_params: Sequence[str] = [],
) -> Sequence[str]:
    """Return json representation of atoms and calculator objects.

    Includes possibility to remove certain keys from the atoms dictionary,
    e.g. for hashing

    Args:
        atoms: The structure to be converted to a json with attached calculator
        ignore_results: If True ignore the results in atoms.calc
        ignore_keys: Ignore keys in this list
        ignore_calc_params: Ignore calculator keys in this list

    Returns:
        atoms_json: Json representation of the atoms dictionary
        calc_json: Json representation of the calculator dictionary
    """

    # dictionary contains all the information in atoms object
    atomsdict = ase_atoms2dict(atoms)

    # remove unwanted keys from atomsdict
    for name in ignore_keys:
        if name in atomsdict:
            atomsdict.pop(name)

    calcdict = {}

    # move physical properties from atomsdict to calcdict if they are wanted
    for key in all_properties:
        if key in atomsdict:
            value = atomsdict.pop(key)
            if not ignore_results:
                calcdict[key] = value

    # clean calculator entries
    if "calculator_parameters" in atomsdict:
        calculator_params = atomsdict["calculator_parameters"]
        for name in [key for key in calculator_params if key in ignore_calc_params]:
            calculator_params.pop(name)

        if "species_dir" in calculator_params:
            calculator_params["species_dir"] = Path(
                calculator_params["species_dir"]
            ).parts[-1]

    for name in ["calculator", "calculator_parameters"]:
        if name in atomsdict:
            calcdict[name] = atomsdict.pop(name)

    return get_json(atomsdict), get_json(calcdict)
