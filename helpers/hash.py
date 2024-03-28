""" Tools for hashing atoms objects """

from configparser import ConfigParser
from hashlib import sha1 as hash_sha
from json import dumps
from pathlib import Path


def hashfunc(string, empty_str="", digest=True):
    """Wrap the sha hash function and check for empty objects

    Parameters
    ----------
    string: str
        string to hash
    emptystr: str
        What an empy string should map to
    digest: bool
        If True digest the hash

    Returns
    -------
    str
        Hash of string
    """
    if string in ("", "[]", "{}", "None"):
        string = empty_str

    if isinstance(string, bytes):
        string = string
    else:
        string = string.encode("utf8")

    if digest:
        return hash_sha(string).hexdigest()

    return hash_sha(string)


def hash_file(file, **kwargs):
    """wrapper for hashfunc when file should be hashed"""
    return hashfunc(open(file, mode="rb").read(), **kwargs)


def hash_atoms(atoms, velocities=False):
    """hash the atoms object as it would be written to trajectory

    Args:
        atoms: Atoms to hash
        velocities: hash velocities as well

    Returns:
        atoms_hash: The hash of the atoms Object
    """
    from .converters import atoms2dict

    a = atoms.copy()
    a.info = {}

    atoms_hash = hash_dict(atoms2dict(a), velocities=velocities)

    return atoms_hash


def hash_dict(dct, velocities=False, ignore_keys=None):
    """hash a dictionary as it would be written to trajectory

    Replace full `species_dir` by only the last bit

    Args:
        dct: dictionary to hash
        velocities: hash velocities

    Returns:
        hash: The hash of the dictionary
    """
    from .converters import dict2json

    if ignore_keys is None:
        ignore_keys = []

    dct = dct.copy()

    dct.pop("info", None)
    for key in ignore_keys:
        dct.pop(key, None)
    # legacy: check for species_dir
    k1 = "calculator_parameters"
    k2 = "species_dir"
    if k1 in dct:
        if k2 in dct[k1]:
            dct[k1][k2] = Path(dct[k1][k2]).parts[-1]

    if not velocities:
        dct.pop("velocities", None)

    rep = dict2json(dct)
    return hashfunc(rep)


def hash_atoms_and_calc(
    atoms,
    ignore_results=True,
    ignore_keys=["unique_id", "info"],
    ignore_calc_params=[],
    ignore_file=None,
):
    """Hash atoms and calculator object, with possible ignores

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to be converted to a json with attached calculator
    ignore_results: bool
        If True ignore the results in atoms.calc
    ignore_keys: list of str
        Ignore all keys in this list
    ignore_calc_params: list of str
        Ignore all keys in this list that represent calculator parameters
    ignore_file: str
        Path to a file with standard keys to ignore

    Returns
    -------
    atomshash: str
        hash of the atoms
    calchash: str
        hash of atoms.calc
    """
    from .converters import atoms_calc2json

    if ignore_file is not None:
        fil = Path(ignore_file)
        if fil.exists():
            configparser = ConfigParser()
            configparser.read(fil)
            ignores = configparser["hash_ignore"]

            ignore_keys += [key for key in ignores if not ignores.getboolean(key)]

            ignore_calc_params = [key for key in ignores if not ignores.getboolean(key)]

    atomsjson, calcjson = atoms_calc2json(
        atoms, ignore_results, ignore_keys, ignore_calc_params
    )

    atomshash = hashfunc(atomsjson)
    calchash = hashfunc(calcjson)

    return atomshash, calchash


def hash_trajectory(calculated_atoms, metadata, hash_meta=False):
    """hash of a trajectory file

    Parameters
    ----------
    calculated_atoms: list of ase.atoms.Atoms
        Atoms objects inside a trajectory file
    metadata: dict
        Metadata for the trajectory
    hash_meta: bool
        if True hash the metadata

    Returns
    -------
    atomshash: str
        hash of all of the elements calculated_atoms
    metahash: str
        hash of the metadata
    """
    from .converters import atoms_calc2json

    calculated_atoms_dct = [atoms_calc2json(at) for at in calculated_atoms]
    dct = dict(metadata, calculated_atoms=calculated_atoms_dct)
    if hash_meta:
        return hashfunc(dumps(dct)), hashfunc(dumps(metadata))
    return hashfunc(dumps(dct))
