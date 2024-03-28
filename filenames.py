"""default filenames"""
import collections
from pathlib import Path

from vibes import keys
from vibes._defaults import DEFAULT_GEOMETRY_FILE

_filename = Path(__file__).stem

_geometry = DEFAULT_GEOMETRY_FILE

_suffixes = "suffixes"
_dct = {
    "next_step": ".next_step",
    "tdep_fc": ".forceconstant",
    "tdep_fc_remapped": ".forceconstant_remapped",
}
suffixes = collections.namedtuple("suffixes", _dct.keys())(**_dct)


# output file names
_output = "output"
_dct = {"aims": "aims.out"}
output = collections.namedtuple(_output, _dct.keys())(**_dct)

# force constant names
_fc = "fc"
_dct = {
    "phonopy": "FORCE_CONSTANTS",
    "phonopy_hdf5": "fc2.hdf5",
    "phonopy_remapped": "FORCE_CONSTANTS_remapped",
}
fc = collections.namedtuple(_fc, _dct.keys())(**_dct)


_dct = {
    "atoms": _geometry,
    "atoms_next": _geometry + suffixes.next_step,
    "primitive": _geometry + ".primitive",
    "supercell": _geometry + ".supercell",
    "trajectory": keys.trajectory + ".son",
    "trajectory_dataset": keys.trajectory + ".nc",
    "deformation": "deformation.dat",
    _output: output,
    _fc: fc,
    _suffixes: suffixes,
}
filenames = collections.namedtuple(_filename, _dct.keys())(**_dct)
