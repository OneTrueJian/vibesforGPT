""" vibes defaults for phonopy """
import collections

from vibes.helpers.dict import AttributeDict as adict

displacement_id_str = "displacement_id"
name = "phonopy"

mandatory = {
    "mandatory_keys": ["machine", "control", "geometry"],
    "mandatory_obj_keys": ["supercell_matrix"],
}

_keys = [
    "supercell_matrix",
    "displacement",
    "symprec",
    "is_diagonal",
    "is_plusminus",
    "q_mesh",
    "workdir",
]
keys = collections.namedtuple(f"{name}_keywords", _keys)(*_keys)

kwargs = adict(
    {
        keys.supercell_matrix: [1, 1, 1],
        keys.displacement: 0.01,
        keys.is_diagonal: False,
        keys.is_plusminus: "auto",
        keys.symprec: 1e-5,
        keys.q_mesh: [45, 45, 45],
        keys.workdir: name,
    }
)

settings_dict = {name: kwargs}
