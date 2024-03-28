""" vibes defaults for md"""
import collections

from vibes.helpers.dict import AttributeDict as adict
from vibes.keys import relaxation, relaxation_options  # noqa: F401
from vibes.konstanten import n_geom_digits, symprec

name = relaxation

mandatory_base = ["machine", "geometry", name]
mandatory_task = ["driver", "fmax"]


_keys = [
    "driver",
    "logfile",
    "unit_cell",
    "fmax",
    "maxstep",
    "hydrostatic_strain",
    "constant_volume",
    "scalar_pressure",
    "decimals",
    "fix_symmetry",
    "symprec",
    "workdir",
    "restart",
]
keys = collections.namedtuple("relaxation_keywords", _keys)(*_keys)

kwargs = adict(
    {
        keys.driver: "BFGS",
        keys.fmax: 0.001,
        keys.unit_cell: True,
        keys.fix_symmetry: False,
        keys.hydrostatic_strain: False,
        keys.constant_volume: False,
        keys.scalar_pressure: 0.0,
        keys.decimals: n_geom_digits,
        keys.symprec: symprec,
        keys.workdir: name,
        # kwargs go to Optimizer, e.g., BFGS(..., **kwargs)
        "kwargs": {
            keys.maxstep: 0.2,
            keys.logfile: "relaxation.log",
            keys.restart: "bfgs.restart",
        },
    }
)

settings_dict = {name: kwargs}
