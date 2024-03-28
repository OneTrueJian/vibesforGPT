""" vibes defaults for md"""
import collections

from vibes.helpers import talk as _talk
from vibes.helpers.dict import AttributeDict as adict


name = "md"

_keys = [
    "driver",
    "timestep",
    "temperature",
    "maxsteps",
    "compute_stresses",
    "logfile",
    "friction",
    # NPTBerendsen
    "taut",
    "taup",
    "pressure",
    "compressibility",
    "workdir",
    "inhomogeneous",
]
keys = collections.namedtuple("md_keywords", _keys)(*_keys)

# Verlet
base_dict = adict(
    {
        keys.driver: "VelocityVerlet",
        keys.timestep: 1,
        keys.maxsteps: 1000,
        keys.compute_stresses: False,
        keys.workdir: name,
        # kwargs go to Dynamics, e.g., Langeving(..., **kwargs)
        "kwargs": {keys.logfile: "md.log"},
    }
)
nve_dict = {name: base_dict.copy()}

# Langevin
kwargs_nvt = {**base_dict}
kwargs_nvt[keys.driver] = "Langevin"
kwargs_nvt["kwargs"] = {
    keys.temperature: 300,
    keys.friction: 0.02,
    **base_dict["kwargs"],
}
nvt_dict = {name: kwargs_nvt}

# Berendsen
kwargs_npt = {**base_dict}
kwargs_npt[keys.driver] = "NPTBerendsen"
kwargs_npt["kwargs"] = {
    keys.temperature: 300,
    keys.taut: 0.5e3,  # * units.fs,
    keys.taup: 1e3,  # * units.fs,
    keys.pressure: 1.01325,  # in bar
    keys.compressibility: 4.57e-5,  # in bar^-1
    keys.inhomogeneous: False,  # use Inhomogeneous_NPTBerendsen?
    **base_dict["kwargs"],
}
npt_dict = {name: kwargs_npt}


calculation_timeout = 30 * 60  # 30 minutes


def talk(msg, verbose=True):
    return _talk(msg, verbose=verbose, prefix=name)
