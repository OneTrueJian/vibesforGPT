"""Wrap son to provide pretty formatted json"""

import son

from vibes.helpers.converters import dict2json
from vibes.helpers import warn


def dump(*args, **kwargs):
    """wrapper for son.dump"""
    return son.dump(*args, **{"dumper": dict2json, **kwargs})


def load(*args, **kwargs):
    """wrapper for son.load"""
    return son.load(*args, **kwargs)


def open(*args, **kwargs):
    """wrapper for son.open"""
    return son.open(*args, **kwargs)


def last_from(file, allow_empty=False):
    """return last entry from son file

    Parameters
    ----------
    file: str
        Path to file to load

    Returns
    -------
    data[-1]: dict
        Last entry in the son file
    """

    _, data = son.load_last(file)
    if not allow_empty and data is None:
        warn(
            f"** trajectory lacking the first step, please CHECK!",
            level=2,
        )

    return data
