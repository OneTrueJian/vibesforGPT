""" vibes defaults for aims"""

from vibes.helpers import talk as _talk
from vibes.keys import calculator as name


def talk(msg, verbose=True):
    """wrapper for helpers.talk with 'aims' prefix"""
    return _talk(msg, prefix=name, verbose=verbose)
