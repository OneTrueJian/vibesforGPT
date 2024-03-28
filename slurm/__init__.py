from vibes.helpers import talk


prefix = "slurm"


def _talk(msg, verbose=True):
    return talk(msg, prefix=prefix, verbose=verbose)
