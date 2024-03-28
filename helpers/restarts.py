""" handling restarts of tasks and workflows """
import subprocess as sp
from os import path

from vibes.helpers import talk, warn
from vibes.settings import Settings


def restart(settings=None, trajectory_file=None, verbose=True):
    """ restart a job according to the restart instructions in the settings

    Args:
        settings (Settings): Settings for the task
        trajectory_file (Path): if given, check if the trajectory exists
        verbose (bool): If True print more logging information

    Returns:
        bool: True if restart was performed
    """
    _prefix = "restart"

    if settings is None:
        settings = Settings()

    if "restart" in settings:
        # check if trajectory exists
        if trajectory_file and not path.exists(trajectory_file):
            msg = "Computation restart request, but no trajectory found. CHECK!"
            warn(msg, level=2)
        if verbose:
            talk(f"Restart task with {settings.restart.command}", prefix=_prefix)
        sp.run(settings.restart.command.split(), stderr=sp.STDOUT)
        return True
    else:
        if verbose:
            talk("Task not completed, please inspect and rerun.", prefix=_prefix)
        return False
