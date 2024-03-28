"""Functions that generate FWActions after performing Aims Calculations"""
from pathlib import Path

import numpy as np

from vibes.filenames import filenames
from vibes.fireworks.tasks.calculate_wrapper import T_S_LINE


def get_calc_times(workdir=".", calc_dirs=None):
    """Get the calculation for a set of Aims Calculations

    Parameters
    ----------
    workdir : str
        working directory with the calculation in it (Default value = ".")
    calc_dirs : list of str
        list of directories where calculations were done (Default value = None)

    Returns
    -------
    calc_times : list of floats
        the wall clock time of all calculations

    """

    if calc_dirs is None:
        workdir = Path(workdir)
        calc_dirs = list(workdir.glob("*/")) + list(workdir.glob("*/*/"))

    calc_times = []

    for direc in calc_dirs:
        aims_out = Path(direc) / filenames.output.aims
        if aims_out.exists():
            lines = np.array(open(str(aims_out)).readlines())
            time_line = np.where(lines == T_S_LINE)[0][0] + 1
            time = float(lines[time_line].split(":")[1].split("s")[0])
            calc_times.append(time)

    return calc_times
