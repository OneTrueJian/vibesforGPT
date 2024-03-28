"""Post processing for FHI-aims calculations"""

import numpy as np


def check_completion(workdir, fmax=0.00):
    """Check if MD completed"""
    relax_sum = np.loadtxt(f"{workdir}/relaxation.log", usecols=(-1,), skiprows=1)

    try:
        relax_sum = relax_sum[np.where(np.isfinite(relax_sum))[0]]
    except IndexError:
        relax_sum = [float(relax_sum)]

    return relax_sum[-1] < fmax
