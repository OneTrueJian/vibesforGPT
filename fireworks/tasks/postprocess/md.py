"""MD postprocessing jobs"""
import numpy as np


def check_completion(workdir, n_steps=-1):
    """Check if MD completed"""
    md_sum = np.genfromtxt(f"{workdir}/md.log", usecols=(0,))
    md_sum = md_sum[np.where(np.isfinite(md_sum))[0]]
    return len(md_sum) > n_steps
