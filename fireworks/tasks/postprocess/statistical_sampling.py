"""Processing statistical sampling calculations"""
from pathlib import Path

import numpy as np
from vibes import anharmonicity_score
from vibes.trajectory import reader


def get_sigma(trajectory_file, return_trajectory=False):
    """Get the sigma value for all temperatures in a sampling trajectory.son file

    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file
    return_trajectory : bool
        Return the trajectory and metadata

    Returns
    -------
    np.ndarray
        array of temperatures and sigma for each temperature
    Trajectory
        The trajectory for the statistical sampling
    dict
        The metadata of the trajectory
    """
    trajectory, meta = reader(file=trajectory_file, get_metadata=True, verbose=False)

    forces_dft = {}
    forces_harmonic = {}

    for ii, sc in enumerate(trajectory):
        temp = float(sc.info["info_str"][1].split("T = ")[1].split(" K")[0])
        if temp not in forces_dft:
            forces_dft[temp] = []
            forces_harmonic[temp] = []
        forces_dft[temp] += list(trajectory.forces[ii].flatten())
        forces_harmonic[temp] += list(trajectory.forces_harmonic[ii].flatten())

    sigma = []
    temp = []

    for key in forces_dft.keys():
        temp.append(key)

        dft = np.array(forces_dft[key])
        ha = np.array(forces_harmonic[key])
        sigma.append(anharmonicity_score.get_sigma(dft, ha))

    with open(f"{Path(trajectory_file).parents[0]}/sigma.dat", "w") as f:
        for t, r in zip(temp, sigma):
            f.write(f"{t}, {r}\n")

    if return_trajectory:
        return sigma, trajectory, meta

    return sigma
