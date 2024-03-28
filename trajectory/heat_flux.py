"""compute and analyze heat fluxes"""
import numpy as np
from ase import units

from vibes.helpers import progressbar
from vibes.helpers.stresses import get_stresses

from . import Timer


def average_atomic_stresses(trajectory, verbose=True):
    """compute average atomic stress from trajectory"""
    atomic_stresses = trajectory.stresses
    return np.mean(atomic_stresses, axis=0)


def get_heat_flux(trajectory):
    """compute heat fluxes from TRAJECTORY and return as xarray

    Args:
        trajectory: list of atoms objects WITH ATOMIC STRESS COMPUTED

    Returns:
        flux ([N_t, N_a, 3]): the time resolved heat flux in eV/AA**3/ps
        avg_flux ([N_t, N_a, 3]): high frequency part of heat flux
    """
    msg = "Trajectory needs to have atomic stress computed, check!"
    assert len(trajectory) == len(trajectory.with_stresses), msg

    # 1) compute average stresses
    avg_stresses = average_atomic_stresses(trajectory)

    # 2) compute J_avg from average stresses and dJ from variance
    timer = Timer("Compute heat flux:")
    fluxes = []
    avg_fluxes = []
    times = trajectory.times
    times -= min(times)
    for a in progressbar(trajectory):
        stresses = get_stresses(a)

        ds = stresses - avg_stresses

        # velocity in \AA / ps
        vs = a.get_velocities() * units.fs * 1000

        fluxes.append((ds @ vs[:, :, None]))
        avg_fluxes.append((avg_stresses @ vs[:, :, None]))

    avg_flux = np.array(avg_fluxes).squeeze()
    flux = np.array(fluxes).squeeze()
    timer()

    return flux, avg_flux
