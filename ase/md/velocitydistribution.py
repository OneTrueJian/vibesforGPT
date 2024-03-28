"""Modify routines from `ase.md.velocitydistribution`"""
from typing import Sequence

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import *  # noqa: F403

from vibes.helpers import warn


def phonon_harmonics(
    force_constants: np.ndarray,
    masses: Sequence[float],
    temp: float,
    rng: np.random.RandomState = np.random,
    quantum: bool = False,
    deterministic: bool = False,
    plus_minus: bool = False,
    gauge_eigenvectors: bool = False,
    return_eigensolution: bool = False,
    failfast: bool = True,
    ignore_negative: bool = True,
) -> Sequence[np.ndarray]:
    r"""Return displacements and velocities that produce a given temperature.

    References:
        [West, Estreicher; PRL 96, 22 (2006)]
        [Zacharias, Giustino; PRB 94, 075125 (2016)]

    Args:
        force_constants: force constants (Hessian) of the system in eV/Å²
        masses: masses of the structure in amu
        temp: Temperature converted to eV (T * units.kB)
        rng: RandomState (or similar) with `rng.rand` function
        quantum: True for Bose-Einstein, False for Maxwell-Boltzmann
        deterministic: Displace atoms deterministically by fixing the phases
        plus_minus: Displace atoms with +/- the amplitude accoding to PRB 94, 075125
        gauge_eigenvectors: gauge eigenvectors such that largest entry is positive
        return_eigensolution: return eigenvalues and eigenvectors of the dynamical matrix
        failfast: If True, raise Error when phonon spectrum is not positive
        ignore_negative: If True freeze out the imaginary modes

    Returns:
        displacements, velocities generated from the eigenmodes,
        (optional: eigenvalues, eigenvectors of dynamical matrix)

    """
    # Build dynamical matrix
    rminv = (masses ** -0.5).repeat(3)
    dynamical_matrix = force_constants * rminv[:, None] * rminv[None, :]

    # Solve eigenvalue problem to compute phonon spectrum and eigenvectors
    w2_s, X_is = np.linalg.eigh(dynamical_matrix)

    # First three modes are translational so ignore:
    last_ignore_mode = 3
    # if ignore_negative ignore all modes with frequency < 0.0 threshold (1e-3)
    if ignore_negative:
        last_ignore_mode = max(len(np.where(w2_s <= 0)[0]), 3)

    # Check for soft modes
    w2min = w2_s[last_ignore_mode:].min()
    if w2min < 0:
        msg = f"Spectrum not positive, e.g. {w2min}. Use `ignore_negative`."
        if failfast:
            raise ValueError(msg)
        else:
            warn(msg, level=1)

    zeros = w2_s[last_ignore_mode - 3 : last_ignore_mode]
    worst_zero = np.abs(zeros).max()
    if worst_zero > 1e-3:
        msg = "Translational deviate from 0 significantly: {}".format(w2_s[:3])
        warn(msg, level=1)

    nw = len(w2_s) - last_ignore_mode
    n_atoms = len(masses)

    w_s = np.sqrt(w2_s[last_ignore_mode:])
    X_acs = X_is[:, last_ignore_mode:].reshape(n_atoms, 3, nw)

    # Assign the amplitudes according to Bose-Einstein distribution
    # or high temperature (== classical) limit
    if quantum:
        hbar = units._hbar * units.J * units.s
        A_s = np.sqrt(hbar * (2 * n_BE(temp, hbar * w_s) + 1) / (2 * w_s))  # noqa: F405
    else:
        A_s = np.sqrt(temp) / w_s

    if ignore_negative:
        A_s[np.where(np.isnan(w_s))[0]] = 0.0
        w_s[np.where(np.isnan(w_s))[0]] = 0.0

    if plus_minus or gauge_eigenvectors:
        # gauge eigenvectors: largest value always positive
        for ii in range(X_acs.shape[-1]):
            vec = X_acs[:, :, ii]
            max_arg = np.argmax(abs(vec))
            X_acs[:, :, ii] *= np.sign(vec.flat[max_arg])

    if deterministic or plus_minus:
        # create samples by multiplying the amplitude with +/-
        # according to Eq. 5 in PRB 94, 075125

        if plus_minus:
            spread = (-1) ** np.arange(nw)
        else:
            spread = np.ones(nw)

        # Create velocities und displacements from the amplitudes and
        # eigenvectors
        A_s *= spread
        phi_s = 2.0 * np.pi * rng.rand(nw)

        # Assign velocities, sqrt(2) compensates for missing sin(phi) in
        # amplitude for displacement
        v_ac = (w_s * A_s * np.sqrt(2) * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]

        # Assign displacements
        d_ac = (A_s * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]

    else:
        # compute the gaussian distribution for the amplitudes
        # We need 0 < P <= 1.0 and not 0 0 <= P < 1.0 for the logarithm
        # to avoid (highly improbable) NaN.

        # Box Muller [en.wikipedia.org/wiki/Box–Muller_transform]:
        spread = np.sqrt(-2.0 * np.log(1.0 - rng.rand(nw)))

        # assign amplitudes and phases
        A_s *= spread
        phi_s = 2.0 * np.pi * rng.rand(nw)

        # Assign velocities and displacements
        v_ac = (w_s * A_s * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]

        d_ac = (A_s * np.sin(phi_s) * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]

    if return_eigensolution:
        return d_ac, v_ac, w2_s, X_is
    # else
    return d_ac, v_ac


def PhononHarmonics(
    atoms: Atoms, force_constants: np.ndarray, temp: float, **kwargs
) -> None:
    r"""Excite phonon modes to specified temperature.

    This will displace atomic positions and set the velocities so as
    to produce a random, phononically correct state with the requested
    temperature.

    Args:
        atoms: the structure
        force_constants: Force constants in eV/Å²
        temp: Temperature in eV (T * units.kB)
        kwargs: kwargs for `phonon_harmonics`

    """

    # Receive displacements and velocities from phonon_harmonics()
    d_ac, v_ac = phonon_harmonics(
        force_constants=force_constants,
        masses=atoms.get_masses(),
        temp=temp,
        return_eigensolution=False,
        **kwargs,
    )

    # Assign new positions (with displacements) and velocities
    atoms.positions += d_ac
    atoms.set_velocities(v_ac)
