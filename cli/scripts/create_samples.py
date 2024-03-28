"""
Script to initialize positions and velocities with force constants.
Similar to canonical_sampling from TDEP.
"""
import numpy as np
from ase import units as u
from ase.io import read

import vibes.ase.md.velocitydistribution as vd
from vibes.harmonic_analysis.dynamical_matrix import get_frequencies
from vibes.helpers import talk
from vibes.structure.io import inform


def generate_samples(
    atoms,
    temperature=None,
    n_samples=None,
    force_constants=None,
    rattle=False,
    quantum=False,
    deterministic=False,
    zacharias=False,
    gauge_eigenvectors=False,
    ignore_negative=False,
    random_seed=None,
    propagate=None,
    failfast=True,
):
    """create samples for Monte Carlo sampling

    Args:
        atoms: the structure
        temperature: temperature in Kelvin
        n_samples: number of samples to create (default: 1)
        force_constants: numpy array w/ force constants
        rattle: use `atoms.rattle`
        quantum: use Bose-Einstein distribution instead of Maxwell-Boltzmann
        deterministic: create sample deterministically
        zacharias: use +/-
        gauge_eigenvectors: make largest entry positive
        ignore_negative: don't check for negative modes
        random_seed: seed for random number generator
        propagate: propagate atoms according to velocities for this many fs
    """

    inform(atoms, verbosity=0)

    seed = random_seed
    temp = temperature
    info_str = []

    if not rattle:
        if temp is None:
            exit("** temperature needs to be given")

    if not seed:
        seed = np.random.randint(2 ** 31)

    rng = np.random.RandomState(seed)

    if force_constants is not None:
        # if 3Nx3N shaped force_constants:
        if np.any(np.array(force_constants.shape, dtype=int) != 3 * len(atoms)):
            exit("other force constants not yet implemented")

        # Check dyn. matrix
        check_frequencies(atoms, force_constants)

        # collect arguments for PhononHarmonics
        phonon_harmonic_args = {
            "force_constants": force_constants,
            "quantum": quantum,
            "temp": temp * u.kB,
            "failfast": failfast,
            "rng": rng,
            "deterministic": deterministic,
            "plus_minus": zacharias,
            "gauge_eigenvectors": gauge_eigenvectors,
            "ignore_negative": ignore_negative,
        }

        info_str += ["created from force constants", f"T = {temp} K"]
        talk(f"Random seed: {seed}")
    else:
        mb_args = {"temp": temp * u.kB, "rng": rng}
        info_str += ["created from MB distrubtion", f"T = {temperature} K"]
        talk(f"Use Maxwell Boltzamnn to set up samples")

    info_str += [
        f"quantum:             {quantum}",
        f"deterministic:       {deterministic}",
        f"plus_minus:          {zacharias}",
        f"gauge_eigenvectors:  {gauge_eigenvectors or zacharias}",
        f"Random seed:         {seed}",
    ]

    sample_list = []

    for ii in range(n_samples):
        talk(f"Sample {ii:3d}:")
        sample_info_str = info_str + [f"Sample number:       {ii + 1}"]
        sample = atoms.copy()

        if force_constants is not None:
            vd.PhononHarmonics(sample, **phonon_harmonic_args)

        elif rattle is not None:
            sample.rattle(rattle)
            sample_info_str = info_str + [f"Rattle with stdev:   {rattle}"]

        else:
            vd.MaxwellBoltzmannDistribution(sample, **mb_args)

        if force_constants is not None:
            d = np.ravel(sample.positions - atoms.positions)
            epot_ha = 0.5 * (force_constants @ d @ d)
            epot_ha_temp = epot_ha / u.kB / len(atoms) / 3 * 2

            ha_epot_str = f"{epot_ha:9.3f}eV ({epot_ha_temp:.2f}K)"

            sample_info_str += [f"Harmonic E_pot:    {ha_epot_str}"]
            talk(f".. harmonic potential energy:   {ha_epot_str})")

        talk(f".. temperature before cleaning: {sample.get_temperature():9.3f}K")
        talk(f".. remove net momentum from sample and force temperature")
        vd.force_temperature(sample, temp)
        vd.Stationary(sample)
        # vd.ZeroRotation(sample)

        if propagate:
            talk(f".. propagate positions for {propagate} fs")
            sample_info_str += [f"Propagated for:      {propagate} fs"]
            sample.positions += sample.get_velocities() * propagate * u.fs

        sample.info["info_str"] = sample_info_str
        sample_list.append(sample)

        talk(f".. temperature in sample {ii}:     {sample.get_temperature():9.3f}K")

    return sample_list


def create_samples(atoms_file, temperature=None, fc_file=None, format="aims", **kwargs):
    """FileIO frontend to `generate_samples`

    Args:
        atoms_file: input geometry file
        temperature: temperature in Kelvin
        fc_file: file holding force constants
        format: The ASE file format for geometry files
        kwargs: kwargs for `generate_samples`
    """

    atoms = read(atoms_file, format=format)
    inform(atoms, verbosity=0)

    fc = None
    if fc_file is not None:
        # if 3Nx3N shaped txt file:
        try:
            fc = np.loadtxt(fc_file)
        except ValueError:
            exit("other force constants not yet implemented")
        talk(f"\nUse force constants from {fc_file} to prepare samples")

    sample_list = generate_samples(
        atoms, temperature=temperature, force_constants=fc, **kwargs
    )

    for ii, sample in enumerate(sample_list):
        talk(f"Sample {ii:3d}:")
        out_file = f"{atoms_file}.{int(temperature):04d}K"
        if len(sample_list) > 1:
            out_file += f".{ii:03d}"

        info_str = sample.info.pop("info_str")
        sample.write(out_file, info_str=info_str, velocities=True, format=format)
        talk(f".. temperature in sample {ii}:     {sample.get_temperature():9.3f}K")
        talk(f".. written to {out_file}")


def check_frequencies(atoms, force_constants):
    """print lowest and highest frequencies obtained from force constants"""
    w2 = get_frequencies(force_constants, masses=atoms.get_masses())

    print("The first 6 frequencies:")
    for ii, freq in enumerate(w2[:6]):
        print(f" {ii + 1:4d}: {freq}")

    print("Highest 6 frequencies")
    for ii, freq in enumerate(w2[:-7:-1]):
        print(f" {len(w2) - ii:4d}: {freq }")
