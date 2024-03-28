"""Functions to calculate anharmonicity scores as described in
    (future reference)"""
import numpy as np
import xarray as xr

from vibes import keys
from vibes.helpers import progressbar, warn
from vibes.spglib.wrapper import get_symmetry_dataset


def _check_shape(f1, f2):
    """check if sizes of input data are compatible"""

    assert np.shape(f1) == np.shape(f2), (
        "Check shape of input arrays!: ",
        f1.shape,
        f2.shape,
    )


def get_sigma(f_data, f_model, rtol=1e-5, axis=None, silent=False):
    """Calculate RMSE / STD

    Args:
        f_data (np.ndarray): input data
        f_model (np.ndarray): input model data
        rtol (float): assert f.mean() / f.std() < rtol
        axis (tuple): axis along which mean and std are taken
        silent (bool): silence warnings
    """
    f1 = np.asarray(f_data)
    f2 = np.asarray(f_model)
    _check_shape(f1, f2)

    # check f_data_mean, should be small
    if np.any(f1.mean(axis=axis) > rtol * f1.std(axis=axis)) and not silent:
        warn(f"f_data.mean(axis=axis) is {f1.mean(axis=axis)}", level=1)

    rmse = (f1 - f2).std(axis=axis)
    std = f1.std(axis=axis)

    return rmse / std


def get_r2(in_f_data, in_f_model, mean=True, silent=False):
    r"""Calculate coefficient of determination between f_data and f_model

    Refrence Website
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions

    Args:
        in_f_data: input data
        in_f_model: input model data
        mean (bool): take out mean of f_data
        silent (bool): make silent

    Returns:
        float: Coefficient of Determination
    """

    f_data = np.ravel(in_f_data)
    f_model = np.ravel(in_f_model)

    _check_shape(f_data, f_model)

    f_data_mean = np.mean(f_data, axis=0)

    # check f_data_mean, should be small
    if f_data_mean > 1e-1 and not silent:
        warn(f"|f_data.mean()| is {f_data_mean}", level=1)

    Sres = (f_data - f_model) @ (f_data - f_model)

    if mean:
        Stot = (f_data - f_data_mean) @ (f_data - f_data_mean)
    else:
        Stot = (f_data) @ (f_data)

    return 1 - Sres / Stot


def get_sigma_per_atom(
    forces_dft, forces_harmonic, ref_structure, reduce_by_symmetry=False
):
    """Compute sigma score per atom in primitive cell. Optionally use symmetry.

    Args:
        forces_dft: forces from dft calculations in shape [Nt, Na, 3]
        forces_harmonic: forces from harmonic approximation in shape [Nt, Na, 3]
        ref_structure: reference structure for symmetry analysis
        reduce_by_symmetry: project on symmetry equivalent instead of primitive

    Returns:
        unique_atoms: the atoms from ref_structure for which r^2 was computed
        sigma_per_atom: sigma score for atoms in unique_atoms
    """

    sds = get_symmetry_dataset(ref_structure)

    # check shape:
    if len(np.shape(forces_dft)) == 2:
        forces_dft = np.expand_dims(forces_dft, axis=0)
        forces_harmonic = np.expand_dims(forces_harmonic, axis=0)

    if reduce_by_symmetry:
        compare_to = sds.equivalent_atoms
    else:
        compare_to = ref_structure.numbers

    if np.shape(forces_dft)[1] == 3 * len(ref_structure):
        compare_to = compare_to.repeat(3)

    symbols = np.array(ref_structure.get_chemical_symbols())
    unique_atoms, counts = np.unique(compare_to, return_counts=True)

    sigma_atom = []
    f_norms = []
    unique_symbols = []
    for u in unique_atoms:
        # which atoms belong to the prototype?
        mask = compare_to == u
        unique_symbols.append(symbols[mask][0])
        # take the forces that belong to this atom
        f_dft = forces_dft[:, mask]
        f_ha = forces_harmonic[:, mask]
        # compute r^2
        sigma_atom.append(get_sigma(f_dft, f_ha))
        f_norms.append(float(f_dft.std()))

    # compute weighted mean
    mean = (np.array(sigma_atom) @ counts) / counts.sum()

    attrs = {
        "unique_atoms": unique_atoms,
        "symbols": unique_symbols,
        "counts": counts,
        "f_norms": f_norms,
        "mean": mean,
    }

    ret = xr.DataArray(np.array(sigma_atom), attrs=attrs)

    return ret


def get_sigma_per_direction(f_data, f_model):
    """compute sigma for each Cartesian direction"""
    assert f_data.shape[-1] == 3
    directions = ("x", "y", "z")
    y = np.asarray(f_data).swapaxes(-1, 0)
    f = np.asarray(f_model).swapaxes(-1, 0)

    sigma_direction = [get_sigma(y[xx], f[xx]) for xx in range(len(directions))]
    return sigma_direction


def get_sigma_dict(f, fh, ref_atoms, by_symmetry=False, per_direction=False):
    """get sigma dictionary for set of forces `f` and harmonic forces `fh`"""
    dct = {}
    dct.update({"sigma": get_sigma(f, fh)})

    sigmas = get_sigma_per_atom(f, fh, ref_atoms, reduce_by_symmetry=by_symmetry)
    dct.update({"sigma_atom_mean": sigmas.attrs["mean"]})
    d = {}
    for sym, r in zip(sigmas.attrs["symbols"], sigmas):
        k = _key(sym, d)
        d[k] = float(r)

    dct.update(d)

    if per_direction:
        directions = ("x", "y", "z")
        sigmas = get_sigma_per_direction(f, fh)
        d = {}
        for ii, xx in enumerate(directions):
            d[f"sigma [{xx}]"] = sigmas[ii]

        dct.update(d)

    return dct


def get_dataframe(dataset, per_sample=False, per_direction=False, by_symmetry=False):
    """return anharmonicity dataframe for xarray.Dataset DS"""
    import pandas as pd
    from vibes.helpers.converters import json2atoms

    DS = dataset

    ref_atoms = json2atoms(DS.attrs[keys.reference_atoms])

    fs = DS.forces
    fhs = DS.forces_harmonic

    name = DS.attrs[keys.system_name]

    dct = {}
    key = f"{name}"
    dct[key] = get_sigma_dict(fs, fhs, ref_atoms, by_symmetry, per_direction)

    # per mode
    dct[key]["sigma_mode"] = get_sigma_mode(dataset)

    # add information per sample
    if per_sample:
        for ii, (f, fh) in enumerate(zip(progressbar(fs), fhs)):
            key = f"{name}.{ii:04d}"
            dct[key] = get_sigma_dict(f, fh, ref_atoms, by_symmetry, per_direction)

    df = pd.DataFrame(dct)

    return df.T


def get_sigma_mode(dataset):
    """return sigma computed for mode resolved forces

    Args:
        dataset (xarray.Dataset): the trajectory dataset including force_constants

    Returns:
        float: sigma
    """
    x, y, f = _get_forces_per_mode(dataset)

    return get_sigma(x[3:], y[3:])


def get_sigma_per_mode(dataset, absolute=False):
    """return frequencies and sigma per mode from trajectory.dataset

    sigma_s = sigma(F^a_s) / sigma(F_s)

    Args:
        dataset (xarray.Dataset): the trajectory dataset including force_constants
        absolute (bool, optional): Don't weight with the force. Defaults to False.

    Returns:
        pandas.Series: omega_s with frequencies as index
    """
    import pandas as pd

    x, y, f = _get_forces_per_mode(dataset)

    std = x.std()

    sigmas = []
    for n in range(x.shape[1]):
        X = pd.Series((x / std)[:, n].flatten())
        Y = pd.Series(((x - y) / std)[:, n].flatten())

        if absolute:
            sigmas.append(Y.std())
        else:
            sigmas.append(Y.std() / X.std())

    series = pd.Series(sigmas, index=f, name=keys.sigma_mode)
    series.index.name = keys.omega

    return series


def _get_forces_per_mode(dataset):
    """return forces and frequencies per mode from trajectory.dataset

    Args:
        dataset (xarray.Dataset): the trajectory dataset including force_constants

    Returns:
        (x, y, f): forces_mode, harmonic_forces_mode, frequencies
    """
    from vibes.helpers.converters import json2atoms
    from vibes.harmonic_analysis.mode_projection import SimpleModeProjection

    ds = dataset

    assert keys.fc_remapped in ds
    fc = ds[keys.fc_remapped].data
    supercell = json2atoms(ds.attrs[keys.reference_atoms])

    proj = SimpleModeProjection(supercell, fc)

    pref = -0.5
    x = proj.project(ds.forces, mass_weight=pref, info="forces")
    y = proj.project(ds.forces_harmonic, mass_weight=pref, info="harmonic forces")
    f = proj.omegas

    return x, y, f


def _key(sym, d):
    key = f"sigma [{sym}]"
    if key in d:
        for ii in range(1, 100):
            key = f"sigma [{sym}{ii}]"
            if key not in d:
                break
    return key
