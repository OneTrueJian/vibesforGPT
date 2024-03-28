"""
Utility functions for working with Brillouin zones
"""

from ase.dft import kpoints
from ase.lattice import FCC

from vibes.helpers.latex import latexify_labels


def get_paths(atoms):
    """Get recommended path connencting high symmetry points in the BZ.

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the recommended high-symmetry point path

    Returns
    -------
    paths: list
        Recommended special points as list >>> ['GXL', 'KU']

    """
    lat = atoms.cell.get_bravais_lattice()
    if isinstance(lat, FCC):
        paths = ["GXU", "KGL"]
    else:
        paths = lat.special_path.split(",")
    return paths


def get_special_points(atoms):
    """return the high symmetry points of the BZ for atoms

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the recommended high-symmetry points

    Returns
    -------
    kpoints: list of np.ndarrays
        The high-symmetry points
    """
    return atoms.cell.get_bravais_lattice().get_special_points()


def get_bands(atoms, paths=None, npoints=50):
    """Get the recommended BZ path(s) for atoms

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the recommended high-symmetry point path
    paths: list of np.ndarray
        Paths connecting high-symmetry points
    npoints: int
        Number of points for each band

    Returns
    -------
    bands: list of np.ndarrays
        The recommended BZ path(s) for atoms
    """
    if paths is None:
        paths = get_paths(atoms)
    bands = []
    for path in paths:
        points = kpoints.parse_path_string(path)[0]  # [:-1]
        ps = [points.pop(0)]
        for _, p in enumerate(points):
            ps.append(p)
            bands.append(atoms.cell.bandpath("".join(ps), npoints=npoints).kpts)
            ps.pop(0)
    return bands


def get_labels(paths, latex=False):
    """Get the labels for a given path for printing them with latex

    Parameters
    ----------
    paths: list of np.ndarray
        Paths connecting high-symmetry points
    latex: bool
        If True convert labels to Latex format

    Returns
    -------
    labels: list of str
        The labels for the high-symmetry path
    """
    if len(paths) == 1:
        labels = kpoints.parse_path_string(paths[0])[0]
        labels.append("|")
    else:
        labels = []
        for path in paths:
            points = kpoints.parse_path_string(path)[0]
            labels.extend(points)
            labels.append("|")

    # discard last |
    labels = labels[:-1]

    for ii, ll in enumerate(labels):
        if ll == "|":
            labels[ii] = f"{labels[ii-1]}|{labels[ii+1]}"
            labels[ii - 1], labels[ii + 1] = "", ""

    labels = [ll for ll in labels if ll]

    if latex:
        return latexify_labels(labels)

    return labels


def get_bands_and_labels(atoms, paths=None, npoints=50, latex=False):
    """Combine get_bands() and get_labels()

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        The structure to get the recommended high-symmetry point path
    paths: list of np.ndarray
        Paths connecting high-symmetry points
    npoints: int
        Number of points for each band
    latex: bool
        If True convert labels to Latex format

    Returns
    -------
    bands: list of np.ndarrays
        The recommended BZ path(s) for atoms
    labels: list of str
        The labels for the high-symmetry path
    """
    if paths is None:
        paths = get_paths(atoms)

    bands = get_bands(atoms, paths, npoints=npoints)
    labels = get_labels(paths, latex=latex)

    return bands, labels
