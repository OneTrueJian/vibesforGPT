import re
from collections import namedtuple

import numpy as np
from vibes.structure.convert import to_Atoms

fp_tup = namedtuple("fp_tup", "frequencies occupancies special_pts nbins")


def get_ener(binning, frequencies, min_e, max_e, nbins):
    """Get the energy bins used for making a fingerprint

    Parameters
    ----------
    binning: bool
        if True use the band/DOS frequencies given as the bin boundaries
    frequencies: list or np.ndarray of floats
        The set of frequencies the band structure or DOS is calculated for
    min_e: float
        minimum energy mode to be included
    max_e: float
        maximum energy mode to be included
    nbins: int
        number of bins for the histogram

    Returns
    -------
    np.ndarray of floats
        energy bin labels (energy of the band or bin center point)
    np.ndarray of floats
        energy bin boundaries
    """
    if binning:
        enerBounds = np.linspace(min_e, max_e, nbins + 1)
        return enerBounds[:-1] + (enerBounds[1] - enerBounds[0]) / 2.0, enerBounds
    else:
        return (
            np.array(frequencies),
            np.append(frequencies, [frequencies[-1] + np.abs(frequencies[-1]) / 10]),
        )


def get_phonon_bs_fp(
    phonon, q_points=None, binning=True, min_e=None, max_e=None, nbins=32
):
    """Generates the phonon band structure fingerprint for a band structure

    Parameters
    ----------
    phonon: phonopy Object
        The phonopy object
    q_points: dict
        A dict of the high symmetry points Keys=labels, Values= high symmetry point
    min_e: float
        The minimum mode energy to include in the fingerprint
    max_e: float
        The maximum mode energy to include in the fingerprint
    nbins:int
        Number of bins to be used in the fingerprint

    Returns
    -------
    namedtuple(fp_tup)
        The phonon band structure fingerprint
    """
    if q_points is None:
        q_points = (
            to_Atoms(phonon.primitive).cell.get_bravais_lattice().get_special_points()
        )

    bands = {}
    for pt in q_points:
        bands[pt] = phonon.get_frequencies(q_points[pt])

    if max_e is None:
        max_e = np.max(np.array([bands[pt] for pt in bands]))

    if min_e is None:
        min_e = np.min(np.array([bands[pt] for pt in bands]))

    freq_list = []
    n_bands = []
    special_pts = []

    for pt in bands:
        special_pts.append(pt)
        ener, enerBounds = get_ener(binning, bands[pt], min_e, max_e, nbins)
        freq_list.append(ener)
        n_bands.append(np.histogram(bands[pt], enerBounds)[0])

    return fp_tup(
        np.array(freq_list), np.array(n_bands), special_pts, len(freq_list[0])
    )


def get_phonon_dos_fp(phonon, binning=True, min_e=None, max_e=None, nbins=256):
    """Generates the DOS fingerprint for a bands structure stored in a phonopy object

    Parameters
    ----------
    phonon: phonopy Object
        The phonopy object
    min_e: float
        The minimum mode energy to include in the fingerprint
    max_e: float
        The maximum mode energy to include in the fingerprint
    nbins:int
        Number of bins to be used in the fingerprint

    Returns
    -------
    namedtuple(fp_tup)
        The phonon density of states fingerprint
    """
    dos = np.array(phonon.get_total_DOS()).transpose()

    if max_e is None:
        max_e = np.max(dos[:, 0])

    if min_e is None:
        min_e = np.min(dos[:, 0])

    if dos.shape[0] < nbins:
        inds = np.where((dos[:, 0] >= min_e) & (dos[:, 0] <= max_e))
        return fp_tup(dos[inds, 0], dos[inds, 1], ["DOS"], dos.shape[0])

    ener, enerBounds = get_ener(binning, dos[:, 0], min_e, max_e, nbins)
    dos_rebin = np.zeros(ener.shape)
    for ii, e1, e2 in zip(range(len(ener)), enerBounds[0:-1], enerBounds[1:]):
        dos_rebin[ii] = np.sum(
            dos[np.where((dos[:, 0] >= e1) & (dos[:, 0] < e2))[0], 1]
        )
    return fp_tup(np.array(ener), dos_rebin, ["DOS"], nbins)


def scalar_product(fp1, fp2, col=0, pt="All", normalize=False, tanimoto=False):
    """Calculates the dot product between two finger prints

    Parameters
    ----------
    fp1: namedtuple(fp_tup)
        The first fingerprint
    fp2: namedtuple(fp_tup)
        The second fingerprint
    col: int
        The item in the fingerprints to take the dot product of (either 0 or 1)
    pt: int or 'All'
        The index of the point that the dot product is to be taken, 'All' flatten arraies
    normalize: bool
        If True normalize the scalar product to 1

    Returns
    -------
    float
        The dot product
    """
    if not isinstance(fp1, dict):
        fp1_dict = to_dict(fp1)
    else:
        fp1_dict = fp1

    if not isinstance(fp2, dict):
        fp2_dict = to_dict(fp2)
    else:
        fp2_dict = fp2

    if pt == "All":
        vec1 = np.array([pt[col] for pt in fp1_dict.values()]).flatten()
        vec2 = np.array([pt[col] for pt in fp2_dict.values()]).flatten()
    else:
        vec1 = fp1_dict[fp1[2][pt]][col]
        vec2 = fp2_dict[fp2[2][pt]][col]

    rescale = 1.0

    if tanimoto:
        rescale = (
            np.linalg.norm(vec1) ** 2 + np.linalg.norm(vec2) ** 2 - np.dot(vec1, vec2)
        )
    elif normalize:
        rescale = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    return np.dot(vec1, vec2) / rescale


def to_dict(fp, to_mongo=False):
    """Converts a fingerprint into a dictionary

    Parameters
    ----------
    fp: namedtuple(fp_tup)
        The fingerprint to be converted into a dictionary
    to_mongo: bool
        True if the database that this will be stored in is a mongo db

    Returns
    -------
    dict
        A dict of the fingerprint Keys=labels, Values=np.ndarray(frequencies, #of states)
    """
    fp_dict = {}
    if not to_mongo:
        if len(fp[2]) > 1:
            for aa in range(len(fp[2])):
                fp_dict[fp[2][aa]] = np.array([fp[0][aa], fp[1][aa]]).T
        else:
            fp_dict[fp[2][0]] = np.array([fp[0], fp[1]]).T
    else:
        if len(fp[2]) > 1:
            for aa in range(len(fp[2])):
                fp_dict[re.sub("[.]", "_", str(fp[2][aa]))] = np.array(
                    [fp[0][aa], fp[1][aa]]
                ).T
        else:
            fp_dict[re.sub("[.]", "_", str(fp[2][0]))] = np.array([fp[0], fp[1]]).T
    return fp_dict
