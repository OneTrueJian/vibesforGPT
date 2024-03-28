"""read FORCE_CONSTANTS and remap to given supercell"""

import numpy as np

from vibes.helpers import talk
from vibes.io import parse_force_constants


def remap_force_constants(
    fc_file="FORCE_CONSTANTS",
    uc_file="geometry.primitive",
    sc_file="geometry.supercell",
    fortran=True,
    eps=1e-13,
    tol=1e-5,
):
    """take phonopy FORCE_CONSTANTS and remap to given supercell

    Parameters
    ----------
    uc_file: str or Path
        The input file for the primitive/unit cell
    sc_file: str or Path
        The input file for the supercell
    fc_file: str or Path
        The phonopy forceconstant file to parse
    eps: float
        finite zero
    tol: float
        tolerance to discern pairs
    format: str
        File format for the input geometries
    """

    fc = parse_force_constants(
        fc_file=fc_file,
        primitive=uc_file,
        supercell=sc_file,
        fortran=fortran,
        two_dim=True,
        eps=eps,
        tol=tol,
    )

    msg = f"remapped force constants from {fc_file}, shape [{fc.shape}]"
    outfile = f"{fc_file}_remapped"
    np.savetxt(outfile, fc, header=msg)

    talk(f".. remapped force constants written to {outfile}")
