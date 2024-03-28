"""vibes CLI wrapper ase kpt density tools"""

from argparse import ArgumentParser as argpars

from vibes.helpers import talk
from vibes.helpers.k_grid import d2k, k2d
from vibes.io import inform, read


def suggest_k_grid(file, density, uneven, format):
    """suggest a k_grid for geometry in FILENAME based on density

    Parameters
    ----------
    file: str or Path
        The input geometry file
    density: float
        The kpoint density
    uneven: bool
        If True allow uneven values of k
    format: str
        The ASE file format for geometry files
    """

    cell = read(file, format=format)
    inform(cell, file=file, verbosity=0)

    k_grid = d2k(cell, kptdensity=density, even=not uneven)

    # the resulting density
    density = k2d(cell, k_grid)

    talk(f"\nSuggested k_grid for kpt-density {density}:")
    k_grid = " ".join([f"{k}" for k in k_grid])
    talk(f"  k_grid {k_grid}")
    rep = "[{}]".format(", ".join([f"{d:.2f}" for d in density]))
    talk(f"Resulting density: {density.mean():.2f} {rep}")


def main():
    """ suggest k_grid """
    parser = argpars(description="Read geometry and suggest k_grid based on density")
    parser.add_argument("geom", type=str, help="geometry input file")
    parser.add_argument("-d", "--density", type=float, default=3.5)
    parser.add_argument("--uneven", action="store_true")
    parser.add_argument("--format", default="aims")
    args = parser.parse_args()

    suggest_k_grid(args.geo, args.density, args.uneven, args.format)


if __name__ == "__main__":
    main()
