""" Update trajectory files of old format """

import shutil
from argparse import ArgumentParser

from vibes.helpers import talk
from vibes.io import read
from vibes.trajectory import reader


def update_trajectory(trajectory_file, uc=None, sc=None, format="aims"):
    """update TRAJECTORY by adding unit cell and supercell"""
    trajectory = reader(trajectory_file)
    new_trajectory = "temp.son"

    if uc:
        atoms = read(uc, format=format)
        trajectory.primitive = atoms

    if sc:
        atoms = read(sc, format=format)
        trajectory.supercell = atoms

    trajectory.write(file=new_trajectory)

    file = f"{trajectory_file}.bak"
    talk(f".. back up old trajectory to {file}")
    shutil.copy(trajectory_file, file)
    talk(f".. write new trajectory to {trajectory_file}")
    shutil.move(new_trajectory, trajectory_file)


def main():
    """ main routine """
    parser = ArgumentParser(description="Update trajectory file")
    parser.add_argument("trajectory")
    parser.add_argument("-uc", help="Add a (primitive) unit cell")
    parser.add_argument("-sc", help="Add the respective supercell")
    parser.add_argument("--format", default="aims")
    args = parser.parse_args()

    update_trajectory(args.trajectory, args.uc, args.sc, args.format)


if __name__ == "__main__":
    main()
