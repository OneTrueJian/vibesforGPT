""" Update trajectory files of old format """

from argparse import ArgumentParser

from vibes.trajectory import reader


def main():
    """ main routine """
    parser = ArgumentParser(description="Update trajectory file")
    parser.add_argument("trajectory")
    parser.add_argument("-s", "--skip", default=100, type=int)
    parser.add_argument("-f", "--file", default="trajectory.xyz")
    args = parser.parse_args()

    trajectory = reader(args.trajectory)

    trajectory.to_xyz(file=args.file)


if __name__ == "__main__":
    main()
