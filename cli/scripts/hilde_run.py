""" create a configuration file and working directory """

from argparse import ArgumentParser

from vibes import Settings, run


def main():
    """ main routine """
    parser = ArgumentParser(description="run the vibes task encoded in settings.in")
    parser.add_argument("--dry", action="store_true", help="only show configuration")
    args = parser.parse_args()

    if args.dry:
        print("Summary of settings:")
        settings = Settings()
        settings.print()

    else:
        run()


if __name__ == "__main__":
    main()
