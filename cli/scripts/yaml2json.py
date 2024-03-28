""" Summarize output from ASE.md class (in md.log) """

import json
from argparse import ArgumentParser

from vibes.helpers import Timer
from vibes.helpers.fileformats import from_yaml, to_json


def main():
    """ main routine """
    parser = ArgumentParser(description="convert yaml file to json")
    parser.add_argument("file", help="md.log input file")
    args = parser.parse_args()

    iname = args.file
    oname = args.file.rsplit(".yaml")[0] + ".json"
    print(f"Convert {iname} to {oname}")

    timer = Timer()
    try:
        data = from_yaml(iname, use_json=True)
    except json.decoder.JSONDecodeError:
        data = from_yaml(iname, use_json=False)
    timer(f"read {iname}")

    to_json(data, oname, mode="w")
    timer(f"write {oname}")


if __name__ == "__main__":
    main()
