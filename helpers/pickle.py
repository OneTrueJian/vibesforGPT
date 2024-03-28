""" Pickle a python object and save it as a compressed file """

import gzip
import pickle
from pathlib import Path


def psave(obj, file="test.pick", compressed=True, verbose=False):
    """save as (compressed) pickled file

    Args:
      obj: python object
      file:  (Default value = "test.pick")
      compressed:  (Default value = True)
      verbose:  (Default value = False)

    """
    if compressed:
        file = str(file) + ".gz"
        with gzip.open(file, "wb", 5) as f:
            pickle.dump(obj, f)
    else:
        with open(file, "wb") as f:
            pickle.dump(obj, f)
    #
    if verbose:
        print(f"List of {len(obj)} objects written to {file}.")


def pread(file):
    """read (compressed) pickled file

    Args:
      file(Path or str): path to pickle file

    Returns:
        python object

    """
    if "gz" in Path(file).suffix:
        with gzip.open(file, "rb") as f:
            return pickle.load(f)
    with open(file, "rb") as f:
        return pickle.load(f)
