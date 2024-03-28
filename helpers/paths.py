import contextlib
import os
import shutil
from pathlib import Path
from warnings import warn


@contextlib.contextmanager
def cwd(path, mkdir=False, debug=False):
    """Change cwd intermediately

    Example
    -------
    >>> with cwd(some_path):
    >>>     do so some stuff in some_path
    >>> do so some other stuff in old cwd

    Parameters
    ----------
    path: str or Path
        Path to change working directory to
    mkdir: bool
        If True make path if it does not exist
    debug: bool
        If True enter debug mode
    """
    CWD = os.getcwd()

    if os.path.exists(path) is False and mkdir:
        os.makedirs(path)

    if debug:
        os.chdir(path)
        yield
        os.chdir(CWD)
        return

    os.chdir(path)
    yield
    os.chdir(CWD)


def move(file, dest, exist_ok=False):
    """Move file to new destination

    Parameters
    ----------
    file: str or Path
        file to move
    dest: str or Path
        destination of the new file
    exist_ok: bool
        if True then it is okay if the dest folder exists
    """
    file = Path(file)
    folder = Path(dest).parent
    if file.exists():
        folder.mkdir(exist_ok=exist_ok)
        file.rename(dest)
    else:
        warn(f"** move: {file} does not exist.")


def move_to_dir(file, folder, exist_ok=False):
    """Move file to new directory

    Parameters
    ----------
    file: str or Path
        file to move
    folder: str or Path
        destination of the new file
    exist_ok: bool
        if True then it is okay if folder exists
    """

    file = Path(file)
    if file.exists():
        folder.mkdir(exist_ok=exist_ok)
        file.rename(folder / file)
    else:
        warn(f"** move_to_dir: {file} does not exist.")


def copy_to_dir(file, folder, exist_ok=False):
    """Copy file to new directory

    Parameters
    ----------
    file: str or Path
        file to move
    folder: str or Path
        destination of the new file
    exist_ok: bool
        if True then it is okay if folder exists
    """
    file = Path(file)
    if file.exists():
        folder.mkdir(exist_ok=exist_ok)
        shutil.copy(file, folder)
    else:
        warn(f"** copy_to_dir: {file} does not exist.")


# would be nice to have?
# def decor_cwd(path, mkdir=False, debug=False):
#     def decorator_cwd(func):
#         @functools.wraps(func)
#         def execute_func(path, *args, mkdir=False, debug=False, **kwargs):
#             with cwd(path, mkdir=mkdir, debug=debug):
#                 values = func(*args, **kwargs)
#             return values
#         return execute_func
#     return decorator_cwd
