""" tools for backup """

import shutil
import tarfile
from pathlib import Path

from vibes.filenames import filenames
from vibes.helpers import talk
from vibes.helpers.aims import peek_aims_uuid
from vibes.keys import default_backup_folder  # noqa: F401

_prefix = "backup"
_default_files = (filenames.output.aims, "control.in", filenames.atoms)


def backup_file(workdir=".", prefix=_prefix):
    """generate a backup file name for the current backup directory"""

    file = lambda counter: Path(workdir) / f"{prefix}.{counter:05d}"

    # get starting counter:
    files = sorted(int(p.stem.split(".")[1]) for p in Path(workdir).glob(f"{prefix}.*"))
    if files:
        counter = files[-1] + 1
    else:
        counter = 0

    return file(counter)


def backup_folder(
    source_dir, target_folder=".", additional_files=None, zip=True, verbose=True
):
    """backup a folder as .tgz

    Args:
        source_dir: path to the source direcotry
        target_folder: Path to where the backups should be stored
        additional_files: Additional files to backup
        zip: True if backup folder should be compressed
        verbose: If True perform more verbose logging

    Returns:
        True if source_dir exists and is not empty
    """

    output_file = backup_file(target_folder)

    if not Path(source_dir).exists():
        talk(f"{source_dir} does not exist, nothing to back up.", prefix=_prefix)
        return False

    try:
        Path(source_dir).rmdir()
        talk(f"{source_dir} is empty, do not backup and remove.", prefix=_prefix)
        return False
    except OSError:
        pass

    if not Path(target_folder).exists():
        Path(target_folder).mkdir()

    # peek aims output file
    info_str = ""
    aims_uuid = peek_aims_uuid(Path(source_dir) / filenames.output.aims)
    if aims_uuid:
        info_str = f"aims_uuid was:      {aims_uuid}"
        output_file = f"{output_file}.{aims_uuid[:8]}"

    if zip:
        output_file = f"{output_file}.tgz"
        make_tarfile(
            output_file,
            source_dir,
            additional_files=additional_files,
            arcname=Path(source_dir).stem,
        )
    else:
        shutil.move(source_dir, output_file)

    message = []
    message += [f"Folder:             {source_dir}"]
    message += [f"was backed up in:   {output_file}"]
    message += [info_str]

    if verbose:
        talk(message, prefix=_prefix)

    return True


def make_tarfile(
    output_file, source_dir, additional_files=None, arcname=None, only_defaults=True
):
    """create a tgz directory

    Args:
        output_file: Path to the output file
        source_dir: Path to the source directory
        additional_files: Additional files to include in the tar file
        arcname: Path to the archive file
        only_defaults: only use the default file names for backup
    """

    outfile = Path(output_file)

    files = Path(source_dir).glob("*")

    with tarfile.open(outfile, "w:gz") as tar:
        for file in files:
            if only_defaults and file.name not in _default_files:
                continue
            tar.add(file, arcname=Path(arcname) / file.name)

        if additional_files is not None:
            for file in additional_files:
                tar.add(file)
