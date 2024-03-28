import subprocess as sp
import time
from pathlib import Path

from . import _talk
from .generate import generate_jobscript


def submit(
    dct,
    command=None,
    submit_command="sbatch",
    file="submit.sh",
    submit_log=".submit.log",
    log_folder="log",
    dry=False,
):
    """submit the job described in dct"""
    Path(log_folder).mkdir(exist_ok=True)

    if command:
        dct.update({"command": command})

    dct.update({"logfile": str(Path(log_folder) / dct["name"])})

    # write jobscribt to file
    generate_jobscript(dct, file=file)

    if dry:
        _talk(f"DRY RUN requested: Jobscript written to {file}. STOP")
        return

    cmd = [submit_command, file]

    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8")

    out = proc.stdout.read()
    err = proc.stderr.read()

    if not out and not err:
        out = "empty (e.g. local computation)"

    if out:
        _talk(out)
    if err:
        _talk(f"ERROR: {err}")

    try:
        timestr = time.strftime("%Y/%m/%d_%H:%M:%S")
        with open(submit_log, "a") as f:
            if err:
                f.write(f"{timestr} [ERROR]: \n{err}\n")
            else:
                f.write(f"{timestr}: {out}\n")
    except (IndexError, ValueError):
        _talk("Error during slurm submission: {:s}".format(out))
