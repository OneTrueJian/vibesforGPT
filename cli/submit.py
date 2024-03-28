"""`vibes run` part of the CLI"""

import click

from .misc import AliasedGroup, complete_files

paths = complete_files
_prefix = "vibes.submit"
_command = lambda c, s: f"vibes run {c} {s}"


def _start(settings_file, name, dry=False):
    """check if settings contain [slurm] and submit"""
    from vibes.settings import Settings
    from vibes.slurm.submit import submit as _submit

    settings = Settings(settings_file=settings_file)
    if "slurm" not in settings:
        raise click.ClickException(f"[slurm] settings not found in {settings_file}")

    dct = settings["slurm"]
    dct["name"] = name

    _submit(dct, command=_command(name, settings_file), dry=dry)


@click.command(cls=AliasedGroup)
@click.option("--dry", is_flag=True)
@click.pass_obj
def submit(obj, dry):
    """submit a vibes workflow to slurm"""
    obj.dry = dry


@submit.command()
@click.argument("file", default="aims.in", type=paths)
@click.pass_obj
def singlepoint(obj, file):
    """submit singlepoint calculations from FILE (default: aims.in)"""

    _start(file, "singlepoint", dry=obj.dry)


@submit.command()
@click.argument("file", default="phonopy.in", type=paths)
@click.pass_obj
def phonopy(obj, file):
    """submit a phonopy calculation from FILE (default: phonopy.in)"""

    _start(file, "phonopy", dry=obj.dry)


@submit.command()
@click.argument("file", default="phono3py.in", type=paths)
@click.pass_obj
def phono3py(obj, file):
    """submit a phono3py calculation for FILE (default: phono3py.in)"""

    _start(file, "phono3py", dry=obj.dry)


@submit.command()
@click.argument("file", default="md.in", type=paths)
@click.pass_obj
def md(obj, file):
    """submit MD simulation from FILE (default: md.in)"""

    _start(file, "md", dry=obj.dry)


@submit.command()
@click.argument("file", default="relaxation.in", type=paths)
@click.pass_obj
def relaxation(obj, file):
    """submit relaxation from FILE (default: relaxation.in)"""

    _start(file, "relaxation", dry=obj.dry)
