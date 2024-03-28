"""CLI for vibes with click"""

import shutil

import click
import click_completion

from vibes import __version__ as vibes_version
from vibes._defaults import DEFAULT_CONFIG_FILE

from . import info, output, run, submit, template, utils
from .cli_tracker import CliTracker
from .misc import AliasedGroup, check_path


click_completion.init()


@click.command(cls=AliasedGroup)
@click.version_option(vibes_version, "-V", "--version")
@click.option("-v", "--verbose", is_flag=True, hidden=True)
@click.option("--silent", is_flag=True, help="set verbosity level to 0")
@click.pass_context
def cli(ctx, verbose, silent):
    """vibes: lattice dynamics with python"""
    if verbose:
        verbosity = 2
    elif silent:
        verbosity = 0
    else:
        verbosity = 1

    ctx.obj = CliTracker(verbose=verbosity)
    ctx.help_option_names = ["-h", "--help"]

    if verbosity > 1:
        click.echo(f"Welcome to vibes!\n")


cli.add_command(info.info)
cli.add_command(template.template)
cli.add_command(run.run)
cli.add_command(utils.utils)
cli.add_command(output.output)
cli.add_command(submit.submit)

try:
    import vibes.fireworks.cli

    cli.add_command(vibes.fireworks.cli.fireworks)
except ImportError:
    pass
except KeyError:
    pass

try:
    import vibes.balsam.cli

    cli.add_command(vibes.balsam.cli.balsam)
except ImportError:
    pass


@cli.command("status", hidden=True)
@click.option("--verbose", is_flag=True)
@click.pass_obj
def vibes_status(obj, verbose):
    """check if everything is set up"""
    from vibes.settings import Configuration

    configfile = DEFAULT_CONFIG_FILE

    check_path(configfile)

    config = Configuration(config_file=configfile)

    if verbose:
        click.echo(f"This is the configuration found in {configfile}")
        click.echo(config)

    click.echo("check if `[machine]` is in `.vibesrc`")
    assert "machine" in config
    click.echo(".. pass")
    click.echo("check basisset location in `machine.basissetloc`")
    check_path(config.machine.basissetloc)
    click.echo(".. pass")
    click.echo("check aims command in `machine.aims_command`")
    check_path(shutil.which(config.machine.aims_command))
    click.echo(".. pass")

    click.secho("It seems we are good to go!", bold=True)
