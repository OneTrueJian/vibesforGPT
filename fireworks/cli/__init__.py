"""`vibes fireworks part of the CLI"""
import click
from fireworks.fw_config import CONFIG_FILE_DIR, LAUNCHPAD_LOC

from vibes.cli.misc import AliasedGroup
from vibes.fireworks._defaults import FW_DEFAULTS
from vibes.helpers import talk


class ListOption(click.Option):
    """A list option for click"""

    def type_cast_value(self, ctx, value):
        """Casts a comma separated list string as a python list

        Parameters
        ----------
        ctx: Context
            context for the operation (necessary, but not used)
        value: str
            value of the option

        Returns
        -------
        value: list
            The comma seperated list as either a list of strs or a list of ints
            depending on the type of the option
        """
        if not value:
            return None
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(self.type, click.types.IntParamType):
            return [int(v) for v in value]
        return value


@click.command(cls=AliasedGroup)
def fireworks():
    """Access to fhi-vibes's FireWorks wrappers"""


@fireworks.command("add_wf")
@click.option("-w", "--workflow", help="The workflow input file", default="workflow.in")
@click.option("-l", "--launchpad", help="path to launchpad file", default=LAUNCHPAD_LOC)
def add_wf(workflow, launchpad):
    """Adds a workflow to the launchpad"""
    from pathlib import Path
    from glob import glob

    from vibes.context import TaskContext
    from vibes.fireworks.workflows.workflow_generator import generate_workflow
    from vibes.settings import Settings
    from vibes.structure.misc import get_sysname

    settings = Settings(settings_file=workflow)
    structure_files = []
    if "files" in settings:
        if "geometries" in settings.files:
            if "/" == settings.files.geometries[0]:
                files = glob(settings.files.pop("geometries"))
            else:
                files = Path.cwd().glob(settings.files.pop("geometries"))
            for file in files:
                structure_files.append(Path(file).relative_to(Path.cwd()))
        if "geometry" in settings.files:
            structure_files.append(settings.files.pop("geometry"))
    else:
        raise IOError("No geometry file was specified")

    for file in structure_files:
        settings = Settings(settings_file=workflow)
        settings["calculator"]["make_species_dir"] = False

        settings.files.pop("geometries", None)
        settings.files["geometry"] = str(file)

        wflow = TaskContext(name=None, settings=settings)
        atoms = wflow.atoms
        atoms.set_calculator(wflow.calculator)

        talk(f"Generating workflow for {get_sysname(atoms)}", prefix="fireworks")
        generate_workflow(wflow, atoms, launchpad)


@fireworks.command("claunch")
@click.pass_context
@click.option(
    "-rh",
    "--remote_host",
    default=[FW_DEFAULTS["remote_host"]],
    multiple=True,
    help="Remote host to exec qlaunch.",
)
@click.option(
    "-rc",
    "--remote_config_dir",
    cls=ListOption,
    default=FW_DEFAULTS["remote_config_dir"],
    help="Remote config dir location(s). Defaults to ~/.fireworks. ",
)
@click.option(
    "-ru",
    "--remote_user",
    default=FW_DEFAULTS["remote_user"],
    help="Username to login to remote host.",
)
@click.option(
    "-rp",
    "--remote_password",
    default=FW_DEFAULTS["remote_password"],
    help="Password for remote host (if necessary). Not recommended",
)
@click.option(
    "-rsh",
    "--remote_shell",
    help="Shell command to use on remote host for running submission.",
    default="/bin/bash -l -c",
)
@click.option(
    "-rs",
    "--remote_setup",
    help="Setup the remote config dir using files in the directory specified by -c.",
    is_flag=True,
)
@click.option("--gss_auth", help="use gss_api authorization", is_flag=True)
@click.option(
    "-rro",
    "--remote_recover_offline",
    is_flag=True,
    help="recover offline jobs from remote host",
)
@click.option(
    "-d",
    "--daemon",
    help="Daemon mode. Command is repeated every x seconds. Defaults to non-daemon mode.",
    type=int,
    default=0,
)
@click.option(
    "--launch_dir", help="directory to launch the job / rapid-fire", default="."
)
@click.option("--logdir", help="path to a directory for logging", default=None)
@click.option(
    "--loglvl", help="level to print log messages", default="CRITICAL", type=str
)
@click.option("-s", "--silencer", help="shortcut to mute log messages", is_flag=True)
@click.option("-r", "--reserve", help="reserve a fw", is_flag=True)
@click.option("-l", "--launchpad_file", help="path to launchpad file")
@click.option("-w", "--fworker_file", help="path to fworker file")
@click.option("-q", "--queueadapter_file", help="path to queueadapter file")
@click.option(
    "-c",
    "--config_dir",
    help="path to a directory containing config files (used if -l, -w, -q unspecified)",
    default=CONFIG_FILE_DIR,
)
@click.option(
    "-fm",
    "--fill_mode",
    help="launch queue submissions even when there is nothing to run",
    is_flag=True,
)
@click.option(
    "-ids",
    "--firework_ids",
    cls=ListOption,
    help="A list of specific ids to run",
    type=int,
    default=None,
)
@click.option(
    "-wf",
    "--wflow",
    cls=ListOption,
    help="A list of the root fw ids of a workflow",
    type=int,
    default=None,
)
@click.option(
    "-m",
    "--maxjobs_queue",
    help="maximum jobs to keep in queue for this user",
    default=FW_DEFAULTS["njobs_queue"],
    type=int,
)
@click.option(
    "-b",
    "--maxjobs_block",
    help="maximum jobs to put in a block",
    default=FW_DEFAULTS["njobs_block"],
    type=int,
)
@click.option(
    "--nlaunches",
    help='num_launches (int or "infinite"; default 0 is all jobs in DB)',
    default=FW_DEFAULTS["nlaunches"],
)
@click.option(
    "--timeout",
    help="timeout (secs) after which to quit (default None)",
    default=None,
    type=int,
)
@click.option(
    "--sleep",
    help="sleep time between loops",
    default=FW_DEFAULTS["sleep_time"],
    type=int,
)
@click.option(
    "-tq",
    "--tasks_to_queue",
    cls=ListOption,
    type=str,
    default=",".join(FW_DEFAULTS["tasks2queue"]),
    help="list of tasks to be sent to the queue",
)
def claunch(
    ctx,
    remote_host,
    remote_config_dir,
    remote_user,
    remote_password,
    remote_shell,
    remote_setup,
    gss_auth,
    remote_recover_offline,
    daemon,
    launch_dir,
    logdir,
    loglvl,
    silencer,
    reserve,
    launchpad_file,
    fworker_file,
    queueadapter_file,
    config_dir,
    fill_mode,
    firework_ids,
    wflow,
    maxjobs_queue,
    maxjobs_block,
    nlaunches,
    timeout,
    sleep,
    tasks_to_queue,
):
    """Launches Fireworks both locally and remotely based on what the tasks func is"""
    from vibes.fireworks.combined_launcher import rapidfire

    # Check if fabric 2.0 is installed
    try:
        import fabric

        if int(fabric.__version__.split(".")[0]) < 2:
            raise ImportError
    except ImportError:
        HAS_FABRIC = False
    else:
        HAS_FABRIC = True

    if remote_host and not HAS_FABRIC:
        raise ImportError(
            "Remote options require the Fabric package v2+ to be installed!"
        )

    from vibes.fireworks.cli.launch_utils import get_fw_files, get_lpad_fworker_qadapter

    launchpad_file, fworker_file, queueadapter_file = get_fw_files(
        config_dir, launchpad_file, fworker_file, queueadapter_file, remote_host
    )
    ctx.obj.remote_host = remote_host
    ctx.obj.remote_config_dir = remote_config_dir
    ctx.obj.remote_user = remote_user
    ctx.obj.remote_password = remote_password
    ctx.obj.remote_shell = remote_shell
    ctx.obj.remote_setup = remote_setup
    ctx.obj.gss_auth = gss_auth
    ctx.obj.remote_recover_offline = remote_recover_offline
    ctx.obj.daemon = daemon
    ctx.obj.launch_dir = launch_dir
    ctx.obj.logdir = logdir
    ctx.obj.loglvl = loglvl
    ctx.obj.silencer = silencer
    ctx.obj.reserve = reserve
    ctx.obj.launchpad_file = launchpad_file
    ctx.obj.fworker_file = fworker_file
    ctx.obj.queueadapter_file = queueadapter_file
    ctx.obj.config_dir = config_dir
    ctx.obj.fill_mode = fill_mode
    ctx.obj.firework_ids = firework_ids
    ctx.obj.wflow = wflow
    ctx.obj.maxjobs_queue = maxjobs_queue
    ctx.obj.maxjobs_block = maxjobs_block
    ctx.obj.nlaunches = nlaunches
    ctx.obj.timeout = timeout
    ctx.obj.sleep = sleep
    ctx.obj.tasks_to_queue = tasks_to_queue

    launchpad, fworker, queueadapter = get_lpad_fworker_qadapter(ctx)

    rapidfire(
        launchpad,
        fworker=fworker,
        qadapter=queueadapter,
        launch_dir=launch_dir,
        nlaunches=nlaunches,
        njobs_queue=maxjobs_queue,
        njobs_block=maxjobs_block,
        sleep_time=sleep,
        reserve=reserve,
        strm_lvl=loglvl,
        timeout=timeout,
        fill_mode=fill_mode,
        firework_ids=firework_ids,
        wflow=wflow,
        tasks2queue=tasks_to_queue,
        gss_auth=gss_auth,
        remote_host=remote_host,
        remote_config_dir=remote_config_dir,
        remote_user=remote_user,
        remote_password=remote_password,
        remote_shell=remote_shell,
        remote_recover_offline=remote_recover_offline,
        daemon=daemon,
    )


@fireworks.command(cls=AliasedGroup)
@click.pass_context
@click.option(
    "-rh",
    "--remote_host",
    default=[FW_DEFAULTS["remote_host"]],
    multiple=True,
    help="Remote host to exec qlaunch.",
)
@click.option(
    "-rc",
    "--remote_config_dir",
    cls=ListOption,
    default=FW_DEFAULTS["remote_config_dir"],
    help="Remote config dir location(s).",
)
@click.option(
    "-ru",
    "--remote_user",
    default=FW_DEFAULTS["remote_user"],
    help="Username to login to remote host.",
)
@click.option(
    "-rp",
    "--remote_password",
    default=FW_DEFAULTS["remote_password"],
    help="Password for remote host (if necessary). Not recommended",
)
@click.option(
    "-rsh",
    "--remote_shell",
    help="Shell command to use on remote host for running submission.",
    default="/bin/bash -l -c",
)
@click.option(
    "-rs",
    "--remote_setup",
    help="Setup the remote config dir using files in the directory specified by -c.",
    is_flag=True,
)
@click.option("-rgss", "--gss_auth", help="use gss_api authorization", is_flag=True)
@click.option(
    "-rro",
    "--remote_recover_offline",
    is_flag=True,
    help="recover offline jobs from remote host",
)
@click.option(
    "-d",
    "--daemon",
    help="Daemon mode. Command is repeated every x seconds. Defaults to non-daemon mode.",
    type=int,
    default=0,
)
@click.option(
    "--launch_dir", help="directory to launch the job / rapid-fire", default="."
)
@click.option("--logdir", help="path to a directory for logging", default=None)
@click.option(
    "--loglvl", help="level to print log messages", default="CRITICAL", type=str
)
@click.option("-s", "--silencer", help="shortcut to mute log messages", is_flag=True)
@click.option("-r", "--reserve", help="reserve a fw", is_flag=True)
@click.option("-l", "--launchpad_file", help="path to launchpad file")
@click.option("-w", "--fworker_file", help="path to fworker file")
@click.option("-q", "--queueadapter_file", help="path to queueadapter file")
@click.option(
    "-c",
    "--config_dir",
    help="path to a directory containing config files (used if -l, -w, -q unspecified)",
    default=CONFIG_FILE_DIR,
)
@click.option(
    "-fm",
    "--fill_mode",
    help="launch queue submissions even when there is nothing to run",
    is_flag=True,
)
def qlaunch(
    ctx,
    remote_host,
    remote_config_dir,
    remote_user,
    remote_password,
    remote_shell,
    remote_setup,
    gss_auth,
    remote_recover_offline,
    daemon,
    launch_dir,
    logdir,
    loglvl,
    silencer,
    reserve,
    launchpad_file,
    fworker_file,
    queueadapter_file,
    config_dir,
    fill_mode,
):
    """Launch FireWorks to the queue"""
    import os

    from vibes.fireworks.cli.launch_utils import get_fw_files

    # Check if fabric 2.0 is installed
    try:
        import fabric

        if int(fabric.__version__.split(".")[0]) < 2:
            raise ImportError
    except ImportError:
        HAS_FABRIC = False
    else:
        HAS_FABRIC = True

    launchpad_file, fworker_file, queueadapter_file = get_fw_files(
        config_dir, launchpad_file, fworker_file, queueadapter_file, remote_host
    )
    ctx.obj.remote_host = remote_host
    ctx.obj.remote_config_dir = remote_config_dir
    ctx.obj.remote_user = remote_user
    ctx.obj.remote_password = remote_password
    ctx.obj.remote_shell = remote_shell
    ctx.obj.remote_setup = remote_setup
    ctx.obj.gss_auth = gss_auth
    ctx.obj.remote_recover_offline = remote_recover_offline
    ctx.obj.daemon = daemon
    ctx.obj.launch_dir = launch_dir
    ctx.obj.logdir = logdir
    ctx.obj.loglvl = loglvl
    ctx.obj.silencer = silencer
    ctx.obj.reserve = reserve
    ctx.obj.launchpad_file = launchpad_file
    ctx.obj.fworker_file = fworker_file
    ctx.obj.queueadapter_file = queueadapter_file
    ctx.obj.config_dir = config_dir
    ctx.obj.fill_mode = fill_mode
    ctx.obj.HAS_FABRIC = HAS_FABRIC

    pre_non_default = []
    for k in ["silencer", "reserve"]:
        v = getattr(ctx.obj, k, None)
        if v:
            pre_non_default.append("--%s" % k)
    pre_non_default = " ".join(pre_non_default)
    ctx.obj.pre_non_default = pre_non_default

    if remote_host and not HAS_FABRIC:
        raise ImportError(
            "Remote options require the Fabric package v2+ to be installed!"
        )

    if remote_setup and remote_host:
        for h in remote_host:
            # conf = fabric.Configuration()
            # conf.run.shell = remote_shell
            with fabric.Connection(
                host=h,
                user=remote_user,
                config=fabric.Config({"run": {"shell": remote_shell}}),
                connect_kwargs={
                    "password": remote_password,
                    "gss_auth": ctx.obj.gss_auth,
                },
            ) as conn:
                for r in remote_config_dir:
                    r = os.path.expanduser(r)
                    conn.run("mkdir -p {}".format(r))
                    for f in os.listdir(ctx.obj.config_dir):
                        if os.path.isfile(f):
                            conn.put(f, os.path.join(r, f))


@qlaunch.command("rapidfire")
@click.pass_context
@click.option(
    "-ids",
    "--firework_ids",
    cls=ListOption,
    help="A list of specific ids to run",
    type=int,
    default=None,
)
@click.option(
    "-wf",
    "--wflow",
    cls=ListOption,
    help="A list of the root fw ids of a workflow",
    type=int,
    default=None,
)
@click.option(
    "-m",
    "--maxjobs_queue",
    help="maximum jobs to keep in queue for this user",
    default=FW_DEFAULTS["njobs_queue"],
    type=int,
)
@click.option(
    "-b",
    "--maxjobs_block",
    help="maximum jobs to put in a block",
    default=FW_DEFAULTS["njobs_block"],
    type=int,
)
@click.option(
    "--nlaunches",
    help='num_launches (int or "infinite"; default 0 is all jobs in DB)',
    default=FW_DEFAULTS["nlaunches"],
)
@click.option(
    "--timeout",
    help="timeout (secs) after which to quit (default None)",
    default=None,
    type=int,
)
@click.option(
    "--sleep",
    help="sleep time between loops",
    default=FW_DEFAULTS["sleep_time"],
    type=int,
)
@click.option(
    "-tq",
    "--tasks_to_queue",
    cls=ListOption,
    type=str,
    default=FW_DEFAULTS["tasks2queue"],
    help="list of tasks to be sent to the queue",
)
def qlaunch_rapidfire(
    ctx,
    firework_ids,
    wflow,
    maxjobs_queue,
    maxjobs_block,
    nlaunches,
    timeout,
    sleep,
    tasks_to_queue,
):
    """Preform a qlaunch rpaidfire"""
    from vibes.fireworks.cli.launch_utils import do_qluanch

    ctx.obj.firework_ids = firework_ids
    ctx.obj.wflow = wflow
    ctx.obj.maxjobs_queue = maxjobs_queue
    ctx.obj.maxjobs_block = maxjobs_block
    ctx.obj.nlaunches = nlaunches
    ctx.obj.timeout = timeout
    ctx.obj.sleep = sleep
    ctx.obj.tasks_to_queue = tasks_to_queue
    ctx.obj.command = "rapidfire"

    non_default = []
    for k in ["maxjobs_queue", "maxjobs_block", "nlaunches", "sleep"]:
        v = getattr(ctx.obj, k, None)
        if v is not None:
            non_default.append("--{} {}".format(k, v))
    val = getattr(ctx.obj, "firework_ids", None)
    if val is not None:
        non_default.append("--{} {}".format("firework_ids", val[0]))
        for v in val[1:]:
            non_default[-1] += ",{}".format(v)
    val = getattr(ctx.obj, "wflow", None)
    if val is not None:
        non_default.append("--{} {}".format("wflow", val[0]))
        for v in val[1:]:
            non_default[-1] += ",{}".format(v)
    do_qluanch(ctx, non_default)


@qlaunch.command("singleshot")
@click.pass_context
@click.option(
    "-f",
    "--firework_id",
    help="specific firework_id to run in reservation mode",
    default=None,
    type=int,
)
def qlaunch_singleshot(ctx, firework_id):
    """preform a qlaunch singleshot"""
    from vibes.fireworks.cli.launch_utils import do_qluanch

    ctx.obj.firework_id = firework_id
    ctx.obj.command = "singleshot"
    non_default = []
    val = getattr(ctx.obj, "firework_id", None)
    if val is not None:
        non_default.append("--{} {}".format("firework_id", val))
    do_qluanch(ctx, non_default)


@fireworks.command(cls=AliasedGroup)
@click.pass_context
@click.option(
    "--loglvl", help="level to print log messages", default="CRITICAL", type=str
)
@click.option("-s", "--silencer", help="shortcut to mute log messages", is_flag=True)
@click.option("-l", "--launchpad_file", help="path to launchpad file")
@click.option("-w", "--fworker_file", help="path to fworker file")
@click.option(
    "-c",
    "--config_dir",
    help="path to a directory containing config files (used if -l, -w, -q unspecified)",
    default=CONFIG_FILE_DIR,
)
def rlaunch(ctx, loglvl, silencer, launchpad_file, fworker_file, config_dir):
    """Launch a rocket locally"""
    from fireworks.utilities.fw_utilities import get_my_host, get_my_ip, get_fw_logger

    from vibes.fireworks.cli.launch_utils import get_fw_files

    launchpad_file, fworker_file, _ = get_fw_files(
        config_dir, launchpad_file, fworker_file, None, None
    )

    loglvl = "CRITICAL" if silencer else loglvl

    ctx.obj.loglvl = loglvl
    ctx.obj.silencer = silencer
    ctx.obj.launchpad_file = launchpad_file
    ctx.obj.fworker_file = fworker_file
    ctx.obj.config_dir = config_dir

    # prime addr lookups
    _log = get_fw_logger("rlaunch", stream_level="INFO")
    _log.info("Hostname/IP lookup (this will take a few seconds)")
    get_my_host()
    get_my_ip()


@rlaunch.command("rapidfire")
@click.pass_context
@click.option(
    "-ids",
    "--firework_ids",
    cls=ListOption,
    help="A list of specific ids to run",
    type=int,
    default=None,
)
@click.option(
    "-wf",
    "--wflow",
    cls=ListOption,
    help="A list of the root fw ids of a workflow",
    type=int,
    default=None,
)
@click.option(
    "--nlaunches",
    help='num_launches (int or "infinite"; default 0 is all jobs in DB)',
    default=FW_DEFAULTS["nlaunches"],
)
@click.option(
    "--timeout",
    help="timeout (secs) after which to quit (default None)",
    default=None,
    type=int,
)
@click.option(
    "--sleep",
    help="sleep time between loops",
    default=FW_DEFAULTS["sleep_time"],
    type=int,
)
@click.option(
    "--max_loops",
    help="after this many sleep loops, quit even in infinite nlaunches mode",
    default=-1,
    type=int,
)
@click.option(
    "--local_redirect",
    help="Redirect stdout and stderr to the launch directory",
    is_flag=True,
)
def rlaunch_rapidfire(
    ctx, firework_ids, wflow, nlaunches, timeout, sleep, max_loops, local_redirect
):
    from vibes.fireworks.cli.launch_utils import get_lpad_fworker_qadapter
    from vibes.fireworks.rocket_launcher import rapidfire

    """preform a rlaunch rpaidfire"""
    launchpad, fworker, _ = get_lpad_fworker_qadapter(ctx)
    rapidfire(
        launchpad,
        fworker=fworker,
        m_dir=None,
        nlaunches=nlaunches,
        max_loops=max_loops,
        sleep_time=sleep,
        strm_lvl=ctx.obj.loglvl,
        timeout=timeout,
        local_redirect=local_redirect,
        firework_ids=firework_ids,
        wflow_id=wflow,
    )


@rlaunch.command("singleshot")
@click.pass_context
@click.option(
    "-f",
    "--fw_id",
    help="specific firework_id to run in reservation mode",
    default=None,
    type=int,
)
@click.option("--offline", help="run in offline mode (FW.json required)", is_flag=True)
@click.option("--pdb", help="shortcut to invoke debugger on error", is_flag=True)
def rlaunch_singleshot(ctx, fw_id, offline, pdb):
    """preform a rlaunch singleshot"""
    from fireworks.core.rocket_launcher import launch_rocket
    from vibes.fireworks.cli.launch_utils import get_lpad_fworker_qadapter

    launchpad, fworker, _ = get_lpad_fworker_qadapter(ctx, offline)

    launch_rocket(launchpad, fworker, fw_id, ctx.obj.loglvl, pdb_on_exception=pdb)


@rlaunch.command("multi")
@click.pass_context
@click.option("--num_jobs", help="the number of jobs to run in parallel", type=int)
@click.option(
    "--nlaunches",
    help='number of FireWorks to run in series per parallel job (int or "infinite"',
    default=0,
)
@click.option(
    "--sleep",
    help="sleep time between loops in infinite launch mode" "(secs)",
    default=None,
    type=int,
)
@click.option(
    "--timeout",
    help="timeout (secs) after which to quit (default None)",
    default=None,
    type=int,
)
@click.option(
    "--nodefile",
    help="nodefile name or environment variable name containing the node file name",
    default=None,
    type=str,
)
@click.option(
    "--ppn",
    help="processors per node (for populating FWData only)",
    default=1,
    type=int,
)
@click.option(
    "--exclude_current_node",
    help="Don't use the script launching node" "as compute node",
    is_flag=True,
)
@click.option(
    "--local_redirect",
    help="Redirect stdout and stderr to the launch directory",
    is_flag=True,
)
def rlaunch_multi(
    ctx,
    num_jobs,
    nlaunches,
    sleep,
    timeout,
    nodefile,
    ppn,
    exclude_current_node,
    local_redirect,
):
    """preform a rlaunch multi"""
    import os
    from fireworks.features.multi_launcher import launch_multiprocess
    from vibes.fireworks.cli.launch_utils import get_lpad_fworker_qadapter

    launchpad, fworker, _ = get_lpad_fworker_qadapter(ctx)

    total_node_list = None
    if nodefile:
        if nodefile in os.environ:
            nodefile = os.environ[nodefile]
        with open(nodefile, "r") as f:
            total_node_list = [line.strip() for line in f.readlines()]
    launch_multiprocess(
        launchpad,
        fworker,
        ctx.obj.loglvl,
        nlaunches,
        num_jobs,
        sleep,
        total_node_list,
        ppn,
        timeout=timeout,
        exclude_current_node=exclude_current_node,
        local_redirect=local_redirect,
    )
