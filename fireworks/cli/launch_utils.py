"""Utitility functions for FireWork Launches"""
import os
import time

from fireworks.core.fworker import FWorker
from fireworks.fw_config import FWORKER_LOC, LAUNCHPAD_LOC, QUEUEADAPTER_LOC

# from fireworks.queue.queue_launcher import launch_rocket_to_queue
from fireworks.utilities.fw_serializers import load_object_from_file

from vibes.fireworks.launchpad import LaunchPad
from vibes.fireworks.queue_launcher import launch_rocket_to_queue
from vibes.fireworks.queue_launcher import rapidfire as q_rapidfire


# Check if fabric 2.0 is installed
try:
    import fabric

    if int(fabric.__version__.split(".")[0]) < 2:
        raise ImportError
except ImportError:
    HAS_FABRIC = False
else:
    HAS_FABRIC = True
    # If fabric 2 is present check if it allows for SSH multiplexing
    from fabric.connection import SSHClient

    SSH_MULTIPLEXING = "controlpath" in SSHClient.connect.__doc__


def get_fw_files(
    config_dir,
    launchpad_file=None,
    fworker_file=None,
    queueadapter_file=None,
    remote_host=None,
):
    """Finds and returns the correct FireWorks config files

    Parameters
    ----------
    config_dir: str
        Directory where FireWorks configure files are stored
    launchpad_file: str
        LaunchPad yaml file
    fworker_file: str
        The FWorker yaml file
    queueadapter_file: str
        The QueueAdapter yaml file
    remote_host: str
        The remote host used for the calculation

    Returns
    -------
    launchpad_file: str
        The updated LaunchPad file. If None was passed try the default one in config_dir
    fworker_file: str
        The updated FWorker file. If None was passed try the default one in config_dir
    queueadapter_file: str
        The updated QueueAdapterfile. If None was passed try the default one in config_dir
    """
    if not launchpad_file and os.path.exists(
        os.path.join(config_dir, "my_launchpad.yaml")
    ):
        launchpad_file = os.path.join(config_dir, "my_launchpad.yaml")
    elif not launchpad_file:
        launchpad_file = LAUNCHPAD_LOC

    if not fworker_file and os.path.exists(os.path.join(config_dir, "my_fworker.yaml")):
        fworker_file = os.path.join(config_dir, "my_fworker.yaml")
    elif not fworker_file:
        fworker_file = FWORKER_LOC

    if remote_host == "localhost" or not remote_host:
        if not queueadapter_file and os.path.exists(
            os.path.join(config_dir, "my_qadapter.yaml")
        ):
            queueadapter_file = os.path.join(config_dir, "my_qadapter.yaml")
        elif not queueadapter_file:
            queueadapter_file = QUEUEADAPTER_LOC
    else:
        queueadapter_file = None
    return launchpad_file, fworker_file, queueadapter_file


def get_lpad_fworker_qadapter(ctx=None, offline=False):
    """Gets the LaunchPad, FWorker, and QueueAdapter from the passed files in ctx

    Parameters
    ----------
    ctx: Context
        Context for the commands (passed from click)
    offline: bool
        If True run is being done in offline mode

    Returns
    -------
    launchpad: LaunchPad
        The LaunchPad object defined by the respective file in ctx
    fworker: FWorker
        The FWorker object defined by the respective file in ctx
    queueadapter: QueueAdapter
        The QueueAdapter object defined by the respective file in ctx
    """
    if ctx.obj.fworker_file:
        fworker = FWorker.from_file(ctx.obj.fworker_file)
    else:
        fworker = FWorker()

    if offline:
        launchpad = None
    else:
        launchpad = (
            LaunchPad.from_file(ctx.obj.launchpad_file)
            if ctx.obj.launchpad_file
            else LaunchPad(strm_lvl=ctx.obj.loglvl)
        )
    if getattr(ctx.obj, "queueadapter_file", None):
        queueadapter = load_object_from_file(ctx.obj.queueadapter_file)
    else:
        queueadapter = None
    return launchpad, fworker, queueadapter


def do_qluanch(ctx, non_default):
    """Takes in a Context ctx and performs the qlaunch for that command

    Parameters
    ----------
    ctx: Context
        Context for the command
    non_default: dict
        A dict of non-default parameters
    """
    interval = ctx.obj.daemon
    while True:
        connect_kwargs = {"gss_auth": ctx.obj.gss_auth}
        if ctx.obj.remote_password is not None:
            connect_kwargs["password"] = ctx.obj.remote_password
        if ctx.obj.remote_host:
            for h in ctx.obj.remote_host:
                with fabric.Connection(
                    host=h,
                    user=ctx.obj.remote_user,
                    config=fabric.Config({"run": {"shell": ctx.obj.remote_shell}}),
                    connect_kwargs=connect_kwargs,
                ) as conn:
                    for r in ctx.obj.remote_config_dir:
                        r = os.path.expanduser(r)
                        with conn.cd(r):
                            conn.run(
                                "qlaunch_vibes {} {} {}".format(
                                    ctx.obj.pre_non_default,
                                    ctx.obj.command,
                                    non_default,
                                )
                            )
        else:
            do_launch(ctx)
        if interval > 0:
            print(
                "Next run in {} seconds... Press Ctrl-C to exit at any "
                "time.".format(interval)
            )
            time.sleep(ctx.obj.daemon)
        else:
            break


def do_launch(ctx):
    """Launches the calculations with parameters defined in ctx

    Parameters
    ----------
    ctx: Context
        The Context for the command
    """
    launchpad, fworker, queueadapter = get_lpad_fworker_qadapter(ctx)
    ctx.obj.loglvl = "CRITICAL" if ctx.obj.silencer else ctx.obj.loglvl
    if ctx.obj.command == "rapidfire":
        q_rapidfire(
            launchpad,
            fworker=fworker,
            qadapter=queueadapter,
            launch_dir=ctx.obj.launch_dir,
            nlaunches=ctx.obj.nlaunches,
            njobs_queue=ctx.obj.maxjobs_queue,
            njobs_block=ctx.obj.maxjobs_block,
            sleep_time=ctx.obj.sleep,
            reserve=ctx.obj.reserve,
            strm_lvl=ctx.obj.loglvl,
            timeout=ctx.obj.timeout,
            fill_mode=ctx.obj.fill_mode,
            firework_ids=ctx.obj.firework_ids,
            wflow_id=ctx.obj.wflow,
        )
    else:
        launch_rocket_to_queue(
            launchpad,
            fworker,
            queueadapter,
            ctx.obj.launch_dir,
            ctx.obj.reserve,
            ctx.obj.loglvl,
            False,
            ctx.obj.fill_mode,
        )
