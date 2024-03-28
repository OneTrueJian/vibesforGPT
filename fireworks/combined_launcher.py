"""Used to combine a queue/rocket launcher based on which FireTasks are in a WorkFlow

FireWorks Copyright (c) 2013, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""
import glob
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.fw_config import (
    ALWAYS_CREATE_NEW_BLOCK,
    QUEUE_UPDATE_INTERVAL,
    RAPIDFIRE_SLEEP_SECS,
)
from fireworks.queue.queue_launcher import (
    _get_number_of_jobs_in_queue,
    _njobs_in_dir,
    launch_rocket_to_queue,
)
from fireworks.utilities.fw_utilities import (
    create_datestamp_dir,
    get_fw_logger,
    log_exception,
)
from vibes.fireworks._defaults import FW_DEFAULTS
from vibes.fireworks.qlaunch_remote import qlaunch_remote
from vibes.helpers import talk


# See if Fabric2 is installed
try:
    import fabric

    if int(fabric.__version__.split(".")[0]) < 2:
        raise ImportError
except ImportError:
    HAS_FABRIC = False
    SSH_MULTIPLEXING = False
else:
    HAS_FABRIC = True
    # If fabric 2 is present check if it allows for SSH multiplexing
    from fabric.connection import SSHClient

    SSH_MULTIPLEXING = "controlpath" in SSHClient.connect.__doc__


def get_ready_firework_ids(wflow):
    """Gets an ordered (with respect to when jobs need to run) list of fws in a WorkFlow wflow

    Parameters
    ----------
    wflow : WorkFlow
        WorkFlow to run

    Returns
    -------
    list of ints
        An ordered list for the desired workflow to run

    """
    ready_fireworks = []

    for fw_id, fw in wflow.id_fw.items():
        if fw.state == "READY":
            ready_fireworks.append(fw_id)
    return ready_fireworks[::-1]


def use_queue_launch(fire_work, tasks2queue):
    """Determines if a particular FireWork should be ran on a remote

    Parameters
    ----------
    fire_work : FireWork
        FireWork to be run locally or remotely
    tasks2queue : list of str
        Paths of functions to run on the queue

    Returns
    -------
    bool
        True if the task should be run on a remote

    """
    for task in fire_work.spec["_tasks"]:
        if task["args"][0] in tasks2queue:
            return True
    return False


def rapidfire(
    launchpad,
    fworker=None,
    qadapter=None,
    launch_dir=".",
    nlaunches=FW_DEFAULTS.nlaunches,
    njobs_queue=FW_DEFAULTS.njobs_queue,
    njobs_block=FW_DEFAULTS.njobs_block,
    sleep_time=FW_DEFAULTS.sleep_time,
    reserve=False,
    strm_lvl="CRITICAL",
    timeout=None,
    fill_mode=False,
    firework_ids=None,
    wflow=None,
    tasks2queue=FW_DEFAULTS.tasks2queue,
    gss_auth=False,
    controlpath=None,
    remote_host=FW_DEFAULTS.remote_host,
    remote_config_dir=FW_DEFAULTS.remote_config_dir,
    remote_user=FW_DEFAULTS.remote_user,
    remote_password=FW_DEFAULTS.remote_password,
    remote_shell="/bin/bash -l -c",
    remote_recover_offline=False,
    daemon=0,
):
    """Submit many jobs to the queue.

    Parameters
    ----------
    launchpad : LaunchPad
        The LaunchPad where the launches are defined
    fworker : FWorker
        The FireWorker used to run the launches (Default value = None)
    qadapter : QueueAdapterBase
        The QueueAdapter for the launches (Default value = None)
    launch_dir : str
        directory where we want to write the blocks (Default value = ".")
    nlaunches : int
        total number of launches desired; "infinite" for loop, 0 for one round (Default value = FW_DEFAULTS.nlaunches)
    njobs_queue : int
        stops submitting jobs when njobs_queue jobs are in the queue, 0 for no limit (Default value = FW_DEFAULTS.njobs_queue)
    njobs_block : int
        automatically write a new block when njobs_block jobs are in a single block (Default value = FW_DEFAULTS.njobs_block)
    sleep_time : int
        secs to sleep between rapidfire loop iterations (Default value = FW_DEFAULTS.sleep_time)
    reserve : bool
        Whether to queue in reservation mode (Default value = False)
    strm_lvl : str
        level at which to stream log messages (Default value = "CRITICAL")
    timeout : int
        # of seconds after which to stop the rapidfire process (Default value = None)
    fill_mode : bool
        True if submit jobs with nothing to run (only in non-reservation mode) (Default value = False)
    firework_ids : list of ints
        a list firework_ids to launch (len(firework_ids) == nlaunches) (Default value = None)
    wflow : WorkFlow
        the workflow this qlauncher is supposed to run (Default value = None)
    tasks2queue : List of str
        List of functions to run on a remote queue (Default value = FW_DEFAULTS.tasks2queue)
    gss_auth : bool
        True if GSS_API should be used to connect to the remote machine (Default value = False)
    controlpath : str
        path the the socket file for MUX connections (Default value = None)
    remote_host : list of str
        list of hosts to attempt to connect to (Default value = FW_DEFAULTS.remote_host)
    remote_config_dir : list of str
        list of directories on the remote machines to findFireWorks configuration files (Default value = FW_DEFAULTS.remote_config_dir)
    remote_user : str
        username for the remote account (Default value = FW_DEFAULTS.remote_user)
    remote_password : str or None
        Password for access to the remote account (Default value = FW_DEFAULTS.remote_password)
    remote_shell : str
        Type of shell on the remote machine (Default value = "/bin/bash -l -c")
    daemon : int
        Daemon mode. Command is repeated every x seconds. Defaults non-daemon mode
    remote_recover_offline :
        (Default value = False)

    Raises
    ------
    RuntimeError
        If launch is not successful
    AttributeError
        If no FireWorker or Queue Adapter is specified
    ValueError
        If the launch directory does not exist

    """
    if tasks2queue is None:
        tasks2queue = [""]
    if remote_config_dir is None:
        remote_config_dir = ["~/.fireworks"]
    r_args = [launchpad]
    r_kwargs = {"fworker": fworker, "strm_lvl": strm_lvl, "pdb_on_exception": False}
    if isinstance(wflow, list):
        wflow_id = wflow
        if not wflow_id:
            wflow_id = None
            wflow = None
        else:
            wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
    elif wflow:
        wflow_id = wflow.root_firework_ids
    else:
        wflow_id = None
    if remote_host == "localhost":
        qlaunch = launch_rocket_to_queue
        remote = False
        if not fworker or not qadapter:
            raise AttributeError(
                "For a direct launch_rocket_to_queue fworker and qadapter "
                + "need to be specified"
            )
        q_args = [launchpad, fworker, qadapter]
        q_kwargs = {
            "launcher_dir": launch_dir,
            "reserve": reserve,
            "strm_lvl": strm_lvl,
            "create_launcher_dir": True,
            "fill_mode": fill_mode,
        }
    else:
        qlaunch = qlaunch_remote
        remote = True
        q_args = ["rapidfire"]
        q_kwargs = {
            "reserve": reserve,
            "gss_auth": gss_auth,
            "controlpath": controlpath,
            "remote_host": remote_host,
            "remote_config_dir": remote_config_dir,
            "remote_user": remote_user,
            "remote_password": remote_password,
            "remote_shell": remote_shell,
            "daemon": daemon,
            "maxjobs_queue": njobs_queue,
            "nlaunches": 1,
            "sleep": sleep_time,
        }
        if launch_dir != ".":
            q_kwargs["launcher_dir"] = launch_dir
        if strm_lvl != "INFO":
            q_kwargs["loglvl"] = strm_lvl
        if njobs_block != 500:
            q_kwargs["maxjobs_block"] = njobs_block
    if firework_ids and len(firework_ids) != nlaunches:
        talk(
            "WARNING: Setting nlaunches to the length of firework_ids.",
            prefix="fireworks",
        )
        nlaunches = len(firework_ids)
    sleep_time = sleep_time if sleep_time else RAPIDFIRE_SLEEP_SECS
    launch_dir = os.path.abspath(launch_dir)
    nlaunches = -1 if nlaunches == "infinite" else int(nlaunches)
    l_logger = get_fw_logger(
        "queue.launcher", l_dir=launchpad.logdir, stream_level=strm_lvl
    )

    # make sure launch_dir exists:
    if not os.path.exists(launch_dir):
        raise ValueError(
            "Desired launch directory {} does not exist!".format(launch_dir)
        )

    num_launched = 0
    start_time = datetime.now()
    skip_check = False
    try:
        if remote_host == "localhost":
            l_logger.info("getting queue adapter")

            prev_blocks = sorted(
                glob.glob(os.path.join(launch_dir, "block_*")), reverse=True
            )
            if prev_blocks and not ALWAYS_CREATE_NEW_BLOCK:
                block_dir = os.path.abspath(os.path.join(launch_dir, prev_blocks[0]))
                l_logger.info("Found previous block, using {}".format(block_dir))
            else:
                block_dir = create_datestamp_dir(launch_dir, l_logger)
            q_kwargs["launcher_dir"] = block_dir
        while True:
            if remote_recover_offline:
                for host in remote_host:
                    connect_kwargs = {"password": remote_password, "gss_auth": gss_auth}
                    if SSH_MULTIPLEXING:
                        controlpath = str(
                            Path.home()
                            / ".ssh"
                            / "sockets"
                            / (remote_user + "@$" + host + "-22")
                        )
                        connect_kwargs["controlpath"] = controlpath
                    try:
                        with fabric.Connection(
                            host=host,
                            user=remote_user,
                            config=fabric.Config({"run": {"shell": remote_shell}}),
                            connect_kwargs=connect_kwargs,
                        ) as conn:
                            for remote in remote_config_dir:
                                remote = os.path.expanduser(remote)
                                with conn.cd(remote):
                                    conn.run("lpad recover_offline")
                    except Exception:
                        pass
            if remote_host == "localhost":
                # get number of jobs in queue
                jobs_in_queue = _get_number_of_jobs_in_queue(
                    qadapter, njobs_queue, l_logger
                )
            if wflow_id:
                wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                nlaunches = len(wflow.fws)
                firework_ids = get_ready_firework_ids(wflow)
            while (
                skip_check
                or launchpad.run_exists(fworker, ids=firework_ids)
                or (fill_mode and not reserve)
            ):
                if timeout and (datetime.now() - start_time).total_seconds() >= timeout:
                    l_logger.info("Timeout reached.")
                    break
                if remote_host == "localhost":
                    if njobs_queue and jobs_in_queue >= njobs_queue:
                        l_logger.info(
                            "Jobs in queue ({}) meets/exceeds "
                            "maximum allowed ({})".format(jobs_in_queue, njobs_queue)
                        )
                        break
                    l_logger.info("Launching a rocket!")

                    # switch to new block dir if it got too big
                    if _njobs_in_dir(block_dir) >= njobs_block:
                        l_logger.info(
                            "Block got bigger than {} jobs.".format(njobs_block)
                        )
                        block_dir = create_datestamp_dir(launch_dir, l_logger)
                return_code = None
                # launch a single job
                if firework_ids or wflow_id:
                    fw_id = firework_ids[0]
                else:
                    fw_id = launchpad._get_a_fw_to_run(
                        fworker.query, fw_id=None, checkout=False
                    ).fw_id
                talk(f"Running FireWork with ID: {fw_id}", prefix="fireworks")
                use_queue = use_queue_launch(launchpad.get_fw_by_id(fw_id), tasks2queue)
                if use_queue:
                    rlaunch = qlaunch
                    args = q_args
                    kwargs = q_kwargs
                    if remote_host == "localhost":
                        kwargs["fw_id"] = fw_id
                    else:
                        kwargs["firework_ids"] = [fw_id]
                else:
                    rlaunch = launch_rocket
                    args = r_args
                    kwargs = r_kwargs
                    kwargs["fw_id"] = fw_id
                return_code = rlaunch(*args, **kwargs)
                if wflow_id:
                    wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                    nlaunches = len(wflow.fws)
                    firework_ids = get_ready_firework_ids(wflow)
                if use_queue:
                    num_launched += 1
                elif return_code is None:
                    l_logger.info("No READY jobs detected...")
                    break
                elif not return_code:
                    raise RuntimeError("Launch unsuccessful!")
                else:
                    num_launched += 1
                if nlaunches > 0 and num_launched == nlaunches:
                    l_logger.info(
                        "Launched allowed number of " "jobs: {}".format(num_launched)
                    )
                    break
                # wait for the queue system to update
                if remote_host == "localhost":
                    l_logger.info(
                        "Sleeping for {} seconds...zzz...".format(QUEUE_UPDATE_INTERVAL)
                    )
                    time.sleep(QUEUE_UPDATE_INTERVAL)
                skip_check = not use_queue and launchpad.run_exists(
                    fworker, ids=firework_ids
                )
            if (
                (nlaunches > 0 and num_launched == nlaunches)
                or (
                    timeout and (datetime.now() - start_time).total_seconds() >= timeout
                )
                or (nlaunches == 0 and not launchpad.future_run_exists(fworker))
            ):
                break

            l_logger.info(
                "Finished a round of launches, sleeping for {} secs".format(sleep_time)
            )
            time.sleep(sleep_time)
            l_logger.info("Checking for Rockets to run...")
    except Exception:
        log_exception(l_logger, "Error with queue launcher rapid fire!")
