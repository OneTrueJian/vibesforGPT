"""Python API for qlaunch to connect to remote hosts

FireWorks Copyright (c) 2013, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""
import os
import time

from vibes.helpers import talk


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

__authors__ = "Anubhav Jain, Shyue Ping Ong, Adapted by Thomas Purcell for use in HiLDe"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Anubhav Jain"
__email__ = "ajain@lbl.gov"
__date__ = "Jan 14, 2013, Adaptation: October 31, 2018"


def convert_input_to_param(param_name, param, param_list):
    """Converts a function input into a qlaunch parameter

    Parameters
    ----------
    param_name : str
        name of the parameter for qlaunch
    param : str
        value of the parameter
    param_list : list of str
        List of all parameters

    Returns
    -------
    str
        command argument str for qlaunch

    """
    if param is not None:
        param_list.append(f"--{param_name} {param}")
    return param_list


def qlaunch_remote(
    command,
    maxjobs_queue=None,
    maxjobs_block=None,
    nlaunches=None,
    sleep=None,
    firework_ids=None,
    fw_id=None,
    wflow=None,
    silencer=False,
    reserve=False,
    launcher_dir=None,
    loglvl=None,
    gss_auth=True,
    controlpath=None,
    remote_host="localhost",
    remote_config_dir=None,
    remote_user=None,
    remote_password=None,
    remote_shell="/bin/bash -l -c",
    daemon=0,
):
    """This function adapts the fireworks script qlaunch to a python function

    Parameters
    ----------
    command : str
        Whether to do a singleshot or rapidfire command
    maxjobs_queue : int
        maximum jobs to keep in queue for this user (Default value = None)
    maxjobs_block : int
        maximum jobs to put in a block (Default value = None)
    nlaunches : int or "infinite"
        maximum number of launches to perform (default 0 is all jobs in DB)
    sleep : int
        sleep time between loops (Default value = None)
    firework_ids : list of int
        specific firework_ids to run in reservation mode (Default value = None)
    fw_id : int
        ID of a specific FireWork to run in reservation mode (Default value = None)
    wflow : list of int or Workflow
        specific Workflow to run in reservation mode (Default value = None)
    silencer : bool
        shortcut to mute log messages (Default value = False)
    reserve : bool
        reserve a fw (Default value = False)
    launcher_dir : str
        Directory to launch rocket from (Default value = None)
    loglvl : str
        How much logging should occur (Default value = None)
    gss_auth : bool
        Allow GSS-API authorization with Kerberos (Default value = True)
    controlpath : str
        Path to a control path for ssh multiplexing (Only possible in modified paramiko)
        (Default value = None)
    remote_host : str
        Remote host to exec qlaunch. Right now, only supports running from a config dir.
        (Default value = "localhost")
    remote_config_dir : list of str
        Remote config dir location(s). Defaults to ~/.fireworks.
    remote_user : str
        Username to login to remote host. (Default value = None)
    remote_password : str
        Password for remote host (if necessary and not recommended)
        (Default value = None)
    remote_shell : str
        Shell command to use on remote host for running submission.
        (Default value = "/bin/bash -l -c")
    daemon : int
        Daemon mode. Command is repeated every x seconds. Defaults to non-daemon mode.

    Raises
    ------
    ImportError
        If Fabric v2+ is not installed
    AssertionError
        If remote_host is localhost

    """
    assert remote_host != "localhost"

    if not HAS_FABRIC:
        raise ImportError(
            "Remote options require the Fabric package v2+ to be installed!"
        )
    if remote_config_dir is None:
        remote_config_dir = ["~/.fireworks"]
    non_default = []
    convert_input_to_param("launch_dir", launcher_dir, non_default)
    if command == "rapidfire":
        convert_input_to_param("maxjobs_queue", maxjobs_queue, non_default)
        convert_input_to_param("maxjobs_block", maxjobs_block, non_default)
        convert_input_to_param("nlaunches", nlaunches, non_default)
        convert_input_to_param("sleep", sleep, non_default)

        if firework_ids:
            non_default.append("--{} {}".format("firework_ids", firework_ids[0]))
            for fire_work in firework_ids[1:]:
                non_default[-1] += " {}".format(fire_work)
                fw_id = fire_work
        if wflow:
            non_default.append("--{} {}".format("wflow", wflow.root_firework_ids[0]))
            for fire_work in wflow.root_firework_ids[1:]:
                non_default[-1] += " {}".format(fire_work)
    else:
        convert_input_to_param("fw_id", fw_id, non_default)
    non_default = " ".join(non_default)

    pre_non_default = []
    if silencer:
        pre_non_default.append("--silencer")
    if reserve:
        pre_non_default.append("--reserve")
    pre_non_default = " ".join(pre_non_default)
    interval = daemon
    while True:
        for host in remote_host:
            connect_kwargs = {"password": remote_password, "gss_auth": gss_auth}
            if SSH_MULTIPLEXING:
                connect_kwargs["controlpath"] = controlpath
            talk(
                f"Launching FireWork {fw_id} on {host} with command: \n"
                + f"\tvibes fireworks qlaunch {pre_non_default} {command} {non_default}",
                prefix="fireworks",
            )
            with fabric.Connection(
                host=host,
                user=remote_user,
                config=fabric.Config({"run": {"shell": remote_shell}}),
                connect_kwargs=connect_kwargs,
            ) as conn:
                for remote in remote_config_dir:
                    remote = os.path.expanduser(remote)
                    with conn.cd(remote):
                        conn.run(
                            "vibes fireworks qlaunch {} {} {}".format(
                                pre_non_default, command, non_default
                            )
                        )
        if interval > 0:
            talk(
                "Next run in {} seconds... Press Ctrl-C to exit at any "
                "time.".format(interval),
                prefix="fireworks",
            )
            time.sleep(daemon)
        else:
            break
