"""This module contains a modified rapid-fire mode.

FireWorks Copyright (c) 2013, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""
# coding: utf-8
from __future__ import unicode_literals

import os
import time
from datetime import datetime

from fireworks.core.rocket_launcher import get_fworker, launch_rocket
from fireworks.fw_config import RAPIDFIRE_SLEEP_SECS
from fireworks.utilities.fw_utilities import (
    create_datestamp_dir,
    get_fw_logger,
    log_multi,
    redirect_local,
)
from vibes.helpers import talk

from .combined_launcher import get_ready_firework_ids


__author__ = "Anubhav Jain, Modified by Thomas Purcell Nov 2, 2018"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Anubhav Jain"
__email__ = "ajain@lbl.gov"
__date__ = "Feb 22, 2013"


def rapidfire(
    launchpad,
    fworker=None,
    m_dir=None,
    nlaunches=0,
    max_loops=-1,
    sleep_time=None,
    strm_lvl="CRITICAL",
    timeout=None,
    local_redirect=False,
    pdb_on_exception=False,
    firework_ids=None,
    wflow_id=None,
):
    """Keeps running Rockets in m_dir until we reach an error.

    Automatically creates subdirectories for each Rocket. Usually stops when we
    run out of FireWorks from the LaunchPad.

    Parameters
    ----------
    launchpad : LaunchPad
        LaunchPad for the launch
    fworker : FWorker
        FireWorker for the launch (Default value = None)
    m_dir : str
        the directory in which to loop Rocket running (Default value = None)
    nlaunches : int
        0 means 'until completion', -1 or "infinite" means to loop until max_loops (Default value = 0)
    max_loops : int
        maximum number of loops (default -1 is infinite)
    sleep_time : int
        secs to sleep between rapidfire loop iterations (Default value = None)
    strm_lvl : str
        level at which to output logs to stdout (Default value = "CRITICAL")
    timeout : int
        of seconds after which to stop the rapidfire process (Default value = None)
    local_redirect : bool
        redirect standard input and output to local file (Default value = False)
    pdb_on_exception : bool
        if set to True, python will start the debugger on a firework exception (Default value = False)
    firework_ids : list of ints
        list of FireWorks to run (Default value = None)
    wflow_id : list of ints
        list of ids of the root nodes of a workflow (Default value = None)

    """
    if firework_ids and len(firework_ids) != nlaunches:
        talk(
            "WARNING: Setting nlaunches to the length of firework_ids.",
            prefix="fireworks",
        )
        nlaunches = len(firework_ids)
    sleep_time = sleep_time if sleep_time else RAPIDFIRE_SLEEP_SECS
    curdir = m_dir if m_dir else os.getcwd()
    l_logger = get_fw_logger(
        "rocket.launcher", l_dir=launchpad.get_logdir(), stream_level=strm_lvl
    )
    nlaunches = -1 if nlaunches == "infinite" else int(nlaunches)
    fworker = get_fworker(fworker)

    num_launched = 0
    start_time = datetime.now()
    num_loops = 0

    def time_ok():
        """Determines if the rapidfire run has timed out"""
        return (
            timeout is None or (datetime.now() - start_time).total_seconds() < timeout
        )

    while num_loops != max_loops and time_ok():
        skip_check = False  # this is used to speed operation
        while (
            skip_check or launchpad.run_exists(fworker, ids=firework_ids)
        ) and time_ok():
            if wflow_id:
                wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                nlaunches = len(wflow.fws)
                firework_ids = get_ready_firework_ids(wflow)
            os.chdir(curdir)
            launcher_dir = create_datestamp_dir(curdir, l_logger, prefix="launcher_")
            os.chdir(launcher_dir)
            if local_redirect:
                with redirect_local():
                    if firework_ids or wflow_id:
                        rocket_ran = launch_rocket(
                            launchpad,
                            fworker,
                            strm_lvl=strm_lvl,
                            pdb_on_exception=pdb_on_exception,
                            fw_id=firework_ids[0],
                        )
                    else:
                        rocket_ran = launch_rocket(
                            launchpad,
                            fworker,
                            strm_lvl=strm_lvl,
                            pdb_on_exception=pdb_on_exception,
                        )
            else:
                if firework_ids or wflow_id:
                    rocket_ran = launch_rocket(
                        launchpad,
                        fworker,
                        strm_lvl=strm_lvl,
                        pdb_on_exception=pdb_on_exception,
                        fw_id=firework_ids[0],
                    )
                else:
                    rocket_ran = launch_rocket(
                        launchpad,
                        fworker,
                        strm_lvl=strm_lvl,
                        pdb_on_exception=pdb_on_exception,
                    )
            if wflow_id:
                wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                nlaunches = len(wflow.fws)
                firework_ids = get_ready_firework_ids(wflow)
            if rocket_ran:
                num_launched += 1
            elif not os.listdir(launcher_dir):
                # remove the empty shell of a directory
                os.chdir(curdir)
                os.rmdir(launcher_dir)
            if nlaunches > 0 and num_launched == nlaunches:
                break
            if launchpad.run_exists(fworker, ids=firework_ids):
                skip_check = True  # don't wait, pull the next FW right away
            else:
                # add a small amount of buffer breathing time for DB to refresh
                time.sleep(0.15)
                skip_check = False
        if nlaunches == 0:
            if not launchpad.future_run_exists(fworker, ids=firework_ids):
                break
        elif num_launched == nlaunches:
            break
        log_multi(l_logger, "Sleeping for {} secs".format(sleep_time))
        time.sleep(sleep_time)
        num_loops += 1
        log_multi(l_logger, "Checking for FWs to run...")
    os.chdir(curdir)
