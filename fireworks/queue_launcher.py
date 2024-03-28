"""A modified version of queue launcher to allow for a rapidfire over a single Workflow

FireWorks Copyright (c) 2013, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""
import glob
import os
import time
from datetime import datetime

import numpy as np
from fireworks import FWorker
from fireworks.fw_config import (
    ALWAYS_CREATE_NEW_BLOCK,
    QSTAT_FREQUENCY,
    QUEUE_JOBNAME_MAXLEN,
    QUEUE_UPDATE_INTERVAL,
    RAPIDFIRE_SLEEP_SECS,
    SUBMIT_SCRIPT_NAME,
)
from fireworks.queue.queue_launcher import (  # launch_rocket_to_queue,
    _get_number_of_jobs_in_queue,
    _njobs_in_dir,
    setup_offline_job,
)
from fireworks.utilities.fw_serializers import load_object
from fireworks.utilities.fw_utilities import (
    create_datestamp_dir,
    get_fw_logger,
    get_slug,
    log_exception,
)
from monty.os import cd, makedirs_p
from vibes.fireworks._defaults import FW_DEFAULTS
from vibes.fireworks.combined_launcher import get_ready_firework_ids
from vibes.helpers import talk
from vibes.helpers.warnings import warn
from vibes.helpers.watchdogs import str2time


__author__ = "Anubhav Jain, Michael Kocher, Modified by Thomas Purcell"
__copyright__ = "Copyright 2012, The Materials Project, Modified 2.11.2018"
__version__ = "0.1"
__maintainer__ = "Anubhav Jain"
__email__ = "ajain@lbl.gov"
__date__ = "Dec 12, 2012"


def launch_rocket_to_queue(
    launchpad,
    fworker,
    qadapter,
    launcher_dir=FW_DEFAULTS.launch_dir,
    reserve=False,
    strm_lvl="INFO",
    create_launcher_dir=False,
    fill_mode=False,
    fw_id=None,
):
    """Submit a single job to the queue.

    Parameters
    ----------
    launchpad : LaunchPad
        LaunchPad for the launch
    fworker : FWorker
        FireWorker for the launch
    qadapter : QueueAdapterBase
        Queue Adapter for the resource
    launcher_dir : str
        The directory where to submit the job (Default value = FW_DEFAULTS.launch_dir)
    reserve : bool
        Whether to queue in reservation mode (Default value = False)
    strm_lvl : str
        level at which to stream log messages (Default value = "INFO")
    create_launcher_dir : bool
        Whether to create a subfolder launcher+timestamp, if needed (Default value = False)
    fill_mode : bool
        If True submit jobs even if there are none to run (only in non-reservation mode) (Default value = False)
    fw_id : int
        specific fw_id to reserve (reservation mode only) (Default value = None)

    Raises
    ------
    RuntimeError
        If launch is not successful OR
        If queue script could not be submitted
    ValueError
        If the launch directory does not exist OR
        If in offline mode and not reservation mode OR
        If in Reservation Mode and not using a singleshot RocketLauncher OR
        If in Reservation Mode and Fill Mode is also requested OR
        If asking to launch a particular FireWork and not in Reservation Mode

    """
    fworker = fworker if fworker else FWorker()
    launcher_dir = os.path.abspath(launcher_dir)
    l_logger = get_fw_logger(
        "queue.launcher", l_dir=launchpad.logdir, stream_level=strm_lvl
    )

    l_logger.debug("getting queue adapter")
    qadapter = load_object(
        qadapter.to_dict()
    )  # make a defensive copy, mainly for reservation mode

    fw, launch_id = None, None  # only needed in reservation mode
    if not os.path.exists(launcher_dir):
        raise ValueError(
            "Desired launch directory {} does not exist!".format(launcher_dir)
        )

    if "--offline" in qadapter["rocket_launch"] and not reserve:
        raise ValueError(
            "Must use reservation mode (-r option) of qlaunch "
            "when using offline option of rlaunch!!"
        )

    if reserve and "singleshot" not in qadapter.get("rocket_launch", ""):
        raise ValueError(
            "Reservation mode of queue launcher only works for singleshot Rocket Launcher"
        )

    if fill_mode and reserve:
        raise ValueError("Fill_mode cannot be used in conjunction with reserve mode!")

    if fw_id and not reserve:
        raise ValueError(
            "qlaunch for specific fireworks may only be used in reservation mode."
        )
    if fill_mode or launchpad.run_exists(fworker):
        launch_id = None
        try:
            if reserve:
                if fw_id:
                    l_logger.debug("finding a FW to reserve...")
                fw = launchpad._get_a_fw_to_run(
                    fworker.query, fw_id=fw_id, checkout=False
                )
                fw_id = fw.fw_id
                if not fw:
                    l_logger.info(
                        "No jobs exist in the LaunchPad for submission to queue!"
                    )
                    return False
                l_logger.info("reserved FW with fw_id: {}".format(fw.fw_id))
                # update qadapter job_name based on FW name
                job_name = get_slug(fw.name)[0:QUEUE_JOBNAME_MAXLEN]
                qadapter.update({"job_name": job_name})
                if "_queueadapter" in fw.spec:
                    l_logger.debug("updating queue params using Firework spec..")
                    ntasks = fw.spec["_queueadapter"].pop("ntasks", 1)
                    nodes = int(np.ceil(ntasks / qadapter["ntasks_per_node"]))
                    if nodes < fw.spec["_queueadapter"].get("nodes", 1):
                        nodes = fw.spec["_queueadapter"].get("nodes", 1)
                    if "queues" in qadapter:
                        nodes_needed = []
                        if "expected_mem" in fw.spec["_queueadapter"]:
                            expect_mem = fw.spec["_queueadapter"]["expected_mem"]
                        else:
                            expect_mem = 1e-10
                        for queue in qadapter["queues"]:
                            if "max_mem" in queue:
                                accessible_mem = queue["max_mem"]
                                if expect_mem > accessible_mem:
                                    nodes_needed.append(
                                        int(np.ceil(expect_mem / accessible_mem))
                                    )
                                else:
                                    nodes_needed.append(nodes)
                            else:
                                nodes_needed.append(nodes)
                        if (
                            "queue" in fw.spec["_queueadapter"]
                            and "walltime" in fw.spec["_queueadapter"]
                        ):
                            queue_ind = -1
                            for ii, qu in enumerate(qadapter["queues"]):
                                if qu["name"] == fw.spec["_queueadapter"]["queue"]:
                                    queue = qu
                                    queue_ind = ii
                            if queue_ind < 0:
                                raise ValueError(
                                    "Queue Name not found for that resources"
                                )
                            if queue["max_nodes"] < nodes_needed[queue_ind]:
                                raise IOError(
                                    "Requested resource does not have enough memory"
                                )
                            sc_wt = str2time(fw.spec["_queueadapter"]["walltime"])
                            if nodes * sc_wt < str2time(queue["max_walltime"]):
                                fw.spec["_queueadapter"]["nodes"] = nodes
                            else:
                                fw.spec["_queueadapter"]["nodes"] = int(
                                    np.ceil(sc_wt / str2time(queue["max_walltime"]))
                                )
                                fw.spec["_queueadapter"]["walltime"] = queue[
                                    "max_walltime"
                                ]
                        if "queue" in fw.spec["_queueadapter"]:
                            queue_ind = -1
                            for ii, qu in enumerate(qadapter["queues"]):
                                if qu["name"] == fw.spec["_queueadapter"]["queue"]:
                                    queue = qu
                                    queue_ind = ii
                            if queue_ind < 0:
                                raise ValueError(
                                    "Queue Name not found for that resources"
                                )
                            if queue["max_nodes"] < nodes_needed[queue_ind]:
                                warn("Requested resource might not have enough memory")
                                fw.spec["_queueadapter"]["nodes"] = queue["max_nodes"]

                            if "walltime" not in fw.spec["_queueadapter"]:
                                fw.spec["_queueadapter"]["walltime"] = queue[
                                    "max_walltime"
                                ]
                        else:
                            if "walltime" in fw.spec["_queueadapter"]:
                                sc_wt = str2time(fw.spec["_queueadapter"]["walltime"])
                            else:
                                sc_wt = 1800
                            queue_wt = 1.0e15
                            queue = None
                            queue_ind = -1
                            for ii, qu in enumerate(qadapter["queues"]):
                                if nodes_needed[ii] > qu["max_nodes"]:
                                    continue
                                max_comp_time = (
                                    str2time(qu["max_walltime"]) * qu["max_nodes"]
                                )
                                if queue_wt > max_comp_time > sc_wt:
                                    queue_wt = max_comp_time
                                    queue = qu
                                    queue_ind = ii
                            if queue is None:
                                warn("Job may not run on requested resource")
                                queue = qu
                                queue_ind = ii 
                            fw.spec["_queueadapter"]["queue"] = queue["name"]
                            if sc_wt < str2time(queue["max_walltime"]):
                                fw.spec["_queueadapter"]["nodes"] = nodes
                            else:
                                fw.spec["_queueadapter"]["nodes"] = int(
                                    np.ceil(sc_wt / str2time(queue["max_walltime"]))
                                )
                                fw.spec["_queueadapter"]["walltime"] = queue[
                                    "max_walltime"
                                ]
                    if nodes_needed[queue_ind] > fw.spec["_queueadapter"]["nodes"]:
                        fw.spec["_queueadapter"]["nodes"] = nodes_needed[queue_ind]

                    if fw.spec["_queueadapter"]["nodes"] > queue["max_nodes"]:
                        warn("Requested nodes too large, reducing them now")
                        fw.spec["_queueadapter"]["nodes"] = queue["max_nodes"]
                    sc_wt = str2time(fw.spec["_queueadapter"]["walltime"])
                    if sc_wt > str2time(queue["max_walltime"]):
                        warn("Requested walltime too long, reducing it now")
                        fw.spec["_queueadapter"]["walltime"] = queue["max_walltime"]

                    del qadapter["queues"]
                    qadapter.update(fw.spec["_queueadapter"])
                    qadapter.pop("ntasks", 1)

                if "walltime" in qadapter:
                    for tt, task in enumerate(fw.spec["_tasks"]):
                        if ("kwargs" in task and "walltime" in task["kwargs"]) or (
                            "calculate_wrapper" in task["args"][0]
                        ):
                            fw.spec["_tasks"][tt]["kwargs"]["walltime"] = (
                                str2time(qadapter["walltime"]) - 180.0
                            )

                launchpad.update_spec([fw.fw_id], fw.spec)
                fw, launch_id = launchpad.reserve_fw(fworker, launcher_dir, fw_id=fw_id)

                # reservation mode includes --fw_id in rocket launch
                qadapter["rocket_launch"] += " --fw_id {}".format(fw.fw_id)

                # update launcher_dir if _launch_dir is selected in reserved fw
                if "_launch_dir" in fw.spec:
                    fw_launch_dir = os.path.expandvars(fw.spec["_launch_dir"])

                    if not os.path.isabs(fw_launch_dir):
                        fw_launch_dir = os.path.join(launcher_dir, fw_launch_dir)

                    launcher_dir = fw_launch_dir

                    makedirs_p(launcher_dir)

                    launchpad.change_launch_dir(launch_id, launcher_dir)
                elif create_launcher_dir:
                    # create launcher_dir
                    launcher_dir = create_datestamp_dir(
                        launcher_dir, l_logger, prefix="launcher_"
                    )
                    launchpad.change_launch_dir(launch_id, launcher_dir)

            elif create_launcher_dir:
                # create launcher_dir
                launcher_dir = create_datestamp_dir(
                    launcher_dir, l_logger, prefix="launcher_"
                )

            # move to the launch directory
            l_logger.info("moving to launch_dir {}".format(launcher_dir))

            with cd(launcher_dir):

                if "--offline" in qadapter["rocket_launch"]:
                    setup_offline_job(launchpad, fw, launch_id)

                l_logger.debug("writing queue script")
                with open(SUBMIT_SCRIPT_NAME, "w") as f:
                    queue_script = qadapter.get_script_str(launcher_dir)
                    f.write(queue_script)

                l_logger.info("submitting queue script")
                reservation_id = qadapter.submit_to_queue(SUBMIT_SCRIPT_NAME)
                if not reservation_id:
                    raise RuntimeError(
                        "queue script could not be submitted, check queue "
                        "script/queue adapter/queue server status!"
                    )
                if reserve:
                    launchpad.set_reservation_id(launch_id, reservation_id)
            return reservation_id
        except Exception:
            log_exception(l_logger, "Error writing/submitting queue script!")
            if reserve and launch_id is not None:
                try:
                    l_logger.info(
                        "Un-reserving FW with fw_id, launch_id: {}, {}".format(
                            fw.fw_id, launch_id
                        )
                    )
                    launchpad.cancel_reservation(launch_id)
                    launchpad.forget_offline(launch_id)
                except Exception:
                    log_exception(
                        l_logger, "Error unreserving FW with fw_id {}".format(fw.fw_id)
                    )

            return False

    else:
        # note: this is a hack (rather than False) to indicate a soft failure to rapidfire
        l_logger.info("No jobs exist in the LaunchPad for submission to queue!")
        return None


def rapidfire(
    launchpad,
    fworker,
    qadapter,
    launch_dir=FW_DEFAULTS.launch_dir,
    nlaunches=FW_DEFAULTS.nlaunches,
    njobs_queue=FW_DEFAULTS.njobs_queue,
    njobs_block=FW_DEFAULTS.njobs_block,
    sleep_time=FW_DEFAULTS.sleep_time,
    reserve=False,
    strm_lvl="CRITICAL",
    timeout=None,
    fill_mode=False,
    firework_ids=None,
    wflow_id=None,
):
    """Submit many jobs to the queue.

    Parameters
    ----------
    launchpad : LaunchPad
        LaunchPad for the launch
    fworker : FWorker
        FireWorker for the launch
    qadapter : QueueAdapterBase
        Queue Adapter for the resource
    launch_dir : str
        directory where we want to write the blocks (Default value = FW_DEFAULTS.launch_dir)
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
        If True submit jobs even if there is nothing to run (only in non-reservation mode) (Default value = False)
    firework_ids : list of ints
        a list firework_ids to launch (len(firework_ids) == nlaunches) (Default value = None)
    wflow_id : list of ints
        a list firework_ids that are a root of the workflow (Default value = None)

    Returns
    -------

    Raises
    ------
    ValueError
        If the luanch directory does not exist

    """
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
    try:
        l_logger.info("getting queue adapter")

        prev_blocks = sorted(
            glob.glob(os.path.join(launch_dir, "block_*")), reverse=True
        )
        if prev_blocks and not ALWAYS_CREATE_NEW_BLOCK:
            block_dir = os.path.abspath(os.path.join(launch_dir, prev_blocks[0]))
            l_logger.info("Found previous block, using {}".format(block_dir))
        else:
            block_dir = create_datestamp_dir(launch_dir, l_logger)
        while True:
            # get number of jobs in queue
            jobs_in_queue = _get_number_of_jobs_in_queue(
                qadapter, njobs_queue, l_logger
            )
            job_counter = 0  # this is for QSTAT_FREQUENCY option
            if wflow_id:
                wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                nlaunches = len(wflow.fws)
                firework_ids = get_ready_firework_ids(wflow)

            while launchpad.run_exists(fworker, ids=firework_ids) or (
                fill_mode and not reserve
            ):
                if timeout and (datetime.now() - start_time).total_seconds() >= timeout:
                    l_logger.info("Timeout reached.")
                    break

                if njobs_queue and jobs_in_queue >= njobs_queue:
                    l_logger.info(
                        "Jobs in queue ({}) meets/exceeds "
                        "maximum allowed ({})".format(jobs_in_queue, njobs_queue)
                    )
                    break

                l_logger.info("Launching a rocket!")

                # switch to new block dir if it got too big
                if _njobs_in_dir(block_dir) >= njobs_block:
                    l_logger.info("Block got bigger than {} jobs.".format(njobs_block))
                    block_dir = create_datestamp_dir(launch_dir, l_logger)
                return_code = None
                # launch a single job
                if firework_ids or wflow_id:
                    return_code = launch_rocket_to_queue(
                        launchpad,
                        fworker,
                        qadapter,
                        block_dir,
                        reserve,
                        strm_lvl,
                        True,
                        fill_mode,
                        firework_ids[0],
                    )
                else:
                    return_code = launch_rocket_to_queue(
                        launchpad,
                        fworker,
                        qadapter,
                        block_dir,
                        reserve,
                        strm_lvl,
                        True,
                        fill_mode,
                    )
                if return_code is None:
                    l_logger.info("No READY jobs detected...")
                    break
                elif not return_code:
                    raise RuntimeError("Launch unsuccessful!")

                if wflow_id:
                    wflow = launchpad.get_wf_by_fw_id(wflow_id[0])
                    nlaunches = len(wflow.fws)
                    firework_ids = get_ready_firework_ids(wflow)
                num_launched += 1
                if nlaunches > 0 and num_launched == nlaunches:
                    l_logger.info(
                        "Launched allowed number of " "jobs: {}".format(num_launched)
                    )
                    break
                # wait for the queue system to update
                l_logger.info(
                    "Sleeping for {} seconds...zzz...".format(QUEUE_UPDATE_INTERVAL)
                )
                time.sleep(QUEUE_UPDATE_INTERVAL)
                jobs_in_queue += 1
                job_counter += 1
                if job_counter % QSTAT_FREQUENCY == 0:
                    job_counter = 0
                    jobs_in_queue = _get_number_of_jobs_in_queue(
                        qadapter, njobs_queue, l_logger
                    )

            if (
                (nlaunches > 0 and num_launched == nlaunches)
                or (
                    timeout and (datetime.now() - start_time).total_seconds() >= timeout
                )
                or (
                    nlaunches == 0
                    and not launchpad.future_run_exists(fworker, ids=firework_ids)
                )
            ):
                break

            l_logger.info(
                "Finished a round of launches, sleeping for {} secs".format(sleep_time)
            )
            time.sleep(sleep_time)
            l_logger.info("Checking for Rockets to run...")
    except Exception:
        log_exception(l_logger, "Error with queue launcher rapid fire!")
