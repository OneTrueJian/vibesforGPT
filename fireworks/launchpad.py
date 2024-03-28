""" Modified Launchpad class from FireWorks

FireWorks Copyright (c) 2013, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject
to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.
"""
from fireworks.core import launchpad
from fireworks.fw_config import LAUNCHPAD_LOC


class LaunchPad(launchpad.LaunchPad):
    """The modified Launchpad that manges the FireWorks database"""

    def __init__(self, *args, **kwargs):
        super(LaunchPad, self).__init__(*args, **kwargs)

    def run_exists(self, fworker=None, ids=None):
        """Checks to see if the database has any FireWorks ready to run in a given set

        Parameters
        ----------
        fworker : FWorker
            FireWorker for the query (Default value = None)
        ids : list of ints
            List of FireWork ids to query over (Default value = None)

        Returns
        -------
        bool
            True if the database has any FireWorks that are ready to run in a given set.

        """
        query = fworker.query if fworker else {}
        if ids:
            query["fw_id"] = {"$in": ids}
        return bool(self._get_a_fw_to_run(query=query, checkout=False))

    def future_run_exists(self, fworker=None, ids=None):
        """heck if database has any current OR future Fireworks available

        Parameters
        ----------
        fworker : FWorker
            FireWorker for the query (Default value = None)
        ids : list of ints
            (Default value = None)

        Returns
        -------
        bool
            True if database has any ready or waiting Fireworks.

        """
        if self.run_exists(fworker, ids):
            # check first to see if any are READY
            return True
        # retrieve all [RUNNING/RESERVED] fireworks
        q = fworker.query if fworker else {}
        if ids:
            q["fw_id"] = {"$in": ids}
        q.update({"state": {"$in": ["RUNNING", "RESERVED"]}})
        active = self.get_fw_ids(q)
        # then check if they have WAITING children
        for fw_id in active:
            children = self.get_wf_by_fw_id_lzyfw(fw_id).links[fw_id]
            if any(self.get_fw_dict_by_id(i)["state"] == "WAITING" for i in children):
                return True
        return False

    @classmethod
    def from_dict(cls, d):
        """Constructs a LaunchPad from a dict

        Parameters
        ----------
        cls : Class
            The class of the LaunchPad
        d : dict
            Dictionary used to define the LaunchPad

        Returns
        -------
        LaunchPad
            The LaunchPad defined by the dict

        """
        port = d.get("port", None)
        name = d.get("name", None)
        username = d.get("username", None)
        password = d.get("password", None)
        logdir = d.get("logdir", None)
        strm_lvl = d.get("strm_lvl", None)
        user_indices = d.get("user_indices", [])
        wf_user_indices = d.get("wf_user_indices", [])
        authsource = d.get("authsource", None)
        uri_mode = d.get("uri_mode", False)
        mongoclient_kwargs = d.get("mongoclient_kwargs", None)
        return LaunchPad(
            d["host"],
            port,
            name,
            username,
            password,
            logdir,
            strm_lvl,
            user_indices,
            wf_user_indices,
            authsource,
            uri_mode,
            mongoclient_kwargs,
        )

    @classmethod
    def auto_load(cls):
        """auto_load from default file

        Parameters
        ----------
        cls : Class
            The class of the LaunchPad

        Returns
        -------
        LaunchPad
            The LaunchPad defined in LAUNCHPAD_LOC or the default LaunchPad

        """
        if LAUNCHPAD_LOC:
            return LaunchPad.from_file(LAUNCHPAD_LOC)
        return LaunchPad()
