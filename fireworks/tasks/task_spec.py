"""Defines a TaskSpec object"""
from pathlib import Path


class TaskSpec:
    """Class used to define a task specification in a standardized way"""

    def __init__(
        self,
        func,
        func_fw_out,
        task_with_atoms_obj,
        func_kwargs=None,
        func_fw_out_kwargs=None,
        args=None,
        inputs=None,
        make_abs_path=False,
        fw_settings=None,
    ):
        """TaskSpec Constructor

        Parameters
        ----------
        func : str or Function
            Function to be wrapped into a PyTask
        func_fw_out : str or Function
            Function that converts the outputs of func into FWActions
        task_with_atoms_obj : bool
            True if calculating using an ASE Atoms object
        func_kwargs : dict
            kwargs for func
        func_fw_out_kwargs : dict
            kwargs for func_fw_out
        args : list
            args for func
        inputs : list of str
            spec_keys for args for func stored in the FireWorks database
        make_abs_path : bool
            If True make all paths absolute
        fw_settings : dict
            The FireWorks specific settings

        """
        if not isinstance(func, str):
            func = f"{func.__module__}.{func.__name__}"
        if not isinstance(func_fw_out, str):
            func_fw_out = f"{func_fw_out.__module__}.{func_fw_out.__name__}"
        self.func = func
        self.func_fw_out = func_fw_out
        self.task_with_atoms_obj = task_with_atoms_obj

        self.fw_settings = fw_settings
        if func_kwargs:
            if "workdir" in func_kwargs and make_abs_path:
                func_kwargs["workdir"] = str(Path(func_kwargs["workdir"]).absolute())
            self.func_kwargs = func_kwargs
        else:
            self.func_kwargs = {}

        if func_fw_out_kwargs:
            if "workdir" in func_fw_out_kwargs and make_abs_path:
                func_fw_out_kwargs["workdir"] = str(
                    Path(func_fw_out_kwargs["workdir"]).absolute()
                )
            self.func_fw_out_kwargs = func_fw_out_kwargs
        else:
            self.func_fw_out_kwargs = {}

        if args:
            self.args = args
        else:
            self.args = []

        if inputs:
            self._inputs = inputs
        else:
            self._inputs = []

        if task_with_atoms_obj:
            self._pt_args = [
                self.func,
                self.func_fw_out,
                self.func_kwargs,
                self.func_fw_out_kwargs,
                *self.args,
            ]
        else:
            self._pt_args = [self.func, self.func_fw_out, *self.args]

    @property
    def pt_args(self):
        """get the PyTask args for the task"""
        return self._pt_args

    @property
    def fw_settings(self):
        """get the fw_settings for the PyTask"""
        return self._fw_settings

    @fw_settings.setter
    def fw_settings(self, fw_set):
        """set the task_specs fw_settings

        Parameters
        ----------
        fw_set : dict
            FireWork settings to be added to the task spec

        Returns
        -------

        """
        self._fw_settings = fw_set

    @property
    def pt_kwargs(self):
        """get the PyTask kwargs for the task"""
        if not self.fw_settings:
            self.fw_settings = {}

        if self.task_with_atoms_obj:
            return {"fw_settings": self.fw_settings}

        to_ret = dict(self.func_kwargs, **self.func_fw_out_kwargs)
        to_ret["fw_settings"] = self.fw_settings

        return to_ret

    @property
    def pt_inputs(self):
        """get the PyTask inputs for the task"""
        return self._inputs
