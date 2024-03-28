"""General purpose FWAction Generators"""
from fireworks import FWAction


def add_additions_to_spec(func, func_fw_out, *args, fw_settings=None, **kwargs):
    """Adds a set of returned Fireworks to the Workflow

    Parameters
    ----------
    func : str
        Path to function that performs the MD like operation
    func_fw_out : str
        Path to this function
    args : list
        List of arguments to pass to func
    fw_settings : dict
        FireWorks specific settings (Default value = None)
    kwargs : dict
        keyword arguments for func

    Returns
    -------
    firworks.FWAction
        A FWAction to add the outputted FireWorks/Workflows as additions

    """
    additions = []
    for out in kwargs["outputs"]:
        additions.append(out)

    return FWAction(additions=additions)


def fireworks_no_mods(
    atoms_dict,
    calculator_dict,
    outputs,
    func,
    func_fw_out,
    func_kwargs,
    func_fw_kwargs,
    fw_settings,
):
    """A function that does not change the FireWorks Workflow upon completion

    Parameters
    ----------
    atoms_dict : dict
        The dictionary representation of the original atoms at the start of this job
    calculator_dict : dict
        The dictionary representation of the original calculator
    outputs : any
        The outputs from the function (assumes to be a single bool output)
    func : str
        Path to function that performs the MD like operation
    func_fw_out : str
        Path to this function
    func_kwargs : dict
        keyword arguments for func
    func_fw_kwargs : dict
        Keyword arguments for fw_out function
    fw_settings : dict
        FireWorks specific settings

    Returns
    -------
    firworks.FWAction
        An empty FWAction

    """
    return FWAction()


def fireworks_no_mods_gen_function(
    func, func_fw_out, *args, fw_settings=None, **kwargs
):
    """A function that does not change the FireWorks Workflow upon completion

    Parameters
    ----------
    func : str
        Path to function that performs the MD like operation
    func_fw_out : str
        Path to this function
    args : list
        List of arguments to pass to func
    fw_settings : dict
        FireWorks specific settings (Default value = None)
    kwargs : dict
        keyword arguments for func

    Returns
    -------
    firworks.FWAction
        An empty FWAction

    """
    return FWAction()
