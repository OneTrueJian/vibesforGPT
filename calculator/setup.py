"""set up ase calculators from settings"""
from ase.calculators.calculator import Calculator, get_calculator_class
from importlib import import_module

from vibes import keys
from vibes.helpers import talk


def from_settings(settings: dict = None) -> Calculator:
    """get calculator class and create the calculator from settings.parameters"""
    calc_dict = settings[keys.calculator]
    calc_name = calc_dict.name
    calc_module = calc_dict.get("module")

    if calc_module is None:  # ase calculator
        cls = get_calculator_class(calc_name.lower())
    else:
        module = import_module(calc_module)
        cls = getattr(module, calc_name)

    parameters = calc_dict.get(keys.parameters, {})

    talk(f"Set up a `{cls}` calculator with the following parameters:")
    for (k, v) in parameters.items():
        talk(f"{k:15}: {v}")

    return cls(**parameters)
