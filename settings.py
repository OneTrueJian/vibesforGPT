""" Settings class for holding settings, based on configparser.ConfigParser """
from pathlib import Path
from typing import Sequence

from jconfigparser import Config
from jconfigparser.dict import DotDict

from vibes import keys
from vibes._defaults import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_FIREWORKS_FILE,
    DEFAULT_GEOMETRY_FILE,
    DEFAULT_SETTINGS_FILE,
)


class SettingsError(Exception):
    """error in settings"""


def verify_key(
    key: str,
    obj: dict,
    hint: str = None,
    section: bool = False,
    allowed_to_fail: bool = False,
):
    """verify that key is in object, otherwise raise SettingsError

    Args:
        key: Key to check if it is in obj
        obj: Dict to see if key is in it
        hint: string representation of obj
        section: If True key is a section in obj
        allowed_to_fail: If True use wannings not errors
    """
    from vibes.helpers.warnings import warn

    if not hint:
        hint = str(obj)

    if key not in obj:
        if section:
            msg = f"\n  section [{key}] is missing in {hint}"
        else:
            msg = f"\n  key '{key}' is missing in {hint}"

        if allowed_to_fail:
            warn(msg, level=1)
        else:
            raise SettingsError(msg)


class Configuration(Config):
    """class to hold the configuration from .vibesrc"""

    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        """Initializer

        Args:
            config_file: Path to the configure file
        """
        from vibes import __version__ as version

        super().__init__(filenames=config_file)

        # include the vibes version tag
        self.update({"vibes": {"version": version}})


class Settings(Config):
    """Class to hold the settings parsed from settings.in (+ the configuration)"""

    def __init__(
        self,
        settings_file: str = None,
        template_dict: dict = None,
        config_files: Sequence[str] = [DEFAULT_CONFIG_FILE],
        fireworks: bool = False,
        dct: dict = None,
        debug: bool = False,
    ):
        """Initialize Settings

        Args:
            settings_file: Path to the settings file
            template_dict: Template dictionary with default settings
            config_files: Path to the configuration files
            fireworks: read fireworks config file
            dct: create Settings from this dictionary
            debug: add verbosity for debugging

        """
        from vibes.helpers.dict import merge
        from vibes.helpers.warnings import warn

        if debug and dct is None and settings_file is None:
            warn("`settings_file` is `None`", level=1)

        # read config, then template, then user settings

        _dct = DotDict()

        # i) config
        if config_files is not None:
            if fireworks:
                config_files.append(DEFAULT_FIREWORKS_FILE)

            for file in config_files:
                _temp = Config(file)
                _dct = merge(_temp, _dct, dict_type=DotDict)

        # ii) read template
        if template_dict:
            _dct = merge(template_dict, _dct, dict_type=DotDict)

        # iii) user settings
        if not dct:
            dct = {}

            if settings_file is not None:
                dct = Config(settings_file)

        _dct = merge(dct, _dct, dict_type=DotDict)

        _dct = legacy_update(_dct)

        super().__init__()
        for key in _dct:
            self[key] = _dct[key]

        if hasattr(dct, "file"):
            self._settings_file = dct.file
        else:
            self._settings_file = settings_file

    @classmethod
    def from_dict(cls, dct):
        """initialize from dictionary"""
        return cls(dct=dct)

    @property
    def file(self):
        """return path to the settings file"""
        if self._settings_file is not None:
            return self._settings_file
        return DEFAULT_SETTINGS_FILE

    def write(self, file=None):
        """write settings to file"""
        from vibes.helpers.warnings import warn

        if not file:
            file = Path(self.file).name

        if not Path(file).exists():
            super().write(Path(file))
        else:
            warn(f"{file} exists, do not overwrite settings.", level=1)

    def print(self, only_settings: bool = False):
        ignore_sections = None
        if only_settings:
            ignore_sections = Configuration().keys()
        super().print(ignore_sections=ignore_sections)

    def read_atoms(self, format="aims"):
        """parse the geometry described in settings.in and return as atoms
        """
        from ase.io import read
        from vibes.helpers.warnings import warn

        if "files" in self and "geometry" in self["files"]:
            path = Path(self.files.geometry)
            file = next(path.parent.glob(path.name))
        else:
            file = DEFAULT_GEOMETRY_FILE

        if Path(file).exists():
            return read(file, format=format)

        warn(f"Geometry file {file} not found.", level=1)

        return None


def legacy_update_aims(settings: dict) -> dict:
    """replace legacy keynames in settings related to aims"""

    # aims
    if "control" in settings:
        if "calculator" not in settings:
            settings["calculator"] = DotDict()

        settings[f"{keys.calculator}.{keys.name}"] = "aims"
        settings[keys.calculator][keys.parameters] = settings.pop("control")

    if "control_kpt" in settings:
        settings[keys.calculator]["kpoints"] = settings.pop("control_kpt")

    if "basissets" in settings:
        settings[keys.calculator]["basissets"] = settings.pop("basissets")

    if "socketio" in settings:
        settings[keys.calculator]["socketio"] = settings.pop("socketio")

    if "basisset" in settings:
        msg = "`basisset.type` is removed in favour of `basissets.default`. Stop"
        raise RuntimeError(msg)


def legacy_update(settings: dict) -> dict:
    """replace legacy keynames in settings"""

    # aims -> calculator.aims
    legacy_update_aims(settings)

    # geometry -> files
    _files = "files"
    if "geometry" in settings:
        if _files not in settings:
            settings[_files] = DotDict()

        if "file" in settings.geometry:
            settings[_files]["geometry"] = settings.geometry.pop("file")
        if "files" in settings.geometry:
            settings[_files]["geometries"] = settings.geometry.pop("files")

        for name in ("primitive", "supercell"):
            if name in settings.geometry:
                settings[_files][name] = settings.geometry.pop(name)

    return settings
