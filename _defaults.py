""" some default naming """

from pathlib import Path

supported_tasks = ["phonopy", "phono3py", "md"]

HOME = Path().home()
ROOT = Path(__file__).parent.absolute()

DEFAULT_CONFIG_FILE = HOME / ".vibesrc"
DEFAULT_FIREWORKS_FILE = HOME / ".fireworksrc"
DEFAULT_GEOMETRY_FILE = "geometry.in"
DEFAULT_SETTINGS_FILE = ROOT / "templates" / "settings.in"
DEFAULT_TEMP_SETTINGS_FILE = "temp_settings.in"
