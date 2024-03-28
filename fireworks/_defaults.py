"""Default definitions for FireWorks"""
from vibes._defaults import DEFAULT_FIREWORKS_FILE
from vibes.helpers.dict import AttributeDict as adict
from vibes.settings import Settings


SETTINGS = Settings(config_files=[DEFAULT_FIREWORKS_FILE])
FIREWORKS = SETTINGS.pop("fireworks")
REMOTE = FIREWORKS.pop("remote")
REMOTE_AUTH = REMOTE.pop("authorization")
REMOTE_LAUNCH = REMOTE.pop("launch")

FW_DEFAULTS = adict(
    {
        "config_dir": FIREWORKS.pop("config_dir"),
        "launch_dir": REMOTE.pop("launch_dir", "."),
        "remote_host": REMOTE.pop("host", None),
        "remote_config_dir": REMOTE.pop("config_dir", "~/.fireworks"),
        "remote_user": REMOTE_AUTH.pop("user", None),
        "remote_password": REMOTE_AUTH.pop("password", None),
        "gss_auth": REMOTE_AUTH.pop("gss_auth"),
        "njobs_queue": REMOTE_LAUNCH.pop("njobs_queue", 0),
        "njobs_block": REMOTE_LAUNCH.pop("njobs_block", 500),
        "nlaunches": REMOTE_LAUNCH.pop("nlaunches", 0),
        "sleep_time": REMOTE_LAUNCH.pop("sleep_time", None),
        "tasks2queue": FIREWORKS.pop("tasks2queue", []),
    }
)
