from .render import render


# args that should be set
args = (
    "name",
    "logfile",
    "mail_type",
    "mail_address",
    "nodes",
    "cores",
    "timeout",
    "queue",
    "command",
)


def _verify_dct(dct):
    for key in args:
        assert key in dct, f"key `{key}` missing"


def generate_jobscript(dct, file=None):
    """generate jobscript according to settings in dct"""

    if "tag" not in dct:
        dct["tag"] = "vibes"

    _verify_dct(dct)

    h, m = min_to_h_min(dct["timeout"])

    dct.update({"h": h, "min": m})

    if file is None:
        return render(dct)
    else:
        with open(file, "w") as f:
            f.write(render(dct))


def min_to_h_min(mins):
    h, m = divmod(mins, 60.0)

    return int(h), int(m)
