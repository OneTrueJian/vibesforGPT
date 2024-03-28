""" Provides messages and warnings naming the origin """
import inspect


def warn(message, level=0):
    """Print warnings to the console

    https://stackoverflow.com/a/2654130/5172579

    Parameters
    ----------
    message: str
        warning message to print
    level: int (0, 1, 2)
        How severe the warning is

    Raises
    ------
    RuntimeError
        If warning level=2 (Error)
    """

    curframe = inspect.currentframe()
    frame = inspect.getouterframes(curframe, 2)[1]

    if level == 0:
        typ = "Message"
    elif level == 1:
        typ = "Warning"
    elif level == 2:
        typ = "Error"

    stars = "*" + "*" * level

    file = frame[1].split("vibes")[-1]

    print(
        f"{stars} {typ} from file vibes{file}, line {frame[2]}, function {frame[3]}:",
        flush=True,
    )
    print(f"--> {message}\n", flush=True)

    if typ == "Error":
        raise RuntimeError("see above")
