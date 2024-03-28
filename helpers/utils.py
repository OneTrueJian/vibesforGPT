"""A simple timer"""
import inspect
import itertools
import os
import signal
import sys
import threading
import time

import click

from vibes.helpers.warnings import warn


# print in bold
def bold(text):
    """ print text in bold face """
    return "\033[1m" + text + "\033[0m"


def talk(message, prefix=None, verbose=True):
    """vibes message output. Use instead of print. Sensitive to CLI context

    https://stackoverflow.com/a/2654130/5172579

    Args:
        message (str): message to print
        prefix (str): prefix for the message
        verbosity (int): verbosity level (0, 1, 2)
    """
    # see if we are in a CLI context
    if verbose is not False:
        try:
            ctx = click.get_current_context()
            verbose = ctx.obj.verbose
        except (RuntimeError, AttributeError):
            pass

    if verbose == 1:
        print_msg(message, prefix=prefix)
    elif verbose > 1:
        curframe = inspect.currentframe()
        frame = inspect.getouterframes(curframe, 2)[1]

        file = frame[1].split("vibes")[-1][1:]

        timestr = time.strftime("%H:%M:%S %Y/%m/%d")

        print(f"[{timestr} from {file}, l. {frame[2]} in {frame[3]}()]", flush=True)
        print_msg(message, prefix=prefix, indent=2)
        print()


def print_msg(message, prefix=None, indent=0, width=15):
    """print for talk

    Args:
        message (str): message to print
        prefix (str): prefix for message
        indent (int): number of spaces to indent by
        width (int): width of prefix
    """
    indent = indent * " "
    if not prefix:
        pref = "[vibes]"
    else:
        pref = f"[{prefix}] "
    if isinstance(message, list):
        for msg in message:
            print(f"{indent}{pref:{width}}{msg}", flush=True)
    else:
        print(f"{indent}{pref:{width}}{message}", flush=True)


class Timer:
    """simple timer"""

    prefix = None

    def __init__(self, message=None, timeout=None, prefix=None, verbose=True):
        """Initialize

        Args:
            message: Message to print at initialization
            timeout: attach a `Timeout` with `timeout` seconds
            prefix: prefix for `talk'
            verboes: be verbose

        """
        self.time = time.time()
        self.verbose = verbose

        self.print = talk

        if prefix:
            self.prefix = prefix

        self.message = message
        if message and verbose:
            self.print(message, prefix=self.prefix)

        self.timeout = Timeout(timeout)

    def wrap(self, func, *args, info_str="", **kwargs):
        result = func(*args, **kwargs)
        self.__call__(info_str=info_str)
        return result

    def __call__(self, info_str="", reset=False):
        """print how much time elapsed, optionally print `info_str`"""
        time_str = f"{time.time() - self.time:.3f}s"

        if info_str.strip() and self.verbose:
            self.print(f".. {info_str} in {time_str}", prefix=self.prefix)
        elif self.verbose:
            self.print(f".. time elapsed: {time_str}", prefix=self.prefix)

        # stop signal alarm if it was initialized
        self.timeout.stop()

        if reset:
            self.time = time.time()

        return float(time_str[:-1])


class Timeout:
    """simple Timeout function"""

    def __init__(self, timeout: int, kill: bool = True, pid: int = None):
        """Initialize a Timeout

        Args:
            timeout: timeout in seconds after which `os.kill()` is called
            kill: kill process instead of raising `TimeoutError`
            pid: terminate this PID

        Timeout partially inspired by
            https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/

        """
        self.pid = pid
        self.kill = kill
        self.time = time.time()
        self.timeout = timeout

        if timeout is not None:
            signal.signal(signal.SIGALRM, self.raise_timeout)
            signal.alarm(timeout)

    def stop(self):
        if self.timeout is not None:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)

    def __call__(self):
        """update timer, i.e., reset timeout"""
        self.stop()

        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.timeout)

    def raise_timeout(self, signum, frame):
        """kill the program"""
        msg = f"Timeout of {self.timeout}s is approaching, stop programm"
        if self.kill:
            warn(msg, level=1)
            pid = self.pid or os.getpid()
            os.kill(pid, signal.SIGKILL)

        raise TimeoutError(msg)


class Spinner:
    """Spinner for command line feedback

    Inspired by:
        https://stackoverflow.com/a/39504463/5172579
    """

    busy = False
    delay = 0.1

    spinning_cursor = itertools.cycle(["-", "|", "\\", "/"])

    def __init__(self, prefix="working", delay=None, file=sys.stdout, verbose=True):
        self.prefix = prefix
        self.file = file
        self.verbose = verbose
        self.msg = ""
        self.isatty = True
        self.spinner_generator = self.spinning_cursor
        if delay and float(delay):
            self.delay = delay

        if not hasattr(file, "isatty") or not file.isatty():
            self.isatty = False

    def get_msg(self):
        msg = f"{self.prefix}: "
        if self.busy:
            if self.isatty:
                msg += f"{next(self.spinner_generator)}"
            else:
                msg += f"working"
        else:
            msg += "finished." + "\n"
        return msg

    def print(self, newline=False):
        if self.verbose:
            self.msg = self.get_msg()
            if newline:
                self.file.write("\n")
            self.file.write(self.msg)
            self.file.flush()

    def rewind(self):
        if self.verbose:
            self.file.write(len(self.msg) * "\b")
            self.file.flush()

    def spinner_task(self):
        while self.busy:
            self.print()
            time.sleep(self.delay)
            self.rewind()
        self.print()

    def __enter__(self):
        self.busy = True
        if self.isatty:
            threading.Thread(target=self.spinner_task).start()
        else:
            self.print()

    def __exit__(self, exception, value, tb):
        self.busy = False
        if self.isatty:
            time.sleep(self.delay)
        else:
            self.print(newline=True)

        if exception is not None:
            raise exception


def progressbar(
    it,
    prefix="progress",
    size=35,
    file=sys.stdout,
    len_it=None,
    n_bars=200,
    verbose=True,
):
    """a simple progress bar to decorate an iterator

    Args:
        it (iterator): show progressbar for this iterator
        prefix (str): prefix for the progressbar
        size (int): size of the progress bar
        file (file): file to write to
        start_count (int): length of iterable
        n_bars (int): show this many bars
        verbose (bool): show the bar
    """
    count = len_it or max(1, len(it))
    n = len(str(count)) + 1

    def show(jj):
        """show the progressbar"""
        x = int(size * jj / count)
        counter = "{:{}d}/{}".format(jj, n, count)
        bar = "{:17s} |{}{}| {}\r".format(
            f"[{prefix}]", "|" * x, " " * (size - x), counter
        )
        if verbose:
            file.write(bar)
            file.flush()

    show(0)

    divider = max(1, count // n_bars)

    ii = 0
    for ii, item in enumerate(it):
        yield item
        if not ii % divider:
            if hasattr(file, "isatty") and file.isatty():
                show(ii)

    show(ii + 1)
    if hasattr(file, "isatty") and file.isatty():
        if verbose:
            file.write("\n")
            file.flush()
