""" socket io helpers """
import socket
from contextlib import closing

import numpy as np
from ase import units

from vibes.helpers import talk
from vibes.helpers.warnings import warn

from . import stresses as stresses_helper


_prefix = "socketio"


def check_socket(host, port):
    """Check if socket is able to bind

    Args:
        host (str): string to the host
        port (int): port for the socket

    Returns:
        bool: True if socket is able to bind
    """

    try:
        socket.getservbyport(port)
        return False
    except OSError:
        pass

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def get_free_port(host, offset=0, min_port_val=10000):
    """Automatically select a free port to use

    Args:
        host (str): String to the host path
        min_port_val (int): First port to check
        offset (int): Select the next + offset free port (in case multiple sequntial runs)

    Returns
        port (int): The available port
    """

    pp = 0
    for port in range(max(1024, min_port_val), 49151):
        if check_socket(host, port):
            pp += 1
        if pp > offset:
            return port
    raise ValueError("No available port found.")


def get_port(host, port, offset=0):
    """Get the port to use for the socketio calculation

    Args:
        host (str): String to the host path
        port (int): Requested port
        offset (int): Select the next + offset free port (in case multiple sequntial runs)

    Returns
        port (int): A free port based on the requested value
    """
    # If a unixsocket is being used then disregard since ports aren't used
    if "UNIX" in host:
        return port

    # If None then do nothing this should be an error
    if port is None:
        return None

    # If automatic find a free port, if port is not free get a new port
    if port == "auto":
        port = get_free_port(host, offset)
    elif port and not check_socket(host, port):
        warn(f"Port {port} in use, changing to the next free port")
        port = get_free_port(host, min_port_val=port)

    return port


def get_socket_info(calculator, prefix=_prefix):
    """return port of the calculator

    Args:
        calculator: calculator to get the port of
        prefix: prefix for messages from this function
    Returns:
        host: The host for socketio
        port: the port for socketio
        unixsocket: get_unixsocket
    """

    port = None
    unixsocket = None
    if not hasattr(calculator, "parameters"):
        warn(f"{prefix} No parameters found in calculator {calculator.name}.", level=1)

    else:
        if "use_pimd_wrapper" in calculator.parameters:
            port = calculator.parameters["use_pimd_wrapper"][1]
            host = calculator.parameters["use_pimd_wrapper"][0]

            if "UNIX:" in host:
                unixsocket = calculator.parameters["use_pimd_wrapper"][0]
                talk(f"Use SocketIO with unixsocket file {unixsocket}", prefix=prefix)
            else:
                talk(f"Use SocketIO with host {host} and port {port}", prefix=prefix)
        else:
            talk(f"Socketio not used with calculator {calculator.name}", prefix=prefix)

    return port, unixsocket


def get_stresses(atoms):
    """Use Socket to get intensive atomic stresses in eV/AA^3 in Nx3x3 shape.
    Raw stresses are supposed to be extensive and the volume is divided out.

    """
    if "socketio" not in atoms.calc.name.lower():
        return stresses_helper.get_stresses(atoms)
    # assume these are extensive stresses
    atoms.calc.server.protocol.sendmsg("GETSTRESSES")
    msg = atoms.calc.server.protocol.recvmsg()
    assert msg == "STRESSREADY"
    natoms = atoms.calc.server.protocol.recv(1, np.int32)
    stresses = atoms.calc.server.protocol.recv((int(natoms), 3, 3), np.float64)
    return stresses * units.Hartree / atoms.get_volume()


def socket_stress_off(calculator):
    """Turn stresses computation off via socket

    Args:
        calculator: ase.calculators.calculator.Calculator
            calculator to turn off stress computation for
    """
    calculator.server.protocol.sendmsg("STRESSES_OFF")


def socket_stress_on(calculator):
    """Turn stresses computation on via socket

    Args:
        calculator: ase.calculators.calculator.Calculator
            calculator to turn on stress computation for
    """
    calculator.server.protocol.sendmsg("STRESSES_ON")
