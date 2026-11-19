"""
Where to reach this server, for the person reading the terminal.

Both server modes bind ``0.0.0.0``, so the app is reachable from any device on
the local network -- but "0.0.0.0" is not an address anyone can type, and
"localhost" is the wrong one to hand to a phone. This module works out the
address that actually reaches this machine from elsewhere on the network, and
formats the two lines a user needs to see at startup.

Kept apart from ``run_web`` because two very different callers need it: the
launcher, which prints it before handing off, and the gunicorn config, whose
worker hook prints it once the server is genuinely accepting connections.
"""

from __future__ import annotations

from typing import List, Optional
import socket


#: Reserved by RFC 5737 for documentation, so it can never be a real host.
#: Used only as a routing hint; nothing is ever sent to it.
_ROUTE_PROBE = ("203.0.113.1", 1)


def lan_address() -> Optional[str]:
    """
    This machine's address on the local network, or ``None`` if it has none.

    Found by asking the routing table which local interface would be used to
    reach an off-link address. That is the only way to pick the right one on a
    host with several interfaces at once -- Wi-Fi, Ethernet, VPN tunnels,
    virtual bridges -- without guessing at interface names, which differ by
    platform and by machine. The socket is UDP, so ``connect`` only assigns a
    local address; no packet is sent and the target need not exist.

    Loopback is reported as ``None``. A server reachable only at 127.0.0.1 is
    not on the network, and printing that as the "network" address would be a
    lie the user would waste time on.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(_ROUTE_PROBE)
        address = probe.getsockname()[0]
    except OSError:
        # No route to anywhere: the machine is off the network entirely.
        return None
    finally:
        probe.close()
    if not address or str(address).startswith("127."):
        return None
    return str(address)


def address_lines(port: int) -> List[str]:
    """The Local and Network lines to show for a server on ``port``."""
    lines = [f"  Local:   http://localhost:{port}/"]
    address = lan_address()
    if address:
        lines.append(f"  Network: http://{address}:{port}/")
    else:
        lines.append("  Network: unavailable (no local network address)")
    return lines


def print_addresses(port: int) -> None:
    """Print the address lines, flushed, so an exec or a redirect cannot eat them."""
    for line in address_lines(port):
        print(line, flush=True)


def port_from_bind(bind: object) -> Optional[int]:
    """
    The port out of a gunicorn ``bind`` setting, which may be a list.

    Handles ``host:port``, a bare ``:port``, and IPv6's ``[::]:port``; returns
    ``None`` for anything else, including a unix socket, where there is no port
    and therefore no URL worth printing.
    """
    if isinstance(bind, (list, tuple)):
        if not bind:
            return None
        bind = bind[0]
    text = str(bind or "")
    if text.startswith("unix:") or ":" not in text:
        return None
    try:
        return int(text.rsplit(":", 1)[1])
    except ValueError:
        return None
