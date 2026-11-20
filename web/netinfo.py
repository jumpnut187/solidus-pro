from __future__ import annotations

from typing import List, Optional
import socket

_ROUTE_PROBE = ("203.0.113.1", 1)

def lan_address() -> Optional[str]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(_ROUTE_PROBE)
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    if not address or str(address).startswith("127."):
        return None
    return str(address)

def address_lines(port: int) -> List[str]:
    lines = [f"  Local:   http://localhost:{port}/"]
    address = lan_address()
    if address:
        lines.append(f"  Network: http://{address}:{port}/")
    else:
        lines.append("  Network: unavailable (no local network address)")
    return lines

def print_addresses(port: int) -> None:
    for line in address_lines(port):
        print(line, flush=True)

def port_from_bind(bind: object) -> Optional[int]:
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
