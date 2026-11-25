import gunicorn.util as _util

_util._setproctitle = lambda title: None

control_socket_disable = True

def pre_fork(server, worker) -> None:
    pass

def post_worker_init(worker) -> None:
    from web.netinfo import address_lines, port_from_bind

    port = port_from_bind(getattr(worker.cfg, "bind", None))
    if port is None:
        return

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  READY", flush=True)
    for line in address_lines(port):
        print(line, flush=True)
    print("=" * 60 + "\n", flush=True)
