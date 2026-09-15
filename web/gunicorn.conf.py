"""Gunicorn configuration for the Solidus web application.

setproctitle's macOS implementation drives CoreFoundation/os_log, which is not
safe across os.fork(). When gunicorn sets process titles (master at startup,
workers right after fork) on macOS 26, workers die with a SIGSEGV inside
libsystem_trace, or with objc_initializeAfterForkError when a CoreFoundation
background thread is mid-initialization at fork time. Neutralizing the title
call before the arbiter starts keeps the master free of those threads, so
worker forks stay clean. Process titles are cosmetic; gunicorn functions
normally without them.
"""

import gunicorn.util as _util

_util._setproctitle = lambda title: None

# Gunicorn 25.1+ starts a control-socket server thread in the master right
# after the first worker is forked (arbiter._start_control_server). From then
# on the master is multithreaded, and on macOS the Objective-C runtime
# refuses to run any class's +initialize in a child forked from a
# multithreaded parent: the replacement worker dies in MLX/Metal setup with
# "+[NSNumber initialize] may have been in progress in another thread when
# fork() was called", the arbiter forks another, and the loop never ends.
# Observed 2026-09-15 after the kernel killed a worker for memory. The first
# worker always booted fine because it was forked before the thread existed.
# Nothing here uses `gunicornc`, so the socket is not needed.
control_socket_disable = True


def pre_fork(server, worker) -> None:
    pass


def post_worker_init(worker) -> None:
    """
    Announce where to reach the app, once it is genuinely ready.

    The launcher already prints a configuration banner, but it prints it before
    handing off to gunicorn -- and everything after that point, the arbiter's
    own log lines plus the hardware, GPU and Taichi banners emitted while the
    worker imports the application, is another three dozen lines. The addresses
    scrolled off the top and the last thing on screen was a compute backend,
    which is why the app looked like it was only ever on localhost.

    This hook runs inside the worker after the WSGI application has finished
    loading, so it is the last thing printed and the first thing the user sees
    when they look back at the terminal. It is also the honest moment to say
    "ready": the listener is bound and the app is importable, so the URL below
    will actually answer.
    """
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
