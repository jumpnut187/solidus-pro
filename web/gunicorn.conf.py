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


def pre_fork(server, worker) -> None:
    pass
