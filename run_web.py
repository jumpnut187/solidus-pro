#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 9)

def check_python_version() -> bool:
    if sys.version_info < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
        print(f"   Current: Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies() -> list[str]:
    required = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'trimesh': 'Trimesh',
        'stl': 'numpy-stl',
        'numba': 'Numba',
        'psutil': 'psutil',
        'gunicorn': 'gunicorn',
    }
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - not installed")
            missing.append(name)

    try:
        __import__('cascadio')
        print("✓ cascadio (STEP support enabled)")
    except ImportError:
        print("○ cascadio (STEP support disabled — install with `pip install cascadio` for STEP files)")

    try:
        __import__('embreex')
        print("✓ embreex (Embree ray casting for part assignment)")
    except ImportError:
        print("⚠ embreex missing — part assignment will use the slow, memory-hungry "
              "pure-Python ray caster. Install with `pip install embreex`.")

    return missing

def install_dependencies(missing: list[str]) -> bool:
    if not missing:
        return True
    print(f"\nMissing packages: {', '.join(missing)}")
    response = input("Install missing packages? [y/N]: ").strip().lower()
    if response != 'y':
        return False
    pip_packages = {
        'Flask': 'flask',
        'Flask-CORS': 'flask-cors',
        'NumPy': 'numpy>=1.24.0',
        'SciPy': 'scipy>=1.10.0',
        'Trimesh': 'trimesh>=3.21.0',
        'numpy-stl': 'numpy-stl>=3.0.1',
        'Numba': 'numba>=0.57.0',
        'psutil': 'psutil>=5.9.5',
        'gunicorn': 'gunicorn>=21.2.0',
    }
    for name in missing:
        if name in pip_packages:
            pkg = pip_packages[name]
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
    return True

def _resolve_numba_threads() -> tuple[int, int, int, int]:
    try:
        from config.hardware_config import hardware
        logical = int(getattr(hardware, "logical_cores", 0) or (os.cpu_count() or 1))
        physical = int(getattr(hardware, "physical_cores", 0) or logical)
    except Exception:
        logical = os.cpu_count() or 4
        physical = logical

    env_override = os.environ.get("CASTSIM_NUMBA_THREADS") or os.environ.get(
        "CASTSIM_SOLIDIFICATION_THREADS"
    ) or "all"
    raw = env_override.strip().lower()
    if raw in ("all", "max", "logical"):
        target = logical
    elif raw in ("physical", "phy"):
        target = physical
    else:
        try:
            target = int(raw)
        except ValueError:
            target = logical

    target = max(1, min(target, logical))
    return target, logical, physical

def _print_startup_banner(port: int, app_module: str, worker_count: int, thread_count: int) -> None:
    numba_target, logical, physical = _resolve_numba_threads()

    try:
        from numba import get_num_threads as _gnt
        numba_active = _gnt()
    except Exception:
        numba_active = "?"

    try:
        from web.services.simulation_queue import default_queue_config
        from web.services.fair_share import compute_thread_share, resolve_usable_cores
        queue_config = default_queue_config()
        usable_cores = resolve_usable_cores(logical, queue_config.reserved_cores)
        share = compute_thread_share(usable_cores, max(1, queue_config.max_concurrent_simulations))
    except Exception:
        queue_config = None
        share = logical
        usable_cores = logical

    print("=" * 60)
    print("  SOLIDUS REACT/BABYLON WEB APPLICATION")
    print(f"  API:     http://localhost:{port}/api/health")
    print("=" * 60)
    print(f"  Server             : gunicorn (production)")
    print(f"  Workers            : {worker_count}")
    print(f"  Threads per worker : {thread_count}")
    print(f"  Hardware threads   : {physical} physical / {logical} logical")
    print(f"  Numba threads      : {numba_target} [active: {numba_active}]")
    print(f"  CASTSIM_NUMBA_THREADS: {os.environ.get('CASTSIM_NUMBA_THREADS', '(default: all)')}")
    if queue_config is not None:
        print(
            f"  Fair share         : {queue_config.max_concurrent_simulations} concurrent sims, "
            f"{queue_config.reserved_cores} reserved core(s), ~{share} cores/sim at full load"
        )
        print(
            f"  Memory reserve     : {queue_config.reserve_memory_mb} MB reserved, "
            f"{queue_config.min_sim_memory_mb} MB min/sim"
        )
    print("=" * 60 + "\n")

def _is_apple_system_python() -> bool:
    return "Xcode.app" in str(sys.executable) or getattr(sys, "_framework", "") == "Python3"

def _run_gunicorn(port: int, project_dir: Path) -> None:
    os.environ.setdefault("CASTSIM_NUMBA_THREADS", "all")

    if _is_apple_system_python():
        print("=" * 60)
        print("  ERROR: gunicorn cannot run on Apple's Xcode system Python.")
        print("  Its fork() is unsafe on current macOS: every worker segfaults")
        print("  in CoreFoundation at boot, causing an infinite crash loop.")
        print("  Use the project venv (built on a Homebrew Python) instead:")
        print("      source venv/bin/activate")
        print("      python run_web.py")
        sys.exit(1)

    if sys.platform == 'darwin':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    app_module = "web.app_factory:create_app()"

    gunicorn_args = [
        sys.executable, "-m", "gunicorn",
        app_module,
        "--config", str(project_dir / "web" / "gunicorn.conf.py"),
        "--bind", f"0.0.0.0:{port}",
        "--workers", "1",
        "--threads", "8",
        "--worker-class", "gthread",
        "--timeout", "0",
        "--keep-alive", "5",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", "info",
    ]

    print("\n" + "=" * 60)
    print("  Starting gunicorn (production WSGI server)...")
    print("  Use `--dev` flag to run Werkzeug dev server instead.")
    print("=" * 60 + "\n")

    _print_startup_banner(port, app_module, worker_count=1, thread_count=8)

    os.chdir(project_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(gunicorn_args[0], gunicorn_args)

def _run_dev_server(port: int, project_dir: Path) -> None:
    if sys.platform == 'darwin':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    os.environ.setdefault("CASTSIM_NUMBA_THREADS", "all")

    from web.app_factory import create_app
    from config.hardware_config import hardware

    numba_target, logical, physical = _resolve_numba_threads()

    try:
        from numba import get_num_threads as _gnt
        numba_threads = _gnt()
    except Exception:
        numba_threads = "?"

    app = create_app()

    print("=" * 60)
    print("  SOLIDUS REACT/BABYLON WEB APPLICATION")
    print(f"  API:     http://localhost:{port}/api/health")
    print(f"  Database: {app.config['DATABASE']}")
    print(f"  Uploads: {app.config['UPLOAD_FOLDER']}")
    print("=" * 60)
    print(f"  Server             : Werkzeug (development)")
    print(f"  Multiprocessing    : spawn")
    print(f"  Hardware threads   : {physical} physical / {logical} logical")
    print(f"  Numba threads      : {numba_target} [active: {numba_threads}]")
    print(f"  CASTSIM_NUMBA_THREADS: {os.environ.get('CASTSIM_NUMBA_THREADS', '(default: all)')}")
    print("=" * 60 + "\n")

    os.chdir(project_dir)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solidus Pro Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web.py               # gunicorn production server (default)
  python run_web.py --dev        # Werkzeug development server
  python run_web.py --port 8080  # Custom port with gunicorn
        """,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use Werkzeug development server instead of gunicorn",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CASTSIM_WEB_PORT", "5050")),
        help="Port to bind to (default: 5050 or CASTSIM_WEB_PORT env var)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CASTING SOLIDIFICATION SIMULATOR - WEB VERSION")
    print("=" * 60 + "\n")

    if not check_python_version():
        sys.exit(1)

    print("\nChecking dependencies...")
    missing = check_dependencies()

    if missing:
        if not install_dependencies(missing):
            print("\n❌ Cannot start without required packages")
            sys.exit(1)

    project_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_dir))

    if args.dev:
        print("\n" + "-" * 60)
        print("Starting Werkzeug development server...")
        print("-" * 60 + "\n")
        _run_dev_server(args.port, project_dir)
    else:
        print("\n" + "-" * 60)
        print("Starting gunicorn production server...")
        print("-" * 60 + "\n")
        _run_gunicorn(args.port, project_dir)

if __name__ == "__main__":
    main()
