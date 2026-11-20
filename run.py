#!/usr/bin/env python3

import sys
import subprocess
import os

def check_python_version():
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    required = ['PySide6', 'numpy', 'vtk', 'trimesh', 'numba']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)

    try:
        __import__('cascadio')
        print("✓ cascadio (STEP support enabled)")
    except ImportError:
        print("○ cascadio (STEP support disabled — install with `pip install cascadio` for STEP files)")

    return len(missing) == 0, missing

def install_dependencies():
    print("\n📦 Installing dependencies...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies")
        print(result.stderr)
        return False

def main():
    print("="*60)
    print("Casting Solidification Simulator - Setup")
    print("="*60)
    print()

    if not check_python_version():
        sys.exit(1)

    print("\n📋 Checking dependencies...")
    deps_ok, missing = check_dependencies()

    if not deps_ok:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ")

        if response.lower() == 'y':
            if not install_dependencies():
                sys.exit(1)
        else:
            print("\nTo install manually:")
            print("  pip install -r requirements.txt")
            sys.exit(1)

    print("\n🚀 Launching simulator...")
    print("="*60)
    print()

    try:
        subprocess.run([sys.executable, 'main.py'])
    except KeyboardInterrupt:
        print("\n\n✓ Simulator closed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
