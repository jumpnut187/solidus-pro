#!/usr/bin/env python3
"""
Launch script for Casting Solidification Simulator Web Application
Checks dependencies and starts the Flask server
"""

import sys
import subprocess
import os
from pathlib import Path

MIN_PYTHON = (3, 9)

def check_python_version():
    if sys.version_info < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
        print(f"   Current: Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    required = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'trimesh': 'Trimesh',
        'stl': 'numpy-stl',
        'numba': 'Numba',
        'psutil': 'psutil',
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            missing.append(name)
    
    return missing

def install_dependencies(missing):
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
    }
    
    for name in missing:
        if name in pip_packages:
            pkg = pip_packages[name]
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
    
    return True

def main():
    print("\n" + "="*60)
    print("  CASTING SOLIDIFICATION SIMULATOR - WEB VERSION")
    print("="*60 + "\n")
    
    if not check_python_version():
        sys.exit(1)
    
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        if not install_dependencies(missing):
            print("\n❌ Cannot start without required packages")
            sys.exit(1)
    
    print("\n" + "-"*60)
    print("Starting web server...")
    print("-"*60 + "\n")
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    sys.path.insert(0, str(project_dir))
    from web.app_factory import create_app
    
    port = int(os.environ.get("CASTSIM_WEB_PORT", "5050"))
    app = create_app()
    print("=" * 60)
    print("  SOLIDUS REACT/BABYLON WEB APPLICATION")
    print(f"  API: http://localhost:{port}/api/health")
    print(f"  Database: {app.config['DATABASE']}")
    print(f"  Uploads: {app.config['UPLOAD_FOLDER']}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)

if __name__ == "__main__":
    main()
