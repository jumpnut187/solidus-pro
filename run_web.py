#!/usr/bin/env python3
"""
Launch script for Casting Solidification Simulator Web Application
Checks dependencies and starts the Flask server
"""

import sys
import subprocess
import os
from pathlib import Path

# Minimum Python version
MIN_PYTHON = (3, 9)

def check_python_version():
    """Verify Python version meets minimum requirements"""
    if sys.version_info < MIN_PYTHON:
        print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
        print(f"   Current: Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
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
    """Offer to install missing dependencies"""
    if not missing:
        return True
    
    print(f"\nMissing packages: {', '.join(missing)}")
    response = input("Install missing packages? [y/N]: ").strip().lower()
    
    if response != 'y':
        return False
    
    # Install packages
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
    
    # Check Python version
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
    
    # Change to project root directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Add to path and start Flask app
    sys.path.insert(0, str(project_dir))
    from web.app import app
    
    print("\n" + "="*60)
    print("  SERVER RUNNING")
    print("  Open http://localhost:5050 in your browser")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5050, threaded=True)

if __name__ == '__main__':
    main()
