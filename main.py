"""
Solidus Pro - Professional Casting Solidification Simulator
Optimized for Apple M3 - Multi-core CPU, GPU, and Neural Engine acceleration
"""

import sys
import os
from ui.qt import QApplication

# Set multiprocessing start method for macOS
if sys.platform == 'darwin':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Initialize error logging early
from core.error_logger import logger as error_logger

from ui.main_window import MainWindow

def main():
    # Log application startup
    error_logger.info(f"Python version: {sys.version}")
    error_logger.info(f"Platform: {sys.platform}")
    
    # Qt 6 scales for high-DPI displays by default; the Qt 5 AA_* attributes are gone.
    
    app = QApplication(sys.argv)
    app.setApplicationName("Solidus Pro")
    app.setOrganizationName("SolidusPro")
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow()
    window.show()
    
    error_logger.info("Application window opened")
    
    exit_code = app.exec()
    error_logger.info(f"Application closed with exit code: {exit_code}")
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
