"""
Solidus Pro - Professional Casting Solidification Simulator
Optimized for Apple M3 - Multi-core CPU, GPU, and Neural Engine acceleration
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

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
    
    # Enable high DPI scaling for Retina displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Solidus Pro")
    app.setOrganizationName("SolidusPro")
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow()
    window.show()
    
    error_logger.info("Application window opened")
    
    exit_code = app.exec_()
    error_logger.info(f"Application closed with exit code: {exit_code}")
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
