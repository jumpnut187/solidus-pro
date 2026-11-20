import sys
import os
from ui.qt import QApplication

if sys.platform == 'darwin':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from core.error_logger import logger as error_logger

from ui.main_window import MainWindow

def main():
    error_logger.info(f"Python version: {sys.version}")
    error_logger.info(f"Platform: {sys.platform}")

    app = QApplication(sys.argv)
    app.setApplicationName("Solidus Pro")
    app.setOrganizationName("SolidusPro")
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    error_logger.info("Application window opened")

    exit_code = app.exec()
    error_logger.info(f"Application closed with exit code: {exit_code}")

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
