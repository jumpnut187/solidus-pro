import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from logging.handlers import RotatingFileHandler
import threading

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "error_log.txt"
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3

class SimulatorLogger:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._log_entries: List[Dict[str, Any]] = []
        self._max_entries = 1000

        LOG_DIR.mkdir(exist_ok=True)

        self.logger = logging.getLogger("CastingSim")
        self.logger.setLevel(logging.DEBUG)

        self.logger.handlers.clear()

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self._install_exception_handler()

        self.info("=" * 60)
        self.info("Casting Solidification Simulator Started")
        self.info(f"Log file: {LOG_FILE}")
        self.info("=" * 60)

    def _install_exception_handler(self):
        def exception_handler(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return

            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_text = ''.join(tb_lines)
            self.critical(f"Unhandled exception:\n{tb_text}")

            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = exception_handler

    def _add_entry(self, level: str, message: str, context: Optional[Dict] = None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'context': context or {}
        }

        with self._lock:
            self._log_entries.append(entry)
            if len(self._log_entries) > self._max_entries:
                self._log_entries = self._log_entries[-self._max_entries:]

    def debug(self, message: str, context: Optional[Dict] = None):
        self.logger.debug(message)
        self._add_entry('DEBUG', message, context)

    def info(self, message: str, context: Optional[Dict] = None):
        self.logger.info(message)
        self._add_entry('INFO', message, context)

    def warning(self, message: str, context: Optional[Dict] = None):
        self.logger.warning(message)
        self._add_entry('WARNING', message, context)

    def error(self, message: str, exc_info: bool = False, context: Optional[Dict] = None):
        if exc_info:
            self.logger.error(message, exc_info=True)
            if context is None:
                context = {}
            context['traceback'] = traceback.format_exc()
        else:
            self.logger.error(message)
        self._add_entry('ERROR', message, context)

    def critical(self, message: str, context: Optional[Dict] = None):
        self.logger.critical(message)
        self._add_entry('CRITICAL', message, context)

    def exception(self, message: str, context: Optional[Dict] = None):
        tb = traceback.format_exc()
        full_message = f"{message}\n{tb}"
        self.logger.exception(message)
        if context is None:
            context = {}
        context['traceback'] = tb
        self._add_entry('ERROR', full_message, context)

    def log_simulation_error(self, stage: str, error: Exception,
                             mesh_info: Optional[Dict] = None):
        context = {
            'stage': stage,
            'error_type': type(error).__name__,
            'mesh_info': mesh_info or {}
        }
        self.error(f"Simulation error in {stage}: {error}", exc_info=True, context=context)

    def log_memory_warning(self, current_mb: float, limit_mb: float,
                           voxel_count: int):
        context = {
            'current_mb': current_mb,
            'limit_mb': limit_mb,
            'voxel_count': voxel_count
        }
        self.warning(
            f"Memory usage {current_mb:.1f}MB approaching limit {limit_mb:.1f}MB "
            f"with {voxel_count:,} voxels",
            context=context
        )

    def log_performance(self, operation: str, duration_seconds: float,
                        voxel_count: Optional[int] = None, count_unit: str = "voxels"):
        context = {
            'operation': operation,
            'duration_seconds': duration_seconds,
            'voxel_count': voxel_count
        }
        rate = ""
        if voxel_count and duration_seconds > 0:
            rate = f" ({voxel_count / duration_seconds:.0f} {count_unit}/sec)"
        self.info(f"{operation} completed in {duration_seconds:.2f}s{rate}", context=context)

    def get_recent_entries(self, count: int = 100,
                           level_filter: Optional[str] = None) -> List[Dict]:
        with self._lock:
            entries = self._log_entries[-count:]
            if level_filter:
                level_filter = level_filter.upper()
                if level_filter == 'ERROR':
                    entries = [e for e in entries if e['level'] in ('ERROR', 'CRITICAL')]
                elif level_filter == 'WARNING':
                    entries = [e for e in entries if e['level'] in ('WARNING', 'ERROR', 'CRITICAL')]
                else:
                    entries = [e for e in entries if e['level'] == level_filter]
            return entries

    def get_log_file_path(self) -> Path:
        return LOG_FILE

    def get_log_file_contents(self, max_lines: int = 500) -> str:
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return ''.join(lines[-max_lines:])
            return "No log file found."
        except Exception as e:
            return f"Error reading log file: {e}"

    def clear_log_file(self):
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("")
            self._log_entries.clear()
            self.info("Log file cleared")
        except Exception as e:
            self.error(f"Failed to clear log file: {e}")

logger = SimulatorLogger()

def debug(message: str, context: Optional[Dict] = None):
    logger.debug(message, context)

def info(message: str, context: Optional[Dict] = None):
    logger.info(message, context)

def warning(message: str, context: Optional[Dict] = None):
    logger.warning(message, context)

def error(message: str, exc_info: bool = False, context: Optional[Dict] = None):
    logger.error(message, exc_info, context)

def critical(message: str, context: Optional[Dict] = None):
    logger.critical(message, context)

def exception(message: str, context: Optional[Dict] = None):
    logger.exception(message, context)

if __name__ == '__main__':
    print(f"Log file location: {LOG_FILE}")

    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Caught test exception")

    print("\nRecent entries:")
    for entry in logger.get_recent_entries(5):
        print(f"  {entry['timestamp']} [{entry['level']}] {entry['message'][:50]}")
