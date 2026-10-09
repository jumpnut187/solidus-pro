"""
Error Logging System for Casting Solidification Simulator

Provides centralized error logging to file and console with:
- Automatic log file rotation
- Stack traces for exceptions
- Severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Timestamps and context information
- Thread-safe logging
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from logging.handlers import RotatingFileHandler
import threading

# Log file location (in project root)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "error_log.txt"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Keep 3 backup files


class SimulatorLogger:
    """
    Centralized logging system for the casting simulator.
    
    Features:
    - Writes to both file and console
    - Captures unhandled exceptions
    - Thread-safe for multi-threaded simulation
    - Provides structured error information
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global logger access"""
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
        self._max_entries = 1000  # Keep last 1000 entries in memory
        
        # Create logs directory
        LOG_DIR.mkdir(exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger("CastingSim")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
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
        
        # Console handler (only warnings and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Install global exception handler
        self._install_exception_handler()
        
        # Log startup
        self.info("=" * 60)
        self.info("Casting Solidification Simulator Started")
        self.info(f"Log file: {LOG_FILE}")
        self.info("=" * 60)
    
    def _install_exception_handler(self):
        """Install global exception handler to catch unhandled exceptions"""
        def exception_handler(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            
            # Log the exception
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_text = ''.join(tb_lines)
            self.critical(f"Unhandled exception:\n{tb_text}")
            
            # Call default handler
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        
        sys.excepthook = exception_handler
    
    def _add_entry(self, level: str, message: str, context: Optional[Dict] = None):
        """Add entry to in-memory log for UI display"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'context': context or {}
        }
        
        with self._lock:
            self._log_entries.append(entry)
            # Trim old entries
            if len(self._log_entries) > self._max_entries:
                self._log_entries = self._log_entries[-self._max_entries:]
    
    def debug(self, message: str, context: Optional[Dict] = None):
        """Log debug message"""
        self.logger.debug(message)
        self._add_entry('DEBUG', message, context)
    
    def info(self, message: str, context: Optional[Dict] = None):
        """Log info message"""
        self.logger.info(message)
        self._add_entry('INFO', message, context)
    
    def warning(self, message: str, context: Optional[Dict] = None):
        """Log warning message"""
        self.logger.warning(message)
        self._add_entry('WARNING', message, context)
    
    def error(self, message: str, exc_info: bool = False, context: Optional[Dict] = None):
        """Log error message, optionally with exception info"""
        if exc_info:
            self.logger.error(message, exc_info=True)
            # Add traceback to context
            if context is None:
                context = {}
            context['traceback'] = traceback.format_exc()
        else:
            self.logger.error(message)
        self._add_entry('ERROR', message, context)
    
    def critical(self, message: str, context: Optional[Dict] = None):
        """Log critical error message"""
        self.logger.critical(message)
        self._add_entry('CRITICAL', message, context)
    
    def exception(self, message: str, context: Optional[Dict] = None):
        """Log exception with full traceback"""
        tb = traceback.format_exc()
        full_message = f"{message}\n{tb}"
        self.logger.exception(message)
        if context is None:
            context = {}
        context['traceback'] = tb
        self._add_entry('ERROR', full_message, context)
    
    def log_simulation_error(self, stage: str, error: Exception, 
                             mesh_info: Optional[Dict] = None):
        """Log simulation-specific error with context"""
        context = {
            'stage': stage,
            'error_type': type(error).__name__,
            'mesh_info': mesh_info or {}
        }
        self.error(f"Simulation error in {stage}: {error}", exc_info=True, context=context)
    
    def log_memory_warning(self, current_mb: float, limit_mb: float, 
                           voxel_count: int):
        """Log memory warning"""
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
        """Log performance metrics. ``count_unit`` names the counted points ("nodes" on tet runs)."""
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
        """Get recent log entries for UI display"""
        with self._lock:
            entries = self._log_entries[-count:]
            if level_filter:
                level_filter = level_filter.upper()
                if level_filter == 'ERROR':
                    # Include ERROR and CRITICAL
                    entries = [e for e in entries if e['level'] in ('ERROR', 'CRITICAL')]
                elif level_filter == 'WARNING':
                    # Include WARNING, ERROR, and CRITICAL
                    entries = [e for e in entries if e['level'] in ('WARNING', 'ERROR', 'CRITICAL')]
                else:
                    entries = [e for e in entries if e['level'] == level_filter]
            return entries
    
    def get_log_file_path(self) -> Path:
        """Get the path to the log file"""
        return LOG_FILE
    
    def get_log_file_contents(self, max_lines: int = 500) -> str:
        """Read the log file contents"""
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return ''.join(lines[-max_lines:])
            return "No log file found."
        except Exception as e:
            return f"Error reading log file: {e}"
    
    def clear_log_file(self):
        """Clear the log file"""
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("")
            self._log_entries.clear()
            self.info("Log file cleared")
        except Exception as e:
            self.error(f"Failed to clear log file: {e}")


# Global logger instance
logger = SimulatorLogger()


# Convenience functions for direct import
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
    # Test the logger
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
