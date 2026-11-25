class DummyLogger:
    def debug(self, msg: str, **kwargs) -> None: pass
    def info(self, msg: str, **kwargs) -> None: pass
    def warning(self, msg: str, **kwargs) -> None: pass
    def error(self, msg: str, **kwargs) -> None: pass
    def critical(self, msg: str, **kwargs) -> None: pass
    def exception(self, msg: str, **kwargs) -> None: pass
    def log_performance(self, *args, **kwargs) -> None: pass
    def log_simulation_error(self, *args, **kwargs) -> None: pass
    def log_memory_warning(self, *args, **kwargs) -> None: pass

logger = DummyLogger()
