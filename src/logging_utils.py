"""
Shared logging setup.

Single utility used by every entrypoint so terminal output is always
mirrored to a log file inside the run's output directory.
"""

import logging
import sys
from pathlib import Path


class _LogStream:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message: str) -> int:
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        if self.buffer:
            self.logger.log(self.level, self.buffer)
            self.buffer = ""


def setup_logging(output_dir: Path, filename: str = "run.log", logger_name: str = "wave_gnn") -> logging.Logger:
    """Configure a logger that writes to both console and a file in output_dir.

    Also installs a sys.excepthook so uncaught exceptions are recorded in the
    log file before the process exits.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = True

    sys.stdout = _LogStream(logger, logging.INFO)
    sys.stderr = _LogStream(logger, logging.ERROR)

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _log_uncaught_exception

    logger.info(f"Logging to: {log_path}")
    return logger
