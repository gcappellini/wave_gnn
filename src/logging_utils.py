"""
Shared logging setup.

Single utility used by every entrypoint so terminal output is always
mirrored to a log file inside the run's output directory.
"""

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path, filename: str = "run.log", logger_name: str = "wave_gnn") -> logging.Logger:
    """Configure a logger that writes to both console and a file in output_dir.

    Also installs a sys.excepthook so uncaught exceptions are recorded in the
    log file before the process exits.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _log_uncaught_exception

    logger.info(f"Logging to: {log_path}")
    return logger
