import logging
import os
import sys
from contextlib import contextmanager


def _configure_rdkit_logger(log_path: str, suppress_console: bool) -> None:
    try:
        from rdkit import RDLogger, rdBase
    except Exception:
        return

    try:
        rdBase.LogToPython()
    except Exception:
        pass

    loggers = [logging.getLogger("rdkit")]
    try:
        loggers.append(RDLogger.logger())
    except Exception:
        pass

    for rdkit_logger in loggers:
        # Clear existing handlers to avoid console spam or duplicates.
        for handler in list(rdkit_logger.handlers):
            rdkit_logger.removeHandler(handler)

        rdkit_logger.propagate = False
        rdkit_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("{asctime} {levelname:<7} {name} - {message}", style="{")
        )
        rdkit_logger.addHandler(file_handler)

        if not suppress_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("{levelname:<7} {message}", style="{"))
            rdkit_logger.addHandler(console_handler)


def setup_clean_logging(log_path: str | None = None, suppress_console: bool = True) -> None:
    """Configure RDKit logging to avoid noisy console output."""
    if not log_path:
        return

    _configure_rdkit_logger(log_path, suppress_console)


@contextmanager
def quiet_operation():
    """Context manager for completely silent operations"""
    with open("/dev/null", "w") as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        try:
            sys.stderr = devnull
            if os.getenv("QUIET_MODE", "false").lower() == "true":
                sys.stdout = devnull
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout
