# logger_setup.py
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path


class _ConsoleFormatterNoTraceback(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        exc_info = record.exc_info
        exc_text = record.exc_text
        record.exc_info = None
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info = exc_info
            record.exc_text = exc_text


def _redirect_stderr_to_log(log_path: str) -> None:
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    os.close(fd)


def setup_logging(
    log_dir: str = "./logs",
    console: bool = True,
    level: str = "WARNING",
    quiet_console: bool = False,  # New parameter
    console_level: str = None,  # Separate console level
    log_path: str = None,  # Optional fixed log path
    suppress_rdkit_console: bool = True,
) -> tuple:
    """
    Setup logging with file and optional console handlers.

    Args:
        log_dir: Directory for log files
        console: Whether to enable console logging
        level: File logging level
        quiet_console: If True, only show minimal console output
        console_level: Separate level for console (defaults to level)
    """
    # Define the logger name
    name = "stereomapper"

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_path is None:
        log_path = str(Path(log_dir) / f"{name}_{ts}.log")

    if console and suppress_rdkit_console:
        _redirect_stderr_to_log(log_path)

    # Route RDKit logs into the same file and keep console output clean.
    try:
        from stereomapper.utils.suppress import setup_clean_logging

        setup_clean_logging(log_path=log_path, suppress_console=True)
    except Exception:
        pass

    # base config: file handler for all logs
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "{asctime} {levelname:<7} {name} - {message}",
                "style": "{",
            },
            "console": {
                "format": "{levelname:<7} {message}",
                "style": "{",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "plain",
                "filename": log_path,
                "encoding": "utf-8",
                "mode": "w",
                "level": "DEBUG",  # capture everything in file
            }
        },
        "loggers": {
            name: {
                "level": level,  # e.g. WARNING
                "handlers": ["file"],
                "propagate": False,
            },
        },
        "root": {"handlers": []},  # keep root empty
    }

    # apply base config
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

    # Get the main project logger
    logger = logging.getLogger(name)

    # Create console formatter
    console_formatter = _ConsoleFormatterNoTraceback("{levelname:<7} {message}", style="{")

    # Create summary logger first
    summary_logger = logging.getLogger(f"{name}.summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.propagate = False

    # Console-only output for summary logger (if enabled)
    if console:
        summary_console_handler = logging.StreamHandler(stream=sys.stdout)
        summary_console_handler.setLevel(logging.INFO)
        summary_console_handler.setFormatter(console_formatter)
        summary_logger.addHandler(summary_console_handler)

    # Handle console logging based on parameters
    if console and not quiet_console:
        # Normal console logging
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_level = console_level or level
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)
    elif console and quiet_console:
        # Minimal console output - only errors and critical
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(console_formatter)

        # Summary logger already has its own console handler
        pass

    logger.info("Logging initialised. File: %s", log_path)
    return logger, summary_logger
