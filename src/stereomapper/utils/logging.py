# logger_setup.py
import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_dir: str = "./logs",
    console: bool = True,
    level: str = "INFO",
    quiet_console: bool = False,  # New parameter
    console_level: str = None,    # Separate console level
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
    log_path = str(Path(log_dir) / f"{name}_{ts}.log")

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
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "plain",
                "filename": log_path,
                "encoding": "utf-8",
                "mode": "w",
                "level": "DEBUG",   # capture everything in file
            }
        },
        "loggers": {
            name: {
                "level": level,          # e.g. WARNING
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
    console_formatter = logging.Formatter("{levelname:<7} {message}", style="{")

    # Create summary logger first
    summary_logger = logging.getLogger(f"{name}.summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.propagate = False

    # Add file handler for summary logger
    fh_summary = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh_summary.setLevel(logging.INFO)
    fh_summary.setFormatter(logging.Formatter("{asctime} SUMMARY - {message}", style="{"))
    summary_logger.addHandler(fh_summary)

    # Handle console logging based on parameters
    if console and not quiet_console:
        # Normal console logging
        console_handler = logging.StreamHandler()
        console_level = console_level or level
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
        summary_logger.addHandler(console_handler)
    elif console and quiet_console:
        # Minimal console output - only errors and critical
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(console_formatter)
        
        # Only add to summary logger for important messages
        summary_logger.addHandler(console_handler)

    logger.info("Logging initialised. File: %s", log_path)
    return logger, summary_logger