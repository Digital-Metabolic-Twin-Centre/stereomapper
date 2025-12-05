"""Setting up timing operations for stereomapper pipeline"""
import logging
import time
from contextlib import contextmanager
from functools import wraps

def _now():
    return time.perf_counter()

@contextmanager
def section_timer(name: str, logger: logging.Logger):
    """Set up a section specific timer"""
    t0 = _now()
    try:
        yield
    finally:
        dt = _now() - t0
        logger.info("TIMER %s took %.3f s", name, dt)

def timeit(logger: logging.Logger, name: str | None = None):
    """Time a specific operation"""
    def deco(fn):
        label = name or fn.__qualname__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = _now()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = _now() - t0
                logger.info("TIMER %s took %.3f s", label, dt)
        return wrapper
    return deco
