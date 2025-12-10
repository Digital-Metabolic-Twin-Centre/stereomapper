# config/resolvers.py
from pathlib import Path
from typing import Optional, Tuple, List
from stereomapper.data.db import connect
import os
from platformdirs import user_cache_dir

APP = "stereomapper"
SCHEMA_VERSION = 1  # increment when schema changes

def default_cache_path() -> Path:
    p = Path(user_cache_dir(APP))
    p.mkdir(parents=True, exist_ok=True)
    return p / f"structures-v{SCHEMA_VERSION}.sqlite"

def _resolve_cache_path(*, relate_with_cache: bool, fresh_cache: bool, cache_path: Optional[str]) -> Optional[Path]:
    """
    Decide which cache DB we are using for this run:
    - relate_with_cache=True: must open an existing DB (either provided or default).
    - fresh_cache=True: create a brand new DB in the given path (or a temp dir).
    - neither: optional cache; create if missing at provided/default path.
    """
    if relate_with_cache and fresh_cache:
        raise ValueError("Cannot set both relate_with_cache=True and fresh_cache=True.")

    if relate_with_cache:
        # Must open an existing DB (either provided or default)
        p = Path(cache_path) if cache_path else default_cache_path()
        if not p.exists():
            raise FileNotFoundError(f"relate_with_cache=True but cache DB not found: {p}")
        return p

    if fresh_cache:
        # Create a brand-new DB (overwrite if exists)
        p = Path(cache_path) if cache_path else default_cache_path()
        if p.exists():
            p.unlink()
        return p

    # Neither relate_with_cache nor fresh_cache: use provided/default path, create if absent
    p = Path(cache_path) if cache_path else default_cache_path()
    return p

def _resolve_inputs_from_cfg(cfg) -> Tuple[str, ...]:
    if cfg.input and cfg.input_dir:
        raise ValueError("Specify either explicit input files or input_dir, not both.")

    if cfg.input_dir:
        base = Path(cfg.input_dir)
        if not base.is_dir():
            raise ValueError(f"--input-dir is not a directory: {base}")
        pattern = "**/*" if cfg.recursive else "*"
        found: List[Path] = []
        # collect matching files
        for p in base.glob(pattern):
            if p.is_file() and p.suffix.lower() in cfg.extensions:
                found.append(p.resolve())
        files = tuple(sorted({str(p) for p in found}))
        if not files:
            rec = " recursively" if cfg.recursive else ""
            exts = ", ".join(cfg.extensions)
            raise ValueError(f"No files with extensions ({exts}) found in {base}{rec}.")
        return files

    if cfg.input:
        # normalize & validate explicit files
        files = []
        for item in cfg.input:
            p = Path(item)
            if not p.is_file():
                raise ValueError(f"Input file not found: {item}")
            if p.suffix.lower() not in cfg.extensions:
                raise ValueError(f"Unsupported input extension for {item} (allowed: {cfg.extensions})")
            files.append(str(p.resolve()))
        # stable & unique
        return tuple(sorted(set(files)))

    raise ValueError("No inputs provided. Use input=... or input_dir=...")