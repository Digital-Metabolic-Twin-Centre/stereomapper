"""Run configuration for ``stereomapper reconcile propose`` (contracts/input-files.md)."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from stereomapper.domain.exceptions import ConfigurationError
from stereomapper.reconciliation.hashing import sha256_file

ALIGN_V1 = "align-v1"
RETRIEVE_V1 = "retrieve-v1"
RANK_V1 = "rank-v1"
COFACTORS_V1 = "cofactors-v1"
OUTPUT_SCHEMA_VERSION = "reconcile-out-v1"
CONFIG_VERSION = 1

_KNOWN_VERSIONS = {
    "alignment": {ALIGN_V1},
    "retrieval": {RETRIEVE_V1},
    "ranking": {RANK_V1},
}
_KNOWN_ROUTES = {"shared_cluster", "related_cluster", "xref"}
_REQUIRED_INPUTS = ("extract_dir", "stereomapper_db", "version_tag")


def default_config() -> dict[str, Any]:
    text = (
        resources.files("stereomapper.reconciliation")
        .joinpath("resources/default_config.json")
        .read_text(encoding="utf-8")
    )
    data: dict[str, Any] = json.loads(text)
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class RunConfig:
    """Resolved configuration. ``raw`` holds the merged JSON as recorded in the manifest."""

    raw: dict[str, Any]
    config_dir: Path
    extract_dir: Path
    stereomapper_db: Path
    version_tag: str
    baseline: Path | None
    baseline_version: str

    @property
    def exact_max_side(self) -> int:
        return int(self.raw["alignment"]["exact_max_side"])

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(self.raw["retrieval"]["routes"])

    @property
    def rhea_status_include(self) -> tuple[str, ...]:
        return tuple(self.raw["scope"]["rhea_status"]["include"])

    @property
    def cofactor_spec(self) -> str:
        return str(self.raw["cofactors"]["list"])

    def relative(self, path: Path) -> str:
        """Path relative to the config file, used so run IDs don't depend on location."""
        return Path(os.path.relpath(path, self.config_dir)).as_posix()


def _resolve(config_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _validate(raw: dict[str, Any]) -> None:
    if raw.get("config_version") != CONFIG_VERSION:
        raise ConfigurationError(
            f"Unsupported config_version {raw.get('config_version')!r}",
            config_field="config_version",
        )
    for section, known in _KNOWN_VERSIONS.items():
        version = raw.get(section, {}).get("version")
        if version not in known:
            raise ConfigurationError(
                f"Unknown {section}.version {version!r}; expected one of {sorted(known)}",
                config_field=f"{section}.version",
            )
    routes = raw["retrieval"].get("routes") or []
    unknown = sorted(set(routes) - _KNOWN_ROUTES)
    if unknown or not routes:
        raise ConfigurationError(
            f"Invalid retrieval.routes {routes!r}", config_field="retrieval.routes"
        )
    if raw["retrieval"].get("cap") is not None:
        raise ConfigurationError(
            "retrieval.cap is not supported in retrieve-v1; a cap needs measured benchmark "
            "impact before adoption (FR-018)",
            config_field="retrieval.cap",
        )
    status = raw["scope"]["rhea_status"]
    allowed = {"approved", "preliminary", "obsolete"}
    for key in ("include", "exclude"):
        bad = sorted(set(status.get(key, [])) - allowed)
        if bad:
            raise ConfigurationError(
                f"Unknown Rhea status values in scope.rhea_status.{key}: {bad}",
                config_field=f"scope.rhea_status.{key}",
            )
    if set(status.get("include", [])) & set(status.get("exclude", [])):
        raise ConfigurationError(
            "A Rhea status cannot be both included and excluded",
            config_field="scope.rhea_status",
        )
    max_side = raw["alignment"].get("exact_max_side")
    if not isinstance(max_side, int) or max_side < 1 or max_side > 20:
        raise ConfigurationError(
            "alignment.exact_max_side must be an integer between 1 and 20",
            config_field="alignment.exact_max_side",
        )
    extra_scope = sorted(set(raw.get("scope", {})) - {"rhea_status"})
    if extra_scope:
        raise ConfigurationError(
            f"Unsupported scope rules {extra_scope}: pathway or subsystem exclusion is not "
            "applied before proposals (FR-005); use `reconcile flag-subsystems` afterwards",
            config_field="scope",
        )


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    try:
        user = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Config file is not valid JSON: {config_path}: {exc}") from exc
    raw = _merge(default_config(), user)
    _validate(raw)

    config_dir = config_path.parent
    inputs = raw["inputs"]
    for key in _REQUIRED_INPUTS:
        if not inputs.get(key):
            raise ConfigurationError(f"inputs.{key} is required", config_field=f"inputs.{key}")

    extract_dir = _resolve(config_dir, inputs["extract_dir"])
    db_path = _resolve(config_dir, inputs["stereomapper_db"])
    if not extract_dir.is_dir():
        raise ConfigurationError(
            f"inputs.extract_dir is not a directory: {extract_dir}",
            config_field="inputs.extract_dir",
        )
    if not db_path.is_file():
        raise ConfigurationError(
            f"inputs.stereomapper_db not found: {db_path}", config_field="inputs.stereomapper_db"
        )
    baseline = _resolve(config_dir, inputs["baseline"]) if inputs.get("baseline") else None
    if baseline is not None and not baseline.is_file():
        raise ConfigurationError(
            f"inputs.baseline not found: {baseline}", config_field="inputs.baseline"
        )

    for rel, expected in sorted((inputs.get("checksums") or {}).items()):
        target = _resolve(config_dir, rel)
        if not target.is_file():
            raise ConfigurationError(
                f"Checksummed input not found: {rel}", config_field="inputs.checksums"
            )
        actual = sha256_file(target)
        if actual != expected:
            raise ConfigurationError(
                f"Checksum mismatch for {rel}: expected {expected}, found {actual}",
                config_field="inputs.checksums",
            )

    return RunConfig(
        raw=raw,
        config_dir=config_dir,
        extract_dir=extract_dir,
        stereomapper_db=db_path,
        version_tag=str(inputs["version_tag"]),
        baseline=baseline,
        baseline_version=str(inputs.get("baseline_version") or ""),
    )
