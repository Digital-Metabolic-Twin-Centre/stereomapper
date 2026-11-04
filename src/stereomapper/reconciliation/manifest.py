"""Run record (run_manifest.json) and run identity (research R10, Principle II)."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from stereomapper.domain.relationship_terms import relationship_term_table
from stereomapper.reconciliation.config import OUTPUT_SCHEMA_VERSION, RunConfig
from stereomapper.reconciliation.evidence import CofactorList
from stereomapper.reconciliation.hashing import canonical_json, sha256_file, sha256_text, short_hash

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VOCAB_CSV = _REPO_ROOT / "docs" / "ontology" / "relationship_terms.csv"

# Fields that describe the execution, not the pinned inputs; excluded from run_id.
_NOT_IN_RUN_ID = {
    "run_id",
    "started_at",
    "python_version",
    "funnel",
    "greedy_alignment_count",
    "trace",
    "nondeterminism",
}


def smro_vocabulary() -> dict[str, str | None]:
    table = sorted(relationship_term_table().items())
    return {
        "terms_sha256": "sha256:" + sha256_text(canonical_json(table)),
        "relationship_terms_csv": sha256_file(_VOCAB_CSV) if _VOCAB_CSV.is_file() else None,
    }


def code_version() -> dict[str, str]:
    try:
        version = metadata.version("stereomapper")
    except metadata.PackageNotFoundError:
        version = "unknown"
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"stereomapper_version": version, "git_commit": commit}


def build_run_record(
    config: RunConfig,
    extract_manifest: dict[str, Any],
    input_paths: dict[str, Path],
    cofactors: CofactorList,
) -> dict[str, Any]:
    """Pinned inputs and configuration. ``run_id`` is a hash of this record (R10)."""
    inputs = {
        name: {"path": config.relative(path), "checksum": sha256_file(path)}
        for name, path in sorted(input_paths.items())
    }
    pinned_config = json.loads(canonical_json(config.raw))
    # Record input paths relative to the config file, and drop the optional checksum map
    # (checksums are recorded per input), so run_id doesn't depend on location (R10).
    pinned_config["inputs"].pop("checksums", None)
    pinned_config["inputs"]["extract_dir"] = config.relative(config.extract_dir)
    pinned_config["inputs"]["stereomapper_db"] = config.relative(config.stereomapper_db)
    if config.baseline is not None:
        pinned_config["inputs"]["baseline"] = config.relative(config.baseline)
    record: dict[str, Any] = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "extract": {
            "kg_snapshot": extract_manifest.get("kg_snapshot"),
            "vmh_release": extract_manifest.get("vmh_release"),
            "rhea_release": extract_manifest.get("rhea_release"),
            "extraction_query": extract_manifest.get("extraction_query"),
            "extracted_at": extract_manifest.get("extracted_at"),
        },
        "inputs": inputs,
        "stereomapper_results": {
            "path": config.relative(config.stereomapper_db),
            "checksum": inputs["stereomapper_db"]["checksum"],
            "version_tag": config.version_tag,
        },
        "baseline_version": config.baseline_version,
        "cofactors": {
            "version": cofactors.version,
            "checksum": cofactors.checksum,
            "source": cofactors.source,
        },
        "smro_vocabulary": smro_vocabulary(),
        "config": pinned_config,
        **code_version(),
    }
    record["run_id"] = "RUN-" + short_hash(
        canonical_json({k: v for k, v in record.items() if k not in _NOT_IN_RUN_ID})
    )
    record["python_version"] = platform.python_version()
    record["nondeterminism"] = []
    record["started_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return record


RUN_OUTPUTS_FROM_INPUTS = {
    "reaction_proposals.tsv": ("reactions", "participants", "stereomapper_db", "baseline"),
    "participant_evidence.tsv": ("reactions", "participants", "stereomapper_db"),
    "alternative_alignments.tsv": ("reactions", "participants", "stereomapper_db"),
    "scope_exclusions.tsv": ("reactions",),
    "vmh_subsystems.tsv": ("subsystems",),
    "funnel.tsv": ("reactions", "participants", "stereomapper_db"),
    "reactions_summary.tsv": ("reactions", "participants", "stereomapper_db"),
    "candidates.tsv": ("reactions", "participants", "stereomapper_db"),
    "baseline_pairs.tsv": ("baseline", "reactions"),
}


def trace_section(record: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map each output file to the checksums of the inputs it derives from (US4-3)."""
    inputs = record["inputs"]
    return {
        output: {name: inputs[name]["checksum"] for name in names if name in inputs}
        for output, names in sorted(RUN_OUTPUTS_FROM_INPUTS.items())
    }
