from pathlib import Path
import re
import shutil

SDF_DIR = Path("/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/data/benchmarking_data/protomer_benchmark_data")
DRY_RUN = False  # set to False after checking the proposed renames


def extract_chebi_id(sdf_path: Path) -> str | None:
    text = sdf_path.read_text(errors="replace")

    match = re.search(
        r">\s*<ChEBI ID>\s*\n\s*(CHEBI[:_]\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    chebi_id = match.group(1).upper()
    chebi_id = chebi_id.replace(":", "_")
    return chebi_id


def unique_target_path(target: Path) -> Path:
    """
    Avoid overwriting if two files contain the same ChEBI ID.
    Example:
      CHEBI_71616.sdf
      CHEBI_71616__2.sdf
    """
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix

    i = 2
    while True:
        candidate = target.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def rename_sdf_files(sdf_dir: Path, dry_run: bool = True) -> None:
    sdf_files = sorted(sdf_dir.glob("*.sdf"))

    renamed = 0
    skipped = 0

    for sdf_path in sdf_files:
        chebi_id = extract_chebi_id(sdf_path)

        if chebi_id is None:
            print(f"SKIP, no ChEBI ID found: {sdf_path.name}")
            skipped += 1
            continue

        target = sdf_path.with_name(f"{chebi_id}.sdf")
        target = unique_target_path(target)

        if sdf_path.name == target.name:
            print(f"SKIP, already named correctly: {sdf_path.name}")
            skipped += 1
            continue

        print(f"{sdf_path.name} -> {target.name}")

        if not dry_run:
            shutil.move(str(sdf_path), str(target))

        renamed += 1

    print()
    print(f"Renamed: {renamed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    rename_sdf_files(SDF_DIR, dry_run=DRY_RUN)