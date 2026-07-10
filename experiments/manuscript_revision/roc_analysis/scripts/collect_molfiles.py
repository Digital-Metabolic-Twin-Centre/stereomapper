"""
Filter the combined relationship benchmarking dataset to rows where both
molecule structures exist, then collect the required mol files from
benchmarking_data subdirectories into one folder.
"""

import pandas as pd
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

COMBINED_DATASET_PATH = DATA_DIR / "combined_relationship_benchmarking_dataset.csv"
FILTERED_OUTPUT_PATH = DATA_DIR / "combined_relationship_benchmarking_dataset_filtered.csv"

BENCHMARKING_MOLFILES_DIR = DATA_DIR / "benchmarking_data"

COLLECTED_MOLFILES_DIR = BASE_DIR / "combined_relationship_dataset_molfiles"

MISSING_REPORT_PATH = DATA_DIR / "missing_molfiles_report.csv"


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def normalise_chebi_id(chem_id: str) -> str:
    """
    Convert supported CheBI ID formats to the local molfile naming format.

    Examples:
        chebi:1234 -> CHEBI_1234
        CHEBI:1234 -> CHEBI_1234
        CHEBI_1234 -> CHEBI_1234
    """
    chem_id = str(chem_id).strip()

    if chem_id.lower().startswith("chebi:"):
        return chem_id.replace("chebi:", "CHEBI_").replace("CHEBI:", "CHEBI_")

    if chem_id.startswith("CHEBI_"):
        return chem_id

    return chem_id


def build_molfile_index(molfiles_root_dir: Path) -> dict[str, Path]:
    """
    Search all subdirectories and build a lookup from CHEBI ID to molfile path.

    Supports:
        CHEBI_1234.mol
        CHEBI_1234.sdf
    """
    if not molfiles_root_dir.exists():
        raise FileNotFoundError(f"Molfile directory does not exist: {molfiles_root_dir}")

    molfile_index = {}

    for molfile_path in sorted(molfiles_root_dir.rglob("*")):
        if molfile_path.suffix.lower() not in {".mol", ".sdf"}:
            continue

        normalised_id = normalise_chebi_id(molfile_path.stem)

        if not normalised_id.startswith("CHEBI_"):
            continue

        if normalised_id in molfile_index:
            print(
                f"Duplicate molfile for {normalised_id}: "
                f"keeping {molfile_index[normalised_id]}, ignoring {molfile_path}"
            )
            continue

        molfile_index[normalised_id] = molfile_path

    print(f"Indexed {len(molfile_index)} unique molfiles from {molfiles_root_dir}")

    return molfile_index


def get_molfile_path(chem_id: str, molfile_index: dict[str, Path]) -> Path | None:
    """Return matching molfile path for a chemical ID."""
    normalised_id = normalise_chebi_id(chem_id)
    return molfile_index.get(normalised_id)


def molfile_exists(chem_id: str, molfile_index: dict[str, Path]) -> bool:
    """Check whether a molfile exists for a chemical ID."""
    return get_molfile_path(chem_id, molfile_index) is not None


def filter_dataset_for_available_molfiles(
    dataset_path: Path,
    filtered_output_path: Path,
    molfile_index: dict[str, Path],
) -> pd.DataFrame:
    """
    Remove rows where id1 or id2 lacks a molfile.

    Returns a dataframe describing the missing entries.
    """
    df = load_csv_data(dataset_path)

    required_columns = {"id1", "id2", "pairkey", "true_label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_path} is missing required columns: {sorted(missing_columns)}"
        )

    missing_records = []
    keep_mask = []

    for _, row in df.iterrows():
        id1 = row["id1"]
        id2 = row["id2"]

        id1_path = get_molfile_path(id1, molfile_index)
        id2_path = get_molfile_path(id2, molfile_index)

        keep_row = id1_path is not None and id2_path is not None
        keep_mask.append(keep_row)

        if id1_path is None:
            missing_records.append(
                {
                    "dataset": dataset_path.name,
                    "pairkey": row.get("pairkey"),
                    "missing_id": id1,
                    "normalised_id": normalise_chebi_id(id1),
                    "true_label": row.get("true_label"),
                }
            )

        if id2_path is None:
            missing_records.append(
                {
                    "dataset": dataset_path.name,
                    "pairkey": row.get("pairkey"),
                    "missing_id": id2,
                    "normalised_id": normalise_chebi_id(id2),
                    "true_label": row.get("true_label"),
                }
            )

    filtered_df = df.loc[keep_mask].reset_index(drop=True)
    filtered_df.to_csv(filtered_output_path, index=False)

    n_removed = len(df) - len(filtered_df)

    print(f"\n{dataset_path.name}")
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Removed rows: {n_removed}")
    print(f"Saved filtered dataset to: {filtered_output_path}")

    return pd.DataFrame(missing_records)


def collect_molfiles_for_dataset(
    dataset_path: Path,
    output_dir: Path,
    molfile_index: dict[str, Path],
) -> None:
    """
    Collect mol files for a filtered dataset.
    """
    df = load_csv_data(dataset_path)

    required_columns = {"id1", "id2"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_path} is missing required columns: {sorted(missing_columns)}"
        )

    unique_ids = set(df["id1"]).union(set(df["id2"]))

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    already_present = 0

    for chem_id in sorted(unique_ids):
        normalised_id = normalise_chebi_id(chem_id)

        if not normalised_id.startswith("CHEBI_"):
            print(f"Unknown chemical ID format: {chem_id}")
            skipped += 1
            continue

        source_path = get_molfile_path(normalised_id, molfile_index)

        if source_path is None:
            print(f"Mol file not found for {chem_id}")
            skipped += 1
            continue

        output_path = output_dir / f"{normalised_id}{source_path.suffix.lower()}"

        if output_path.exists():
            already_present += 1
            continue

        shutil.copy(source_path, output_path)
        copied += 1

    print(f"\nCopied {copied} molfiles to {output_dir}")
    print(f"Already present: {already_present}")
    print(f"Skipped {skipped} IDs.")


def summarise_filtered_dataset(dataset_path: Path) -> None:
    """
    Print counts by true_label after filtering.
    """
    df = load_csv_data(dataset_path)

    print(f"\nSummary for {dataset_path.name}")

    if "true_label" in df.columns:
        print("True label counts:")
        print(df["true_label"].value_counts(dropna=False))

    print(f"\nUnique molecule IDs: {len(set(df['id1']).union(set(df['id2'])))}")

    if "pairkey" in df.columns:
        print(f"Unique pairkeys: {df['pairkey'].nunique()}")


def main() -> None:
    molfile_index = build_molfile_index(BENCHMARKING_MOLFILES_DIR)

    missing_report = filter_dataset_for_available_molfiles(
        dataset_path=COMBINED_DATASET_PATH,
        filtered_output_path=FILTERED_OUTPUT_PATH,
        molfile_index=molfile_index,
    )

    if len(missing_report) > 0:
        missing_report = missing_report.drop_duplicates()
        missing_report.to_csv(MISSING_REPORT_PATH, index=False)
        print(f"\nSaved missing molfile report to: {MISSING_REPORT_PATH}")
    else:
        print("\nNo missing molfiles found.")

    collect_molfiles_for_dataset(
        dataset_path=FILTERED_OUTPUT_PATH,
        output_dir=COLLECTED_MOLFILES_DIR,
        molfile_index=molfile_index,
    )

    summarise_filtered_dataset(FILTERED_OUTPUT_PATH)


if __name__ == "__main__":
    main()