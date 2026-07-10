"""
Create one-vs-rest benchmarking datasets for confidence score ROC evaluation.

For each relationship, the target class is treated as positive and all other
relationship classes are treated as negative.

Each output row contains:
    - id1
    - id2
    - pairkey
    - true_label
    - target_relationship
    - binary_label

Example:
    Enantiomer ROC dataset:
        positives = enantiomer pairs
        negatives = diastereomer, stereo-resolution, and protomer pairs
"""

import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent.parent / "benchmarking" / "data"

ENANTIOMER_DATASET_PATH = DATA_DIR / "enantiomer_control_set.csv"
DIASTEREOMER_DATASET_PATH = DATA_DIR / "diastereomer_control_set.csv"
STEREO_RES_DATASET_PATH = DATA_DIR / "stereo_resolution_pairs.csv"
PROTOMER_DATASET_PATH = DATA_DIR / "protomers_control_set.csv"

NUM_SAMPLES_PER_CLASS = 1000
REMOVE_CONTRADICTORY_PAIRS = True


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def make_undirected_pairkey(row: pd.Series) -> str:
    """
    Create an undirected pairkey.

    This treats A_B and B_A as the same molecular pair.
    """
    id1 = str(row["id1"])
    id2 = str(row["id2"])

    return "_".join(sorted([id1, id2]))


def standardise_common_columns(df: pd.DataFrame, true_label: str) -> pd.DataFrame:
    """
    Standardise common columns across relationship datasets.
    """
    df = df.copy()

    if "label" in df.columns and "true_label" not in df.columns:
        df = df.rename(columns={"label": "true_label"})

    if "true_label" not in df.columns:
        df["true_label"] = true_label
    else:
        df["true_label"] = true_label

    if "id1" not in df.columns or "id2" not in df.columns:
        raise ValueError(
            f"Dataset labelled '{true_label}' must contain id1 and id2 columns after standardisation."
        )

    df["id1"] = df["id1"].astype(str)
    df["id2"] = df["id2"].astype(str)

    df["pairkey"] = df.apply(make_undirected_pairkey, axis=1)

    return df


def standardise_enantiomer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise enantiomer data."""
    return standardise_common_columns(df, true_label="Enantiomers")


def standardise_diastereomer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise diastereomer data."""
    return standardise_common_columns(df, true_label="Diastereomers")


def standardise_stereo_res_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise stereo-resolution data.

    parent_label and child_label are converted to id1 and id2.
    """
    df = df.copy()

    df = df.drop(
        columns=[
            col
            for col in ["mnxparent_label", "mnxchild_label"]
            if col in df.columns
        ],
        errors="ignore",
    )

    if "parent_label" in df.columns and "child_label" in df.columns:
        df["id1"] = df["parent_label"]
        df["id2"] = df["child_label"]

        df["id1"] = df["id1"].astype(str).str.replace("chebi:", "CHEBI_", regex=False)
        df["id2"] = df["id2"].astype(str).str.replace("chebi:", "CHEBI_", regex=False)

    df = df.drop(
        columns=[
            col
            for col in ["parent_label", "child_label"]
            if col in df.columns
        ],
        errors="ignore",
    )

    return standardise_common_columns(df, true_label="Stereo-resolution pairs")


def standardise_protomer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise protomer data."""
    df = df.copy()

    if "id_left" in df.columns and "id_right" in df.columns:
        df["id1"] = df["id_left"]
        df["id2"] = df["id_right"]

    df = df.drop(
        columns=[
            col
            for col in ["id_left", "id_right"]
            if col in df.columns
        ],
        errors="ignore",
    )

    return standardise_common_columns(df, true_label="Protomers")


def report_duplicate_pairkeys(name: str, df: pd.DataFrame) -> None:
    """Report duplicate pairkeys within a single relationship dataset."""
    duplicate_count = df.duplicated(subset=["pairkey"]).sum()

    if duplicate_count > 0:
        print(f"{name}: found {duplicate_count} duplicate pairkeys within this dataset.")
    else:
        print(f"{name}: no duplicate pairkeys within this dataset.")


def remove_internal_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate pairkeys within one relationship dataset.

    Keeps the first occurrence.
    """
    return df.drop_duplicates(subset=["pairkey"], keep="first").reset_index(drop=True)


def find_contradictory_pairkeys(datasets: dict[str, pd.DataFrame]) -> set[str]:
    """
    Find pairkeys assigned to more than one true_label across datasets.
    """
    combined = pd.concat(datasets.values(), ignore_index=True)

    pairkey_label_counts = (
        combined.groupby("pairkey")["true_label"]
        .nunique()
        .reset_index(name="n_labels")
    )

    contradictory_pairkeys = set(
        pairkey_label_counts.loc[
            pairkey_label_counts["n_labels"] > 1,
            "pairkey"
        ]
    )

    return contradictory_pairkeys


def remove_pairkeys(df: pd.DataFrame, pairkeys_to_remove: set[str]) -> pd.DataFrame:
    """Remove rows with pairkeys found in pairkeys_to_remove."""
    return df.loc[~df["pairkey"].isin(pairkeys_to_remove)].reset_index(drop=True)


def create_full_combined_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all relationship datasets into one full benchmark table.

    Each pair appears once after duplicate and contradictory pair removal.
    """
    combined = pd.concat(
        datasets.values(),
        ignore_index=True,
    )

    combined = combined.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    return combined


def write_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the combined benchmark dataset with key columns first.
    """
    priority_columns = [
        "id1",
        "id2",
        "pairkey",
        "true_label",
    ]

    existing_priority_columns = [
        col for col in priority_columns if col in df.columns
    ]

    remaining_columns = [
        col for col in df.columns if col not in existing_priority_columns
    ]

    df = df[existing_priority_columns + remaining_columns]

    df.to_csv(output_path, index=False)


def main() -> None:
    enantiomer_data = load_csv_data(ENANTIOMER_DATASET_PATH)
    diastereomer_data = load_csv_data(DIASTEREOMER_DATASET_PATH)
    stereo_res_data = load_csv_data(STEREO_RES_DATASET_PATH)
    protomer_data = load_csv_data(PROTOMER_DATASET_PATH)

    print(f"Raw enantiomer dataset: {len(enantiomer_data)} rows.")
    print(f"Raw diastereomer dataset: {len(diastereomer_data)} rows.")
    print(f"Raw stereo-resolution dataset: {len(stereo_res_data)} rows.")
    print(f"Raw protomer dataset: {len(protomer_data)} rows.")

    enantiomer_data = standardise_enantiomer_data(enantiomer_data)
    diastereomer_data = standardise_diastereomer_data(diastereomer_data)
    stereo_res_data = standardise_stereo_res_data(stereo_res_data)
    protomer_data = standardise_protomer_data(protomer_data)

    datasets = {
        "Enantiomers": enantiomer_data,
        "Diastereomers": diastereomer_data,
        "Stereo-resolution pairs": stereo_res_data,
        "Protomers": protomer_data,
    }

    print("\nChecking duplicate pairkeys within each dataset.")
    for name, df in datasets.items():
        report_duplicate_pairkeys(name, df)

    datasets = {
        name: remove_internal_duplicates(df)
        for name, df in datasets.items()
    }

    contradictory_pairkeys = find_contradictory_pairkeys(datasets)

    print(
        f"\nFound {len(contradictory_pairkeys)} pairkeys assigned to more than one relationship class."
    )

    if contradictory_pairkeys and REMOVE_CONTRADICTORY_PAIRS:
        print("Removing contradictory pairkeys from all datasets.")

        datasets = {
            name: remove_pairkeys(df, contradictory_pairkeys)
            for name, df in datasets.items()
        }

    elif contradictory_pairkeys:
        print("Contradictory pairkeys were found but not removed.")

    print("\nFinal standardised dataset sizes:")
    for name, df in datasets.items():
        print(f"{name}: {len(df)} rows.")

    combined_dataset = create_full_combined_dataset(datasets)

    output_path = BASE_DIR / "combined_relationship_benchmarking_dataset.csv"

    write_dataset(
        combined_dataset,
        output_path,
    )

    print(
        f"\nSaved {output_path.name} with {len(combined_dataset)} rows."
    )

    print("\nClass counts:")
    print(combined_dataset["true_label"].value_counts())


if __name__ == "__main__":
    main()