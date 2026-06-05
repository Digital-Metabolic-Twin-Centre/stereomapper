#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

FILTERED_BENCHMARK_PATH = (
    DATA_DIR / "combined_relationship_benchmarking_dataset_filtered.csv"
)

MAPPED_RESULTS_PATH = (
    "/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/results/all_benchmarking_data_for_threshold_analysis.csv"
)

OUTPUT_PATH = (
    DATA_DIR / "benchmark_to_confidence_reconciliation_table.csv"
)


def normalise_label(value):
    if pd.isna(value):
        return value

    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    aliases = {
        "enantiomers": "enantiomer",
        "diastereomers": "diastereomer",
        "stereo_resolution_pair": "stereo_resolution",
        "stereo_resolution_pairs": "stereo_resolution",
        "stereoresolution": "stereo_resolution",
        "protomers": "protomer",
    }

    return aliases.get(value, value)


def make_pair_id(df):
    if "pair_id" in df.columns:
        return df["pair_id"].astype(str)

    if "pairkey" in df.columns:
        return df["pairkey"].astype(str)

    if {"id1", "id2"}.issubset(df.columns):
        return df.apply(
            lambda row: "_".join(sorted([str(row["id1"]), str(row["id2"])])),
            axis=1,
        )

    raise ValueError("No pair_id, pairkey, or id1/id2 columns found.")


def main():
    benchmark_df = pd.read_csv(FILTERED_BENCHMARK_PATH)
    mapped_df = pd.read_csv(MAPPED_RESULTS_PATH)

    benchmark_df = benchmark_df.copy()
    mapped_df = mapped_df.copy()

    benchmark_df["pair_id"] = make_pair_id(benchmark_df)
    mapped_df["pair_id"] = make_pair_id(mapped_df)

    benchmark_df["true_label_norm"] = benchmark_df["true_class"].apply(normalise_label)
    mapped_df["true_label_norm"] = mapped_df["true_class"].apply(normalise_label)
    mapped_df["predicted_class_norm"] = mapped_df["predicted_class"].apply(normalise_label)

    mapped_keep = mapped_df[
        [
            "pair_id",
            "predicted_class",
            "predicted_class_norm",
            "score_S",
        ]
    ].drop_duplicates(subset=["pair_id"], keep="first")

    merged = benchmark_df.merge(
        mapped_keep,
        on="pair_id",
        how="left",
    )

    merged["has_prediction"] = merged["predicted_class"].notna()
    merged["has_score"] = pd.to_numeric(
        merged["score_S"],
        errors="coerce",
    ).notna()

    merged["is_scored_prediction"] = (
        merged["has_prediction"] & merged["has_score"]
    )

    merged["is_correct_scored_prediction"] = (
        merged["is_scored_prediction"]
        & (merged["true_label_norm"] == merged["predicted_class_norm"])
    )

    merged["is_incorrect_scored_prediction"] = (
        merged["is_scored_prediction"]
        & (merged["true_label_norm"] != merged["predicted_class_norm"])
    )

    rows = []

    for label, group in merged.groupby("true_label_norm", dropna=False):
        initial_pairs_after_molfile_filter = len(group)

        scored = group[group["is_scored_prediction"]]
        unscored_or_unclassified = group[~group["is_scored_prediction"]]

        correct_scored = group["is_correct_scored_prediction"].sum()
        incorrect_scored = group["is_incorrect_scored_prediction"].sum()

        rows.append(
            {
                "relationship": label,
                "pairs_after_molfile_filter": initial_pairs_after_molfile_filter,
                "scored_predictions": len(scored),
                "unscored_or_unclassified": len(unscored_or_unclassified),
                "correct_scored_predictions": int(correct_scored),
                "incorrect_scored_predictions": int(incorrect_scored),
                "scored_prediction_coverage": len(scored) / initial_pairs_after_molfile_filter,
                "scored_prediction_precision": (
                    correct_scored / len(scored) if len(scored) else None
                ),
            }
        )

    summary_df = pd.DataFrame(rows)

    total_row = {
        "relationship": "all",
        "pairs_after_molfile_filter": len(merged),
        "scored_predictions": int(merged["is_scored_prediction"].sum()),
        "unscored_or_unclassified": int((~merged["is_scored_prediction"]).sum()),
        "correct_scored_predictions": int(merged["is_correct_scored_prediction"].sum()),
        "incorrect_scored_predictions": int(merged["is_incorrect_scored_prediction"].sum()),
        "scored_prediction_coverage": merged["is_scored_prediction"].sum() / len(merged),
        "scored_prediction_precision": (
            merged["is_correct_scored_prediction"].sum()
            / merged["is_scored_prediction"].sum()
        ),
    }

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([total_row])],
        ignore_index=True,
    )

    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(summary_df)
    print(f"\nSaved reconciliation table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()