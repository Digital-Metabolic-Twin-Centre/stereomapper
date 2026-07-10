#!/usr/bin/env python3

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


RELATIONSHIP_CLASSES = [
    "enantiomer",
    "diastereomer",
    "stereo_resolution",
    "protomer",
]


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def normalise_class_name(value):
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
        "stereo_resolution_pairs": "stereo_resolution",
        "protomers": "protomer",
    }

    return aliases.get(value, value)


def validate_input(df):
    required_columns = {
        "pair_id",
        "true_class",
        "predicted_class",
        "score_S",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    if df["score_S"].isna().any():
        raise ValueError("Input file contains missing score_S values.")

    df["score_S"] = pd.to_numeric(df["score_S"], errors="coerce")

    if df["score_S"].isna().any():
        raise ValueError("score_S contains non-numeric values.")

    df["true_class"] = df["true_class"].apply(normalise_class_name)
    df["predicted_class"] = df["predicted_class"].apply(normalise_class_name)

    if df["true_class"].isna().any():
        raise ValueError("Input file contains missing true_class values.")

    if df["predicted_class"].isna().any():
        raise ValueError("Input file contains missing predicted_class values.")

    observed_true = set(df["true_class"].unique())
    observed_predicted = set(df["predicted_class"].unique())

    expected = set(RELATIONSHIP_CLASSES)

    missing_true = expected - observed_true
    missing_predicted = expected - observed_predicted

    if missing_true:
        print(
            "Warning: these expected true classes were not found:",
            sorted(missing_true),
        )

    if missing_predicted:
        print(
            "Warning: these expected predicted classes were not found:",
            sorted(missing_predicted),
        )

    return df


def calculate_threshold_metrics(y_true, scores, thresholds):
    rows = []

    for threshold in thresholds:
        y_pred_high_confidence = scores >= threshold

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred_high_confidence,
            labels=[0, 1],
        ).ravel()

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        sensitivity = recall
        specificity = safe_divide(tn, tn + fp)
        false_positive_rate = safe_divide(fp, fp + tn)
        false_negative_rate = safe_divide(fn, fn + tp)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        if not np.isnan(sensitivity) and not np.isnan(specificity):
            balanced_accuracy = (sensitivity + specificity) / 2
            youden_j = sensitivity + specificity - 1
        else:
            balanced_accuracy = np.nan
            youden_j = np.nan

        n_high_confidence = tp + fp
        high_confidence_fraction = safe_divide(n_high_confidence, len(y_true))

        rows.append(
            {
                "threshold": threshold,
                "n_predictions_for_class": len(y_true),
                "n_high_confidence": n_high_confidence,
                "high_confidence_fraction": high_confidence_fraction,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "youden_j": youden_j,
            }
        )

    return pd.DataFrame(rows)


def select_precision_threshold(metrics_df, target_precision, min_true_positives):
    eligible = metrics_df[
        (metrics_df["precision"] >= target_precision)
        & (metrics_df["tp"] >= min_true_positives)
        & (metrics_df["n_high_confidence"] > 0)
    ].copy()

    if eligible.empty:
        return None

    # Choose the lowest threshold that reaches target precision.
    # This maximises retained high-confidence predictions while meeting reliability.
    selected = eligible.sort_values(
        ["threshold", "recall", "n_high_confidence"],
        ascending=[True, False, False],
    ).iloc[0]

    return selected


def select_best_f1_threshold(metrics_df):
    valid = metrics_df.dropna(subset=["f1"]).copy()

    if valid.empty:
        return None

    selected = valid.sort_values(
        ["f1", "precision", "recall", "n_high_confidence"],
        ascending=[False, False, False, False],
    ).iloc[0]

    return selected


def plot_roc_curve(y_true, scores, target_class, output_dir):
    if len(np.unique(y_true)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve: predicted {target_class}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path = output_dir / f"{target_class}_roc_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auroc


def plot_precision_recall_curve(y_true, scores, target_class, output_dir):
    if len(np.unique(y_true)) < 2:
        return np.nan

    precision, recall, _ = precision_recall_curve(y_true, scores)
    auprc = average_precision_score(y_true, scores)

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve: predicted {target_class}")
    plt.legend(loc="lower left")
    plt.tight_layout()

    output_path = output_dir / f"{target_class}_precision_recall_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auprc


def analyse_predicted_class(
    df,
    target_class,
    output_dir,
    target_precision,
    min_true_positives,
):
    class_df = df[df["predicted_class"] == target_class].copy()

    if class_df.empty:
        print(f"Warning: no predictions found for class: {target_class}")
        return None

    class_df["correct_prediction"] = (
        class_df["true_class"] == class_df["predicted_class"]
    ).astype(int)

    y_true = class_df["correct_prediction"].to_numpy(dtype=int)
    scores = class_df["score_S"].to_numpy(dtype=float)

    n_predictions = len(class_df)
    n_correct = int(np.sum(y_true == 1))
    n_incorrect = int(np.sum(y_true == 0))

    thresholds = np.arange(0, 101, 1)

    metrics_df = calculate_threshold_metrics(
        y_true=y_true,
        scores=scores,
        thresholds=thresholds,
    )

    threshold_path = output_dir / f"{target_class}_threshold_metrics.csv"
    metrics_df.to_csv(threshold_path, index=False)

    class_rows_path = output_dir / f"{target_class}_predicted_rows.csv"
    class_df.to_csv(class_rows_path, index=False)

    if len(np.unique(y_true)) < 2:
        print(
            f"Warning: predicted {target_class} has only one correctness label. "
            "ROC and PR curves were skipped."
        )
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = plot_roc_curve(
            y_true=y_true,
            scores=scores,
            target_class=target_class,
            output_dir=output_dir,
        )

        auprc = plot_precision_recall_curve(
            y_true=y_true,
            scores=scores,
            target_class=target_class,
            output_dir=output_dir,
        )

    selected = select_precision_threshold(
        metrics_df=metrics_df,
        target_precision=target_precision,
        min_true_positives=min_true_positives,
    )

    fallback = False

    if selected is None:
        selected = select_best_f1_threshold(metrics_df)
        fallback = True

    if selected is None:
        print(f"Warning: no valid threshold found for class: {target_class}")
        return None

    summary = {
        "target_predicted_class": target_class,
        "n_predictions_for_class": n_predictions,
        "n_correct_predictions": n_correct,
        "n_incorrect_predictions": n_incorrect,
        "observed_class_precision_without_threshold": safe_divide(
            n_correct,
            n_predictions,
        ),
        "auroc": auroc,
        "auprc": auprc,
        "target_precision": target_precision,
        "min_true_positives": min_true_positives,
        "selected_threshold": selected["threshold"],
        "threshold_selection": (
            f"lowest threshold with precision >= {target_precision}"
            if not fallback
            else "fallback: maximum F1 because target precision was not reached"
        ),
        "tp": int(selected["tp"]),
        "fp": int(selected["fp"]),
        "tn": int(selected["tn"]),
        "fn": int(selected["fn"]),
        "n_high_confidence": int(selected["n_high_confidence"]),
        "high_confidence_fraction": selected["high_confidence_fraction"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "sensitivity": selected["sensitivity"],
        "specificity": selected["specificity"],
        "false_positive_rate": selected["false_positive_rate"],
        "false_negative_rate": selected["false_negative_rate"],
        "f1": selected["f1"],
        "balanced_accuracy": selected["balanced_accuracy"],
        "youden_j": selected["youden_j"],
    }

    return summary


def make_confusion_matrix(df, output_dir):
    labels = sorted(set(df["true_class"]).union(set(df["predicted_class"])))

    confusion = pd.crosstab(
        df["true_class"],
        df["predicted_class"],
        rownames=["true_class"],
        colnames=["predicted_class"],
        dropna=False,
    )

    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)

    output_path = output_dir / "class_confusion_matrix.csv"
    confusion.to_csv(output_path)

    return confusion


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate high-confidence thresholds for StereoMapper predictions. "
            "Calibration is conditional on predicted_class."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV file with pair_id, true_class, predicted_class and score_S columns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where calibration outputs will be written.",
    )

    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Precision target for high-confidence threshold selection.",
    )

    parser.add_argument(
        "--min-true-positives",
        type=int,
        default=1,
        help=(
            "Minimum number of true-positive high-confidence calls required "
            "for a threshold to be eligible."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = validate_input(df)

    df.to_csv(output_dir / "normalised_input.csv", index=False)

    confusion = make_confusion_matrix(df, output_dir)
    print("\nClass confusion matrix:")
    print(confusion)

    summaries = []

    for target_class in RELATIONSHIP_CLASSES:
        summary = analyse_predicted_class(
            df=df,
            target_class=target_class,
            output_dir=output_dir,
            target_precision=args.target_precision,
            min_true_positives=args.min_true_positives,
        )

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No class summaries were generated.")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "high_confidence_threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nCalibration complete.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()