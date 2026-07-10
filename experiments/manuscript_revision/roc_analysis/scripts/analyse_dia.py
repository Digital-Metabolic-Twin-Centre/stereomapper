import pandas as pd

df = pd.read_csv("/home/jackmcgoldrick/repos/stereomapper_original_release/experiments/manuscript_revision/roc_analysis/results/all_benchmarking_data_for_threshold_analysis.csv")

def norm(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    aliases = {
        "enantiomers": "enantiomer",
        "diastereomers": "diastereomer",
        "stereo_resolution_pair": "stereo_resolution",
        "stereo_resolution_pairs": "stereo_resolution",
        "stereoresolution": "stereo_resolution",
        "protomers": "protomer",
    }
    return aliases.get(x, x)

df["true_norm"] = df["true_class"].apply(norm)
df["pred_norm"] = df["predicted_class"].apply(norm)
df["score_S"] = pd.to_numeric(df["score_S"], errors="coerce")

dia = df[
    (df["true_norm"] == "diastereomer")
    & df["pred_norm"].notna()
    & df["score_S"].notna()
].copy()

print(pd.crosstab(dia["true_norm"], dia["pred_norm"]))

wrong_dia = dia[dia["pred_norm"] != "diastereomer"].copy()

print(wrong_dia["pred_norm"].value_counts())
print(wrong_dia.groupby("pred_norm")["score_S"].describe())

wrong_dia.to_csv("true_diastereomers_misclassified_with_scores.csv", index=False)