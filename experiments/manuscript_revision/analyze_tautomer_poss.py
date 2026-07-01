import pandas as pd

df = pd.read_csv("/home/jackmcgoldrick/repos/stereomapper_original_release/cluster_output_with_molfile_paths_and_raw_smiles.tsv", sep="\t")

identity_col = "identity_key_strict"
id_col = "member_curie"

group_summary = (
    df.groupby(identity_col)
    .agg(
        n_members=(id_col, "nunique"),
        n_raw_forms=("raw_canonical_smiles", "nunique"),
        formula=("formula", "first"),
        formal_charge=("formal_charge", "first"),
        members=(id_col, lambda x: ", ".join(sorted(set(map(str, x)))[:6])),
    )
    .reset_index()
    .sort_values(["n_raw_forms", "n_members"], ascending=[False, True])
)

print(group_summary.head(30).to_string(index=False))

if __name__ == "__main__":
    group_summary.to_csv("tautomer_groups_summary.tsv", sep="\t", index=False)