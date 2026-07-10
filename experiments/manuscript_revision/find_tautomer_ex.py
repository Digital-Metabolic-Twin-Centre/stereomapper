from pathlib import Path
import re
import pandas as pd
import sqlite3

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


# ----------------------------
# Settings
# ----------------------------

db_path = "/home/jackmcgoldrick/2026_05_20_full_chebi_vs_vmh_stereomapper/results/2026_06_02_vmh_chebi_mappings.sqlite"
molfile_root = Path("/home/jackmcgoldrick/Downloads/stereomapper_structures")

id_col = "member_curie"
identity_col = "identity_key_strict"


# ----------------------------
# Query database directly
# ----------------------------

query = """
select 
cm.cluster_id,
cm.member_curie,
c.identity_key_strict
from clusters c 
JOIN cluster_members cm 
on c.cluster_id = cm.cluster_id
LIMIT 5000
"""

with sqlite3.connect(db_path) as conn:
    df = pd.read_sql_query(query, conn)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print("Columns loaded from database:")
print(df.columns.tolist())

if id_col not in df.columns:
    raise ValueError(f"Missing column: {id_col}. Found columns: {df.columns.tolist()}")

if identity_col not in df.columns:
    raise ValueError(f"Missing column: {identity_col}. Found columns: {df.columns.tolist()}")

df[id_col] = df[id_col].astype(str).str.strip()
df[identity_col] = df[identity_col].astype(str).str.strip()

# ----------------------------
# ID normalisation
# ----------------------------

def normalise_member_curie(x):
    """
    Converts common variants to one key style.

    Examples:
    CHEBI:80858 -> CHEBI_80858
    CHEBI_80858 -> CHEBI_80858
    """
    x = str(x).strip()
    x = x.replace(":", "_")
    return x.upper()


def extract_curie_from_path(path):
    """
    Extracts IDs from filenames or full paths.

    Supports:
    CHEBI_80858
    CHEBI:80858
    CHEBI-80858
    HMDB0000001
    MNXM123
    """
    text = str(path)

    chebi = re.search(r"CHEBI[_:\-]?(\d+)", text, flags=re.IGNORECASE)
    if chebi:
        return f"CHEBI_{chebi.group(1)}"

    hmdb = re.search(r"HMDB\d+", text, flags=re.IGNORECASE)
    if hmdb:
        return hmdb.group(0).upper()

    mnx = re.search(r"MNXM\d+", text, flags=re.IGNORECASE)
    if mnx:
        return mnx.group(0).upper()

    return None


df["molfile_key"] = df[id_col].apply(normalise_member_curie)


# ----------------------------
# Recursively index molfiles
# ----------------------------

molfile_paths = (
    list(molfile_root.rglob("*.mol")) +
    list(molfile_root.rglob("*.sdf"))
)

path_index = pd.DataFrame({
    "molfile_path": [str(p) for p in molfile_paths]
})

path_index["molfile_key"] = path_index["molfile_path"].apply(extract_curie_from_path)

path_index = path_index.dropna(subset=["molfile_key"])

print(f"Found {len(path_index):,} indexed molfile paths")


# ----------------------------
# Check duplicate molfile matches
# ----------------------------

dups = (
    path_index
    .groupby("molfile_key")
    .size()
    .reset_index(name="n_paths")
    .query("n_paths > 1")
)

print(f"Duplicate molfile keys: {len(dups):,}")

if len(dups) > 0:
    print(
        path_index[
            path_index["molfile_key"].isin(dups["molfile_key"].head(10))
        ].sort_values("molfile_key").to_string(index=False)
    )

# Keep first path per key for now.
# If duplicates matter, inspect them before using this.
path_index_unique = path_index.drop_duplicates("molfile_key", keep="first")


# ----------------------------
# Join molfile paths to cluster table
# ----------------------------

df = df.merge(
    path_index_unique,
    on="molfile_key",
    how="left"
)

print(f"Rows in cluster table: {len(df):,}")
print(f"Rows linked to molfiles: {df['molfile_path'].notna().sum():,}")
print(f"Rows missing molfiles: {df['molfile_path'].isna().sum():,}")


# ----------------------------
# RDKit helpers
# ----------------------------

tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def mol_from_file(path):
    if pd.isna(path):
        return None

    path = Path(path)

    if not path.exists():
        return None

    if path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(
            str(path),
            sanitize=True,
            removeHs=False,
            strictParsing=False
        )

        for mol in supplier:
            if mol is not None:
                return mol

        return None

    return Chem.MolFromMolFile(
        str(path),
        sanitize=True,
        removeHs=False,
        strictParsing=False
    )


def raw_canonical_smiles(mol):
    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


def formula(mol):
    if mol is None:
        return None

    return rdMolDescriptors.CalcMolFormula(mol)


def formal_charge(mol):
    if mol is None:
        return None

    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def rdkit_canonical_tautomer(mol):
    if mol is None:
        return None

    taut = tautomer_enumerator.Canonicalize(mol)

    return Chem.MolToSmiles(
        taut,
        canonical=True,
        isomericSmiles=True
    )


# ----------------------------
# Convert molfiles to raw structure descriptors
# ----------------------------

mols = df["molfile_path"].apply(mol_from_file)

df["raw_canonical_smiles"] = mols.apply(raw_canonical_smiles)
df["formula"] = mols.apply(formula)
df["formal_charge"] = mols.apply(formal_charge)
df["rdkit_canonical_tautomer"] = mols.apply(rdkit_canonical_tautomer)

df.to_csv(
    "cluster_output_with_molfile_paths_and_raw_smiles.tsv",
    sep="\t",
    index=False
)

if __name__ == "__main__":
    print(df[[
        id_col,
        identity_col,
        "molfile_path",
        "raw_canonical_smiles",
        "formula",
        "formal_charge",
        "rdkit_canonical_tautomer"
    ]].head(10).to_string(index=False))