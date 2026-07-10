import re
import time
import requests
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from urllib.parse import quote
from io import StringIO

BASE_URL = "https://webservice.bridgedb.org"
ORG = "Human"
SOURCE = "Ce"
# helper function to normalise hmdb ids to avoid duplicates due to different formatting (e.g. HMDB00001 vs HMDB00001)
def normalise_hmdb_id(x):
    x = str(x).strip()

    match = re.match(r"^HMDB0*(\d+)$", x)
    if not match:
        return x

    return f"HMDB{int(match.group(1)):07d}"

# another helper just in case for chebi
def normalise_chebi_id(x):
    x = str(x).strip()

    if x.upper().startswith("CHEBI:"):
        return "CHEBI:" + x.split(":", 1)[1]

    if x.upper().startswith("CHEBI_"):
        return "CHEBI:" + x.split("_", 1)[1]

    return "CHEBI:" + x

# single use for fetching mapping
def get_hmdb_from_chebi(chebi_id, session=None, timeout=30):
    session = session or requests.Session()

    chebi_id = normalise_chebi_id(chebi_id)
   # print(f"Fetching HMDB mapping for {chebi_id} from BridgeDb...")
    encoded_id = quote(chebi_id, safe="")
    url = f"{BASE_URL}/{ORG}/xrefs/{SOURCE}/{encoded_id}"

    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    bridgedb_df = pd.read_csv(
        StringIO(response.text),
        sep="\t",
        header=None,
        names=["mapped_id", "database"],
    )

    hmdb_mappings = (
        bridgedb_df[bridgedb_df["database"].eq("HMDB")]
        .rename(columns={"mapped_id": "hmdb_id"})
        [["hmdb_id"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if hmdb_mappings.empty:
        return pd.DataFrame(columns=["chebi_id", "hmdb_id"])

    hmdb_mappings["chebi_id"] = chebi_id
    hmdb_mappings["hmdb_id"] = hmdb_mappings["hmdb_id"].apply(normalise_hmdb_id)

    hmdb_mappings = (
        hmdb_mappings[["chebi_id", "hmdb_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return hmdb_mappings

# batch mapper
def get_hmdb_from_chebi_batch(
    chebi_ids,
    sleep_s=0.1,
    timeout=30,
    progress_every=100,
):
    chebi_ids = [normalise_chebi_id(x) for x in chebi_ids if pd.notna(x)]
    chebi_ids = sorted(set(chebi_ids))

    all_results = []
    failed = []
    no_mapping = []

    with requests.Session() as session:
        for i, chebi_id in enumerate(chebi_ids, start=1):
            try:
                result = get_hmdb_from_chebi(
                    chebi_id=chebi_id,
                    session=session,
                    timeout=timeout,
                )

                if result.empty:
                    no_mapping.append(chebi_id)
                else:
                    all_results.append(result)

            except requests.RequestException as e:
                failed.append(
                    {
                        "chebi_id": chebi_id,
                        "error": str(e),
                    }
                )

            if progress_every and i % progress_every == 0:
                print(f"Processed {i:,}/{len(chebi_ids):,}")

            time.sleep(sleep_s)

    mappings_df = (
        pd.concat(all_results, ignore_index=True).drop_duplicates()
        if all_results
        else pd.DataFrame(columns=["chebi_id", "hmdb_id"])
    )

    failed_df = pd.DataFrame(failed)
    no_mapping_df = pd.DataFrame({"chebi_id": no_mapping})

    return mappings_df, no_mapping_df, failed_df