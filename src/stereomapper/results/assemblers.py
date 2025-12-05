# result/assemblers.py
from collections import defaultdict
from typing import Iterable, Mapping, Tuple, Optional
import json
import hashlib
from rdkit import Chem
from pathlib import Path
from stereomapper.domain.chemistry import ChemistryOperations, ChemistryValidator

def cluster_rows(
        rows: Iterable[Mapping]) -> Iterable[Tuple]:
    """
    Input rows must contain:
    inchikey_first, smiles (canonical),
    is_undef_sru, is_def_sru, sru_repeat_count, accession_curies(list)

    Yields tuples matching INSERT columns:
      (ik_first, identity_key_strict,
       is_undef_sru, is_def_sru, sru_repeat_count,
       member_count, members_json, members_hash)
    """
    by_smi = defaultdict(list)

    # ---------- build (ik_first, smiles) groups ----------
    for r in rows:
        ik_first = (r.get("inchikey_first") or "").strip()
        smi      = (r.get("smiles") or "").strip()  # identity_key_strict
        # Skip if either key is missing
        if not ik_first or not smi:
            continue

        is_undef = bool(r.get("is_undef_sru", False))
        is_def   = bool(r.get("is_def_sru", False))
        accessions = r.get("accession_curies", [])
        rep_cnt  = r.get("sru_repeat_count", None)
        # Normalise repeat count
        if rep_cnt in (None, "", "null", "None"):
            rep_cnt = None
        else:
            try:
                rep_cnt = int(rep_cnt)
            except (TypeError, ValueError):
                rep_cnt = None

        by_smi[(ik_first, smi)].append({
            "accession_curies": accessions if isinstance(accessions, list) else [],
            "is_def": is_def,
            "is_undef": is_undef,
            "rep_cnt": rep_cnt,
        })

    # ---------- emit clusters ----------
    for (ik_first, identity_key_strict), members_all in by_smi.items():
        def_by_k = defaultdict(list)
        undef = []
        none_sru = []

        for m in members_all:
            if m["is_def"] and m["rep_cnt"] is not None:
                def_by_k[m["rep_cnt"]].append(m)
            elif m["is_undef"]:
                # never merged with defined
                undef.append(m)
            else:
                # candidate to merge with defined (single k)
                none_sru.append(m)

        if def_by_k:
            ks = sorted(def_by_k.keys())
            if len(ks) == 1:
                # Merge no-SRU into the single defined-k cluster
                k = ks[0]
                merged = def_by_k[k] + none_sru

                member_ids = sorted({f for m in merged for f in m["accession_curies"]})
                member_count = len(member_ids)
                members_json = json.dumps(member_ids) if member_count else None
                members_hash = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()

                yield (
                    ik_first, identity_key_strict,
                    0, 1, k,  # undef=0, def=1
                    member_count, members_json, members_hash
                )
            else:
                # Multiple distinct defined counts: keep each separate; keep no-SRU separate
                for k in ks:
                    grp = def_by_k[k]
                    member_ids = sorted({f for m in grp for f in m["accession_curies"]})
                    member_count = len(member_ids)
                    members_json = json.dumps(member_ids) if member_count else None
                    members_hash = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()

                    yield (
                        ik_first, identity_key_strict,
                        0, 1, k,
                        member_count, members_json, members_hash
                    )

                if none_sru:
                    member_ids = sorted({f for m in none_sru for f in m["accession_curies"]})
                    member_count = len(member_ids)
                    members_json = json.dumps(member_ids) if member_count else None
                    members_hash = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()

                    yield (
                        ik_first, identity_key_strict,
                        0, 0, None,
                        member_count, members_json, members_hash
                    )
        else:
            # No defined SRU at all in this (ik_first, smiles) group
            if undef:
                member_ids = sorted({f for m in undef for f in m["accession_curies"]})
                member_count = len(member_ids)
                members_json = json.dumps(member_ids) if member_count else None
                members_hash = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()

                yield (
                    ik_first, identity_key_strict,
                    1, 0, None,
                    member_count, members_json, members_hash
                )

            if none_sru:
                member_ids = sorted({f for m in none_sru for f in m["accession_curies"]})
                member_count = len(member_ids)
                members_json = json.dumps(member_ids) if member_count else None
                members_hash = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()

                yield (
                    ik_first, identity_key_strict,
                    0, 0, None,
                    member_count, members_json, members_hash
                )

def build_mols_for_reps(reps, logger):
    """
    reps: Iterable[(cluster_id, smiles)]
    Returns:
      mol_by_cid: {int cid -> RDKit Mol}
      smi_by_cid: {int cid -> canonical SMILES}
      props_by_cid: {int cid -> (charge:int, is_radioactive:bool)}
      fallback_candidates: {int cid -> SMILES string}
    """
    mol_by_cid = {}
    smi_by_cid = {}
    props_by_cid = {}
    fallback_candidates = {}

    for cid_raw, smi in reps:
        try:
            cid = int(cid_raw)
        except Exception:
            cid = cid_raw  # keep as-is, but prefer ints consistently

        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            # Only add to fallback candidates, NOT to mol_by_cid
            fallback_candidates[cid] = smi
            logger.warning(f"Added cluster {cid} with SMILES {smi} to fallback candidates.")
            
            # Still need to store properties and SMILES for fallback
            smi_by_cid[cid] = smi
            # For properties, we can't compute charge without a valid molecule
            props_by_cid[cid] = (0, False)  # Default charge=0, not radioactive
            continue

        # Only add valid molecules to mol_by_cid
        mol_by_cid[cid] = mol
        smi_by_cid[cid] = smi

        is_radioactive = bool(ChemistryValidator.is_radioactive(mol))
        if is_radioactive:
            logger.warning(f"[radioactive] cluster {cid} with SMILES {smi} contains radioactive atoms.")

        # Compute & store props for valid molecules
        formal_charge = ChemistryOperations.get_formal_charge(mol)
        props_by_cid[cid] = (formal_charge, is_radioactive)

    logger.info(f"Built {len(mol_by_cid)} valid molecules and {len(fallback_candidates)} fallback candidates")
    return mol_by_cid, smi_by_cid, props_by_cid, fallback_candidates

def _coerce_scalar(x):
    # Accept None
    if x is None:
        return None
    # Accept numpy scalars, floats, ints
    try:
        val = float(x)
    except Exception:
        return None
    # Normalize NaN to None for DB NULL
    if val != val:  # NaN check
        return None
    return val

def _normalise_classification(res):
    """
    Return a plain dict with at least:
      - classification: str
      - score: float | None           # now taken from confidence.score
      - score_details: JSON-safe dict
    Works if res is:
      - StereoClassification
      - dict-like
      - a (res, confidence) tuple (old bug path) -> uses the first item
    """
    if res is None:
        return None

    # If some paths still return tuples, unwrap the first element.
    if isinstance(res, tuple) and res:
        res = res[0]

    # Prefer a dict view
    if hasattr(res, "to_dict"):
        d = res.to_dict()
    elif isinstance(res, dict):
        d = dict(res)
    else:
        # generic attribute fallback
        d = {
            "classification": getattr(res, "classification", None),
            "rmsd": getattr(res, "rmsd", None),
            "stereo_penalties": getattr(res, "stereo_penalties", None),
            "confidence_score": getattr(res, "confidence_score", None),
            "confidence": getattr(res, "confidence", None),
            "stereo_score": getattr(res, "stereo_score", None),
            "details": getattr(res, "details", None),
        }

    cls = d.get("classification")

    # --- NEW: pull the confidence score ---
    score = d.get("confidence_score", None)

    if score is None:
        conf = d.get("confidence")
        if isinstance(conf, dict):
            score = conf.get("score", None)
        else:
            # attribute-like confidence object
            score = getattr(conf, "score", None) if conf is not None else None

    # Fallback (for older objects) – only if nothing found yet
    if score is None:
        score = d.get("stereo_score", None)

    # Make score JSON-safe scalar or None
    score = _coerce_scalar(score)

    # Keep details compact and JSON-serializable
    score_details = {
        "confidence": d.get("confidence", None),
        "rmsd": d.get("rmsd", None),
        "stereo_penalties": d.get("stereo_penalties", None),
        "details": d.get("details", None),
    }

    # Try to surface an explanatory message if available
    extra_info = None
    # common fields at top level
    if isinstance(d, dict):
        for key in ("reason", "message", "note"):
            val = d.get(key)
            if val:
                extra_info = str(val)
                break
        # also look inside details dict if present
        if not extra_info:
            det = d.get("details") or {}
            if isinstance(det, dict):
                for key in ("reason", "message", "note"):
                    val = det.get(key)
                    if val:
                        extra_info = str(val)
                        break

    return {
        "classification": cls,
        "score": score,                  # <— this is now conf.score
        "score_details": score_details,
        "extra_info": extra_info,
    }

def _details_from_res(res_dict: dict) -> dict:
    conf = res_dict.get("confidence") or {}
    if not isinstance(conf, dict):
        conf = {"bin": getattr(res_dict.get("confidence"), "bin", None)}
    return {"confidence_bin": conf.get("bin")}


# set the prefix rules 
PREFIX_RULES = {
    '/vmh_structures/': 'vmhM:',
    '/kegg_structures/': 'kegg:',
    '/chebi_structures/': 'chebi:',
    '/hmdb_structures/': 'hmdb:',
    '/lipidmaps_structures/': 'lipidmapsM:',
    '/swisslipids_structures/': 'slm:',
    '/modelseed_structures/': 'modelseed:'
}

def detect_prefix(molfile_path: str) -> str:
    p = str(Path(molfile_path).expanduser().resolve())
    for key, prefix in PREFIX_RULES.items():
        if key in p:
            return prefix
    return 'unknown:'

def prefixed_identifier(molfile_path: str, basename: str) -> str:
    prefix = detect_prefix(molfile_path)

    cleaned = basename
    if ":" in cleaned:
        ns, local = cleaned.split(":", 1)
        if f"{ns.lower()}:" == prefix.lower() or ns:
            cleaned = local

    return f"{prefix}{cleaned}"


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# in cache_db.py
def make_molecule_key(*, std_version: str,
                      inchikey_full: Optional[str] = None,
                      isomeric_smiles: Optional[str] = None,
                      formal_charge: Optional[int] = None) -> str:
    base = inchikey_full or isomeric_smiles
    if not base:
        raise ValueError("make_molecule_key requires inchikey_full or isomeric_smiles")
    q = "NA" if formal_charge is None else formal_charge
    return hashlib.blake2b(f"{std_version}|{base}|q={q}".encode(), digest_size=16).hexdigest()    
