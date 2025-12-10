import os
import threading
import json
import io
import logging
import sqlite3
from typing import List, Dict, Set, Tuple, Optional, Any
from stereomapper.data import results_repo
from stereomapper.results import assemblers
from stereomapper.classification import RelationshipAnalyser, InChIFallbackAnalyser
from stereomapper.utils.suppress import setup_clean_logging
from stereomapper.utils.logging import setup_logging

setup_clean_logging()

logger, summary_logger = setup_logging(
    console=True,
    level="WARNING",
)

def _preload_cluster_members(results_db_path: str, cluster_ids: List[str]) -> Dict:
    """Preload cluster member information for all clusters."""
    if not cluster_ids:
        return {}
    
    placeholders = ",".join(["?"] * len(cluster_ids))
    sql = f"""
        SELECT cluster_id, members_json, member_count
        FROM clusters
        WHERE cluster_id IN ({placeholders})
    """
    
    cluster_members = {}
    with sqlite3.connect(results_db_path) as conn:
        for cluster_id, members_json, member_count in conn.execute(sql, cluster_ids):
            cluster_members[cluster_id] = {
                'members_json': members_json,
                'member_count': member_count
            }
    
    return cluster_members

def _load_cluster_data(results_db_path: str, inchikey_first: str, version_tag: str) -> Optional[Tuple[Dict, List, Set]]:
    """Load and validate cluster data."""
    reps = results_repo.fetch_cluster_reps_for_inchikey(results_db_path, inchikey_first)
        
    if len(reps) < 2:
        return None

    mol_by_cid, smi_by_cid, props_by_cid, fallback_candidates = assemblers.build_mols_for_reps(reps, logger)
    
    # COMPLETE THE FUNCTION:
    all_cids = set(mol_by_cid.keys()).union(set(fallback_candidates.keys()))
    if len(all_cids) < 2:
        logger.warning(f"Only {len(all_cids)} clusters available for comparison")
        return None

    cluster_ids = sorted(all_cids)
    processed_pairs = results_repo.preload_processed_pairs(results_db_path, version_tag, cluster_ids)
    processed_pairs = {(min(a, b), max(a, b)) for (a, b) in processed_pairs}

    logger.info(f"Loaded {len(mol_by_cid)} RDKit molecules and {len(fallback_candidates)} fallback candidates")
    
    return (mol_by_cid, props_by_cid, fallback_candidates, cluster_ids), cluster_ids, processed_pairs

def _is_valid_primary_result(result: Any) -> bool:
    """Check if primary analysis result is valid and meaningful."""
    if result is None:
        #print(f"DEBUG _is_valid_primary_result: FAILED - result is None")
        return False
    
    # Check if this is a SimilarityResult object
    if not hasattr(result, 'classification'):
        #print(f"DEBUG _is_valid_primary_result: FAILED - no classification attribute")
        return False
    
    classification = getattr(result, 'classification', None)
    #print(f"DEBUG _is_valid_primary_result: classification = '{classification}'")
    
    # Instead of string matching, check for explicit failure cases
    failure_cases = [
        None,
        "RMSD_ERROR", 
        "RMSD ERROR",
        "FAILED",
        "ERROR", 
        "ALIGNMENT_FAILED"
    ]
    
    if classification in failure_cases:
        #print(f"DEBUG _is_valid_primary_result: FAILED - explicit failure case: {classification}")
        return False
    
    # Check if it's a "No classification" case (the only enum value that should trigger fallback)
    if classification == "No classification":  # StereoClass.NO_CLASSIFICATION.value
        #print(f"DEBUG _is_valid_primary_result: FAILED - no classification available")
        return False
    
    # If we have any other classification string, consider it valid
    # This includes all the valid StereoClass enum values:
    # "Enantiomers", "Diastereomers", "Identical structures", etc.
    if isinstance(classification, str) and classification.strip():
        #print(f"DEBUG _is_valid_primary_result: PASSED - valid classification: {classification}")
        return True
    
    #print(f"DEBUG _is_valid_primary_result: FAILED - empty or invalid classification")
    return False

def _analyze_pair(analyser: RelationshipAnalyser, fallback_analyser: InChIFallbackAnalyser,
                 mol_by_cid: Dict, props_by_cid: Dict, fallback_candidates: Dict, 
                 sru_by_cid: Dict, cid_a: str, cid_b: str) -> Optional[Any]:
    """Analyze relationship between two clusters with fallback support."""
    
    # Get molecules and properties
    mol_a = mol_by_cid.get(cid_a)
    mol_b = mol_by_cid.get(cid_b)
    
    # DEBUG: Check why primary analysis is skipped
    summary_logger.info(f"DEBUG _analyze_pair: Starting analysis for clusters {cid_a} vs {cid_b}")
    summary_logger.info(f"DEBUG _analyze_pair: mol_a exists = {mol_a is not None}, mol_b exists = {mol_b is not None}")
    
    if cid_a not in props_by_cid:
        summary_logger.info(f"DEBUG _analyze_pair: MISSING props for {cid_a}")
    if cid_b not in props_by_cid:
        summary_logger.info(f"DEBUG _analyze_pair: MISSING props for {cid_b}")
    
    if mol_a is None:
        summary_logger.info(f"DEBUG _analyze_pair: mol_a is None for cluster {cid_a}")
    if mol_b is None:
        summary_logger.info(f"DEBUG _analyze_pair: mol_b is None for cluster {cid_b}")
    
    # If either molecule is missing, skip to fallback immediately
    if mol_a is None or mol_b is None:
        summary_logger.info(f"DEBUG _analyze_pair: Skipping primary analysis - missing molecules")
        # Go directly to fallback section...
    
    charge_a, is_radio_a = props_by_cid[cid_a]
    charge_b, is_radio_b = props_by_cid[cid_b]

    # Get SRU data
    sru_a = sru_by_cid.get(cid_a, {"has_sru": False, "is_undef": False, "rep": None})
    sru_b = sru_by_cid.get(cid_b, {"has_sru": False, "is_undef": False, "rep": None})
    has_sru_a, is_undef_sru_a, rep_a = sru_a["has_sru"], sru_a["is_undef"], sru_a["rep"]
    has_sru_b, is_undef_sru_b, rep_b = sru_b["has_sru"], sru_b["is_undef"], sru_b["rep"]

    # Try primary analysis first if both molecules are available
    primary_result = None
    alignment_failed = False
    
    if mol_a is not None and mol_b is not None:
        summary_logger.info(f"DEBUG _analyze_pair: Attempting primary analysis for clusters {cid_a} vs {cid_b}")
        try:            
            primary_result = analyser.calc_relationship(
                mol_a, mol_b, charge_a, charge_b,
                cid1=cid_a, cid2=cid_b,
                isRadio1=is_radio_a, isRadio2=is_radio_b,
                has_sru1=has_sru_a, has_sru2=has_sru_b,
                is_undef_sru1=is_undef_sru_a, is_undef_sru2=is_undef_sru_b,
                sru_repeat_count1=rep_a, sru_repeat_count2=rep_b,
            )
            
            summary_logger.info(f"DEBUG _analyze_pair: Primary analysis completed for clusters {cid_a} vs {cid_b}")
            
            out = _is_valid_primary_result(primary_result)
            summary_logger.info(f"DEBUG _analyze_pair: primary_result valid = {out} for clusters {cid_a} vs {cid_b}")
            
            # Check if primary result is valid and alignment didn't fail
            if primary_result is not None and _is_valid_primary_result(primary_result):
                summary_logger.info(f"[compare] primary analysis succeeded for clusters {cid_a} vs {cid_b}")
                return primary_result
            else:
                summary_logger.info(f"[compare] primary analysis returned invalid result for clusters {cid_a} vs {cid_b}")                
        except Exception as e:
            summary_logger.info(f"DEBUG _analyze_pair: Primary analysis EXCEPTION for clusters {cid_a} vs {cid_b}: {e}")
            logger.warning("[compare] primary analysis exception for clusters %s vs %s: %s", cid_a, cid_b, e)
    else:
        summary_logger.info(f"DEBUG _analyze_pair: Skipping primary analysis - mol_a={mol_a is not None}, mol_b={mol_b is not None}")

    # PRIMARY FAILED OR ALIGNMENT FAILED - Use fallback
    summary_logger.info(f"DEBUG _analyze_pair: Using fallback for clusters {cid_a} vs {cid_b}")
    # ... rest of fallback code

    # PRIMARY FAILED OR ALIGNMENT FAILED - Use fallback
    if mol_a is not None and mol_b is not None:
        summary_logger.info("[compare] using InChI fallback analysis for clusters %s vs %s", cid_a, cid_b)
        try:
            fallback_result = fallback_analyser.analyze_relationship_fallback(
                mol_a, mol_b, charge_a, charge_b, cid_a, cid_b
            )
            
            if fallback_result is not None:
                if hasattr(fallback_result, 'penalties'):
                    fallback_result.penalties = fallback_result.penalties or {}
                    fallback_result.penalties['used_fallback_method'] = True
                summary_logger.info("[compare] fallback analysis succeeded for clusters %s vs %s", cid_a, cid_b)
                return fallback_result
            else:
                logger.warning("[compare] fallback analysis returned None for clusters %s vs %s", cid_a, cid_b)
                
        except Exception as fallback_error:
            logger.error("[compare] fallback analysis failed for clusters %s vs %s: %s", 
                        cid_a, cid_b, fallback_error)

    logger.warning("[compare] no valid analysis method available for clusters %s vs %s", cid_a, cid_b)
    return None


def _process_result(res: Any, cid_a: str, cid_b: str, version_tag: str,
                    cluster_a_members: str, cluster_b_members: str,
                    cluster_a_size: int, cluster_b_size: int) -> Optional[Tuple]:
    """Process analysis result into database format."""
    if res is None:
        pid = os.getpid()
        logger.warning(
            "[compare] no result from calc_stereo_similarity for pair (%s,%s); skipping (pid=%s tid=%s)",
            cid_a, cid_b, pid, threading.get_ident()
        )
        return None

    norm = assemblers._normalise_classification(res)
    cls = None if norm is None else norm.get("classification")

    if not cls:
        reason = getattr(res, "reason", None)
        logger.warning(
            "[pair] missing classification for pair (%s,%s); skipping. res=%s%s",
            cid_a, cid_b, res, (f" reason={reason}" if reason else "")
        )
        return None

    score = None
    if norm is not None and "score" in norm:
        score = assemblers._coerce_scalar(norm["score"])

    extra_info = None
    if norm is not None:
        extra_info = norm.get("extra_info")
        # Fallback to attribute if present
        if not extra_info:
            extra_info = getattr(res, "reason", None)

    try:
        details_source = res.to_dict() if hasattr(res, "to_dict") else res
        details = assemblers._details_from_res(details_source)
        details_json = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        logger.exception("[pair] could not get score_details for (%s,%s); skipping", cid_a, cid_b)
        return None

    return (cid_a, cid_b, cluster_a_members, cluster_b_members, 
            cluster_a_size, cluster_b_size, cls, score, details_json, extra_info, version_tag)

def compare_cluster_relationships(*, results_db_path: str, inchikey_first: str, version_tag: str, logger):
    """Compare cluster relationships and store results."""
    # Load data with fallback support
    data_result = _load_cluster_data(results_db_path, inchikey_first, version_tag)
    if data_result is None:
        return

    (mol_by_cid, props_by_cid, fallback_candidates, cluster_ids), cluster_ids, processed_pairs = data_result
    sru_by_cid = results_repo.preload_cluster_sru(results_db_path, cluster_ids)
    
    # Preload cluster member information
    cluster_members_by_id = _preload_cluster_members(results_db_path, cluster_ids)
    
    # Initialize analyzers
    analyser = RelationshipAnalyser()
    fallback_analyser = InChIFallbackAnalyser()
    to_insert = []

    for i, cid_a in enumerate(cluster_ids):
        for cid_b in cluster_ids[i+1:]:
            key = (min(cid_a, cid_b), max(cid_a, cid_b))
            if key in processed_pairs:
                continue

            res = _analyze_pair(analyser, fallback_analyser, mol_by_cid, props_by_cid, fallback_candidates, sru_by_cid, cid_a, cid_b)
            
            # Get cluster member info
            cluster_a_info = cluster_members_by_id.get(cid_a, {})
            cluster_b_info = cluster_members_by_id.get(cid_b, {})
            
            processed_result = _process_result(
                res, cid_a, cid_b, version_tag,
                cluster_a_info.get('members_json'),
                cluster_b_info.get('members_json'),
                cluster_a_info.get('member_count', 0),
                cluster_b_info.get('member_count', 0)
            )
            
            if processed_result is not None:
                to_insert.append(processed_result)

    # Save results
    if to_insert:
        results_repo.batch_insert_cluster_pairs(results_db_path, to_insert)
