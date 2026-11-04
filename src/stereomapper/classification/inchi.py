"""InChI based classification fallback analyser."""

import re
import traceback
from typing import Any, Optional

from rdkit import Chem  # pylint: disable=no-member

from stereomapper.domain.chemistry import ChemistryOperations
from stereomapper.domain.models import SimilarityResult
from stereomapper.models.stereo_classification import StereoClassification
from stereomapper.scoring.features import FeatureBuilder
from stereomapper.utils.logging import setup_logging

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",  # Detailed logging to files
    quiet_console=True,  # Minimal console output during progress bar
    console_level="ERROR",  # Only errors to console
)


def _to_similarity_result(stereo_class: StereoClassification) -> SimilarityResult:
    return SimilarityResult.from_stereo_classification(stereo_class)


T_RE = re.compile(r"/t([^/]+)")
M_RE = re.compile(r"/m(\d+)")
S_RE = re.compile(r"/s(\d+)")


def parse_tms_with_unknowns(inchi: str) -> tuple[dict[str, str], str, str]:
    """
    t_dict: { center_index(str): sign in {'+','-','?'} }
    m_val: '0','1','2',''
    s_val: '1','2','3',''
    """
    t_dict: dict[str, str] = {}
    m_val = s_val = ""

    try:
        if m := T_RE.search(inchi):
            for part in re.split(r"[;,]", m.group(1)):
                part = part.strip()
                mm = re.match(r"^(\d+)([+\-?])$", part)  # accept '?'
                if mm:
                    idx, sign = mm.groups()
                    t_dict[idx] = sign

        if m := M_RE.search(inchi):
            m_val = m.group(1)
        if m := S_RE.search(inchi):
            s_val = m.group(1)
    except Exception as e:
        logger.warning("Error parsing InChI TMS layers: %s", e)

    return t_dict, m_val, s_val


def undefined_fraction_from_inchi(inchi: str) -> float:
    """
    Returns a number in [0,1]. If /s != '1', we conservatively return 1.0.
    If /t is absent and /s==1, returns 0.0 (no tetra stereo present).
    """
    try:
        t_dict, _m, s = parse_tms_with_unknowns(inchi)
        db_dict = _extract_double_bond_stereo(inchi)

        # Combine tetra and double bond stereo info
        all_stereo = {}
        all_stereo.update(t_dict)
        all_stereo.update(db_dict)

        total = len(all_stereo)
        if total == 0:
            # No stereo centers listed; treat as 0.0 undefined fraction
            return 0.0

        # FIX: Default s to "1" if empty or missing (InChI spec)
        if s == "" or s is None:
            s = "1"

        # Stereo status guard: if stereo is not fully specified, call it fully undefined.
        if s != "1":
            return 1.0

        undefined = sum(1 for v in all_stereo.values() if v == "?")
        return undefined / total
    except Exception as e:
        logger.warning("Failed to parse InChI for undefined fraction: %s", e)
        return 1.0  # Conservative fallback for malformed InChI


def _extract_double_bond_stereo(inchi: str) -> dict:
    """Extract double bond stereochemistry from /b layer."""
    b_dict = {}
    try:
        b_match = re.search(r"/b([^/]+)", inchi)
        if b_match:
            for part in re.split(r"[;,]", b_match.group(1)):
                part = part.strip()
                # Match patterns like "9-8-", "11-10+", etc.
                match = re.match(r"^(\d+-\d+)([+\-?])$", part)
                if match:
                    bond_desc, sign = match.groups()
                    # Use the bond description as key, or extract second number
                    bond_num = bond_desc.split("-")[1]  # Get the second number
                    b_dict[bond_num] = sign
    except Exception as e:
        logger.warning("Error parsing InChI /b layer: %s", e)
    return b_dict


def _has_unknown(t: dict[str, str]) -> bool:
    if t is None:
        return False
    return any(sign == "?" for sign in t.values())


def _has_unknown_stereo(t: dict[str, str], db: dict[str, str]) -> bool:
    """Check if there are unknown stereochemistry assignments
    in either tetrahedral or double bond."""
    tetra_unknown = _has_unknown(t)
    db_unknown = _has_unknown(db)
    return tetra_unknown or db_unknown


def _defined_subset(t: dict[str, str]) -> dict[str, str]:
    if t is None:
        return {}
    return {k: v for k, v in t.items() if v in {"+", "-"}}


def _all_signs_inverted_defined(
    t1: dict[str, str], db1: dict[str, str], t2: dict[str, str], db2: dict[str, str]
) -> bool:
    """Require same set of defined centers and full inversion on them."""
    if t1 is None:
        t1 = {}
    if db1 is None:
        db1 = {}
    if t2 is None:
        t2 = {}
    if db2 is None:
        db2 = {}

    # Check that double bond centers match between molecules
    if set(db1.keys()) != set(db2.keys()):
        return False

    # Check tetrahedral centers
    d1_tetra, d2_tetra = _defined_subset(t1), _defined_subset(t2)
    if set(d1_tetra.keys()) != set(d2_tetra.keys()):
        return False

    # Check double bond centers (only defined ones)
    d1_db = _defined_subset(db1)
    d2_db = _defined_subset(db2)
    if set(d1_db.keys()) != set(d2_db.keys()):
        return False

    inv = {"+": "-", "-": "+"}

    # All tetrahedral centers must be inverted
    tetra_inverted = all(d2_tetra[k] == inv[d1_tetra[k]] for k in d1_tetra)

    # All double bond centers must match (NOT be inverted) for enantiomers
    db_match = all(d2_db[k] == d1_db[k] for k in d1_db)

    return tetra_inverted and db_match


def _any_defined_signs_different(
    t1: dict[str, str], t2: dict[str, str], db1: dict[str, str] = None, db2: dict[str, str] = None
) -> bool:
    """Look only at centers that are defined in both."""
    if t1 is None:
        t1 = {}
    if t2 is None:
        t2 = {}
    if db1 is None:
        db1 = {}
    if db2 is None:
        db2 = {}

    # Check tetrahedral centers
    d1_tetra, d2_tetra = _defined_subset(t1), _defined_subset(t2)
    shared_tetra = set(d1_tetra) & set(d2_tetra)
    tetra_different = any(d1_tetra[k] != d2_tetra[k] for k in shared_tetra)

    # Check double bond centers
    d1_db, d2_db = _defined_subset(db1), _defined_subset(db2)
    shared_db = set(d1_db) & set(d2_db)
    db_different = any(d1_db[k] != d2_db[k] for k in shared_db)

    return tetra_different or db_different


def _resolution_direction_from_inchi(inchi_a: str, inchi_b: str) -> tuple[str | None, dict]:
    t_a, _m_a, _s_a = parse_tms_with_unknowns(inchi_a)
    t_b, _m_b, _s_b = parse_tms_with_unknowns(inchi_b)
    db_a = _extract_double_bond_stereo(inchi_a)
    db_b = _extract_double_bond_stereo(inchi_b)

    defined_a = sum(1 for v in t_a.values() if v in {"+", "-"}) + sum(
        1 for v in db_a.values() if v in {"+", "-"}
    )
    defined_b = sum(1 for v in t_b.values() if v in {"+", "-"}) + sum(
        1 for v in db_b.values() if v in {"+", "-"}
    )

    total_a = len(t_a) + len(db_a)
    total_b = len(t_b) + len(db_b)

    direction = None
    if defined_a > defined_b:
        direction = "A_to_B"
    elif defined_b > defined_a:
        direction = "B_to_A"

    return direction, {
        "defined_a": defined_a,
        "defined_b": defined_b,
        "total_a": total_a,
        "total_b": total_b,
        "resolution_direction": direction,
    }


class InChIFallbackAnalyser:
    """Fallback analyser using InChI layer comparison when alignment fails."""

    def __init__(self, confidence_penalty: float = 0.3):
        self.confidence_penalty = confidence_penalty
        self.builder = FeatureBuilder()

    def _extract_inchi_layers(
        self, mol: Chem.Mol
    ) -> Optional[dict[str, str]]:  # pylint: disable=no-member
        """Extract selected InChI layers from a mol object (via external InChI)."""
        try:
            inchi = Chem.MolToInchi(mol)  # returns std inchi from rdkit
            if not inchi or not inchi.startswith("InChI="):
                return None

            parts = inchi.split("/")
            layers: dict[str, str] = {"inchi": inchi, "formula": parts[1] if len(parts) > 1 else ""}

            for part in parts[2:]:
                if part.startswith("c"):
                    layers["connectivity"] = part[1:]  # strip leading 'c'
                elif part.startswith("h"):
                    layers["hydrogen"] = part[1:]
                elif part.startswith("q"):
                    layers["charge"] = part[1:]
                elif part.startswith("p"):
                    layers["proton"] = part[1:]
                elif part.startswith("t"):
                    layers["stereochemistry_sub1"] = part[1:]  # strip leading 't'
                elif part.startswith("m"):
                    layers["stereochemistry_sub2"] = part[1:]  # strip 'm'
                elif part.startswith("s"):
                    layers["stereochemistry_sub3"] = part[1:]  # strip 's'
                elif part.startswith("b"):
                    layers["double_bond"] = part[1:]
                elif part.startswith("i"):
                    layers["isotope"] = part[1:]
                elif part.startswith("f"):
                    layers["fixed_h"] = part[1:]
                elif part.startswith("r"):
                    layers["reconnected"] = part[1:]

            return layers
        except Exception as e:
            logger.warning("Failed to extract InChI layers: %s", e)
            return None

    def _get_inchikey_layers(self, mol: Chem.Mol) -> Optional[dict[str, str]]:
        """Get InChIKey blocks from a molfile path."""
        try:
            inchikey = Chem.MolToInchiKey(mol)  # returns std inchikey from rdkit
            if not inchikey or inchikey.count("-") != 2:
                return None
            first, second, third = inchikey.split("-")
            return {"first": first, "second": second, "third": third}
        except Exception as e:
            logger.warning(f"Failed to get InChIKey layers: {e}")
            return None

    @staticmethod
    def _calculate_stereo_stats(t_a: dict, t_b: dict, inchi_a: str, inchi_b: str) -> dict:
        """Calculate stereochemistry statistics from InChI data."""
        # pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-return-statements

        # Extract double bond stereochemistry from /b layer
        db_a = _extract_double_bond_stereo(inchi_a)
        db_b = _extract_double_bond_stereo(inchi_b)

        # Extract m-layer values to handle m0/m1 enantiomer case
        _, m_a, _ = parse_tms_with_unknowns(inchi_a)
        _, m_b, _ = parse_tms_with_unknowns(inchi_b)
        m_a = m_a or ""
        m_b = m_b or ""

        # Calculate tetrahedral center statistics
        common_tetra_centers = set(t_a.keys()) & set(t_b.keys())
        defined_common_tetra = {
            k for k in common_tetra_centers if t_a.get(k) in {"+", "-"} and t_b.get(k) in {"+", "-"}
        }

        # Handle m0/m1 enantiomer case: identical t-layers but different m-values
        if {m_a, m_b} == {"0", "1"} and str(sorted(t_a.items())) == str(sorted(t_b.items())):
            # In m0/m1 case, all tetrahedral centers are conceptually flipped
            tetra_matches = 0
            tetra_flips = len(defined_common_tetra)
        else:
            # Normal case: compare signs directly
            tetra_matches = sum(1 for k in defined_common_tetra if t_a[k] == t_b[k])
            tetra_flips = sum(1 for k in defined_common_tetra if t_a[k] != t_b[k])

        # Calculate double bond statistics
        common_db_centers = set(db_a.keys()) & set(db_b.keys())
        defined_common_db = {
            k for k in common_db_centers if db_a.get(k) in {"+", "-"} and db_b.get(k) in {"+", "-"}
        }

        db_matches = sum(1 for k in defined_common_db if db_a[k] == db_b[k])
        db_flips = sum(1 for k in defined_common_db if db_a[k] != db_b[k])

        # Count missing centers: any position where at least one molecule has '?'
        # This includes both '?' vs '?' and '?' vs defined cases
        missing_tetra = sum(
            1 for k in common_tetra_centers if t_a.get(k) == "?" or t_b.get(k) == "?"
        )
        missing_db = sum(
            1
            for k in set(db_a.keys()) & set(db_b.keys())
            if db_a.get(k) == "?" or db_b.get(k) == "?"
        )

        total_missing = missing_tetra + missing_db

        # Total stereogenic elements (all unique centers, not double-counted)
        all_tetra_centers = set(t_a.keys()) | set(t_b.keys())
        all_db_centers = set(db_a.keys()) | set(db_b.keys())

        total_stereogenic_elements = len(all_tetra_centers) + len(all_db_centers)

        stats = {
            "num_stereogenic_elements": total_stereogenic_elements,
            "num_tetra_matches": tetra_matches,
            "num_tetra_flips": tetra_flips,
            "num_db_matches": db_matches,
            "num_db_flips": db_flips,
            "num_missing": total_missing,
        }

        # Debug logging
        summary_logger.info("Stereo stats calculated: %s (m_a=%s, m_b=%s)", stats, m_a, m_b)
        summary_logger.info("t_a: %s, t_b: %s", t_a, t_b)
        summary_logger.info("db_a: %s, db_b: %s", db_a, db_b)

        return stats

    @staticmethod
    def _classify_stereo_from_inchi(inchi_a: str, inchi_b: str) -> str:
        """Classify stereochemical relationship from InChI strings."""
        # pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-return-statements
        try:
            t_a, m_a, s_a = parse_tms_with_unknowns(inchi_a)
            t_b, m_b, s_b = parse_tms_with_unknowns(inchi_b)

            db_a = _extract_double_bond_stereo(inchi_a)
            db_b = _extract_double_bond_stereo(inchi_b)

            frac_a = undefined_fraction_from_inchi(inchi_a)
            frac_b = undefined_fraction_from_inchi(inchi_b)

            # Ensure fractions are never None
            if frac_a is None:
                frac_a = 1.0
            if frac_b is None:
                frac_b = 1.0

            # Ensure m and s values are strings for comparison
            m_a = m_a or ""
            m_b = m_b or ""
            s_a = s_a or ""
            s_b = s_b or ""

            # Ensure t dictionaries are never None
            if t_a is None:
                t_a = {}
            if t_b is None:
                t_b = {}

            # Nothing to compare
            if not t_a and not t_b and not db_a and not db_b:
                return "STEREO_UNDEFINED"

            # Check for PLANAR VS STEREO cases
            t_a_all_tetra_undefined = not t_a or all(sign == "?" for sign in t_a.values())
            t_b_all_tetra_undefined = not t_b or all(sign == "?" for sign in t_b.values())
            t_a_has_tetra_defined = t_a and any(sign in {"+", "-"} for sign in t_a.values())
            db_a_has_defined = db_a and any(sign in {"+", "-"} for sign in db_a.values())
            t_a_has_defined = t_a_has_tetra_defined or db_a_has_defined
            t_b_has_tetra_defined = t_b and any(sign in {"+", "-"} for sign in t_b.values())
            db_b_has_defined = db_b and any(sign in {"+", "-"} for sign in db_b.values())
            t_b_has_defined = t_b_has_tetra_defined or db_b_has_defined

            # account for new method of assigning parent-child
            if (t_a_all_tetra_undefined and t_b_has_tetra_defined) or (
                t_a_has_tetra_defined and t_b_all_tetra_undefined
            ):
                return "PLANAR_VS_STEREO"
            if (t_a_has_defined and not t_b_has_defined) or (
                not t_a_has_defined and t_b_has_defined
            ):
                return "PLANAR_VS_STEREO"

            # Check for unknown stereochemistry in either tetrahedral or double bond
            has_unknown_a = _has_unknown_stereo(t_a, db_a)
            has_unknown_b = _has_unknown_stereo(t_b, db_b)

            # If either has unknown parity at any center OR s-layer ≠ 1, we can't make strong claims
            if has_unknown_a or has_unknown_b or s_a != "1" or s_b != "1":
                max_frac = max(frac_a, frac_b)
                if max_frac > 0.4:
                    return "PLANAR_VS_STEREO"
                return "PLANAR_VS_STEREO"

            if m_a == "2" or m_b == "2":
                return "RACEMIC_OR_MIXTURE"

            # Classic: same m, full inversion on all defined centers (including double bonds)
            if m_a == m_b and _all_signs_inverted_defined(t_a, db_a, t_b, db_b):
                return "ENANTIOMERS"

            # ChEBI-observed: identical /t and /b text but m differs 0 vs 1
            t_a_str = str(sorted(t_a.items())) if t_a else ""
            t_b_str = str(sorted(t_b.items())) if t_b else ""
            db_a_str = str(sorted(db_a.items())) if db_a else ""
            db_b_str = str(sorted(db_b.items())) if db_b else ""

            if t_a_str == t_b_str and db_a_str == db_b_str and {m_a, m_b} == {"0", "1"}:
                return "ENANTIOMERS"

            # Any defined differences (including double bonds) → diastereomers
            if _any_defined_signs_different(t_a, t_b, db_a, db_b) or (
                set(_defined_subset(t_a)) != set(_defined_subset(t_b))
                or set(_defined_subset(db_a)) != set(_defined_subset(db_b))
            ):
                return "DIASTEREOMERS"

            return "IDENTICAL_STEREO"

        except Exception as e:
            logger.error("Error in _classify_stereo_from_inchi: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())
            return "STEREO_UNDEFINED"

    def _compare_full_inchi_stereochemistry(self, molfile_a: str, molfile_b: str) -> Optional[str]:
        try:
            la = self._extract_inchi_layers(molfile_a)
            lb = self._extract_inchi_layers(molfile_b)
            if not la or not lb or "inchi" not in la or "inchi" not in lb:
                return None
            return self._classify_stereo_from_inchi(la["inchi"], lb["inchi"])
        except Exception as e:
            logger.warning("Failed to compare full InChI stereochemistry: %s", e)
            return "STEREO_UNDEFINED"

    def _build_fallback_confidence(
        self,
        classification: str,
        charge1,
        charge2,
        tanimoto2d,
        ik_first_eq,
        ik_stereo_layer_eq,
        ik_protonation_layer_eq,
        stereo_stats: dict = None,
    ) -> Any:
        """Build confidence features for fallback classification."""
        # Handle None charges
        if charge1 is None:
            charge1 = 0
        if charge2 is None:
            charge2 = 0

        if stereo_stats is None:
            stereo_stats = {
                "num_stereogenic_elements": 0,
                "num_tetra_matches": 0,
                "num_tetra_flips": 0,
                "num_db_matches": 0,
                "num_db_flips": 0,
                "num_missing": 0,
            }

        conf = self.builder.build_features_for_confidence(
            classification,
            rmsd=None,  # can't calculate RMSD in fallback
            charge1=charge1,
            charge2=charge2,
            num_stereogenic_elements=stereo_stats["num_stereogenic_elements"],
            num_tetra_matches=stereo_stats["num_tetra_matches"],
            num_tetra_flips=stereo_stats["num_tetra_flips"],
            num_db_matches=stereo_stats["num_db_matches"],
            num_db_flips=stereo_stats["num_db_flips"],
            num_missing=stereo_stats["num_missing"],
            tanimoto2d=tanimoto2d,
            ik_first_eq=ik_first_eq,
            ik_stereo_layer_eq=ik_stereo_layer_eq,
            ik_protonation_layer_eq=ik_protonation_layer_eq,
        )
        if hasattr(conf, "score") and conf.score is not None:
            conf.score = max(0.0, conf.score - self.confidence_penalty)
        return conf

    def analyze_relationship_fallback(
        self, mol_a, mol_b, charge_a, charge_b, cid_a: str = "", cid_b: str = ""
    ) -> Optional[SimilarityResult]:
        """
        Note: ensure mol_a and mol_b are *molfile paths* for ChemistryOperations.*_software calls,
        or wrap RDKit mols into temp molfiles before calling.
        """
        logger.info("Using InChI fallback analysis for pair (%s, %s)", cid_a, cid_b)

        # compute tanimoto2d
        try:
            tanimoto2d = ChemistryOperations.fingerprint_tanimoto(mol_a, mol_b)
        except Exception as e:
            logger.warning("Failed to compute Tanimoto2D: %s", e)
            tanimoto2d = None
        # Compute stereo stats
        try:
            # Extract InChI layers first
            layers_a_temp = self._extract_inchi_layers(mol_a)
            layers_b_temp = self._extract_inchi_layers(mol_b)

            if layers_a_temp and layers_b_temp:
                t_a, _, _ = parse_tms_with_unknowns(layers_a_temp["inchi"])
                t_b, _, _ = parse_tms_with_unknowns(layers_b_temp["inchi"])
                stereo_stats = self._calculate_stereo_stats(
                    t_a, t_b, layers_a_temp["inchi"], layers_b_temp["inchi"]
                )
            else:
                stereo_stats = {
                    "num_stereogenic_elements": 0,
                    "num_tetra_matches": 0,
                    "num_tetra_flips": 0,
                    "num_db_matches": 0,
                    "num_db_flips": 0,
                }
        except Exception as e:
            logger.warning("Failed to calculate stereo stats: %s", e)
            stereo_stats = {
                "num_stereogenic_elements": 0,
                "num_tetra_matches": 0,
                "num_tetra_flips": 0,
                "num_db_matches": 0,
                "num_db_flips": 0,
            }

        # Handle None charges early
        if charge_a is None:
            charge_a = 0
        if charge_b is None:
            charge_b = 0

        # EXPECTING paths; if you pass RDKit mols today, convert them before this point.
        layers_a = self._get_inchikey_layers(mol_a)
        layers_b = self._get_inchikey_layers(mol_b)
        if not layers_a or not layers_b:
            logger.warning("Failed to extract InChIKey layers for pair (%s, %s)", cid_a, cid_b)
            return _to_similarity_result(StereoClassification.no_classification())

        # First block: connectivity/skeleton
        if layers_a["first"] != layers_b["first"]:
            logger.info("Different molecular skeletons for pair (%s, %s)", cid_a, cid_b)
            return _to_similarity_result(StereoClassification.no_classification())

        second_diff = layers_a["second"] != layers_b["second"]  # stereo+isotopes
        third_diff = layers_a["third"] != layers_b["third"]  # protonation

        if third_diff and not second_diff:
            # Protomers
            logger.info("Protomers detected via fallback for pair (%s, %s)", cid_a, cid_b)
            conf = self._build_fallback_confidence(
                "PROTOMERS",
                charge_a,
                charge_b,
                tanimoto2d,
                ik_first_eq=True,
                ik_protonation_layer_eq=False,
                ik_stereo_layer_eq=True,
                stereo_stats=stereo_stats,
            )
            return _to_similarity_result(
                StereoClassification.protomers(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={
                        "fallback_method": True,
                        "confidence_penalty": self.confidence_penalty,
                    },
                )
            )

        elif second_diff and not third_diff:
            # Stereo differs; decide enantiomer vs diastereomer via full InChI
            if charge_a != charge_b:
                logger.info(
                    "Different charges with stereochemistry differences for pair (%s, %s)",
                    cid_a,
                    cid_b,
                )
                # Provide reason for UI/DB
                base = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=base.classification,
                    rmsd=base.rmsd,
                    confidence_score=base.confidence_score,
                    confidence_bin=base.confidence_bin,
                    confidence=base.confidence,
                    details={
                        **(base.details or {}),
                        "reason": "Stereo undefined via fallback; no classification",
                    },
                )

            stereo_class = self._compare_full_inchi_stereochemistry(mol_a, mol_b)
            if stereo_class == "ENANTIOMERS":
                logger.info("Enantiomers detected via fallback for pair (%s, %s)", cid_a, cid_b)
                conf = self._build_fallback_confidence(
                    "ENANTIOMERS",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=True,
                    ik_stereo_layer_eq=False,
                    stereo_stats=stereo_stats,
                )
                return _to_similarity_result(
                    StereoClassification.enantiomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=None,
                        penalties={
                            "fallback_method": True,
                            "confidence_penalty": self.confidence_penalty,
                        },
                    )
                )
            elif stereo_class == "DIASTEREOMERS":
                logger.info("Diastereomers detected via fallback for pair (%s, %s)", cid_a, cid_b)
                conf = self._build_fallback_confidence(
                    "DIASTEREOMERS",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_stereo_layer_eq=False,
                    ik_protonation_layer_eq=True,
                    stereo_stats=stereo_stats,
                )
                return _to_similarity_result(
                    StereoClassification.diastereomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=None,
                        penalties={
                            "fallback_method": True,
                            "confidence_penalty": self.confidence_penalty,
                        },
                    )
                )
            elif stereo_class == "PLANAR_VS_STEREO":
                logger.info(
                    "Planar vs Stereo detected via fallback for pair (%s, %s)", cid_a, cid_b
                )
                direction, detail_payload = (None, {})
                if layers_a_temp and layers_b_temp:
                    direction, detail_payload = _resolution_direction_from_inchi(
                        layers_a_temp.get("inchi", ""), layers_b_temp.get("inchi", "")
                    )
                conf = self._build_fallback_confidence(
                    "PLANAR_VS_STEREO",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=True,
                    ik_stereo_layer_eq=False,
                    stereo_stats=stereo_stats,
                )
                return _to_similarity_result(
                    StereoClassification.planar_vs_stereo(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=None,
                        details=detail_payload,
                        penalties={
                            "fallback_method": True,
                            "confidence_penalty": self.confidence_penalty,
                        },
                    )
                )
            else:
                logger.info(
                    "Stereo undefined or failed to classify via fallback for pair (%s, %s)",
                    cid_a,
                    cid_b,
                )
                base = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=base.classification,
                    rmsd=base.rmsd,
                    confidence_score=base.confidence_score,
                    confidence_bin=base.confidence_bin,
                    confidence=base.confidence,
                    details={
                        **(base.details or {}),
                        "reason": "Stereo and charge both differ; no classification",
                    },
                )

        elif second_diff and third_diff:
            if charge_a != charge_b:
                logger.info("Complex differences (stereo + charge) for pair (%s, %s)", cid_a, cid_b)
                base = _to_similarity_result(StereoClassification.no_classification())
                return SimilarityResult(
                    classification=base.classification,
                    rmsd=base.rmsd,
                    confidence_score=base.confidence_score,
                    confidence_bin=base.confidence_bin,
                    confidence=base.confidence,
                    details={
                        **(base.details or {}),
                        "reason": "Stereo and charge differ (complex); no classification",
                    },
                )

        else:
            # Same InChIKey across all 3 blocks
            if charge_a != charge_b:
                logger.info(
                    "Protomers (same InChIKey, diff charges) detected via fallback for pair (%s, %s)",
                    cid_a,
                    cid_b,
                )
                conf = self._build_fallback_confidence(
                    "PROTOMERS",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=False,
                    ik_stereo_layer_eq=True,
                    stereo_stats=stereo_stats,
                )
                return _to_similarity_result(
                    StereoClassification.protomers(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=None,
                        penalties={
                            "fallback_method": True,
                            "confidence_penalty": self.confidence_penalty,
                        },
                    )
                )
            else:
                logger.info(
                    "Identical structures detected via fallback for pair (%s, %s)", cid_a, cid_b
                )
                # FIX: Use _build_fallback_confidence instead of direct call
                conf = self._build_fallback_confidence(
                    "IDENTICAL",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=True,
                    ik_stereo_layer_eq=True,
                    stereo_stats=stereo_stats,
                )
                return _to_similarity_result(
                    StereoClassification.unresolved(
                        stereo_score=conf.score,
                        confidence=conf.as_dict(),
                        rmsd=None,
                        penalties={"fallback_method": True},
                        details={
                            "reason": "Possible pipeline error - should be no identical pairs here"
                        },
                    )
                )
