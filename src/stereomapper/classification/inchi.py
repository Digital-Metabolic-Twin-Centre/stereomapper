"""InChI based classification fallback analyser."""
from typing import Optional, Dict, Any
import re
import traceback
from rdkit import Chem  # pylint: disable=no-member
from stereomapper.domain.chemistry import ChemistryOperations
from stereomapper.domain.models import SimilarityResult
from stereomapper.models.stereo_classification import StereoClassification
from stereomapper.scoring.features import FeatureBuilder
from stereomapper.utils.logging import setup_logging

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",
    quiet_console=True,
    console_level="ERROR"
)


def _to_similarity_result(stereo_class: StereoClassification) -> SimilarityResult:
    return SimilarityResult.from_stereo_classification(stereo_class)


T_RE = re.compile(r"/t([^/]+)")
M_RE = re.compile(r"/m(\d+)")
S_RE = re.compile(r"/s(\d+)")


def parse_tms_with_unknowns(inchi: str) -> tuple[dict[str, str], str, str]:
    """
    t_dict: { centre_index(str): sign in {'+','-','?'} }
    m_val: '0','1','2',''
    s_val: '1','2','3',''
    """
    t_dict: dict[str, str] = {}
    m_val = ""
    s_val = ""

    try:
        match_t = T_RE.search(inchi)
        if match_t:
            for part in re.split(r"[;,]", match_t.group(1)):
                part = part.strip()
                mm = re.match(r"^(\d+)([+\-?])$", part)
                if mm:
                    idx, sign = mm.groups()
                    t_dict[idx] = sign

        match_m = M_RE.search(inchi)
        if match_m:
            m_val = match_m.group(1)

        match_s = S_RE.search(inchi)
        if match_s:
            s_val = match_s.group(1)

    except Exception as exc:
        logger.warning("Error parsing InChI TMS layers: %s", exc)

    return t_dict, m_val, s_val


def undefined_fraction_from_inchi(inchi: str) -> float:
    """
    Returns a number in [0,1]. If /s is not '1', we conservatively return 1.0.
    If /t is absent and /s is '1', returns 0.0.
    """
    try:
        t_dict, _m, s_val = parse_tms_with_unknowns(inchi)
        db_dict = _extract_double_bond_stereo(inchi)

        all_stereo = {}
        all_stereo.update(t_dict)
        all_stereo.update(db_dict)

        total = len(all_stereo)
        if total == 0:
            return 0.0

        if not s_val:
            s_val = "1"

        if s_val != "1":
            return 1.0

        undefined = sum(1 for v in all_stereo.values() if v == "?")
        return undefined / total

    except Exception as exc:
        logger.warning("Failed to parse InChI for undefined fraction: %s", exc)
        return 1.0


def _extract_double_bond_stereo(inchi: str) -> dict:
    """Extract double bond stereochemistry from /b layer."""
    b_dict = {}
    try:
        match_b = re.search(r"/b([^/]+)", inchi)
        if match_b:
            for part in re.split(r"[;,]", match_b.group(1)):
                part = part.strip()
                match = re.match(r"^(\d+-\d+)([+\-?])$", part)
                if match:
                    bond_desc, sign = match.groups()
                    bond_num = bond_desc.split("-")[1]
                    b_dict[bond_num] = sign
    except Exception as exc:
        logger.warning("Error parsing InChI /b layer: %s", exc)
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


def _all_signs_inverted_defined(t1: dict[str, str], db1: dict[str, str],
                                t2: dict[str, str], db2: dict[str, str]) -> bool:
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

    d1_tetra = _defined_subset(t1)
    d2_tetra = _defined_subset(t2)
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


def _any_defined_signs_different(t1: dict[str, str], t2: dict[str, str],
                                 db1: dict[str, str] = None,
                                 db2: dict[str, str] = None) -> bool:
    if t1 is None:
        t1 = {}
    if t2 is None:
        t2 = {}
    if db1 is None:
        db1 = {}
    if db2 is None:
        db2 = {}

    d1_tetra = _defined_subset(t1)
    d2_tetra = _defined_subset(t2)
    shared_tetra = set(d1_tetra) & set(d2_tetra)
    tetra_different = any(d1_tetra[k] != d2_tetra[k] for k in shared_tetra)

    d1_db = _defined_subset(db1)
    d2_db = _defined_subset(db2)
    shared_db = set(d1_db) & set(d2_db)
    db_different = any(d1_db[k] != d2_db[k] for k in shared_db)

    return tetra_different or db_different


class InChIFallbackAnalyser:
    """Fallback analyser using InChI layer comparison when alignment fails."""

    def __init__(self, confidence_penalty: float = 0.3):
        self.confidence_penalty = confidence_penalty
        self.builder = FeatureBuilder()

    def _extract_inchi_layers(self, mol: Chem.Mol) -> Optional[Dict[str, str]]:  # noqa
        """Extract selected InChI layers from an RDKit mol."""
        try:
            inchi = Chem.MolToInchi(mol)
            if not inchi or not inchi.startswith("InChI="):
                return None

            parts = inchi.split("/")
            layers: Dict[str, str] = {
                "inchi": inchi,
                "formula": parts[1] if len(parts) > 1 else ""
            }

            for part in parts[2:]:
                prefix = part[:1]
                if prefix == "c":
                    layers["connectivity"] = part[1:]
                elif prefix == "h":
                    layers["hydrogen"] = part[1:]
                elif prefix == "q":
                    layers["charge"] = part[1:]
                elif prefix == "p":
                    layers["proton"] = part[1:]
                elif prefix == "t":
                    layers["stereochemistry_sub1"] = part[1:]
                elif prefix == "m":
                    layers["stereochemistry_sub2"] = part[1:]
                elif prefix == "s":
                    layers["stereochemistry_sub3"] = part[1:]
                elif prefix == "b":
                    layers["double_bond"] = part[1:]
                elif prefix == "i":
                    layers["isotope"] = part[1:]
                elif prefix == "f":
                    layers["fixed_h"] = part[1:]
                elif prefix == "r":
                    layers["reconnected"] = part[1:]

            return layers

        except Exception as exc:
            logger.warning("Failed to extract InChI layers: %s", exc)
            return None

    def _get_inchikey_layers(self, mol: Chem.Mol) -> Optional[Dict[str, str]]:  # noqa
        """Get InChIKey blocks from an RDKit mol."""
        try:
            inchikey = Chem.MolToInchiKey(mol)
            if not inchikey or inchikey.count("-") != 2:
                return None
            first, second, third = inchikey.split("-")
            return {"first": first, "second": second, "third": third}

        except Exception as exc:
            logger.warning("Failed to get InChIKey layers: %s", exc)
            return None

    @staticmethod
    def _calculate_stereo_stats(t_A: dict, t_B: dict,
                                inchi_a: str, inchi_b: str) -> dict:
        """Calculate stereochemistry statistics."""
        # pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-return-statements

        db_a = _extract_double_bond_stereo(inchi_a)
        db_b = _extract_double_bond_stereo(inchi_b)

        _, m_A, _ = parse_tms_with_unknowns(inchi_a)
        _, m_B, _ = parse_tms_with_unknowns(inchi_b)
        m_A = m_A or ""
        m_B = m_B or ""

        common_tetra = set(t_A) & set(t_B)
        defined_common = {
            k for k in common_tetra
            if t_A.get(k) in {"+", "-"} and t_B.get(k) in {"+", "-"}
        }

        if {m_A, m_B} == {"0", "1"} and str(sorted(t_A.items())) == str(sorted(t_B.items())):
            tetra_matches = 0
            tetra_flips = len(defined_common)
        else:
            tetra_matches = sum(1 for k in defined_common if t_A[k] == t_B[k])
            tetra_flips = sum(1 for k in defined_common if t_A[k] != t_B[k])

        common_db = set(db_a) & set(db_b)
        defined_db = {
            k for k in common_db
            if db_a.get(k) in {"+", "-"} and db_b.get(k) in {"+", "-"}
        }

        db_matches = sum(1 for k in defined_db if db_a[k] == db_b[k])
        db_flips = sum(1 for k in defined_db if db_a[k] != db_b[k])

        missing_tetra = sum(
            1 for k in common_tetra if t_A.get(k) == "?" or t_B.get(k) == "?"
        )
        missing_db = sum(
            1 for k in common_db if db_a.get(k) == "?" or db_b.get(k) == "?"
        )

        total_missing = missing_tetra + missing_db

        all_tetra = set(t_A) | set(t_B)
        all_db = set(db_a) | set(db_b)
        total_stereo = len(all_tetra) + len(all_db)

        stats = {
            "num_stereogenic_elements": total_stereo,
            "num_tetra_matches": tetra_matches,
            "num_tetra_flips": tetra_flips,
            "num_db_matches": db_matches,
            "num_db_flips": db_flips,
            "num_missing": total_missing
        }

        summary_logger.info("Stereo stats calculated: %s (m_A=%s, m_B=%s)",
                            stats, m_A, m_B)
        summary_logger.info("t_A: %s, t_B: %s", t_A, t_B)
        summary_logger.info("db_a: %s, db_b: %s", db_a, db_b)

        return stats


    @staticmethod
    def _classify_stereo_from_inchi(inchi_a: str, inchi_b: str) -> str:
        """Classify stereochemical relationship from two InChI strings."""
        # pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-return-statements

        try:
            t_A, m_A, s_A = parse_tms_with_unknowns(inchi_a)
            t_B, m_B, s_B = parse_tms_with_unknowns(inchi_b)
            t_A, m_A, s_A = parse_tms_with_unknowns(inchi_a)
            t_B, m_B, s_B = parse_tms_with_unknowns(inchi_b)

            db_a = _extract_double_bond_stereo(inchi_a)
            db_b = _extract_double_bond_stereo(inchi_b)

            frac_a = undefined_fraction_from_inchi(inchi_a)
            frac_b = undefined_fraction_from_inchi(inchi_b)

            if frac_a is None:
                frac_a = 1.0
            if frac_b is None:
                frac_b = 1.0

            m_A = m_A or ""
            m_B = m_B or ""
            s_A = s_A or ""
            s_B = s_B or ""

            if t_A is None:
                t_A = {}
            if t_B is None:
                t_B = {}

            if not t_A and not t_B and not db_a and not db_b:
                return "STEREO_UNDEFINED"

            t_A_undef = not t_A or all(v == "?" for v in t_A.values())
            t_B_undef = not t_B or all(v == "?" for v in t_B.values())
            t_A_def = t_A and any(v in {"+", "-"} for v in t_A.values())
            db_A_def = db_a and any(v in {"+", "-"} for v in db_a.values())
            t_A_has_def = t_A_def or db_A_def

            t_B_def = t_B and any(v in {"+", "-"} for v in t_B.values())
            db_B_def = db_b and any(v in {"+", "-"} for v in db_b.values())
            t_B_has_def = t_B_def or db_B_def

            if ((t_A_undef and t_B_def) or (t_A_def and t_B_undef)):
                return "PLANAR_VS_STEREO"
            if ((t_A_has_def and not t_B_has_def) or (not t_A_has_def and t_B_has_def)):
                return "PLANAR_VS_STEREO"

            has_unknown_A = _has_unknown_stereo(t_A, db_a)
            has_unknown_B = _has_unknown_stereo(t_B, db_b)

            if has_unknown_A or has_unknown_B or s_A != "1" or s_B != "1":
                max_frac = max(frac_a, frac_b)
                if max_frac > 0.4:
                    return "PLANAR_VS_STEREO"
                return "PLANAR_VS_STEREO"

            if m_A == "2" or m_B == "2":
                return "RACEMIC_OR_MIXTURE"

            if m_A == m_B and _all_signs_inverted_defined(t_A, db_a, t_B, db_b):
                return "ENANTIOMERS"

            t_A_str = str(sorted(t_A.items())) if t_A else ""
            t_B_str = str(sorted(t_B.items())) if t_B else ""
            dbA_str = str(sorted(db_a.items())) if db_a else ""
            dbB_str = str(sorted(db_b.items())) if db_b else ""

            if t_A_str == t_B_str and dbA_str == dbB_str and {m_A, m_B} == {"0", "1"}:
                return "ENANTIOMERS"

            if _any_defined_signs_different(t_A, t_B, db_a, db_b) or (
                set(_defined_subset(t_A)) != set(_defined_subset(t_B)) or
                set(_defined_subset(db_a)) != set(_defined_subset(db_b))
            ):
                return "DIASTEREOMERS"

            return "IDENTICAL_STEREO"

        except Exception as exc:
            logger.error("Error in _classify_stereo_from_inchi: %s", exc)
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
            stereo_stats: dict = None) -> Any:
        """Build confidence features for fallback classification."""
        # Handle None charges
        if charge1 is None:
            charge1 = 0
        if charge2 is None:
            charge2 = 0

        if stereo_stats is None:
            stereo_stats = {
                "num_stereogenic_elements": 0, "num_tetra_matches": 0,
                "num_tetra_flips": 0, "num_db_matches": 0,
                "num_db_flips": 0, "num_missing": 0}

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
        if hasattr(conf, 'score') and conf.score is not None:
            conf.score = max(0.0, conf.score - self.confidence_penalty)
        return conf

    def analyze_relationship_fallback(self, mol_a, mol_b, charge_a, charge_b,
                                  cid_a: str = "", cid_b: str = "") -> Optional[SimilarityResult]:
        """
        Note: ensure mol_a and mol_b are *molfile paths* for ChemistryOperations.*_software calls,
        or wrap RDKit mols into temp molfiles before calling.
        """
        logger.info("Using InChI fallback analysis for pair (%s, %s)", cid_a, cid_b)

        # compute tanimoto2d
        try:
            tanimoto2d = ChemistryOperations.fingerprint_tanimoto(mol_a, mol_b)
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
                t_A, _, _ = parse_tms_with_unknowns(layers_a_temp["inchi"])
                t_B, _, _ = parse_tms_with_unknowns(layers_b_temp["inchi"])
                stereo_stats = self._calculate_stereo_stats(
                    t_A,
                    t_B,
                    layers_a_temp["inchi"],
                    layers_b_temp["inchi"])
            else:
                stereo_stats = {
                "num_stereogenic_elements": 0, "num_tetra_matches": 0,
                "num_tetra_flips": 0, "num_db_matches": 0, "num_db_flips": 0}
        except Exception as e:
            logger.warning("Failed to calculate stereo stats: %s", e)
            stereo_stats = {
            "num_stereogenic_elements": 0, "num_tetra_matches": 0,
            "num_tetra_flips": 0, "num_db_matches": 0, "num_db_flips": 0}

        # Handle None charges early
        if charge_a is None:
            charge_a = 0
        if charge_b is None:
            charge_b = 0

        # EXPECTING paths; if you pass RDKit mols today, convert them before this point.
        layers_a = self._get_inchikey_layers(mol_a)
        layers_b = self._get_inchikey_layers(mol_b)
        if not layers_a or not layers_b:
            logger.warning("Failed to extract InChIKey layers for pair (%s, %s)", cid_a, cid_b )
            return _to_similarity_result(StereoClassification.no_classification())

        # First block: connectivity/skeleton
        if layers_a["first"] != layers_b["first"]:
            logger.info("Different molecular skeletons for pair (%s, %s)", cid_a, cid_b)
            return _to_similarity_result(StereoClassification.no_classification())

        second_diff = layers_a["second"] != layers_b["second"]  # stereo+isotopes
        third_diff = layers_a["third"] != layers_b["third"]     # protonation

        if third_diff and not second_diff:
            # Protomers
            logger.info("Protomers detected via fallback for pair (%s, %s)",
                        cid_a, cid_b)
            conf = self._build_fallback_confidence(
                "PROTOMERS", charge_a, charge_b, tanimoto2d,
                ik_first_eq=True, ik_protonation_layer_eq=False,
                ik_stereo_layer_eq=True, stereo_stats=stereo_stats)
            return _to_similarity_result(StereoClassification.protomers(
                stereo_score=conf.score, confidence=conf.as_dict(), rmsd=None,
                penalties={"fallback_method": True, "confidence_penalty": self.confidence_penalty}
            ))

        elif second_diff and not third_diff:
            # Stereo differs; decide enantiomer vs diastereomer via full InChI
            if charge_a != charge_b:
                logger.info(
                    "Different charges with stereochemistry differences for pair (%s, %s)",
                    cid_a, cid_b
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
                        'reason': 'Stereo undefined via fallback; no classification'
                        }
                )

            stereo_class = self._compare_full_inchi_stereochemistry(mol_a, mol_b)
            if stereo_class == "ENANTIOMERS":
                logger.info(
                    "Enantiomers detected via fallback for pair (%s, %s)",
                    cid_a, cid_b
                    )
                conf = self._build_fallback_confidence(
                    "ENANTIOMERS", charge_a, charge_b,
                    tanimoto2d, ik_first_eq=True,
                    ik_protonation_layer_eq=True,
                    ik_stereo_layer_eq=False,
                    stereo_stats=stereo_stats
                    )
                return _to_similarity_result(StereoClassification.enantiomers(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={
                        "fallback_method": True,
                        "confidence_penalty": self.confidence_penalty
                        }
                ))
            elif stereo_class == "DIASTEREOMERS":
                logger.info(
                    "Diastereomers detected via fallback for pair (%s, %s)",
                    cid_a, cid_b)
                conf = self._build_fallback_confidence(
                    "DIASTEREOMERS", charge_a, charge_b,
                    tanimoto2d, ik_first_eq=True,
                    ik_stereo_layer_eq=False,
                    ik_protonation_layer_eq=True,
                    stereo_stats=stereo_stats)
                return _to_similarity_result(StereoClassification.diastereomers(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={
                        "fallback_method": True,
                        "confidence_penalty": self.confidence_penalty
                        }
                ))
            elif stereo_class == "PLANAR VS_STEREO":
                logger.info(
                    "Planar vs Stereo detected via fallback for pair (%s, %s)",
                    cid_a, cid_b
                    )
                conf = self._build_fallback_confidence(
                    "PLANAR_VS_STEREO",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=True,
                    ik_stereo_layer_eq=False,
                    stereo_stats=stereo_stats
                    )
                return _to_similarity_result(StereoClassification.planar_vs_stereo(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={
                        "fallback_method": True,
                        "confidence_penalty": self.confidence_penalty
                        }
                ))
            else:
                logger.info(
                    "Stereo undefined or failed to classify via fallback for pair (%s, %s)",
                    cid_a, cid_b
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
                        'reason': 'Stereo and charge both differ; no classification'
                        }
                )

        elif second_diff and third_diff:
            if charge_a != charge_b:
                logger.info(
                    "Complex differences (stereo + charge) for pair (%s, %s)",
                    cid_a, cid_b
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
                        'reason': 'Stereo and charge differ (complex); no classification'
                        }
                )

        else:
            # Same InChIKey across all 3 blocks
            if charge_a != charge_b:
                logger.info(
                    """
                    Protomers (same InChIKey, diff charges) 
                    detected via fallback for pair (%s, %s)
                    """,
                    cid_a, cid_b
                    )
                conf = self._build_fallback_confidence(
                    "PROTOMERS",
                    charge_a,
                    charge_b,
                    tanimoto2d,
                    ik_first_eq=True,
                    ik_protonation_layer_eq=False,
                    ik_stereo_layer_eq=True,
                    stereo_stats=stereo_stats
                    )
                return _to_similarity_result(StereoClassification.protomers(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={
                        "fallback_method": True,
                        "confidence_penalty": self.confidence_penalty
                        }
                ))
            else:
                logger.info(
                    "Identical structures detected via fallback for pair (%s, %s)",
                    cid_a, cid_b
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
                    stereo_stats=stereo_stats
                    )
                return _to_similarity_result(StereoClassification.unresolved(
                    stereo_score=conf.score,
                    confidence=conf.as_dict(),
                    rmsd=None,
                    penalties={"fallback_method": True},
                    details={
                        "reason": "Possible pipeline error - should be no identical pairs here"
                        }
                ))
