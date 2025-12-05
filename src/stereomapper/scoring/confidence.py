# stereomapper/logic/confidence.py
"""Core confidence scoring logic."""

import math
from typing import Dict, Any, Optional
from stereomapper.utils.logging import setup_logging
from stereomapper.models.stereo_elements import StereoCounts
from stereomapper.models.stereo_elements import StereoCounts

from .models import ConfidenceResult

logger, summary_logger = setup_logging(
    console=True,
    level="INFO",           # Detailed logging to files
    quiet_console=True,     # Minimal console output during progress bar
    console_level="ERROR"   # Only errors to console
)

# Move all the helper functions here (unchanged)
def _isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))

def _nz(x, default=0.0):
    return x if _isnum(x) else default

def clamp01(x: float) -> float:
    return 0.0 if x is None else max(0.0, min(1.0, x))

def exp_decay_rmsd_class_aware(rmsd: Optional[float], assigned_class: str) -> float:
    """Class-aware RMSD scoring with different scales."""
    if not _isnum(rmsd) or rmsd < 0:
        return 0.0

    if assigned_class in {"ENANTIOMERS", "ENANTIOMER"}:
        scale = 0.3  # Stricter for enantiomers
    elif assigned_class in {"IDENTICAL", "IDENTICALS", "IDENTICAL_MOLECULES"}:
        scale = 0.1  # Very strict for identical
    else:
        scale = 0.75  # Default

    return clamp01(math.exp(-rmsd / scale))

def undef_sites_penalty(n_undef: Optional[int]) -> float:
    """0 -> 1.0, 1 -> 0.67, 2 -> 0.5, 3 -> 0.4 …"""
    if n_undef is None or not isinstance(n_undef, (int, float)) or n_undef < 0:
        return 1.0
    return 1.0 / (1.0 + 0.5 * float(n_undef))

BIN_RULES = [
    ("high",      90),
    ("medium",    70),
    ("low",       40),
]

def _bin_class_aware(score: int, assigned_class: str) -> str:
    """Bin confidence scores using standard thresholds."""
    for label, thr in BIN_RULES:
        if score >= thr:
            return label
    return "very_low"

class ConfidenceScorer:
    """Main confidence scoring engine."""

    def score(
        self,
        assigned_class: str,
        *,
        rmsd: Optional[float] = None,
        charge_match: Optional[int] = None,
        tanimoto2d: Optional[float] = None,
        ik_first_eq: Optional[int] = None,
        ik_stereo_layer_eq: Optional[int] = None,
        ik_protonation_layer_eq: Optional[int] = None,
        counts: StereoCounts = None,
        **kwargs,
    ) -> ConfidenceResult:
        """
        Heuristic, class-aware confidence. Robust to None/NaN inputs.
        """

        counts = self._build_counts(counts, kwargs)
        # Normalised helpers
        geom = exp_decay_rmsd_class_aware(rmsd, assigned_class)

        # compute relevant stereo fractions
        denom = max(1, _nz(counts.num_stereogenic_elements, 0))
        stereo_match_frac = (_nz(counts.num_tetra_matches,0) + _nz(counts.num_db_matches,0)) / denom
        stereo_opposite_frac = (_nz(counts.num_tetra_flips,0) + _nz(counts.num_db_flips,0)) / denom
        # ---- Core score components ----

        # Safe numbers
        sm = clamp01(_nz(stereo_match_frac, 0.0))
        so = clamp01(_nz(stereo_opposite_frac, 0.0))
        tan2d = clamp01(_nz(tanimoto2d, 0.0))
        ik1 = 1 if _nz(ik_first_eq, 0)==1 else 0
        ikS = 1 if _nz(ik_stereo_layer_eq, 0)==1 else 0
        ikP = 1 if _nz(ik_protonation_layer_eq, 0)==1 else 0
        chg = 1 if _nz(charge_match, 1)==1 else 0

        contributors: Dict[str, float] = {}
        # deductions: Dict[str, float] = {}
        # boosts: Dict[str, float] = {}
        expectations: Dict[str, Any] = {}

        # ---- Class-specific support & expectations ----
        assigned_class = str(assigned_class).upper()

        if assigned_class in {"IDENTICAL", "IDENTICALS", "IDENTICAL_MOLECULES", "UNRESOLVED"}:
            # ensure only matches are present if stereogenic elements exits
            if counts.num_stereogenic_elements and counts.num_stereogenic_elements > 0:
                # ensure no opposite stereo
                if so > 0:
                    stereo_consistency_mult = 0.1  # Heavily penalize any opposite stereo
                elif sm < 0.99:
                    stereo_consistency_mult = 0.5  # Penalize if not near perfect match
                else:
                    stereo_consistency_mult = 1.0  # Perfect
            else:
                stereo_consistency_mult = 1.0  # No stereogenic elements, no penalty needed
            S = (0.50*stereo_consistency_mult + 0.25*geom + 0.25*tan2d)
            contributors.update({
                "stereo_consistency_mult*0.50": 0.50*stereo_consistency_mult,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":1, "ik_protonation_layer_eq":1, "charge_match":1}

        elif assigned_class in {"PROTOMERS", "IDENTICAL_DIFFERENT_CHARGE", "IDENTICAL_CHARGE_DIFF"}:
            # similar to identical but allow charge mismatch
            charge_match = _nz(charge_match, 0)
            chg = 1 if charge_match==1 else 0

            if charge_match == 1:
                # penalise heavily for charge match, expect mismatch here
                charge_mult = 0.2
            else:
                charge_mult = 1.0
            # ensure only matches are present if stereogenic elements exists
            if counts.num_stereogenic_elements and counts.num_stereogenic_elements > 0:
                # ensure no opposite stereo
                if so > 0:
                    stereo_consistency_mult = 0.1  # Heavily penalize any opposite stereo
                elif sm < 0.99:
                    stereo_consistency_mult = 0.5  # Penalize if not near perfect match
                else:
                    stereo_consistency_mult = 1.0  # Perfect
            else:
                stereo_consistency_mult = 1.0  # No stereogenic elements, no penalty needed

            S = (0.50*charge_mult + 0.25*geom + 0.25*tan2d) * stereo_consistency_mult
            contributors.update({
                "charge_mult*0.50": 0.50*charge_mult,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
                "*stereo_consistency_mult": stereo_consistency_mult,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":1, "charge_match":0}

        elif assigned_class in {"IDENTICAL_MISSING_CHARGE", "IDENTICAL_MISSING_CHARGE"}:
            S = (0.50*sm + 0.25*geom + 0.25*tan2d)
            contributors.update({
                "stereo_match*0.50": 0.50*sm,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":1}

        elif assigned_class in {"ENANTIOMERS", "ENANTIOMER"}:
            expected_stereo = _nz(counts.num_stereogenic_elements, 0)
            actual_flips = _nz(counts.num_tetra_flips, 0)
            actual_matches = _nz(counts.num_tetra_matches, 0)
            actual_db_flips = _nz(counts.num_db_flips, 0)
            actual_db_matches = _nz(counts.num_db_matches, 0)
            num_missing = _nz(counts.num_missing, 0)

            # explicit check for db flips, should == 0
            if actual_db_flips > 0:
                stereo_consistency_mult = 0.1
                partial_penalty = max(0, sm * 2.0)
            elif num_missing > 0:
                stereo_consistency_mult = 0.2
                partial_penalty = max(0, sm * 2.0)
            elif ((actual_flips + actual_db_matches) == expected_stereo) and actual_matches == 0:
                stereo_consistency_mult = 1.0
                partial_penalty = 0.0
            elif actual_matches > 0:
                stereo_consistency_mult = 0.3
                partial_penalty = max(0, sm * 2.0)
            else:
                stereo_consistency_mult = 0.6
                partial_penalty = max(0, sm * 2.0)

            S = (0.50*stereo_consistency_mult + 0.25*geom + 0.25*tan2d - 0.10*partial_penalty)
            contributors.update({
                "stereo_consistency_mult*0.50": 0.50*stereo_consistency_mult,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
                "partial_match_penalty*-0.10": -0.10*partial_penalty,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":0, "charge_match":1}

        elif assigned_class in {"DIASTEREOMERS", "DIASTEREOMER"}:
            # diff term should be a weighted calculation, dependent on number of stereo elements

            # diff to enantiomers, expect some matches
            num_missing = _nz(counts.num_missing, 0)
            num_tetra_flips = _nz(counts.num_tetra_flips, 0)
            num_tetra_matches = _nz(counts.num_tetra_matches, 0)
            num_db_flips = _nz(counts.num_db_flips, 0)
            num_db_matches = _nz(counts.num_db_matches, 0)
            expected_stereo_elements= _nz(counts.num_stereogenic_elements, 0)
            actual_elements = num_tetra_flips + num_tetra_matches + num_db_flips + num_db_matches

            # cant calc expected flips, but should be all defined, with some matches , no missing
            if num_missing > 0:
                # penalise heavily
                stereo_consistency_mult = 0.2
            elif actual_elements == expected_stereo_elements:
                # what we want , dont penalise
                stereo_consistency_mult = 1.0
            elif actual_elements < expected_stereo_elements:
                # not all defined, penalise
                stereo_consistency_mult = 0.5
            else:
                # too many defined, penalise
                stereo_consistency_mult = 0.5

            S = (0.50*stereo_consistency_mult + 0.25*geom  + 0.25*tan2d) # using diff term unfairly penalises diastereomers with many stereo centres matching stereochemistry

            contributors.update({
                "stereo_consistency*0.50": 0.50*stereo_consistency_mult,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":0, "charge_match":1}

        elif assigned_class in {"PLANAR_VS_STEREO", "2D vs 3D structures", "2D vs 3D", "Parent-child",
                                "STEREO-RESOLUTION PAIRS", "STEREO RESOLUTION PAIRS", "Stereo-resolution pairs"}:
            # expect one to be more stereochemically defined than the other
            num_missing = _nz(counts.num_missing, 0)
            # if structure A has more defined stereo elements than B, assign mult of 1.0
            if num_missing > 0:
                stereo_consistency_mult = 1.0
            elif num_missing == 0:
                stereo_consistency_mult = 0.2  # heavily penalise if both fully defined
            else:
                stereo_consistency_mult = 0.5  # penalise if both partially defined

            S = (0.50*stereo_consistency_mult + 0.25*geom + 0.25*tan2d)
            contributors.update({
                "stereo_opposite*0.50": 0.50*stereo_consistency_mult,  # Match what you're actually using
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
            })
            expect = {"ik_first_eq":1, "charge_match":1} # would expect one of strucs to have 2nd layer 'UHHF..

        else:  # fallback
            diff_term = so if so > 0 else (1.0 - sm)
            S = (0.50*diff_term + 0.25*geom + 0.25*tan2d)
            contributors.update({
                "stereo_diff*0.50": 0.50*diff_term,
                "geom*0.25": 0.25*geom,
                "tanimoto2d*0.25": 0.25*tan2d,
            })
            expect = {"ik_first_eq":1, "ik_stereo_layer_eq":0, "charge_match":1}

        score = int(round(100 * clamp01(S)))

        # Then in your score method:
        bin_label = _bin_class_aware(score, assigned_class)
        expectations = {
            "expected": expect,
            "observed": {
                "ik_first_eq": ik1, "ik_stereo_layer_eq": ikS, "ik_protonation_layer_eq": ikP, "charge_match": chg
            }
        }
        return ConfidenceResult(
            score=score,
            bin=bin_label,
            contributors=contributors,
            expectations=expectations,
        )

    def _build_counts(self, counts: StereoCounts, kwargs: dict) -> StereoCounts:
        if counts is not None:
            return counts

        return StereoCounts(
            num_stereogenic_elements=kwargs.get("num_stereogenic_elements", 0),
            num_tetra_matches=kwargs.get("num_tetra_matches", 0),
            num_tetra_flips=kwargs.get("num_tetra_flips", 0),
            num_db_matches=kwargs.get("num_db_matches", 0),
            num_db_flips=kwargs.get("num_db_flips", 0),
            num_missing=kwargs.get("num_missing", 0),
            num_unspecified=kwargs.get("num_unspecified", 0),
        )