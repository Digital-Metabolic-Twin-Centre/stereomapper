import math

import pytest

from stereomapper.scoring.confidence import (
    ConfidenceScorer,
    clamp01,
    exp_decay_rmsd_class_aware,
    undef_sites_penalty,
    _bin_class_aware,
    _isnum,
    _nz,
)
from stereomapper.scoring.models import ConfidenceResult


@pytest.fixture
def scorer() -> ConfidenceScorer:
    return ConfidenceScorer()


def test_isnum_handles_common_edge_cases():
    assert not _isnum(None)
    assert not _isnum(float("nan"))
    assert _isnum(0.0)
    assert _isnum(5)


def test_nz_returns_default_for_missing_values():
    assert _nz(None, 2.5) == 2.5
    assert _nz(float("nan"), 42) == 42
    assert _nz(7, 1) == 7


def test_clamp01_limits_values_between_zero_and_one():
    assert clamp01(None) == 0.0
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.4) == 0.4
    assert clamp01(1.5) == 1.0


def test_exp_decay_rmsd_class_aware_uses_class_specific_scales():
    assert exp_decay_rmsd_class_aware(None, "IDENTICAL") == 0.0
    assert exp_decay_rmsd_class_aware(-0.1, "IDENTICAL") == 0.0
    assert exp_decay_rmsd_class_aware(0.0, "IDENTICAL") == 1.0
    identical = exp_decay_rmsd_class_aware(0.1, "IDENTICAL")
    assert identical == pytest.approx(math.exp(-0.1 / 0.1))
    enantiomer = exp_decay_rmsd_class_aware(0.3, "ENANTIOMERS")
    assert enantiomer == pytest.approx(math.exp(-0.3 / 0.3))
    fallback = exp_decay_rmsd_class_aware(0.75, "OTHER")
    assert fallback == pytest.approx(math.exp(-0.75 / 0.75))


def test_undef_sites_penalty_monotonically_decreases():
    assert undef_sites_penalty(None) == 1.0
    assert undef_sites_penalty("invalid") == 1.0
    assert undef_sites_penalty(-5) == 1.0
    assert undef_sites_penalty(0) == 1.0
    assert undef_sites_penalty(1) == pytest.approx(1.0 / 1.5)
    assert undef_sites_penalty(2) == pytest.approx(1.0 / 2.0)


def test_bin_class_aware_thresholds_are_applied():
    assert _bin_class_aware(95, "IDENTICAL") == "high"
    assert _bin_class_aware(75, "IDENTICAL") == "medium"
    assert _bin_class_aware(45, "IDENTICAL") == "low"
    assert _bin_class_aware(10, "IDENTICAL") == "very_low"


def test_confidence_result_as_dict_round_trips():
    result = ConfidenceResult(
        score=88,
        bin="medium",
        contributors={"geom": 0.4},
        expectations={"expected": {"ik_first_eq": 1}},
    )
    assert result.as_dict() == {
        "score": 88,
        "bin": "medium",
        "contributors": {"geom": 0.4},
        "expectations": {"expected": {"ik_first_eq": 1}},
    }


def test_identical_class_with_perfect_alignment_scores_high(scorer: ConfidenceScorer):
    result = scorer.score(
        "IDENTICAL",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=1,
        ik_protonation_layer_eq=1,
    )

    assert result.score == 100
    assert result.bin == "high"
    assert result.contributors["stereo_consistency_mult*0.50"] == pytest.approx(0.5)
    assert result.contributors["geom*0.25"] == pytest.approx(0.25)
    assert result.contributors["tanimoto2d*0.25"] == pytest.approx(0.25)
    assert result.expectations["expected"] == {
        "ik_first_eq": 1,
        "ik_stereo_layer_eq": 1,
        "ik_protonation_layer_eq": 1,
        "charge_match": 1,
    }
    assert result.expectations["observed"] == {
        "ik_first_eq": 1,
        "ik_stereo_layer_eq": 1,
        "ik_protonation_layer_eq": 1,
        "charge_match": 1,
    }


def test_identical_class_penalises_stereo_mismatches(scorer: ConfidenceScorer):
    result = scorer.score(
        "IDENTICAL",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=1,
        ik_protonation_layer_eq=1,
        num_stereogenic_elements=2,
        num_tetra_matches=1,
        num_tetra_flips=1,
    )

    assert result.score == 55
    assert result.bin == "low"
    assert result.contributors["stereo_consistency_mult*0.50"] == pytest.approx(0.05)
    assert result.expectations["expected"]["ik_stereo_layer_eq"] == 1
    assert result.expectations["observed"]["ik_stereo_layer_eq"] == 1


def test_protomers_expect_charge_mismatch(scorer: ConfidenceScorer):
    result = scorer.score(
        "PROTOMERS",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=0,
        ik_first_eq=1,
        ik_stereo_layer_eq=1,
    )

    assert result.score == 100
    assert result.bin == "high"
    assert result.contributors["charge_mult*0.50"] == pytest.approx(0.5)
    assert result.contributors["*stereo_consistency_mult"] == pytest.approx(1.0)
    assert result.expectations["expected"]["charge_match"] == 0
    assert result.expectations["observed"]["charge_match"] == 0


def test_enantiomers_full_flip_scores_high(scorer: ConfidenceScorer):
    result = scorer.score(
        "ENANTIOMERS",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=0,
        num_stereogenic_elements=2,
        num_tetra_flips=2,
        num_tetra_matches=0,
        num_db_matches=0,
        num_db_flips=0,
        num_missing=0,
    )

    assert result.score == 100
    assert result.bin == "high"
    assert result.contributors["stereo_consistency_mult*0.50"] == pytest.approx(0.5)
    assert result.contributors["partial_match_penalty*-0.10"] == pytest.approx(0.0)
    assert result.expectations["expected"]["ik_stereo_layer_eq"] == 0
    assert result.expectations["observed"]["ik_stereo_layer_eq"] == 0


def test_enantiomers_partial_matches_are_penalised(scorer: ConfidenceScorer):
    result = scorer.score(
        "ENANTIOMERS",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=0,
        num_stereogenic_elements=2,
        num_tetra_flips=1,
        num_tetra_matches=1,
        num_db_matches=0,
        num_db_flips=0,
        num_missing=0,
    )

    assert result.score == 55
    assert result.bin == "low"
    assert result.contributors["stereo_consistency_mult*0.50"] == pytest.approx(0.15)
    assert result.contributors["partial_match_penalty*-0.10"] == pytest.approx(-0.10)


def test_diastereomers_missing_sites_reduce_score(scorer: ConfidenceScorer):
    result = scorer.score(
        "DIASTEREOMERS",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=0,
        num_stereogenic_elements=3,
        num_tetra_matches=1,
        num_tetra_flips=1,
        num_db_matches=0,
        num_db_flips=0,
        num_missing=1,
    )

    assert result.score == 60
    assert result.bin == "low"
    assert result.contributors["stereo_consistency*0.50"] == pytest.approx(0.10)
    assert result.expectations["expected"]["ik_stereo_layer_eq"] == 0


def test_planar_vs_stereo_prefers_missing_centres(scorer: ConfidenceScorer):
    result = scorer.score(
        "PLANAR_VS_STEREO",
        rmsd=0.0,
        tanimoto2d=1.0,
        charge_match=1,
        ik_first_eq=1,
        num_stereogenic_elements=3,
        num_missing=3,
    )

    assert result.score == 100
    assert result.bin == "high"
    assert result.contributors["stereo_opposite*0.50"] == pytest.approx(0.5)


def test_fallback_class_uses_difference_term(scorer: ConfidenceScorer):
    result = scorer.score(
        "AMBIGUOUS_CLASS",
        rmsd=0.0,
        tanimoto2d=0.8,
        charge_match=1,
        ik_first_eq=1,
        ik_stereo_layer_eq=0,
        num_stereogenic_elements=2,
        num_tetra_matches=1,
        num_tetra_flips=0,
    )

    assert result.score == 70
    assert result.bin == "medium"
    assert result.contributors["stereo_diff*0.50"] == pytest.approx(0.25)
    assert result.contributors["geom*0.25"] == pytest.approx(0.25)
    assert result.contributors["tanimoto2d*0.25"] == pytest.approx(0.2)
    assert result.expectations["expected"] == {
        "ik_first_eq": 1,
        "ik_stereo_layer_eq": 0,
        "charge_match": 1,
    }
