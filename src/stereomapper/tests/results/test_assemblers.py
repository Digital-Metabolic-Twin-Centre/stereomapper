import hashlib
from pathlib import Path

import pytest

from stereomapper.results import assemblers


def test_prefixed_identifier_strips_existing_namespace(tmp_path: Path):
    mol_path = tmp_path / "swisslipids_structures" / "SLM-folder" / "SLM_496071.mol"
    mol_path.parent.mkdir(parents=True)
    mol_path.touch()

    prefixed = assemblers.prefixed_identifier(str(mol_path), "SLM:496071")

    assert prefixed == "slm:496071"


def test_prefixed_identifier_handles_unknown_directory(tmp_path: Path):
    mol_path = tmp_path / "misc" / "CHEBI_17994.mol"
    mol_path.parent.mkdir(parents=True)
    mol_path.touch()

    prefixed = assemblers.prefixed_identifier(str(mol_path), "CHEBI:17994")

    assert prefixed == "unknown:17994"


def test_coerce_scalar_normalises_values():
    assert assemblers._coerce_scalar(None) is None
    assert assemblers._coerce_scalar(5) == pytest.approx(5.0)
    assert assemblers._coerce_scalar("3.5") == pytest.approx(3.5)
    assert assemblers._coerce_scalar("not-a-number") is None
    assert assemblers._coerce_scalar(float("nan")) is None


def test_details_from_res_accepts_object_confidence():
    class DummyConfidence:
        bin = "medium"

    res = {"confidence": DummyConfidence()}
    assert assemblers._details_from_res(res) == {"confidence_bin": "medium"}


def test_normalise_classification_extracts_score_and_extra_info():
    res = {
        "classification": "IDENTICAL",
        "confidence_score": 98,
        "details": {"note": "perfect match"},
    }

    summary = assemblers._normalise_classification(res)

    assert summary["classification"] == "IDENTICAL"
    assert summary["score"] == pytest.approx(98.0)
    assert summary["extra_info"] == "perfect match"
    assert summary["score_details"]["confidence"] is None


def test_cluster_rows_merges_defined_and_plain_members():
    rows = [
        {
            "inchikey_first": "AAAAAA",
            "smiles": "C",
            "is_undef_sru": 0,
            "is_def_sru": 1,
            "sru_repeat_count": 2,
            "accession_curies": ["chebi:1"],
        },
        {
            "inchikey_first": "AAAAAA",
            "smiles": "C",
            "is_undef_sru": 0,
            "is_def_sru": 0,
            "sru_repeat_count": None,
            "accession_curies": ["chebi:2"],
        },
    ]

    result = list(assemblers.cluster_rows(rows))

    assert len(result) == 1
    ik, smi, is_undef, is_def, rep_cnt, member_count, members_json, members_hash = result[0]
    assert ik == "AAAAAA"
    assert smi == "C"
    assert is_def == 1 and is_undef == 0
    assert rep_cnt == 2
    assert member_count == 2
    assert members_json == '["chebi:1", "chebi:2"]'
    expected_hash = hashlib.sha256("chebi:1\nchebi:2".encode("utf-8")).hexdigest()
    assert members_hash == expected_hash


def test_hash_file_returns_md5(tmp_path: Path):
    target = tmp_path / "example.mol"
    target.write_text("hello world", encoding="utf-8")

    digest = assemblers.hash_file(str(target))

    assert digest == hashlib.md5(b"hello world").hexdigest()


def test_make_molecule_key_uses_provided_identifiers():
    expected = hashlib.blake2b(
        "1|ABCDEFGHIJKLMN|q=1".encode("utf-8"), digest_size=16
    ).hexdigest()

    key = assemblers.make_molecule_key(
        std_version="1",
        inchikey_full="ABCDEFGHIJKLMN",
        formal_charge=1,
    )

    assert key == expected
