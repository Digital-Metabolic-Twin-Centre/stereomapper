from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from stereomapper.reconciliation.inputs import (
    InputContractError,
    load_extract,
    read_flag_list,
    read_tsv,
)
from stereomapper.tests.reconciliation.conftest import Scenario, write_tsv


def _basic(scenario: Scenario) -> Path:
    scenario.vmh("R1", [("a", "chebi:1")], [("b", "chebi:2")])
    scenario.rhea(10, [("CHEBI:1", "chebi:1")], [("CHEBI:2", "chebi:2")])
    scenario.build()
    return scenario.root / "extract"


def _rewrite(path: Path, transform: Callable[[list[str]], list[str]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(transform(lines)) + "\n", encoding="utf-8")


def test_valid_extract_loads(scenario: Scenario) -> None:
    extract = load_extract(_basic(scenario))
    assert len(extract.reactions.rows) == 2
    assert len(extract.participants.rows) == 4
    assert extract.manifest["vmh_release"] == "vmh-test"


def test_missing_required_column_names_file_and_column(scenario: Scenario) -> None:
    extract_dir = _basic(scenario)
    path = extract_dir / "participants.tsv"
    _rewrite(
        path,
        lambda lines: [
            ln.replace("coefficient", "coef", 1) if i == 0 else ln for i, ln in enumerate(lines)
        ],
    )
    with pytest.raises(InputContractError) as err:
        load_extract(extract_dir)
    assert err.value.file == "participants.tsv"
    assert err.value.field_name == "coefficient"


@pytest.mark.parametrize(
    "column,bad",
    [("rhea_status", "retired"), ("direction", "sideways")],
)
def test_invalid_reaction_enumerations_rejected(scenario: Scenario, column: str, bad: str) -> None:
    extract_dir = _basic(scenario)
    path = extract_dir / "reactions.tsv"
    header = path.read_text().splitlines()[0].split("\t")
    idx = header.index(column)
    row_pos = 2 if column == "rhea_status" else 1

    def change(lines: list[str]) -> list[str]:
        cells = lines[row_pos].split("\t")
        cells[idx] = bad
        lines[row_pos] = "\t".join(cells)
        return lines

    _rewrite(path, change)
    with pytest.raises(InputContractError) as err:
        load_extract(extract_dir)
    assert err.value.field_name == column
    assert err.value.row == row_pos + 1


def test_invalid_kind_rejected(scenario: Scenario) -> None:
    scenario.vmh("R1", [{"metabolite_id": "a", "kind": "protein"}], [("b", "chebi:2")])
    scenario.build()
    with pytest.raises(InputContractError) as err:
        load_extract(scenario.root / "extract")
    assert err.value.field_name == "kind"


@pytest.mark.parametrize("coefficient", ["0", "-1", "abc"])
def test_bad_coefficient_rejected(scenario: Scenario, coefficient: str) -> None:
    scenario.vmh("R1", [("a", "chebi:1", coefficient)], [("b", "chebi:2")])
    scenario.build()
    with pytest.raises(InputContractError) as err:
        load_extract(scenario.root / "extract")
    assert err.value.field_name == "coefficient"


def test_symbolic_coefficient_allowed_for_polymer(scenario: Scenario) -> None:
    scenario.rhea(
        10,
        [{"metabolite_id": "CHEBI:9", "kind": "polymer", "coefficient": "n"}],
        [("CHEBI:2", "chebi:2")],
    )
    scenario.build()
    extract = load_extract(scenario.root / "extract")
    assert extract.participants.rows[0]["coefficient"] == "n"


def test_duplicate_entry_key_rejected(scenario: Scenario) -> None:
    scenario.vmh("R1", [("a", "chebi:1"), ("a", "chebi:1")], [("b", "chebi:2")])
    scenario.build()
    with pytest.raises(InputContractError) as err:
        load_extract(scenario.root / "extract")
    assert "duplicate participant" in str(err.value)


def test_extra_columns_kept(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv"
    write_tsv(path, ["a", "b", "extra"], [{"a": "1", "b": "2", "extra": "x"}])
    table = read_tsv(path, ("a", "b"))
    assert table.rows[0]["extra"] == "x"


def test_flag_list_version_parsed_and_rationale_required(tmp_path: Path) -> None:
    path = tmp_path / "flags.tsv"
    path.write_text(
        "# list_version: v1\nsubsystem\tcategory\trationale\n"
        "Glycolysis/gluconeogenesis\tsugar\tsugar reactions are highly similar\n",
        encoding="utf-8",
    )
    flags = read_flag_list(path)
    assert flags.list_version == "v1"
    assert flags.entries[0]["category"] == "sugar"

    path.write_text(
        "# list_version: v1\nsubsystem\tcategory\trationale\nDrug metabolism\tdrug\t\n",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError) as err:
        read_flag_list(path)
    assert err.value.field_name == "rationale"


def test_flag_list_requires_version(tmp_path: Path) -> None:
    path = tmp_path / "flags.tsv"
    path.write_text("subsystem\tcategory\trationale\nX\tdrug\tr\n", encoding="utf-8")
    with pytest.raises(InputContractError) as err:
        read_flag_list(path)
    assert err.value.field_name == "list_version"
