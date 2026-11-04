from __future__ import annotations

from typing import Any

from stereomapper.tests.reconciliation.conftest import Scenario, propose, read_rows

S, P = ("s", "vmh:s"), ("p", "vmh:p")
S_R, P_R = ("CHEBI:s", "chebi:s"), ("CHEBI:p", "chebi:p")


def _pair(
    scenario: Scenario, vmh_id: str, master: int, status: str = "approved", **kw: Any
) -> None:
    scenario.vmh(vmh_id, [S], [P], **kw)
    scenario.rhea(master, [S_R], [P_R], status=status)


def test_obsolete_rhea_excluded_and_recorded(scenario: Scenario) -> None:  # FR-005a
    _pair(scenario, "E1", 100, status="obsolete")
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = propose(scenario.build())
    assert not out.has_proposal("E1", "RHEA:100")
    exclusions = read_rows(out.run_dir / "scope_exclusions.tsv")
    assert exclusions == [
        {
            "rule": "rhea_obsolete",
            "source": "rhea",
            "reaction_id": "RHEA:100",
            "rule_parameter": "approved;preliminary",
            "evidence": "obsolete",
            "reason": "Rhea status 'obsolete' is outside the configured statuses",
        }
    ]
    assert out.funnel["rhea_excluded_obsolete"] == ("1", "1")


def test_preliminary_rhea_proposed_and_flagged(scenario: Scenario) -> None:  # FR-005a
    _pair(scenario, "E1", 100, status="preliminary")
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    row = propose(scenario.build()).proposal("E1", "RHEA:100")
    assert row["rhea_preliminary"] == "true"


def test_every_subsystem_is_proposed(scenario: Scenario) -> None:  # FR-005
    _pair(scenario, "G1", 100, subsystems=("Glycolysis/gluconeogenesis",))
    scenario.vmh("D1", [S], [P], subsystems=("Drug metabolism", "Other"))
    scenario.same("vmh:s", "chebi:s").same("vmh:p", "chebi:p")
    out = propose(scenario.build())
    assert out.proposal("G1", "RHEA:100")["vmh_subsystems"] == "Glycolysis/gluconeogenesis"
    assert out.proposal("D1", "RHEA:100")["vmh_subsystems"] == "Drug metabolism;Other"
    assert out.funnel["vmh_in_scope"] == ("2", "2")
    assert read_rows(out.run_dir / "scope_exclusions.tsv") == []
