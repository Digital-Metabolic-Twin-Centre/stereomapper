"""Scope rules applied before retrieval (FR-005, FR-005a).

Only the Rhea status rule is applied here. No reaction is excluded by pathway or subsystem;
subsystem flags are applied after proposals by ``flags.py`` (FR-005b).
"""

from __future__ import annotations

from stereomapper.reconciliation.model import ExclusionRule, ScopeExclusion, SourceReaction


def apply_rhea_status_rule(
    rhea_masters: dict[str, SourceReaction], include: tuple[str, ...]
) -> tuple[dict[str, SourceReaction], list[ScopeExclusion]]:
    in_scope: dict[str, SourceReaction] = {}
    exclusions: list[ScopeExclusion] = []
    parameter = ";".join(include)
    for rid in sorted(rhea_masters):
        reaction = rhea_masters[rid]
        status = reaction.rhea_status.value if reaction.rhea_status else ""
        if status in include:
            in_scope[rid] = reaction
            continue
        exclusions.append(
            ScopeExclusion(
                source="rhea",
                reaction_id=rid,
                rule=ExclusionRule.RHEA_OBSOLETE,
                rule_parameter=parameter,
                evidence=status,
                reason=f"Rhea status {status!r} is outside the configured statuses",
            )
        )
    return in_scope, exclusions
