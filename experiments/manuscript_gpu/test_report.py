"""Results of the test evaluations and the decisions of the approved protocols.

Writes results/test_summary.{md,json}. Intervals resample the routes of the readable test traversals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from mapper_runs import TEST_REPRODUCTION_VARIANT, TEST_VARIANTS
from report import by_route, load, loop_rows, split_section, table, unit_counts
from stream import RESULTS, Sequence, readable_test
from summarize import bootstrap, counts

F1 = 2


@dataclass(frozen=True)
class Hypothesis:
    """Tested once every hypothesis in ``requires`` is confirmed; confirmed if the interval of F1(a) − F1(b) clears ``bound``."""

    name: str
    a: str
    b: str
    bound: float
    requires: tuple[str, ...] = ()


PROTOCOLS = {
    "First evaluation (docs/TEST_SET_PROTOCOL_DRAFT_2026-11-19.md)": [
        Hypothesis("primary", "topo", "previous_system", 0.0),
        Hypothesis("non-inferiority VLAD", "topo", "anyloc_vlad_g", -0.10, ("primary",)),
        Hypothesis("non-inferiority GeM", "topo", "anyloc_gem_g", -0.10, ("primary",)),
    ],
    "Second evaluation (docs/TEST_PROTOCOL_DEBIASED_2026-11-20.md)": [
        Hypothesis("H1", "topo_debiased", "topo", 0.0),
        Hypothesis("H2", "topo_debiased", "previous_system", 0.0, ("H1",)),
        Hypothesis("H3", "topo_debiased", "anyloc_vlad_g", 0.0, ("H1", "H2")),
        Hypothesis("H4", "topo_debiased", "anyloc_gem_g", 0.0, ("H1", "H2", "H3")),
    ],
}
RECORDED_PREVIOUS_SYSTEM_2M = (74, 30, 147)
LOOP_RULE_PROTOCOL = "docs/TEST_PROTOCOL_PREDECESSOR_EXCLUSION_2026-11-21.md"
DOMAIN_SUBSETS = {
    "COLD": lambda s: s.dataset == "cold",
    "CID-SIMS": lambda s: s.dataset == "cid",
    "COLD Ljubljana": lambda s: "ljubljana" in s.recording,
}


def check_reproduction(results: dict, seqs: list[Sequence]) -> None:
    total = sum(counts(results[(s.name, TEST_REPRODUCTION_VARIANT)], 2.0) for s in seqs)
    got = tuple(int(x) for x in total[:3])
    if got != RECORDED_PREVIOUS_SYSTEM_2M:
        raise SystemExit(f"{TEST_REPRODUCTION_VARIANT} does not reproduce previous_system of the first evaluation: "
                         f"TP/FP/FN {got} instead of {RECORDED_PREVIOUS_SYSTEM_2M}")


def decisions(results: dict, seqs: list[Sequence]) -> tuple[list[str], dict]:
    variants = sorted({v for hypotheses in PROTOCOLS.values() for h in hypotheses for v in (h.a, h.b)})
    per_route = unit_counts(results, seqs, variants, by_route)
    lines = ["## Decisions of the protocols", "", f"Routes resampled: {len(per_route)}. Traversals: {len(seqs)}.", ""]
    data: dict = {"routes": len(per_route)}
    for protocol, hypotheses in PROTOCOLS.items():
        lines += [f"### {protocol}", ""]
        outcome: dict[str, dict] = {}
        for h in hypotheses:
            result = bootstrap(per_route, h.a, h.b, F1)
            tested = all(outcome[r]["confirmed"] for r in h.requires)
            result["confirmed"] = tested and result["ci95"][0] > h.bound
            verdict = ("CONFIRMED" if result["confirmed"] else "NOT CONFIRMED") if tested else "not tested"
            outcome[h.name] = result
            lines.append(f"- {h.name}, F1 {h.a} − {h.b} (bound {h.bound:+.2f}): {result['point']:+.3f} "
                         f"[{result['ci95'][0]:+.3f}; {result['ci95'][1]:+.3f}] → {verdict}")
        data[protocol] = outcome
        lines.append("")
    return lines, data


def domain_sections(results: dict, seqs: list[Sequence]) -> tuple[list[str], dict]:
    lines, data = ["## By domain at 2 m (descriptive)", ""], {}
    header = ["Variant", "Nodes", "Opp.", "TP", "FP", "FN", "P", "R", "F1"]
    for name, belongs in DOMAIN_SUBSETS.items():
        subset = [s for s in seqs if belongs(s)]
        rows, data[name] = loop_rows(results, subset, list(TEST_VARIANTS))
        lines += [f"### {name} ({len(subset)} traversals)", "", *table(header, rows)]
    return lines, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seqs = readable_test()
    results = load(args.out, seqs)
    missing = [(s.name, v) for s in seqs for v in (*TEST_VARIANTS, TEST_REPRODUCTION_VARIANT)
               if (s.name, v) not in results]
    if missing:
        raise SystemExit(f"{len(missing)} test runs are missing, for example {missing[:3]}")
    check_reproduction(results, seqs)
    lines, data = ["# Test results", "", f"Loop-closure rule of {LOOP_RULE_PROTOCOL}.", ""], {}
    for name, (section, payload) in {
        "decisions": decisions(results, seqs),
        "test": split_section(results, seqs, f"Test traversals ({len(seqs)})", list(TEST_VARIANTS), by_route,
                              "routes"),
        "domains": domain_sections(results, seqs),
    }.items():
        lines += section
        data[name] = payload
    (RESULTS / "test_summary.md").write_text("\n".join(lines))
    (RESULTS / "test_summary.json").write_text(json.dumps(data, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
