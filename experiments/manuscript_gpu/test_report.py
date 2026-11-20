"""Results of the single test evaluation and the decisions of the approved protocol.

Writes results/test_summary.{md,json}. Intervals resample the routes of the readable test traversals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapper_runs import TEST_VARIANTS
from report import by_route, load, loop_rows, split_section, table, unit_counts
from stream import RESULTS, Sequence, readable_test
from summarize import bootstrap

METHOD = "topo"
PREVIOUS = "previous_system"
ANYLOC = ("anyloc_vlad_g", "anyloc_gem_g")
NON_INFERIORITY_MARGIN = 0.10
F1 = 2
DOMAIN_SUBSETS = {
    "COLD": lambda s: s.dataset == "cold",
    "CID-SIMS": lambda s: s.dataset == "cid",
    "COLD Ljubljana": lambda s: "ljubljana" in s.recording,
}


def decisions(results: dict, seqs: list[Sequence]) -> tuple[list[str], dict]:
    per_route = unit_counts(results, seqs, [METHOD, PREVIOUS, *ANYLOC], by_route)
    primary = bootstrap(per_route, METHOD, PREVIOUS, F1)
    primary["confirmed"] = primary["ci95"][0] > 0.0
    data = {"routes": len(per_route), "primary": primary, "non_inferiority": {}}
    lines = ["## Decisions of the protocol", "",
             f"Routes resampled: {len(per_route)}. Traversals: {len(seqs)}.", "",
             f"- Primary, F1 {METHOD} − {PREVIOUS}: {primary['point']:+.3f} "
             f"[{primary['ci95'][0]:+.3f}; {primary['ci95'][1]:+.3f}] → "
             f"{'CONFIRMED' if primary['confirmed'] else 'NOT CONFIRMED'}"]
    for comparator in ANYLOC:
        result = bootstrap(per_route, METHOD, comparator, F1)
        if primary["confirmed"]:
            result["confirmed"] = result["ci95"][0] > -NON_INFERIORITY_MARGIN
            verdict = "CONFIRMED" if result["confirmed"] else "NOT CONFIRMED"
        else:
            result["confirmed"] = None
            verdict = "not tested (primary not confirmed)"
        data["non_inferiority"][comparator] = result
        lines.append(f"- Non-inferiority (margin {NON_INFERIORITY_MARGIN}), F1 {METHOD} − {comparator}: "
                     f"{result['point']:+.3f} [{result['ci95'][0]:+.3f}; {result['ci95'][1]:+.3f}] → {verdict}")
    return lines + [""], data


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
    missing = [(s.name, v) for s in seqs for v in TEST_VARIANTS if (s.name, v) not in results]
    if missing:
        raise SystemExit(f"{len(missing)} test runs are missing, for example {missing[:3]}")
    lines, data = ["# Test results", ""], {}
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
