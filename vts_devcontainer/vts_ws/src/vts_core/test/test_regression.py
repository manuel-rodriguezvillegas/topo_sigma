"""Equivalence with the recorded manuscript-v1 runs (git tag ``manuscript-v1``).

The fixture holds the tagged code's segments, gate decisions (141 places,
38 accepted revisits), geometry-only decisions, loop counts and posterior
records on the seven C1 sessions; ``fixtures/make_manuscript_v1_c1.py``
regenerates it from a checkout of the tag. Needs the development session
stores (``toposigma frames import-legacy``); skipped otherwise.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from vts_core import evaluation as ev
from vts_core.association import (
    CalibrationPrior,
    GateAssociator,
    PosteriorAssociator,
    PosteriorConfig,
    run,
)
from vts_core.places import summarize
from vts_core.provenance import load_config
from vts_core.segmentation import segment
from vts_core.session import open_session, store_directory

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "manuscript_v1_c1.json").read_text())
CONFIG = load_config()
VARIANTS = {"posterior": PosteriorConfig(), "posterior+multiview": PosteriorConfig(appearance="multiview_cosine"),
            "posterior_view": PosteriorConfig(view_mixture=True),
            "posterior_view_prior": PosteriorConfig(view_mixture=True, use_prior=True)}


def stored(name):
    if not (store_directory(CONFIG, name) / "manifest.json").is_file():
        pytest.skip(f"session store {name} not available")
    session = open_session(CONFIG, name)
    descriptors = session.descriptors()
    poses, budgets = session.odometry
    segments = segment(descriptors)
    return session, segments, summarize(segments, descriptors, poses, budgets)


@pytest.mark.parametrize("name", sorted(FIXTURE["sessions"]))
def test_segments_and_gate_decisions_are_identical(name):
    expected = FIXTURE["sessions"][name]
    session, segments, places = stored(name)
    assert [list(s) for s in segments] == expected["segments"]
    assert [p.medoid for p in places] == expected["medoids"]
    for method, associator in (("gate", GateAssociator()), ("geometry", GateAssociator(use_appearance=False))):
        got = [[d.query, d.matched, d.accepted, d.score, d.reason] for d in run(associator, places)]
        assert got == expected[method]
        gt = ev.place_gt(places, session.gt)[:, :2]
        for tolerance, counts in expected[method + "_counts"].items():
            result = ev.loop_counts(run(GateAssociator(use_appearance=method == "gate"), places), gt, float(tolerance))
            assert {k: result[k] for k in ("tp", "fp", "fn")} == counts


@pytest.mark.parametrize("name", sorted(FIXTURE["sessions"]))
def test_posterior_records_match(name):
    expected = FIXTURE["sessions"][name]
    _, _, places = stored(name)
    p = expected["prior"]
    prior = CalibrationPrior(tuple(p["view"]), tuple(p["diff"]), p["confirmed"], p["shared"])
    for variant, config in VARIANTS.items():
        got = run(PosteriorAssociator(config, prior), places)
        want = expected[variant]
        assert [d.matched for d in got] == [r[1] for r in want]
        assert np.allclose([d.posterior for d in got], [r[2] for r in want], rtol=0, atol=1e-12)


def test_totals():
    sessions = FIXTURE["sessions"].values()
    assert sum(len(s["medoids"]) for s in sessions) == 141
    assert sum(d[2] for s in sessions for d in s["gate"]) == 38
