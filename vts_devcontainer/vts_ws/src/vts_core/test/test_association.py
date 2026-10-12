import numpy as np
import pytest
from conftest import make_place

from vts_core.association import (
    GateAssociator,
    PosteriorAssociator,
    PosteriorConfig,
    SimilarityRows,
    geometric_log_lr,
    log_likelihood_ratio,
    multiview_similarity,
    run,
)
from vts_core.mapping import MapperConfig, OnlineMapper, map_session


def loop_route(rng, laps=2, n=12, dim=16, radius=5.0):
    """Places around a circle, visited ``laps`` times with growing budgets."""
    centres = rng.normal(size=(n, dim))
    places = []
    for k in range(laps * n):
        angle = 2 * np.pi * (k % n) / n
        pose = [radius * np.cos(angle), radius * np.sin(angle), angle + np.pi / 2]
        places.append(make_place(pose, centres[k % n] + 0.2 * rng.normal(size=dim), budget=0.02 * k, start=k))
    return places


@pytest.mark.parametrize("make", [GateAssociator, lambda: GateAssociator(use_appearance=False),
                                  lambda: PosteriorAssociator(PosteriorConfig()),
                                  lambda: PosteriorAssociator(PosteriorConfig(view_mixture=True))])
def test_decisions_are_causal(rng, make):
    places = loop_route(rng)
    full = run(make(), places)
    for cut in (5, 13, 20):
        prefix = run(make(), places[:cut])
        assert [vars(d) for d in prefix] == [vars(d) for d in full[: len(prefix)]]


def test_gate_closes_the_second_lap(rng):
    places = loop_route(rng)
    decisions = run(GateAssociator(), places)
    second_lap = [d for d in decisions if d.query >= 12]
    assert sum(d.accepted and d.matched == d.query - 12 for d in second_lap) >= 8
    assert decisions[0].reason == "no_eligible_candidate"


def test_posterior_is_a_distribution(rng):
    for decision in run(PosteriorAssociator(PosteriorConfig()), loop_route(rng)):
        assert 0.0 <= decision.posterior <= 1.0 and decision.posterior + decision.new <= 1.0 + 1e-12


def test_likelihood_ratios_are_monotone_and_bounded():
    scores = np.linspace(-1, 1, 50)
    lr = log_likelihood_ratio(scores, (0.8, 0.1), (0.3, 0.1))
    assert np.all(np.diff(lr) >= 0)
    assert np.all(log_likelihood_ratio(scores, (0.3, 0.1), (0.8, 0.1)) == 0)  # uninformative
    geo = geometric_log_lr(np.array([0.0, 10.0, 1e9]), area=100.0, eps=0.05)
    assert np.all(geo >= np.log(0.05) - 1e-12) and np.all(np.diff(geo) <= 0)


def test_incremental_similarity_matches_reference(rng):
    places = [make_place([0, 0, 0], rng.normal(size=8), views=rng.normal(size=(1 + k % 3, 8)), start=k)
              for k in range(10)]
    for dtype in (np.float32, np.float64):
        rows = SimilarityRows("multiview_cosine", dtype)
        for i, place in enumerate(places):
            row = rows.add(place)
            reference = [multiview_similarity(rows.features(place), rows.features(places[j])) for j in range(i)]
            # Bit-exactness at the real dimension is covered by test_regression.
            assert np.allclose(row, reference, rtol=0, atol=1e-6)
    assert multiview_similarity(places[1].views, places[2].views) == pytest.approx(
        multiview_similarity(places[2].views, places[1].views))


def test_online_mapper_matches_offline(rng):
    from test_segmentation import scenes

    frames = scenes(rng, lengths=(60, 50, 70, 60, 50, 60))
    poses = np.column_stack([np.linspace(0, 30, len(frames)), np.zeros(len(frames)), np.zeros(len(frames))])
    budgets = np.linspace(0, 1, len(frames))[:, None, None] * np.eye(3)
    offline = map_session(frames, poses, budgets, MapperConfig())
    online = OnlineMapper(MapperConfig())
    for d, p, b in zip(frames, poses, budgets):
        online.push(d, p, b)
    online.finish()
    assert [(p.start, p.stop, p.medoid, p.view_frames) for p in online.places] == \
        [(p.start, p.stop, p.medoid, p.view_frames) for p in offline.places]
    assert [vars(d) for d in online.decisions] == [vars(d) for d in offline.decisions]
