import numpy as np
from conftest import make_place

from vts_core.multisession import AnchorTracker, distance_sequence, fit_se2, similarity_matrix


def se2(theta, t):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.asarray(t, float)


def test_fit_se2_recovers_transform(rng):
    rotation, translation = se2(0.7, [3.0, -2.0])
    source = rng.normal(size=(6, 2)) * 4
    r, t = fit_se2(source, source @ rotation.T + translation)
    assert np.allclose(r, rotation) and np.allclose(t, translation)


def sessions(rng, theta=1.1, shift=(8.0, -3.0), n=30, noise=0.1):
    """Map A along a corridor loop; B traverses the same places in another frame."""
    path = np.column_stack([np.linspace(0, 20, n), 3 * np.sin(np.linspace(0, 3, n))])
    rotation, translation = se2(theta, shift)
    a = [make_place([*xy, 0.0], np.eye(n)[k], budget=0.01 * k, start=k) for k, xy in enumerate(path)]
    b_xy = (path - translation) @ rotation  # inverse transform: A = R B + t
    b = [make_place([*(xy + rng.normal(0, noise, 2)), 0.0], np.eye(n)[k], budget=0.01 * k, start=k)
         for k, xy in enumerate(b_xy)]
    return a, b, rotation, translation


def test_anchor_found_with_true_candidates_and_distractors(rng):
    a, b, rotation, translation = sessions(rng)
    tracker = AnchorTracker(b, a)
    anchor = None
    for i in range(len(b)):
        distractor = int(rng.integers(len(a)))
        anchor = tracker.observe(i, [(distractor, float(rng.normal(-0.5, 1.0))), (i, 2.0)])
        if anchor:
            break
    assert anchor is not None and i < 10
    assert np.allclose(anchor.rotation, rotation, atol=0.05)
    assert np.allclose(anchor.translation, translation, atol=0.5)


def test_no_anchor_from_uninformative_candidates():
    """Random candidates with background appearance never anchor (a corridor alias trap)."""
    accepted = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        a, b, *_ = sessions(rng)
        tracker = AnchorTracker(b, a)
        for i in range(len(b)):
            tracker.observe(i, [(int(j), float(rng.normal(-0.5, 1.0))) for j in rng.choice(len(a), 2, replace=False)])
        accepted += tracker.anchor is not None
    assert accepted == 0


def test_similarity_kinds_agree_for_single_views(rng):
    q = [make_place([0, 0, 0], rng.normal(size=6)) for _ in range(4)]
    r = [make_place([0, 0, 0], rng.normal(size=6)) for _ in range(5)]
    assert np.allclose(similarity_matrix(q, r, "mean_cosine"), similarity_matrix(q, r, "multiview_cosine"))


def test_distance_sequence_prefers_consistent_history():
    n = 10
    distance = np.arange(n, dtype=float)
    scores = np.full((n, n), 0.1)
    np.fill_diagonal(scores, 0.9)
    scores[5, 2] = 0.95  # an alias stronger than the true match for query 5
    seq = distance_sequence(scores, distance, distance, length_m=3.0, step_m=1.0, jump_m=0.4)
    assert seq[5].argmax() == 5
    reverse = distance_sequence(scores[:, ::-1], distance, distance[::-1].copy() * 0 + distance, 3.0, 1.0, 0.4)
    assert reverse.shape == scores.shape


def test_cross_session_associator_is_causal_and_anchors(rng):
    from vts_core.multisession import AppearanceModel, CrossConfig, CrossSessionAssociator

    a, b, rotation, translation = sessions(rng, n=30)
    # Appearance: the true place scores high, with random aliases.
    for place in b:
        place.mean[:] = place.mean + 0.3 * rng.normal(size=place.mean.size)
        place.mean[:] /= np.linalg.norm(place.mean)
    model = AppearanceModel((0.9, 0.1), (0.1, 0.1))
    full = CrossSessionAssociator(a, model, CrossConfig())
    decisions = [full.observe(p) for p in b]
    prefix = CrossSessionAssociator(a, model, CrossConfig())
    assert [vars(prefix.observe(p)) for p in b[:12]] == [vars(d) for d in decisions[:12]]
    assert full.tracker.anchor is not None and full.anchor_query < 15
    assert np.allclose(full.tracker.anchor.rotation, rotation, atol=0.1)
    assert sum(d.matched == d.query for d in decisions) >= 25
    assert all(0.0 <= d.posterior <= 1.0 and d.posterior + d.new <= 1 + 1e-12 for d in decisions)
