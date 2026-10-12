import numpy as np

from vts_core.segmentation import Segmenter, lambda2_series, segment


def scenes(rng, lengths=(80, 60, 90, 70, 100), dim=32, noise=0.15):
    frames = []
    for length in lengths:
        centre = rng.normal(size=dim)
        block = centre + noise * rng.normal(size=(length, dim)) * np.linalg.norm(centre) / np.sqrt(dim)
        frames.append(block)
    frames = np.concatenate(frames)
    return (frames / np.linalg.norm(frames, axis=1, keepdims=True)).astype(np.float32)


def streamed(frames, window=30, k=2.0):
    segmenter = Segmenter(window, k)
    out = [s for s in map(segmenter.push, frames) if s]
    final = segmenter.flush()
    return out + ([final] if final else []), segmenter


def test_offline_and_streaming_are_identical(rng):
    frames = scenes(rng)
    offline = segment(frames)
    online, segmenter = streamed(frames)
    assert offline == online
    assert np.array_equal(lambda2_series(frames, 30), np.asarray(segmenter.lambda2))


def test_segments_partition_the_stream(rng):
    frames = scenes(rng)
    segments = segment(frames)
    assert segments[0][0] == 0 and segments[-1][1] == len(frames)
    assert all(a[1] == b[0] and a[0] < a[1] for a, b in zip(segments, segments[1:]))
    assert len(segments) >= 3


def test_causality_of_completed_segments(rng):
    frames = scenes(rng)
    full = segment(frames)
    for cut in (150, 230, 333):
        prefix, _ = streamed(frames[:cut])
        closed = prefix[:-1]  # the last one is the end-of-stream flush
        assert full[: len(closed)] == closed


def test_short_streams():
    single = np.ones((1, 4), dtype=np.float32) / 2
    assert segment(single) == [(0, 1)]
    assert lambda2_series(single, 30).size == 0
