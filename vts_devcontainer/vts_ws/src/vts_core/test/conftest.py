import numpy as np
import pytest

from vts_core.places import Place


def make_place(pose, mean, budget=0.0, views=None, start=0) -> Place:
    mean = np.asarray(mean, dtype=np.float64)
    mean = mean / np.linalg.norm(mean)
    views = np.atleast_2d(mean if views is None else views)
    views = views / np.linalg.norm(views, axis=1, keepdims=True)
    pose = np.asarray(pose, dtype=np.float64)
    return Place(start, start + 1, start, tuple(range(start, start + len(views))), mean, views, 1.0, pose,
                 np.eye(2) * budget, np.repeat(pose[None], len(views), axis=0))


@pytest.fixture
def rng():
    return np.random.default_rng(7)
