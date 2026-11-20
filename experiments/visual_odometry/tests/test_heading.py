import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from diagnostic import IMAGE_SIZE, visual_yaw  # noqa: E402


def _rotation_about_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _project(points: np.ndarray, focal: float) -> np.ndarray:
    return focal * points[:, :2] / points[:, 2:] + np.array(IMAGE_SIZE) / 2


@pytest.mark.parametrize("yaw_deg", [-12.0, -3.0, 4.0, 15.0])
def test_visual_yaw_recovers_a_planar_turn(yaw_deg: float) -> None:
    rng = np.random.default_rng(0)
    focal = 520.0
    points = np.column_stack([rng.uniform(-4, 4, 300), rng.uniform(-1.5, 1.0, 300), rng.uniform(3, 12, 300)])
    camera_rotation = _rotation_about_y(np.radians(yaw_deg))
    camera_centre = np.array([0.05, 0.0, 0.3])
    second = (points - camera_centre) @ camera_rotation
    visible = (second[:, 2] > 0.5)
    first_px, second_px = _project(points[visible], focal), _project(second[visible], focal)
    noise = rng.normal(0.0, 0.3, first_px.shape)

    yaw, inliers = visual_yaw(first_px, second_px + noise, focal)

    assert inliers > 0.8 * visible.sum()
    assert np.degrees(yaw) == pytest.approx(-yaw_deg, abs=0.2)
