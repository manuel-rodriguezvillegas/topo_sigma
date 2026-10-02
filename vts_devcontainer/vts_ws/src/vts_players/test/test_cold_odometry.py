import tempfile
import unittest
from pathlib import Path

import numpy as np

from vts_players.cold_odometry import load_cold_odometry


class ColdOdometryTest(unittest.TestCase):
    def test_parse_and_angle_aware_interpolation(self) -> None:
        rows = (
            "0 0 0 10 0 2 0 6 1.0 2.0 0 3.10 0 0\n"
            "0 0 0 11 0 2 0 6 3.0 4.0 0 -3.10 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            odometry = load_cold_odometry(str(path))

        pose, covariance = odometry.at(10.5)
        np.testing.assert_allclose(pose[:2], [2.0, 3.0])
        self.assertAlmostEqual(abs(pose[2]), np.pi, places=2)
        self.assertEqual(covariance.shape, (3, 3))

    def test_large_interpolation_gap_is_rejected(self) -> None:
        rows = (
            "0 0 0 10 0 2 0 6 0 0 0 0 0 0\n"
            "0 0 0 12 0 2 0 6 1 0 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            odometry = load_cold_odometry(str(path))
        with self.assertRaises(ValueError):
            odometry.at(11.0, max_gap_s=1.0)

    def test_microsecond_wrap_without_second_increment_is_repaired(self) -> None:
        # Pattern found in COLD Ljubljana logs: "10 896000" followed by "10 0".
        rows = (
            "0 0 0 10 792000 2 0 6 0 0 0 0 0 0\n"
            "0 0 0 10 896000 2 0 6 1 0 0 0 0 0\n"
            "0 0 0 10 0 2 0 6 2 0 0 0 0 0\n"
            "0 0 0 11 104000 2 0 6 3 0 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            odometry = load_cold_odometry(str(path))
        np.testing.assert_allclose(odometry.timestamps, [10.792, 10.896, 11.0, 11.104])

    def test_genuinely_decreasing_seconds_still_fail(self) -> None:
        rows = (
            "0 0 0 11 0 2 0 6 0 0 0 0 0 0\n"
            "0 0 0 10 500000 2 0 6 1 0 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cold_odometry(str(path))


if __name__ == "__main__":
    unittest.main()
