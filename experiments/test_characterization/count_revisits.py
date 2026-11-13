"""Count in-session revisits from ground truth only (no method is run).

Used to characterize candidate test data for plan B before writing its
protocol. Each trajectory is resampled every ``STEP`` metres of arc length (a
proxy for places). A sample is a revisit opportunity if an earlier sample at
least ``MIN_ARC`` metres of travel before it lies within ``RADIUS`` metres. It
is *aligned* if one such sample has a heading within 45 degrees, otherwise
*opposed*. Development sessions are counted with the same definition, for
comparison.

    python count_revisits.py OPENLORIS_PER_SEQUENCE_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "vts_devcontainer" / "vts_ws" / "src" / "vts_core"))

STEP, MIN_ARC, RADIUS, ALIGNED = 1.5, 6.0, 2.0, np.radians(45.0)
SKIP_START_S = 10.0  # OpenLORIS starts with a short back-and-forth push for time sync


def resample(t, xy, yaw, skip_s=0.0):
    keep = t >= t[0] + skip_s
    t, xy, yaw = t[keep], xy[keep], np.unwrap(yaw[keep])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    arc += np.arange(len(arc)) * 1e-9
    s = np.arange(0.0, arc[-1], STEP)
    return s, np.column_stack([np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])]), np.interp(s, arc, yaw)


def count(s, xy, yaw):
    aligned = opposed = 0
    for i in range(len(s)):
        earlier = np.flatnonzero(s <= s[i] - MIN_ARC)
        near = earlier[np.linalg.norm(xy[earlier] - xy[i], axis=1) <= RADIUS]
        if not len(near):
            continue
        dyaw = np.abs((yaw[near] - yaw[i] + np.pi) % (2 * np.pi) - np.pi)
        aligned += bool((dyaw < ALIGNED).any())
        opposed += not (dyaw < ALIGNED).any()
    return aligned, opposed


def yaw_from_quaternion(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def row(name, s, aligned, opposed):
    length = s[-1] if len(s) else 0.0
    return f"| {name} | {length:.0f} | {len(s)} | {aligned} | {opposed} | {100 * opposed / max(length, 1):.1f} |"


def main() -> None:
    lines = ["| Secuencia | Longitud (m) | Muestras | Revisitas alineadas | Revisitas opuestas | Opuestas / 100 m |",
             "|---|---|---|---|---|---|"]
    totals = {}
    for path in sorted(Path(sys.argv[1]).glob("*/groundtruth.txt")):
        data = np.loadtxt(path, comments="#")
        s, xy, yaw = resample(data[:, 0], data[:, 1:3], yaw_from_quaternion(data[:, 4:8]), SKIP_START_S)
        a, o = count(s, xy, yaw)
        scene = path.parent.name.split("1-")[0]
        totals.setdefault(f"openloris_{scene}", [0.0, 0, 0])
        totals[f"openloris_{scene}"] = [totals[f"openloris_{scene}"][0] + (s[-1] if len(s) else 0),
                                        totals[f"openloris_{scene}"][1] + a, totals[f"openloris_{scene}"][2] + o]
        lines.append(row(path.parent.name, s, a, o))

    from vts_core.provenance import load_config
    from vts_core.session import open_session, stored_sessions

    config = load_config()
    for name in stored_sessions(config):
        session = open_session(config, name)
        s, xy, yaw = resample(session.timestamps, session.gt[:, :2], session.gt[:, 2])
        a, o = count(s, xy, yaw)
        group = "dev_cold" if name.startswith("cold") else "dev_cid"
        current = totals.setdefault(group, [0.0, 0, 0])
        totals[group] = [current[0] + s[-1], current[1] + a, current[2] + o]
    lines += ["", "| Grupo | Longitud (m) | Alineadas | Opuestas | Opuestas / 100 m |", "|---|---|---|---|---|"]
    for group, (length, a, o) in totals.items():
        lines.append(f"| {group} | {length:.0f} | {a} | {o} | {100 * o / max(length, 1):.1f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
