"""Which test traversals yield a frame stream; writes results/test_inventory.json and prints the readable ones.

Only frames, odometry and ground truth are read: no mapping method is run.
"""

from __future__ import annotations

import json
import sys

from stream import RESULTS, TEST, frames


def main() -> None:
    inventory = {}
    for seq in TEST:
        if not seq.path.exists():
            inventory[seq.name] = {"route": seq.route, "error": "not downloaded"}
            continue
        try:
            inventory[seq.name] = {"route": seq.route, "frames": sum(1 for _ in frames(seq))}
        except (ValueError, OSError) as error:
            inventory[seq.name] = {"route": seq.route, "error": str(error)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_inventory.json").write_text(json.dumps(inventory, indent=1))
    for name, entry in inventory.items():
        print(f"{name:40s} {entry.get('frames', entry.get('error'))}", file=sys.stderr)
    print(" ".join(name for name, entry in inventory.items() if "frames" in entry))


if __name__ == "__main__":
    main()
