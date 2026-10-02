import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from vts_core.artifacts import atomic_json, file_digest, identity
from vts_evaluation.node_segmentation import _match_boundaries
from vts_evaluation.run_experiments import map_run, source_identity
from vts_evaluation.summarize_experiments import compatible_reports


def test_boundary_matching_maximizes_count():
    assert _match_boundaries([2, 5], [0, 3], 2) == 2
    assert _match_boundaries([], [0, 3], 2) == 0


def test_changes_outside_mapper_invalidate_source_identity(tmp_path):
    workspace = tmp_path / "vts_ws"
    core = workspace / "src/vts_core/vts_core"
    core.mkdir(parents=True)
    for path in (tmp_path / "pyproject.toml", tmp_path / "uv.lock", core / "directional.py"):
        path.write_text("original")
    first = identity(source_identity(workspace))
    (core / "directional.py").write_text("corrected")
    assert identity(source_identity(workspace)) != first


def test_launch_success_without_completion_cannot_replace_old_map(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "final_graph.pkl").write_bytes(b"previous result")
    with patch("vts_evaluation.run_experiments.subprocess.run"):
        with pytest.raises(RuntimeError, match="Incomplete"):
            map_run({"mapping": {}}, destination, {"schema": 1})
    assert (destination / "final_graph.pkl").read_bytes() == b"previous result"


def test_gate_compatibility_uses_identity_not_report_mtime(tmp_path):
    manifest = {
        "fingerprint": "unused",
        "sources": {"directional": "hash"},
        "parameters": {"mapping": {"gate_mode": "both", "visual_model": "vmf"}},
    }
    for name, mode in (("main", "both"), ("visual", "visual")):
        directory = tmp_path / name
        directory.mkdir()
        graph = directory / "final_graph.pkl"
        graph.write_bytes(name.encode())
        variant = copy.deepcopy(manifest)
        variant.pop("fingerprint")
        variant["parameters"]["mapping"]["gate_mode"] = mode
        fingerprint = identity(variant)
        atomic_json(
            directory / "completion.json",
            {
                "fingerprint": fingerprint,
                "frames": 1,
                "artifacts": {"final_graph.pkl": file_digest(graph)},
            },
        )
        atomic_json(
            directory / "metrics_report.json",
            {"provenance": {"run_fingerprint": fingerprint, "graph_sha256": file_digest(graph)}},
        )
        atomic_json(directory / "run_manifest.json", {"fingerprint": fingerprint, **variant})
    assert compatible_reports(tmp_path / "main", tmp_path / "visual")
    variant["parameters"]["mapping"]["visual_model"] = "cosine"
    atomic_json(tmp_path / "visual/run_manifest.json", variant)
    assert not compatible_reports(tmp_path / "main", tmp_path / "visual")


@pytest.mark.parametrize("mean", [[0.485, 0.456, 0.406], (0.485, 0.456, 0.406)])
def test_completed_run_is_published_with_original_manifest(tmp_path, mean):
    destination = tmp_path / "run"
    manifest = {"schema": 1, "encoder": {"model": "fixture", "processor": {"image_mean": mean}}}
    fingerprint = identity(manifest)

    def launch(command, **kwargs):
        config_path = Path(
            next(arg.removeprefix("config:=") for arg in command if arg.startswith("config:="))
        )
        directory = config_path.parent
        artifact = directory / "final_graph.pkl"
        artifact.write_bytes(b"complete")
        atomic_json(
            directory / "completion.json",
            {
                "fingerprint": fingerprint,
                "frames": 1,
                "encoder_identity": manifest["encoder"],
                "artifacts": {"final_graph.pkl": file_digest(artifact)},
            },
        )

    with patch("vts_evaluation.run_experiments.subprocess.run", side_effect=launch) as run:
        map_run({"mapping": {}}, destination, manifest)
        map_run({"mapping": {}}, destination, manifest)
    assert run.call_count == 1
    assert (destination / "run_manifest.json").exists()


def test_different_encoder_is_still_rejected(tmp_path):
    destination = tmp_path / "run"
    manifest = {"encoder": {"model": "requested", "processor": {"image_mean": (0.5,)}}}

    def launch(command, **kwargs):
        config_path = Path(
            next(arg.removeprefix("config:=") for arg in command if arg.startswith("config:="))
        )
        artifact = config_path.parent / "final_graph.pkl"
        artifact.write_bytes(b"complete")
        atomic_json(
            config_path.parent / "completion.json",
            {
                "fingerprint": identity(manifest),
                "frames": 1,
                "encoder_identity": {"model": "different", "processor": {"image_mean": [0.5]}},
                "artifacts": {"final_graph.pkl": file_digest(artifact)},
            },
        )

    with patch("vts_evaluation.run_experiments.subprocess.run", side_effect=launch):
        with pytest.raises(RuntimeError, match="loaded encoder differs"):
            map_run({"mapping": {}}, destination, manifest)
    assert not destination.exists()
