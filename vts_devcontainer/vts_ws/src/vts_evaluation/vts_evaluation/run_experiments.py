"""Run and resume experiments using content manifests and verified completion."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from importlib.metadata import version
from pathlib import Path

import yaml

from vts_core.artifacts import atomic_json, completed_run, file_digest, identity
from vts_evaluation.summarize_experiments import ENVIRONMENTS


def source_identity(workspace: Path) -> dict[str, str]:
    paths = [
        path
        for path in (workspace / "src").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".xml", ".cfg"}
    ]
    paths += [workspace.parent / name for name in ("pyproject.toml", "uv.lock")]
    return {str(path.relative_to(workspace.parent)): file_digest(path) for path in sorted(paths)}


def input_identity(player: dict, cid: bool) -> dict:
    """Hash the actual input bytes once per sequence per suite invocation."""
    root = Path(player["dataset_root"]) / player["sequences"][0]
    if cid:
        images = root / player["color_subdir"]
        paths = [
            *images.glob("*.png"),
            root / player["ground_truth_file"],
            root / player["odometry_file"],
        ]
    else:
        images = root / player["images_subdir"]
        paths = [path for path in images.iterdir() if path.is_file()]
        paths += [root / player["labels_subpath"], root / player["odometry_subpath"]]
    if len(paths) <= 2:
        raise RuntimeError(f"No input images under {images}")
    return {
        "root": str(root.resolve()),
        "files": {str(path.relative_to(root)): file_digest(path) for path in sorted(paths)},
    }


def mapping_parameters(config: dict) -> dict:
    params = copy.deepcopy(config)
    params.pop("language", None)
    for key in ("output_dir", "run_name", "descriptor_cache_dir", "run_fingerprint"):
        params["mapping"].pop(key, None)
    return params


def map_run(config: dict, directory: Path, manifest: dict, force: bool = False) -> None:
    # Compare the same JSON representation used by persisted manifests and
    # completion markers. Processor metadata can contain tuples (mean/std),
    # which JSON represents as lists without changing the encoder identity.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    fingerprint = identity(manifest)
    if directory.is_symlink():
        raise ValueError("result directory must not be a symlink")
    try:
        manifest_matches = json.loads((directory / "run_manifest.json").read_text()) == {
            "fingerprint": fingerprint,
            **manifest,
        }
    except (OSError, ValueError):
        manifest_matches = False
    if not force and manifest_matches and completed_run(directory, fingerprint):
        print(f"Reusing verified run: {directory.name}", flush=True)
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Stage every artifact together. A failed launch cannot expose a previous graph.
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as staging:
        stage = Path(staging)
        effective = copy.deepcopy(config)
        effective["mapping"]["output_dir"] = str(stage.resolve())
        effective["mapping"]["run_fingerprint"] = fingerprint
        config_path = stage / "effective_config.yaml"
        config_path.write_text(yaml.safe_dump(effective, sort_keys=True))
        atomic_json(stage / "run_manifest.json", {"fingerprint": fingerprint, **manifest})
        print(f"Mapping {directory.name}", flush=True)
        subprocess.run(
            [
                "ros2",
                "launch",
                "vts_bringup",
                "pipeline.launch.py",
                f"config:={config_path.resolve()}",
                "mode:=building",
            ],
            check=True,
        )
        marker = completed_run(stage, fingerprint)
        if marker is None:
            raise RuntimeError(
                f"Incomplete mapping run: {directory.name}; previous results preserved"
            )
        if marker["encoder_identity"] != manifest["encoder"]:
            raise RuntimeError("loaded encoder differs from requested run manifest")
        if directory.exists():
            archive = directory.parent / ".archive"
            archive.mkdir(exist_ok=True)
            directory.rename(archive / f"{directory.name}-{uuid.uuid4().hex}")
        stage.rename(directory)


def module(name: str, *args: object, output: Path | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"vts_evaluation.{name}", *map(str, args)],
        check=True,
        stdout=subprocess.PIPE if output else None,
    )
    if output:
        # Parse before publishing: logs or interrupted JSON never become a report.
        atomic_json(output, json.loads(result.stdout))


def evaluate(config: dict, directory: Path, cid: bool, main: bool) -> None:
    player, mapping = config["player"], config["mapping"]
    sequence = Path(player["dataset_root"]) / player["sequences"][0]
    gt = sequence / (player["ground_truth_file"] if cid else player["images_subdir"])
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    args = [
        "--graph",
        directory / "final_graph.pkl",
        "--node-gt",
        directory / "graph_0_node_gt.json",
        "--gt-trajectory",
        gt,
    ]
    if main:
        args += ["--viz-path", figures / "topological_map.pdf"]
        if directory.name == "freiburg_a":
            args += ["--legend-loc", "upper right", "--legend-ncol", 2]
        if not cid:
            module(
                "make_queries",
                "--labels-file",
                sequence / player["labels_subpath"],
                "--out",
                directory / "queries.json",
            )
            args += ["--queries", directory / "queries.json"]
    else:
        args += ["--no-viz"]
    module("evaluate_run", *args, output=directory / "metrics_report.json")
    if main:
        module(
            "plot_lambda2",
            "--lambda2",
            directory / "graph_0_lambda2.npy",
            "--valley-mode",
            mapping["valley_mode"],
            "--valley-k",
            mapping["valley_k"],
            "--valley-delta",
            mapping["valley_delta"],
            "--skip",
            mapping["window_size"],
            "--out",
            figures / "lambda2_valleys.pdf",
        )
        odom_args = ["--odom", sequence / player["odometry_file"]] if cid else []
        module(
            "plot_odometry",
            "--gt-trajectory",
            gt,
            *odom_args,
            "--out",
            figures / "recorded_odometry.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("cold", "cid-sims"))
    parser.add_argument("environments", nargs="*")
    args = parser.parse_args()
    workspace = Path.cwd()
    if not (workspace / "src/vts_bringup").is_dir():
        raise SystemExit(
            "Run this command from vts_devcontainer/vts_ws after sourcing install/setup.bash"
        )
    cid = args.suite == "cid-sims"
    environments = [name for name in ENVIRONMENTS if name.startswith("cid_sims") == cid]
    selected = [
        name
        for name in environments
        if not args.environments
        or name in args.environments
        or name.removeprefix("cid_sims_") in args.environments
    ]
    if not selected:
        raise SystemExit("No matching environments")
    suffix = os.environ.get("VARIANT_SUFFIX", "")
    if suffix and (
        not suffix.startswith("_")
        or any(
            c not in "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
            for c in suffix
        )
    ):
        raise SystemExit(
            "VARIANT_SUFFIX must start with '_' and contain only letters, digits, '.', '-' or '_'"
        )
    overrides = {
        key.lower(): os.environ[key]
        for key in (
            "FEATURE_BACKEND",
            "VISUAL_MODEL",
            "DINO_MODEL",
            "DINO_LAYER",
            "DINO_DEVICE",
            "VALLEY_MODE",
            "VALLEY_K",
            "VALLEY_DELTA",
        )
        if os.environ.get(key)
    }
    if (
        any(
            key in overrides
            for key in (
                "valley_mode",
                "valley_k",
                "valley_delta",
                "feature_backend",
                "visual_model",
                "dino_model",
                "dino_layer",
            )
        )
        and not suffix
    ):
        raise SystemExit(
            "Model/detector overrides require VARIANT_SUFFIX to keep the main method separate"
        )
    root = Path(os.environ.get("OUTPUT_ROOT", "output/submission")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents competing suites from replacing each other's artifacts.
    import fcntl

    with (root / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = source_identity(workspace)
        runtime = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("torch", "torchvision", "transformers", "numpy", "scipy", "pillow")
            },
        }
        for environment in selected:
            config_name = environment if cid else f"cold_{environment}"
            config = yaml.safe_load(
                (workspace / f"src/vts_bringup/config/{config_name}.yaml").read_text()
            )
            config["player"]["dataset_root"] = os.environ.get(
                "CID_SIMS_ROOT" if cid else "DATA_ROOT", config["player"]["dataset_root"]
            )
            config["player"]["dataset_root"] = str(Path(config["player"]["dataset_root"]).resolve())
            if os.environ.get("PLAYER_RATE_HZ"):
                config["player"]["publish_rate_hz"] = float(os.environ["PLAYER_RATE_HZ"])
            mapping = config["mapping"]
            for key, value in {
                "feature_backend": "dino_cls",
                "visual_model": "cosine",
                "sequence_policy": "self_only",
                "require_visual_verification": False,
                "recovery_mode": 'off',
                "recovery_top_k": 5,
                "recovery_horizon": 2,
                "recovery_residual_m": 0.75,
                "recovery_min_span_m": 1.0,
                "recovery_min_turn_deg": 15.0,
                "recovery_visual_z": 2.0,
                "geometric_gate_scale": 1.0,
                "verification_threshold": 0.0,
                "defer_nodes": 2,
                "fixed_concentration": 384.0,
                "valley_mode": "adaptive",
                "valley_delta": 0.1,
                "dino_layer": -1,
                "dino_device": "auto",
            }.items():
                mapping.setdefault(key, value)
            mapping.update(overrides)
            for key in ("valley_k", "valley_delta"):
                mapping[key] = float(mapping[key])
            mapping["dino_layer"] = int(mapping["dino_layer"])
            cache_dir = os.environ.get("DESCRIPTOR_CACHE_DIR", "output/cache/descriptors")
            mapping["descriptor_cache_dir"] = str(Path(cache_dir).resolve()) if cache_dir else ""
            # Resolve weights before fingerprinting so first runs and cache hits share identity.
            from vts_core.features import build_extractor

            extractor = build_extractor(
                mapping["dino_model"],
                device=mapping["dino_device"],
                backend=mapping["feature_backend"],
                layer=mapping["dino_layer"],
            )
            encoder = extractor.model_identity
            del extractor
            gc.collect()
            print(f"Hashing inputs for {environment}", flush=True)
            inputs = input_identity(config["player"], cid)
            modes = (
                ("both", "visual", "geometric", "threshold")
                if os.environ.get("RUN_ABLATIONS") == "1"
                else ("both",)
            )
            for mode in modes:
                mapping["gate_mode"] = mode
                name = environment + (f"_{mode}" if mode != "both" else "") + suffix
                mapping["run_name"] = name
                manifest = {
                    "schema": 1,
                    "sources": sources,
                    "runtime": runtime,
                    "encoder": encoder,
                    "inputs": inputs,
                    "parameters": mapping_parameters(config),
                }
                directory = root / name
                map_run(config, directory, manifest, os.environ.get("FORCE") == "1")
                evaluate(config, directory, cid, mode == "both")
        if not suffix:
            module("summarize_experiments", "--root", root)


if __name__ == "__main__":
    main()
