"""``toposigma``: one command line for stores, mapping, association and evaluation.

    toposigma frames list
    toposigma frames build SESSION... [--encoder dinov3] [--all-missing]
    toposigma frames import-legacy output/regime_study_v1/frames
    toposigma map SESSION... [--method gate|posterior] [--all]
    toposigma associate --pairs all|A:B ... [--kind mean_cosine] [--sequence]
    toposigma evaluate loops|revisits|cross|stages ...
    toposigma merge MAP QUERY [--tau 0.5]

Every output JSON carries a run record (code hash, parameters, input manifests).
Test sessions are refused unless ``--allow-test`` is given.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vts_core import evaluation as ev
from vts_core.datasets import DataQualityError, catalogue, data_root, read_recording
from vts_core.graph import TopoGraph
from vts_core.mapping import MapperConfig, map_session
from vts_core.multisession import distance_sequence, place_distance, similarity_matrix
from vts_core.odometry import path_length
from vts_core.places import summarize
from vts_core.provenance import atomic_json, load_config, resolve, run_record
from vts_core.segmentation import segment
from vts_core.session import (
    add_descriptors,
    build_frames,
    import_legacy,
    open_session,
    store_directory,
    stored_sessions,
)


def _pool(jobs: int, function, items: list):
    if jobs <= 1 or len(items) <= 1:
        return list(map(function, items))
    # One BLAS thread per worker: nested threading made parallel replays slower than serial.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(function, items))


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def frames_list(config: dict, args) -> None:
    stored = set(stored_sessions(config, role=None))
    for name, recording in catalogue(config).items():
        status = "stored" if name in stored else "-"
        quality = ""
        if args.check and recording.role != "test" and name not in stored:
            try:
                quality = f"{len(read_recording(recording, config).timestamps)} frames"
            except DataQualityError as error:
                quality = f"DEGRADED ({error})"
        encoders = ",".join(open_session(config, name).encoders) if name in stored and recording.role != "test" else ""
        print(f"{name:42s} {recording.environment:22s} {recording.role:11s} {status:6s} {encoders:16s} {quality}")


def frames_build(config: dict, args) -> None:
    from vts_core.descriptors import build_encoder, encode_paths

    names = list(args.sessions)
    if args.all_missing:
        stored = set(stored_sessions(config, role=None))
        names += [n for n, r in catalogue(config).items() if r.role == "development" and n not in stored]
    encoder = None
    for name in names:
        directory = store_directory(config, name)
        if not (directory / "manifest.json").is_file():
            try:
                build_frames(config, name, args.allow_test)
            except DataQualityError as error:
                print(f"{name}: skipped, {error}", flush=True)
                continue
        session = open_session(config, name, args.allow_test)
        if args.encoder in session.encoders:
            print(f"{name}: {args.encoder} already stored", flush=True)
            continue
        encoder = encoder or build_encoder(args.encoder, config["encoders"]["device"])
        root = data_root(config) / config["datasets"][session.recording.dataset]["subdir"]
        descriptors = encode_paths(encoder, [root / p for p in session.image_paths],
                                   config["encoders"]["batch_size"], name)
        add_descriptors(directory, args.encoder, descriptors,
                        {**encoder.identity, "batch_size": config["encoders"]["batch_size"]})
        print(f"{name}: {len(descriptors)} {args.encoder} descriptors stored", flush=True)


def frames_import(config: dict, args) -> None:
    for meta in sorted(Path(args.directory).glob("*.json")):
        if "__" in meta.stem:
            continue
        try:
            print(f"{meta.stem} -> {import_legacy(config, Path(args.directory), meta.stem)}", flush=True)
        except PermissionError as error:
            print(f"{meta.stem}: skipped ({error})", flush=True)


# --------------------------------------------------------------------------- #
# map
# --------------------------------------------------------------------------- #
def _map_one(task: tuple[dict, str, str, bool]) -> dict:
    config, name, encoder, allow_test = task
    session = open_session(config, name, allow_test)
    structure = session.descriptors("dinov3")
    appearance = None if encoder == "dinov3" else session.descriptors(encoder)
    poses, budgets = session.odometry
    mapper = MapperConfig.from_dict(config["mapping"])
    result = map_session(structure, poses, budgets, mapper, appearance)
    graph = TopoGraph()
    graph.add_session(name, result.places, result.decisions)
    out = resolve(config, "output_root") / "maps" / f"{name}__{mapper.method}__{encoder}"
    graph.save(out.with_suffix(".npz"))
    record = run_record("map", {"mapping": config["mapping"], "encoder": encoder},
                        {name: session.manifest["files"]["frames.npz"]})
    atomic_json(out.with_suffix(".json"), {"run": record, "places": len(result.places),
                                           "accepted": sum(d.accepted for d in result.decisions)})
    return {"session": name, "places": len(result.places), "accepted": sum(d.accepted for d in result.decisions)}


def map_command(config: dict, args) -> None:
    if args.method:
        config["mapping"]["method"] = args.method
    names = [n for n in stored_sessions(config) if "dinov3" in open_session(config, n).encoders] if args.all \
        else args.sessions
    for row in _pool(args.jobs, _map_one, [(config, n, args.encoder, args.allow_test) for n in names]):
        print(f"{row['session']}: {row['places']} places, {row['accepted']} accepted revisits", flush=True)


# --------------------------------------------------------------------------- #
# Cross-session pairs
# --------------------------------------------------------------------------- #
def development_pairs(config: dict) -> list[tuple[str, str]]:
    """Every ordered (map, query) pair of stored development sessions sharing an environment."""
    by_env = defaultdict(list)
    for name in stored_sessions(config):
        session = open_session(config, name)
        if "dinov3" in session.encoders:  # stores being extracted have no descriptors yet
            by_env[session.recording.environment].append(name)
    return [pair for names in by_env.values() for pair in itertools.permutations(sorted(names), 2)]


def test_pairs(config: dict) -> list[tuple[str, str]]:
    """Ordered pairs within an environment that involve at least one stored test session."""
    by_env = defaultdict(list)
    for name in stored_sessions(config, role=None):
        session = open_session(config, name, allow_test=True)
        if "dinov3" in session.encoders:
            by_env[session.recording.environment].append((name, session.recording.role))
    return [(a, b) for members in by_env.values() for (a, ra), (b, rb) in itertools.permutations(sorted(members), 2)
            if "test" in (ra, rb)]


def _parse_pairs(config: dict, pairs: list[str]) -> list[tuple[str, str]]:
    if pairs == ["all"]:
        return development_pairs(config)
    if pairs == ["test"]:
        return test_pairs(config)
    return [tuple(p.split(":")) for p in pairs]


def _places(config: dict, name: str, encoder: str, allow_test: bool):
    session = open_session(config, name, allow_test)
    structure = session.descriptors("dinov3")
    appearance = None if encoder == "dinov3" else session.descriptors(encoder)
    poses, budgets = session.odometry
    mapping = config["mapping"]
    places = summarize(segment(structure, mapping["window"], mapping["valley_k"]), structure, poses, budgets,
                       appearance)
    return session, places, place_distance(places, path_length(poses))


def _associate_one(task) -> dict:
    config, (map_name, query_name), encoder, kind, sequence, allow_test = task
    map_session_, map_places, map_distance = _places(config, map_name, encoder, allow_test)
    query_session, query_places, query_distance = _places(config, query_name, encoder, allow_test)
    scores = similarity_matrix(query_places, map_places, kind)
    if sequence:
        scores = distance_sequence(scores, query_distance, map_distance)
    return {"map": map_name, "query": query_name, "scores": scores,
            "map_gt": ev.place_gt(map_places, map_session_.gt), "query_gt": ev.place_gt(query_places, query_session.gt),
            "inputs": {map_name: map_session_.manifest["files"], query_name: query_session.manifest["files"]}}


def associate_command(config: dict, args) -> None:
    pairs = _parse_pairs(config, args.pairs)
    tasks = [(config, pair, args.encoder, args.kind, args.sequence, args.allow_test) for pair in pairs]
    results = _pool(args.jobs, _associate_one, tasks)
    label = f"{args.kind}__{args.encoder}{'__seq' if args.sequence else ''}"
    out = resolve(config, "output_root") / "associate" / label
    out.mkdir(parents=True, exist_ok=True)
    for result in results:
        np.savez_compressed(out / f"{result['map']}__{result['query']}.npz", scores=result["scores"],
                            map_gt=result["map_gt"], query_gt=result["query_gt"])
    record = run_record("associate", {"kind": args.kind, "encoder": args.encoder, "sequence": args.sequence,
                                      "mapping": config["mapping"]},
                        {f"{r['map']}:{r['query']}": r["inputs"] for r in results})
    atomic_json(out / "run.json", {"run": record, "pairs": [f"{r['map']}:{r['query']}" for r in results]})
    print(f"{len(results)} pairs -> {out}")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def evaluate_loops(config: dict, args) -> dict:
    """In-session loop closure metrics of stored map outputs, split by revisit type."""
    report = {}
    for path in sorted((resolve(config, "output_root") / "maps").glob(f"*__{args.method}__{args.encoder}.npz")):
        graph = TopoGraph.load(path)
        (name,) = graph.sessions
        session = open_session(config, name, args.allow_test)
        gt = ev.place_gt(graph.places[name], session.gt)
        report[name] = {}
        for tolerance in config["evaluation"]["tolerances_m"]:
            kinds = ev.revisit_kinds(gt, gt, tolerance, config["evaluation"]["aligned_deg"],
                                     ev.in_session_eligible(len(gt)))
            counts = ev.loop_counts(graph.decisions[name], gt[:, :2], tolerance, kinds)
            report[name][str(tolerance)] = {**counts, **ev.prf(counts["tp"], counts["fp"], counts["fn"])}
    total = {}
    for tolerance in config["evaluation"]["tolerances_m"]:
        summed = defaultdict(int)
        for rows in report.values():
            for key, value in rows[str(tolerance)].items():
                if key in {"tp", "fp", "fn", "tp_aligned", "fn_aligned", "tp_opposed", "fn_opposed"}:
                    summed[key] += value
        total[str(tolerance)] = {**summed, **ev.prf(summed["tp"], summed["fp"], summed["fn"])}
    return {"sessions": report, "total": total}


def evaluate_revisits(config: dict, args) -> dict:
    """Step 3 diagnostic: how many cross-session revisits are aligned or opposed."""
    places = {}
    rows = []
    for map_name, query_name in _parse_pairs(config, args.pairs):
        for name in (map_name, query_name):
            if name not in places:
                session, ps, _ = _places(config, name, "dinov3", args.allow_test)
                places[name] = (session.recording.environment, ev.place_gt(ps, session.gt))
        env, map_gt = places[map_name]
        _, query_gt = places[query_name]
        row = {"map": map_name, "query": query_name, "environment": env, "query_places": len(query_gt),
               "map_places": len(map_gt)}
        for tolerance in config["evaluation"]["tolerances_m"]:
            kinds = ev.revisit_kinds(query_gt, map_gt, tolerance, config["evaluation"]["aligned_deg"])
            row[str(tolerance)] = {"aligned": kinds.count("aligned"), "opposed": kinds.count("opposed"),
                                   "none": kinds.count(None)}
        rows.append(row)
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in rows:
        for tolerance in config["evaluation"]["tolerances_m"]:
            for kind, count in row[str(tolerance)].items():
                summary[row["environment"]][str(tolerance)][kind] += count
                summary["all"][str(tolerance)][kind] += count
    return {"pairs": rows, "summary": summary}


def evaluate_cross(config: dict, args) -> dict:
    """Top-1 retrieval PR of stored association scores, with a bootstrap over pairs."""
    directory = resolve(config, "output_root") / "associate" / args.label
    report, per_pair = {}, defaultdict(list)
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path)
        distance = np.linalg.norm(data["query_gt"][:, None, :2] - data["map_gt"][None, :, :2], axis=2)
        pair = {}
        for tolerance in config["evaluation"]["tolerances_m"]:
            kinds = ev.revisit_kinds(data["query_gt"], data["map_gt"], tolerance, config["evaluation"]["aligned_deg"])
            positive = distance.min(axis=1) <= tolerance
            for statistic in ("raw", "margin"):
                top, value = ev.top1_statistic(data["scores"], statistic)
                correct = distance[np.arange(len(top)), top] <= tolerance
                curve = ev.pr_curve(correct, positive, value)
                aligned = np.asarray([k == "aligned" for k in kinds])
                curve["recall_at_1_aligned"] = float(correct[aligned].mean()) if aligned.any() else None
                opposed = np.asarray([k == "opposed" for k in kinds])
                curve["recall_at_1_opposed"] = float(correct[opposed].mean()) if opposed.any() else None
                curve["chance_top1"] = ev.chance_top1(distance, tolerance)
                pair[f"{statistic}|{tolerance}"] = curve
                per_pair[f"{statistic}|{tolerance}"].append(curve["average_precision"])
        report[path.stem] = pair
    summary = {key: dict(zip(("mean_ap", "ci_low", "ci_high"),
                             ev.bootstrap_interval(values, resamples=config["evaluation"]["bootstrap"])))
               for key, values in per_pair.items()}
    return {"label": args.label, "pairs": report, "summary": summary}


def evaluate_stages(config: dict, args) -> dict:
    """Plan B step 4: E0 baseline, E1 calibrated appearance, E3 with anchored geometry."""
    from vts_core import stages
    from vts_core.multisession import CrossConfig

    data = stages.pair_data(config, _parse_pairs(config, args.pairs),
                            lambda name: _places(config, name, "dinov3", args.allow_test)[:2])
    frozen = stages.load_model(args.model) if args.calibration == "frozen" else None
    out = {}
    for tolerance in config["evaluation"]["tolerances_m"]:
        if args.jobs > 1:
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                os.environ[variable] = "1"
            with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                result = stages.study(data, CrossConfig(), tolerance, config["evaluation"]["aligned_deg"],
                                      args.calibration, pool.map, config["evaluation"]["bootstrap"], frozen)
        else:
            result = stages.study(data, CrossConfig(), tolerance, config["evaluation"]["aligned_deg"],
                                  args.calibration, map, config["evaluation"]["bootstrap"], frozen)
        out[str(tolerance)] = result
    return {"calibration": args.calibration, "summary": {t: r["summary"] for t, r in out.items()},
            "pairs": {t: r["pairs"] for t, r in out.items()}}


def merge_command(config: dict, args) -> None:
    """Associate a query session with a map session online and save the merged graph."""
    from vts_core import stages
    from vts_core.multisession import CrossSessionAssociator

    map_session_, map_places, _ = _places(config, args.map, "dinov3", args.allow_test)
    query_session, query_places, _ = _places(config, args.query, "dinov3", args.allow_test)
    model = None
    if args.calibration == "supervised":
        env = map_session_.recording.environment
        others = [p for p in development_pairs(config) if open_session(config, p[0]).recording.environment != env]
        data = stages.pair_data(config, others, lambda name: _places(config, name, "dinov3", False)[:2])
        model = stages.supervised_model(stages.supervised_samples(data, "mean_cosine"), env)
    associator = CrossSessionAssociator(map_places, model)
    decisions = [associator.observe(p) for p in query_places]
    graph = TopoGraph()
    mapper = MapperConfig.from_dict(config["mapping"])
    for name, session, places in ((args.map, map_session_, map_places), (args.query, query_session, query_places)):
        graph.add_session(name, places, map_session(session.descriptors("dinov3"), *session.odometry, mapper).decisions)
    merged = [d for d in decisions if d.posterior >= args.tau]
    for d in merged:
        graph.add_edge((args.map, d.matched), (args.query, d.query), "merge")
    out = resolve(config, "output_root") / "merge" / f"{args.map}__{args.query}"
    graph.save(out.with_suffix(".npz"))
    anchor = associator.tracker.anchor
    atomic_json(out.with_suffix(".json"), {
        "run": run_record("merge", {"tau": args.tau, "calibration": args.calibration}, {}),
        "decisions": [vars(d) for d in decisions], "anchor_query": associator.anchor_query,
        "transform": None if anchor is None else {"rotation": anchor.rotation, "translation": anchor.translation}})
    print(f"{len(merged)}/{len(decisions)} query places merged; anchor at query {associator.anchor_query} -> {out}.npz")


def evaluate_command(config: dict, args) -> None:
    result = {"loops": evaluate_loops, "revisits": evaluate_revisits, "cross": evaluate_cross,
              "stages": evaluate_stages}[args.report](config, args)
    record = run_record(f"evaluate {args.report}", {k: v for k, v in vars(args).items() if k != "func"}, {})
    suffix = {"cross": "_" + args.label, "stages": "_" + args.calibration}.get(args.report, "") + args.tag
    out = resolve(config, "output_root") / "evaluate" / f"{args.report}{suffix}.json"
    atomic_json(out, {"run": record, **result})
    summary = result.get("total") or result.get("summary")
    print(json.dumps(summary, indent=1, default=float)[:4000])
    print(f"-> {out}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML deep-merged over the defaults")
    common.add_argument("--allow-test", action="store_true", help="permit test sessions (needs a protocol)")
    common.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser = argparse.ArgumentParser(prog="toposigma", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    frames = sub.add_parser("frames").add_subparsers(dest="action", required=True)
    listing = frames.add_parser("list", parents=[common])
    listing.add_argument("--check", action="store_true", help="read raw data to report quality")
    listing.set_defaults(func=frames_list)
    build = frames.add_parser("build", parents=[common])
    build.add_argument("sessions", nargs="*")
    build.add_argument("--all-missing", action="store_true")
    build.add_argument("--encoder", default="dinov3")
    build.set_defaults(func=frames_build)
    legacy = frames.add_parser("import-legacy", parents=[common])
    legacy.add_argument("directory")
    legacy.set_defaults(func=frames_import)

    mapping = sub.add_parser("map", parents=[common])
    mapping.add_argument("sessions", nargs="*")
    mapping.add_argument("--all", action="store_true")
    mapping.add_argument("--method", choices=("gate", "posterior"))
    mapping.add_argument("--encoder", default="dinov3")
    mapping.set_defaults(func=map_command)

    associate = sub.add_parser("associate", parents=[common])
    associate.add_argument("--pairs", nargs="+", default=["all"], help="'all', 'test' or MAP:QUERY ...")
    associate.add_argument("--kind", choices=("mean_cosine", "multiview_cosine"), default="mean_cosine")
    associate.add_argument("--encoder", default="dinov3")
    associate.add_argument("--sequence", action="store_true", help="distance-aligned sequence evidence")
    associate.set_defaults(func=associate_command)

    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("report", choices=("loops", "revisits", "cross", "stages"))
    evaluate.add_argument("--calibration", choices=("supervised", "self", "frozen"), default="supervised",
                          help="stages: leave-one-environment-out GT, self-calibrated, or a frozen --model file")
    evaluate.add_argument("--model", type=Path, help="frozen appearance model JSON (calibration=frozen)")
    evaluate.add_argument("--tag", default="", help="suffix for the output file")
    evaluate.add_argument("--pairs", nargs="+", default=["all"])
    evaluate.add_argument("--method", default="gate")
    evaluate.add_argument("--encoder", default="dinov3")
    evaluate.add_argument("--label", default="mean_cosine__dinov3", help="associate output (cross)")
    evaluate.set_defaults(func=evaluate_command)

    merge = sub.add_parser("merge", parents=[common])
    merge.add_argument("map")
    merge.add_argument("query")
    merge.add_argument("--tau", type=float, default=0.5)
    merge.add_argument("--calibration", choices=("supervised", "self"), default="supervised")
    merge.set_defaults(func=merge_command)

    args = parser.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
