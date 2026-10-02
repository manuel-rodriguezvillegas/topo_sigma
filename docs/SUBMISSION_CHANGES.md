# Submission corrections

The revised implementation removes the optional pose solver and its dependency,
launch/configuration options, factor state, diagnostics and dedicated tests.
Sequential and accepted loop edges now describe topology only. Node poses remain
recorded odometry poses. The manuscript paragraph and its unused citation were
removed as well.

## Six corrections

1. **Motion uncertainty.** `path_length_budget_v1` accumulates XY variance as
   `alpha3 * distance + alpha4 * abs(wrapped_rotation)` and angular variance as
   `alpha1 * abs(wrapped_rotation) + alpha2 * distance`. Units of alpha1–4 are
   rad²/rad, rad²/m, m²/m and m²/rad. Subdivision of the same straight motion or
   monotonic turn leaves this budget unchanged, assuming each resolved angular
   increment is smaller than pi. Only alpha3/4 affect the current XY gate;
   alpha1/2 track the angular budget in the adapter stream. It is a heuristic budget, not
   a calibrated pose marginal or a complete SE(2) covariance propagation model.
   Adapters must declare this contract; non-finite, asymmetric or decreasing
   budgets fail explicitly. The 5.991 gate is nominal, not a validated 95%
   coverage guarantee. Coefficients need independent calibration.
2. **Controlled ablations.** Visual-only now uses the selected `visual_model`,
   including vMF. The cosine comparator uses symmetric averages of each view's
   best counterpart, followed by bidirectional sequence averaging. Candidate
   ordering includes both Mahalanobis distance and log determinant. The complete
   weaker-score background remains present when geometry is ambiguous.
3. **Pending-frame memory.** Frames are losslessly spooled into a temporary
   SQLite store with a small page cache. Streaming sums, exact medoid selection
   and streaming spatial moments preserve segment boundaries and representative
   images. Only selected images are loaded when a node closes. Disk usage grows
   with an unfinished segment; graph images and diagnostic traces still grow
   with the run. This is not a constant-memory claim for the whole system.
4. **Reproducibility.** Runs record effective parameters, input-byte hashes,
   source-file hashes, package versions, pinned model commits and checkpoint
   identity. Results are staged, atomically written, and published only after
   the complete artifact set is verified. Existing results require an identical
   fingerprint. Changed runs are archived under `.archive/`; historical results
   are never renamed into invented method baselines. Ablation compatibility
   depends on provenance and evaluation protocol, not report timestamps.
5. **Evaluation and measurements.** Temporal exclusions are removed before
   retrieval top-k; no eligible candidates means a miss. Boundary matching now
   maximizes the match count within tolerance. Place-recognition caching uses
   image/model identity. Compact maps round-trip as `TopoGraph` and include the
   mean direction, view descriptors, poses, budgets, edge types and any semantic
   embeddings. Three 384-D float32 views plus the mean need 6,144 bytes before
   other state. Array payload, compact/full disk size, callback p95/p99,
   encoder-plus-mapper time, observed throughput and process peak RSS have
   separate names. Undefined diagnostic numbers serialize as JSON null.
6. **Language destinations.** The margin is a relative distinctiveness
   diagnostic and does not establish destination presence. Singleton maps are
   indeterminate. The ROS interface returns suggestions first and requires
   confirmation of a candidate from the current query before publishing a
   target. Semantic embedding caches retain and validate model identity.

The bundled ROS players also join evaluation labels by frame key, wait for all
required connections, ignore stale acknowledgements, and fail on acknowledgement
or startup timeouts. Unreadable COLD images fail immediately. Ground truth remains
outside online mapping decisions. Callback tests use transport stubs; real DDS
integration still needs the devcontainer run.

## Performance improvements 1–3

- Frozen descriptors are shared across gate and detector experiments through a
  cache keyed by decoded image bytes, model/checkpoint, preprocessing and runtime.
- Geometry is evaluated first. Geometric-only searches and geometric rejection
  do no visual scoring; unique full-method candidates compute just their logged
  score. Ambiguous searches retain the original background population.
- Normalized immutable views and per-node vMF normalizers are cached. Candidate
  geometry is vectorized; one regularized solve serves distance and likelihood.

## Validation and limits

Native Python 3.12 validation passed all 83 regression tests and Ruff. DINOv2 ViT-S,
AnyLoc-GeM ViT-S and the pinned CLIP model ran successfully with cached weights.
The evaluation CLI produced valid JSON including language metrics. The edited
manuscript source compiled successfully; its numerical content remains a draft.

All 149 saved full-method nodes gave the same accepted/rejected associations and
proposed destinations as the original implementation when keeping their original
budgets. A 1,000-node no-candidate search measured approximately 207 ms before
and 0.63 ms after the geometry-first changes (native diagnostic, five-run median;
not a container benchmark). A 1,500-frame constant-scene test at 320×240 kept
peak RSS near 56.6 MB, versus 345.6 MB of logical pending-image payload.

A separate diagnostic replay recomputed budgets at the saved representative
frames: aggregate TP/FP/FN changed from **23/1/25 to 25/2/23** at 2 m. This reuses
previous visual nodes and is not a replacement mapping experiment. The extra
false closure shows that rate consistency is not equivalent to better precision.
Do not tune these coefficients against the same seven evaluation sequences.

Docker's daemon was unavailable on the review host. A clean Jazzy build, DDS
execution, fresh numerical tables and final manuscript layout remain to be
validated in the devcontainer. Historical output was preserved; the manuscript
explicitly marks its previous numerical results as pending regeneration.

## Devcontainer command

Rebuild the devcontainer to apply the updated dependency lock (including the
pinned test dependency), then run:

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws
bash verify_submission.sh
```

This builds all six packages, runs all tests, and regenerates the seven main
maps, gate and cosine ablations, AnyLoc comparison, node sensitivity, tolerance
sweeps and language benchmark under `output/submission/`. It resumes only verified
matching runs. The label-free fixed-prominence baseline reads its fresh
calibration from `node_segmentation_experiment.json`. Update manuscript tables,
figures and comparative conclusions from these artifacts before removing its
draft validation notice. The figure generator now reads the new result root and
controlled cosine directories rather than historical archives.

Use `bash verify_submission.sh --test-only` for build and tests. `DATA_ROOT`,
`CID_SIMS_ROOT`, `OUTPUT_ROOT` and `DESCRIPTOR_CACHE_DIR` override data/cache paths.
`FORCE=1` rebuilds runs. `SKIP_EXISTING` no longer bypasses validation.
The full script uses an upper replay rate of 100 Hz with one-frame acknowledgement
flow control. This changes wall-clock pacing, not the recorded image selection
or native odometry integration; override `PLAYER_RATE_HZ` to change it.

For fresh encoder timing, explicitly disable descriptor caching and force a run:

```bash
DESCRIPTOR_CACHE_DIR='' FORCE=1 PLAYER_RATE_HZ=100 ./run_experiments.sh freiburg_a
```

Compare cache-miss timings with cache-miss timings. Callback latency excludes DDS
queue waiting; observed throughput spans first processed frame through final
flush and graph serialization. Peak RSS includes model loading and is process-wide.
