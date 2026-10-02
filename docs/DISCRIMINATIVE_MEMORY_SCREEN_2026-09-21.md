# Local image evidence and discriminative memory: exploratory screen

Date: 2026-09-21. **Neither proposed component beats the saved 2 m baseline.**
The production mapper and its configuration were not changed. Observation
selection (proposal 3) was not attempted because proposals 1 and 2 did not
succeed.

## Protocol

Seven existing DINOv3 Small runs, 141 fixed nodes and the original 52 node
revisit opportunities. Every variant uses the same node poses, boundaries,
mean descriptors, odometry, chronological candidate pool and evaluation code.
Source graphs and cached image descriptors are checked against the recorded
artifacts. Ground truth is used only after associations are finished. The
`original_baseline` decisions and metrics reproduce `recovery_a` exactly in all
seven environments.

**1. Local image evidence.** For a rejected query, retrieve its five most
similar historical nodes by the existing symmetric multi-view cosine. Match
up to three query and three reference views per candidate using SIFT (1,200
features), mutual Lowe-ratio matches (0.75) and RANSAC homography (3 px).
An image pair needs at least 12 inliers spanning 5% of the image area. A node
wins only if its best pair scores at least 1.5 times its best rival. Existing
accepted decisions are preserved because most currently accepted closures are
correct. This is an additive recovery test.

A second version counts only inlier features whose descriptor matches the
selected candidate more strongly than the corresponding features of the other
retrieved candidates (ratio 0.8). This is genuine feature-level competition,
not just a comparison of final node scores. It still cannot establish a metric
camera displacement: homography support means scene overlap, not proximity
within 2 m.

**2. View memory.** Keep exactly three views per node, always including its
recorded representative. The remaining two are chosen greedily from the
original segment frames. `diverse` maximizes how well chosen descriptors
cover segment descriptors. `discriminative` adds a fixed-weight (0.2) reward
for views dissimilar to historical competitors while still explaining the
current segment. The immediately previous node is excluded from competitors
because it cannot be a loop candidate. Selection sees only previously emitted
nodes and current segment frames. No new DINO inference, GT or later nodes
enter selection. Resulting views are mapped back to verified source images for
local matching. This uses global-descriptor distinctiveness; it does not yet
optimize local patch distinctiveness when storing the views.

The initial memory screen incorrectly included the immediately preceding node
as a competitor. `distinctive_memory_v1` and `distinctive_competitive_v2` are
retained as exploratory traces. `distinctive_competitive_v3` is the corrected
comparison below. The correction followed the mapper's existing temporal
exclusion rule, not a search over thresholds or view budget.

## Results at 2 m

| Memory / verifier | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original / existing decisions | 33 | 5 | 19 | 0.868 | 0.635 | **0.733** |
| Original / plain SIFT | 35 | 15 | 17 | 0.700 | 0.673 | 0.686 |
| Original / rival-aware SIFT | 34 | 13 | 18 | 0.723 | 0.654 | 0.687 |
| Diverse views / existing decisions | 32 | 6 | 20 | 0.842 | 0.615 | 0.711 |
| Diverse views / rival-aware SIFT | 32 | 12 | 20 | 0.727 | 0.615 | 0.667 |
| Discriminative views / existing decisions | 33 | 5 | 19 | 0.868 | 0.635 | **0.733** |
| Discriminative views / rival-aware SIFT | 33 | 9 | 19 | 0.786 | 0.635 | 0.702 |

Rival-aware SIFT on the original memory adds one correct closure at 0.24 m
and eight false ones. False new closures include 2.45, 2.67, 2.85, 3.44,
5.25, 6.30 and 7.04 m examples. Some locations share an actual visible
scene while their representative camera poses lie beyond the evaluation
threshold. Descriptor competition removes two false positives from the
plain-SIFT variant but does not solve the distance problem.

The memory change alone trades decisions between environments: it gains one
correct match in CID apartment 1, loses one in Freiburg A, and removes one
false match in Saarbruecken A while adding one in Freiburg A. The aggregate
tie is not evidence of improvement. Adding rival-aware SIFT to that memory
adds false closures and reduces F1.

A separate diagnostic of the 38 currently accepted closures found that plain
SIFT supported only 10 of 33 correct closures and one of five false ones.
In several COLD sequences it supported no correct closures. Using SIFT as a
mandatory gate would therefore discard many valid existing decisions.
This diagnostic is recorded in `accepted_pair_audit.json` under the final
output directory.

## Interpretation and next decision

The two components were actually tested independently and together at equal
memory. Neither improves the main metric, so there is no success condition
for trying proposal 3. The baseline remains active. These results do not
refute all forms of discriminative memory or scene verification; they show
that the tested DINO descriptor selection plus SIFT homography cannot provide
the **metric proximity evidence** needed by this evaluation.

A stronger next hypothesis would require a relative-distance estimate or a
representation/evaluation explicitly based on shared places instead of just
representative poses within 2 m. That would be a new study with its own
comparison and literature check; it should not be claimed from this run.

## Artifacts and validation

Main result: `vts_devcontainer/vts_ws/output/distinctive_competitive_v3/summary.json`.
Plain-SIFT control: `output/distinctive_local_v1/summary.json`.
All run manifests include fixed parameters, source hashes and study code hashes.

From `vts_devcontainer/vts_ws`, in an environment with the project dependencies:

```bash
PYTHONPATH=src/vts_core:src/vts_evaluation:src/vts_players python3 -m vts_evaluation.visual_verification_study --stage memory --verification competitive --output output/distinctive_competitive_repeat
```

Validation: 132 tests and 3 subtests passed across core, evaluation and mapping
contracts. A synthetic image test checks geometric matching and the rival
comparison; memory tests check its three-view budget and causal selection.
No ROS simulation or production mapper change was needed for this screen.
