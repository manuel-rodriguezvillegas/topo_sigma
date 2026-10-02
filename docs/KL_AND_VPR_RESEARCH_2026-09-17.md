# KL experiment and research directions

Research checked September 17, 2026. This is a design discussion, not manuscript
text. Benefits below are hypotheses for this indoor system; published benchmark
rankings do not establish performance on our data.

## What the experiment establishes

See [the focused study](FOCUSED_CLOSURE_COMPARISON.md) for protocol and results.
The inexpensive replay separates self-pair removal from mandatory verification.
On seven environments at 2 m, corrected multi-view cosine yields F1 0.733;
estimated-concentration KL yields 0.682, existing vMF overlap 0.674 and
mean cosine/fixed KL 0.635. Mandatory verification remains detrimental.

This supports investigating distribution summaries but does not support replacing
multi-view cosine. A single distribution may compress useful view-dependent
structure. Also, low concentration may indicate several legitimate viewpoints,
rather than unreliable imagery. These are hypotheses, not diagnosed causes yet.

## Closest prior work

[KappaPlace (May 2026 preprint)](https://arxiv.org/abs/2605.19435) represents
VPR descriptors with vMF uncertainty and learns a concentration head using
prototype supervision. Its matching uncertainty uses the resultant of natural
parameter vectors, rather than our symmetric KL. Our concentration comes from
within-segment dispersion; theirs is learned. This is close prior art, so neither
vMF place representations nor concentration-based confidence alone is a credible
novelty claim.

[Kitagawa and Rowley (2022)](https://arxiv.org/abs/2202.05192) derive analytical
vMF divergences. The implemented comparison follows this established family:

`D_sym(p,q) = 0.5 * (theta_p - theta_q) dot (eta_p - eta_q)`

where `theta = kappa * mu` and `eta = A_d(kappa) * mu` are natural and expectation
parameters. This is half the Jeffreys divergence. Fixed concentration reduces it
to a scaled cosine distance, explaining the identical decisions in our control.

## Ranked ideas

Costs are relative engineering estimates, not measured completion times.

### 1. Compare better match-confidence signals (low cost)

[Through the Lens of Doubt (2025 preprint; 2026 journal record)](https://arxiv.org/abs/2510.13464)
introduces training-free measures based on similarity-score separation and
competition among candidates. These can be evaluated on existing descriptors.
[SUE, CVPR 2024](https://arxiv.org/abs/2404.00546) additionally uses reference-map
poses when estimating match uncertainty and provides useful simple baselines.

Proposal: compare score separation/ratio-spread with our MAD test, log confidence
before making it a hard gate, and test recall at fixed precision. Use a fixed
candidate background: do not select only weaker competitors based on the winner.
For SUE-inspired spatial clustering use estimated map coordinates, never GT;
odometry drift means its pose assumptions require care here. Calibrate any
probability conversion on separate validation sequences. Confidence need not be
a hard veto: it can decide when to gather more evidence.

### 2. Weight temporal evidence and retain several hypotheses (low–medium cost)

[Carson, Ford and Milford (2023)](https://arxiv.org/abs/2307.01464) combine
training-free quality prediction with weighted sequence matching. This is an
established relevant method, not a claim about the newest leaderboard winner.

Proposal: use frame- or keyframe-level context inside a node, variable-speed
alignment, both traversal directions, and a bounded beam of candidate histories.
Retain a new-place hypothesis. Weight independent observations by quality and
exclude self-pairs. Our fixed one-node-per-step continuation can fail when return
traversals have different segmentation. Initially use saved view descriptors;
full temporal alignment may require caching more per-frame descriptors.

### 3. Preserve multiple views in the probabilistic representation (medium cost)

Inspired by the vMF literature above: compare a small mixture or view-conditioned
set of distributions against the single fitted vMF. This is our proposed extension,
not something established by KappaPlace. Separately test shrinkage of unreliable
concentration estimates toward a shared prior; strongly correlated video frames
should not be counted as independent evidence.

A mixture could retain room-facing and corridor-facing modes. However, the closed
form for a single-vMF KL does not become a mixture-KL formula: use justified bounds,
sampling or an explicitly different set-matching objective. Current compact nodes
retain representative features, not enough statistics for every possible mixture;
training a learned confidence head would require additional labeled data.

### 4. Verify ambiguous pairs using local image geometry (medium cost)

[LightGlue, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)
provides adaptive sparse feature matching. [FoL, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32811)
focuses retrieval and re-ranking on discriminative image regions.

Proposal: retain/cache representative images and local features; run a matcher
and robust geometric verification only for a few ambiguous candidates. Log
inlier ratio, spatial coverage and degeneracy. This tests actual shared visual
structure, unlike merely requiring a larger global cosine score. Lack of overlap
must remain inconclusive rather than universally proving different places.

[To Match or Not to Match, CVPR workshop 2025](https://arxiv.org/abs/2504.06116)
finds that re-ranking can hurt already strong retrieval. That supports testing
selective verification and motivates measuring both precision and lost recall.
Compact graph files omit images, so this needs an additional cache pass. Monocular
correspondences alone do not generally provide metric translation scale.

### 5. Test a VPR-trained descriptor/aggregator (medium cost; inference required)

[SALAD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.html)
uses optimal transport aggregation with a fine-tuned DINOv2 backbone.
[BoQ, CVPR 2024](https://arxiv.org/abs/2405.07364) learns global queries for feature
aggregation. These are strong task-specific references, not guaranteed winners
for indoor short loops.

Proposal: extract alternative descriptors for exactly the same frame segments,
keeping segmentation and policy fixed. Use a published checkpoint with its intended
backbone; replacing DINOv2 with DINOv3 in a trained aggregator is not automatically
compatible. Benchmark extraction and matching memory/time on the target CPU before
considering a full suite. FoL is a more recent candidate if regional evidence proves
useful. No need to change language/CLIP embeddings for this comparison.

### 6. Allocate verification effort by expected information gain (medium–high cost)

[Information-based Compact Pose SLAM, T-RO 2010](https://www.iri.upc.edu/publications/show/1096)
selects informative links and nonredundant poses. This is established precedent,
not novel merely because it involves an information matrix.

Proposal: spend local-matching compute where it is expected to reduce association
uncertainty most, relative to cost. After introducing valid relative-pose factors
and joint covariance, a separate pose-information objective can rank useful
closures. A Gaussian log-determinant reduction requires an anchored, observable
state and appropriate measurement covariance. The mapper's cumulative odometry
budgets are not joint posterior covariances and cannot be substituted into that
calculation. Adding an optimizer without measured constraints would not repair it.

## A defensible information-geometric contribution

KL compares normalized distributions, accounting for changes in direction and
concentration. Its local second-order form is the Fisher information metric; this
connection is established information geometry, not novelty by itself. See
[Nielsen's introduction (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7650632/).

Single-vMF comparisons require O(d) vector work plus cached scalar functions;
multi-view matching can require O(v_a * v_b * d). That is a possible association
cost advantage, not evidence of end-to-end acceleration: inference may dominate.
And information theory (entropy reduction), information matrices in SLAM and
Fisher–Rao geometry are related concepts, not interchangeable justifications.

A research hypothesis worth pursuing is: use calibrated directional uncertainty
to allocate a fixed verification budget while maintaining closure precision and
recovering short loops. Compare to cosine margins, the recent training-free
confidence baselines and a fixed-budget verifier. Measure precision/recall, false
closures per trajectory, confirmation latency, verifier calls, memory and wall
time on held-out environments. Keep segmentation, descriptors and candidates
controlled. Only a demonstrable benefit and a fuller novelty search would justify
presenting this as the paper's contribution.

## Recommendation

First compare confidence signals and weighted temporal evidence using saved
features. Keep KL as an inexpensive experimental scorer. If confidence identifies
specific ambiguous cases, add selective local verification. Test a VPR-specific
encoder as a separate performance baseline. Defer mixtures, learned concentration
and pose-information scheduling until simpler comparisons identify the bottleneck.
