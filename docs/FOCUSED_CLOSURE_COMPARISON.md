# Focused closure comparison (v2)

Run inside the existing development container:

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws
bash ./compare_closures.sh
```

The default replays five appearance scorers across four decision policies in all
seven completed DINOv3 environments. These are inexpensive association replays,
not image inference, ROS playback, segmentation, language evaluation or the full
experiment suite. All 140 replays took about 0.82 s of association computation in
the September 17 Docker screening (file I/O and evaluation excluded).

Source manifests, encoder identities and graph hashes are checked. Legacy cosine
and vMF replay must reproduce the corresponding recorded graph edges exactly.
Ground truth, room labels, source loop edges and future nodes do not enter mapping.

## Independent policies

| Profile | Sequence evidence | Mandatory visual verification |
|---|---|---|
| `legacy` | Original behavior, including self-pairs | No |
| `self_only` | Skip identical observation pairs; retain other context | No |
| `verification_only` | Original sequence | Yes |
| `both` | Skip identical observation pairs | Yes |

The mapper/config defaults are `sequence_policy: self_only` and
`require_visual_verification: false`. This removes invalid self-pair evidence
without treating all overlapping windows as invalid. It is a correctness choice,
not a demonstrated performance improvement. `legacy` remains a reproducibility
control. The earlier overbroad policy is available as mapper option `disjoint`
for diagnosis but is excluded from the default screening.

Strict verification requires relative visual distinctiveness and a configurable
multi-view cosine floor (default 0, strict inequality). Rejected hypotheses can
obtain independent historical continuation support within two subsequent nodes;
otherwise they expire. This remains experimental and is off by default.
The existing unique-geometric-candidate acceptance is preserved when strict
verification is off; such decisions do not test the appearance model.

## Appearance comparisons

- Multi-view cosine (reference).
- Mean-direction cosine.
- Existing vMF overlap evidence.
- Negative symmetric KL with fixed concentration (default 384).
- Negative symmetric KL with fitted per-node concentration.

Symmetric KL means half the sum of forward/reverse KL. With common concentration,
`D_sym = kappa * A_d(kappa) * (1 - cosine)`. Fixed KL therefore controls numerical
consistency; it is not an independent improvement over mean cosine. Concentration
here describes within-segment dispersion, not a calibrated probability of a
correct match. Strict verification refuses zero-concentration distribution pairs.

Select a smaller screening, if needed:

```bash
bash ./compare_closures.sh --profiles self_only --output output/closure_self_only
```

## Results and limits

`output/closure_comparison_v2/summary.json` contains decisions, confirmation delays,
association runtimes and closure metrics at 0.5, 1, 2 and 3 metres. Fingerprinted
protocol and replayed graphs are saved alongside it. Old outputs are preserved;
use another `--output` directory if the protocol changes.

September 17, all seven environments, 2 m micro-aggregate:

| Policy / scorer | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Recorded cosine | 34 | 4 | 18 | 0.756 |
| Self-only / cosine | 33 | 5 | 19 | 0.733 |
| Self-only / mean cosine | 27 | 6 | 25 | 0.635 |
| Self-only / vMF overlap | 29 | 5 | 23 | 0.674 |
| Self-only / fixed KL | 27 | 6 | 25 | 0.635 |
| Self-only / estimated KL | 29 | 4 | 23 | 0.682 |
| Both / cosine | 25 | 4 | 27 | 0.617 |
| Both / estimated KL | 15 | 3 | 37 | 0.429 |

Estimated KL beats the fitted-vMF overlap and mean-cosine controls on F1 in the
self-only policy, but loses to multi-view cosine. It also loses to multi-view
cosine at every evaluated tolerance. Mandatory verification reduces recall.
There is no basis for a general performance claim or an expensive full rerun.

This study freezes segmentation and odometry and measures association behavior
only. It does not measure end-to-end runtime or generalization. Further parameter
selection requires separate validation data and should not optimize these test
sequences. No paper text has been changed.
