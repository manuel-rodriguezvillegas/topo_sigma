Manuscript update audit — verification run of 5–6 September 2026

Compared the current manuscript source with `output/submission/`, using `output/revised/` to check historical mapping results and `output/cosine_2026-08-20_final/language_benchmark_100.json` for historical language results. No manuscript or result figures were edited. Line references refer to the source as inspected on 6 September.

Evidence: all 77 run completion manifests and every listed artifact passed SHA-256 verification; evaluation reports match their run fingerprints and graph hashes. All runs share one source snapshot, which matches the current workspace. All 77 cases are present in the four-radius sweep, and their 2 m counts match the individual evaluation reports. The seven primary runs processed 13,196 frames. The final language report is present. These are artifact checks, not a new execution of ROS or pytest; a persisted pytest result was not used to assert a test count.

Aggregate scores below are micro-averages recomputed from TP/FP/FN, not averages of rounded per-sequence scores. Three-decimal presentation can introduce rounding ties; use one rounding convention consistently in the final tables.

1. Rewrite the headline interpretation ([manuscript.tex:40](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:40), [manuscript.tex:463](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:463), [manuscript.tex:484](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:484), [manuscript.tex:521](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:521), [manuscript.tex:595](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:595), [manuscript.tex:601](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:601)).

The claim that TopoSIGMA beats AnyLoc at every radius is false after the corrections. It wins at 0.5 and 2 m, and loses at 1 and 3 m. The 2 m result improves from F1 0.639 to 0.667, but precision falls from 0.958 to 0.926 and recall rises from 0.479 to 0.521. The full method is a precision-oriented operating point: geometry-only has higher F1 at all four radii. The evidence still strongly supports geometric restriction versus unrestricted appearance. Avoid describing these deterministic seven-sequence comparisons as statistically significant or universal.

| Tolerance (m) | Old TopoSIGMA F1 | New TopoSIGMA F1 | Old AnyLoc F1 | New AnyLoc F1 | New winner |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.375 | 0.286 | 0.370 | 0.250 | TopoSIGMA |
| 1.0 | 0.511 | 0.320 | 0.476 | 0.436 | AnyLoc |
| 2.0 | 0.639 | 0.667 | 0.433 | 0.575 | TopoSIGMA |
| 3.0 | 0.479 | 0.505 | 0.430 | 0.587 | AnyLoc |

2. Replace the loop-results table (`tab:loops`, [manuscript.tex:364](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:364)–[manuscript.tex:384](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:384)) and update the paragraph introducing it.

149 nodes remain unchanged; accepted loops increase from 24 to 27. At 2 and 3 m there are now 25 true edges out of 27, not 23 out of 24. Freiburg A loses one true closure and gains one false closure at 2 m; Apartments 1 and 2 gain two and one true closures respectively. Apartment 3 changes edge endpoints despite unchanged 2 m counts.

| Sequence | Nodes | Loops | F1 @0.5 m | F1 @1 m | F1 @2 m | F1 @3 m |
| --- | --- | --- | --- | --- | --- | --- |
| Freiburg A | 30 | 3 | 0.400 | 0.333 | 0.200 | 0.125 |
| Freiburg Extended | 22 | 3 | 0.800 | 0.571 | 0.600 | 0.500 |
| Saarbrücken A | 32 | 3 | 0.000 | 0.286 | 0.500 | 0.375 |
| Saarbrücken Extended | 16 | 2 | 0.000 | 0.667 | 1.000 | 0.500 |
| Apartment 1 | 18 | 8 | 0.364 | 0.308 | 0.941 | 0.800 |
| Apartment 2 | 11 | 2 | 0.000 | 0.667 | 0.571 | 0.500 |
| Apartment 3 | 20 | 6 | 0.000 | 0.000 | 0.800 | 0.632 |
| Aggregate | 149 | 27 | 0.286 | 0.320 | 0.667 | 0.505 |

3. Replace both AnyLoc tables (`tab:baseline`, [manuscript.tex:467](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:467), and `tab:tolerance_baseline`, [manuscript.tex:502](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:502)), including all boldface winner markings.

| Radius | COLD TopoSIGMA | COLD AnyLoc | CID-SIMS TopoSIGMA | CID-SIMS AnyLoc | All TopoSIGMA | All AnyLoc |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.400 | 0.133 | 0.200 | 0.320 | 0.286 | 0.250 |
| 1 | 0.435 | 0.261 | 0.222 | 0.562 | 0.320 | 0.436 |
| 2 | 0.500 | 0.276 | 0.821 | 0.773 | 0.667 | 0.575 |
| 3 | 0.346 | 0.400 | 0.681 | 0.731 | 0.505 | 0.587 |

| Radius | TopoSIGMA P | TopoSIGMA R | TopoSIGMA F1 | AnyLoc P | AnyLoc R | AnyLoc F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.185 | 0.625 | 0.286 | 0.156 | 0.625 | 0.250 |
| 1 | 0.296 | 0.348 | 0.320 | 0.375 | 0.522 | 0.436 |
| 2 | 0.926 | 0.521 | 0.667 | 0.656 | 0.512 | 0.575 |
| 3 | 0.926 | 0.347 | 0.505 | 0.844 | 0.450 | 0.587 |

COLD favors TopoSIGMA through 2 m and AnyLoc at 3 m. CID-SIMS favors TopoSIGMA only at 2 m. At 0.5 m recall ties and TopoSIGMA has higher precision; remove the opposite precision claim and the old 0.005 F1-gap statement. At 1 m AnyLoc has higher precision and recall; at 3 m TopoSIGMA has higher precision but lower recall. The AnyLoc node detector has the same settings but different descriptors and node boundaries: at 2 m the opportunity counts are 48 for TopoSIGMA and 41 for AnyLoc. Present this as a mapper-level comparison, not a paired classifier test on identical opportunities.

4. Replace the gate-ablation table and interpretation (`tab:ablation`, [manuscript.tex:525](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:525)–[manuscript.tex:548](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:548)). Explicitly state the 2 m evaluation radius in its caption.

| Gate | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Geometry + vMF | 25 | 2 | 23 | 0.926 | 0.521 | 0.667 |
| Geometry + cosine | 22 | 4 | 26 | 0.846 | 0.458 | 0.595 |
| Geometry only | 29 | 6 | 19 | 0.829 | 0.604 | 0.699 |
| Fixed threshold | 9 | 28 | 39 | 0.243 | 0.188 | 0.212 |
| Visual only (vMF) | 6 | 57 | 42 | 0.095 | 0.125 | 0.108 |

Old → new counts: full 23/1/25 → 25/2/23; cosine 20/1/28 → 22/4/26; geometry-only 24/5/24 → 29/6/19; visual-only 9/66/39 → 6/57/42. Fixed threshold remains 9/28/39. Geometry-only now has the highest TP, recall and F1; full vMF has the fewest FP and highest precision. Relative to geometry-only, vMF gives up four TPs and removes four FPs: P 0.829 → 0.926, R 0.604 → 0.521, F1 0.699 → 0.667. Replace “one more true closure” and “five false shortcuts” with four and six. Replace 66 visual-only false closures with 57. Do not attribute that visual-only numerical change solely to motion uncertainty: the old visual-only comparator was also corrected to use vMF.

5. Rewrite the vMF-versus-cosine explanation and caption ([manuscript.tex:489](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:489), [manuscript.tex:519](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:519), [manuscript.tex:548](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:548)).

| Sequence | vMF F1 @2 m | Cosine F1 @2 m |
| --- | --- | --- |
| Freiburg A | 0.200 | 0.333 |
| Freiburg Extended | 0.600 | 0.600 |
| Saarbrücken A | 0.500 | 0.462 |
| Saarbrücken Extended | 1.000 | 1.000 |
| Apartment 1 | 0.941 | 0.800 |
| Apartment 2 | 0.571 | 0.571 |
| Apartment 3 | 0.800 | 0.615 |

The graphs differ in four sequences: Freiburg A, Saarbrücken A, Apartment 1 and Apartment 3. They agree in Freiburg Extended, Saarbrücken Extended and Apartment 2. vMF is worse on Freiburg A and better on the other three differing sequences. Apartment 1 is now 0.800 → 0.941, with two additional TPs (6 → 8), not 0.500 → 0.800 with three. Aggregate vMF improves TP by three and reduces FP by two versus cosine, raising F1 0.595 → 0.667. At 0.5 and 1 m cosine has higher aggregate F1, so qualify the gain by radius. The controlled comparator uses symmetric multi-view cosine, not one representative descriptor; remove claims that its failures demonstrate information hidden by “one representative descriptor.” Model dispersion may explain specific outcomes, but the new aggregate numbers alone do not prove that causal explanation; re-inspect candidate scores before retaining it.

6. Keep the node-segmentation table but rewrite its end-to-end sensitivity discussion ([manuscript.tex:332](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:332)–[manuscript.tex:360](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:360), [manuscript.tex:593](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:593)).

The diagnostic summary and displayed table values are reproduced: 217/183/149/119/99 nodes for k=1/1.5/2/2.5/3; 158 for fixed prominence; the CV, transition recall, purity and fragmentation figures also agree. Fresh fixed calibration is δ=0.1301148376011585, still 0.130 when displayed, matching 29 Freiburg A boundaries and 30 nodes. These quantities need refreshed provenance rather than invented numerical changes. The segmentation JSON replays saved λ2 signals; the separately completed sensitivity maps provide end-to-end loop results.

| Detector | F1 @0.5 m | F1 @1 m | F1 @2 m | F1 @3 m |
| --- | --- | --- | --- | --- |
| k=1 | 0.333 | 0.400 | 0.571 | 0.506 |
| k=1.5 | 0.314 | 0.421 | 0.577 | 0.485 |
| k=2 | 0.286 | 0.320 | 0.667 | 0.505 |
| k=2.5 | 0.133 | 0.222 | 0.526 | 0.339 |
| k=3 | 0.167 | 0.154 | 0.400 | 0.261 |
| Fixed | 0.056 | 0.275 | 0.543 | 0.442 |

k=1 leads at two radii (0.5 and 3 m), not three; k=1.5 leads at 1 m and k=2 leads at 2 m. The 3 m advantage of k=1 over k=2 is only about 0.001. k=2 uses 149 rather than 217 nodes, but describe it as a compactness/2 m trade-off rather than uniformly close to the best score. Adaptive k=2 still beats fixed prominence in P, R and F1 at every radius; its F1 advantage is now 0.045–0.230, not 0.079–0.188. The claim about repeating the gate/AnyLoc comparisons at k=1 is historical: verify_submission does not generate those crossed variants. Remove it or rerun that study with the revised implementation. Likewise, the claim that recalibrating fixed prominence to k=1 produces 301 nodes is not supported by this fresh artifact set; remove it or repeat that calibration. Avoid selecting or retuning k against these same evaluation results while implying independent validation.

7. Update three path-distortion cells in `tab:mapquality` ([manuscript.tex:433](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:433)–[manuscript.tex:452](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:452)).

| Sequence | Old D_path | New D_path |
| --- | --- | --- |
| Freiburg A | 0.649 | 0.713 |
| Apartment 2 | 0.547 | 0.431 |
| Apartment 3 | 0.471 | 0.463 |

All seven representative-frame ground-truth files are identical to the historical set. Coverage at the four fixed radii and node-placement RMSE remain unchanged at manuscript precision. The other path-distortion cells remain unchanged. The coverage/RMSE ranges and Saarbrücken A distortion discussion can remain. Do not use the generic `graph.coverage` field for C_map@0.5 m: that field uses an automatically chosen radius. Use `coverage_at_tolerance` from the sweep.

8. Refresh the empirical figures and their interpretation ([manuscript.tex:325](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:325), [manuscript.tex:389](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:389), [manuscript.tex:488](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:488), [manuscript.tex:495](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:495), [manuscript.tex:564](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:564)).

Required changed assets: `images/map_freiburg_a.pdf`, `map_apartment_1.pdf`, `map_apartment_2.pdf`, `map_apartment_3.pdf`, `vmf_cosine_comparison.pdf`, and `baseline_comparison.pdf`. The maps of Freiburg Extended, Saarbrücken A and Saarbrücken Extended have unchanged loop-edge sets; regenerate or verify all seven against the fresh outputs for consistent provenance. Freiburg A replaces loop (22,26) with (12,18), now has two false loops at 2 and 3 m, and cannot be described only through the old single through-wall example. Recheck wall-crossing statements against the fresh plot: metric FP status alone does not prove a wall crossing. Apartment 1 changes several endpoints as well as gaining two loops; Apartment 3 replaces (7,15) with (8,14). Apartment 2 adds (0,9). Node IDs here are the stored zero-based graph IDs.

The Freiburg A λ2 array is byte-identical to the old one; 29 valleys and 30 nodes remain valid. The language metrics reproduce the old values, but verify the four pictured requests and view choices against the fresh map before keeping the example montage. Conceptual diagrams and synthetic vMF illustrations do not require numerical regeneration. `report/generate_result_figures.py` already reads submission results and the controlled cosine directories, but only generates the two comparison charts; it does not refresh map, λ2 or language-example PDFs. The verification command does not run this manuscript figure generator or compile the manuscript.

9. Replace historical timings and clarify playback versus sampling ([manuscript.tex:278](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:278), [manuscript.tex:280](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:280), [manuscript.tex:463](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:463)).

The manifests confirm Python 3.12.3, Linux aarch64, CPU PyTorch 2.13.0+cpu, NumPy 1.26.4, SciPy 1.17.1, torchvision 0.28.0+cpu and transformers 5.15.1. Retain the hardware description only as the known host setup; the run manifests do not identify the Mac model independently. Record the pinned encoder revision/checkpoint identity in reproducibility material. The selected image stream represents approximately 5 Hz dataset time, while the runner requests 100 Hz playback with one-frame acknowledgement flow control. Do not say the rerun actually processed five frames per wall-clock second.

| Frame-weighted processing mean | TopoSIGMA | AnyLoc adaptation |
| --- | --- | --- |
| Old encoder + mapper (ms/frame) | 72.54 | 76.18 |
| New encoder + mapper (ms/frame) | 128.95 | 126.81 |
| New encoder portion (ms/frame) | 124.93 | 123.21 |
| New mapper portion (ms/frame) | 4.02 | 3.60 |

Both primary configurations had 13,196 cache misses and zero hits, so the new timings include encoder computation. They also include the enabled cache-miss path, rather than a cache-disabled timing experiment. Cosine had 13,196 hits and zero misses; its 8.16 ms/frame mean is not a like-for-like fresh-inference comparison. AnyLoc is slightly faster in this run, so remove any implied TopoSIGMA speed advantage. Do not turn a single sequential run into a causal speed-regression claim; runtime conditions and instrumentation changed.

Observed throughput ranges: TopoSIGMA 1.28–1.80 frames/s; AnyLoc 1.50–1.97. Per-run callback p95 ranges: 148.40–202.65 ms and 150.44–160.68 ms; p99 ranges: 168.24–290.25 ms and 165.56–195.79 ms. A weighted mean of per-run percentiles is not a pooled percentile. Report ranges or retain per-run values. Callback time excludes DDS waiting, and observed throughput includes the measured first-frame-to-final-flush/serialization interval. The present run does not demonstrate end-to-end 5 Hz throughput.

Keep the corrected 6,144-byte visual-array calculation (three views plus separate mean). If adding measured footprint results: primary compact maps are 72,649–210,366 bytes, full maps 30,487,906–88,691,109 bytes, and process peak RSS 635,506,688–673,214,464 bytes (about 606–642 MiB). These compact-map measurements precede semantic embedding enrichment (`semantic_embedding_bytes=0`), so do not describe them as a language-ready map footprint. They are distinct from logical array payload and resident memory. Keep the existing qualified disk-spooling paragraph and the absence of a whole-system constant-memory claim.

10. Keep the language numbers but correct the decision terminology ([manuscript.tex:267](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:267), [manuscript.tex:554](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:554), [manuscript.tex:558](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:558), [manuscript.tex:571](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:571), [manuscript.tex:603](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:603)).

The fresh report reproduces the overall, per-map and category metrics: 400 pairs, 238 present; R@1 0.4370, R@3 0.7395, MRR 0.5658, present selection coverage 0.3445, selected accuracy 0.5732, absent rejection 0.6049, absent false-selection rate 0.3951. Preserve the current table at its existing rounding. Remove “historical” from the reproduced diagnostic and describe C_ret/A_ans/R_rej as margin-heuristic selection/rejection statistics, not measured autonomous navigation safety or confidence of destination presence. The implementation now requires confirmation before publishing a target; there is no confirmation-user-study result here. The future-work sentence about adding confirmation must distinguish the already implemented confirmation from future clarification dialogue and calibrated absence detection.

11. Tighten experimental-protocol and methodology wording ([manuscript.tex:217](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:217), [manuscript.tex:274](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:274), [manuscript.tex:280](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:280), [manuscript.tex:308](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:308), [manuscript.tex:334](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:334)).

The geometric figure caption should say “nominal 95% Gaussian contour,” consistent with the body’s explicit heuristic-budget caveat. Ground truth and labels are attached during replay for evaluation bookkeeping; replace “accessed only after a run has finished” with “used only for evaluation, never for online node-boundary or association decisions.” State the actual temporal exclusion (two most recent nodes) and that opportunities count query events with at least one eligible prior node, not every nearby pair. A wrong accepted edge can count as both FP and a missed valid opportunity (FN). Aggregates pool counts; semantic segmentation metrics are macro-averages over the four COLD sequences. Specify one-to-one maximum-cardinality boundary matching within the 15-frame tolerance. Qualify “same parameters in every experiment” to exclude the explicit detector/gate/backend ablations.

The uncertainty-budget equations, controlled visual-only vMF definition, symmetric multi-view cosine definition, Gaussian candidate ordering, no-pose-optimization behavior, temporary-disk buffer and confirmation-first language interface are already reflected in the source. Preserve those corrections instead of reverting to the old methodology.

12. Remove the draft notice only after the manuscript itself is refreshed and checked ([manuscript.tex:49](/Users/manue/Manuel/tfg/topo_sigma/report/manuscript.tex:49)).

Recompile `report/manuscript.tex`, inspect all affected tables/figures, update the abstract/discussion/conclusion together, and replace the stale `report/manuscript.pdf`. Its existing modification date is 4 September, earlier than both the revised source and the new experiment outputs. Do not equate completed verification outputs with an updated manuscript. If a final submission says tests passed, retain the actual devcontainer build/test log rather than carrying over the native 83-test claim from the earlier correction notes.

Related release documentation also needs synchronization: README result summaries, `docs/VMF_VALIDATION.md`, `docs/ANYLOC_BASELINE.md`, and the “pending regeneration / Docker unavailable” historical status in `docs/SUBMISSION_CHANGES.md`. Keep those statements as dated history or append a clearly dated completion note rather than silently representing the old status as current.

Suggested replacement scientific summary: “Across seven traversals, geometric restriction sharply reduces false loop closures relative to appearance-only decisions. At 2 m, the full vMF method achieves 25 true positives and two false positives (precision 0.926, recall 0.521, F1 0.667). Visual validation reduces false closures relative to geometry-only at a cost in recall and aggregate F1. Comparisons with multi-view cosine and the AnyLoc-inspired descriptor depend on sequence and spatial tolerance. Adaptive node creation retains more consistent graph density than fixed prominence, while the natural-language margin remains an exploratory ranking diagnostic.”
