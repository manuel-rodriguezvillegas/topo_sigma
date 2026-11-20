# Manuscript results on the GPU

## Odometry calibration (development ground truth only)

Pairs per traversal: {'freiburg_a': 594, 'freiburg_ext': 309, 'saarbruecken_a': 961, 'saarbruecken_ext': 267, 'cid_sims_apartment1_1': 293, 'cid_sims_apartment2_1': 291, 'cid_sims_apartment3_1': 342, 'heldout_freiburg_a_seq1': 363, 'heldout_freiburg_a_seq2': 594, 'heldout_freiburg_b_seq3': 285, 'heldout_saarbruecken_a_seq1': 249, 'heldout_saarbruecken_a_seq2': 945, 'heldout_saarbruecken_b_seq3': 145, 'heldout_saarbruecken_b_seq4': 261, 'heldout_ljubljana_a_seq1': 1137, 'heldout_ljubljana_a_seq2': 1314, 'heldout_cid_apartment1_2': 370, 'heldout_cid_apartment2_2': 412, 'heldout_cid_apartment3_2': 398}

| Model | α1 | α2 | α3 | α4 | floor [m] | mean NLL |
|---|---|---|---|---|---|---|
| isotropic, original | 0.025 | 0.005 | 0.01 | 0.0025 | 1.336 | – |
| isotropic, calibrated | 0.025 (unused) | 0.005 (unused) | 0.08883 [0.0498; 0.148] | 5.998e-07 [6e-07; 6e-07] | 0.000 [0.000; 0.000] | 2.071 |
| propagated, calibrated | 0.01157 [0.000434; 0.0175] | 0.0004837 [1.52e-08; 0.00586] | 0.07525 [0.00908; 0.182] | 2.925e-08 [2.92e-08; 2.93e-08] | 0.069 [0.019; 0.176] | 1.573 |

Leave-one-traversal-out mean NLL (lower is better):

| Traversal | isotropic_default | isotropic_calibrated | propagated_calibrated |
|---|---|---|---|
| freiburg_a | 4.355 | 2.406 | 1.508 |
| freiburg_ext | 3.396 | 2.079 | 1.249 |
| saarbruecken_a | 4.801 | 2.551 | 1.722 |
| saarbruecken_ext | 2.872 | 1.387 | 0.879 |
| cid_sims_apartment1_1 | 2.630 | 1.323 | 1.411 |
| cid_sims_apartment2_1 | 2.615 | 1.296 | 1.351 |
| cid_sims_apartment3_1 | 2.624 | 1.253 | 1.051 |
| heldout_freiburg_a_seq1 | 13.056 | 9.421 | 20.058 |
| heldout_freiburg_a_seq2 | 4.777 | 2.650 | 1.639 |
| heldout_freiburg_b_seq3 | 3.319 | 1.970 | 1.240 |
| heldout_saarbruecken_a_seq1 | 2.747 | 1.260 | 0.832 |
| heldout_saarbruecken_a_seq2 | 4.982 | 2.594 | 1.698 |
| heldout_saarbruecken_b_seq3 | 2.607 | 0.977 | 0.506 |
| heldout_saarbruecken_b_seq4 | 2.868 | 1.366 | 0.879 |
| heldout_ljubljana_a_seq1 | 2.950 | 1.689 | 1.451 |
| heldout_ljubljana_a_seq2 | 7.000 | 3.389 | 2.074 |
| heldout_cid_apartment1_2 | 2.695 | 1.368 | 1.156 |
| heldout_cid_apartment2_2 | 2.673 | 1.372 | 1.128 |
| heldout_cid_apartment3_2 | 2.778 | 1.573 | 1.713 |
| mean | 3.987 | 2.206 | 2.292 |

Fraction of true drifts inside the 95 % gate (with σ_place), by travelled distance [m]:

| Gate | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 32.0 | 64.0 | 128.0 |
|---|---|---|---|---|---|---|---|---|
| isotropic_default | 1.00 | 0.99 | 0.99 | 0.97 | 0.77 | 0.47 | 0.17 | 0.01 |
| isotropic_calibrated | 1.00 | 0.99 | 0.99 | 0.99 | 0.97 | 0.81 | 0.49 | 0.35 |
| propagated_calibrated | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.88 | 0.94 |

## Node creation sensitivity (development)

Fixed δ calibrated on freiburg_a: 0.0683

| Detector | Nodes | ρ | CV | R_trans | Purity | Seg./episode |
|---|---|---|---|---|---|---|
| dinov2_s/minimum adaptive 1 | 217 | 16.943 | 0.134 | 0.643 | 0.908 | 3.508 |
| dinov2_s/minimum adaptive 1.5 | 183 | 14.265 | 0.134 | 0.580 | 0.895 | 3.267 |
| dinov2_s/minimum adaptive 2 | 149 | 11.464 | 0.138 | 0.467 | 0.859 | 2.943 |
| dinov2_s/minimum adaptive 2.5 | 119 | 9.057 | 0.103 | 0.417 | 0.826 | 2.534 |
| dinov2_s/minimum adaptive 3 | 99 | 7.318 | 0.160 | 0.368 | 0.765 | 2.253 |
| dinov2_s/minimum fixed 0.1301 | 158 | 12.341 | 0.235 | 0.596 | 0.896 | 3.099 |
| dinov2_s/cut adaptive 1 | 217 | 16.943 | 0.134 | 0.379 | 0.891 | 3.508 |
| dinov2_s/cut adaptive 1.5 | 183 | 14.265 | 0.134 | 0.333 | 0.888 | 3.267 |
| dinov2_s/cut adaptive 2 | 149 | 11.464 | 0.138 | 0.317 | 0.869 | 2.943 |
| dinov2_s/cut adaptive 2.5 | 119 | 9.057 | 0.103 | 0.286 | 0.824 | 2.534 |
| dinov2_s/cut adaptive 3 | 99 | 7.318 | 0.160 | 0.254 | 0.770 | 2.253 |
| dinov2_s/cut fixed 0.1301 | 158 | 12.341 | 0.235 | 0.333 | 0.883 | 3.099 |
| dinov3_s/minimum adaptive 1 | 217 | 17.363 | 0.174 | 0.596 | 0.900 | 3.405 |
| dinov3_s/minimum adaptive 1.5 | 178 | 14.079 | 0.149 | 0.547 | 0.886 | 3.117 |
| dinov3_s/minimum adaptive 2 | 141 | 10.779 | 0.159 | 0.482 | 0.861 | 2.795 |
| dinov3_s/minimum adaptive 2.5 | 110 | 8.282 | 0.137 | 0.402 | 0.821 | 2.429 |
| dinov3_s/minimum adaptive 3 | 92 | 6.760 | 0.168 | 0.368 | 0.775 | 2.194 |
| dinov3_s/minimum fixed 0.06835 | 207 | 16.721 | 0.238 | 0.612 | 0.903 | 3.213 |
| dinov3_s/cut adaptive 1 | 217 | 17.363 | 0.174 | 0.426 | 0.890 | 3.437 |
| dinov3_s/cut adaptive 1.5 | 178 | 14.079 | 0.149 | 0.364 | 0.875 | 3.133 |
| dinov3_s/cut adaptive 2 | 141 | 10.779 | 0.159 | 0.364 | 0.863 | 2.811 |
| dinov3_s/cut adaptive 2.5 | 110 | 8.282 | 0.137 | 0.270 | 0.821 | 2.445 |
| dinov3_s/cut adaptive 3 | 92 | 6.760 | 0.168 | 0.252 | 0.774 | 2.210 |
| dinov3_s/cut fixed 0.06835 | 207 | 16.721 | 0.238 | 0.397 | 0.885 | 3.245 |

Loop closure F1 of the mapper run with each detector:

| Detector | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 |
|---|---|---|---|---|
| valley_k1.0 | 0.280 | 0.486 | 0.856 | 0.776 |
| valley_k1.5 | 0.269 | 0.465 | 0.815 | 0.711 |
| topo | 0.432 | 0.612 | 0.725 | 0.581 |
| valley_k2.5 | 0.435 | 0.600 | 0.682 | 0.515 |
| valley_k3.0 | 0.000 | 0.571 | 0.640 | 0.400 |
| valley_fixed | 0.312 | 0.583 | 0.837 | 0.766 |
| boundary_cut | 0.432 | 0.583 | 0.676 | 0.562 |

## Development traversals (7)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 141 | 9 | 8 | 20 | 1 | 0.286 | 0.889 | 0.432 |
| geometry_only | 141 | 9 | 8 | 19 | 1 | 0.296 | 0.889 | 0.444 |
| vmf | 141 | 9 | 8 | 22 | 1 | 0.267 | 0.889 | 0.410 |
| isotropic_default | 141 | 9 | 8 | 32 | 1 | 0.200 | 0.889 | 0.327 |
| isotropic_calibrated | 141 | 9 | 7 | 21 | 2 | 0.250 | 0.778 | 0.378 |
| gate_propagated | 141 | 9 | 4 | 38 | 5 | 0.095 | 0.444 | 0.157 |
| boundary_cut | 141 | 9 | 8 | 20 | 1 | 0.286 | 0.889 | 0.432 |
| dinov2_s | 149 | 8 | 6 | 33 | 2 | 0.154 | 0.750 | 0.255 |
| anyloc_vlad_g | 142 | 8 | 7 | 25 | 1 | 0.219 | 0.875 | 0.350 |
| anyloc_gem_g | 132 | 10 | 9 | 19 | 1 | 0.321 | 0.900 | 0.474 |
| previous_system | 149 | 8 | 5 | 22 | 3 | 0.185 | 0.625 | 0.286 |
| threshold | 141 | 9 | 3 | 22 | 6 | 0.120 | 0.333 | 0.176 |
| visual_only | 141 | 9 | 3 | 73 | 6 | 0.039 | 0.333 | 0.071 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 141 | 21 | 15 | 13 | 6 | 0.536 | 0.714 | 0.612 |
| geometry_only | 141 | 21 | 15 | 12 | 6 | 0.556 | 0.714 | 0.625 |
| vmf | 141 | 21 | 15 | 15 | 6 | 0.500 | 0.714 | 0.588 |
| isotropic_default | 141 | 21 | 16 | 24 | 5 | 0.400 | 0.762 | 0.525 |
| isotropic_calibrated | 141 | 21 | 13 | 15 | 8 | 0.464 | 0.619 | 0.531 |
| gate_propagated | 141 | 21 | 11 | 31 | 10 | 0.262 | 0.524 | 0.349 |
| boundary_cut | 141 | 20 | 14 | 14 | 6 | 0.500 | 0.700 | 0.583 |
| dinov2_s | 149 | 23 | 13 | 26 | 10 | 0.333 | 0.565 | 0.419 |
| anyloc_vlad_g | 142 | 21 | 16 | 16 | 5 | 0.500 | 0.762 | 0.604 |
| anyloc_gem_g | 132 | 17 | 13 | 15 | 4 | 0.464 | 0.765 | 0.578 |
| previous_system | 149 | 23 | 8 | 19 | 15 | 0.296 | 0.348 | 0.320 |
| threshold | 141 | 21 | 3 | 22 | 18 | 0.120 | 0.143 | 0.130 |
| visual_only | 141 | 21 | 5 | 71 | 16 | 0.066 | 0.238 | 0.103 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 141 | 41 | 25 | 3 | 16 | 0.893 | 0.610 | 0.725 |
| geometry_only | 141 | 41 | 26 | 1 | 15 | 0.963 | 0.634 | 0.765 |
| vmf | 141 | 41 | 27 | 3 | 14 | 0.900 | 0.659 | 0.761 |
| isotropic_default | 141 | 41 | 30 | 10 | 11 | 0.750 | 0.732 | 0.741 |
| isotropic_calibrated | 141 | 41 | 22 | 6 | 19 | 0.786 | 0.537 | 0.638 |
| gate_propagated | 141 | 41 | 18 | 24 | 23 | 0.429 | 0.439 | 0.434 |
| boundary_cut | 141 | 40 | 23 | 5 | 17 | 0.821 | 0.575 | 0.676 |
| dinov2_s | 149 | 48 | 35 | 4 | 13 | 0.897 | 0.729 | 0.805 |
| anyloc_vlad_g | 142 | 40 | 23 | 9 | 17 | 0.719 | 0.575 | 0.639 |
| anyloc_gem_g | 132 | 38 | 24 | 4 | 14 | 0.857 | 0.632 | 0.727 |
| previous_system | 149 | 48 | 25 | 2 | 23 | 0.926 | 0.521 | 0.667 |
| threshold | 141 | 41 | 6 | 19 | 35 | 0.240 | 0.146 | 0.182 |
| visual_only | 141 | 41 | 11 | 65 | 30 | 0.145 | 0.268 | 0.188 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 141 | 65 | 27 | 1 | 38 | 0.964 | 0.415 | 0.581 |
| geometry_only | 141 | 65 | 27 | 0 | 38 | 1.000 | 0.415 | 0.587 |
| vmf | 141 | 65 | 29 | 1 | 36 | 0.967 | 0.446 | 0.611 |
| isotropic_default | 141 | 65 | 36 | 4 | 29 | 0.900 | 0.554 | 0.686 |
| isotropic_calibrated | 141 | 65 | 26 | 2 | 39 | 0.929 | 0.400 | 0.559 |
| gate_propagated | 141 | 65 | 28 | 14 | 37 | 0.667 | 0.431 | 0.523 |
| boundary_cut | 141 | 68 | 27 | 1 | 41 | 0.964 | 0.397 | 0.562 |
| dinov2_s | 149 | 72 | 38 | 1 | 34 | 0.974 | 0.528 | 0.685 |
| anyloc_vlad_g | 142 | 67 | 28 | 4 | 39 | 0.875 | 0.418 | 0.566 |
| anyloc_gem_g | 132 | 59 | 27 | 1 | 32 | 0.964 | 0.458 | 0.621 |
| previous_system | 149 | 72 | 25 | 2 | 47 | 0.926 | 0.347 | 0.505 |
| threshold | 141 | 65 | 10 | 15 | 55 | 0.400 | 0.154 | 0.222 |
| visual_only | 141 | 65 | 19 | 57 | 46 | 0.250 | 0.292 | 0.270 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.893 [0.774; 1.000], R 0.610 [0.394; 0.773], F 0.725 [0.542; 0.841]
- geometry_only: P 0.963 [0.892; 1.000], R 0.634 [0.417; 0.792], F 0.765 [0.582; 0.873]
- vmf: P 0.900 [0.795; 1.000], R 0.659 [0.429; 0.812], F 0.761 [0.577; 0.864]
- isotropic_default: P 0.750 [0.638; 0.889], R 0.732 [0.515; 0.885], F 0.741 [0.602; 0.847]
- isotropic_calibrated: P 0.786 [0.667; 0.909], R 0.537 [0.333; 0.744], F 0.638 [0.453; 0.781]
- gate_propagated: P 0.429 [0.250; 0.657], R 0.439 [0.298; 0.561], F 0.434 [0.274; 0.600]
- boundary_cut: P 0.821 [0.731; 0.917], R 0.575 [0.333; 0.806], F 0.676 [0.462; 0.833]
- dinov2_s: P 0.897 [0.780; 0.978], R 0.729 [0.585; 0.846], F 0.805 [0.704; 0.879]
- anyloc_vlad_g: P 0.719 [0.619; 0.811], R 0.575 [0.390; 0.737], F 0.639 [0.493; 0.747]
- anyloc_gem_g: P 0.857 [0.778; 0.955], R 0.632 [0.433; 0.784], F 0.727 [0.571; 0.833]
- previous_system: P 0.926 [0.727; 1.000], R 0.521 [0.325; 0.722], F 0.667 [0.455; 0.837]
- threshold: P 0.240 [0.026; 0.632], R 0.146 [0.024; 0.255], F 0.182 [0.026; 0.360]
- visual_only: P 0.145 [0.064; 0.276], R 0.268 [0.140; 0.413], F 0.188 [0.089; 0.324]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.736 [0.522; 0.910]
- geometry_only: AP 0.739 [0.554; 0.862]
- vmf: AP 0.751 [0.533; 0.925]
- isotropic_default: AP 0.708 [0.526; 0.841]
- isotropic_calibrated: AP 0.675 [0.486; 0.824]
- boundary_cut: AP 0.692 [0.480; 0.859]
- dinov2_s: AP 0.777 [0.615; 0.929]
- anyloc_vlad_g: AP 0.658 [0.468; 0.812]
- anyloc_gem_g: AP 0.702 [0.623; 0.748]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − previous_system: +0.058 [-0.189; +0.338]
- P topo − previous_system: -0.033 [-0.214; +0.184]
- R topo − previous_system: +0.089 [-0.164; +0.384]
- F1 topo − geometry_only: -0.040 [-0.091; +0.000]
- F1 topo − vmf: -0.036 [-0.071; +0.000]
- F1 topo − isotropic_calibrated: +0.087 [+0.008; +0.208]
- F1 topo − isotropic_default: -0.016 [-0.170; +0.095]
- F1 topo − gate_propagated: +0.291 [+0.114; +0.442]
- F1 topo − boundary_cut: +0.048 [-0.094; +0.152]
- F1 topo − dinov2_s: -0.080 [-0.247; +0.061]
- F1 topo − anyloc_vlad_g: +0.086 [+0.008; +0.165]
- F1 topo − anyloc_gem_g: -0.003 [-0.197; +0.181]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 12 | 7 | 29 | 18 | 3 |
| geometry_only | 12 | 8 | 29 | 18 | 1 |
| vmf | 12 | 9 | 29 | 18 | 3 |
| isotropic_default | 12 | 10 | 29 | 20 | 10 |
| isotropic_calibrated | 12 | 6 | 29 | 16 | 6 |
| gate_propagated | 12 | 7 | 29 | 11 | 24 |
| boundary_cut | 12 | 8 | 28 | 15 | 5 |
| dinov2_s | 15 | 14 | 33 | 21 | 4 |
| anyloc_vlad_g | 10 | 8 | 30 | 15 | 9 |
| anyloc_gem_g | 13 | 11 | 25 | 13 | 4 |
| previous_system | 15 | 11 | 33 | 14 | 2 |
| threshold | 12 | 5 | 29 | 1 | 19 |
| visual_only | 12 | 5 | 29 | 6 | 65 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| freiburg_a | 32 | 8 | 0.400 | 0.714 | 0.875 | 0.636 | 0.530 | 0.767 | 0.981 | 1.000 | 2.915 |
| freiburg_ext | 18 | 4 | 0.400 | 0.857 | 0.800 | 0.727 | 0.526 | 0.830 | 0.981 | 1.000 | 2.080 |
| saarbruecken_a | 29 | 1 | 1.000 | 0.667 | 0.286 | 0.182 | 0.394 | 0.647 | 0.901 | 0.964 | 4.033 |
| saarbruecken_ext | 16 | 3 | 0.500 | 0.800 | 1.000 | 0.667 | 0.537 | 0.805 | 1.000 | 1.000 | 1.096 |
| cid_sims_apartment1_1 | 16 | 6 | 0.444 | 0.364 | 0.615 | 0.667 | 0.566 | 0.941 | 1.000 | 1.000 | 0.468 |
| cid_sims_apartment2_1 | 10 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.424 | 0.688 | 0.963 | 1.000 | 0.376 |
| cid_sims_apartment3_1 | 20 | 6 | 0.286 | 0.444 | 0.800 | 0.706 | 0.657 | 0.920 | 1.000 | 1.000 | 0.619 |

## Validation traversals (12)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 222 | 12 | 8 | 24 | 4 | 0.250 | 0.667 | 0.364 |
| geometry_only | 222 | 12 | 7 | 24 | 5 | 0.226 | 0.583 | 0.326 |
| vmf | 222 | 12 | 8 | 25 | 4 | 0.242 | 0.667 | 0.356 |
| isotropic_default | 222 | 12 | 8 | 34 | 4 | 0.190 | 0.667 | 0.296 |
| isotropic_calibrated | 222 | 12 | 8 | 27 | 4 | 0.229 | 0.667 | 0.340 |
| gate_propagated | 222 | 12 | 3 | 46 | 9 | 0.061 | 0.250 | 0.098 |
| boundary_cut | 222 | 14 | 11 | 21 | 3 | 0.344 | 0.786 | 0.478 |
| dinov2_s | 229 | 10 | 8 | 28 | 2 | 0.222 | 0.800 | 0.348 |
| anyloc_vlad_g | 214 | 7 | 5 | 24 | 2 | 0.172 | 0.714 | 0.278 |
| anyloc_gem_g | 219 | 8 | 5 | 25 | 3 | 0.167 | 0.625 | 0.263 |
| previous_system | 229 | 10 | 6 | 21 | 4 | 0.222 | 0.600 | 0.324 |
| threshold | 222 | 12 | 2 | 40 | 10 | 0.048 | 0.167 | 0.074 |
| visual_only | 222 | 12 | 3 | 107 | 9 | 0.027 | 0.250 | 0.049 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 222 | 24 | 15 | 17 | 9 | 0.469 | 0.625 | 0.536 |
| geometry_only | 222 | 24 | 14 | 17 | 10 | 0.452 | 0.583 | 0.509 |
| vmf | 222 | 24 | 15 | 18 | 9 | 0.455 | 0.625 | 0.526 |
| isotropic_default | 222 | 24 | 15 | 27 | 9 | 0.357 | 0.625 | 0.455 |
| isotropic_calibrated | 222 | 24 | 15 | 20 | 9 | 0.429 | 0.625 | 0.508 |
| gate_propagated | 222 | 24 | 8 | 41 | 16 | 0.163 | 0.333 | 0.219 |
| boundary_cut | 222 | 26 | 19 | 13 | 7 | 0.594 | 0.731 | 0.655 |
| dinov2_s | 229 | 19 | 14 | 22 | 5 | 0.389 | 0.737 | 0.509 |
| anyloc_vlad_g | 214 | 22 | 13 | 16 | 9 | 0.448 | 0.591 | 0.510 |
| anyloc_gem_g | 219 | 25 | 15 | 15 | 10 | 0.500 | 0.600 | 0.545 |
| previous_system | 229 | 19 | 10 | 17 | 9 | 0.370 | 0.526 | 0.435 |
| threshold | 222 | 24 | 2 | 40 | 22 | 0.048 | 0.083 | 0.061 |
| visual_only | 222 | 24 | 5 | 105 | 19 | 0.045 | 0.208 | 0.075 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 222 | 47 | 28 | 4 | 19 | 0.875 | 0.596 | 0.709 |
| geometry_only | 222 | 47 | 27 | 4 | 20 | 0.871 | 0.574 | 0.692 |
| vmf | 222 | 47 | 28 | 5 | 19 | 0.848 | 0.596 | 0.700 |
| isotropic_default | 222 | 47 | 30 | 12 | 17 | 0.714 | 0.638 | 0.674 |
| isotropic_calibrated | 222 | 47 | 28 | 7 | 19 | 0.800 | 0.596 | 0.683 |
| gate_propagated | 222 | 47 | 16 | 33 | 31 | 0.327 | 0.340 | 0.333 |
| boundary_cut | 222 | 48 | 29 | 3 | 19 | 0.906 | 0.604 | 0.725 |
| dinov2_s | 229 | 52 | 30 | 6 | 22 | 0.833 | 0.577 | 0.682 |
| anyloc_vlad_g | 214 | 43 | 23 | 6 | 20 | 0.793 | 0.535 | 0.639 |
| anyloc_gem_g | 219 | 49 | 25 | 5 | 24 | 0.833 | 0.510 | 0.633 |
| previous_system | 229 | 52 | 19 | 8 | 33 | 0.704 | 0.365 | 0.481 |
| threshold | 222 | 47 | 3 | 39 | 44 | 0.071 | 0.064 | 0.067 |
| visual_only | 222 | 47 | 10 | 100 | 37 | 0.091 | 0.213 | 0.127 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 222 | 75 | 32 | 0 | 43 | 1.000 | 0.427 | 0.598 |
| geometry_only | 222 | 75 | 31 | 0 | 44 | 1.000 | 0.413 | 0.585 |
| vmf | 222 | 75 | 33 | 0 | 42 | 1.000 | 0.440 | 0.611 |
| isotropic_default | 222 | 75 | 39 | 3 | 36 | 0.929 | 0.520 | 0.667 |
| isotropic_calibrated | 222 | 75 | 32 | 3 | 43 | 0.914 | 0.427 | 0.582 |
| gate_propagated | 222 | 75 | 28 | 21 | 47 | 0.571 | 0.373 | 0.452 |
| boundary_cut | 222 | 77 | 32 | 0 | 45 | 1.000 | 0.416 | 0.587 |
| dinov2_s | 229 | 83 | 35 | 1 | 48 | 0.972 | 0.422 | 0.588 |
| anyloc_vlad_g | 214 | 72 | 26 | 3 | 46 | 0.897 | 0.361 | 0.515 |
| anyloc_gem_g | 219 | 74 | 29 | 1 | 45 | 0.967 | 0.392 | 0.558 |
| previous_system | 229 | 83 | 22 | 5 | 61 | 0.815 | 0.265 | 0.400 |
| threshold | 222 | 75 | 5 | 37 | 70 | 0.119 | 0.067 | 0.085 |
| visual_only | 222 | 75 | 18 | 92 | 57 | 0.164 | 0.240 | 0.195 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.875 [0.781; 1.000], R 0.596 [0.417; 0.744], F 0.709 [0.566; 0.809]
- geometry_only: P 0.871 [0.773; 1.000], R 0.574 [0.373; 0.736], F 0.692 [0.520; 0.804]
- vmf: P 0.848 [0.750; 0.955], R 0.596 [0.417; 0.744], F 0.700 [0.558; 0.794]
- isotropic_default: P 0.714 [0.608; 0.879], R 0.638 [0.464; 0.790], F 0.674 [0.556; 0.795]
- isotropic_calibrated: P 0.800 [0.676; 1.000], R 0.596 [0.417; 0.744], F 0.683 [0.557; 0.786]
- gate_propagated: P 0.327 [0.186; 0.529], R 0.340 [0.233; 0.455], F 0.333 [0.208; 0.482]
- boundary_cut: P 0.906 [0.750; 1.000], R 0.604 [0.444; 0.735], F 0.725 [0.581; 0.835]
- dinov2_s: P 0.833 [0.778; 0.923], R 0.577 [0.450; 0.724], F 0.682 [0.584; 0.787]
- anyloc_vlad_g: P 0.793 [0.684; 0.938], R 0.535 [0.423; 0.654], F 0.639 [0.545; 0.743]
- anyloc_gem_g: P 0.833 [0.706; 1.000], R 0.510 [0.411; 0.655], F 0.633 [0.538; 0.762]
- previous_system: P 0.704 [0.483; 1.000], R 0.365 [0.236; 0.525], F 0.481 [0.329; 0.667]
- threshold: P 0.071 [0.000; 0.219], R 0.064 [0.000; 0.171], F 0.067 [0.000; 0.184]
- visual_only: P 0.091 [0.042; 0.136], R 0.213 [0.125; 0.269], F 0.127 [0.064; 0.177]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.695 [0.578; 0.816]
- geometry_only: AP 0.676 [0.555; 0.797]
- vmf: AP 0.689 [0.573; 0.808]
- isotropic_default: AP 0.618 [0.461; 0.780]
- isotropic_calibrated: AP 0.675 [0.513; 0.834]
- boundary_cut: AP 0.685 [0.513; 0.846]
- dinov2_s: AP 0.754 [0.641; 0.870]
- anyloc_vlad_g: AP 0.639 [0.493; 0.808]
- anyloc_gem_g: AP 0.720 [0.566; 0.902]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − previous_system: +0.228 [+0.006; +0.397]
- P topo − previous_system: +0.171 [-0.118; +0.371]
- R topo − previous_system: +0.230 [+0.014; +0.408]
- F1 topo − geometry_only: +0.017 [+0.000; +0.073]
- F1 topo − vmf: +0.009 [+0.000; +0.035]
- F1 topo − isotropic_calibrated: +0.026 [+0.000; +0.063]
- F1 topo − isotropic_default: +0.035 [-0.068; +0.094]
- F1 topo − gate_propagated: +0.376 [+0.221; +0.516]
- F1 topo − boundary_cut: -0.016 [-0.074; +0.042]
- F1 topo − dinov2_s: +0.027 [-0.097; +0.122]
- F1 topo − anyloc_vlad_g: +0.070 [-0.080; +0.178]
- F1 topo − anyloc_gem_g: +0.076 [-0.109; +0.202]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 5 | 2 | 42 | 26 | 4 |
| geometry_only | 5 | 1 | 42 | 26 | 4 |
| vmf | 5 | 2 | 42 | 26 | 5 |
| isotropic_default | 5 | 2 | 42 | 28 | 12 |
| isotropic_calibrated | 5 | 2 | 42 | 26 | 7 |
| gate_propagated | 5 | 2 | 42 | 14 | 33 |
| boundary_cut | 5 | 3 | 43 | 26 | 3 |
| dinov2_s | 7 | 4 | 45 | 26 | 6 |
| anyloc_vlad_g | 3 | 3 | 40 | 20 | 6 |
| anyloc_gem_g | 4 | 3 | 45 | 22 | 5 |
| previous_system | 7 | 1 | 45 | 18 | 8 |
| threshold | 5 | 2 | 42 | 1 | 39 |
| visual_only | 5 | 2 | 42 | 8 | 100 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_freiburg_a_seq1 | 21 | 5 | 0.000 | 0.750 | 0.727 | 0.833 | 0.557 | 0.777 | 0.932 | 1.000 | 1.787 |
| heldout_freiburg_a_seq2 | 33 | 10 | 0.462 | 0.429 | 0.800 | 0.833 | 0.491 | 0.770 | 0.987 | 1.000 | 3.670 |
| heldout_freiburg_b_seq3 | 21 | 5 | 0.750 | 0.600 | 0.909 | 0.769 | 0.535 | 0.831 | 0.989 | 0.999 | 1.722 |
| heldout_saarbruecken_a_seq1 | 12 | 2 | 0.000 | 0.500 | 0.800 | 0.667 | 0.378 | 0.610 | 0.827 | 0.986 | 0.974 |
| heldout_saarbruecken_a_seq2 | 30 | 2 | 0.000 | 0.667 | 0.400 | 0.286 | 0.382 | 0.612 | 0.904 | 0.979 | 4.556 |
| heldout_saarbruecken_b_seq3 | 11 | 1 | 0.000 | 0.000 | 0.667 | 0.500 | 0.656 | 0.887 | 0.988 | 1.000 | 0.524 |
| heldout_saarbruecken_b_seq4 | 13 | 1 | 0.000 | 0.000 | 0.000 | 0.400 | 0.517 | 0.775 | 1.000 | 1.000 | 1.070 |
| heldout_ljubljana_a_seq1 | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.152 | 0.268 | 0.462 | 0.576 | 2.168 |
| heldout_ljubljana_a_seq2 | 23 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.272 | 0.397 | 0.614 | 0.718 | 5.979 |
| heldout_cid_apartment1_2 | 9 | 1 | 0.667 | 0.667 | 0.500 | 0.400 | 0.251 | 0.465 | 0.827 | 0.943 | 0.948 |
| heldout_cid_apartment2_2 | 18 | 3 | 0.333 | 0.571 | 0.667 | 0.400 | 0.483 | 0.854 | 1.000 | 1.000 | 0.938 |
| heldout_cid_apartment3_2 | 16 | 2 | 0.000 | 0.000 | 1.000 | 0.571 | 0.458 | 0.765 | 0.991 | 1.000 | 1.237 |

## Language retrieval (google/siglip2-base-patch16-224)

Fields per environment and overall:

```
{
 "overall": {
  "absent": {
   "correct_rejection_rate": 0.5741,
   "false_answer_rate": 0.4259,
   "n": 162
  },
  "heuristic_decision_accuracy": 0.31,
  "n_map_query_pairs": 400,
  "present": {
   "answer_coverage": 0.3025,
   "mrr": 0.5784,
   "n": 238,
   "precision_at_1_when_answered": 0.4306,
   "recall_at_1": 0.4328,
   "recall_at_3": 0.7605,
   "successful_answer_rate": 0.1303
  }
 },
 "environments": {
  "freiburg_a": {
   "absent": {
    "correct_rejection_rate": 0.6667,
    "false_answer_rate": 0.3333,
    "n": 27
   },
   "heuristic_decision_accuracy": 0.3,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.2192,
    "mrr": 0.5822,
    "n": 73,
    "precision_at_1_when_answered": 0.75,
    "recall_at_1": 0.4795,
    "recall_at_3": 0.7123,
    "successful_answer_rate": 0.1644
   }
  },
  "freiburg_ext": {
   "absent": {
    "correct_rejection_rate": 0.537,
    "false_answer_rate": 0.463,
    "n": 54
   },
   "heuristic_decision_accuracy": 0.32,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.4348,
    "mrr": 0.5181,
    "n": 46,
    "precision_at_1_when_answered": 0.15,
    "recall_at_1": 0.2826,
    "recall_at_3": 0.8043,
    "successful_answer_rate": 0.0652
   }
  },
  "saarbruecken_a": {
   "absent": {
    "correct_rejection_rate": 0.4815,
    "false_answer_rate": 0.5185,
    "n": 27
   },
   "heuristic_decision_accuracy": 0.16,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.2466,
    "mrr": 0.5457,
    "n": 73,
    "precision_at_1_when_answered": 0.1667,
    "recall_at_1": 0.3973,
    "recall_at_3": 0.726,
    "successful_answer_rate": 0.0411
   }
  },
  "saarbruecken_ext": {
   "absent": {
    "correct_rejection_rate": 0.6111,
    "false_answer_rate": 0.3889,
    "n": 54
   },
   "heuristic_decision_accuracy": 0.46,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.3913,
    "mrr": 0.6848,
    "n": 46,
    "precision_at_1_when_answered": 0.7222,
    "recall_at_1": 0.5652,
    "recall_at_3": 0.8478,
    "successful_answer_rate": 0.2826
   }
  }
 },
 "present_by_category": {
  "command": {
   "answer_coverage": 0.4118,
   "mrr": 0.5784,
   "n": 51,
   "precision_at_1_when_answered": 0.4286,
   "recall_at_1": 0.4314,
   "recall_at_3": 0.7647,
   "successful_answer_rate": 0.1765
  },
  "direct": {
   "answer_coverage": 0.2308,
   "mrr": 0.7051,
   "n": 26,
   "precision_at_1_when_answered": 0.5,
   "recall_at_1": 0.5769,
   "recall_at_3": 0.8462,
   "successful_answer_rate": 0.1154
  },
  "function": {
   "answer_coverage": 0.375,
   "mrr": 0.5,
   "n": 32,
   "precision_at_1_when_answered": 0.5833,
   "recall_at_1": 0.4062,
   "recall_at_3": 0.625,
   "successful_answer_rate": 0.2188
  },
  "indirect": {
   "answer_coverage": 0.0882,
   "mrr": 0.5735,
   "n": 34,
   "precision_at_1_when_answered": 0.3333,
   "recall_at_1": 0.4412,
   "recall_at_3": 0.7353,
   "successful_answer_rate": 0.0294
  },
  "object": {
   "answer_coverage": 0.4545,
   "mrr": 0.4924,
   "n": 22,
   "precision_at_1_when_answered": 0.2,
   "recall_at_1": 0.2727,
   "recall_at_3": 0.7727,
   "successful_answer_rate": 0.0909
  },
  "paraphrase": {
   "answer_coverage": 0.2619,
   "mrr": 0.5595,
   "n": 42,
   "precision_at_1_when_answered": 0.3636,
   "recall_at_1": 0.4048,
   "recall_at_3": 0.7619,
   "successful_answer_rate": 0.0952
  },
  "scene": {
   "answer_coverage": 0.2903,
   "mrr": 0.6452,
   "n": 31,
   "precision_at_1_when_answered": 0.5556,
   "recall_at_1": 0.4839,
   "recall_at_3": 0.8387,
   "successful_answer_rate": 0.1613
  }
 }
}
```
