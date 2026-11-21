# Manuscript results on the GPU

## Odometry calibration (development ground truth only)

Pairs per traversal: {'freiburg_a': 594, 'freiburg_ext': 291, 'saarbruecken_a': 955, 'saarbruecken_ext': 267, 'cid_sims_apartment1_1': 293, 'cid_sims_apartment2_1': 291, 'cid_sims_apartment3_1': 342, 'heldout_freiburg_a_seq1': 315, 'heldout_freiburg_a_seq2': 594, 'heldout_freiburg_b_seq3': 273, 'heldout_saarbruecken_a_seq1': 249, 'heldout_saarbruecken_a_seq2': 945, 'heldout_saarbruecken_b_seq3': 145, 'heldout_saarbruecken_b_seq4': 261, 'heldout_ljubljana_a_seq1': 1137, 'heldout_ljubljana_a_seq2': 1314, 'heldout_cid_apartment1_2': 370, 'heldout_cid_apartment2_2': 412, 'heldout_cid_apartment3_2': 398}

| Model | α1 | α2 | α3 | α4 | floor [m] | mean NLL |
|---|---|---|---|---|---|---|
| isotropic, original | 0.025 | 0.005 | 0.01 | 0.0025 | 0.950 | – |
| isotropic, calibrated | 0.025 (unused) | 0.005 (unused) | 0.002345 [0.000993; 0.0599] | 0.1202 [9.74e-07; 0.177] | 0.000 [0.000; 0.000] | 1.623 |
| propagated, calibrated | 0.00948 [2.65e-07; 0.0141] | 0.0009416 [1.52e-08; 0.00631] | 1.523e-08 [1.52e-08; 1.52e-08] | 0.02419 [0.0175; 0.0309] | 0.111 [0.080; 0.153] | 0.902 |
| propagated, debiased heading | 0.001561 [2.38e-08; 0.00393] | 2.279e-05 [1.52e-08; 0.000596] | 1.523e-08 [1.52e-08; 1.52e-08] | 0.02438 [0.0189; 0.0321] | 0.145 [0.111; 0.178] | 0.485 |

Leave-one-traversal-out mean NLL (lower is better):

| Traversal | isotropic_default | isotropic_calibrated | propagated_calibrated | propagated_debiased |
|---|---|---|---|---|
| freiburg_a | 5.009 | 2.372 | 1.360 | 0.129 |
| freiburg_ext | 3.395 | 1.657 | 0.577 | -0.102 |
| saarbruecken_a | 5.389 | 2.424 | 1.413 | 0.561 |
| saarbruecken_ext | 2.493 | 1.086 | 0.546 | 0.137 |
| cid_sims_apartment1_1 | 2.088 | 1.287 | 1.269 | 1.349 |
| cid_sims_apartment2_1 | 2.062 | 1.448 | 1.184 | 0.950 |
| cid_sims_apartment3_1 | 2.076 | 1.136 | 0.736 | 0.743 |
| heldout_freiburg_a_seq1 | 3.061 | 1.519 | 0.669 | 0.032 |
| heldout_freiburg_a_seq2 | 5.799 | 2.729 | 1.487 | 0.186 |
| heldout_freiburg_b_seq3 | 3.295 | 1.650 | 0.639 | -0.002 |
| heldout_saarbruecken_a_seq1 | 2.283 | 1.004 | 0.364 | -0.122 |
| heldout_saarbruecken_a_seq2 | 5.778 | 2.500 | 1.440 | 0.510 |
| heldout_saarbruecken_b_seq3 | 2.049 | 0.619 | 0.214 | -0.113 |
| heldout_saarbruecken_b_seq4 | 2.485 | 1.124 | 0.520 | 0.110 |
| heldout_ljubljana_a_seq1 | 2.549 | 1.386 | 0.561 | -0.002 |
| heldout_ljubljana_a_seq2 | 9.093 | 4.685 | 1.577 | 2.671 |
| heldout_cid_apartment1_2 | 2.188 | 1.429 | 0.842 | 0.674 |
| heldout_cid_apartment2_2 | 2.150 | 1.601 | 0.827 | 0.632 |
| heldout_cid_apartment3_2 | 2.330 | 1.605 | 1.644 | 1.990 |
| mean | 3.451 | 1.751 | 0.941 | 0.544 |

Fraction of true drifts inside the 95 % gate (with σ_place), by travelled distance [m]:

| Gate | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 32.0 | 64.0 | 128.0 |
|---|---|---|---|---|---|---|---|---|
| isotropic_default | 1.00 | 1.00 | 1.00 | 0.98 | 0.77 | 0.47 | 0.17 | 0.01 |
| isotropic_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.76 | 0.39 | 0.13 |
| propagated_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.82 | 0.91 |
| propagated_debiased | 1.00 | 1.00 | 1.00 | 0.99 | 0.97 | 0.94 | 0.93 | 0.94 |

## Node creation sensitivity (development)

Fixed δ calibrated on freiburg_a: 0.0683

| Detector | Nodes | ρ | CV | R_trans | Purity | Seg./episode |
|---|---|---|---|---|---|---|
| dinov2_s/minimum adaptive 1 | 214 | 16.770 | 0.149 | 0.643 | 0.907 | 3.425 |
| dinov2_s/minimum adaptive 1.5 | 180 | 14.088 | 0.142 | 0.580 | 0.894 | 3.183 |
| dinov2_s/minimum adaptive 2 | 147 | 11.365 | 0.135 | 0.467 | 0.858 | 2.888 |
| dinov2_s/minimum adaptive 2.5 | 116 | 8.858 | 0.104 | 0.417 | 0.824 | 2.451 |
| dinov2_s/minimum adaptive 3 | 95 | 7.031 | 0.130 | 0.368 | 0.764 | 2.142 |
| dinov2_s/minimum fixed 0.1301 | 158 | 12.402 | 0.232 | 0.596 | 0.895 | 3.099 |
| dinov2_s/cut adaptive 1 | 214 | 16.770 | 0.149 | 0.379 | 0.890 | 3.425 |
| dinov2_s/cut adaptive 1.5 | 180 | 14.088 | 0.142 | 0.333 | 0.887 | 3.183 |
| dinov2_s/cut adaptive 2 | 147 | 11.365 | 0.135 | 0.317 | 0.868 | 2.888 |
| dinov2_s/cut adaptive 2.5 | 116 | 8.858 | 0.104 | 0.286 | 0.823 | 2.451 |
| dinov2_s/cut adaptive 3 | 95 | 7.031 | 0.130 | 0.254 | 0.769 | 2.142 |
| dinov2_s/cut fixed 0.1301 | 158 | 12.402 | 0.232 | 0.333 | 0.882 | 3.099 |
| dinov3_s/minimum adaptive 1 | 214 | 17.196 | 0.184 | 0.596 | 0.899 | 3.322 |
| dinov3_s/minimum adaptive 1.5 | 176 | 13.977 | 0.158 | 0.547 | 0.885 | 3.061 |
| dinov3_s/minimum adaptive 2 | 139 | 10.668 | 0.165 | 0.482 | 0.861 | 2.739 |
| dinov3_s/minimum adaptive 2.5 | 108 | 8.164 | 0.132 | 0.402 | 0.821 | 2.373 |
| dinov3_s/minimum adaptive 3 | 90 | 6.635 | 0.157 | 0.337 | 0.765 | 2.138 |
| dinov3_s/minimum fixed 0.06835 | 207 | 16.789 | 0.233 | 0.612 | 0.903 | 3.213 |
| dinov3_s/cut adaptive 1 | 214 | 17.196 | 0.184 | 0.426 | 0.890 | 3.353 |
| dinov3_s/cut adaptive 1.5 | 176 | 13.977 | 0.158 | 0.364 | 0.875 | 3.078 |
| dinov3_s/cut adaptive 2 | 139 | 10.668 | 0.165 | 0.364 | 0.863 | 2.756 |
| dinov3_s/cut adaptive 2.5 | 108 | 8.164 | 0.132 | 0.270 | 0.821 | 2.390 |
| dinov3_s/cut adaptive 3 | 90 | 6.635 | 0.157 | 0.221 | 0.766 | 2.155 |
| dinov3_s/cut fixed 0.06835 | 207 | 16.789 | 0.233 | 0.397 | 0.884 | 3.245 |

Loop closure F1 of the mapper run with each detector:

| Detector | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 |
|---|---|---|---|---|
| valley_k1.0 | 0.241 | 0.506 | 0.866 | 0.840 |
| valley_k1.5 | 0.322 | 0.531 | 0.833 | 0.773 |
| topo | 0.320 | 0.492 | 0.813 | 0.690 |
| valley_k2.5 | 0.312 | 0.524 | 0.814 | 0.634 |
| valley_k3.0 | 0.000 | 0.480 | 0.757 | 0.582 |
| valley_fixed | 0.260 | 0.522 | 0.844 | 0.830 |
| boundary_cut | 0.333 | 0.462 | 0.766 | 0.656 |

## Development traversals (7)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 10 | 8 | 32 | 2 | 0.200 | 0.800 | 0.320 |
| topo_debiased | 139 | 10 | 10 | 39 | 0 | 0.204 | 1.000 | 0.339 |
| anyloc_vlad_g | 138 | 8 | 6 | 43 | 2 | 0.122 | 0.750 | 0.211 |
| anyloc_gem_g | 131 | 15 | 13 | 29 | 2 | 0.310 | 0.867 | 0.456 |
| previous_system | 147 | 12 | 7 | 26 | 5 | 0.212 | 0.583 | 0.311 |
| geometry_only | 139 | 10 | 8 | 33 | 2 | 0.195 | 0.800 | 0.314 |
| vmf | 139 | 10 | 9 | 31 | 1 | 0.225 | 0.900 | 0.360 |
| isotropic_default | 139 | 10 | 8 | 40 | 2 | 0.167 | 0.800 | 0.276 |
| isotropic_calibrated | 139 | 10 | 7 | 29 | 3 | 0.194 | 0.700 | 0.304 |
| gate_propagated | 139 | 10 | 8 | 38 | 2 | 0.174 | 0.800 | 0.286 |
| boundary_cut | 139 | 11 | 9 | 34 | 2 | 0.209 | 0.818 | 0.333 |
| dinov2_s | 147 | 12 | 9 | 43 | 3 | 0.173 | 0.750 | 0.281 |
| threshold | 139 | 10 | 4 | 25 | 6 | 0.138 | 0.400 | 0.205 |
| visual_only | 139 | 10 | 1 | 85 | 9 | 0.012 | 0.100 | 0.021 |
| anyloc_vlad_g_debiased | 138 | 8 | 8 | 45 | 0 | 0.151 | 1.000 | 0.262 |
| anyloc_gem_g_debiased | 131 | 15 | 13 | 37 | 2 | 0.260 | 0.867 | 0.400 |
| dinov2_s_debiased | 147 | 12 | 10 | 48 | 2 | 0.172 | 0.833 | 0.286 |
| geometry_only_debiased | 139 | 10 | 9 | 41 | 1 | 0.180 | 0.900 | 0.300 |
| vmf_debiased | 139 | 10 | 10 | 39 | 0 | 0.204 | 1.000 | 0.339 |
| boundary_cut_debiased | 139 | 11 | 10 | 36 | 1 | 0.217 | 0.909 | 0.351 |
| gate_propagated_debiased | 139 | 10 | 9 | 33 | 1 | 0.214 | 0.900 | 0.346 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 25 | 16 | 24 | 9 | 0.400 | 0.640 | 0.492 |
| topo_debiased | 139 | 25 | 20 | 29 | 5 | 0.408 | 0.800 | 0.541 |
| anyloc_vlad_g | 138 | 27 | 22 | 27 | 5 | 0.449 | 0.815 | 0.579 |
| anyloc_gem_g | 131 | 24 | 20 | 22 | 4 | 0.476 | 0.833 | 0.606 |
| previous_system | 147 | 30 | 14 | 19 | 16 | 0.424 | 0.467 | 0.444 |
| geometry_only | 139 | 25 | 16 | 25 | 9 | 0.390 | 0.640 | 0.485 |
| vmf | 139 | 25 | 17 | 23 | 8 | 0.425 | 0.680 | 0.523 |
| isotropic_default | 139 | 25 | 16 | 32 | 9 | 0.333 | 0.640 | 0.438 |
| isotropic_calibrated | 139 | 25 | 15 | 21 | 10 | 0.417 | 0.600 | 0.492 |
| gate_propagated | 139 | 25 | 15 | 31 | 10 | 0.326 | 0.600 | 0.423 |
| boundary_cut | 139 | 22 | 15 | 28 | 7 | 0.349 | 0.682 | 0.462 |
| dinov2_s | 147 | 30 | 20 | 32 | 10 | 0.385 | 0.667 | 0.488 |
| threshold | 139 | 25 | 4 | 25 | 21 | 0.138 | 0.160 | 0.148 |
| visual_only | 139 | 25 | 2 | 84 | 23 | 0.023 | 0.080 | 0.036 |
| anyloc_vlad_g_debiased | 138 | 27 | 24 | 29 | 3 | 0.453 | 0.889 | 0.600 |
| anyloc_gem_g_debiased | 131 | 24 | 21 | 29 | 3 | 0.420 | 0.875 | 0.568 |
| dinov2_s_debiased | 147 | 30 | 21 | 37 | 9 | 0.362 | 0.700 | 0.477 |
| geometry_only_debiased | 139 | 25 | 19 | 31 | 6 | 0.380 | 0.760 | 0.507 |
| vmf_debiased | 139 | 25 | 20 | 29 | 5 | 0.408 | 0.800 | 0.541 |
| boundary_cut_debiased | 139 | 22 | 17 | 29 | 5 | 0.370 | 0.773 | 0.500 |
| gate_propagated_debiased | 139 | 25 | 17 | 25 | 8 | 0.405 | 0.680 | 0.507 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 51 | 37 | 3 | 14 | 0.925 | 0.725 | 0.813 |
| topo_debiased | 139 | 51 | 45 | 4 | 6 | 0.918 | 0.882 | 0.900 |
| anyloc_vlad_g | 138 | 55 | 37 | 12 | 18 | 0.755 | 0.673 | 0.712 |
| anyloc_gem_g | 131 | 53 | 37 | 5 | 16 | 0.881 | 0.698 | 0.779 |
| previous_system | 147 | 62 | 31 | 2 | 31 | 0.939 | 0.500 | 0.653 |
| geometry_only | 139 | 51 | 37 | 4 | 14 | 0.902 | 0.725 | 0.804 |
| vmf | 139 | 51 | 38 | 2 | 13 | 0.950 | 0.745 | 0.835 |
| isotropic_default | 139 | 51 | 38 | 10 | 13 | 0.792 | 0.745 | 0.768 |
| isotropic_calibrated | 139 | 51 | 32 | 4 | 19 | 0.889 | 0.627 | 0.736 |
| gate_propagated | 139 | 51 | 30 | 16 | 21 | 0.652 | 0.588 | 0.619 |
| boundary_cut | 139 | 51 | 36 | 7 | 15 | 0.837 | 0.706 | 0.766 |
| dinov2_s | 147 | 62 | 48 | 4 | 14 | 0.923 | 0.774 | 0.842 |
| threshold | 139 | 51 | 9 | 20 | 42 | 0.310 | 0.176 | 0.225 |
| visual_only | 139 | 51 | 13 | 73 | 38 | 0.151 | 0.255 | 0.190 |
| anyloc_vlad_g_debiased | 138 | 55 | 43 | 10 | 12 | 0.811 | 0.782 | 0.796 |
| anyloc_gem_g_debiased | 131 | 53 | 43 | 7 | 10 | 0.860 | 0.811 | 0.835 |
| dinov2_s_debiased | 147 | 62 | 52 | 6 | 10 | 0.897 | 0.839 | 0.867 |
| geometry_only_debiased | 139 | 51 | 44 | 6 | 7 | 0.880 | 0.863 | 0.871 |
| vmf_debiased | 139 | 51 | 45 | 4 | 6 | 0.918 | 0.882 | 0.900 |
| boundary_cut_debiased | 139 | 51 | 42 | 4 | 9 | 0.913 | 0.824 | 0.866 |
| gate_propagated_debiased | 139 | 51 | 34 | 8 | 17 | 0.810 | 0.667 | 0.731 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 76 | 40 | 0 | 36 | 1.000 | 0.526 | 0.690 |
| topo_debiased | 139 | 76 | 49 | 0 | 27 | 1.000 | 0.645 | 0.784 |
| anyloc_vlad_g | 138 | 78 | 47 | 2 | 31 | 0.959 | 0.603 | 0.740 |
| anyloc_gem_g | 131 | 74 | 42 | 0 | 32 | 1.000 | 0.568 | 0.724 |
| previous_system | 147 | 89 | 31 | 2 | 58 | 0.939 | 0.348 | 0.508 |
| geometry_only | 139 | 76 | 41 | 0 | 35 | 1.000 | 0.539 | 0.701 |
| vmf | 139 | 76 | 40 | 0 | 36 | 1.000 | 0.526 | 0.690 |
| isotropic_default | 139 | 76 | 45 | 3 | 31 | 0.938 | 0.592 | 0.726 |
| isotropic_calibrated | 139 | 76 | 35 | 1 | 41 | 0.972 | 0.461 | 0.625 |
| gate_propagated | 139 | 76 | 35 | 11 | 41 | 0.761 | 0.461 | 0.574 |
| boundary_cut | 139 | 82 | 41 | 2 | 41 | 0.953 | 0.500 | 0.656 |
| dinov2_s | 147 | 89 | 51 | 1 | 38 | 0.981 | 0.573 | 0.723 |
| threshold | 139 | 76 | 13 | 16 | 63 | 0.448 | 0.171 | 0.248 |
| visual_only | 139 | 76 | 29 | 57 | 47 | 0.337 | 0.382 | 0.358 |
| anyloc_vlad_g_debiased | 138 | 78 | 52 | 1 | 26 | 0.981 | 0.667 | 0.794 |
| anyloc_gem_g_debiased | 131 | 74 | 49 | 1 | 25 | 0.980 | 0.662 | 0.790 |
| dinov2_s_debiased | 147 | 89 | 57 | 1 | 32 | 0.983 | 0.640 | 0.776 |
| geometry_only_debiased | 139 | 76 | 49 | 1 | 27 | 0.980 | 0.645 | 0.778 |
| vmf_debiased | 139 | 76 | 48 | 1 | 28 | 0.980 | 0.632 | 0.768 |
| boundary_cut_debiased | 139 | 82 | 46 | 0 | 36 | 1.000 | 0.561 | 0.719 |
| gate_propagated_debiased | 139 | 76 | 41 | 1 | 35 | 0.976 | 0.539 | 0.695 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.925 [0.828; 0.980], R 0.725 [0.486; 0.873], F 0.813 [0.623; 0.908]
- topo_debiased: P 0.918 [0.795; 0.984], R 0.882 [0.739; 0.970], F 0.900 [0.784; 0.972]
- anyloc_vlad_g: P 0.755 [0.634; 0.833], R 0.673 [0.429; 0.859], F 0.712 [0.526; 0.828]
- anyloc_gem_g: P 0.881 [0.776; 1.000], R 0.698 [0.532; 0.889], F 0.779 [0.653; 0.918]
- previous_system: P 0.939 [0.778; 1.000], R 0.500 [0.314; 0.677], F 0.653 [0.451; 0.806]
- geometry_only: P 0.902 [0.806; 0.978], R 0.725 [0.486; 0.882], F 0.804 [0.615; 0.911]
- vmf: P 0.950 [0.857; 1.000], R 0.745 [0.510; 0.904], F 0.835 [0.633; 0.939]
- isotropic_default: P 0.792 [0.688; 0.930], R 0.745 [0.550; 0.889], F 0.768 [0.630; 0.880]
- isotropic_calibrated: P 0.889 [0.720; 0.974], R 0.627 [0.417; 0.789], F 0.736 [0.548; 0.848]
- gate_propagated: P 0.652 [0.453; 0.903], R 0.588 [0.444; 0.746], F 0.619 [0.455; 0.778]
- boundary_cut: P 0.837 [0.690; 0.926], R 0.706 [0.455; 0.870], F 0.766 [0.552; 0.895]
- dinov2_s: P 0.923 [0.836; 0.983], R 0.774 [0.667; 0.873], F 0.842 [0.760; 0.915]
- threshold: P 0.310 [0.062; 0.679], R 0.176 [0.043; 0.288], F 0.225 [0.051; 0.400]
- visual_only: P 0.151 [0.060; 0.278], R 0.255 [0.135; 0.333], F 0.190 [0.083; 0.302]
- anyloc_vlad_g_debiased: P 0.811 [0.737; 0.889], R 0.782 [0.586; 0.931], F 0.796 [0.673; 0.895]
- anyloc_gem_g_debiased: P 0.860 [0.812; 0.943], R 0.811 [0.706; 0.927], F 0.835 [0.772; 0.906]
- dinov2_s_debiased: P 0.897 [0.815; 0.969], R 0.839 [0.768; 0.900], F 0.867 [0.809; 0.922]
- geometry_only_debiased: P 0.880 [0.787; 0.968], R 0.863 [0.723; 0.964], F 0.871 [0.778; 0.954]
- vmf_debiased: P 0.918 [0.846; 0.975], R 0.882 [0.739; 0.970], F 0.900 [0.812; 0.961]
- boundary_cut_debiased: P 0.913 [0.793; 0.983], R 0.824 [0.615; 0.977], F 0.866 [0.709; 0.969]
- gate_propagated_debiased: P 0.810 [0.615; 0.957], R 0.667 [0.556; 0.782], F 0.731 [0.593; 0.845]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.819 [0.621; 0.924]
- topo_debiased: AP 0.957 [0.874; 0.995]
- anyloc_vlad_g: AP 0.787 [0.613; 0.918]
- anyloc_gem_g: AP 0.819 [0.722; 0.947]
- geometry_only: AP 0.810 [0.623; 0.927]
- vmf: AP 0.848 [0.642; 0.960]
- isotropic_default: AP 0.781 [0.600; 0.912]
- isotropic_calibrated: AP 0.752 [0.551; 0.871]
- boundary_cut: AP 0.812 [0.621; 0.951]
- dinov2_s: AP 0.833 [0.722; 0.941]
- anyloc_vlad_g_debiased: AP 0.870 [0.765; 0.966]
- anyloc_gem_g_debiased: AP 0.884 [0.802; 0.976]
- dinov2_s_debiased: AP 0.898 [0.833; 0.959]
- geometry_only_debiased: AP 0.935 [0.848; 0.994]
- vmf_debiased: AP 0.979 [0.929; 0.996]
- boundary_cut_debiased: AP 0.911 [0.782; 0.987]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo_debiased − topo: +0.087 [+0.023; +0.190]
- P topo_debiased − topo: -0.007 [-0.077; +0.053]
- R topo_debiased − topo: +0.157 [+0.063; +0.295]
- F1 topo_debiased − anyloc_vlad_g_debiased: +0.104 [+0.019; +0.174]
- F1 topo_debiased − anyloc_gem_g_debiased: +0.065 [-0.086; +0.145]
- F1 topo_debiased − dinov2_s_debiased: +0.033 [-0.073; +0.124]
- F1 topo_debiased − geometry_only_debiased: +0.029 [-0.023; +0.075]
- F1 topo_debiased − vmf_debiased: +0.000 [-0.031; +0.024]
- F1 topo_debiased − boundary_cut_debiased: +0.034 [-0.059; +0.123]
- F1 topo_debiased − gate_propagated_debiased: +0.169 [+0.075; +0.249]
- F1 topo − anyloc_vlad_g: +0.102 [-0.020; +0.198]
- F1 topo − anyloc_gem_g: +0.034 [-0.220; +0.197]
- F1 topo − previous_system: +0.161 [-0.010; +0.392]
- P topo − previous_system: -0.014 [-0.167; +0.182]
- R topo − previous_system: +0.225 [+0.023; +0.447]
- F1 topo − gate_propagated: +0.195 [-0.024; +0.306]
- F1 topo − geometry_only: +0.009 [-0.057; +0.061]
- F1 topo − vmf: -0.022 [-0.064; +0.000]
- F1 topo − isotropic_calibrated: +0.078 [+0.019; +0.150]
- F1 topo − isotropic_default: +0.046 [-0.088; +0.131]
- F1 topo − boundary_cut: +0.047 [-0.049; +0.123]
- F1 topo − dinov2_s: -0.029 [-0.167; +0.074]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 12 | 10 | 28 | 19 | 3 |
| topo_debiased | 12 | 12 | 28 | 25 | 4 |
| anyloc_vlad_g | 10 | 9 | 29 | 16 | 12 |
| anyloc_gem_g | 13 | 11 | 24 | 14 | 5 |
| previous_system | 15 | 11 | 33 | 14 | 2 |
| geometry_only | 12 | 9 | 28 | 19 | 4 |
| vmf | 12 | 11 | 28 | 19 | 2 |
| isotropic_default | 12 | 10 | 28 | 20 | 10 |
| isotropic_calibrated | 12 | 6 | 28 | 18 | 4 |
| gate_propagated | 12 | 7 | 28 | 19 | 16 |
| boundary_cut | 12 | 10 | 27 | 18 | 7 |
| dinov2_s | 15 | 14 | 33 | 21 | 4 |
| threshold | 12 | 5 | 28 | 3 | 20 |
| visual_only | 12 | 3 | 28 | 6 | 73 |
| anyloc_vlad_g_debiased | 10 | 9 | 29 | 21 | 10 |
| anyloc_gem_g_debiased | 13 | 12 | 24 | 18 | 7 |
| dinov2_s_debiased | 15 | 14 | 33 | 25 | 6 |
| geometry_only_debiased | 12 | 10 | 28 | 25 | 6 |
| vmf_debiased | 12 | 12 | 28 | 25 | 4 |
| boundary_cut_debiased | 12 | 11 | 27 | 22 | 4 |
| gate_propagated_debiased | 12 | 10 | 28 | 21 | 8 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| freiburg_a | 32 | 10 | 0.462 | 0.737 | 0.952 | 0.714 | 0.530 | 0.767 | 0.981 | 1.000 | 2.915 |
| freiburg_ext | 16 | 4 | 0.000 | 0.571 | 0.800 | 0.727 | 0.478 | 0.768 | 0.959 | 0.986 | 1.858 |
| saarbruecken_a | 29 | 2 | 0.667 | 0.500 | 0.444 | 0.308 | 0.394 | 0.647 | 0.901 | 0.964 | 4.029 |
| saarbruecken_ext | 16 | 3 | 0.500 | 0.800 | 0.857 | 0.600 | 0.537 | 0.805 | 1.000 | 1.000 | 1.096 |
| cid_sims_apartment1_1 | 16 | 9 | 0.333 | 0.286 | 0.889 | 0.818 | 0.566 | 0.941 | 1.000 | 1.000 | 0.468 |
| cid_sims_apartment2_1 | 10 | 1 | 0.000 | 0.000 | 0.000 | 0.286 | 0.424 | 0.688 | 0.963 | 1.000 | 0.376 |
| cid_sims_apartment3_1 | 20 | 11 | 0.167 | 0.267 | 0.909 | 0.880 | 0.657 | 0.920 | 1.000 | 1.000 | 0.619 |

## Validation traversals (12)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 17 | 13 | 43 | 4 | 0.232 | 0.765 | 0.356 |
| topo_debiased | 217 | 17 | 16 | 50 | 1 | 0.242 | 0.941 | 0.386 |
| anyloc_vlad_g | 210 | 12 | 10 | 35 | 2 | 0.222 | 0.833 | 0.351 |
| anyloc_gem_g | 214 | 12 | 10 | 41 | 2 | 0.196 | 0.833 | 0.317 |
| previous_system | 223 | 16 | 8 | 25 | 8 | 0.242 | 0.500 | 0.327 |
| geometry_only | 217 | 17 | 12 | 45 | 5 | 0.211 | 0.706 | 0.324 |
| vmf | 217 | 17 | 13 | 45 | 4 | 0.224 | 0.765 | 0.347 |
| isotropic_default | 217 | 17 | 13 | 52 | 4 | 0.200 | 0.765 | 0.317 |
| isotropic_calibrated | 217 | 17 | 12 | 41 | 5 | 0.226 | 0.706 | 0.343 |
| gate_propagated | 217 | 17 | 11 | 43 | 6 | 0.204 | 0.647 | 0.310 |
| boundary_cut | 217 | 22 | 19 | 36 | 3 | 0.345 | 0.864 | 0.494 |
| dinov2_s | 223 | 16 | 13 | 43 | 3 | 0.232 | 0.812 | 0.361 |
| threshold | 217 | 17 | 2 | 42 | 15 | 0.045 | 0.118 | 0.066 |
| visual_only | 217 | 17 | 4 | 123 | 13 | 0.031 | 0.235 | 0.056 |
| anyloc_vlad_g_debiased | 210 | 12 | 12 | 40 | 0 | 0.231 | 1.000 | 0.375 |
| anyloc_gem_g_debiased | 214 | 12 | 10 | 50 | 2 | 0.167 | 0.833 | 0.278 |
| dinov2_s_debiased | 223 | 16 | 14 | 48 | 2 | 0.226 | 0.875 | 0.359 |
| geometry_only_debiased | 217 | 17 | 13 | 49 | 4 | 0.210 | 0.765 | 0.329 |
| vmf_debiased | 217 | 17 | 15 | 50 | 2 | 0.231 | 0.882 | 0.366 |
| boundary_cut_debiased | 217 | 22 | 22 | 42 | 0 | 0.344 | 1.000 | 0.512 |
| gate_propagated_debiased | 217 | 17 | 11 | 35 | 6 | 0.239 | 0.647 | 0.349 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 36 | 28 | 28 | 8 | 0.500 | 0.778 | 0.609 |
| topo_debiased | 217 | 36 | 34 | 32 | 2 | 0.515 | 0.944 | 0.667 |
| anyloc_vlad_g | 210 | 30 | 22 | 23 | 8 | 0.489 | 0.733 | 0.587 |
| anyloc_gem_g | 214 | 30 | 22 | 29 | 8 | 0.431 | 0.733 | 0.543 |
| previous_system | 223 | 29 | 14 | 19 | 15 | 0.424 | 0.483 | 0.452 |
| geometry_only | 217 | 36 | 27 | 30 | 9 | 0.474 | 0.750 | 0.581 |
| vmf | 217 | 36 | 28 | 30 | 8 | 0.483 | 0.778 | 0.596 |
| isotropic_default | 217 | 36 | 27 | 38 | 9 | 0.415 | 0.750 | 0.535 |
| isotropic_calibrated | 217 | 36 | 26 | 27 | 10 | 0.491 | 0.722 | 0.584 |
| gate_propagated | 217 | 36 | 21 | 33 | 15 | 0.389 | 0.583 | 0.467 |
| boundary_cut | 217 | 37 | 29 | 26 | 8 | 0.527 | 0.784 | 0.630 |
| dinov2_s | 223 | 29 | 24 | 32 | 5 | 0.429 | 0.828 | 0.565 |
| threshold | 217 | 36 | 2 | 42 | 34 | 0.045 | 0.056 | 0.050 |
| visual_only | 217 | 36 | 8 | 119 | 28 | 0.063 | 0.222 | 0.098 |
| anyloc_vlad_g_debiased | 210 | 30 | 27 | 25 | 3 | 0.519 | 0.900 | 0.659 |
| anyloc_gem_g_debiased | 214 | 30 | 26 | 34 | 4 | 0.433 | 0.867 | 0.578 |
| dinov2_s_debiased | 223 | 29 | 26 | 36 | 3 | 0.419 | 0.897 | 0.571 |
| geometry_only_debiased | 217 | 36 | 31 | 31 | 5 | 0.500 | 0.861 | 0.633 |
| vmf_debiased | 217 | 36 | 33 | 32 | 3 | 0.508 | 0.917 | 0.653 |
| boundary_cut_debiased | 217 | 37 | 35 | 29 | 2 | 0.547 | 0.946 | 0.693 |
| gate_propagated_debiased | 217 | 36 | 22 | 24 | 14 | 0.478 | 0.611 | 0.537 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 72 | 50 | 6 | 22 | 0.893 | 0.694 | 0.781 |
| topo_debiased | 217 | 72 | 60 | 6 | 12 | 0.909 | 0.833 | 0.870 |
| anyloc_vlad_g | 210 | 60 | 39 | 6 | 21 | 0.867 | 0.650 | 0.743 |
| anyloc_gem_g | 214 | 64 | 42 | 9 | 22 | 0.824 | 0.656 | 0.730 |
| previous_system | 223 | 80 | 26 | 7 | 54 | 0.788 | 0.325 | 0.460 |
| geometry_only | 217 | 72 | 50 | 7 | 22 | 0.877 | 0.694 | 0.775 |
| vmf | 217 | 72 | 51 | 7 | 21 | 0.879 | 0.708 | 0.785 |
| isotropic_default | 217 | 72 | 53 | 12 | 19 | 0.815 | 0.736 | 0.774 |
| isotropic_calibrated | 217 | 72 | 46 | 7 | 26 | 0.868 | 0.639 | 0.736 |
| gate_propagated | 217 | 72 | 35 | 19 | 37 | 0.648 | 0.486 | 0.556 |
| boundary_cut | 217 | 71 | 50 | 5 | 21 | 0.909 | 0.704 | 0.794 |
| dinov2_s | 223 | 80 | 50 | 6 | 30 | 0.893 | 0.625 | 0.735 |
| threshold | 217 | 72 | 4 | 40 | 68 | 0.091 | 0.056 | 0.069 |
| visual_only | 217 | 72 | 24 | 103 | 48 | 0.189 | 0.333 | 0.241 |
| anyloc_vlad_g_debiased | 210 | 60 | 48 | 4 | 12 | 0.923 | 0.800 | 0.857 |
| anyloc_gem_g_debiased | 214 | 64 | 52 | 8 | 12 | 0.867 | 0.812 | 0.839 |
| dinov2_s_debiased | 223 | 80 | 57 | 5 | 23 | 0.919 | 0.713 | 0.803 |
| geometry_only_debiased | 217 | 72 | 57 | 5 | 15 | 0.919 | 0.792 | 0.851 |
| vmf_debiased | 217 | 72 | 59 | 6 | 13 | 0.908 | 0.819 | 0.861 |
| boundary_cut_debiased | 217 | 71 | 60 | 4 | 11 | 0.938 | 0.845 | 0.889 |
| gate_propagated_debiased | 217 | 72 | 35 | 11 | 37 | 0.761 | 0.486 | 0.593 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 100 | 55 | 1 | 45 | 0.982 | 0.550 | 0.705 |
| topo_debiased | 217 | 100 | 66 | 0 | 34 | 1.000 | 0.660 | 0.795 |
| anyloc_vlad_g | 210 | 99 | 43 | 2 | 56 | 0.956 | 0.434 | 0.597 |
| anyloc_gem_g | 214 | 103 | 49 | 2 | 54 | 0.961 | 0.476 | 0.636 |
| previous_system | 223 | 106 | 29 | 4 | 77 | 0.879 | 0.274 | 0.417 |
| geometry_only | 217 | 100 | 56 | 1 | 44 | 0.982 | 0.560 | 0.713 |
| vmf | 217 | 100 | 57 | 1 | 43 | 0.983 | 0.570 | 0.722 |
| isotropic_default | 217 | 100 | 62 | 3 | 38 | 0.954 | 0.620 | 0.752 |
| isotropic_calibrated | 217 | 100 | 50 | 3 | 50 | 0.943 | 0.500 | 0.654 |
| gate_propagated | 217 | 100 | 46 | 8 | 54 | 0.852 | 0.460 | 0.597 |
| boundary_cut | 217 | 98 | 54 | 1 | 44 | 0.982 | 0.551 | 0.706 |
| dinov2_s | 223 | 106 | 55 | 1 | 51 | 0.982 | 0.519 | 0.679 |
| threshold | 217 | 100 | 6 | 38 | 94 | 0.136 | 0.060 | 0.083 |
| visual_only | 217 | 100 | 38 | 89 | 62 | 0.299 | 0.380 | 0.335 |
| anyloc_vlad_g_debiased | 210 | 99 | 52 | 0 | 47 | 1.000 | 0.525 | 0.689 |
| anyloc_gem_g_debiased | 214 | 103 | 60 | 0 | 43 | 1.000 | 0.583 | 0.736 |
| dinov2_s_debiased | 223 | 106 | 62 | 0 | 44 | 1.000 | 0.585 | 0.738 |
| geometry_only_debiased | 217 | 100 | 62 | 0 | 38 | 1.000 | 0.620 | 0.765 |
| vmf_debiased | 217 | 100 | 65 | 0 | 35 | 1.000 | 0.650 | 0.788 |
| boundary_cut_debiased | 217 | 98 | 64 | 0 | 34 | 1.000 | 0.653 | 0.790 |
| gate_propagated_debiased | 217 | 100 | 42 | 4 | 58 | 0.913 | 0.420 | 0.575 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.893 [0.833; 0.957], R 0.694 [0.580; 0.794], F 0.781 [0.699; 0.845]
- topo_debiased: P 0.909 [0.830; 0.971], R 0.833 [0.746; 0.909], F 0.870 [0.789; 0.936]
- anyloc_vlad_g: P 0.867 [0.789; 0.968], R 0.650 [0.527; 0.759], F 0.743 [0.645; 0.836]
- anyloc_gem_g: P 0.824 [0.745; 0.892], R 0.656 [0.529; 0.778], F 0.730 [0.629; 0.819]
- previous_system: P 0.788 [0.651; 1.000], R 0.325 [0.217; 0.433], F 0.460 [0.336; 0.578]
- geometry_only: P 0.877 [0.818; 0.937], R 0.694 [0.560; 0.795], F 0.775 [0.681; 0.837]
- vmf: P 0.879 [0.821; 0.939], R 0.708 [0.598; 0.800], F 0.785 [0.709; 0.840]
- isotropic_default: P 0.815 [0.750; 0.907], R 0.736 [0.636; 0.818], F 0.774 [0.711; 0.836]
- isotropic_calibrated: P 0.868 [0.790; 0.974], R 0.639 [0.508; 0.746], F 0.736 [0.646; 0.795]
- gate_propagated: P 0.648 [0.545; 0.784], R 0.486 [0.365; 0.568], F 0.556 [0.452; 0.634]
- boundary_cut: P 0.909 [0.821; 0.978], R 0.704 [0.578; 0.803], F 0.794 [0.692; 0.867]
- dinov2_s: P 0.893 [0.820; 0.977], R 0.625 [0.495; 0.732], F 0.735 [0.632; 0.811]
- threshold: P 0.091 [0.000; 0.240], R 0.056 [0.000; 0.125], F 0.069 [0.000; 0.161]
- visual_only: P 0.189 [0.132; 0.271], R 0.333 [0.246; 0.425], F 0.241 [0.177; 0.321]
- anyloc_vlad_g_debiased: P 0.923 [0.846; 1.000], R 0.800 [0.677; 0.912], F 0.857 [0.757; 0.944]
- anyloc_gem_g_debiased: P 0.867 [0.780; 0.935], R 0.812 [0.707; 0.931], F 0.839 [0.752; 0.920]
- dinov2_s_debiased: P 0.919 [0.860; 0.981], R 0.713 [0.580; 0.825], F 0.803 [0.701; 0.887]
- geometry_only_debiased: P 0.919 [0.829; 0.985], R 0.792 [0.681; 0.882], F 0.851 [0.753; 0.923]
- vmf_debiased: P 0.908 [0.824; 0.971], R 0.819 [0.724; 0.905], F 0.861 [0.773; 0.934]
- boundary_cut_debiased: P 0.938 [0.885; 0.983], R 0.845 [0.756; 0.913], F 0.889 [0.825; 0.937]
- gate_propagated_debiased: P 0.761 [0.611; 0.875], R 0.486 [0.353; 0.580], F 0.593 [0.457; 0.674]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.839 [0.766; 0.912]
- topo_debiased: AP 0.905 [0.808; 0.976]
- anyloc_vlad_g: AP 0.809 [0.649; 0.939]
- anyloc_gem_g: AP 0.831 [0.701; 0.939]
- geometry_only: AP 0.810 [0.715; 0.903]
- vmf: AP 0.825 [0.757; 0.902]
- isotropic_default: AP 0.776 [0.687; 0.862]
- isotropic_calibrated: AP 0.784 [0.680; 0.874]
- boundary_cut: AP 0.831 [0.715; 0.932]
- dinov2_s: AP 0.866 [0.763; 0.957]
- anyloc_vlad_g_debiased: AP 0.936 [0.845; 0.991]
- anyloc_gem_g_debiased: AP 0.918 [0.838; 0.978]
- dinov2_s_debiased: AP 0.905 [0.833; 0.976]
- geometry_only_debiased: AP 0.915 [0.812; 0.986]
- vmf_debiased: AP 0.903 [0.806; 0.972]
- boundary_cut_debiased: AP 0.916 [0.855; 0.964]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo_debiased − topo: +0.088 [+0.041; +0.138]
- P topo_debiased − topo: +0.016 [-0.064; +0.083]
- R topo_debiased − topo: +0.139 [+0.075; +0.216]
- F1 topo_debiased − anyloc_vlad_g_debiased: +0.012 [-0.080; +0.092]
- F1 topo_debiased − anyloc_gem_g_debiased: +0.031 [-0.045; +0.100]
- F1 topo_debiased − dinov2_s_debiased: +0.067 [-0.032; +0.152]
- F1 topo_debiased − geometry_only_debiased: +0.019 [-0.012; +0.073]
- F1 topo_debiased − vmf_debiased: +0.008 [+0.000; +0.026]
- F1 topo_debiased − boundary_cut_debiased: -0.019 [-0.065; +0.028]
- F1 topo_debiased − gate_propagated_debiased: +0.276 [+0.204; +0.388]
- F1 topo − anyloc_vlad_g: +0.038 [-0.071; +0.119]
- F1 topo − anyloc_gem_g: +0.051 [-0.062; +0.144]
- F1 topo − previous_system: +0.321 [+0.198; +0.414]
- P topo − previous_system: +0.105 [-0.093; +0.232]
- R topo − previous_system: +0.369 [+0.233; +0.462]
- F1 topo − gate_propagated: +0.226 [+0.141; +0.313]
- F1 topo − geometry_only: +0.006 [-0.022; +0.044]
- F1 topo − vmf: -0.003 [-0.027; +0.019]
- F1 topo − isotropic_calibrated: +0.045 [+0.008; +0.090]
- F1 topo − isotropic_default: +0.008 [-0.047; +0.045]
- F1 topo − boundary_cut: -0.012 [-0.068; +0.046]
- F1 topo − dinov2_s: +0.046 [-0.021; +0.108]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 5 | 2 | 41 | 27 | 6 |
| topo_debiased | 5 | 4 | 41 | 35 | 6 |
| anyloc_vlad_g | 3 | 3 | 39 | 22 | 6 |
| anyloc_gem_g | 5 | 5 | 44 | 25 | 9 |
| previous_system | 8 | 2 | 45 | 20 | 7 |
| geometry_only | 5 | 1 | 41 | 27 | 7 |
| vmf | 5 | 2 | 41 | 27 | 7 |
| isotropic_default | 5 | 2 | 41 | 29 | 12 |
| isotropic_calibrated | 5 | 1 | 41 | 27 | 7 |
| gate_propagated | 5 | 4 | 41 | 22 | 19 |
| boundary_cut | 5 | 3 | 43 | 30 | 5 |
| dinov2_s | 8 | 5 | 45 | 30 | 6 |
| threshold | 5 | 2 | 41 | 1 | 40 |
| visual_only | 5 | 2 | 41 | 7 | 103 |
| anyloc_vlad_g_debiased | 3 | 3 | 39 | 31 | 4 |
| anyloc_gem_g_debiased | 5 | 5 | 44 | 34 | 8 |
| dinov2_s_debiased | 8 | 6 | 45 | 35 | 5 |
| geometry_only_debiased | 5 | 1 | 41 | 35 | 5 |
| vmf_debiased | 5 | 3 | 41 | 35 | 6 |
| boundary_cut_debiased | 5 | 4 | 43 | 39 | 4 |
| gate_propagated_debiased | 5 | 4 | 41 | 23 | 11 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_freiburg_a_seq1 | 17 | 5 | 0.000 | 0.500 | 0.909 | 0.769 | 0.454 | 0.697 | 0.918 | 1.000 | 1.643 |
| heldout_freiburg_a_seq2 | 33 | 14 | 0.444 | 0.636 | 0.857 | 0.812 | 0.491 | 0.770 | 0.987 | 1.000 | 3.670 |
| heldout_freiburg_b_seq3 | 20 | 8 | 0.545 | 0.714 | 0.875 | 0.889 | 0.517 | 0.831 | 0.985 | 0.994 | 1.648 |
| heldout_saarbruecken_a_seq1 | 12 | 4 | 0.000 | 0.571 | 0.889 | 0.889 | 0.378 | 0.610 | 0.827 | 0.986 | 0.974 |
| heldout_saarbruecken_a_seq2 | 30 | 5 | 0.000 | 0.800 | 0.667 | 0.556 | 0.382 | 0.612 | 0.904 | 0.979 | 4.556 |
| heldout_saarbruecken_b_seq3 | 11 | 3 | 0.000 | 0.000 | 0.750 | 0.750 | 0.656 | 0.887 | 0.988 | 1.000 | 0.524 |
| heldout_saarbruecken_b_seq4 | 13 | 3 | 0.000 | 0.500 | 0.571 | 0.667 | 0.517 | 0.775 | 1.000 | 1.000 | 1.070 |
| heldout_ljubljana_a_seq1 | 15 | 1 | 1.000 | 1.000 | 1.000 | 0.667 | 0.152 | 0.268 | 0.462 | 0.576 | 2.168 |
| heldout_ljubljana_a_seq2 | 23 | 2 | 0.000 | 0.000 | 0.500 | 0.500 | 0.272 | 0.397 | 0.614 | 0.718 | 5.979 |
| heldout_cid_apartment1_2 | 9 | 1 | 0.667 | 0.667 | 0.500 | 0.400 | 0.251 | 0.465 | 0.827 | 0.943 | 0.948 |
| heldout_cid_apartment2_2 | 18 | 6 | 0.545 | 0.667 | 0.667 | 0.600 | 0.483 | 0.854 | 1.000 | 1.000 | 0.938 |
| heldout_cid_apartment3_2 | 16 | 4 | 0.400 | 0.400 | 0.889 | 0.615 | 0.458 | 0.765 | 0.991 | 1.000 | 1.237 |

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
  "heuristic_decision_accuracy": 0.3075,
  "n_map_query_pairs": 400,
  "present": {
   "answer_coverage": 0.3109,
   "mrr": 0.5784,
   "n": 238,
   "precision_at_1_when_answered": 0.4054,
   "recall_at_1": 0.4286,
   "recall_at_3": 0.7647,
   "successful_answer_rate": 0.1261
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
    "correct_rejection_rate": 0.5185,
    "false_answer_rate": 0.4815,
    "n": 54
   },
   "heuristic_decision_accuracy": 0.3,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.4783,
    "mrr": 0.529,
    "n": 46,
    "precision_at_1_when_answered": 0.0909,
    "recall_at_1": 0.2826,
    "recall_at_3": 0.8261,
    "successful_answer_rate": 0.0435
   }
  },
  "saarbruecken_a": {
   "absent": {
    "correct_rejection_rate": 0.5185,
    "false_answer_rate": 0.4815,
    "n": 27
   },
   "heuristic_decision_accuracy": 0.17,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.2466,
    "mrr": 0.5388,
    "n": 73,
    "precision_at_1_when_answered": 0.1667,
    "recall_at_1": 0.3836,
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
   "mrr": 0.6859,
   "n": 26,
   "precision_at_1_when_answered": 0.5,
   "recall_at_1": 0.5385,
   "recall_at_3": 0.8462,
   "successful_answer_rate": 0.1154
  },
  "function": {
   "answer_coverage": 0.4062,
   "mrr": 0.5156,
   "n": 32,
   "precision_at_1_when_answered": 0.5385,
   "recall_at_1": 0.4062,
   "recall_at_3": 0.6562,
   "successful_answer_rate": 0.2188
  },
  "indirect": {
   "answer_coverage": 0.1176,
   "mrr": 0.5735,
   "n": 34,
   "precision_at_1_when_answered": 0.25,
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
   "precision_at_1_when_answered": 0.2727,
   "recall_at_1": 0.4048,
   "recall_at_3": 0.7619,
   "successful_answer_rate": 0.0714
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
