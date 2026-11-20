# Manuscript results on the GPU

## Odometry calibration (development ground truth only)

Pairs per traversal: {'freiburg_a': 594, 'freiburg_ext': 291, 'saarbruecken_a': 955, 'saarbruecken_ext': 267, 'cid_sims_apartment1_1': 293, 'cid_sims_apartment2_1': 291, 'cid_sims_apartment3_1': 342, 'heldout_freiburg_a_seq1': 315, 'heldout_freiburg_a_seq2': 594, 'heldout_freiburg_b_seq3': 273, 'heldout_saarbruecken_a_seq1': 249, 'heldout_saarbruecken_a_seq2': 945, 'heldout_saarbruecken_b_seq3': 145, 'heldout_saarbruecken_b_seq4': 261, 'heldout_ljubljana_a_seq1': 1137, 'heldout_ljubljana_a_seq2': 1314, 'heldout_cid_apartment1_2': 370, 'heldout_cid_apartment2_2': 412, 'heldout_cid_apartment3_2': 398}

| Model | α1 | α2 | α3 | α4 | floor [m] | mean NLL |
|---|---|---|---|---|---|---|
| isotropic, original | 0.025 | 0.005 | 0.01 | 0.0025 | 0.950 | – |
| isotropic, calibrated | 0.025 (unused) | 0.005 (unused) | 0.002345 [0.000993; 0.0599] | 0.1202 [9.74e-07; 0.177] | 0.000 [0.000; 0.000] | 1.623 |
| propagated, calibrated | 0.00948 [2.65e-07; 0.0141] | 0.0009416 [1.52e-08; 0.00631] | 1.523e-08 [1.52e-08; 1.52e-08] | 0.02419 [0.0175; 0.0309] | 0.111 [0.080; 0.153] | 0.902 |

Leave-one-traversal-out mean NLL (lower is better):

| Traversal | isotropic_default | isotropic_calibrated | propagated_calibrated |
|---|---|---|---|
| freiburg_a | 5.009 | 2.372 | 1.360 |
| freiburg_ext | 3.395 | 1.657 | 0.577 |
| saarbruecken_a | 5.389 | 2.424 | 1.413 |
| saarbruecken_ext | 2.493 | 1.086 | 0.546 |
| cid_sims_apartment1_1 | 2.088 | 1.287 | 1.269 |
| cid_sims_apartment2_1 | 2.062 | 1.448 | 1.184 |
| cid_sims_apartment3_1 | 2.076 | 1.136 | 0.736 |
| heldout_freiburg_a_seq1 | 3.061 | 1.519 | 0.669 |
| heldout_freiburg_a_seq2 | 5.799 | 2.729 | 1.487 |
| heldout_freiburg_b_seq3 | 3.295 | 1.650 | 0.639 |
| heldout_saarbruecken_a_seq1 | 2.283 | 1.004 | 0.364 |
| heldout_saarbruecken_a_seq2 | 5.778 | 2.500 | 1.440 |
| heldout_saarbruecken_b_seq3 | 2.049 | 0.619 | 0.214 |
| heldout_saarbruecken_b_seq4 | 2.485 | 1.124 | 0.520 |
| heldout_ljubljana_a_seq1 | 2.549 | 1.386 | 0.561 |
| heldout_ljubljana_a_seq2 | 9.093 | 4.685 | 1.577 |
| heldout_cid_apartment1_2 | 2.188 | 1.429 | 0.842 |
| heldout_cid_apartment2_2 | 2.150 | 1.601 | 0.827 |
| heldout_cid_apartment3_2 | 2.330 | 1.605 | 1.644 |
| mean | 3.451 | 1.751 | 0.941 |

Fraction of true drifts inside the 95 % gate (with σ_place), by travelled distance [m]:

| Gate | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 32.0 | 64.0 | 128.0 |
|---|---|---|---|---|---|---|---|---|
| isotropic_default | 1.00 | 1.00 | 1.00 | 0.98 | 0.77 | 0.47 | 0.17 | 0.01 |
| isotropic_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.76 | 0.39 | 0.13 |
| propagated_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.82 | 0.91 |

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
| valley_k1.0 | 0.216 | 0.472 | 0.863 | 0.802 |
| valley_k1.5 | 0.250 | 0.462 | 0.812 | 0.748 |
| topo | 0.341 | 0.566 | 0.778 | 0.646 |
| valley_k2.5 | 0.320 | 0.500 | 0.711 | 0.567 |
| valley_k3.0 | 0.000 | 0.500 | 0.692 | 0.476 |
| valley_fixed | 0.272 | 0.567 | 0.843 | 0.792 |
| boundary_cut | 0.368 | 0.571 | 0.676 | 0.562 |

## Development traversals (7)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 9 | 7 | 25 | 2 | 0.219 | 0.778 | 0.341 |
| anyloc_vlad_g | 138 | 7 | 5 | 29 | 2 | 0.147 | 0.714 | 0.244 |
| anyloc_gem_g | 131 | 9 | 8 | 20 | 1 | 0.286 | 0.889 | 0.432 |
| previous_system | 147 | 9 | 5 | 22 | 4 | 0.185 | 0.556 | 0.278 |
| geometry_only | 139 | 9 | 7 | 24 | 2 | 0.226 | 0.778 | 0.350 |
| vmf | 139 | 9 | 8 | 24 | 1 | 0.250 | 0.889 | 0.390 |
| isotropic_default | 139 | 9 | 7 | 32 | 2 | 0.179 | 0.778 | 0.292 |
| isotropic_calibrated | 139 | 9 | 6 | 19 | 3 | 0.240 | 0.667 | 0.353 |
| gate_propagated | 139 | 9 | 7 | 32 | 2 | 0.179 | 0.778 | 0.292 |
| boundary_cut | 139 | 9 | 7 | 22 | 2 | 0.241 | 0.778 | 0.368 |
| dinov2_s | 147 | 9 | 6 | 35 | 3 | 0.146 | 0.667 | 0.240 |
| threshold | 139 | 9 | 4 | 20 | 5 | 0.167 | 0.444 | 0.242 |
| visual_only | 139 | 9 | 4 | 73 | 5 | 0.052 | 0.444 | 0.093 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 21 | 15 | 17 | 6 | 0.469 | 0.714 | 0.566 |
| anyloc_vlad_g | 138 | 20 | 15 | 19 | 5 | 0.441 | 0.750 | 0.556 |
| anyloc_gem_g | 131 | 16 | 13 | 15 | 3 | 0.464 | 0.812 | 0.591 |
| previous_system | 147 | 23 | 8 | 19 | 15 | 0.296 | 0.348 | 0.320 |
| geometry_only | 139 | 21 | 15 | 16 | 6 | 0.484 | 0.714 | 0.577 |
| vmf | 139 | 21 | 16 | 16 | 5 | 0.500 | 0.762 | 0.604 |
| isotropic_default | 139 | 21 | 15 | 24 | 6 | 0.385 | 0.714 | 0.500 |
| isotropic_calibrated | 139 | 21 | 12 | 13 | 9 | 0.480 | 0.571 | 0.522 |
| gate_propagated | 139 | 21 | 14 | 25 | 7 | 0.359 | 0.667 | 0.467 |
| boundary_cut | 139 | 20 | 14 | 15 | 6 | 0.483 | 0.700 | 0.571 |
| dinov2_s | 147 | 23 | 13 | 28 | 10 | 0.317 | 0.565 | 0.406 |
| threshold | 139 | 21 | 4 | 20 | 17 | 0.167 | 0.190 | 0.178 |
| visual_only | 139 | 21 | 6 | 71 | 15 | 0.078 | 0.286 | 0.122 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 40 | 28 | 4 | 12 | 0.875 | 0.700 | 0.778 |
| anyloc_vlad_g | 138 | 39 | 23 | 11 | 16 | 0.676 | 0.590 | 0.630 |
| anyloc_gem_g | 131 | 37 | 24 | 4 | 13 | 0.857 | 0.649 | 0.738 |
| previous_system | 147 | 48 | 25 | 2 | 23 | 0.926 | 0.521 | 0.667 |
| geometry_only | 139 | 40 | 27 | 4 | 13 | 0.871 | 0.675 | 0.761 |
| vmf | 139 | 40 | 29 | 3 | 11 | 0.906 | 0.725 | 0.806 |
| isotropic_default | 139 | 40 | 29 | 10 | 11 | 0.744 | 0.725 | 0.734 |
| isotropic_calibrated | 139 | 40 | 20 | 5 | 20 | 0.800 | 0.500 | 0.615 |
| gate_propagated | 139 | 40 | 24 | 15 | 16 | 0.615 | 0.600 | 0.608 |
| boundary_cut | 139 | 39 | 23 | 6 | 16 | 0.793 | 0.590 | 0.676 |
| dinov2_s | 147 | 48 | 35 | 6 | 13 | 0.854 | 0.729 | 0.787 |
| threshold | 139 | 40 | 6 | 18 | 34 | 0.250 | 0.150 | 0.188 |
| visual_only | 139 | 40 | 11 | 66 | 29 | 0.143 | 0.275 | 0.188 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 139 | 64 | 31 | 1 | 33 | 0.969 | 0.484 | 0.646 |
| anyloc_vlad_g | 138 | 66 | 32 | 2 | 34 | 0.941 | 0.485 | 0.640 |
| anyloc_gem_g | 131 | 59 | 27 | 1 | 32 | 0.964 | 0.458 | 0.621 |
| previous_system | 147 | 72 | 25 | 2 | 47 | 0.926 | 0.347 | 0.505 |
| geometry_only | 139 | 64 | 31 | 0 | 33 | 1.000 | 0.484 | 0.653 |
| vmf | 139 | 64 | 30 | 2 | 34 | 0.938 | 0.469 | 0.625 |
| isotropic_default | 139 | 64 | 35 | 4 | 29 | 0.897 | 0.547 | 0.680 |
| isotropic_calibrated | 139 | 64 | 23 | 2 | 41 | 0.920 | 0.359 | 0.517 |
| gate_propagated | 139 | 64 | 27 | 12 | 37 | 0.692 | 0.422 | 0.524 |
| boundary_cut | 139 | 67 | 27 | 2 | 40 | 0.931 | 0.403 | 0.562 |
| dinov2_s | 147 | 72 | 39 | 2 | 33 | 0.951 | 0.542 | 0.690 |
| threshold | 139 | 64 | 9 | 15 | 55 | 0.375 | 0.141 | 0.205 |
| visual_only | 139 | 64 | 18 | 59 | 46 | 0.234 | 0.281 | 0.255 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.875 [0.727; 0.976], R 0.700 [0.421; 0.889], F 0.778 [0.545; 0.912]
- anyloc_vlad_g: P 0.676 [0.545; 0.800], R 0.590 [0.388; 0.750], F 0.630 [0.475; 0.743]
- anyloc_gem_g: P 0.857 [0.778; 0.957], R 0.649 [0.435; 0.804], F 0.738 [0.571; 0.849]
- previous_system: P 0.926 [0.727; 1.000], R 0.521 [0.325; 0.722], F 0.667 [0.455; 0.837]
- geometry_only: P 0.871 [0.816; 0.955], R 0.675 [0.425; 0.840], F 0.761 [0.576; 0.853]
- vmf: P 0.906 [0.833; 0.975], R 0.725 [0.441; 0.911], F 0.806 [0.593; 0.925]
- isotropic_default: P 0.744 [0.635; 0.879], R 0.725 [0.500; 0.882], F 0.734 [0.590; 0.842]
- isotropic_calibrated: P 0.800 [0.647; 0.950], R 0.500 [0.324; 0.674], F 0.615 [0.444; 0.746]
- gate_propagated: P 0.615 [0.451; 0.840], R 0.600 [0.462; 0.763], F 0.608 [0.465; 0.776]
- boundary_cut: P 0.793 [0.714; 0.857], R 0.590 [0.333; 0.805], F 0.676 [0.459; 0.810]
- dinov2_s: P 0.854 [0.725; 0.955], R 0.729 [0.585; 0.846], F 0.787 [0.682; 0.870]
- threshold: P 0.250 [0.029; 0.640], R 0.150 [0.024; 0.260], F 0.188 [0.027; 0.367]
- visual_only: P 0.143 [0.062; 0.275], R 0.275 [0.140; 0.426], F 0.188 [0.088; 0.328]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.743 [0.524; 0.920]
- anyloc_vlad_g: AP 0.658 [0.472; 0.803]
- anyloc_gem_g: AP 0.725 [0.622; 0.770]
- geometry_only: AP 0.732 [0.540; 0.857]
- vmf: AP 0.783 [0.558; 0.935]
- isotropic_default: AP 0.699 [0.515; 0.838]
- isotropic_calibrated: AP 0.672 [0.458; 0.859]
- boundary_cut: AP 0.661 [0.449; 0.817]
- dinov2_s: AP 0.784 [0.628; 0.927]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − anyloc_vlad_g: +0.148 [+0.014; +0.272]
- F1 topo − anyloc_gem_g: +0.039 [-0.213; +0.235]
- F1 topo − previous_system: +0.111 [-0.142; +0.359]
- P topo − previous_system: -0.051 [-0.259; +0.167]
- R topo − previous_system: +0.179 [-0.087; +0.425]
- F1 topo − gate_propagated: +0.170 [-0.072; +0.329]
- F1 topo − geometry_only: +0.017 [-0.092; +0.110]
- F1 topo − vmf: -0.028 [-0.092; +0.024]
- F1 topo − isotropic_calibrated: +0.162 [+0.036; +0.279]
- F1 topo − isotropic_default: +0.044 [-0.144; +0.159]
- F1 topo − boundary_cut: +0.101 [-0.042; +0.219]
- F1 topo − dinov2_s: -0.009 [-0.193; +0.119]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 12 | 10 | 28 | 18 | 4 |
| anyloc_vlad_g | 10 | 9 | 29 | 14 | 11 |
| anyloc_gem_g | 13 | 11 | 24 | 13 | 4 |
| previous_system | 15 | 11 | 33 | 14 | 2 |
| geometry_only | 12 | 9 | 28 | 18 | 4 |
| vmf | 12 | 11 | 28 | 18 | 3 |
| isotropic_default | 12 | 10 | 28 | 19 | 10 |
| isotropic_calibrated | 12 | 5 | 28 | 15 | 5 |
| gate_propagated | 12 | 8 | 28 | 16 | 15 |
| boundary_cut | 12 | 8 | 27 | 15 | 6 |
| dinov2_s | 15 | 14 | 33 | 21 | 6 |
| threshold | 12 | 5 | 28 | 1 | 18 |
| visual_only | 12 | 5 | 28 | 6 | 66 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| freiburg_a | 32 | 8 | 0.400 | 0.714 | 0.875 | 0.636 | 0.530 | 0.767 | 0.981 | 1.000 | 2.915 |
| freiburg_ext | 16 | 3 | 0.000 | 0.667 | 0.750 | 0.667 | 0.478 | 0.768 | 0.959 | 0.986 | 1.858 |
| saarbruecken_a | 29 | 1 | 1.000 | 0.667 | 0.286 | 0.182 | 0.394 | 0.647 | 0.901 | 0.964 | 4.029 |
| saarbruecken_ext | 16 | 3 | 0.500 | 0.800 | 1.000 | 0.667 | 0.537 | 0.805 | 1.000 | 1.000 | 1.096 |
| cid_sims_apartment1_1 | 16 | 7 | 0.400 | 0.500 | 0.714 | 0.737 | 0.566 | 0.941 | 1.000 | 1.000 | 0.468 |
| cid_sims_apartment2_1 | 10 | 1 | 0.000 | 0.000 | 0.000 | 0.333 | 0.424 | 0.688 | 0.963 | 1.000 | 0.376 |
| cid_sims_apartment3_1 | 20 | 9 | 0.200 | 0.333 | 1.000 | 0.900 | 0.657 | 0.920 | 1.000 | 1.000 | 0.619 |

## Validation traversals (12)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 12 | 9 | 26 | 3 | 0.257 | 0.750 | 0.383 |
| anyloc_vlad_g | 210 | 7 | 5 | 23 | 2 | 0.179 | 0.714 | 0.286 |
| anyloc_gem_g | 214 | 8 | 6 | 29 | 2 | 0.171 | 0.750 | 0.279 |
| previous_system | 223 | 10 | 6 | 20 | 4 | 0.231 | 0.600 | 0.333 |
| geometry_only | 217 | 12 | 7 | 27 | 5 | 0.206 | 0.583 | 0.304 |
| vmf | 217 | 12 | 8 | 27 | 4 | 0.229 | 0.667 | 0.340 |
| isotropic_default | 217 | 12 | 8 | 34 | 4 | 0.190 | 0.667 | 0.296 |
| isotropic_calibrated | 217 | 12 | 8 | 26 | 4 | 0.235 | 0.667 | 0.348 |
| gate_propagated | 217 | 12 | 8 | 33 | 4 | 0.195 | 0.667 | 0.302 |
| boundary_cut | 217 | 14 | 11 | 23 | 3 | 0.324 | 0.786 | 0.458 |
| dinov2_s | 223 | 10 | 8 | 30 | 2 | 0.211 | 0.800 | 0.333 |
| threshold | 217 | 12 | 2 | 39 | 10 | 0.049 | 0.167 | 0.075 |
| visual_only | 217 | 12 | 3 | 102 | 9 | 0.029 | 0.250 | 0.051 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 24 | 17 | 18 | 7 | 0.486 | 0.708 | 0.576 |
| anyloc_vlad_g | 210 | 22 | 14 | 14 | 8 | 0.500 | 0.636 | 0.560 |
| anyloc_gem_g | 214 | 25 | 16 | 19 | 9 | 0.457 | 0.640 | 0.533 |
| previous_system | 223 | 19 | 9 | 17 | 10 | 0.346 | 0.474 | 0.400 |
| geometry_only | 217 | 24 | 15 | 19 | 9 | 0.441 | 0.625 | 0.517 |
| vmf | 217 | 24 | 16 | 19 | 8 | 0.457 | 0.667 | 0.542 |
| isotropic_default | 217 | 24 | 15 | 27 | 9 | 0.357 | 0.625 | 0.455 |
| isotropic_calibrated | 217 | 24 | 15 | 19 | 9 | 0.441 | 0.625 | 0.517 |
| gate_propagated | 217 | 24 | 13 | 28 | 11 | 0.317 | 0.542 | 0.400 |
| boundary_cut | 217 | 26 | 19 | 15 | 7 | 0.559 | 0.731 | 0.633 |
| dinov2_s | 223 | 19 | 14 | 24 | 5 | 0.368 | 0.737 | 0.491 |
| threshold | 217 | 24 | 2 | 39 | 22 | 0.049 | 0.083 | 0.062 |
| visual_only | 217 | 24 | 5 | 100 | 19 | 0.048 | 0.208 | 0.078 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 46 | 29 | 6 | 17 | 0.829 | 0.630 | 0.716 |
| anyloc_vlad_g | 210 | 42 | 24 | 4 | 18 | 0.857 | 0.571 | 0.686 |
| anyloc_gem_g | 214 | 49 | 29 | 6 | 20 | 0.829 | 0.592 | 0.690 |
| previous_system | 223 | 53 | 18 | 8 | 35 | 0.692 | 0.340 | 0.456 |
| geometry_only | 217 | 46 | 27 | 7 | 19 | 0.794 | 0.587 | 0.675 |
| vmf | 217 | 46 | 28 | 7 | 18 | 0.800 | 0.609 | 0.691 |
| isotropic_default | 217 | 46 | 30 | 12 | 16 | 0.714 | 0.652 | 0.682 |
| isotropic_calibrated | 217 | 46 | 27 | 7 | 19 | 0.794 | 0.587 | 0.675 |
| gate_propagated | 217 | 46 | 21 | 20 | 25 | 0.512 | 0.457 | 0.483 |
| boundary_cut | 217 | 48 | 31 | 3 | 17 | 0.912 | 0.646 | 0.756 |
| dinov2_s | 223 | 53 | 32 | 6 | 21 | 0.842 | 0.604 | 0.703 |
| threshold | 217 | 46 | 3 | 38 | 43 | 0.073 | 0.065 | 0.069 |
| visual_only | 217 | 46 | 10 | 95 | 36 | 0.095 | 0.217 | 0.132 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 217 | 72 | 34 | 1 | 38 | 0.971 | 0.472 | 0.636 |
| anyloc_vlad_g | 210 | 70 | 26 | 2 | 44 | 0.929 | 0.371 | 0.531 |
| anyloc_gem_g | 214 | 71 | 32 | 3 | 39 | 0.914 | 0.451 | 0.604 |
| previous_system | 223 | 80 | 21 | 5 | 59 | 0.808 | 0.263 | 0.396 |
| geometry_only | 217 | 72 | 33 | 1 | 39 | 0.971 | 0.458 | 0.623 |
| vmf | 217 | 72 | 34 | 1 | 38 | 0.971 | 0.472 | 0.636 |
| isotropic_default | 217 | 72 | 39 | 3 | 33 | 0.929 | 0.542 | 0.684 |
| isotropic_calibrated | 217 | 72 | 31 | 3 | 41 | 0.912 | 0.431 | 0.585 |
| gate_propagated | 217 | 72 | 32 | 9 | 40 | 0.780 | 0.444 | 0.566 |
| boundary_cut | 217 | 74 | 33 | 1 | 41 | 0.971 | 0.446 | 0.611 |
| dinov2_s | 223 | 80 | 37 | 1 | 43 | 0.974 | 0.463 | 0.627 |
| threshold | 217 | 72 | 5 | 36 | 67 | 0.122 | 0.069 | 0.088 |
| visual_only | 217 | 72 | 18 | 87 | 54 | 0.171 | 0.250 | 0.203 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.829 [0.742; 0.952], R 0.630 [0.439; 0.775], F 0.716 [0.578; 0.811]
- anyloc_vlad_g: P 0.857 [0.786; 0.955], R 0.571 [0.471; 0.676], F 0.686 [0.600; 0.772]
- anyloc_gem_g: P 0.829 [0.675; 0.966], R 0.592 [0.458; 0.738], F 0.690 [0.564; 0.814]
- previous_system: P 0.692 [0.478; 1.000], R 0.340 [0.229; 0.475], F 0.456 [0.323; 0.620]
- geometry_only: P 0.794 [0.700; 0.909], R 0.587 [0.377; 0.758], F 0.675 [0.512; 0.780]
- vmf: P 0.800 [0.714; 0.917], R 0.609 [0.421; 0.766], F 0.691 [0.553; 0.787]
- isotropic_default: P 0.714 [0.611; 0.878], R 0.652 [0.469; 0.809], F 0.682 [0.560; 0.808]
- isotropic_calibrated: P 0.794 [0.660; 1.000], R 0.587 [0.417; 0.733], F 0.675 [0.553; 0.793]
- gate_propagated: P 0.512 [0.345; 0.720], R 0.457 [0.318; 0.568], F 0.483 [0.338; 0.621]
- boundary_cut: P 0.912 [0.778; 1.000], R 0.646 [0.477; 0.788], F 0.756 [0.613; 0.867]
- dinov2_s: P 0.842 [0.732; 0.966], R 0.604 [0.471; 0.727], F 0.703 [0.590; 0.810]
- threshold: P 0.073 [0.000; 0.222], R 0.065 [0.000; 0.173], F 0.069 [0.000; 0.187]
- visual_only: P 0.095 [0.043; 0.143], R 0.217 [0.129; 0.273], F 0.132 [0.066; 0.183]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.751 [0.625; 0.880]
- anyloc_vlad_g: AP 0.702 [0.525; 0.885]
- anyloc_gem_g: AP 0.779 [0.603; 0.943]
- geometry_only: AP 0.709 [0.548; 0.866]
- vmf: AP 0.731 [0.607; 0.863]
- isotropic_default: AP 0.671 [0.508; 0.819]
- isotropic_calibrated: AP 0.682 [0.527; 0.833]
- boundary_cut: AP 0.744 [0.559; 0.926]
- dinov2_s: AP 0.797 [0.676; 0.930]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − anyloc_vlad_g: +0.030 [-0.104; +0.122]
- F1 topo − anyloc_gem_g: +0.026 [-0.132; +0.159]
- F1 topo − previous_system: +0.260 [+0.053; +0.409]
- P topo − previous_system: +0.136 [-0.123; +0.310]
- R topo − previous_system: +0.291 [+0.063; +0.470]
- F1 topo − gate_propagated: +0.233 [+0.114; +0.353]
- F1 topo − geometry_only: +0.041 [+0.000; +0.108]
- F1 topo − vmf: +0.025 [+0.000; +0.068]
- F1 topo − isotropic_calibrated: +0.041 [+0.000; +0.083]
- F1 topo − isotropic_default: +0.034 [-0.066; +0.093]
- F1 topo − boundary_cut: -0.040 [-0.097; +0.003]
- F1 topo − dinov2_s: +0.013 [-0.084; +0.084]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 5 | 3 | 41 | 26 | 6 |
| anyloc_vlad_g | 3 | 3 | 39 | 21 | 4 |
| anyloc_gem_g | 5 | 5 | 44 | 24 | 6 |
| previous_system | 8 | 1 | 45 | 17 | 8 |
| geometry_only | 5 | 1 | 41 | 26 | 7 |
| vmf | 5 | 2 | 41 | 26 | 7 |
| isotropic_default | 5 | 2 | 41 | 28 | 12 |
| isotropic_calibrated | 5 | 2 | 41 | 25 | 7 |
| gate_propagated | 5 | 3 | 41 | 18 | 20 |
| boundary_cut | 5 | 3 | 43 | 28 | 3 |
| dinov2_s | 8 | 5 | 45 | 27 | 6 |
| threshold | 5 | 2 | 41 | 1 | 38 |
| visual_only | 5 | 2 | 41 | 8 | 95 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_freiburg_a_seq1 | 17 | 4 | 0.000 | 0.857 | 0.889 | 0.889 | 0.454 | 0.697 | 0.918 | 1.000 | 1.643 |
| heldout_freiburg_a_seq2 | 33 | 11 | 0.429 | 0.400 | 0.762 | 0.800 | 0.491 | 0.770 | 0.987 | 1.000 | 3.670 |
| heldout_freiburg_b_seq3 | 20 | 6 | 0.667 | 0.727 | 0.833 | 0.923 | 0.517 | 0.831 | 0.985 | 0.994 | 1.648 |
| heldout_saarbruecken_a_seq1 | 12 | 2 | 0.000 | 0.500 | 0.800 | 0.667 | 0.378 | 0.610 | 0.827 | 0.986 | 0.974 |
| heldout_saarbruecken_a_seq2 | 30 | 2 | 0.000 | 0.667 | 0.400 | 0.286 | 0.382 | 0.612 | 0.904 | 0.979 | 4.556 |
| heldout_saarbruecken_b_seq3 | 11 | 1 | 0.000 | 0.000 | 0.667 | 0.500 | 0.656 | 0.887 | 0.988 | 1.000 | 0.524 |
| heldout_saarbruecken_b_seq4 | 13 | 1 | 0.000 | 0.000 | 0.000 | 0.400 | 0.517 | 0.775 | 1.000 | 1.000 | 1.070 |
| heldout_ljubljana_a_seq1 | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.152 | 0.268 | 0.462 | 0.576 | 2.168 |
| heldout_ljubljana_a_seq2 | 23 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.272 | 0.397 | 0.614 | 0.718 | 5.979 |
| heldout_cid_apartment1_2 | 9 | 1 | 0.667 | 0.667 | 0.500 | 0.400 | 0.251 | 0.465 | 0.827 | 0.943 | 0.948 |
| heldout_cid_apartment2_2 | 18 | 5 | 0.500 | 0.667 | 0.727 | 0.588 | 0.483 | 0.854 | 1.000 | 1.000 | 0.938 |
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
