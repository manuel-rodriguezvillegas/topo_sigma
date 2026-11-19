# Manuscript results on the GPU

## Odometry calibration (development ground truth only)

Pairs per traversal: {'freiburg_a': 594, 'freiburg_ext': 309, 'saarbruecken_a': 961, 'saarbruecken_ext': 267, 'cid_sims_apartment1_1': 293, 'cid_sims_apartment2_1': 291, 'cid_sims_apartment3_1': 342}

| Model | α1 | α2 | α3 | α4 | floor [m] | mean NLL |
|---|---|---|---|---|---|---|
| isotropic, original | 0.025 | 0.005 | 0.01 | 0.0025 | 0.859 | – |
| isotropic, calibrated | 0.025 (unused) | 0.005 (unused) | 0.06281 [0.00138; 0.0835] | 9.544e-05 [2.76e-08; 0.095] | 0.000 [0.000; 0.000] | 1.689 |
| propagated, calibrated | 0.001697 [1.52e-08; 0.00922] | 0.005295 [0.00136; 0.00729] | 0.01631 [0.00844; 0.0282] | 1.357e-07 [1.36e-07; 1.36e-07] | 0.171 [0.100; 0.227] | 1.092 |

Leave-one-traversal-out mean NLL (lower is better):

| Traversal | isotropic_default | isotropic_calibrated | propagated_calibrated |
|---|---|---|---|
| freiburg_a | 5.865 | 2.662 | 1.533 |
| freiburg_ext | 3.679 | 2.329 | 1.968 |
| saarbruecken_a | 6.156 | 2.769 | 1.555 |
| saarbruecken_ext | 2.451 | 1.239 | 0.676 |
| cid_sims_apartment1_1 | 2.086 | 1.148 | 1.110 |
| cid_sims_apartment2_1 | 2.065 | 1.115 | 1.091 |
| cid_sims_apartment3_1 | 2.078 | 1.047 | 0.679 |
| mean | 3.483 | 1.759 | 1.230 |

Fraction of true drifts inside the 95 % gate (with σ_place), by travelled distance [m]:

| Gate | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 32.0 | 64.0 | 128.0 |
|---|---|---|---|---|---|---|---|---|
| isotropic_default | 1.00 | 1.00 | 1.00 | 0.99 | 0.75 | 0.40 | 0.04 | 0.00 |
| isotropic_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.69 | 0.21 | 0.00 |
| propagated_calibrated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.66 | 0.71 |

## Node creation sensitivity (development)

Fixed δ calibrated on freiburg_a: 0.1301

| Detector | Nodes | ρ | CV | R_trans | Purity | Seg./episode |
|---|---|---|---|---|---|---|
| adaptive 1 | 217 | 16.943 | 0.134 | 0.643 | 0.908 | 3.508 |
| adaptive 1.5 | 183 | 14.265 | 0.134 | 0.580 | 0.895 | 3.267 |
| adaptive 2 | 149 | 11.464 | 0.138 | 0.467 | 0.859 | 2.943 |
| adaptive 2.5 | 119 | 9.057 | 0.103 | 0.417 | 0.826 | 2.534 |
| adaptive 3 | 99 | 7.318 | 0.160 | 0.368 | 0.765 | 2.253 |
| fixed 0.1301 | 158 | 12.341 | 0.235 | 0.596 | 0.896 | 3.099 |

Loop closure F1 of the mapper run with each detector:

| Detector | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 |
|---|---|---|---|---|
| valley_k1.0 | 0.321 | 0.496 | 0.823 | 0.748 |
| valley_k1.5 | 0.243 | 0.444 | 0.791 | 0.730 |
| topo | 0.217 | 0.361 | 0.791 | 0.691 |
| valley_k2.5 | 0.167 | 0.296 | 0.723 | 0.618 |
| valley_k3.0 | 0.250 | 0.353 | 0.690 | 0.520 |
| valley_fixed | 0.275 | 0.455 | 0.750 | 0.689 |

## Development traversals (7)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 149 | 8 | 5 | 33 | 3 | 0.132 | 0.625 | 0.217 |
| topo_visual | 149 | 8 | 6 | 34 | 2 | 0.150 | 0.750 | 0.250 |
| topo_visual_cosine | 149 | 8 | 6 | 34 | 2 | 0.150 | 0.750 | 0.250 |
| disk_isotropic_default | 149 | 8 | 5 | 40 | 3 | 0.111 | 0.625 | 0.189 |
| disk_isotropic_calibrated | 149 | 8 | 5 | 32 | 3 | 0.135 | 0.625 | 0.222 |
| gate_propagated | 149 | 8 | 4 | 30 | 4 | 0.118 | 0.500 | 0.190 |
| gate_propagated_geometry_only | 149 | 8 | 7 | 46 | 1 | 0.132 | 0.875 | 0.230 |
| paper | 149 | 8 | 5 | 22 | 3 | 0.185 | 0.625 | 0.286 |
| paper_geometry_only | 149 | 8 | 7 | 28 | 1 | 0.200 | 0.875 | 0.326 |
| threshold | 149 | 8 | 1 | 36 | 7 | 0.027 | 0.125 | 0.044 |
| visual_only | 149 | 8 | 0 | 63 | 8 | 0.000 | 0.000 | 0.000 |
| anyloc_vlad_g | 142 | 8 | 4 | 28 | 4 | 0.125 | 0.500 | 0.200 |
| anyloc_vlad_g_visual | 142 | 8 | 7 | 27 | 1 | 0.206 | 0.875 | 0.333 |
| anyloc_gem_g | 132 | 10 | 7 | 19 | 3 | 0.269 | 0.700 | 0.389 |
| anyloc_gem_g_visual | 132 | 10 | 9 | 20 | 1 | 0.310 | 0.900 | 0.462 |
| dinov3_s | 141 | 9 | 8 | 24 | 1 | 0.250 | 0.889 | 0.390 |
| dinov3_s_visual | 141 | 9 | 9 | 24 | 0 | 0.273 | 1.000 | 0.429 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 149 | 23 | 11 | 27 | 12 | 0.289 | 0.478 | 0.361 |
| topo_visual | 149 | 23 | 13 | 27 | 10 | 0.325 | 0.565 | 0.413 |
| topo_visual_cosine | 149 | 23 | 13 | 27 | 10 | 0.325 | 0.565 | 0.413 |
| disk_isotropic_default | 149 | 23 | 14 | 31 | 9 | 0.311 | 0.609 | 0.412 |
| disk_isotropic_calibrated | 149 | 23 | 11 | 26 | 12 | 0.297 | 0.478 | 0.367 |
| gate_propagated | 149 | 23 | 6 | 28 | 17 | 0.176 | 0.261 | 0.211 |
| gate_propagated_geometry_only | 149 | 23 | 15 | 38 | 8 | 0.283 | 0.652 | 0.395 |
| paper | 149 | 23 | 8 | 19 | 15 | 0.296 | 0.348 | 0.320 |
| paper_geometry_only | 149 | 23 | 16 | 19 | 7 | 0.457 | 0.696 | 0.552 |
| threshold | 149 | 23 | 5 | 32 | 18 | 0.135 | 0.217 | 0.167 |
| visual_only | 149 | 23 | 1 | 62 | 22 | 0.016 | 0.043 | 0.023 |
| anyloc_vlad_g | 142 | 21 | 14 | 18 | 7 | 0.438 | 0.667 | 0.528 |
| anyloc_vlad_g_visual | 142 | 21 | 16 | 18 | 5 | 0.471 | 0.762 | 0.582 |
| anyloc_gem_g | 132 | 17 | 12 | 14 | 5 | 0.462 | 0.706 | 0.558 |
| anyloc_gem_g_visual | 132 | 17 | 14 | 15 | 3 | 0.483 | 0.824 | 0.609 |
| dinov3_s | 141 | 21 | 16 | 16 | 5 | 0.500 | 0.762 | 0.604 |
| dinov3_s_visual | 141 | 21 | 17 | 16 | 4 | 0.515 | 0.810 | 0.630 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 149 | 48 | 34 | 4 | 14 | 0.895 | 0.708 | 0.791 |
| topo_visual | 149 | 48 | 35 | 5 | 13 | 0.875 | 0.729 | 0.795 |
| topo_visual_cosine | 149 | 48 | 36 | 4 | 12 | 0.900 | 0.750 | 0.818 |
| disk_isotropic_default | 149 | 48 | 35 | 10 | 13 | 0.778 | 0.729 | 0.753 |
| disk_isotropic_calibrated | 149 | 48 | 32 | 5 | 16 | 0.865 | 0.667 | 0.753 |
| gate_propagated | 149 | 48 | 21 | 13 | 27 | 0.618 | 0.438 | 0.512 |
| gate_propagated_geometry_only | 149 | 48 | 31 | 22 | 17 | 0.585 | 0.646 | 0.614 |
| paper | 149 | 48 | 25 | 2 | 23 | 0.926 | 0.521 | 0.667 |
| paper_geometry_only | 149 | 48 | 29 | 6 | 19 | 0.829 | 0.604 | 0.699 |
| threshold | 149 | 48 | 9 | 28 | 39 | 0.243 | 0.188 | 0.212 |
| visual_only | 149 | 48 | 6 | 57 | 42 | 0.095 | 0.125 | 0.108 |
| anyloc_vlad_g | 142 | 40 | 23 | 9 | 17 | 0.719 | 0.575 | 0.639 |
| anyloc_vlad_g_visual | 142 | 40 | 24 | 10 | 16 | 0.706 | 0.600 | 0.649 |
| anyloc_gem_g | 132 | 38 | 23 | 3 | 15 | 0.885 | 0.605 | 0.719 |
| anyloc_gem_g_visual | 132 | 38 | 25 | 4 | 13 | 0.862 | 0.658 | 0.746 |
| dinov3_s | 141 | 41 | 28 | 4 | 13 | 0.875 | 0.683 | 0.767 |
| dinov3_s_visual | 141 | 41 | 31 | 2 | 10 | 0.939 | 0.756 | 0.838 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 149 | 72 | 38 | 0 | 34 | 1.000 | 0.528 | 0.691 |
| topo_visual | 149 | 72 | 38 | 2 | 34 | 0.950 | 0.528 | 0.679 |
| topo_visual_cosine | 149 | 72 | 38 | 2 | 34 | 0.950 | 0.528 | 0.679 |
| disk_isotropic_default | 149 | 72 | 40 | 5 | 32 | 0.889 | 0.556 | 0.684 |
| disk_isotropic_calibrated | 149 | 72 | 35 | 2 | 37 | 0.946 | 0.486 | 0.642 |
| gate_propagated | 149 | 72 | 27 | 7 | 45 | 0.794 | 0.375 | 0.509 |
| gate_propagated_geometry_only | 149 | 72 | 36 | 17 | 36 | 0.679 | 0.500 | 0.576 |
| paper | 149 | 72 | 25 | 2 | 47 | 0.926 | 0.347 | 0.505 |
| paper_geometry_only | 149 | 72 | 30 | 5 | 42 | 0.857 | 0.417 | 0.561 |
| threshold | 149 | 72 | 14 | 23 | 58 | 0.378 | 0.194 | 0.257 |
| visual_only | 149 | 72 | 18 | 45 | 54 | 0.286 | 0.250 | 0.267 |
| anyloc_vlad_g | 142 | 67 | 31 | 1 | 36 | 0.969 | 0.463 | 0.626 |
| anyloc_vlad_g_visual | 142 | 67 | 31 | 3 | 36 | 0.912 | 0.463 | 0.614 |
| anyloc_gem_g | 132 | 59 | 26 | 0 | 33 | 1.000 | 0.441 | 0.612 |
| anyloc_gem_g_visual | 132 | 59 | 28 | 1 | 31 | 0.966 | 0.475 | 0.636 |
| dinov3_s | 141 | 65 | 32 | 0 | 33 | 1.000 | 0.492 | 0.660 |
| dinov3_s_visual | 141 | 65 | 32 | 1 | 33 | 0.970 | 0.492 | 0.653 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.895 [0.806; 0.969], R 0.708 [0.547; 0.846], F 0.791 [0.667; 0.879]
- topo_visual: P 0.875 [0.774; 0.957], R 0.729 [0.585; 0.846], F 0.795 [0.693; 0.879]
- topo_visual_cosine: P 0.900 [0.805; 0.978], R 0.750 [0.615; 0.869], F 0.818 [0.723; 0.896]
- disk_isotropic_default: P 0.778 [0.648; 0.886], R 0.729 [0.585; 0.846], F 0.753 [0.634; 0.857]
- disk_isotropic_calibrated: P 0.865 [0.741; 0.953], R 0.667 [0.491; 0.814], F 0.753 [0.613; 0.849]
- gate_propagated: P 0.618 [0.448; 0.765], R 0.438 [0.283; 0.600], F 0.512 [0.351; 0.658]
- gate_propagated_geometry_only: P 0.585 [0.375; 0.818], R 0.646 [0.458; 0.840], F 0.614 [0.415; 0.817]
- paper: P 0.926 [0.727; 1.000], R 0.521 [0.325; 0.722], F 0.667 [0.455; 0.837]
- paper_geometry_only: P 0.829 [0.600; 1.000], R 0.604 [0.409; 0.811], F 0.699 [0.500; 0.885]
- threshold: P 0.243 [0.059; 0.519], R 0.188 [0.055; 0.327], F 0.212 [0.057; 0.388]
- visual_only: P 0.095 [0.010; 0.281], R 0.125 [0.018; 0.269], F 0.108 [0.013; 0.271]
- anyloc_vlad_g: P 0.719 [0.542; 0.838], R 0.575 [0.333; 0.773], F 0.639 [0.419; 0.789]
- anyloc_vlad_g_visual: P 0.706 [0.560; 0.828], R 0.600 [0.400; 0.763], F 0.649 [0.481; 0.765]
- anyloc_gem_g: P 0.885 [0.833; 1.000], R 0.605 [0.391; 0.733], F 0.719 [0.556; 0.800]
- anyloc_gem_g_visual: P 0.862 [0.781; 0.958], R 0.658 [0.455; 0.810], F 0.746 [0.588; 0.853]
- dinov3_s: P 0.875 [0.818; 0.960], R 0.683 [0.440; 0.841], F 0.767 [0.593; 0.857]
- dinov3_s_visual: P 0.939 [0.886; 1.000], R 0.756 [0.500; 0.923], F 0.838 [0.655; 0.941]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.761 [0.618; 0.880]
- topo_visual: AP 0.781 [0.627; 0.926]
- topo_visual_cosine: AP 0.810 [0.656; 0.957]
- disk_isotropic_default: AP 0.736 [0.578; 0.892]
- disk_isotropic_calibrated: AP 0.747 [0.605; 0.877]
- anyloc_vlad_g: AP 0.669 [0.468; 0.803]
- anyloc_vlad_g_visual: AP 0.654 [0.477; 0.783]
- anyloc_gem_g: AP 0.653 [0.533; 0.731]
- anyloc_gem_g_visual: AP 0.704 [0.641; 0.751]
- dinov3_s: AP 0.738 [0.560; 0.863]
- dinov3_s_visual: AP 0.789 [0.578; 0.935]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − paper: +0.124 [-0.003; +0.294]
- P topo − paper: -0.031 [-0.167; +0.136]
- R topo − paper: +0.188 [+0.055; +0.341]
- F1 topo − disk_isotropic_default: +0.038 [-0.029; +0.092]
- F1 topo − disk_isotropic_calibrated: +0.038 [+0.008; +0.080]
- F1 topo − gate_propagated: +0.279 [+0.154; +0.388]
- F1 topo_visual − topo: +0.005 [-0.025; +0.048]
- F1 topo_visual_cosine − topo: +0.027 [-0.025; +0.132]
- F1 topo − anyloc_vlad_g: +0.152 [+0.046; +0.310]
- F1 topo_visual_cosine − anyloc_vlad_g_visual: +0.170 [+0.038; +0.319]
- F1 topo − anyloc_gem_g: +0.072 [+0.028; +0.133]
- F1 topo_visual_cosine − anyloc_gem_g_visual: +0.072 [+0.005; +0.150]
- F1 topo − dinov3_s: +0.024 [-0.056; +0.109]
- F1 topo_visual − dinov3_s_visual: -0.042 [-0.129; +0.075]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 15 | 12 | 33 | 22 | 4 |
| topo_visual | 15 | 14 | 33 | 21 | 5 |
| topo_visual_cosine | 15 | 14 | 33 | 22 | 4 |
| disk_isotropic_default | 15 | 13 | 33 | 22 | 10 |
| disk_isotropic_calibrated | 15 | 11 | 33 | 21 | 5 |
| gate_propagated | 15 | 9 | 33 | 12 | 13 |
| gate_propagated_geometry_only | 15 | 12 | 33 | 19 | 22 |
| paper | 15 | 11 | 33 | 14 | 2 |
| paper_geometry_only | 15 | 12 | 33 | 17 | 6 |
| threshold | 15 | 7 | 33 | 2 | 28 |
| visual_only | 15 | 6 | 33 | 0 | 57 |
| anyloc_vlad_g | 10 | 8 | 30 | 15 | 9 |
| anyloc_vlad_g_visual | 10 | 9 | 30 | 15 | 10 |
| anyloc_gem_g | 13 | 10 | 25 | 13 | 3 |
| anyloc_gem_g_visual | 13 | 11 | 25 | 14 | 4 |
| dinov3_s | 12 | 9 | 29 | 19 | 4 |
| dinov3_s_visual | 12 | 12 | 29 | 19 | 2 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| freiburg_a | 30 | 6 | 0.500 | 0.667 | 0.769 | 0.632 | 0.491 | 0.771 | 0.977 | 1.000 | 3.792 |
| freiburg_ext | 22 | 5 | 0.571 | 0.444 | 0.833 | 0.714 | 0.646 | 0.893 | 0.969 | 1.000 | 1.886 |
| saarbruecken_a | 32 | 4 | 0.000 | 0.250 | 0.615 | 0.471 | 0.453 | 0.685 | 0.911 | 0.998 | 4.179 |
| saarbruecken_ext | 16 | 3 | 0.000 | 0.500 | 0.800 | 0.667 | 0.545 | 0.808 | 0.999 | 1.000 | 1.039 |
| cid_sims_apartment1_1 | 18 | 8 | 0.182 | 0.308 | 0.941 | 0.800 | 0.630 | 0.894 | 1.000 | 1.000 | 0.439 |
| cid_sims_apartment2_1 | 11 | 3 | 0.000 | 0.000 | 0.500 | 0.667 | 0.409 | 0.623 | 0.930 | 0.994 | 0.364 |
| cid_sims_apartment3_1 | 20 | 9 | 0.000 | 0.286 | 0.889 | 0.818 | 0.707 | 0.914 | 1.000 | 1.000 | 0.502 |

## Unseen traversals (12)

### 0.5 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 229 | 10 | 6 | 32 | 4 | 0.158 | 0.600 | 0.250 |
| topo_visual | 229 | 10 | 6 | 33 | 4 | 0.154 | 0.600 | 0.245 |
| topo_visual_cosine | 229 | 10 | 8 | 32 | 2 | 0.200 | 0.800 | 0.320 |
| disk_isotropic_default | 229 | 10 | 6 | 37 | 4 | 0.140 | 0.600 | 0.226 |
| disk_isotropic_calibrated | 229 | 10 | 6 | 33 | 4 | 0.154 | 0.600 | 0.245 |
| gate_propagated | 229 | 10 | 7 | 38 | 3 | 0.156 | 0.700 | 0.255 |
| gate_propagated_geometry_only | 229 | 10 | 10 | 50 | 0 | 0.167 | 1.000 | 0.286 |
| paper | 229 | 10 | 6 | 21 | 4 | 0.222 | 0.600 | 0.324 |
| paper_geometry_only | 229 | 10 | 7 | 24 | 3 | 0.226 | 0.700 | 0.341 |
| threshold | 229 | 10 | 3 | 33 | 7 | 0.083 | 0.300 | 0.130 |
| visual_only | 229 | 10 | 3 | 98 | 7 | 0.030 | 0.300 | 0.054 |
| anyloc_vlad_g | 214 | 7 | 4 | 25 | 3 | 0.138 | 0.571 | 0.222 |
| anyloc_vlad_g_visual | 214 | 7 | 5 | 25 | 2 | 0.167 | 0.714 | 0.270 |
| anyloc_gem_g | 219 | 8 | 4 | 28 | 4 | 0.125 | 0.500 | 0.200 |
| anyloc_gem_g_visual | 219 | 8 | 6 | 30 | 2 | 0.167 | 0.750 | 0.273 |
| dinov3_s | 222 | 12 | 7 | 30 | 5 | 0.189 | 0.583 | 0.286 |
| dinov3_s_visual | 222 | 12 | 8 | 30 | 4 | 0.211 | 0.667 | 0.320 |

### 1.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 229 | 19 | 11 | 27 | 8 | 0.289 | 0.579 | 0.386 |
| topo_visual | 229 | 19 | 11 | 28 | 8 | 0.282 | 0.579 | 0.379 |
| topo_visual_cosine | 229 | 19 | 13 | 27 | 6 | 0.325 | 0.684 | 0.441 |
| disk_isotropic_default | 229 | 19 | 12 | 31 | 7 | 0.279 | 0.632 | 0.387 |
| disk_isotropic_calibrated | 229 | 19 | 12 | 27 | 7 | 0.308 | 0.632 | 0.414 |
| gate_propagated | 229 | 19 | 12 | 33 | 7 | 0.267 | 0.632 | 0.375 |
| gate_propagated_geometry_only | 229 | 19 | 16 | 44 | 3 | 0.267 | 0.842 | 0.405 |
| paper | 229 | 19 | 10 | 17 | 9 | 0.370 | 0.526 | 0.435 |
| paper_geometry_only | 229 | 19 | 12 | 19 | 7 | 0.387 | 0.632 | 0.480 |
| threshold | 229 | 19 | 3 | 33 | 16 | 0.083 | 0.158 | 0.109 |
| visual_only | 229 | 19 | 4 | 97 | 15 | 0.040 | 0.211 | 0.067 |
| anyloc_vlad_g | 214 | 22 | 13 | 16 | 9 | 0.448 | 0.591 | 0.510 |
| anyloc_vlad_g_visual | 214 | 22 | 14 | 16 | 8 | 0.467 | 0.636 | 0.538 |
| anyloc_gem_g | 219 | 25 | 14 | 18 | 11 | 0.438 | 0.560 | 0.491 |
| anyloc_gem_g_visual | 219 | 25 | 16 | 20 | 9 | 0.444 | 0.640 | 0.525 |
| dinov3_s | 222 | 24 | 14 | 23 | 10 | 0.378 | 0.583 | 0.459 |
| dinov3_s_visual | 222 | 24 | 15 | 23 | 9 | 0.395 | 0.625 | 0.484 |

### 2.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 229 | 52 | 30 | 8 | 22 | 0.789 | 0.577 | 0.667 |
| topo_visual | 229 | 52 | 29 | 10 | 23 | 0.744 | 0.558 | 0.637 |
| topo_visual_cosine | 229 | 52 | 31 | 9 | 21 | 0.775 | 0.596 | 0.674 |
| disk_isotropic_default | 229 | 52 | 30 | 13 | 22 | 0.698 | 0.577 | 0.632 |
| disk_isotropic_calibrated | 229 | 52 | 29 | 10 | 23 | 0.744 | 0.558 | 0.637 |
| gate_propagated | 229 | 52 | 23 | 22 | 29 | 0.511 | 0.442 | 0.474 |
| gate_propagated_geometry_only | 229 | 52 | 28 | 32 | 24 | 0.467 | 0.538 | 0.500 |
| paper | 229 | 52 | 19 | 8 | 33 | 0.704 | 0.365 | 0.481 |
| paper_geometry_only | 229 | 52 | 21 | 10 | 31 | 0.677 | 0.404 | 0.506 |
| threshold | 229 | 52 | 4 | 32 | 48 | 0.111 | 0.077 | 0.091 |
| visual_only | 229 | 52 | 10 | 91 | 42 | 0.099 | 0.192 | 0.131 |
| anyloc_vlad_g | 214 | 43 | 23 | 6 | 20 | 0.793 | 0.535 | 0.639 |
| anyloc_vlad_g_visual | 214 | 43 | 24 | 6 | 19 | 0.800 | 0.558 | 0.658 |
| anyloc_gem_g | 219 | 49 | 27 | 5 | 22 | 0.844 | 0.551 | 0.667 |
| anyloc_gem_g_visual | 219 | 49 | 29 | 7 | 20 | 0.806 | 0.592 | 0.682 |
| dinov3_s | 222 | 47 | 29 | 8 | 18 | 0.784 | 0.617 | 0.690 |
| dinov3_s_visual | 222 | 47 | 30 | 8 | 17 | 0.789 | 0.638 | 0.706 |

### 3.0 m

| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| topo | 229 | 83 | 36 | 2 | 47 | 0.947 | 0.434 | 0.595 |
| topo_visual | 229 | 83 | 37 | 2 | 46 | 0.949 | 0.446 | 0.607 |
| topo_visual_cosine | 229 | 83 | 38 | 2 | 45 | 0.950 | 0.458 | 0.618 |
| disk_isotropic_default | 229 | 83 | 37 | 6 | 46 | 0.860 | 0.446 | 0.587 |
| disk_isotropic_calibrated | 229 | 83 | 35 | 4 | 48 | 0.897 | 0.422 | 0.574 |
| gate_propagated | 229 | 83 | 30 | 15 | 53 | 0.667 | 0.361 | 0.469 |
| gate_propagated_geometry_only | 229 | 83 | 36 | 24 | 47 | 0.600 | 0.434 | 0.503 |
| paper | 229 | 83 | 22 | 5 | 61 | 0.815 | 0.265 | 0.400 |
| paper_geometry_only | 229 | 83 | 24 | 7 | 59 | 0.774 | 0.289 | 0.421 |
| threshold | 229 | 83 | 5 | 31 | 78 | 0.139 | 0.060 | 0.084 |
| visual_only | 229 | 83 | 15 | 86 | 68 | 0.149 | 0.181 | 0.163 |
| anyloc_vlad_g | 214 | 72 | 27 | 2 | 45 | 0.931 | 0.375 | 0.535 |
| anyloc_vlad_g_visual | 214 | 72 | 27 | 3 | 45 | 0.900 | 0.375 | 0.529 |
| anyloc_gem_g | 219 | 74 | 30 | 2 | 44 | 0.938 | 0.405 | 0.566 |
| anyloc_gem_g_visual | 219 | 74 | 34 | 2 | 40 | 0.944 | 0.459 | 0.618 |
| dinov3_s | 222 | 75 | 36 | 1 | 39 | 0.973 | 0.480 | 0.643 |
| dinov3_s_visual | 222 | 75 | 37 | 1 | 38 | 0.974 | 0.493 | 0.655 |

### 2 m with 95 % CI (bootstrap over traversals)

- topo: P 0.789 [0.681; 0.921], R 0.577 [0.412; 0.739], F 0.667 [0.525; 0.794]
- topo_visual: P 0.744 [0.647; 0.875], R 0.558 [0.403; 0.719], F 0.637 [0.507; 0.764]
- topo_visual_cosine: P 0.775 [0.700; 0.882], R 0.596 [0.463; 0.737], F 0.674 [0.571; 0.779]
- disk_isotropic_default: P 0.698 [0.552; 0.903], R 0.577 [0.411; 0.756], F 0.632 [0.490; 0.794]
- disk_isotropic_calibrated: P 0.744 [0.603; 0.921], R 0.558 [0.391; 0.730], F 0.637 [0.493; 0.782]
- gate_propagated: P 0.511 [0.323; 0.771], R 0.442 [0.286; 0.629], F 0.474 [0.308; 0.667]
- gate_propagated_geometry_only: P 0.467 [0.303; 0.750], R 0.538 [0.396; 0.692], F 0.500 [0.351; 0.685]
- paper: P 0.704 [0.483; 1.000], R 0.365 [0.236; 0.525], F 0.481 [0.329; 0.667]
- paper_geometry_only: P 0.677 [0.455; 1.000], R 0.404 [0.254; 0.571], F 0.506 [0.349; 0.697]
- threshold: P 0.111 [0.000; 0.320], R 0.077 [0.000; 0.146], F 0.091 [0.000; 0.196]
- visual_only: P 0.099 [0.031; 0.183], R 0.192 [0.068; 0.296], F 0.131 [0.043; 0.221]
- anyloc_vlad_g: P 0.793 [0.710; 0.905], R 0.535 [0.423; 0.654], F 0.639 [0.545; 0.738]
- anyloc_vlad_g_visual: P 0.800 [0.714; 0.909], R 0.558 [0.462; 0.667], F 0.658 [0.571; 0.750]
- anyloc_gem_g: P 0.844 [0.683; 1.000], R 0.551 [0.438; 0.698], F 0.667 [0.551; 0.795]
- anyloc_gem_g_visual: P 0.806 [0.650; 0.960], R 0.592 [0.463; 0.737], F 0.682 [0.559; 0.812]
- dinov3_s: P 0.784 [0.681; 0.909], R 0.617 [0.438; 0.776], F 0.690 [0.557; 0.795]
- dinov3_s_visual: P 0.789 [0.688; 0.913], R 0.638 [0.464; 0.790], F 0.706 [0.581; 0.811]

### Average precision at 2 m (decisions ranked by probability)

- topo: AP 0.759 [0.606; 0.932]
- topo_visual: AP 0.758 [0.641; 0.900]
- topo_visual_cosine: AP 0.770 [0.653; 0.906]
- disk_isotropic_default: AP 0.686 [0.527; 0.881]
- disk_isotropic_calibrated: AP 0.698 [0.546; 0.886]
- anyloc_vlad_g: AP 0.672 [0.522; 0.857]
- anyloc_vlad_g_visual: AP 0.657 [0.501; 0.829]
- anyloc_gem_g: AP 0.757 [0.603; 0.927]
- anyloc_gem_g_visual: AP 0.719 [0.565; 0.904]
- dinov3_s: AP 0.649 [0.482; 0.801]
- dinov3_s_visual: AP 0.678 [0.544; 0.812]

### Paired differences at 2 m (bootstrap over traversals)

- F1 topo − paper: +0.186 [-0.032; +0.339]
- P topo − paper: +0.086 [-0.136; +0.255]
- R topo − paper: +0.212 [+0.000; +0.377]
- F1 topo − disk_isotropic_default: +0.035 [-0.032; +0.079]
- F1 topo − disk_isotropic_calibrated: +0.029 [+0.000; +0.058]
- F1 topo − gate_propagated: +0.192 [-0.038; +0.362]
- F1 topo_visual − topo: -0.029 [-0.072; +0.000]
- F1 topo_visual_cosine − topo: +0.007 [-0.054; +0.068]
- F1 topo − anyloc_vlad_g: +0.028 [-0.103; +0.155]
- F1 topo_visual_cosine − anyloc_vlad_g_visual: +0.016 [-0.110; +0.146]
- F1 topo − anyloc_gem_g: +0.000 [-0.146; +0.114]
- F1 topo_visual_cosine − anyloc_gem_g_visual: -0.008 [-0.154; +0.118]
- F1 topo − dinov3_s: -0.024 [-0.146; +0.099]
- F1 topo_visual − dinov3_s_visual: -0.069 [-0.221; +0.077]

### Aligned / opposed revisits at 2 m

| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |
|---|---|---|---|---|---|
| topo | 7 | 2 | 45 | 28 | 8 |
| topo_visual | 7 | 2 | 45 | 27 | 10 |
| topo_visual_cosine | 7 | 4 | 45 | 27 | 9 |
| disk_isotropic_default | 7 | 2 | 45 | 28 | 13 |
| disk_isotropic_calibrated | 7 | 2 | 45 | 27 | 10 |
| gate_propagated | 7 | 3 | 45 | 20 | 22 |
| gate_propagated_geometry_only | 7 | 4 | 45 | 24 | 32 |
| paper | 7 | 1 | 45 | 18 | 8 |
| paper_geometry_only | 7 | 1 | 45 | 20 | 10 |
| threshold | 7 | 3 | 45 | 1 | 32 |
| visual_only | 7 | 3 | 45 | 7 | 91 |
| anyloc_vlad_g | 3 | 2 | 40 | 21 | 6 |
| anyloc_vlad_g_visual | 3 | 3 | 40 | 21 | 6 |
| anyloc_gem_g | 4 | 1 | 45 | 26 | 5 |
| anyloc_gem_g_visual | 4 | 4 | 45 | 25 | 7 |
| dinov3_s | 5 | 1 | 42 | 28 | 8 |
| dinov3_s_visual | 5 | 2 | 42 | 28 | 8 |

### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE

| Traversal | Nodes | Loops | F1@0.5 | F1@1.0 | F1@2.0 | F1@3.0 | C@0.5 | C@1.0 | C@2.0 | C@3.0 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_freiburg_a_seq1 | 22 | 5 | 0.333 | 0.750 | 0.889 | 0.615 | 0.575 | 0.809 | 0.989 | 1.000 | 2.731 |
| heldout_freiburg_a_seq2 | 32 | 8 | 0.400 | 0.545 | 0.667 | 0.667 | 0.500 | 0.764 | 0.970 | 1.000 | 3.199 |
| heldout_freiburg_b_seq3 | 23 | 5 | 0.571 | 0.667 | 0.833 | 0.714 | 0.566 | 0.915 | 0.990 | 0.999 | 1.372 |
| heldout_saarbruecken_a_seq1 | 12 | 2 | 0.000 | 0.500 | 0.800 | 0.800 | 0.307 | 0.556 | 0.879 | 1.000 | 0.984 |
| heldout_saarbruecken_a_seq2 | 30 | 4 | 0.000 | 0.000 | 0.500 | 0.571 | 0.368 | 0.610 | 0.948 | 0.999 | 4.690 |
| heldout_saarbruecken_b_seq3 | 12 | 1 | 0.000 | 0.000 | 0.500 | 0.400 | 0.654 | 0.896 | 1.000 | 1.000 | 0.520 |
| heldout_saarbruecken_b_seq4 | 12 | 1 | 0.000 | 0.000 | 1.000 | 0.500 | 0.494 | 0.746 | 0.978 | 1.000 | 1.059 |
| heldout_ljubljana_a_seq1 | 19 | 1 | 0.000 | 0.000 | 0.667 | 0.500 | 0.210 | 0.404 | 0.577 | 0.661 | 1.974 |
| heldout_ljubljana_a_seq2 | 19 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.253 | 0.397 | 0.577 | 0.673 | 6.304 |
| heldout_cid_apartment1_2 | 10 | 1 | 0.000 | 0.000 | 0.000 | 0.333 | 0.291 | 0.476 | 0.835 | 0.977 | 0.989 |
| heldout_cid_apartment2_2 | 19 | 6 | 0.000 | 0.000 | 0.571 | 0.632 | 0.602 | 0.944 | 1.000 | 1.000 | 0.988 |
| heldout_cid_apartment3_2 | 19 | 4 | 0.400 | 0.400 | 1.000 | 0.533 | 0.525 | 0.797 | 0.939 | 1.000 | 1.161 |

## Language retrieval (google/siglip2-base-patch16-224)

Fields per environment and overall:

```
{
 "overall": {
  "absent": {
   "correct_rejection_rate": 0.6605,
   "false_answer_rate": 0.3395,
   "n": 162
  },
  "heuristic_decision_accuracy": 0.36,
  "n_map_query_pairs": 400,
  "present": {
   "answer_coverage": 0.2899,
   "mrr": 0.5539,
   "n": 238,
   "precision_at_1_when_answered": 0.5362,
   "recall_at_1": 0.4202,
   "recall_at_3": 0.7227,
   "successful_answer_rate": 0.1555
  }
 },
 "environments": {
  "freiburg_a": {
   "absent": {
    "correct_rejection_rate": 0.963,
    "false_answer_rate": 0.037,
    "n": 27
   },
   "heuristic_decision_accuracy": 0.36,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.3425,
    "mrr": 0.4612,
    "n": 73,
    "precision_at_1_when_answered": 0.4,
    "recall_at_1": 0.3014,
    "recall_at_3": 0.6575,
    "successful_answer_rate": 0.137
   }
  },
  "freiburg_ext": {
   "absent": {
    "correct_rejection_rate": 0.5741,
    "false_answer_rate": 0.4259,
    "n": 54
   },
   "heuristic_decision_accuracy": 0.33,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.1522,
    "mrr": 0.5761,
    "n": 46,
    "precision_at_1_when_answered": 0.2857,
    "recall_at_1": 0.4348,
    "recall_at_3": 0.7609,
    "successful_answer_rate": 0.0435
   }
  },
  "saarbruecken_a": {
   "absent": {
    "correct_rejection_rate": 0.5926,
    "false_answer_rate": 0.4074,
    "n": 27
   },
   "heuristic_decision_accuracy": 0.31,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.2466,
    "mrr": 0.5457,
    "n": 73,
    "precision_at_1_when_answered": 0.8333,
    "recall_at_1": 0.4384,
    "recall_at_3": 0.6849,
    "successful_answer_rate": 0.2055
   }
  },
  "saarbruecken_ext": {
   "absent": {
    "correct_rejection_rate": 0.6296,
    "false_answer_rate": 0.3704,
    "n": 54
   },
   "heuristic_decision_accuracy": 0.44,
   "n_map_query_pairs": 100,
   "present": {
    "answer_coverage": 0.413,
    "mrr": 0.692,
    "n": 46,
    "precision_at_1_when_answered": 0.5263,
    "recall_at_1": 0.5652,
    "recall_at_3": 0.8478,
    "successful_answer_rate": 0.2174
   }
  }
 },
 "present_by_category": {
  "command": {
   "answer_coverage": 0.3333,
   "mrr": 0.5719,
   "n": 51,
   "precision_at_1_when_answered": 0.4706,
   "recall_at_1": 0.4118,
   "recall_at_3": 0.7647,
   "successful_answer_rate": 0.1569
  },
  "direct": {
   "answer_coverage": 0.3462,
   "mrr": 0.6154,
   "n": 26,
   "precision_at_1_when_answered": 0.6667,
   "recall_at_1": 0.4615,
   "recall_at_3": 0.8077,
   "successful_answer_rate": 0.2308
  },
  "function": {
   "answer_coverage": 0.4062,
   "mrr": 0.5104,
   "n": 32,
   "precision_at_1_when_answered": 0.6154,
   "recall_at_1": 0.4375,
   "recall_at_3": 0.625,
   "successful_answer_rate": 0.25
  },
  "indirect": {
   "answer_coverage": 0.1765,
   "mrr": 0.4608,
   "n": 34,
   "precision_at_1_when_answered": 0.3333,
   "recall_at_1": 0.2941,
   "recall_at_3": 0.6471,
   "successful_answer_rate": 0.0588
  },
  "object": {
   "answer_coverage": 0.3636,
   "mrr": 0.4697,
   "n": 22,
   "precision_at_1_when_answered": 0.75,
   "recall_at_1": 0.3636,
   "recall_at_3": 0.5909,
   "successful_answer_rate": 0.2727
  },
  "paraphrase": {
   "answer_coverage": 0.1905,
   "mrr": 0.5675,
   "n": 42,
   "precision_at_1_when_answered": 0.375,
   "recall_at_1": 0.4524,
   "recall_at_3": 0.7381,
   "successful_answer_rate": 0.0714
  },
  "scene": {
   "answer_coverage": 0.2581,
   "mrr": 0.6613,
   "n": 31,
   "precision_at_1_when_answered": 0.5,
   "recall_at_1": 0.5161,
   "recall_at_3": 0.8387,
   "successful_answer_rate": 0.129
  }
 }
}
```
