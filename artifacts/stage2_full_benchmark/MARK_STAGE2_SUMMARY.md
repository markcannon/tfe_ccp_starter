# Stage-2 softmax bound benchmark

This report is generated automatically by `benchmark_stage2_full.py`.
It benchmarks Stage 2 only; it is not yet a closed-loop box-CCP controller comparison.

## Real CCP P-box experiment

The baseline simplex CCP is run first. Each optimized Stage-1 simplex is replaced only for diagnostic purposes by its exact axis-aligned bounding box. All Stage-2 methods are then evaluated on the same P box.

| method | mean gap to true min | worst gap | invalid | median full-matrix ms | mean R-box width |
|---|---:|---:|---:|---:|---:|
| p_exact | 1.373597e-09 | 2.169431e-08 | 0 | 0.062417 | 1.782776e-05 |
| common_o_n | 9.001030e-04 | 2.428205e-03 | 0 | 0.349237 | 9.179294e-04 |
| q_box | 1.782864e-05 | 1.238379e-04 | 0 | 0.072754 | 3.565503e-05 |
| rough | 1.782918e-05 | 1.238432e-04 | 0 | 0.048548 | 3.565557e-05 |
| u_upper | 4.576995e-10 | 9.463678e-09 | 0 | 0.039716 | nan |

Common-scalar enclosure width / original simplex-box width: mean 6168.886, max 67235.165.

## Probe sanity

Maximum P(U^(k+1)) bounding-box violation: 0.000e+00.  Maximum Stage-1 simplex violation: 0.000e+00.
The linearisation point P^k is reported separately and is not required to belong to the newly optimized simplex/box.

## Synthetic scaling experiment

Synthetic boxes preserve the common-width / different-centres geometry observed in the baseline Stage-1 simplex boxes. Their centre and P^k-offset statistics are calibrated from the real probe.

| N | method | median full-matrix ms | mean gap to true min | invalid |
|---:|---|---:|---:|---:|
| 4 | p_exact | 0.060166 | 2.224594e-07 | 0 |
| 4 | common_o_n | 0.338483 | 2.715854e-02 | 0 |
| 4 | q_box | 0.070583 | 1.269158e-07 | 0 |
| 4 | rough | 0.047817 | 3.241905e-07 | 0 |
| 4 | u_upper | 0.038000 | 7.415326e-08 | 0 |
| 6 | p_exact | 0.113500 | 2.701218e-07 | 0 |
| 6 | common_o_n | 0.759417 | 2.226304e-02 | 0 |
| 6 | q_box | 0.106958 | 1.777281e-05 | 0 |
| 6 | rough | 0.073033 | 1.801943e-05 | 0 |
| 6 | u_upper | 0.062042 | 5.401698e-08 | 0 |
| 8 | p_exact | 0.310708 | 2.047792e-07 | 0 |
| 8 | common_o_n | 1.324950 | 2.242889e-02 | 0 |
| 8 | q_box | 0.142500 | 7.713128e-05 | 0 |
| 8 | rough | 0.094967 | 7.728840e-05 | 0 |
| 8 | u_upper | 0.086617 | 2.925894e-08 | 0 |
| 10 | p_exact | 1.024708 | 2.801198e-07 | 0 |
| 10 | common_o_n | 2.053875 | 1.826709e-02 | 0 |
| 10 | q_box | 0.177908 | 2.286130e-05 | 0 |
| 10 | rough | 0.118708 | 2.312887e-05 | 0 |
| 10 | u_upper | 0.114442 | 3.112532e-08 | 0 |
| 12 | p_exact | 5.230333 | 2.347003e-07 | 0 |
| 12 | common_o_n | 2.936617 | 1.691045e-02 | 0 |
| 12 | q_box | 0.214342 | 4.368929e-05 | 0 |
| 12 | rough | 0.142108 | 4.390870e-05 | 0 |
| 12 | u_upper | 0.144392 | 2.133922e-08 | 0 |
| 14 | p_exact | 33.320333 | 2.957956e-07 | 0 |
| 14 | common_o_n | 3.963858 | 1.532688e-02 | 0 |
| 14 | q_box | 0.250333 | 1.304931e-07 | 0 |
| 14 | rough | 0.165167 | 4.245035e-07 | 0 |
| 14 | u_upper | 0.177608 | 2.275351e-08 | 0 |
| 16 | common_o_n | 5.172092 | 1.784569e-02 | 0 |
| 16 | q_box | 0.286183 | 1.582949e-05 | 0 |
| 16 | rough | 0.188200 | 1.610262e-05 | 0 |
| 16 | u_upper | 0.213000 | 1.852034e-08 | 0 |
| 24 | common_o_n | 11.841817 | 1.655455e-02 | 0 |
| 24 | q_box | 0.435417 | 6.293703e-05 | 0 |
| 24 | rough | 0.284533 | 6.322738e-05 | 0 |
| 24 | u_upper | 0.379367 | 1.303704e-08 | 0 |
| 32 | common_o_n | 21.003683 | 1.763335e-02 | 0 |
| 32 | q_box | 0.578283 | 1.150482e-06 | 0 |
| 32 | rough | 0.381750 | 1.451228e-06 | 0 |
| 32 | u_upper | 0.592208 | 9.715549e-09 | 0 |
| 48 | common_o_n | 47.601417 | 1.899662e-02 | 0 |
| 48 | q_box | 0.883200 | 1.033321e-04 | 0 |
| 48 | rough | 0.581075 | 1.036200e-04 | 0 |
| 48 | u_upper | 1.149583 | 6.299122e-09 | 0 |
| 64 | common_o_n | 86.252625 | 1.934902e-02 | 0 |
| 64 | q_box | 1.203683 | 2.526963e-05 | 0 |
| 64 | rough | 0.789700 | 2.557440e-05 | 0 |
| 64 | u_upper | 1.983283 | 4.860763e-09 | 0 |

`p_exact` is intentionally stopped once vertex enumeration exceeds the configured exact-size limit; the other methods continue to larger N.

## Interpretation notes

- `p_exact`: exact minimum of the original paper lower function L(P; P^k) on the supplied P hyperrectangle, using vertex enumeration. It is the small-N P-space reference.
- `common_o_n`: exact on the enlarged common-scalar box, but potentially conservative relative to the original componentwise box.
- `q_box`: transformed-Q critical-vertex method; cheap but potentially conservative because the selected Q-box vertex need not be reachable from one P vertex.
- `rough`: termwise closed-form bound on 1-L; cheap and conservative.
- `u_upper`: direct critical P-vertex for the upper function U; no gradient-magnitude bound is used.

## Next step

After selecting a Stage-2 formulation from these results, integrate it with the Stage-1 and Stage-3 hyperrectangle propagation to build a separate box-CCP controller. Then compare full CCP solve time, iteration count, tracking performance and closed-loop behavior against the existing simplex `ExactConicCCP` baseline.
