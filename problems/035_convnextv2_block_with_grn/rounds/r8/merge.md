# Per-workload merge for round r8

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r7-merge's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r7-merge portal us (rented) | r8-d-e1-allwarp-epi | r8-g-e2-singletile | r8-h-dataflow-queue | routed to |
|---|---|---|---|---|---|
| 1,192,28,28 | 23.2 (18.7) | 23.1 (-0.3%) | 117.2 (+388.5%) | 112.5 (+385.0%) new | r7-merge |
| 1,96,56,56 | 23.8 (19.4) | 22.6 (-5.1%) | 44.9 (+80.5%) | 79.2 (+232.7%) new | r8-d-e1-allwarp-epi |
| 2,768,7,7 | 33.3 (26.8) | 32.9 (-1.2%) | 160.7 (+368.8%) | 154.4 (+363.5%) new | r7-merge |
| 2,96,56,56 | 33.5 (26.9) | 31.9 (-4.8%) | 45.8 (+32.9%) | 125.3 (+274.0%) new | r8-d-e1-allwarp-epi |
| 1,160,449,7 | 34.2 (26.7) | 33.0 (-3.4%) | 143.9 (+320.8%) | 142.6 (+317.0%) new | r8-d-e1-allwarp-epi |
| 4,192,28,28 | 38.5 (30.9) | 36.7 (-4.8%) | 180.2 (+354.3%) | 174.4 (+353.0%) new | r8-d-e1-allwarp-epi |
| 8,768,7,7 | 43.2 (34.6) | 41.3 (-4.5%) | 339.5 (+666.4%) | 82.4 (+90.8%) new | r8-d-e1-allwarp-epi |
| 8,384,14,14 | 49.6 (37.6) | 44.5 (-10.3%) | 351.8 (+630.0%) | 98.7 (+99.0%) new | r8-d-e1-allwarp-epi |
| 8,192,28,28 | 57.0 (45.0) | 54.5 (-4.4%) | 281.1 (+387.2%) | 276.5 (+385.1%) new | r8-d-e1-allwarp-epi |
| 4,96,56,56 | 57.4 (41.4) | 50.9 (-6.8%) | 48.3 (-9.1%) | 198.5 (+245.9%) new | r8-g-e2-singletile |
| 16,768,7,7 | 58.9 (46.6) | 55.8 (-5.3%) | 738.1 (+1172.8%) | 103.8 (+76.3%) new | r8-d-e1-allwarp-epi |
| 16,384,14,14 | 76.7 (59.2) | 67.2 (-12.4%) | 455.2 (+499.5%) | 118.8 (+54.8%) new | r8-d-e1-allwarp-epi |
| 32,768,7,7 | 87.5 (65.6) | 83.1 (-2.1%) | 1528.0 (+1773.8%) | 166.6 (+90.5%) new | r8-d-e1-allwarp-epi |
| 8,320,32,32 | 99.7 (73.2) | 812.1 (+756.8%) | 87.1 (-7.2%) | 853.9 (+756.5%) new | r8-g-e2-singletile |
| 32,128,64,64 | 292.1 (227.9) | 2158.6 (+631.6%) | 293.4 (+0.5%) | 2135.3 (+631.0%) new | r7-merge |
| 64,96,112,112 | 1082.8 (870.5) | 7294.6 (+547.2%) | 1087.2 (+0.4%) | 7003.6 (+546.8%) new | r7-merge |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r8/candidates/r8-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7333 against the best's 0.7207 (#66740).
