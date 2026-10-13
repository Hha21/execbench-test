# Per-workload merge for round r9

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r8-merge's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r8-merge portal us (rented) | r9-d-e1-asynca | r9-d-e2-pair-smemg | r9-g-e3-tc05-p1s | routed to |
|---|---|---|---|---|---|
| 1,96,56,56 | 23.8 (18.2) | 23.6 (-0.8%) | 23.8 (+0.0%) | 45.1 (+120.1%) | r8-merge |
| 1,192,28,28 | 24.4 (18.5) | 24.0 (-1.8%) | 24.3 (-0.3%) | 109.3 (+385.1%) | r8-merge |
| 2,96,56,56 | 33.1 (25.5) | 32.4 (-2.0%) | 33.1 (-0.1%) | 46.4 (+61.0%) | r8-merge |
| 1,160,449,7 | 33.3 (25.5) | 32.8 (-1.4%) | 33.3 (-0.1%) | 135.1 (+333.9%) | r8-merge |
| 2,768,7,7 | 33.5 (27.1) | 30.1 (-10.2%) | 32.9 (-1.7%) | 149.3 (+351.3%) | r9-d-e1-asynca |
| 4,192,28,28 | 38.5 (29.7) | 38.4 (-0.2%) | 38.4 (-0.3%) | 176.2 (+373.6%) | r8-merge |
| 8,768,7,7 | 41.9 (32.9) | 41.3 (-1.5%) | 42.2 (+0.6%) | 332.5 (+708.4%) | r8-merge |
| 8,384,14,14 | 43.5 (33.6) | 44.3 (+1.9%) | 43.6 (+0.3%) | 339.6 (+706.4%) | r8-merge |
| 4,96,56,56 | 47.5 (42.1) | 49.0 (-10.0%) | 49.1 (-8.8%) | 47.5 (-0.0%) | r8-merge |
| 8,192,28,28 | 55.1 (43.1) | 54.4 (-1.3%) | 55.3 (+0.4%) | 270.7 (+402.0%) | r8-merge |
| 16,768,7,7 | 56.0 (44.4) | 55.9 (-0.1%) | 55.3 (-1.3%) | 731.3 (+1232.6%) | r8-merge |
| 16,384,14,14 | 66.9 (51.7) | 66.9 (+0.0%) | 66.9 (+0.0%) | 442.0 (+582.9%) | r8-merge |
| 32,768,7,7 | 82.1 (63.9) | 84.5 (+2.9%) | 79.9 (-2.7%) | 1511.3 (+1814.6%) | r9-d-e2-pair-smemg |
| 8,320,32,32 | 87.6 (71.9) | 804.6 (+771.6%) | 804.4 (+771.4%) | 75.5 (-13.8%) | r9-g-e3-tc05-p1s |
| 32,128,64,64 | 290.2 (231.9) | 2133.7 (+616.3%) | 2134.1 (+616.4%) | 290.9 (+0.2%) | r8-merge |
| 64,96,112,112 | 1072.6 (868.4) | 7239.1 (+548.9%) | 7236.4 (+548.6%) | 1076.3 (+0.3%) | r8-merge |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r9/candidates/r9-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7351 against the best's 0.7313 (#66855).
