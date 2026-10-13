# Per-workload merge for round r7

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r6-merge's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r6-merge portal us (rented) | r7-d-e2-smallc | r7-g-e1-fusedp1-mma | routed to |
|---|---|---|---|---|
| 2,768,7,7 | 33.2 (26.8) | 33.1 (-0.4%) | 155.9 (+369.5%) new | r6-merge |
| 1,96,56,56 | 35.5 (28.5) | 23.5 (-33.3%) | 74.5 (+109.8%) new | r7-d-e2-smallc |
| 1,192,28,28 | 38.1 (30.2) | 23.9 (-36.1%) | 113.7 (+198.5%) new | r7-d-e2-smallc |
| 8,768,7,7 | 43.7 (34.5) | 43.8 (+0.1%) | 333.1 (+662.3%) new | r6-merge |
| 2,96,56,56 | 46.4 (31.2) | 33.2 (-14.1%) | 91.1 (+96.4%) new | r7-d-e2-smallc |
| 8,384,14,14 | 48.1 (38.4) | 48.2 (+0.2%) | 340.0 (+606.9%) new | r6-merge |
| 1,160,449,7 | 49.1 (32.4) | 33.1 (-17.6%) | 167.7 (+241.5%) new | r7-d-e2-smallc |
| 4,192,28,28 | 50.0 (33.8) | 37.5 (-10.4%) | 205.7 (+311.4%) new | r7-d-e2-smallc |
| 4,96,56,56 | 57.4 (41.4) | 185.3 (+245.5%) | 85.8 (+49.4%) new | r6-merge |
| 16,768,7,7 | 60.1 (46.4) | 60.2 (+0.2%) | 763.9 (+1171.0%) new | r6-merge |
| 8,192,28,28 | 66.0 (45.6) | 57.7 (-2.2%) | 312.1 (+372.9%) new | r7-d-e2-smallc |
| 16,384,14,14 | 77.4 (59.8) | 77.5 (+0.1%) | 455.2 (+488.1%) new | r6-merge |
| 32,768,7,7 | 88.9 (63.7) | 88.2 (+6.9%) | 1702.5 (+1815.1%) new | r6-merge |
| 8,320,32,32 | 100.9 (72.6) | 806.2 (+757.8%) | 866.5 (+758.7%) new | r6-merge |
| 32,128,64,64 | 322.5 (252.0) | 2138.5 (+555.5%) | 288.8 (-10.5%) new | r7-g-e1-fusedp1-mma |
| 64,96,112,112 | 1281.7 (1019.0) | 7236.5 (+448.5%) | 1084.9 (-15.4%) new | r7-g-e1-fusedp1-mma |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r7/candidates/r7-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7208 against the best's 0.6908 (#66643).
