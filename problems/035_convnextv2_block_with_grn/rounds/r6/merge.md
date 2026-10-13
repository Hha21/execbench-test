# Per-workload merge for round r6

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r5-merge-b's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r5-merge-b portal us (rented) | r6-d-e2-epi-fix | r6-f-e1-libpipe | routed to |
|---|---|---|---|---|
| 1,96,56,56 | 35.8 (28.2) | 84.0 (+127.3%) | 80.7 (+125.5%) new | r5-merge-b |
| 1,192,28,28 | 37.8 (29.7) | 117.7 (+202.7%) | 41.0 (+8.6%) new | r5-merge-b |
| 2,768,7,7 | 44.8 (34.2) | 35.9 (-19.9%) | 53.6 (+19.6%) new | r6-d-e2-epi-fix |
| 2,96,56,56 | 46.2 (37.9) | 130.6 (+163.0%) | 121.6 (+163.2%) new | r5-merge-b |
| 1,160,449,7 | 49.2 (40.1) | 146.0 (+177.7%) | 136.0 (+176.5%) new | r5-merge-b |
| 4,192,28,28 | 49.4 (38.6) | 182.4 (+260.8%) | 54.5 (+10.4%) new | r5-merge-b |
| 4,96,56,56 | 57.6 (46.8) | 185.0 (+201.7%) | 174.1 (+202.3%) new | r5-merge-b |
| 8,768,7,7 | 60.5 (48.8) | 45.0 (-29.5%) | 56.2 (-7.2%) new | r6-d-e2-epi-fix |
| 8,384,14,14 | 64.1 (52.9) | 49.9 (-28.0%) | 58.1 (-9.3%) new | r6-d-e2-epi-fix |
| 8,192,28,28 | 66.5 (53.1) | 283.3 (+306.6%) | 67.3 (+1.1%) new | r5-merge-b |
| 16,768,7,7 | 68.4 (55.8) | 60.9 (-16.7%) | 64.4 (-5.8%) new | r6-d-e2-epi-fix |
| 16,384,14,14 | 82.0 (62.8) | 78.9 (-4.2%) | 87.5 (+6.7%) new | r6-d-e2-epi-fix |
| 32,768,7,7 | 88.7 (70.7) | 88.5 (-4.5%) | 90.3 (+1.8%) new | r5-merge-b |
| 8,320,32,32 | 100.9 (80.1) | 815.3 (+677.0%) | 784.3 (+677.3%) new | r5-merge-b |
| 32,128,64,64 | 322.9 (258.3) | 2161.2 (+538.4%) | 2058.3 (+537.5%) new | r5-merge-b |
| 64,96,112,112 | 1281.5 (1028.6) | 7322.1 (+443.1%) | 6975.8 (+444.4%) new | r5-merge-b |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r6/candidates/r6-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.6884 against the best's 0.6747 (#66402).
