# Per-workload merge for round r5

Side-by-side rented-B200 times (median of 4 passes each); a workload moves to a round kernel only when it is more than 2% faster there than the current best.

| workload | r4-merge-m3-a4 (us, rented) | r5-a-e2-ringk2 | r5-e1-cutlass-evt-tf32 | routed to |
|---|---|---|---|---|
| 1,96,56,56 | 28.8 | -7.5% | +125.4% | r5-a-e2-ringk2 |
| 1,192,28,28 | 30.1 | -5.2% | +197.9% | r5-a-e2-ringk2 |
| 2,768,7,7 | 34.4 | +41.3% | fail | r4-merge-m3-a4 |
| 2,96,56,56 | 35.4 | -11.7% | +184.5% | r5-a-e2-ringk2 |
| 1,160,449,7 | 34.5 | -5.7% | +221.7% | r5-a-e2-ringk2 |
| 4,192,28,28 | 37.2 | -9.4% | +274.1% | r5-a-e2-ringk2 |
| 4,96,56,56 | 43.9 | -6.0% | +223.8% | r5-a-e2-ringk2 |
| 8,768,7,7 | 44.2 | -4.7% | fail | r5-a-e2-ringk2 |
| 8,384,14,14 | 54.4 | -19.0% | fail | r5-a-e2-ringk2 |
| 8,192,28,28 | 52.1 | -11.8% | +317.4% | r5-a-e2-ringk2 |
| 16,768,7,7 | 53.1 | -9.7% | +15.9% | r5-a-e2-ringk2 |
| 32,768,7,7 | 70.3 | -7.1% | +37.1% | r5-a-e2-ringk2 |
| 16,384,14,14 | 72.1 | -20.1% | fail | r5-a-e2-ringk2 |
| 8,320,32,32 | 89.8 | -18.9% | +595.8% | r5-a-e2-ringk2 |
| 32,128,64,64 | 291.5 | -13.0% | +467.4% | r5-a-e2-ringk2 |
| 64,96,112,112 | 1149.1 | -10.9% | +387.7% | r5-a-e2-ringk2 |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r5/candidates/r5-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.6802 against the best's 0.6553 (#65972).
