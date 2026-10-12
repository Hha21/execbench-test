# Portal shortlist for round r4

Reference: r3-merge-a3-b3-d3 (B200 score 0.6355). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r4-a-e1-bulkplane-wprep.json` | B200r (emulator, 3 kernels) | -14%/-4%/-17% | 0.6714 ± 0.0071 | 100% |
| `r4-coop-dataflow-s.json` | B200r (emulator, 3 kernels) | +49%/+56%/+64% | 0.5258 ± 0.0079 | 0% |
| `r4-d-tc05-swapab.json` | B200r (emulator, 3 kernels) | +160%/+79%/+125% | 0.4428 ± 0.0073 | 0% |
