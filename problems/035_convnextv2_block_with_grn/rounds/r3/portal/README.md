# Portal shortlist for round r3

Reference: r0-convnext-coop-fp32-anchor (B200 score 0.3038). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r3-merge-a3-b3-d3.json` | B200r (emulator, 1 kernels) | -62%/-73%/-84% | 0.6335 ± 0.0072 | 100% |
| `r3-a-e1-mma-fused-3launch.json` | B200r (emulator, 1 kernels) | -59%/-68%/-84% | 0.6146 ± 0.0072 | 100% |
| `r3-coop-flat-bulkrows.json` | B200r (emulator, 1 kernels) | -60%/-65%/-74% | 0.5639 ± 0.0074 | 100% |
