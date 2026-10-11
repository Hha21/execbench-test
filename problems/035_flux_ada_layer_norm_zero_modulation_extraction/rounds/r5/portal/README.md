# Portal shortlist for round r5

Reference: r3-e2-tc-pair-kbs2 (B200 score 0.6166). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r5-e2-tc-ws-uniform-issue.json` | B200r (emulator, 3 kernels) | -1%/-7%/-9% | 0.6378 ± 0.0112 | 98% |
| `r5-e1-merged-dispatch.json` | B200r (emulator, 3 kernels) | -1%/-2%/-5% | 0.6294 ± 0.0111 | 87% |
| `r5-e3-tc-k48-pair.json` | B200r (emulator, 3 kernels) | -0%/-2%/-5% | 0.6277 ± 0.0113 | 84% |
