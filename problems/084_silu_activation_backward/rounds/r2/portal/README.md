# Portal shortlist for round r2

Reference: r1-silu-bw-cluster2-small (B200 score 0.5166). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-silu-bw-c16coop-c1coop.json` | B200r (emulator) | -2%/-6%/+1% | 0.5218 ± 0.0023 | 99% |
| `r2-silu-bw-width-ordering-closed.json` | B200r (emulator) | +1%/+1%/-0% | 0.5145 ± 0.0022 | 17% |
| `r2-silu-bw-s-wbulk-e2.json` | B200r (emulator) | +15%/+1%/+0% | 0.5037 ± 0.0022 | 0% |
