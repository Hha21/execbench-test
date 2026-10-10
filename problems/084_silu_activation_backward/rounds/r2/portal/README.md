# Portal shortlist for round r2

Reference: r2-silu-bw-c16coop-c1coop (B200 score 0.5219). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-silu-bw-width-ordering-closed.json` | B200r (emulator, 3 kernels) | +3%/+8%/-1% | 0.5148 ± 0.0087 | 21% |
| `r2-silu-bw-s-wbulk-e2.json` | B200r (emulator, 3 kernels) | +17%/+7%/-1% | 0.5044 ± 0.0086 | 2% |
