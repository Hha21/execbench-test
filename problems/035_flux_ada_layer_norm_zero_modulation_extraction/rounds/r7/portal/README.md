# Portal shortlist for round r7

Reference: r6-e1-fp16-tch-fwd (B200 score 0.6680). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r7-e1-fp16-rot3.json` | B200r (emulator, 6 kernels) | +0%/-1%/-7% | 0.6741 ± 0.0070 | 81% |
