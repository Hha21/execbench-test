# Portal shortlist for round r8

Reference: r7-e1-fp16-rot3 (B200 score 0.6848). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r8-e1-fused-emb-persist.json` | B200r (emulator, 7 kernels) | -0%/-3%/-6% | 0.6849 ± 0.0076 | 50% |
