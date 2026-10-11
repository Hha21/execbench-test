# Portal shortlist for round r9

Reference: r8-e1-fused-emb-persist (B200 score 0.6853). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r9-e1-skew-uniform-s.json` | B200r (emulator, 8 kernels) | +0%/-1%/+0% | 0.6874 ± 0.0058 | 63% |
