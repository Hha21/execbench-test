# Portal shortlist for round r3

Reference: r2-e1-tc05-ef-small (B200 score 0.6108). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r3-e2-tc-pair-kbs2.json` | B200r (emulator, 2 kernels) | +0%/-3%/-2% | 0.6180 ± 0.0111 | 74% |
| `r3-e3-tc05-emb-mcast2.json` | B200r (emulator, 2 kernels) | +1%/-0%/-2% | 0.6137 ± 0.0112 | 60% |
