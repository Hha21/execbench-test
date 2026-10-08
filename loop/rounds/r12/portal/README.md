# Portal shortlist for round r12

Reference: c2-cluster2-s-only (B200 score 0.6125). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r12-c2-natural-order.json` | B200r (emulator) | -0%/+0%/-0% | 0.6116 ± 0.0045 | 42% |
| `r12-c3-c2-no-prefetch.json` | B200r (emulator) | -0%/+0%/+0% | 0.6116 ± 0.0045 | 42% |
| `r12-e1-h18-efcurve-disp.json` | B200r (emulator) | +0%/+0%/+0% | 0.6116 ± 0.0044 | 42% |
