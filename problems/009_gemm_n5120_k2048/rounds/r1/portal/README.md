# Portal shortlist for round r1

Reference: r0-skinny16-cublas (B200 score 0.4870). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1b-lt-tiles.json` | B200r (emulator) | -2%/-3%/-1% | 0.4922 ± 0.0047 | 86% |
| `r1-skinny16-mmout.json` | B200r (emulator) | -1%/-0%/-1% | 0.4891 ± 0.0048 | 67% |
| `r1-splitk4-cluster-m17-64.json` | B200r (emulator) | +2%/+48%/-2% | 0.4438 ± 0.0047 | 0% |
