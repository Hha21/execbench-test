# Portal shortlist for round r2

Reference: r1-rope-v4nc-s128 (B200 score 0.6772). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-rope-pair2-magic-occ10.json` | B200r (emulator, 2 kernels) | -3%/-2%/+0% | 0.6814 ± 0.0109 | 65% |
| `r2-rope-psst-m.json` | B200r (emulator, 2 kernels) | -0%/+1%/+0% | 0.6766 ± 0.0108 | 48% |
| `r2-rope-ceiling-v4.json` | B200r (emulator, 2 kernels) | +0%/+0%/+0% | 0.6764 ± 0.0108 | 47% |
