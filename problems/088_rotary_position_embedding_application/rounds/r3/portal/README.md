# Portal shortlist for round r3

Reference: r2-rope-pair2-magic-occ10 (B200 score 0.6827). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r3-rope-hints-dispatch.json` | B200r (emulator, 3 kernels) | +1%/+0%/-1% | 0.6818 ± 0.0107 | 47% |
| `r3-rope-quad-d2-m.json` | B200r (emulator, 3 kernels) | +0%/+1%/-1% | 0.6820 ± 0.0107 | 47% |
| `r3-rope-pair2-e3-closed.json` | B200r (emulator, 3 kernels) | +1%/+1%/+0% | 0.6802 ± 0.0107 | 40% |
