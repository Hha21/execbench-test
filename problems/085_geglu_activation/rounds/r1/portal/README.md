# Portal shortlist for round r1

Reference: r0-geglu-ldg256-2d-stel (B200 score 0.6742). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-geglu-efld-m.json` | B200r (emulator, 1 kernels) | -1%/-5%/+0% | 0.6802 ± 0.0107 | 71% |
| `r1-geglu-s-lean32.json` | B200r (emulator, 1 kernels) | +0%/+0%/+1% | 0.6734 ± 0.0106 | 46% |
| `r1-geglu-bulkring-s3k8-l.json` | B200r (emulator, 1 kernels) | +0%/+0%/+9% | 0.6591 ± 0.0104 | 7% |
