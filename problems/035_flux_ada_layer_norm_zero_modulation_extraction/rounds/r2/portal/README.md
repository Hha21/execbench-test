# Portal shortlist for round r2

Reference: r1-adamod-cutlass-tf32-2sm-chunkepi (B200 score 0.5884). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-e1-tc05-ef-small.json` | B200r (emulator, 1 kernels) | -17%/-11%/+1% | 0.6268 ± 0.0112 | 100% |
| `r2-e3-tf32-ceiling-r1tiles.json` | B200r (emulator, 1 kernels) | +0%/-1%/-6% | 0.6148 ± 0.0112 | 99% |
| `r2-adamod-stream16-cutlass-tf32.json` | B200r (emulator, 1 kernels) | -6%/+0%/+1% | 0.6064 ± 0.0111 | 95% |
