# Portal shortlist for round r1

Reference: r0-convnext-coop-fp32-anchor (B200 score 0.3038). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-lib-cublaslt-fp16-v3.json` | B200r (emulator, 1 kernels) | -54%/-68%/-82% | 0.5938 ± 0.0074 | 100% |
| `r1-lib-cublaslt-fp16-v3fix.json` | B200r (emulator, 1 kernels) | -48%/-65%/-81% | 0.5720 ± 0.0075 | 100% |
| `r1-convnext-coop-hmma-ring.json` | B200r (emulator, 1 kernels) | -59%/-65%/-74% | 0.5622 ± 0.0074 | 100% |
