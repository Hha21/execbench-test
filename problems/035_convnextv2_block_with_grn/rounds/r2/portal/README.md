# Portal shortlist for round r2

Reference: r0-convnext-coop-fp32-anchor (B200 score 0.3038). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-lib-cublaslt-fp16-tmaconv.json` | B200r (emulator, 1 kernels) | -58%/-67%/-84% | 0.6113 ± 0.0073 | 100% |
| `r2-coop-flat-s.json` | B200r (emulator, 1 kernels) | -60%/-65%/-74% | 0.5644 ± 0.0074 | 100% |
| `r2-e3-onchip-hidden-v2.json` | B200r (emulator, 1 kernels) | -51%/-28%/-39% | 0.4161 ± 0.0067 | 100% |
