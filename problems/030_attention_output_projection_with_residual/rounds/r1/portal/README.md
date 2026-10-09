# Portal shortlist for round r1

Reference: r0-oproj-cublaslt-fused-v0 (B200 score 0.4024). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-oproj-cutlass-t224-dispatch.json` | B200r (emulator) | -2%/-4%/-13% | 0.4258 ± 0.0047 | 100% |
| `r1-oproj-cublaslt-forced-algo.json` | B200r (emulator) | -5%/-0%/-4% | 0.4129 ± 0.0055 | 98% |
| `r1-oproj-e1-baseline-id.json` | B200r (emulator) | -0%/-0%/+0% | 0.4028 ± 0.0043 | 53% |
