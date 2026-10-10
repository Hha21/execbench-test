# Portal shortlist for round r1

Reference: r0-adamod-dispatch-skinny-fp32-3xtf32 (B200 score 0.2349). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-adamod-cutlass-tf32-2sm-chunkepi.json` | B200r (emulator, 1 kernels), parent's features | -61%/-79%/-77% | 0.6023 ± 0.0111 | 100% |
| `r1-ef-bulk-tf32-stream-b16.json` | B200r (emulator, 1 kernels) | -62%/-75%/-71% | 0.5434 ± 0.0105 | 100% |
| `r1-e1-lt-fasttf32-bias-scatter.json` | B200r (emulator, 1 kernels) | -58%/-75%/-71% | 0.5347 ± 0.0105 | 100% |
