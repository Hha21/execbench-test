# Portal shortlist for round r4

Reference: r2-e1-tc05-ef-small (B200 score 0.6108). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r4-e1-tc-pair2-twoacc.json` | B200r (emulator, 2 kernels) | -1%/-3%/-5% | 0.6247 ± 0.0113 | 89% |
| `r4-e3-ingest-probe-pair-parent.json` | B200r (emulator, 2 kernels) | +0%/-3%/-2% | 0.6177 ± 0.0111 | 73% |
| `r4-e2-tc-pair2-nb2.json` | B200r (emulator, 2 kernels) | +0%/-2%/-2% | 0.6176 ± 0.0112 | 73% |
