# Portal shortlist for round r2

Reference: r1-oproj-cutlass-t224-dispatch (B200 score 0.4529). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-oproj-t224-epi-ab.json` | B200r (emulator) | +2%/+4%/+7% | 0.4225 ± 0.0043 | 0% |
| `r2-oproj-rinit-tmem-ab.json` | B200r (emulator) | +1%/+4%/+8% | 0.4210 ± 0.0043 | 0% |
| `r2-oproj-e1-t192-ab4096.json` | B200r (emulator) | +1%/+8%/+8% | 0.4169 ± 0.0043 | 0% |
