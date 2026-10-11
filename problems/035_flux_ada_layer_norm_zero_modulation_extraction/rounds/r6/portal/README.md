# Portal shortlist for round r6

Reference: r5-e2-tc-ws-uniform-issue (B200 score 0.6256). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r6-e1-fp16-tch-fwd.json` | B200r (emulator, 5 kernels) | +1%/+1%/-15% | 0.6541 ± 0.0089 | 100% |
