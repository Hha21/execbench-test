# Portal shortlist for round r13

Reference: c2-cluster2-s-only (B200 score 0.6125). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r13-e1-h22-residue-c2.json` | B200r (emulator) | +0%/+0%/+0% | 0.6115 ± 0.0044 | 41% |
| `r13-e2-h21-c2-control.json` | B200r (emulator) | +1%/+0%/+0% | 0.6108 ± 0.0044 | 36% |
| `r13-e3-bulkstore-off.json` | B200r (emulator) | +1%/+1%/+0% | 0.6107 ± 0.0044 | 34% |
