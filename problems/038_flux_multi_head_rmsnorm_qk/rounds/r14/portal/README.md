# Portal shortlist for round r14

Reference: c2-cluster2-s-only (B200 score 0.6125). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r14-a1-h21-audit-c2.json` | B200r (emulator) | +0%/+0%/+0% | 0.6119 ± 0.0043 | 44% |
| `r14-f2-ldg256-ef-el-ilv.json` | B200r (emulator) | +5%/+1%/+1% | 0.6071 ± 0.0059 | 19% |
| `r14-plain256-ef-el.json` | B200r (emulator) | +3%/+0%/-0% | 0.6059 ± 0.0055 | 12% |
