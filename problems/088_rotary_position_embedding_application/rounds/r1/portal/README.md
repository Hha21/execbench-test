# Portal shortlist for round r1

Reference: r0-rope-ldg256-os-stel (B200 score 0.6351). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-rope-v4nc-s128.json` | B200r (emulator, 1 kernels) | -9%/-6%/-6% | 0.6643 ± 0.0107 | 100% |
| `r1-rope-ldg256-os64-tabel.json` | B200r (emulator, 1 kernels) | +1%/+1%/-5% | 0.6413 ± 0.0106 | 72% |
| `r1-rope-pers-tstat-m.json` | B200r (emulator, 1 kernels) | +2%/+17%/-0% | 0.6133 ± 0.0105 | 2% |
