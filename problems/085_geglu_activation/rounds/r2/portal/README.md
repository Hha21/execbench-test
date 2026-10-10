# Portal shortlist for round r2

Reference: r1-geglu-efld-m (B200 score 0.6755). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r2-geglu-row640-efm.json` | B200r (emulator, 2 kernels) | +0%/-0%/-0% | 0.6781 ± 0.0109 | 59% |
| `r2-geglu-cpol.json` | B200r (emulator, 2 kernels) | -0%/+0%/-0% | 0.6767 ± 0.0106 | 54% |
| `r2-geglu-die-probe-ctl.json` | B200r (emulator, 2 kernels) | -1%/+0%/+0% | 0.6766 ± 0.0106 | 53% |
