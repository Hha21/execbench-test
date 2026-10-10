# Portal shortlist for round r1

Reference: r0-silu-bw-cuda-ldg128-os-efel (B200 score 0.4941). Predicted scores project each candidate's speed relative to the reference, per workload, on the rented B200 when measured there (else its most representative cheap GPU), onto the reference's B200 per-workload times.

| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |
|---|---|---|---|---|
| `r1-silu-bw-cluster2-small.json` | B200r (emulator) | -16%/-7%/+0% | 0.5106 ± 0.0019 | 100% |
| `r1-silu-bw-triton-b512w4-efel.json` | B200r (emulator) | -12%/-2%/-1% | 0.5054 ± 0.0019 | 100% |
| `r1-silu-bw-cuda-ldg128-os-efel-e3.json` | B200r (emulator) | -12%/-2%/-0% | 0.5044 ± 0.0019 | 100% |
