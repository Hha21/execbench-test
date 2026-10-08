# Diagnostic d1

| file | what it tests | expected |
|---|---|---|
| `d1-os-r8w4-nohint.json` | `g2-os-r8w4` (0.577) with every cache hint removed; 128-bit loads, Triton | about 0.588 if the hints, not the 256-bit width, explain the CuTe kernel's gain |

Checked on A100 through the round-r5 tool server: 16/16 workloads pass; the sm_100a build uses plain LDG.128/STG.128
(32 registers, 16 CTAs/SM).
