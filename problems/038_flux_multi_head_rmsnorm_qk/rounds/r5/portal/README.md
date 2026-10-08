# Portal candidate from round r5 (first interactive design session)

| file | change vs r3-cute-ldg256-os-r16 (0.588) | evidence |
|---|---|---|
| `r5-ldg256-os-r16-stel.json` | CUDA C++ port, 256-bit loads with default policy, output stores marked L2 `evict_last` | A100, same GPU as reference: S −6.4%, M −3.6%, L −3.8%; control with normal stores +0.8/+2.4/+1.1%; projected 0.605 |

The session's transcript is in `../sessions/`. The submitted code equals the tested draft 2 minus an unused
evict_first load option (now being re-tested on A100/H200 by the round's test job).
