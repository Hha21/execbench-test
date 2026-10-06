# Portal round 1 (generation 2)

All five passed all 16 workloads on A100 through NVIDIA's harness (6 October). Submit privately to #38 on B200,
save each result page into `html_results/`, then run `python3 poc/ingest_portal.py html_results/<page>.html`.

| File | Question it answers on B200 | A100 vs v039 |
|---|---|---|
| `g2-os-r16w8.json` | v039's exact tile without the persistent loop: does the loop cost us on B200? | −5% |
| `g2-os-r8w4.json` | Smaller one-shot tiles (8 rows, 4 warps): best on A100; does it beat v039 on small inputs? | −5% |
| `g2-ws-hb16w8s3.json` | Weights loaded once + 3-stage cp.async pipeline: does it lift large-input bandwidth? | +1% (−3% on large) |
| `g2-tma-r8s3.json` | TMA tiles with a 3-deep ring (playbook recipe): first TMA result on B200 | not representative on A100 |
| `g2-tmaws-hb8s4.json` | TMA + weights held once, 4-deep ring: the most promising large-input design on paper | not representative on A100 |

Reference: v039 scored 0.531 (28.56 µs geomean): small 2.67 TB/s, medium 4.90, large 5.77; the baseline gets 6.11
on large. Round 2 will combine the best small-input and large-input kernels behind a size dispatcher.
