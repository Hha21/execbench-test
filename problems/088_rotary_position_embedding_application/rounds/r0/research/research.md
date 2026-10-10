# Design session research

Task: {"round": "r0"}


### probe_b200 (1): probe API + how io.py generates cos/sin


### probe_b200 (2): first RoPE kernel: 8 lanes/row, v8 loads; configs x sizes, correctness vs reference


### probe_b200 (3): inspect b200probe launcher/timer source


### probe_b200 (4): diagnose CUPTI 'No timing results' with trivial kernel vs torch op


### probe_b200 (5): read bench_gpu_time_with_cupti; retry rope kernel alone


### probe_b200 (6): isolate: scalar arg marshalling vs v8 load/store path


### probe_b200 (7): localise the rope1 wrongness: which rows/cols differ


### probe_b200 (8): v8 ordering test; float4 control variant of rope


### probe_b200 (9): six outstanding v8 loads; rope variants (no .nc, no restrict, float4)


### probe_b200 (10): fixed K-row offset; config sweep x sizes with correctness


### run_tests (draft 1): r0 first kernel: v8 loads, 8 lanes/row, EL stores, 64/256-thread dispatch


### probe_b200 (11): alternatives: float4 16 lanes/row (low regs) and 2-head cos/sin reuse; plus empty-kernel floor


### probe_b200 (12): byte/FLOP table; 512-thread variant and empty-grid cost at L sizes
