# Design session 20261013-183732_01-ling-specialisation-all

Task: {"round": "r8"}


### compile_b200 (draft 1): r8 G E2 first build (kernel.cu + binding.cpp only, anchor omitted for compile check)


### probe_b200 (1): check harness_time signature


### probe_b200 (2): r8 G E2: build g_p1s, precision log, per-launch times, stamps for 4,96 and 8,320


### probe_b200 (3): r8 G E2: variants prefetch-x / pingpong, faster K3, G2 algo sweep


### probe_b200 (4): noop (check)


### probe_b200 (5): r8 G E2: variants prefetch-x / pingpong, faster K3, G2 algo sweep (retry wrapper)


### probe_b200 (6): r8 G E2: 4,96 DWS on/off, K3 launch_bounds(512,1/2), prologue stamp


### run_tests (draft 2): r8 G E2 shipped build: g_p1s for 4,96 (DWS off) and 8,320 (streamed W1'), K3 one-wave; r7 g_pass1 for 64,96 / 32,128; anchor elsewhere
