# Design session research

Task: {"round": "r0"}


### probe_b200 (1): inspect probe API


### probe_b200 (2): elementwise variants: float4 vs 256-bit, unroll 1/2/4, one-shot 256 threads


### probe_b200 (3): large sizes: 128 vs 256-bit, unroll, evict_last stores (TB/s in parens)


### probe_b200 (4): tiny sizes: why 256-bit is slower; block size; scalar; tail loop cost


### probe_b200 (5): load-side cache hints with evict_last stores, all sizes


### compile_b200 (draft 1): first draft: float4 one-shot, EL stores, two-file layout


### run_tests (draft 2): first kernel: float4 one-shot, EF loads + EL stores


### probe_b200 (6): block size / unroll under EF+EL hints; torch.compile baseline estimate


### compile_b200 (draft 3): static stats of the tested kernel


### probe_b200 (7): repeat L-band block-size check; in-process torch.compile baseline estimate
