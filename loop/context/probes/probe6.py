import os, time, traceback
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
t0 = time.time()
try:
    import torch, cutlass, cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    print("cutlass-dsl", getattr(cutlass, "__version__", "?"))

    @cute.kernel
    def k(gA: cute.Tensor, gB: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        gB[i] = gA[i] * 2.0

    @cute.jit
    def host(mA: cute.Tensor, mB: cute.Tensor):
        k(mA, mB).launch(grid=[mA.shape[0] // 128, 1, 1], block=[128, 1, 1])

    a = torch.zeros(1024, dtype=torch.float32)
    b = torch.zeros(1024, dtype=torch.float32)
    c = cute.compile(host, from_dlpack(a), from_dlpack(b), options="--keep-ptx --keep-cubin" if False else "")
    print("compiled OK", type(c), round(time.time() - t0, 1), "s")
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:800])
