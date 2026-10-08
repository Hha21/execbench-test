"""Helpers for quick B200 experiments (the design sessions' probe_b200 tool). Seconds per probe, no harness build.

    import torch, b200probe as bp
    SRC = r'''
    extern "C" __global__ void copy(const float* __restrict__ x, float* __restrict__ y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i];
    }'''
    k = bp.cuda_kernel(SRC, "copy")                       # NVRTC -> cubin -> loaded function (~1 s)
    x = torch.randn(1 << 22, device="cuda"); y = torch.empty_like(x)
    n = x.numel()
    print(bp.harness_time(lambda x, y: k((n // 256,), (256,), x, y, n), inputs=[x], outputs=[y]))   # µs
    print(bp.resources(SRC)); print(bp.sass(SRC)[:1500])

cuda_kernel: write kernels as extern "C" __global__. Launch args: torch tensors pass as pointers, Python ints as int32
(wrap with bp.i64(v) for 64-bit), floats as float32. Extra NVRTC options via opts=[...].
harness_time: NVIDIA's own timing (CUPTI kernel spans, 2x-L2 zero-fill before every iteration, input/output pointers
shifted each iteration); returns the median in µs. fn receives the (shifted) inputs then outputs.
"""

import ctypes
import os
import subprocess
import tempfile

import torch
from cuda.bindings import driver, nvrtc

ARCH = "sm_100a"
CUDA = os.environ.get("CUDA_HOME", "/usr/local/cuda")


class i64(int):
    """Marks a launch argument as a 64-bit integer."""


def _ok(res, what):
    """cuda-python calls return (err,) or (err, value); raise on error, else return the value."""
    err, *rest = res
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return rest[0] if len(rest) == 1 else None


def compile_cubin(src, opts=()):
    """CUDA C++ source -> cubin bytes for sm_100a with NVRTC. Raises with the compile log on error."""
    prog = _ok(nvrtc.nvrtcCreateProgram(src.encode(), b"probe.cu", 0, [], []), "nvrtcCreateProgram")
    options = [f"--gpu-architecture={ARCH}".encode(), b"-std=c++17", f"-I{CUDA}/include".encode(),
               *[o.encode() for o in opts]]
    err = nvrtc.nvrtcCompileProgram(prog, len(options), options)[0]
    log_size = _ok(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log.decode(errors="replace"))
    size = _ok(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")
    cubin = b" " * size
    _ok(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    return cubin


class Kernel:
    def __init__(self, cubin, name):
        torch.zeros(1, device="cuda")                    # make torch's primary context current
        self.ctx = _ok(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
        self.module = _ok(driver.cuModuleLoadData(cubin), "cuModuleLoadData")
        self.fn = _ok(driver.cuModuleGetFunction(self.module, name.encode()), "cuModuleGetFunction")
        self.name = name

    def set_smem(self, nbytes):
        """Allow more than 48 KB of dynamic shared memory."""
        attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _ok(driver.cuFuncSetAttribute(self.fn, attr, nbytes), "cuFuncSetAttribute")

    def __call__(self, grid, block, *args, smem=0, stream=None):
        _ok(driver.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")   # driver contexts are per thread
        vals, types = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                vals.append(a.data_ptr()); types.append(ctypes.c_void_p)
            elif isinstance(a, i64):
                vals.append(int(a)); types.append(ctypes.c_int64)
            elif isinstance(a, int):
                vals.append(a); types.append(ctypes.c_int32)
            elif isinstance(a, float):
                vals.append(a); types.append(ctypes.c_float)
            else:
                raise TypeError(f"unsupported launch argument {type(a)}")
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        s = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        _ok(driver.cuLaunchKernel(self.fn, *g, *b, smem, driver.CUstream(s), (tuple(vals), tuple(types)), 0),
            f"cuLaunchKernel({self.name})")


def cuda_kernel(src, name, opts=()):
    return Kernel(compile_cubin(src, opts), name)


def _cuobjdump(src, flag, opts=()):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(compile_cubin(src, opts))
        f.flush()
        return subprocess.run([f"{CUDA}/bin/cuobjdump", flag, f.name], capture_output=True, text=True).stdout


def sass(src, opts=()):
    """SASS of every kernel in the source, for sm_100a."""
    return _cuobjdump(src, "-sass", opts)


def resources(src, opts=()):
    """Registers, shared memory, stack and local memory per kernel."""
    return _cuobjdump(src, "-res-usage", opts)


def flush_l2():
    """Zero a buffer twice the L2 size, as the harness does before every timed call."""
    from sol_execbench.core.bench.timing import get_l2_cache_size
    torch.empty(2 * get_l2_cache_size("cuda"), dtype=torch.int8, device="cuda").zero_()
    torch.cuda.synchronize()


def harness_time(fn, inputs=(), outputs=(), warmup=10, rep=50, tries=3):
    """Median µs of fn(*inputs, *outputs) with the harness's CUPTI methodology (cold L2, shifted pointers)."""
    from sol_execbench.core.bench.timing import time_runnable
    for attempt in range(tries):
        try:
            ms = time_runnable(fn, list(inputs), list(outputs), "cuda:0", warmup=warmup, rep=rep,
                               return_mode="median", methodology="cupti")
            return ms * 1e3
        except ValueError as e:                          # CUPTI occasionally drops an iteration's records here
            if not any(m in str(e) for m in ("activity sequence", "No timing results")) or attempt == tries - 1:
                raise
