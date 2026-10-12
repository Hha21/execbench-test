"""Build r4-merge-a4-d3: a per-shape dispatch over two lineages' unchanged kernel files.

Each lineage's kernel.cu becomes its own translation unit, wrapped in a namespace (B3 and D3 share struct and device
function names from a common ancestor) with distinct extern "C" entry points. binding.cpp picks the lineage by shape.
"""
import json
import re
import sys
from pathlib import Path

R = Path("/home/lain/execbench-test/problems/035_convnextv2_block_with_grn/rounds")
OUT = R / "r4/candidates"
src = {k: json.loads((R / f).read_text()) for k, f in dict(
    A="r4/candidates/r4-a-e1-bulkplane-wprep.json",
    D="r3/candidates/r3-ws-coop-widec.json").items()}
cu = {k: next(x["content"] for x in s["sources"] if x["path"] == "kernel.cu") for k, s in src.items()}


def wrap(text, ns):
    """Open namespace `ns` after the last top-level preprocessor include / namespace alias line, close it at the end."""
    lines = text.split("\n")
    last = max(i for i, l in enumerate(lines) if l.startswith("#include") or l.startswith("namespace cg ="))
    assert not any(l.startswith("#include") for l in lines[last + 1:]), ns
    return "\n".join(lines[:last + 1] + [f"namespace {ns} {{  // merged build: lineage-private symbols"]
                     + lines[last + 1:] + [f"}}  // namespace {ns}", ""])


# D3's own fp32-anchor fallback exports convnext_launch, as does B3: rename D3's (unused in the merge).
assert cu["D"].count('extern "C" const char* convnext_launch(') == 1
cu["D"] = cu["D"].replace('extern "C" const char* convnext_launch(', 'extern "C" const char* lin_d_anchor_launch(')
assert "convnext_launch" not in cu["D"]
files = {"lin_a.cu": wrap(cu["A"], "lin_a"), "lin_d.cu": wrap(cu["D"], "lin_d")}

binding = r'''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Per-shape dispatch over two lineages (each in its own translation unit, unchanged):
//   D (lin_d.cu, r3-ws-coop-widec weight-streaming cooperative kernel): 2,768,7,7
//   A (lin_a.cu, r4-a-e1-bulkplane-wprep: bulk-plane conv K1 + cuBLASLt pipeline): every other shape

// ---- lineage A
extern "C" size_t convblk_ws_bytes(int B, int C, int H, int W);
extern "C" int convblk_run(const float* x, const float* dww, const float* dwb, const float* lnw, const float* lnb,
                           const float* w1, const float* b1, const float* gw, const float* gb, const float* w2,
                           const float* b2, float* out, void* ws, int B, int C, int H, int W, float grn_eps,
                           float ln_eps, cudaStream_t st);

// ---- lineage D (layout must match lin_d::WSArgs)
struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};
extern "C" long long ws_workspace_bytes(int B, int C, int H, int W);
extern "C" const char* ws_launch(const WSArgs* a, void* stream);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

static bool is_shape(int B, int C, int H, int W, int b, int c, int h, int w) { return B == b && C == c && H == h && W == w; }

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();

  if (is_shape(B, C, H, W, 2, 768, 7, 7)) {
    const long long wsb = ws_workspace_bytes(B, C, H, W);
    TORCH_CHECK(wsb > 0, "lineage D: unsupported shape");
    torch::Tensor ws = torch::empty({(long)wsb}, x.options().dtype(torch::kUInt8));
    WSArgs a;
    a.x = x.data_ptr<float>(); a.dww = dwconv_weight.data_ptr<float>(); a.dwb = dwconv_bias.data_ptr<float>();
    a.lnw = layernorm_weight.data_ptr<float>(); a.lnb = layernorm_bias.data_ptr<float>();
    a.w1 = pwconv1_weight.data_ptr<float>(); a.b1 = pwconv1_bias.data_ptr<float>();
    a.gw = grn_weight.data_ptr<float>(); a.gb = grn_bias.data_ptr<float>();
    a.w2 = pwconv2_weight.data_ptr<float>(); a.b2 = pwconv2_bias.data_ptr<float>();
    a.out = output.data_ptr<float>(); a.ws = ws.data_ptr(); a.stamps = nullptr;
    a.B = B; a.C = C; a.H = H; a.W = W; a.eps = (float)eps; a.ln_eps = (float)layer_norm_eps;
    const char* err = ws_launch(&a, (void*)stream);
    if (err != nullptr) throw std::runtime_error(std::string("ws_launch failed: ") + err);
    return;
  }

  size_t wb = convblk_ws_bytes(B, C, H, W);
  TORCH_CHECK(wb > 0, "unsupported shape or cuBLASLt heuristic failure");
  auto ws = torch::empty({(int64_t)wb}, x.options().dtype(torch::kUInt8));
  int rc = convblk_run(x.data_ptr<float>(), dwconv_weight.data_ptr<float>(), dwconv_bias.data_ptr<float>(),
                       layernorm_weight.data_ptr<float>(), layernorm_bias.data_ptr<float>(), pwconv1_weight.data_ptr<float>(),
                       pwconv1_bias.data_ptr<float>(), grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(),
                       pwconv2_weight.data_ptr<float>(), pwconv2_bias.data_ptr<float>(), output.data_ptr<float>(),
                       ws.data_ptr(), B, C, H, W, (float)eps, (float)layer_norm_eps, stream);
  TORCH_CHECK(rc == 0, "convblk_run failed with code ", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (per-shape dispatch over lineages A, D)"); }
'''

# the struct layouts copied into binding.cpp must match each lineage's own binding
bD = next(x["content"] for x in src["D"]["sources"] if x["path"] == "binding.cpp")
body = lambda t, name: re.search(r"struct " + name + r" \{(.*?)\};", t, re.S).group(1)
assert body(bD, "WSArgs") == body(binding, "WSArgs"), "D WSArgs layout"

cid = "r4-merge-a4-d3"
sol = dict(src["A"])
sol["name"] = cid
sol["description"] = ("Per-shape dispatch over two lineages' unchanged kernels: D r3-ws-coop-widec at 2,768,7,7; "
                      "A r4-a-e1-bulkplane-wprep elsewhere. Each lineage is its own translation unit in a private "
                      "namespace.")
sol["sources"] = [dict(path="binding.cpp", content=binding)] + [dict(path=k, content=v) for k, v in files.items()]
(OUT / f"{cid}.json").write_text(json.dumps(sol, indent=1))
print("wrote", OUT / f"{cid}.json", [(s["path"], len(s["content"])) for s in sol["sources"]])
