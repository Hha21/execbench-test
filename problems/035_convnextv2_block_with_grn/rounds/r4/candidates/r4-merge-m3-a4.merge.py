"""Build r4-merge-m3-a4: r3-merge-a3-b3-d3 (portal 0.6355) plus lineage A r4 (r4-a-e1-bulkplane-wprep) on the six rows
where an interleaved same-container A/B against the r3 merge showed it faster (loop/ab_container.py, 2026-10-12):
1,160,449,7 -23%; 8,768 -20%; 16,768 -21%; 32,768 -21%; 16,384 -14%; 4,192 -2%. Kernel files unchanged; A4 is its
own namespaced translation unit with its extern "C" entry points prefixed a4_."""
import json
import re
from pathlib import Path

R = Path("/home/lain/execbench-test/problems/035_convnextv2_block_with_grn/rounds")
m3 = json.loads((R / "r3/candidates/r3-merge-a3-b3-d3.json").read_text())
a4 = json.loads((R / "r4/candidates/r4-a-e1-bulkplane-wprep.json").read_text())
a4cu = next(x["content"] for x in a4["sources"] if x["path"] == "kernel.cu")

n_ext = len(re.findall(r'extern "C" (\w[\w\s\*]*?)\bconvblk_', a4cu))
a4cu = re.sub(r'(extern "C" [\w\s\*]*?)\bconvblk_', r'\1a4_convblk_', a4cu)
assert n_ext >= 2 and not re.search(r'extern "C" [\w\s\*]*?\bconvblk_', a4cu)
lines = a4cu.split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("#include"))
a4cu = "\n".join(lines[:last + 1] + ["namespace lin_a4 {  // merged build: lineage-private symbols"] + lines[last + 1:]
                 + ["}  // namespace lin_a4", ""])

binding = next(x["content"] for x in m3["sources"] if x["path"] == "binding.cpp")
decl = '''// ---- lineage A, r4 (lin_a4.cu: bulk-plane conv K1 + cuBLASLt pipeline) on the W=7/14 and 449x7 rows it wins
extern "C" size_t a4_convblk_ws_bytes(int B, int C, int H, int W);
extern "C" int a4_convblk_run(const float* x, const float* dww, const float* dwb, const float* lnw, const float* lnb,
                              const float* w1, const float* b1, const float* gw, const float* gb, const float* w2,
                              const float* b2, float* out, void* ws, int B, int C, int H, int W, float grn_eps,
                              float ln_eps, cudaStream_t st);

// ---- lineage B'''
assert binding.count("// ---- lineage B") == 1
binding = binding.replace("// ---- lineage B", decl, 1)
branch = '''  if (is_shape(B, C, H, W, 1, 160, 449, 7) || is_shape(B, C, H, W, 8, 768, 7, 7) || is_shape(B, C, H, W, 16, 768, 7, 7) ||
      is_shape(B, C, H, W, 32, 768, 7, 7) || is_shape(B, C, H, W, 16, 384, 14, 14) || is_shape(B, C, H, W, 4, 192, 28, 28)) {
    size_t wb = a4_convblk_ws_bytes(B, C, H, W);
    TORCH_CHECK(wb > 0, "lineage A r4: unsupported shape or cuBLASLt heuristic failure");
    auto ws = torch::empty({(int64_t)wb}, x.options().dtype(torch::kUInt8));
    int rc = a4_convblk_run(x.data_ptr<float>(), dwconv_weight.data_ptr<float>(), dwconv_bias.data_ptr<float>(),
                            layernorm_weight.data_ptr<float>(), layernorm_bias.data_ptr<float>(), pwconv1_weight.data_ptr<float>(),
                            pwconv1_bias.data_ptr<float>(), grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(),
                            pwconv2_weight.data_ptr<float>(), pwconv2_bias.data_ptr<float>(), output.data_ptr<float>(),
                            ws.data_ptr(), B, C, H, W, (float)eps, (float)layer_norm_eps, stream);
    TORCH_CHECK(rc == 0, "a4_convblk_run failed with code ", rc);
    return;
  }

  size_t wb = convblk_ws_bytes(B, C, H, W);'''
assert binding.count("  size_t wb = convblk_ws_bytes(B, C, H, W);") == 1
binding = binding.replace("  size_t wb = convblk_ws_bytes(B, C, H, W);", branch, 1)
binding = binding.replace("//   A (lin_a.cu, TMA conv + cuBLASLt / mma.sync pipeline): every other shape",
                          "//   A r4 (lin_a4.cu): 1,160,449,7  8/16/32,768,7,7  16,384,14,14  4,192,28,28\n"
                          "//   A r3 (lin_a.cu, TMA conv + cuBLASLt / mma.sync pipeline): every other shape")
binding = binding.replace("(per-shape dispatch over lineages A, B, D)", "(per-shape dispatch over lineages A r3, A r4, B, D)")

cid = "r4-merge-m3-a4"
sol = dict(m3)
sol["name"] = cid
sol["description"] = ("r3-merge-a3-b3-d3 (D at 2,768/8,384, B at 1,192/1,96, A r3 elsewhere) plus lineage A r4's bulk-plane "
                      "conv pipeline on 1,160,449,7, 8/16/32,768,7,7, 16,384,14,14 and 4,192,28,28. Kernel files unchanged, "
                      "one namespaced translation unit per lineage.")
sol["sources"] = ([dict(path="binding.cpp", content=binding)]
                  + [x for x in m3["sources"] if x["path"] != "binding.cpp"] + [dict(path="lin_a4.cu", content=a4cu)])
(R / f"r4/candidates/{cid}.json").write_text(json.dumps(sol, indent=1))
print("wrote", cid, [(s["path"], len(s["content"])) for s in sol["sources"]])
