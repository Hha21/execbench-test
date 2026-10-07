# quack_cutedsl_inline_ptx_excerpt.py
#
# Source:  https://github.com/Dao-AILab/quack/blob/35266c3298f0e9bf6d5f46c30aace2eaeae517e3/microbenchmarks/global_memory_coalescing.py
#          (results: https://github.com/Dao-AILab/quack/blob/35266c3298f0e9bf6d5f46c30aace2eaeae517e3/AI/global_memory_coalescing_notes.md)
# Commit:  35266c3298f0 (2026-09-12)
# Licence: Apache-2.0. Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao. Excerpt, unmodified except that
#          lines are elided ("...") and comments starting "# [solx]" are ours.
#
# Why it matters for #38:
# - Shows how a CuTe DSL kernel emits an arbitrary PTX global load/store (llvm.inline_asm inside a
#   @dsl_user_op). CopyUniversalOp has no cache-hint parameter, so this is the route to an L2 hint on our
#   256-bit loads while keeping the rest of the r3 CuTe kernel unchanged.
# - The H100 study it comes from: a warp instruction covering ONE contiguous 512 B span is fastest; four far
#   128 B pieces cost ~8% on loads; sparse/partial-sector stores are much worse than sparse loads. Our
#   half-warp-per-512-B-row layout is the good case; FlashInfer's 4-threads-per-row layout is not.
# - [solx] Untested adaptation for #38 (needs an sm_100a compile check): a 256-bit variant would use
#   "ld.global.L2::cache_hint.v8.b32 {$0,...,$7}, [$8], $9;" with constraints "=r"x8 + ",l,l" and the policy
#   0x12F0000000000000 (CUTLASS EVICT_FIRST), or the direct form "ld.global.L2::evict_first.v8.b32" (ptxas
#   accepts .L2::evict_first on loads only for 256-bit .v8.b32/.v4.b64, per TransformerEngine issue #3601).
#   Values come back as Int32 bit patterns; reinterpret to Float32 without a conversion.
#   has_side_effects=True also pins program order, so issue every load before the first use yourself.

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

...

@dsl_user_op
def load_global_v4_u32(
    gmem_ptr: cute.Pointer,
    load_mode: str,
    *,
    loc=None,
    ip=None,
) -> tuple[Int32, Int32, Int32, Int32]:
    """Inline PTX 16-byte global load returning four u32 values.

    `has_side_effects=True` deliberately keeps repeated cache-resident loads in
    the timed loop instead of letting LLVM treat the inline asm as hoistable pure
    computation.  The returned values are also accumulated into a checksum.
    """
    asm_by_mode = {
        "global": "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "ca": "ld.global.ca.v4.u32 {$0, $1, $2, $3}, [$4];",
        "cg": "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "volatile": "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
    }
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [gmem_ptr.llvm_ptr],
        asm_by_mode[load_mode],
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def load_global_scalar4_u32(
...

@dsl_user_op
def store_global_v4_u32(
    gmem_ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    store_mode: str,
    *,
    loc=None,
    ip=None,
) -> None:
    asm_by_mode = {
        "global": "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "wb": "st.global.wb.v4.u32 [$0], {$1, $2, $3, $4};",
        "cg": "st.global.cg.v4.u32 [$0], {$1, $2, $3, $4};",
        "cs": "st.global.cs.v4.u32 [$0], {$1, $2, $3, $4};",
        "wt": "st.global.wt.v4.u32 [$0], {$1, $2, $3, $4};",
        "volatile": "st.volatile.global.v4.u32 [$0], {$1, $2, $3, $4};",
    }
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            Int32(v0).ir_value(loc=loc, ip=ip),
            Int32(v1).ir_value(loc=loc, ip=ip),
            Int32(v2).ir_value(loc=loc, ip=ip),
            Int32(v3).ir_value(loc=loc, ip=ip),
        ],
        asm_by_mode[store_mode],
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
