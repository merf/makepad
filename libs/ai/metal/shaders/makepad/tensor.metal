// Makepad's tensor kernels: fills, copies and the elementwise ops (metal/src/stream/kernels.rs
// names them; metal/src/stream/tensor_ops.rs launches them). Tensors are bound at any 4-byte
// offset, so they are `device` pointers, never `constant`, and read one word at a time.
//
// Fills and copies move 32-bit words, not floats, so every bit pattern (NaN payloads, u32 data)
// arrives unchanged. add, mul and relu are one IEEE operation each: bit-identical to CUDA's and to
// the host's, with nothing to contract.
#include "makepad_common.h"

struct TensorCount {
    uint n;
};

struct TensorFill {
    uint n;
    uint bits;
};

// A rows x cols block from src to dst, each with its own row stride (ld) and start, in words.
struct TensorCopy2d {
    uint rows;
    uint cols;
    uint src_ld;
    uint dst_ld;
    uint src_offset;
    uint dst_offset;
};

kernel void tensor_fill_u32(constant TensorFill& p [[buffer(0)]],
                            device uint* dst [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        dst[i] = p.bits;
    }
}

kernel void tensor_copy2d_u32(constant TensorCopy2d& p [[buffer(0)]],
                              device const uint* src [[buffer(1)]],
                              device uint* dst [[buffer(2)]],
                              uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.cols || pos.y >= p.rows) {
        return;
    }
    const ulong from = ulong(p.src_offset) + ulong(pos.y) * ulong(p.src_ld) + ulong(pos.x);
    const ulong to = ulong(p.dst_offset) + ulong(pos.y) * ulong(p.dst_ld) + ulong(pos.x);
    dst[to] = src[from];
}

kernel void tensor_add_f32(constant TensorCount& p [[buffer(0)]],
                           device const float* a [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = a[i] + b[i];
    }
}

kernel void tensor_mul_f32(constant TensorCount& p [[buffer(0)]],
                           device const float* a [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = a[i] * b[i];
    }
}

// CUDA's fmaxf(x, 0) (diffusion_ops.cu): a NaN input gives 0.
kernel void tensor_relu_f32(constant TensorCount& p [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = fmax(x[i], 0.0f);
    }
}
