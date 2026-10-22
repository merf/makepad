// Cutie's pooling and resizing (cutie-metal family (b)): MaxPool2d(3, stride 2, padding 1) and
// F.interpolate(mode='bilinear', align_corners=False) of planar [planes, height*width] tensors,
// ported from cuda/kernels/cutie.cu (cutie_maxpool3x3s2_kernel, cutie_resize_bilinear_kernel) and
// launched by metal/src/cutie_ops/pool_resize.rs, which checks every shape and size first.
//
// Both are held to CUDA's bits (EXACT in the ops check): the max-pool only compares, and the
// resize does CUDA's fp32 operations one at a time in CUDA's order. Where cutie.cu writes
// __fadd_rn, __fsub_rn and __fmul_rn, each operation here is a statement of its own, and the
// contraction setting in makepad_common.h (step 7) keeps the compiler from fusing a product into
// the sum after it. Tensors are bound at any 4-byte offset and read one float at a time.
// One thread an output; the host makes sure every element count fits a uint, so no index here
// overflows.
#include "makepad_common.h"

// The input's plane, the output's, and the outputs over every plane (ow * oh * planes).
struct CutieMaxpool {
    uint width;
    uint height;
    uint ow;
    uint oh;
    uint n;
};

// Padding never wins (PyTorch pads with -inf), so a window takes the max of its in-bounds taps,
// rows then columns; a NaN tap wins, the last one in that order when there are several, as in
// PyTorch's max_pool2d (`val > maxval || isnan(val)`).
kernel void cutie_maxpool3x3s2_f32(constant CutieMaxpool& p [[buffer(0)]],
                                   device const float* x [[buffer(1)]],
                                   device float* out [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint plane = p.ow * p.oh;
    const uint ox = i % p.ow;
    const uint oy = (i % plane) / p.ow;
    device const float* src = x + (i / plane) * (p.width * p.height);
    float best = -INFINITY;
    // A tap's row and column are counted from 1 here (the window starts one before 2 * o), so a
    // tap before the image is 0 and needs no signed index.
    for (uint ky = 0; ky < 3; ++ky) {
        const uint y1 = oy * 2 + ky;
        if (y1 == 0 || y1 > p.height) {
            continue;
        }
        device const float* row = src + (y1 - 1) * p.width;
        for (uint kx = 0; kx < 3; ++kx) {
            const uint x1 = ox * 2 + kx;
            if (x1 == 0 || x1 > p.width) {
                continue;
            }
            const float v = row[x1 - 1];
            if (v > best || isnan(v)) {
                best = v;
            }
        }
    }
    out[i] = best;
}

// The input's plane, the output's, each axis's scale (in / out as a double, made float on the
// host, as PyTorch's CUDA multiply takes it), and the outputs over every plane.
struct CutieResizeBilinear {
    uint in_w;
    uint in_h;
    uint out_w;
    uint out_h;
    float scale_x;
    float scale_y;
    uint n;
};

// One axis's source position for output index `dst`, as torch._decomp's `_upsample_linear` does
// it with each step its own fp32 op: scale * (dst + 0.5) - 0.5, clamped at 0; the lower index
// the position truncated, the upper one past it but at the edge; lambda the position less the
// lower index.
static inline void cutie_resize_source(float scale, uint dst, uint size_in, thread uint& lower, thread uint& upper,
                                       thread float& lambda) {
    const float center = float(dst) + 0.5f;
    const float scaled = scale * center;
    float src = scaled - 0.5f;
    src = src < 0.0f ? 0.0f : src;
    lower = uint(src);
    upper = lower + (lower < size_in - 1 ? 1u : 0u);
    lambda = src - float(lower);
}

// a + (b - a) * t in three fp32 ops (the decomposition's `v1 + torch.mul(v2 - v1, xscale)`).
static inline float cutie_resize_lerp(float a, float b, float t) {
    const float diff = b - a;
    const float step = diff * t;
    return a + step;
}

// Lerps along the width first (each of the two source rows), then the height.
kernel void cutie_resize_bilinear_f32(constant CutieResizeBilinear& p [[buffer(0)]],
                                      device const float* x [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint out_plane = p.out_w * p.out_h;
    const uint at = i % out_plane;
    const uint oy = at / p.out_w;
    const uint ox = at % p.out_w;
    uint y0, y1, x0, x1;
    float h, w;
    cutie_resize_source(p.scale_y, oy, p.in_h, y0, y1, h);
    cutie_resize_source(p.scale_x, ox, p.in_w, x0, x1, w);
    device const float* src = x + (i / out_plane) * (p.in_w * p.in_h);
    device const float* row0 = src + y0 * p.in_w;
    device const float* row1 = src + y1 * p.in_w;
    const float top = cutie_resize_lerp(row0[x0], row0[x1], w);
    const float bottom = cutie_resize_lerp(row1[x0], row1[x1], w);
    out[i] = cutie_resize_lerp(top, bottom, h);
}
