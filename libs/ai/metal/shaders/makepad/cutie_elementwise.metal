// Cutie's elementwise and shape kernels (RationalVisions docs/cutie-metal.md, family (a)): the
// sigmoid, the sensory GRU's update, others, pad, crop, add_broadcast, normalise_rows, rgb_planes
// and object_values, ported from cuda/kernels/cutie.cu. metal/src/cutie_ops/elementwise.rs names
// and launches them, its params structs field for field these.
//
// Exactness (D8):
// - The library compiles with fast math off; exp and tanh are precise::, the divisions IEEE.
// - Where cutie.cu writes __fadd_rn, __fsub_rn, __fmul_rn or __fdiv_rn, each operation is its own
//   statement, so a product is never contracted into the sum after it (a contraction setting that
//   stays inside one expression cannot reach it; makepad_common.h's step-7 setting covers the
//   rest).
// - One thread an element (others: one a column) where CUDA loops over a capped grid; each element
//   is computed alone, so the results are the same.
// - Pad and crop move 32-bit words, not floats, so every bit pattern arrives unchanged, as CUDA's
//   float copies leave it.
// - Indices into an input that can be longer than the output are 64-bit, as CUDA's size_t.
//
// Names are prefixed so they never meet a helper in makepad_common.h.
#include "makepad_common.h"

// PyTorch's sigmoid for f32 (cutie.cu's cutie_sigmoid): precise exp, an IEEE division.
static inline float elementwise_sigmoid(float x) {
    return 1.0f / (1.0f + precise::exp(-x));
}

// --- Elementwise ---

struct CutieSigmoid {
    uint n;
};

kernel void cutie_sigmoid_f32(constant CutieSigmoid& p [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = elementwise_sigmoid(x[i]);
    }
}

// Cutie's sensory GRU: per object o, values rows [o*3c, o*3c+c) forget, then update, then new;
// new_h = forget * h * (1 - update) + update * tanh(new), each product and sum rounded as PyTorch's
// separate tensor ops round them.
struct CutieGru {
    uint cp;  // channels * plane: one object's h
    uint n;   // objects * cp
};

kernel void cutie_gru_f32(constant CutieGru& p [[buffer(0)]],
                          device const float* h [[buffer(1)]],
                          device const float* values [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint o = i / p.cp;
    const uint rem = i % p.cp;
    device const float* v = values + ulong(o) * 3 * ulong(p.cp);
    const float forget = elementwise_sigmoid(v[rem]);
    const float update = elementwise_sigmoid(v[ulong(p.cp) + rem]);
    const float fresh = precise::tanh(v[2 * ulong(p.cp) + rem]);
    const float forget_h = forget * h[i];
    const float keep = 1.0f - update;
    const float kept = forget_h * keep;
    const float added = update * fresh;
    out[i] = kept + added;
}

// For each object's mask [objects, pixels], the others' sum (all objects summed in PyTorch's
// order, then the object's own subtracted), clamped to 0..1 with NaN passing through. The sum is
// cutie.cu's cutie_reduce4: four accumulators, term o into o % 4, then ((a0 + a1) + a2) + a3.
struct CutieOthers {
    uint objects;
    uint pixels;
};

kernel void cutie_others_f32(constant CutieOthers& p [[buffer(0)]],
                             device const float* masks [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p.pixels) {
        return;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint o = 0; o < p.objects; o += 4) {
        for (uint j = 0; j < 4; ++j) {
            if (o + j < p.objects) {
                acc[j] = acc[j] + masks[ulong(o + j) * p.pixels + i];
            }
        }
    }
    const float first_two = acc[0] + acc[1];
    const float first_three = first_two + acc[2];
    const float sum = first_three + acc[3];
    for (uint o = 0; o < p.objects; ++o) {
        const ulong at = ulong(o) * p.pixels + i;
        const float d = sum - masks[at];
        out[at] = d < 0.0f ? 0.0f : (d > 1.0f ? 1.0f : d);
    }
}

// --- Shapes ---

// Zero padding of planar [planes, h*w] to [planes, nh*nw], the image at (left, top).
struct CutiePad {
    uint w;
    uint h;
    uint nw;
    uint nh;
    uint left;
    uint top;
    uint n;  // planes * nw * nh
};

kernel void cutie_pad_f32(constant CutiePad& p [[buffer(0)]],
                          device const uint* x [[buffer(1)]],
                          device uint* out [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint plane = p.nw * p.nh;
    const uint at_plane = i / plane;
    const uint at = i % plane;
    const int y = int(at / p.nw) - int(p.top);
    const int xx = int(at % p.nw) - int(p.left);
    const bool inside = y >= 0 && y < int(p.h) && xx >= 0 && xx < int(p.w);
    out[i] = inside ? x[(ulong(at_plane) * p.h + uint(y)) * p.w + uint(xx)] : 0u;
}

// A window of planar [planes, h*w]: out_w x out_h from (left, top).
struct CutieCrop {
    uint w;
    uint h;
    uint left;
    uint top;
    uint out_w;
    uint out_h;
    uint n;  // planes * out_w * out_h
};

kernel void cutie_crop_f32(constant CutieCrop& p [[buffer(0)]],
                           device const uint* x [[buffer(1)]],
                           device uint* out [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint plane = p.out_w * p.out_h;
    const uint at_plane = i / plane;
    const uint at = i % plane;
    const ulong y = ulong(at / p.out_w) + p.top;
    const ulong xx = ulong(at % p.out_w) + p.left;
    out[i] = x[(ulong(at_plane) * p.h + y) * p.w + xx];
}

// --- The per-frame step ---

// out[i] = a[i % a_len] + b[i]: one block `a` added to each block of `b` (Cutie's x + g of one
// image's features broadcast over the objects).
struct CutieAddBroadcast {
    uint a_len;
    uint n;
};

kernel void cutie_add_broadcast_f32(constant CutieAddBroadcast& p [[buffer(0)]],
                                    device const float* a [[buffer(1)]],
                                    device const float* b [[buffer(2)]],
                                    device float* out [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = a[i % p.a_len] + b[i];
    }
}

// (x - sub[row]) / divisor[row] over planar [rows, plane], a true division (CUTIE.encode_image's
// (image - pixel_mean) / pixel_std, each op rounded).
struct CutieNormaliseRows {
    uint plane;
    uint n;  // rows * plane
};

kernel void cutie_normalise_rows_f32(constant CutieNormaliseRows& p [[buffer(0)]],
                                     device const float* x [[buffer(1)]],
                                     device const float* sub [[buffer(2)]],
                                     device const float* divisor [[buffer(3)]],
                                     device float* out [[buffer(4)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint row = i / p.plane;
    const float centred = x[i] - sub[row];
    out[i] = centred / divisor[row];
}

// RGB bytes (plane pixels of 3) to planar f32 [3, plane], each byte times `scale` (PyTorch's CUDA
// x / 255 multiplies by the reciprocal). The bytes are a u32 tensor's words, read as bytes.
struct CutieRgbPlanes {
    uint plane;
    uint n;  // 3 * plane
    float scale;
};

kernel void cutie_rgb_planes_f32(constant CutieRgbPlanes& p [[buffer(0)]],
                                 device const uchar* rgb [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint c = i / p.plane;
    const uint px = i % p.plane;
    out[i] = float(rgb[ulong(px) * 3 + c]) * p.scale;
}

// The object transformer's values from the object memory [objects*queries, width+1] (each
// summary's sums, then its area): sums / (area + eps), true divisions, written interleaved
// [queries*objects, width] (row q*objects + o).
struct CutieObjectValues {
    uint objects;
    uint queries;
    uint width;
    float eps;
    uint n;  // objects * queries * width
};

kernel void cutie_object_values_f32(constant CutieObjectValues& p [[buffer(0)]],
                                    device const float* memory [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint row = i / p.width;
    const uint c = i % p.width;
    const uint q = row / p.objects;
    const uint o = row % p.objects;
    device const float* src = memory + (ulong(o) * p.queries + q) * (ulong(p.width) + 1);
    const float area = src[p.width] + p.eps;
    out[i] = src[c] / area;
}
