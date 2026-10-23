// The convolution on MPS (MAKEPAD_CUTIE_CONV=mpscnn, metal/src/cutie_ops/matmul_conv.rs): the
// planar tensors in and out of the rgba32Float texture arrays MPSCNNConvolution reads and
// writes. An MPSImage of C feature channels holds each image as ceil(C/4) slices, channels
// 4s..4s+3 in slice s's r, g, b and a; image i's slice s is the array's layer i*slices + s. The
// channels past C in the last slice are written 0 and never read back.
//
// Coming out, the residual is added and then the ReLU taken (fmax(x, 0), so a NaN becomes 0, as
// gpu_birefnet_relu), in PyTorch's order after the convolution and its bias, which MPS adds.
// One thread a pixel and slice; the host makes sure every count fits a uint.
#include "makepad_common.h"

// The images' planar shape: images, channels, the plane, and slices an image (ceil(channels/4)).
struct CutieImageShape {
    uint images;
    uint channels;
    uint width;
    uint height;
    uint slices;
    uint has_residual;
    uint relu;
};

kernel void cutie_planar_to_image_f32(constant CutieImageShape& p [[buffer(0)]],
                                      device const float* x [[buffer(1)]],
                                      texture2d_array<float, access::write> image [[texture(0)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.images * p.slices) {
        return;
    }
    const uint n = gid.z / p.slices;
    const uint slice = gid.z % p.slices;
    const uint plane = p.width * p.height;
    const uint at = gid.y * p.width + gid.x;
    float4 v = float4(0.0f);
    for (uint j = 0; j < 4; ++j) {
        const uint c = slice * 4 + j;
        if (c < p.channels) {
            v[j] = x[(n * p.channels + c) * plane + at];
        }
    }
    image.write(v, uint2(gid.x, gid.y), gid.z);
}

kernel void cutie_image_to_planar_f32(constant CutieImageShape& p [[buffer(0)]],
                                      device float* out [[buffer(1)]],
                                      device const float* residual [[buffer(2)]],
                                      texture2d_array<float, access::read> image [[texture(0)]],
                                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height || gid.z >= p.images * p.slices) {
        return;
    }
    const uint n = gid.z / p.slices;
    const uint slice = gid.z % p.slices;
    const uint plane = p.width * p.height;
    const uint at = gid.y * p.width + gid.x;
    const float4 v = image.read(uint2(gid.x, gid.y), gid.z);
    for (uint j = 0; j < 4; ++j) {
        const uint c = slice * 4 + j;
        if (c < p.channels) {
            const uint i = (n * p.channels + c) * plane + at;
            float r = v[j];
            if (p.has_residual != 0) {
                r = r + residual[i];
            }
            if (p.relu != 0) {
                r = fmax(r, 0.0f);
            }
            out[i] = r;
        }
    }
}

// --- The convolution on MPSGraph (an fp16 step's, metal/src/stream/mps.rs) ---
//
// MPSGraph reads and writes buffers from their start, in NCHW, which is the planar layout. In:
// the planar input rounded to half into the graph's input buffer (round to nearest even). Out:
// the graph's fp32 result (its bias added in the graph) into the planar output, with the
// residual added and then the ReLU taken, as cutie_image_to_planar_f32.

struct CutieGraphCopy {
    uint n;             // the elements
    uint has_residual;
    uint relu;
};

kernel void cutie_planar_to_half_f32(constant CutieGraphCopy& p [[buffer(0)]],
                                     device const float* x [[buffer(1)]],
                                     device half* y [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        y[i] = half(x[i]);
    }
}

kernel void cutie_graph_to_planar_f32(constant CutieGraphCopy& p [[buffer(0)]],
                                      device const float* y [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      device const float* residual [[buffer(3)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    float r = y[i];
    if (p.has_residual != 0) {
        r = r + residual[i];
    }
    if (p.relu != 0) {
        r = fmax(r, 0.0f);
    }
    out[i] = r;
}
