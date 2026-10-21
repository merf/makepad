// Cutie (video object segmentation) device ops, in exact fp32: the native port is checked against
// a PyTorch fp32 oracle with TF32 off (models/vision/src/cutie*.rs). The Rust wrappers are in
// src/cutie_ops.rs; each op there has a host reference beside it that these kernels must match.
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// MaxPool2d(kernel 3, stride 2, padding 1) on planar [planes, height*width]: padding never wins
// (PyTorch pads with -inf), so a window takes the max of its in-bounds taps.
static __global__ void cutie_maxpool3x3s2_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    uint32_t width, uint32_t height, uint32_t ow, uint32_t oh, uint32_t planes) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = size_t(ow) * oh;
    if (i >= plane * planes) return;
    const size_t p = i / plane;
    const int ox = int(i % ow), oy = int((i % plane) / ow);
    const float* src = x + p * size_t(width) * height;
    float best = -INFINITY;
    for (int ky = 0; ky < 3; ++ky) {
        const int y = oy * 2 - 1 + ky;
        if (y < 0 || y >= int(height)) continue;
        for (int kx = 0; kx < 3; ++kx) {
            const int xx = ox * 2 - 1 + kx;
            if (xx < 0 || xx >= int(width)) continue;
            best = fmaxf(best, src[size_t(y) * width + xx]);
        }
    }
    out[i] = best;
}

extern "C" cudaError_t makepad_cuda_cutie_maxpool3x3s2_f32(
    const float* x, float* out, uint32_t width, uint32_t height, uint32_t ow, uint32_t oh,
    uint32_t planes, cudaStream_t stream) {
    const size_t n = size_t(ow) * oh * planes;
    if (!n) return cudaSuccess;
    cutie_maxpool3x3s2_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, out, width, height, ow, oh, planes);
    return cudaGetLastError();
}
