// PyTorch's LayerNorm (RationalVisions docs/cutie-metal.md §6): a port of cuda/kernels/
// diffusion_ops.cu's makepad_cuda_layer_norm_pytorch_f32_kernel in the same order of evaluation,
// launched by metal/src/layer_norm.rs for gpu_layer_norm_pytorch (every model's, on macOS) and by
// cutie_ops/softmax_norm.rs for gpu_cutie_layer_norm (its affine cached), so the two agree to the
// bit as they do on CUDA. A library of its own: a fault in one of Cutie's libraries breaks Cutie's
// ops, never the layer norms of the models that are not Cutie.
//
// Exactness (D8): fast math is off.
// - Where nvcc contracted (its PTX at -O3 has fma.rn for each Welford update, the combine's mean
//   and sigma2, and the affine) the kernel calls fma() on nvcc's operands, and no product is left
//   that feeds a sum. The bits are then CUDA's but for rsqrt: CUDA's rsqrtf is rsqrt.approx, this
//   is precise::rsqrt.
// - The reductions keep CUDA's trees: the shuffle-down tree in each SIMD-group (simd_shuffle_down,
//   never simd_sum) and PyTorch's compact tree across the four groups. They take CUDA's 32 lanes:
//   the host refuses a GPU whose SIMD-groups are another width.
// Indices that multiply two sizes are ulong, as CUDA's size_t. No 16-bit float anywhere.
#include "makepad_common.h"

// This library's helpers, in a namespace of their own so they never meet makepad_common.h's.
namespace layer_norm {

// A row's running moments in PyTorch's LayerNorm (diffusion_ops.cu's makepad_cuda_welford_ln).
struct Welford {
    float mean;
    float sigma2;
    float count;
};

// One value into the moments (cuWelfordOnlineSum): nvcc fuses both updates.
static inline Welford welford_online(float value, Welford current) {
    const float delta = value - current.mean;
    const float new_count = current.count + 1.0f;
    const float new_mean = fma(delta, 1.0f / new_count, current.mean);
    Welford next;
    next.mean = new_mean;
    next.sigma2 = fma(delta, value - new_mean, current.sigma2);
    next.count = new_count;
    return next;
}

// Two threads' moments into one (cuWelfordCombine), `b` the thread's own and `a` the one it
// takes in: mean = n_a * a.mean + n_b * b.mean, nvcc fusing the first product (fma(n_a, a.mean,
// rounded n_b * b.mean)); sigma2 = a.sigma2 + b.sigma2 + delta * delta * a.count * n_b, the last
// product fused. No moments at all give zeros.
static inline Welford welford_combine(Welford b, Welford a) {
    const float delta = b.mean - a.mean;
    const float count = a.count + b.count;
    Welford out;
    if (count <= 0.0f) {
        out.mean = 0.0f;
        out.sigma2 = 0.0f;
        out.count = 0.0f;
        return out;
    }
    const float coefficient = 1.0f / count;
    const float n_a = a.count * coefficient;
    const float n_b = b.count * coefficient;
    const float b_part = n_b * b.mean;
    const float delta2 = delta * delta;
    const float spread = delta2 * a.count;
    const float sigmas = a.sigma2 + b.sigma2;
    out.mean = fma(n_a, a.mean, b_part);
    out.sigma2 = fma(spread, n_b, sigmas);
    out.count = count;
    return out;
}

}  // namespace layer_norm

// x and out [rows, cols], cols % 4 == 0; gamma and beta [cols].
struct LayerNormPytorch {
    uint cols;
    float eps;
};

// nn.LayerNorm over each row as PyTorch's CUDA fast path computes it, a threadgroup [32, 4] a row
// (PyTorch's dim3(warp_size, 4)): each thread's Welford moments over its four-value steps (the
// values of a float4 in order, read here one at a time since a view may sit at any 4 bytes), a
// shuffle-down tree in each SIMD-group, then PyTorch's compact tree across the four groups through
// `stats` (each pair's mean and sigma2, then the counts), the biased variance sigma2 / cols, and
// gamma * (rstd * (x - mean)) + beta with the last product fused, as nvcc does.
kernel void layer_norm_pytorch_f32(constant LayerNormPytorch& p [[buffer(0)]],
                                   device const float* x [[buffer(1)]],
                                   device const float* gamma [[buffer(2)]],
                                   device const float* beta [[buffer(3)]],
                                   device float* out [[buffer(4)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint2 thread_in_group [[thread_position_in_threadgroup]],
                                   uint2 group_size [[threads_per_threadgroup]]) {
    threadgroup float stats[6];
    const uint row = group.x;
    const uint lane = thread_in_group.x;
    const uint warp = thread_in_group.y;
    const uint flat_thread = lane + warp * group_size.x;
    const uint flat_threads = group_size.x * group_size.y;
    const uint vectors = p.cols / 4;
    device const float* input = x + ulong(row) * p.cols;
    device float* output = out + ulong(row) * p.cols;

    layer_norm::Welford wd;
    wd.mean = 0.0f;
    wd.sigma2 = 0.0f;
    wd.count = 0.0f;
    for (uint index = flat_thread; index < vectors; index += flat_threads) {
        const uint at = 4 * index;
        wd = layer_norm::welford_online(input[at], wd);
        wd = layer_norm::welford_online(input[at + 1], wd);
        wd = layer_norm::welford_online(input[at + 2], wd);
        wd = layer_norm::welford_online(input[at + 3], wd);
    }
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        layer_norm::Welford other;
        other.mean = simd_shuffle_down(wd.mean, offset);
        other.sigma2 = simd_shuffle_down(wd.sigma2, offset);
        other.count = simd_shuffle_down(wd.count, offset);
        wd = layer_norm::welford_combine(wd, other);
    }

    // PyTorch allocates warps * 3 / 2 floats: each pair's mean and sigma2, then the counts.
    threadgroup float* mean_sigma = stats;
    threadgroup float* counts = stats + group_size.y;
    for (uint offset = group_size.y / 2; offset > 0; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            const uint write_y = warp - offset;
            mean_sigma[2 * write_y] = wd.mean;
            mean_sigma[2 * write_y + 1] = wd.sigma2;
            counts[write_y] = wd.count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && warp < offset) {
            layer_norm::Welford other;
            other.mean = mean_sigma[2 * warp];
            other.sigma2 = mean_sigma[2 * warp + 1];
            other.count = counts[warp];
            wd = layer_norm::welford_combine(wd, other);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0 && warp == 0) {
        mean_sigma[0] = wd.mean;
        mean_sigma[1] = wd.sigma2 / float(p.cols);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = mean_sigma[0];
    const float rstd = precise::rsqrt(mean_sigma[1] + p.eps);
    for (uint index = flat_thread; index < vectors; index += flat_threads) {
        for (uint k = 0; k < 4; ++k) {
            const uint at = 4 * index + k;
            const float centred = input[at] - mean;
            const float scaled = rstd * centred;
            output[at] = fma(gamma[at], scaled, beta[at]);
        }
    }
}
