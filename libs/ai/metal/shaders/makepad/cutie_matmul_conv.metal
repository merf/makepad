// Cutie's convolutions and memory reading (family (d) of RationalVisions docs/cutie-metal.md, step
// 10): the f32 columns a convolution's GEMM multiplies (im2col), the pass after it (the bias, a
// residual and a ReLU; the side-by-side batch put back in place), and the similarity's operands
// and epilogue around its two GEMMs. The GEMMs are MetalStream::gemm's (gemm.metal). Ported kernel
// for kernel from cuda/kernels/cutie.cu; metal/src/cutie_ops/matmul_conv.rs plans and launches
// them, with the same shapes, grids and loops as CUDA's wrappers.
//
// Tensors and weights are bound at any 4-byte offset, so they are `device` pointers read one
// float at a time. Indices into a tensor are ulong; sizes and row counts are uint (the wrappers
// check they fit).
//
// Exactness (D8): every value is one IEEE operation in the order cutie.cu writes it with its _rn
// intrinsics, and a NaN goes where CUDA's goes. im2col and the unstack only move floats; the
// convolution's pass only adds, then fmax; the similarity's epilogue multiplies only after its
// sums. Two lines would change if a product were contracted into the add that follows it: the
// similarity's sums b_sq and a_sq (cutie_similarity_prep_f32). They rely on makepad_common.h's
// contraction setting (step 7). So does -sim + 2*x in the epilogue, where contraction changes the
// result only when 2*x overflows.
#include "makepad_common.h"

// The columns of `images` planar images from `first_image` on, for a square k x k window with
// stride and zero padding (cutie.cu's cutie_im2col_kernel).
struct CutieIm2col {
    uint images;
    uint first_image;
    uint channels;
    uint width;
    uint height;
    uint ow;
    uint oh;
    uint k;
    uint pad;
    uint stride;
    uint rows;          // images * channels * k * k
    uint side_by_side;  // 0: col[b][r][q], one GEMM an image; else col[r][b][q], one GEMM for all
};

// The pass after a convolution's GEMM, over `rows` = images * channels output rows of `plane`
// pixels. `images` is read only by the unstack.
struct CutieConvEpilogue {
    uint images;
    uint channels;
    uint plane;
    uint rows;
    uint has_bias;
    uint has_residual;
    uint relu;
};

// The similarity's sizes: memory keys [ck, m], query keys [ck, p]; has_qe: a selection is given.
struct CutieSimilarityPrep {
    uint ck;
    uint m;
    uint p;
    uint has_qe;
};

// The similarity [m, p] from its GEMMs: has_two_ab with a selection, has_ms with a shrinkage.
struct CutieSimilarityEpilogue {
    uint m;
    uint p;
    float inv_scale;
    uint has_two_ab;
    uint has_ms;
};

// --- Convolution: f32 im2col, then MetalStream::gemm, then the bias, residual and ReLU ---

// Row r = c*k*k + ky*k + kx and column q = oy*ow + ox of each image's columns: the layout
// PyTorch's [out][in][k][k] weights multiply as an [out, in*k*k] matrix. Per image (col[b][r][q])
// or, side by side, the images' columns next to each other (col[r][b][q]). Threads along x are
// output pixels; threadgroups along y take rows, looping past the grid's 65535 as CUDA's do.
kernel void cutie_im2col_f32(constant CutieIm2col& p [[buffer(0)]],
                             device const float* x [[buffer(1)]],
                             device float* col [[buffer(2)]],
                             uint2 pos [[thread_position_in_grid]],
                             uint2 groups [[threadgroups_per_grid]]) {
    const ulong opix = ulong(p.ow) * p.oh;
    const uint q = pos.x;
    if (q >= opix) {
        return;
    }
    const int oy = int(q / p.ow), ox = int(q % p.ow);
    const uint kk = p.k * p.k, kdim = p.channels * kk;
    for (uint row = pos.y; row < p.rows; row += groups.y) {
        const uint b = row / kdim;
        const uint r = row % kdim;
        const uint c = r / kk, ky = (r % kk) / p.k, kx = r % p.k;
        const int iy = oy * int(p.stride) - int(p.pad) + int(ky);
        const int ix = ox * int(p.stride) - int(p.pad) + int(kx);
        float v = 0.0f;
        if (iy >= 0 && iy < int(p.height) && ix >= 0 && ix < int(p.width)) {
            v = x[((ulong(p.first_image + b) * p.channels + c) * p.height + uint(iy)) * p.width + uint(ix)];
        }
        const ulong at = p.side_by_side != 0 ? ulong(r) * p.images + b : ulong(row);
        col[at * opix + q] = v;
    }
}

// What a convolution does after its product, in one pass: the bias added, then the residual (a
// skip connection, the output's layout), then the ReLU, each rounded as the separate ops it
// replaces round it (gpu_add's a + b, gpu_birefnet_relu's fmax(x, 0), which makes a NaN 0), so the
// result is theirs to the bit (cutie.cu's cutie_conv_epilogue).
static inline float cutie_conv_epilogue(float v,
                                        constant CutieConvEpilogue& p,
                                        device const float* bias,
                                        uint c,
                                        device const float* residual,
                                        ulong at) {
    if (p.has_bias != 0) {
        v = v + bias[c];
    }
    if (p.has_residual != 0) {
        v = v + residual[at];
    }
    return p.relu != 0 ? fmax(v, 0.0f) : v;
}

// The side-by-side GEMM's result [channels][images][plane] to the images' planar layout
// [images][channels][plane], with the epilogue (cutie.cu's cutie_unstack_epilogue_kernel).
kernel void cutie_unstack_epilogue_f32(constant CutieConvEpilogue& p [[buffer(0)]],
                                       device const float* src [[buffer(1)]],
                                       device float* out [[buffer(2)]],
                                       device const float* bias [[buffer(3)]],
                                       device const float* residual [[buffer(4)]],
                                       uint2 pos [[thread_position_in_grid]],
                                       uint2 groups [[threadgroups_per_grid]]) {
    const uint q = pos.x;
    if (q >= p.plane) {
        return;
    }
    for (uint row = pos.y; row < p.rows; row += groups.y) {
        const uint b = row / p.channels, c = row % p.channels;
        const ulong at = ulong(row) * p.plane + q;
        const float v = src[(ulong(c) * p.images + b) * p.plane + q];
        out[at] = cutie_conv_epilogue(v, p, bias, c, residual, at);
    }
}

// The epilogue in place over [rows, plane], the bias by row % channels (cutie.cu's
// cutie_conv_epilogue_kernel).
kernel void cutie_conv_epilogue_f32(constant CutieConvEpilogue& p [[buffer(0)]],
                                    device float* out [[buffer(1)]],
                                    device const float* bias [[buffer(2)]],
                                    device const float* residual [[buffer(3)]],
                                    uint2 pos [[thread_position_in_grid]],
                                    uint2 groups [[threadgroups_per_grid]]) {
    const uint q = pos.x;
    if (q >= p.plane) {
        return;
    }
    for (uint row = pos.y; row < p.rows; row += groups.y) {
        const ulong at = ulong(row) * p.plane + q;
        out[at] = cutie_conv_epilogue(out[at], p, bias, row % p.channels, residual, at);
    }
}

// --- Memory reading ---

// The similarity's operands before its GEMMs, as PyTorch forms them: mk^2 [ck, m] and, per query
// pixel, qk * qe [ck, p] and b_sq = sum_c qe * qk^2 [p]; without a selection, a_sq = sum_c mk^2
// per memory token [m]. One thread a token and a pixel (max(m, p) threads); the channel sums run
// in channel order (cutie.cu's cutie_similarity_prep_kernel).
kernel void cutie_similarity_prep_f32(constant CutieSimilarityPrep& dims [[buffer(0)]],
                                      device const float* mk [[buffer(1)]],
                                      device const float* qk [[buffer(2)]],
                                      device const float* qe [[buffer(3)]],
                                      device float* mk_sq [[buffer(4)]],
                                      device float* qkqe [[buffer(5)]],
                                      device float* b_sq [[buffer(6)]],
                                      device float* a_sq [[buffer(7)]],
                                      uint t [[thread_position_in_grid]]) {
    if (dims.has_qe != 0) {
        if (t < dims.m) {
            for (uint c = 0; c < dims.ck; ++c) {
                const float v = mk[ulong(c) * dims.m + t];
                mk_sq[ulong(c) * dims.m + t] = v * v;
            }
        }
        if (t < dims.p) {
            float sum = 0.0f;
            for (uint c = 0; c < dims.ck; ++c) {
                const ulong at = ulong(c) * dims.p + t;
                const float k = qk[at], e = qe[at];
                qkqe[at] = k * e;
                // Not contracted (step 7's setting): CUDA's __fadd_rn(sum, __fmul_rn(e, __fmul_rn(k, k))).
                const float term = e * (k * k);
                sum = sum + term;
            }
            b_sq[t] = sum;
        }
    } else if (t < dims.m) {
        float sum = 0.0f;
        for (uint c = 0; c < dims.ck; ++c) {
            const float v = mk[ulong(c) * dims.m + t];
            // Not contracted (step 7's setting): CUDA's __fadd_rn(sum, __fmul_rn(v, v)).
            const float term = v * v;
            sum = sum + term;
        }
        a_sq[t] = sum;
    }
}

// The similarity from its GEMMs, in the reference's order: with a selection, `sim` holds a_sq and
// `two_ab` mk^T (qk*qe): ((-a_sq + 2 two_ab) - b_sq); without, `sim` holds mk^T qk and a_sq is per
// token: (-a_sq + 2 sim). Then * ms (when given), then * inv_scale (PyTorch divides by the scalar
// sqrt(ck) as a multiply by its reciprocal). In place in `sim` (cutie.cu's
// cutie_similarity_epilogue_kernel). Threads along x are pixels; threadgroups along y take memory
// tokens, looping past 65535.
kernel void cutie_similarity_epilogue_f32(constant CutieSimilarityEpilogue& e [[buffer(0)]],
                                          device float* sim [[buffer(1)]],
                                          device const float* two_ab [[buffer(2)]],
                                          device const float* b_sq [[buffer(3)]],
                                          device const float* a_sq [[buffer(4)]],
                                          device const float* ms [[buffer(5)]],
                                          uint2 pos [[thread_position_in_grid]],
                                          uint2 groups [[threadgroups_per_grid]]) {
    const uint px = pos.x;
    if (px >= e.p) {
        return;
    }
    for (uint j = pos.y; j < e.m; j += groups.y) {
        const ulong i = ulong(j) * e.p + px;
        float s;
        if (e.has_two_ab != 0) {
            const float twice = 2.0f * two_ab[i];
            s = (-sim[i] + twice) - b_sq[px];
        } else {
            const float twice = 2.0f * sim[i];
            s = -a_sq[j] + twice;
        }
        if (e.has_ms != 0) {
            s = s * ms[j];
        }
        sim[i] = s * e.inv_scale;
    }
}
