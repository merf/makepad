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

// cutie_conv_epilogue with its params in thread memory (cutie_conv_gemm_f32 builds them).
static inline float cutie_conv_epilogue_p(float v,
                                          thread const CutieConvEpilogue& p,
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

// --- Convolution in one pass: im2col, the GEMM and the epilogue together ---

// A convolution of `images` images as one GEMM, n = images * plane columns: the [out, kdim]
// weights times the columns im2col would write, read here straight from the input as each step of
// k is staged, and the epilogue applied as each output is written to its image's plane. The tile,
// threads and walk down k are gemm.metal's gemm_tiled_f32's (32x32, four SIMD groups, 16 deep,
// then 8 at a time, zero past its edge), so each output is the same multiply-accumulates in the
// same order as im2col then MetalStream::gemm on the tiled kernel: the same bits, with the columns
// never written or read and the epilogue's pass gone (gemm_result is acc itself for alpha 1, beta
// 0). matmul_conv.rs uses it only where that GEMM would run on the tiled kernel.
struct CutieConvGemm {
    uint images;
    uint channels;      // input channels
    uint width;
    uint height;
    uint ow;
    uint oh;
    uint k;
    uint pad;
    uint stride;
    uint out_channels;
    uint kdim;          // channels * k * k
    uint plane;         // ow * oh
    uint has_bias;
    uint has_residual;
    uint relu;
    // A step of k (CONV_TILE_K rows) as whole channels, rows and columns of the window:
    // CONV_TILE_K = step_c * k * k + step_y * k + step_x.
    uint step_c;
    uint step_y;
    uint step_x;
};

#define CONV_TILE 32u
#define CONV_TILE_K 16u
#define CONV_THREADS 128u
#define CONV_A_LD (CONV_TILE_K + 4u)
#define CONV_B_LD (CONV_TILE + 4u)
#define CONV_A_ROWS_PER_PASS (CONV_THREADS / CONV_TILE_K)
#define CONV_A_PASSES (CONV_TILE / CONV_A_ROWS_PER_PASS)
#define CONV_B_ROWS_PER_PASS (CONV_THREADS / CONV_TILE)
#define CONV_B_PASSES (CONV_TILE_K / CONV_B_ROWS_PER_PASS)
#define CONV_STAGE (CONV_TILE * CONV_A_LD + CONV_TILE_K * CONV_B_LD)

// (x, y) = (column tile, row tile); 128 threads, four SIMD groups of 32.
kernel void cutie_conv_gemm_f32(constant CutieConvGemm& p [[buffer(0)]],
                                device const float* x [[buffer(1)]],
                                device const float* w [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                device const float* bias [[buffer(4)]],
                                device const float* residual [[buffer(5)]],
                                uint3 tile [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tiles[CONV_STAGE];
    threadgroup float* a_tile = tiles;
    threadgroup float* b_tile = tiles + CONV_TILE * CONV_A_LD;

    const uint m = p.out_channels;
    const uint n = p.images * p.plane;
    const uint row0 = tile.y * CONV_TILE;
    const uint col0 = tile.x * CONV_TILE;

    // This thread's staging: weights column a_kk of rows a_r + 8 * pass, and column b_j's pixel
    // (image b_img at oy, ox) for rows b_kk + 4 * pass of k.
    const uint a_kk = tid % CONV_TILE_K;
    const uint a_r = tid / CONV_TILE_K;
    const uint b_j = tid % CONV_TILE;
    const uint b_kk = tid / CONV_TILE;
    const uint j = col0 + b_j;
    const bool b_in = j < n;
    const uint b_img = b_in ? j / p.plane : 0u;
    const uint q = b_in ? j % p.plane : 0u;
    const int oy = int(q / p.ow), ox = int(q % p.ow);
    device const float* x_img = x + ulong(b_img) * p.channels * p.height * p.width;

    // Each of this thread's rows of k as its channel and window position, worked out once and
    // moved on by a step at a time: no division in the loop.
    uint row_c[CONV_B_PASSES], row_y[CONV_B_PASSES], row_x[CONV_B_PASSES];
#pragma clang loop unroll(full)
    for (uint pass = 0; pass < CONV_B_PASSES; ++pass) {
        const uint kk = b_kk + pass * CONV_B_ROWS_PER_PASS;
        row_c[pass] = kk / (p.k * p.k);
        row_y[pass] = (kk % (p.k * p.k)) / p.k;
        row_x[pass] = kk % p.k;
    }

    const uint sub_row = (sg / 2u) * 16u;
    const uint sub_col = (sg % 2u) * 16u;
    simdgroup_float8x8 acc[4];
    for (uint s = 0; s < 4; ++s) {
        acc[s] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < p.kdim; k0 += CONV_TILE_K) {
        const uint kk_a = k0 + a_kk;
#pragma clang loop unroll(full)
        for (uint pass = 0; pass < CONV_A_PASSES; ++pass) {
            const uint r = a_r + pass * CONV_A_ROWS_PER_PASS;
            const uint i = row0 + r;
            a_tile[r * CONV_A_LD + a_kk] = (i < m && kk_a < p.kdim) ? w[ulong(i) * p.kdim + kk_a] : 0.0f;
        }
#pragma clang loop unroll(full)
        for (uint pass = 0; pass < CONV_B_PASSES; ++pass) {
            const uint kr = b_kk + pass * CONV_B_ROWS_PER_PASS;
            const uint kk = k0 + kr;
            float v = 0.0f;
            if (b_in && kk < p.kdim) {
                const int iy = oy * int(p.stride) - int(p.pad) + int(row_y[pass]);
                const int ix = ox * int(p.stride) - int(p.pad) + int(row_x[pass]);
                if (iy >= 0 && iy < int(p.height) && ix >= 0 && ix < int(p.width)) {
                    v = x_img[(ulong(row_c[pass]) * p.height + uint(iy)) * p.width + uint(ix)];
                }
            }
            b_tile[kr * CONV_B_LD + b_j] = v;
            // The next step's row: CONV_TILE_K further down k.
            row_x[pass] += p.step_x;
            row_y[pass] += p.step_y;
            row_c[pass] += p.step_c;
            if (row_x[pass] >= p.k) {
                row_x[pass] -= p.k;
                row_y[pass] += 1u;
            }
            if (row_y[pass] >= p.k) {
                row_y[pass] -= p.k;
                row_c[pass] += 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < CONV_TILE_K; s += 8) {
            simdgroup_float8x8 am[2];
            simdgroup_float8x8 bm[2];
            simdgroup_load(am[0], a_tile + (sub_row + 0) * CONV_A_LD + s, CONV_A_LD);
            simdgroup_load(am[1], a_tile + (sub_row + 8) * CONV_A_LD + s, CONV_A_LD);
            simdgroup_load(bm[0], b_tile + s * CONV_B_LD + sub_col + 0, CONV_B_LD);
            simdgroup_load(bm[1], b_tile + s * CONV_B_LD + sub_col + 8, CONV_B_LD);
            simdgroup_multiply_accumulate(acc[0], am[0], bm[0], acc[0]);
            simdgroup_multiply_accumulate(acc[1], am[0], bm[1], acc[1]);
            simdgroup_multiply_accumulate(acc[2], am[1], bm[0], acc[2]);
            simdgroup_multiply_accumulate(acc[3], am[1], bm[1], acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc[0], tiles + (sub_row + 0) * CONV_TILE + sub_col + 0, CONV_TILE);
    simdgroup_store(acc[1], tiles + (sub_row + 0) * CONV_TILE + sub_col + 8, CONV_TILE);
    simdgroup_store(acc[2], tiles + (sub_row + 8) * CONV_TILE + sub_col + 0, CONV_TILE);
    simdgroup_store(acc[3], tiles + (sub_row + 8) * CONV_TILE + sub_col + 8, CONV_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    CutieConvEpilogue e;
    e.images = p.images;
    e.channels = m;
    e.plane = p.plane;
    e.rows = p.images * m;
    e.has_bias = p.has_bias;
    e.has_residual = p.has_residual;
    e.relu = p.relu;
    for (uint t = tid; t < CONV_TILE * CONV_TILE; t += CONV_THREADS) {
        const uint i = row0 + t / CONV_TILE;
        const uint jj = col0 + t % CONV_TILE;
        if (i < m && jj < n) {
            const uint img = jj / p.plane, qq = jj % p.plane;
            const ulong at = (ulong(img) * m + i) * p.plane + qq;
            const float v = tiles[t];
            out[at] = (p.has_bias | p.has_residual | p.relu) != 0 ? cutie_conv_epilogue_p(v, e, bias, i, residual, at) : v;
        }
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
