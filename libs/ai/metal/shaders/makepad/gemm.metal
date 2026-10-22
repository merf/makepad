// Makepad's fp32 GEMM and the two kernels its users add around it (RationalVisions
// docs/cutie-metal.md, section 7): metal/src/stream/gemm.rs names them and plans each GEMM's
// launch, MetalStream::gemm launches it, metal/src/linear.rs adds a bias after one and
// metal/src/attention.rs runs the softmax between two.
//
// C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i] for every batch i. Every operand is row-major
// at any 4-byte offset, so it is a `device` pointer read a word at a time (never float4), and is
// read through a row stride and a column stride (a transpose swaps them; gemm.rs sets them from
// the leading dimension), from its offset, a batch stride apart. Every product and sum is fp32,
// never staged through half (ggml's mul_mm stages both operands as half: D6):
// - gemm_naive_f32: one thread an output, an fma at a time down k in order. The baseline.
// - gemm_tiled_f32: a 32x32 output tile a threadgroup of four 32-wide SIMD groups, op(A) and
//   op(B) staged as fp32 in threadgroup memory 16 deep in k (zero past their edges), each SIMD
//   group multiplying 16x16 of the tile with simdgroup_float8x8 (fp32 multiply-accumulate).
// The two round differently (the order inside each 8-deep SIMD-group product is the hardware's),
// as cuBLAS differs from both: Cutie's checks hold a GEMM to its sum in f64 (TIGHT), not to bits.
//
// Exactness (D8): the multiply-accumulate is an explicit fma, and the epilogue is alpha * acc
// then one explicit fma with beta * C, so no contraction setting changes them. With beta 0, C is
// never read (an Out's contents are undefined, NaN with poisoned allocations); with k 0 there is no
// product and C becomes beta * C, as BLAS (gemm.rs passes k 0 for alpha 0 too).
#include "makepad_common.h"

// The output tile of gemm_tiled_f32 (rows and columns), its depth in k, and its threads (four
// SIMD groups of 32, arranged 2x2 over the tile). gemm.rs's TILE and TILED_THREADS match.
#define GEMM_TILE 32u
#define GEMM_TILE_K 16u
#define GEMM_TILED_THREADS 128u

// gemm.rs's GemmParams, field for field. op(A)[i][kk] is a[a_offset + batch * a_batch_stride +
// i * a_row_stride + kk * a_col_stride]; op(B)[kk][j] likewise; C[i][j] is at c_offset +
// batch * c_batch_stride + i * ldc + j.
struct GemmParams {
    uint m;
    uint n;
    uint k;
    uint batch;
    uint a_offset;
    uint a_row_stride;
    uint a_col_stride;
    uint a_batch_stride;
    uint b_offset;
    uint b_row_stride;
    uint b_col_stride;
    uint b_batch_stride;
    uint c_offset;
    uint ldc;
    uint c_batch_stride;
    float alpha;
    float beta;
};

// alpha * acc + beta * C into C, as cuBLAS's epilogue rounds it: the product, then one fma with
// beta * C. C is read only when beta is not 0.
static inline float gemm_result(constant GemmParams& p, float acc, device const float* c) {
    if (p.k == 0) {
        return p.beta == 0.0f ? 0.0f : p.beta * *c;
    }
    const float product = p.alpha * acc;
    return p.beta == 0.0f ? product : fma(p.beta, *c, product);
}

// One thread an output: (x, y, z) = (column, row, batch).
kernel void gemm_naive_f32(constant GemmParams& p [[buffer(0)]],
                           device const float* a [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* c [[buffer(3)]],
                           uint3 pos [[thread_position_in_grid]]) {
    const uint j = pos.x;
    const uint i = pos.y;
    if (i >= p.m || j >= p.n || pos.z >= p.batch) {
        return;
    }
    const ulong batch = pos.z;
    device const float* a_row = a + (ulong(p.a_offset) + batch * p.a_batch_stride + ulong(i) * p.a_row_stride);
    device const float* b_col = b + (ulong(p.b_offset) + batch * p.b_batch_stride + ulong(j) * p.b_col_stride);
    float acc = 0.0f;
    for (uint kk = 0; kk < p.k; ++kk) {
        acc = fma(a_row[ulong(kk) * p.a_col_stride], b_col[ulong(kk) * p.b_row_stride], acc);
    }
    device float* out = c + (ulong(p.c_offset) + batch * p.c_batch_stride + ulong(i) * p.ldc + j);
    *out = gemm_result(p, acc, out);
}

// A threadgroup a 32x32 tile of one batch's output: (x, y, z) = (column tile, row tile, batch).
// 128 threads, four SIMD groups of 32 (gemm.rs refuses another SIMD width).
kernel void gemm_tiled_f32(constant GemmParams& p [[buffer(0)]],
                           device const float* a [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* c [[buffer(3)]],
                           uint3 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    // op(A)'s 32x16 block then op(B)'s 16x32 block, each row-major; after the k loop the same
    // memory holds the 32x32 result for the epilogue.
    threadgroup float tiles[GEMM_TILE * GEMM_TILE];
    threadgroup float* a_tile = tiles;
    threadgroup float* b_tile = tiles + GEMM_TILE * GEMM_TILE_K;

    const uint row0 = tile.y * GEMM_TILE;
    const uint col0 = tile.x * GEMM_TILE;
    const ulong batch = tile.z;
    device const float* a_batch = a + (ulong(p.a_offset) + batch * p.a_batch_stride);
    device const float* b_batch = b + (ulong(p.b_offset) + batch * p.b_batch_stride);

    // SIMD group sg multiplies the tile's rows (sg / 2) * 16 and columns (sg % 2) * 16, 16x16:
    // acc[2 * r + q] is its 8x8 block at row half r, column half q.
    const uint sub_row = (sg / 2u) * 16u;
    const uint sub_col = (sg % 2u) * 16u;
    simdgroup_float8x8 acc[4];
    for (uint s = 0; s < 4; ++s) {
        acc[s] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < p.k; k0 += GEMM_TILE_K) {
        for (uint e = tid; e < GEMM_TILE * GEMM_TILE_K; e += GEMM_TILED_THREADS) {
            const uint i = row0 + e / GEMM_TILE_K;
            const uint kk = k0 + e % GEMM_TILE_K;
            a_tile[e] = (i < p.m && kk < p.k)
                ? a_batch[ulong(i) * p.a_row_stride + ulong(kk) * p.a_col_stride]
                : 0.0f;
        }
        for (uint e = tid; e < GEMM_TILE_K * GEMM_TILE; e += GEMM_TILED_THREADS) {
            const uint kk = k0 + e / GEMM_TILE;
            const uint j = col0 + e % GEMM_TILE;
            b_tile[e] = (kk < p.k && j < p.n)
                ? b_batch[ulong(kk) * p.b_row_stride + ulong(j) * p.b_col_stride]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < GEMM_TILE_K; s += 8) {
            simdgroup_float8x8 am[2];
            simdgroup_float8x8 bm[2];
            simdgroup_load(am[0], a_tile + (sub_row + 0) * GEMM_TILE_K + s, GEMM_TILE_K);
            simdgroup_load(am[1], a_tile + (sub_row + 8) * GEMM_TILE_K + s, GEMM_TILE_K);
            simdgroup_load(bm[0], b_tile + s * GEMM_TILE + sub_col + 0, GEMM_TILE);
            simdgroup_load(bm[1], b_tile + s * GEMM_TILE + sub_col + 8, GEMM_TILE);
            simdgroup_multiply_accumulate(acc[0], am[0], bm[0], acc[0]);
            simdgroup_multiply_accumulate(acc[1], am[0], bm[1], acc[1]);
            simdgroup_multiply_accumulate(acc[2], am[1], bm[0], acc[2]);
            simdgroup_multiply_accumulate(acc[3], am[1], bm[1], acc[3]);
        }
        // Every SIMD group has read this step's blocks before the next step (or the result)
        // overwrites them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc[0], tiles + (sub_row + 0) * GEMM_TILE + sub_col + 0, GEMM_TILE);
    simdgroup_store(acc[1], tiles + (sub_row + 0) * GEMM_TILE + sub_col + 8, GEMM_TILE);
    simdgroup_store(acc[2], tiles + (sub_row + 8) * GEMM_TILE + sub_col + 0, GEMM_TILE);
    simdgroup_store(acc[3], tiles + (sub_row + 8) * GEMM_TILE + sub_col + 8, GEMM_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* c_batch = c + (ulong(p.c_offset) + batch * p.c_batch_stride);
    for (uint e = tid; e < GEMM_TILE * GEMM_TILE; e += GEMM_TILED_THREADS) {
        const uint i = row0 + e / GEMM_TILE;
        const uint j = col0 + e % GEMM_TILE;
        if (i < p.m && j < p.n) {
            device float* out = c_batch + (ulong(i) * p.ldc + j);
            *out = gemm_result(p, tiles[e], out);
        }
    }
}

// --- The bias after a linear ---

// gemm.rs's GemmAddRows.
struct GemmAddRows {
    uint rows;
    uint cols;
};

// out[r][c] = out[r][c] + bias[c]: CUDA's makepad_cuda_add_rows_vec_f32 (diffusion_ops.cu), one
// fp32 add, so bit-identical. (x, y) = (column, row).
kernel void gemm_add_rows_f32(constant GemmAddRows& p [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              device const float* bias [[buffer(2)]],
                              uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= p.cols || pos.y >= p.rows) {
        return;
    }
    const ulong i = ulong(pos.y) * p.cols + pos.x;
    out[i] = out[i] + bias[pos.x];
}

// --- Attention's softmax ---

// gemm.rs's GemmSoftmaxRows. The bias, when has_bias is not 0, is laid out as the scores.
struct GemmSoftmaxRows {
    uint rows;
    uint row_stride;
    uint cols;
    uint has_bias;
};

// CUDA's makepad_cuda_diff_warp_reduce_sum and _max (diffusion_ops.cu): shuffle down 16, 8, 4,
// 2, 1 within a 32-wide SIMD group; lane 0 ends with the group's result. Never simd_sum, whose
// order differs (D8).
static float gemm_warp_sum(float value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        value += simd_shuffle_down(value, offset);
    }
    return value;
}

static float gemm_warp_max(float value) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const float other = simd_shuffle_down(value, offset);
        value = value > other ? value : other;
    }
    return value;
}

// CUDA's makepad_cuda_diff_block_reduce_sum and _max: each SIMD group's result into `shared`,
// then the first SIMD group reduces those. Thread 0 ends with the threadgroup's result.
static float gemm_block_sum(float value, threadgroup float* shared, uint tid, uint threads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;
    value = gemm_warp_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = tid < (threads + 31u) / 32u ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = gemm_warp_sum(value);
    }
    return value;
}

static float gemm_block_max(float value, threadgroup float* shared, uint tid, uint threads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;
    value = gemm_warp_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = tid < (threads + 31u) / 32u ? shared[lane] : -INFINITY;
    if (warp == 0) {
        value = gemm_warp_max(value);
    }
    return value;
}

// A row's logit: its score, plus its bias when there is one (CUDA's makepad_cuda_add_f32_precise
// pass before the softmax, fused: the same single fp32 add).
static inline float gemm_logit(constant GemmSoftmaxRows& p, device const float* row, device const float* bias,
                               uint row_index, uint idx) {
    const float score = row[idx];
    if (p.has_bias == 0) {
        return score;
    }
    return score + bias[ulong(row_index) * p.row_stride + idx];
}

// Each row's softmax over its first `cols` entries, in place: CUDA's
// makepad_cuda_softmax_rows_precise_f32 (diffusion_ops.cu), one threadgroup a row, threads
// striding the row; the max, exp(x - max) summed, then exp(x - max) / sum. A NaN never wins the
// max (`v > m ? v : m`, as CUDA's), and a row that is -inf everywhere comes out NaN, as CUDA's.
// precise::exp where CUDA has expf: the result is CUDA's to within exp's rounding, not its bits.
kernel void gemm_softmax_rows_f32(constant GemmSoftmaxRows& p [[buffer(0)]],
                                  device float* scores [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  uint row_index [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial_max[32];
    threadgroup float partial_sum[32];
    // The row's max, then its sum, as thread 0 found them.
    threadgroup float row_stats[2];
    if (row_index >= p.rows) {
        return;
    }
    device float* row = scores + ulong(row_index) * p.row_stride;

    float max_value = -INFINITY;
    for (uint idx = tid; idx < p.cols; idx += threads) {
        const float value = gemm_logit(p, row, bias, row_index, idx);
        max_value = value > max_value ? value : max_value;
    }
    max_value = gemm_block_max(max_value, partial_max, tid, threads);
    if (tid == 0) {
        row_stats[0] = max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = row_stats[0];

    float sum = 0.0f;
    for (uint idx = tid; idx < p.cols; idx += threads) {
        sum += precise::exp(gemm_logit(p, row, bias, row_index, idx) - row_max);
    }
    sum = gemm_block_sum(sum, partial_sum, tid, threads);
    if (tid == 0) {
        row_stats[1] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_sum = row_stats[1];

    for (uint idx = tid; idx < p.cols; idx += threads) {
        row[idx] = precise::exp(gemm_logit(p, row, bias, row_index, idx) - row_max) / row_sum;
    }
}
