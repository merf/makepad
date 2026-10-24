// Cutie's top-k softmax and attention mask (RationalVisions docs/cutie-metal.md, family (e) of
// step 10): ports of cuda/kernels/cutie.cu's cutie_topk_partial_kernel, cutie_topk_merge_kernel,
// cutie_row_sum_kernel, cutie_attn_foreground_kernel and cutie_attn_mask_kernel, in the same
// order of evaluation, launched by metal/src/cutie_ops/topk_mask.rs. Each op's host reference is
// in cuda/src/cutie_reference.rs.
//
// Exactness (D8): fast math is off and nothing here adds a product, so there is nothing to
// contract; exp and log are precise::, a division is IEEE's, and the sums PyTorch orders (the
// top-k denominator, the background product) keep its four accumulators. The merge's pick and
// the row sums go round a SIMD-group with simd_shuffle_xor in CUDA's butterfly order, so they
// take CUDA's 32 lanes: the host refuses a GPU whose SIMD-groups are another width. Indices that
// multiply two sizes are ulong, as CUDA's size_t. No half anywhere.
#include "makepad_common.h"

// This library's helpers, in a namespace of their own so they never meet makepad_common.h's
// (which may take over cutie_reduce4, the radix key and max_nan for every family).
namespace topk_mask {

// An empty place in a top-k list: key 0 is below every float's key (-inf's is 0x007fffff), so
// every row beats it and it is never chosen while k <= m.
constant uint EMPTY_KEY = 0u;
constant uint EMPTY_ROW = 0xffffffffu;

// PyTorch's radix key of a float (TopKTypeConfig<float>::convert): the keys' unsigned order is
// torch.topk's order of the values, so -0 ranks below +0 and every NaN (either sign) above every
// number.
static inline uint radix_key(float v) {
    const uint x = as_type<uint>(v);
    if (isnan(v)) {
        return 0xffffffffu;
    }
    return x ^ ((x & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

// The value back from its key (every NaN comes back as the one NaN 0x7fffffff).
static inline float radix_value(uint key) {
    return as_type<float>((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

// torch.topk's order: the larger key first, equal keys by the lower row.
static inline bool ranks_before(uint ka, uint ia, uint kb, uint ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// PyTorch's sigmoid for f32 (precise exp, IEEE division).
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + precise::exp(-x));
}

// Cutie's aggregate logit: clamp to 1e-7..1-1e-7 (NaN passes through, as torch.clamp), then
// log(q / (1 - q)).
static inline float logit(float q) {
    const float lo = 1e-7f;
    const float hi = 1.0f - 1e-7f;
    q = q < lo ? lo : (q > hi ? hi : q);
    return precise::log(q / (1.0f - q));
}

// The larger of two values, a NaN in either winning (PyTorch's max_propagate_nan).
static inline float max_nan(float a, float b) {
    return isnan(b) ? b : (isnan(a) ? a : fmax(a, b));
}

// The background of pixel i: the product over the n objects of 1 - sigmoid(aux), in PyTorch's
// order for a reduction that is not the innermost (cutie.cu's cutie_reduce4<true>): four
// accumulators, term o into o % 4, then ((a0 * a1) * a2) * a3.
static inline float background(device const float* aux, uint n, ulong p, ulong i) {
    float acc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    for (uint o = 0; o < n; o += 4) {
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            if (o + j < n) {
                acc[j] = acc[j] * (1.0f - sigmoid(aux[ulong(o + j) * p + i]));
            }
        }
    }
    return ((acc[0] * acc[1]) * acc[2]) * acc[3];
}

}  // namespace topk_mask

// --- Top-k softmax ---

struct CutieTopkPartial {
    uint m;
    uint p;
    uint k;
    uint chunk;
    uint slices;
};

// Each thread takes one column's rows [slice*chunk, (slice+1)*chunk) of sim [m, p] and keeps the
// best K of them in registers as keys, sorted (a new entry is carried down the list, swapping
// past every entry it beats); reads are coalesced across the columns of a threadgroup row. The
// first k of each list go to keys/idx [slices][k][p]. K (32 or 64, the instances below) is at
// least k.
template <int K>
kernel void cutie_topk_partial(constant CutieTopkPartial& prm [[buffer(0)]],
                               device const float* sim [[buffer(1)]],
                               device uint* keys [[buffer(2)]],
                               device uint* idx [[buffer(3)]],
                               uint2 tid [[thread_position_in_threadgroup]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint2 width [[threads_per_threadgroup]]) {
    const uint col = group.x * width.x + tid.x;
    const uint slice = group.y * width.y + tid.y;
    if (col >= prm.p || slice >= prm.slices) {
        return;
    }
    uint tk[K];
    uint ti[K];
#pragma unroll
    for (int t = 0; t < K; ++t) {
        tk[t] = topk_mask::EMPTY_KEY;
        ti[t] = topk_mask::EMPTY_ROW;
    }
    const uint start = slice * prm.chunk;
    const uint end = min(prm.m, start + prm.chunk);
    device const float* src = sim + col;
    // Rows are read four at a time so the loads are in flight together.
    for (uint j = start; j < end; j += 4) {
        uint batch[4];
#pragma unroll
        for (uint u = 0; u < 4; ++u) {
            batch[u] = j + u < end ? topk_mask::radix_key(src[ulong(j + u) * prm.p]) : topk_mask::EMPTY_KEY;
        }
#pragma unroll
        for (uint u = 0; u < 4; ++u) {
            uint row = j + u;
            uint key = batch[u];
            if (row >= end || !topk_mask::ranks_before(key, row, tk[K - 1], ti[K - 1])) {
                continue;
            }
#pragma unroll
            for (int t = 0; t < K; ++t) {
                const bool before = topk_mask::ranks_before(key, row, tk[t], ti[t]);
                const uint sk = tk[t];
                const uint si = ti[t];
                tk[t] = before ? key : sk;
                ti[t] = before ? row : si;
                key = before ? sk : key;
                row = before ? si : row;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < K; ++t) {
        if (uint(t) < prm.k) {
            const ulong at = (ulong(slice) * prm.k + uint(t)) * prm.p + col;
            keys[at] = tk[t];
            idx[at] = ti[t];
        }
    }
}

typedef decltype(cutie_topk_partial<32>) cutie_topk_partial_t;

template [[host_name("cutie_topk_partial_32")]] kernel cutie_topk_partial_t cutie_topk_partial<32>;
template [[host_name("cutie_topk_partial_64")]] kernel cutie_topk_partial_t cutie_topk_partial<64>;

struct CutieTopkMerge {
    uint m;
    uint p;
    uint k;
    uint slices;
};

// One SIMD-group a column: lane l holds the heads of slices l and l+32. Each of the k picks is
// the SIMD-group's best head (a butterfly in the key order; rows are unique, so one lane owns it
// and advances). Then the softmax as Cutie's do_softmax: exp of the picked values (no max
// subtracted), summed in PyTorch's order over the picks (best first, four accumulators, pick t
// into t % 4), normalised, scattered into aff [m, p] (zeroed before). Every lane runs every
// shuffle: the loop over the picks is the same for the whole SIMD-group, and so is the column.
//
// PICKS (an fp16 step's sparse readout): instead of scattering into aff, pick t's row and weight
// go to rows[t * p + col] and aff[t * p + col] ([k, p]), every pick written.
template <bool PICKS>
kernel void cutie_topk_merge_t(constant CutieTopkMerge& prm [[buffer(0)]],
                               device const uint* keys [[buffer(1)]],
                               device const uint* idx [[buffer(2)]],
                               device float* aff [[buffer(3)]],
                               device uint* rows [[buffer(4)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint simdgroup [[simdgroup_index_in_threadgroup]],
                               uint simdgroups [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint col = group * simdgroups + simdgroup;
    if (col >= prm.p) {
        return;
    }
    const uint k = prm.k;
    const ulong p = prm.p;
    const uint s0 = lane;
    const uint s1 = lane + 32;
    uint h0 = 0;
    uint h1 = 0;
    uint key0 = topk_mask::EMPTY_KEY;
    uint key1 = topk_mask::EMPTY_KEY;
    uint i0 = topk_mask::EMPTY_ROW;
    uint i1 = topk_mask::EMPTY_ROW;
    if (s0 < prm.slices) {
        key0 = keys[ulong(s0) * k * p + col];
        i0 = idx[ulong(s0) * k * p + col];
    }
    if (s1 < prm.slices) {
        key1 = keys[ulong(s1) * k * p + col];
        i1 = idx[ulong(s1) * k * p + col];
    }
    // Lane l keeps picks l and l+32: their exp and row.
    float e0 = 0.0f;
    float e1 = 0.0f;
    uint r0 = topk_mask::EMPTY_ROW;
    uint r1 = topk_mask::EMPTY_ROW;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    for (uint t = 0; t < k; ++t) {
        const bool second = topk_mask::ranks_before(key1, i1, key0, i0);
        uint bk = second ? key1 : key0;
        uint bi = second ? i1 : i0;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            const uint ok = simd_shuffle_xor(bk, offset);
            const uint oi = simd_shuffle_xor(bi, offset);
            if (topk_mask::ranks_before(ok, oi, bk, bi)) {
                bk = ok;
                bi = oi;
            }
        }
        if (bi == i0 && s0 < prm.slices) {
            ++h0;
            key0 = h0 < k ? keys[(ulong(s0) * k + h0) * p + col] : topk_mask::EMPTY_KEY;
            i0 = h0 < k ? idx[(ulong(s0) * k + h0) * p + col] : topk_mask::EMPTY_ROW;
        } else if (bi == i1 && s1 < prm.slices) {
            ++h1;
            key1 = h1 < k ? keys[(ulong(s1) * k + h1) * p + col] : topk_mask::EMPTY_KEY;
            i1 = h1 < k ? idx[(ulong(s1) * k + h1) * p + col] : topk_mask::EMPTY_ROW;
        }
        const float e = precise::exp(topk_mask::radix_value(bk));
        const uint slot = t & 3u;
        a0 = slot == 0 ? a0 + e : a0;
        a1 = slot == 1 ? a1 + e : a1;
        a2 = slot == 2 ? a2 + e : a2;
        a3 = slot == 3 ? a3 + e : a3;
        if (t % 32 == lane) {
            if (t < 32) {
                e0 = e;
                r0 = bi;
            } else {
                e1 = e;
                r1 = bi;
            }
        }
    }
    const float sum = ((a0 + a1) + a2) + a3;
    if (PICKS) {
        if (lane < k) {
            rows[ulong(lane) * p + col] = r0;
            aff[ulong(lane) * p + col] = e0 / sum;
        }
        if (lane + 32 < k) {
            rows[ulong(lane + 32) * p + col] = r1;
            aff[ulong(lane + 32) * p + col] = e1 / sum;
        }
        return;
    }
    // A pick is always a real row while k <= m; the bound keeps a stray one off memory.
    if (lane < k && r0 < prm.m) {
        aff[ulong(r0) * p + col] = e0 / sum;
    }
    if (lane + 32 < k && r1 < prm.m) {
        aff[ulong(r1) * p + col] = e1 / sum;
    }
}

typedef decltype(cutie_topk_merge_t<false>) cutie_topk_merge_type;

template [[host_name("cutie_topk_merge")]] kernel cutie_topk_merge_type cutie_topk_merge_t<false>;
template [[host_name("cutie_topk_merge_picks")]] kernel cutie_topk_merge_type cutie_topk_merge_t<true>;

// --- The sparse readout (an fp16 step's) ---

struct CutieSparseReadout {
    uint v;       // value rows (objects * 256)
    uint p;       // pixels
    uint k;       // picks a pixel
    uint n_long;  // long-term memory's tokens (its values first), then working memory's
    uint n_work;
};

// readout[r, col] = sum over the picks t (best first, one accumulator) of weight[t, col] times
// the picked token's value in row r: long-term memory's values [v, n_long] for a token below
// n_long, else working memory's [v, n_work]. A thread a pixel and value row, pixels across a
// threadgroup's rows so a row's reads stay in one value row.
kernel void cutie_sparse_readout(constant CutieSparseReadout& prm [[buffer(0)]],
                                 device const uint* rows [[buffer(1)]],
                                 device const float* weights [[buffer(2)]],
                                 device const float* long_values [[buffer(3)]],
                                 device const float* work_values [[buffer(4)]],
                                 device float* out [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= prm.p || r >= prm.v) {
        return;
    }
    const ulong p = prm.p;
    float acc = 0.0f;
    for (uint t = 0; t < prm.k; ++t) {
        const uint j = rows[ulong(t) * p + col];
        const float w = weights[ulong(t) * p + col];
        const float x = j < prm.n_long ? long_values[ulong(r) * prm.n_long + j] : work_values[ulong(r) * prm.n_work + (j - prm.n_long)];
        acc = acc + w * x;
    }
    out[ulong(r) * p + col] = acc;
}

struct CutiePickUsage {
    uint p;
    uint k;
    uint m;
    float scale;  // fixed-point units a unit of weight
};

// Each token's usage, the picks' weights summed over the pixels, deterministically: each weight
// rounded to fixed point (`scale` units, chosen so p of them fit a uint) and added with an
// integer atomic, so the order does not change the sum. counts [m] is zeroed before.
kernel void cutie_pick_usage(constant CutiePickUsage& prm [[buffer(0)]],
                             device const uint* rows [[buffer(1)]],
                             device const float* weights [[buffer(2)]],
                             device atomic_uint* counts [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= prm.p * prm.k) {
        return;
    }
    const uint j = rows[i];
    if (j < prm.m) {
        atomic_fetch_add_explicit(&counts[j], uint(rint(weights[i] * prm.scale)), memory_order_relaxed);
    }
}

// The counts back to fp32 usage [m].
kernel void cutie_pick_usage_out(constant CutiePickUsage& prm [[buffer(0)]],
                                 device const uint* counts [[buffer(1)]],
                                 device float* usage [[buffer(2)]],
                                 uint j [[thread_position_in_grid]]) {
    if (j < prm.m) {
        usage[j] = float(counts[j]) / prm.scale;
    }
}

struct CutieRowSum {
    uint rows;
    uint cols;
};

// The sum of each row of x [rows, cols] into out [rows]: one SIMD-group a row, each lane summing
// every 32nd column from its own, then a butterfly. Not PyTorch's order for affinity.sum(dim=2)
// (a vectorised innermost reduce), so the usage is fp32 rounding apart from PyTorch's, as on CUDA.
kernel void cutie_row_sum(constant CutieRowSum& prm [[buffer(0)]],
                          device const float* x [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                          uint simdgroups [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * simdgroups + simdgroup;
    if (row >= prm.rows) {
        return;
    }
    device const float* src = x + ulong(row) * prm.cols;
    float sum = 0.0f;
    for (uint i = lane; i < prm.cols; i += 32) {
        sum += src[i];
    }
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        sum += simd_shuffle_xor(sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

// --- Attention mask ---

struct CutieAttnForeground {
    ulong p;
    uint n;
    uint pad;
};

// The attention mask's foreground: per pixel, each object's aggregated logit (of the sigmoid of
// its aux logit) against the largest of all rows, the background's included. fg [n, p] (0/1);
// seen[2o] set when object o has any foreground, seen[2o+1] when it has any background (seen
// zeroed before). A grid-stride loop, as CUDA's.
kernel void cutie_attn_foreground(constant CutieAttnForeground& prm [[buffer(0)]],
                                  device const float* aux [[buffer(1)]],
                                  device uint* fg [[buffer(2)]],
                                  device atomic_uint* seen [[buffer(3)]],
                                  uint index [[thread_position_in_grid]],
                                  uint count [[threads_per_grid]]) {
    const ulong p = prm.p;
    const uint n = prm.n;
    for (ulong i = index; i < p; i += count) {
        const float bg = topk_mask::background(aux, n, p, i);
        // A NaN anywhere makes the max NaN (logits.max propagates it), and then no row is >= it.
        float largest = topk_mask::logit(bg);
        for (uint o = 0; o < n; ++o) {
            largest = topk_mask::max_nan(largest, topk_mask::logit(topk_mask::sigmoid(aux[ulong(o) * p + i])));
        }
        for (uint o = 0; o < n; ++o) {
            const uint is_fg = topk_mask::logit(topk_mask::sigmoid(aux[ulong(o) * p + i])) >= largest ? 1u : 0u;
            fg[ulong(o) * p + i] = is_fg;
            atomic_fetch_or_explicit(&seen[2 * ulong(o) + (is_fg ? 0 : 1)], 1u, memory_order_relaxed);
        }
    }
}

struct CutieAttnMask {
    ulong p;
    ulong rows;
    uint heads;
    uint queries;
};

// The additive mask [n*heads*queries, p]: the first half of each object's queries see only its
// foreground, the rest only the rest; a row that would see nothing sees everything. A threadgroup
// row of pixels, stepping down the rows by the grid's height (CUDA's gridDim.y loop).
kernel void cutie_attn_mask(constant CutieAttnMask& prm [[buffer(0)]],
                            device const uint* fg [[buffer(1)]],
                            device const uint* seen [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 width [[threads_per_threadgroup]],
                            uint2 groups [[threadgroups_per_grid]]) {
    const ulong i = ulong(group.x) * width.x + tid.x;
    if (i >= prm.p) {
        return;
    }
    const ulong per_object = ulong(prm.heads) * prm.queries;
    for (ulong row = group.y; row < prm.rows; row += groups.y) {
        const ulong o = row / per_object;
        const uint q = uint(row % prm.queries);
        const bool foreground_query = q < prm.queries / 2;
        const bool open_all = foreground_query ? seen[2 * o] == 0 : seen[2 * o + 1] == 0;
        const bool blocked = !open_all && ((fg[o * prm.p + i] != 0) != foreground_query);
        out[row * prm.p + i] = blocked ? -INFINITY : 0.0f;
    }
}
