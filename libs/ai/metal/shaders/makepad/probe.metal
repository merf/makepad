// The Metal stream's bring-up probes (RationalVisions docs/cutie-metal.md, step 7), run by the
// Mac's tests/stream.rs, stream_fault.rs and stream_probe.rs: what this GPU and compiler do with
// fast math off (contraction, precise:: transcendentals, NaN in fmax, denormals), and kernels that
// exercise every kind of binding, function constants, threadgroup memory and a forced fault.
// Nothing outside the tests calls them.
#include "makepad_common.h"

struct ProbeCount {
    uint n;
};

// The MSL version this library compiled as (__METAL_VERSION__, e.g. 320 for 3.2). No params.
kernel void probe_version(device uint* out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i == 0) {
        out[0] = __METAL_VERSION__;
    }
}

// a*b+c as written: the host compares with the separately rounded result and with fma.
kernel void probe_mad(constant ProbeCount& p [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device const float* c [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// The fused reference: one rounding.
kernel void probe_fma(constant ProbeCount& p [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device const float* c [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = fma(a[i], b[i], c[i]);
    }
}

// One precise:: function, chosen by PROBE_FN: 0 exp, 1 log, 2 tanh, 3 sin, 4 cos.
constant uint PROBE_FN [[function_constant(0)]];

kernel void probe_unary(constant ProbeCount& p [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const float v = x[i];
    float r;
    switch (PROBE_FN) {
        case 0: r = precise::exp(v); break;
        case 1: r = precise::log(v); break;
        case 2: r = precise::tanh(v); break;
        case 3: r = precise::sin(v); break;
        default: r = precise::cos(v); break;
    }
    out[i] = r;
}

kernel void probe_fmax(constant ProbeCount& p [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = fmax(a[i], b[i]);
    }
}

// Each threadgroup sums its slice of x in order through threadgroup memory (one float a
// thread at [[threadgroup(0)]]); thread 0 writes the group's sum.
kernel void probe_threadgroup_sum(constant ProbeCount& p [[buffer(0)]],
                                  device const float* x [[buffer(1)]],
                                  device float* out [[buffer(2)]],
                                  threadgroup float* tile [[threadgroup(0)]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint width [[threads_per_threadgroup]]) {
    const uint i = group * width + tid;
    tile[tid] = i < p.n ? x[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < width; ++k) {
            sum += tile[k];
        }
        out[group] = sum;
    }
}

struct ProbeBinds {
    uint n;
    uint table_len;
};

// Every kind of binding at once: a tensor, a setBytes table, an absent slot, weights at a float
// offset (the binding's), and the output. out[i] = (x[i] + table[i % table_len]) + w[i].
kernel void probe_binds(constant ProbeBinds& p [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        constant float* table [[buffer(2)]],
                        device const float* absent [[buffer(3)]],
                        device const float* w [[buffer(4)]],
                        device float* out [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = (x[i] + table[i % p.table_len]) + w[i];
    }
}

// Read and written: io[i] = io[i] * 2 + x[i]. Doubling is exact, so fused or not, one rounding.
kernel void probe_inout(constant ProbeCount& p [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* io [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        io[i] = io[i] * 2.0f + x[i];
    }
}

struct ProbeFault {
    ulong far_words;
};

// Writes far past its buffer, to an address nothing maps: the command buffer fails with a GPU
// address fault (status 5), which the stream must report at its next wait with the op's label.
kernel void probe_fault(constant ProbeFault& p [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[p.far_words + i] = 1.0f;
}

struct ProbeSpin {
    uint n;
    uint iterations;
};

// Slow on purpose, for the residency tests (tests/residency.rs): `iterations` dependent steps of
// v = v * 0.5 + 1, which reach 2 exactly and stay there, then out = x + (v - 2), which is x (for x
// not -0). So the result is the input's bits, and a command buffer holding it stays pending for as
// long as the iterations take.
kernel void probe_spin(constant ProbeSpin& p [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    float v = float(i & 1023u);
    for (uint k = 0; k < p.iterations; ++k) {
        v = v * 0.5f + 1.0f;
    }
    out[i] = x[i] + (v - 2.0f);
}
