// probe_mad under MSL 3.2's `#pragma METAL fp contract(off)`, in a library of its own so that a
// compiler that rejects the pragma breaks only this probe. Below MSL 3.2 the pragma is left out
// and the kernel is probe_mad again; probe_version says which.
#include "makepad_common.h"

struct ProbeCount {
    uint n;
};

kernel void probe_mad_contract_off(constant ProbeCount& p [[buffer(0)]],
                                   device const float* a [[buffer(1)]],
                                   device const float* b [[buffer(2)]],
                                   device const float* c [[buffer(3)]],
                                   device float* out [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
#if __METAL_VERSION__ >= 320
#pragma METAL fp contract(off)
#endif
    if (i < p.n) {
        out[i] = a[i] * b[i] + c[i];
    }
}
