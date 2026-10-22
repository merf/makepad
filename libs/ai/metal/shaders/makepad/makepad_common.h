// Shared by every Makepad kernel library (shaders/makepad/*.metal). The Metal stream compiles
// libraries from text at run time, so it splices this file in where a source has the line
// #include "makepad_common.h" (metal/src/stream/source.rs), framed by #line directives.
//
// Two machines filled this file in at the same time (RationalVisions docs/cutie-metal.md, section
// 11), each only between its own markers below: the Mac the contraction setting (step 7), the PC
// the helpers (step 8).
#ifndef MAKEPAD_COMMON_H
#define MAKEPAD_COMMON_H

#include <metal_stdlib>
using namespace metal;

// ==== Contraction (step 7, the Mac) ====
// Exactness (D8), from step 7's probe on an M5 (macOS 26.6.2, MSL 4.0; tests/stream_probe.rs):
// with fast math off alone, a*b+c written in a kernel is contracted into an fma (probe_mad: all
// 1048576 as fused), and math mode Safe changes nothing; MSL 3.2's `#pragma METAL fp
// contract(off)` stops it (probe_mad_contract_off: all as separate). So every library that
// includes this header is compiled with contraction off, and a kernel calls fma() where nvcc
// contracts (CUDA's own fmaf, or a product it fuses), which the pragma leaves alone. Also found:
// denormals are flushed to zero (products and sums), fmax(+0, -0) is -0 (the host's max gives +0),
// fmax with one NaN gives the other operand, and precise:: exp is within 1 ulp, log and tanh 2,
// sin and cos 3.
#pragma METAL fp contract(off)
// ==== End of contraction ====

// ==== Helpers (step 8, the PC) ====
// Empty: cutie_reduce4 (PyTorch's order, cuda/kernels/cutie.cu), the top-k radix key and max_nan
// are kept in each family's own namespace (cutie_elementwise.metal, cutie_softmax_norm.metal,
// cutie_topk_mask.metal) until one copy moves here.
// ==== End of helpers ====

#endif
