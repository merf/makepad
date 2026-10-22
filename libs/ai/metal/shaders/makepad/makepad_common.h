// Shared by every Makepad kernel library (shaders/makepad/*.metal). The Metal stream compiles
// libraries from text at run time, so it splices this file in where a source has the line
// #include "makepad_common.h" (metal/src/stream/source.rs), framed by #line directives.
//
// Two machines fill this file in at the same time (RationalVisions docs/cutie-metal.md, section
// 11), each only between its own markers below, so their commits never touch the same lines: the
// Mac writes the contraction setting (step 7), the PC the helpers (step 8).
#ifndef MAKEPAD_COMMON_H
#define MAKEPAD_COMMON_H

#include <metal_stdlib>
using namespace metal;

// ==== Contraction (step 7, the Mac) ====
// Exactness (D8): every library compiles with fast math off. Whether a*b+c written in a kernel is
// still contracted into an fma, and which setting stops it (MSL 3.2's `#pragma METAL fp
// contract(off)`, else math mode Safe, else fast math off alone), is what step 7's probe finds
// (probe.metal and probe_contract.metal, run by tests/stream_probe.rs); the chosen setting goes
// here. Until then, a kernel that must not contract writes its products and sums so that
// contraction cannot change them, or calls fma() where nvcc contracts.
// ==== End of contraction ====

// ==== Helpers (step 8, the PC) ====
// cutie_reduce4 in PyTorch's order (cuda/kernels/cutie.cu), the top-k radix key and max_nan.
// ==== End of helpers ====

#endif
