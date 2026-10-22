// Shared by every Makepad kernel library (shaders/makepad/*.metal). The Metal stream compiles
// libraries from text at run time, so it splices this file in where a source has the line
// #include "makepad_common.h" (metal/src/stream/source.rs), framed by #line directives.
//
// Exactness (RationalVisions docs/cutie-metal.md, D8): every library compiles with fast math off.
// Whether a*b+c written in a kernel is still contracted into an fma, and which setting stops it
// (MSL 3.2's `#pragma METAL fp contract(off)`, else math mode Safe, else fast math off alone), is
// what step 7's probe finds (probe.metal and probe_contract.metal, run by tests/stream_probe.rs);
// the chosen setting goes here. Until then, a kernel that must not contract writes its products
// and sums so that contraction cannot change them, or calls fma() where nvcc contracts.
//
// Step 8 adds here: cutie_reduce4 in PyTorch's order (cuda/kernels/cutie.cu), the top-k radix key
// and max_nan.
#ifndef MAKEPAD_COMMON_H
#define MAKEPAD_COMMON_H

#include <metal_stdlib>
using namespace metal;

#endif
