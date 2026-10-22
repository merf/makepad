//! The stream's own kernel libraries (`shaders/makepad/`), their kernels and their params structs,
//! each field for field its MSL twin. Cutie's libraries join them in step 8.

use super::kernel::{Kernel, KernelLib, KernelParams};

/// Fills, copies and the elementwise ops (`tensor.metal`).
pub static TENSOR: KernelLib = KernelLib {
    name: "tensor",
    source: include_str!("../../shaders/makepad/tensor.metal"),
};

/// `dst[i] = bits` for `i < n`, as 32-bit words ([`TensorFill`]).
pub static TENSOR_FILL: Kernel = Kernel { lib: &TENSOR, function: "tensor_fill_u32" };
/// A `rows x cols` block of words between two row strides ([`TensorCopy2d`]).
pub static TENSOR_COPY2D: Kernel = Kernel { lib: &TENSOR, function: "tensor_copy2d_u32" };
/// `out = a + b` elementwise ([`TensorCount`]).
pub static TENSOR_ADD: Kernel = Kernel { lib: &TENSOR, function: "tensor_add_f32" };
/// `out = a * b` elementwise ([`TensorCount`]).
pub static TENSOR_MUL: Kernel = Kernel { lib: &TENSOR, function: "tensor_mul_f32" };
/// `out = fmax(x, 0)` ([`TensorCount`]).
pub static TENSOR_RELU: Kernel = Kernel { lib: &TENSOR, function: "tensor_relu_f32" };

/// The bring-up probes (`probe.metal`); only the tests launch them.
pub static PROBE: KernelLib = KernelLib {
    name: "probe",
    source: include_str!("../../shaders/makepad/probe.metal"),
};

/// The contraction probe under MSL 3.2's pragma (`probe_contract.metal`), apart so a compiler that
/// rejects the pragma breaks only it.
pub static PROBE_CONTRACT: KernelLib = KernelLib {
    name: "probe_contract",
    source: include_str!("../../shaders/makepad/probe_contract.metal"),
};

pub static PROBE_VERSION: Kernel = Kernel { lib: &PROBE, function: "probe_version" };
pub static PROBE_MAD: Kernel = Kernel { lib: &PROBE, function: "probe_mad" };
pub static PROBE_FMA: Kernel = Kernel { lib: &PROBE, function: "probe_fma" };
/// Takes `FnConst::U32(0, f)`: 0 exp, 1 log, 2 tanh, 3 sin, 4 cos (all `precise::`).
pub static PROBE_UNARY: Kernel = Kernel { lib: &PROBE, function: "probe_unary" };
pub static PROBE_FMAX: Kernel = Kernel { lib: &PROBE, function: "probe_fmax" };
/// One float of threadgroup memory a thread at `[[threadgroup(0)]]`.
pub static PROBE_THREADGROUP_SUM: Kernel = Kernel { lib: &PROBE, function: "probe_threadgroup_sum" };
pub static PROBE_BINDS: Kernel = Kernel { lib: &PROBE, function: "probe_binds" };
pub static PROBE_INOUT: Kernel = Kernel { lib: &PROBE, function: "probe_inout" };
pub static PROBE_FAULT: Kernel = Kernel { lib: &PROBE, function: "probe_fault" };
pub static PROBE_MAD_CONTRACT_OFF: Kernel = Kernel { lib: &PROBE_CONTRACT, function: "probe_mad_contract_off" };

/// Every kernel above, for the bring-up test that compiles them all.
pub static ALL_KERNELS: &[&Kernel] = &[
    &TENSOR_FILL,
    &TENSOR_COPY2D,
    &TENSOR_ADD,
    &TENSOR_MUL,
    &TENSOR_RELU,
    &PROBE_VERSION,
    &PROBE_MAD,
    &PROBE_FMA,
    &PROBE_UNARY,
    &PROBE_FMAX,
    &PROBE_THREADGROUP_SUM,
    &PROBE_BINDS,
    &PROBE_INOUT,
    &PROBE_FAULT,
    &PROBE_MAD_CONTRACT_OFF,
];

/// `TensorCount` / `ProbeCount`: the element count.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorCount {
    pub n: u32,
}
// SAFETY: #[repr(C)], 4 bytes, no pointers; `TensorCount` and `ProbeCount` in MSL.
unsafe impl KernelParams for TensorCount {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorFill {
    pub n: u32,
    pub bits: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `TensorFill` in MSL.
unsafe impl KernelParams for TensorFill {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorCopy2d {
    pub rows: u32,
    pub cols: u32,
    pub src_ld: u32,
    pub dst_ld: u32,
    pub src_offset: u32,
    pub dst_offset: u32,
}
// SAFETY: #[repr(C)], 24 bytes, no pointers; `TensorCopy2d` in MSL.
unsafe impl KernelParams for TensorCopy2d {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProbeBinds {
    pub n: u32,
    pub table_len: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `ProbeBinds` in MSL.
unsafe impl KernelParams for ProbeBinds {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProbeFault {
    pub far_words: u64,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `ProbeFault` in MSL (ulong).
unsafe impl KernelParams for ProbeFault {}
