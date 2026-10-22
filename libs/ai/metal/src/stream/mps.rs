//! The GEMM on Metal Performance Shaders (`MPSMatrixMultiplication`), the default on macOS
//! (MAKEPAD_METAL_GEMM unset, auto or mps; `kernels` for `gemm.metal`'s): Apple's fp32 matrix
//! multiply, about twice the tiled kernel's speed on an M5 at Cutie's shapes, 15 to 26% off a
//! Cutie frame (RationalVisions docs/cutie-metal.md §11). It is encoded on
//! the stream's open command buffer between its compute encoders (`StreamCtx::begin_buffer_work`),
//! so it is ordered with the kernels around it as any dispatch is. Its sums are MPS's own, not
//! the tiled kernel's, so results move by rounding: judged by `cutie-validate all` within
//! `Tol::STAGE`, not against the golden bit for bit.

use super::MetalResult;
use crate::objc_util::StrongId;
use makepad_objc_sys::runtime::{ObjcId, BOOL, NO, YES};
use makepad_objc_sys::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

/// `MPSDataTypeFloat32` (`MPSDataTypeFloatBit | 32`).
const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0000 | 32;

/// One matrix as MPS reads or writes it: a buffer, the byte offset of its first element, its
/// stored rows and columns, and the bytes from one row to the next.
pub(crate) struct MpsMatrix {
    pub buffer: ObjcId,
    pub offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
}

/// A product's shape and scalars: what an `MPSMatrixMultiplication` is made for.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct KernelKey {
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: u32,
    beta: u32,
}

/// The calling thread's `MPSMatrixMultiplication` kernels, one per shape and scalars, made at
/// first use.
#[derive(Default)]
pub(crate) struct MpsCache {
    kernels: HashMap<KernelKey, StrongId>,
}

impl MpsCache {
    /// C = alpha * op(A) @ op(B) + beta * C for one batch, encoded on `cb` (no compute encoder
    /// open on it). The caller has checked the shapes and bindings (`gemm::plan`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        device: ObjcId,
        cb: ObjcId,
        (trans_a, trans_b): (bool, bool),
        (m, n, k): (usize, usize, usize),
        (alpha, beta): (f32, f32),
        a: &MpsMatrix,
        b: &MpsMatrix,
        c: &MpsMatrix,
    ) -> MetalResult<()> {
        let key = KernelKey { trans_a, trans_b, m, n, k, alpha: alpha.to_bits(), beta: beta.to_bits() };
        if !self.kernels.contains_key(&key) {
            let bool_of = |v: bool| -> BOOL { if v { YES } else { NO } };
            let kernel = unsafe {
                let alloc: ObjcId = msg_send![class!(MPSMatrixMultiplication), alloc];
                let kernel: ObjcId = msg_send![
                    alloc,
                    initWithDevice: device
                    transposeLeft: bool_of(trans_a)
                    transposeRight: bool_of(trans_b)
                    resultRows: m as u64
                    resultColumns: n as u64
                    interiorColumns: k as u64
                    alpha: alpha as f64
                    beta: beta as f64
                ];
                StrongId::from_owned(kernel)
            }
            .ok_or("MPSMatrixMultiplication init returned nil")?;
            self.kernels.insert(key, kernel);
        }
        let kernel = self.kernels[&key].as_id();
        let (a, b, c) = (matrix(a)?, matrix(b)?, matrix(c)?);
        unsafe {
            let _: () = msg_send![
                kernel,
                encodeToCommandBuffer: cb
                leftMatrix: a.as_id()
                rightMatrix: b.as_id()
                resultMatrix: c.as_id()
            ];
        }
        Ok(())
    }
}

/// An `MPSMatrix` over `m`'s buffer at its offset (the command buffer retains what it encodes).
fn matrix(m: &MpsMatrix) -> MetalResult<StrongId> {
    unsafe {
        let descriptor: ObjcId = msg_send![
            class!(MPSMatrixDescriptor),
            matrixDescriptorWithRows: m.rows as u64
            columns: m.cols as u64
            rowBytes: m.row_bytes as u64
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        if descriptor.is_null() {
            return Err("MPSMatrixDescriptor returned nil".into());
        }
        let alloc: ObjcId = msg_send![class!(MPSMatrix), alloc];
        let matrix: ObjcId = msg_send![alloc, initWithBuffer: m.buffer offset: m.offset descriptor: descriptor];
        StrongId::from_owned(matrix).ok_or_else(|| "MPSMatrix init returned nil".into())
    }
}
