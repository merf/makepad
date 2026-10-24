//! The MPSGraph ops the Cutie graphs (`transformer`, `memory_read`) are built from: thin
//! wrappers over MPSGraph's selectors, each tensor autoreleased (the graph keeps it).

use super::graph_shape;
use makepad_objc_sys::runtime::ObjcId;
use makepad_objc_sys::{class, msg_send, sel, sel_impl};

pub(super) const F16: u32 = super::MPS_DATA_TYPE_FLOAT16;
pub(super) const F32: u32 = super::MPS_DATA_TYPE_FLOAT32;

/// The graph being built, its tensors autoreleased (the graph keeps them).
pub(super) struct G(pub(super) ObjcId);

pub(super) fn nil() -> ObjcId {
    std::ptr::null_mut()
}

impl G {
    pub(super) unsafe fn constant(&self, data: &[f32], dims: &[usize]) -> ObjcId {
        let bytes: ObjcId = msg_send![class!(NSData), dataWithBytes: data.as_ptr() as *const std::ffi::c_void length: (data.len() * 4) as u64];
        msg_send![self.0, constantWithData: bytes shape: graph_shape(dims) dataType: F32]
    }
    pub(super) unsafe fn scalar(&self, v: f64, ty: u32) -> ObjcId {
        msg_send![self.0, constantWithScalar: v dataType: ty]
    }
    pub(super) unsafe fn cast(&self, x: ObjcId, ty: u32) -> ObjcId {
        msg_send![self.0, castTensor: x toType: ty name: nil()]
    }
    pub(super) unsafe fn half(&self, x: ObjcId) -> ObjcId {
        self.cast(x, F16)
    }
    pub(super) unsafe fn float(&self, x: ObjcId) -> ObjcId {
        self.cast(x, F32)
    }
    pub(super) unsafe fn add(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, additionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn sub(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, subtractionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn mul(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, multiplicationWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn div(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, divisionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn maximum(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, maximumWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn relu(&self, x: ObjcId) -> ObjcId {
        msg_send![self.0, reLUWithTensor: x name: nil()]
    }
    pub(super) unsafe fn sigmoid(&self, x: ObjcId) -> ObjcId {
        msg_send![self.0, sigmoidWithTensor: x name: nil()]
    }
    pub(super) unsafe fn reshape(&self, x: ObjcId, dims: &[usize]) -> ObjcId {
        msg_send![self.0, reshapeTensor: x withShape: graph_shape(dims) name: nil()]
    }
    pub(super) unsafe fn transpose(&self, x: ObjcId, a: usize, b: usize) -> ObjcId {
        msg_send![self.0, transposeTensor: x dimension: a as u64 withDimension: b as u64 name: nil()]
    }
    pub(super) unsafe fn matmul(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, matrixMultiplicationWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn mean(&self, x: ObjcId, axes: &[usize]) -> ObjcId {
        msg_send![self.0, meanOfTensor: x axes: graph_shape(axes) name: nil()]
    }
    pub(super) unsafe fn softmax(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, softMaxWithTensor: x axis: axis as i64 name: nil()]
    }
    pub(super) unsafe fn max_along(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, reductionMaximumWithTensor: x axis: axis as i64 name: nil()]
    }
    pub(super) unsafe fn prod_along(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, reductionProductWithTensor: x axis: axis as i64 name: nil()]
    }
    pub(super) unsafe fn clamp(&self, x: ObjcId, lo: f64, hi: f64) -> ObjcId {
        let (lo, hi) = (self.scalar(lo, F32), self.scalar(hi, F32));
        msg_send![self.0, clampWithTensor: x minValueTensor: lo maxValueTensor: hi name: nil()]
    }
    pub(super) unsafe fn ge(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, greaterThanOrEqualToWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    pub(super) unsafe fn broadcast(&self, x: ObjcId, dims: &[usize]) -> ObjcId {
        msg_send![self.0, broadcastTensor: x toShape: graph_shape(dims) name: nil()]
    }
    pub(super) unsafe fn concat(&self, a: ObjcId, b: ObjcId, dim: usize) -> ObjcId {
        msg_send![self.0, concatTensor: a withTensor: b dimension: dim as i64 name: nil()]
    }
    pub(super) unsafe fn slice(&self, x: ObjcId, dim: usize, start: usize, len: usize) -> ObjcId {
        msg_send![self.0, sliceTensor: x dimension: dim as u64 start: start as i64 length: len as i64 name: nil()]
    }
    pub(super) unsafe fn neg(&self, x: ObjcId) -> ObjcId {
        msg_send![self.0, negativeWithTensor: x name: nil()]
    }
    pub(super) unsafe fn sum_along(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, reductionSumWithTensor: x axis: axis as i64 name: nil()]
    }
    /// The `k` largest along the last axis: (values, int32 indices).
    pub(super) unsafe fn top_k(&self, x: ObjcId, k: usize) -> (ObjcId, ObjcId) {
        let pair: ObjcId = msg_send![self.0, topKWithSourceTensor: x k: k as u64 name: nil()];
        let values: ObjcId = msg_send![pair, objectAtIndex: 0u64];
        let indices: ObjcId = msg_send![pair, objectAtIndex: 1u64];
        (values, indices)
    }
    /// `x` gathered along `axis` at `indices` (their shape in place of that axis).
    pub(super) unsafe fn gather(&self, x: ObjcId, indices: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, gatherWithUpdatesTensor: x indicesTensor: indices axis: axis as u64 batchDimensions: 0u64 name: nil()]
    }
    /// `updates` [m] added into zeros of `dims` at `indices` [m, 1] (MPSGraphScatterModeAdd).
    pub(super) unsafe fn scatter_add(&self, updates: ObjcId, indices: ObjcId, dims: &[usize]) -> ObjcId {
        msg_send![self.0, scatterNDWithUpdatesTensor: updates indicesTensor: indices shape: graph_shape(dims) batchDimensions: 0u64 mode: 0i64 name: nil()]
    }
}
