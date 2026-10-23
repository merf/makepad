//! The GEMM on Metal Performance Shaders (`MPSMatrixMultiplication`), the default on macOS
//! (MAKEPAD_METAL_GEMM unset, auto or mps; `kernels` for `gemm.metal`'s): Apple's fp32 matrix
//! multiply, about twice the tiled kernel's speed on an M5 at Cutie's shapes, 15 to 26% off a
//! Cutie frame (RationalVisions docs/cutie-metal.md §11). It is encoded on
//! the stream's open command buffer between its compute encoders (`StreamCtx::begin_buffer_work`),
//! so it is ordered with the kernels around it as any dispatch is. Its sums are MPS's own, not
//! the tiled kernel's, so results move by rounding: judged by `cutie-validate all` within
//! `Tol::STAGE`, not against the golden bit for bit.
//!
//! Also Cutie's convolutions on `MPSCNNConvolution` (MAKEPAD_CUTIE_CONV=mpscnn, not the default
//! yet): the input planes copied into an rgba32Float texture array, the convolution and its bias on
//! MPS, and the result copied back with the residual and the ReLU (`cutie_mps.metal`,
//! `MetalStream::conv_mpscnn`). No im2col and no pass of its own for the epilogue.

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

/// The calling thread's `MPSMatrixMultiplication` kernels, one per shape and scalars; its
/// `MPSCNNConvolution`s, one per layer (weights key); and the convolutions' texture arrays, one
/// per role (in, out) and size, reused from one convolution to the next (Metal's hazard tracking
/// orders their uses). All made at first use.
#[derive(Default)]
pub(crate) struct MpsCache {
    kernels: HashMap<KernelKey, StrongId>,
    convs: HashMap<String, StrongId>,
    textures: HashMap<(u8, usize, usize, usize), StrongId>,
    /// `MPSMatrixDescriptor`s by rows, columns and row bytes.
    descriptors: HashMap<(usize, usize, usize), StrongId>,
}

/// `MTLTextureType2DArray`, `MTLPixelFormatRGBA32Float`, read and write usage, private storage.
const MTL_TEXTURE_TYPE_2D_ARRAY: u64 = 3;
const MTL_PIXEL_FORMAT_RGBA32_FLOAT: u64 = 125;
const MTL_TEXTURE_USAGE_READ_WRITE: u64 = 1 | 2;
const MTL_STORAGE_MODE_PRIVATE: u64 = 2;
/// `MPSNNConvolutionAccumulatorPrecisionOptionFloat`: fp32 sums (the default is half).
const MPS_ACCUMULATOR_FLOAT: u64 = 1;

/// `MPSOffset`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MpsOffset {
    x: i64,
    y: i64,
    z: i64,
}

/// One `MPSCNNConvolution` layer: its weights key, PyTorch's [out][in][k][k] weights, the bias,
/// the channels, the square kernel and the stride. Padding is PyTorch's k/2 on every side.
pub(crate) struct MpsConvLayer<'a> {
    pub key: &'a str,
    pub weights: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub k: usize,
    pub stride: usize,
}

impl MpsCache {
    /// The texture array for `role` (0 in, 1 out) of `width` x `height` and `layers` layers.
    pub(crate) fn texture(&mut self, device: ObjcId, role: u8, width: usize, height: usize, layers: usize) -> MetalResult<ObjcId> {
        let key = (role, width, height, layers);
        if !self.textures.contains_key(&key) {
            let texture = unsafe {
                let desc: ObjcId = msg_send![class!(MTLTextureDescriptor), new];
                let desc = StrongId::from_owned(desc).ok_or("MTLTextureDescriptor new returned nil")?;
                let d = desc.as_id();
                let _: () = msg_send![d, setTextureType: MTL_TEXTURE_TYPE_2D_ARRAY];
                let _: () = msg_send![d, setPixelFormat: MTL_PIXEL_FORMAT_RGBA32_FLOAT];
                let _: () = msg_send![d, setWidth: width as u64];
                let _: () = msg_send![d, setHeight: height as u64];
                let _: () = msg_send![d, setArrayLength: layers as u64];
                let _: () = msg_send![d, setUsage: MTL_TEXTURE_USAGE_READ_WRITE];
                let _: () = msg_send![d, setStorageMode: MTL_STORAGE_MODE_PRIVATE];
                let texture: ObjcId = msg_send![device, newTextureWithDescriptor: d];
                StrongId::from_owned(texture)
            }
            .ok_or_else(|| format!("could not allocate a {width}x{height}x{layers} rgba32Float texture array"))?;
            self.textures.insert(key, texture);
        }
        Ok(self.textures[&key].as_id())
    }

    /// The layer's `MPSCNNConvolution`, made at first use: its weights reordered to MPS's
    /// [out][k][k][in], fp32 sums, zero outside the image, offset 0 (each output centred on its
    /// source pixel times the stride, PyTorch's k/2 padding for an odd k).
    pub(crate) fn convolution(&mut self, device: ObjcId, layer: &MpsConvLayer<'_>) -> MetalResult<ObjcId> {
        if !self.convs.contains_key(layer.key) {
            let (o, i, k) = (layer.out_channels, layer.in_channels, layer.k);
            if layer.weights.len() != o * i * k * k || layer.bias.is_some_and(|b| b.len() != o) {
                return Err(format!("MPSCNNConvolution {}: {} weights for {o}x{i}x{k}x{k}", layer.key, layer.weights.len()));
            }
            let mut ohwi = vec![0.0f32; layer.weights.len()];
            for oc in 0..o {
                for ic in 0..i {
                    for y in 0..k {
                        for x in 0..k {
                            ohwi[((oc * k + y) * k + x) * i + ic] = layer.weights[((oc * i + ic) * k + y) * k + x];
                        }
                    }
                }
            }
            let conv = unsafe {
                let nil: ObjcId = std::ptr::null_mut();
                let desc: ObjcId = msg_send![
                    class!(MPSCNNConvolutionDescriptor),
                    cnnConvolutionDescriptorWithKernelWidth: k as u64
                    kernelHeight: k as u64
                    inputFeatureChannels: i as u64
                    outputFeatureChannels: o as u64
                    neuronFilter: nil
                ];
                if desc.is_null() {
                    return Err("MPSCNNConvolutionDescriptor returned nil".into());
                }
                let _: () = msg_send![desc, setStrideInPixelsX: layer.stride as u64];
                let _: () = msg_send![desc, setStrideInPixelsY: layer.stride as u64];
                let bias = layer.bias.map_or(std::ptr::null(), |b| b.as_ptr());
                let alloc: ObjcId = msg_send![class!(MPSCNNConvolution), alloc];
                let conv: ObjcId = msg_send![
                    alloc,
                    initWithDevice: device
                    convolutionDescriptor: desc
                    kernelWeights: ohwi.as_ptr()
                    biasTerms: bias
                    flags: 0u64
                ];
                StrongId::from_owned(conv)
            }
            .ok_or_else(|| format!("MPSCNNConvolution {}: init returned nil", layer.key))?;
            unsafe {
                let c = conv.as_id();
                if crate::objc_util::responds_to(c, sel!(setAccumulatorPrecisionOption:)) {
                    let _: () = msg_send![c, setAccumulatorPrecisionOption: MPS_ACCUMULATOR_FLOAT];
                }
                let _: () = msg_send![c, setOffset: MpsOffset { x: 0, y: 0, z: 0 }];
                // MPSImageEdgeModeZero.
                let _: () = msg_send![c, setEdgeMode: 0u64];
            }
            self.convs.insert(layer.key.to_string(), conv);
        }
        Ok(self.convs[layer.key].as_id())
    }

    /// `conv` from the image in `source` (`in_channels` feature channels) to `destination`
    /// (`out_channels`), encoded on `cb` (no compute encoder open on it).
    pub(crate) fn encode_convolution(
        &self,
        conv: ObjcId,
        cb: ObjcId,
        (source, in_channels): (ObjcId, usize),
        (destination, out_channels): (ObjcId, usize),
    ) -> MetalResult<()> {
        let image = |texture: ObjcId, channels: usize| unsafe {
            let alloc: ObjcId = msg_send![class!(MPSImage), alloc];
            let image: ObjcId = msg_send![alloc, initWithTexture: texture featureChannels: channels as u64];
            StrongId::from_owned(image).ok_or_else(|| "MPSImage init returned nil".to_string())
        };
        let (src, dst) = (image(source, in_channels)?, image(destination, out_channels)?);
        unsafe {
            let _: () = msg_send![conv, encodeToCommandBuffer: cb sourceImage: src.as_id() destinationImage: dst.as_id()];
        }
        Ok(())
    }

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
        let (a, b, c) = (self.matrix(a)?, self.matrix(b)?, self.matrix(c)?);
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

    /// An `MPSMatrix` over `m`'s buffer at its offset (the command buffer retains what it
    /// encodes); its descriptor kept by shape, as a GEMM's shapes repeat every step.
    fn matrix(&mut self, m: &MpsMatrix) -> MetalResult<StrongId> {
        let key = (m.rows, m.cols, m.row_bytes);
        if !self.descriptors.contains_key(&key) {
            let descriptor = unsafe {
                let descriptor: ObjcId = msg_send![
                    class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows: m.rows as u64
                    columns: m.cols as u64
                    rowBytes: m.row_bytes as u64
                    dataType: MPS_DATA_TYPE_FLOAT32
                ];
                StrongId::retain(descriptor)
            }
            .ok_or("MPSMatrixDescriptor returned nil")?;
            self.descriptors.insert(key, descriptor);
        }
        let descriptor = self.descriptors[&key].as_id();
        unsafe {
            let alloc: ObjcId = msg_send![class!(MPSMatrix), alloc];
            let matrix: ObjcId = msg_send![alloc, initWithBuffer: m.buffer offset: m.offset descriptor: descriptor];
            StrongId::from_owned(matrix).ok_or_else(|| "MPSMatrix init returned nil".into())
        }
    }
}
