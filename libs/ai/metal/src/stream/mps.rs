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

#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
extern "C" {}
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
    textures: HashMap<(u8, usize, usize, usize, u64), StrongId>,
    /// `MPSMatrixDescriptor`s by rows, columns and row bytes.
    descriptors: HashMap<(usize, usize, usize), StrongId>,
    /// An fp16 step's convolutions on MPSGraph, one graph a layer and input shape.
    graphs: HashMap<GraphKey, GraphConv>,
    /// Their input (half) and output (fp32) buffers, one per role and size, shared as the
    /// textures are.
    graph_buffers: HashMap<(u8, usize), StrongId>,
}

/// A graph's layer (weights key), images, and input width and height.
type GraphKey = (String, usize, usize, usize);

/// One layer's convolution as an MPSGraph: an fp16 input placeholder, the fp32 weights as a
/// constant cast to half in the graph, the convolution (fp16), its result cast to fp32 and the
/// bias added in fp32. The input and output `MPSGraphTensorData` wrap the shared buffers.
pub(crate) struct GraphConv {
    graph: StrongId,
    feeds: StrongId,
    results: StrongId,
    pub(crate) input: ObjcId,
    pub(crate) output: ObjcId,
}

/// `MPSDataTypeFloat16`.
const MPS_DATA_TYPE_FLOAT16: u32 = 0x1000_0000 | 16;
/// `MPSGraphPaddingStyleExplicit`, `MPSGraphTensorNamedDataLayoutNCHW` and `OIHW`.
const GRAPH_PADDING_EXPLICIT: u64 = 0;
const GRAPH_LAYOUT_NCHW: u64 = 0;
const GRAPH_LAYOUT_OIHW: u64 = 2;

/// An `NSArray` of `NSNumber`s: a graph tensor's shape (autoreleased).
fn graph_shape(dims: &[usize]) -> ObjcId {
    unsafe {
        let numbers: Vec<ObjcId> = dims.iter().map(|&d| msg_send![class!(NSNumber), numberWithLongLong: d as i64]).collect();
        msg_send![class!(NSArray), arrayWithObjects: numbers.as_ptr() count: numbers.len() as u64]
    }
}

/// A graph's input or output size: images, channels, width and height.
pub(crate) struct GraphSide {
    pub images: usize,
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

/// `MTLTextureType2DArray`, `MTLPixelFormatR32Float`, `RG32Float` and `RGBA32Float`, read and
/// write usage, private storage.
const MTL_TEXTURE_TYPE_2D_ARRAY: u64 = 3;
const MTL_PIXEL_FORMAT_R32_FLOAT: u64 = 55;
const MTL_PIXEL_FORMAT_RG32_FLOAT: u64 = 105;
const MTL_PIXEL_FORMAT_RGBA32_FLOAT: u64 = 125;
const MTL_PIXEL_FORMAT_R16_FLOAT: u64 = 25;
const MTL_PIXEL_FORMAT_RG16_FLOAT: u64 = 65;
const MTL_PIXEL_FORMAT_RGBA16_FLOAT: u64 = 115;

/// The pixel format an `MPSImage` of `channels` feature channels must have: one or two channels
/// want a texture of as many (MPSImage asserts "Texture pixel format doesnt match number of
/// feature channels" on an rgba one), three or more rgba slices of four. In fp32, or in half
/// (`half`: Cutie's fp16 steps, the convolution's sums still fp32).
fn pixel_format(channels: usize, half: bool) -> u64 {
    match (channels, half) {
        (1, false) => MTL_PIXEL_FORMAT_R32_FLOAT,
        (2, false) => MTL_PIXEL_FORMAT_RG32_FLOAT,
        (_, false) => MTL_PIXEL_FORMAT_RGBA32_FLOAT,
        (1, true) => MTL_PIXEL_FORMAT_R16_FLOAT,
        (2, true) => MTL_PIXEL_FORMAT_RG16_FLOAT,
        (_, true) => MTL_PIXEL_FORMAT_RGBA16_FLOAT,
    }
}
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
    /// The texture array for `role` (0 in, 1 out) of `width` x `height` and `layers` layers, for an
    /// image of `channels` feature channels, in half when `half` ([`pixel_format`]).
    pub(crate) fn texture(
        &mut self, device: ObjcId, role: u8, width: usize, height: usize, layers: usize, channels: usize, half: bool,
    ) -> MetalResult<ObjcId> {
        let format = pixel_format(channels, half);
        let key = (role, width, height, layers, format);
        if !self.textures.contains_key(&key) {
            let texture = unsafe {
                let desc: ObjcId = msg_send![class!(MTLTextureDescriptor), new];
                let desc = StrongId::from_owned(desc).ok_or("MTLTextureDescriptor new returned nil")?;
                let d = desc.as_id();
                let _: () = msg_send![d, setTextureType: MTL_TEXTURE_TYPE_2D_ARRAY];
                let _: () = msg_send![d, setPixelFormat: format];
                let _: () = msg_send![d, setWidth: width as u64];
                let _: () = msg_send![d, setHeight: height as u64];
                let _: () = msg_send![d, setArrayLength: layers as u64];
                let _: () = msg_send![d, setUsage: MTL_TEXTURE_USAGE_READ_WRITE];
                let _: () = msg_send![d, setStorageMode: MTL_STORAGE_MODE_PRIVATE];
                let texture: ObjcId = msg_send![device, newTextureWithDescriptor: d];
                StrongId::from_owned(texture)
            }
            .ok_or_else(|| format!("could not allocate a {width}x{height}x{layers} texture array (pixel format {format})"))?;
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

    /// The shared buffer for `role` (0 the input in half, 1 the output in fp32) of `bytes`.
    fn graph_buffer(&mut self, device: ObjcId, role: u8, bytes: usize) -> MetalResult<ObjcId> {
        if !self.graph_buffers.contains_key(&(role, bytes)) {
            let buffer = unsafe {
                // MTLResourceStorageModePrivate.
                let b: ObjcId = msg_send![device, newBufferWithLength: bytes.max(4) as u64 options: (MTL_STORAGE_MODE_PRIVATE << 4)];
                StrongId::from_owned(b)
            }
            .ok_or_else(|| format!("could not allocate a {bytes}-byte buffer for MPSGraph"))?;
            self.graph_buffers.insert((role, bytes), buffer);
        }
        Ok(self.graph_buffers[&(role, bytes)].as_id())
    }

    /// The layer's convolution on MPSGraph for an input of `input` and an output of `output`,
    /// made at first use (its graph compiles at its first encode), and the buffers it reads and
    /// writes: its input in half and its output in fp32, NCHW from their start.
    pub(crate) fn graph_convolution(
        &mut self, device: ObjcId, layer: &MpsConvLayer<'_>, input: &GraphSide, output: &GraphSide,
    ) -> MetalResult<(&GraphConv, ObjcId, ObjcId)> {
        let (o, i, k) = (layer.out_channels, layer.in_channels, layer.k);
        if layer.weights.len() != o * i * k * k || layer.bias.is_some_and(|b| b.len() != o) || input.channels != i || output.channels != o {
            return Err(format!("MPSGraph convolution {}: {} weights for {o}x{i}x{k}x{k}", layer.key, layer.weights.len()));
        }
        let in_bytes = input.images * i * input.width * input.height * 2;
        let out_bytes = output.images * o * output.width * output.height * 4;
        let in_buf = self.graph_buffer(device, 0, in_bytes)?;
        let out_buf = self.graph_buffer(device, 1, out_bytes)?;
        let key = (layer.key.to_string(), input.images, input.width, input.height);
        if !self.graphs.contains_key(&key) {
            let pad = (k / 2) as u64;
            let conv = unsafe {
                let nil: ObjcId = std::ptr::null_mut();
                let graph: ObjcId = msg_send![class!(MPSGraph), new];
                let graph = StrongId::from_owned(graph).ok_or("MPSGraph new returned nil")?;
                let g = graph.as_id();
                let in_shape = graph_shape(&[input.images, i, input.height, input.width]);
                let out_shape = graph_shape(&[output.images, o, output.height, output.width]);
                let x: ObjcId = msg_send![g, placeholderWithShape: in_shape dataType: MPS_DATA_TYPE_FLOAT16 name: nil];
                let wdata: ObjcId = msg_send![class!(NSData), dataWithBytes: layer.weights.as_ptr() as *const std::ffi::c_void length: (layer.weights.len() * 4) as u64];
                let w32: ObjcId = msg_send![g, constantWithData: wdata shape: graph_shape(&[o, i, k, k]) dataType: MPS_DATA_TYPE_FLOAT32];
                let w16: ObjcId = msg_send![g, castTensor: w32 toType: MPS_DATA_TYPE_FLOAT16 name: nil];
                let desc: ObjcId = msg_send![
                    class!(MPSGraphConvolution2DOpDescriptor),
                    descriptorWithStrideInX: layer.stride as u64
                    strideInY: layer.stride as u64
                    dilationRateInX: 1u64
                    dilationRateInY: 1u64
                    groups: 1u64
                    paddingLeft: pad
                    paddingRight: pad
                    paddingTop: pad
                    paddingBottom: pad
                    paddingStyle: GRAPH_PADDING_EXPLICIT
                    dataLayout: GRAPH_LAYOUT_NCHW
                    weightsLayout: GRAPH_LAYOUT_OIHW
                ];
                if desc.is_null() {
                    return Err("MPSGraphConvolution2DOpDescriptor returned nil".into());
                }
                let y16: ObjcId = msg_send![g, convolution2DWithSourceTensor: x weightsTensor: w16 descriptor: desc name: nil];
                let mut y: ObjcId = msg_send![g, castTensor: y16 toType: MPS_DATA_TYPE_FLOAT32 name: nil];
                if let Some(bias) = layer.bias {
                    let bdata: ObjcId = msg_send![class!(NSData), dataWithBytes: bias.as_ptr() as *const std::ffi::c_void length: (bias.len() * 4) as u64];
                    let b: ObjcId = msg_send![g, constantWithData: bdata shape: graph_shape(&[1, o, 1, 1]) dataType: MPS_DATA_TYPE_FLOAT32];
                    y = msg_send![g, additionWithPrimaryTensor: y secondaryTensor: b name: nil];
                }
                let xd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                let xd: ObjcId = msg_send![xd, initWithMTLBuffer: in_buf shape: in_shape dataType: MPS_DATA_TYPE_FLOAT16];
                let xd = StrongId::from_owned(xd).ok_or("MPSGraphTensorData init returned nil")?;
                let yd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                let yd: ObjcId = msg_send![yd, initWithMTLBuffer: out_buf shape: out_shape dataType: MPS_DATA_TYPE_FLOAT32];
                let yd = StrongId::from_owned(yd).ok_or("MPSGraphTensorData init returned nil")?;
                let feeds: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: xd.as_id() forKey: x];
                let results: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: yd.as_id() forKey: y];
                let feeds = StrongId::retain(feeds).ok_or("NSDictionary returned nil")?;
                let results = StrongId::retain(results).ok_or("NSDictionary returned nil")?;
                GraphConv { graph, feeds, results, input: x, output: y }
            };
            self.graphs.insert(key.clone(), conv);
        }
        Ok((&self.graphs[&key], in_buf, out_buf))
    }

    /// A program of convolutions (`cutie_ops::graph`) as one graph in half, made at first use
    /// under `key` and the input's shape, and the buffers it reads (the input in half) and writes
    /// (each output in fp32), NCHW from their start. Values are half between the ops; each
    /// output is cast to fp32.
    pub(crate) fn program_graph(
        &mut self, device: ObjcId, key: &str, ops: &[crate::cutie_ops::CutieGraphOp<'_>], shapes: &[crate::cutie_ops::graph::GraphShape],
        outputs: &[usize],
    ) -> MetalResult<(ObjcId, Vec<ObjcId>)> {
        use crate::cutie_ops::CutieGraphOp;
        let size = |s: &crate::cutie_ops::graph::GraphShape| s.channels * s.width * s.height;
        let in_buf = self.graph_buffer(device, 2, size(&shapes[0]) * 2)?;
        let mut out_bufs = Vec::with_capacity(outputs.len());
        for (i, &o) in outputs.iter().enumerate() {
            out_bufs.push(self.graph_buffer(device, 10 + i as u8, size(&shapes[o]) * 4)?);
        }
        let gkey = (format!("program::{key}"), 1, shapes[0].width, shapes[0].height);
        if !self.graphs.contains_key(&gkey) {
            let built = unsafe {
                let nil: ObjcId = std::ptr::null_mut();
                let graph: ObjcId = msg_send![class!(MPSGraph), new];
                let graph = StrongId::from_owned(graph).ok_or("MPSGraph new returned nil")?;
                let g = graph.as_id();
                let nchw = |s: &crate::cutie_ops::graph::GraphShape| graph_shape(&[1, s.channels, s.height, s.width]);
                let x: ObjcId = msg_send![g, placeholderWithShape: nchw(&shapes[0]) dataType: MPS_DATA_TYPE_FLOAT16 name: nil];
                let mut values: Vec<ObjcId> = vec![x];
                for op in ops {
                    let v: ObjcId = match *op {
                        CutieGraphOp::Conv { src, conv, residual, relu } => {
                            let (o, i, k, pad) = (conv.out_channels, conv.in_channels, conv.k, conv.pad as u64);
                            let wdata: ObjcId = msg_send![class!(NSData), dataWithBytes: conv.weights.as_ptr() as *const std::ffi::c_void length: (conv.weights.len() * 4) as u64];
                            let w32: ObjcId = msg_send![g, constantWithData: wdata shape: graph_shape(&[o, i, k, k]) dataType: MPS_DATA_TYPE_FLOAT32];
                            let w16: ObjcId = msg_send![g, castTensor: w32 toType: MPS_DATA_TYPE_FLOAT16 name: nil];
                            let desc: ObjcId = msg_send![
                                class!(MPSGraphConvolution2DOpDescriptor),
                                descriptorWithStrideInX: conv.stride as u64
                                strideInY: conv.stride as u64
                                dilationRateInX: 1u64
                                dilationRateInY: 1u64
                                groups: 1u64
                                paddingLeft: pad
                                paddingRight: pad
                                paddingTop: pad
                                paddingBottom: pad
                                paddingStyle: GRAPH_PADDING_EXPLICIT
                                dataLayout: GRAPH_LAYOUT_NCHW
                                weightsLayout: GRAPH_LAYOUT_OIHW
                            ];
                            if desc.is_null() {
                                return Err(format!("MPSGraph program {key}: a convolution descriptor returned nil"));
                            }
                            let mut y: ObjcId = msg_send![g, convolution2DWithSourceTensor: values[src] weightsTensor: w16 descriptor: desc name: nil];
                            if let Some(bias) = conv.bias {
                                let bdata: ObjcId = msg_send![class!(NSData), dataWithBytes: bias.as_ptr() as *const std::ffi::c_void length: (bias.len() * 4) as u64];
                                let b32: ObjcId = msg_send![g, constantWithData: bdata shape: graph_shape(&[1, o, 1, 1]) dataType: MPS_DATA_TYPE_FLOAT32];
                                let b16: ObjcId = msg_send![g, castTensor: b32 toType: MPS_DATA_TYPE_FLOAT16 name: nil];
                                y = msg_send![g, additionWithPrimaryTensor: y secondaryTensor: b16 name: nil];
                            }
                            if let Some(r) = residual {
                                y = msg_send![g, additionWithPrimaryTensor: y secondaryTensor: values[r] name: nil];
                            }
                            if relu {
                                y = msg_send![g, reLUWithTensor: y name: nil];
                            }
                            y
                        }
                        CutieGraphOp::MaxPool3x3s2 { src } => {
                            let desc: ObjcId = msg_send![
                                class!(MPSGraphPooling2DOpDescriptor),
                                descriptorWithKernelWidth: 3u64
                                kernelHeight: 3u64
                                strideInX: 2u64
                                strideInY: 2u64
                                dilationRateInX: 1u64
                                dilationRateInY: 1u64
                                paddingLeft: 1u64
                                paddingRight: 1u64
                                paddingTop: 1u64
                                paddingBottom: 1u64
                                paddingStyle: GRAPH_PADDING_EXPLICIT
                                dataLayout: GRAPH_LAYOUT_NCHW
                            ];
                            if desc.is_null() {
                                return Err(format!("MPSGraph program {key}: the pooling descriptor returned nil"));
                            }
                            msg_send![g, maxPooling2DWithSourceTensor: values[src] descriptor: desc name: nil]
                        }
                    };
                    values.push(v);
                }
                let xd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                let xd: ObjcId = msg_send![xd, initWithMTLBuffer: in_buf shape: nchw(&shapes[0]) dataType: MPS_DATA_TYPE_FLOAT16];
                let xd = StrongId::from_owned(xd).ok_or("MPSGraphTensorData init returned nil")?;
                let mut keys: Vec<ObjcId> = Vec::new();
                let mut datas: Vec<StrongId> = Vec::new();
                for (&o, &buf) in outputs.iter().zip(&out_bufs) {
                    let y: ObjcId = msg_send![g, castTensor: values[o] toType: MPS_DATA_TYPE_FLOAT32 name: nil];
                    let yd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                    let yd: ObjcId = msg_send![yd, initWithMTLBuffer: buf shape: nchw(&shapes[o]) dataType: MPS_DATA_TYPE_FLOAT32];
                    datas.push(StrongId::from_owned(yd).ok_or("MPSGraphTensorData init returned nil")?);
                    keys.push(y);
                }
                let objects: Vec<ObjcId> = datas.iter().map(StrongId::as_id).collect();
                let feeds: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: xd.as_id() forKey: x];
                let results: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObjects: objects.as_ptr() forKeys: keys.as_ptr() count: keys.len() as u64];
                let feeds = StrongId::retain(feeds).ok_or("NSDictionary returned nil")?;
                let results = StrongId::retain(results).ok_or("NSDictionary returned nil")?;
                GraphConv { graph, feeds, results, input: x, output: keys.first().copied().unwrap_or(x) }
            };
            self.graphs.insert(gkey, built);
        }
        Ok((in_buf, out_bufs))
    }

    /// The program graph made for `key` and an input of `input`'s shape, if `program_graph` made it.
    pub(crate) fn program(&self, key: &str, input: crate::cutie_ops::graph::GraphShape) -> Option<&GraphConv> {
        self.graphs.get(&(format!("program::{key}"), 1, input.width, input.height))
    }

    /// The graph made for `key` (layer, images, width, height), if `graph_convolution` made it.
    pub(crate) fn graph(&self, key: &GraphKey) -> Option<&GraphConv> {
        self.graphs.get(key)
    }

    /// `graph` encoded on `cb` (no compute encoder open on it), through an `MPSCommandBuffer` round
    /// it; `Err` if MPSGraph moved on to a command buffer of its own (it committed `cb`).
    pub(crate) fn encode_graph(graph: &GraphConv, cb: ObjcId) -> MetalResult<()> {
        unsafe {
            let mcb: ObjcId = msg_send![class!(MPSCommandBuffer), commandBufferWithCommandBuffer: cb];
            if mcb.is_null() {
                return Err("MPSCommandBuffer returned nil".into());
            }
            let nil: ObjcId = std::ptr::null_mut();
            // An execution descriptor that never commits the command buffer to start the GPU
            // early (`enableCommitAndContinue`, as PyTorch's MPS stream can turn it off): MPSGraph
            // otherwise does for a large graph, and the stream's command buffer is the stream's.
            let descriptor: ObjcId = msg_send![class!(MPSGraphExecutionDescriptor), new];
            let descriptor = StrongId::from_owned(descriptor).ok_or("MPSGraphExecutionDescriptor new returned nil")?;
            if crate::objc_util::responds_to(descriptor.as_id(), sel!(setEnableCommitAndContinue:)) {
                let _: () = msg_send![descriptor.as_id(), setEnableCommitAndContinue: NO];
            }
            let _: () = msg_send![
                graph.graph.as_id(),
                encodeToCommandBuffer: mcb
                feeds: graph.feeds.as_id()
                targetOperations: nil
                resultsDictionary: graph.results.as_id()
                executionDescriptor: descriptor.as_id()
            ];
            let now: ObjcId = msg_send![mcb, commandBuffer];
            if now != cb {
                return Err("MPSGraph committed the stream's command buffer and went on to another".into());
            }
        }
        Ok(())
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
