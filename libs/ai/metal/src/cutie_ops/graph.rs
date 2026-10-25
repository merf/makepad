//! A chain of Cutie's convolutions as one MPSGraph in half (an fp16 step's, macOS): a small
//! program over numbered values, value 0 the input, each op adding the next value. One graph runs
//! the whole chain with its activations in half between the convolutions, instead of a graph a
//! convolution with a copy in and out of each (`gpu_cutie_conv2d_fused_f16`), so the pixel
//! encoder's 50-odd convolutions cost one copy in and one out a stage they return.
//!
//! A convolution adds its bias, then the residual (an earlier value of its own shape), then takes
//! the ReLU, in that order, as `gpu_cutie_conv2d_fused` rounds them; all in half here, as
//! PyTorch's autocast runs them. The max-pool is 3x3, stride 2, padding 1, the padding never the
//! maximum (PyTorch's -inf padding). The types compile on every OS; the op runs on macOS only.

use crate::stream::MetalResult;

/// One convolution of a program: PyTorch's [out][in][k][k] weights, the bias, square, symmetric
/// padding. `key` names it in errors.
#[derive(Clone, Copy)]
pub struct CutieGraphConv<'a> {
    pub key: &'a str,
    pub weights: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub k: usize,
    pub pad: usize,
    pub stride: usize,
}

/// One op of a program, reading earlier values by index (0 the input).
#[derive(Clone, Copy)]
pub enum CutieGraphOp<'a> {
    /// relu?(conv(src) + bias + residual?).
    Conv { src: usize, conv: CutieGraphConv<'a>, residual: Option<usize>, relu: bool },
    /// Max-pool 3x3, stride 2, padding 1.
    MaxPool3x3s2 { src: usize },
}

/// A value's shape: channels, width and height (one image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphShape {
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

/// A convolution's output size along one axis (PyTorch's floor rule).
fn out_size(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad).saturating_sub(k) / stride + 1
}

/// Every value's shape, value 0 `input`'s; `Err` naming the op where a source is not an earlier
/// value, a convolution's channels are not its source's, or a residual is not the output's shape.
pub fn graph_shapes(input: GraphShape, ops: &[CutieGraphOp<'_>]) -> MetalResult<Vec<GraphShape>> {
    let mut shapes = vec![input];
    for (i, op) in ops.iter().enumerate() {
        let at = |v: usize| shapes.get(v).copied().ok_or_else(|| format!("cutie graph op {i}: value {v} is not made yet"));
        let shape = match *op {
            CutieGraphOp::Conv { src, conv, residual, .. } => {
                let s = at(src)?;
                if s.channels != conv.in_channels || conv.k == 0 || conv.stride == 0 {
                    return Err(format!("cutie graph op {i} ({}): {} channels in, the conv takes {}", conv.key, s.channels, conv.in_channels));
                }
                if conv.weights.len() != conv.out_channels * conv.in_channels * conv.k * conv.k || conv.bias.is_some_and(|b| b.len() != conv.out_channels) {
                    return Err(format!("cutie graph op {i} ({}): {} weights for {}x{}x{}x{}", conv.key, conv.weights.len(), conv.out_channels, conv.in_channels, conv.k, conv.k));
                }
                let out = GraphShape {
                    channels: conv.out_channels,
                    width: out_size(s.width, conv.k, conv.pad, conv.stride),
                    height: out_size(s.height, conv.k, conv.pad, conv.stride),
                };
                if let Some(r) = residual {
                    if at(r)? != out {
                        return Err(format!("cutie graph op {i} ({}): the residual is not the output's shape", conv.key));
                    }
                }
                out
            }
            CutieGraphOp::MaxPool3x3s2 { src } => {
                let s = at(src)?;
                GraphShape { channels: s.channels, width: out_size(s.width, 3, 1, 2), height: out_size(s.height, 3, 1, 2) }
            }
        };
        shapes.push(shape);
    }
    Ok(shapes)
}

/// `ops` run on `x` (one image, [channels, height * width]) as one graph in half, cached under
/// `key` and the input's size: the values `outputs` names, each a new tensor [channels, h * w] in
/// fp32.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_graph_f16(
    x: &crate::gpu_types::GpuTensor, width: usize, height: usize, key: &str, ops: &[CutieGraphOp<'_>], outputs: &[usize],
) -> MetalResult<Vec<crate::gpu_types::GpuTensor>> {
    gpu_cutie_graph(x, width, height, key, ops, outputs, true)
}

/// `gpu_cutie_graph_f16` in fp32 throughout (an fp32 step's), cached apart from it.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_graph_f32(
    x: &crate::gpu_types::GpuTensor, width: usize, height: usize, key: &str, ops: &[CutieGraphOp<'_>], outputs: &[usize],
) -> MetalResult<Vec<crate::gpu_types::GpuTensor>> {
    gpu_cutie_graph(x, width, height, &format!("{key}::f32"), ops, outputs, false)
}

#[cfg(target_os = "macos")]
fn gpu_cutie_graph(
    x: &crate::gpu_types::GpuTensor, width: usize, height: usize, key: &str, ops: &[CutieGraphOp<'_>], outputs: &[usize], half: bool,
) -> MetalResult<Vec<crate::gpu_types::GpuTensor>> {
    use crate::gpu_types::GpuTensor;
    use crate::stream::{with_metal_stream, MpsGraphRun};
    let op = if half { "gpu_cutie_graph_f16" } else { "gpu_cutie_graph_f32" };
    if x.cols() != width * height {
        return Err(format!("{op} {key}: {}x{} for {width}x{height}", x.rows(), x.cols()));
    }
    let shapes = graph_shapes(GraphShape { channels: x.rows(), width, height }, ops)?;
    for &o in outputs {
        if o >= shapes.len() {
            return Err(format!("{op} {key}: output {o} of {} values", shapes.len()));
        }
    }
    with_metal_stream(op, |s| {
        let outs: Vec<GpuTensor> =
            outputs.iter().map(|&o| GpuTensor::alloc(shapes[o].channels, shapes[o].width * shapes[o].height)).collect::<MetalResult<_>>()?;
        let refs: Vec<&GpuTensor> = outs.iter().collect();
        s.graph_mpsgraph(&MpsGraphRun {
            key,
            x,
            ops,
            shapes: &shapes,
            outputs,
            outs: &refs,
            to_half: if half { &super::matmul_conv::PLANAR_TO_HALF } else { &super::matmul_conv::GRAPH_TO_PLANAR },
            from_graph: &super::matmul_conv::GRAPH_TO_PLANAR,
            half,
        })?;
        Ok(outs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_follow_the_convolutions_and_the_pool() {
        let w1 = vec![0.0; 8 * 3 * 7 * 7];
        let w2 = vec![0.0; 8 * 8];
        let conv1 = CutieGraphConv { key: "c1", weights: &w1, bias: None, in_channels: 3, out_channels: 8, k: 7, pad: 3, stride: 2 };
        let conv2 = CutieGraphConv { key: "c2", weights: &w2, bias: None, in_channels: 8, out_channels: 8, k: 1, pad: 0, stride: 1 };
        let ops = [
            CutieGraphOp::Conv { src: 0, conv: conv1, residual: None, relu: true },
            CutieGraphOp::MaxPool3x3s2 { src: 1 },
            CutieGraphOp::Conv { src: 2, conv: conv2, residual: Some(2), relu: true },
        ];
        let shapes = graph_shapes(GraphShape { channels: 3, width: 960, height: 544 }, &ops).unwrap();
        assert_eq!(shapes[1], GraphShape { channels: 8, width: 480, height: 272 });
        assert_eq!(shapes[2], GraphShape { channels: 8, width: 240, height: 136 });
        assert_eq!(shapes[3], shapes[2]);
        let bad = [CutieGraphOp::Conv { src: 0, conv: conv2, residual: None, relu: false }];
        assert!(graph_shapes(GraphShape { channels: 3, width: 4, height: 4 }, &bad).is_err());
    }
}
