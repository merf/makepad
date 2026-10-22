//! Cutie's softmax down columns, aggregate, ECA, layer norm and sine encoding as Metal kernels
//! (RationalVisions docs/cutie-metal.md, family (c)), at the signatures of the macOS
//! `gpu_cutie_*` functions (models/common's gpu_cutie_host.rs, the same as CUDA's in
//! cuda/src/cutie_ops.rs), and `gpu_layer_norm_pytorch` on the same layer norm kernel, so the ops
//! check's EXACT comparison of the two holds by construction as it does on CUDA. The kernels are
//! `shaders/makepad/cutie_softmax_norm.metal`, ported from cuda/kernels/cutie.cu and, for the
//! layer norm, diffusion_ops.cu; each op's host reference is in cuda/src/cutie_reference.rs.
//!
//! Each op makes the checks CUDA's wrapper makes, in its order and with its messages, before
//! anything is encoded, and launches CUDA's kernels with CUDA's threadgroups where the order of a
//! sum depends on them (the softmax tile, the ECA row mean, the layer norm). Two departures:
//! - The kernels index in u32 where CUDA's count a few sizes in size_t, so an op whose elements,
//!   or any size its kernel takes, pass u32 is `Err` here (`cutie_u32`'s message) where CUDA
//!   would run it: four billion floats, never Cutie's.
//! - ECA with more than [`CUTIE_ECA_MAX_TAPS`] taps is `Err`: CUDA runs the host reference for
//!   those (never Cutie's 3 or 5), which this crate cannot call, so the caller does
//!   (models/common's gpu_cutie_host.rs).
//!
//! The checks, the kernels' params and their grids are plain functions of the sizes, compiled and
//! tested on any OS, with the kernels' reduction trees run on the host as the MSL writes them; the
//! ops are macOS only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::{Grid, Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::stream::{upload, with_metal_stream, Bind, MetalStream};
#[cfg(target_os = "macos")]
use crate::GpuTensor;

/// Cutie's softmax, aggregate, ECA, layer norm and sine kernels (`cutie_softmax_norm.metal`).
pub static CUTIE_SOFTMAX_NORM: KernelLib = KernelLib {
    name: "cutie_softmax_norm",
    source: include_str!("../../shaders/makepad/cutie_softmax_norm.metal"),
};

/// Softmax down each column, a thread a column (`CutieSoftmaxAxis0`).
pub static CUTIE_SOFTMAX_AXIS0: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_softmax_axis0_f32" };
/// The same, a threadgroup a tile of columns over many rows (`CutieSoftmaxAxis0`).
pub static CUTIE_SOFTMAX_AXIS0_TILE: Kernel =
    Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_softmax_axis0_tile_f32" };
/// Probabilities to logits with a background row first (`CutieAggregate`).
pub static CUTIE_AGGREGATE: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_aggregate_f32" };
/// ECA's channel means, a threadgroup a row (`CutieEcaRowMean`).
pub static CUTIE_ECA_ROW_MEAN: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_eca_row_mean_f32" };
/// ECA's per-channel scale (`CutieEcaScale`).
pub static CUTIE_ECA_SCALE: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_eca_scale_f32" };
/// x * scale + residual (`CutieEcaApply`).
pub static CUTIE_ECA_APPLY: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_eca_apply_f32" };
/// PyTorch's LayerNorm, a threadgroup a row (`CutieLayerNorm`).
pub static CUTIE_LAYER_NORM: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_layer_norm_f32" };
/// Cutie's sine positional encoding (`CutieSineEncoding`).
pub static CUTIE_SINE_ENCODING: Kernel = Kernel { lib: &CUTIE_SOFTMAX_NORM, function: "cutie_sine_encoding_f32" };

/// Every kernel above, for a bring-up that compiles them all.
pub static SOFTMAX_NORM_KERNELS: &[&Kernel] = &[
    &CUTIE_SOFTMAX_AXIS0,
    &CUTIE_SOFTMAX_AXIS0_TILE,
    &CUTIE_AGGREGATE,
    &CUTIE_ECA_ROW_MEAN,
    &CUTIE_ECA_SCALE,
    &CUTIE_ECA_APPLY,
    &CUTIE_LAYER_NORM,
    &CUTIE_SINE_ENCODING,
];

/// The most taps ECA's kernel takes by value (Cutie's ECA has 3 or 5), as CUDA's
/// `CUTIE_ECA_MAX_TAPS`. CUDA runs the host reference beyond it; here that is the caller's choice.
pub const CUTIE_ECA_MAX_TAPS: usize = 15;

/// Threads to a threadgroup, as CUDA's launches (256).
const THREADS: u32 = 256;
/// The column softmax's threadgroup, as CUDA's (128).
const SOFTMAX_COLUMN_THREADS: u32 = 128;
/// The tile softmax's threadgroup: columns, and thread rows striding down them (CUDA's
/// CUTIE_SOFTMAX_TILE_COLS and _ROWS).
const SOFTMAX_TILE_COLS: u32 = 8;
const SOFTMAX_TILE_ROWS: u32 = 128;
/// The layer norm's threadgroup: PyTorch's dim3(warp_size, 4).
const LAYER_NORM_THREADS: [u32; 3] = [32, 4, 1];
/// The SIMD width CUDA's butterflies and shuffle-down trees are written for.
const WARP: u32 = 32;

/// `cutie_softmax_axis0_f32`'s and `cutie_softmax_axis0_tile_f32`'s params: x [rows, cols].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieSoftmaxAxis0 {
    pub rows: u32,
    pub cols: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieSoftmaxAxis0` in MSL.
unsafe impl KernelParams for CutieSoftmaxAxis0 {}

/// `cutie_aggregate_f32`'s params: prob [objects, pixels].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieAggregate {
    pub objects: u32,
    pub pixels: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieAggregate` in MSL.
unsafe impl KernelParams for CutieAggregate {}

/// `cutie_eca_row_mean_f32`'s params: x [rows, plane], a threadgroup a row.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieEcaRowMean {
    pub plane: u32,
}
// SAFETY: #[repr(C)], 4 bytes, no pointers; `CutieEcaRowMean` in MSL.
unsafe impl KernelParams for CutieEcaRowMean {}

/// `cutie_eca_scale_f32`'s params: the taps by value (zero past `k`), and means [rows] of images
/// of `channels` channels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieEcaScale {
    pub taps: [f32; CUTIE_ECA_MAX_TAPS],
    pub k: u32,
    pub channels: u32,
    pub rows: u32,
}
// SAFETY: #[repr(C)], 72 bytes, no pointers; `CutieEcaScale` in MSL.
unsafe impl KernelParams for CutieEcaScale {}

/// `cutie_eca_apply_f32`'s params: rows of `plane`, `n` elements in all.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieEcaApply {
    pub plane: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieEcaApply` in MSL.
unsafe impl KernelParams for CutieEcaApply {}

/// `cutie_layer_norm_f32`'s params: rows of `cols` (a multiple of 4), and eps.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieLayerNorm {
    pub cols: u32,
    pub eps: f32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieLayerNorm` in MSL.
unsafe impl KernelParams for CutieLayerNorm {}

/// `cutie_sine_encoding_f32`'s params: `freqs` frequencies at `w x h`, the position scale and
/// eps, and `n` = w * h * 4 * freqs entries.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieSineEncoding {
    pub freqs: u32,
    pub w: u32,
    pub h: u32,
    pub scale: f32,
    pub eps: f32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 24 bytes, no pointers; `CutieSineEncoding` in MSL.
unsafe impl KernelParams for CutieSineEncoding {}

/// A size as the kernels take it (CUDA's `cutie_u32`, with its message).
fn cutie_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// The product of `sizes` as a kernel's u32 count: 0 when any size is 0, otherwise `cutie_u32`'s
/// message when it does not fit.
fn cutie_count(op: &str, sizes: &[usize]) -> MetalResult<u32> {
    if sizes.contains(&0) {
        return Ok(0);
    }
    match sizes.iter().try_fold(1usize, |n, &size| n.checked_mul(size)) {
        Some(n) => cutie_u32(op, n),
        None => {
            let sizes: Vec<String> = sizes.iter().map(|size| size.to_string()).collect();
            Err(format!("{op}: {} does not fit the kernel's 32-bit sizes", sizes.join("x")))
        }
    }
}

// --- Softmax down columns ---

/// `x` [m, p] as (rows, cols).
fn softmax_axis0_params(rows: usize, cols: usize) -> MetalResult<CutieSoftmaxAxis0> {
    let op = "gpu_cutie_softmax_axis0";
    let (m, p) = (cutie_u32(op, rows)?, cutie_u32(op, cols)?);
    cutie_count(op, &[rows, cols])?;
    Ok(CutieSoftmaxAxis0 { rows: m, cols: p })
}

/// The kernel and grid CUDA's launcher chooses: nothing for an empty `x`; over more than 64 rows
/// and fewer than 8192 columns (consolidation's candidates by prototypes) a threadgroup of
/// [8, 128] a tile of 8 columns; otherwise a thread a column, 128 to a threadgroup.
fn softmax_axis0_launch(params: &CutieSoftmaxAxis0) -> MetalResult<(&'static Kernel, Grid)> {
    let (m, p) = (params.rows as usize, params.cols as usize);
    if m > 64 && p < 8192 {
        let groups = p.div_ceil(SOFTMAX_TILE_COLS as usize);
        let grid = Grid::groups([groups, 1, 1], [SOFTMAX_TILE_COLS, SOFTMAX_TILE_ROWS, 1])?;
        return Ok((&CUTIE_SOFTMAX_AXIS0_TILE, grid));
    }
    let groups = if m == 0 { 0 } else { p.div_ceil(SOFTMAX_COLUMN_THREADS as usize) };
    Ok((&CUTIE_SOFTMAX_AXIS0, Grid::groups([groups, 1, 1], [SOFTMAX_COLUMN_THREADS, 1, 1])?))
}

// --- Objects ---

/// `prob` [n, p] as (rows, cols); the output has a row more. No objects is a background alone.
fn aggregate_params(rows: usize, cols: usize) -> MetalResult<CutieAggregate> {
    let op = "gpu_cutie_aggregate";
    let objects = cutie_u32(op, rows)?;
    let pixels = cutie_u32(op, cols)?;
    cutie_count(op, &[rows + 1, cols])?;
    Ok(CutieAggregate { objects, pixels })
}

// --- ECA ---

/// ECA's three launches: each channel's mean, its scale, then x * scale + residual.
#[derive(Clone, Copy, Debug, PartialEq)]
struct EcaPlan {
    row_mean: CutieEcaRowMean,
    scale: CutieEcaScale,
    apply: CutieEcaApply,
}

impl EcaPlan {
    /// The rows (images * channels) a mean and a scale each.
    fn rows(&self) -> u32 {
        self.scale.rows
    }

    /// The three grids, as CUDA's launcher: a threadgroup of 256 a row for the means, 256 a
    /// threadgroup for the scales and for the elements; nothing when there are no elements.
    fn grids(&self) -> MetalResult<[Grid; 3]> {
        let live = if self.apply.n == 0 { 0 } else { self.rows() as usize };
        Ok([
            Grid::groups([live, 1, 1], [THREADS, 1, 1])?,
            Grid::linear(live, THREADS)?,
            Grid::linear(self.apply.n as usize, THREADS)?,
        ])
    }
}

/// `x` and `residual` as (rows, cols), `n` images of `c` channels, the taps `conv_w`.
fn eca_params(x: (usize, usize), residual: (usize, usize), n: usize, c: usize, conv_w: &[f32]) -> MetalResult<EcaPlan> {
    let op = "gpu_cutie_eca";
    if n.checked_mul(c) != Some(x.0) || residual.0 != x.0 || residual.1 != x.1 || conv_w.len() % 2 == 0 {
        return Err("gpu_cutie_eca shape mismatch".into());
    }
    if conv_w.len() > CUTIE_ECA_MAX_TAPS {
        return Err(format!(
            "{op}: {} taps, the kernel takes at most {CUTIE_ECA_MAX_TAPS} by value (CUDA runs the host reference for more)",
            conv_w.len()
        ));
    }
    cutie_u32(op, n)?;
    let channels = cutie_u32(op, c)?;
    let rows = cutie_u32(op, x.0)?;
    let plane = cutie_u32(op, x.1)?;
    let elements = cutie_count(op, &[x.0, x.1])?;
    let mut taps = [0.0f32; CUTIE_ECA_MAX_TAPS];
    taps[..conv_w.len()].copy_from_slice(conv_w);
    Ok(EcaPlan {
        row_mean: CutieEcaRowMean { plane },
        scale: CutieEcaScale { taps, k: conv_w.len() as u32, channels, rows },
        apply: CutieEcaApply { plane, n: elements },
    })
}

// --- Layer norm ---

/// `gpu_cutie_layer_norm`'s check for `x` [rows, cols] under weight key `key`, with CUDA's
/// message.
fn cutie_layer_norm_params(
    key: &str,
    gamma: usize,
    beta: usize,
    rows: usize,
    cols: usize,
    eps: f32,
) -> MetalResult<CutieLayerNorm> {
    let op = "gpu_cutie_layer_norm";
    if gamma != cols || beta != cols || cols % 4 != 0 {
        return Err(format!("{op} {key}: {gamma} and {beta} affine values for {cols} columns"));
    }
    layer_norm_params(op, rows, cols, eps)
}

/// `gpu_layer_norm_pytorch`'s check, with CUDA's message.
fn layer_norm_pytorch_params(scale: usize, bias: usize, rows: usize, cols: usize, eps: f32) -> MetalResult<CutieLayerNorm> {
    if scale != cols || bias != cols || cols % 4 != 0 {
        return Err("gpu_layer_norm_pytorch width mismatch".to_string());
    }
    layer_norm_params("gpu_layer_norm_pytorch", rows, cols, eps)
}

fn layer_norm_params(op: &str, rows: usize, cols: usize, eps: f32) -> MetalResult<CutieLayerNorm> {
    cutie_u32(op, rows)?;
    let cols_u = cutie_u32(op, cols)?;
    cutie_count(op, &[rows, cols])?;
    Ok(CutieLayerNorm { cols: cols_u, eps })
}

/// A threadgroup of [32, 4] a row, as PyTorch's launch; nothing when there are no elements.
fn layer_norm_grid(rows: usize, cols: usize) -> MetalResult<Grid> {
    let groups = if cols == 0 { 0 } else { rows };
    Grid::groups([groups, 1, 1], LAYER_NORM_THREADS)
}

// --- Sine encoding ---

/// `freqs` frequencies at `w x h`, with CUDA's check and message.
fn sine_encoding_params(freqs: usize, w: usize, h: usize, scale: f32, eps: f32) -> MetalResult<CutieSineEncoding> {
    let op = "gpu_cutie_sine_encoding";
    if freqs == 0 || w == 0 || h == 0 {
        return Err(format!("{op}: {freqs} frequencies at {w}x{h}"));
    }
    let d = |v: usize| cutie_u32(op, v);
    let (freqs_u, w_u, h_u) = (d(freqs)?, d(w)?, d(h)?);
    let n = cutie_count(op, &[w, h, 4, freqs])?;
    Ok(CutieSineEncoding { freqs: freqs_u, w: w_u, h: h_u, scale, eps, n })
}

// --- The ops (macOS) ---

#[cfg(target_os = "macos")]
fn cutie_check_f32(op: &str, x: &GpuTensor) -> MetalResult<()> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn shape(t: &GpuTensor) -> (usize, usize) {
    (t.rows(), t.cols())
}

/// `Err` unless `kernel`'s SIMD-groups are 32 wide: its butterfly or shuffle-down tree is CUDA's
/// over a warp's 32 lanes, and another width would sum in another order (or miss lanes).
#[cfg(target_os = "macos")]
fn require_warp_width(s: &mut MetalStream<'_>, op: &str, kernel: &Kernel) -> MetalResult<()> {
    let width = s.limits(kernel, &[])?.simd_width;
    if width != WARP {
        return Err(format!(
            "{op} ({}): the kernel reduces over CUDA's {WARP}-lane warps; this GPU's SIMD-groups are {width} wide",
            kernel.function
        ));
    }
    Ok(())
}

/// Softmax down each column of `x` [m, p]: the max, exp(x - max) summed in row order, divided, as
/// PyTorch's spatial softmax does. Over more than 64 rows and fewer than 8192 columns
/// (consolidation's candidates by prototypes) a threadgroup a tile of columns sums in a fixed
/// tree: neither that nor PyTorch's order there is sequential, so it is fp32 rounding apart.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_softmax_axis0(x: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_softmax_axis0";
    cutie_check_f32(op, x)?;
    let params = softmax_axis0_params(x.rows(), x.cols())?;
    let (kernel, grid) = softmax_axis0_launch(&params).map_err(|e| format!("{op}: {e}"))?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        s.launch(kernel, &[], &params, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// Cutie's `aggregate`: objects' probabilities `prob` [n, p] to logits with a background row first
/// ([n+1, p]): the background is prod(1 - prob) (in PyTorch's order, `cutie_reduce4`), every row
/// clamped to 1e-7..1-1e-7, then log(q / (1 - q)).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_aggregate(prob: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_aggregate";
    cutie_check_f32(op, prob)?;
    let params = aggregate_params(prob.rows(), prob.cols())?;
    let grid = Grid::linear(params.pixels as usize, THREADS).map_err(|e| format!("{op}: {e}"))?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(prob.rows() + 1, prob.cols())?;
        s.launch(&CUTIE_AGGREGATE, &[], &params, &[Bind::In(prob), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// The tail of Cutie's channel-attention block (ECA): per image b of `x` [n*c, p], the mean of
/// each channel, a 1-D convolution over channels (`conv_w`, odd length k, zero padding (k-1)/2,
/// no bias), a sigmoid, then x * that + `residual` ([n*c, p]). More than [`CUTIE_ECA_MAX_TAPS`]
/// taps is `Err` (CUDA runs the host reference for those).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_eca(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_eca";
    cutie_check_f32(op, x)?;
    cutie_check_f32(op, residual)?;
    let plan = eca_params(shape(x), shape(residual), n, c, conv_w)?;
    let [mean_grid, scale_grid, apply_grid] = plan.grids().map_err(|e| format!("{op}: {e}"))?;
    with_metal_stream(op, |s| {
        require_warp_width(s, op, &CUTIE_ECA_ROW_MEAN)?;
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        let means = GpuTensor::alloc(plan.rows() as usize, 1)?;
        let scale = GpuTensor::alloc(plan.rows() as usize, 1)?;
        s.launch(&CUTIE_ECA_ROW_MEAN, &[], &plan.row_mean, &[Bind::In(x), Bind::Out(&means)], mean_grid)?;
        s.launch(&CUTIE_ECA_SCALE, &[], &plan.scale, &[Bind::In(&means), Bind::Out(&scale)], scale_grid)?;
        let binds = [Bind::In(x), Bind::In(residual), Bind::In(&scale), Bind::Out(&out)];
        s.launch(&CUTIE_ECA_APPLY, &[], &plan.apply, &binds, apply_grid)?;
        Ok(out)
    })
}

/// One launch of the layer norm kernel over `x` into `out`, gamma and beta bound as given.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn launch_layer_norm(
    s: &mut MetalStream<'_>,
    op: &str,
    x: &GpuTensor,
    gamma: Bind<'_>,
    beta: Bind<'_>,
    out: &GpuTensor,
    params: &CutieLayerNorm,
    grid: Grid,
) -> MetalResult<()> {
    require_warp_width(s, op, &CUTIE_LAYER_NORM)?;
    s.launch(&CUTIE_LAYER_NORM, &[], params, &[Bind::In(x), gamma, beta, Bind::Out(out)], grid)
}

/// nn.LayerNorm over each row of `x` [rows, cols] (cols % 4 == 0), the kernel of
/// `gpu_layer_norm_pytorch` (PyTorch's CUDA order), with `gamma` and `beta` kept on the device
/// under `{cache_namespace}::{key}::cutie_ln` (gamma, then beta) instead of uploaded on every call.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_layer_norm(
    x: &GpuTensor,
    cache_namespace: &str,
    key: &str,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_layer_norm";
    cutie_check_f32(op, x)?;
    let params = cutie_layer_norm_params(key, gamma.len(), beta.len(), x.rows(), x.cols(), eps)?;
    let grid = layer_norm_grid(x.rows(), x.cols()).map_err(|e| format!("{op}: {e}"))?;
    let cols = x.cols();
    with_metal_stream(op, |s| {
        let qualified_key = format!("{cache_namespace}::{key}::cutie_ln");
        let affine = s.weights(&qualified_key, 2 * cols, |dst| {
            let (gamma_dst, beta_dst) = dst.split_at_mut(cols);
            gamma_dst.copy_from_slice(gamma);
            beta_dst.copy_from_slice(beta);
            Ok(())
        })?;
        let out = GpuTensor::alloc(x.rows(), cols)?;
        launch_layer_norm(s, op, x, Bind::Weights(&affine, 0), Bind::Weights(&affine, cols), &out, &params, grid)?;
        Ok(out)
    })
}

/// nn.LayerNorm over each row of `x` [rows, cols] (cols % 4 == 0) as PyTorch's CUDA fast path
/// computes it: float4 steps of online Welford moments, its fixed tree over four 32-lane groups,
/// and gamma * (rstd * (x - mean)) + beta; CUDA's `gpu_layer_norm_pytorch`, on the kernel
/// `gpu_cutie_layer_norm` runs, the affine uploaded on every call.
#[cfg(target_os = "macos")]
pub fn gpu_layer_norm_pytorch(x: &GpuTensor, scale: &[f32], bias: &[f32], eps: f32) -> Result<GpuTensor, String> {
    let op = "gpu_layer_norm_pytorch";
    let params = layer_norm_pytorch_params(scale.len(), bias.len(), x.rows(), x.cols(), eps)?;
    let grid = layer_norm_grid(x.rows(), x.cols()).map_err(|e| format!("{op}: {e}"))?;
    let gamma = upload(scale, 1, x.cols())?;
    let beta = upload(bias, 1, x.cols())?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        launch_layer_norm(s, op, x, Bind::In(&gamma), Bind::In(&beta), &out, &params, grid)?;
        Ok(out)
    })
}

/// Cutie's PositionalEncoding (channel-last, normalised) at `w x h` with the frequencies
/// `inv_freq` (F of them): tokens [h*w, 4F], token y*w + x, the x encoding in channels 0..2F and
/// the y encoding in 2F..4F, sin and cos interleaved per frequency. Each axis' positions are
/// i / ((len - 1) + `eps`) * `scale`, times each frequency, every op rounded as PyTorch's tensor
/// ops round it; then precise::sin and cos where CUDA calls its own sinf and cosf (which are what
/// torch.sin and torch.cos give on CUDA, and are not correctly rounded: fp32 rounding apart).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_sine_encoding(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_sine_encoding";
    let params = sine_encoding_params(inv_freq.len(), w, h, scale, eps)?;
    let grid = Grid::linear(params.n as usize, THREADS).map_err(|e| format!("{op}: {e}"))?;
    let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
    with_metal_stream(op, |s| {
        let freqs = s.upload_bytes(&bytes)?;
        let out = GpuTensor::alloc(w * h, 4 * inv_freq.len())?;
        s.launch(&CUTIE_SINE_ENCODING, &[], &params, &[Bind::In(&freqs), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// The checks, params and grids on any OS: CUDA's messages and launches, Cutie's shapes, each
/// params struct field for field its MSL twin, and the kernels' reduction trees run on the host
/// as the MSL writes them, against sums in f64.
#[cfg(test)]
mod tests {
    use super::*;

    /// The fields of `struct name { ... };` in the library's source, as (type, name).
    fn msl_struct(name: &str) -> Vec<(String, String)> {
        let source = CUTIE_SOFTMAX_NORM.source;
        let start = source.find(&format!("struct {name} {{")).unwrap_or_else(|| panic!("no struct {name} in the source"));
        let body = &source[start..];
        let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
        body.lines()
            .map(|line| line.split("//").next().unwrap().trim())
            .filter(|line| !line.is_empty())
            .map(|line| {
                let line = line.strip_suffix(';').unwrap_or_else(|| panic!("{name}: odd field line {line:?}"));
                let (ty, field) = line.split_once(' ').unwrap_or_else(|| panic!("{name}: odd field line {line:?}"));
                (ty.trim().to_string(), field.trim().to_string())
            })
            .collect()
    }

    fn fields(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter().map(|&(ty, name)| (ty.to_string(), name.to_string())).collect()
    }

    #[test]
    fn every_kernel_is_in_the_library_and_the_library_includes_the_header() {
        let source = CUTIE_SOFTMAX_NORM.source;
        for kernel in SOFTMAX_NORM_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &CUTIE_SOFTMAX_NORM));
            assert!(source.contains(&format!("kernel void {}(", kernel.function)), "{} is not declared", kernel.function);
        }
        assert_eq!(source.matches("kernel void ").count(), SOFTMAX_NORM_KERNELS.len(), "a kernel the list misses");
        assert!(source.lines().any(|line| line.trim() == "#include \"makepad_common.h\""));
        let half = source.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).any(|word| word.starts_with("half"));
        assert!(!half, "no half anywhere (D8)");
        // CUDA's tile and taps, as the MSL defines them.
        for define in [
            format!("#define CUTIE_SOFTMAX_TILE_COLS {SOFTMAX_TILE_COLS}"),
            format!("#define CUTIE_SOFTMAX_TILE_ROWS {SOFTMAX_TILE_ROWS}"),
            format!("#define CUTIE_ECA_MAX_TAPS {CUTIE_ECA_MAX_TAPS}"),
        ] {
            assert!(source.lines().any(|line| line.trim() == define), "no `{define}` in the source");
        }
    }

    #[test]
    fn params_are_their_msl_twins_field_for_field() {
        let twins: [(&str, &[&Kernel], Vec<(String, String)>, usize, usize); 7] = [
            (
                "CutieSoftmaxAxis0",
                &[&CUTIE_SOFTMAX_AXIS0, &CUTIE_SOFTMAX_AXIS0_TILE],
                fields(&[("uint", "rows"), ("uint", "cols")]),
                size_of::<CutieSoftmaxAxis0>(),
                8,
            ),
            ("CutieAggregate", &[&CUTIE_AGGREGATE], fields(&[("uint", "objects"), ("uint", "pixels")]), size_of::<CutieAggregate>(), 8),
            ("CutieEcaRowMean", &[&CUTIE_ECA_ROW_MEAN], fields(&[("uint", "plane")]), size_of::<CutieEcaRowMean>(), 4),
            (
                "CutieEcaScale",
                &[&CUTIE_ECA_SCALE],
                fields(&[("float", "taps[CUTIE_ECA_MAX_TAPS]"), ("uint", "k"), ("uint", "channels"), ("uint", "rows")]),
                size_of::<CutieEcaScale>(),
                4 * CUTIE_ECA_MAX_TAPS + 12,
            ),
            ("CutieEcaApply", &[&CUTIE_ECA_APPLY], fields(&[("uint", "plane"), ("uint", "n")]), size_of::<CutieEcaApply>(), 8),
            ("CutieLayerNorm", &[&CUTIE_LAYER_NORM], fields(&[("uint", "cols"), ("float", "eps")]), size_of::<CutieLayerNorm>(), 8),
            (
                "CutieSineEncoding",
                &[&CUTIE_SINE_ENCODING],
                fields(&[
                    ("uint", "freqs"),
                    ("uint", "w"),
                    ("uint", "h"),
                    ("float", "scale"),
                    ("float", "eps"),
                    ("uint", "n"),
                ]),
                size_of::<CutieSineEncoding>(),
                24,
            ),
        ];
        let mut kernels = 0;
        for (name, users, want, bytes, msl_bytes) in twins {
            assert_eq!(msl_struct(name), want, "{name}: the MSL struct is not the Rust struct's twin");
            assert_eq!(bytes, msl_bytes, "{name}: size");
            for kernel in users {
                // The kernel reads this struct at [[buffer(0)]].
                let signature = format!("kernel void {}(constant {name}& p [[buffer(0)]]", kernel.function);
                assert!(CUTIE_SOFTMAX_NORM.source.contains(&signature), "no `{signature}` in the source");
                kernels += 1;
            }
        }
        assert_eq!(kernels, SOFTMAX_NORM_KERNELS.len());
        // The Rust fields in the same order (Debug prints them in declaration order).
        let debug = format!("{:?}", CutieSineEncoding { freqs: 1, w: 2, h: 3, scale: 4.0, eps: 5.0, n: 6 });
        assert_eq!(debug, "CutieSineEncoding { freqs: 1, w: 2, h: 3, scale: 4.0, eps: 5.0, n: 6 }");
        let debug = format!("{:?}", CutieLayerNorm { cols: 1, eps: 2.0 });
        assert_eq!(debug, "CutieLayerNorm { cols: 1, eps: 2.0 }");
        assert_eq!(std::mem::offset_of!(CutieEcaScale, k), 4 * CUTIE_ECA_MAX_TAPS);
        assert_eq!(std::mem::offset_of!(CutieEcaScale, rows), 4 * CUTIE_ECA_MAX_TAPS + 8);
    }

    #[test]
    fn shape_errors_are_cudas() {
        let eca = |x, r, n, c, k: usize| eca_params(x, r, n, c, &vec![0.5; k]);
        let mismatch = "gpu_cutie_eca shape mismatch";
        assert_eq!(eca((6, 10), (6, 10), 2, 4, 3).unwrap_err(), mismatch, "rows are not n * c");
        assert_eq!(eca((8, 10), (7, 10), 2, 4, 3).unwrap_err(), mismatch);
        assert_eq!(eca((8, 10), (8, 9), 2, 4, 3).unwrap_err(), mismatch);
        assert_eq!(eca((8, 10), (8, 10), 2, 4, 4).unwrap_err(), mismatch, "an even number of taps");
        assert_eq!(eca((8, 10), (8, 10), 2, 4, 0).unwrap_err(), mismatch, "no taps");
        assert_eq!(eca((0, 10), (0, 10), usize::MAX, 2, 3).unwrap_err(), mismatch, "n * c overflows");
        assert_eq!(
            eca((8, 10), (8, 10), 2, 4, 17).unwrap_err(),
            "gpu_cutie_eca: 17 taps, the kernel takes at most 15 by value (CUDA runs the host reference for more)"
        );
        // The shape is checked before the taps' count, as on CUDA.
        assert_eq!(eca((6, 10), (6, 10), 2, 4, 17).unwrap_err(), mismatch);
        assert_eq!(eca((8, 10), (8, 10), 2, 4, 15).unwrap().scale.k, 15);

        assert_eq!(
            cutie_layer_norm_params("ln", 255, 256, 3, 256, 1e-5).unwrap_err(),
            "gpu_cutie_layer_norm ln: 255 and 256 affine values for 256 columns"
        );
        assert_eq!(
            cutie_layer_norm_params("ln", 256, 255, 3, 256, 1e-5).unwrap_err(),
            "gpu_cutie_layer_norm ln: 256 and 255 affine values for 256 columns"
        );
        assert_eq!(
            cutie_layer_norm_params("q norm", 6, 6, 3, 6, 1e-5).unwrap_err(),
            "gpu_cutie_layer_norm q norm: 6 and 6 affine values for 6 columns"
        );
        let width = "gpu_layer_norm_pytorch width mismatch";
        assert_eq!(layer_norm_pytorch_params(255, 256, 3, 256, 1e-5).unwrap_err(), width);
        assert_eq!(layer_norm_pytorch_params(256, 255, 3, 256, 1e-5).unwrap_err(), width);
        assert_eq!(layer_norm_pytorch_params(6, 6, 3, 6, 1e-5).unwrap_err(), width);

        let sine = "gpu_cutie_sine_encoding";
        assert_eq!(sine_encoding_params(0, 24, 14, 32.0, 1e-6).unwrap_err(), format!("{sine}: 0 frequencies at 24x14"));
        assert_eq!(sine_encoding_params(64, 0, 14, 32.0, 1e-6).unwrap_err(), format!("{sine}: 64 frequencies at 0x14"));
        assert_eq!(sine_encoding_params(64, 24, 0, 32.0, 1e-6).unwrap_err(), format!("{sine}: 64 frequencies at 24x0"));
    }

    #[test]
    fn the_softmax_takes_cudas_kernel_and_grid() {
        const FRAME: usize = 384 * 224;
        // The decoder's softmax over background and objects: a thread a column, 128 to a group.
        let params = softmax_axis0_params(4, FRAME).unwrap();
        assert_eq!(params, CutieSoftmaxAxis0 { rows: 4, cols: FRAME as u32 });
        let (kernel, grid) = softmax_axis0_launch(&params).unwrap();
        assert_eq!(kernel.function, CUTIE_SOFTMAX_AXIS0.function);
        assert_eq!((grid.groups, grid.threads), ([FRAME.div_ceil(128) as u32, 1, 1], [128, 1, 1]));
        // Consolidation's candidates by 128 prototypes: tiles of 8 columns, 128 thread rows.
        for m in [1344, 5 * 112 * 63, 65] {
            let (kernel, grid) = softmax_axis0_launch(&softmax_axis0_params(m, 128).unwrap()).unwrap();
            assert_eq!(kernel.function, CUTIE_SOFTMAX_AXIS0_TILE.function, "m{m}");
            assert_eq!((grid.groups, grid.threads), ([16, 1, 1], [8, 128, 1]), "m{m}");
        }
        // CUDA's bounds: more than 64 rows, fewer than 8192 columns.
        let kernel_for = |m, p| softmax_axis0_launch(&softmax_axis0_params(m, p).unwrap()).unwrap().0.function;
        assert_eq!(kernel_for(64, 128), CUTIE_SOFTMAX_AXIS0.function);
        assert_eq!(kernel_for(65, 8191), CUTIE_SOFTMAX_AXIS0_TILE.function);
        assert_eq!(kernel_for(65, 8192), CUTIE_SOFTMAX_AXIS0.function);
        assert_eq!(kernel_for(1, 7), CUTIE_SOFTMAX_AXIS0.function);
        let (_, odd) = softmax_axis0_launch(&softmax_axis0_params(100, 13).unwrap()).unwrap();
        assert_eq!(odd.groups, [2, 1, 1], "a tile partly past the columns");
    }

    #[test]
    fn params_at_cuties_shapes() {
        // Stride 16 of the padded 384x224 frame, stride 4, and the frame itself.
        const P16: usize = 24 * 14;
        const P4: usize = 96 * 56;
        const FRAME: usize = 384 * 224;
        assert_eq!(aggregate_params(3, FRAME).unwrap(), CutieAggregate { objects: 3, pixels: FRAME as u32 });
        assert_eq!(aggregate_params(9, 37).unwrap(), CutieAggregate { objects: 9, pixels: 37 });

        let plan = eca_params((3 * 256, P16), (3 * 256, P16), 3, 256, &[0.25, -0.5, 1.0, 0.5, -0.25]).unwrap();
        assert_eq!(plan.row_mean, CutieEcaRowMean { plane: P16 as u32 });
        assert_eq!(plan.apply, CutieEcaApply { plane: P16 as u32, n: (3 * 256 * P16) as u32 });
        assert_eq!((plan.scale.k, plan.scale.channels, plan.scale.rows), (5, 256, 768));
        let mut taps = [0.0f32; CUTIE_ECA_MAX_TAPS];
        taps[..5].copy_from_slice(&[0.25, -0.5, 1.0, 0.5, -0.25]);
        assert_eq!(plan.scale.taps, taps, "the taps by value, zero past k");
        let [mean, scale, apply] = plan.grids().unwrap();
        assert_eq!((mean.groups, mean.threads), ([768, 1, 1], [256, 1, 1]), "a threadgroup a row");
        assert_eq!((scale.groups, scale.threads), ([3, 1, 1], [256, 1, 1]));
        assert_eq!((apply.groups, apply.threads), ([(768 * P16).div_ceil(256) as u32, 1, 1], [256, 1, 1]));

        let ln = cutie_layer_norm_params("ln", 256, 256, P4, 256, 1e-5).unwrap();
        assert_eq!(ln, CutieLayerNorm { cols: 256, eps: 1e-5 });
        assert_eq!(layer_norm_pytorch_params(256, 256, P4, 256, 1e-5).unwrap(), ln, "one kernel, one params");
        let grid = layer_norm_grid(P4, 256).unwrap();
        assert_eq!((grid.groups, grid.threads), ([P4 as u32, 1, 1], [32, 4, 1]), "PyTorch's dim3(32, 4) a row");

        let sine = sine_encoding_params(64, 24, 14, 32.0, 1e-6).unwrap();
        assert_eq!(sine, CutieSineEncoding { freqs: 64, w: 24, h: 14, scale: 32.0, eps: 1e-6, n: (P16 * 256) as u32 });
    }

    #[test]
    fn empty_tensors_launch_nothing() {
        let (_, grid) = softmax_axis0_launch(&softmax_axis0_params(0, 7).unwrap()).unwrap();
        assert!(grid.is_empty(), "no rows");
        let (_, grid) = softmax_axis0_launch(&softmax_axis0_params(3, 0).unwrap()).unwrap();
        assert!(grid.is_empty(), "no columns");
        let (kernel, grid) = softmax_axis0_launch(&softmax_axis0_params(100, 0).unwrap()).unwrap();
        assert!(grid.is_empty() && kernel.function == CUTIE_SOFTMAX_AXIS0_TILE.function, "no columns, tiled");
        assert_eq!(aggregate_params(0, 37).unwrap(), CutieAggregate { objects: 0, pixels: 37 }, "a background alone");
        assert_eq!(aggregate_params(3, 0).unwrap().pixels, 0);
        for (x, n, c) in [((0, 10), 0, 4), ((8, 0), 2, 4), ((0, 10), 2, 0)] {
            let plan = eca_params(x, x, n, c, &[1.0, 2.0, 3.0]).unwrap();
            assert!(plan.grids().unwrap().iter().all(Grid::is_empty), "{x:?}: CUDA returns before any launch");
        }
        assert!(layer_norm_grid(0, 256).unwrap().is_empty());
        assert!(layer_norm_grid(3, 0).unwrap().is_empty());
        assert_eq!(layer_norm_pytorch_params(0, 0, 3, 0, 1e-5).unwrap().cols, 0, "no columns is 4-aligned");
    }

    #[test]
    fn sizes_past_u32_are_errors_with_cudas_message() {
        let big = 1usize << 32;
        let past = format!("{big} does not fit the kernel's 32-bit sizes");
        assert_eq!(softmax_axis0_params(big, 1).unwrap_err(), format!("gpu_cutie_softmax_axis0: {past}"));
        assert_eq!(softmax_axis0_params(1 << 16, 1 << 16).unwrap_err(), format!("gpu_cutie_softmax_axis0: {past}"));
        assert_eq!(aggregate_params(big, 1).unwrap_err(), format!("gpu_cutie_aggregate: {past}"));
        // [n + 1, p] is the output's count.
        assert!(aggregate_params(u32::MAX as usize, 1).is_err());
        assert_eq!(aggregate_params((u32::MAX - 1) as usize, 1).unwrap().objects, u32::MAX - 1);
        assert_eq!(
            eca_params((big, 1), (big, 1), 1, big, &[1.0]).unwrap_err(),
            format!("gpu_cutie_eca: {past}"),
            "c past u32"
        );
        assert_eq!(
            cutie_layer_norm_params("ln", 1 << 16, 1 << 16, 1 << 16, 1 << 16, 1e-5).unwrap_err(),
            format!("gpu_cutie_layer_norm: {past}")
        );
        assert_eq!(
            sine_encoding_params(1 << 10, 1 << 11, 1 << 9, 32.0, 1e-6).unwrap_err(),
            format!("gpu_cutie_sine_encoding: {past}")
        );
        assert_eq!(
            sine_encoding_params(usize::MAX, 2, 1, 32.0, 1e-6).unwrap_err(),
            format!("gpu_cutie_sine_encoding: {} does not fit the kernel's 32-bit sizes", usize::MAX)
        );
    }

    // --- The kernels' trees on the host, as the MSL writes them ---

    /// xorshift64*, as the ops check's.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Uniform in [lo, hi), 24 bits.
        fn uniforms(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)).collect()
        }
    }

    /// `got` within `rel` of `want`, relative to the larger of |want| and `floor`.
    fn assert_close(what: &str, got: &[f32], want: &[f64], rel: f64, floor: f64) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let tol = rel * w.abs().max(floor);
            assert!((g as f64 - w).abs() <= tol, "{what}: element {i} is {g:e}, f64 gives {w:e} (tolerance {tol:e})");
        }
    }

    /// `cutie_softmax_axis0_tile_f32` as the MSL writes it: per tile of 8 columns, each of the
    /// [8, 128] threads' strided max and sum, then the tree over thread rows through `shared`.
    fn softmax_tile_as_msl(x: &[f32], m: usize, p: usize) -> Vec<f32> {
        let (cols, rows) = (SOFTMAX_TILE_COLS as usize, SOFTMAX_TILE_ROWS as usize);
        let mut out = vec![f32::NAN; m * p];
        let tree = |shared: &mut Vec<[f32; 8]>, f: fn(f32, f32) -> f32| {
            let mut s = rows / 2;
            while s > 0 {
                for ty in 0..s {
                    for tx in 0..cols {
                        shared[ty][tx] = f(shared[ty][tx], shared[ty + s][tx]);
                    }
                }
                s >>= 1;
            }
        };
        for group in 0..p.div_ceil(cols) {
            let mut shared = vec![[0.0f32; 8]; rows];
            for ty in 0..rows {
                for tx in 0..cols {
                    let col = group * cols + tx;
                    let mut largest = f32::NEG_INFINITY;
                    if col < p {
                        for j in (ty..m).step_by(rows) {
                            largest = largest.max(x[j * p + col]);
                        }
                    }
                    shared[ty][tx] = largest;
                }
            }
            tree(&mut shared, f32::max);
            let largest = shared[0];
            for ty in 0..rows {
                for tx in 0..cols {
                    let col = group * cols + tx;
                    let mut sum = 0.0f32;
                    if col < p {
                        for j in (ty..m).step_by(rows) {
                            let e = (x[j * p + col] - largest[tx]).exp();
                            out[j * p + col] = e;
                            sum += e;
                        }
                    }
                    shared[ty][tx] = sum;
                }
            }
            tree(&mut shared, |a, b| a + b);
            for tx in 0..cols {
                let col = group * cols + tx;
                if col < p {
                    for j in 0..m {
                        out[j * p + col] /= shared[0][tx];
                    }
                }
            }
        }
        out
    }

    /// Softmax down columns with the sum in f64 (the reference's way).
    fn softmax_f64(x: &[f32], m: usize, p: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * p];
        for col in 0..p {
            let largest = (0..m).map(|j| x[j * p + col]).fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = (0..m).map(|j| (x[j * p + col] as f64 - largest).exp()).sum();
            for j in 0..m {
                out[j * p + col] = (x[j * p + col] as f64 - largest).exp() / sum;
            }
        }
        out
    }

    #[test]
    fn the_softmax_tiles_tree_is_a_softmax() {
        let mut rng = Rng(0x5eed_c001);
        // Consolidation's shape, rows not a multiple of the thread rows, a tile past the columns,
        // and a few -inf entries (never a whole column).
        for (m, p) in [(1344, 16), (65, 13), (300, 9)] {
            let mut x = rng.uniforms(m * p, -25.0, 0.0);
            for i in (0..m * p).step_by(97) {
                x[i] = f32::NEG_INFINITY;
            }
            assert_close(&format!("tile m{m} p{p}"), &softmax_tile_as_msl(&x, m, p), &softmax_f64(&x, m, p), 1e-5, 1e-30);
        }
    }

    /// `cutie_eca_row_mean_f32` for one row as the MSL writes it: 256 threads' strided sums, the
    /// xor butterfly in each 32-lane group, the groups' sums butterflied in the first group.
    fn row_mean_as_msl(row: &[f32]) -> f32 {
        let threads = THREADS as usize;
        let butterfly = |lanes: &mut [f32]| {
            let mut offset = 16;
            while offset > 0 {
                let before = lanes.to_vec();
                for lane in 0..32 {
                    lanes[lane] = before[lane] + before[lane ^ offset];
                }
                offset >>= 1;
            }
        };
        let mut sums: Vec<f32> = (0..threads).map(|t| row.iter().skip(t).step_by(threads).fold(0.0f32, |s, &v| s + v)).collect();
        for warp in sums.chunks_mut(32) {
            butterfly(warp);
        }
        let mut first: Vec<f32> = (0..32).map(|lane| if lane < threads.div_ceil(32) { sums[lane * 32] } else { 0.0 }).collect();
        butterfly(&mut first);
        first[0] / row.len() as f32
    }

    #[test]
    fn the_row_means_butterfly_is_a_mean() {
        let mut rng = Rng(0x5eed_c002);
        for plane in [1, 9, 31, 256, 257, 336, 1000, 5376] {
            let row: Vec<f32> = rng.uniforms(plane, -1.0, 1.0).iter().map(|v| v + 3.0).collect();
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / plane as f64;
            assert_close(&format!("row mean p{plane}"), &[row_mean_as_msl(&row)], &[mean], 1e-6, 1.0);
        }
    }

    /// One thread's moments, as `softmax_norm::Welford`.
    #[derive(Clone, Copy, Debug)]
    struct Welford {
        mean: f32,
        sigma2: f32,
        count: f32,
    }

    const NO_MOMENTS: Welford = Welford { mean: 0.0, sigma2: 0.0, count: 0.0 };

    /// `welford_online`: `mul_add` where the MSL calls fma().
    fn welford_online(value: f32, current: Welford) -> Welford {
        let delta = value - current.mean;
        let new_count = current.count + 1.0;
        let new_mean = delta.mul_add(1.0 / new_count, current.mean);
        Welford { mean: new_mean, sigma2: delta.mul_add(value - new_mean, current.sigma2), count: new_count }
    }

    /// `welford_combine`, `b` the thread's own.
    fn welford_combine(b: Welford, a: Welford) -> Welford {
        let delta = b.mean - a.mean;
        let count = a.count + b.count;
        if count <= 0.0 {
            return NO_MOMENTS;
        }
        let coefficient = 1.0 / count;
        let (n_a, n_b) = (a.count * coefficient, b.count * coefficient);
        Welford {
            mean: n_a.mul_add(a.mean, n_b * b.mean),
            sigma2: (delta * delta * a.count).mul_add(n_b, a.sigma2 + b.sigma2),
            count,
        }
    }

    /// `cutie_layer_norm_f32` for one row as the MSL writes it, thread by thread over [32, 4]:
    /// each thread's moments over its four-value steps, simd_shuffle_down's tree in each group
    /// (lanes past the group keep their own), the compact tree through `stats`, then the affine.
    /// rsqrt is 1 / sqrt here, so the mean and variance are the kernel's and the output within an
    /// ulp or two of it.
    fn layer_norm_as_msl(row: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> (f32, f32, Vec<f32>) {
        let [width, groups, _] = LAYER_NORM_THREADS.map(|t| t as usize);
        let threads = width * groups;
        let vectors = row.len() / 4;
        let mut wd = vec![NO_MOMENTS; threads];
        for (t, moments) in wd.iter_mut().enumerate() {
            for index in (t..vectors).step_by(threads) {
                for k in 0..4 {
                    *moments = welford_online(row[4 * index + k], *moments);
                }
            }
        }
        let mut offset = 16;
        while offset > 0 {
            let before = wd.clone();
            for warp in 0..groups {
                for lane in 0..width {
                    let other = if lane + offset < width { before[warp * width + lane + offset] } else { before[warp * width + lane] };
                    wd[warp * width + lane] = welford_combine(before[warp * width + lane], other);
                }
            }
            offset >>= 1;
        }
        let mut stats = [0.0f32; 6];
        let mut offset = groups / 2;
        while offset > 0 {
            for warp in offset..2 * offset {
                let write_y = warp - offset;
                stats[2 * write_y] = wd[warp * width].mean;
                stats[2 * write_y + 1] = wd[warp * width].sigma2;
                stats[groups + write_y] = wd[warp * width].count;
            }
            for warp in 0..offset {
                let other = Welford { mean: stats[2 * warp], sigma2: stats[2 * warp + 1], count: stats[groups + warp] };
                wd[warp * width] = welford_combine(wd[warp * width], other);
            }
            offset >>= 1;
        }
        let (mean, var) = (wd[0].mean, wd[0].sigma2 / row.len() as f32);
        let rstd = 1.0 / (var + eps).sqrt();
        let out = row.iter().zip(gamma.iter().zip(beta)).map(|(&v, (&g, &b))| g.mul_add(rstd * (v - mean), b)).collect();
        (mean, var, out)
    }

    #[test]
    fn the_layer_norms_welford_tree_is_a_layer_norm() {
        let mut rng = Rng(0x5eed_c003);
        // The object transformer's 256 channels, fewer vectors than threads, more vectors than
        // threads (each thread two steps), and one vector.
        for (cols, offset) in [(256, 2.0f32), (64, -5.0), (1024, 0.5), (516, 30.0), (4, 1.0)] {
            let row: Vec<f32> = rng.uniforms(cols, -3.0, 3.0).iter().map(|v| v + offset).collect();
            let gamma = rng.uniforms(cols, -1.0, 1.0);
            let beta = rng.uniforms(cols, -0.5, 0.5);
            let eps = 1e-5f32;
            let (mean, var, out) = layer_norm_as_msl(&row, &gamma, &beta, eps);
            let mean64 = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var64 = row.iter().map(|&v| (v as f64 - mean64).powi(2)).sum::<f64>() / cols as f64;
            assert_close(&format!("mean c{cols}"), &[mean], &[mean64], 1e-6, 1.0);
            assert_close(&format!("variance c{cols}"), &[var], &[var64], 1e-5, 1e-6);
            let rstd = 1.0 / (var64 + eps as f64).sqrt();
            let want: Vec<f64> = (0..cols).map(|i| gamma[i] as f64 * (rstd * (row[i] as f64 - mean64)) + beta[i] as f64).collect();
            assert_close(&format!("layer norm c{cols}"), &out, &want, 1e-5, 1.0);
        }
    }
}

/// The ops on the Mac against the host, in the host binding: each against f64 (or the host's
/// libm) within fp32 rounding, the cached layer norm against `gpu_layer_norm_pytorch` bit for bit,
/// and the errors CUDA's wrappers give. `cargo test -p makepad-ai-metal --lib
/// cutie_ops::softmax_norm`, and again with MAKEPAD_METAL_MISALIGN=1.
#[cfg(all(test, target_os = "macos"))]
mod metal_tests {
    use super::*;
    use crate::stream;

    /// xorshift64*, as the ops check's.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn uniforms(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)).collect()
        }
    }

    fn up(values: &[f32], rows: usize, cols: usize) -> GpuTensor {
        stream::upload(values, rows, cols).unwrap()
    }

    fn down(t: &GpuTensor) -> Vec<f32> {
        stream::download(t).unwrap()
    }

    /// An op's error (a tensor is not Debug, so no unwrap_err).
    fn err(result: Result<GpuTensor, String>) -> String {
        match result {
            Ok(t) => panic!("expected an Err, got a {}x{} tensor", t.rows(), t.cols()),
            Err(e) => e,
        }
    }

    /// `got` within `rel` of `want`, relative to the larger of |want| and `floor`.
    fn assert_close(what: &str, got: &[f32], want: &[f64], rel: f64, floor: f64) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let tol = rel * w.abs().max(floor);
            assert!((g as f64 - w).abs() <= tol, "{what}: element {i} is {g:e}, f64 gives {w:e} (tolerance {tol:e})");
        }
    }

    fn sigmoid64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn logit64(q: f32) -> f64 {
        let q = q.clamp(1e-7, 1.0 - 1e-7) as f64;
        (q / (1.0 - q)).ln()
    }

    #[test]
    fn every_kernel_compiles_with_32_lane_simd_groups() {
        assert!(stream::library_available(&CUTIE_SOFTMAX_NORM), "cutie_softmax_norm does not compile");
        with_metal_stream("cutie softmax norm: pipelines", |s| {
            for kernel in SOFTMAX_NORM_KERNELS {
                s.limits(kernel, &[])?;
            }
            require_warp_width(s, "cutie softmax norm", &CUTIE_ECA_ROW_MEAN)?;
            require_warp_width(s, "cutie softmax norm", &CUTIE_LAYER_NORM)?;
            let tile = s.limits(&CUTIE_SOFTMAX_AXIS0_TILE, &[])?;
            assert!(tile.max_threads >= SOFTMAX_TILE_COLS * SOFTMAX_TILE_ROWS, "the tile's 1024 threads: {tile:?}");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn softmax_down_columns_on_both_kernels() {
        let mut rng = Rng(0x5eed_c101);
        // The decoder's few rows (a thread a column), consolidation's many (tiles), one row.
        for (m, p) in [(4, 1000), (2, 8200), (1344, 128), (65, 13), (1, 7)] {
            let mut v = rng.uniforms(m * p, -25.0, 5.0);
            if m > 1 {
                for i in (0..p).step_by(97) {
                    v[i] = f32::NEG_INFINITY;
                }
            }
            let got = down(&gpu_cutie_softmax_axis0(&up(&v, m, p)).unwrap());
            let mut want = vec![0.0f64; m * p];
            for col in 0..p {
                let largest = (0..m).map(|j| v[j * p + col]).fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = (0..m).map(|j| (v[j * p + col] as f64 - largest).exp()).sum();
                for j in 0..m {
                    want[j * p + col] = (v[j * p + col] as f64 - largest).exp() / sum;
                }
            }
            assert_close(&format!("softmax m{m} p{p}"), &got, &want, 1e-5, 1e-30);
        }
        let empty = gpu_cutie_softmax_axis0(&up(&[], 0, 5)).unwrap();
        assert_eq!((empty.rows(), empty.cols()), (0, 5));
    }

    #[test]
    fn aggregate_against_the_host() {
        let mut rng = Rng(0x5eed_c102);
        // 5 objects on, PyTorch's product over them is no longer sequential; 0 is a background
        // alone; probabilities at and past the clamp.
        for (n, p) in [(0, 9), (1, 37), (3, 1000), (5, 300), (9, 37)] {
            let mut prob = rng.uniforms(n * p, 0.0, 1.0);
            for (i, q) in prob.iter_mut().enumerate() {
                match i % 13 {
                    0 => *q = 0.0,
                    1 => *q = 1.0,
                    _ => {}
                }
            }
            let got = down(&gpu_cutie_aggregate(&up(&prob, n, p)).unwrap());
            let mut want = vec![0.0f64; (n + 1) * p];
            for i in 0..p {
                let mut acc = [1.0f32; 4];
                for o in 0..n {
                    acc[o % 4] *= 1.0 - prob[o * p + i];
                }
                want[i] = logit64(acc[0] * acc[1] * acc[2] * acc[3]);
                for o in 0..n {
                    want[(o + 1) * p + i] = logit64(prob[o * p + i]);
                }
            }
            assert_close(&format!("aggregate n{n} p{p}"), &got, &want, 1e-5, 1.0);
        }
    }

    #[test]
    fn eca_against_the_host() {
        let mut rng = Rng(0x5eed_c103);
        for (n, c, p, k) in [(3, 256, 336, 5), (2, 64, 1000, 3), (2, 7, 13, 3), (1, 3, 9, 5), (1, 2, 5, 15)] {
            let mut x = rng.uniforms(n * c * p, -1.0, 1.0);
            let offsets = rng.uniforms(n * c, -1.0, 1.0);
            for (row, off) in offsets.iter().enumerate() {
                x[row * p..(row + 1) * p].iter_mut().for_each(|v| *v += off);
            }
            let r = rng.uniforms(n * c * p, -1.0, 1.0);
            let w = rng.uniforms(k, -1.0, 1.0);
            let got = down(&gpu_cutie_eca(&up(&x, n * c, p), &up(&r, n * c, p), n, c, &w).unwrap());
            let pad = (k - 1) / 2;
            let mut want = vec![0.0f64; n * c * p];
            for o in 0..n {
                let means: Vec<f64> =
                    (0..c).map(|ch| x[(o * c + ch) * p..(o * c + ch + 1) * p].iter().map(|&v| v as f64).sum::<f64>() / p as f64).collect();
                for ch in 0..c {
                    let mut y = 0.0f64;
                    for (j, &tap) in w.iter().enumerate() {
                        let src = ch as isize + j as isize - pad as isize;
                        if src >= 0 && (src as usize) < c {
                            y += tap as f64 * means[src as usize];
                        }
                    }
                    let s = sigmoid64(y);
                    for i in 0..p {
                        let at = (o * c + ch) * p + i;
                        want[at] = x[at] as f64 * s + r[at] as f64;
                    }
                }
            }
            assert_close(&format!("eca n{n} c{c} p{p} k{k}"), &got, &want, 1e-5, 1.0);
        }
        let x = up(&[1.0; 8], 2, 4);
        let taps = err(gpu_cutie_eca(&x, &x, 1, 2, &[0.5; 17]));
        assert!(taps.contains("17 taps"), "{taps}");
        assert_eq!(err(gpu_cutie_eca(&x, &x, 1, 2, &[0.5; 4])), "gpu_cutie_eca shape mismatch");
    }

    #[test]
    fn layer_norm_cached_is_layer_norm_pytorch_bit_for_bit() {
        let mut rng = Rng(0x5eed_c104);
        for (rows, cols) in [(16, 256), (336, 256), (7, 516), (3, 4), (2, 1024)] {
            let x = rng.uniforms(rows * cols, -3.0, 3.0);
            let gamma = rng.uniforms(cols, -1.0, 1.0);
            let beta = rng.uniforms(cols, -0.5, 0.5);
            let key = format!("softmax_norm test {rows}x{cols}");
            let xt = up(&x, rows, cols);
            let cached = down(&gpu_cutie_layer_norm(&xt, "metal_tests", &key, &gamma, &beta, 1e-5).unwrap());
            let again = down(&gpu_cutie_layer_norm(&xt, "metal_tests", &key, &gamma, &beta, 1e-5).unwrap());
            let plain = down(&gpu_layer_norm_pytorch(&xt, &gamma, &beta, 1e-5).unwrap());
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&cached), bits(&plain), "{rows}x{cols}: cached against uploaded affine");
            assert_eq!(bits(&again), bits(&plain), "{rows}x{cols}: from the cache");
            let mut want = vec![0.0f64; rows * cols];
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
                let rstd = 1.0 / (var + 1e-5).sqrt();
                for i in 0..cols {
                    want[r * cols + i] = gamma[i] as f64 * (rstd * (row[i] as f64 - mean)) + beta[i] as f64;
                }
            }
            assert_close(&format!("layer norm {rows}x{cols}"), &cached, &want, 1e-5, 1.0);
        }
        stream::evict_weights("metal_tests::softmax_norm test", false).unwrap();
        let x = up(&[1.0; 12], 2, 6);
        assert_eq!(
            err(gpu_cutie_layer_norm(&x, "metal_tests", "odd", &[1.0; 6], &[0.0; 6], 1e-5)),
            "gpu_cutie_layer_norm odd: 6 and 6 affine values for 6 columns"
        );
        assert_eq!(err(gpu_layer_norm_pytorch(&x, &[1.0; 6], &[0.0; 6], 1e-5)), "gpu_layer_norm_pytorch width mismatch");
    }

    #[test]
    fn sine_encoding_against_f64() {
        // Cutie's 64 frequencies at stride 16 of the padded frame and of the oracle's 60x34.
        let inv_freq: Vec<f32> = (0..64).map(|i| 1.0 / 10000f32.powf(i as f32 / 64.0)).collect();
        for (w, h) in [(24, 14), (60, 34), (1, 1), (3, 1)] {
            let got = down(&gpu_cutie_sine_encoding(&inv_freq, w, h, 32.0, 1e-6).unwrap());
            let axis = |len: usize| -> Vec<f64> {
                let last = (len - 1) as f32 + 1e-6;
                (0..len)
                    .flat_map(|i| {
                        let pos = i as f32 / last * 32.0;
                        inv_freq.iter().flat_map(move |&f| {
                            let angle = (pos * f) as f64;
                            [angle.sin(), angle.cos()]
                        })
                    })
                    .collect()
            };
            let (ex, ey) = (axis(w), axis(h));
            let mut want = Vec::with_capacity(w * h * 256);
            for y in 0..h {
                for x in 0..w {
                    want.extend_from_slice(&ex[x * 128..(x + 1) * 128]);
                    want.extend_from_slice(&ey[y * 128..(y + 1) * 128]);
                }
            }
            // Values in [-1, 1]: a few ulps of 1 apart from sin and cos of the same fp32 angle.
            assert_close(&format!("sine {w}x{h}"), &got, &want, 4.0 * f32::EPSILON as f64, 1.0);
        }
        assert_eq!(
            err(gpu_cutie_sine_encoding(&[], 24, 14, 32.0, 1e-6)),
            "gpu_cutie_sine_encoding: 0 frequencies at 24x14"
        );
    }
}
