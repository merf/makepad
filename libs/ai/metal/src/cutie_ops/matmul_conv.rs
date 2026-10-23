//! Cutie's matrix products, memory reading and convolutions on the Metal stream (family (d) of
//! RationalVisions docs/cutie-metal.md, steps 8 and 10): `gpu_cutie_matmul`,
//! `gpu_cutie_similarity`, `gpu_cutie_conv2d` and `gpu_cutie_conv2d_fused`, at the signatures of
//! CUDA's (`cuda/src/cutie_ops.rs`) with its checks, messages, weight keys and choices, and
//! `MetalStream::gemm` (fp32 multiply-accumulate, never half-staged) where CUDA calls cuBLAS. Their
//! kernels are in `shaders/makepad/cutie_matmul_conv.metal`, ported from `cuda/kernels/cutie.cu`.
//!
//! A convolution runs as CUDA's does: the images' f32 columns (im2col), a GEMM with the [out,
//! in*k*k] weights, then the bias, a residual and a ReLU in one pass. Several small images (stride 8
//! and 16) go side by side through one GEMM and that pass puts them back in place; otherwise a
//! strided-batched GEMM over the images, in groups when their columns outgrow 256 MB, and a 1x1
//! stride-1 unpadded convolution reads its input as its columns. The weights then the bias are
//! cached on the device as one buffer under `{namespace}::{key}::cutie_conv`, as on CUDA.
//!
//! Every shape check and every GEMM's numbers are worked out on the host before anything is
//! encoded (`ConvPlan`, `GemmDims`), so the plans are unit-tested on any OS against a direct
//! convolution; the ops themselves run on macOS only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::{Grid, Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::gpu_types::GpuTensor;
#[cfg(target_os = "macos")]
use crate::stream::{with_metal_stream, Bind, Gemm, GemmOperand, MetalStream, MpsConv};

/// The family's kernel library (`cutie_matmul_conv.metal`).
pub static CUTIE_MATMUL_CONV: KernelLib = KernelLib {
    name: "cutie_matmul_conv",
    source: include_str!("../../shaders/makepad/cutie_matmul_conv.metal"),
};

/// A convolution's columns ([`CutieIm2col`]).
static IM2COL: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_im2col_f32" };
/// The side-by-side GEMM's result back in the images' planes, with the epilogue
/// ([`CutieConvEpilogue`]).
static UNSTACK_EPILOGUE: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_unstack_epilogue_f32" };
/// The bias, residual and ReLU in place ([`CutieConvEpilogue`]).
static CONV_EPILOGUE: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_conv_epilogue_f32" };
/// im2col, the GEMM and the epilogue in one pass ([`CutieConvGemm`]).
static CONV_GEMM: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_conv_gemm_f32" };
/// The similarity's operands before its GEMMs ([`CutieSimilarityPrep`]).
static SIMILARITY_PREP: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_similarity_prep_f32" };
/// The similarity from its GEMMs, in place ([`CutieSimilarityEpilogue`]).
static SIMILARITY_EPILOGUE: Kernel = Kernel { lib: &CUTIE_MATMUL_CONV, function: "cutie_similarity_epilogue_f32" };

/// The layout kernels around `MPSCNNConvolution` (`cutie_mps.metal`), a library of their own so a
/// fault there touches only MAKEPAD_CUTIE_CONV=mpscnn. Not among the gate's libraries: compiled
/// only when that path runs.
pub static CUTIE_MPS: KernelLib = KernelLib { name: "cutie_mps", source: include_str!("../../shaders/makepad/cutie_mps.metal") };
/// Planar tensors into an rgba32Float texture array.
pub static PLANAR_TO_IMAGE: Kernel = Kernel { lib: &CUTIE_MPS, function: "cutie_planar_to_image_f32" };
/// Back, with the residual and the ReLU.
pub static IMAGE_TO_PLANAR: Kernel = Kernel { lib: &CUTIE_MPS, function: "cutie_image_to_planar_f32" };
/// Both, for the Mac's test that compiles them.
pub static MPS_CONV_KERNELS: &[&Kernel] = &[&PLANAR_TO_IMAGE, &IMAGE_TO_PLANAR];

/// Whether convolutions run on `MPSCNNConvolution` (MAKEPAD_CUTIE_CONV=mpscnn; read once): where
/// the padding is k/2 and k is odd, which is every one of Cutie's. Off by default until it is
/// measured on the Mac.
#[cfg(target_os = "macos")]
fn conv_mpscnn_enabled() -> bool {
    if let Some(on) = CONV_MPSCNN_OVERRIDE.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MAKEPAD_CUTIE_CONV").is_ok_and(|v| v.trim() == "mpscnn"))
}

thread_local! {
    static CONV_MPSCNN_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Runs the calling thread's convolutions on `MPSCNNConvolution` or not, whatever
/// MAKEPAD_CUTIE_CONV says (`None`: the variable again), so a test compares both in one process.
pub fn set_conv_mpscnn_override(on: Option<bool>) {
    CONV_MPSCNN_OVERRIDE.with(|cell| cell.set(on));
}

/// Every kernel in [`CUTIE_MATMUL_CONV`], for a bring-up test that compiles them all.
pub static MATMUL_CONV_KERNELS: &[&Kernel] =
    &[&IM2COL, &UNSTACK_EPILOGUE, &CONV_EPILOGUE, &CONV_GEMM, &SIMILARITY_PREP, &SIMILARITY_EPILOGUE];

/// `CutieIm2col` in MSL: the columns of `images` images from `first_image` on.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CutieIm2col {
    images: u32,
    first_image: u32,
    channels: u32,
    width: u32,
    height: u32,
    ow: u32,
    oh: u32,
    k: u32,
    pad: u32,
    stride: u32,
    /// images * channels * k * k.
    rows: u32,
    side_by_side: u32,
}
// SAFETY: #[repr(C)], 48 bytes, no pointers; `CutieIm2col` in MSL.
unsafe impl KernelParams for CutieIm2col {}

/// `CutieConvEpilogue` in MSL: `rows` = images * channels output rows of `plane` pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CutieConvEpilogue {
    images: u32,
    channels: u32,
    plane: u32,
    rows: u32,
    has_bias: u32,
    has_residual: u32,
    relu: u32,
}
// SAFETY: #[repr(C)], 28 bytes, no pointers; `CutieConvEpilogue` in MSL.
unsafe impl KernelParams for CutieConvEpilogue {}

/// `CutieConvGemm` in MSL: a convolution of `images` images as one GEMM with im2col and the
/// epilogue inside it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CutieConvGemm {
    images: u32,
    channels: u32,
    width: u32,
    height: u32,
    ow: u32,
    oh: u32,
    k: u32,
    pad: u32,
    stride: u32,
    out_channels: u32,
    /// channels * k * k.
    kdim: u32,
    /// ow * oh.
    plane: u32,
    has_bias: u32,
    has_residual: u32,
    relu: u32,
    /// A step of k ([`CONV_GEMM_TILE_K`] rows) as whole channels, window rows and columns.
    step_c: u32,
    step_y: u32,
    step_x: u32,
}
// SAFETY: #[repr(C)], 72 bytes, no pointers; `CutieConvGemm` in MSL.
unsafe impl KernelParams for CutieConvGemm {}

/// The fused kernel's tile and threads (`CONV_TILE`, `CONV_THREADS`): the tiled GEMM's.
const CONV_GEMM_TILE: usize = 32;
const CONV_GEMM_THREADS: u32 = 128;
/// Its depth in k a step (`CONV_TILE_K`).
const CONV_GEMM_TILE_K: usize = 16;

/// Whether convolutions run in one pass ([`CONV_GEMM`]); MAKEPAD_CUTIE_CONV_FUSED=0 keeps the three
/// (im2col, MetalStream::gemm, the epilogue), to compare or bisect. Read once.
#[cfg(target_os = "macos")]
fn conv_gemm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MAKEPAD_CUTIE_CONV_FUSED").map_or(true, |v| v.trim() != "0"))
}

/// `CutieSimilarityPrep` in MSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CutieSimilarityPrep {
    ck: u32,
    m: u32,
    p: u32,
    has_qe: u32,
}
// SAFETY: #[repr(C)], 16 bytes, no pointers; `CutieSimilarityPrep` in MSL.
unsafe impl KernelParams for CutieSimilarityPrep {}

/// `CutieSimilarityEpilogue` in MSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieSimilarityEpilogue {
    m: u32,
    p: u32,
    inv_scale: f32,
    has_two_ab: u32,
    has_ms: u32,
}
// SAFETY: #[repr(C)], 20 bytes, no pointers; `CutieSimilarityEpilogue` in MSL.
unsafe impl KernelParams for CutieSimilarityEpilogue {}

/// Threads to a threadgroup along x (CUDA's blocks of 256).
const THREADS: u32 = 256;
/// The most threadgroups along y; the kernels loop over the rows beyond, as CUDA's grid caps y.
const MAX_GROUPS_Y: usize = 65535;

/// When a convolution's images go side by side, and how many share a GEMM.
#[derive(Clone, Copy, Debug)]
struct ConvLimits {
    /// The im2col scratch a convolution may hold at once; batches beyond it run in groups.
    scratch_bytes: usize,
    /// Batches of images this small (output pixels each) run as one GEMM, side by side.
    side_by_side_pixels: usize,
}

/// CUDA's limits (`CUTIE_CONV_SCRATCH_BYTES`, `CUTIE_CONV_SIDE_BY_SIDE_PIXELS`).
const CUTIE_CONV_LIMITS: ConvLimits = ConvLimits { scratch_bytes: 256 << 20, side_by_side_pixels: 4096 };

/// A dimension as the kernels take it.
fn cutie_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// A convolution's output size along one axis (PyTorch's floor rule).
fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

/// 1 / sqrt(ck) as PyTorch applies `x / math.sqrt(ck)` on CUDA: the reciprocal of the Python
/// float taken in f64, rounded to f32, then a multiply (`cutie_reference.rs`'s).
fn cutie_inv_sqrt(ck: usize) -> f32 {
    (1.0 / (ck as f64).sqrt()) as f32
}

#[cfg(target_os = "macos")]
fn cutie_check_f32(op: &str, x: &GpuTensor) -> MetalResult<()> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// A GEMM's error under a convolution's key, as CUDA's "{op} {key}: <cuBLAS's error>": the
/// stream's own "{op}: " taken off, so the op is named once.
fn keyed_error(op: &str, key: &str, e: &str) -> String {
    let detail = e.strip_prefix(op).and_then(|rest| rest.strip_prefix(": ")).unwrap_or(e);
    format!("{op} {key}: {detail}")
}

/// CUDA's launch over `rows` rows of `plane` values: threadgroups of 256 along the plane, a
/// threadgroup a row along y up to 65535, the kernel looping over the rows beyond.
fn rows_grid(plane: usize, rows: usize) -> MetalResult<Grid> {
    Grid::groups([plane.div_ceil(THREADS as usize), rows.min(MAX_GROUPS_Y), 1], [THREADS, 1, 1])
}

/// One operand of a planned GEMM: `GemmOperand` without its binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OperandDims {
    offset: usize,
    ld: usize,
    trans: bool,
    batch_stride: usize,
}

/// A planned GEMM, C[i] = op(A[i]) @ op(B[i]) for i < batch, every matrix row-major: `Gemm`'s
/// numbers without its bindings (alpha 1, beta 0: C is only written).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GemmDims {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a: OperandDims,
    b: OperandDims,
    c_offset: usize,
    ldc: usize,
    c_batch_stride: usize,
}

impl GemmDims {
    /// CUDA's `cutie_sgemm`: c [m, n] = op(a) @ op(b), `a_cols` and `b_cols` the stored widths
    /// (their leading dimensions), c packed.
    fn product(m: usize, n: usize, k: usize, a_cols: usize, trans_a: bool, b_cols: usize, trans_b: bool) -> GemmDims {
        GemmDims {
            m,
            n,
            k,
            batch: 1,
            a: OperandDims { offset: 0, ld: a_cols, trans: trans_a, batch_stride: 0 },
            b: OperandDims { offset: 0, ld: b_cols, trans: trans_b, batch_stride: 0 },
            c_offset: 0,
            ldc: n,
            c_batch_stride: 0,
        }
    }

    /// CUDA's `cutie_sgemm` refuses a product it would run (not empty, k not 0) past cuBLAS's
    /// int sizes; the same here, before anything is encoded. With k = 0 it zeroes C, which the
    /// stream's fill counts in u32 (CUDA's memset does not): C's elements must fit one.
    fn check(&self) -> MetalResult<()> {
        if self.m == 0 || self.n == 0 {
            return Ok(());
        }
        if self.k == 0 {
            let op = "cutie sgemm";
            let words =
                self.m.checked_mul(self.n).ok_or_else(|| format!("{op}: {}x{} does not fit the kernel's 32-bit sizes", self.m, self.n))?;
            return cutie_u32(op, words).map(|_| ());
        }
        for v in [self.n, self.m, self.k, self.b.ld, self.a.ld] {
            i32::try_from(v).map_err(|_| format!("cutie sgemm: {v} exceeds the GEMM's sizes (cuBLAS's on CUDA)"))?;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl GemmDims {
    /// The GEMM with its operands bound.
    fn bind<'a>(&self, a: Bind<'a>, b: Bind<'a>, c: Bind<'a>) -> Gemm<'a> {
        let operand = |src: Bind<'a>, d: OperandDims| GemmOperand {
            src,
            offset: d.offset,
            ld: d.ld,
            trans: d.trans,
            batch_stride: d.batch_stride,
        };
        Gemm {
            m: self.m,
            n: self.n,
            k: self.k,
            batch: self.batch,
            alpha: 1.0,
            beta: 0.0,
            a: operand(a, self.a),
            b: operand(b, self.b),
            c,
            c_offset: self.c_offset,
            ldc: self.ldc,
            c_batch_stride: self.c_batch_stride,
        }
    }
}

/// CUDA's `cutie_sgemm` on the stream, into the whole of `c`: nothing for an empty product, zeros
/// when k is 0 (a new tensor's contents are undefined), otherwise the GEMM.
#[cfg(target_os = "macos")]
fn cutie_sgemm(s: &mut MetalStream<'_>, d: &GemmDims, a: Bind<'_>, b: Bind<'_>, c: &GpuTensor) -> MetalResult<()> {
    if d.m == 0 || d.n == 0 {
        return Ok(());
    }
    if d.k == 0 {
        return s.fill(c, 0);
    }
    s.gemm(&d.bind(a, b, Bind::Out(c)))
}

// --- Convolution ---

/// How a convolution runs (CUDA's choice in `gpu_cutie_conv2d_fused`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvPath {
    /// Every image's columns side by side, [kdim, batch*opix], one GEMM into [out, batch*opix],
    /// then the unstack puts each image's channels in place with the epilogue.
    SideBySide,
    /// `group` images a strided-batched GEMM straight into the output, then the epilogue in place
    /// when there is one; `direct`: the input is its own columns (1x1, stride 1, no padding).
    Grouped { group: usize, direct: bool },
}

/// A convolution's checked shape and path. Every size a kernel takes fits a u32 and every GEMM
/// size an i32 (cuBLAS's on CUDA, kept here so a layer fails alike on both).
#[derive(Clone, Copy, Debug)]
struct ConvPlan {
    batch: usize,
    in_channels: usize,
    in_w: usize,
    in_h: usize,
    out_channels: usize,
    k: usize,
    pad: usize,
    stride: usize,
    ow: usize,
    oh: usize,
    /// Output pixels an image.
    opix: usize,
    /// Rows of an image's columns: in_channels * k * k.
    kdim: usize,
    path: ConvPath,
}

impl ConvPlan {
    /// CUDA's checks and messages for `x` [x_rows, x_cols] (`residual`: its shape), then its
    /// choice of path.
    #[allow(clippy::too_many_arguments)]
    fn new(
        op: &str, key: &str, x_rows: usize, x_cols: usize, batch: usize, in_w: usize, in_h: usize, weights: usize,
        bias: Option<usize>, out_channels: usize, k: usize, pad: usize, stride: usize, residual: Option<(usize, usize)>,
        limits: ConvLimits,
    ) -> MetalResult<ConvPlan> {
        if batch == 0
            || x_rows == 0
            || !x_rows.is_multiple_of(batch)
            || x_cols != in_w * in_h
            || k == 0
            || stride == 0
            || in_w + 2 * pad < k
            || in_h + 2 * pad < k
        {
            return Err(format!(
                "{op} {key}: x {x_rows}x{x_cols} for batch {batch} at {in_w}x{in_h}, k {k} pad {pad} stride {stride}"
            ));
        }
        let in_channels = x_rows / batch;
        let kdim = in_channels * k * k;
        if weights != out_channels * kdim || bias.is_some_and(|b| b != out_channels) {
            return Err(format!("{op} {key}: {weights} weights for {out_channels}x{in_channels}x{k}x{k}"));
        }
        let (ow, oh) = (cutie_conv_out(in_w, k, pad, stride), cutie_conv_out(in_h, k, pad, stride));
        let opix = ow * oh;
        if let Some((rows, cols)) = residual {
            if (rows, cols) != (batch * out_channels, opix) {
                return Err(format!(
                    "{op} {key}: residual {rows}x{cols} for an output {}x{opix}",
                    batch * out_channels
                ));
            }
        }
        let dim = |v: usize| {
            i32::try_from(v).map(|_| ()).map_err(|_| format!("{op} {key}: {v} exceeds the GEMM's sizes (cuBLAS's on CUDA)"))
        };
        dim(opix)?;
        dim(out_channels)?;
        dim(kdim)?;
        for v in [in_channels, in_w, in_h, ow, oh, k, pad, stride, batch, out_channels] {
            cutie_u32(op, v)?;
        }
        let per_image = kdim * opix;
        let side_by_side = batch > 1
            && opix <= limits.side_by_side_pixels
            && batch * per_image * size_of::<f32>() <= limits.scratch_bytes;
        let (path, col_rows) = if side_by_side {
            dim(batch * opix)?;
            (ConvPath::SideBySide, batch * kdim)
        } else if k == 1 && stride == 1 && pad == 0 {
            (ConvPath::Grouped { group: batch, direct: true }, 0)
        } else {
            // Images a GEMM: all of them, unless their columns outgrow the scratch.
            let group = (limits.scratch_bytes / size_of::<f32>() / per_image.max(1)).clamp(1, batch);
            (ConvPath::Grouped { group, direct: false }, group * kdim)
        };
        // The kernels' row counts: a launch's columns (at most the scratch's, or one image's kdim)
        // and the output's rows. Neither can fail for a tensor that fits in memory; checked so the
        // launches convert them without a check.
        cutie_u32(op, col_rows)?;
        cutie_u32(op, batch * out_channels)?;
        Ok(ConvPlan { batch, in_channels, in_w, in_h, out_channels, k, pad, stride, ow, oh, opix, kdim, path })
    }

    /// The output's rows: images * out channels.
    fn out_rows(&self) -> usize {
        self.batch * self.out_channels
    }

    /// Whether every GEMM this plan's path runs would be on the tiled kernel (gemm.rs's choice in
    /// the calling thread's mode), so that the one-pass kernel, which multiplies as that kernel
    /// does, gives the same bits.
    #[cfg(target_os = "macos")]
    fn fused_matches_tiled(&self) -> MetalResult<bool> {
        use crate::stream::gemm::{mode, GemmVariant};
        let mode = mode()?;
        let tiled = |n: usize, batch: usize| GemmVariant::choose(mode, self.out_channels, n, self.kdim, batch) == GemmVariant::Tiled;
        Ok(match self.path {
            ConvPath::SideBySide => tiled(self.batch * self.opix, 1),
            // Groups of images, the last perhaps smaller: one image is the least work any has.
            ConvPath::Grouped { .. } => tiled(self.opix, 1),
        })
    }

    /// The one-pass kernel's params for the whole batch.
    fn conv_gemm(&self, has_bias: bool, has_residual: bool, relu: bool) -> CutieConvGemm {
        CutieConvGemm {
            images: self.batch as u32,
            channels: self.in_channels as u32,
            width: self.in_w as u32,
            height: self.in_h as u32,
            ow: self.ow as u32,
            oh: self.oh as u32,
            k: self.k as u32,
            pad: self.pad as u32,
            stride: self.stride as u32,
            out_channels: self.out_channels as u32,
            kdim: self.kdim as u32,
            plane: self.opix as u32,
            has_bias: has_bias as u32,
            has_residual: has_residual as u32,
            relu: relu as u32,
            step_c: (CONV_GEMM_TILE_K / (self.k * self.k)) as u32,
            step_y: (CONV_GEMM_TILE_K % (self.k * self.k) / self.k) as u32,
            step_x: (CONV_GEMM_TILE_K % self.k) as u32,
        }
    }

    /// The columns of `count` images from `first` on: per image ([count*kdim, opix]) or side by
    /// side ([kdim, count*opix]).
    fn im2col(&self, first: usize, count: usize, side_by_side: bool) -> CutieIm2col {
        CutieIm2col {
            images: count as u32,
            first_image: first as u32,
            channels: self.in_channels as u32,
            width: self.in_w as u32,
            height: self.in_h as u32,
            ow: self.ow as u32,
            oh: self.oh as u32,
            k: self.k as u32,
            pad: self.pad as u32,
            stride: self.stride as u32,
            rows: (count * self.kdim) as u32,
            side_by_side: side_by_side as u32,
        }
    }

    /// The side-by-side GEMM: the [out, kdim] weights @ the [kdim, batch*opix] columns =
    /// [out, batch*opix], the images still side by side.
    fn side_by_side_gemm(&self) -> GemmDims {
        let wide = self.batch * self.opix;
        GemmDims {
            m: self.out_channels,
            n: wide,
            k: self.kdim,
            batch: 1,
            a: OperandDims { offset: 0, ld: self.kdim, trans: false, batch_stride: 0 },
            b: OperandDims { offset: 0, ld: wide, trans: false, batch_stride: 0 },
            c_offset: 0,
            ldc: wide,
            c_batch_stride: 0,
        }
    }

    /// The GEMM of `count` images from `first` on, one product an image: the [out, kdim] weights
    /// @ its [kdim, opix] columns (the scratch's, from its start, or, `direct`, the input's own
    /// planes) = its [out, opix] place in the output.
    fn group_gemm(&self, first: usize, count: usize, direct: bool) -> GemmDims {
        let per_image = self.kdim * self.opix;
        let image = self.in_channels * self.in_w * self.in_h;
        GemmDims {
            m: self.out_channels,
            n: self.opix,
            k: self.kdim,
            batch: count,
            a: OperandDims { offset: 0, ld: self.kdim, trans: false, batch_stride: 0 },
            b: OperandDims {
                offset: if direct { first * image } else { 0 },
                ld: self.opix,
                trans: false,
                batch_stride: per_image,
            },
            c_offset: first * self.out_channels * self.opix,
            ldc: self.opix,
            c_batch_stride: self.out_channels * self.opix,
        }
    }

    /// The pass after the GEMM over the output's rows.
    fn epilogue(&self, has_bias: bool, has_residual: bool, relu: bool) -> CutieConvEpilogue {
        CutieConvEpilogue {
            images: self.batch as u32,
            channels: self.out_channels as u32,
            plane: self.opix as u32,
            rows: self.out_rows() as u32,
            has_bias: has_bias as u32,
            has_residual: has_residual as u32,
            relu: relu as u32,
        }
    }
}

/// Conv2d over a batch of planar images: `x` is [batch*in_channels, in_h*in_w], `weights`
/// PyTorch's [out][in][k][k], `bias` per output channel (None: no bias). Square kernel, symmetric
/// padding and stride. Out: [batch*out_channels, out_h*out_w], with out_h = (in_h + 2*pad -
/// k)/stride + 1. Weights then bias are cached on the device as one buffer under
/// `{cache_namespace}::{key}::cutie_conv` (one key per layer).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str, weights: &[f32],
    bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    gpu_cutie_conv2d_fused(
        x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, None, false,
    )
}

/// `gpu_cutie_conv2d`, then `residual` added (when given: the output's shape, [batch*out_channels,
/// out_h*out_w]) and a ReLU (when `relu`), in the pass that adds the bias: relu((conv + bias) +
/// residual), each add rounded as `gpu_add` rounds it and the ReLU `gpu_birefnet_relu`'s (fmax(x,
/// 0)), so the result is those ops' to the bit, the output read and written once instead of three
/// times (a ResNet block's tail).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str, weights: &[f32],
    bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize, residual: Option<&GpuTensor>,
    relu: bool,
) -> Result<GpuTensor, String> {
    conv2d_fused(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu, false)
}

/// `gpu_cutie_conv2d_fused` for Cutie's fp16 steps (CUDA's half convolution): on
/// `MPSCNNConvolution` where the padding is k/2 and k is odd (every one of Cutie's), which keeps
/// its weights in half and sums in fp32, about rel 2e-4 from the fp32 path (judged by the masks, as
/// CUDA's fp16 is, not by `Tol::STAGE`); elsewhere the fp32 path.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused_f16(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str, weights: &[f32],
    bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize, residual: Option<&GpuTensor>,
    relu: bool,
) -> Result<GpuTensor, String> {
    conv2d_fused(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu, true)
}

/// The two above: `half` runs the convolution on `MPSCNNConvolution` (as MAKEPAD_CUTIE_CONV=mpscnn
/// does for every convolution, to measure).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn conv2d_fused(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str, weights: &[f32],
    bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize, residual: Option<&GpuTensor>,
    relu: bool, half: bool,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_conv2d";
    cutie_check_f32(op, x)?;
    if let Some(r) = residual {
        cutie_check_f32(op, r)?;
    }
    let plan = ConvPlan::new(
        op,
        key,
        x.rows(),
        x.cols(),
        batch,
        in_w,
        in_h,
        weights.len(),
        bias.map(<[f32]>::len),
        out_channels,
        k,
        pad,
        stride,
        residual.map(|r| (r.rows(), r.cols())),
        CUTIE_CONV_LIMITS,
    )?;
    let qualified_key = format!("{cache_namespace}::{key}::cutie_conv");
    let floats = weights.len() + bias.map_or(0, <[f32]>::len);
    let epilogue = plan.epilogue(bias.is_some(), residual.is_some(), relu);
    let epilogue_grid = rows_grid(plan.opix, plan.out_rows())?;
    let residual_bind = residual.map_or(Bind::Absent, Bind::In);
    let gemm_err = |e: String| keyed_error(op, key, &e);
    with_metal_stream(op, |s| {
        // Weights then bias, one cached buffer: no host upload after the first call.
        let w = s.weights(&qualified_key, floats, |dst| {
            let (w_dst, b_dst) = dst.split_at_mut(weights.len());
            w_dst.copy_from_slice(weights);
            if let Some(b) = bias {
                b_dst.copy_from_slice(b);
            }
            Ok(())
        })?;
        let bias_bind = if bias.is_some() { Bind::Weights(&w, weights.len()) } else { Bind::Absent };
        let out = GpuTensor::alloc(plan.out_rows(), plan.opix)?;
        if plan.out_rows() == 0 {
            // No output channels: nothing to compute (cuBLAS returns at once on CUDA).
            return Ok(out);
        }
        if (half || conv_mpscnn_enabled()) && plan.k % 2 == 1 && plan.pad == plan.k / 2 {
            s.conv_mpscnn(&MpsConv {
                key: &qualified_key,
                weights,
                bias,
                x,
                out: &out,
                residual,
                relu,
                batch: plan.batch,
                in_channels: plan.in_channels,
                in_w: plan.in_w,
                in_h: plan.in_h,
                out_channels: plan.out_channels,
                ow: plan.ow,
                oh: plan.oh,
                k: plan.k,
                stride: plan.stride,
                to_image: &PLANAR_TO_IMAGE,
                from_image: &IMAGE_TO_PLANAR,
            })?;
            return Ok(out);
        }
        if conv_gemm_enabled() && plan.fused_matches_tiled()? {
            // One pass: the same bits as the three below on the tiled GEMM (cutie_conv_gemm_f32).
            let params = plan.conv_gemm(bias.is_some(), residual.is_some(), relu);
            let n = plan.batch * plan.opix;
            let grid = Grid::groups(
                [n.div_ceil(CONV_GEMM_TILE), plan.out_channels.div_ceil(CONV_GEMM_TILE), 1],
                [CONV_GEMM_THREADS, 1, 1],
            )?;
            crate::stream::gemm::check_simd_width(s, op, &CONV_GEMM)?;
            let binds = [Bind::In(x), Bind::Weights(&w, 0), Bind::Out(&out), bias_bind, residual_bind];
            s.launch(&CONV_GEMM, &[], &params, &binds, grid)?;
            return Ok(out);
        }
        match plan.path {
            ConvPath::SideBySide => {
                // Small images: one GEMM over every image's columns side by side keeps the GEMM's
                // tiles full, then each image's channels are put back in place with the epilogue.
                let gemm = plan.side_by_side_gemm();
                let col = GpuTensor::alloc(plan.kdim, gemm.n)?;
                let stacked = GpuTensor::alloc(plan.out_channels, gemm.n)?;
                let im2col = plan.im2col(0, plan.batch, true);
                let grid = rows_grid(plan.opix, im2col.rows as usize)?;
                s.launch(&IM2COL, &[], &im2col, &[Bind::In(x), Bind::Out(&col)], grid)?;
                s.gemm(&gemm.bind(Bind::Weights(&w, 0), Bind::In(&col), Bind::Out(&stacked))).map_err(gemm_err)?;
                let binds = [Bind::In(&stacked), Bind::Out(&out), bias_bind, residual_bind];
                s.launch(&UNSTACK_EPILOGUE, &[], &epilogue, &binds, epilogue_grid)?;
            }
            ConvPath::Grouped { group, direct } => {
                let col = if direct { None } else { Some(GpuTensor::alloc(group * plan.kdim, plan.opix)?) };
                let mut first = 0;
                while first < plan.batch {
                    let count = group.min(plan.batch - first);
                    let columns = match &col {
                        Some(col) => {
                            let im2col = plan.im2col(first, count, false);
                            let grid = rows_grid(plan.opix, im2col.rows as usize)?;
                            s.launch(&IM2COL, &[], &im2col, &[Bind::In(x), Bind::Out(col)], grid)?;
                            Bind::In(col)
                        }
                        None => Bind::In(x),
                    };
                    let gemm = plan.group_gemm(first, count, direct);
                    s.gemm(&gemm.bind(Bind::Weights(&w, 0), columns, Bind::Out(&out))).map_err(gemm_err)?;
                    first += count;
                }
                if bias.is_some() || residual.is_some() || relu {
                    let binds = [Bind::InOut(&out), bias_bind, residual_bind];
                    s.launch(&CONV_EPILOGUE, &[], &epilogue, &binds, epilogue_grid)?;
                }
            }
        }
        Ok(out)
    })
}

// --- Matrix products and memory reading ---

/// op(a) @ op(b), op transposing when asked: op(a) [m, k], op(b) [k, n]; out [m, n]. fp32
/// multiply-accumulate.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_matmul(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_matmul", a)?;
    cutie_check_f32("gpu_cutie_matmul", b)?;
    let (m, k) = if trans_a { (a.cols(), a.rows()) } else { (a.rows(), a.cols()) };
    let (k2, n) = if trans_b { (b.cols(), b.rows()) } else { (b.rows(), b.cols()) };
    if k != k2 {
        return Err(format!("gpu_cutie_matmul inner dims {k} != {k2}"));
    }
    let gemm = GemmDims::product(m, n, k, a.cols(), trans_a, b.cols(), trans_b);
    gemm.check()?;
    with_metal_stream("gpu_cutie_matmul", |s| {
        let out = GpuTensor::alloc(m, n)?;
        cutie_sgemm(s, &gemm, Bind::In(a), Bind::In(b), &out)?;
        Ok(out)
    })
}

/// Cutie's anisotropic L2 similarity (XMem's appendix): memory keys `mk` [ck, m], shrinkage `ms`
/// [1, m] (None: 1), query keys `qk` [ck, p], selection `qe` [ck, p] (None: plain L2 as STCN).
/// Out [m, p]: (-a_sq + two_ab - b_sq) * ms / sqrt(ck), with a_sq = mk^2 . qe, two_ab =
/// 2 mk . (qk*qe), b_sq = sum_c qe*qk^2 (without qe: a_sq = |mk|^2, two_ab = 2 mk . qk, no b_sq).
///
/// As PyTorch forms it: a_sq and two_ab are GEMMs over the channels, each rounded before they are
/// combined, so cancellation between them is PyTorch's own; the division by sqrt(ck) is a multiply
/// by its reciprocal (`cutie_inv_sqrt`). The channel sums (b_sq, and a_sq without qe) are
/// sequential, PyTorch's split over warps: fp32 rounding apart.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_similarity(
    mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_similarity";
    cutie_check_f32(op, mk)?;
    cutie_check_f32(op, qk)?;
    for t in ms.iter().chain(qe.iter()) {
        cutie_check_f32(op, t)?;
    }
    let (ck, m, p) = (mk.rows(), mk.cols(), qk.cols());
    if qk.rows() != ck || qe.is_some_and(|e| e.rows() != ck || e.cols() != p) || ms.is_some_and(|s| s.rows() * s.cols() != m)
    {
        return Err("gpu_cutie_similarity shape mismatch".into());
    }
    let (ck_u, m_u, p_u) = (cutie_u32(op, ck)?, cutie_u32(op, m)?, cutie_u32(op, p)?);
    let inv_scale = cutie_inv_sqrt(ck);
    // Both GEMMs are mk-shaped [ck, m] transposed against a [ck, p] operand.
    let gemm = GemmDims::product(m, p, ck, m, true, p, false);
    gemm.check()?;
    let prep_grid = Grid::linear(m.max(p), THREADS)?;
    let epilogue_grid = rows_grid(p, m)?;
    let ms_bind = ms.map_or(Bind::Absent, Bind::In);
    let has_ms = ms.is_some() as u32;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(m, p)?;
        match qe {
            Some(qe) => {
                let mk_sq = GpuTensor::alloc(ck, m)?;
                let qkqe = GpuTensor::alloc(ck, p)?;
                let b_sq = GpuTensor::alloc(1, p)?;
                let two_ab = GpuTensor::alloc(m, p)?;
                let prep = CutieSimilarityPrep { ck: ck_u, m: m_u, p: p_u, has_qe: 1 };
                let binds = [
                    Bind::In(mk),
                    Bind::In(qk),
                    Bind::In(qe),
                    Bind::Out(&mk_sq),
                    Bind::Out(&qkqe),
                    Bind::Out(&b_sq),
                    Bind::Absent,
                ];
                s.launch(&SIMILARITY_PREP, &[], &prep, &binds, prep_grid)?;
                // a_sq = (mk^2)^T qe into the output; two_ab's product mk^T (qk*qe) beside it.
                cutie_sgemm(s, &gemm, Bind::In(&mk_sq), Bind::In(qe), &out)?;
                cutie_sgemm(s, &gemm, Bind::In(mk), Bind::In(&qkqe), &two_ab)?;
                let epilogue = CutieSimilarityEpilogue { m: m_u, p: p_u, inv_scale, has_two_ab: 1, has_ms };
                let binds = [Bind::InOut(&out), Bind::In(&two_ab), Bind::In(&b_sq), Bind::Absent, ms_bind];
                s.launch(&SIMILARITY_EPILOGUE, &[], &epilogue, &binds, epilogue_grid)?;
            }
            None => {
                let a_sq = GpuTensor::alloc(1, m)?;
                let prep = CutieSimilarityPrep { ck: ck_u, m: m_u, p: p_u, has_qe: 0 };
                let binds = [
                    Bind::In(mk),
                    Bind::In(qk),
                    Bind::Absent,
                    Bind::Absent,
                    Bind::Absent,
                    Bind::Absent,
                    Bind::Out(&a_sq),
                ];
                s.launch(&SIMILARITY_PREP, &[], &prep, &binds, prep_grid)?;
                // mk^T qk into the output.
                cutie_sgemm(s, &gemm, Bind::In(mk), Bind::In(qk), &out)?;
                let epilogue = CutieSimilarityEpilogue { m: m_u, p: p_u, inv_scale, has_two_ab: 0, has_ms };
                let binds = [Bind::InOut(&out), Bind::Absent, Bind::Absent, Bind::In(&a_sq), ms_bind];
                s.launch(&SIMILARITY_EPILOGUE, &[], &epilogue, &binds, epilogue_grid)?;
            }
        }
        Ok(out)
    })
}

/// The plans on any OS: each kernel's loop and the GEMM's contract (`MetalStream::gemm`) run on
/// the host from a plan's numbers, against a direct convolution, the reference's similarity and a
/// plain product; and the paths CUDA takes at the ops check's sizes. The kernels themselves are
/// checked on the Mac (`cutie-validate ops`).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::source::library_source;

    /// A small deterministic generator (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// `n` values in [-1, 1).
        fn values(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect()
        }
    }

    /// `cutie_im2col_f32`, thread for thread.
    fn im2col(x: &[f32], p: &CutieIm2col, col: &mut [f32]) {
        let opix = p.ow as usize * p.oh as usize;
        let (kk, kdim) = (p.k * p.k, p.channels * p.k * p.k);
        for q in 0..opix {
            let (oy, ox) = ((q / p.ow as usize) as i64, (q % p.ow as usize) as i64);
            for row in 0..p.rows {
                let (b, r) = (row / kdim, row % kdim);
                let (c, ky, kx) = (r / kk, (r % kk) / p.k, r % p.k);
                let iy = oy * p.stride as i64 - p.pad as i64 + ky as i64;
                let ix = ox * p.stride as i64 - p.pad as i64 + kx as i64;
                let inside = iy >= 0 && iy < p.height as i64 && ix >= 0 && ix < p.width as i64;
                let at = |image: u32| {
                    (((image as usize * p.channels as usize + c as usize) * p.height as usize + iy as usize)
                        * p.width as usize)
                        + ix as usize
                };
                let v = if inside { x[at(p.first_image + b)] } else { 0.0 };
                let to = if p.side_by_side != 0 { r as usize * p.images as usize + b as usize } else { row as usize };
                col[to * opix + q] = v;
            }
        }
    }

    /// `MetalStream::gemm`'s contract: C[i] = op(A[i]) @ op(B[i]), row-major, op(A)[r][l] at
    /// `offset + i*batch_stride + r*ld + l` (transposed: `l*ld + r`), in k order.
    fn gemm(d: &GemmDims, a: &[f32], b: &[f32], c: &mut [f32]) {
        let at = |o: &OperandDims, i: usize, row: usize, col: usize| {
            let base = o.offset + i * o.batch_stride;
            if o.trans { base + col * o.ld + row } else { base + row * o.ld + col }
        };
        for i in 0..d.batch {
            for r in 0..d.m {
                for j in 0..d.n {
                    let mut sum = 0.0f32;
                    for l in 0..d.k {
                        sum += a[at(&d.a, i, r, l)] * b[at(&d.b, i, l, j)];
                    }
                    c[d.c_offset + i * d.c_batch_stride + r * d.ldc + j] = sum;
                }
            }
        }
    }

    fn epilogue_value(v: f32, p: &CutieConvEpilogue, bias: &[f32], c: usize, residual: &[f32], at: usize) -> f32 {
        let mut v = v;
        if p.has_bias != 0 {
            v += bias[c];
        }
        if p.has_residual != 0 {
            v += residual[at];
        }
        if p.relu != 0 { v.max(0.0) } else { v }
    }

    /// `cutie_unstack_epilogue_f32`, thread for thread.
    fn unstack(src: &[f32], out: &mut [f32], bias: &[f32], residual: &[f32], p: &CutieConvEpilogue) {
        let plane = p.plane as usize;
        for q in 0..plane {
            for row in 0..p.rows as usize {
                let (b, c) = (row / p.channels as usize, row % p.channels as usize);
                let at = row * plane + q;
                out[at] = epilogue_value(src[(c * p.images as usize + b) * plane + q], p, bias, c, residual, at);
            }
        }
    }

    /// `cutie_conv_epilogue_f32`, thread for thread.
    fn conv_epilogue(out: &mut [f32], bias: &[f32], residual: &[f32], p: &CutieConvEpilogue) {
        let plane = p.plane as usize;
        for q in 0..plane {
            for row in 0..p.rows as usize {
                let at = row * plane + q;
                out[at] = epilogue_value(out[at], p, bias, row % p.channels as usize, residual, at);
            }
        }
    }

    /// `gpu_cutie_conv2d_fused`'s launches from its plan, on the host.
    fn run_conv(plan: &ConvPlan, x: &[f32], w: &[f32], bias: &[f32], residual: &[f32], relu: bool) -> Vec<f32> {
        let epilogue = plan.epilogue(!bias.is_empty(), !residual.is_empty(), relu);
        let mut out = vec![f32::NAN; plan.out_rows() * plan.opix];
        match plan.path {
            ConvPath::SideBySide => {
                let gemm = plan.side_by_side_gemm();
                let mut col = vec![f32::NAN; plan.kdim * gemm.n];
                let mut stacked = vec![f32::NAN; plan.out_channels * gemm.n];
                im2col(x, &plan.im2col(0, plan.batch, true), &mut col);
                self::gemm(&gemm, w, &col, &mut stacked);
                unstack(&stacked, &mut out, bias, residual, &epilogue);
            }
            ConvPath::Grouped { group, direct } => {
                let mut col = vec![f32::NAN; if direct { 0 } else { group * plan.kdim * plan.opix }];
                let mut first = 0;
                while first < plan.batch {
                    let count = group.min(plan.batch - first);
                    let gemm = plan.group_gemm(first, count, direct);
                    if direct {
                        self::gemm(&gemm, w, x, &mut out);
                    } else {
                        im2col(x, &plan.im2col(first, count, false), &mut col);
                        self::gemm(&gemm, w, &col, &mut out);
                    }
                    first += count;
                }
                if !bias.is_empty() || !residual.is_empty() || relu {
                    conv_epilogue(&mut out, bias, residual, &epilogue);
                }
            }
        }
        out
    }

    /// A direct convolution summed in f64, then the residual and the ReLU.
    fn direct_conv(plan: &ConvPlan, x: &[f32], w: &[f32], bias: &[f32], residual: &[f32], relu: bool) -> Vec<f32> {
        let (cin, k) = (plan.in_channels, plan.k);
        let mut out = Vec::with_capacity(plan.out_rows() * plan.opix);
        for b in 0..plan.batch {
            for o in 0..plan.out_channels {
                for oy in 0..plan.oh {
                    for ox in 0..plan.ow {
                        let mut sum = 0.0f64;
                        for c in 0..cin {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = (oy * plan.stride + ky) as i64 - plan.pad as i64;
                                    let ix = (ox * plan.stride + kx) as i64 - plan.pad as i64;
                                    if iy < 0 || iy >= plan.in_h as i64 || ix < 0 || ix >= plan.in_w as i64 {
                                        continue;
                                    }
                                    let xv = x[((b * cin + c) * plan.in_h + iy as usize) * plan.in_w + ix as usize];
                                    sum += xv as f64 * w[((o * cin + c) * k + ky) * k + kx] as f64;
                                }
                            }
                        }
                        let mut v = sum as f32 + bias.get(o).copied().unwrap_or(0.0);
                        if let Some(r) = residual.get(out.len()) {
                            v += r;
                        }
                        out.push(if relu { v.max(0.0) } else { v });
                    }
                }
            }
        }
        out
    }

    fn assert_close(what: &str, actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tol, "{what}: [{i}] {a} against {e}");
        }
    }

    /// Small limits, so every path runs at sizes a test can afford: side by side up to 64 pixels
    /// an image, a scratch of 4096 floats.
    const SMALL: ConvLimits = ConvLimits { scratch_bytes: 4096 * 4, side_by_side_pixels: 64 };

    #[test]
    fn every_conv_path_computes_the_convolution() {
        // (batch, in, w, h, out, k, pad, stride, bias, residual, relu, the path SMALL gives)
        type Case = (usize, usize, usize, usize, usize, usize, usize, usize, bool, bool, bool, ConvPath);
        let grouped = |group: usize, direct: bool| ConvPath::Grouped { group, direct };
        let cases: [Case; 10] = [
            (3, 2, 7, 5, 4, 3, 1, 1, true, true, true, ConvPath::SideBySide),
            (2, 3, 8, 6, 2, 2, 0, 2, false, false, false, ConvPath::SideBySide),
            (4, 2, 3, 2, 3, 3, 1, 1, true, false, true, ConvPath::SideBySide),
            (2, 2, 13, 9, 3, 5, 2, 2, true, false, true, ConvPath::SideBySide),
            (2, 3, 15, 9, 2, 1, 0, 2, true, true, false, ConvPath::SideBySide),
            (3, 2, 5, 4, 2, 1, 0, 1, true, true, true, ConvPath::SideBySide),
            (1, 3, 9, 7, 2, 3, 1, 1, true, false, false, grouped(1, false)),
            (2, 4, 10, 9, 3, 1, 0, 1, true, true, false, grouped(2, true)),
            (3, 2, 11, 8, 3, 3, 1, 1, true, true, true, grouped(2, false)),
            (3, 4, 12, 10, 2, 3, 1, 1, false, false, false, grouped(1, false)),
        ];
        let mut rng = Rng(0x5eed_c0de);
        for (batch, cin, w, h, cout, k, pad, stride, bias, residual, relu, path) in cases {
            let x = rng.values(batch * cin * w * h);
            let weights = rng.values(cout * cin * k * k);
            let b = if bias { rng.values(cout) } else { Vec::new() };
            let (ow, oh) = (cutie_conv_out(w, k, pad, stride), cutie_conv_out(h, k, pad, stride));
            let r = if residual { rng.values(batch * cout * ow * oh) } else { Vec::new() };
            let shape = residual.then_some((batch * cout, ow * oh));
            let plan = ConvPlan::new(
                "conv", "case", batch * cin, w * h, batch, w, h, weights.len(), bias.then_some(cout), cout, k, pad, stride,
                shape, SMALL,
            )
            .unwrap();
            let what = format!("{batch}x{cin}x{w}x{h} -> {cout} k{k} p{pad} s{stride}");
            assert_eq!(plan.path, path, "{what}: path");
            let actual = run_conv(&plan, &x, &weights, &b, &r, relu);
            let expected = direct_conv(&plan, &x, &weights, &b, &r, relu);
            assert_close(&what, &actual, &expected, 1e-4);
        }
    }

    #[test]
    fn conv_paths_are_cudas_at_the_ops_check_sizes() {
        let path = |batch: usize, cin: usize, w: usize, h: usize, cout: usize, k: usize, pad: usize, stride: usize| {
            let plan = ConvPlan::new(
                "conv", "case", batch * cin, w * h, batch, w, h, cout * cin * k * k, Some(cout), cout, k, pad, stride,
                None, CUTIE_CONV_LIMITS,
            )
            .unwrap();
            plan.path
        };
        let grouped = |group: usize, direct: bool| ConvPath::Grouped { group, direct };
        // One image never goes side by side; its columns, or its input for a 1x1.
        assert_eq!(path(1, 3, 384, 224, 64, 7, 3, 2), grouped(1, false));
        assert_eq!(path(1, 1024, 24, 14, 256, 1, 0, 1), grouped(1, true));
        // Small images (stride 16 and 8) side by side; a strided 1x1 needs its columns.
        assert_eq!(path(3, 256, 24, 14, 256, 3, 1, 1), ConvPath::SideBySide);
        assert_eq!(path(2, 512, 24, 14, 768, 3, 1, 1), ConvPath::SideBySide);
        assert_eq!(path(2, 64, 96, 56, 128, 1, 0, 2), ConvPath::SideBySide);
        // Larger images as a strided batch, a 1x1 reading its input.
        assert_eq!(path(2, 64, 96, 56, 64, 1, 0, 1), grouped(2, true));
        assert_eq!(path(3, 64, 96, 56, 64, 3, 1, 1), grouped(3, false));
        // Columns past the 256 MB scratch: two images at a time, or one at the worker's 1000p.
        assert_eq!(path(3, 64, 224, 192, 64, 3, 1, 1), grouped(2, false));
        assert_eq!(path(2, 5, 1792, 1008, 64, 7, 3, 2), grouped(1, false));
    }

    #[test]
    fn conv_checks_are_cudas() {
        let plan = |x_rows: usize, x_cols: usize, batch: usize, weights: usize, bias: Option<usize>, residual| {
            ConvPlan::new("gpu_cutie_conv2d", "l", x_rows, x_cols, batch, 4, 3, weights, bias, 2, 3, 1, 1, residual, CUTIE_CONV_LIMITS)
                .map(|p| p.path)
        };
        assert_eq!(
            plan(5, 12, 2, 2 * 9, None, None).unwrap_err(),
            "gpu_cutie_conv2d l: x 5x12 for batch 2 at 4x3, k 3 pad 1 stride 1"
        );
        assert!(plan(0, 12, 1, 0, None, None).is_err());
        assert!(plan(4, 12, 0, 2 * 9, None, None).is_err());
        assert!(plan(4, 11, 2, 2 * 2 * 9, None, None).is_err());
        assert_eq!(plan(4, 12, 2, 2 * 2 * 9 + 1, None, None).unwrap_err(), "gpu_cutie_conv2d l: 37 weights for 2x2x3x3");
        assert!(plan(4, 12, 2, 2 * 2 * 9, Some(3), None).is_err());
        assert_eq!(
            plan(4, 12, 2, 2 * 2 * 9, Some(2), Some((4, 11))).unwrap_err(),
            "gpu_cutie_conv2d l: residual 4x11 for an output 4x12"
        );
        assert_eq!(plan(4, 12, 2, 2 * 2 * 9, Some(2), Some((4, 12))).unwrap(), ConvPath::SideBySide);
        // A window larger than the padded image.
        let too_small = ConvPlan::new("gpu_cutie_conv2d", "l", 1, 4, 1, 2, 2, 25, None, 1, 5, 1, 1, None, CUTIE_CONV_LIMITS);
        assert!(too_small.is_err());
    }

    #[test]
    fn products_follow_cutie_sgemm() {
        let mut rng = Rng(0xabad_cafe);
        let (m, k, n) = (5, 7, 3);
        for (trans_a, trans_b) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = rng.values(m * k);
            let b = rng.values(k * n);
            let a_cols = if trans_a { m } else { k };
            let b_cols = if trans_b { k } else { n };
            let mut actual = vec![f32::NAN; m * n];
            gemm(&GemmDims::product(m, n, k, a_cols, trans_a, b_cols, trans_b), &a, &b, &mut actual);
            let av = |i: usize, l: usize| if trans_a { a[l * m + i] } else { a[i * k + l] };
            let bv = |l: usize, j: usize| if trans_b { b[j * k + l] } else { b[l * n + j] };
            let expected: Vec<f32> = (0..m * n)
                .map(|ij| (0..k).map(|l| av(ij / n, l) as f64 * bv(l, ij % n) as f64).sum::<f64>() as f32)
                .collect();
            assert_close(&format!("matmul {trans_a} {trans_b}"), &actual, &expected, 1e-5);
        }
    }

    /// `gpu_cutie_similarity`'s launches on the host: `cutie_similarity_prep_f32`, the two
    /// products, then `cutie_similarity_epilogue_f32`.
    fn run_similarity(ck: usize, m: usize, p: usize, mk: &[f32], ms: Option<&[f32]>, qk: &[f32], qe: Option<&[f32]>) -> Vec<f32> {
        let gemm = GemmDims::product(m, p, ck, m, true, p, false);
        let inv_scale = cutie_inv_sqrt(ck);
        let mut sim = vec![f32::NAN; m * p];
        let (mut two_ab, mut b_sq, mut a_sq) = (Vec::new(), Vec::new(), vec![0.0f32; m]);
        match qe {
            Some(qe) => {
                let mk_sq: Vec<f32> = mk.iter().map(|v| v * v).collect();
                let qkqe: Vec<f32> = qk.iter().zip(qe).map(|(k, e)| k * e).collect();
                b_sq = (0..p).map(|t| (0..ck).fold(0.0f32, |sum, c| sum + qe[c * p + t] * (qk[c * p + t] * qk[c * p + t]))).collect();
                self::gemm(&gemm, &mk_sq, qe, &mut sim);
                two_ab = vec![f32::NAN; m * p];
                self::gemm(&gemm, mk, &qkqe, &mut two_ab);
            }
            None => {
                a_sq = (0..m).map(|t| (0..ck).fold(0.0f32, |sum, c| sum + mk[c * m + t] * mk[c * m + t])).collect();
                self::gemm(&gemm, mk, qk, &mut sim);
            }
        }
        for (i, v) in sim.iter_mut().enumerate() {
            let (j, px) = (i / p, i % p);
            let mut s = if qe.is_some() { (-*v + 2.0 * two_ab[i]) - b_sq[px] } else { -a_sq[j] + 2.0 * *v };
            if let Some(ms) = ms {
                s *= ms[j];
            }
            *v = s * inv_scale;
        }
        sim
    }

    #[test]
    fn similarity_follows_the_reference() {
        let mut rng = Rng(0x51_3117);
        let (ck, m, p) = (8, 5, 6);
        let mk = rng.values(ck * m);
        let qk = rng.values(ck * p);
        let qe: Vec<f32> = rng.values(ck * p).iter().map(|v| v * 0.5 + 0.5).collect();
        let ms: Vec<f32> = rng.values(m).iter().map(|d| d * d + 1.0).collect();
        for (shrinkage, selection) in [(false, false), (true, false), (false, true), (true, true)] {
            let ms = shrinkage.then_some(ms.as_slice());
            let qe = selection.then_some(qe.as_slice());
            let actual = run_similarity(ck, m, p, &mk, ms, &qk, qe);
            // cutie_reference.rs's gpu_cutie_similarity_reference.
            let mut expected = vec![0.0f32; m * p];
            for j in 0..m {
                for i in 0..p {
                    let (mut a_sq, mut two_ab, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
                    for c in 0..ck {
                        let (mkc, qkc) = (mk[c * m + j] as f64, qk[c * p + i] as f64);
                        let qec = qe.map_or(1.0, |e| e[c * p + i] as f64);
                        a_sq += mkc * mkc * qec;
                        two_ab += mkc * (qkc * qec);
                        b_sq += qec * qkc * qkc;
                    }
                    let b_sq = if qe.is_some() { b_sq as f32 } else { 0.0 };
                    let sim = -(a_sq as f32) + 2.0 * (two_ab as f32) - b_sq;
                    expected[j * p + i] = ms.map_or(sim, |s| sim * s[j]) * cutie_inv_sqrt(ck);
                }
            }
            assert_close(&format!("similarity ms {shrinkage} qe {selection}"), &actual, &expected, 1e-4);
        }
    }

    #[test]
    fn a_gemm_is_checked_before_anything_is_encoded() {
        let big = 1usize << 31;
        let e = GemmDims::product(big, 1, 1, big, true, 1, false).check().unwrap_err();
        assert_eq!(e, format!("cutie sgemm: {big} exceeds the GEMM's sizes (cuBLAS's on CUDA)"));
        assert!(GemmDims::product(big, 1, 0, big, true, 1, false).check().is_ok(), "k = 0 zeroes C, of 2^31 words");
        // k = 0 zeroes C with the stream's fill, which counts in u32 (CUDA's memset does not).
        let e = GemmDims::product(1 << 16, 1 << 16, 0, 1, false, 1 << 16, false).check().unwrap_err();
        assert_eq!(e, "cutie sgemm: 4294967296 does not fit the kernel's 32-bit sizes");
        assert!(GemmDims::product(0, 1 << 40, 0, 1, false, 1, false).check().is_ok(), "an empty product does nothing");
    }

    #[test]
    fn a_conv_gemms_error_names_the_op_once_with_the_key() {
        let e = "gpu_cutie_conv2d: gemm reads float 9 of B (3x3, ld 3, offset 0, 1 batches 0 apart), its binding holds 9";
        assert_eq!(
            keyed_error("gpu_cutie_conv2d", "enc.layer1.0.conv1", e),
            "gpu_cutie_conv2d enc.layer1.0.conv1: gemm reads float 9 of B (3x3, ld 3, offset 0, 1 batches 0 apart), its binding holds 9"
        );
        assert_eq!(keyed_error("gpu_cutie_conv2d", "k", "no Metal stream"), "gpu_cutie_conv2d k: no Metal stream");
    }

    #[test]
    fn rows_grid_is_cudas() {
        let grid = rows_grid(1000, 3).unwrap();
        assert_eq!(grid.groups, [4, 3, 1]);
        assert_eq!(grid.threads, [256, 1, 1]);
        assert_eq!(rows_grid(1, 100_000).unwrap().groups, [1, 65535, 1]);
        assert!(rows_grid(0, 5).unwrap().is_empty());
        assert!(rows_grid(5, 0).unwrap().is_empty());
    }

    #[test]
    fn every_kernel_is_in_the_library_and_the_library_includes_the_header() {
        for kernel in MATMUL_CONV_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &CUTIE_MATMUL_CONV));
            let declared = format!("kernel void {}(", kernel.function);
            assert!(CUTIE_MATMUL_CONV.source.contains(&declared), "{} is not declared", kernel.function);
        }
        assert!(CUTIE_MATMUL_CONV.source.contains("#include \"makepad_common.h\""));
        let spliced = library_source(&CUTIE_MATMUL_CONV, None).unwrap();
        assert!(spliced.contains("MAKEPAD_COMMON_H"));
        assert!(!CUTIE_MATMUL_CONV.source.contains("half"), "no half anywhere (D8)");
    }

    #[test]
    fn params_structs_have_their_msl_sizes() {
        assert_eq!(size_of::<CutieIm2col>(), 48);
        assert_eq!(size_of::<CutieConvEpilogue>(), 28);
        assert_eq!(size_of::<CutieSimilarityPrep>(), 16);
        assert_eq!(size_of::<CutieSimilarityEpilogue>(), 20);
    }
}
