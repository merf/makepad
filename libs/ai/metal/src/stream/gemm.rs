//! `MetalStream::gemm` (RationalVisions docs/cutie-metal.md §7): C = alpha·op(A)·op(B) + beta·C
//! in fp32, never staged through half, over row-major operands at any leading dimension, float
//! offset and batch stride (CUDA's strided-batched GemmEx covers packed heads and row views the
//! same way). Two kernels in `gemm.metal` sit behind the one call: a naive one (one thread an
//! output, an fma at a time down k: the correctness baseline) and a tiled one (simdgroup_float8x8
//! over fp32 tiles), chosen by shape, or by MAKEPAD_METAL_GEMM=naive|tiled|auto. With
//! MAKEPAD_METAL_GEMM=mps the product runs on Metal Performance Shaders instead (`mps.rs`), after
//! the same checks: fp32 too, but MPS's own order of sums, so not the kernels' bits.
//!
//! The checks and the plan (which kernel, its params and grid) are platform-free and unit-tested
//! anywhere; the launch is macOS's. A GEMM is refused before anything is encoded when:
//! - an operand is not read through `Bind::In` or `Bind::Weights`, or C not written through
//!   `Bind::Out` or `Bind::InOut`; beta is not 0 and C is not `InOut` (an Out's contents are
//!   undefined);
//! - a leading dimension is less than its stored matrix's columns (as cuBLAS refuses), or an
//!   operand's last element read (C's written) is past what its binding holds;
//! - a size is past the kernels' 32 bits (m, n, k and batch at most i32::MAX, as cuBLAS's ints;
//!   offsets, leading dimensions and strides at most u32::MAX).
//!
//! C's batches must not share an element (as cuBLAS); nothing checks it. k = 0 writes beta·C (0
//! with beta 0), and so does alpha = 0, without reading A or B, as BLAS.
//!
//! The library also holds the two kernels the GEMM's users add around it: the bias after a linear
//! (`GEMM_ADD_ROWS`, `linear.rs`) and attention's row softmax (`GEMM_SOFTMAX_ROWS`,
//! `attention.rs`), so the family compiles and fails as one library (cutie-metal.md §6).

use super::grid::Grid;
use super::kernel::{Kernel, KernelLib, KernelParams};
use super::MetalResult;
use std::cell::Cell;
use std::sync::OnceLock;

/// The GEMM family's library (`gemm.metal`).
pub static GEMM: KernelLib = KernelLib {
    name: "gemm",
    source: include_str!("../../shaders/makepad/gemm.metal"),
};

/// One thread an output ([`GemmParams`]).
pub static GEMM_NAIVE: Kernel = Kernel { lib: &GEMM, function: "gemm_naive_f32" };
/// 32x32 output tiles of four SIMD groups ([`GemmParams`]).
pub static GEMM_TILED: Kernel = Kernel { lib: &GEMM, function: "gemm_tiled_f32" };
/// `out[r][c] += bias[c]` ([`GemmAddRows`]): a linear's bias.
pub static GEMM_ADD_ROWS: Kernel = Kernel { lib: &GEMM, function: "gemm_add_rows_f32" };
/// Each row's softmax in place, with an optional additive bias ([`GemmSoftmaxRows`]).
pub static GEMM_SOFTMAX_ROWS: Kernel = Kernel { lib: &GEMM, function: "gemm_softmax_rows_f32" };

/// Every kernel above, for the Mac's test that compiles them all.
pub static GEMM_KERNELS: &[&Kernel] = &[&GEMM_NAIVE, &GEMM_TILED, &GEMM_ADD_ROWS, &GEMM_SOFTMAX_ROWS];

/// The SIMD-group width the tiled kernel and the softmax's reductions are written for (Apple
/// GPUs'); a pipeline with another is refused.
pub const SIMD_WIDTH: u32 = 32;
/// The tiled kernel's output tile, rows and columns (`GEMM_TILE` in `gemm.metal`).
const TILE: usize = 32;
/// The tiled kernel's threads: four SIMD groups (`GEMM_TILED_THREADS`).
const TILED_THREADS: u32 = 128;
/// The naive kernel's threadgroup: 16 columns by 16 rows.
const NAIVE_THREADS: [u32; 3] = [16, 16, 1];
/// Auto picks the tiled kernel from this much work (m·n·k·batch) on, when m and n are at least
/// [`TILED_MIN_SIDE`]: below it, or for thinner products, most of a tile would be idle.
const TILED_MIN_WORK: usize = 32 * 32 * 32;
const TILED_MIN_SIDE: usize = 8;

/// The largest m, n, k or batch: cuBLAS takes them as `int`, and the kernels index in 32 bits.
const MAX_DIM: usize = i32::MAX as usize;

/// `GemmParams` in `gemm.metal`, field for field. op(A)[i][kk] is read at `a_offset + batch *
/// a_batch_stride + i * a_row_stride + kk * a_col_stride` of A's binding; op(B)[kk][j] likewise;
/// C[i][j] is at `c_offset + batch * c_batch_stride + i * ldc + j`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub a_offset: u32,
    pub a_row_stride: u32,
    pub a_col_stride: u32,
    pub a_batch_stride: u32,
    pub b_offset: u32,
    pub b_row_stride: u32,
    pub b_col_stride: u32,
    pub b_batch_stride: u32,
    pub c_offset: u32,
    pub ldc: u32,
    pub c_batch_stride: u32,
    pub alpha: f32,
    pub beta: f32,
}
// SAFETY: #[repr(C)], 68 bytes, no pointers; `GemmParams` in MSL.
unsafe impl KernelParams for GemmParams {}

/// `GemmAddRows` in `gemm.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmAddRows {
    pub rows: u32,
    pub cols: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `GemmAddRows` in MSL.
unsafe impl KernelParams for GemmAddRows {}

/// `GemmSoftmaxRows` in `gemm.metal`: `rows` rows `row_stride` apart, the softmax over each one's
/// first `cols`; with `has_bias` not 0 the bias at `[[buffer(2)]]` is added first, laid out as the
/// scores.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmSoftmaxRows {
    pub rows: u32,
    pub row_stride: u32,
    pub cols: u32,
    pub has_bias: u32,
}
// SAFETY: #[repr(C)], 16 bytes, no pointers; `GemmSoftmaxRows` in MSL.
unsafe impl KernelParams for GemmSoftmaxRows {}

/// Which kernel a GEMM runs on: MAKEPAD_METAL_GEMM=naive|tiled|auto|mps (auto, the default,
/// chooses by shape; mps is Metal Performance Shaders' `MPSMatrixMultiplication`, `mps.rs`), or
/// [`set_mode_override`] on the calling thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    Naive,
    Tiled,
    Auto,
    Mps,
}

impl GemmMode {
    /// A value of MAKEPAD_METAL_GEMM; anything else is an error, never a quiet `Auto`.
    pub fn parse(value: &str) -> MetalResult<GemmMode> {
        match value.trim() {
            "naive" => Ok(GemmMode::Naive),
            "tiled" => Ok(GemmMode::Tiled),
            "auto" | "" => Ok(GemmMode::Auto),
            "mps" => Ok(GemmMode::Mps),
            other => Err(format!("MAKEPAD_METAL_GEMM={other}: expected naive, tiled, auto or mps")),
        }
    }
}

/// What a GEMM runs on: one of `gemm.metal`'s kernels, or MPS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmVariant {
    Naive,
    Tiled,
    Mps,
}

impl GemmVariant {
    /// The `gemm.metal` kernel it runs on; `None` for MPS.
    pub fn kernel(self) -> Option<&'static Kernel> {
        match self {
            GemmVariant::Naive => Some(&GEMM_NAIVE),
            GemmVariant::Tiled => Some(&GEMM_TILED),
            GemmVariant::Mps => None,
        }
    }

    /// What `mode` gives an m x n x k product in `batch` batches: auto takes the tiled kernel
    /// for enough work with both sides at least 8, the naive one otherwise (and for k = 0, which
    /// only writes beta·C; mps too, so MPS always multiplies).
    pub fn choose(mode: GemmMode, m: usize, n: usize, k: usize, batch: usize) -> GemmVariant {
        match mode {
            GemmMode::Naive => GemmVariant::Naive,
            GemmMode::Tiled => GemmVariant::Tiled,
            GemmMode::Mps if k > 0 => GemmVariant::Mps,
            GemmMode::Mps => GemmVariant::Naive,
            GemmMode::Auto => {
                let work = m.saturating_mul(n).saturating_mul(k).saturating_mul(batch);
                if k > 0 && m >= TILED_MIN_SIDE && n >= TILED_MIN_SIDE && work >= TILED_MIN_WORK {
                    GemmVariant::Tiled
                } else {
                    GemmVariant::Naive
                }
            }
        }
    }
}

thread_local! {
    static MODE_OVERRIDE: Cell<Option<GemmMode>> = const { Cell::new(None) };
}

/// Runs the calling thread's GEMMs on `mode` whatever MAKEPAD_METAL_GEMM says (`None`: the
/// variable again), so a test or a bench compares both kernels in one process.
pub fn set_mode_override(mode: Option<GemmMode>) {
    MODE_OVERRIDE.with(|cell| cell.set(mode));
}

/// The calling thread's mode: its override, else MAKEPAD_METAL_GEMM (read once; unset is auto).
pub fn mode() -> MetalResult<GemmMode> {
    if let Some(mode) = MODE_OVERRIDE.with(Cell::get) {
        return Ok(mode);
    }
    static FROM_ENV: OnceLock<MetalResult<GemmMode>> = OnceLock::new();
    FROM_ENV
        .get_or_init(|| match std::env::var("MAKEPAD_METAL_GEMM") {
            Ok(value) => GemmMode::parse(&value),
            Err(_) => Ok(GemmMode::Auto),
        })
        .clone()
}

/// `v` as one of a GEMM's sizes (m, n, k or batch), or `Err` naming `op` and `what`: cuBLAS
/// takes them as `int`. Callers check their sizes with it before `with_metal_stream`.
pub fn gemm_dim(op: &str, what: &str, v: usize) -> MetalResult<u32> {
    if v > MAX_DIM {
        return Err(format!("{op}: gemm {what} {v} exceeds the kernels' sizes (at most {MAX_DIM})"));
    }
    Ok(v as u32)
}

/// `v` as a float offset, leading dimension or stride, or `Err`.
fn gemm_word(op: &str, what: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: gemm {what} {v} does not fit the kernels' 32-bit offsets"))
}

/// One operand as the plan sees it: the floats its binding holds from where it is bound, and how
/// the GEMM reads it (C's `trans` is false).
#[derive(Clone, Copy, Debug)]
pub(crate) struct OperandPlan {
    pub floats: usize,
    pub offset: usize,
    pub ld: usize,
    pub trans: bool,
    pub batch_stride: usize,
}

/// A GEMM as the plan sees it, its bindings reduced to their sizes. `c_reads`: C is bound
/// `InOut`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub alpha: f32,
    pub beta: f32,
    pub a: OperandPlan,
    pub b: OperandPlan,
    pub c: OperandPlan,
    pub c_reads: bool,
}

/// What a GEMM launches: the kernel, its params and its grid (empty when m, n or batch is 0).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmLaunch {
    pub variant: GemmVariant,
    pub params: GemmParams,
    pub grid: Grid,
}

/// An operand's offset and its row, column and batch strides as the kernels take them, after
/// checking it: its leading dimension covers its stored `rows x cols`, and the last element read
/// (`verb`) is inside what its binding holds.
fn operand_words(
    op: &str,
    name: &str,
    verb: &str,
    o: &OperandPlan,
    rows: usize,
    cols: usize,
    batch: usize,
) -> MetalResult<[u32; 4]> {
    if o.ld < cols {
        return Err(format!("{op}: gemm {name}'s leading dimension {} is less than its {cols} columns", o.ld));
    }
    if batch > 0 && rows > 0 && cols > 0 {
        let last = (batch - 1)
            .checked_mul(o.batch_stride)
            .and_then(|v| v.checked_add((rows - 1).checked_mul(o.ld)?))
            .and_then(|v| v.checked_add(o.offset))
            .and_then(|v| v.checked_add(cols - 1))
            .ok_or_else(|| format!("{op}: gemm {name}'s extent overflows"))?;
        if last >= o.floats {
            return Err(format!(
                "{op}: gemm {verb} float {last} of {name} ({rows}x{cols}, ld {}, offset {}, {batch} batches {} apart), \
                 its binding holds {}",
                o.ld, o.offset, o.batch_stride, o.floats
            ));
        }
    }
    let offset = gemm_word(op, &format!("{name} offset"), o.offset)?;
    let ld = gemm_word(op, &format!("{name} leading dimension"), o.ld)?;
    let batch_stride = gemm_word(op, &format!("{name} batch stride"), o.batch_stride)?;
    // Stored row-major with `ld`: untransposed, a row is `ld` apart and a column 1; transposed,
    // the other way round.
    let (row_stride, col_stride) = if o.trans { (1, ld) } else { (ld, 1) };
    Ok([offset, row_stride, col_stride, batch_stride])
}

/// Checks a GEMM and plans its launch on `mode` (module docs: what is refused).
pub(crate) fn plan(op: &str, g: &GemmPlan, mode: GemmMode) -> MetalResult<GemmLaunch> {
    let m = gemm_dim(op, "m", g.m)?;
    let n = gemm_dim(op, "n", g.n)?;
    let k = gemm_dim(op, "k", g.k)?;
    let batch = gemm_dim(op, "batch", g.batch)?;
    if g.beta != 0.0 && !g.c_reads {
        return Err(format!("{op}: gemm with beta {} reads C: bind it Bind::InOut", g.beta));
    }
    // Stored shapes: op(A) is m x k, op(B) k x n.
    let (a_rows, a_cols) = if g.a.trans { (g.k, g.m) } else { (g.m, g.k) };
    let (b_rows, b_cols) = if g.b.trans { (g.n, g.k) } else { (g.k, g.n) };
    let a = operand_words(op, "A", "reads", &g.a, a_rows, a_cols, g.batch)?;
    let b = operand_words(op, "B", "reads", &g.b, b_rows, b_cols, g.batch)?;
    let c_plan = OperandPlan { trans: false, ..g.c };
    let c = operand_words(op, "C", "writes", &c_plan, g.m, g.n, g.batch)?;
    let params = GemmParams {
        m,
        n,
        // alpha 0 reads neither A nor B: beta·C alone, as k = 0 (BLAS's rule).
        k: if g.alpha == 0.0 { 0 } else { k },
        batch,
        a_offset: a[0],
        a_row_stride: a[1],
        a_col_stride: a[2],
        a_batch_stride: a[3],
        b_offset: b[0],
        b_row_stride: b[1],
        b_col_stride: b[2],
        b_batch_stride: b[3],
        c_offset: c[0],
        ldc: c[1],
        c_batch_stride: c[3],
        alpha: g.alpha,
        beta: g.beta,
    };
    let variant = GemmVariant::choose(mode, g.m, g.n, params.k as usize, g.batch);
    let grid = match variant {
        // MPS launches its own; the grid only says whether there is anything to do.
        GemmVariant::Naive | GemmVariant::Mps => Grid::cover([g.n, g.m, g.batch], NAIVE_THREADS)?,
        GemmVariant::Tiled => Grid::groups([g.n.div_ceil(TILE), g.m.div_ceil(TILE), g.batch], [TILED_THREADS, 1, 1])?,
    };
    Ok(GemmLaunch { variant, params, grid })
}

#[cfg(target_os = "macos")]
use super::{Bind, Gemm, GemmOperand, MetalStream};

/// `operand`'s plan: what its binding holds from where it is bound.
#[cfg(target_os = "macos")]
fn operand_plan(op: &str, name: &str, operand: &GemmOperand<'_>) -> MetalResult<OperandPlan> {
    let floats = match operand.src {
        Bind::In(t) => t.rows() * t.cols(),
        Bind::Weights(w, offset) => w.floats().saturating_sub(offset),
        _ => return Err(format!("{op}: gemm reads {name} through Bind::In or Bind::Weights")),
    };
    Ok(OperandPlan {
        floats,
        offset: operand.offset,
        ld: operand.ld,
        trans: operand.trans,
        batch_stride: operand.batch_stride,
    })
}

/// `MetalStream::gemm`: checks `g`, plans it on the calling thread's mode and launches it with A,
/// B and C at `[[buffer(1)]]` to `[[buffer(3)]]`.
#[cfg(target_os = "macos")]
pub(crate) fn run(s: &mut MetalStream<'_>, op: &'static str, g: &Gemm<'_>) -> MetalResult<()> {
    let a = operand_plan(op, "A", &g.a)?;
    let b = operand_plan(op, "B", &g.b)?;
    let (c_floats, c_reads) = match g.c {
        Bind::Out(t) => (t.rows() * t.cols(), false),
        Bind::InOut(t) => (t.rows() * t.cols(), true),
        _ => return Err(format!("{op}: gemm writes C through Bind::Out or Bind::InOut")),
    };
    let c = OperandPlan { floats: c_floats, offset: g.c_offset, ld: g.ldc, trans: false, batch_stride: g.c_batch_stride };
    let shape = GemmPlan { m: g.m, n: g.n, k: g.k, batch: g.batch, alpha: g.alpha, beta: g.beta, a, b, c, c_reads };
    let launch = plan(op, &shape, mode().map_err(|e| format!("{op}: {e}"))?)?;
    let Some(kernel) = launch.variant.kernel() else {
        return if launch.grid.is_empty() { Ok(()) } else { s.gemm_mps(g, &shape) };
    };
    if launch.variant == GemmVariant::Tiled {
        check_simd_width(s, op, kernel)?;
    }
    s.launch(kernel, &[], &launch.params, &[g.a.src, g.b.src, g.c], launch.grid)
}

/// `Err` unless `kernel`'s SIMD groups are [`SIMD_WIDTH`] wide, as the tiled GEMM and the
/// softmax's reductions are written for.
#[cfg(target_os = "macos")]
pub(crate) fn check_simd_width(s: &mut MetalStream<'_>, op: &str, kernel: &Kernel) -> MetalResult<()> {
    let limits = s.limits(kernel, &[])?;
    if limits.simd_width != SIMD_WIDTH {
        return Err(format!(
            "{op} ({}): SIMD groups of {} threads; the kernel is written for {SIMD_WIDTH}",
            kernel.function, limits.simd_width
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operand(floats: usize, offset: usize, ld: usize, trans: bool, batch_stride: usize) -> OperandPlan {
        OperandPlan { floats, offset, ld, trans, batch_stride }
    }

    /// A plain m x n x k product, every operand packed and exactly as big as it needs.
    fn packed(m: usize, n: usize, k: usize) -> GemmPlan {
        GemmPlan {
            m,
            n,
            k,
            batch: 1,
            alpha: 1.0,
            beta: 0.0,
            a: operand(m * k, 0, k, false, 0),
            b: operand(k * n, 0, n, false, 0),
            c: operand(m * n, 0, n, false, 0),
            c_reads: false,
        }
    }

    #[test]
    fn params_structs_have_their_msl_sizes() {
        assert_eq!(std::mem::size_of::<GemmParams>(), 68);
        assert_eq!(std::mem::size_of::<GemmAddRows>(), 8);
        assert_eq!(std::mem::size_of::<GemmSoftmaxRows>(), 16);
    }

    /// The fields of `struct name { ... };` in `source`, as (type, name).
    fn msl_fields(source: &str, name: &str) -> Vec<(String, String)> {
        let start = source.find(&format!("struct {name} {{")).unwrap_or_else(|| panic!("no struct {name}"));
        let body = &source[start..];
        let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
        body.lines()
            .map(|line| line.split("//").next().unwrap().trim())
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut words = line.trim_end_matches(';').split_whitespace();
                (words.next().unwrap().to_string(), words.next().unwrap().to_string())
            })
            .collect()
    }

    #[test]
    fn params_are_their_msl_twins_field_for_field() {
        let fields = |list: &[&str], ty: &str| list.iter().map(|f| (ty.to_string(), f.to_string())).collect::<Vec<_>>();
        let mut gemm = fields(
            &[
                "m", "n", "k", "batch", "a_offset", "a_row_stride", "a_col_stride", "a_batch_stride", "b_offset",
                "b_row_stride", "b_col_stride", "b_batch_stride", "c_offset", "ldc", "c_batch_stride",
            ],
            "uint",
        );
        gemm.extend(fields(&["alpha", "beta"], "float"));
        assert_eq!(msl_fields(GEMM.source, "GemmParams"), gemm);
        assert_eq!(msl_fields(GEMM.source, "GemmAddRows"), fields(&["rows", "cols"], "uint"));
        assert_eq!(
            msl_fields(GEMM.source, "GemmSoftmaxRows"),
            fields(&["rows", "row_stride", "cols", "has_bias"], "uint")
        );
        // The tile the grid is planned for is the kernel's.
        assert!(GEMM.source.contains(&format!("#define GEMM_TILE {TILE}u")));
        assert!(GEMM.source.contains(&format!("#define GEMM_TILED_THREADS {TILED_THREADS}u")));
    }

    #[test]
    fn every_gemm_kernel_is_in_the_library_and_it_includes_the_header() {
        for kernel in GEMM_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &GEMM));
            assert!(GEMM.source.contains(&format!("kernel void {}(", kernel.function)), "{}", kernel.function);
        }
        assert!(GEMM.source.contains("#include \"makepad_common.h\""));
        // D8: no half anywhere, no fast math, no simd_sum.
        for banned in ["half", "fast::", "simd_sum"] {
            let code: String = GEMM.source.lines().map(|line| line.split("//").next().unwrap()).collect();
            assert!(!code.contains(banned), "gemm.metal's code uses {banned}");
        }
    }

    #[test]
    fn mode_parses_its_four_values_and_nothing_else() {
        assert_eq!(GemmMode::parse("naive"), Ok(GemmMode::Naive));
        assert_eq!(GemmMode::parse("tiled"), Ok(GemmMode::Tiled));
        assert_eq!(GemmMode::parse("auto"), Ok(GemmMode::Auto));
        assert_eq!(GemmMode::parse("mps"), Ok(GemmMode::Mps));
        assert_eq!(GemmMode::parse(" tiled\n"), Ok(GemmMode::Tiled));
        assert_eq!(GemmMode::parse(""), Ok(GemmMode::Auto));
        let err = GemmMode::parse("cublas").unwrap_err();
        assert!(err.contains("MAKEPAD_METAL_GEMM=cublas") && err.contains("naive, tiled, auto or mps"), "{err}");
    }

    #[test]
    fn the_override_is_the_threads_own() {
        set_mode_override(Some(GemmMode::Tiled));
        assert_eq!(mode(), Ok(GemmMode::Tiled));
        std::thread::spawn(|| {
            set_mode_override(Some(GemmMode::Naive));
            assert_eq!(mode(), Ok(GemmMode::Naive));
        })
        .join()
        .unwrap();
        assert_eq!(mode(), Ok(GemmMode::Tiled), "another thread's override is its own");
        set_mode_override(None);
    }

    #[test]
    fn auto_tiles_products_that_fill_tiles() {
        use GemmVariant::*;
        let auto = |m, n, k, batch| GemmVariant::choose(GemmMode::Auto, m, n, k, batch);
        assert_eq!(auto(256, 256, 256, 1), Tiled);
        assert_eq!(auto(32, 32, 32, 1), Tiled);
        assert_eq!(auto(16, 336, 32, 8), Tiled, "attention's scores: 16 queries, 8 heads");
        assert_eq!(auto(31, 32, 32, 1), Naive, "below the work");
        assert_eq!(auto(4, 4, 4, 1), Naive);
        assert_eq!(auto(7, 4096, 4096, 1), Naive, "a side under 8");
        assert_eq!(auto(4096, 1, 4096, 1), Naive);
        assert_eq!(auto(256, 256, 0, 1), Naive, "k = 0 only writes beta C");
        assert_eq!(auto(usize::MAX, usize::MAX, 2, 1), Tiled, "the work saturates");
        assert_eq!(GemmVariant::choose(GemmMode::Naive, 256, 256, 256, 1), Naive);
        assert_eq!(GemmVariant::choose(GemmMode::Tiled, 1, 1, 0, 1), Tiled);
        assert_eq!(GemmVariant::choose(GemmMode::Mps, 7, 3, 5, 1), Mps, "MPS takes every shape");
        assert_eq!(GemmVariant::choose(GemmMode::Mps, 256, 256, 0, 1), Naive, "k = 0 only writes beta C");
        assert_eq!(Naive.kernel().unwrap().function, "gemm_naive_f32");
        assert_eq!(Tiled.kernel().unwrap().function, "gemm_tiled_f32");
        assert!(Mps.kernel().is_none());
    }

    #[test]
    fn plan_sets_strides_from_ld_and_transposes() {
        let mut g = packed(5, 7, 3);
        g.a = operand(100, 4, 9, false, 0);
        g.b = operand(100, 2, 11, true, 0); // stored 7 x 3 (n x k), ld 11
        g.c = operand(100, 1, 8, false, 0);
        let launch = plan("op", &g, GemmMode::Naive).unwrap();
        let p = launch.params;
        assert_eq!((p.m, p.n, p.k, p.batch), (5, 7, 3, 1));
        assert_eq!((p.a_offset, p.a_row_stride, p.a_col_stride), (4, 9, 1));
        assert_eq!((p.b_offset, p.b_row_stride, p.b_col_stride), (2, 1, 11));
        assert_eq!((p.c_offset, p.ldc), (1, 8));
        assert_eq!((p.alpha, p.beta), (1.0, 0.0));
        assert_eq!(launch.variant, GemmVariant::Naive);
        assert_eq!(launch.grid.groups, [1, 1, 1]);
        assert_eq!(launch.grid.threads, NAIVE_THREADS);
        // A transposed A is stored k x m.
        let mut g = packed(5, 7, 3);
        g.a = operand(15, 0, 5, true, 0);
        let p = plan("op", &g, GemmMode::Naive).unwrap().params;
        assert_eq!((p.a_row_stride, p.a_col_stride), (1, 5));
        assert_eq!((p.b_row_stride, p.b_col_stride), (7, 1));
    }

    #[test]
    fn plan_grids_cover_the_output_for_each_kernel() {
        let mut g = packed(65, 33, 40);
        g.batch = 3;
        g.a.batch_stride = 65 * 40;
        g.a.floats *= 3;
        g.c.batch_stride = 65 * 33;
        g.c.floats *= 3;
        let naive = plan("op", &g, GemmMode::Naive).unwrap();
        assert_eq!(naive.grid.groups, [3, 5, 3]);
        let tiled = plan("op", &g, GemmMode::Tiled).unwrap();
        assert_eq!(tiled.variant, GemmVariant::Tiled);
        assert_eq!(tiled.grid.groups, [2, 3, 3]);
        assert_eq!(tiled.grid.threads, [TILED_THREADS, 1, 1]);
        assert_eq!(tiled.grid.threadgroup_bytes, [0; 4], "the tiles are the kernel's own");
        assert_eq!(tiled.params.batch, 3);
        // Nothing to write: an empty grid, whichever kernel.
        for (m, n, batch) in [(0, 5, 1), (5, 0, 1), (5, 5, 0)] {
            let mut g = packed(m, n, 4);
            g.batch = batch;
            for mode in [GemmMode::Naive, GemmMode::Tiled] {
                assert!(plan("op", &g, mode).unwrap().grid.is_empty(), "{m}x{n} batch {batch}");
            }
        }
    }

    #[test]
    fn k_zero_and_alpha_zero_read_neither_operand() {
        // k = 0: the operands may be empty and their ld 0; C is still written.
        let mut g = packed(4, 3, 0);
        g.a = operand(0, 0, 0, false, 0);
        g.b = operand(0, 0, 3, false, 0);
        let launch = plan("op", &g, GemmMode::Auto).unwrap();
        assert_eq!(launch.params.k, 0);
        assert!(!launch.grid.is_empty());
        // alpha = 0 runs as k = 0 (the operands are still checked).
        let mut g = packed(4, 3, 5);
        g.alpha = 0.0;
        assert_eq!(plan("op", &g, GemmMode::Naive).unwrap().params.k, 0);
        assert_eq!(plan("op", &g, GemmMode::Auto).unwrap().variant, GemmVariant::Naive);
        g.a.floats -= 1;
        assert!(plan("op", &g, GemmMode::Naive).is_err());
    }

    #[test]
    fn beta_reads_c_only_through_inout() {
        let mut g = packed(4, 4, 4);
        g.beta = 0.5;
        let err = plan("op", &g, GemmMode::Naive).unwrap_err();
        assert!(err.starts_with("op: gemm with beta 0.5") && err.contains("Bind::InOut"), "{err}");
        g.c_reads = true;
        assert_eq!(plan("op", &g, GemmMode::Naive).unwrap().params.beta, 0.5);
        g.beta = -0.0;
        g.c_reads = false;
        assert!(plan("op", &g, GemmMode::Naive).is_ok(), "-0 is 0");
    }

    #[test]
    fn leading_dimensions_cover_the_stored_columns() {
        let mut g = packed(5, 7, 3);
        g.a.ld = 2;
        let err = plan("op", &g, GemmMode::Naive).unwrap_err();
        assert!(err.contains("A's leading dimension 2 is less than its 3 columns"), "{err}");
        let mut g = packed(5, 7, 3);
        g.b = operand(100, 0, 2, true, 0); // stored 7 x 3
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("B's leading dimension 2"));
        let mut g = packed(5, 7, 3);
        g.c.ld = 6;
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("C's leading dimension 6"));
        // A transposed A of m = 5 is stored with 5 columns.
        let mut g = packed(5, 7, 3);
        g.a = operand(100, 0, 4, true, 0);
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("less than its 5 columns"));
    }

    #[test]
    fn operands_are_read_and_written_only_inside_their_bindings() {
        // Exactly inside: the last row ends at the last float.
        let mut g = packed(5, 7, 3);
        g.a = operand(4 * 10 + 3 + 6, 6, 10, false, 0);
        assert!(plan("op", &g, GemmMode::Naive).is_ok());
        g.a.floats -= 1;
        let err = plan("op", &g, GemmMode::Naive).unwrap_err();
        assert!(err.contains("reads float 48 of A") && err.contains("holds 48"), "{err}");
        // Batches count: the last batch's last element.
        let mut g = packed(2, 2, 2);
        g.batch = 3;
        g.b = operand(2 * 4 + 4, 0, 2, false, 4);
        g.a.batch_stride = 0; // broadcast, as a convolution's weights
        g.c = operand(3 * 4, 0, 2, false, 4);
        assert!(plan("op", &g, GemmMode::Naive).is_ok());
        g.c.floats -= 1;
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("writes float 11 of C"));
        // Packed heads: ld the hidden width, a head's columns head_dim apart.
        let (q_len, heads, head_dim) = (6, 4, 8);
        let hidden = heads * head_dim;
        let mut g = packed(q_len, q_len, head_dim);
        g.batch = heads;
        g.a = operand(q_len * hidden, 0, hidden, false, head_dim);
        g.b = operand(q_len * hidden, 0, hidden, true, head_dim);
        g.c = operand(heads * q_len * q_len, 0, q_len, false, q_len * q_len);
        assert!(plan("op", &g, GemmMode::Tiled).is_ok());
        g.a.offset = 1;
        assert!(plan("op", &g, GemmMode::Tiled).is_err());
        // Overflow is an error, not a wrap.
        let mut g = packed(2, 2, 2);
        g.a.offset = usize::MAX;
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("overflows"));
    }

    #[test]
    fn sizes_past_32_bits_are_refused() {
        let err = gemm_dim("op", "m", MAX_DIM + 1).unwrap_err();
        assert!(err.starts_with("op: gemm m") && err.contains("exceeds"), "{err}");
        assert_eq!(gemm_dim("op", "m", MAX_DIM), Ok(i32::MAX as u32));
        let mut g = packed(1, 1, 1);
        g.a = operand(usize::MAX, (u32::MAX as usize) + 1, 1, false, 0);
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("A offset"));
        let mut g = packed(1, 1, 1);
        g.batch = MAX_DIM + 1;
        assert!(plan("op", &g, GemmMode::Naive).unwrap_err().contains("gemm batch"));
    }
}
