//! The generic `gpu_*` ops Cutie calls, on Makepad's own Metal kernels (RationalVisions
//! docs/cutie-metal.md §9, step 8), at the signatures of models/common's macOS `gpu_*` functions
//! (the same as CUDA's in cuda/src/launch.rs), with CUDA's checks and messages:
//! - `slice_cols` and `concat_cols`: copies of blocks of words (`tensor.metal`'s
//!   `tensor_copy2d_u32`, CUDA's `makepad_cuda_copy_submatrix_f32`);
//! - `pixal_pool`, `planar_tokens_transpose` (and BiRefNet's `tokens_to_planar`, the same
//!   transpose), `realesrgan_scale_add`, `add_cols_broadcast` and `gather_cols`:
//!   `generic_ops.metal`, ported from CUDA's kernels with their arithmetic, so every result is
//!   CUDA's to the bit (moves are moves, the pool sums its window in row order and divides once,
//!   the scale add is CUDA's one `fmaf`, the broadcast add one IEEE add);
//! - `concat_rows` and `concat_rows_many`: parts adjacent in one storage, in order, are a view of
//!   it, as on CUDA; otherwise one copy (`tensor_copy2d_u32`) a part into a new tensor.
//!
//! The library is apart from `tensor.metal`, so a fault in it breaks these ops alone, not the
//! stream's fills and copies. models/common's gpu.rs calls them on
//! macOS, or the host versions in `gpu_tensor.rs` where MAKEPAD_CUTIE_METAL_REF names the op.
//!
//! Every check and u32 conversion is made before anything is encoded, so a failure is an `Err`
//! with nothing launched; where CUDA's kernels count in size_t, a size past the kernels' u32 is an
//! `Err` here. The library, its params and the checks compile on every OS and are tested anywhere;
//! the ops are macOS only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::kernels::TensorCopy2d;
use crate::stream::{Grid, Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::stream::kernels::TENSOR_COPY2D;
#[cfg(target_os = "macos")]
use crate::stream::{view, with_metal_stream, Bind};
#[cfg(target_os = "macos")]
use crate::GpuTensor;
#[cfg(target_os = "macos")]
use std::sync::Arc;

/// The pool, the transpose, the scale add, the broadcast add and the column gather
/// (`generic_ops.metal`).
pub static GENERIC_OPS: KernelLib = KernelLib {
    name: "generic_ops",
    source: include_str!("../shaders/makepad/generic_ops.metal"),
};

/// Adaptive average pooling of planar tensors, one thread an output ([`GenericPixalPool`]).
pub static GENERIC_PIXAL_POOL: Kernel = Kernel { lib: &GENERIC_OPS, function: "generic_pixal_pool_f32" };
/// `[rows, cols]` to `[cols, rows]`, a word moved a thread ([`GenericTranspose`]).
pub static GENERIC_TRANSPOSE: Kernel = Kernel { lib: &GENERIC_OPS, function: "generic_transpose_u32" };
/// `fma(scale, delta, base)` ([`GenericScaleAdd`]).
pub static GENERIC_SCALE_ADD: Kernel = Kernel { lib: &GENERIC_OPS, function: "generic_scale_add_f32" };
/// `x + bias[col]` over `[rows, cols]` ([`GenericAddColsBroadcast`]).
pub static GENERIC_ADD_COLS_BROADCAST: Kernel =
    Kernel { lib: &GENERIC_OPS, function: "generic_add_cols_broadcast_f32" };
/// Columns of `[rows, cols]` by an index table, a word moved a thread ([`GenericGatherCols`]).
pub static GENERIC_GATHER_COLS: Kernel = Kernel { lib: &GENERIC_OPS, function: "generic_gather_cols_u32" };

/// Every kernel above, for the bring-up test that compiles them all.
pub static GENERIC_KERNELS: &[&Kernel] = &[
    &GENERIC_PIXAL_POOL,
    &GENERIC_TRANSPOSE,
    &GENERIC_SCALE_ADD,
    &GENERIC_ADD_COLS_BROADCAST,
    &GENERIC_GATHER_COLS,
];

/// Threads to a threadgroup for the one-thread-an-output kernels.
const THREADS: u32 = 256;
/// The copy's threadgroup: 32 columns by 8 rows, so a part one column wide idles less of it.
const COPY_THREADS: [u32; 3] = [32, 8, 1];
/// The largest side `gpu_pixal_pool` takes (CUDA's), so its window arithmetic fits a u32.
const PIXAL_POOL_MAX_SIDE: usize = 65535;

/// `generic_pixal_pool_f32`'s params: the input's plane, the output's, and the outputs over every
/// channel (`out_width * out_height * channels`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenericPixalPool {
    pub width: u32,
    pub height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 20 bytes, no pointers; `GenericPixalPool` in MSL.
unsafe impl KernelParams for GenericPixalPool {}

/// `generic_transpose_u32`'s params: the input's shape and its element count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenericTranspose {
    pub rows: u32,
    pub cols: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 12 bytes, no pointers; `GenericTranspose` in MSL.
unsafe impl KernelParams for GenericTranspose {}

/// `generic_scale_add_f32`'s params: the element count and the scale.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenericScaleAdd {
    pub n: u32,
    pub scale: f32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `GenericScaleAdd` in MSL.
unsafe impl KernelParams for GenericScaleAdd {}

/// `generic_add_cols_broadcast_f32`'s params: the columns (the bias's length) and the element
/// count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenericAddColsBroadcast {
    pub cols: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `GenericAddColsBroadcast` in MSL.
unsafe impl KernelParams for GenericAddColsBroadcast {}

/// `generic_gather_cols_u32`'s params: the input's columns, the indices' count (the output's
/// columns) and the output's element count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenericGatherCols {
    pub cols: u32,
    pub count: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 12 bytes, no pointers; `GenericGatherCols` in MSL.
unsafe impl KernelParams for GenericGatherCols {}

/// A size as the kernels take it (the message of CUDA's `cutie_u32`).
fn kernel_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// The element count of a `[rows, cols]` tensor as a kernel's u32 index.
fn kernel_count(op: &str, rows: usize, cols: usize) -> MetalResult<u32> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{op}: {rows}x{cols} does not fit the kernel's 32-bit sizes"))?;
    kernel_u32(op, n)
}

/// One launch of the copy: its params and its grid.
#[derive(Clone, Copy, Debug)]
struct CopyBlock {
    params: TensorCopy2d,
    grid: Grid,
}

/// The copy of a `rows x cols` block of words from `src_offset` of a source `src_ld` words wide to
/// `dst_offset` of a destination `dst_ld` wide.
fn copy_block(
    op: &str,
    rows: usize,
    cols: usize,
    src_ld: usize,
    src_offset: usize,
    dst_ld: usize,
    dst_offset: usize,
) -> MetalResult<CopyBlock> {
    let d = |v: usize| kernel_u32(op, v);
    let params = TensorCopy2d {
        rows: d(rows)?,
        cols: d(cols)?,
        src_ld: d(src_ld)?,
        dst_ld: d(dst_ld)?,
        src_offset: d(src_offset)?,
        dst_offset: d(dst_offset)?,
    };
    let grid = Grid::cover([cols, rows, 1], COPY_THREADS).map_err(|e| format!("{op}: {e}"))?;
    Ok(CopyBlock { params, grid })
}

/// `gpu_slice_cols`'s copy of columns `start..start + len` of a `[rows, cols]` tensor, after
/// CUDA's check.
fn slice_cols_plan(rows: usize, cols: usize, start: usize, len: usize) -> MetalResult<CopyBlock> {
    let op = "gpu_slice_cols";
    if start.checked_add(len).is_none_or(|end| end > cols) {
        return Err("gpu_slice_cols out of range".to_string());
    }
    kernel_count(op, rows, cols)?;
    copy_block(op, rows, len, cols, start, len, 0)
}

/// `gpu_concat_cols`'s rows, its columns in all, and one copy a part (each into its columns of
/// the output), after CUDA's check: at least one row, the same in every part.
fn concat_cols_plan(parts: &[(usize, usize)]) -> MetalResult<(usize, usize, Vec<CopyBlock>)> {
    let op = "gpu_concat_cols";
    let rows = parts.first().map_or(0, |&(rows, _)| rows);
    if rows == 0 || parts.iter().any(|&(r, _)| r != rows) {
        return Err("gpu_concat_cols shape mismatch".to_string());
    }
    let total = parts
        .iter()
        .try_fold(0usize, |total, &(_, cols)| total.checked_add(cols))
        .ok_or_else(|| format!("{op}: the parts' columns overflow"))?;
    kernel_count(op, rows, total)?;
    let mut at = 0;
    let mut copies = Vec::with_capacity(parts.len());
    for &(_, cols) in parts {
        copies.push(copy_block(op, rows, cols, cols, 0, total, at)?);
        at += cols;
    }
    Ok((rows, total, copies))
}

/// `gpu_pixal_pool`'s params for a `[rows, cols]` input of `width x height` planes to
/// `out_width x out_height`, after CUDA's checks.
fn pixal_pool_params(
    rows: usize,
    cols: usize,
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> MetalResult<GenericPixalPool> {
    let op = "gpu_pixal_pool";
    if width == 0
        || height == 0
        || out_width == 0
        || out_height == 0
        || width.checked_mul(height) != Some(cols)
        || out_width.checked_mul(out_height).is_none()
        || [width, height, out_width, out_height, rows].iter().any(|&n| n > PIXAL_POOL_MAX_SIDE)
    {
        return Err("gpu_pixal_pool shape/dtype mismatch".to_string());
    }
    kernel_count(op, rows, cols)?;
    let n = kernel_count(op, rows, out_width * out_height)?;
    // Every side is at most 65535 (checked above): they and the window's arithmetic fit a u32.
    Ok(GenericPixalPool {
        width: width as u32,
        height: height as u32,
        out_width: out_width as u32,
        out_height: out_height as u32,
        n,
    })
}

/// `gpu_planar_tokens_transpose`'s params for a `[rows, cols]` input.
fn transpose_params(rows: usize, cols: usize) -> MetalResult<GenericTranspose> {
    let op = "gpu_planar_tokens_transpose";
    let n = kernel_count(op, rows, cols)?;
    Ok(GenericTranspose { rows: rows as u32, cols: cols as u32, n })
}

/// `gpu_add_cols_broadcast`'s params for `x` `[rows, cols]` and a bias of `bias` (rows, cols),
/// after CUDA's check: as many bias values as `x` has columns.
fn add_cols_broadcast_params(x: (usize, usize), bias: (usize, usize)) -> MetalResult<GenericAddColsBroadcast> {
    if bias.0.checked_mul(bias.1) != Some(x.1) {
        return Err(format!(
            "gpu_add_cols_broadcast shape mismatch {}x{} half=false vs bias {}x{} half=false",
            x.0, x.1, bias.0, bias.1
        ));
    }
    let n = kernel_count("gpu_add_cols_broadcast", x.0, x.1)?;
    // The columns are at most the element count, which fits a u32 (just checked), or there are no
    // rows and nothing runs.
    let cols = kernel_u32("gpu_add_cols_broadcast", x.1)?;
    Ok(GenericAddColsBroadcast { cols, n })
}

/// `gpu_gather_cols`'s params for `x` `[rows, cols]` and `indices`, after CUDA's checks, in its
/// order and with its messages: a table, and every index inside the columns.
fn gather_cols_params(rows: usize, cols: usize, indices: &[u32]) -> MetalResult<GenericGatherCols> {
    if indices.is_empty() {
        return Err("gpu_gather_cols: empty index table".to_string());
    }
    if let Some(&index) = indices.iter().find(|&&index| index as usize >= cols) {
        return Err(format!("gpu_gather_cols index {index} out of range (cols {cols})"));
    }
    let op = "gpu_gather_cols";
    kernel_count(op, rows, cols)?;
    let n = kernel_count(op, rows, indices.len())?;
    Ok(GenericGatherCols { cols: kernel_u32(op, cols)?, count: kernel_u32(op, indices.len())?, n })
}

/// `gpu_concat_rows_many`'s plan for parts of these row counts, all `cols` wide: one copy a part
/// into its rows of the output (an empty grid for a part with no rows).
fn concat_rows_plan(op: &str, cols: usize, rows: &[usize]) -> MetalResult<Vec<CopyBlock>> {
    let total = rows
        .iter()
        .try_fold(0usize, |total, &r| total.checked_add(r))
        .ok_or_else(|| format!("{op}: the parts' rows overflow"))?;
    kernel_count(op, total, cols)?;
    let mut at = 0;
    let mut copies = Vec::with_capacity(rows.len());
    for &r in rows {
        copies.push(copy_block(op, r, cols, cols, 0, cols, at * cols)?);
        at += r;
    }
    Ok(copies)
}

/// `gpu_realesrgan_scale_add`'s params for `base` and `delta` of these shapes, after CUDA's check.
fn scale_add_params(base: (usize, usize), delta: (usize, usize), scale: f32) -> MetalResult<GenericScaleAdd> {
    if base != delta {
        return Err(format!("gpu_realesrgan_scale_add {}x{} vs {}x{}", base.0, base.1, delta.0, delta.1));
    }
    let n = kernel_count("gpu_realesrgan_scale_add", base.0, base.1)?;
    Ok(GenericScaleAdd { n, scale })
}

/// Columns `start..start + len` of `x` [rows, cols] into a new `[rows, len]` (CUDA's
/// `gpu_slice_cols`).
#[cfg(target_os = "macos")]
pub fn slice_cols(x: &GpuTensor, start: usize, len: usize) -> MetalResult<GpuTensor> {
    let op = "gpu_slice_cols";
    let copy = slice_cols_plan(x.rows(), x.cols(), start, len)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), len)?;
        s.launch(&TENSOR_COPY2D, &[], &copy.params, &[Bind::In(x), Bind::Out(&out)], copy.grid)?;
        Ok(out)
    })
}

/// `parts` side by side: `[rows, sum of their cols]`, each part's columns after the one before's
/// (CUDA's `gpu_concat_cols`). A part may appear more than once.
#[cfg(target_os = "macos")]
pub fn concat_cols(parts: &[&GpuTensor]) -> MetalResult<GpuTensor> {
    let op = "gpu_concat_cols";
    if parts.iter().any(|part| part.is_half()) {
        return Err("gpu_concat_cols shape mismatch".to_string());
    }
    let shapes: Vec<(usize, usize)> = parts.iter().map(|part| (part.rows(), part.cols())).collect();
    let (rows, total, copies) = concat_cols_plan(&shapes)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(rows, total)?;
        for (part, copy) in parts.iter().zip(&copies) {
            s.launch(&TENSOR_COPY2D, &[], &copy.params, &[Bind::In(part), Bind::Out(&out)], copy.grid)?;
        }
        Ok(out)
    })
}

/// Adaptive average pooling of planar `x` [channels, height*width] to [channels,
/// out_height*out_width] (CUDA's `gpu_pixal_pool`): each output the mean of its window
/// `[floor(i*in/out), ceil((i+1)*in/out))`, summed in row order.
#[cfg(target_os = "macos")]
pub fn pixal_pool(
    x: &GpuTensor,
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> MetalResult<GpuTensor> {
    let op = "gpu_pixal_pool";
    if x.is_half() {
        return Err("gpu_pixal_pool shape/dtype mismatch".to_string());
    }
    let params = pixal_pool_params(x.rows(), x.cols(), width, height, out_width, out_height)?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), out_width * out_height)?;
        s.launch(&GENERIC_PIXAL_POOL, &[], &params, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// `x` [rows, cols] transposed to [cols, rows]: planar [C, HW] to tokens [HW, C] and back (CUDA's
/// `gpu_planar_tokens_transpose`).
#[cfg(target_os = "macos")]
pub fn planar_tokens_transpose(x: &GpuTensor) -> MetalResult<GpuTensor> {
    transpose("gpu_planar_tokens_transpose", x)
}

/// BiRefNet's tokens `[HW, C]` to planar `[C, HW]` (CUDA's `gpu_birefnet_tokens_to_planar`): the
/// same transpose.
#[cfg(target_os = "macos")]
pub fn tokens_to_planar(x: &GpuTensor) -> MetalResult<GpuTensor> {
    transpose("gpu_birefnet_tokens_to_planar", x)
}

/// `x` transposed, named `op`.
#[cfg(target_os = "macos")]
fn transpose(op: &'static str, x: &GpuTensor) -> MetalResult<GpuTensor> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    let params = transpose_params(x.rows(), x.cols())?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.cols(), x.rows())?;
        s.launch(&GENERIC_TRANSPOSE, &[], &params, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// `base + scale * delta` elementwise with one rounding (CUDA's `gpu_realesrgan_scale_add`, its
/// `fmaf(scale, delta, base)`).
#[cfg(target_os = "macos")]
pub fn realesrgan_scale_add(base: &GpuTensor, delta: &GpuTensor, scale: f32) -> MetalResult<GpuTensor> {
    let op = "gpu_realesrgan_scale_add";
    if base.is_half() || delta.is_half() {
        return Err("gpu_realesrgan_scale_add is f32-only".to_string());
    }
    let params = scale_add_params((base.rows(), base.cols()), (delta.rows(), delta.cols()), scale)?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(base.rows(), base.cols())?;
        s.launch(&GENERIC_SCALE_ADD, &[], &params, &[Bind::In(base), Bind::In(delta), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// `x` [rows, cols] plus `bias` (`cols` values, any shape) on every row (CUDA's
/// `gpu_add_cols_broadcast`).
#[cfg(target_os = "macos")]
pub fn add_cols_broadcast(x: &GpuTensor, bias: &GpuTensor) -> MetalResult<GpuTensor> {
    let op = "gpu_add_cols_broadcast";
    let params = add_cols_broadcast_params((x.rows(), x.cols()), (bias.rows(), bias.cols()))?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        s.launch(&GENERIC_ADD_COLS_BROADCAST, &[], &params, &[Bind::In(x), Bind::In(bias), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// Columns `indices` of `x` [rows, cols], in that order, into [rows, indices.len()] (CUDA's
/// `gpu_gather_cols`); every index is checked first.
#[cfg(target_os = "macos")]
pub fn gather_cols(x: &GpuTensor, indices: &[u32]) -> MetalResult<GpuTensor> {
    let op = "gpu_gather_cols";
    if x.is_half() {
        return Err("gpu_gather_cols is f32-only".to_string());
    }
    let params = gather_cols_params(x.rows(), x.cols(), indices)?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    let table: Vec<u8> = indices.iter().flat_map(|index| index.to_le_bytes()).collect();
    with_metal_stream(op, |s| {
        let idx = s.upload_bytes(&table)?;
        let out = GpuTensor::alloc(x.rows(), indices.len())?;
        s.launch(&GENERIC_GATHER_COLS, &[], &params, &[Bind::In(x), Bind::In(&idx), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// `a`'s rows, then `b`'s (CUDA's `gpu_concat_rows`): a view when `b` follows `a` in one storage,
/// otherwise a copy of each.
#[cfg(target_os = "macos")]
pub fn concat_rows(a: &GpuTensor, b: &GpuTensor) -> MetalResult<GpuTensor> {
    if a.cols() != b.cols() || a.is_half() != b.is_half() {
        return Err("gpu_concat_rows shape mismatch".to_string());
    }
    stack_rows("gpu_concat_rows", &[a, b])
}

/// `parts`' rows one after another (CUDA's `gpu_concat_rows_many`): one part is a view of it;
/// parts adjacent in one storage, in order, are a view of it; otherwise one copy a part.
#[cfg(target_os = "macos")]
pub fn concat_rows_many(parts: &[&GpuTensor]) -> MetalResult<GpuTensor> {
    let Some(first) = parts.first() else {
        return Err("gpu_concat_rows_many empty".to_string());
    };
    if parts.len() == 1 {
        return crate::stream::slice_rows(first, 0, first.rows());
    }
    if parts.iter().any(|part| part.cols() != first.cols() || part.is_half() != first.is_half()) {
        return Err("gpu_concat_rows_many shape mismatch".to_string());
    }
    stack_rows("gpu_concat_rows_many", parts)
}

/// `parts` (all as wide) stacked: a view of their storage when each starts where the one before
/// it ends there, else a new tensor with one copy a part.
#[cfg(target_os = "macos")]
fn stack_rows(op: &'static str, parts: &[&GpuTensor]) -> MetalResult<GpuTensor> {
    let first = parts[0];
    let cols = first.cols();
    let mut next = first.offset;
    let adjacent = parts.iter().all(|part| {
        let follows = Arc::ptr_eq(&part.storage, &first.storage) && part.offset == next;
        next = part.offset + part.rows() * cols;
        follows
    });
    if adjacent {
        // They lie one after another in one storage, so their rows add up inside it.
        let total = parts.iter().map(|part| part.rows()).sum();
        return Ok(view(first, 0, total, cols));
    }
    let rows: Vec<usize> = parts.iter().map(|part| part.rows()).collect();
    let copies = concat_rows_plan(op, cols, &rows)?;
    let total: usize = rows.iter().sum();
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(total, cols)?;
        for (part, copy) in parts.iter().zip(&copies) {
            s.launch(&TENSOR_COPY2D, &[], &copy.params, &[Bind::In(part), Bind::Out(&out)], copy.grid)?;
        }
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    //! What can be checked without a GPU: the library declares its kernels and includes the
    //! header, each params struct is its MSL twin, the checks and messages are CUDA's, the copies
    //! land where CUDA's do, and the pool, as the MSL computes it in u32, is the host version's to
    //! the bit, its window's arithmetic fitting a u32 up to CUDA's limit.
    use super::*;

    /// The fields of `struct name { ... };` in `source`, as (type, name), in order.
    fn msl_struct_fields(source: &str, name: &str) -> Vec<(String, String)> {
        let start = source
            .find(&format!("struct {name} {{"))
            .unwrap_or_else(|| panic!("struct {name} is not in {}.metal", GENERIC_OPS.name));
        let body = &source[start..];
        let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
        body.split(';')
            .map(str::trim)
            .filter(|field| !field.is_empty())
            .map(|field| {
                let mut words = field.split_whitespace();
                (words.next().unwrap().to_string(), words.next().unwrap().to_string())
            })
            .collect()
    }

    fn fields(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter().map(|&(ty, name)| (ty.to_string(), name.to_string())).collect()
    }

    #[test]
    fn every_kernel_is_in_the_library_and_the_library_includes_the_header() {
        for kernel in GENERIC_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &GENERIC_OPS));
            let declared = format!("kernel void {}(", kernel.function);
            assert!(GENERIC_OPS.source.contains(&declared), "{} is not declared", kernel.function);
        }
        assert!(GENERIC_OPS.source.lines().any(|line| line.trim() == "#include \"makepad_common.h\""));
        let words = || GENERIC_OPS.source.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
        assert!(!words().any(|word| word.starts_with("half")), "no half anywhere (D8)");
        assert!(!words().any(|word| word == "fast"), "no fast:: functions (D8)");
    }

    #[test]
    fn params_structs_are_their_msl_twins() {
        let source = GENERIC_OPS.source;
        assert_eq!(std::mem::size_of::<GenericPixalPool>(), 20);
        assert_eq!(
            msl_struct_fields(source, "GenericPixalPool"),
            fields(&[
                ("uint", "width"),
                ("uint", "height"),
                ("uint", "out_width"),
                ("uint", "out_height"),
                ("uint", "n"),
            ])
        );
        assert_eq!(std::mem::size_of::<GenericTranspose>(), 12);
        assert_eq!(
            msl_struct_fields(source, "GenericTranspose"),
            fields(&[("uint", "rows"), ("uint", "cols"), ("uint", "n")])
        );
        assert_eq!(std::mem::size_of::<GenericScaleAdd>(), 8);
        assert_eq!(msl_struct_fields(source, "GenericScaleAdd"), fields(&[("uint", "n"), ("float", "scale")]));
        assert_eq!(std::mem::size_of::<GenericAddColsBroadcast>(), 8);
        assert_eq!(msl_struct_fields(source, "GenericAddColsBroadcast"), fields(&[("uint", "cols"), ("uint", "n")]));
        assert_eq!(std::mem::size_of::<GenericGatherCols>(), 12);
        assert_eq!(
            msl_struct_fields(source, "GenericGatherCols"),
            fields(&[("uint", "cols"), ("uint", "count"), ("uint", "n")])
        );
    }

    #[test]
    fn slices_and_concatenations_check_and_copy_as_cuda_does() {
        let range = "gpu_slice_cols out of range";
        assert_eq!(slice_cols_plan(3, 10, 4, 7).unwrap_err(), range);
        assert_eq!(slice_cols_plan(3, 10, usize::MAX, 2).unwrap_err(), range, "start + len overflows");
        let copy = slice_cols_plan(3, 10, 4, 6).unwrap();
        assert_eq!(copy_fields(&copy.params), [3, 6, 10, 6, 4, 0]);
        assert_eq!((copy.grid.groups, copy.grid.threads), ([1, 1, 1], COPY_THREADS));
        assert!(slice_cols_plan(3, 10, 10, 0).unwrap().grid.is_empty(), "an empty slice launches nothing");
        assert!(slice_cols_plan(0, 10, 2, 3).unwrap().grid.is_empty(), "no rows: nothing");

        let mismatch = "gpu_concat_cols shape mismatch";
        assert_eq!(concat_cols_plan(&[]).unwrap_err(), mismatch);
        assert_eq!(concat_cols_plan(&[(0, 4), (0, 3)]).unwrap_err(), mismatch, "no rows");
        assert_eq!(concat_cols_plan(&[(2, 4), (3, 3)]).unwrap_err(), mismatch);
        let (rows, total, copies) = concat_cols_plan(&[(5, 4), (5, 1), (5, 0), (5, 3)]).unwrap();
        assert_eq!((rows, total), (5, 8));
        let offsets: Vec<(u32, u32, u32, u32)> =
            copies.iter().map(|c| (c.params.cols, c.params.src_ld, c.params.dst_ld, c.params.dst_offset)).collect();
        assert_eq!(offsets, [(4, 4, 8, 0), (1, 1, 8, 4), (0, 0, 8, 5), (3, 3, 8, 5)]);
        assert!(copies[2].grid.is_empty(), "a part with no columns launches nothing");
        assert!(copies.iter().all(|c| c.params.rows == 5 && c.params.src_offset == 0));

        let wide = slice_cols_plan(1, 1 << 32, 0, 1).unwrap_err();
        assert_eq!(wide, "gpu_slice_cols: 4294967296 does not fit the kernel's 32-bit sizes");
        let many = concat_cols_plan(&[(1 << 20, 1 << 11), (1 << 20, 1 << 11)]).unwrap_err();
        assert!(many.contains("does not fit the kernel's 32-bit sizes"), "{many}");
    }

    #[test]
    fn pool_transpose_and_scale_add_check_as_cuda_does() {
        let mismatch = "gpu_pixal_pool shape/dtype mismatch";
        assert_eq!(pixal_pool_params(3, 12, 4, 3, 0, 2).unwrap_err(), mismatch);
        assert_eq!(pixal_pool_params(3, 12, 4, 3, 2, 0).unwrap_err(), mismatch);
        assert_eq!(pixal_pool_params(3, 0, 0, 3, 2, 2).unwrap_err(), mismatch);
        assert_eq!(pixal_pool_params(3, 13, 4, 3, 2, 2).unwrap_err(), mismatch);
        assert_eq!(pixal_pool_params(65536, 12, 4, 3, 2, 2).unwrap_err(), mismatch, "CUDA's 65535");
        assert_eq!(pixal_pool_params(1, 65536, 65536, 1, 2, 2).unwrap_err(), mismatch);
        assert_eq!(pixal_pool_params(1, 4, 2, 2, 65536, 1).unwrap_err(), mismatch);
        assert_eq!(
            pixal_pool_params(2, 12, 4, 3, 3, 2).unwrap(),
            GenericPixalPool { width: 4, height: 3, out_width: 3, out_height: 2, n: 12 }
        );
        assert_eq!(pixal_pool_params(0, 12, 4, 3, 3, 2).unwrap().n, 0, "no channels: an empty grid");
        let many = pixal_pool_params(65535, 4, 2, 2, 65535, 2).unwrap_err();
        assert_eq!(many, "gpu_pixal_pool: 8589672450 does not fit the kernel's 32-bit sizes");

        assert_eq!(transpose_params(3, 5).unwrap(), GenericTranspose { rows: 3, cols: 5, n: 15 });
        assert_eq!(transpose_params(0, 5).unwrap().n, 0);
        let big = transpose_params(1 << 16, 1 << 16).unwrap_err();
        assert_eq!(big, "gpu_planar_tokens_transpose: 4294967296 does not fit the kernel's 32-bit sizes");

        assert_eq!(scale_add_params((2, 3), (3, 2), 1.0).unwrap_err(), "gpu_realesrgan_scale_add 2x3 vs 3x2");
        let p = scale_add_params((2, 3), (2, 3), -1.0).unwrap();
        assert_eq!((p.n, p.scale.to_bits()), (6, (-1.0f32).to_bits()));
    }

    #[test]
    fn the_broadcast_add_and_the_gather_check_as_cuda_does() {
        let err = add_cols_broadcast_params((4, 3), (1, 4)).unwrap_err();
        assert_eq!(err, "gpu_add_cols_broadcast shape mismatch 4x3 half=false vs bias 1x4 half=false");
        assert_eq!(add_cols_broadcast_params((4, 3), (3, 1)).unwrap(), GenericAddColsBroadcast { cols: 3, n: 12 });
        assert_eq!(add_cols_broadcast_params((0, 3), (1, 3)).unwrap().n, 0, "no rows: an empty grid");
        let big = add_cols_broadcast_params((1 << 20, 1 << 12), (1, 1 << 12)).unwrap_err();
        assert!(big.contains("does not fit the kernel's 32-bit sizes"), "{big}");

        assert_eq!(gather_cols_params(2, 5, &[]).unwrap_err(), "gpu_gather_cols: empty index table");
        assert_eq!(gather_cols_params(2, 5, &[0, 5, 9]).unwrap_err(), "gpu_gather_cols index 5 out of range (cols 5)");
        assert_eq!(gather_cols_params(2, 5, &[4, 0, 4]).unwrap(), GenericGatherCols { cols: 5, count: 3, n: 6 });
        assert_eq!(gather_cols_params(0, 5, &[1]).unwrap().n, 0);
    }

    #[test]
    fn concatenated_rows_copy_each_part_below_the_one_before() {
        let copies = concat_rows_plan("gpu_concat_rows_many", 7, &[2, 0, 3]).unwrap();
        let fields: Vec<[u32; 6]> = copies.iter().map(|c| copy_fields(&c.params)).collect();
        assert_eq!(fields, [[2, 7, 7, 7, 0, 0], [0, 7, 7, 7, 0, 14], [3, 7, 7, 7, 0, 14]]);
        assert!(copies[1].grid.is_empty(), "a part with no rows launches nothing");
        let wide = concat_rows_plan("gpu_concat_rows", 1 << 16, &[1 << 15, 1 << 15]).unwrap_err();
        assert_eq!(wide, "gpu_concat_rows: 4294967296 does not fit the kernel's 32-bit sizes");
    }

    /// `generic_pixal_pool_f32` as the MSL writes it, thread by thread, in u32 arithmetic (a
    /// wrap would panic in a debug build).
    #[allow(clippy::manual_div_ceil)] // the kernel's ceiling, written as it is
    fn pool_as_msl(x: &[f32], p: &GenericPixalPool) -> Vec<f32> {
        (0..p.n)
            .map(|i| {
                let plane = p.out_width * p.out_height;
                let (c, px, py) = (i / plane, i % p.out_width, (i % plane) / p.out_width);
                let x0 = px * p.width / p.out_width;
                let x1 = ((px + 1) * p.width + p.out_width - 1) / p.out_width;
                let y0 = py * p.height / p.out_height;
                let y1 = ((py + 1) * p.height + p.out_height - 1) / p.out_height;
                let mut sum = 0.0f32;
                for y in y0..y1 {
                    let row = (c as u64 * p.height as u64 + y as u64) * p.width as u64;
                    for xx in x0..x1 {
                        sum += x[(row + xx as u64) as usize];
                    }
                }
                sum / ((x1 - x0) * (y1 - y0)) as f32
            })
            .collect()
    }

    /// The host version's loop (`gpu_tensor::pool_area`), in usize.
    #[allow(clippy::manual_div_ceil)] // the host version's ceiling, written as it is
    pub(super) fn pool_host(x: &[f32], channels: usize, w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; channels * ow * oh];
        for c in 0..channels {
            for py in 0..oh {
                let (y0, y1) = (py * h / oh, ((py + 1) * h + oh - 1) / oh);
                for px in 0..ow {
                    let (x0, x1) = (px * w / ow, ((px + 1) * w + ow - 1) / ow);
                    let mut sum = 0.0f32;
                    for y in y0..y1 {
                        for xx in x0..x1 {
                            sum += x[(c * h + y) * w + xx];
                        }
                    }
                    out[c * ow * oh + py * ow + px] = sum / ((x1 - x0) * (y1 - y0)) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn the_pool_kernels_u32_window_is_the_hosts() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        };
        // Cutie's (the mask to stride 16 and the value's size), odd ones, and up and down.
        for (w, h, ow, oh) in [(384, 224, 24, 14), (96, 56, 24, 14), (7, 5, 3, 2), (5, 7, 9, 4), (1, 1, 3, 3), (13, 1, 4, 1)] {
            let channels = 3;
            let x: Vec<f32> = (0..channels * w * h).map(|_| next()).collect();
            let p = pixal_pool_params(channels, w * h, w, h, ow, oh).unwrap();
            let bits = |v: &[f32]| v.iter().map(|t| t.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&pool_as_msl(&x, &p)), bits(&pool_host(&x, channels, w, h, ow, oh)), "{w}x{h} to {ow}x{oh}");
        }
        // At CUDA's limit the window's arithmetic still fits a u32: the last output's right edge.
        let side = PIXAL_POOL_MAX_SIDE as u32;
        let px = side - 1;
        assert!((px + 1).checked_mul(side).and_then(|v| v.checked_add(side - 1)).is_some());
    }

    /// A copy's params as `[rows, cols, src_ld, dst_ld, src_offset, dst_offset]`.
    fn copy_fields(p: &TensorCopy2d) -> [u32; 6] {
        [p.rows, p.cols, p.src_ld, p.dst_ld, p.src_offset, p.dst_offset]
    }
}

/// The ops on the GPU against the host, bit for bit (they move words, sum in the host's order or
/// make CUDA's one fma). `cargo test -p makepad-ai-metal --lib generic_ops`, and again with
/// MAKEPAD_METAL_MISALIGN=1.
#[cfg(all(test, target_os = "macos"))]
mod metal_tests {
    use super::*;
    use crate::stream::{download, library_available, upload};

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                if i % 17 == 9 {
                    f32::from_bits(0x7fc0_0000 | (v & 0x003f_ffff) | 1)
                } else {
                    (v >> 8) as f32 / (1u32 << 23) as f32 - 1.0
                }
            })
            .collect()
    }

    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            panic!("{what}: element {i} is {:#010x}, the host's {:#010x}", got[i].to_bits(), want[i].to_bits());
        }
    }

    /// As `assert_bits`, but any NaN matches any NaN: the GPU's arithmetic gives the canonical
    /// NaN (0x7fc00000, as CUDA's does), not the host's propagated payload. For ops that compute
    /// (the pool, the scale add); ops that only move words keep the payload and use `assert_bits`.
    fn assert_bits_any_nan(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        if let Some(i) = (0..got.len()).find(|&i| !same(got[i], want[i])) {
            panic!("{what}: element {i} is {:#010x}, the host's {:#010x}", got[i].to_bits(), want[i].to_bits());
        }
    }

    #[test]
    fn every_kernel_compiles() {
        assert!(library_available(&GENERIC_OPS), "generic_ops.metal does not compile");
        with_metal_stream("generic_ops: pipelines", |s| {
            for kernel in GENERIC_KERNELS {
                s.limits(kernel, &[])?;
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn slices_and_concatenations_move_every_word() {
        let (rows, cols) = (37, 29);
        let xs = values(rows * cols, 1);
        let x = upload(&xs, rows, cols).unwrap();
        for (start, len) in [(0, 29), (0, 1), (5, 7), (28, 1), (29, 0), (13, 16)] {
            let got = download(&slice_cols(&x, start, len).unwrap()).unwrap();
            let want: Vec<f32> = (0..rows).flat_map(|r| xs[r * cols + start..r * cols + start + len].to_vec()).collect();
            assert_bits(&format!("slice {start}+{len}"), &got, &want);
        }
        assert_eq!(slice_cols(&x, 20, 10).err().unwrap(), "gpu_slice_cols out of range");

        let ones = upload(&vec![1.0; rows], rows, 1).unwrap();
        let ys = values(rows * 3, 2);
        let y = upload(&ys, rows, 3).unwrap();
        let got = download(&concat_cols(&[&x, &ones, &y, &x]).unwrap()).unwrap();
        let mut want = Vec::new();
        for r in 0..rows {
            want.extend_from_slice(&xs[r * cols..(r + 1) * cols]);
            want.push(1.0);
            want.extend_from_slice(&ys[r * 3..(r + 1) * 3]);
            want.extend_from_slice(&xs[r * cols..(r + 1) * cols]);
        }
        assert_bits("concat", &got, &want);
        assert_eq!(concat_cols(&[&x, &upload(&[0.0; 3], 3, 1).unwrap()]).err().unwrap(), "gpu_concat_cols shape mismatch");

        let got = download(&concat_rows_many(&[&y, &y]).unwrap()).unwrap();
        let want: Vec<f32> = ys.iter().chain(ys.iter()).copied().collect();
        assert_bits("concat rows", &got, &want);
        assert_eq!(concat_rows_many(&[]).err().unwrap(), "gpu_concat_rows_many empty");
        assert_eq!(concat_rows(&x, &y).err().unwrap(), "gpu_concat_rows shape mismatch");
    }

    #[test]
    fn the_pool_is_the_hosts_and_the_transpose_and_scale_add_cudas() {
        for (channels, w, h, ow, oh) in [(4, 384, 224, 24, 14), (3, 7, 5, 3, 2), (2, 5, 7, 9, 4), (1, 1, 1, 3, 3)] {
            let xs = values(channels * w * h, 3);
            let x = upload(&xs, channels, w * h).unwrap();
            let got = download(&pixal_pool(&x, w, h, ow, oh).unwrap()).unwrap();
            let want = super::tests::pool_host(&xs, channels, w, h, ow, oh);
            assert_bits_any_nan(&format!("pool {w}x{h} to {ow}x{oh}"), &got, &want);
        }

        for (rows, cols) in [(1, 1), (3, 5), (64, 1000), (257, 3)] {
            let xs = values(rows * cols, 4);
            let x = upload(&xs, rows, cols).unwrap();
            let t = planar_tokens_transpose(&x).unwrap();
            assert_eq!((t.rows(), t.cols()), (cols, rows));
            let want: Vec<f32> = (0..cols).flat_map(|c| (0..rows).map(move |r| (r, c))).map(|(r, c)| xs[r * cols + c]).collect();
            assert_bits(&format!("transpose {rows}x{cols}"), &download(&t).unwrap(), &want);
        }

        let n = 1000;
        let (bs, ds) = (values(n, 5), values(n, 6));
        let (b, d) = (upload(&bs, 1, n).unwrap(), upload(&ds, 1, n).unwrap());
        for scale in [-1.0f32, 0.2, 3.0e-3] {
            let got = download(&realesrgan_scale_add(&b, &d, scale).unwrap()).unwrap();
            let want: Vec<f32> = bs.iter().zip(&ds).map(|(&b, &d)| scale.mul_add(d, b)).collect();
            assert_bits_any_nan(&format!("scale add {scale}"), &got, &want);
        }
    }

    #[test]
    fn the_broadcast_add_the_gather_and_tokens_to_planar_are_cudas() {
        let (rows, cols) = (37, 29);
        let xs = values(rows * cols, 7);
        let x = upload(&xs, rows, cols).unwrap();
        for bias_rows in [1, 29] {
            let bs = values(cols, 8);
            let bias = upload(&bs, bias_rows, cols / bias_rows).unwrap();
            let got = download(&add_cols_broadcast(&x, &bias).unwrap()).unwrap();
            let want: Vec<f32> = (0..rows * cols).map(|i| xs[i] + bs[i % cols]).collect();
            assert_bits_any_nan(&format!("add_cols_broadcast, bias {bias_rows} rows"), &got, &want);
        }
        let bad = upload(&values(cols + 1, 9), 1, cols + 1).unwrap();
        assert!(add_cols_broadcast(&x, &bad).err().unwrap().starts_with("gpu_add_cols_broadcast shape mismatch"));

        let indices = [28u32, 0, 13, 13, 5];
        let got = download(&gather_cols(&x, &indices).unwrap()).unwrap();
        let want: Vec<f32> = (0..rows).flat_map(|r| indices.iter().map(move |&j| (r, j as usize))).map(|(r, j)| xs[r * cols + j]).collect();
        assert_bits("gather_cols", &got, &want);
        assert_eq!(gather_cols(&x, &[3, 29]).err().unwrap(), "gpu_gather_cols index 29 out of range (cols 29)");
        assert_eq!(gather_cols(&x, &[]).err().unwrap(), "gpu_gather_cols: empty index table");

        let t = tokens_to_planar(&x).unwrap();
        assert_eq!((t.rows(), t.cols()), (cols, rows));
        let want: Vec<f32> = (0..cols).flat_map(|c| (0..rows).map(move |r| (r, c))).map(|(r, c)| xs[r * cols + c]).collect();
        assert_bits("tokens_to_planar", &download(&t).unwrap(), &want);
    }
}
