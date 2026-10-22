//! Cutie's elementwise and shape ops as Metal kernels (RationalVisions docs/cutie-metal.md, family
//! (a)): the sigmoid, the sensory GRU's update, others, pad, crop, add_broadcast, normalise_rows,
//! rgb_planes and object_values, at the signatures of the macOS `gpu_cutie_*` functions
//! (models/common's gpu_cutie_host.rs, the same as CUDA's in cuda/src/cutie_ops.rs). The kernels
//! are `shaders/makepad/cutie_elementwise.metal`, ported from cuda/kernels/cutie.cu; each op's
//! host reference is in cuda/src/cutie_reference.rs.
//!
//! Each op makes the checks CUDA's wrapper makes, in its order and with its messages, before
//! anything is encoded. The kernels index in u32 where CUDA's count a few sizes in size_t, so an op
//! whose elements, or any size its kernel takes, pass u32 is `Err` here (`cutie_u32`'s message)
//! where CUDA would run it: four billion floats, never Cutie's. The checks and the kernels' params
//! are plain functions of the sizes, compiled and tested on any OS; the ops are macOS only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::{Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::stream::{with_metal_stream, Bind, Grid};
#[cfg(target_os = "macos")]
use crate::GpuTensor;

/// Cutie's elementwise and shape kernels (`cutie_elementwise.metal`).
pub static CUTIE_ELEMENTWISE: KernelLib = KernelLib {
    name: "cutie_elementwise",
    source: include_str!("../../shaders/makepad/cutie_elementwise.metal"),
};

/// 1 / (1 + exp(-x)) (`CutieSigmoid`).
pub static CUTIE_SIGMOID: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_sigmoid_f32" };
/// The sensory GRU's new hidden state (`CutieGru`).
pub static CUTIE_GRU: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_gru_f32" };
/// Each object's others' masks summed and clamped (`CutieOthers`).
pub static CUTIE_OTHERS: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_others_f32" };
/// Zero padding of planar images (`CutiePad`).
pub static CUTIE_PAD: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_pad_f32" };
/// A window of planar images (`CutieCrop`).
pub static CUTIE_CROP: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_crop_f32" };
/// One block added to each block of another (`CutieAddBroadcast`).
pub static CUTIE_ADD_BROADCAST: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_add_broadcast_f32" };
/// x less one value a row, divided by another (`CutieNormaliseRows`).
pub static CUTIE_NORMALISE_ROWS: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_normalise_rows_f32" };
/// RGB bytes to planar f32 (`CutieRgbPlanes`).
pub static CUTIE_RGB_PLANES: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_rgb_planes_f32" };
/// The object transformer's values from the object memory (`CutieObjectValues`).
pub static CUTIE_OBJECT_VALUES: Kernel = Kernel { lib: &CUTIE_ELEMENTWISE, function: "cutie_object_values_f32" };

/// Every kernel above, for a bring-up that compiles them all.
pub static ELEMENTWISE_KERNELS: &[&Kernel] = &[
    &CUTIE_SIGMOID,
    &CUTIE_GRU,
    &CUTIE_OTHERS,
    &CUTIE_PAD,
    &CUTIE_CROP,
    &CUTIE_ADD_BROADCAST,
    &CUTIE_NORMALISE_ROWS,
    &CUTIE_RGB_PLANES,
    &CUTIE_OBJECT_VALUES,
];

/// Threads to a threadgroup, as CUDA's launches (256).
const THREADS: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieSigmoid {
    n: u32,
}
// SAFETY: #[repr(C)], 4 bytes, no pointers; `CutieSigmoid` in MSL.
unsafe impl KernelParams for CutieSigmoid {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieGru {
    /// channels * plane: one object's rows of h.
    cp: u32,
    /// objects * cp.
    n: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieGru` in MSL.
unsafe impl KernelParams for CutieGru {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieOthers {
    objects: u32,
    pixels: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieOthers` in MSL.
unsafe impl KernelParams for CutieOthers {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutiePad {
    w: u32,
    h: u32,
    nw: u32,
    nh: u32,
    left: u32,
    top: u32,
    /// planes * nw * nh.
    n: u32,
}
// SAFETY: #[repr(C)], 28 bytes, no pointers; `CutiePad` in MSL.
unsafe impl KernelParams for CutiePad {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieCrop {
    w: u32,
    h: u32,
    left: u32,
    top: u32,
    out_w: u32,
    out_h: u32,
    /// planes * out_w * out_h.
    n: u32,
}
// SAFETY: #[repr(C)], 28 bytes, no pointers; `CutieCrop` in MSL.
unsafe impl KernelParams for CutieCrop {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieAddBroadcast {
    a_len: u32,
    n: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieAddBroadcast` in MSL.
unsafe impl KernelParams for CutieAddBroadcast {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieNormaliseRows {
    plane: u32,
    /// rows * plane.
    n: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieNormaliseRows` in MSL.
unsafe impl KernelParams for CutieNormaliseRows {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieRgbPlanes {
    plane: u32,
    /// 3 * plane.
    n: u32,
    scale: f32,
}
// SAFETY: #[repr(C)], 12 bytes, no pointers; `CutieRgbPlanes` in MSL.
unsafe impl KernelParams for CutieRgbPlanes {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct CutieObjectValues {
    objects: u32,
    queries: u32,
    width: u32,
    eps: f32,
    /// objects * queries * width.
    n: u32,
}
// SAFETY: #[repr(C)], 20 bytes, no pointers; `CutieObjectValues` in MSL.
unsafe impl KernelParams for CutieObjectValues {}

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

fn sigmoid_params(rows: usize, cols: usize) -> MetalResult<CutieSigmoid> {
    Ok(CutieSigmoid { n: cutie_count("gpu_cutie_sigmoid", &[rows, cols])? })
}

/// `h` [n*c, p] and `values` [n*3c, p], as (rows, cols).
fn gru_params(h: (usize, usize), values: (usize, usize), n: usize, c: usize) -> MetalResult<CutieGru> {
    let op = "gpu_cutie_gru";
    if h.0 != n * c || values.0 != n * 3 * c || h.1 != values.1 {
        return Err("gpu_cutie_gru shape mismatch".into());
    }
    cutie_u32(op, n)?;
    cutie_u32(op, c)?;
    Ok(CutieGru { cp: cutie_count(op, &[c, h.1])?, n: cutie_count(op, &[n, c, h.1])? })
}

/// `masks` [n, p] as (rows, cols). The kernel runs a thread a column, none when there are no
/// objects.
fn others_params(rows: usize, cols: usize) -> MetalResult<CutieOthers> {
    let op = "gpu_cutie_others";
    let objects = cutie_u32(op, rows)?;
    Ok(CutieOthers { objects, pixels: cutie_u32(op, cols)? })
}

/// `x` [planes, h*w] as (rows, cols).
fn pad_params(
    x: (usize, usize), w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize,
) -> MetalResult<CutiePad> {
    let op = "gpu_cutie_pad";
    if x.1 != w * h {
        return Err("gpu_cutie_pad shape mismatch".into());
    }
    let (nw, nh) = (w + left + right, h + top + bottom);
    let d = |v: usize| cutie_u32(op, v);
    let (_planes, wu, hu, l, _r, t, _b) = (d(x.0)?, d(w)?, d(h)?, d(left)?, d(right)?, d(top)?, d(bottom)?);
    let (nwu, nhu) = (d(nw)?, d(nh)?);
    Ok(CutiePad { w: wu, h: hu, nw: nwu, nh: nhu, left: l, top: t, n: cutie_count(op, &[x.0, nw, nh])? })
}

/// `x` [planes, h*w] as (rows, cols).
fn crop_params(
    x: (usize, usize), w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize,
) -> MetalResult<CutieCrop> {
    let op = "gpu_cutie_crop";
    if x.1 != w * h || left + out_w > w || top + out_h > h {
        return Err("gpu_cutie_crop shape mismatch".into());
    }
    let d = |v: usize| cutie_u32(op, v);
    let (_planes, wu, hu, l, t, owu, ohu) = (d(x.0)?, d(w)?, d(h)?, d(left)?, d(top)?, d(out_w)?, d(out_h)?);
    Ok(CutieCrop { w: wu, h: hu, left: l, top: t, out_w: owu, out_h: ohu, n: cutie_count(op, &[x.0, out_w, out_h])? })
}

/// `a` [r, c] and `b` [k*r, c] as (rows, cols).
fn add_broadcast_params(a: (usize, usize), b: (usize, usize)) -> MetalResult<CutieAddBroadcast> {
    let op = "gpu_cutie_add_broadcast";
    if a.0 == 0 || a.1 != b.1 || !b.0.is_multiple_of(a.0) {
        return Err(format!("{op}: {}x{} over {}x{}", a.0, a.1, b.0, b.1));
    }
    // CUDA's launcher refuses an `a` of no elements under a `b` of some, which these checks
    // already rule out (a.cols == b.cols), so the kernel's `i % a_len` never divides by 0.
    Ok(CutieAddBroadcast { a_len: cutie_count(op, &[a.0, a.1])?, n: cutie_count(op, &[b.0, b.1])? })
}

/// `x` [rows, cols] as (rows, cols); `sub` and `div` their element counts.
fn normalise_rows_params(x: (usize, usize), sub: usize, div: usize) -> MetalResult<CutieNormaliseRows> {
    let op = "gpu_cutie_normalise_rows";
    if sub != x.0 || div != x.0 {
        return Err(format!("{op}: {sub} and {div} values for {} rows", x.0));
    }
    cutie_u32(op, x.0)?;
    Ok(CutieNormaliseRows { plane: cutie_u32(op, x.1)?, n: cutie_count(op, &[x.0, x.1])? })
}

fn rgb_planes_params(bytes: usize, plane: usize, scale: f32) -> MetalResult<CutieRgbPlanes> {
    let op = "gpu_cutie_rgb_planes";
    if plane == 0 || bytes != plane * 3 {
        return Err(format!("{op}: {bytes} bytes for {plane} RGB pixels"));
    }
    Ok(CutieRgbPlanes { plane: cutie_u32(op, plane)?, n: cutie_count(op, &[3, plane])?, scale })
}

/// `memory` [objects*queries, width+1] as (rows, cols).
fn object_values_params(memory: (usize, usize), objects: usize, eps: f32) -> MetalResult<CutieObjectValues> {
    let op = "gpu_cutie_object_values";
    if objects == 0 || !memory.0.is_multiple_of(objects) || memory.1 < 2 {
        return Err(format!("{op}: {}x{} for {objects} objects", memory.0, memory.1));
    }
    let (queries, width) = (memory.0 / objects, memory.1 - 1);
    let d = |v: usize| cutie_u32(op, v);
    Ok(CutieObjectValues {
        objects: d(objects)?,
        queries: d(queries)?,
        width: d(width)?,
        eps,
        n: cutie_count(op, &[objects, queries, width])?,
    })
}

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

/// One launch of `kernel` over `threads` threads ([`THREADS`] to a threadgroup) with `params`,
/// reading `inputs` from [[buffer(1)]] on and writing a new [rows, cols] tensor bound after them.
#[cfg(target_os = "macos")]
fn launch_one<P: KernelParams>(
    op: &'static str, kernel: &'static Kernel, params: &P, threads: u32, inputs: &[&GpuTensor], rows: usize,
    cols: usize,
) -> MetalResult<GpuTensor> {
    let grid = Grid::linear(threads as usize, THREADS).map_err(|e| format!("{op}: {e}"))?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(rows, cols)?;
        let binds: Vec<Bind<'_>> = inputs.iter().map(|&t| Bind::In(t)).chain([Bind::Out(&out)]).collect();
        s.launch(kernel, &[], params, &binds, grid)?;
        Ok(out)
    })
}

// --- Elementwise ---

/// 1 / (1 + exp(-x)), elementwise.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_sigmoid(x: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_sigmoid";
    cutie_check_f32(op, x)?;
    let params = sigmoid_params(x.rows(), x.cols())?;
    launch_one(op, &CUTIE_SIGMOID, &params, params.n, &[x], x.rows(), x.cols())
}

/// Cutie's GRU-like sensory update, per object: `values` [n*3c, p] holds each object's forget,
/// update and new-value rows (c each); `h` [n*c, p]. new_h = sigmoid(f)*h*(1-sigmoid(u)) +
/// sigmoid(u)*tanh(v), evaluated in that order.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_gru(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_gru";
    cutie_check_f32(op, h)?;
    cutie_check_f32(op, values)?;
    let params = gru_params(shape(h), shape(values), n, c)?;
    launch_one(op, &CUTIE_GRU, &params, params.n, &[h, values], h.rows(), h.cols())
}

/// For each object's mask in `masks` [n, p], the sum of the other objects' masks, clamped to 0..1
/// (Cutie's `_get_others`: all objects summed in PyTorch's order, `cutie_reduce4`, then the
/// object's own subtracted).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_others(masks: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_others";
    cutie_check_f32(op, masks)?;
    let params = others_params(masks.rows(), masks.cols())?;
    let threads = if params.objects == 0 { 0 } else { params.pixels };
    launch_one(op, &CUTIE_OTHERS, &params, threads, &[masks], masks.rows(), masks.cols())
}

// --- Shapes ---

/// Zero padding of planar [planes, h*w] by (left, right, top, bottom). Out [planes, (h+t+b)*(w+l+r)].
#[cfg(target_os = "macos")]
pub fn gpu_cutie_pad(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_pad";
    cutie_check_f32(op, x)?;
    let params = pad_params(shape(x), w, h, left, right, top, bottom)?;
    let cols = (w + left + right) * (h + top + bottom);
    launch_one(op, &CUTIE_PAD, &params, params.n, &[x], x.rows(), cols)
}

/// A window of planar [planes, h*w]: `out_w` x `out_h` from (`left`, `top`).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_crop(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_crop";
    cutie_check_f32(op, x)?;
    let params = crop_params(shape(x), w, h, left, top, out_w, out_h)?;
    launch_one(op, &CUTIE_CROP, &params, params.n, &[x], x.rows(), out_w * out_h)
}

// --- The per-frame step ---

/// The sum of a block `a` [r, c] and each of the k blocks of `b` [k*r, c]: out = a + b_i, one
/// rounding each (Cutie's x + g, one image's features broadcast over the objects, without
/// repeating them first).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_add_broadcast(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_add_broadcast";
    cutie_check_f32(op, a)?;
    cutie_check_f32(op, b)?;
    let params = add_broadcast_params(shape(a), shape(b))?;
    launch_one(op, &CUTIE_ADD_BROADCAST, &params, params.n, &[a, b], b.rows(), b.cols())
}

/// (x - sub[r]) / div[r] for each row r of `x` [rows, cols], `sub` and `div` one value a row on
/// the device: CUTIE.encode_image's (image - pixel_mean) / pixel_std, each op rounded, a true
/// division (multiplying by 1 / div would be up to an ulp off).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_normalise_rows(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_normalise_rows";
    for t in [x, sub, div] {
        cutie_check_f32(op, t)?;
    }
    let params = normalise_rows_params(shape(x), sub.rows() * sub.cols(), div.rows() * div.cols())?;
    launch_one(op, &CUTIE_NORMALISE_ROWS, &params, params.n, &[x, sub, div], x.rows(), x.cols())
}

/// RGB bytes (`plane` pixels, 3 bytes each) to planar f32 [3, plane], each byte times `scale`.
/// The bytes are what goes to the device. Cutie's image is the worker's `x / 255` in PyTorch,
/// which on CUDA multiplies by the f32 reciprocal: `scale` 1.0f32 / 255.0.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_rgb_planes(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_rgb_planes";
    let params = rgb_planes_params(rgb.len(), plane, scale)?;
    let grid = Grid::linear(params.n as usize, THREADS).map_err(|e| format!("{op}: {e}"))?;
    with_metal_stream(op, |s| {
        // The bytes as u32 words, which the kernel reads as bytes.
        let bytes = s.upload_bytes(rgb)?;
        let out = GpuTensor::alloc(3, plane)?;
        s.launch(&CUTIE_RGB_PLANES, &[], &params, &[Bind::In(&bytes), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// The object transformer's values from the object memory `memory` [objects*queries, width+1]
/// (object-major, each summary's sums then its area): sums / (area + `eps`) as
/// `object_transformer.py`'s `obj_sums / (obj_area + 1e-4)` rounds it (the add, then a true
/// division), interleaved [queries*objects, width] (row q*objects + o).
#[cfg(target_os = "macos")]
pub fn gpu_cutie_object_values(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_object_values";
    cutie_check_f32(op, memory)?;
    let params = object_values_params(shape(memory), objects, eps)?;
    launch_one(op, &CUTIE_OBJECT_VALUES, &params, params.n, &[memory], memory.rows(), memory.cols() - 1)
}

/// The checks and params on any OS: CUDA's messages, Cutie's shapes, and each params struct
/// field for field its MSL twin in `cutie_elementwise.metal`.
#[cfg(test)]
mod tests {
    use super::*;

    /// The fields of `struct name { ... };` in the library's source, as (type, name).
    fn msl_struct(name: &str) -> Vec<(String, String)> {
        let source = CUTIE_ELEMENTWISE.source;
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

    /// The fields of a params struct from its Debug text, every field set to 1: (MSL type, name)
    /// in declaration order, `float` for an f32 (printed "1.0") and `uint` for a u32.
    fn rust_struct(debug: &str) -> Vec<(String, String)> {
        let body = &debug[debug.find('{').unwrap() + 1..debug.rfind('}').unwrap()];
        body.split(',')
            .map(|field| {
                let (name, value) = field.split_once(':').unwrap();
                let ty = if value.contains('.') { "float" } else { "uint" };
                (ty.to_string(), name.trim().to_string())
            })
            .collect()
    }

    #[test]
    fn params_are_their_msl_twins_field_for_field() {
        let twins: [(&str, &Kernel, String, usize); 9] = [
            ("CutieSigmoid", &CUTIE_SIGMOID, format!("{:?}", CutieSigmoid { n: 1 }), size_of::<CutieSigmoid>()),
            ("CutieGru", &CUTIE_GRU, format!("{:?}", CutieGru { cp: 1, n: 1 }), size_of::<CutieGru>()),
            ("CutieOthers", &CUTIE_OTHERS, format!("{:?}", CutieOthers { objects: 1, pixels: 1 }), size_of::<CutieOthers>()),
            (
                "CutiePad",
                &CUTIE_PAD,
                format!("{:?}", CutiePad { w: 1, h: 1, nw: 1, nh: 1, left: 1, top: 1, n: 1 }),
                size_of::<CutiePad>(),
            ),
            (
                "CutieCrop",
                &CUTIE_CROP,
                format!("{:?}", CutieCrop { w: 1, h: 1, left: 1, top: 1, out_w: 1, out_h: 1, n: 1 }),
                size_of::<CutieCrop>(),
            ),
            (
                "CutieAddBroadcast",
                &CUTIE_ADD_BROADCAST,
                format!("{:?}", CutieAddBroadcast { a_len: 1, n: 1 }),
                size_of::<CutieAddBroadcast>(),
            ),
            (
                "CutieNormaliseRows",
                &CUTIE_NORMALISE_ROWS,
                format!("{:?}", CutieNormaliseRows { plane: 1, n: 1 }),
                size_of::<CutieNormaliseRows>(),
            ),
            (
                "CutieRgbPlanes",
                &CUTIE_RGB_PLANES,
                format!("{:?}", CutieRgbPlanes { plane: 1, n: 1, scale: 1.0 }),
                size_of::<CutieRgbPlanes>(),
            ),
            (
                "CutieObjectValues",
                &CUTIE_OBJECT_VALUES,
                format!("{:?}", CutieObjectValues { objects: 1, queries: 1, width: 1, eps: 1.0, n: 1 }),
                size_of::<CutieObjectValues>(),
            ),
        ];
        assert_eq!(twins.len(), ELEMENTWISE_KERNELS.len());
        for (name, kernel, debug, bytes) in twins {
            let msl = msl_struct(name);
            assert_eq!(rust_struct(&debug), msl, "{name}: the Rust struct and the MSL struct differ");
            assert_eq!(bytes, 4 * msl.len(), "{name}: 4 bytes a field");
            // The kernel exists and reads this struct at [[buffer(0)]].
            let signature = format!("kernel void {}(constant {name}& p [[buffer(0)]]", kernel.function);
            assert!(CUTIE_ELEMENTWISE.source.contains(&signature), "no `{signature}` in the source");
            assert!(ELEMENTWISE_KERNELS.iter().any(|k| k.function == kernel.function));
        }
    }

    #[test]
    fn shape_errors_are_cudas() {
        assert_eq!(gru_params((6, 10), (18, 10), 2, 4).unwrap_err(), "gpu_cutie_gru shape mismatch");
        assert_eq!(gru_params((8, 10), (23, 10), 2, 4).unwrap_err(), "gpu_cutie_gru shape mismatch");
        assert_eq!(gru_params((8, 10), (24, 9), 2, 4).unwrap_err(), "gpu_cutie_gru shape mismatch");
        assert_eq!(pad_params((3, 99), 10, 10, 1, 1, 1, 1).unwrap_err(), "gpu_cutie_pad shape mismatch");
        assert_eq!(crop_params((3, 99), 10, 10, 0, 0, 5, 5).unwrap_err(), "gpu_cutie_crop shape mismatch");
        assert_eq!(crop_params((3, 100), 10, 10, 6, 0, 5, 5).unwrap_err(), "gpu_cutie_crop shape mismatch");
        assert_eq!(crop_params((3, 100), 10, 10, 0, 6, 5, 5).unwrap_err(), "gpu_cutie_crop shape mismatch");
        assert_eq!(add_broadcast_params((0, 4), (8, 4)).unwrap_err(), "gpu_cutie_add_broadcast: 0x4 over 8x4");
        assert_eq!(add_broadcast_params((2, 4), (8, 5)).unwrap_err(), "gpu_cutie_add_broadcast: 2x4 over 8x5");
        assert_eq!(add_broadcast_params((3, 4), (8, 4)).unwrap_err(), "gpu_cutie_add_broadcast: 3x4 over 8x4");
        assert_eq!(
            normalise_rows_params((3, 10), 2, 3).unwrap_err(),
            "gpu_cutie_normalise_rows: 2 and 3 values for 3 rows"
        );
        assert_eq!(
            normalise_rows_params((3, 10), 3, 4).unwrap_err(),
            "gpu_cutie_normalise_rows: 3 and 4 values for 3 rows"
        );
        assert_eq!(rgb_planes_params(0, 0, 1.0).unwrap_err(), "gpu_cutie_rgb_planes: 0 bytes for 0 RGB pixels");
        assert_eq!(rgb_planes_params(20, 7, 1.0).unwrap_err(), "gpu_cutie_rgb_planes: 20 bytes for 7 RGB pixels");
        assert_eq!(object_values_params((32, 257), 0, 1e-4).unwrap_err(), "gpu_cutie_object_values: 32x257 for 0 objects");
        assert_eq!(object_values_params((32, 257), 3, 1e-4).unwrap_err(), "gpu_cutie_object_values: 32x257 for 3 objects");
        assert_eq!(object_values_params((32, 1), 2, 1e-4).unwrap_err(), "gpu_cutie_object_values: 32x1 for 2 objects");
    }

    #[test]
    fn params_at_cuties_shapes() {
        // Stride 16 of the padded 384x224 frame, the frame itself.
        const P16: usize = 24 * 14;
        const FRAME: usize = 384 * 224;
        assert_eq!(sigmoid_params(3, FRAME).unwrap(), CutieSigmoid { n: 3 * FRAME as u32 });
        assert_eq!(gru_params((3 * 256, P16), (9 * 256, P16), 3, 256).unwrap(), CutieGru { cp: 256 * P16 as u32, n: 3 * 256 * P16 as u32 });
        assert_eq!(others_params(3, FRAME).unwrap(), CutieOthers { objects: 3, pixels: FRAME as u32 });
        assert_eq!(others_params(0, 37).unwrap(), CutieOthers { objects: 0, pixels: 37 });
        // The processor pads 384x216 to 384x224 (4 rows each side) and crops back.
        assert_eq!(
            pad_params((3, 384 * 216), 384, 216, 0, 0, 4, 4).unwrap(),
            CutiePad { w: 384, h: 216, nw: 384, nh: 224, left: 0, top: 4, n: 3 * FRAME as u32 }
        );
        assert_eq!(
            crop_params((4, FRAME), 384, 224, 0, 4, 384, 216).unwrap(),
            CutieCrop { w: 384, h: 224, left: 0, top: 4, out_w: 384, out_h: 216, n: 4 * 384 * 216 }
        );
        assert_eq!(crop_params((1, 31 * 19), 31, 19, 30, 18, 1, 1).unwrap().n, 1);
        assert_eq!(add_broadcast_params((256, P16), (768, P16)).unwrap(), CutieAddBroadcast { a_len: 256 * P16 as u32, n: 768 * P16 as u32 });
        assert_eq!(normalise_rows_params((3, FRAME), 3, 3).unwrap(), CutieNormaliseRows { plane: FRAME as u32, n: 3 * FRAME as u32 });
        let rgb = rgb_planes_params(3 * FRAME, FRAME, 1.0 / 255.0).unwrap();
        assert_eq!(rgb, CutieRgbPlanes { plane: FRAME as u32, n: 3 * FRAME as u32, scale: 1.0 / 255.0 });
        assert_eq!(
            object_values_params((3 * 16, 257), 3, 1e-4).unwrap(),
            CutieObjectValues { objects: 3, queries: 16, width: 256, eps: 1e-4, n: 3 * 16 * 256 }
        );
    }

    #[test]
    fn empty_tensors_launch_nothing() {
        assert_eq!(sigmoid_params(0, 7).unwrap().n, 0);
        assert_eq!(gru_params((0, 5), (0, 5), 0, 4).unwrap().n, 0);
        assert_eq!(pad_params((0, 4), 2, 2, 1, 1, 1, 1).unwrap().n, 0);
        assert_eq!(crop_params((2, 4), 2, 2, 1, 1, 0, 1).unwrap().n, 0);
        assert_eq!(add_broadcast_params((2, 0), (4, 0)).unwrap(), CutieAddBroadcast { a_len: 0, n: 0 });
        assert_eq!(normalise_rows_params((3, 0), 3, 3).unwrap().n, 0);
    }

    #[test]
    fn sizes_past_u32_are_errors_with_cudas_message() {
        let big = 1usize << 20;
        assert_eq!(
            sigmoid_params(big, 1 << 12).unwrap_err(),
            "gpu_cutie_sigmoid: 4294967296 does not fit the kernel's 32-bit sizes"
        );
        assert_eq!(
            others_params(1usize << 32, 3).unwrap_err(),
            "gpu_cutie_others: 4294967296 does not fit the kernel's 32-bit sizes"
        );
        assert_eq!(
            pad_params((1 << 12, big), big, 1, 0, 0, 0, 0).unwrap_err(),
            "gpu_cutie_pad: 4294967296 does not fit the kernel's 32-bit sizes"
        );
        assert_eq!(
            cutie_count("op", &[usize::MAX, 2, 0]).unwrap(),
            0,
            "a zero size is no elements whatever the others"
        );
        assert_eq!(
            cutie_count("op", &[usize::MAX, 2]).unwrap_err(),
            format!("op: {}x2 does not fit the kernel's 32-bit sizes", usize::MAX)
        );
        assert_eq!(cutie_count("op", &[u32::MAX as usize, 1]).unwrap(), u32::MAX);
    }
}

/// The ops on the Mac against the host, in the host binding: exact where CUDA's kernels only move,
/// add, multiply or divide (the ops check's EXACT), within a few ulps where they call exp or tanh
/// (precise:: against the host's libm). `cargo test -p makepad-ai-metal --lib cutie_ops::elementwise`,
/// and again with MAKEPAD_METAL_MISALIGN=1.
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

        /// Uniform in [0, 1), 24 bits.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        fn uniforms(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * self.unit()).collect()
        }
    }

    fn up(values: &[f32], rows: usize, cols: usize) -> GpuTensor {
        stream::upload(values, rows, cols).unwrap()
    }

    fn down(t: &GpuTensor) -> Vec<f32> {
        stream::download(t).unwrap()
    }

    /// `got` and `want` are the same bits (NaN for NaN, -0 for -0), naming the first difference.
    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            panic!("{what}: element {i} is {:e} ({:#010x}), the host's is {:e} ({:#010x})", got[i], got[i].to_bits(), want[i], want[i].to_bits());
        }
    }

    /// `got` within `ulps` ulps of `want`, scaled by `scale` (the size of the terms that made it).
    fn assert_close(what: &str, got: &[f32], want: &[f32], scale: f32, ulps: f32) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let tol = ulps * f32::EPSILON * scale;
        let within = |i: usize| (got[i] - want[i]).abs() <= tol * want[i].abs().max(1.0);
        if let Some(i) = (0..got.len()).find(|&i| !within(i)) {
            panic!("{what}: element {i} is {:e}, the host's is {:e} (tolerance {tol:e})", got[i], want[i]);
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn every_kernel_compiles() {
        assert!(stream::library_available(&CUTIE_ELEMENTWISE), "cutie_elementwise does not compile");
        with_metal_stream("cutie elementwise: pipelines", |s| {
            for kernel in ELEMENTWISE_KERNELS {
                s.limits(kernel, &[])?;
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn sigmoid_and_gru_are_the_hosts_within_a_few_ulps() {
        let mut rng = Rng(0x5eed_0001);
        for (rows, cols, scale) in [(3, 1000, 4.0f32), (7, 33, 40.0), (1, 1, 1.0)] {
            let x = rng.uniforms(rows * cols, -scale, scale);
            let got = down(&gpu_cutie_sigmoid(&up(&x, rows, cols)).unwrap());
            let want: Vec<f32> = x.iter().map(|&v| sigmoid(v)).collect();
            assert_close(&format!("sigmoid {rows}x{cols}"), &got, &want, 1.0, 4.0);
        }
        for (n, c, p) in [(1, 256, 336), (3, 7, 17), (2, 5, 1)] {
            let h = rng.uniforms(n * c * p, -2.0, 2.0);
            let values = rng.uniforms(n * 3 * c * p, -4.0, 4.0);
            let got = down(&gpu_cutie_gru(&up(&h, n * c, p), &up(&values, n * 3 * c, p), n, c).unwrap());
            let mut want = vec![0.0f32; n * c * p];
            for o in 0..n {
                for ch in 0..c {
                    for i in 0..p {
                        let v = |part: usize| values[((o * 3 + part) * c + ch) * p + i];
                        let (f, u, fresh) = (sigmoid(v(0)), sigmoid(v(1)), v(2).tanh());
                        let at = (o * c + ch) * p + i;
                        want[at] = f * h[at] * (1.0 - u) + u * fresh;
                    }
                }
            }
            assert_close(&format!("gru n{n} c{c} p{p}"), &got, &want, 2.0, 8.0);
        }
    }

    #[test]
    fn others_is_the_hosts_bit_for_bit_in_pytorchs_order() {
        let mut rng = Rng(0x5eed_0002);
        // 5 objects on, PyTorch's sum over them is no longer sequential.
        for (n, p) in [(1, 37), (2, 1000), (3, 513), (5, 300), (9, 37), (0, 5)] {
            let mut masks = rng.uniforms(n * p, 0.0, 1.0);
            for (i, m) in masks.iter_mut().enumerate() {
                match i % 11 {
                    0 => *m = 0.0,
                    1 => *m = 1.0,
                    _ => {}
                }
            }
            if n > 1 {
                masks[3] = f32::NAN;
            }
            let got = down(&gpu_cutie_others(&up(&masks, n, p)).unwrap());
            let mut want = vec![0.0f32; n * p];
            for i in 0..p {
                let mut acc = [0.0f32; 4];
                for o in 0..n {
                    acc[o % 4] += masks[o * p + i];
                }
                let sum = ((acc[0] + acc[1]) + acc[2]) + acc[3];
                for o in 0..n {
                    let d = sum - masks[o * p + i];
                    // NaN passes, -0 stays -0: the kernel's d < 0 ? 0 : (d > 1 ? 1 : d).
                    want[o * p + i] = d.clamp(0.0, 1.0);
                }
            }
            assert_bits(&format!("others n{n} p{p}"), &got, &want);
        }
    }

    #[test]
    fn pad_and_crop_move_every_bit() {
        let mut rng = Rng(0x5eed_0003);
        for (planes, w, h, l, r, t, b) in [(3, 384, 216, 0, 0, 4, 4), (2, 13, 7, 1, 2, 3, 0), (1, 5, 5, 0, 3, 0, 0), (4, 1, 1, 2, 2, 2, 2), (0, 3, 3, 1, 1, 1, 1)] {
            let mut x = rng.uniforms(planes * w * h, -1.0, 1.0);
            if let Some(v) = x.first_mut() {
                *v = f32::from_bits(0x7fc0_1234);
            }
            let (nw, nh) = (w + l + r, h + t + b);
            let got = down(&gpu_cutie_pad(&up(&x, planes, w * h), w, h, l, r, t, b).unwrap());
            let mut want = vec![0.0f32; planes * nw * nh];
            for p in 0..planes {
                for y in 0..h {
                    let at = p * nw * nh + (y + t) * nw + l;
                    want[at..at + w].copy_from_slice(&x[p * w * h + y * w..][..w]);
                }
            }
            assert_bits(&format!("pad {planes} planes {w}x{h} by l{l} r{r} t{t} b{b}"), &got, &want);
        }
        for (planes, w, h, l, t, ow, oh) in [(4, 384, 224, 0, 4, 384, 216), (3, 17, 11, 3, 2, 9, 5), (2, 9, 7, 0, 0, 9, 7), (1, 31, 19, 30, 18, 1, 1), (2, 5, 5, 1, 1, 0, 3)] {
            let x = rng.uniforms(planes * w * h, -1.0, 1.0);
            let got = down(&gpu_cutie_crop(&up(&x, planes, w * h), w, h, l, t, ow, oh).unwrap());
            let mut want = Vec::with_capacity(planes * ow * oh);
            for p in 0..planes {
                for y in 0..oh {
                    want.extend_from_slice(&x[p * w * h + (y + t) * w + l..][..ow]);
                }
            }
            assert_bits(&format!("crop {planes} planes {w}x{h} to {ow}x{oh} at {l},{t}"), &got, &want);
        }
    }

    #[test]
    fn add_broadcast_normalise_and_object_values_are_the_hosts_bit_for_bit() {
        let mut rng = Rng(0x5eed_0004);
        for (rows, cols, k) in [(256, 336, 3), (7, 13, 1), (1, 1, 5)] {
            let a = rng.uniforms(rows * cols, -1.0, 1.0);
            let b = rng.uniforms(k * rows * cols, -1.0, 1.0);
            let got = down(&gpu_cutie_add_broadcast(&up(&a, rows, cols), &up(&b, k * rows, cols)).unwrap());
            let want: Vec<f32> = b.iter().enumerate().map(|(i, &g)| a[i % a.len()] + g).collect();
            assert_bits(&format!("add_broadcast {rows}x{cols} onto {k}"), &got, &want);
        }
        for (rows, cols) in [(3, 384 * 224), (5, 17)] {
            let x = rng.uniforms(rows * cols, 0.0, 1.0);
            let (sub, div) = if rows == 3 {
                (vec![0.485f32, 0.456, 0.406], vec![0.229f32, 0.224, 0.225])
            } else {
                (rng.uniforms(rows, 0.0, 1.0), rng.uniforms(rows, 0.1, 1.0))
            };
            let got = down(&gpu_cutie_normalise_rows(&up(&x, rows, cols), &up(&sub, rows, 1), &up(&div, 1, rows)).unwrap());
            let want: Vec<f32> = x.iter().enumerate().map(|(i, &v)| (v - sub[i / cols]) / div[i / cols]).collect();
            assert_bits(&format!("normalise {rows}x{cols}"), &got, &want);
        }
        const QUERIES: usize = 16;
        const WIDTH: usize = 257;
        for objects in [1, 3] {
            let mut memory = rng.uniforms(objects * QUERIES * WIDTH, -300.0, 300.0);
            for (r, row) in memory.chunks_mut(WIDTH).enumerate() {
                row[WIDTH - 1] = match r % 4 {
                    0 => 0.0,
                    1 => 1e-6,
                    _ => rng.unit() * 336.0,
                };
            }
            let got = down(&gpu_cutie_object_values(&up(&memory, objects * QUERIES, WIDTH), objects, 1e-4).unwrap());
            let mut want = vec![0.0f32; objects * QUERIES * (WIDTH - 1)];
            for o in 0..objects {
                for q in 0..QUERIES {
                    let row = &memory[(o * QUERIES + q) * WIDTH..][..WIDTH];
                    let area = row[WIDTH - 1] + 1e-4;
                    for c in 0..WIDTH - 1 {
                        want[(q * objects + o) * (WIDTH - 1) + c] = row[c] / area;
                    }
                }
            }
            assert_bits(&format!("object_values n{objects}"), &got, &want);
        }
    }

    #[test]
    fn rgb_planes_reads_bytes_at_any_count() {
        let mut rng = Rng(0x5eed_0005);
        let scale = 1.0f32 / 255.0f32;
        // 7 pixels: 21 bytes, not whole words.
        for plane in [384 * 216, 7, 1] {
            let rgb: Vec<u8> = (0..plane * 3).map(|_| rng.next() as u8).collect();
            let got = down(&gpu_cutie_rgb_planes(&rgb, plane, scale).unwrap());
            let mut want = vec![0.0f32; 3 * plane];
            for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    want[c * plane + i] = px[c] as f32 * scale;
                }
            }
            assert_bits(&format!("rgb {plane} pixels"), &got, &want);
        }
    }

    #[test]
    fn bad_shapes_encode_nothing_and_say_so() {
        let x = up(&[0.0; 12], 3, 4);
        assert_eq!(gpu_cutie_pad(&x, 3, 3, 1, 1, 1, 1).err().expect("an error"), "gpu_cutie_pad shape mismatch");
        assert_eq!(gpu_cutie_crop(&x, 2, 2, 0, 0, 3, 1).err().expect("an error"), "gpu_cutie_crop shape mismatch");
        assert_eq!(gpu_cutie_rgb_planes(&[0; 5], 2, 1.0).err().expect("an error"), "gpu_cutie_rgb_planes: 5 bytes for 2 RGB pixels");
    }
}
