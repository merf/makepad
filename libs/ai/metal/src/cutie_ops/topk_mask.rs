//! Cutie's top-k softmax and attention mask as Metal kernels (RationalVisions docs/cutie-metal.md,
//! family (e)): `gpu_cutie_topk_softmax` and `gpu_cutie_attn_mask`, at the signatures of the macOS
//! `gpu_cutie_*` functions (models/common's gpu_cutie_host.rs, the same as CUDA's in
//! cuda/src/cutie_ops.rs). The kernels are `shaders/makepad/cutie_topk_mask.metal`, ported from
//! cuda/kernels/cutie.cu; each op's host reference is in cuda/src/cutie_reference.rs.
//!
//! Each op makes the checks CUDA's wrapper makes, in its order and with its messages, before
//! anything is encoded. The kernels index in 64 bits wherever CUDA's do (size_t); the one size
//! limit CUDA lacks is the stream's zero fill (CUDA's is a memset), which counts its words in u32:
//! a similarity of more than u32::MAX elements, or a mask of 2^31 objects or more, is `Err` here
//! (`cutie_u32`'s message), never Cutie's sizes. The merge and the row sums go round a SIMD-group
//! in CUDA's butterfly order, written for its 32 lanes (Apple's SIMD-groups are 32 wide): a
//! pipeline of another width is an `Err`, never another order. The checks, the kernels' params and
//! their grids are plain functions of the sizes, compiled and tested on any OS; the ops are macOS
//! only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::{Grid, Kernel, KernelLib, KernelLimits, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::stream::{with_metal_stream, Bind};
#[cfg(target_os = "macos")]
use crate::GpuTensor;

/// Cutie's top-k softmax and attention mask kernels (`cutie_topk_mask.metal`).
pub static CUTIE_TOPK_MASK: KernelLib = KernelLib {
    name: "cutie_topk_mask",
    source: include_str!("../../shaders/makepad/cutie_topk_mask.metal"),
};

/// Each column's slices' best 32 (`CutieTopkPartial`); for k up to 32.
pub static CUTIE_TOPK_PARTIAL_32: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_topk_partial_32" };
/// Each column's slices' best 64 (`CutieTopkPartial`); for k from 33 to 64.
pub static CUTIE_TOPK_PARTIAL_64: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_topk_partial_64" };
/// Each column's slices merged, the softmax scattered into the zeroed affinity (`CutieTopkMerge`).
pub static CUTIE_TOPK_MERGE: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_topk_merge" };
/// Each row's sum: the top-k's usage (`CutieRowSum`).
pub static CUTIE_ROW_SUM: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_row_sum" };
/// Each object's foreground flags, and whether it has any foreground and any background
/// (`CutieAttnForeground`).
pub static CUTIE_ATTN_FOREGROUND: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_attn_foreground" };
/// The additive attention mask from the flags (`CutieAttnMask`).
pub static CUTIE_ATTN_MASK: Kernel = Kernel { lib: &CUTIE_TOPK_MASK, function: "cutie_attn_mask" };

/// Every kernel above, for a bring-up that compiles them all.
pub static TOPK_MASK_KERNELS: &[&Kernel] = &[
    &CUTIE_TOPK_PARTIAL_32,
    &CUTIE_TOPK_PARTIAL_64,
    &CUTIE_TOPK_MERGE,
    &CUTIE_ROW_SUM,
    &CUTIE_ATTN_FOREGROUND,
    &CUTIE_ATTN_MASK,
];

/// The largest k of the top-k kernel (Cutie reads 30).
const CUTIE_TOPK_MAX: usize = 64;
/// Columns to a threadgroup of the partial top-k: a threadgroup row.
const TOPK_COLS: u32 = 32;
/// Slices to a threadgroup of the partial top-k, fewer when the pipeline takes fewer threads.
const TOPK_SLICES_PER_GROUP: u32 = 4;
/// About how many rows a slice of a column holds.
const TOPK_SLICE_ROWS: u32 = 256;
/// The most slices a column is split into: the merge's SIMD-group holds two a lane.
const TOPK_MAX_SLICES: u32 = 64;
/// Columns the merge takes to a threadgroup, a SIMD-group each (CUDA's 128-thread blocks).
const TOPK_MERGE_COLUMNS: u32 = 4;
/// Rows the row sum takes to a threadgroup, a SIMD-group each (CUDA's 256-thread blocks).
const ROW_SUM_ROWS: u32 = 8;
/// The lanes the butterflies are written for: CUDA's warp.
const SIMD_WIDTH: u32 = 32;
/// Threads to a threadgroup of the attention mask's kernels, as CUDA's launches.
const THREADS: u32 = 256;
/// The most threadgroups of the foreground's grid-stride loop (CUDA's `cutie_blocks`).
const FOREGROUND_MAX_GROUPS: usize = 65536;
/// The most threadgroups down the mask's rows; each steps on by the grid's height.
const MASK_MAX_ROW_GROUPS: usize = 65535;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TopkPartial {
    m: u32,
    p: u32,
    k: u32,
    /// Rows a slice holds: ceil(m / slices).
    chunk: u32,
    slices: u32,
}
// SAFETY: #[repr(C)], 20 bytes, no pointers; `CutieTopkPartial` in MSL.
unsafe impl KernelParams for TopkPartial {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct TopkMerge {
    m: u32,
    p: u32,
    k: u32,
    slices: u32,
}
// SAFETY: #[repr(C)], 16 bytes, no pointers; `CutieTopkMerge` in MSL.
unsafe impl KernelParams for TopkMerge {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct RowSum {
    rows: u32,
    cols: u32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `CutieRowSum` in MSL.
unsafe impl KernelParams for RowSum {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct AttnForeground {
    p: u64,
    n: u32,
    pad: u32,
}
// SAFETY: #[repr(C)], 16 bytes, no pointers; `CutieAttnForeground` in MSL (ulong, uint, uint).
unsafe impl KernelParams for AttnForeground {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct AttnMask {
    p: u64,
    rows: u64,
    heads: u32,
    queries: u32,
}
// SAFETY: #[repr(C)], 24 bytes, no pointers; `CutieAttnMask` in MSL (ulong, ulong, uint, uint).
unsafe impl KernelParams for AttnMask {}

/// A size as the kernels take it (CUDA's `cutie_u32`, with its message).
fn cutie_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// The slices one column of `m` rows is split into: about [`TOPK_SLICE_ROWS`] rows each, at most
/// [`TOPK_MAX_SLICES`], at least one (CUDA's `cutie_topk_slices`).
fn topk_slices(m: u32) -> u32 {
    m.div_ceil(TOPK_SLICE_ROWS).clamp(1, TOPK_MAX_SLICES)
}

/// A top-k softmax over `sim` [m, p] after CUDA's checks: k = min(k, m), and the kernels' params.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TopkPlan {
    k: usize,
    partial: TopkPartial,
    merge: TopkMerge,
    row_sum: RowSum,
}

impl TopkPlan {
    /// Words each of the scratch keys and rows holds: every slice's best k of every column.
    fn entries(&self) -> usize {
        self.partial.slices as usize * self.k * self.partial.p as usize
    }

    /// Whether there are picks to make, so the partial top-k and the merge run (CUDA's `if (k)`
    /// after `if (!m || !p) return`; k > 0 means m > 0).
    fn picks(&self) -> bool {
        self.k > 0 && self.partial.p > 0
    }

    /// The partial top-k's kernel: the one whose lists hold k.
    fn partial_kernel(&self) -> &'static Kernel {
        if self.k <= 32 {
            &CUTIE_TOPK_PARTIAL_32
        } else {
            &CUTIE_TOPK_PARTIAL_64
        }
    }

    /// The partial top-k's grid: [`TOPK_COLS`] columns by [`TOPK_SLICES_PER_GROUP`] slices a
    /// threadgroup, fewer slices when the pipeline takes fewer threads (lists of 64 in registers
    /// can lower what it takes).
    fn partial_grid(&self, max_threads: u32) -> MetalResult<Grid> {
        let per_group = (max_threads / TOPK_COLS).clamp(1, TOPK_SLICES_PER_GROUP);
        Grid::groups(
            [(self.partial.p as usize).div_ceil(TOPK_COLS as usize), self.partial.slices.div_ceil(per_group) as usize, 1],
            [TOPK_COLS, per_group, 1],
        )
    }
}

fn topk_plan(m: usize, p: usize, k: usize) -> MetalResult<TopkPlan> {
    let op = "gpu_cutie_topk_softmax";
    let k = k.min(m);
    if k > CUTIE_TOPK_MAX {
        return Err(format!("{op}: top {k} of {m} rows; the kernel takes at most {CUTIE_TOPK_MAX} (Cutie's top_k is 30)"));
    }
    let (m_u, p_u) = (cutie_u32(op, m)?, cutie_u32(op, p)?);
    // The affinity is zeroed by the stream's fill, which counts its words in u32.
    let words = m.checked_mul(p).ok_or_else(|| format!("{op}: {m}x{p} does not fit the kernel's 32-bit sizes"))?;
    cutie_u32(op, words)?;
    let (m, p) = (m_u, p_u);
    let slices = topk_slices(m);
    let partial = TopkPartial { m, p, k: k as u32, chunk: m.div_ceil(slices), slices };
    Ok(TopkPlan {
        k,
        partial,
        merge: TopkMerge { m, p, k: k as u32, slices },
        row_sum: RowSum { rows: m, cols: p },
    })
}

/// A grid of one SIMD-group an item (a column, a row) over `items`, `want` to a threadgroup, fewer
/// when the pipeline takes fewer threads. `Err` unless the pipeline's SIMD-groups are 32 wide, the
/// width its butterflies are written for.
fn simd_grid(op: &str, kernel: &Kernel, limits: KernelLimits, items: usize, want: u32) -> MetalResult<Grid> {
    if limits.simd_width != SIMD_WIDTH {
        return Err(format!(
            "{op} ({}): written for {SIMD_WIDTH}-wide SIMD-groups, this GPU's are {}",
            kernel.function, limits.simd_width
        ));
    }
    let per_group = want.min(limits.max_threads / SIMD_WIDTH).max(1);
    Grid::groups([items.div_ceil(per_group as usize), 1, 1], [per_group * SIMD_WIDTH, 1, 1])
}

/// An attention mask from `aux` [n, p] after CUDA's checks: its rows (n * heads * queries) and the
/// kernels' params.
#[derive(Clone, Copy, Debug, PartialEq)]
struct AttnPlan {
    n: usize,
    p: usize,
    rows: usize,
    foreground: AttnForeground,
    mask: AttnMask,
}

impl AttnPlan {
    /// No rows or no pixels: the mask is empty and nothing runs (as on CUDA).
    fn is_empty(&self) -> bool {
        self.rows == 0 || self.p == 0
    }

    /// The foreground's grid-stride grid: a thread a pixel, at most [`FOREGROUND_MAX_GROUPS`]
    /// threadgroups.
    fn foreground_grid(&self) -> MetalResult<Grid> {
        Grid::groups([self.p.div_ceil(THREADS as usize).min(FOREGROUND_MAX_GROUPS), 1, 1], [THREADS, 1, 1])
    }

    /// The mask's grid: a thread a pixel across, at most [`MASK_MAX_ROW_GROUPS`] threadgroups
    /// down the rows.
    fn mask_grid(&self) -> MetalResult<Grid> {
        Grid::groups([self.p.div_ceil(THREADS as usize), self.rows.min(MASK_MAX_ROW_GROUPS), 1], [THREADS, 1, 1])
    }
}

fn attn_plan(n: usize, p: usize, heads: usize, queries: usize) -> MetalResult<AttnPlan> {
    let op = "gpu_cutie_attn_mask";
    let (n_u, heads_u, queries_u) = (cutie_u32(op, n)?, cutie_u32(op, heads)?, cutie_u32(op, queries)?);
    let rows = n
        .checked_mul(heads)
        .and_then(|rows| rows.checked_mul(queries))
        .ok_or_else(|| format!("{op}: {n} objects x {heads} heads x {queries} queries overflows"))?;
    // Each object's two flags are zeroed by the stream's fill, which counts its words in u32.
    let flags = n.checked_mul(2).ok_or_else(|| format!("{op}: {n} objects overflows"))?;
    cutie_u32(op, flags)?;
    Ok(AttnPlan {
        n,
        p,
        rows,
        foreground: AttnForeground { p: p as u64, n: n_u, pad: 0 },
        mask: AttnMask { p: p as u64, rows: rows as u64, heads: heads_u, queries: queries_u },
    })
}

#[cfg(target_os = "macos")]
fn cutie_check_f32(op: &str, x: &GpuTensor) -> MetalResult<()> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// Cutie's top-k softmax down each column of `sim` [m, p]: the k largest (k = min(k, m)), their
/// exp normalised to sum 1 (no max subtracted, as Cutie's `do_softmax`), every other entry 0.
/// Also each row's sum of the result ([1, m]): how much each memory token was used.
///
/// The picks are torch.topk's: values in PyTorch's radix-key order (-0 below +0, a NaN of either
/// sign above every number, so a NaN similarity is picked and its column's affinities come out
/// NaN, as PyTorch's do), equal keys by the lower row. The denominator sums the picks best first
/// in PyTorch's order (four accumulators), so the affinities are PyTorch's up to the bits of the
/// exp (Metal's `precise::exp` is not CUDA's). The usage sums each row in another order than
/// PyTorch's innermost reduce: fp32 rounding apart. With no columns the usage is 0, as the
/// reference's (CUDA leaves it unwritten).
///
/// Each column's rows are split into slices; a thread a column and slice keeps the slice's top k
/// (reads coalesced across columns), a SIMD-group a column merges them and does the softmax, then
/// a SIMD-group a row sums the usage. At most 64 a column are kept in registers: a larger k (after
/// k = min(k, m)) is an error, as on CUDA. Cutie's top_k is 30.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_topk_softmax(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    let op = "gpu_cutie_topk_softmax";
    cutie_check_f32(op, sim)?;
    let (m, p) = (sim.rows(), sim.cols());
    let plan = topk_plan(m, p, k)?;
    with_metal_stream(op, |s| {
        let aff = GpuTensor::alloc(m, p)?;
        let usage = GpuTensor::alloc(1, m)?;
        s.fill(&aff, 0)?;
        if plan.picks() {
            // Each slice's best k per column: their keys and rows.
            let keys = GpuTensor::alloc_u32(plan.entries())?;
            let idx = GpuTensor::alloc_u32(plan.entries())?;
            let partial = plan.partial_kernel();
            let grid = plan.partial_grid(s.limits(partial, &[])?.max_threads)?;
            s.launch(partial, &[], &plan.partial, &[Bind::In(sim), Bind::Out(&keys), Bind::Out(&idx)], grid)?;
            let grid = simd_grid(op, &CUTIE_TOPK_MERGE, s.limits(&CUTIE_TOPK_MERGE, &[])?, p, TOPK_MERGE_COLUMNS)?;
            let binds = [Bind::In(&keys), Bind::In(&idx), Bind::Out(&aff)];
            s.launch(&CUTIE_TOPK_MERGE, &[], &plan.merge, &binds, grid)?;
        }
        let grid = simd_grid(op, &CUTIE_ROW_SUM, s.limits(&CUTIE_ROW_SUM, &[])?, m, ROW_SUM_ROWS)?;
        s.launch(&CUTIE_ROW_SUM, &[], &plan.row_sum, &[Bind::In(&aff), Bind::Out(&usage)], grid)?;
        Ok((aff, usage))
    })
}

/// The object transformer's attention mask from its auxiliary logits `aux` [n, p] (one row per
/// object): sigmoid, aggregated with a background (as `gpu_cutie_aggregate`); a pixel is an
/// object's foreground where its logit is the largest of all rows (a NaN makes the largest NaN,
/// as `logits.max` does, and the pixel no object's foreground). For each object, each of
/// `heads` heads and each of `queries` queries a row of p: the first half of the queries may only
/// see the object's foreground, the second half only the rest; a row that would see nothing sees
/// everything. Out [n*heads*queries, p], additive: 0 where seen, -inf where blocked.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_attn_mask(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_attn_mask";
    cutie_check_f32(op, aux)?;
    let plan = attn_plan(aux.rows(), aux.cols(), heads, queries)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(plan.rows, plan.p)?;
        if plan.is_empty() {
            return Ok(out);
        }
        // Each object's foreground flags [n, p], then whether it has any foreground and any
        // background (zeroed: the kernel only sets them).
        let fg = GpuTensor::alloc_u32(plan.n * plan.p)?;
        let seen = GpuTensor::alloc_u32(2 * plan.n)?;
        s.fill(&seen, 0)?;
        let binds = [Bind::In(aux), Bind::Out(&fg), Bind::InOut(&seen)];
        s.launch(&CUTIE_ATTN_FOREGROUND, &[], &plan.foreground, &binds, plan.foreground_grid()?)?;
        let binds = [Bind::In(&fg), Bind::In(&seen), Bind::Out(&out)];
        s.launch(&CUTIE_ATTN_MASK, &[], &plan.mask, &binds, plan.mask_grid()?)?;
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The fields of `struct name { ... };` in the library's source, as (type, name).
    fn msl_struct(name: &str) -> Vec<(String, String)> {
        let source = CUTIE_TOPK_MASK.source;
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

    /// Checks MSL struct `name` has `fields` (MSL type, name) in order, and that the Rust twin
    /// lays them out as MSL does (natural alignment: uint 4 bytes, ulong 8): `offsets` and `size`.
    fn check_twin(name: &str, fields: &[(&str, &str)], offsets: &[usize], size: usize) {
        let msl = msl_struct(name);
        let want: Vec<(String, String)> = fields.iter().map(|(t, n)| (t.to_string(), n.to_string())).collect();
        assert_eq!(msl, want, "{name}: MSL fields");
        let mut at = 0usize;
        let mut align = 1usize;
        for ((ty, field), &offset) in fields.iter().zip(offsets) {
            let bytes = match *ty {
                "uint" => 4,
                "ulong" => 8,
                other => panic!("{name}: no layout for {other}"),
            };
            at = at.next_multiple_of(bytes);
            align = align.max(bytes);
            assert_eq!(offset, at, "{name}.{field}: offset");
            at += bytes;
        }
        assert_eq!(size, at.next_multiple_of(align), "{name}: size");
    }

    #[test]
    fn params_are_field_for_field_their_msl_twins() {
        check_twin(
            "CutieTopkPartial",
            &[("uint", "m"), ("uint", "p"), ("uint", "k"), ("uint", "chunk"), ("uint", "slices")],
            &[
                offset_of!(TopkPartial, m),
                offset_of!(TopkPartial, p),
                offset_of!(TopkPartial, k),
                offset_of!(TopkPartial, chunk),
                offset_of!(TopkPartial, slices),
            ],
            size_of::<TopkPartial>(),
        );
        check_twin(
            "CutieTopkMerge",
            &[("uint", "m"), ("uint", "p"), ("uint", "k"), ("uint", "slices")],
            &[
                offset_of!(TopkMerge, m),
                offset_of!(TopkMerge, p),
                offset_of!(TopkMerge, k),
                offset_of!(TopkMerge, slices),
            ],
            size_of::<TopkMerge>(),
        );
        check_twin(
            "CutieRowSum",
            &[("uint", "rows"), ("uint", "cols")],
            &[offset_of!(RowSum, rows), offset_of!(RowSum, cols)],
            size_of::<RowSum>(),
        );
        check_twin(
            "CutieAttnForeground",
            &[("ulong", "p"), ("uint", "n"), ("uint", "pad")],
            &[offset_of!(AttnForeground, p), offset_of!(AttnForeground, n), offset_of!(AttnForeground, pad)],
            size_of::<AttnForeground>(),
        );
        check_twin(
            "CutieAttnMask",
            &[("ulong", "p"), ("ulong", "rows"), ("uint", "heads"), ("uint", "queries")],
            &[
                offset_of!(AttnMask, p),
                offset_of!(AttnMask, rows),
                offset_of!(AttnMask, heads),
                offset_of!(AttnMask, queries),
            ],
            size_of::<AttnMask>(),
        );
    }

    #[test]
    fn every_kernel_is_in_the_source() {
        for kernel in TOPK_MASK_KERNELS {
            let named = CUTIE_TOPK_MASK.source.contains(&format!("kernel void {}(", kernel.function))
                || CUTIE_TOPK_MASK.source.contains(&format!("[[host_name(\"{}\")]]", kernel.function));
            assert!(named, "{} is not a kernel of {}", kernel.function, CUTIE_TOPK_MASK.name);
        }
    }

    #[test]
    fn topk_takes_cudas_k() {
        // k is min(k, m) first, so a k over 64 is refused only past 64 rows, with CUDA's message.
        let e = topk_plan(100, 9, 65).unwrap_err();
        assert_eq!(e, "gpu_cutie_topk_softmax: top 65 of 100 rows; the kernel takes at most 64 (Cutie's top_k is 30)");
        assert_eq!(topk_plan(20, 37, 30).unwrap().k, 20);
        assert_eq!(topk_plan(30000, 16, 64).unwrap().partial_kernel().function, "cutie_topk_partial_64");
        assert_eq!(topk_plan(3500, 16, 30).unwrap().partial_kernel().function, "cutie_topk_partial_32");
        assert_eq!(topk_plan(1, 33, 30).unwrap().k, 1);
        let none = topk_plan(50, 9, 0).unwrap();
        assert!(!none.picks());
        assert_eq!(none.entries(), 0);
        assert!(!topk_plan(0, 9, 30).unwrap().picks());
        assert!(!topk_plan(40, 0, 30).unwrap().picks());
        let e = topk_plan(1 << 32, 1, 30).unwrap_err();
        assert_eq!(e, "gpu_cutie_topk_softmax: 4294967296 does not fit the kernel's 32-bit sizes");
        // The affinity's zero fill counts in u32: m * p must fit, before anything is encoded.
        let e = topk_plan(1 << 16, 1 << 16, 30).unwrap_err();
        assert_eq!(e, "gpu_cutie_topk_softmax: 4294967296 does not fit the kernel's 32-bit sizes");
        assert_eq!(topk_plan(65535, 65537, 30).unwrap().partial.p, 65537, "u32::MAX elements");
    }

    #[test]
    fn topk_slices_cover_every_row() {
        // The ops check's sizes: Cutie's reads, long-term memory, the specials over 3 slices.
        let cases = [(336, 2, 168), (3500, 14, 250), (30000, 64, 469), (700, 3, 234), (1, 1, 1), (256, 1, 256)];
        for (m, slices, chunk) in cases {
            let plan = topk_plan(m, 8, 30).unwrap();
            assert_eq!((plan.partial.slices, plan.partial.chunk), (slices, chunk), "m {m}");
            assert_eq!(plan.merge.slices, slices, "m {m}");
        }
        for m in (1..=40_000u32).chain([1 << 20, u32::MAX]) {
            let slices = topk_slices(m);
            let chunk = m.div_ceil(slices);
            assert!((1..=64).contains(&slices), "m {m}: {slices} slices");
            assert!(slices as u64 * chunk as u64 >= m as u64, "m {m}: {slices} slices of {chunk} miss rows");
        }
    }

    #[test]
    fn topk_grids() {
        let plan = topk_plan(30000, 2500, 64).unwrap();
        let grid = plan.partial_grid(1024).unwrap();
        assert_eq!((grid.groups, grid.threads), ([79, 16, 1], [32, 4, 1]));
        // A pipeline that takes 96 threads: three slices a threadgroup.
        let grid = plan.partial_grid(96).unwrap();
        assert_eq!((grid.groups, grid.threads), ([79, 22, 1], [32, 3, 1]));
        let limits = KernelLimits { max_threads: 1024, simd_width: 32 };
        let grid = simd_grid("op", &CUTIE_TOPK_MERGE, limits, 2500, TOPK_MERGE_COLUMNS).unwrap();
        assert_eq!((grid.groups, grid.threads), ([625, 1, 1], [128, 1, 1]));
        let grid = simd_grid("op", &CUTIE_ROW_SUM, limits, 30000, ROW_SUM_ROWS).unwrap();
        assert_eq!((grid.groups, grid.threads), ([3750, 1, 1], [256, 1, 1]));
        let small = KernelLimits { max_threads: 64, simd_width: 32 };
        let grid = simd_grid("op", &CUTIE_ROW_SUM, small, 5, ROW_SUM_ROWS).unwrap();
        assert_eq!((grid.groups, grid.threads), ([3, 1, 1], [64, 1, 1]));
        assert!(simd_grid("op", &CUTIE_ROW_SUM, limits, 0, ROW_SUM_ROWS).unwrap().is_empty());
        let other = KernelLimits { max_threads: 1024, simd_width: 64 };
        let e = simd_grid("op", &CUTIE_TOPK_MERGE, other, 9, TOPK_MERGE_COLUMNS).unwrap_err();
        assert_eq!(e, "op (cutie_topk_merge): written for 32-wide SIMD-groups, this GPU's are 64");
    }

    #[test]
    fn attn_mask_plans() {
        let plan = attn_plan(3, 37, 8, 16).unwrap();
        assert_eq!(plan.rows, 384);
        assert!(!plan.is_empty());
        assert_eq!(plan.mask, AttnMask { p: 37, rows: 384, heads: 8, queries: 16 });
        assert_eq!(plan.foreground, AttnForeground { p: 37, n: 3, pad: 0 });
        let grid = plan.mask_grid().unwrap();
        assert_eq!((grid.groups, grid.threads), ([1, 384, 1], [256, 1, 1]));
        // Past CUDA's grid height the rows are stepped through; past its blocks, the pixels.
        let tall = attn_plan(9, 7056, 100, 100).unwrap();
        assert_eq!(tall.mask_grid().unwrap().groups, [28, 65535, 1]);
        let wide = attn_plan(1, 1 << 25, 1, 1).unwrap();
        assert_eq!(wide.foreground_grid().unwrap().groups, [65536, 1, 1]);
        assert!(attn_plan(0, 37, 8, 16).unwrap().is_empty());
        assert!(attn_plan(2, 0, 8, 16).unwrap().is_empty());
        assert!(attn_plan(2, 37, 8, 0).unwrap().is_empty());
        let e = attn_plan(1 << 32, 1, 1, 1).unwrap_err();
        assert_eq!(e, "gpu_cutie_attn_mask: 4294967296 does not fit the kernel's 32-bit sizes");
        let e = attn_plan(u32::MAX as usize, 1, u32::MAX as usize, u32::MAX as usize).unwrap_err();
        assert!(e.ends_with("queries overflows"), "{e}");
        // The flags' zero fill counts in u32: two words an object.
        let e = attn_plan(1 << 31, 1, 1, 1).unwrap_err();
        assert_eq!(e, "gpu_cutie_attn_mask: 4294967296 does not fit the kernel's 32-bit sizes");
        assert_eq!(attn_plan((1 << 31) - 1, 1, 1, 1).unwrap().n, (1 << 31) - 1);
    }
}
