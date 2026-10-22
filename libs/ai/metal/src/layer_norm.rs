//! PyTorch's LayerNorm on Makepad's Metal kernel (RationalVisions docs/cutie-metal.md §6):
//! `gpu_layer_norm_pytorch` at the signature of models/common's macOS function (CUDA's in
//! cuda/src/launch.rs), with CUDA's checks and messages, for every model that calls it. Cutie's
//! `gpu_cutie_layer_norm` (`cutie_ops::softmax_norm`) launches the same kernel with its affine
//! cached, so the ops check's EXACT comparison of the two holds by construction, as on CUDA.
//!
//! The kernel is `shaders/makepad/layer_norm.metal`, ported from cuda/kernels/diffusion_ops.cu's
//! Welford tree (PyTorch's CUDA fast path) with CUDA's threadgroup of [32, 4]. It is a library of
//! its own, apart from Cutie's: a model that is not Cutie gets its layer norm from this one
//! kernel, so until the Mac has compiled them, a fault in a Cutie library cannot break it.
//!
//! The kernels index in u32 where CUDA's count in size_t, so a tensor of more than u32::MAX
//! elements is `Err` here (`cutie_u32`'s message) where CUDA would run it: four billion floats.
//! The checks, the params and the grid are plain functions of the sizes, compiled and tested on
//! any OS, with the kernel's reduction tree run on the host as the MSL writes it; the op is macOS
//! only.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use crate::stream::{Grid, Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::stream::gemm::check_simd_width;
#[cfg(target_os = "macos")]
use crate::stream::{upload, with_metal_stream, Bind, MetalStream};
#[cfg(target_os = "macos")]
use crate::GpuTensor;

/// PyTorch's LayerNorm (`layer_norm.metal`).
pub static LAYER_NORM: KernelLib = KernelLib {
    name: "layer_norm",
    source: include_str!("../shaders/makepad/layer_norm.metal"),
};

/// nn.LayerNorm over each row, a threadgroup a row ([`LayerNormPytorch`]).
pub static LAYER_NORM_PYTORCH: Kernel = Kernel { lib: &LAYER_NORM, function: "layer_norm_pytorch_f32" };

/// Every kernel above, for the bring-up test that compiles them all.
pub static LAYER_NORM_KERNELS: &[&Kernel] = &[&LAYER_NORM_PYTORCH];

/// The threadgroup: PyTorch's dim3(warp_size, 4).
const LAYER_NORM_THREADS: [u32; 3] = [32, 4, 1];

/// `layer_norm_pytorch_f32`'s params: rows of `cols` (a multiple of 4), and eps.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerNormPytorch {
    pub cols: u32,
    pub eps: f32,
}
// SAFETY: #[repr(C)], 8 bytes, no pointers; `LayerNormPytorch` in MSL.
unsafe impl KernelParams for LayerNormPytorch {}

/// A size as the kernel takes it (CUDA's `cutie_u32`, with its message).
fn kernel_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// The params for `x` [rows, cols] under `op`, once its caller has made its own shape check:
/// every size and the element count as the kernel's u32.
pub(crate) fn layer_norm_params(op: &str, rows: usize, cols: usize, eps: f32) -> MetalResult<LayerNormPytorch> {
    kernel_u32(op, rows)?;
    let cols_u = kernel_u32(op, cols)?;
    if rows != 0 && cols != 0 {
        let n = rows.checked_mul(cols).ok_or_else(|| format!("{op}: {rows}x{cols} does not fit the kernel's 32-bit sizes"))?;
        kernel_u32(op, n)?;
    }
    Ok(LayerNormPytorch { cols: cols_u, eps })
}

/// `gpu_layer_norm_pytorch`'s check, with CUDA's message.
pub(crate) fn layer_norm_pytorch_params(
    scale: usize,
    bias: usize,
    rows: usize,
    cols: usize,
    eps: f32,
) -> MetalResult<LayerNormPytorch> {
    if scale != cols || bias != cols || !cols.is_multiple_of(4) {
        return Err("gpu_layer_norm_pytorch width mismatch".to_string());
    }
    layer_norm_params("gpu_layer_norm_pytorch", rows, cols, eps)
}

/// A threadgroup of [32, 4] a row, as PyTorch's launch; nothing when there are no elements.
pub(crate) fn layer_norm_grid(rows: usize, cols: usize) -> MetalResult<Grid> {
    let groups = if cols == 0 { 0 } else { rows };
    Grid::groups([groups, 1, 1], LAYER_NORM_THREADS)
}

/// One launch of the kernel over `x` into `out` under `op`, gamma and beta bound as the caller
/// keeps them (uploaded, or cached weights). `Err` unless the pipeline's SIMD-groups are 32 wide:
/// its shuffle-down tree is CUDA's over a warp's 32 lanes.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_layer_norm(
    s: &mut MetalStream<'_>,
    op: &str,
    x: &GpuTensor,
    gamma: Bind<'_>,
    beta: Bind<'_>,
    out: &GpuTensor,
    params: &LayerNormPytorch,
    grid: Grid,
) -> MetalResult<()> {
    check_simd_width(s, op, &LAYER_NORM_PYTORCH)?;
    s.launch(&LAYER_NORM_PYTORCH, &[], params, &[Bind::In(x), gamma, beta, Bind::Out(out)], grid)
}

/// nn.LayerNorm over each row of `x` [rows, cols] (cols % 4 == 0) as PyTorch's CUDA fast path
/// computes it: float4 steps of online Welford moments, its fixed tree over four 32-lane groups,
/// and gamma * (rstd * (x - mean)) + beta; CUDA's `gpu_layer_norm_pytorch`, the affine uploaded on
/// every call.
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

/// The checks, params and grid on any OS: CUDA's messages and launch, the params struct field for
/// field its MSL twin, and the kernel's reduction tree run on the host as the MSL writes it,
/// against sums in f64.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_is_its_kernel_and_its_params() {
        let source = LAYER_NORM.source;
        for kernel in LAYER_NORM_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &LAYER_NORM));
            assert!(source.contains(&format!("kernel void {}(", kernel.function)), "{} is not declared", kernel.function);
        }
        assert_eq!(source.matches("kernel void ").count(), LAYER_NORM_KERNELS.len(), "a kernel the list misses");
        assert!(source.lines().any(|line| line.trim() == "#include \"makepad_common.h\""));
        let half = source.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).any(|word| word.starts_with("half"));
        assert!(!half, "no half anywhere (D8)");
        // The params struct, field for field, read at [[buffer(0)]].
        let start = source.find("struct LayerNormPytorch {").expect("no struct LayerNormPytorch");
        let body = &source[start..];
        let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
        let fields: Vec<&str> = body.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
        assert_eq!(fields, ["uint cols;", "float eps;"]);
        assert_eq!(size_of::<LayerNormPytorch>(), 8);
        assert_eq!(format!("{:?}", LayerNormPytorch { cols: 1, eps: 2.0 }), "LayerNormPytorch { cols: 1, eps: 2.0 }");
        let signature = "kernel void layer_norm_pytorch_f32(constant LayerNormPytorch& p [[buffer(0)]]";
        assert!(source.contains(signature), "no `{signature}` in the source");
    }

    #[test]
    fn shape_errors_are_cudas() {
        let width = "gpu_layer_norm_pytorch width mismatch";
        assert_eq!(layer_norm_pytorch_params(255, 256, 3, 256, 1e-5).unwrap_err(), width);
        assert_eq!(layer_norm_pytorch_params(256, 255, 3, 256, 1e-5).unwrap_err(), width);
        assert_eq!(layer_norm_pytorch_params(6, 6, 3, 6, 1e-5).unwrap_err(), width, "6 columns are not a multiple of 4");
        assert_eq!(layer_norm_pytorch_params(256, 256, 7, 256, 1e-5).unwrap(), LayerNormPytorch { cols: 256, eps: 1e-5 });
    }

    #[test]
    fn a_threadgroup_a_row_and_nothing_when_empty() {
        let grid = layer_norm_grid(96 * 56, 256).unwrap();
        assert_eq!((grid.groups, grid.threads), ([96 * 56, 1, 1], [32, 4, 1]), "PyTorch's dim3(32, 4) a row");
        assert!(layer_norm_grid(0, 256).unwrap().is_empty());
        assert!(layer_norm_grid(3, 0).unwrap().is_empty());
        assert_eq!(layer_norm_pytorch_params(0, 0, 3, 0, 1e-5).unwrap().cols, 0, "no columns is 4-aligned");
        assert_eq!(layer_norm_pytorch_params(256, 256, 0, 256, 1e-5).unwrap().cols, 256, "no rows");
    }

    #[test]
    fn sizes_past_u32_are_errors_with_cudas_message() {
        let big = 1usize << 32;
        let op = "gpu_layer_norm_pytorch";
        assert_eq!(layer_norm_params(op, big, 4, 1e-5).unwrap_err(), format!("{op}: {big} does not fit the kernel's 32-bit sizes"));
        assert_eq!(
            layer_norm_params(op, 1 << 16, 1 << 16, 1e-5).unwrap_err(),
            format!("{op}: {big} does not fit the kernel's 32-bit sizes"),
            "the element count"
        );
        assert_eq!(
            layer_norm_pytorch_params(1 << 16, 1 << 16, 1 << 16, 1 << 16, 1e-5).unwrap_err(),
            format!("{op}: {big} does not fit the kernel's 32-bit sizes")
        );
        assert!(layer_norm_params(op, 1 << 16, (1 << 16) - 4, 1e-5).is_ok());
    }

    // --- The kernel's tree on the host, as the MSL writes it ---

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

    /// One thread's moments, as `layer_norm::Welford`.
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

    /// `layer_norm_pytorch_f32` for one row as the MSL writes it, thread by thread over [32, 4]:
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
                for &value in &row[4 * index..4 * index + 4] {
                    *moments = welford_online(value, *moments);
                }
            }
        }
        let mut offset = 16;
        while offset > 0 {
            let before = wd.clone();
            for (warp, lanes) in wd.chunks_mut(width).enumerate() {
                let before = &before[warp * width..(warp + 1) * width];
                for (lane, moments) in lanes.iter_mut().enumerate() {
                    let other = before.get(lane + offset).copied().unwrap_or(before[lane]);
                    *moments = welford_combine(before[lane], other);
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
    fn the_welford_tree_is_a_layer_norm() {
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

/// The op on the Mac against f64, in the host binding, and the errors CUDA's wrapper gives.
/// `cargo test -p makepad-ai-metal --lib layer_norm`, and again with MAKEPAD_METAL_MISALIGN=1.
/// Cutie's cached layer norm against this one, bit for bit, is `cutie_ops::softmax_norm`'s.
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

    #[test]
    fn the_kernel_compiles_with_32_lane_simd_groups() {
        assert!(stream::library_available(&LAYER_NORM), "layer_norm does not compile");
        with_metal_stream("layer norm: pipeline", |s| check_simd_width(s, "layer norm", &LAYER_NORM_PYTORCH)).unwrap();
    }

    #[test]
    fn layer_norm_pytorch_against_f64() {
        let mut rng = Rng(0x5eed_c105);
        // SAM's and DINO's widths, Cutie's 256, an odd number of vectors, one vector.
        for (rows, cols) in [(16, 256), (7, 516), (3, 4), (2, 1024), (5, 1280)] {
            let x = rng.uniforms(rows * cols, -3.0, 3.0);
            let gamma = rng.uniforms(cols, -1.0, 1.0);
            let beta = rng.uniforms(cols, -0.5, 0.5);
            let got = stream::download(&gpu_layer_norm_pytorch(&stream::upload(&x, rows, cols).unwrap(), &gamma, &beta, 1e-5).unwrap()).unwrap();
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
                let rstd = 1.0 / (var + 1e-5).sqrt();
                for i in 0..cols {
                    let want = gamma[i] as f64 * (rstd * (row[i] as f64 - mean)) + beta[i] as f64;
                    let g = got[r * cols + i] as f64;
                    assert!((g - want).abs() <= 1e-5 * want.abs().max(1.0), "{rows}x{cols} [{r}][{i}]: {g:e}, f64 gives {want:e}");
                }
            }
        }
        let empty = gpu_layer_norm_pytorch(&stream::upload(&[], 0, 8).unwrap(), &[1.0; 8], &[0.0; 8], 1e-5).unwrap();
        assert_eq!((empty.rows(), empty.cols()), (0, 8));
        let x = stream::upload(&[1.0; 12], 2, 6).unwrap();
        match gpu_layer_norm_pytorch(&x, &[1.0; 6], &[0.0; 6], 1e-5) {
            Ok(_) => panic!("6 columns are not a multiple of 4"),
            Err(e) => assert_eq!(e, "gpu_layer_norm_pytorch width mismatch"),
        }
    }
}
