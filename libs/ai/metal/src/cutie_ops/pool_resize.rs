//! Cutie's pooling and resizing on the Metal stream (RationalVisions docs/cutie-metal.md, step 8,
//! family (b)): `gpu_cutie_maxpool3x3s2` and `gpu_cutie_resize_bilinear` at the signatures of
//! cuda/src/cutie_ops.rs, with its checks and messages, launching the kernels of
//! `shaders/makepad/cutie_pool_resize.metal` (ported from cuda/kernels/cutie.cu). Both are held to
//! their host references' bits (cuda/src/cutie_reference.rs): the max-pool only compares, and the
//! resize is PyTorch's deterministic CUDA bilinear (torch._decomp's `_upsample_linear`), one fp32
//! operation at a time in the same order.
//!
//! The library, its kernels, their params and the checks compile on every OS and are tested
//! anywhere; the ops themselves are macOS only. Every size is checked, and every element count
//! made a u32, before anything is encoded: a failure is an `Err` with nothing launched.

use crate::stream::{Kernel, KernelLib, KernelParams, MetalResult};
#[cfg(target_os = "macos")]
use crate::gpu_types::GpuTensor;
#[cfg(target_os = "macos")]
use crate::stream::{with_metal_stream, Bind, Grid};

/// The max-pool and the bilinear resize (`cutie_pool_resize.metal`).
pub static CUTIE_POOL_RESIZE: KernelLib = KernelLib {
    name: "cutie_pool_resize",
    source: include_str!("../../shaders/makepad/cutie_pool_resize.metal"),
};

/// MaxPool2d(3, stride 2, padding 1), one thread an output ([`CutieMaxpool`]).
pub static CUTIE_MAXPOOL3X3S2: Kernel = Kernel { lib: &CUTIE_POOL_RESIZE, function: "cutie_maxpool3x3s2_f32" };
/// Bilinear resize, align_corners false, one thread an output ([`CutieResizeBilinear`]).
pub static CUTIE_RESIZE_BILINEAR: Kernel = Kernel { lib: &CUTIE_POOL_RESIZE, function: "cutie_resize_bilinear_f32" };

/// Every kernel above, for the bring-up test that compiles them all.
pub static POOL_RESIZE_KERNELS: &[&Kernel] = &[&CUTIE_MAXPOOL3X3S2, &CUTIE_RESIZE_BILINEAR];

/// Threads to a threadgroup: one thread an output.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const THREADS: u32 = 256;

/// `cutie_maxpool3x3s2_f32`'s params: the input's plane, the output's, and the outputs over
/// every plane (`ow * oh * planes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieMaxpool {
    pub width: u32,
    pub height: u32,
    pub ow: u32,
    pub oh: u32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 20 bytes, no pointers; `CutieMaxpool` in MSL.
unsafe impl KernelParams for CutieMaxpool {}

/// `cutie_resize_bilinear_f32`'s params: the input's plane, the output's, each axis's scale, and
/// the outputs over every plane (`out_w * out_h * planes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutieResizeBilinear {
    pub in_w: u32,
    pub in_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub n: u32,
}
// SAFETY: #[repr(C)], 28 bytes, no pointers; `CutieResizeBilinear` in MSL.
unsafe impl KernelParams for CutieResizeBilinear {}

/// A size as the kernels take it, with CUDA's message (`cutie_u32` in cuda/src/cutie_ops.rs).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cutie_u32(op: &str, v: usize) -> MetalResult<u32> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// The element count of a `[rows, cols]` tensor as a kernel's u32 index, with `cutie_u32`'s
/// message: the kernels index every element with a uint.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cutie_count(op: &str, rows: usize, cols: usize) -> MetalResult<u32> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{op}: {rows}x{cols} does not fit the kernel's 32-bit sizes"))?;
    cutie_u32(op, n)
}

/// A convolution's output size along one axis (PyTorch's floor rule; `cutie_conv_out` in
/// cuda/src/cutie_reference.rs).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

/// One axis's scale for the resize: Python's in / out (a double), made f32 as PyTorch's CUDA
/// multiply takes a scalar (`cutie_bilinear_scale` in cuda/src/cutie_reference.rs).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cutie_bilinear_scale(size_in: usize, size_out: usize) -> f32 {
    (size_in as f64 / size_out as f64) as f32
}

/// The max-pool's params for a `[rows, cols]` input of `w x h` planes, after CUDA's checks.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn maxpool_params(rows: usize, cols: usize, w: usize, h: usize) -> MetalResult<CutieMaxpool> {
    let op = "gpu_cutie_maxpool3x3s2";
    if w.checked_mul(h) != Some(cols) || w == 0 || h == 0 {
        return Err("gpu_cutie_maxpool3x3s2 shape mismatch".into());
    }
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    let (width, height) = (cutie_u32(op, w)?, cutie_u32(op, h)?);
    cutie_count(op, rows, cols)?;
    let n = cutie_count(op, rows, ow * oh)?;
    Ok(CutieMaxpool { width, height, ow: ow as u32, oh: oh as u32, n })
}

/// The resize's params for a `[rows, cols]` input of `in_w x in_h` planes to `out_w x out_h`,
/// after CUDA's checks.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn resize_params(
    rows: usize,
    cols: usize,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> MetalResult<CutieResizeBilinear> {
    let op = "gpu_cutie_resize_bilinear";
    if in_w.checked_mul(in_h) != Some(cols) || in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err(format!("{op}: {rows}x{cols} for {in_w}x{in_h} to {out_w}x{out_h}"));
    }
    let d = |v: usize| cutie_u32(op, v);
    let (_, iw, ih, ow, oh) = (d(rows)?, d(in_w)?, d(in_h)?, d(out_w)?, d(out_h)?);
    cutie_count(op, rows, cols)?;
    let out_plane = out_w
        .checked_mul(out_h)
        .ok_or_else(|| format!("{op}: {out_w}x{out_h} does not fit the kernel's 32-bit sizes"))?;
    let n = cutie_count(op, rows, out_plane)?;
    Ok(CutieResizeBilinear {
        in_w: iw,
        in_h: ih,
        out_w: ow,
        out_h: oh,
        scale_x: cutie_bilinear_scale(in_w, out_w),
        scale_y: cutie_bilinear_scale(in_h, out_h),
        n,
    })
}

#[cfg(target_os = "macos")]
fn cutie_check_f32(op: &str, x: &GpuTensor) -> MetalResult<()> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// MaxPool2d(3, stride 2, padding 1) on planar [planes, h*w]. Out [planes, oh*ow], oh = (h-1)/2+1.
/// A NaN in a window wins, as in PyTorch.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_maxpool3x3s2(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_maxpool3x3s2";
    cutie_check_f32(op, x)?;
    let params = maxpool_params(x.rows(), x.cols(), w, h)?;
    let out_cols = params.ow as usize * params.oh as usize;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), out_cols)?;
        s.launch(&CUTIE_MAXPOOL3X3S2, &[], &params, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

/// `F.interpolate(mode='bilinear', align_corners=False)` of planar [planes, in_h*in_w] to
/// [planes, out_h*out_w], bit for bit as PyTorch computes it on CUDA with deterministic
/// algorithms on (cuda/src/cutie_ops.rs says how, and what it was checked against): each axis's
/// scale in / out (a double made f32), the source scale * (dst + 0.5) - 0.5 clamped at 0, the
/// lower index truncated, the upper the next one but at the edge, the lambda source - lower, then
/// lerps a + (b - a) * t along the width (each source row), then the height.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_resize_bilinear(
    x: &GpuTensor,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_resize_bilinear";
    cutie_check_f32(op, x)?;
    let params = resize_params(x.rows(), x.cols(), in_w, in_h, out_w, out_h)?;
    let grid = Grid::linear(params.n as usize, THREADS)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(x.rows(), out_w * out_h)?;
        s.launch(&CUTIE_RESIZE_BILINEAR, &[], &params, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    //! What can be checked without a GPU: the library declares its kernels and includes the
    //! header, each params struct is its MSL twin, the checks are CUDA's, and the kernels'
    //! arithmetic, as the MSL writes it, run on the host: the max-pool's unsigned window against
    //! the reference's signed one, the resize against PyTorch's own values.
    use super::*;

    /// The fields of `struct name { ... };` in `source`, as (type, name), in order.
    fn msl_struct_fields(source: &str, name: &str) -> Vec<(String, String)> {
        let start = source
            .find(&format!("struct {name} {{"))
            .unwrap_or_else(|| panic!("struct {name} is not in {}.metal", CUTIE_POOL_RESIZE.name));
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
        for kernel in POOL_RESIZE_KERNELS {
            assert!(std::ptr::eq(kernel.lib, &CUTIE_POOL_RESIZE));
            let declared = format!("kernel void {}(", kernel.function);
            assert!(CUTIE_POOL_RESIZE.source.contains(&declared), "{} is not declared", kernel.function);
        }
        assert!(CUTIE_POOL_RESIZE.source.lines().any(|line| line.trim() == "#include \"makepad_common.h\""));
        let half = CUTIE_POOL_RESIZE
            .source
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|word| word.starts_with("half"));
        assert!(!half, "no half anywhere (D8)");
    }

    #[test]
    fn params_structs_are_their_msl_twins() {
        let source = CUTIE_POOL_RESIZE.source;
        assert_eq!(std::mem::size_of::<CutieMaxpool>(), 20);
        assert_eq!(
            msl_struct_fields(source, "CutieMaxpool"),
            fields(&[("uint", "width"), ("uint", "height"), ("uint", "ow"), ("uint", "oh"), ("uint", "n")])
        );
        assert_eq!(std::mem::size_of::<CutieResizeBilinear>(), 28);
        assert_eq!(
            msl_struct_fields(source, "CutieResizeBilinear"),
            fields(&[
                ("uint", "in_w"),
                ("uint", "in_h"),
                ("uint", "out_w"),
                ("uint", "out_h"),
                ("float", "scale_x"),
                ("float", "scale_y"),
                ("uint", "n"),
            ])
        );
    }

    #[test]
    fn maxpool_checks_as_cuda_does() {
        let mismatch = "gpu_cutie_maxpool3x3s2 shape mismatch";
        assert_eq!(maxpool_params(2, 10, 3, 3).unwrap_err(), mismatch);
        assert_eq!(maxpool_params(2, 0, 0, 4).unwrap_err(), mismatch);
        assert_eq!(maxpool_params(2, 0, 4, 0).unwrap_err(), mismatch);
        assert_eq!(maxpool_params(1, 0, usize::MAX, 2).unwrap_err(), mismatch, "w * h overflows");
        let p = maxpool_params(64, 192 * 112, 192, 112).unwrap();
        assert_eq!(p, CutieMaxpool { width: 192, height: 112, ow: 96, oh: 56, n: 64 * 96 * 56 });
        for (w, h, ow, oh) in [(1, 1, 1, 1), (2, 3, 1, 2), (15, 9, 8, 5), (7, 1, 4, 1), (16, 16, 8, 8)] {
            let p = maxpool_params(3, w * h, w, h).unwrap();
            assert_eq!((p.ow, p.oh, p.n), (ow, oh, 3 * ow * oh), "{w}x{h}");
        }
        assert_eq!(maxpool_params(0, 12, 4, 3).unwrap().n, 0, "no planes: an empty grid");
        let wide = maxpool_params(1, 1 << 32, 1 << 32, 1).unwrap_err();
        assert!(wide.contains("4294967296 does not fit the kernel's 32-bit sizes"), "{wide}");
        let many = maxpool_params(1 << 21, 1 << 11, 1 << 11, 1).unwrap_err();
        assert!(many.contains("does not fit the kernel's 32-bit sizes"), "{many}");
    }

    #[test]
    fn resize_checks_as_cuda_does() {
        let op = "gpu_cutie_resize_bilinear";
        assert_eq!(resize_params(3, 35, 7, 4, 5, 8).unwrap_err(), format!("{op}: 3x35 for 7x4 to 5x8"));
        assert_eq!(resize_params(3, 0, 0, 5, 5, 8).unwrap_err(), format!("{op}: 3x0 for 0x5 to 5x8"));
        assert_eq!(resize_params(3, 35, 7, 5, 0, 8).unwrap_err(), format!("{op}: 3x35 for 7x5 to 0x8"));
        assert_eq!(resize_params(3, 35, 7, 5, 5, 0).unwrap_err(), format!("{op}: 3x35 for 7x5 to 5x0"));
        let p = resize_params(3, 1920 * 1080, 1920, 1080, 1777, 1000).unwrap();
        assert_eq!((p.in_w, p.in_h, p.out_w, p.out_h, p.n), (1920, 1080, 1777, 1000, 3 * 1777 * 1000));
        assert_eq!(p.scale_x.to_bits(), ((1920.0f64 / 1777.0) as f32).to_bits());
        assert_eq!(p.scale_y.to_bits(), ((1080.0f64 / 1000.0) as f32).to_bits());
        assert_eq!(resize_params(0, 35, 7, 5, 5, 8).unwrap().n, 0, "no planes: an empty grid");
        let tall = resize_params(1, 7, 7, 1, 5, 1 << 32).unwrap_err();
        assert_eq!(tall, format!("{op}: 4294967296 does not fit the kernel's 32-bit sizes"));
        let many = resize_params(1 << 12, 4, 2, 2, 1 << 10, 1 << 10).unwrap_err();
        assert!(many.contains("does not fit the kernel's 32-bit sizes"), "{many}");
    }

    /// A small deterministic generator (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Values in [-1, 1), every 13th a NaN with its own payload, so which NaN wins shows.
        fn values(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let r = self.next();
                    if i % 13 == 5 {
                        f32::from_bits(0x7fc0_0000 | (r as u32 & 0x003f_ffff) | 1)
                    } else {
                        (r >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                    }
                })
                .collect()
        }
    }

    /// `cutie_maxpool3x3s2_f32` as the MSL writes it, thread by thread, in u32 arithmetic.
    fn maxpool_as_msl(x: &[f32], p: &CutieMaxpool) -> Vec<f32> {
        (0..p.n)
            .map(|i| {
                let plane = p.ow * p.oh;
                let (ox, oy) = (i % p.ow, (i % plane) / p.ow);
                let src = &x[((i / plane) * (p.width * p.height)) as usize..];
                let mut best = f32::NEG_INFINITY;
                for ky in 0..3 {
                    let y1 = oy * 2 + ky;
                    if y1 == 0 || y1 > p.height {
                        continue;
                    }
                    let row = &src[((y1 - 1) * p.width) as usize..];
                    for kx in 0..3 {
                        let x1 = ox * 2 + kx;
                        if x1 == 0 || x1 > p.width {
                            continue;
                        }
                        let v = row[(x1 - 1) as usize];
                        if v > best || v.is_nan() {
                            best = v;
                        }
                    }
                }
                best
            })
            .collect()
    }

    /// The reference's loop (cuda/src/cutie_reference.rs), with its signed window.
    fn maxpool_reference(x: &[f32], planes: usize, w: usize, h: usize) -> Vec<f32> {
        let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
        let mut out = vec![0.0f32; planes * ow * oh];
        for p in 0..planes {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    for ky in 0..3 {
                        let y = (oy * 2 + ky) as isize - 1;
                        if y < 0 || y >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let xx = (ox * 2 + kx) as isize - 1;
                            if xx < 0 || xx >= w as isize {
                                continue;
                            }
                            let tap = x[p * w * h + y as usize * w + xx as usize];
                            if tap > best || tap.is_nan() {
                                best = tap;
                            }
                        }
                    }
                    out[p * ow * oh + oy * ow + ox] = best;
                }
            }
        }
        out
    }

    #[test]
    fn the_maxpool_kernels_unsigned_window_is_the_references() {
        let mut rng = Rng(0x5eed_9001);
        let mut sizes: Vec<(usize, usize)> = (1..=9).flat_map(|w| (1..=9).map(move |h| (w, h))).collect();
        sizes.extend([(37, 23), (192, 112), (2, 31)]);
        for (w, h) in sizes {
            let planes = 3;
            let x = rng.values(planes * w * h);
            let p = maxpool_params(planes, w * h, w, h).unwrap();
            let got = maxpool_as_msl(&x, &p);
            let want = maxpool_reference(&x, planes, w, h);
            let bits = |v: &[f32]| v.iter().map(|t| t.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{w}x{h}");
        }
    }

    /// `cutie_resize_bilinear_f32` as the MSL writes it, thread by thread: u32 indices, and each
    /// fp32 operation rounded on its own (Rust never contracts).
    fn resize_as_msl(x: &[f32], p: &CutieResizeBilinear) -> Vec<f32> {
        let source = |scale: f32, dst: u32, size_in: u32| {
            let center = dst as f32 + 0.5;
            let scaled = scale * center;
            let mut src = scaled - 0.5;
            src = if src < 0.0 { 0.0 } else { src };
            let lower = src as u32;
            let upper = lower + if lower < size_in - 1 { 1 } else { 0 };
            (lower, upper, src - lower as f32)
        };
        let lerp = |a: f32, b: f32, t: f32| {
            let diff = b - a;
            let step = diff * t;
            a + step
        };
        (0..p.n)
            .map(|i| {
                let out_plane = p.out_w * p.out_h;
                let at = i % out_plane;
                let (oy, ox) = (at / p.out_w, at % p.out_w);
                let (y0, y1, h) = source(p.scale_y, oy, p.in_h);
                let (x0, x1, w) = source(p.scale_x, ox, p.in_w);
                let src = ((i / out_plane) * (p.in_w * p.in_h)) as usize;
                let row0 = &x[src + (y0 * p.in_w) as usize..];
                let row1 = &x[src + (y1 * p.in_w) as usize..];
                let top = lerp(row0[x0 as usize], row0[x1 as usize], w);
                let bottom = lerp(row1[x0 as usize], row1[x1 as usize], w);
                lerp(top, bottom, h)
            })
            .collect()
    }

    /// torch 2.11 (CUDA, fp32, deterministic algorithms) `F.interpolate(x, size=(8, 5),
    /// mode='bilinear', align_corners=False)` of golden(35) as [1, 1, 5, 7]: the ops check's
    /// `PINNED_RESIZE` (models/cutie/src/ops_check.rs).
    const PINNED_RESIZE: [u32; 40] = [
        0x3e31c5fb, 0x3ee0b187, 0x3f67733a, 0x3ebd1b60, 0x3f2274f2, 0x3ea1f6cd, 0x3f14e2aa, 0x3f1bfd20,
        0x3f031798, 0x3edaca84, 0x3f01faec, 0x3f42aefb, 0x3e805fac, 0x3f2db0b8, 0x3e399322, 0x3f162da2,
        0x3f36e1ae, 0x3ee8c524, 0x3f01e36d, 0x3ec52ef7, 0x3ef4c0ae, 0x3f1b1464, 0x3f289549, 0x3ecc2c42,
        0x3f06ca2c, 0x3e3a4c2a, 0x3ede8e36, 0x3f5cc7fb, 0x3eb491a8, 0x3f1afcd9, 0x3eabf1e0, 0x3f19e02f,
        0x3f00faa4, 0x3f08151e, 0x3eb1925f, 0x3ef505a6, 0x3f3e6a0e, 0x3e561200, 0x3f2c9f00, 0x3e0ee608,
    ];

    /// The ops check's `golden`: i * 0.618034 + 0.05, its fractional part.
    fn golden(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let x = i as f32 * 0.618034 + 0.05;
                x - x.floor()
            })
            .collect()
    }

    #[test]
    fn the_resize_kernels_arithmetic_gives_pytorchs_bits() {
        let p = resize_params(1, 35, 7, 5, 5, 8).unwrap();
        let got: Vec<u32> = resize_as_msl(&golden(35), &p).iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, PINNED_RESIZE);
    }
}
