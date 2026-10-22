//! The fp32 GEMM and its users on the Mac (RationalVisions docs/cutie-metal.md, step 9), in the
//! host binding: `MetalStream::gemm` on both kernels against products summed in f64 (transposes,
//! leading dimensions past the columns, offsets, batch strides and broadcast operands, k = 0,
//! alpha and beta, odd sizes, NaN and inf), the refusals before anything is encoded, and
//! `linear_f32_resident` and the fp32 attention against f64 as Cutie's ops check judges them
//! (TIGHT: rel 1e-5, cosine 0.9999999).
//!
//! Every float of an operand's tensor that the GEMM must not read is a NaN, so a read outside the
//! matrix shows as a NaN in the result; every float of an InOut C outside the matrix must come
//! back with its bits.
//!
//! `cargo test -p makepad-ai-metal --test gemm`, again with MAKEPAD_METAL_MISALIGN=1 (inputs bound
//! 4 bytes in). The kernels are chosen per test (`gemm::set_mode_override`), whatever
//! MAKEPAD_METAL_GEMM says.
#![cfg(target_os = "macos")]

use makepad_ai_metal::stream::gemm::{self, GemmMode, GEMM, GEMM_KERNELS, SIMD_WIDTH};
use makepad_ai_metal::stream::{self, with_metal_stream, Bind, Gemm, GemmOperand};
use makepad_ai_metal::{attention, linear, GpuTensor};

/// fp32's unit roundoff, 2^-24.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    fn value(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.value()).collect()
    }

    /// Roughly normal (the sum of four uniforms, scaled to unit variance).
    fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| (self.value() + self.value() + self.value() + self.value()) * (0.75f32).sqrt() * scale).collect()
    }
}

/// Sets the calling thread's GEMM kernel while it lives (libtest may run tests one after another
/// on one thread).
struct ModeGuard;

impl ModeGuard {
    fn new(mode: GemmMode) -> ModeGuard {
        gemm::set_mode_override(Some(mode));
        ModeGuard
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        gemm::set_mode_override(None);
    }
}

const MODES: [GemmMode; 2] = [GemmMode::Naive, GemmMode::Tiled];

fn upload(values: &[f32]) -> GpuTensor {
    stream::upload(values, values.len(), 1).unwrap()
}

fn download(t: &GpuTensor) -> Vec<f32> {
    stream::download(t).unwrap()
}

/// Where an operand's matrices lie in its tensor: the first at `offset`, each row `pad` floats
/// wider than its columns (ld = cols + pad), the next batch `gap` floats after the last row of
/// the one before; with `broadcast`, one matrix for every batch (batch stride 0).
#[derive(Clone, Copy, Debug)]
struct Layout {
    offset: usize,
    pad: usize,
    gap: usize,
    broadcast: bool,
}

const PACKED: Layout = Layout { offset: 0, pad: 0, gap: 0, broadcast: false };
const LOOSE: Layout = Layout { offset: 5, pad: 3, gap: 7, broadcast: false };

/// One operand on the host: its tensor's floats and how the GEMM reads them.
struct Operand {
    data: Vec<f32>,
    offset: usize,
    ld: usize,
    trans: bool,
    batch_stride: usize,
}

impl Operand {
    /// `batch` stored `rows x cols` matrices laid out as `layout`, random values, NaN everywhere
    /// else (two more after the last).
    fn new(rng: &mut Rng, rows: usize, cols: usize, batch: usize, layout: Layout, trans: bool) -> Operand {
        let ld = cols + layout.pad;
        let stride = if layout.broadcast { 0 } else { rows * ld + layout.gap };
        let matrices = if layout.broadcast { 1 } else { batch.max(1) };
        let len = layout.offset + (matrices - 1) * stride + rows * ld + 2;
        let mut data = vec![f32::NAN; len];
        for b in 0..matrices {
            for r in 0..rows {
                for c in 0..cols {
                    data[layout.offset + b * stride + r * ld + c] = rng.value();
                }
            }
        }
        Operand { data, offset: layout.offset, ld, trans, batch_stride: stride }
    }

    /// The float index of op()'s element (i, j) of batch `b`.
    fn index(&self, b: usize, i: usize, j: usize) -> usize {
        let (r, c) = if self.trans { (j, i) } else { (i, j) };
        self.offset + b * self.batch_stride + r * self.ld + c
    }

    fn get(&self, b: usize, i: usize, j: usize) -> f32 {
        self.data[self.index(b, i, j)]
    }

    fn set(&mut self, b: usize, i: usize, j: usize, value: f32) {
        let at = self.index(b, i, j);
        self.data[at] = value;
    }

    fn gemm<'a>(&self, src: Bind<'a>) -> GemmOperand<'a> {
        GemmOperand { src, offset: self.offset, ld: self.ld, trans: self.trans, batch_stride: self.batch_stride }
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    alpha: f32,
    beta: f32,
}

impl Shape {
    fn new(m: usize, n: usize, k: usize) -> Shape {
        Shape { m, n, k, batch: 1, alpha: 1.0, beta: 0.0 }
    }
}

/// The operands of one case: A stored m x k (k x m transposed), B k x n (n x k), C m x n.
struct Case {
    shape: Shape,
    a: Operand,
    b: Operand,
    c: Operand,
}

impl Case {
    fn new(rng: &mut Rng, shape: Shape, ta: bool, tb: bool, layouts: [Layout; 3]) -> Case {
        let Shape { m, n, k, batch, .. } = shape;
        let (a_rows, a_cols) = if ta { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if tb { (n, k) } else { (k, n) };
        Case {
            shape,
            a: Operand::new(rng, a_rows, a_cols, batch, layouts[0], ta),
            b: Operand::new(rng, b_rows, b_cols, batch, layouts[1], tb),
            c: Operand::new(rng, m, n, batch, layouts[2], false),
        }
    }

    /// Runs the case through `MetalStream::gemm` in an op of its own; C is an `InOut` starting
    /// from `c.data` when `inout` (always when beta is not 0), else an `Out`. C's tensor after it.
    fn run(&self, op: &'static str, inout: bool) -> Result<Vec<f32>, String> {
        let (ta, tb, tc) = (upload(&self.a.data), upload(&self.b.data), upload(&self.c.data));
        let s = self.shape;
        let out = with_metal_stream(op, |stream| {
            let c = GpuTensor::alloc(self.c.data.len(), 1)?;
            if inout {
                stream.copy(&tc, &c)?;
            }
            stream.gemm(&Gemm {
                m: s.m,
                n: s.n,
                k: s.k,
                batch: s.batch,
                alpha: s.alpha,
                beta: s.beta,
                a: self.a.gemm(Bind::In(&ta)),
                b: self.b.gemm(Bind::In(&tb)),
                c: if inout { Bind::InOut(&c) } else { Bind::Out(&c) },
                c_offset: self.c.offset,
                ldc: self.c.ld,
                c_batch_stride: self.c.batch_stride,
            })?;
            Ok(c)
        })?;
        Ok(download(&out))
    }

    /// `got` (C's tensor after the GEMM) against the product summed in f64, element by element
    /// within fp32's error bound for a k-term dot product and the epilogue's two roundings; with
    /// `inout`, every float outside the matrices still has its bits.
    fn check(&self, what: &str, got: &[f32], inout: bool) {
        let Shape { m, n, k, batch, alpha, beta } = self.shape;
        assert_eq!(got.len(), self.c.data.len(), "{what}: C's length");
        let mut written = vec![false; got.len()];
        for bt in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let at = self.c.index(bt, i, j);
                    written[at] = true;
                    let (mut sum, mut abs) = (0.0f64, 0.0f64);
                    for kk in 0..k {
                        let p = self.a.get(bt, i, kk) as f64 * self.b.get(bt, kk, j) as f64;
                        sum += p;
                        abs += p.abs();
                    }
                    let beta_c = if beta == 0.0 { 0.0 } else { beta as f64 * self.c.data[at] as f64 };
                    // With k or alpha 0 neither A nor B is read (they may hold NaN).
                    let reads = k > 0 && alpha != 0.0;
                    let want = if reads { alpha as f64 * sum + beta_c } else { beta_c };
                    let product = if reads { alpha.abs() as f64 * abs } else { 0.0 };
                    let bound = (k as f64 + 2.0) * U * (product + beta_c.abs()) * 1.01 + 1e-38;
                    let g = got[at] as f64;
                    assert!(
                        (g - want).abs() <= bound,
                        "{what}: C[{bt}][{i}][{j}] is {g:e}, f64 says {want:e} (bound {bound:e})"
                    );
                }
            }
        }
        if inout {
            if let Some(at) = (0..got.len()).find(|&at| !written[at] && got[at].to_bits() != self.c.data[at].to_bits()) {
                panic!("{what}: float {at} of C, outside the matrices, changed to {:e}", got[at]);
            }
        }
    }
}

const SIZES: [(usize, usize, usize); 14] = [
    (1, 1, 1),
    (1, 1, 7),
    (7, 5, 3),
    (13, 7, 5),
    (8, 8, 8),
    (31, 33, 17),
    (32, 32, 32),
    (33, 31, 47),
    (64, 64, 64),
    (65, 97, 129),
    (1, 200, 300),
    (200, 1, 30),
    (128, 256, 256),
    (257, 63, 513),
];

#[test]
fn every_gemm_pipeline_compiles() {
    assert!(stream::library_available(&GEMM), "gemm.metal does not compile");
    with_metal_stream("gemm test: pipelines", |s| {
        for kernel in GEMM_KERNELS {
            let limits = s.limits(kernel, &[])?;
            // The tiled GEMM's threadgroup is 128 threads; the softmax wants 256 a row (1024 from
            // 1024 keys), and takes fewer if it must.
            let wanted = if kernel.function == "gemm_tiled_f32" { 128 } else { 256 };
            assert_eq!(limits.simd_width, SIMD_WIDTH, "{}", kernel.function);
            assert!(limits.max_threads >= wanted, "{}: {limits:?}", kernel.function);
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn both_kernels_match_f64_over_transposes_and_odd_sizes() {
    let mut rng = Rng(0x6e44_0001);
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        for (m, n, k) in SIZES {
            for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                let case = Case::new(&mut rng, Shape::new(m, n, k), ta, tb, [PACKED; 3]);
                let got = case.run("gemm test: sizes", false).unwrap();
                case.check(&format!("{mode:?} {m}x{n}x{k} ta {ta} tb {tb}"), &got, false);
            }
        }
    }
}

#[test]
fn leading_dimensions_offsets_and_batch_strides() {
    let mut rng = Rng(0x6e44_0002);
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        for (m, n, k) in [(7, 5, 3), (33, 31, 47), (65, 97, 129)] {
            for (ta, tb) in [(false, false), (true, true), (false, true)] {
                for batch in [1, 3] {
                    let shape = Shape { batch, ..Shape::new(m, n, k) };
                    let case = Case::new(&mut rng, shape, ta, tb, [LOOSE; 3]);
                    let got = case.run("gemm test: layouts", false).unwrap();
                    case.check(&format!("{mode:?} loose {m}x{n}x{k} ta {ta} tb {tb} batch {batch}"), &got, false);
                }
            }
        }
        // One operand for every batch, as a convolution's weights (batch stride 0).
        let broadcast = Layout { broadcast: true, ..LOOSE };
        let shape = Shape { batch: 4, ..Shape::new(40, 36, 27) };
        let case = Case::new(&mut rng, shape, false, false, [broadcast, LOOSE, LOOSE]);
        let got = case.run("gemm test: broadcast", false).unwrap();
        case.check(&format!("{mode:?} broadcast A"), &got, false);
        let case = Case::new(&mut rng, shape, true, false, [LOOSE, broadcast, LOOSE]);
        let got = case.run("gemm test: broadcast", false).unwrap();
        case.check(&format!("{mode:?} broadcast B"), &got, false);
    }
}

#[test]
fn packed_heads_are_read_and_written_in_place() {
    // Attention's layout: Q and K [len, heads * head_dim], each head's columns head_dim apart
    // (batch stride head_dim, ld the hidden width), the output's heads likewise.
    let mut rng = Rng(0x6e44_0003);
    let (q_len, kv_len, heads, head_dim) = (19, 45, 4, 12);
    let hidden = heads * head_dim;
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        let q = Operand { data: rng.values(q_len * hidden), offset: 0, ld: hidden, trans: false, batch_stride: head_dim };
        let k = Operand { data: rng.values(kv_len * hidden), offset: 0, ld: hidden, trans: true, batch_stride: head_dim };
        let out = Operand { data: vec![f32::NAN; q_len * kv_len * heads], offset: 0, ld: kv_len, trans: false, batch_stride: q_len * kv_len };
        let case = Case { shape: Shape { batch: heads, alpha: 0.25, ..Shape::new(q_len, kv_len, head_dim) }, a: q, b: k, c: out };
        let got = case.run("gemm test: heads", false).unwrap();
        case.check(&format!("{mode:?} scores"), &got, false);
        // P V back into the heads' columns of a [q_len, hidden] output.
        let p = Operand { data: rng.values(heads * q_len * kv_len), offset: 0, ld: kv_len, trans: false, batch_stride: q_len * kv_len };
        let v = Operand { data: rng.values(kv_len * hidden), offset: 0, ld: hidden, trans: false, batch_stride: head_dim };
        let out = Operand { data: vec![f32::NAN; q_len * hidden], offset: 0, ld: hidden, trans: false, batch_stride: head_dim };
        let case = Case { shape: Shape { batch: heads, ..Shape::new(q_len, head_dim, kv_len) }, a: p, b: v, c: out };
        let got = case.run("gemm test: heads", false).unwrap();
        case.check(&format!("{mode:?} output"), &got, false);
        assert!(got.iter().all(|v| v.is_finite()), "{mode:?}: every column of the output is some head's");
    }
}

#[test]
fn alpha_beta_and_k_zero() {
    let mut rng = Rng(0x6e44_0004);
    for mode in [GemmMode::Naive, GemmMode::Tiled, GemmMode::Auto] {
        let _mode = ModeGuard::new(mode);
        for (m, n, k) in [(7, 5, 3), (40, 33, 65)] {
            // alpha and beta, C an InOut whose padding must come back unchanged.
            let shape = Shape { alpha: 0.37, beta: -1.25, batch: 2, ..Shape::new(m, n, k) };
            let case = Case::new(&mut rng, shape, false, true, [PACKED, LOOSE, LOOSE]);
            let got = case.run("gemm test: alpha beta", true).unwrap();
            case.check(&format!("{mode:?} alpha beta {m}x{n}x{k}"), &got, true);
            // k = 0: beta C, and 0 with beta 0, on an Out holding alloc's NaNs.
            for beta in [0.0, 1.5] {
                let shape = Shape { beta, ..Shape::new(m, n, 0) };
                let case = Case::new(&mut rng, shape, false, false, [PACKED, PACKED, LOOSE]);
                let got = case.run("gemm test: k = 0", beta != 0.0).unwrap();
                case.check(&format!("{mode:?} k = 0 beta {beta}"), &got, beta != 0.0);
            }
            // beta 0 never reads C: an InOut of NaNs comes out finite.
            let mut case = Case::new(&mut rng, Shape::new(m, n, k), false, false, [PACKED; 3]);
            case.c.data.iter_mut().for_each(|v| *v = f32::NAN);
            let got = case.run("gemm test: beta 0", true).unwrap();
            case.check(&format!("{mode:?} beta 0 over NaN"), &got, true);
            // alpha 0 reads neither A nor B: beta C, even with NaN in A.
            let shape = Shape { alpha: 0.0, beta: 2.0, ..Shape::new(m, n, k) };
            let mut case = Case::new(&mut rng, shape, false, false, [PACKED; 3]);
            case.a.data.iter_mut().for_each(|v| *v = f32::NAN);
            let got = case.run("gemm test: alpha 0", true).unwrap();
            case.check(&format!("{mode:?} alpha 0"), &got, true);
        }
    }
}

#[test]
fn nan_and_inf_propagate_as_in_f64() {
    let mut rng = Rng(0x6e44_0005);
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        let (m, n, k) = (37, 35, 41);
        let mut case = Case::new(&mut rng, Shape::new(m, n, k), false, false, [PACKED; 3]);
        case.a.set(0, 1, 2, f32::INFINITY);
        case.a.set(0, 3, 0, f32::NAN);
        case.b.set(0, 2, 4, 0.0); // inf * 0 at C[1][4]
        case.b.set(0, 40, 34, f32::NEG_INFINITY); // the last k and column
        let got = case.run("gemm test: specials", false).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|kk| case.a.get(0, i, kk) as f64 * case.b.get(0, kk, j) as f64).sum();
                let g = got[case.c.index(0, i, j)];
                if want.is_nan() || want.is_infinite() {
                    assert!(
                        (want.is_nan() && g.is_nan()) || g as f64 == want,
                        "{mode:?}: C[{i}][{j}] is {g}, f64 says {want}"
                    );
                } else {
                    assert!(g.is_finite(), "{mode:?}: C[{i}][{j}] is {g}, f64 says {want}");
                }
            }
        }
        assert!(got[case.c.index(0, 1, 4)].is_nan(), "{mode:?}: inf * 0");
        assert!(got[case.c.index(0, 3, 9)].is_nan(), "{mode:?}: a NaN in A's row");
    }
}

#[test]
fn weights_operands_and_an_output_written_in_parts() {
    // A convolution's shape: A the weights (out_channels x kdim), bound 3 floats into the cached
    // buffer and read from 2 floats past that, the same for every batch; B batched; C written in
    // two GEMMs, the second at an offset (as attention writes its chunks of queries).
    let mut rng = Rng(0x6e44_0006);
    let (oc, kdim, opix, batch) = (24, 27, 50, 3);
    let weights: Vec<f32> = rng.values(5 + oc * kdim);
    let cols = rng.values(batch * kdim * opix);
    let tcols = upload(&cols);
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        let out = with_metal_stream("gemm test: weights", |s| {
            let w = s.weights("gemm-test::conv::cutie_conv", weights.len(), |dst| {
                dst.copy_from_slice(&weights);
                Ok(())
            })?;
            let out = GpuTensor::alloc(batch * oc, opix)?;
            for (first, count) in [(0, 2), (2, 1)] {
                s.gemm(&Gemm {
                    m: oc,
                    n: opix,
                    k: kdim,
                    batch: count,
                    alpha: 1.0,
                    beta: 0.0,
                    a: GemmOperand { src: Bind::Weights(&w, 3), offset: 2, ld: kdim, trans: false, batch_stride: 0 },
                    b: GemmOperand { src: Bind::In(&tcols), offset: first * kdim * opix, ld: opix, trans: false, batch_stride: kdim * opix },
                    c: Bind::Out(&out),
                    c_offset: first * oc * opix,
                    ldc: opix,
                    c_batch_stride: oc * opix,
                })?;
            }
            Ok(out)
        })
        .unwrap();
        let got = download(&out);
        for b in 0..batch {
            for o in 0..oc {
                for p in 0..opix {
                    let (mut sum, mut abs) = (0.0f64, 0.0f64);
                    for kk in 0..kdim {
                        let v = weights[5 + o * kdim + kk] as f64 * cols[b * kdim * opix + kk * opix + p] as f64;
                        sum += v;
                        abs += v.abs();
                    }
                    let g = got[(b * oc + o) * opix + p] as f64;
                    let bound = (kdim as f64 + 2.0) * U * abs * 1.01;
                    assert!((g - sum).abs() <= bound, "{mode:?}: image {b} channel {o} pixel {p}: {g:e}, f64 {sum:e}");
                }
            }
        }
    }
    assert_eq!(stream::evict_weights("gemm-test::", false).unwrap(), 1);
}

/// One thing wrong with a GEMM, for the refusals.
#[derive(Clone, Copy, Debug)]
enum Broken {
    Nothing,
    BetaOnAnOut,
    AnOutAsA,
    AnInAsC,
    NoC,
    ShortLd,
    PastB,
    PastC,
    BatchesPastA,
}

#[test]
fn a_gemm_is_refused_before_anything_is_encoded() {
    let x = stream::upload(&[1.0; 12], 3, 4).unwrap();
    // A 3x4 @ (3x4)^T into 3x3 of 4s, with `broken` broken.
    let run = |op: &'static str, broken: Broken| {
        with_metal_stream(op, |s| {
            let c = GpuTensor::alloc(3, 3)?;
            let mut g = Gemm {
                m: 3,
                n: 3,
                k: 4,
                batch: 1,
                alpha: 1.0,
                beta: 0.0,
                a: GemmOperand { src: Bind::In(&x), offset: 0, ld: 4, trans: false, batch_stride: 0 },
                b: GemmOperand { src: Bind::In(&x), offset: 0, ld: 4, trans: true, batch_stride: 0 },
                c: Bind::Out(&c),
                c_offset: 0,
                ldc: 3,
                c_batch_stride: 0,
            };
            match broken {
                Broken::Nothing => {}
                Broken::BetaOnAnOut => g.beta = 1.0,
                Broken::AnOutAsA => g.a.src = Bind::Out(&c),
                Broken::AnInAsC => g.c = Bind::In(&x),
                Broken::NoC => g.c = Bind::Absent,
                Broken::ShortLd => g.a.ld = 3,
                Broken::PastB => g.b.offset = 1,
                Broken::PastC => g.c_offset = 1,
                Broken::BatchesPastA => {
                    g.batch = 2;
                    g.a.batch_stride = 1;
                }
            }
            s.gemm(&g)?;
            Ok(c)
        })
    };
    assert_eq!(download(&run("gemm test: valid", Broken::Nothing).unwrap()), vec![4.0; 9]);
    let refusals = [
        ("gemm test: beta on an Out", Broken::BetaOnAnOut, "with beta 1 reads C: bind it Bind::InOut"),
        ("gemm test: A an Out", Broken::AnOutAsA, "reads A through Bind::In or Bind::Weights"),
        ("gemm test: C an In", Broken::AnInAsC, "writes C through Bind::Out or Bind::InOut"),
        ("gemm test: no C", Broken::NoC, "writes C through Bind::Out or Bind::InOut"),
        ("gemm test: short ld", Broken::ShortLd, "A's leading dimension 3 is less than its 4 columns"),
        ("gemm test: past B", Broken::PastB, "reads float 12 of B"),
        ("gemm test: past C", Broken::PastC, "writes float 9 of C"),
        ("gemm test: batches past A", Broken::BatchesPastA, "reads float 12 of A"),
    ];
    for (op, broken, expect) in refusals {
        let err = run(op, broken).err().unwrap_or_else(|| panic!("{op}: no error"));
        assert!(err.starts_with(op) && err.contains(expect), "{broken:?}: {err}");
    }
    // Refused before anything was encoded: nothing pending, and the stream goes on.
    stream::synchronize().unwrap();
    assert_eq!(download(&run("gemm test: after", Broken::Nothing).unwrap()), vec![4.0; 9]);
}

/// Asserts `got` is `want` (computed in f64) as Cutie's ops check judges TIGHT: the largest
/// difference at most 1e-5 of the largest reference and cosine at least 0.9999999; non-finite
/// values only as the same infinity or both NaN.
fn assert_tight(what: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let (mut max_abs, mut max_ref, mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = g as f64;
        if !g.is_finite() || !w.is_finite() {
            assert!((g.is_nan() && w.is_nan()) || g == w, "{what}: element {i} is {g}, f64 says {w}");
            continue;
        }
        max_abs = max_abs.max((g - w).abs());
        max_ref = max_ref.max(w.abs());
        dot += g * w;
        ng += g * g;
        nw += w * w;
    }
    let rel = max_abs / max_ref.max(1e-12);
    let cos = if ng == 0.0 && nw == 0.0 { 1.0 } else { dot / (ng.sqrt() * nw.sqrt()).max(1e-300) };
    let zeros = max_ref == 0.0 && max_abs < 1e-6;
    assert!(zeros || (rel <= 1e-5 && cos >= 0.9999999), "{what}: rel {rel:e}, cos {cos} (max abs {max_abs:e})");
}

/// x @ w^T + b in f64.
fn linear_f64(x: &[f32], w: &[f32], b: Option<&[f32]>, rows: usize, in_f: usize, out_f: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * out_f);
    for r in 0..rows {
        for o in 0..out_f {
            let dot: f64 = (0..in_f).map(|i| x[r * in_f + i] as f64 * w[o * in_f + i] as f64).sum();
            out.push(dot + b.map_or(0.0, |b| b[o] as f64));
        }
    }
    out
}

#[test]
fn linear_matches_f64_as_the_ops_check_judges_it() {
    let mut rng = Rng(0x6e44_0007);
    for mode in [GemmMode::Naive, GemmMode::Tiled, GemmMode::Auto] {
        let _mode = ModeGuard::new(mode);
        // Cutie's object transformer (256 wide, 2048 in its FFN), Body-like odd sizes, and edges.
        for (rows, in_f, out_f, bias) in
            [(1, 1, 1, true), (5, 7, 3, true), (336, 256, 256, true), (16, 256, 2048, false), (100, 3, 64, true), (33, 129, 65, false)]
        {
            let x = rng.normals(rows * in_f, 1.0);
            let w = rng.normals(out_f * in_f, 1.0 / (in_f as f32).sqrt());
            let b = rng.normals(out_f, 0.1);
            let bias_t = bias.then(|| stream::upload(&b, 1, out_f).unwrap());
            let got = linear::linear_f32_resident(
                &stream::upload(&x, rows, in_f).unwrap(),
                &stream::upload(&w, out_f, in_f).unwrap(),
                bias_t.as_ref(),
            )
            .unwrap();
            assert_eq!((got.rows(), got.cols()), (rows, out_f));
            let want = linear_f64(&x, &w, bias.then_some(&b[..]), rows, in_f, out_f);
            assert_tight(&format!("{mode:?} linear {rows}x{in_f} -> {out_f} bias {bias}"), &download(&got), &want);
        }
    }
}

#[test]
fn linears_bias_is_one_add_after_the_product() {
    // As CUDA's: the product rounded, then the bias added in a pass of its own. So the linear
    // with a bias is, bit for bit, the linear without one plus the bias in f32; a [n, 1] bias
    // reads as a [1, n] one.
    let mut rng = Rng(0x6e44_0008);
    let (rows, in_f, out_f) = (37, 70, 45);
    let (x, w, b) = (rng.normals(rows * in_f, 1.0), rng.normals(out_f * in_f, 0.1), rng.normals(out_f, 1.0));
    let (tx, tw) = (stream::upload(&x, rows, in_f).unwrap(), stream::upload(&w, out_f, in_f).unwrap());
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        let plain = download(&linear::linear_f32_resident(&tx, &tw, None).unwrap());
        for (r, c) in [(1, out_f), (out_f, 1)] {
            let biased = download(&linear::linear_f32_resident(&tx, &tw, Some(&stream::upload(&b, r, c).unwrap())).unwrap());
            for (i, (&got, &p)) in biased.iter().zip(&plain).enumerate() {
                assert_eq!(got.to_bits(), (p + b[i % out_f]).to_bits(), "{mode:?} element {i}");
            }
        }
        // And the same call twice gives the same bits (the weight's device copy is reused).
        assert_eq!(download(&linear::linear_f32_resident(&tx, &tw, None).unwrap()), plain);
    }
}

#[test]
fn linear_edges_and_cudas_refusals() {
    // k = 0: the bias alone; no rows: an empty output.
    let x = stream::upload(&[], 4, 0).unwrap();
    let w = stream::upload(&[], 3, 0).unwrap();
    let b = stream::upload(&[1.0, -2.0, 0.5], 1, 3).unwrap();
    assert_eq!(download(&linear::linear_f32_resident(&x, &w, Some(&b)).unwrap()), [1.0, -2.0, 0.5].repeat(4));
    assert_eq!(download(&linear::linear_f32_resident(&x, &w, None).unwrap()), vec![0.0; 12]);
    let empty = linear::linear_f32_resident(&stream::upload(&[], 0, 5).unwrap(), &stream::upload(&[0.0; 10], 2, 5).unwrap(), None).unwrap();
    assert_eq!((empty.rows(), empty.cols()), (0, 2));
    // CUDA's messages.
    let x = stream::upload(&[1.0; 6], 2, 3).unwrap();
    let w = stream::upload(&[1.0; 8], 2, 4).unwrap();
    let err = linear::linear_f32_resident(&x, &w, None).err().unwrap();
    assert_eq!(err, "gpu_linear_f32_resident k mismatch: x 2x3, w 2x4");
    let w = stream::upload(&[1.0; 6], 2, 3).unwrap();
    let err = linear::linear_f32_resident(&x, &w, Some(&stream::upload(&[0.0; 3], 1, 3).unwrap())).err().unwrap();
    assert_eq!(err, "gpu_linear_f32_resident bias mismatch");
}

/// softmax(scale q k^T + bias) v per head in f64 (the ops check's reference), heads packed along
/// the columns, the bias [heads * q_len, kv_len].
#[allow(clippy::too_many_arguments)]
fn attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<&[f32]>,
    heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f64> {
    let hidden = heads * head_dim;
    let mut out = vec![0.0f64; q_len * hidden];
    for h in 0..heads {
        for i in 0..q_len {
            let scores: Vec<f64> = (0..kv_len)
                .map(|j| {
                    let dot: f64 = (0..head_dim).map(|d| q[i * hidden + h * head_dim + d] as f64 * k[j * hidden + h * head_dim + d] as f64).sum();
                    scale as f64 * dot + bias.map_or(0.0, |b| b[(h * q_len + i) * kv_len + j] as f64)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..head_dim {
                out[i * hidden + h * head_dim + d] = (0..kv_len).map(|j| exps[j] / sum * v[j * hidden + h * head_dim + d] as f64).sum();
            }
        }
    }
    out
}

/// Q, K, V for `heads` heads of 32: normal values, as the ops check makes them.
fn qkv(rng: &mut Rng, heads: usize, q_len: usize, kv_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hidden = heads * 32;
    (rng.normals(q_len * hidden, 1.0), rng.normals(kv_len * hidden, 1.0), rng.normals(kv_len * hidden, 1.0))
}

#[test]
fn attention_matches_f64_as_the_ops_check_judges_it() {
    const HEAD_DIM: usize = 32;
    let mut rng = Rng(0x6e44_0009);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for mode in [GemmMode::Naive, GemmMode::Tiled, GemmMode::Auto] {
        let _mode = ModeGuard::new(mode);
        // Cutie's object transformer (8 heads, 16 queries, 336 pixels and back), one key, one
        // query, and past 1024 keys (the softmax's 1024 threads).
        for (heads, q_len, kv_len, masked) in
            [(8, 16, 336, false), (8, 16, 336, true), (8, 336, 16, true), (1, 33, 1, false), (2, 1, 1000, true), (4, 7, 1500, false)]
        {
            let (q, k, v) = qkv(&mut rng, heads, q_len, kv_len);
            let hidden = heads * HEAD_DIM;
            // Every 7th entry blocked, never a row's first: no row is blocked everywhere.
            let mask: Vec<f32> = (0..heads * q_len * kv_len)
                .map(|i| if i % 7 == 3 && i % kv_len != 0 { f32::NEG_INFINITY } else { 0.0 })
                .collect();
            let (tq, tk, tv) =
                (stream::upload(&q, q_len, hidden).unwrap(), stream::upload(&k, kv_len, hidden).unwrap(), stream::upload(&v, kv_len, hidden).unwrap());
            let got = if masked {
                attention::attention_packed_cross_bias(&tq, &tk, &tv, heads, scale, &stream::upload(&mask, heads * q_len, kv_len).unwrap())
            } else {
                attention::attention_packed_cross_f32(&tq, &tk, &tv, heads, scale)
            }
            .unwrap();
            assert_eq!((got.rows(), got.cols()), (q_len, hidden));
            let want = attention_f64(&q, &k, &v, masked.then_some(&mask[..]), heads, q_len, kv_len, HEAD_DIM, scale);
            assert_tight(&format!("{mode:?} attention heads {heads} q {q_len} kv {kv_len} masked {masked}"), &download(&got), &want);
        }
        // SAM 3.1's kind of bias: any values, not only a mask.
        let (heads, q_len, kv_len) = (4, 21, 77);
        let (q, k, v) = qkv(&mut rng, heads, q_len, kv_len);
        let bias = rng.normals(heads * q_len * kv_len, 2.0);
        let hidden = heads * HEAD_DIM;
        let got = attention::attention_packed_cross_bias(
            &stream::upload(&q, q_len, hidden).unwrap(),
            &stream::upload(&k, kv_len, hidden).unwrap(),
            &stream::upload(&v, kv_len, hidden).unwrap(),
            heads,
            scale,
            &stream::upload(&bias, heads * q_len, kv_len).unwrap(),
        )
        .unwrap();
        let want = attention_f64(&q, &k, &v, Some(&bias), heads, q_len, kv_len, HEAD_DIM, scale);
        assert_tight(&format!("{mode:?} attention with a bias"), &download(&got), &want);
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn attention_chunks_and_a_zero_bias_change_no_bit() {
    // The queries in chunks (CUDA's bound on the scores, made small) give what one pass gives:
    // on the naive kernel the same bits, each output being the same fmas in the same order; on
    // the tiled one a query lands at another row of its tile, so only within TIGHT unless the
    // SIMD-group product rounds alike at every position (printed). And a zero bias adds nothing
    // (x + 0 is x), so the biased attention is the plain one, bit for bit, on either kernel.
    let mut rng = Rng(0x6e44_000a);
    let (heads, q_len, kv_len) = (3, 10, 41);
    let hidden = heads * 32;
    let (q, k, v) = qkv(&mut rng, heads, q_len, kv_len);
    let (tq, tk, tv) =
        (stream::upload(&q, q_len, hidden).unwrap(), stream::upload(&k, kv_len, hidden).unwrap(), stream::upload(&v, kv_len, hidden).unwrap());
    let zero = stream::upload(&vec![0.0; heads * q_len * kv_len], heads * q_len, kv_len).unwrap();
    for mode in MODES {
        let _mode = ModeGuard::new(mode);
        let whole = download(&attention::attention_packed_cross_f32(&tq, &tk, &tv, heads, 0.2).unwrap());
        let whole_f64: Vec<f64> = whole.iter().map(|&v| v as f64).collect();
        for cap in [heads * kv_len * 3, heads * kv_len, 1] {
            let chunked = download(&attention::attention_packed_cross_f32_capped(&tq, &tk, &tv, heads, 0.2, cap).unwrap());
            let what = format!("{mode:?}: chunks of {} queries", (cap / (heads * kv_len)).max(1));
            if mode == GemmMode::Naive {
                assert_eq!(bits(&chunked), bits(&whole), "{what}");
            } else {
                assert_tight(&what, &chunked, &whole_f64);
                eprintln!("{what}: the same bits as one pass: {}", bits(&chunked) == bits(&whole));
            }
        }
        let biased = download(&attention::attention_packed_cross_bias(&tq, &tk, &tv, heads, 0.2, &zero).unwrap());
        assert_eq!(bits(&biased), bits(&whole), "{mode:?}: a zero bias");
    }
}

#[test]
fn attention_edges_and_cudas_refusals() {
    let _mode = ModeGuard::new(GemmMode::Auto);
    let mut rng = Rng(0x6e44_000b);
    let (heads, q_len, kv_len) = (2, 3, 5);
    let hidden = heads * 32;
    let (q, k, v) = qkv(&mut rng, heads, q_len, kv_len);
    let (tq, tk, tv) =
        (stream::upload(&q, q_len, hidden).unwrap(), stream::upload(&k, kv_len, hidden).unwrap(), stream::upload(&v, kv_len, hidden).unwrap());
    // A row blocked everywhere comes out NaN, as CUDA's (exp(-inf - -inf)); the others do not.
    let mut mask = vec![0.0f32; heads * q_len * kv_len];
    mask[kv_len..2 * kv_len].iter_mut().for_each(|m| *m = f32::NEG_INFINITY); // head 0, query 1
    let got = download(&attention::attention_packed_cross_bias(&tq, &tk, &tv, heads, 0.2, &stream::upload(&mask, heads * q_len, kv_len).unwrap()).unwrap());
    for i in 0..q_len {
        for c in 0..hidden {
            let blocked = i == 1 && c < 32;
            assert_eq!(got[i * hidden + c].is_nan(), blocked, "query {i} column {c}");
        }
    }
    // No keys: zeros (the output GEMM's k is 0). No queries: an empty output.
    let none = stream::upload(&[], 0, hidden).unwrap();
    let out = attention::attention_packed_cross_f32(&tq, &none, &none, heads, 0.2).unwrap();
    assert_eq!(download(&out), vec![0.0; q_len * hidden]);
    let out = attention::attention_packed_cross_f32(&none, &tk, &tv, heads, 0.2).unwrap();
    assert_eq!((out.rows(), out.cols()), (0, hidden));
    // CUDA's messages.
    let narrow = stream::upload(&[0.0; 5 * 32], 5, 32).unwrap();
    assert_eq!(attention::attention_packed_cross_f32(&tq, &narrow, &tv, heads, 0.2).err().unwrap(), "gpu_attention_cross shape mismatch");
    assert_eq!(attention::attention_packed_cross_f32(&tq, &tk, &tv, 3, 0.2).err().unwrap(), "gpu_attention_cross head mismatch");
    assert_eq!(attention::attention_packed_cross_f32(&tq, &tk, &tv, 0, 0.2).err().unwrap(), "gpu_attention_cross head mismatch");
    let short = stream::upload(&[0.0; 4], 2, 2).unwrap();
    assert_eq!(
        attention::attention_packed_cross_bias(&tq, &tk, &tv, heads, 0.2, &short).err().unwrap(),
        "gpu_attention_cross_bias bias length mismatch"
    );
    assert_eq!(
        attention::attention_packed_cross_bias(&tq, &tk, &narrow, heads, 0.2, &short).err().unwrap(),
        "gpu_attention_cross_bias shape mismatch"
    );
}
