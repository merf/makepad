//! Stage `ops`: each Cutie device op (cuda/src/cutie_ops.rs) against its host reference, on
//! random inputs at Cutie's shapes (several objects, odd sizes, the padded oracle sizes) and the
//! edge cases each op has (k > rows for top-k, ties, a mask row blocked everywhere, padding at
//! every border for max-pool and the strided convs).
//!
//! The shapes are the oracle's: frames 384x216 padded to 384x224, so strides 2, 4, 8 and 16 are
//! 192x112, 96x56, 48x28 and 24x14 (336 tokens); 1 to 3 objects; memory of 336 to 3500 tokens.
//! Ops that only move or compare values must match exactly; ops that sum or call exp, tanh or
//! log must be within fp32 rounding (`TIGHT`); top-k must also choose the same set (the kernel
//! breaks ties as the reference does: the lower row). Each op at a real shape is timed against
//! its reference too (a note per case; indicative only, the GPU is shared).

use crate::model::{dev, CUTIE_NAMESPACE};
use crate::oracle::{OracleTap, Report, Tol};
use crate::Result;
use makepad_ai_common::gpu::{
    gpu_cutie_aggregate, gpu_cutie_aggregate_reference, gpu_cutie_attn_mask, gpu_cutie_attn_mask_reference,
    gpu_cutie_conv2d, gpu_cutie_conv2d_reference, gpu_cutie_crop, gpu_cutie_crop_reference, gpu_cutie_eca,
    gpu_cutie_eca_reference, gpu_cutie_gru, gpu_cutie_gru_reference, gpu_cutie_matmul, gpu_cutie_matmul_reference,
    gpu_cutie_maxpool3x3s2, gpu_cutie_maxpool3x3s2_reference, gpu_cutie_others, gpu_cutie_others_reference, gpu_cutie_pad,
    gpu_cutie_pad_reference, gpu_cutie_sigmoid, gpu_cutie_sigmoid_reference, gpu_cutie_similarity,
    gpu_cutie_similarity_reference, gpu_cutie_softmax_axis0, gpu_cutie_softmax_axis0_reference, gpu_cutie_topk_softmax,
    gpu_cutie_topk_softmax_reference, gpu_download, gpu_synchronize, gpu_upload, gpu_weight_cache_evict_prefix, GpuTensor,
};
use std::time::Instant;

const STAGE: &str = "ops";

/// Ops that only move, pick or compare values: bit for bit.
const EXACT: Tol = Tol { rel: 0.0, cos: 1.0 - 1e-12, bool_mismatch: 0.0 };
/// Ops that sum, or call exp, tanh or log: fp32 rounding apart.
const TIGHT: Tol = Tol { rel: 1e-5, cos: 0.9999999, bool_mismatch: 0.0 };

/// The conv cases' weight cache keys start with this (under Cutie's namespace), each case its
/// own key: random weights under a reused key would read a stale cached copy. Evicted after.
const CONV_KEY_PREFIX: &str = "ops-check";

/// Deterministic inputs (xorshift64*), so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, 1), 24 bits.
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn uniforms(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| lo + (hi - lo) * self.uniform()).collect()
    }

    /// Normal (Box-Muller) times `scale`.
    fn normals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let u = (self.uniform() as f64).max(1e-12);
                let v = self.uniform() as f64;
                ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32 * scale
            })
            .collect()
    }
}

/// How long `f` takes on the device: after a warm run, the best of `rounds` rounds of 3 runs
/// each (their mean, synchronised), since the GPU is shared and other work only adds time.
fn time_ms<T>(rounds: usize, mut f: impl FnMut() -> std::result::Result<T, String>) -> Result<f64> {
    const RUNS: usize = 3;
    f().map_err(|e| dev("ops timing", e))?;
    gpu_synchronize().map_err(|e| dev("ops timing", e))?;
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..RUNS {
            f().map_err(|e| dev("ops timing", e))?;
        }
        gpu_synchronize().map_err(|e| dev("ops timing", e))?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3 / RUNS as f64);
    }
    Ok(best)
}

/// A case's inputs: uploaded tensors, every op checked through the same few comparisons.
struct OpsCheck<'a> {
    report: &'a mut Report,
    rng: Rng,
    conv_cases: usize,
}

fn up(values: &[f32], rows: usize, cols: usize) -> Result<GpuTensor> {
    gpu_upload(values, rows, cols).map_err(|e| dev("ops upload", e))
}

fn down(t: &GpuTensor) -> Result<Vec<f32>> {
    gpu_download(t).map_err(|e| dev("ops download", e))
}

impl OpsCheck<'_> {
    /// The kernel's result against the reference's, as an oracle tap would be.
    fn compare(&mut self, op: &str, case: &str, actual: &GpuTensor, expected: &GpuTensor, tol: Tol) -> Result<()> {
        let tap = OracleTap { shape: vec![expected.rows(), expected.cols()], dtype: "f32".into(), values: down(expected)? };
        self.report.compare(STAGE, op, case, &down(actual)?, &tap, tol);
        Ok(())
    }

    /// Where two results are non-zero (an additive mask's blocked entries, top-k's chosen set),
    /// compared exactly.
    fn compare_nonzero(&mut self, op: &str, case: &str, actual: &GpuTensor, expected: &GpuTensor) -> Result<()> {
        let tap = OracleTap { shape: vec![expected.rows(), expected.cols()], dtype: "bool".into(), values: down(expected)? };
        self.report.compare(STAGE, op, case, &down(actual)?, &tap, EXACT);
        Ok(())
    }

    fn timing(&mut self, op: &str, case: &str, kernel_ms: f64, reference_ms: f64) {
        self.report.note(format!(
            "ops time {op:<10} {case:<44} kernel {kernel_ms:>9.3} ms  reference {reference_ms:>10.3} ms  ({:.0}x)",
            reference_ms / kernel_ms.max(1e-9)
        ));
    }

    // --- Convolution ---

    /// A conv of `batch` images with fresh random weights (scaled so outputs stay near 1).
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, what: &str, batch: usize, in_c: usize, w: usize, h: usize, out_c: usize, k: usize, pad: usize, stride: usize, bias: bool, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(batch * in_c * w * h, 1.0), batch * in_c, w * h)?;
        let weights = self.rng.normals(out_c * in_c * k * k, 1.0 / ((in_c * k * k) as f32).sqrt());
        let b = bias.then(|| self.rng.normals(out_c, 0.1));
        self.conv_cases += 1;
        let key = format!("{CONV_KEY_PREFIX}:{}:{what}", self.conv_cases);
        let case = format!("{what} {k}x{k}s{stride}p{pad} {in_c}->{out_c} {w}x{h} n{batch}{}", if bias { "" } else { " nobias" });
        let run = |reference: bool| {
            let f = if reference { gpu_cutie_conv2d_reference } else { gpu_cutie_conv2d };
            f(&x, batch, w, h, CUTIE_NAMESPACE, &key, &weights, b.as_deref(), out_c, k, pad, stride)
        };
        let actual = run(false).map_err(|e| dev(&case, e))?;
        let expected = run(true).map_err(|e| dev(&case, e))?;
        self.compare("conv2d", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || run(false))?;
            let reference = time_ms(2, || run(true))?;
            self.timing("conv2d", &case, kernel, reference);
        }
        Ok(())
    }

    fn convs(&mut self) -> Result<()> {
        // The pixel encoder (ResNet-50 to layer3), one image: the stem, each stage's 1x1s and
        // 3x3s (the stride on the first 3x3), the strided 1x1 downsamples, the projections.
        self.conv("stem", 1, 3, 384, 224, 64, 7, 3, 2, true, true)?;
        self.conv("res2", 1, 64, 96, 56, 64, 1, 0, 1, true, true)?;
        self.conv("res2", 1, 64, 96, 56, 64, 3, 1, 1, true, true)?;
        self.conv("res2", 1, 64, 96, 56, 256, 1, 0, 1, true, true)?;
        self.conv("layer2", 1, 128, 96, 56, 128, 3, 1, 2, true, true)?;
        self.conv("layer2 ds", 1, 256, 96, 56, 512, 1, 0, 2, true, true)?;
        self.conv("layer2", 1, 128, 48, 28, 128, 3, 1, 1, true, true)?;
        self.conv("layer3", 1, 256, 48, 28, 256, 3, 1, 2, true, true)?;
        self.conv("layer3 ds", 1, 512, 48, 28, 1024, 1, 0, 2, true, true)?;
        self.conv("layer3", 1, 256, 24, 14, 1024, 1, 0, 1, true, true)?;
        self.conv("pix_feat", 1, 1024, 24, 14, 256, 1, 0, 1, true, true)?;
        self.conv("key", 1, 256, 24, 14, 64, 3, 1, 1, true, true)?;
        self.conv("d_proj", 1, 256, 24, 14, 1, 3, 1, 1, true, false)?;
        // The mask encoder (ResNet-18 on 5 channels), per object.
        self.conv("mask stem", 3, 5, 384, 224, 64, 7, 3, 2, true, true)?;
        self.conv("mask stem", 2, 5, 384, 224, 64, 7, 3, 2, true, false)?;
        self.conv("mask l1", 3, 64, 96, 56, 64, 3, 1, 1, true, true)?;
        self.conv("mask l2", 2, 64, 96, 56, 128, 3, 1, 2, true, false)?;
        self.conv("mask l2 ds", 2, 64, 96, 56, 128, 1, 0, 2, true, false)?;
        self.conv("mask l3", 3, 128, 48, 28, 256, 3, 1, 2, true, true)?;
        // Fusion blocks, the sensory update, pixel fusion, the decoder.
        self.conv("ca block", 3, 256, 24, 14, 256, 3, 1, 1, true, true)?;
        self.conv("fuser x", 1, 256, 24, 14, 256, 1, 0, 1, true, false)?;
        self.conv("sensory", 2, 512, 24, 14, 768, 3, 1, 1, true, true)?;
        self.conv("compress", 3, 258, 24, 14, 256, 1, 0, 1, true, false)?;
        self.conv("up_16_8", 2, 256, 48, 28, 128, 3, 1, 1, true, true)?;
        self.conv("up ds", 2, 256, 48, 28, 128, 1, 0, 1, true, false)?;
        self.conv("up_8_4", 3, 128, 96, 56, 128, 3, 1, 1, true, true)?;
        self.conv("pred", 2, 128, 96, 56, 1, 3, 1, 1, true, false)?;
        self.conv("g4_conv", 2, 129, 24, 14, 256, 1, 0, 1, true, false)?;
        // Several big images run as a strided batch (several small ones side by side, above):
        // a 1x1 there reads its input directly.
        self.conv("batched 1x1", 2, 64, 96, 56, 64, 1, 0, 1, true, false)?;
        // Odd sizes: strided windows past every border, k == stride, even k, padded 1x1, no bias.
        self.conv("odd", 2, 7, 13, 9, 5, 3, 1, 2, true, false)?;
        self.conv("odd", 1, 3, 17, 11, 4, 5, 2, 1, false, false)?;
        self.conv("odd", 3, 5, 37, 23, 6, 7, 3, 2, true, false)?;
        self.conv("odd", 1, 4, 20, 14, 3, 3, 0, 3, true, false)?;
        self.conv("odd", 2, 6, 15, 9, 7, 1, 0, 2, true, false)?;
        self.conv("odd", 1, 3, 8, 6, 2, 2, 0, 2, true, false)?;
        self.conv("odd", 2, 2, 5, 4, 3, 1, 1, 1, true, false)?;
        self.conv("odd", 1, 3, 7, 7, 2, 7, 3, 2, false, false)?;
        self.conv("odd", 4, 9, 3, 2, 5, 3, 1, 1, true, false)?;
        Ok(())
    }

    // --- Pooling and elementwise ---

    fn maxpool(&mut self, planes: usize, w: usize, h: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(planes * w * h, 1.0), planes, w * h)?;
        let case = format!("{planes} planes {w}x{h}");
        let actual = gpu_cutie_maxpool3x3s2(&x, w, h).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_maxpool3x3s2_reference(&x, w, h).map_err(|e| dev(&case, e))?;
        self.compare("maxpool", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_maxpool3x3s2(&x, w, h))?;
            let reference = time_ms(2, || gpu_cutie_maxpool3x3s2_reference(&x, w, h))?;
            self.timing("maxpool", &case, kernel, reference);
        }
        Ok(())
    }

    fn sigmoid(&mut self, rows: usize, cols: usize, scale: f32, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(rows * cols, scale), rows, cols)?;
        let case = format!("{rows}x{cols} scale {scale}");
        let actual = gpu_cutie_sigmoid(&x).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_sigmoid_reference(&x).map_err(|e| dev(&case, e))?;
        self.compare("sigmoid", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_sigmoid(&x))?;
            let reference = time_ms(2, || gpu_cutie_sigmoid_reference(&x))?;
            self.timing("sigmoid", &case, kernel, reference);
        }
        Ok(())
    }

    fn gru(&mut self, n: usize, c: usize, p: usize, timed: bool) -> Result<()> {
        let h = up(&self.rng.normals(n * c * p, 1.0), n * c, p)?;
        let values = up(&self.rng.normals(n * 3 * c * p, 2.0), n * 3 * c, p)?;
        let case = format!("n{n} c{c} p{p}");
        let actual = gpu_cutie_gru(&h, &values, n, c).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_gru_reference(&h, &values, n, c).map_err(|e| dev(&case, e))?;
        self.compare("gru", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_gru(&h, &values, n, c))?;
            let reference = time_ms(2, || gpu_cutie_gru_reference(&h, &values, n, c))?;
            self.timing("gru", &case, kernel, reference);
        }
        Ok(())
    }

    fn eca(&mut self, n: usize, c: usize, p: usize, k: usize, timed: bool) -> Result<()> {
        // A channel's mean is offset per channel, as conv outputs are, so the ECA's scales vary.
        let mut xv = self.rng.normals(n * c * p, 1.0);
        let offsets = self.rng.normals(n * c, 1.0);
        for (row, off) in offsets.iter().enumerate() {
            xv[row * p..(row + 1) * p].iter_mut().for_each(|v| *v += off);
        }
        let x = up(&xv, n * c, p)?;
        let r = up(&self.rng.normals(n * c * p, 1.0), n * c, p)?;
        let w = self.rng.normals(k, 1.0);
        let case = format!("n{n} c{c} p{p} k{k}");
        let actual = gpu_cutie_eca(&x, &r, n, c, &w).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_eca_reference(&x, &r, n, c, &w).map_err(|e| dev(&case, e))?;
        self.compare("eca", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_eca(&x, &r, n, c, &w))?;
            let reference = time_ms(2, || gpu_cutie_eca_reference(&x, &r, n, c, &w))?;
            self.timing("eca", &case, kernel, reference);
        }
        Ok(())
    }

    /// Masks in 0..1 with exact 0s and 1s among them (one in eight each).
    fn masks(&mut self, n: usize, p: usize) -> Vec<f32> {
        (0..n * p)
            .map(|_| match self.next_choice(8) {
                0 => 0.0,
                1 => 1.0,
                _ => self.rng.uniform(),
            })
            .collect()
    }

    fn next_choice(&mut self, n: u64) -> u64 {
        self.rng.next_u64() % n
    }

    fn others(&mut self, n: usize, p: usize, timed: bool) -> Result<()> {
        let m = up(&self.masks(n, p), n, p)?;
        let case = format!("n{n} p{p}");
        let actual = gpu_cutie_others(&m).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_others_reference(&m).map_err(|e| dev(&case, e))?;
        self.compare("others", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_others(&m))?;
            let reference = time_ms(2, || gpu_cutie_others_reference(&m))?;
            self.timing("others", &case, kernel, reference);
        }
        Ok(())
    }

    // --- Shapes ---

    #[allow(clippy::too_many_arguments)]
    fn pad(&mut self, planes: usize, w: usize, h: usize, l: usize, r: usize, t: usize, b: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(planes * w * h, 1.0), planes, w * h)?;
        let case = format!("{planes} planes {w}x{h} by l{l} r{r} t{t} b{b}");
        let actual = gpu_cutie_pad(&x, w, h, l, r, t, b).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_pad_reference(&x, w, h, l, r, t, b).map_err(|e| dev(&case, e))?;
        self.compare("pad", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_pad(&x, w, h, l, r, t, b))?;
            let reference = time_ms(2, || gpu_cutie_pad_reference(&x, w, h, l, r, t, b))?;
            self.timing("pad", &case, kernel, reference);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn crop(&mut self, planes: usize, w: usize, h: usize, l: usize, t: usize, ow: usize, oh: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(planes * w * h, 1.0), planes, w * h)?;
        let case = format!("{planes} planes {w}x{h} to {ow}x{oh} at {l},{t}");
        let actual = gpu_cutie_crop(&x, w, h, l, t, ow, oh).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_crop_reference(&x, w, h, l, t, ow, oh).map_err(|e| dev(&case, e))?;
        self.compare("crop", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_crop(&x, w, h, l, t, ow, oh))?;
            let reference = time_ms(2, || gpu_cutie_crop_reference(&x, w, h, l, t, ow, oh))?;
            self.timing("crop", &case, kernel, reference);
        }
        Ok(())
    }

    // --- Matrix products and memory reading ---

    /// op(a) [m, k] @ op(b) [k, n], `positive` b as an affinity (non-negative, columns of
    /// mostly small values) rather than signed.
    #[allow(clippy::too_many_arguments)]
    fn matmul(&mut self, what: &str, m: usize, k: usize, n: usize, trans_a: bool, trans_b: bool, positive: bool, timed: bool) -> Result<()> {
        let a = self.rng.normals(m * k, 1.0);
        let b = if positive { self.rng.uniforms(k * n, 0.0, 1.0).iter().map(|v| v * v * v).collect() } else { self.rng.normals(k * n, 1.0) };
        let a = if trans_a { up(&a, k, m)? } else { up(&a, m, k)? };
        let b = if trans_b { up(&b, n, k)? } else { up(&b, k, n)? };
        let case = format!("{what} [{m}x{k}]{} @ [{k}x{n}]{}", if trans_a { "T" } else { "" }, if trans_b { "T" } else { "" });
        let actual = gpu_cutie_matmul(&a, trans_a, &b, trans_b).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_matmul_reference(&a, trans_a, &b, trans_b).map_err(|e| dev(&case, e))?;
        self.compare("matmul", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_matmul(&a, trans_a, &b, trans_b))?;
            let reference = time_ms(1, || gpu_cutie_matmul_reference(&a, trans_a, &b, trans_b))?;
            self.timing("matmul", &case, kernel, reference);
        }
        Ok(())
    }

    /// Memory keys [ck, m] against query keys [ck, p], with shrinkage (d^2 + 1 >= 1) and
    /// selection (a sigmoid, 0..1) when asked. Half the queries are near some memory token, as a
    /// tracked object's are, so the similarity has close pairs where its terms cancel.
    #[allow(clippy::too_many_arguments)]
    fn similarity(&mut self, what: &str, ck: usize, m: usize, p: usize, shrinkage: bool, selection: bool, timed: bool) -> Result<()> {
        let mk = self.rng.normals(ck * m, 1.0);
        let mut qk = self.rng.normals(ck * p, 1.0);
        let near = self.rng.normals(ck * p, 0.05);
        for i in (0..p).step_by(2) {
            let j = (self.rng.next_u64() % m as u64) as usize;
            for c in 0..ck {
                qk[c * p + i] = mk[c * m + j] + near[c * p + i];
            }
        }
        let mk = up(&mk, ck, m)?;
        let qk = up(&qk, ck, p)?;
        let ms = if shrinkage { Some(up(&self.rng.normals(m, 1.0).iter().map(|d| d * d + 1.0).collect::<Vec<_>>(), 1, m)?) } else { None };
        let qe = if selection { Some(up(&self.rng.uniforms(ck * p, 0.0, 1.0), ck, p)?) } else { None };
        let case = format!("{what} ck{ck} m{m} p{p}{}{}", if shrinkage { " ms" } else { "" }, if selection { " qe" } else { "" });
        let actual = gpu_cutie_similarity(&mk, ms.as_ref(), &qk, qe.as_ref()).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_similarity_reference(&mk, ms.as_ref(), &qk, qe.as_ref()).map_err(|e| dev(&case, e))?;
        self.compare("similar", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_similarity(&mk, ms.as_ref(), &qk, qe.as_ref()))?;
            let reference = time_ms(1, || gpu_cutie_similarity_reference(&mk, ms.as_ref(), &qk, qe.as_ref()))?;
            self.timing("similarity", &case, kernel, reference);
        }
        Ok(())
    }

    /// Top-k softmax of `sim` [m, p]; `values` makes the column values.
    fn topk(&mut self, what: &str, m: usize, p: usize, k: usize, values: Vec<f32>, timed: bool) -> Result<()> {
        let sim = up(&values, m, p)?;
        let case = format!("{what} m{m} p{p} k{k}");
        let (aff, usage) = gpu_cutie_topk_softmax(&sim, k).map_err(|e| dev(&case, e))?;
        let (aff_ref, usage_ref) = gpu_cutie_topk_softmax_reference(&sim, k).map_err(|e| dev(&case, e))?;
        self.compare_nonzero("topk", &format!("{case} chosen"), &aff, &aff_ref)?;
        self.compare("topk", &format!("{case} affinity"), &aff, &aff_ref, TIGHT)?;
        self.compare("topk", &format!("{case} usage"), &usage, &usage_ref, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_topk_softmax(&sim, k))?;
            let reference = time_ms(1, || gpu_cutie_topk_softmax_reference(&sim, k))?;
            self.timing("topk", &case, kernel, reference);
        }
        Ok(())
    }

    /// Similarities as Cutie's are: negative, spread over a few tens.
    fn similarities(&mut self, n: usize) -> Vec<f32> {
        self.rng.uniforms(n, -25.0, 0.0)
    }

    fn softmax_axis0(&mut self, what: &str, m: usize, p: usize, timed: bool) -> Result<()> {
        let mut v = self.rng.normals(m * p, 5.0);
        // A few -inf entries (never a whole column): exp gives 0 there.
        if m > 1 {
            for i in (0..p).step_by(97) {
                v[i] = f32::NEG_INFINITY;
            }
        }
        let x = up(&v, m, p)?;
        let case = format!("{what} m{m} p{p}");
        let actual = gpu_cutie_softmax_axis0(&x).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_softmax_axis0_reference(&x).map_err(|e| dev(&case, e))?;
        self.compare("softmax0", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_softmax_axis0(&x))?;
            let reference = time_ms(2, || gpu_cutie_softmax_axis0_reference(&x))?;
            self.timing("softmax0", &case, kernel, reference);
        }
        Ok(())
    }

    // --- Objects ---

    fn aggregate(&mut self, n: usize, p: usize, timed: bool) -> Result<()> {
        let prob = up(&self.masks(n, p), n, p)?;
        let case = format!("n{n} p{p}");
        let actual = gpu_cutie_aggregate(&prob).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_aggregate_reference(&prob).map_err(|e| dev(&case, e))?;
        self.compare("aggregate", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_aggregate(&prob))?;
            let reference = time_ms(2, || gpu_cutie_aggregate_reference(&prob))?;
            self.timing("aggregate", &case, kernel, reference);
        }
        Ok(())
    }

    /// The attention mask of aux logits `aux` [n, p], compared exactly (blocked or not).
    #[allow(clippy::too_many_arguments)]
    fn attn_mask(&mut self, what: &str, n: usize, p: usize, heads: usize, queries: usize, aux: Vec<f32>, timed: bool) -> Result<()> {
        let aux = up(&aux, n, p)?;
        let case = format!("{what} n{n} p{p} heads{heads} queries{queries}");
        let actual = gpu_cutie_attn_mask(&aux, heads, queries).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_attn_mask_reference(&aux, heads, queries).map_err(|e| dev(&case, e))?;
        self.compare_nonzero("attn_mask", &case, &actual, &expected)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_attn_mask(&aux, heads, queries))?;
            let reference = time_ms(2, || gpu_cutie_attn_mask_reference(&aux, heads, queries))?;
            self.timing("attn_mask", &case, kernel, reference);
        }
        Ok(())
    }
}

pub(crate) fn check(report: &mut Report) -> Result<()> {
    let mut ck = OpsCheck { report, rng: Rng(0x0dd_c0ffee_5eed), conv_cases: 0 };
    let result = run(&mut ck);
    // The conv cases' random weights leave the device cache whether or not they passed.
    let evicted = gpu_weight_cache_evict_prefix(&format!("{CUTIE_NAMESPACE}::{CONV_KEY_PREFIX}"));
    result?;
    evicted.map_err(|e| dev("ops evict", e))?;
    Ok(())
}

fn run(ck: &mut OpsCheck) -> Result<()> {
    // Stride 16 of the padded frame, the frame itself, and stride 4.
    const P16: usize = 24 * 14;
    const FRAME: usize = 384 * 224;
    const P4: usize = 96 * 56;

    ck.convs()?;

    // Max-pool after both stems; odd sizes put the window past every border.
    ck.maxpool(64, 192, 112, true)?;
    ck.maxpool(192, 192, 112, false)?;
    for (planes, w, h) in [(5, 15, 9), (3, 1, 1), (2, 2, 3), (4, 16, 16), (2, 7, 1), (3, 1, 6)] {
        ck.maxpool(planes, w, h, false)?;
    }

    ck.sigmoid(3, FRAME, 4.0, true)?;
    ck.sigmoid(7, 33, 40.0, false)?;

    for n in 1..=3 {
        ck.gru(n, 256, P16, n == 3)?;
    }
    ck.gru(2, 5, 17, false)?;

    ck.eca(3, 256, P16, 5, true)?;
    ck.eca(1, 256, P16, 5, false)?;
    ck.eca(2, 64, 1000, 3, false)?;
    ck.eca(2, 7, 13, 3, false)?;
    ck.eca(1, 3, 9, 5, false)?;

    for n in 1..=3 {
        ck.others(n, FRAME, n == 3)?;
    }
    ck.others(2, 37, false)?;

    // The processor pads 384x216 to 384x224 (4 rows each side) and crops back.
    ck.pad(3, 384, 216, 0, 0, 4, 4, true)?;
    ck.pad(2, 13, 7, 1, 2, 3, 0, false)?;
    ck.pad(1, 5, 5, 0, 3, 0, 0, false)?;
    ck.pad(4, 1, 1, 2, 2, 2, 2, false)?;
    ck.crop(4, 384, 224, 0, 4, 384, 216, true)?;
    ck.crop(3, 17, 11, 3, 2, 9, 5, false)?;
    ck.crop(2, 100, 60, 13, 7, 50, 41, false)?;
    ck.crop(2, 9, 7, 0, 0, 9, 7, false)?;
    ck.crop(1, 31, 19, 30, 18, 1, 1, false)?;

    // The memory readout ([objects*256, N] @ affinity [N, p]), the object summarizer
    // (weights^T @ features), consolidation's readout, and every transpose at odd sizes.
    for tokens in [336, 1680, 3500] {
        ck.matmul("readout", 3 * 256, tokens, P16, false, false, true, tokens == 3500)?;
    }
    ck.matmul("readout", 256, 1680, P16, false, false, true, false)?;
    ck.matmul("summary", 16, P16, 256, true, false, false, true)?;
    ck.matmul("consolidate", 2 * 256, 1344, 128, false, false, true, false)?;
    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
        ck.matmul("odd", 13, 7, 5, ta, tb, false, false)?;
        ck.matmul("odd", 1, 33, 17, ta, tb, false, false)?;
    }

    // Reading memory (key 64 against 336 to 3500 tokens), consolidation (candidates against
    // 128 prototypes), STCN's plain L2 (no selection), no shrinkage, odd sizes.
    for tokens in [336, 1680, 3500] {
        ck.similarity("read", 64, tokens, P16, true, true, tokens == 3500)?;
    }
    ck.similarity("consolidate", 64, 1344, 128, true, true, false)?;
    ck.similarity("plain", 64, 1680, P16, true, false, false)?;
    ck.similarity("no ms", 64, 336, P16, false, true, false)?;
    ck.similarity("no ms plain", 64, 336, P16, false, false, false)?;
    ck.similarity("odd", 7, 13, 5, true, true, false)?;

    // Top-k: Cutie's k 30 over 336 to 3500 tokens; long-term memory full (30000) at k 64; a big
    // frame's pixels; k > rows; one row; k 0; many exact ties (the lower row wins in both).
    for tokens in [336, 1680, 3500] {
        let v = ck.similarities(tokens * P16);
        ck.topk("read", tokens, P16, 30, v, tokens == 3500)?;
    }
    let v = ck.similarities(30000 * P16);
    ck.topk("long", 30000, P16, 64, v, true)?;
    let v = ck.similarities(3500 * 2500);
    ck.topk("big frame", 3500, 2500, 30, v, true)?;
    let v = ck.similarities(20 * 37);
    ck.topk("k > rows", 20, 37, 30, v, false)?;
    let v = ck.similarities(33);
    ck.topk("one row", 1, 33, 30, v, false)?;
    let v = ck.similarities(50 * 9);
    ck.topk("k 0", 50, 9, 0, v, false)?;
    let v: Vec<f32> = (0..300 * 64).map(|_| -0.5 * (ck.rng.next_u64() % 12) as f32).collect();
    ck.topk("ties", 300, 64, 30, v, false)?;

    // The decoder's softmax over background and objects at the frame; consolidation's over
    // candidates; a single row.
    for n in 1..=3 {
        ck.softmax_axis0("frame", n + 1, FRAME, n == 3)?;
    }
    ck.softmax_axis0("consolidate", 1344, 128, false)?;
    ck.softmax_axis0("one row", 1, 7, false)?;

    for n in 1..=3 {
        ck.aggregate(n, FRAME, n == 3)?;
    }
    ck.aggregate(2, P4, false)?;

    // The object transformer's masks (8 heads, 16 queries) for 1 to 3 objects; an object with
    // no foreground and one with no background (rows blocked everywhere, let through); objects
    // with identical logits (exact ties); odd sizes.
    for n in 1..=3 {
        let aux = ck.rng.normals(n * P16, 4.0);
        ck.attn_mask("aux", n, P16, 8, 16, aux, n == 3)?;
    }
    let mut aux = ck.rng.normals(3 * P16, 4.0);
    aux[P16..2 * P16].iter_mut().for_each(|v| *v = -12.0);
    ck.attn_mask("no foreground", 3, P16, 8, 16, aux, false)?;
    ck.attn_mask("no background", 1, P16, 8, 16, vec![20.0; P16], false)?;
    let row = ck.rng.normals(P16, 4.0);
    let aux = [row.clone(), row].concat();
    ck.attn_mask("tied objects", 2, P16, 8, 16, aux, false)?;
    let aux = ck.rng.normals(2 * 37, 4.0);
    ck.attn_mask("odd", 2, 37, 3, 6, aux, false)?;
    Ok(())
}
