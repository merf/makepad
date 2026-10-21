//! Stage `ops`: each Cutie device op (cuda/src/cutie_ops.rs) against its host reference, on
//! seeded random inputs at Cutie's shapes (several objects, odd sizes, the padded oracle sizes)
//! and the edge cases each op has (k > rows for top-k, ties, NaN, a mask row blocked everywhere,
//! padding at every border for max-pool and the strided convs). The oracle is not read: every
//! clip gives the same result.
//!
//! The shapes are the oracle's: frames 384x216 padded to 384x224, so strides 2, 4, 8 and 16 are
//! 192x112, 96x56, 48x28 and 24x14 (336 tokens); 1 to 3 objects, and 5 to 9 where PyTorch's
//! order over objects stops being sequential; memory of 336 to 3500 tokens; and a few at the
//! worker's max_internal_size (1000 on the short side: 1792x1008 padded).
//! Ops that only move or compare values must match exactly; ops that sum or call exp, tanh or
//! log must be within fp32 rounding (`TIGHT`); top-k must also choose the same set (the kernel
//! breaks ties as the reference does: the lower row). Each op at a real shape is timed against
//! its reference too (a note per case; indicative only, the GPU is shared).
//!
//! The references share the kernels' reading of PyTorch's semantics, so a few results are also
//! pinned to PyTorch's own (`pinned`): values torch 2.11 computed on CUDA for inputs where its
//! summation order differs from a sequential one, where its bilinear resize (as the oracle runs
//! it, deterministic algorithms on: torch._decomp's lerps) differs from the default CUDA kernel's
//! fused sums, and where CUDA's sinf and cosf are not correctly rounded.

use crate::model::{dev, CUTIE_NAMESPACE};
use crate::oracle::{OracleTap, Report, Tol};
use crate::Result;
use makepad_ai_common::gpu::{
    gpu_cutie_add_broadcast, gpu_cutie_add_broadcast_reference, gpu_cutie_aggregate, gpu_cutie_aggregate_reference,
    gpu_cutie_attn_mask, gpu_cutie_attn_mask_reference, gpu_cutie_conv2d, gpu_cutie_conv2d_reference, gpu_cutie_crop,
    gpu_cutie_crop_reference, gpu_cutie_eca, gpu_cutie_eca_reference, gpu_cutie_gru, gpu_cutie_gru_reference,
    gpu_cutie_layer_norm, gpu_cutie_layer_norm_reference, gpu_cutie_matmul, gpu_cutie_matmul_reference,
    gpu_cutie_maxpool3x3s2, gpu_cutie_maxpool3x3s2_reference, gpu_cutie_normalise_rows, gpu_cutie_normalise_rows_reference,
    gpu_cutie_object_values, gpu_cutie_object_values_reference, gpu_cutie_others, gpu_cutie_others_reference, gpu_cutie_pad,
    gpu_cutie_pad_reference, gpu_cutie_resize_bilinear, gpu_cutie_resize_bilinear_reference, gpu_cutie_rgb_planes,
    gpu_cutie_rgb_planes_reference, gpu_cutie_sigmoid, gpu_cutie_sigmoid_reference, gpu_cutie_similarity,
    gpu_cutie_similarity_reference, gpu_cutie_sine_encoding, gpu_cutie_sine_encoding_reference, gpu_cutie_softmax_axis0,
    gpu_cutie_softmax_axis0_reference, gpu_cutie_topk_softmax, gpu_cutie_topk_softmax_reference, gpu_download,
    gpu_synchronize, gpu_upload, gpu_weight_cache_evict_prefix, GpuTensor,
};
use std::time::Instant;

const STAGE: &str = "ops";

/// Ops that only move, pick or compare values: bit for bit.
const EXACT: Tol = Tol { rel: 0.0, cos: 1.0 - 1e-12, bool_mismatch: 0.0 };
/// Ops that sum, or call exp, tanh or log: fp32 rounding apart.
const TIGHT: Tol = Tol { rel: 1e-5, cos: 0.9999999, bool_mismatch: 0.0 };

/// The conv and layer norm cases' weight cache keys start with this (under Cutie's namespace),
/// each case its own key: random weights under a reused key would read a stale cached copy.
/// Evicted after.
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

/// Inputs where PyTorch's order over objects or picks (four accumulators) and a sequential one
/// round differently: the golden-ratio sequence fract(i * 0.618034 + 0.05) in f32, as the probe
/// that computed the `PINNED_*` values made it.
fn golden(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let x = i as f32 * 0.618034 + 0.05;
            x - x.floor()
        })
        .collect()
}

/// torch 2.11 (CUDA, fp32) `(masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)` for 5 objects of
/// 8 pixels, masks = golden(40) * 0.15 as [1, 5, 1, 8]; its sums differ from sequential ones at
/// pixels 1, 4, 6 and 7.
const PINNED_OTHERS: [u32; 40] = [
    0x3f0be166, 0x3ea26bca, 0x3db452e1, 0x3eeaf0e2, 0x3e6b33b5, 0x3e99dc72, 0x3ebe1f02, 0x3e118ff0,
    0x3ecf3da6, 0x3ea6b371, 0x3dc57181, 0x3eef3888, 0x3e73c302, 0x3e9e2419, 0x3ec266a8, 0x3e1a1f44,
    0x3ed3854e, 0x3eaafb17, 0x3dd69021, 0x3ef38030, 0x3e7c5258, 0x3ea26bc1, 0x3ec6ae50, 0x3e22ae94,
    0x3ed7ccf6, 0x3eaf42bf, 0x3de7aed4, 0x3ef7c7dd, 0x3e8270d4, 0x3e33cd38, 0x3ecaf602, 0x3e2b3dee,
    0x3edc14a3, 0x3eb38a6c, 0x3df8cd61, 0x3efc0f8a, 0x3e86b880, 0x3e3c5c91, 0x3ecf3da5, 0x3e33cd34,
];

/// torch 2.11 (CUDA, fp32) Cutie's `aggregate(masks, dim=1)` of the same masks ([6, 8], the
/// background first); its products differ from sequential ones at pixels 3, 4, 6 and 7.
const PINNED_AGGREGATE: [u32; 48] = [
    0x3e53501b, 0x3f1a73bc, 0x3ff98bf2, 0x3dfcb514, 0x3f7b3c0d, 0x3f63ccfa, 0x3ecfcf7f, 0x3fbee105,
    0xc09c5493, 0xc00c7a06, 0xc046b4d8, 0xbfed14c0, 0xc01dc8f1, 0xc075cf30, 0xc0036771, 0xc034428c,
    0xbfdee4be, 0xc012a4b0, 0xc055214a, 0xbff674d2, 0xc0259613, 0xc08b64db, 0xc008e402, 0xc03f1591,
    0xbfe78730, 0xc019563b, 0xc06768bd, 0xc0002fcc, 0xc02e4d80, 0xc0ae2e6e, 0xc00ec698, 0xc04bde44,
    0xbff09a43, 0xc020ab81, 0xc080479b, 0xc00574eb, 0xc038341c, 0xbfe2249c, 0xc01521f0, 0xc05b8c1e,
    0xbffa2d2d, 0xc028cb95, 0xc094caba, 0xc00b1691, 0xc043b269, 0xbfeaf07e, 0xc01c0e21, 0xc06ff04a,
];

/// torch 2.11 (CUDA, fp32) Cutie's `do_softmax(sim, top_k=30)` of one column of 32, sim =
/// golden(32) * -25 as [1, 32, 1]; its denominator differs from a sequential sum's.
const PINNED_TOPK: [u32; 32] = [
    0x3e9298b5, 0x336fa8db, 0x3a4d443f, 0x2f27c995, 0x360fb560, 0x3cf62bc4, 0x31c938fa, 0x38ac580b,
    0x00000000, 0x347151c4, 0x3b4eb054, 0x3028f2d2, 0x3710b40f, 0x3df7e005, 0x32ca9d3d, 0x39ad8a3b,
    0x2e8dda2c, 0x3572fd14, 0x3c501eee, 0x312a1e70, 0x3811b525, 0x3ef997ac, 0x33cc04aa, 0x3aaebe0b,
    0x2f8ed5d0, 0x3674ac11, 0x3d519159, 0x322b4d39, 0x3912b797, 0x00000000, 0x34cd70fd, 0x3baff50f,
];

/// torch 2.11 (CUDA, fp32, deterministic algorithms on, as the oracle runs it)
/// `F.interpolate(x, size=(8, 5), mode='bilinear', align_corners=False)` of x = golden(35) as
/// [1, 1, 5, 7]: 7x5 down to 5 wide and up to 8 high. Torch then runs torch._decomp's
/// `_upsample_linear` (lerps, each its own fp32 op); its default CUDA kernel, whose weighted sums
/// nvcc fuses, gives entries 4, 24, 29, 38 and 39 1 ulp apart from these.
const PINNED_RESIZE: [u32; 40] = [
    0x3e31c5fb, 0x3ee0b187, 0x3f67733a, 0x3ebd1b60, 0x3f2274f2, 0x3ea1f6cd, 0x3f14e2aa, 0x3f1bfd20,
    0x3f031798, 0x3edaca84, 0x3f01faec, 0x3f42aefb, 0x3e805fac, 0x3f2db0b8, 0x3e399322, 0x3f162da2,
    0x3f36e1ae, 0x3ee8c524, 0x3f01e36d, 0x3ec52ef7, 0x3ef4c0ae, 0x3f1b1464, 0x3f289549, 0x3ecc2c42,
    0x3f06ca2c, 0x3e3a4c2a, 0x3ede8e36, 0x3f5cc7fb, 0x3eb491a8, 0x3f1afcd9, 0x3eabf1e0, 0x3f19e02f,
    0x3f00faa4, 0x3f08151e, 0x3eb1925f, 0x3ef505a6, 0x3f3e6a0e, 0x3e561200, 0x3f2c9f00, 0x3e0ee608,
];

/// torch 2.11 (CUDA) Cutie's `PositionalEncoding(16, scale=32, temperature=128)` at 3x2 ([1, 2,
/// 3, 16]) and its `inv_freq`: 10 of its 96 entries are 1 ulp from sin and cos rounded from f64.
const PINNED_SINE_FREQS: [u32; 4] = [0x3f800000, 0x3e9837f0, 0x3db504f3, 0x3cd744fd];
const PINNED_SINE: [u32; 96] = [
    0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000,
    0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000,
    0xbe93671b, 0xbf752951, 0xbf7fbf4e, 0x3d35f42e, 0x3f7cde39, 0x3e1fafe5, 0x3ed0fb97, 0x3f69b438,
    0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000,
    0x3f0d2977, 0x3f559037, 0xbdb5c632, 0xbf7efd5a, 0x3e9dbbc3, 0xbf738c81, 0x3f3ec81c, 0x3f2ab31a,
    0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000,
    0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000,
    0x3f0d28a2, 0x3f5590c4, 0xbdb5c3b5, 0xbf7efd61, 0x3e9dbbea, 0xbf738c7b, 0x3f3ec818, 0x3f2ab31e,
    0xbe93671b, 0xbf752951, 0xbf7fbf4e, 0x3d35f42e, 0x3f7cde39, 0x3e1fafe5, 0x3ed0fb97, 0x3f69b438,
    0x3f0d28a2, 0x3f5590c4, 0xbdb5c3b5, 0xbf7efd61, 0x3e9dbbea, 0xbf738c7b, 0x3f3ec818, 0x3f2ab31e,
    0x3f0d2977, 0x3f559037, 0xbdb5c632, 0xbf7efd5a, 0x3e9dbbc3, 0xbf738c81, 0x3f3ec81c, 0x3f2ab31a,
    0x3f0d28a2, 0x3f5590c4, 0xbdb5c3b5, 0xbf7efd61, 0x3e9dbbea, 0xbf738c7b, 0x3f3ec818, 0x3f2ab31e,
];

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

    /// A result against values PyTorch computed (`PINNED_*`, as bits).
    fn compare_pinned(&mut self, op: &str, case: &str, actual: &GpuTensor, bits: &[u32], tol: Tol) -> Result<()> {
        let tap = OracleTap {
            shape: vec![actual.rows(), actual.cols()],
            dtype: "f32".into(),
            values: bits.iter().map(|&b| f32::from_bits(b)).collect(),
        };
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
        // Images whose columns outgrow the im2col scratch go through in groups: a frame at the
        // worker's max_internal_size (1792x1008 padded), one image a GEMM; three images that fit
        // two at a time (groups of 2, then 1).
        self.conv("mask stem 1000p", 2, 5, 1792, 1008, 64, 7, 3, 2, true, true)?;
        self.conv("grouped", 3, 64, 224, 192, 64, 3, 1, 1, true, false)?;
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

    /// Max-pool of random planes; `nan_every` puts a NaN at every so many taps (0: none), which
    /// wins its windows as in PyTorch.
    fn maxpool(&mut self, planes: usize, w: usize, h: usize, nan_every: usize, timed: bool) -> Result<()> {
        let mut v = self.rng.normals(planes * w * h, 1.0);
        if nan_every > 0 {
            v.iter_mut().step_by(nan_every).for_each(|t| *t = f32::NAN);
        }
        let x = up(&v, planes, w * h)?;
        let case = format!("{planes} planes {w}x{h}{}", if nan_every > 0 { " nan" } else { "" });
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

    // --- The per-frame step ---

    /// One block [rows, cols] added to each of `k` blocks.
    fn add_broadcast(&mut self, rows: usize, cols: usize, k: usize, timed: bool) -> Result<()> {
        let a = up(&self.rng.normals(rows * cols, 1.0), rows, cols)?;
        let b = up(&self.rng.normals(k * rows * cols, 1.0), k * rows, cols)?;
        let case = format!("{rows}x{cols} onto {k}");
        let actual = gpu_cutie_add_broadcast(&a, &b).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_add_broadcast_reference(&a, &b).map_err(|e| dev(&case, e))?;
        self.compare("add_bcast", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_add_broadcast(&a, &b))?;
            let reference = time_ms(2, || gpu_cutie_add_broadcast_reference(&a, &b))?;
            self.timing("add_bcast", &case, kernel, reference);
        }
        Ok(())
    }

    /// (x - sub[r]) / div[r] of 0..1 planes; `sub` and `div` ImageNet's for 3 rows, else random.
    fn normalise_rows(&mut self, rows: usize, cols: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.uniforms(rows * cols, 0.0, 1.0), rows, cols)?;
        let (sub, div) = if rows == 3 {
            (vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
        } else {
            (self.rng.uniforms(rows, 0.0, 1.0), self.rng.uniforms(rows, 0.1, 1.0))
        };
        let (sub, div) = (up(&sub, rows, 1)?, up(&div, rows, 1)?);
        let case = format!("{rows}x{cols}");
        let actual = gpu_cutie_normalise_rows(&x, &sub, &div).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_normalise_rows_reference(&x, &sub, &div).map_err(|e| dev(&case, e))?;
        self.compare("normalise", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_normalise_rows(&x, &sub, &div))?;
            let reference = time_ms(2, || gpu_cutie_normalise_rows_reference(&x, &sub, &div))?;
            self.timing("normalise", &case, kernel, reference);
        }
        Ok(())
    }

    /// Random RGB bytes w x h to planes, times 1/255 as the processor does.
    fn rgb_planes(&mut self, w: usize, h: usize, timed: bool) -> Result<()> {
        let rgb: Vec<u8> = (0..w * h * 3).map(|_| self.rng.next_u64() as u8).collect();
        let (plane, scale) = (w * h, 1.0f32 / 255.0f32);
        let case = format!("{w}x{h}");
        let actual = gpu_cutie_rgb_planes(&rgb, plane, scale).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_rgb_planes_reference(&rgb, plane, scale).map_err(|e| dev(&case, e))?;
        self.compare("rgb", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_rgb_planes(&rgb, plane, scale))?;
            let reference = time_ms(2, || gpu_cutie_rgb_planes_reference(&rgb, plane, scale))?;
            self.timing("rgb", &case, kernel, reference);
        }
        Ok(())
    }

    /// Bilinear resize of random 0..1 planes, in_w x in_h to out_w x out_h.
    #[allow(clippy::too_many_arguments)]
    fn resize(&mut self, planes: usize, in_w: usize, in_h: usize, out_w: usize, out_h: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.uniforms(planes * in_w * in_h, 0.0, 1.0), planes, in_w * in_h)?;
        let case = format!("{planes} planes {in_w}x{in_h} to {out_w}x{out_h}");
        let actual = gpu_cutie_resize_bilinear(&x, in_w, in_h, out_w, out_h).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_resize_bilinear_reference(&x, in_w, in_h, out_w, out_h).map_err(|e| dev(&case, e))?;
        self.compare("resize", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_resize_bilinear(&x, in_w, in_h, out_w, out_h))?;
            let reference = time_ms(1, || gpu_cutie_resize_bilinear_reference(&x, in_w, in_h, out_w, out_h))?;
            self.timing("resize", &case, kernel, reference);
        }
        Ok(())
    }

    /// The object transformer's values from `objects` objects' memory [objects*16, 257]: sums of
    /// a few hundred, areas from 0 (only the 1e-4 left) up to the frame's tokens.
    fn object_values(&mut self, objects: usize, timed: bool) -> Result<()> {
        const QUERIES: usize = 16;
        const WIDTH: usize = 257;
        let mut memory = self.rng.normals(objects * QUERIES * WIDTH, 100.0);
        for (r, row) in memory.chunks_mut(WIDTH).enumerate() {
            row[WIDTH - 1] = match r % 4 {
                0 => 0.0,
                1 => 1e-6,
                _ => self.rng.uniform() * 336.0,
            };
        }
        let memory = up(&memory, objects * QUERIES, WIDTH)?;
        let case = format!("n{objects}");
        let actual = gpu_cutie_object_values(&memory, objects, 1e-4).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_object_values_reference(&memory, objects, 1e-4).map_err(|e| dev(&case, e))?;
        self.compare("obj_values", &case, &actual, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_object_values(&memory, objects, 1e-4))?;
            let reference = time_ms(2, || gpu_cutie_object_values_reference(&memory, objects, 1e-4))?;
            self.timing("obj_values", &case, kernel, reference);
        }
        Ok(())
    }

    /// Cutie's sine encoding (64 frequencies 1 / 128^(2i/128), scale 32) at w x h.
    fn sine_encoding(&mut self, w: usize, h: usize, timed: bool) -> Result<()> {
        let inv_freq: Vec<f32> = (0..64).map(|i| (1.0 / 128f64.powf(2.0 * i as f64 / 128.0)) as f32).collect();
        let case = format!("{w}x{h}");
        let actual = gpu_cutie_sine_encoding(&inv_freq, w, h, 32.0, 1e-6).map_err(|e| dev(&case, e))?;
        let expected = gpu_cutie_sine_encoding_reference(&inv_freq, w, h, 32.0, 1e-6).map_err(|e| dev(&case, e))?;
        self.compare("sine", &case, &actual, &expected, TIGHT)?;
        if timed {
            let kernel = time_ms(5, || gpu_cutie_sine_encoding(&inv_freq, w, h, 32.0, 1e-6))?;
            let reference = time_ms(2, || gpu_cutie_sine_encoding_reference(&inv_freq, w, h, 32.0, 1e-6))?;
            self.timing("sine", &case, kernel, reference);
        }
        Ok(())
    }

    /// LayerNorm of rows [rows, cols] with its affine cached against `gpu_layer_norm_pytorch`.
    fn layer_norm(&mut self, rows: usize, cols: usize, timed: bool) -> Result<()> {
        let x = up(&self.rng.normals(rows * cols, 3.0), rows, cols)?;
        let gamma = self.rng.normals(cols, 1.0);
        let beta = self.rng.normals(cols, 0.5);
        self.conv_cases += 1;
        let key = format!("{CONV_KEY_PREFIX}:{}:layer norm", self.conv_cases);
        let case = format!("{rows}x{cols}");
        let run = |reference: bool| {
            let f = if reference { gpu_cutie_layer_norm_reference } else { gpu_cutie_layer_norm };
            f(&x, CUTIE_NAMESPACE, &key, &gamma, &beta, 1e-5)
        };
        let actual = run(false).map_err(|e| dev(&case, e))?;
        // Again, from the cached affine.
        let again = run(false).map_err(|e| dev(&case, e))?;
        let expected = run(true).map_err(|e| dev(&case, e))?;
        self.compare("layer_norm", &case, &actual, &expected, EXACT)?;
        self.compare("layer_norm", &format!("{case} cached"), &again, &expected, EXACT)?;
        if timed {
            let kernel = time_ms(5, || run(false))?;
            let reference = time_ms(5, || run(true))?;
            self.timing("layer_norm", &case, kernel, reference);
        }
        Ok(())
    }

    // --- PyTorch's own results ---

    /// The kernels against PyTorch's values (`PINNED_*`) where its order over objects and picks
    /// is not sequential, where its bilinear resize is lerps (deterministic algorithms on, as the
    /// oracle runs it) rather than the default kernel's fused sums, and where its sin
    /// and cos are not correctly rounded: exactly (the same fp32 operations in the same order,
    /// CUDA's own expf, logf, sinf and cosf). The references too: exactly where they only add,
    /// multiply and fuse, within `TIGHT` where they call the host's exp, ln, sin or cos.
    fn pinned(&mut self) -> Result<()> {
        let masks: Vec<f32> = golden(40).iter().map(|v| v * 0.15).collect();
        let masks = up(&masks, 5, 8)?;
        let case = "torch n5 p8";
        let others = gpu_cutie_others(&masks).map_err(|e| dev(case, e))?;
        self.compare_pinned("others", &format!("{case} kernel"), &others, &PINNED_OTHERS, EXACT)?;
        let others = gpu_cutie_others_reference(&masks).map_err(|e| dev(case, e))?;
        self.compare_pinned("others", &format!("{case} reference"), &others, &PINNED_OTHERS, EXACT)?;
        let agg = gpu_cutie_aggregate(&masks).map_err(|e| dev(case, e))?;
        self.compare_pinned("aggregate", &format!("{case} kernel"), &agg, &PINNED_AGGREGATE, EXACT)?;
        let agg = gpu_cutie_aggregate_reference(&masks).map_err(|e| dev(case, e))?;
        self.compare_pinned("aggregate", &format!("{case} reference"), &agg, &PINNED_AGGREGATE, TIGHT)?;

        let sim: Vec<f32> = golden(32).iter().map(|v| v * -25.0).collect();
        let sim = up(&sim, 32, 1)?;
        let case = "torch m32 p1 k30";
        let (aff, _) = gpu_cutie_topk_softmax(&sim, 30).map_err(|e| dev(case, e))?;
        self.compare_pinned("topk", &format!("{case} kernel"), &aff, &PINNED_TOPK, EXACT)?;
        let (aff, _) = gpu_cutie_topk_softmax_reference(&sim, 30).map_err(|e| dev(case, e))?;
        self.compare_pinned("topk", &format!("{case} reference"), &aff, &PINNED_TOPK, TIGHT)?;

        let x = up(&golden(35), 1, 35)?;
        let case = "torch 7x5 to 5x8";
        let out = gpu_cutie_resize_bilinear(&x, 7, 5, 5, 8).map_err(|e| dev(case, e))?;
        self.compare_pinned("resize", &format!("{case} kernel"), &out, &PINNED_RESIZE, EXACT)?;
        let out = gpu_cutie_resize_bilinear_reference(&x, 7, 5, 5, 8).map_err(|e| dev(case, e))?;
        self.compare_pinned("resize", &format!("{case} reference"), &out, &PINNED_RESIZE, EXACT)?;

        // The kernel is CUDA's sinf and cosf, PyTorch's to the bit; the reference rounds from f64.
        let inv_freq: Vec<f32> = PINNED_SINE_FREQS.iter().map(|&b| f32::from_bits(b)).collect();
        let case = "torch dim 16 at 3x2";
        let pe = gpu_cutie_sine_encoding(&inv_freq, 3, 2, 32.0, 1e-6).map_err(|e| dev(case, e))?;
        self.compare_pinned("sine", &format!("{case} kernel"), &pe, &PINNED_SINE, EXACT)?;
        let pe = gpu_cutie_sine_encoding_reference(&inv_freq, 3, 2, 32.0, 1e-6).map_err(|e| dev(case, e))?;
        self.compare_pinned("sine", &format!("{case} reference"), &pe, &PINNED_SINE, TIGHT)?;
        Ok(())
    }
}

/// Top-k similarities with the values torch.topk orders specially: [m, 8] columns of
/// `similarities` except column 0 all NaN, column 1 NaN but for its first 10 rows (fewer finite
/// rows than k), column 2 a negative and a positive NaN (both picked first), column 3 29 rows of
/// 1 then -0 at row 3 and +0 at row 9 (the 30th pick is the +0: torch.topk's key puts it above
/// -0), column 4 a +inf and a -inf (inf/inf is NaN), column 5 all -inf (0/0 is NaN).
fn topk_specials(mut v: Vec<f32>, m: usize) -> Vec<f32> {
    const P: usize = 8;
    for j in 0..m {
        v[j * P] = f32::NAN;
        if j >= 10 {
            v[j * P + 1] = f32::NAN;
        }
        v[j * P + 3] = if (100..129).contains(&j) { 1.0 } else { -1.0 };
        v[j * P + 5] = f32::NEG_INFINITY;
    }
    v[5 * P + 2] = f32::from_bits(0xffc0_0000);
    v[(m - 1) * P + 2] = f32::NAN;
    v[3 * P + 3] = -0.0;
    v[9 * P + 3] = 0.0;
    v[7 * P + 4] = f32::INFINITY;
    v[(m / 2) * P + 4] = f32::NEG_INFINITY;
    v
}

pub(crate) fn check(report: &mut Report) -> Result<()> {
    report.note("ops: seeded inputs against the host references and values pinned from torch; the oracle is not read");
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

    ck.pinned()?;

    // Max-pool after both stems; odd sizes put the window past every border; NaN taps.
    ck.maxpool(64, 192, 112, 0, true)?;
    ck.maxpool(192, 192, 112, 0, false)?;
    for (planes, w, h) in [(5, 15, 9), (3, 1, 1), (2, 2, 3), (4, 16, 16), (2, 7, 1), (3, 1, 6)] {
        ck.maxpool(planes, w, h, 0, false)?;
    }
    ck.maxpool(4, 37, 23, 97, false)?;

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

    // 1 to 3 objects, and 5, 6 and 9, where PyTorch's sum over objects is no longer sequential.
    for n in 1..=3 {
        ck.others(n, FRAME, n == 3)?;
    }
    ck.others(2, 37, false)?;
    ck.others(5, FRAME, false)?;
    ck.others(6, P16, false)?;
    ck.others(9, 37, false)?;

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
    // NaN, signed zeros and infinities, over 3 slices of rows so the merge sees them too.
    let v = topk_specials(ck.similarities(700 * 8), 700);
    ck.topk("specials", 700, 8, 30, v, false)?;

    // The decoder's softmax over background and objects at the frame; consolidation's over
    // candidates; a single row.
    for n in 1..=3 {
        ck.softmax_axis0("frame", n + 1, FRAME, n == 3)?;
    }
    ck.softmax_axis0("consolidate", 1344, 128, false)?;
    ck.softmax_axis0("consolidate 1000p", 5 * 112 * 63, 128, true)?;
    ck.softmax_axis0("one row", 1, 7, false)?;

    for n in 1..=3 {
        ck.aggregate(n, FRAME, n == 3)?;
    }
    ck.aggregate(2, P4, false)?;
    ck.aggregate(5, FRAME, false)?;
    ck.aggregate(6, P4, false)?;
    ck.aggregate(9, 37, false)?;

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
    // 5, 6 and 9 objects (PyTorch's background product is no longer sequential); NaN logits
    // (the pixel is no object's foreground).
    for n in [5, 6] {
        let aux = ck.rng.normals(n * P16, 4.0);
        ck.attn_mask("aux", n, P16, 8, 16, aux, false)?;
    }
    let aux = ck.rng.normals(9 * 37, 4.0);
    ck.attn_mask("odd", 9, 37, 3, 6, aux, false)?;
    let mut aux = ck.rng.normals(2 * P16, 4.0);
    aux.iter_mut().step_by(29).for_each(|v| *v = f32::NAN);
    ck.attn_mask("nan", 2, P16, 8, 16, aux, false)?;

    // The per-frame step: the fusion blocks' and the decoder's x + g (1 to 3 objects at strides
    // 16, 8 and 4), the summarizer's encoding added to each object's tokens; odd sizes.
    for n in 1..=3 {
        ck.add_broadcast(256, P16, n, n == 3)?;
    }
    ck.add_broadcast(128, 48 * 28, 2, false)?;
    ck.add_broadcast(128, P4, 3, true)?;
    ck.add_broadcast(P16, 256, 2, false)?;
    ck.add_broadcast(5, 37, 4, false)?;
    // The image normalised: the oracle's frame, a 1000p one, odd rows.
    ck.normalise_rows(3, FRAME, true)?;
    ck.normalise_rows(3, 1792 * 1008, true)?;
    ck.normalise_rows(7, 33, false)?;
    // The frame's bytes to planes.
    ck.rgb_planes(384, 216, true)?;
    ck.rgb_planes(1920, 1080, true)?;
    ck.rgb_planes(7, 5, false)?;
    ck.rgb_planes(1, 1, false)?;
    // A frame over max_internal_size resized in and the probabilities back out (1080p and 4K,
    // portrait), and odd sizes: down, up, mixed, one pixel either way.
    ck.resize(3, 1920, 1080, 1777, 1000, true)?;
    ck.resize(3, 1777, 1000, 1920, 1080, true)?;
    ck.resize(2, 1080, 1920, 1000, 1777, false)?;
    ck.resize(1, 3840, 2160, 1777, 1000, false)?;
    ck.resize(2, 1440, 1080, 1333, 1000, false)?;
    for (planes, iw, ih, ow, oh) in [(2, 7, 5, 4, 3), (2, 4, 3, 9, 7), (1, 1, 1, 3, 2), (1, 3, 2, 1, 1), (3, 5, 1, 2, 1), (1, 13, 9, 13, 4)] {
        ck.resize(planes, iw, ih, ow, oh, false)?;
    }
    // The decoder's x2 (strides 16 to 8 and 8 to 4, every object's channels) and x4 (the
    // logits to the padded frame).
    ck.resize(2 * 256, 24, 14, 48, 28, false)?;
    ck.resize(3 * 128, 48, 28, 96, 56, true)?;
    ck.resize(3, 96, 56, 384, 224, true)?;
    // The object transformer's values for 1 to 3 objects, and 6.
    for n in 1..=3 {
        ck.object_values(n, n == 3)?;
    }
    ck.object_values(6, false)?;
    // The sine encoding at the oracle's stride-16 grids (384x224, 960x544), 1000p's, a pixel.
    ck.sine_encoding(24, 14, true)?;
    ck.sine_encoding(60, 34, false)?;
    ck.sine_encoding(112, 63, true)?;
    ck.sine_encoding(1, 1, false)?;
    // The object transformer's LayerNorms: queries (16 an object) and odd rows.
    ck.layer_norm(16 * 3, 256, true)?;
    ck.layer_norm(16, 256, false)?;
    ck.layer_norm(7, 12, false)?;
    Ok(())
}
