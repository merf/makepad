//! Cutie's kernel libraries on the Mac (RationalVisions docs/cutie-metal.md, steps 9 and 10):
//! every Cutie library (the layer norm's among them) and the generic ops'
//! library compile, and every one of their kernels makes a pipeline. The GEMM's are
//! `tests/gemm.rs`'s. Families (a) and (c), the layer norm and the generic ops also have unit
//! tests against the host (`cargo test -p makepad-ai-metal --lib cutie_ops::elementwise`,
//! `cutie_ops::softmax_norm`, `layer_norm`, `generic_ops`); every op is checked against its
//! reference by `cutie-validate <oracle> ops`, the only check of families (b), (d) and (e).
//!
//! `cargo test -p makepad-ai-metal --test cutie_ops -- --nocapture` (it prints every pipeline's
//! limits). With MAKEPAD_METAL_KERNELS_SRC=<checkout>/libs/ai/metal/shaders/makepad the sources
//! are read from disk, for an edit loop without a rebuild. Also the gate (step 14): `cutie_ready`.
#![cfg(target_os = "macos")]

use makepad_ai_metal::cutie_ops::{
    cutie_kernels, cutie_ready, gpu_cutie_conv2d_fused, set_conv_mpscnn_override, CUTIE_LIBRARIES, CUTIE_MPS, MPS_CONV_KERNELS,
};
use makepad_ai_metal::generic_ops::{GENERIC_KERNELS, GENERIC_OPS};
use makepad_ai_metal::stream::{self, with_metal_stream, Kernel};

#[test]
fn cutie_is_ready_on_this_thread() {
    cutie_ready().unwrap();
}

/// MAKEPAD_CUTIE_CONV=mpscnn against the default path, on Cutie's shapes: the stem (7x7, stride
/// 2), 3x3 at strides 1 and 2, 1x1, channels not a multiple of 4, two images, a residual and a
/// ReLU. MPS keeps the weights in half and sums in its own order, so within rel 5e-4 of the largest output, not the same bits;
/// a wrong layout, offset or stride misses by far more.
#[test]
fn mpscnn_convolutions_match_the_default_path() {
    assert!(stream::library_available(&CUTIE_MPS), "cutie_mps.metal does not compile");
    let mut seed = 0x5eed_c0d5u64;
    let mut values = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    // (batch, in, w, h, out, k, stride, bias, residual, relu)
    let cases = [
        (1, 3, 97, 61, 64, 7, 2, true, false, true),
        (1, 64, 48, 28, 64, 3, 1, true, true, true),
        (2, 128, 40, 24, 128, 3, 1, true, false, true),
        (1, 128, 48, 28, 256, 3, 2, true, false, true),
        (2, 256, 24, 14, 512, 1, 1, true, true, false),
        (1, 5, 33, 17, 7, 3, 1, false, false, false),
        (2, 129, 15, 9, 1, 3, 1, true, false, false),
    ];
    for (i, &(batch, in_c, w, h, out_c, k, stride, bias, residual, relu)) in cases.iter().enumerate() {
        let pad = k / 2;
        let (ow, oh) = ((w + 2 * pad - k) / stride + 1, (h + 2 * pad - k) / stride + 1);
        let x = stream::upload(&values(batch * in_c * w * h, 1.0), batch * in_c, w * h).unwrap();
        let weights = values(out_c * in_c * k * k, 1.0 / ((in_c * k * k) as f32).sqrt());
        let b = bias.then(|| values(out_c, 0.1));
        let r = residual.then(|| stream::upload(&values(batch * out_c * ow * oh, 1.0), batch * out_c, ow * oh).unwrap());
        let key = format!("mpscnn test {i}");
        let run = |mpscnn: bool| {
            set_conv_mpscnn_override(Some(mpscnn));
            let out = gpu_cutie_conv2d_fused(&x, batch, w, h, "test", &key, &weights, b.as_deref(), out_c, k, pad, stride, r.as_ref(), relu)
                .unwrap();
            set_conv_mpscnn_override(None);
            stream::download(&out).unwrap()
        };
        let (expected, got) = (run(false), run(true));
        assert_eq!(got.len(), expected.len(), "case {i}");
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max) / scale;
        println!("mpscnn case {i} ({batch}x{in_c} {w}x{h} -> {out_c} k{k} s{stride}): worst rel {worst:.3e}");
        // MPSCNNConvolution keeps its weights in half (fp16-representable weights bring these
        // cases to 1e-5; a data source declaring fp32 does not change it), so Metal's fp16:
        // within 5e-4, not fp32's 1e-4.
        assert!(worst < 5e-4, "case {i}: worst rel {worst:.3e} against the default path");
    }
}

#[test]
fn every_cutie_library_and_pipeline_compiles() {
    let mut failures = Vec::new();
    for lib in CUTIE_LIBRARIES.iter().copied().chain([&GENERIC_OPS, &CUTIE_MPS]) {
        if !stream::library_available(lib) {
            failures.push(format!("library {} does not compile", lib.name));
        }
    }
    let kernels: Vec<&Kernel> =
        cutie_kernels().chain(GENERIC_KERNELS.iter().copied()).chain(MPS_CONV_KERNELS.iter().copied()).collect();
    with_metal_stream("cutie bring-up: pipelines", |s| {
        for kernel in &kernels {
            match s.limits(kernel, &[]) {
                Ok(limits) => println!(
                    "{:<22} {:<34} max threads {:>4}, SIMD width {}",
                    kernel.lib.name, kernel.function, limits.max_threads, limits.simd_width
                ),
                Err(e) => failures.push(e),
            }
        }
        Ok(())
    })
    .unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
