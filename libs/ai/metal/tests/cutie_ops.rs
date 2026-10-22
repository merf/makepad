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

use makepad_ai_metal::cutie_ops::{cutie_kernels, cutie_ready, CUTIE_LIBRARIES};
use makepad_ai_metal::generic_ops::{GENERIC_KERNELS, GENERIC_OPS};
use makepad_ai_metal::stream::{self, with_metal_stream, Kernel};

#[test]
fn cutie_is_ready_on_this_thread() {
    cutie_ready().unwrap();
}

#[test]
fn every_cutie_library_and_pipeline_compiles() {
    let mut failures = Vec::new();
    for lib in CUTIE_LIBRARIES.iter().copied().chain([&GENERIC_OPS]) {
        if !stream::library_available(lib) {
            failures.push(format!("library {} does not compile", lib.name));
        }
    }
    let kernels: Vec<&Kernel> = cutie_kernels().chain(GENERIC_KERNELS.iter().copied()).collect();
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
