//! Cutie, video object segmentation (Cheng et al., "Putting the Object Back into Video Object
//! Segmentation", the `merf/Cutie` fork at ec5cdd4), natively: the `cutie-base-mega` checkpoint
//! read from its `.pth`, the network on Makepad's CUDA tensors in fp32, and Cutie's memory and
//! per-frame step. Checked part by part against a PyTorch oracle (RationalVisions'
//! `tools/cutie-oracle`) with `cutie-validate`. On macOS Cutie's ops are Makepad's Metal kernels
//! (makepad-ai-metal's `cutie_ops`, dispatched by `gpu_cutie_host.rs` in makepad-ai-common), each
//! one variable away from its host reference (MAKEPAD_CUTIE_METAL_REF=all or op,op): the same
//! checks.
//!
//! ```ignore
//! let cutie = Cutie::load(Path::new("cutie-base-mega.pth"))?;
//! let mut p = cutie.processor(objects);
//! p.step(&rgb, w, h, Some(&masks), true, false)?;             // commit: a memory frame
//! // A propagation starts from its memory frame: the sensory memory cleared, the memory frame
//! // stepped again with its mask (which makes the sensory memory anew), then the frames after it
//! // without one.
//! p.clear_sensory_memory();
//! p.step(&rgb, w, h, Some(&masks), false, false)?;
//! let prob = p.step(&next_rgb, w, h, None, false, false)?;   // propagated
//! let labels = CutieProcessor::labels(&prob, objects + 1, w * h); // each pixel's object (0: none)
//! ```
//!
//! Stepping without a mask straight after `clear_sensory_memory` is an error: nothing has made
//! the sensory memory again (PyTorch's `InferenceCore` fails there too).
//!
//! **Threads**: the device tensors, the weight cache and the activation pool belong to the thread
//! that made them. Load, step and unload (`unload_cutie`, then `gpu_pool_clear` if wanted) on one
//! thread kept for Cutie. A `Cutie` moved to another thread still works there, but uploads every
//! convolution's weights again on that thread, and `unload_cutie` frees only the calling thread's
//! copy. On macOS the new thread also compiles the Metal kernel libraries again at first use
//! (`Cutie::load`'s check compiled them on the loading thread only), and a tensor that queued work
//! on the old thread's stream still uses is refused there: call `gpu_synchronize` on the old
//! thread before the move. A `CutieProcessor` borrows its `Cutie` and stays on its thread (it is
//! not `Send`).
//!
//! **Errors**: a step that fails part way (the device out of memory, a frame of another size than
//! the memory's) may leave the processor's counters advanced and its usage counted, as
//! PyTorch's `InferenceCore` does: after an error, `clear_memory` (and memorise again) or make a
//! new processor.
//!
//! Click corrections (Cutie's RITM interaction) are not part of this.

#![allow(dead_code)] // reference ops and checks sit beside the shipping paths

pub use makepad_ai_common::backend;
pub use makepad_ai_common::torch_pth;
pub use makepad_ai_common::{DiffusionError, Result};

mod bench;
mod decoder;
mod encoder;
mod mask;
mod memory;
mod model;
mod ops_check;
mod oracle;
mod processor;
mod transformer;

use crate::model::{CutieNet, Weights, CUTIE_NAMESPACE};
use crate::oracle::{Oracle, Report};
pub use crate::processor::CutieProcessor;
use std::path::Path;

/// Cutie's inference settings, as the RationalVisions worker runs it (Cutie's GUI config with
/// Jit's internal size).
#[derive(Clone, Debug)]
pub struct CutieConfig {
    /// A frame is memorised every this many steps.
    pub mem_every: usize,
    /// How many memory tokens each pixel reads from: at most 64 (the top-k kernel's limit; a
    /// step with more is an error).
    pub top_k: usize,
    /// How many steps between memorised frames update the sensory memory.
    pub stagger_updates: usize,
    /// The shorter side is resized down to this before anything else (0: never).
    pub max_internal_size: usize,
    pub use_long_term: bool,
    /// Without long-term memory, working memory keeps this many frames (one of them the
    /// permanent memory), dropping the oldest (Cutie's top-level `max_mem_frames`).
    pub max_mem_frames_without_long_term: usize,
    /// Long-term memory counts its tokens' use.
    pub count_long_term_usage: bool,
    /// With long-term memory, working memory holds this many frames (one of them the permanent
    /// memory) before it consolidates (`long_term.max_mem_frames`).
    pub max_mem_frames: usize,
    /// What consolidation leaves in working memory (`long_term.min_mem_frames`).
    pub min_mem_frames: usize,
    pub num_prototypes: usize,
    pub max_long_tokens: usize,
    pub buffer_tokens: usize,
}

impl Default for CutieConfig {
    fn default() -> Self {
        Self {
            mem_every: 5,
            top_k: 30,
            stagger_updates: 5,
            max_internal_size: 1000,
            use_long_term: true,
            max_mem_frames_without_long_term: 5,
            count_long_term_usage: true,
            max_mem_frames: 10,
            min_mem_frames: 5,
            num_prototypes: 128,
            max_long_tokens: 10000,
            buffer_tokens: 2000,
        }
    }
}

/// The network, loaded and resident on the device.
pub struct Cutie {
    pub(crate) net: CutieNet,
    pub config: CutieConfig,
}

impl Cutie {
    /// Reads the checkpoint and builds the network on the device. Fails at once where Cutie's
    /// ops do not run (`gpu_cutie_available`): a Linux or Windows build without their CUDA kernels
    /// (no nvcc), no device, or on macOS a Metal kernel library of theirs that does not compile or
    /// a GPU without the 32-wide SIMD groups some of their kernels are written for (logged). On
    /// macOS that check compiles every library on the calling thread, which is then Cutie's (see
    /// Threads above).
    pub fn load(path: &Path) -> Result<Self> {
        if !makepad_ai_common::gpu::gpu_cutie_available() {
            return Err(DiffusionError::model(
                "cutie's ops do not run here: CUDA without Cutie's kernels (no nvcc at build time), no device, or Metal kernels that do not build or run on this GPU (logged)",
            ));
        }
        let mut weights = Weights::load(path)?;
        Ok(Self { net: CutieNet::load(&mut weights)?, config: CutieConfig::default() })
    }

    /// A processor for `objects` objects (the worker's slots), with its own memory.
    pub fn processor(&self, objects: usize) -> CutieProcessor<'_> {
        CutieProcessor::new(&self.net, self.config.clone(), objects)
    }
}

/// Frees Cutie's cached device weights and the idle activation pool: the calling thread's, which
/// must be the model's (see Threads above). A thread that never used the device has nothing to
/// free, and no device context is made for it.
pub fn unload_cutie() -> Result<usize> {
    let evicted =
        crate::backend::gpu_weight_cache_evict_prefix_if_loaded(&format!("{CUTIE_NAMESPACE}::")).map_err(DiffusionError::model)?;
    crate::backend::gpu_pool_clear();
    Ok(evicted)
}

/// The validation stages, in the port's order, then "all" (every stage) and "bench" (timings, not
/// part of "all": `bench.rs`).
pub const VALIDATE_STAGES: [&str; 9] = ["ops", "encoder", "mask", "transformer", "decoder", "memory", "run", "all", "bench"];

/// What `validate` found.
pub struct ValidateSummary {
    pub comparisons: usize,
    pub failures: usize,
}

/// Checks the port against the oracle in `oracle_dir` (a clip's folder of RationalVisions'
/// `corpus/cutie-oracle`), stage by stage (`VALIDATE_STAGES`; "all" for every one), printing a
/// line per comparison. A stage that errors counts as a failure and the rest still run.
pub fn validate(oracle_dir: &Path, weights_path: &Path, stages: &[String]) -> Result<ValidateSummary> {
    let oracle = Oracle::open(oracle_dir)?;
    let mut report = Report::default();
    let all = stages.iter().any(|s| s == "all");
    let want = |s: &str| all || stages.iter().any(|x| x == s);
    for stage in &VALIDATE_STAGES[..7] {
        if !want(stage) {
            continue;
        }
        let started = std::time::Instant::now();
        let result = match *stage {
            "ops" => crate::ops_check::check(&mut report),
            stage => {
                let mut weights = Weights::load(weights_path)?;
                match stage {
                    "encoder" => crate::encoder::check(&oracle, &mut report, &mut weights),
                    "mask" => crate::mask::check(&oracle, &mut report, &mut weights),
                    "transformer" => crate::transformer::check(&oracle, &mut report, &mut weights),
                    "decoder" => crate::decoder::check(&oracle, &mut report, &mut weights),
                    "memory" => crate::memory::check(&oracle, &mut report, &mut weights),
                    _ => crate::processor::check(&oracle, &mut report, &mut weights),
                }
            }
        };
        if let Err(e) = result {
            report.fail(stage, "-", "(stage)", e.to_string());
        }
        report.note(format!("stage {stage}: {:.3} s", started.elapsed().as_secs_f64()));
    }
    if stages.iter().any(|s| s == "bench") {
        if let Err(e) = crate::bench::run(&oracle, weights_path, &mut report) {
            report.fail("bench", "-", "(stage)", e.to_string());
        }
    }
    report.print();
    let summary = ValidateSummary { comparisons: report.rows.len(), failures: report.failures };
    println!("CUTIE_VALIDATE comparisons={} failures={}", summary.comparisons, summary.failures);
    Ok(summary)
}
