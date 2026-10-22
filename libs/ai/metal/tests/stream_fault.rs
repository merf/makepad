//! A forced command-buffer failure on the Mac (RationalVisions docs/cutie-metal.md, step 7): a
//! kernel writes far past its buffer, the GPU faults, and the stream reports it at its next wait
//! with the op's label, then carries on. In the host binding that wait is the launch's own, which
//! acknowledges the failure (nothing else was in flight, and nothing it wrote was copied back), so
//! the synchronize after it is Ok. From step 12 a failure stays pending until a synchronize
//! reports it. A test binary of its own, so a GPU fault cannot touch the other tests' processes.
//!
//! `cargo test -p makepad-ai-metal --test stream_fault -- --nocapture`
#![cfg(target_os = "macos")]

use makepad_ai_metal::stream::kernels::{ProbeFault, PROBE_FAULT};
use makepad_ai_metal::stream::{self, tensor_ops, with_metal_stream, Bind, Grid};
use makepad_ai_metal::GpuTensor;

/// 2^40 floats (4 TB) past the buffer: no allocation of this process is there.
const FAR_WORDS: u64 = 1 << 40;

#[test]
fn a_gpu_fault_surfaces_at_the_next_wait_with_its_label() {
    let result = with_metal_stream("stream-fault: probe_fault", |s| {
        let out = GpuTensor::alloc(64, 1)?;
        s.launch(&PROBE_FAULT, &[], &ProbeFault { far_words: FAR_WORDS }, &[Bind::Out(&out)], Grid::linear(64, 64)?)
    });
    let err = match result {
        Ok(()) => panic!(
            "probe_fault wrote {FAR_WORDS} floats past its buffer and the command buffer completed: this GPU \
             did not fault, so the test needs another way to fail a command buffer"
        ),
        Err(err) => err,
    };
    eprintln!("the forced failure: {err}");
    assert!(err.contains("stream-fault: probe_fault"), "the error names the op's command buffer: {err}");
    assert!(err.contains("failed"), "{err}");

    // Reported by the launch, which acknowledged it: the synchronize has nothing to report.
    stream::synchronize().expect("the failure was already reported");

    // And the stream carries on.
    let after = tensor_ops::fill(4, 4, 7).and_then(|t| stream::download_u32(&t));
    match after {
        Ok(words) => assert_eq!(words, vec![7; 16]),
        Err(e) => panic!("the stream did not recover after the fault (macOS may ignore a faulting process's later work): {e}"),
    }
}
