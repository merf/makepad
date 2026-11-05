//! A forced command-buffer failure on the Mac (RationalVisions docs/cutie-metal.md, steps 7 and
//! 13): a kernel writes far past its buffer, the GPU faults, and the stream reports it at its next
//! wait with the op's label, then carries on. The launch only enqueues: the failure is the next
//! wait's (here the download of what the op wrote), and every wait's after it until a synchronize
//! acknowledges it; what the failed command buffer wrote stays untrusted after that. With
//! MAKEPAD_METAL_STREAM_SYNC=1 the op's own end synchronizes, so the op itself returns it. A test
//! binary of its own, so a GPU fault cannot touch the other tests' processes.
//!
//! `cargo test -p makepad-ai-metal --test stream_fault -- --nocapture`. On an M5 the probe's write
//! does not fault (cutie-metal.md §11, step 7): the test then says so and fails.
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
        s.launch(&PROBE_FAULT, &[], &ProbeFault { far_words: FAR_WORDS }, &[Bind::Out(&out)], Grid::linear(64, 64)?)?;
        Ok(out)
    });
    // The op enqueued and returned; its download is the first wait. Under SYNC the op's own end
    // waited, acknowledged the failure and returned it.
    let (out, err) = match result {
        Ok(out) => match stream::download(&out) {
            Ok(_) => did_not_fault(),
            Err(err) => (Some(out), err),
        },
        Err(err) if err.contains("failed") => (None, err),
        Err(err) => panic!("the op failed before its command buffer ran: {err}"),
    };
    eprintln!("the forced failure: {err}");
    assert!(err.contains("stream-fault: probe_fault"), "the error names the op's command buffer: {err}");
    assert!(err.contains("failed"), "{err}");

    if let Some(out) = &out {
        // Sticky until a synchronize acknowledges it, which returns it once more.
        let again = stream::synchronize().expect_err("the failure is still pending acknowledgement");
        assert!(again.contains("stream-fault: probe_fault"), "{again}");
        // What the failed command buffer wrote stays untrusted after that.
        let untrusted = stream::download(out).expect_err("written by the failed command buffer");
        assert!(untrusted.contains("not trusted"), "{untrusted}");
    }
    stream::synchronize().expect("acknowledged: nothing more to report");

    // And the stream carries on.
    let after = tensor_ops::fill(4, 4, 7).and_then(|t| stream::download_u32(&t));
    match after {
        Ok(words) => assert_eq!(words, vec![7; 16]),
        Err(e) => panic!("the stream did not recover after the fault (macOS may ignore a faulting process's later work): {e}"),
    }
}

fn did_not_fault() -> ! {
    panic!(
        "probe_fault wrote {FAR_WORDS} floats past its buffer and the command buffer completed: this GPU \
         did not fault, so the test needs another way to fail a command buffer"
    )
}
