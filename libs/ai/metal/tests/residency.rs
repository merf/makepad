//! Residency on the Mac (RationalVisions docs/cutie-metal.md, steps 12 and 13): tensors live in
//! pooled Shared buffers, ops enqueue and return, and the host waits only for what it reads. These
//! check the rules that make that safe, each on a thread of its own so its stream, ring and pool
//! are its own:
//! - 10k dependent ops reuse the pool's buffers and commit on the budget, and come out exact;
//! - an upload while an older read of a dropped buffer is pending gets another buffer;
//! - `upload_into` a buffer queued work still reads goes through a staging copy queued after it;
//! - a tensor pending on its owner's stream is refused on another thread until the owner
//!   synchronizes, then adopted; a tensor dropped on another thread is released, not pooled;
//! - a buffer released while queued work still reads it (dropped on another thread, trimmed from
//!   the pool, or over the pool's cap) lives until that work is done: command buffers retain what
//!   they bind;
//! - unaligned row views are read and written where they lie, and adjacent ones concatenate to a
//!   view;
//! - a download waits for its own writer only;
//! - the ring never holds more than the in-flight cap.
//!
//! The pending cases need work that is still running when the host looks: `probe_spin`, a kernel
//! calibrated on each thread to take about 200 ms. With MAKEPAD_METAL_STREAM_SYNC=1 every op is
//! waited for at its end, so nothing is ever pending: those tests check what they can and say so.
//! The tests take turns on the GPU (`gpu_turn`), so one test's spin never holds up another's
//! command buffers or its calibration.
//!
//! `cargo test -p makepad-ai-metal --release --test residency -- --nocapture`, again with
//! MAKEPAD_METAL_MISALIGN=1 and with MAKEPAD_METAL_STREAM_SYNC=1, and once with Metal's validation
//! (MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1), which also catches a buffer used after it is freed.
#![cfg(target_os = "macos")]

use makepad_ai_metal::cutie_ops::gpu_cutie_sigmoid;
use makepad_ai_metal::layer_norm::gpu_layer_norm_pytorch;
use makepad_ai_metal::stream::kernels::{ProbeSpin, PROBE_SPIN};
use makepad_ai_metal::stream::{self, tensor_ops, with_metal_stream, Bind, Grid};
use makepad_ai_metal::{generic_ops, linear, GpuTensor};
use std::sync::mpsc::channel;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// The spin's threads: one float each.
const SPIN_THREADS: usize = 1 << 16;

/// How long a calibrated spin should keep its command buffer running.
const SPIN_MS: f64 = 200.0;

/// Held for the whole of each test. cargo runs tests side by side, each on its own queue, and
/// Apple GPUs switch between command buffers, not inside one: another test's 200 ms spin would hold
/// up a small command buffer whose download is timed, and a calibration made while another spin
/// runs comes out short.
static GPU: Mutex<()> = Mutex::new(());

/// The GPU to this test alone until the guard drops (a test that panicked holding it passes it on).
fn gpu_turn() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `f` on a thread of its own (its own stream, ring and pool), allocations not poisoned
/// whatever MAKEPAD_METAL_POISON says, with the GPU to itself, and passes on its panic.
fn on_own_thread(f: impl FnOnce() + Send + 'static) {
    let _turn = gpu_turn();
    std::thread::spawn(move || {
        stream::set_poison_allocs(Some(false));
        f()
    })
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

/// MAKEPAD_METAL_STREAM_SYNC=1: every op is waited for at its end.
fn sync_mode() -> bool {
    std::env::var("MAKEPAD_METAL_STREAM_SYNC").is_ok_and(|v| v.trim() != "" && v.trim() != "0")
}

/// Values in [1, 2), no zero and no NaN, so a sum or a move keeps every bit.
fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(0x9e37_79b9));
            1.0 + (v >> 9) as f32 / (1u32 << 23) as f32
        })
        .collect()
}

fn upload(values: &[f32]) -> GpuTensor {
    stream::upload(values, values.len(), 1).unwrap()
}

fn download(t: &GpuTensor) -> Vec<f32> {
    stream::download(t).unwrap()
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!("{what}: element {i} is {:e} ({:#010x}), want {:e} ({:#010x})", got[i], got[i].to_bits(), want[i], want[i].to_bits());
    }
}

/// `probe_spin` over `x`: `iterations` dependent steps a thread, then `x` back, in a new tensor.
fn spin(x: &GpuTensor, iterations: u32) -> GpuTensor {
    let n = x.rows() * x.cols();
    with_metal_stream("residency: spin", |s| {
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        s.launch(&PROBE_SPIN, &[], &ProbeSpin { n: n as u32, iterations }, &[Bind::In(x), Bind::Out(&out)], Grid::linear(n, 256)?)?;
        Ok(out)
    })
    .unwrap()
}

/// Iterations of a spin over `SPIN_THREADS` that take at least `ms` on this GPU, doubling until
/// they do (on the calling thread's stream). One short spin runs untimed first: the first launch
/// on a thread compiles the probe library and its pipeline, which would otherwise count as
/// spinning. (64 iterations: the spin gives `x` back once its `v` has reached 2, which takes at
/// most 33 steps from any start below 1024, with or without contraction.)
fn spin_iterations(ms: f64) -> u32 {
    let x = upload(&values(SPIN_THREADS, 99));
    assert_bits("the warm-up spin", &download(&spin(&x, 64)), &values(SPIN_THREADS, 99));
    let mut iterations = 1u32 << 12;
    loop {
        let started = Instant::now();
        let out = spin(&x, iterations);
        assert_bits("spin", &download(&out), &values(SPIN_THREADS, 99));
        let took = started.elapsed().as_secs_f64() * 1e3;
        if took >= ms || iterations >= 1 << 30 {
            eprintln!("[residency] probe_spin: {iterations} iterations take {took:.0} ms");
            return iterations;
        }
        iterations *= 2;
    }
}

/// Whether `t`'s last writer has not finished (after retiring what has).
fn write_pending(t: &GpuTensor) -> bool {
    stream::residency(t).last_write > stream::ring_state().completed
}

/// Whether work still uses `t`'s storage.
fn use_pending(t: &GpuTensor) -> bool {
    stream::residency(t).last_use > stream::ring_state().completed
}

#[test]
fn ten_thousand_dependent_ops_reuse_the_pool_and_commit_on_the_budget() {
    on_own_thread(|| {
        let n = 4099;
        let budget = std::env::var("MAKEPAD_METAL_STREAM_OPS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64u64).max(1);
        let one = stream::upload(&vec![1.0; n], n, 1).unwrap();
        let mut x = stream::upload(&vec![0.0; n], n, 1).unwrap();
        stream::stats(true);
        x = tensor_ops::add(&x, &one).unwrap();
        let first = stream::stats(false);
        if !sync_mode() && budget > 1 {
            assert_eq!((first.commits, first.waits), (0, 0), "an op enqueues and returns: {first:?}");
            assert!(stream::ring_state().open.is_some(), "its command buffer is still open");
        }
        for _ in 1..10_000 {
            x = tensor_ops::add(&x, &one).unwrap();
        }
        let enqueued = stream::stats(false);
        let got = download(&x);
        if let Some(i) = got.iter().position(|&v| v != 10_000.0) {
            panic!("element {i} is {}, not 10000", got[i]);
        }
        let stats = stream::stats(false);
        // Each add takes the buffer the add before last let go (its last use still queued: the
        // serial encoder orders the reuse): one fresh buffer, the first output's.
        assert!(stats.pool_fresh_alloc_count <= 2, "the pool reuses: {stats:?}");
        if sync_mode() {
            assert_eq!(enqueued.commits, 10_000, "SYNC commits every op: {enqueued:?}");
        } else {
            assert_eq!(enqueued.commits, 10_000 / budget, "one commit a budget of dispatches: {enqueued:?}");
            assert_eq!(stats.commits, 10_000u64.div_ceil(budget), "and the download commits the rest: {stats:?}");
        }
        eprintln!("[residency] 10k adds: {stats:?}");
    });
}

#[test]
fn an_upload_while_an_older_read_is_pending_takes_another_buffer() {
    on_own_thread(|| {
        let iterations = spin_iterations(SPIN_MS);
        let a_values = values(SPIN_THREADS, 1);
        let a = upload(&a_values);
        let out = spin(&a, iterations);
        stream::flush().unwrap();
        let a_at = stream::residency(&a);
        let pending = use_pending(&a);
        drop(a); // back to the pool, the spin still reading it
        let b_values = values(SPIN_THREADS, 2);
        let b = upload(&b_values);
        if pending {
            assert_ne!(stream::residency(&b).buffer, a_at.buffer, "the host wrote a buffer a pending read uses");
        } else {
            // The spin was waited for at its op: a's buffer is idle, and the upload takes it.
            assert!(sync_mode(), "the spin finished before the upload: calibrate longer");
            assert_eq!(stream::residency(&b).buffer, a_at.buffer, "an idle buffer of the bucket is taken again");
            eprintln!("[residency] SYNC: nothing pending, the upload took the dropped buffer at once");
        }
        assert_bits("what the spin read", &download(&out), &a_values);
        assert_bits("the upload", &download(&b), &b_values);
        if pending {
            // Once that read has finished, the host may have the buffer again.
            let c = upload(&values(SPIN_THREADS, 3));
            assert_eq!(stream::residency(&c).buffer, a_at.buffer, "an idle buffer of the bucket is taken again");
        }
    });
}

#[test]
fn upload_into_a_buffer_queued_work_reads_is_staged_behind_that_work() {
    on_own_thread(|| {
        let iterations = spin_iterations(SPIN_MS);
        let (a_values, b_values, c_values) = (values(SPIN_THREADS, 4), values(SPIN_THREADS, 5), values(SPIN_THREADS, 6));
        let t = upload(&a_values);
        let out = spin(&t, iterations);
        stream::flush().unwrap();
        let pending = use_pending(&t);
        let buffer = stream::residency(&t).buffer;
        stream::upload_into(&t, &b_values).unwrap();
        if pending {
            assert!(stream::residency(&t).last_write > 0, "staged: written by a copy queued after the spin");
        } else {
            assert!(sync_mode(), "the spin finished before upload_into: calibrate longer");
        }
        assert_eq!(stream::residency(&t).buffer, buffer, "in place, the same storage");
        assert_bits("what the spin read", &download(&out), &a_values);
        assert_bits("after upload_into", &download(&t), &b_values);

        // Idle storage takes the values in place, on the host.
        stream::synchronize().unwrap();
        stream::upload_into(&t, &c_values).unwrap();
        assert_eq!(stream::residency(&t).last_write, 0, "written by the host");
        assert_bits("after an idle upload_into", &download(&t), &c_values);

        // copy_into: queued after what uses either.
        let src = upload(&a_values);
        let out = spin(&t, iterations);
        stream::copy_into(&src, &t).unwrap();
        assert_bits("what the second spin read", &download(&out), &c_values);
        assert_bits("after copy_into", &download(&t), &a_values);

        // The refusals: a view shares the storage, the shape is not the tensor's, a read is live.
        let view = stream::slice_rows(&t, 1, 2).unwrap();
        let err = stream::upload_into(&t, &c_values).unwrap_err();
        assert!(err.contains("shared with a view"), "{err}");
        let err = stream::copy_into(&src, &view).unwrap_err();
        assert!(err.contains("gpu_copy_into shape mismatch"), "{err}");
        let err = stream::copy_into(&src, &t).unwrap_err();
        assert!(err.contains("shared with a view"), "{err}");
        drop(view);
        let err = stream::upload_into(&t, &c_values[..10]).unwrap_err();
        assert!(err.contains("gpu_upload_into shape mismatch: 10 values"), "{err}");
        let err = stream::with_host(&t, |_| stream::upload_into(&t, &c_values)).unwrap().unwrap_err();
        assert!(err.contains("being read on the host"), "{err}");
        stream::upload_into(&t, &c_values).unwrap();
        assert_bits("after the refusals", &download(&t), &c_values);
    });
}

#[test]
fn a_pending_tensor_is_refused_on_another_thread_until_its_owner_synchronizes() {
    let _turn = gpu_turn();
    // The tensor, and whether the owner's spin over it was still running when it was handed over.
    let (to_other, from_owner) = channel::<(GpuTensor, bool)>();
    let (refused, wait_refused) = channel::<()>();
    let (synced, wait_synced) = channel::<()>();
    let a_values = values(SPIN_THREADS, 7);
    let want = a_values.clone();
    let owner = std::thread::spawn(move || {
        stream::set_poison_allocs(Some(false));
        let iterations = spin_iterations(SPIN_MS);
        let t = upload(&a_values);
        let out = spin(&t, iterations);
        stream::flush().unwrap();
        let pending = use_pending(&t);
        to_other.send((t, pending)).unwrap();
        if pending {
            wait_refused.recv().unwrap();
        }
        stream::synchronize().unwrap();
        assert_bits("the owner's spin", &download(&out), &a_values);
        synced.send(()).unwrap();
        pending
    });
    let other = std::thread::spawn(move || {
        stream::set_poison_allocs(Some(false));
        let (t, pending) = from_owner.recv().unwrap();
        if pending {
            let err = tensor_ops::add(&t, &t).err().expect("refused");
            assert!(err.contains("pending on another thread's Metal stream"), "{err}");
            let err = stream::download(&t).unwrap_err();
            assert!(err.contains("pending on another thread's Metal stream"), "{err}");
            refused.send(()).unwrap();
        }
        wait_synced.recv().unwrap();
        // Finished there: adopted here.
        let doubled: Vec<f32> = want.iter().map(|v| v + v).collect();
        assert_bits("adopted", &download(&tensor_ops::add(&t, &t).unwrap()), &doubled);
        assert_eq!(stream::residency(&t).owner, stream::ring_state().stream, "this thread's stream owns it now");
    });
    let pending = owner.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    other.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    if !pending {
        assert!(sync_mode(), "the spin finished before the hand-over: calibrate longer");
        eprintln!("[residency] SYNC: nothing pending at the hand-over, the refusal is not checked");
    }
}

#[test]
fn a_tensor_dropped_on_another_thread_is_released_not_pooled() {
    on_own_thread(|| {
        stream::pool_trim();
        let t = upload(&values(1000, 8));
        let idle = stream::ring_state().idle_bytes;
        std::thread::spawn(move || {
            assert!(stream::available());
            stream::pool_trim();
            let before = stream::ring_state().idle_bytes;
            drop(t);
            assert_eq!(stream::ring_state().idle_bytes, before, "a foreign drop pools nothing here");
        })
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        assert_eq!(stream::ring_state().idle_bytes, idle, "nor in its owner's pool");
        // Dropped on its own thread, a buffer goes back to the pool, its use still recorded.
        let u = upload(&values(1000, 9));
        let bytes = stream::residency(&u).bytes;
        drop(u);
        assert_eq!(stream::ring_state().idle_bytes, idle + bytes);
    });
}

#[test]
fn a_buffer_released_while_queued_work_reads_it_lives_until_that_work_is_done() {
    // Each of three spins reads a buffer that is released for real while the spin is pending:
    // dropped on another thread, dropped here and then trimmed from the pool, dropped here with the
    // pool's cap at 0. Nothing in the stream keeps those buffers; the command buffers that bind
    // them do (retained references). New buffers of the same size, filled with other values, are
    // then taken from the device, so a freed buffer's memory would likely be written again before
    // the spins read it.
    on_own_thread(|| {
        let iterations = spin_iterations(SPIN_MS);
        let (a_values, b_values, c_values) = (values(SPIN_THREADS, 20), values(SPIN_THREADS, 21), values(SPIN_THREADS, 22));
        let mut pending = Vec::new();

        let a = upload(&a_values);
        let out_a = spin(&a, iterations);
        stream::flush().unwrap();
        pending.push(use_pending(&a));
        std::thread::spawn(move || drop(a)).join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));

        let b = upload(&b_values);
        let b_bytes = stream::residency(&b).bytes;
        let out_b = spin(&b, iterations);
        stream::flush().unwrap();
        pending.push(use_pending(&b));
        drop(b);
        assert!(stream::pool_trim() >= b_bytes, "the trim releases the dropped buffer");
        assert_eq!(stream::ring_state().idle_bytes, 0);

        stream::pool_cap_override(Some(0));
        let c = upload(&c_values);
        let out_c = spin(&c, iterations);
        stream::flush().unwrap();
        pending.push(use_pending(&c));
        drop(c);
        assert_eq!(stream::ring_state().idle_bytes, 0, "over the cap: released, not pooled");
        stream::pool_cap_override(None);

        let decoys: Vec<GpuTensor> = (0..8).map(|i| upload(&vec![-1.0 - i as f32; SPIN_THREADS])).collect();
        assert_bits("the spin over the buffer dropped on another thread", &download(&out_a), &a_values);
        assert_bits("the spin over the trimmed buffer", &download(&out_b), &b_values);
        assert_bits("the spin over the buffer over the cap", &download(&out_c), &c_values);
        for (i, decoy) in decoys.iter().enumerate() {
            assert_bits("a new buffer", &download(decoy), &vec![-1.0 - i as f32; SPIN_THREADS]);
        }
        if pending.contains(&false) {
            assert!(sync_mode(), "a spin finished before its buffer was released ({pending:?}): calibrate longer");
            eprintln!("[residency] SYNC: nothing pending at the releases, retention is not checked");
        }
    });
}

#[test]
fn unaligned_views_are_read_and_written_where_they_lie() {
    on_own_thread(|| {
        let (rows, cols) = (37, 13);
        let xs = values(rows * cols, 10);
        let x = stream::upload(&xs, rows, cols).unwrap();
        let v = stream::slice_rows(&x, 3, 20).unwrap();
        let vs = xs[3 * cols..23 * cols].to_vec();
        let (x_at, v_at) = (stream::residency(&x), stream::residency(&v));
        assert_eq!((v_at.buffer, v_at.offset), (x_at.buffer, x_at.offset + 3 * cols), "a view of x's storage");
        assert!(!v.is_aligned16(), "39 words in (40 misaligned): not on a 16-byte boundary");
        assert_bits("the view", &download(&v), &vs);
        let copy = stream::upload(&vs, 20, cols).unwrap();

        // Each op on the view gives what it gives on a fresh copy of the same values.
        let same = |what: &str, on_view: GpuTensor, on_copy: GpuTensor| {
            assert_eq!((on_view.rows(), on_view.cols()), (on_copy.rows(), on_copy.cols()), "{what}: shape");
            assert_bits(what, &download(&on_view), &download(&on_copy));
        };
        same("add", tensor_ops::add(&v, &v).unwrap(), tensor_ops::add(&copy, &copy).unwrap());
        same("relu", tensor_ops::relu(&v).unwrap(), tensor_ops::relu(&copy).unwrap());
        same("transpose", generic_ops::planar_tokens_transpose(&v).unwrap(), generic_ops::planar_tokens_transpose(&copy).unwrap());
        same("slice_cols", generic_ops::slice_cols(&v, 2, 5).unwrap(), generic_ops::slice_cols(&copy, 2, 5).unwrap());
        same("gather_cols", generic_ops::gather_cols(&v, &[12, 0, 7, 7]).unwrap(), generic_ops::gather_cols(&copy, &[12, 0, 7, 7]).unwrap());
        let bias = stream::slice_rows(&x, 30, 1).unwrap();
        let bias_copy = stream::upload(&xs[30 * cols..31 * cols], 1, cols).unwrap();
        same(
            "add_cols_broadcast",
            generic_ops::add_cols_broadcast(&v, &bias).unwrap(),
            generic_ops::add_cols_broadcast(&copy, &bias_copy).unwrap(),
        );
        // The layer norm takes widths that are multiples of 4, and a row view of such a width lies
        // on a 16-byte boundary: its view starts one word into a column instead.
        let (ln_rows, ln_cols) = (5, 16);
        let ln_values = values(1 + ln_rows * ln_cols, 11);
        let ln_column = stream::upload(&ln_values, ln_values.len(), 1).unwrap();
        let ln_view = stream::reshape(stream::slice_rows(&ln_column, 1, ln_rows * ln_cols).unwrap(), ln_rows, ln_cols).unwrap();
        assert!(!ln_view.is_aligned16(), "one word into the column: not on a 16-byte boundary");
        let ln_copy = stream::upload(&ln_values[1..], ln_rows, ln_cols).unwrap();
        let gamma = values(ln_cols, 12);
        let beta = values(ln_cols, 13);
        same(
            "layer_norm_pytorch",
            gpu_layer_norm_pytorch(&ln_view, &gamma, &beta, 1e-5).unwrap(),
            gpu_layer_norm_pytorch(&ln_copy, &gamma, &beta, 1e-5).unwrap(),
        );
        same("sigmoid", gpu_cutie_sigmoid(&v).unwrap(), gpu_cutie_sigmoid(&copy).unwrap());
        // A linear with x, W and the bias all views at odd offsets.
        let w = stream::slice_rows(&x, 25, 9).unwrap();
        let w_copy = stream::upload(&xs[25 * cols..34 * cols], 9, cols).unwrap();
        let column = values(27, 17);
        let lb = stream::slice_rows(&stream::upload(&column, 27, 1).unwrap(), 5, 9).unwrap();
        let lb_copy = stream::upload(&column[5..14], 9, 1).unwrap();
        same(
            "linear",
            linear::linear_f32_resident(&v, &w, Some(&lb)).unwrap(),
            linear::linear_f32_resident(&copy, &w_copy, Some(&lb_copy)).unwrap(),
        );

        // A reshape keeps the view's offset; adjacent views concatenate to a view, others copy.
        let reshaped = stream::reshape(stream::slice_rows(&x, 3, 20).unwrap(), cols, 20).unwrap();
        assert_eq!(stream::residency(&reshaped).offset, v_at.offset);
        assert_bits("reshaped", &download(&reshaped), &vs);
        let head = stream::slice_rows(&x, 0, 3).unwrap();
        let joined = generic_ops::concat_rows_many(&[&head, &v]).unwrap();
        let joined_at = stream::residency(&joined);
        assert_eq!((joined_at.buffer, joined_at.offset, joined.rows()), (x_at.buffer, x_at.offset, 23), "a view");
        assert_bits("joined", &download(&joined), &xs[..23 * cols]);
        let swapped = generic_ops::concat_rows_many(&[&v, &head]).unwrap();
        assert_ne!(stream::residency(&swapped).buffer, x_at.buffer, "out of order: a copy");
        let want: Vec<f32> = vs.iter().chain(&xs[..3 * cols]).copied().collect();
        assert_bits("swapped", &download(&swapped), &want);
        let one = generic_ops::concat_rows_many(&[&v]).unwrap();
        assert_eq!(stream::residency(&one).buffer, x_at.buffer, "one part: a view");
    });
}

#[test]
fn a_download_waits_for_its_own_writer_only() {
    on_own_thread(|| {
        let iterations = spin_iterations(SPIN_MS);
        let (a_values, b_values) = (values(4096, 13), values(4096, 14));
        let (a, b) = (upload(&a_values), upload(&b_values));
        let sum = tensor_ops::add(&a, &b).unwrap();
        stream::flush().unwrap();
        let slow = spin(&upload(&values(SPIN_THREADS, 15)), iterations);
        stream::flush().unwrap();
        let started = Instant::now();
        let want: Vec<f32> = a_values.iter().zip(&b_values).map(|(x, y)| x + y).collect();
        assert_bits("the sum", &download(&sum), &want);
        let waited = started.elapsed();
        if sync_mode() {
            eprintln!("[residency] SYNC: the spin was waited for at its op, nothing to check");
        } else {
            assert!(write_pending(&slow), "the download of the sum waited for the spin queued after it");
            assert!(waited < Duration::from_secs_f64(SPIN_MS / 2e3), "the sum's download took {waited:?}");
        }
        assert_bits("the spin", &download(&slow), &values(SPIN_THREADS, 15));
    });
}

#[test]
fn the_ring_holds_no_more_than_the_in_flight_cap() {
    on_own_thread(|| {
        let iterations = spin_iterations(SPIN_MS / 4.0);
        let cap = std::env::var("MAKEPAD_METAL_MAX_IN_FLIGHT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3usize).max(1);
        let x = upload(&values(SPIN_THREADS, 16));
        stream::stats(true);
        let mut outs = Vec::new();
        for i in 0..cap + 4 {
            outs.push(spin(&x, iterations));
            stream::flush().unwrap();
            let state = stream::ring_state();
            assert!(state.in_flight <= cap, "commit {i}: {} in flight, the cap is {cap}", state.in_flight);
        }
        let stats = stream::stats(false);
        if !sync_mode() {
            assert!(stats.waits >= 4, "each commit past the cap waits on the oldest: {stats:?}");
        }
        for out in &outs {
            assert_bits("a spin", &download(out), &values(SPIN_THREADS, 16));
        }
    });
}
