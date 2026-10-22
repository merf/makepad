//! The Metal stream's bring-up on the Mac (RationalVisions docs/cutie-metal.md, step 7), in the
//! host binding: every library and pipeline compiles; fill, copy, add, mul and relu match the host
//! bit for bit; every kind of binding, function constants, threadgroup memory and the weight cache
//! work; empty grids encode nothing; and the seam's rules come back as errors, not panics.
//!
//! `cargo test -p makepad-ai-metal --test stream`. The forced command-buffer failure is
//! `--test stream_fault` (a process of its own), the probe `--test stream_probe -- --nocapture`.
#![cfg(target_os = "macos")]

use makepad_ai_metal::stream::kernels::*;
use makepad_ai_metal::stream::{
    self, tensor_ops, with_metal_stream, Bind, FnConst, Grid, Kernel, KernelLib, KernelParams,
};
use makepad_ai_metal::GpuTensor;
use std::cell::Cell;

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A unit in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Finite normal values of either sign from 2^-12 to 2^13, so sums and products stay normal.
    fn value(&mut self) -> f32 {
        let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        let exponent = (self.next() % 25) as i32 - 12;
        sign * (1.0 + self.unit()) * 2f32.powi(exponent)
    }

    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.value()).collect()
    }
}

fn upload(values: &[f32]) -> GpuTensor {
    stream::upload(values, values.len(), 1).unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Asserts `got` and `want` are the same bits, naming the first difference.
fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{what}: element {i} is {:e} ({:#010x}), the host's is {:e} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

const SIZES: [usize; 6] = [1, 7, 255, 256, 257, 100_003];

#[test]
fn every_library_and_pipeline_compiles() {
    let mut failures = Vec::new();
    for lib in [&TENSOR, &PROBE, &PROBE_CONTRACT] {
        if !stream::library_available(lib) {
            failures.push(format!("library {} does not compile", lib.name));
        }
    }
    with_metal_stream("bring-up: pipelines", |s| {
        for kernel in ALL_KERNELS {
            let consts: &[FnConst] = if kernel.function == "probe_unary" { &[FnConst::U32(0, 0)] } else { &[] };
            match s.limits(kernel, consts) {
                Ok(limits) if limits.max_threads >= 32 && limits.simd_width >= 4 => {}
                Ok(limits) => failures.push(format!("{}: odd limits {limits:?}", kernel.function)),
                Err(e) => failures.push(e),
            }
        }
        Ok(())
    })
    .unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_library_that_does_not_compile_is_an_error_even_for_an_empty_grid() {
    static BROKEN: KernelLib = KernelLib {
        name: "stream_test_broken",
        source: "#include \"makepad_common.h\"\nkernel void broken(device float* out [[buffer(1)]] {\n",
    };
    static BROKEN_KERNEL: Kernel = Kernel { lib: &BROKEN, function: "broken" };
    assert!(!stream::library_available(&BROKEN));
    let err = with_metal_stream("bring-up: broken", |s| {
        let out = GpuTensor::alloc(0, 1)?;
        s.launch(&BROKEN_KERNEL, &[], &(), &[Bind::Out(&out)], Grid::linear(0, 64)?)
    })
    .err().expect("an error");
    assert!(err.contains("stream_test_broken") && err.contains("bring-up: broken"), "{err}");
    // A missing function in a library that compiles is an error too.
    static MISSING: Kernel = Kernel { lib: &TENSOR, function: "no_such_kernel" };
    let err = with_metal_stream("bring-up: missing", |s| s.limits(&MISSING, &[]).map(|_| ())).err().expect("an error");
    assert!(err.contains("no_such_kernel"), "{err}");
}

#[test]
fn fill_writes_every_bit_pattern() {
    for word in [0u32, 0x3f80_0000, 0x8000_0000, 0x7fc0_1234, 0x7f80_0001, 0xffc0_0000, u32::MAX] {
        for (rows, cols) in [(1, 1), (3, 85), (1, 256), (1, 257), (317, 331)] {
            let t = tensor_ops::fill(rows, cols, word).unwrap();
            assert_eq!((t.rows(), t.cols()), (rows, cols));
            let words = stream::download_u32(&t).unwrap();
            assert_eq!(words.len(), rows * cols);
            if let Some(i) = words.iter().position(|&w| w != word) {
                panic!("fill {word:#010x} into {rows}x{cols}: word {i} is {:#010x}", words[i]);
            }
        }
    }
}

#[test]
fn copy_is_word_for_word() {
    let mut rng = Rng(0x5eed_0001);
    for n in SIZES {
        let mut values = rng.values(n);
        // NaN payloads and signed zeros arrive unchanged.
        values[0] = f32::from_bits(0x7f80_0001);
        values[n / 2] = -0.0;
        let src = upload(&values);
        let copied = tensor_ops::copy(&src).unwrap();
        assert_eq!(bits(&stream::download(&copied).unwrap()), bits(&values), "copy of {n}");
    }
    // Equal element counts in other shapes copy; unequal ones are refused before anything runs.
    let src = stream::upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    let reshaped = with_metal_stream("bring-up: copy reshaped", |s| {
        let out = GpuTensor::alloc(3, 2)?;
        s.copy(&src, &out)?;
        Ok(out)
    })
    .unwrap();
    assert_eq!(stream::download(&reshaped).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let err = with_metal_stream("bring-up: copy mismatch", |s| {
        let out = GpuTensor::alloc(5, 1)?;
        s.copy(&src, &out)
    })
    .err().expect("an error");
    assert!(err.contains("copy of 6 elements"), "{err}");
}

#[test]
fn add_mul_and_relu_match_the_host_bit_for_bit() {
    let mut rng = Rng(0x5eed_0002);
    for n in SIZES {
        let a = rng.values(n);
        let mut b = rng.values(n);
        b[0] = 0.0;
        let (ta, tb) = (upload(&a), upload(&b));
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let product: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        let relu: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
        assert_bits(&format!("add {n}"), &stream::download(&tensor_ops::add(&ta, &tb).unwrap()).unwrap(), &sum);
        assert_bits(&format!("mul {n}"), &stream::download(&tensor_ops::mul(&ta, &tb).unwrap()).unwrap(), &product);
        assert_bits(&format!("relu {n}"), &stream::download(&tensor_ops::relu(&ta).unwrap()).unwrap(), &relu);
        assert_bits(&format!("relu of +0 {n}"), &stream::download(&tensor_ops::relu(&tb).unwrap()).unwrap()[..1], &[0.0]);
    }
}

#[test]
fn add_and_mul_refuse_unequal_shapes() {
    let a = stream::upload(&[1.0; 6], 2, 3).unwrap();
    let b = stream::upload(&[1.0; 6], 3, 2).unwrap();
    let err = tensor_ops::add(&a, &b).err().expect("an error");
    assert!(err.contains("gpu_add shape mismatch 2x3 vs 3x2"), "{err}");
    assert!(tensor_ops::mul(&a, &b).is_err());
}

#[test]
fn empty_grids_encode_nothing() {
    // Its own thread, so the counters are this test's alone.
    std::thread::spawn(|| {
        let empty = with_metal_stream("bring-up: empty fill", |s| {
            let out = GpuTensor::alloc(0, 5)?;
            s.fill(&out, 1)?;
            Ok(out)
        })
        .unwrap();
        assert_eq!((empty.rows(), empty.cols()), (0, 5));
        let sum = tensor_ops::add(&empty, &empty).unwrap();
        assert_eq!(stream::download(&sum).unwrap(), Vec::<f32>::new());
        let stats = stream::stats(false);
        assert_eq!(stats.commits, 0, "an empty grid must not commit a command buffer");
        assert!(stats.mem_total_bytes > 0, "the stream started: {stats:?}");
        // The same op with one element commits once and waits once (the host binding).
        let one = stream::upload(&[2.0], 1, 1).unwrap();
        assert_eq!(stream::download(&tensor_ops::add(&one, &one).unwrap()).unwrap(), vec![4.0]);
        let stats = stream::stats(false);
        assert_eq!((stats.commits, stats.waits), (1, 1), "{stats:?}");
    })
    .join()
    .unwrap();
}

#[test]
fn every_kind_of_binding_reaches_the_kernel() {
    let mut rng = Rng(0x5eed_0003);
    let n = 1000;
    let x = rng.values(n);
    let table = [1.5f32, -2.25, 0.125, 3.0];
    // The weights start three floats in, behind values the kernel must not see.
    let mut w = vec![1e30f32; 3];
    w.extend(rng.values(n));
    let tx = upload(&x);
    let out = with_metal_stream("bring-up: binds", |s| {
        let weights = s.weights("stream-test::binds::w", w.len(), |dst| {
            dst.copy_from_slice(&w);
            Ok(())
        })?;
        assert_eq!(weights.floats(), n + 3);
        let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = GpuTensor::alloc(n, 1)?;
        s.launch(
            &PROBE_BINDS,
            &[],
            &ProbeBinds { n: n as u32, table_len: table.len() as u32 },
            &[Bind::In(&tx), Bind::Bytes(&table_bytes), Bind::Absent, Bind::Weights(&weights, 3), Bind::Out(&out)],
            Grid::linear(n, 64)?,
        )?;
        Ok(out)
    })
    .unwrap();
    let want: Vec<f32> = (0..n).map(|i| (x[i] + table[i % table.len()]) + w[3 + i]).collect();
    assert_bits("probe_binds", &stream::download(&out).unwrap(), &want);
}

#[test]
fn an_op_can_write_a_tensor_then_read_and_write_it_again() {
    let mut rng = Rng(0x5eed_0004);
    let n = 777;
    let (start, x) = (rng.values(n), rng.values(n));
    let (tstart, tx) = (upload(&start), upload(&x));
    let io = with_metal_stream("bring-up: inout", |s| {
        let io = GpuTensor::alloc(n, 1)?;
        s.copy(&tstart, &io)?; // Out: io now holds `start` under a new content id
        s.launch(&PROBE_INOUT, &[], &TensorCount { n: n as u32 }, &[Bind::In(&tx), Bind::InOut(&io)], Grid::linear(n, 128)?)?;
        Ok(io)
    })
    .unwrap();
    let want: Vec<f32> = (0..n).map(|i| start[i] * 2.0 + x[i]).collect();
    assert_bits("probe_inout", &stream::download(&io).unwrap(), &want);
}

#[test]
fn function_constants_choose_the_pipeline() {
    let x: Vec<f32> = (0..512).map(|i| i as f32 / 64.0 - 4.0).collect();
    let tx = upload(&x);
    let run = |f: u32| {
        with_metal_stream("bring-up: function constants", |s| {
            let out = GpuTensor::alloc(x.len(), 1)?;
            s.launch(
                &PROBE_UNARY,
                &[FnConst::U32(0, f)],
                &TensorCount { n: x.len() as u32 },
                &[Bind::In(&tx), Bind::Out(&out)],
                Grid::linear(x.len(), 64)?,
            )?;
            Ok(out)
        })
        .unwrap()
        .host()
    };
    let (exp, sin) = (run(0), run(3));
    let (exp_again, sin_again) = (run(0), run(3));
    assert_eq!(bits(&exp), bits(&exp_again));
    assert_eq!(bits(&sin), bits(&sin_again));
    for i in 0..x.len() {
        let (e, s) = ((x[i] as f64).exp(), (x[i] as f64).sin());
        assert!((exp[i] as f64 - e).abs() <= e.abs() * 1e-6, "exp({}) = {} (f64 {e})", x[i], exp[i]);
        assert!((sin[i] as f64 - s).abs() <= 1e-6, "sin({}) = {} (f64 {s})", x[i], sin[i]);
    }
}

/// A tensor's values, downloaded after its op.
trait Download {
    fn host(self) -> Vec<f32>;
}

impl Download for GpuTensor {
    fn host(self) -> Vec<f32> {
        stream::download(&self).unwrap()
    }
}

#[test]
fn threadgroup_memory_is_set_per_launch() {
    let mut rng = Rng(0x5eed_0005);
    let (n, width) = (64 * 37 + 5, 64usize);
    let x = rng.values(n);
    let tx = upload(&x);
    let groups = n.div_ceil(width);
    let sums = with_metal_stream("bring-up: threadgroup", |s| {
        let out = GpuTensor::alloc(groups, 1)?;
        let grid = Grid::linear(n, width as u32)?.threadgroup(0, (width * 4) as u32);
        s.launch(&PROBE_THREADGROUP_SUM, &[], &TensorCount { n: n as u32 }, &[Bind::In(&tx), Bind::Out(&out)], grid)?;
        Ok(out)
    })
    .unwrap()
    .host();
    let want: Vec<f32> = (0..groups)
        .map(|g| {
            let mut sum = 0.0f32;
            for k in 0..width {
                sum += x.get(g * width + k).copied().unwrap_or(0.0);
            }
            sum
        })
        .collect();
    assert_bits("probe_threadgroup_sum", &sums, &want);
}

#[test]
fn weights_fill_once_and_evict_by_prefix() {
    std::thread::spawn(|| {
        let fills = Cell::new(0);
        with_metal_stream("bring-up: weights", |s| {
            let first = s.weights("stream-test::w::cutie_conv", 10, |dst| {
                fills.set(fills.get() + 1);
                dst.fill(1.0);
                Ok(())
            })?;
            let again = s.weights("stream-test::w::cutie_conv", 10, |_| {
                fills.set(fills.get() + 1);
                Ok(())
            })?;
            assert_eq!((first.floats(), again.floats()), (10, 10));
            // Another size replaces the entry (the old one stays alive in `first`).
            let bigger = s.weights("stream-test::w::cutie_conv", 12, |dst| {
                fills.set(fills.get() + 1);
                dst.fill(2.0);
                Ok(())
            })?;
            assert_eq!(bigger.floats(), 12);
            // A fill that fails leaves nothing behind.
            let failed = s.weights("stream-test::w::cutie_ln", 4, |_| Err("refused".to_string()));
            assert_eq!(failed.err().as_deref(), Some("refused"));
            s.weights("other::w::cutie_ln", 4, |dst| {
                dst.fill(3.0);
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fills.get(), 2, "a hit must not fill");
        assert_eq!(stream::stats(false).weight_bytes, (10 + 12 + 4) * 4);
        assert_eq!(stream::evict_weights("stream-test::", false).unwrap(), 1);
        assert_eq!(stream::evict_weights("stream-test::", true).unwrap(), 0);
        assert_eq!(stream::evict_weights("other::", true).unwrap(), 1);
        assert_eq!(stream::stats(true).weight_bytes, 104);
        assert_eq!(stream::stats(false).weight_bytes, 0, "reset");
    })
    .join()
    .unwrap();
    // A thread with no stream: nothing to evict, nothing to wait for, and no stream started.
    std::thread::spawn(|| {
        assert_eq!(stream::evict_weights("stream-test::", true).unwrap(), 0);
        stream::synchronize().unwrap();
        assert_eq!(stream::stats(false).mem_total_bytes, 0, "no stream may start for these");
    })
    .join()
    .unwrap();
}

#[test]
fn the_seams_rules_are_errors_not_panics() {
    let x = stream::upload(&[1.0, 2.0], 2, 1).unwrap();
    // A nested op and a host call inside an op.
    let err = with_metal_stream("bring-up: outer", |_| with_metal_stream("bring-up: inner", |_| Ok(()))).err().expect("an error");
    assert!(err.contains("bring-up: inner") && err.contains("in use"), "{err}");
    let err = with_metal_stream("bring-up: host call", |_| stream::download(&x).map(|_| ())).err().expect("an error");
    assert!(err.contains("host call inside with_metal_stream"), "{err}");
    // Writing a tensor the op did not allocate.
    let err = with_metal_stream("bring-up: foreign out", |s| s.fill(&x, 0)).err().expect("an error");
    assert!(err.contains("did not allocate"), "{err}");
    // Params and bytes over 4096, a threadgroup too big, an input aliasing the output.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Big([u32; 1025]);
    // SAFETY: repr(C), no pointers; deliberately over the limit, never reaches a kernel.
    unsafe impl KernelParams for Big {}
    let err = with_metal_stream("bring-up: big params", |s| {
        let out = GpuTensor::alloc(1, 1)?;
        s.launch(&TENSOR_FILL, &[], &Big([0; 1025]), &[Bind::Out(&out)], Grid::linear(1, 1)?)
    })
    .err().expect("an error");
    assert!(err.contains("4100 bytes of params"), "{err}");
    let big_bytes = vec![0u8; 4097];
    let err = with_metal_stream("bring-up: big bytes", |s| {
        let out = GpuTensor::alloc(1, 1)?;
        s.launch(&TENSOR_FILL, &[], &TensorFill { n: 1, bits: 0 }, &[Bind::Out(&out), Bind::Bytes(&big_bytes)], Grid::linear(1, 1)?)
    })
    .err().expect("an error");
    assert!(err.contains("4097 bytes"), "{err}");
    let err = with_metal_stream("bring-up: wide group", |s| {
        let out = GpuTensor::alloc(4096, 1)?;
        s.launch(&TENSOR_FILL, &[], &TensorFill { n: 4096, bits: 0 }, &[Bind::Out(&out)], Grid::linear(4096, 4096)?)
    })
    .err().expect("an error");
    assert!(err.contains("4096 threads"), "{err}");
    let err = with_metal_stream("bring-up: alias", |s| {
        let out = GpuTensor::alloc(2, 1)?;
        s.launch(&TENSOR_ADD, &[], &TensorCount { n: 2 }, &[Bind::In(&out), Bind::In(&x), Bind::Out(&out)], Grid::linear(2, 64)?)
    })
    .err().expect("an error");
    assert!(err.contains("bound again"), "{err}");
    // And the stream goes on.
    assert_eq!(stream::download(&tensor_ops::add(&x, &x).unwrap()).unwrap(), vec![2.0, 4.0]);
    assert!(with_metal_stream("bring-up: gemm", |s| s.gemm(&gemm_stub(&x))).is_err(), "gemm arrives in step 8");
}

fn gemm_stub(x: &GpuTensor) -> stream::Gemm<'_> {
    let operand = stream::GemmOperand { src: Bind::In(x), offset: 0, ld: 1, trans: false, batch_stride: 0 };
    stream::Gemm {
        m: 1,
        n: 1,
        k: 1,
        batch: 1,
        alpha: 1.0,
        beta: 0.0,
        a: operand,
        b: operand,
        c: Bind::Absent,
        c_offset: 0,
        ldc: 1,
        c_batch_stride: 0,
    }
}

#[test]
fn u32_words_and_bytes_round_trip() {
    let words = [0u32, 1, 0x7f80_0001, 0xffff_fffe, u32::MAX];
    let t = stream::upload_u32(&words).unwrap();
    assert_eq!((t.rows(), t.cols()), (5, 1));
    assert_eq!(stream::download_u32(&t).unwrap(), words);
    assert_eq!(stream::download_u32(&tensor_ops::copy(&t).unwrap()).unwrap(), words);
    let from_bytes = with_metal_stream("bring-up: bytes", |s| s.upload_bytes(&[1, 0, 0, 0, 0xff])).unwrap();
    assert_eq!((from_bytes.rows(), from_bytes.cols()), (2, 1));
    assert_eq!(stream::download_u32(&from_bytes).unwrap(), vec![1, 0xff]);
}

#[test]
fn the_pool_reuses_buffers_and_inputs_are_copied_once() {
    std::thread::spawn(|| {
        let mut rng = Rng(0x5eed_0006);
        let (a, b) = (upload(&rng.values(5000)), upload(&rng.values(5000)));
        let first = tensor_ops::add(&a, &b).unwrap();
        let after_first = stream::stats(false);
        assert_eq!(after_first.pool_fresh_alloc_count, 3, "two input copies and the output: {after_first:?}");
        let second = tensor_ops::add(&a, &b).unwrap();
        let after_second = stream::stats(false);
        assert_eq!(after_second.pool_fresh_alloc_count, 3, "the inputs' copies are kept, the output's buffer pooled");
        assert_eq!(bits(&first.host()), bits(&second.host()));
        assert!(after_second.gpu_ms >= 0.0 && after_second.commits == 2 && after_second.waits == 2, "{after_second:?}");
        assert!(stream::pool_trim() > 0, "the input copies and the idle output buffer");
        stream::pool_clear();
        assert_eq!(stream::pool_trim(), 0);
    })
    .join()
    .unwrap();
}

#[test]
fn threads_run_their_own_streams() {
    let handles: Vec<_> = (0..4u64)
        .map(|seed| {
            std::thread::spawn(move || {
                let mut rng = Rng(0x5eed_1000 + seed);
                for _ in 0..20 {
                    let (a, b) = (rng.values(4099), rng.values(4099));
                    let sum = tensor_ops::add(&upload(&a), &upload(&b)).unwrap().host();
                    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                    assert_bits("threaded add", &sum, &want);
                }
                stream::synchronize().unwrap();
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
