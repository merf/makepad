//! The probe of RationalVisions docs/cutie-metal.md, step 7: what this Mac's GPU and Metal
//! compiler do with fast math off, for D8 and `makepad_common.h`. It records, it does not judge:
//! every line starts with PROBE, and only a kernel that fails to run fails the test.
//!
//! - which MSL version the libraries compile as;
//! - whether `a*b+c` written in a kernel is contracted into an fma: as compiled, and under
//!   `#pragma METAL fp contract(off)` (MSL 3.2);
//! - the ULP error of `precise::` exp, log, tanh, sin and cos against f64, over 1M inputs each;
//! - `fmax` with NaN and signed zeros (CUDA's ReLU is fmaxf(x, 0));
//! - whether denormals survive a multiply and an add.
//!
//! `cargo test -p makepad-ai-metal --release --test stream_probe -- --nocapture`, and again with
//! MAKEPAD_METAL_MATH_MODE=safe (the compile option, where the OS has it) to compare.
#![cfg(target_os = "macos")]

use makepad_ai_metal::stream::kernels::*;
use makepad_ai_metal::stream::{self, tensor_ops, with_metal_stream, Bind, FnConst, Grid, Kernel};
use makepad_ai_metal::GpuTensor;

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A uniform value in [lo, hi), in f64 then rounded.
    fn uniform(&mut self, lo: f64, hi: f64) -> f32 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (lo + (hi - lo) * unit) as f32
    }

    /// Mantissas in [1, 2), either sign, exponents from 2^-8 to 2^8.
    fn normal(&mut self) -> f32 {
        let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        let exponent = (self.next() % 17) as i32 - 8;
        sign * self.uniform(1.0, 2.0) * 2f32.powi(exponent)
    }
}

const N: usize = 1 << 20;

fn upload(values: &[f32]) -> GpuTensor {
    stream::upload(values, values.len(), 1).unwrap()
}

/// Runs `kernel` over `inputs` (bound in order) into a new output of `n` floats.
fn run(kernel: &Kernel, consts: &[FnConst], inputs: &[&GpuTensor], n: usize) -> Result<Vec<f32>, String> {
    let out = with_metal_stream("stream-probe", |s| {
        let out = GpuTensor::alloc(n, 1)?;
        let mut binds: Vec<Bind<'_>> = inputs.iter().map(|t| Bind::In(t)).collect();
        binds.push(Bind::Out(&out));
        s.launch(kernel, consts, &TensorCount { n: n as u32 }, &binds, Grid::linear(n, 256)?)?;
        Ok(out)
    })?;
    stream::download(&out)
}

/// Distance in units in the last place, +0 and -0 equal; NaN against NaN is 0, against a number
/// the most.
fn ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() { 0 } else { u64::MAX };
    }
    let ordered = |x: f32| {
        let bits = x.to_bits();
        if bits & 0x8000_0000 != 0 {
            -((bits & 0x7fff_ffff) as i64)
        } else {
            bits as i64
        }
    };
    (ordered(a) - ordered(b)).unsigned_abs()
}

#[test]
fn probe_report() {
    println!(
        "PROBE math_mode {}",
        std::env::var("MAKEPAD_METAL_MATH_MODE").unwrap_or_else(|_| "(default: fast math off only)".to_string())
    );
    let version = with_metal_stream("stream-probe version", |s| {
        let out = GpuTensor::alloc_u32(1)?;
        s.launch(&PROBE_VERSION, &[], &(), &[Bind::Out(&out)], Grid::linear(1, 1)?)?;
        Ok(out)
    })
    .and_then(|t| stream::download_u32(&t));
    println!("PROBE msl_version {version:?}");

    contraction();
    transcendentals();
    fmax_and_relu();
    denormals();
}

fn contraction() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let a: Vec<f32> = (0..N).map(|_| rng.normal()).collect();
    let b: Vec<f32> = (0..N).map(|_| rng.normal()).collect();
    let c: Vec<f32> = (0..N).map(|_| rng.normal()).collect();
    let separate: Vec<f32> = (0..N).map(|i| a[i] * b[i] + c[i]).collect();
    let fused: Vec<f32> = (0..N).map(|i| a[i].mul_add(b[i], c[i])).collect();
    let differ = (0..N).filter(|&i| separate[i].to_bits() != fused[i].to_bits()).count();
    println!("PROBE contract inputs: separate and fused rounding differ on {differ} of {N}");
    let (ta, tb, tc) = (upload(&a), upload(&b), upload(&c));
    for (name, kernel) in [("probe_mad", &PROBE_MAD), ("probe_mad_contract_off", &PROBE_MAD_CONTRACT_OFF), ("probe_fma", &PROBE_FMA)] {
        match run(kernel, &[], &[&ta, &tb, &tc], N) {
            Ok(got) => {
                let as_separate = (0..N).filter(|&i| got[i].to_bits() == separate[i].to_bits()).count();
                let as_fused = (0..N).filter(|&i| got[i].to_bits() == fused[i].to_bits()).count();
                let neither = (0..N)
                    .filter(|&i| got[i].to_bits() != separate[i].to_bits() && got[i].to_bits() != fused[i].to_bits())
                    .count();
                let verdict = if as_separate == N {
                    "not contracted"
                } else if as_fused == N {
                    "contracted (fma)"
                } else {
                    "mixed"
                };
                println!("PROBE contract {name}: {verdict}: as separate {as_separate}, as fused {as_fused}, neither {neither}");
            }
            Err(e) => println!("PROBE contract {name}: did not run: {e}"),
        }
    }
}

fn transcendentals() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    type Reference = fn(f64) -> f64;
    let cases: [(u32, &str, f64, f64, bool, Reference); 7] = [
        (0, "exp", -87.0, 88.0, false, f64::exp),
        (1, "log", -30.0, 30.0, true, f64::ln),
        (2, "tanh", -10.0, 10.0, false, f64::tanh),
        (3, "sin", -100.0, 100.0, false, f64::sin),
        (4, "cos", -100.0, 100.0, false, f64::cos),
        (3, "sin(wide)", -1.0e4, 1.0e4, false, f64::sin),
        (4, "cos(wide)", -1.0e4, 1.0e4, false, f64::cos),
    ];
    for (function, name, lo, hi, log_uniform, reference) in cases {
        // log takes 10^u for u uniform, so every decade is tried.
        let x: Vec<f32> = (0..N)
            .map(|_| {
                let u = rng.uniform(lo, hi);
                if log_uniform {
                    10f64.powf(u as f64) as f32
                } else {
                    u
                }
            })
            .collect();
        let got = match run(&PROBE_UNARY, &[FnConst::U32(0, function)], &[&upload(&x)], N) {
            Ok(got) => got,
            Err(e) => {
                println!("PROBE precise::{name}: did not run: {e}");
                continue;
            }
        };
        let mut worst = (0u64, 0usize);
        let (mut exact, mut over_one, mut sum) = (0usize, 0usize, 0f64);
        for i in 0..N {
            let want = reference(x[i] as f64) as f32;
            let d = ulps(got[i], want);
            if d == 0 {
                exact += 1;
            }
            if d > 1 {
                over_one += 1;
            }
            sum += d.min(1 << 20) as f64;
            if d > worst.0 {
                worst = (d, i);
            }
        }
        let (d, i) = worst;
        println!(
            "PROBE precise::{name} over [{lo}, {hi}]{}: max {d} ulp at x = {:e} (got {:e}, f64 {:e}), mean {:.3} ulp, \
             correctly rounded {exact}, over 1 ulp {over_one}, of {N}",
            if log_uniform { " (as powers of 10)" } else { "" },
            x[i],
            got[i],
            reference(x[i] as f64),
            sum / N as f64
        );
    }
}

fn fmax_and_relu() {
    let nan = f32::NAN;
    let snan = f32::from_bits(0x7f80_0001);
    let pairs: [(&str, f32, f32); 8] = [
        ("NaN, 1", nan, 1.0),
        ("1, NaN", 1.0, nan),
        ("NaN, NaN", nan, nan),
        ("sNaN, 1", snan, 1.0),
        ("-0, +0", -0.0, 0.0),
        ("+0, -0", 0.0, -0.0),
        ("-1, -0", -1.0, -0.0),
        ("-NaN, 0", -nan, 0.0),
    ];
    let a: Vec<f32> = pairs.iter().map(|p| p.1).collect();
    let b: Vec<f32> = pairs.iter().map(|p| p.2).collect();
    match run(&PROBE_FMAX, &[], &[&upload(&a), &upload(&b)], pairs.len()) {
        Ok(got) => {
            for (i, (name, x, y)) in pairs.iter().enumerate() {
                let host = x.max(*y);
                println!(
                    "PROBE fmax({name}) = {:e} ({:#010x}); host f32::max {:e} ({:#010x})",
                    got[i],
                    got[i].to_bits(),
                    host,
                    host.to_bits()
                );
            }
        }
        Err(e) => println!("PROBE fmax: did not run: {e}"),
    }
    let x = [-0.0f32, nan, -nan, snan, -1.0];
    match tensor_ops::relu(&upload(&x)).and_then(|t| stream::download(&t)) {
        Ok(got) => {
            for (i, v) in x.iter().enumerate() {
                println!(
                    "PROBE relu({:#010x}) = {:#010x}; host max(x, 0) {:#010x}",
                    v.to_bits(),
                    got[i].to_bits(),
                    v.max(0.0).to_bits()
                );
            }
        }
        Err(e) => println!("PROBE relu: did not run: {e}"),
    }
}

fn denormals() {
    // 2^-70 * 2^-70 = 2^-140, a denormal; 2^-140 + 2^-140 from denormal inputs.
    let small = 2f32.powi(-70);
    let tiny = f32::from_bits(1 << 9); // 2^-140
    let product = tensor_ops::mul(&upload(&[small]), &upload(&[small])).and_then(|t| stream::download(&t));
    let sum = tensor_ops::add(&upload(&[tiny]), &upload(&[tiny])).and_then(|t| stream::download(&t));
    let describe = |got: Result<Vec<f32>, String>, want: f32| match got {
        Ok(v) if v[0].to_bits() == want.to_bits() => format!("kept ({:e})", v[0]),
        Ok(v) if v[0] == 0.0 => "flushed to zero".to_string(),
        Ok(v) => format!("other: {:e} ({:#010x}), want {:e}", v[0], v[0].to_bits(), want),
        Err(e) => format!("did not run: {e}"),
    };
    println!("PROBE denormal product 2^-70 * 2^-70: {}", describe(product, small * small));
    println!("PROBE denormal sum 2^-140 + 2^-140: {}", describe(sum, tiny + tiny));
}
