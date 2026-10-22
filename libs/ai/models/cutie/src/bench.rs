//! `cutie-validate <oracle-dir> bench`: how fast the port runs an oracle's run, and where the time
//! goes. Not part of `all`, and nothing is compared with the oracle: its frames, seed masks and
//! schedule are only the workload (the worker's own: commit frame 0, clear the sensory memory,
//! frame 0 again with its mask, then frames 1.. propagated).
//!
//! - **Load**: the checkpoint read (`Weights::load`) and the network built (`CutieNet::load`, the
//!   linears uploaded; the convolutions' weights go up on their first use, in the first pass).
//! - **Passes**: the whole run replayed `MAKEPAD_CUTIE_BENCH_PASSES` times (5 unless set), each
//!   with a fresh processor on the same network, taps off, every step synchronised before and
//!   after (`gpu_synchronize`; a step ends with its probabilities downloaded anyway). A step's
//!   time is `CutieProcessor::step` as the worker calls it: RGB bytes in, every object's
//!   probability at the frame's size on the host out. The first pass is cold (weights uploaded,
//!   kernels and cuBLAS loaded); the others are summarised by kind of step: the commit, the
//!   restart (frame 0 again with its mask), frames propagated without memorising, memorised, and
//!   memorised with a consolidation. On macOS, also how many times a warm step waited on the
//!   Metal stream and how many command buffers it committed (`gpu_stream_counts`).
//! - **Memory**: the convolutions' weights (the weight cache's uploads), the activation pool's
//!   allocations less what it freed (it keeps freed buffers for reuse: what the run holds on the
//!   device at most, besides the weights), and the whole device's use over the baseline taken
//!   before loading (cudaMemGetInfo after every step: other processes move it).
//! - **Parts**: three more passes with `step_in_parts`, the step taken apart with a
//!   synchronisation after each part (the frame in, `encode_image`, `transform_key`, the memory
//!   read's similarity, top-k softmax and readout, pixel fusion, the object transformer,
//!   `segment`, `encode_mask`, `add_memory`, the frame out): each part's time to finish, and how
//!   long the host took to enqueue it (when that is most of it, the part waits on the host, not
//!   the GPU). Their outputs must be the plain passes' to the bit (reported), and every step's
//!   probabilities hashed together give the run's digest, which a change that keeps the results
//!   to the bit leaves alone across builds.
//! - **Convolutions**: the pixel encoder's, the decoder's and the mask encoder's convolutions at
//!   the frame's size (the last two for every object), each alone (`conv_profile`): as the
//!   network runs it (the bias, the residual and the ReLU in one pass), with the residual's add
//!   and the ReLU as passes of their own, and the bare GEMM of the same shape (what is left is
//!   im2col and the epilogue).
//!
//! Prints `BENCH` lines and one `CUTIE_BENCH {json}` line. The GPU may be shared: compare runs
//! made close together, several times. The Python worker's side is RationalVisions'
//! `tools/cutie-oracle/cutie_oracle.py bench`, on the same oracle.

use crate::memory::{KeyValueStore, Permanence};
use crate::model::{add, dev, relu, Conv2d, CutieNet, Feat, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::processor::{CutieProcessor, Frame, WorkerMasks};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    cutie_conv_out, gpu_cutie_matmul, gpu_cutie_maxpool3x3s2, gpu_cutie_similarity, gpu_cutie_topk_softmax, gpu_download, gpu_perf_stats,
    gpu_rife_fill, gpu_stream_counts, gpu_synchronize, gpu_upload,
};
use std::path::Path;
use std::time::Instant;

/// Passes timed in parts after the plain ones.
const PART_PASSES: usize = 3;
const MB: f64 = 1024.0 * 1024.0;

/// What a step of the run did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Frame 0 with the seed masks, memorised as permanent memory.
    Commit,
    /// Frame 0 again with its mask after the sensory memory is cleared: the propagation's start.
    Restart,
    /// Segmented from memory, not memorised.
    Propagate,
    /// Segmented and memorised.
    Memorise,
    /// Segmented and memorised, working memory consolidating into long-term memory.
    Consolidate,
}

impl Kind {
    const ALL: [Kind; 5] = [Kind::Commit, Kind::Restart, Kind::Propagate, Kind::Memorise, Kind::Consolidate];

    fn name(self) -> &'static str {
        match self {
            Kind::Commit => "commit",
            Kind::Restart => "restart",
            Kind::Propagate => "propagate",
            Kind::Memorise => "memorise",
            Kind::Consolidate => "consolidate",
        }
    }
}

/// One part of a step timed in parts: from the end of the last part to the host having enqueued
/// this one, and to the GPU having finished it.
#[derive(Clone, Copy, Debug)]
struct Lap {
    name: &'static str,
    enqueue_ms: f64,
    ms: f64,
}

/// The parts of a step as they finish (`lap` synchronises); does nothing when off.
struct Laps {
    on: bool,
    mark: Instant,
    parts: Vec<Lap>,
}

impl Laps {
    fn new(on: bool) -> Self {
        Self { on, mark: Instant::now(), parts: Vec::new() }
    }

    fn lap(&mut self, name: &'static str) -> Result<()> {
        if !self.on {
            return Ok(());
        }
        let enqueue = self.mark.elapsed();
        gpu_synchronize().map_err(|e| dev("bench sync", e))?;
        let finished = self.mark.elapsed();
        self.parts.push(Lap { name, enqueue_ms: enqueue.as_secs_f64() * 1e3, ms: finished.as_secs_f64() * 1e3 });
        self.mark = Instant::now();
        Ok(())
    }
}

/// A step of a pass.
struct Step {
    kind: Kind,
    ms: f64,
    /// The probabilities' bits hashed, to compare passes.
    digest: u64,
    parts: Vec<Lap>,
    /// On macOS, the Metal stream's command buffers committed and waits inside the step (the
    /// synchronisations around it left out; in parts, the laps' own counted too).
    stream: Option<(u64, u64)>,
}

/// The device's memory in use, sampled after load and after every step.
struct MemoryWatch {
    /// The whole device's use before loading (bytes).
    baseline: u64,
    peak: u64,
}

impl MemoryWatch {
    fn used() -> u64 {
        let s = gpu_perf_stats(false);
        s.mem_total_bytes.saturating_sub(s.mem_free_bytes)
    }

    fn new() -> Self {
        let baseline = Self::used();
        Self { baseline, peak: baseline }
    }

    fn sample(&mut self) {
        self.peak = self.peak.max(Self::used());
    }
}

/// The median, 90th percentile (nearest rank), least and largest of some times.
#[derive(Clone, Copy, Debug)]
struct Stats {
    n: usize,
    median: f64,
    p90: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let median = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
        let p90 = v[((n as f64 * 0.9).ceil() as usize).clamp(1, n) - 1];
        Some(Self { n, median, p90, min: v[0], max: v[n - 1] })
    }

    fn line(&self) -> String {
        format!("n={:<4} median {:>8.2} ms  p90 {:>8.2}  min {:>8.2}  max {:>8.2}", self.n, self.median, self.p90, self.min, self.max)
    }

    fn json(&self) -> String {
        format!("{{\"n\":{},\"median\":{:.3},\"p90\":{:.3},\"min\":{:.3},\"max\":{:.3}}}", self.n, self.median, self.p90, self.min, self.max)
    }
}

/// FNV-1a over the values' bits.
fn digest(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Stage `bench` (see the module doc): prints its findings; errors only when a step fails.
pub(crate) fn run(oracle: &Oracle, weights_path: &Path, report: &mut Report) -> Result<()> {
    let passes = std::env::var("MAKEPAD_CUTIE_BENCH_PASSES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5).max(2);
    let (w, h, n) = (oracle.width, oracle.height, oracle.objects);
    let frames_path = oracle.dir.join("raw").join("frames.u8");
    let frames = std::fs::read(&frames_path).map_err(|e| DiffusionError::model(format!("{}: {e}", frames_path.display())))?;
    if frames.len() != oracle.frames * w * h * 3 {
        return Err(DiffusionError::model("cutie bench: raw/frames.u8 has the wrong size"));
    }
    let masks = WorkerMasks::new(oracle)?;
    let frame = Frame::new(w, h, oracle.config.max_internal_size);
    let (pw, ph) = frame.padded();
    let name = oracle.dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    println!(
        "BENCH {name}: {w}x{h} ({}x{} inside, padded {pw}x{ph}, {} tokens a frame), {n} objects, {} frames, {} steps; {passes} passes (the first cold), then {PART_PASSES} in parts",
        frame.w,
        frame.h,
        (pw / 16) * (ph / 16),
        oracle.frames,
        oracle.schedule().len()
    );

    let mut watch = MemoryWatch::new();
    let started = Instant::now();
    let mut weights = Weights::load(weights_path)?;
    let read_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let net = CutieNet::load(&mut weights)?;
    gpu_synchronize().map_err(|e| dev("bench sync", e))?;
    let build_ms = started.elapsed().as_secs_f64() * 1e3;
    drop(weights);
    watch.sample();
    println!("BENCH load: checkpoint read {read_ms:.1} ms, network built (linears uploaded) {build_ms:.1} ms");

    let mut runs = Vec::with_capacity(passes);
    for _ in 0..passes {
        runs.push(replay(oracle, &net, &frames, &masks, false, &mut watch)?);
    }
    let stats = gpu_perf_stats(false);
    let conv_weights_mb = stats.weight_stream_bytes as f64 / MB;
    let pool_mb = stats.pool_fresh_alloc_bytes.saturating_sub(stats.pool_overcap_free_bytes) as f64 / MB;
    let mut part_runs = Vec::with_capacity(PART_PASSES);
    for _ in 0..PART_PASSES {
        part_runs.push(replay(oracle, &net, &frames, &masks, true, &mut watch)?);
    }

    // The cold pass.
    let cold = &runs[0];
    let first = |kind: Kind| cold.iter().find(|s| s.kind == kind).map_or(f64::NAN, |s| s.ms);
    let first_propagated = cold.iter().find(|s| !matches!(s.kind, Kind::Commit | Kind::Restart)).map_or(f64::NAN, |s| s.ms);
    let cold_run: f64 = cold.iter().map(|s| s.ms).sum();
    println!(
        "BENCH cold pass: commit {:.1} ms, restart {:.1} ms, first frame propagated {first_propagated:.1} ms, whole run {cold_run:.1} ms",
        first(Kind::Commit),
        first(Kind::Restart)
    );

    // The warm passes, by kind of step.
    let warm = &runs[1..];
    let run_ms: Vec<f64> = warm.iter().map(|r| r.iter().map(|s| s.ms).sum()).collect();
    let run_stats = Stats::of(&run_ms).expect("at least one warm pass");
    let first_warm: Vec<f64> =
        warm.iter().filter_map(|r| r.iter().find(|s| !matches!(s.kind, Kind::Commit | Kind::Restart)).map(|s| s.ms)).collect();
    let segmented = |r: &Vec<Step>| r.iter().filter(|s| !matches!(s.kind, Kind::Commit | Kind::Restart)).map(|s| s.ms).sum::<f64>();
    let frames_propagated = cold.iter().filter(|s| !matches!(s.kind, Kind::Commit | Kind::Restart)).count();
    let fps = frames_propagated as f64 / (warm.iter().map(segmented).sum::<f64>() / warm.len() as f64 / 1e3);
    println!(
        "BENCH warm passes ({}): whole run median {:.1} ms (min {:.1}, max {:.1}); {frames_propagated} frames propagated at {fps:.1} a second; first frame propagated median {:.2} ms",
        warm.len(),
        run_stats.median,
        run_stats.min,
        run_stats.max,
        Stats::of(&first_warm).map_or(f64::NAN, |s| s.median)
    );
    let mut kinds_json = Vec::new();
    for kind in Kind::ALL {
        let times: Vec<f64> = warm.iter().flatten().filter(|s| s.kind == kind).map(|s| s.ms).collect();
        if let Some(s) = Stats::of(&times) {
            println!("BENCH   {:<12} {}", kind.name(), s.line());
            kinds_json.push(format!("\"{}\":{}", kind.name(), s.json()));
        }
    }
    // macOS: how often a warm step commits and waits on the Metal stream (Cutie's goal is one
    // wait a step, its probabilities' download; the in-flight cap and host reads add to it).
    let counted: Vec<(u64, u64)> = warm.iter().flatten().filter_map(|s| s.stream).collect();
    let stream_json = match (
        Stats::of(&counted.iter().map(|c| c.1 as f64).collect::<Vec<_>>()),
        Stats::of(&counted.iter().map(|c| c.0 as f64).collect::<Vec<_>>()),
    ) {
        (Some(waits), Some(commits)) => {
            println!(
                "BENCH   Metal stream a step: waits median {:.0} (min {:.0}, max {:.0}), command buffers committed median {:.0} (min {:.0}, max {:.0})",
                waits.median, waits.min, waits.max, commits.median, commits.min, commits.max
            );
            format!(",\"stream\":{{\"waits\":{},\"commits\":{}}}", waits.json(), commits.json())
        }
        _ => String::new(),
    };

    let device_mb = watch.peak.saturating_sub(watch.baseline) as f64 / MB;
    println!(
        "BENCH memory: convolution weights {conv_weights_mb:.1} MB; activation pool {pool_mb:.1} MB allocated less freed (linears, memory, activations; buffers kept for reuse); whole device +{device_mb:.0} MB at peak over the baseline (other processes move it)"
    );

    // The passes in parts.
    let same_plain = runs.iter().all(|r| r.iter().zip(&runs[0]).all(|(a, b)| a.digest == b.digest));
    let same_parts = part_runs.iter().all(|r| r.iter().zip(&runs[0]).all(|(a, b)| a.digest == b.digest));
    let mut parts_json = Vec::new();
    for kind in [Kind::Propagate, Kind::Memorise] {
        let steps: Vec<&Step> = part_runs.iter().flatten().filter(|s| s.kind == kind).collect();
        if steps.is_empty() {
            continue;
        }
        let plain: Vec<f64> = warm.iter().flatten().filter(|s| s.kind == kind).map(|s| s.ms).collect();
        let whole = Stats::of(&steps.iter().map(|s| s.ms).collect::<Vec<_>>()).map_or(f64::NAN, |s| s.median);
        println!(
            "BENCH parts, frames that {} (n={}, each part synchronised; median ms to finish, median ms the host took to enqueue it):",
            if kind == Kind::Propagate { "do not memorise" } else { "memorise without consolidating" },
            steps.len()
        );
        let mut names: Vec<&'static str> = Vec::new();
        for s in &steps {
            for lap in &s.parts {
                if !names.contains(&lap.name) {
                    names.push(lap.name);
                }
            }
        }
        let mut sum = 0.0;
        let mut json = Vec::new();
        for part in names {
            let times: Vec<f64> = steps.iter().flat_map(|s| s.parts.iter().filter(|l| l.name == part).map(|l| l.ms)).collect();
            let enqueue: Vec<f64> = steps.iter().flat_map(|s| s.parts.iter().filter(|l| l.name == part).map(|l| l.enqueue_ms)).collect();
            let (Some(t), Some(e)) = (Stats::of(&times), Stats::of(&enqueue)) else { continue };
            sum += t.median;
            println!("BENCH   {part:<16} {:>8.3} ms   enqueue {:>8.3} ms   (p90 {:.3})", t.median, e.median, t.p90);
            json.push(format!("\"{part}\":{{\"ms\":{:.4},\"enqueue_ms\":{:.4}}}", t.median, e.median));
        }
        let plain_median = Stats::of(&plain).map_or(f64::NAN, |s| s.median);
        println!("BENCH   parts' medians sum to {sum:.2} ms; the step in parts {whole:.2} ms; the plain step {plain_median:.2} ms");
        parts_json.push(format!("\"{}\":{{{}}}", kind.name(), json.join(",")));
    }
    // Every step's probabilities hashed together: equal across builds when a change keeps the
    // results to the bit.
    let run_digest = runs[0].iter().fold(0xcbf2_9ce4_8422_2325u64, |h, s| (h ^ s.digest).wrapping_mul(0x0100_0000_01b3));
    println!(
        "BENCH outputs: every plain pass's probabilities identical to the first's: {}; the passes in parts identical too: {}; the run's digest {run_digest:016x}",
        if same_plain { "yes" } else { "NO" },
        if same_parts { "yes" } else { "NO" }
    );
    if !same_plain || !same_parts {
        report.note("bench: passes gave different probabilities (see the BENCH outputs line)");
    }
    // The convolutions alone, on frame 1.
    let at = (1 % oracle.frames) * w * h * 3;
    let image = Feat::new(frame.to_internal(&frame.image(&frames[at..at + w * h * 3])?)?, 1, 3, pw, ph)?;
    let convs = conv_profile(&net, &image, n)?;
    println!("BENCH convolutions alone at {pw}x{ph} (median of 5, synchronised): GFLOP; ms (TFLOPS) as run; ms with the add and ReLU apart; ms (TFLOPS) of the bare GEMM");
    let mut convs_json = Vec::new();
    for part in ["encoder", "decoder", "mask encoder"] {
        let mut sums = (0.0, 0.0, 0.0, 0.0);
        for c in convs.iter().filter(|c| c.part == part) {
            println!(
                "BENCH   {:<44} {:>6.2} GF  run {:>7.3} ms ({:>5.1})  apart {:>7.3}  GEMM {:>7.3} ({:>5.1})",
                c.name,
                c.gflop,
                c.fused_ms,
                c.gflop / c.fused_ms,
                c.apart_ms,
                c.gemm_ms,
                c.gflop / c.gemm_ms
            );
            sums = (sums.0 + c.gflop, sums.1 + c.fused_ms, sums.2 + c.apart_ms, sums.3 + c.gemm_ms);
        }
        println!(
            "BENCH   {part} total: {:.1} GFLOP, run {:.2} ms ({:.1} TFLOPS), apart {:.2} ms, bare GEMMs {:.2} ms ({:.1} TFLOPS): im2col and the epilogue {:.2} ms",
            sums.0,
            sums.1,
            sums.0 / sums.1,
            sums.2,
            sums.3,
            sums.0 / sums.3,
            sums.1 - sums.3
        );
        convs_json.push(format!(
            "\"{}\":{{\"gflop\":{:.2},\"run_ms\":{:.3},\"apart_ms\":{:.3},\"gemm_ms\":{:.3}}}",
            part.replace(' ', "_"),
            sums.0,
            sums.1,
            sums.2,
            sums.3
        ));
    }

    println!(
        "CUTIE_BENCH {{\"oracle\":\"{name}\",\"width\":{w},\"height\":{h},\"objects\":{n},\"frames\":{},\"passes\":{passes},\"load_read_ms\":{read_ms:.1},\"load_build_ms\":{build_ms:.1},\"cold\":{{\"commit\":{:.2},\"restart\":{:.2},\"first_propagated\":{first_propagated:.2},\"run\":{cold_run:.1}}},\"warm_run\":{},\"fps\":{fps:.2},\"steps\":{{{}}}{stream_json},\"memory_mb\":{{\"conv_weights\":{conv_weights_mb:.1},\"pool\":{pool_mb:.1},\"device_peak\":{device_mb:.0}}},\"parts\":{{{}}},\"deterministic\":{},\"digest\":\"{run_digest:016x}\",\"convs\":{{{}}}}}",
        oracle.frames,
        first(Kind::Commit),
        first(Kind::Restart),
        run_stats.json(),
        kinds_json.join(","),
        parts_json.join(","),
        same_plain && same_parts,
        convs_json.join(",")
    );
    Ok(())
}

/// One convolution timed alone (`conv_profile`).
struct ConvTime {
    /// "encoder" or "decoder".
    part: &'static str,
    name: String,
    gflop: f64,
    /// As the network runs it: the bias, residual and ReLU fused.
    fused_ms: f64,
    /// The convolution with its bias, then the residual's add and the ReLU as passes of their own.
    apart_ms: f64,
    /// cuBLAS SGEMM of the same M, N and K alone, on constant columns.
    gemm_ms: f64,
}

/// The median of five runs of `f`, each synchronised before and after, after one untimed.
fn time_median<T>(mut f: impl FnMut() -> Result<T>) -> Result<f64> {
    f()?;
    let mut times = Vec::with_capacity(5);
    for _ in 0..5 {
        gpu_synchronize().map_err(|e| dev("bench sync", e))?;
        let started = Instant::now();
        f()?;
        gpu_synchronize().map_err(|e| dev("bench sync", e))?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    Ok(Stats::of(&times).map_or(f64::NAN, |s| s.median))
}

/// Times `conv` on `x` (with `residual` and `relu_after` as the network uses it) three ways
/// (`ConvTime`), and gives what the network would get.
#[allow(clippy::too_many_arguments)]
fn profile_conv(
    times: &mut Vec<ConvTime>, part: &'static str, name: String, conv: &Conv2d, x: &Feat, residual: Option<&Feat>, relu_after: bool,
) -> Result<Feat> {
    let fused_ms = time_median(|| conv.forward_fused(x, residual, relu_after))?;
    let apart_ms = time_median(|| {
        let mut y = conv.forward(x)?;
        if let Some(r) = residual {
            y = add(&y, r)?;
        }
        if relu_after {
            y = relu(&y)?;
        }
        Ok(y)
    })?;
    let kdim = conv.in_c * conv.k * conv.k;
    let m = x.n * cutie_conv_out(x.w, conv.k, conv.pad, conv.stride) * cutie_conv_out(x.h, conv.k, conv.pad, conv.stride);
    let weights = gpu_upload(&conv.weights, conv.out_c, kdim).map_err(|e| dev("bench gemm", e))?;
    let columns = gpu_rife_fill(kdim, m, 0.5).map_err(|e| dev("bench gemm", e))?;
    let gemm_ms = time_median(|| gpu_cutie_matmul(&weights, false, &columns, false).map_err(|e| dev("bench gemm", e)))?;
    let gflop = 2.0 * (conv.out_c * kdim * m) as f64 / 1e9;
    let label = format!("{name} {}->{} {}x{}s{} n{} {}x{}", conv.in_c, conv.out_c, conv.k, conv.k, conv.stride, x.n, x.w, x.h);
    times.push(ConvTime { part, name: label, gflop, fused_ms, apart_ms, gemm_ms });
    conv.forward_fused(x, residual, relu_after)
}

/// Every convolution of the pixel encoder on `image` (one padded frame in 0..1), then of the
/// decoder for `objects` objects on its features (the memory readout and sensory memory
/// constant), then of the mask encoder (its inputs constant), each timed alone (`profile_conv`).
fn conv_profile(net: &CutieNet, image: &Feat, objects: usize) -> Result<Vec<ConvTime>> {
    let mut times = Vec::new();
    let pe = &net.pixel_encoder;
    let x = pe.normalise(image)?;
    let stem = profile_conv(&mut times, "encoder", "stem (relu apart in the network)".into(), &pe.conv1, &x, None, true)?;
    let (w, h) = (cutie_conv_out(stem.w, 3, 1, 2), cutie_conv_out(stem.h, 3, 1, 2));
    let mut x = Feat::new(gpu_cutie_maxpool3x3s2(&stem.t, stem.w, stem.h).map_err(|e| dev("bench maxpool", e))?, 1, stem.c, w, h)?;
    let (mut f4, mut f8) = (None, None);
    for (stage, layer) in [("res2", &pe.res2), ("layer2", &pe.layer2), ("layer3", &pe.layer3)] {
        for (i, block) in layer.blocks.iter().enumerate() {
            let a = profile_conv(&mut times, "encoder", format!("{stage}.{i}.conv1"), &block.conv1, &x, None, true)?;
            let b = profile_conv(&mut times, "encoder", format!("{stage}.{i}.conv2"), &block.conv2, &a, None, true)?;
            let residual = match &block.downsample {
                Some(d) => profile_conv(&mut times, "encoder", format!("{stage}.{i}.downsample"), d, &x, None, false)?,
                None => x.view()?,
            };
            x = profile_conv(&mut times, "encoder", format!("{stage}.{i}.conv3"), &block.conv3, &b, Some(&residual), true)?;
        }
        match stage {
            "res2" => f4 = Some(x.view()?),
            "layer2" => f8 = Some(x.view()?),
            _ => {}
        }
    }
    profile_conv(&mut times, "encoder", "pix_feat_proj".into(), &pe.pix_feat_proj, &x, None, false)?;

    let missing = |what: &str| DiffusionError::model(format!("cutie bench: no {what}"));
    let (f4, f8) = (f4.ok_or_else(|| missing("f4"))?, f8.ok_or_else(|| missing("f8"))?);
    let constant = |n: usize, c: usize, w: usize, h: usize| -> Result<Feat> {
        Feat::new(gpu_rife_fill(n * c, w * h, 0.1).map_err(|e| dev("bench", e))?, n, c, w, h)
    };
    let d = &net.mask_decoder;
    profile_conv(&mut times, "decoder", "feat_proc8".into(), &d.feat_proc8, &f8, None, false)?;
    profile_conv(&mut times, "decoder", "feat_proc4".into(), &d.feat_proc4, &f4, None, false)?;
    let (w16, h16) = (x.w, x.h);
    // up_16_8's block on the objects' features at stride 8, up_8_4's at stride 4.
    for (name, block, c, w, h) in [("up_16_8", &d.up_16_8.out_conv, 256, f8.w, f8.h), ("up_8_4", &d.up_8_4.out_conv, 128, f4.w, f4.h)] {
        let g = constant(objects, c, w, h)?;
        let a = profile_conv(&mut times, "decoder", format!("{name}.conv1"), &block.conv1, &relu(&g)?, None, true)?;
        let residual = match &block.downsample {
            Some(ds) => profile_conv(&mut times, "decoder", format!("{name}.downsample"), ds, &g, None, false)?,
            None => g.view()?,
        };
        profile_conv(&mut times, "decoder", format!("{name}.conv2"), &block.conv2, &a, Some(&residual), false)?;
    }
    profile_conv(&mut times, "decoder", "pred".into(), &d.pred, &constant(objects, 128, f4.w, f4.h)?, None, false)?;
    let s = &d.sensory_update;
    profile_conv(&mut times, "decoder", "sensory g16".into(), &s.g16_conv, &constant(objects, 256, w16, h16)?, None, false)?;
    profile_conv(&mut times, "decoder", "sensory g8".into(), &s.g8_conv, &constant(objects, 128, w16, h16)?, None, false)?;
    profile_conv(&mut times, "decoder", "sensory g4".into(), &s.g4_conv, &constant(objects, 129, w16, h16)?, None, false)?;
    profile_conv(&mut times, "decoder", "sensory transform".into(), &s.transform, &constant(objects, 512, w16, h16)?, None, false)?;

    // The mask encoder (a memorised frame), every object: [image, mask, others] at the padded
    // size through ResNet-18 to stride 16, the fuser, the deep update.
    let m = &net.mask_encoder;
    let stem = profile_conv(&mut times, "mask encoder", "stem".into(), &m.conv1, &constant(objects, 5, image.w, image.h)?, None, false)?;
    let (w, h) = (cutie_conv_out(stem.w, 3, 1, 2), cutie_conv_out(stem.h, 3, 1, 2));
    let pooled = gpu_cutie_maxpool3x3s2(&stem.t, stem.w, stem.h).map_err(|e| dev("bench maxpool", e))?;
    let mut g = relu(&Feat::new(pooled, objects, stem.c, w, h)?)?;
    for (stage, layer) in m.layers.iter().enumerate() {
        for (i, block) in layer.blocks.iter().enumerate() {
            let a = profile_conv(&mut times, "mask encoder", format!("layer{}.{i}.conv1", stage + 1), &block.conv1, &g, None, true)?;
            let residual = match &block.downsample {
                Some(d) => profile_conv(&mut times, "mask encoder", format!("layer{}.{i}.downsample", stage + 1), d, &g, None, false)?,
                None => g.view()?,
            };
            g = profile_conv(&mut times, "mask encoder", format!("layer{}.{i}.conv2", stage + 1), &block.conv2, &a, Some(&residual), true)?;
        }
    }
    let f = &m.fuser;
    profile_conv(&mut times, "mask encoder", "fuser x_transform".into(), &f.x_transform, &constant(1, 256, w16, h16)?, None, false)?;
    profile_conv(&mut times, "mask encoder", "fuser g_transform".into(), &f.g_transform, &g, None, false)?;
    for (b, block) in [("block1", &f.block1), ("block2", &f.block2)] {
        let x = constant(objects, 256, w16, h16)?;
        let a = profile_conv(&mut times, "mask encoder", format!("fuser {b}.conv1"), &block.conv1, &relu(&x)?, None, true)?;
        profile_conv(&mut times, "mask encoder", format!("fuser {b}.conv2"), &block.conv2, &a, None, false)?;
    }
    profile_conv(&mut times, "mask encoder", "sensory transform".into(), &m.sensory_transform, &constant(objects, 512, w16, h16)?, None, false)?;
    Ok(times)
}

/// The oracle's run once, as stage `run` drives it (without taps or comparisons), with a fresh
/// processor: each step timed, synchronised before and after; in parts when `in_parts`.
fn replay(oracle: &Oracle, net: &CutieNet, frames: &[u8], masks: &WorkerMasks, in_parts: bool, watch: &mut MemoryWatch) -> Result<Vec<Step>> {
    let (w, h, n) = (oracle.width, oracle.height, oracle.objects);
    let (plane, size) = (w * h, w * h * 3);
    let mut processor = CutieProcessor::new(net, oracle.config.clone(), n);
    // Each frame's latest label map, frame 0's the seed's to begin with (the worker's masks).
    let mut frame_labels: Vec<Option<Vec<u8>>> = vec![None; oracle.frames];
    frame_labels[0] = Some(masks.seed.clone());
    let mut steps = Vec::with_capacity(oracle.schedule().len());
    for row in oracle.schedule() {
        processor.apply_clears(&row.clears())?;
        let rgb = &frames[row.frame * size..(row.frame + 1) * size];
        let mask = match row.with_mask {
            true => Some(masks.one_hot(
                frame_labels[row.frame].as_ref().ok_or_else(|| DiffusionError::model(format!("cutie bench: no mask for frame {}", row.frame)))?,
            )),
            false => None,
        };
        let consolidations = processor.memory.consolidations;
        gpu_synchronize().map_err(|e| dev("bench sync", e))?;
        let mut laps = Laps::new(in_parts);
        let counts = gpu_stream_counts();
        let started = Instant::now();
        let prob = if in_parts {
            step_in_parts(&mut processor, rgb, w, h, mask.as_deref(), row.force_permanent, &mut laps)?
        } else {
            processor.step(rgb, w, h, mask.as_deref(), row.force_permanent, false)?
        };
        let stream = gpu_stream_counts()
            .zip(counts)
            .map(|(after, before)| (after.0.saturating_sub(before.0), after.1.saturating_sub(before.1)));
        gpu_synchronize().map_err(|e| dev("bench sync", e))?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        watch.sample();
        let memorised = processor.schedule.last_mem_ti == processor.schedule.curr_ti;
        let kind = match (row.force_permanent, row.with_mask, memorised, processor.memory.consolidations != consolidations) {
            (true, ..) => Kind::Commit,
            (_, true, ..) => Kind::Restart,
            (_, _, _, true) => Kind::Consolidate,
            (_, _, true, _) => Kind::Memorise,
            _ => Kind::Propagate,
        };
        frame_labels[row.frame] = Some(CutieProcessor::labels(&prob, n + 1, plane));
        steps.push(Step { kind, ms, digest: digest(&prob), parts: laps.parts, stream });
    }
    Ok(steps)
}

/// `CutieProcessor::step_tapped` without taps (the end flag off), a lap after each part; the
/// memory read (`MemoryManager::read`) taken apart into the similarity, the top-k softmax with
/// the usage counted, the readout, pixel fusion and the object transformer. Must give what
/// `step` gives, to the bit (`run` checks).
fn step_in_parts(
    p: &mut CutieProcessor, rgb: &[u8], width: usize, height: usize, mask: Option<&[f32]>, force_permanent: bool, laps: &mut Laps,
) -> Result<Vec<f32>> {
    let n = p.objects;
    let net = p.net;
    let mut no_taps = Tap::none();
    let tap = &mut no_taps;
    let frame = Frame::new(width, height, p.config.max_internal_size);
    let image = frame.image(rgb)?;
    let mask = mask.map(|m| gpu_upload(m, n, width * height)).transpose().map_err(|e| dev("step mask", e))?;
    let plan = p.schedule.next(mask.is_some(), false);
    let (pw, ph) = frame.padded();
    let image = Feat::new(frame.to_internal(&image)?, 1, 3, pw, ph)?;
    laps.lap("frame in")?;
    let feats = net.encode_image(&image, tap)?;
    laps.lap("encode_image")?;
    let keys = net.transform_key(&feats.f16, tap)?;
    laps.lap("transform_key")?;

    let prob = match &mask {
        None => {
            if !p.memory.engaged() {
                return Err(DiffusionError::model("cutie step: segmenting without any memory"));
            }
            let missing = |what: &str| DiffusionError::model(format!("cutie bench step: no {what}"));
            let last_mask = p.last_mask.as_ref().ok_or_else(|| missing("last mask"))?;
            let config = &p.config;
            let memory = &mut p.memory;
            let long_size = if config.use_long_term { memory.long.size() } else { 0 };
            let (long_key, long_shrinkage) = if long_size > 0 { (memory.long.key.as_ref(), memory.long.shrinkage.as_ref()) } else { (None, None) };
            let memory_key = KeyValueStore::cat(&[long_key, memory.work.key.as_ref()])?.ok_or_else(|| missing("memory key"))?;
            let memory_shrinkage = KeyValueStore::cat(&[long_shrinkage, memory.work.shrinkage.as_ref()])?.ok_or_else(|| missing("shrinkage"))?;
            let similarity = gpu_cutie_similarity(&memory_key, Some(&memory_shrinkage), &keys.key.t, Some(&keys.selection.t))
                .map_err(|e| dev("memory similarity", e))?;
            laps.lap("similarity")?;
            let (affinity, usage) = gpu_cutie_topk_softmax(&similarity, config.top_k).map_err(|e| dev("memory top-k", e))?;
            if config.use_long_term {
                memory.work.update_usage(&usage, long_size)?;
                if long_size > 0 && config.count_long_term_usage {
                    memory.long.update_usage(&usage, 0)?;
                }
            }
            laps.lap("top-k softmax")?;
            let visual_readout = memory.read_values(&affinity, &keys, tap)?;
            laps.lap("readout")?;
            let sensory = memory.sensory()?;
            let pixel = net.pixel_fusion(&feats.pix_feat, &visual_readout, &sensory, last_mask, tap)?;
            laps.lap("pixel fusion")?;
            let obj_memory = memory.obj_memory().ok_or_else(|| missing("object memory"))?;
            let (readout, _aux_logits) = net.readout_query(&pixel, obj_memory, tap)?;
            laps.lap("transformer")?;
            let (sensory, _logits, prob) = net.segment(&feats, &readout, &sensory, plan.update_sensory, tap)?;
            if plan.update_sensory {
                memory.update_sensory(sensory)?;
            }
            laps.lap("segment")?;
            prob
        }
        Some(mask) => {
            let prob = CutieProcessor::mask_prob(&frame.to_internal(mask)?)?;
            laps.lap("mask in")?;
            prob
        }
    };
    if prob.rows() != n + 1 {
        return Err(DiffusionError::model(format!("cutie step: {} probabilities for {n} objects", prob.rows())));
    }
    p.last_mask = Some(CutieProcessor::last_mask(&prob, pw, ph)?);

    if plan.memorise || force_permanent {
        let as_permanent = if force_permanent { Permanence::All } else { Permanence::First };
        p.memory.initialize_sensory_if_needed(&keys, n)?;
        let last_mask = p.last_mask.as_ref().ok_or_else(|| DiffusionError::model("cutie step: no mask to memorise"))?;
        let (value, sensory, obj_value) = net.encode_mask(&feats.image, &feats.pix_feat, &p.memory.sensory()?, last_mask, true, tap)?;
        laps.lap("encode_mask")?;
        p.memory.add_memory(&keys, &value, &obj_value, as_permanent, tap)?;
        p.schedule.memorised();
        p.memory.update_sensory(sensory)?;
        laps.lap("add_memory")?;
    }

    let out = frame.to_given(&prob)?;
    let values = gpu_download(&out).map_err(|e| dev("step out", e))?;
    laps.lap("frame out")?;
    Ok(values)
}
