//! Reading Cutie's oracle and comparing against it. The oracle is RationalVisions'
//! `tools/cutie-oracle` (PyTorch, fp32, TF32 off): a folder per clip with
//!
//! - `taps/<step>/manifest.tsv` (name, dtype, dims a line) and `taps/<step>/<name>.f32` (little-
//!   endian f32 in PyTorch's memory order) for the steps `commit_f0`, `prop_f0`, `prop_f1`,
//!   `prop_f5`, `prop_f40` (working memory consolidates) and `prop_f41` (the first read through
//!   long-term memory);
//! - `raw/meta.txt` (frames, width, height, objects), `raw/frames.u8` (frames x h x w x 3 RGB),
//!   `raw/seed.u8` (objects x h x w, 0/1), `raw/labels.u8` (frames x h x w, the masks out as
//!   label maps), `raw/schedule.tsv` (each processor step: frame, memorised, consolidated, the
//!   memory's sizes).
//!
//! Names: `net.<module>` is a module's output by its name in `CUTIE`; `encode_image.*`, `key.*`,
//! `encode_mask.*`, `pixel_fusion.*`, `object_transformer.*`, `segment.*` the network methods'
//! inputs and outputs; `memory.read.*` and `memory.consolidate.*` the memory's similarity,
//! affinity, usage and readouts; `state.before.*` / `state.after.*` the memory between steps;
//! `step.*` the processor step's input and output. A name seen twice in a step has `#1` added.

use crate::model::Tap;
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{gpu_download, GpuTensor};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The tapped steps, in order.
pub(crate) const STEPS: [&str; 6] = ["commit_f0", "prop_f0", "prop_f1", "prop_f5", "prop_f40", "prop_f41"];

pub(crate) struct Oracle {
    pub(crate) dir: PathBuf,
    pub(crate) frames: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) objects: usize,
    manifests: HashMap<String, Vec<(String, String, Vec<usize>)>>,
}

/// A tap's shape and values.
pub(crate) struct OracleTap {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
    pub(crate) values: Vec<f32>,
}

/// A row of `raw/schedule.tsv`.
#[derive(Clone, Debug)]
pub(crate) struct ScheduleRow {
    pub(crate) step: usize,
    pub(crate) tap: Option<String>,
    pub(crate) frame: usize,
    pub(crate) ti: i64,
    pub(crate) with_mask: bool,
    pub(crate) force_permanent: bool,
    pub(crate) memorised: bool,
    pub(crate) consolidated: bool,
    pub(crate) work_tokens: usize,
    pub(crate) work_permanent: usize,
    pub(crate) long_tokens: usize,
    pub(crate) mask_area: Vec<usize>,
}

fn io(path: &Path, e: impl std::fmt::Display) -> DiffusionError {
    DiffusionError::model(format!("cutie oracle {}: {e}", path.display()))
}

impl Oracle {
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let meta_path = dir.join("raw").join("meta.txt");
        let meta = fs::read_to_string(&meta_path).map_err(|e| io(&meta_path, e))?;
        let get = |key: &str| -> Result<usize> {
            meta.lines()
                .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
                .and_then(|v| v.trim().parse().ok())
                .ok_or_else(|| io(&meta_path, format!("no {key}")))
        };
        let mut manifests = HashMap::new();
        for step in STEPS {
            let path = dir.join("taps").join(step).join("manifest.tsv");
            let text = fs::read_to_string(&path).map_err(|e| io(&path, e))?;
            let rows = text
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| {
                    let mut parts = l.split('\t');
                    let name = parts.next().unwrap_or("").to_string();
                    let dtype = parts.next().unwrap_or("").to_string();
                    let dims = parts.next().unwrap_or("").split(',').filter(|d| !d.is_empty()).filter_map(|d| d.parse().ok()).collect();
                    (name, dtype, dims)
                })
                .collect();
            manifests.insert(step.to_string(), rows);
        }
        Ok(Self { dir: dir.to_path_buf(), frames: get("frames")?, width: get("width")?, height: get("height")?, objects: get("objects")?, manifests })
    }

    /// The names tapped in `step`, in the order they were tapped.
    pub(crate) fn names(&self, step: &str) -> Vec<&str> {
        self.manifests.get(step).map(|m| m.iter().map(|(n, _, _)| n.as_str()).collect()).unwrap_or_default()
    }

    pub(crate) fn has(&self, step: &str, name: &str) -> bool {
        self.manifests.get(step).is_some_and(|m| m.iter().any(|(n, _, _)| n == name))
    }

    pub(crate) fn tap(&self, step: &str, name: &str) -> Result<OracleTap> {
        let (_, dtype, shape) = self
            .manifests
            .get(step)
            .and_then(|m| m.iter().find(|(n, _, _)| n == name))
            .ok_or_else(|| DiffusionError::model(format!("cutie oracle: no tap {step}/{name}")))?;
        let path = self.dir.join("taps").join(step).join(format!("{name}.f32"));
        let bytes = fs::read(&path).map_err(|e| io(&path, e))?;
        let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if values.len() != shape.iter().product::<usize>() {
            return Err(io(&path, format!("{} values for {shape:?}", values.len())));
        }
        Ok(OracleTap { shape: shape.clone(), dtype: dtype.clone(), values })
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        let path = self.dir.join("raw").join(name);
        fs::read(&path).map_err(|e| io(&path, e))
    }

    /// Frame `i` as RGB bytes, h x w x 3.
    pub(crate) fn frame_rgb(&self, i: usize) -> Result<Vec<u8>> {
        let size = self.width * self.height * 3;
        Ok(self.raw("frames.u8")?[i * size..(i + 1) * size].to_vec())
    }

    /// The seed masks, objects x h x w, 0 or 1.
    pub(crate) fn seed(&self) -> Result<Vec<u8>> {
        self.raw("seed.u8")
    }

    /// The oracle's masks out as label maps, frames x h x w (the object's number, or 0).
    pub(crate) fn labels(&self) -> Result<Vec<u8>> {
        self.raw("labels.u8")
    }

    pub(crate) fn schedule(&self) -> Result<Vec<ScheduleRow>> {
        let path = self.dir.join("raw").join("schedule.tsv");
        let text = fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        let mut rows = Vec::new();
        for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() < 12 {
                return Err(io(&path, format!("short line {line:?}")));
            }
            let num = |s: &str| s.parse::<i64>().map_err(|e| io(&path, e));
            rows.push(ScheduleRow {
                step: num(c[0])? as usize,
                tap: (c[1] != "-").then(|| c[1].to_string()),
                frame: num(c[2])? as usize,
                ti: num(c[3])?,
                with_mask: c[4] == "1",
                force_permanent: c[5] == "1",
                memorised: c[6] == "1",
                consolidated: c[7] == "1",
                work_tokens: num(c[8])? as usize,
                work_permanent: num(c[9])? as usize,
                long_tokens: num(c[10])? as usize,
                mask_area: c[11].split(',').filter_map(|v| v.parse().ok()).collect(),
            });
        }
        Ok(rows)
    }
}

// --- Comparing ---

/// How close a tap must be: the largest difference relative to the expected tensor's largest
/// magnitude, and the cosine similarity. Boolean taps (attention masks) are compared exactly,
/// `bool_mismatch` the fraction of entries allowed to differ.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tol {
    pub(crate) rel: f32,
    pub(crate) cos: f64,
    pub(crate) bool_mismatch: f64,
}

impl Tol {
    /// A stage fed the oracle's own inputs (teacher-forced): only fp32 rounding apart.
    pub(crate) const STAGE: Tol = Tol { rel: 1e-3, cos: 0.99999, bool_mismatch: 0.0 };
    /// After many steps of the native pipeline's own state: top-k picks near ties differently,
    /// and a swapped pick moves that pixel's readout and sensory memory locally by up to about a
    /// tenth of their range (YouTubeComp-15 at 960x540: 7%, cosine 0.9999997). The whole tensor
    /// must still agree (cosine), and the masks are judged by IoU.
    pub(crate) const RUN: Tol = Tol { rel: 0.15, cos: 0.9999, bool_mismatch: 0.01 };
    /// Reported, never failing.
    pub(crate) const INFO: Tol = Tol { rel: f32::INFINITY, cos: f64::NEG_INFINITY, bool_mismatch: 1.0 };
}

#[derive(Clone, Debug)]
pub(crate) struct Row {
    pub(crate) stage: String,
    pub(crate) step: String,
    pub(crate) name: String,
    pub(crate) n: usize,
    pub(crate) max_abs: f32,
    pub(crate) max_ref: f32,
    pub(crate) rel: f32,
    pub(crate) cos: f64,
    pub(crate) pass: bool,
    pub(crate) note: String,
}

/// Every comparison made, and anything else a check wants to say.
#[derive(Default)]
pub(crate) struct Report {
    pub(crate) rows: Vec<Row>,
    pub(crate) notes: Vec<String>,
    pub(crate) failures: usize,
}

impl Report {
    /// Compares `actual` with the oracle's `expected`, both in PyTorch's memory order.
    pub(crate) fn compare(&mut self, stage: &str, step: &str, name: &str, actual: &[f32], expected: &OracleTap, tol: Tol) -> bool {
        let mut row = Row {
            stage: stage.into(),
            step: step.into(),
            name: name.into(),
            n: expected.values.len(),
            max_abs: 0.0,
            max_ref: 0.0,
            rel: 0.0,
            cos: 1.0,
            pass: true,
            note: String::new(),
        };
        if actual.len() != expected.values.len() {
            row.pass = false;
            row.note = format!("{} values, oracle {} {:?}", actual.len(), expected.values.len(), expected.shape);
        } else if expected.dtype == "bool" {
            let differ = actual.iter().zip(&expected.values).filter(|(a, e)| (**a != 0.0) != (**e != 0.0)).count();
            let frac = differ as f64 / actual.len().max(1) as f64;
            row.cos = 1.0 - frac;
            row.pass = frac <= tol.bool_mismatch;
            row.note = format!("{differ} entries differ");
        } else {
            let (mut dot, mut na, mut ne) = (0.0f64, 0.0f64, 0.0f64);
            let mut nonfinite = 0usize;
            for (&a, &e) in actual.iter().zip(&expected.values) {
                if !a.is_finite() || !e.is_finite() {
                    if a.is_finite() != e.is_finite() || (a.is_nan() != e.is_nan()) {
                        nonfinite += 1;
                    }
                    continue;
                }
                row.max_abs = row.max_abs.max((a - e).abs());
                row.max_ref = row.max_ref.max(e.abs());
                dot += a as f64 * e as f64;
                na += a as f64 * a as f64;
                ne += e as f64 * e as f64;
            }
            row.rel = row.max_abs / row.max_ref.max(1e-12);
            row.cos = if na == 0.0 && ne == 0.0 { 1.0 } else { dot / (na.sqrt() * ne.sqrt()).max(1e-300) };
            let tiny = row.max_ref < 1e-6 && row.max_abs < 1e-6;
            row.pass = nonfinite == 0 && (tiny || (row.rel <= tol.rel && row.cos >= tol.cos));
            if nonfinite > 0 {
                row.note = format!("{nonfinite} non-finite entries differ");
            }
        }
        if !row.pass {
            self.failures += 1;
        }
        let pass = row.pass;
        self.rows.push(row);
        pass
    }

    pub(crate) fn note(&mut self, text: impl Into<String>) {
        self.notes.push(text.into());
    }

    pub(crate) fn fail(&mut self, stage: &str, step: &str, name: &str, why: impl Into<String>) {
        self.failures += 1;
        self.rows.push(Row {
            stage: stage.into(),
            step: step.into(),
            name: name.into(),
            n: 0,
            max_abs: f32::NAN,
            max_ref: f32::NAN,
            rel: f32::NAN,
            cos: f64::NAN,
            pass: false,
            note: why.into(),
        });
    }

    /// One line a comparison, then the notes.
    pub(crate) fn print(&self) {
        for r in &self.rows {
            println!(
                "{} {:<11} {:<9} {:<64} n={:<8} max_abs={:<11.4e} rel={:<11.4e} cos={:<12.9} {}",
                if r.pass { "PASS" } else { "FAIL" },
                r.stage,
                r.step,
                r.name,
                r.n,
                r.max_abs,
                r.rel,
                r.cos,
                r.note
            );
        }
        for n in &self.notes {
            println!("NOTE {n}");
        }
    }
}

/// Compares every tap put with the oracle's of the same name in `step` (names the oracle lacks
/// are noted, not failed). `tol` chooses the tolerance by name.
pub(crate) struct Comparer<'a> {
    pub(crate) oracle: &'a Oracle,
    pub(crate) report: &'a mut Report,
    pub(crate) stage: String,
    pub(crate) step: String,
    pub(crate) tol: &'a dyn Fn(&str) -> Tol,
    pub(crate) seen: HashMap<String, usize>,
}

impl<'a> Comparer<'a> {
    pub(crate) fn new(oracle: &'a Oracle, report: &'a mut Report, stage: &str, step: &str, tol: &'a dyn Fn(&str) -> Tol) -> Self {
        Self { oracle, report, stage: stage.into(), step: step.into(), tol, seen: HashMap::new() }
    }

    /// Takes a tap: the oracle's `#n` suffix is added for a name put again.
    pub(crate) fn take(&mut self, name: &str, t: &GpuTensor) -> Result<()> {
        let count = self.seen.entry(name.to_string()).or_insert(0);
        let name = if *count == 0 { name.to_string() } else { format!("{name}#{count}") };
        *count += 1;
        if !self.oracle.has(&self.step, &name) {
            self.report.note(format!("{}/{}: {name} is not in the oracle", self.stage, self.step));
            return Ok(());
        }
        let expected = self.oracle.tap(&self.step, &name)?;
        let actual = gpu_download(t).map_err(|e| DiffusionError::model(format!("cutie tap {name}: {e}")))?;
        let tol = (self.tol)(&name);
        self.report.compare(&self.stage, &self.step, &name, &actual, &expected, tol);
        Ok(())
    }

    /// Runs `f` with a tap that compares into this.
    pub(crate) fn with<T>(&mut self, f: impl FnOnce(&mut Tap) -> Result<T>) -> Result<T> {
        let mut sink = |name: &str, t: &GpuTensor| self.take(name, t);
        let mut tap = Tap::new(&mut sink);
        f(&mut tap)
    }
}
