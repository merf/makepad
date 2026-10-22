//! Reading Cutie's oracle and comparing against it. The oracle is RationalVisions'
//! `tools/cutie-oracle` (PyTorch, fp32, TF32 off): a folder per clip with
//!
//! - `taps/<step>/manifest.tsv` (name, dtype, dims a line) and `taps/<step>/<name>.f32` (little-
//!   endian f32 in PyTorch's memory order) for the tapped steps, the schedule's `tap` column:
//!   `commit_f0`, `prop_f0`, `prop_f1`, `prop_f5`, `prop_f40` (working memory consolidates) and
//!   `prop_f41` (the first read through long-term memory) with the default settings, others
//!   with others (`prop_f165` where long-term memory first drops its least used tokens), and the
//!   `edits` scenario's own (`commit_f10` after propagating, `back_f10` and `back_f9` after
//!   `clear_non_permanent_memory`, `restore_f10`, `prop2_f11`, `fresh_f16` into empty memory,
//!   `fresh_f17`);
//! - `raw/meta.txt` (frames, width, height, objects, then every setting Cutie ran with as
//!   `config.<key>=<value>`, `long_term.<key>` too; older oracles have none and ran with the
//!   defaults), `raw/frames.u8` (frames x h x w x 3 RGB), `raw/seed.u8` (objects x h x w, 0/1),
//!   `raw/labels.u8` (frames x h x w, the masks out as label maps, each frame's last),
//!   `raw/schedule.tsv` (each processor step: frame, memorised, consolidated, the memory's sizes,
//!   and in newer oracles whether long-term memory dropped its least used tokens and the clears
//!   before the step), in newer oracles `raw/step_labels.u8` (steps x h x w, each step's label
//!   map: the `edits` scenario steps frames more than once, backwards too).
//!
//! Names: `net.<module>` is a module's output by its name in `CUTIE`; `encode_image.*`, `key.*`,
//! `encode_mask.*`, `pixel_fusion.*`, `object_transformer.*`, `segment.*` the network methods'
//! inputs and outputs; `memory.read.*` and `memory.consolidate.*` the memory's similarity,
//! affinity, usage and readouts; `state.before.*` / `state.after.*` the memory between steps;
//! `step.*` the processor step's input and output. A name seen twice in a step has `#1` added.

use crate::model::{Feat, Tap};
use crate::{CutieConfig, DiffusionError, Result};
use makepad_ai_common::gpu::{gpu_download, GpuTensor};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) struct Oracle {
    pub(crate) dir: PathBuf,
    pub(crate) frames: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) objects: usize,
    /// The settings Cutie ran with (`config.*` in `raw/meta.txt`; the defaults where absent).
    pub(crate) config: CutieConfig,
    /// How the oracle's `torch.topk` broke exact ties, from the device it ran on (`TieOrder`).
    pub(crate) tie_order: TieOrder,
    /// Every processor step of the run (`raw/schedule.tsv`).
    schedule: Vec<ScheduleRow>,
    manifests: HashMap<String, Vec<(String, String, Vec<usize>)>>,
}

/// A tap's shape and values.
pub(crate) struct OracleTap {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
    pub(crate) values: Vec<f32>,
}

impl OracleTap {
    /// A planar tap, [n, c, h, w] or Cutie's grouped [1, n, c, h, w], on the device as n images
    /// of c channels.
    pub(crate) fn feat(&self) -> Result<Feat> {
        match self.shape[..] {
            [n, c, h, w] | [1, n, c, h, w] => Feat::upload(&self.values, n, c, w, h),
            _ => Err(DiffusionError::model(format!("cutie oracle: {:?} is not [n, c, h, w] or [1, n, c, h, w]", self.shape))),
        }
    }

    /// A tap of `n` images of `c` channels whose last two dimensions are the height and the
    /// width, whatever the leading ones say (a batch of masks [1, n, H, W] is n images of one
    /// channel), on the device.
    pub(crate) fn feat_as(&self, n: usize, c: usize) -> Result<Feat> {
        let dims = self.shape.len();
        let (h, w) = if dims >= 2 { (self.shape[dims - 2], self.shape[dims - 1]) } else { (0, 0) };
        if self.values.len() != n * c * h * w || self.values.is_empty() {
            return Err(DiffusionError::model(format!("cutie oracle: {:?} is not {n} images of {c} channels", self.shape)));
        }
        Feat::upload(&self.values, n, c, w, h)
    }
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
    /// Long-term memory dropped its least used tokens before consolidating
    /// (`remove_obsolete_features`); None in oracles that did not record it.
    pub(crate) obsolete_removed: Option<bool>,
    /// The processor's clears before the step, in order; None in oracles that did not record
    /// them (`clears` says what those did).
    pub(crate) recorded_clears: Option<Vec<Clear>>,
}

/// A clear the worker gives the processor between steps (`InferenceCore`'s `clear_memory`,
/// `clear_non_permanent_memory`, `clear_sensory_memory`). Each resets the step counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Clear {
    All,
    NonPermanent,
    Sensory,
}

impl ScheduleRow {
    /// The clears before this step: as recorded (the schedule's `clear` column), or, in older
    /// oracles, which ran only the worker's commit and one propagation, the sensory memory
    /// cleared where a mask starts the propagation (a mask on a step that is not a commit).
    pub(crate) fn clears(&self) -> Vec<Clear> {
        match &self.recorded_clears {
            Some(clears) => clears.clone(),
            None if self.with_mask && !self.force_permanent => vec![Clear::Sensory],
            None => Vec::new(),
        }
    }
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
        let objects = get("objects")?;
        let config = Self::read_config(&meta, objects).map_err(|e| io(&meta_path, e))?;
        // Oracles from before meta.txt carried the settings record the device in oracle.json
        // only (read only then: it is large).
        let json = match Self::device_in(&meta, None) {
            Some(_) => None,
            None => fs::read_to_string(dir.join("oracle.json")).ok(),
        };
        let tie_order = TieOrder::of_device(Self::device_in(&meta, json.as_deref()).as_deref());
        let schedule = Self::read_schedule(dir)?;
        let mut manifests = HashMap::new();
        for step in schedule.iter().filter_map(|r| r.tap.as_deref()) {
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
        Ok(Self {
            dir: dir.to_path_buf(),
            frames: get("frames")?,
            width: get("width")?,
            height: get("height")?,
            objects,
            config,
            tie_order,
            schedule,
            manifests,
        })
    }

    /// The device the oracle ran on: `meta.txt`'s `config.device`, else the top-level `"device"`
    /// of `oracle.json` (`json`, written with one space of indent; the settings' own `"device"`,
    /// deeper, may say `auto`), which every oracle has, older ones without `config.*` lines too.
    /// None when neither names one.
    fn device_in(meta: &str, json: Option<&str>) -> Option<String> {
        let named = |v: &str| -> Option<String> {
            let v = v.trim().trim_end_matches(',').trim().trim_matches('"');
            (!v.is_empty() && v != "auto").then(|| v.to_string())
        };
        meta.lines()
            .find_map(|l| l.strip_prefix("config.device="))
            .and_then(named)
            .or_else(|| json?.lines().find_map(|l| l.strip_prefix(" \"device\":")).and_then(named))
    }

    /// The settings in `meta.txt`'s `config.<key>=<value>` lines over the defaults (an oracle
    /// without them ran with the defaults). Settings that do not change what Cutie computes (the
    /// device, the weights' file names, `amp`, which the oracle runs in fp32 anyway, `save_aux`)
    /// are passed over; one the port does not do (`flip_aug`), or does not know, is an error
    /// rather than a check of something else. So is a `chunk_size` from 1 to fewer than the
    /// `objects`: PyTorch's memory read then runs pixel fusion and the object transformer a chunk
    /// of objects at a time, each object's others and the attention masks' aggregate taken over
    /// its chunk only, while the port always passes every object (as `chunk_size` -1, the
    /// worker's, does; a chunk of every object or more is the same).
    fn read_config(meta: &str, objects: usize) -> std::result::Result<CutieConfig, String> {
        let mut config = CutieConfig::default();
        for (key, value) in meta.lines().filter_map(|l| l.strip_prefix("config.")).filter_map(|l| l.split_once('=')) {
            let value = value.trim();
            let count = || value.parse::<usize>().map_err(|e| format!("config.{key}={value}: {e}"));
            let flag = || match value {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(format!("config.{key}={value}: not true or false")),
            };
            match key {
                "device" | "weights" | "ritm_weights" | "amp" | "save_aux" => {}
                "flip_aug" if !flag()? => {}
                "chunk_size" => {
                    let chunk = value.parse::<i64>().map_err(|e| format!("config.{key}={value}: {e}"))?;
                    if chunk >= 1 && (chunk as usize) < objects {
                        return Err(format!("config.{key}={value}: chunks of fewer than the {objects} objects, which the port does not do"));
                    }
                }
                // Cutie resizes only when this is positive.
                "max_internal_size" => {
                    config.max_internal_size = value.parse::<i64>().map_err(|e| format!("config.{key}={value}: {e}"))?.max(0) as usize
                }
                "use_long_term" => config.use_long_term = flag()?,
                "mem_every" => config.mem_every = count()?,
                "top_k" => config.top_k = count()?,
                "stagger_updates" => config.stagger_updates = count()?,
                "max_mem_frames" => config.max_mem_frames_without_long_term = count()?,
                "long_term.count_usage" => config.count_long_term_usage = flag()?,
                "long_term.max_mem_frames" => config.max_mem_frames = count()?,
                "long_term.min_mem_frames" => config.min_mem_frames = count()?,
                "long_term.num_prototypes" => config.num_prototypes = count()?,
                "long_term.max_num_tokens" => config.max_long_tokens = count()?,
                "long_term.buffer_tokens" => config.buffer_tokens = count()?,
                _ => return Err(format!("config.{key}={value}: a setting the port does not do")),
            }
        }
        Ok(config)
    }

    /// The tapped steps, in the run's order.
    pub(crate) fn steps(&self) -> Vec<&str> {
        self.schedule.iter().filter_map(|r| r.tap.as_deref()).collect()
    }

    /// The names tapped in `step`, in the order they were tapped.
    pub(crate) fn names(&self, step: &str) -> Vec<&str> {
        self.manifests.get(step).map(|m| m.iter().map(|(n, _, _)| n.as_str()).collect()).unwrap_or_default()
    }

    pub(crate) fn has(&self, step: &str, name: &str) -> bool {
        self.manifests.get(step).is_some_and(|m| m.iter().any(|(n, _, _)| n == name))
    }

    /// A tap's manifest line: its dtype and shape.
    fn entry(&self, step: &str, name: &str) -> Result<(&str, &[usize])> {
        self.manifests
            .get(step)
            .and_then(|m| m.iter().find(|(n, _, _)| n == name))
            .map(|(_, dtype, shape)| (dtype.as_str(), shape.as_slice()))
            .ok_or_else(|| DiffusionError::model(format!("cutie oracle: no tap {step}/{name}")))
    }

    /// A tap's shape, from the manifest (nothing is read).
    pub(crate) fn shape(&self, step: &str, name: &str) -> Result<&[usize]> {
        Ok(self.entry(step, name)?.1)
    }

    /// A planar tap on the device (`OracleTap::feat`).
    pub(crate) fn feat(&self, step: &str, name: &str) -> Result<Feat> {
        self.tap(step, name)?.feat().map_err(|e| DiffusionError::model(format!("{step}/{name}: {e}")))
    }

    /// A tap of `n` images of `c` channels on the device (`OracleTap::feat_as`).
    pub(crate) fn feat_as(&self, step: &str, name: &str, n: usize, c: usize) -> Result<Feat> {
        self.tap(step, name)?.feat_as(n, c).map_err(|e| DiffusionError::model(format!("{step}/{name}: {e}")))
    }

    pub(crate) fn tap(&self, step: &str, name: &str) -> Result<OracleTap> {
        let (dtype, shape) = self.entry(step, name)?;
        let path = self.dir.join("taps").join(step).join(format!("{name}.f32"));
        let bytes = fs::read(&path).map_err(|e| io(&path, e))?;
        let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if values.len() != shape.iter().product::<usize>() {
            return Err(io(&path, format!("{} values for {shape:?}", values.len())));
        }
        Ok(OracleTap { shape: shape.to_vec(), dtype: dtype.to_string(), values })
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

    /// Every processor step of the run, in order.
    pub(crate) fn schedule(&self) -> &[ScheduleRow] {
        &self.schedule
    }

    /// Each step's label map as the oracle's run made it (`raw/step_labels.u8`, steps x h x w:
    /// a frame stepped twice has two), in newer oracles; None in older ones, which step each
    /// frame once but frame 0 (`labels`).
    pub(crate) fn step_labels(&self) -> Result<Option<Vec<u8>>> {
        let path = self.dir.join("raw").join("step_labels.u8");
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| io(&path, e))?;
        if bytes.len() != self.schedule.len() * self.width * self.height {
            return Err(io(&path, format!("{} bytes for {} steps of {}x{}", bytes.len(), self.schedule.len(), self.width, self.height)));
        }
        Ok(Some(bytes))
    }

    /// `raw/schedule.tsv`: a header, then a step a line. The 13th column, `obsolete_removed`, and
    /// the 14th, `clear`, are in newer oracles only.
    fn read_schedule(dir: &Path) -> Result<Vec<ScheduleRow>> {
        let path = dir.join("raw").join("schedule.tsv");
        let text = fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        let mut lines = text.lines().filter(|l| !l.is_empty());
        let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
        let obsolete_column = header.iter().position(|&h| h == "obsolete_removed");
        let clear_column = header.iter().position(|&h| h == "clear");
        let mut rows = Vec::new();
        for line in lines {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() < 12 || obsolete_column.into_iter().chain(clear_column).any(|i| c.len() <= i) {
                return Err(io(&path, format!("short line {line:?}")));
            }
            let num = |s: &str| s.parse::<i64>().map_err(|e| io(&path, e));
            let recorded_clears = match clear_column.map(|i| c[i]) {
                None => None,
                Some("-") => Some(Vec::new()),
                Some(list) => Some(
                    list.split('+')
                        .map(|name| match name {
                            "all" => Ok(Clear::All),
                            "non_permanent" => Ok(Clear::NonPermanent),
                            "sensory" => Ok(Clear::Sensory),
                            _ => Err(io(&path, format!("clear {name:?}"))),
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
            };
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
                obsolete_removed: obsolete_column.map(|i| c[i] == "1"),
                recorded_clears,
            });
        }
        Ok(rows)
    }
}

// --- Comparing ---

/// How close a tap must be: the largest difference relative to the expected tensor's largest
/// magnitude, and the cosine similarity. Boolean taps (attention masks) are compared exactly,
/// `bool_mismatch` the fraction of entries allowed to differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tol {
    pub(crate) rel: f32,
    pub(crate) cos: f64,
    pub(crate) bool_mismatch: f64,
}

impl Tol {
    /// A stage fed the oracle's own inputs (teacher-forced): only fp32 rounding apart, and tight
    /// enough to tell TF32 from fp32. Over the seven oracles fp32 rounding reaches rel 4.8e-5
    /// (the decoder's `segment.out.sensory` and well-conditioned logits, the end of a long chain;
    /// most taps 1e-7 to 5e-6) and 1 - cosine 1.5e-11 (`Report::margins` prints each stage's).
    /// One layer in TF32 (its weights and input rounded to 10-bit mantissas) gives rel 2e-4 to
    /// 1e-3 and 1 - cosine about 2e-8: the cosine bound fails it by 20 times while leaving fp32
    /// 60 times of room. (Was rel 1e-3, cosine 0.99999, which TF32 in the pixel fuser and the key
    /// projection passed.)
    pub(crate) const STAGE: Tol = Tol { rel: 2e-4, cos: 1.0 - 1e-9, bool_mismatch: 0.0 };
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
    /// The tolerance a tensor comparison was judged by (None for `record` rows).
    pub(crate) tol: Option<Tol>,
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
            tol: Some(tol),
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
                    // A pair with a non-finite side matches only as two NaNs or the same
                    // infinity (+inf against -inf differs).
                    if !(a.is_nan() && e.is_nan()) && a != e {
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
            // Where the oracle is all zeros, no relative measure exists: a result within 1e-6 of
            // zero passes. Anything else is relative, however small (life counts start at 1e-7).
            let zeros = row.max_ref == 0.0 && row.max_abs < 1e-6;
            row.pass = nonfinite == 0 && (zeros || (row.rel <= tol.rel && row.cos >= tol.cos));
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

    /// A row that is not a tensor comparison (a label map's IoU, the schedule, a pairing),
    /// passing or failing as the check judged it; `note` says what was found.
    pub(crate) fn record(&mut self, stage: &str, step: &str, name: &str, pass: bool, note: impl Into<String>) {
        if !pass {
            self.failures += 1;
        }
        self.rows.push(Row {
            stage: stage.into(),
            step: step.into(),
            name: name.into(),
            n: 0,
            max_abs: f32::NAN,
            max_ref: f32::NAN,
            rel: f32::NAN,
            cos: f64::NAN,
            pass,
            note: note.into(),
            tol: None,
        });
    }

    pub(crate) fn fail(&mut self, stage: &str, step: &str, name: &str, why: impl Into<String>) {
        self.record(stage, step, name, false, why);
    }

    /// One line a comparison (rows that are not one show no numbers), then the notes, then each
    /// stage's margin under `Tol::STAGE` (`margins`).
    pub(crate) fn print(&self) {
        for r in &self.rows {
            let numbers = if r.max_abs.is_nan() && r.cos.is_nan() {
                format!("{:<10} {:<19} {:<15} {:<19}", "n=-", "max_abs=-", "rel=-", "cos=-")
            } else {
                format!("n={:<8} max_abs={:<11.4e} rel={:<11.4e} cos={:<15.12}", r.n, r.max_abs, r.rel, r.cos)
            };
            println!("{} {:<11} {:<9} {:<64} {numbers} {}", if r.pass { "PASS" } else { "FAIL" }, r.stage, r.step, r.name, r.note);
        }
        for n in &self.notes {
            println!("NOTE {n}");
        }
        for m in self.margins() {
            println!("NOTE {m}");
        }
    }

    /// Per stage, the rows judged by `Tol::STAGE` that came closest to it: the largest rel and
    /// the largest 1 - cosine, with where. How much room fp32 rounding leaves under the
    /// tolerance, which must stay tight enough to tell TF32 or fp16 from fp32.
    pub(crate) fn margins(&self) -> Vec<String> {
        let mut stages: Vec<&str> = Vec::new();
        for r in self.rows.iter().filter(|r| r.tol == Some(Tol::STAGE)) {
            if !stages.contains(&r.stage.as_str()) {
                stages.push(&r.stage);
            }
        }
        stages
            .into_iter()
            .map(|stage| {
                let rows: Vec<&Row> = self.rows.iter().filter(|r| r.stage == stage && r.tol == Some(Tol::STAGE) && r.cos.is_finite()).collect();
                let worst_rel = rows.iter().max_by(|a, b| a.rel.total_cmp(&b.rel));
                let worst_cos = rows.iter().min_by(|a, b| a.cos.total_cmp(&b.cos));
                let at = |r: Option<&&Row>| r.map_or("-".to_string(), |r| format!("{}/{}", r.step, r.name));
                format!(
                    "margin {stage}: {} rows at Tol::STAGE (rel {:.0e}, 1-cos {:.0e}); largest rel {:.3e} at {}; largest 1-cos {:.3e} at {}",
                    rows.len(),
                    Tol::STAGE.rel,
                    1.0 - Tol::STAGE.cos,
                    worst_rel.map_or(0.0, |r| r.rel),
                    at(worst_rel),
                    worst_cos.map_or(0.0, |r| 1.0 - r.cos),
                    at(worst_cos),
                )
            })
            .collect()
    }
}

/// How an oracle's `torch.topk` broke exact ties, which PyTorch leaves open: from the device it
/// ran on (`Oracle::device_in`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TieOrder {
    /// CUDA's: the lower index first, as the port's top-ks (`gpu_cutie_topk_softmax`). On such an
    /// oracle the port is held to it: a tie it broke otherwise fails (`TieJudgement::pass`).
    LowerIndex,
    /// CPU torch's (or the device is not recorded): an order of its own, so the picks at an exact
    /// tie are judged as a set.
    Open,
}

impl TieOrder {
    fn of_device(device: Option<&str>) -> Self {
        match device {
            Some(d) if d.starts_with("cuda") => TieOrder::LowerIndex,
            _ => TieOrder::Open,
        }
    }
}

/// Exact ties at the cut of a top-k taken down each column of a score matrix: the memory read's
/// similarity [N, h*w], each pixel keeping its k best memory tokens. `torch.topk` leaves open
/// which of equal values it keeps: CUDA's keeps the lower index, as the port does
/// (`gpu_cutie_topk_softmax`), and CPU torch's an order of its own (about half of the tied
/// columns the higher index, on the PC's x86 and the Mac's M5 alike). Working memory holds the
/// first frame twice (the commit's permanent copy, then the propagation's first temporary
/// frame), so pairs of identical tokens tie, and the cut falls inside a pair wherever an odd
/// number of tokens score above it: at `prop_f40` 50 of Solege-03's 336 columns, 32 of
/// YouTubeComp-15's, 40 of lawn_cut's 432; in the edits, 100 at `back_f9` and 98 at `prop2_f11`.
/// So where the oracle's own k-th and (k+1)-th scores are exactly equal, a top-k's output is
/// judged with its picks among the tied tokens taken as a set (`align`); every other column, and
/// every token outside the tie, exactly as before. On an oracle made on CUDA (`TieOrder::
/// LowerIndex`) the picks are still aligned, but a column moved fails the judgement's row, so the
/// port's lower-index rule stays checked against CUDA's torch.
pub(crate) struct TopKTies {
    rows: usize,
    cols: usize,
    /// Each tied column with its tied tokens (the rows scoring exactly the k-th best), ascending.
    columns: Vec<(usize, Vec<usize>)>,
}

impl TopKTies {
    /// The columns of `scores` (an oracle tap whose last two dimensions are the rows and the
    /// columns, any before them 1) whose k-th and (k+1)-th largest are exactly equal.
    pub(crate) fn find(scores: &OracleTap, k: usize) -> Result<Self> {
        let dims = scores.shape.len();
        let (rows, cols) = if dims >= 2 { (scores.shape[dims - 2], scores.shape[dims - 1]) } else { (0, 0) };
        if dims < 2 || rows * cols != scores.values.len() {
            return Err(DiffusionError::model(format!("cutie oracle: top-k scores {:?} are not one matrix", scores.shape)));
        }
        let mut columns = Vec::new();
        // Without a (k+1)-th score every token is kept: no cut, no tie.
        if k == 0 || k >= rows {
            return Ok(Self { rows, cols, columns });
        }
        let at = |r: usize, c: usize| scores.values[r * cols + c];
        let mut column = vec![0.0f32; rows];
        for c in 0..cols {
            for (r, v) in column.iter_mut().enumerate() {
                *v = at(r, c);
            }
            // Largest first: the k largest end up before index k, the (k+1)-th at it.
            column.select_nth_unstable_by(k, |a, b| b.total_cmp(a));
            let next = column[k];
            let kth = column[..k].iter().copied().min_by(|a, b| a.total_cmp(b)).unwrap_or(f32::NAN);
            if kth == next {
                columns.push((c, (0..rows).filter(|&r| at(r, c) == kth).collect()));
            }
        }
        Ok(Self { rows, cols, columns })
    }

    /// Judges `actual`, a top-k's output down the columns of the scores ([rows, cols] in the same
    /// order, zero where a token was not kept: the read's affinity), against the oracle's
    /// `expected`. In each tied column where it kept other tied tokens than the oracle did, as
    /// many of them, their weights are moved onto the oracle's, in order (they are equal: equal
    /// scores through one softmax). Every token outside the tie, every column without one, and a
    /// tied column that kept another number of the tied tokens than the oracle are left as they
    /// are, to be compared as before.
    pub(crate) fn align(&self, actual: &mut [f32], expected: &[f32]) -> TieJudgement {
        let mut judgement = TieJudgement { columns: self.cols, tied: self.columns.len(), ..TieJudgement::default() };
        let n = self.rows * self.cols;
        if actual.len() != n || expected.len() != n {
            // The comparison fails on the size; nothing to judge.
            judgement.unmatched = self.columns.len();
            return judgement;
        }
        let cols = self.cols;
        for (c, tied) in &self.columns {
            let kept = |t: &[f32]| -> Vec<usize> { tied.iter().copied().filter(|&r| t[r * cols + c] != 0.0).collect() };
            let (ours, theirs) = (kept(actual), kept(expected));
            if ours == theirs {
                continue;
            }
            if ours.len() != theirs.len() {
                judgement.unmatched += 1;
                continue;
            }
            let weights: Vec<f32> = ours.iter().map(|&r| actual[r * cols + c]).collect();
            for &r in &ours {
                actual[r * cols + c] = 0.0;
            }
            for ((&from, &to), &w) in ours.iter().zip(&theirs).zip(&weights) {
                actual[to * cols + c] = w;
                judgement.moves.push((from, to, w));
            }
            judgement.moved += 1;
        }
        judgement
    }
}

/// What `TopKTies::align` found and did.
#[derive(Default)]
pub(crate) struct TieJudgement {
    /// The top-k's columns, and those tied at its cut in the oracle's scores.
    pub(crate) columns: usize,
    pub(crate) tied: usize,
    /// Tied columns that kept other tied tokens than the oracle, as many, judged as sets: their
    /// weights were moved onto the oracle's tokens.
    pub(crate) moved: usize,
    /// Tied columns that kept another number of the tied tokens than the oracle (or a top-k of
    /// another size), left to the comparison.
    pub(crate) unmatched: usize,
    /// Each weight moved: from row, to row, the weight.
    moves: Vec<(usize, usize, f32)>,
}

impl TieJudgement {
    /// `sums`, the top-k output's sums along each row (the read's usage, [rows]), with the
    /// weights `align` moved moved too (each row's change summed in f64, then added once).
    pub(crate) fn align_sums(&self, sums: &mut [f32]) {
        let mut change: HashMap<usize, f64> = HashMap::new();
        for &(from, to, w) in &self.moves {
            *change.entry(from).or_insert(0.0) -= w as f64;
            *change.entry(to).or_insert(0.0) += w as f64;
        }
        for (row, d) in change {
            if let Some(s) = sums.get_mut(row).filter(|_| d != 0.0) {
                *s = (*s as f64 + d) as f32;
            }
        }
    }

    /// Whether the judgement's row passes, the oracle having broken ties in `order`. Where it kept
    /// the lower index (CUDA's torch.topk, as the port's top-k does), the port must have kept the
    /// oracle's own tied tokens: a column `moved` fails, and so does one `unmatched` where
    /// `unmatched_fails` (stage `memory`, which reads with the oracle's own keys; stage `run`
    /// reads its own, whose near ties may move the cut, as its affinity row allows). Anywhere else
    /// the picks at a tie are judged as sets and the row passes, saying what was moved.
    pub(crate) fn pass(&self, order: TieOrder, unmatched_fails: bool) -> bool {
        match order {
            TieOrder::LowerIndex => self.moved == 0 && (!unmatched_fails || self.unmatched == 0),
            TieOrder::Open => true,
        }
    }

    /// What was judged, for a report row.
    pub(crate) fn note(&self, k: usize, order: TieOrder) -> String {
        let rule = match order {
            TieOrder::LowerIndex => "a CUDA oracle: the lower index must be kept",
            TieOrder::Open => "not a CUDA oracle: tied picks judged as sets",
        };
        format!(
            "{} of {} columns tied at the top {k}'s cut in the oracle's scores; {} kept other tied tokens than the oracle ({} weights moved onto the oracle's), {} another number of them ({rule})",
            self.tied,
            self.columns,
            self.moved,
            self.moves.len(),
            self.unmatched
        )
    }
}

/// A `Comparer`'s judging of the memory read's exact top-k ties (`TopKTies`), for stage `run`:
/// `memory.read.affinity` is aligned to the oracle's picks among the tied tokens before it is
/// compared, then `memory.read.usage` by the same moves. (Stage `memory` aligns the read itself,
/// and carries the oracle's picks on into the usage counted and the readout.)
pub(crate) struct ReadTies {
    pub(crate) top_k: usize,
    /// The step's judgement, once its affinity is taken.
    pub(crate) judgement: Option<TieJudgement>,
}

impl ReadTies {
    pub(crate) fn new(top_k: usize) -> Self {
        Self { top_k, judgement: None }
    }

    /// Aligns `actual`, the tap `name` of `step`, when it is the read's affinity or usage.
    fn align(&mut self, oracle: &Oracle, step: &str, name: &str, actual: &mut [f32], expected: &[f32]) -> Result<()> {
        match name {
            "memory.read.affinity" if oracle.has(step, "memory.read.similarity") => {
                let ties = TopKTies::find(&oracle.tap(step, "memory.read.similarity")?, self.top_k)?;
                self.judgement = Some(ties.align(actual, expected));
            }
            "memory.read.usage" => {
                if let Some(judgement) = &self.judgement {
                    judgement.align_sums(actual);
                }
            }
            _ => {}
        }
        Ok(())
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
    /// When set, the read's affinity and usage are judged at exact top-k ties (`ReadTies`).
    pub(crate) read_ties: Option<ReadTies>,
}

impl<'a> Comparer<'a> {
    pub(crate) fn new(oracle: &'a Oracle, report: &'a mut Report, stage: &str, step: &str, tol: &'a dyn Fn(&str) -> Tol) -> Self {
        Self { oracle, report, stage: stage.into(), step: step.into(), tol, seen: HashMap::new(), read_ties: None }
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
        let mut actual = gpu_download(t).map_err(|e| DiffusionError::model(format!("cutie tap {name}: {e}")))?;
        if let Some(ties) = &mut self.read_ties {
            ties.align(self.oracle, &self.step, &name, &mut actual, &expected.values)?;
        }
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

    /// Every name put so far as the oracle names it: a name put again is `name#1`, `name#2`, ...
    pub(crate) fn put_names(&self) -> HashSet<String> {
        let mut put = HashSet::new();
        for (name, &count) in &self.seen {
            put.insert(name.clone());
            put.extend((1..count).map(|i| format!("{name}#{i}")));
        }
        put
    }

    /// Fails, under this comparer's stage and step, every oracle tap under `owned` that nothing
    /// put (but those in `not_compared`: names the stage owns and does not compare, such as inputs
    /// it feeds and does not put back), and every name put that the oracle lacks. So a renamed
    /// tap, on either side, or a regenerated oracle cannot drop comparisons and still pass:
    /// `take` alone only notes a name the oracle lacks.
    pub(crate) fn check_coverage(&mut self, owned: &[&str], not_compared: &[&str]) {
        let (stage, step) = (self.stage.clone(), self.step.clone());
        for name in self.uncompared(owned) {
            if !not_compared.contains(&name.as_str()) {
                self.report.fail(&stage, &step, &name, "in the oracle, not put by the port");
            }
        }
        let mut extra: Vec<String> = self.put_names().into_iter().filter(|name| !self.oracle.has(&step, name)).collect();
        extra.sort();
        for name in extra {
            self.report.fail(&stage, &step, &name, "put by the port, not in the oracle");
        }
    }

    /// The step's oracle taps under any of `prefixes` ("" for all) that nothing put, in the
    /// oracle's order. A zero-size tap is what an absent native tensor is (PyTorch keeps emptied
    /// tensors, e.g. after `clear_non_permanent_memory`), so it is not listed.
    pub(crate) fn uncompared(&self, prefixes: &[&str]) -> Vec<String> {
        let put = self.put_names();
        self.oracle
            .names(&self.step)
            .into_iter()
            .filter(|name| prefixes.iter().any(|p| name.starts_with(p)) && !put.contains(*name))
            .filter(|name| self.oracle.shape(&self.step, name).map_or(true, |s| s.iter().product::<usize>() > 0))
            .map(String::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores [1, rows, cols] from columns.
    fn scores(columns: &[&[f32]]) -> OracleTap {
        let (rows, cols) = (columns[0].len(), columns.len());
        let values = (0..rows * cols).map(|i| columns[i % cols][i / cols]).collect();
        OracleTap { shape: vec![1, rows, cols], dtype: "float32".into(), values }
    }

    /// A top-k output [rows, cols] keeping `kept[c]`'s rows of column c, each at weight `w`.
    fn kept(rows: usize, kept: &[&[usize]], w: f32) -> Vec<f32> {
        let cols = kept.len();
        let mut out = vec![0.0; rows * cols];
        for (c, rows) in kept.iter().enumerate() {
            for &r in *rows {
                out[r * cols + c] = w;
            }
        }
        out
    }

    #[test]
    fn only_a_tie_at_the_cut_is_a_tie() {
        let s = scores(&[
            &[5.0, 4.0, 4.0, 1.0, 0.0],  // 2nd and 3rd equal: tied, rows 1 and 2
            &[5.0, 4.0, 3.0, 3.0, 0.0],  // equal below the cut
            &[4.0, 4.0, 1.0, 0.0, 0.0],  // equal inside the top 2
            &[2.0, 0.0, 2.0, -1.0, 2.0], // three equal across the cut: rows 0, 2, 4
        ]);
        let ties = TopKTies::find(&s, 2).unwrap();
        assert_eq!(ties.columns, vec![(0, vec![1, 2]), (3, vec![0, 2, 4])]);
        assert!(TopKTies::find(&s, 5).unwrap().columns.is_empty(), "every token kept: no cut");
        assert!(TopKTies::find(&s, 0).unwrap().columns.is_empty());
        let not_a_matrix = OracleTap { shape: vec![4, 5], dtype: "float32".into(), values: vec![0.0; 19] };
        assert!(TopKTies::find(&not_a_matrix, 2).is_err());
    }

    #[test]
    fn other_tied_picks_are_moved_and_nothing_else() {
        let s = scores(&[&[5.0, 4.0, 4.0, 1.0, 0.0], &[5.0, 4.0, 3.0, 3.0, 0.0], &[2.0, 0.0, 2.0, -1.0, 2.0]]);
        let ties = TopKTies::find(&s, 2).unwrap();
        // The oracle kept the higher of each tie; the port the lower, and in the untied column
        // another token than the oracle.
        let expected = kept(5, &[&[0, 2], &[0, 1], &[2, 4]], 0.5);
        let mut actual = kept(5, &[&[0, 1], &[0, 2], &[0, 2]], 0.5);
        let mut usage: Vec<f32> = (0..5).map(|r| (0..3).map(|c| actual[r * 3 + c]).sum()).collect();
        let judgement = ties.align(&mut actual, &expected);
        assert_eq!((judgement.tied, judgement.moved, judgement.unmatched), (2, 2, 0));
        let column = |t: &[f32], c: usize| (0..5).map(|r| t[r * 3 + c]).collect::<Vec<_>>();
        assert_eq!(column(&actual, 0), column(&expected, 0));
        assert_eq!(column(&actual, 2), column(&expected, 2));
        assert_eq!(column(&actual, 1), vec![0.5, 0.0, 0.5, 0.0, 0.0], "an untied column is left to the comparison");
        judgement.align_sums(&mut usage);
        let sums: Vec<f32> = (0..5).map(|r| (0..3).map(|c| actual[r * 3 + c]).sum()).collect();
        assert_eq!(usage, sums, "the usage follows the moves");
    }

    #[test]
    fn a_tie_kept_in_another_number_is_left_alone() {
        let s = scores(&[&[5.0, 4.0, 4.0, 1.0, 0.0]]);
        let ties = TopKTies::find(&s, 2).unwrap();
        let expected = kept(5, &[&[0, 2]], 0.5);
        for picks in [&[1usize, 2][..], &[0, 3][..]] {
            let mut actual = kept(5, &[picks], 0.5);
            let before = actual.clone();
            let judgement = ties.align(&mut actual, &expected);
            assert_eq!((judgement.moved, judgement.unmatched), (0, 1), "{picks:?}");
            assert_eq!(actual, before);
        }
        let mut same = expected.clone();
        let judgement = ties.align(&mut same, &expected);
        assert_eq!((judgement.tied, judgement.moved, judgement.unmatched), (1, 0, 0));
        let mut short = vec![0.0; 4];
        assert_eq!(ties.align(&mut short, &expected).unmatched, 1, "another size is not judged");
    }

    #[test]
    fn a_cuda_oracle_holds_the_port_to_the_lower_index() {
        let s = scores(&[&[5.0, 4.0, 4.0, 1.0, 0.0]]);
        let ties = TopKTies::find(&s, 2).unwrap();
        let lower = kept(5, &[&[0, 1]], 0.5);
        let (cuda, cpu) = (TieOrder::LowerIndex, TieOrder::Open);
        // The port kept the higher of the twins, CUDA's oracle the lower: moved, which fails there.
        let moved = ties.align(&mut kept(5, &[&[0, 2]], 0.5), &lower);
        assert_eq!(moved.moved, 1);
        assert!(!moved.pass(cuda, true) && !moved.pass(cuda, false));
        assert!(moved.pass(cpu, true));
        // Both twins kept: another number, failing on the oracle's own keys only.
        let unmatched = ties.align(&mut kept(5, &[&[1, 2]], 0.5), &lower);
        assert_eq!((unmatched.moved, unmatched.unmatched), (0, 1));
        assert!(!unmatched.pass(cuda, true) && unmatched.pass(cuda, false) && unmatched.pass(cpu, true));
        let same = ties.align(&mut lower.clone(), &lower);
        assert!(same.pass(cuda, true));
    }

    #[test]
    fn the_device_gives_the_tie_order() {
        let order = |meta: &str, json: Option<&str>| TieOrder::of_device(Oracle::device_in(meta, json).as_deref());
        let json = |device: &str| format!("{{\n \"what\": \"oracle\",\n \"device\": \"{device}\",\n \"settings\": {{\n  \"device\": \"auto\"\n }}\n}}\n");
        assert_eq!(order("frames=42\nconfig.device=cuda\n", None), TieOrder::LowerIndex);
        assert_eq!(order("frames=42\nconfig.device=cpu\n", Some(&json("cuda"))), TieOrder::Open, "meta.txt first");
        // Older oracles: no config lines, the device in oracle.json, not its settings' "auto".
        assert_eq!(order("frames=42\n", Some(&json("cuda"))), TieOrder::LowerIndex);
        assert_eq!(order("frames=42\n", Some(&json("cpu"))), TieOrder::Open);
        assert_eq!(order("config.device=auto\n", Some(&json("cuda"))), TieOrder::LowerIndex);
        assert_eq!(order("frames=42\n", Some("{\n \"settings\": {\n  \"device\": \"cuda\"\n }\n}\n")), TieOrder::Open);
        assert_eq!(order("frames=42\n", None), TieOrder::Open, "not recorded");
    }
}
