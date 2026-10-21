//! Native Cutie against its PyTorch oracle.
//!
//! `cutie-validate <oracle-dir> [stage...] [--weights <cutie-base-mega.pth>]`
//!
//! `<oracle-dir>` is a clip's folder of RationalVisions' `corpus/cutie-oracle` (made by
//! `tools/cutie-oracle`). Stages: ops, encoder, mask, transformer, decoder, memory, run, all (the
//! default); `bench` times the oracle's run instead (not part of all; `MAKEPAD_CUTIE_BENCH_PASSES`
//! passes, 5 unless set). The weights default to `<oracle-dir>/../../cutie/weights/cutie-base-mega.pth` (the
//! same checkout's Cutie), or `MAKEPAD_CUTIE_WEIGHTS`. Exits 1 when any comparison fails.

use makepad_ai_cutie::{validate, VALIDATE_STAGES};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut oracle: Option<PathBuf> = None;
    let mut weights: Option<PathBuf> = std::env::var_os("MAKEPAD_CUTIE_WEIGHTS").map(PathBuf::from);
    let mut stages = Vec::new();
    while let Some(a) = args.next() {
        if a == "--weights" {
            weights = args.next().map(PathBuf::from);
        } else if oracle.is_none() {
            oracle = Some(PathBuf::from(a));
        } else if VALIDATE_STAGES.contains(&a.as_str()) {
            stages.push(a);
        } else {
            eprintln!("unknown stage {a:?}; stages: {}", VALIDATE_STAGES.join(", "));
            std::process::exit(2);
        }
    }
    let Some(oracle) = oracle else {
        eprintln!("usage: cutie-validate <oracle-dir> [stage...] [--weights <cutie-base-mega.pth>]");
        std::process::exit(2);
    };
    if stages.is_empty() {
        stages.push("all".to_string());
    }
    let weights = weights.unwrap_or_else(|| oracle.join("..").join("..").join("cutie").join("weights").join("cutie-base-mega.pth"));
    match validate(&oracle, &weights, &stages) {
        Ok(summary) => std::process::exit(if summary.failures == 0 { 0 } else { 1 }),
        Err(e) => {
            eprintln!("cutie-validate: {e}");
            std::process::exit(1);
        }
    }
}
