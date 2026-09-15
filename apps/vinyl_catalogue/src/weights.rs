//! Find Whisper/Qwen on disk even when the app was not launched from the repo root.
//!
//! Hub STT only looks at cwd, the executable directory, and
//! `~/.makepad/weights/stt/`. `./tools/download_voice.sh` writes the ggml
//! file at the checkout root, so a double-clicked `target/release/vinyl`
//! reports "Whisper missing". Point `MAKEPAD_VOICE_MODEL` at the checkout
//! copy before VoiceWave starts.

use std::path::{Path, PathBuf};

pub const WHISPER_FILE: &str = "ggml-large-v3-turbo.bin";

pub fn ensure_whisper() {
    if makepad_ai_hub::speech::weights::whisper_model_path().is_some() {
        return;
    }
    let Some(path) = find_whisper() else { return };
    std::env::set_var(
        makepad_ai_hub::speech::weights::WHISPER_MODEL_ENV,
        &path,
    );
}

pub fn whisper_ready() -> bool {
    ensure_whisper();
    makepad_ai_hub::speech::weights::whisper_model_path().is_some()
}

fn find_whisper() -> Option<PathBuf> {
    const REL: [&str; 3] = [
        WHISPER_FILE,
        "local/ggml-large-v3-turbo.bin",
        "local/models/ggml-large-v3-turbo.bin",
    ];
    for rel in REL {
        let path = Path::new(rel);
        if path.is_file() {
            return Some(path.to_path_buf());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for base in exe.ancestors() {
            for rel in REL {
                let candidate = base.join(rel);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)?;
    for rel in REL {
        let candidate = checkout.join(rel);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
