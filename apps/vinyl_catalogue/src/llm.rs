//! In-process Qwen via AiHub — same engine as Files, vinyl-only prompt/tools.

use makepad_ai_hub::{
    hub::{AiHub, ChatConfig},
    hub_chat::HubChatSession,
    local_llm::LocalLlmConfig,
};
use makepad_widgets::makepad_platform::thread::SignalToUI;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub use makepad_ai_hub::local_llm::ChatEvent;

pub const MODEL_FILE: &str = "local/models/Qwen3.5-9B-UD-Q4_K_XL.gguf";
pub const MODEL_ENV: &str = "MAKEPAD_FILES_CHAT_MODEL";

pub const SYSTEM_PROMPT: &str = "\
You split a spoken vinyl catalogue into records.
The speaker lists Artist, Title, Label, Catalogue number, optional notes, then the word next.
Reply with ONLY a JSON array. No markdown, no commentary, no tool calls.
Never invent a field. Use \"\" when something was not said.
Preserve proper names (Surgeon, Tresor, Chain Reaction, Underground Resistance, Rephlex, Peacefrog, Jeff Mills, Basic Channel).

Example input:
Surgeon Magneze Tresor T-179 next Jeff Mills The Bells Purpose Maker PM-001 next

Example output:
[{\"artist\":\"Surgeon\",\"title\":\"Magneze\",\"label\":\"Tresor\",\"catalogue_number\":\"T-179\",\"notes\":\"\"},{\"artist\":\"Jeff Mills\",\"title\":\"The Bells\",\"label\":\"Purpose Maker\",\"catalogue_number\":\"PM-001\",\"notes\":\"\"}]";

pub struct ChatAgent {
    session: HubChatSession,
}

impl ChatAgent {
    pub fn start(model: PathBuf) -> Self {
        let mut llm = LocalLlmConfig::new(model);
        llm.max_new_tokens = 2048;
        llm.max_context = 16384;
        let config = ChatConfig {
            llm,
            system_prompt: SYSTEM_PROMPT.to_string(),
            tools: Vec::new(),
            wake: Some(Arc::new(SignalToUI::set_ui_signal)),
        };
        Self {
            session: AiHub::in_process().start_local_chat(config),
        }
    }

    pub fn send_user_turn(&self, text: String) {
        self.session.send_user_turn(text);
    }

    pub fn send_tool_results(&self, results: Vec<(String, bool)>) {
        self.session.send_tool_results(results);
    }

    pub fn poll(&self) -> Vec<ChatEvent> {
        self.session.poll()
    }
}

pub fn model_path() -> Option<PathBuf> {
    if let Some(from_env) = std::env::var_os(MODEL_ENV) {
        let path = PathBuf::from(from_env);
        return path.is_file().then_some(path);
    }
    let relative = Path::new(MODEL_FILE);
    if relative.is_file() {
        return Some(relative.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        for base in exe.ancestors() {
            let candidate = base.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join(relative))?;
    checkout.is_file().then_some(checkout)
}
