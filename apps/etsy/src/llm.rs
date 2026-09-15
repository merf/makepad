//! In-process Qwen: title-clean one-shot + Ask chat with tools.

use crate::ask_tools;
use makepad_ai_hub::{
    hub::{AiHub, ChatConfig},
    hub_chat::HubChatSession,
    local_llm::{LocalLlmConfig, ToolSpec},
};
use makepad_widgets::makepad_platform::thread::SignalToUI;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub use makepad_ai_hub::local_llm::ChatEvent;

pub const MODEL_FILE: &str = "local/models/Qwen3.5-9B-UD-Q4_K_XL.gguf";
pub const MODEL_ENV: &str = "MAKEPAD_FILES_CHAT_MODEL";

pub const TITLE_SYSTEM_PROMPT: &str = "\
You trim Etsy greeting-card listing titles for display.
Remove SEO fluff only: repeated audience words, keyword stuffing, marketing filler, \
and catalogue/SKU-looking codes (e.g. ABC-1234, SKU123, long alphanumeric product ids).
Do NOT rephrase, synonymise, or invent. Keep the remaining words in their original wording and order.
Reply with ONLY a JSON array. No markdown, no commentary, no tool calls.
Each element: {\"url\":\"...\",\"title_clean\":\"...\"}.

Example input:
[
{\"url\":\"https://etsy.com/listing/1\",\"title\":\"Funny Birthday Card for Him Dad Brother Friend Husband Hilarious Joke Card GC-88421\"},
{\"url\":\"https://etsy.com/listing/2\",\"title\":\"Pack of 4 Cute Cat Birthday Cards\"}
]

Example output:
[
{\"url\":\"https://etsy.com/listing/1\",\"title_clean\":\"Funny Birthday Card\"},
{\"url\":\"https://etsy.com/listing/2\",\"title_clean\":\"Pack of 4 Cute Cat Birthday Cards\"}
]";

pub const ASK_SYSTEM_PROMPT: &str = "\
You are a market-research assistant over the user's LOCAL Etsy greeting-card listings in this app.

How to answer:
- Interrogate the RAW rows. Prefer compute_stats for averages/proportions and query_listings for examples.
- Do not invent listings, prices, or review counts. If a tool result is empty, say so.
- Never invent 'quality' or completeness percentages as business facts.

Postage vocabulary (per listing):
- item£ = listed card price
- postage£ = shipping charge from the search card
- postage_kind free = postage is £0; paid = postage>0 charged ON TOP of item; unknown = not captured
- delivered£ = item+postage when postage is known
So “we know the postage field” is NOT the same as “postage is included in the item price”. \
Free postage means postage_kind=free. Paid-on-top means postage_kind=paid.

shop_reviews / review_count = shop popularity on the search card, not listing sales.
Be concise. Prefer delivered £ and £/card when comparing what buyers actually pay.
Edits (set_title_clean, clear_title_clean, delete_search) need a UI confirm before applying.";

/// Shared session wrapper used for both title-clean and Ask.
pub struct ChatAgent {
    session: HubChatSession,
}

impl ChatAgent {
    pub fn start_title_clean(model: PathBuf) -> Self {
        Self::start(model, TITLE_SYSTEM_PROMPT.to_string(), Vec::new(), 2048)
    }

    pub fn start_ask(model: PathBuf) -> Self {
        Self::start(
            model,
            ASK_SYSTEM_PROMPT.to_string(),
            ask_tools::tool_specs(),
            1024,
        )
    }

    fn start(model: PathBuf, system_prompt: String, tools: Vec<ToolSpec>, max_new: usize) -> Self {
        let mut llm = LocalLlmConfig::new(model);
        llm.max_new_tokens = max_new;
        llm.max_context = 16384;
        let config = ChatConfig {
            llm,
            system_prompt,
            tools,
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

    pub fn cancel(&self) {
        self.session.cancel();
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

/// Parse `[{url, title_clean}, …]` from model output.
pub fn parse_title_clean_batch(text: &str) -> Vec<(String, String)> {
    let blob = extract_json_array(text).unwrap_or_else(|| text.trim().to_string());
    let value = match makepad_strict_json::parse(blob.as_bytes()) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(arr) = value.as_arr() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        let Some(obj) = (match item {
            makepad_strict_json::Value::Obj(_) => Some(item),
            _ => None,
        }) else {
            continue;
        };
        let url = json_str(obj, "url").unwrap_or_default();
        let title_clean = json_str(obj, "title_clean").unwrap_or_default();
        if !url.is_empty() && !title_clean.is_empty() {
            out.push((url, title_clean));
        }
    }
    out
}

fn json_str(obj: &makepad_strict_json::Value, key: &str) -> Option<String> {
    match obj.get(key)? {
        makepad_strict_json::Value::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        _ => None,
    }
}

fn extract_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end <= start {
        return None;
    }
    Some(text[start..=end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_batch() {
        let rows = parse_title_clean_batch(
            r#"Here you go:
[{"url":"https://etsy.com/listing/1","title_clean":"Funny card"}]"#,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "https://etsy.com/listing/1");
        assert_eq!(rows[0].1, "Funny card");
    }
}
