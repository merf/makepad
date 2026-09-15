//! One-window catalogue: armed mic capture loop, Discogs confirm modals, grid.

use crate::extract;
use crate::model::{Extracted, Record};
use makepad_widgets::*;
use makepad_widgets::makepad_platform::file_dialogs::{FileDialog, FileDialogAction};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "native")]
use crate::db::Db;
#[cfg(feature = "native")]
use crate::discogs::{self, MasterCandidate, VersionsBundle};
#[cfg(feature = "native")]
use crate::llm::{ChatAgent, ChatEvent};
#[cfg(feature = "native")]
use makepad_widgets::makepad_platform::thread::{Lane, ToUIReceiver};

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    mod.widgets.VinylBase = #(Vinyl::register_widget(vm))

    let Panel = SolidView{
        draw_bg +: { color: #x14161d }
    }

    let Dim = Label{
        draw_text +: {
            color: #x9aa7b4
            text_style: theme.font_regular{font_size: 8.5}
        }
    }

    let Title = Label{
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_bold{font_size: 13}
        }
    }

    let Chip = Button{
        height: 28
        padding: Inset{left: 12, right: 12, top: 5, bottom: 5}
        draw_bg +: {
            color: #x1b1e27
            color_hover: #x272a35
            color_down: #xc45c2644
            color_focus: #x1b1e27
            border_radius: 6.0
            border_size: 0.0
        }
        draw_text +: {
            color: #x9aa7b4
            color_hover: #xf2f4f8
            color_down: #xf2f4f8
            color_focus: #x9aa7b4
            text_style: theme.font_regular{font_size: 9}
        }
    }

    let Primary = Button{
        height: 28
        padding: Inset{left: 16, right: 16, top: 5, bottom: 5}
        draw_bg +: {
            color: #xc45c26
            color_hover: #xd46a32
            color_down: #xa44c1e
            color_focus: #xc45c26
            border_radius: 6.0
            border_size: 0.0
        }
        draw_text +: {
            color: #xffffff
            color_hover: #xffffff
            color_down: #xffffff
            color_focus: #xffffff
            text_style: theme.font_bold{font_size: 9.5}
        }
    }

    let Field = TextInput{
        height: 28
        draw_bg +: {
            color: #x1b1e27
            border_radius: 4.0
        }
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_regular{font_size: 9.5}
        }
    }

    let Catalogue = DataGrid{
        width: Fill
        height: Fill
        rows: 0
        cols: 8
        default_col_width: 140.0
        default_row_height: 52.0
        col_header_height: 26.0
        row_header_width: 0.0
        cell_pad_x: 8.0
        zebra_stripes: true
        color_bg: #x0c0d12
        color_cell: #x0c0d12
        color_cell_alt: #x101219
        color_text: #xf2f4f8
        color_header: #x14161d
        color_header_active: #x1b1e27
        color_header_text: #x9aa7b4
        color_selection: #xc45c2633
        color_selection_border: #xc45c26
        color_drag_marker: #xc45c26
        color_resize_guide: #xc45c2644
        scroll_bar_h: mod.widgets.ScrollBar{
            draw_bg +: {
                color: uniform(#xffffff26)
                color_hover: uniform(#xffffff42)
                color_drag: uniform(#xffffff66)
            }
        }
        scroll_bar_v: mod.widgets.ScrollBar{
            draw_bg +: {
                color: uniform(#xffffff26)
                color_hover: uniform(#xffffff42)
                color_drag: uniform(#xffffff66)
            }
        }
        ThumbCell := View{
            width: Fill
            height: Fill
            align: Align{x: 0.5 y: 0.5}
            padding: Inset{left: 4, right: 4, top: 4, bottom: 4}
            thumb := Image{
                width: 44
                height: 44
                fit: ImageFit.Smallest
            }
        }
        DiscogsLink := View{
            width: Fill
            height: Fill
            align: Align{x: 0.0 y: 0.5}
            padding: Inset{left: 8, right: 4}
            link := LinkLabel{
                width: Fit
                height: Fit
                margin: 0
                padding: 0
                draw_text +: {
                    color: #xc45c26
                    color_hover: #xd46a32
                    color_down: #xa44c1e
                    color_focus: #xc45c26
                    text_style: theme.font_regular{font_size: 9.5}
                }
                draw_bg +: {
                    color: #xc45c26
                    color_hover: #xd46a32
                    color_down: #xa44c1e
                    color_focus: #xc45c26
                }
            }
        }
    }

    mod.widgets.Vinyl = mod.widgets.VinylBase{
        width: Fill
        height: Fill
        flow: Down
        draw_bg +: { color: #x0c0d12 }

        header := Panel{
            width: Fill
            height: Fit
            flow: Down
            padding: Inset{left: 18, right: 18, top: 14, bottom: 10}
            spacing: 4
            Title{ text: "Vinyl catalogue" }
            status := Dim{ text: "starting…" }
        }

        capture := Panel{
            width: Fill
            height: Fit
            flow: Right
            padding: Inset{left: 18, right: 18, top: 8, bottom: 8}
            spacing: 10
            align: Align{y: 0.5}
            capture_wave := VoiceWave{
                width: 48
                height: 48
                margin: 0
            }
            cat_quick := Field{
                width: Fill
                empty_text: "Catalogue number — type and Enter to add & search"
            }
            process_btn := Primary{ text: "Process" }
            add_btn := Chip{ text: "Add row" }
            discogs_btn := Chip{ text: "Discogs" }
            prices_btn := Chip{ text: "Prices" }
            export_btn := Chip{ text: "Export CSV" }
            delete_btn := Chip{ text: "Delete" }
        }

        transcript_wrap := View{
            width: Fill
            height: 72
            padding: Inset{left: 18, right: 18, bottom: 8}
            transcript := Field{
                width: Fill
                height: Fill
                empty_text: "Heard text lands here — arm the mic, or paste and Process"
            }
        }

        records_grid := Catalogue{}

        editor := Panel{
            width: Fill
            height: Fit
            flow: Right
            padding: Inset{left: 18, right: 18, top: 8, bottom: 12}
            spacing: 8
            align: Align{y: 0.5}
            edit_artist := Field{ width: Fill empty_text: "Artist" }
            edit_title := Field{ width: Fill empty_text: "Title" }
            edit_label := Field{ width: Fill empty_text: "Label" }
            edit_cat := Field{ width: Fill empty_text: "Cat no." }
            edit_notes := Field{ width: Fill empty_text: "Notes" }
            edit_condition := Field{ width: 90 empty_text: "Cond." }
        }

        confirm_modal := Modal{
            content +: {
                width: 520
                height: Fit
                flow: Down
                show_bg: true
                padding: Inset{left: 16, right: 16, top: 14, bottom: 14}
                spacing: 6
                draw_bg +: { color: #x1b1e27 }
                confirm_title := Title{ text: "Discogs" }
                confirm_head := Dim{ width: Fill text: "" }
                cand0 := View{
                    width: Fill height: Fit flow: Right spacing: 10
                    align: Align{y: 0.5}
                    padding: Inset{top: 4, bottom: 4}
                    cand0_thumb := Image{ width: 52 height: 52 fit: ImageFit.Smallest }
                    cand0_text := Dim{ width: Fill text: "" }
                }
                cand1 := View{
                    width: Fill height: Fit flow: Right spacing: 10
                    align: Align{y: 0.5}
                    padding: Inset{top: 4, bottom: 4}
                    cand1_thumb := Image{ width: 52 height: 52 fit: ImageFit.Smallest }
                    cand1_text := Dim{ width: Fill text: "" }
                }
                cand2 := View{
                    width: Fill height: Fit flow: Right spacing: 10
                    align: Align{y: 0.5}
                    padding: Inset{top: 4, bottom: 4}
                    cand2_thumb := Image{ width: 52 height: 52 fit: ImageFit.Smallest }
                    cand2_text := Dim{ width: Fill text: "" }
                }
                cand3 := View{
                    width: Fill height: Fit flow: Right spacing: 10
                    align: Align{y: 0.5}
                    padding: Inset{top: 4, bottom: 4}
                    cand3_thumb := Image{ width: 52 height: 52 fit: ImageFit.Smallest }
                    cand3_text := Dim{ width: Fill text: "" }
                }
                confirm_hint := Dim{
                    width: Fill
                    text: "Say OK, a number, skip, or unresolved"
                }
                confirm_actions := View{
                    width: Fill
                    height: Fit
                    flow: Right
                    spacing: 8
                    pick1 := Chip{ text: "1" }
                    pick2 := Chip{ text: "2" }
                    pick3 := Chip{ text: "3" }
                    pick4 := Chip{ text: "4" }
                    ok_btn := Primary{ text: "OK" }
                    unresolved_btn := Chip{ text: "Unresolved" }
                    skip_btn := Chip{ text: "Skip" }
                }
            }
        }
    }
}

const COL_LABELS: [&str; 9] = [
    "",
    "#",
    "Artist",
    "Title",
    "Label",
    "Cat",
    "Price",
    "Discogs",
    "Status",
];
const COL_WIDTHS: [f64; 9] = [56.0, 36.0, 130.0, 140.0, 110.0, 80.0, 64.0, 90.0, 80.0];

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum ConfirmKind {
    #[default]
    None,
    Master,
}

#[derive(Script, ScriptHook, Widget)]
pub struct Vinyl {
    #[deref]
    view: View,
    #[rust]
    started: bool,
    #[rust]
    records: Vec<Record>,
    #[rust]
    selected: Option<usize>,
    #[rust]
    syncing: bool,
    #[rust]
    status: String,
    #[rust]
    mic_on: bool,
    #[cfg(feature = "native")]
    #[rust]
    capture_parts: Vec<String>,
    #[cfg(feature = "native")]
    #[rust]
    db: Option<Db>,
    #[cfg(feature = "native")]
    #[rust]
    agent: Option<ChatAgent>,
    #[cfg(feature = "native")]
    #[rust]
    agent_ready: bool,
    #[cfg(feature = "native")]
    #[rust]
    pending_transcript: Option<String>,
    #[cfg(feature = "native")]
    #[rust]
    llm_delta: String,
    #[cfg(feature = "native")]
    #[rust]
    applied_this_turn: bool,
    #[cfg(feature = "native")]
    #[rust]
    busy: bool,
    #[cfg(feature = "native")]
    #[rust]
    capture_one: bool,
    #[cfg(feature = "native")]
    #[rust]
    discogs_rx: ToUIReceiver<DiscogsMsg>,
    #[cfg(feature = "native")]
    #[rust]
    discogs_generation: u64,
    #[cfg(feature = "native")]
    #[rust]
    discogs_searching: bool,
    #[cfg(feature = "native")]
    #[rust]
    discogs_record_id: i64,
    #[cfg(feature = "native")]
    #[rust]
    confirm_kind: ConfirmKind,
    #[cfg(feature = "native")]
    #[rust]
    masters: Vec<MasterCandidate>,
    #[cfg(feature = "native")]
    #[rust]
    confirm_heading: String,
    #[cfg(feature = "native")]
    #[rust]
    pending_confirm: Option<VoiceCmd>,
    #[cfg(feature = "native")]
    #[rust]
    pending_search_ids: Vec<i64>,
}

#[cfg(feature = "native")]
enum DiscogsMsg {
    Masters {
        generation: u64,
        record_id: i64,
        heading: String,
        result: Result<Vec<MasterCandidate>, String>,
    },
    BatchPrices {
        generation: u64,
        /// True when Discogs returned condition suggestions (seller settings OK).
        suggestions_ok: bool,
        results: Vec<(i64, Result<VersionsBundle, String>)>,
    },
}

#[cfg(feature = "native")]
#[derive(Clone, Copy)]
enum VoiceCmd {
    Ok,
    Pick(usize),
    Skip,
    Unresolved,
}

impl Vinyl {
    fn now() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    fn set_status(&mut self, cx: &mut Cx, text: impl Into<String>) {
        self.status = text.into();
        self.label(cx, ids!(status)).set_text(cx, &self.status);
    }

    fn refresh_status_counts(&mut self, cx: &mut Cx) {
        #[cfg(feature = "native")]
        {
            let whisper = if crate::weights::whisper_ready() {
                "Whisper ready"
            } else {
                "Whisper missing — paste a transcript"
            };
            let qwen = if self.agent_ready {
                "Qwen loaded"
            } else if crate::llm::model_path().is_some() {
                "Qwen on disk — Process loads it"
            } else {
                "Qwen missing — Process uses next/comma split"
            };
            let loop_state = if self.confirm_kind != ConfirmKind::None {
                " · confirming"
            } else if self.mic_on {
                " · armed — speak a record"
            } else {
                ""
            };
            self.set_status(
                cx,
                format!(
                    "{whisper} · {qwen} · {} records{loop_state}",
                    self.records.len()
                ),
            );
            return;
        }
        #[cfg(not(feature = "native"))]
        {
            self.set_status(cx, format!("{} records", self.records.len()));
        }
    }

    fn start(&mut self, cx: &mut Cx) {
        if self.started {
            return;
        }
        self.started = true;
        #[cfg(feature = "native")]
        crate::weights::ensure_whisper();
        #[cfg(feature = "native")]
        {
            let dir = crate::discogs::vinyl_dir();
            if let Err(error) = std::fs::create_dir_all(&dir) {
                self.set_status(cx, format!("cannot create {}: {error}", dir.display()));
                return;
            }
            let _ = std::fs::create_dir_all(crate::discogs::thumbs_dir());
            let path = dir.join("vinyl.db");
            match Db::open(&path) {
                Ok(mut db) => {
                    match db.load() {
                        Ok(records) => self.records = records,
                        Err(error) => self.status = format!("load failed: {error}"),
                    }
                    self.db = Some(db);
                    if self.status.is_empty() {
                        self.refresh_status_counts(cx);
                    } else {
                        self.label(cx, ids!(status)).set_text(cx, &self.status);
                    }
                }
                Err(error) => self.set_status(cx, format!("cannot open {}: {error}", path.display())),
            }
        }
        #[cfg(not(feature = "native"))]
        {
            self.set_status(cx, "native AI/SQLite disabled");
        }
        if self.status.is_empty() {
            self.refresh_status_counts(cx);
        }
        self.text_input(cx, ids!(cat_quick)).take_key_focus(cx);
    }

    fn next_number(&self) -> i64 {
        self.records.iter().map(|r| r.record_number).max().unwrap_or(0) + 1
    }

    fn persist_insert(&mut self, record: &mut Record) -> Result<(), String> {
        #[cfg(feature = "native")]
        {
            if let Some(db) = &mut self.db {
                return db.insert(record);
            }
            return Err("no database".into());
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = record;
            Ok(())
        }
    }

    fn persist_update(&mut self, record: &Record) -> Result<(), String> {
        #[cfg(feature = "native")]
        {
            if let Some(db) = &mut self.db {
                return db.update(record);
            }
            return Err("no database".into());
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = record;
            Ok(())
        }
    }

    fn persist_delete(&mut self, id: i64) -> Result<(), String> {
        #[cfg(feature = "native")]
        {
            if let Some(db) = &mut self.db {
                return db.delete(id);
            }
            return Err("no database".into());
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = id;
            Ok(())
        }
    }

    fn apply_extracted(&mut self, cx: &mut Cx, rows: Vec<Extracted>, raw: &str, via: &str) {
        let now = Self::now();
        let mut number = self.next_number();
        let mut added = 0;
        let mut last_id = None;
        for row in rows {
            if row.is_empty() {
                continue;
            }
            let mut record = Record::blank(number, now);
            record.raw_transcript = raw.to_string();
            record.artist = row.artist;
            record.title = row.title;
            record.label = row.label;
            record.catalogue_number = row.catalogue_number;
            record.notes = row.notes;
            record.status = "structured".to_string();
            match self.persist_insert(&mut record) {
                Ok(()) => {
                    last_id = Some(record.id);
                    self.records.push(record);
                    number += 1;
                    added += 1;
                }
                Err(error) => {
                    self.set_status(cx, format!("insert failed: {error}"));
                    return;
                }
            }
        }
        if added > 0 {
            self.selected = Some(self.records.len() - 1);
            self.sync_editor(cx);
            self.set_status(
                cx,
                format!("{via}: added {added} · {} in catalogue", self.records.len()),
            );
            #[cfg(feature = "native")]
            if self.capture_one {
                self.capture_one = false;
                if let Some(id) = last_id {
                    self.start_master_search_for(cx, id);
                }
            }
        } else {
            self.capture_one = false;
            self.set_status(cx, format!("{via}: no rows — check the transcript"));
        }
        self.redraw(cx);
    }

    fn process_transcript(&mut self, cx: &mut Cx) {
        let transcript = self.widget(cx, ids!(transcript)).text();
        let transcript = transcript.trim().to_string();
        if transcript.is_empty() {
            self.set_status(cx, "Nothing to process — speak or paste a transcript");
            return;
        }
        #[cfg(feature = "native")]
        {
            if self.busy {
                self.set_status(cx, "Already processing");
                return;
            }
            if let Some(path) = crate::llm::model_path() {
                self.busy = true;
                self.pending_transcript = Some(transcript);
                self.llm_delta.clear();
                self.applied_this_turn = false;
                if self.agent.is_none() {
                    self.set_status(cx, "Loading Qwen…");
                    self.agent = Some(ChatAgent::start(path));
                    return;
                }
                if self.agent_ready {
                    self.send_pending(cx);
                } else {
                    self.set_status(cx, "Waiting for Qwen…");
                }
                return;
            }
        }
        let rows = extract::naive_split(&transcript);
        self.apply_extracted(cx, rows, &transcript, "comma/next split");
    }

    #[cfg(feature = "native")]
    fn process_capture_utterance(&mut self, cx: &mut Cx, spoken: &str) {
        let spoken = spoken.trim();
        if spoken.is_empty() {
            return;
        }
        self.append_transcript(cx, spoken);
        if self.confirm_kind != ConfirmKind::None {
            if let Some(cmd) = parse_voice_command(spoken) {
                self.pending_confirm = None;
                self.apply_voice_cmd(cx, cmd);
            } else {
                self.set_status(cx, "Confirm with OK, a number, skip, or unresolved");
            }
            return;
        }
        // Spoken while search/LLM is still running — queue confirm cmds for the modal.
        if self.discogs_searching || self.busy {
            if let Some(cmd) = parse_voice_command(spoken) {
                self.pending_confirm = Some(cmd);
                self.set_status(cx, "Got it — applying when the confirm popup is ready");
            } else {
                self.set_status(cx, "Still working — wait, then speak again");
            }
            return;
        }
        self.pending_confirm = None;

        // Field-by-field dictation: assemble artist / title / label across pauses.
        if let Some(assembled) = self.push_capture_part(cx, spoken) {
            self.commit_capture_record(cx, &assembled);
        }
    }

    /// Returns `Some(full line)` when the buffer is ready to search.
    #[cfg(feature = "native")]
    fn push_capture_part(&mut self, cx: &mut Cx, spoken: &str) -> Option<String> {
        let t = normalize_voice_cmd(spoken);
        if matches!(
            t.as_str(),
            "go" | "done" | "search" | "send" | "thats it" | "that's it" | "ready"
        ) {
            if self.capture_parts.is_empty() {
                self.set_status(cx, "Nothing buffered — say artist, then title, then label");
                return None;
            }
            let joined = self.capture_parts.join(", ");
            self.capture_parts.clear();
            return Some(joined);
        }
        if matches!(t.as_str(), "clear" | "reset" | "start over" | "scratch that") {
            self.capture_parts.clear();
            self.set_status(cx, "Cleared — say artist, then title, then label");
            return None;
        }

        // One utterance already looks like artist, title, label[, cat].
        let commas = spoken.matches(',').count();
        if commas >= 2 {
            self.capture_parts.clear();
            return Some(spoken.trim().trim_end_matches('.').to_string());
        }

        self.capture_parts.push(spoken.trim().trim_end_matches('.').to_string());
        if self.capture_parts.len() >= 3 {
            let joined = self.capture_parts.join(", ");
            self.capture_parts.clear();
            return Some(joined);
        }

        let preview = self.capture_parts.join(" · ");
        let need = match self.capture_parts.len() {
            1 => "title",
            2 => "label (or say go)",
            _ => "next field",
        };
        self.set_status(cx, format!("Heard: {preview} — now say {need}"));
        None
    }

    #[cfg(feature = "native")]
    fn commit_capture_record(&mut self, cx: &mut Cx, spoken: &str) {
        self.capture_one = true;
        if self.agent_ready {
            if crate::llm::model_path().is_some() {
                self.busy = true;
                self.pending_transcript = Some(spoken.to_string());
                self.llm_delta.clear();
                self.applied_this_turn = false;
                self.send_pending(cx);
                return;
            }
        }
        if let Some(row) = extract::one_record(spoken) {
            self.apply_extracted(cx, vec![row], spoken, "spoken");
        } else {
            self.capture_one = false;
            self.set_status(cx, "Could not parse that utterance");
        }
    }

    #[cfg(feature = "native")]
    fn send_pending(&mut self, cx: &mut Cx) {
        let Some(text) = self.pending_transcript.clone() else { return };
        self.set_status(cx, "Extracting records…");
        if let Some(agent) = &self.agent {
            agent.send_user_turn(format!(
                "Transcript:\n{text}\n\nJSON array of records. Keys: artist, title, label, catalogue_number, notes."
            ));
        }
    }

    #[cfg(feature = "native")]
    fn on_chat_event(&mut self, cx: &mut Cx, event: ChatEvent) {
        match event {
            ChatEvent::Loading { phase, fraction } => {
                self.set_status(cx, format!("loading — {phase} {:.0}%", fraction * 100.0));
            }
            ChatEvent::Ready { secs, .. } => {
                self.agent_ready = true;
                self.set_status(cx, format!("Qwen ready in {secs:.1}s"));
                if self.pending_transcript.is_some() {
                    self.send_pending(cx);
                }
            }
            ChatEvent::Failed(error) => {
                self.agent = None;
                self.agent_ready = false;
                self.busy = false;
                self.capture_one = false;
                self.pending_transcript = None;
                self.set_status(cx, format!("Qwen failed ({error}) — weights found but did not load"));
            }
            ChatEvent::Delta(text) => {
                self.llm_delta.push_str(&text);
            }
            ChatEvent::ToolCall { name, args } => {
                log!("vinyl tool {name} args={args:?}");
                if !self.applied_this_turn {
                    let raw = self.pending_transcript.clone().unwrap_or_default();
                    let rows = extract::from_llm_payload(&args, &self.llm_delta);
                    if !rows.is_empty() {
                        self.apply_extracted(cx, rows, &raw, "Qwen");
                        self.applied_this_turn = true;
                    }
                }
                if let Some(agent) = &self.agent {
                    agent.send_tool_results(vec![("ok".to_string(), false)]);
                }
            }
            ChatEvent::TurnDone { tool_calls, .. } => {
                if tool_calls > 0 {
                    return;
                }
                if !self.applied_this_turn {
                    let raw = self.pending_transcript.clone().unwrap_or_default();
                    let rows = extract::from_llm_payload(&[], &self.llm_delta);
                    if rows.is_empty() {
                        let preview: String = self.llm_delta.chars().take(180).collect();
                        log!("vinyl Qwen produced no JSON: {preview}");
                        self.capture_one = false;
                        self.set_status(
                            cx,
                            format!(
                                "Qwen did not return JSON — {}",
                                if preview.is_empty() {
                                    "empty reply".to_string()
                                } else {
                                    preview.replace('\n', " ")
                                }
                            ),
                        );
                    } else {
                        self.apply_extracted(cx, rows, &raw, "Qwen");
                    }
                }
                self.pending_transcript = None;
                self.llm_delta.clear();
                self.applied_this_turn = false;
                self.busy = false;
            }
            ChatEvent::ContextFull => {
                self.busy = false;
                self.capture_one = false;
                self.pending_transcript = None;
                self.set_status(cx, "Qwen context full — reopen the app");
            }
        }
    }

    fn sync_editor(&mut self, cx: &mut Cx) {
        self.syncing = true;
        let (artist, title, label, cat, notes, condition) =
            match self.selected.and_then(|i| self.records.get(i)) {
                Some(record) => (
                    record.artist.clone(),
                    record.title.clone(),
                    record.label.clone(),
                    record.catalogue_number.clone(),
                    record.notes.clone(),
                    record.condition.clone(),
                ),
                None => (
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                ),
            };
        self.text_input(cx, ids!(edit_artist)).set_text(cx, &artist);
        self.text_input(cx, ids!(edit_title)).set_text(cx, &title);
        self.text_input(cx, ids!(edit_label)).set_text(cx, &label);
        self.text_input(cx, ids!(edit_cat)).set_text(cx, &cat);
        self.text_input(cx, ids!(edit_notes)).set_text(cx, &notes);
        self.text_input(cx, ids!(edit_condition)).set_text(cx, &condition);
        self.syncing = false;
    }

    fn write_selected_field(&mut self, cx: &mut Cx, field: Field, value: String) {
        if self.syncing {
            return;
        }
        let Some(index) = self.selected else { return };
        let Some(record) = self.records.get_mut(index) else { return };
        match field {
            Field::Artist => record.artist = value,
            Field::Title => record.title = value,
            Field::Label => record.label = value,
            Field::Cat => record.catalogue_number = value,
            Field::Notes => record.notes = value,
            Field::Condition => {
                record.condition = value;
                if let Some(price) = record.shown_price() {
                    record.display_price = price;
                }
            }
        }
        record.updated_at = Self::now();
        let snapshot = record.clone();
        if let Err(error) = self.persist_update(&snapshot) {
            self.set_status(cx, format!("save failed: {error}"));
        }
        self.redraw(cx);
    }

    fn add_row(&mut self, cx: &mut Cx) {
        let mut record = Record::blank(self.next_number(), Self::now());
        match self.persist_insert(&mut record) {
            Ok(()) => {
                self.records.push(record);
                self.selected = Some(self.records.len() - 1);
                self.sync_editor(cx);
                self.refresh_status_counts(cx);
                self.redraw(cx);
            }
            Err(error) => self.set_status(cx, format!("insert failed: {error}")),
        }
    }

    fn add_from_catno(&mut self, cx: &mut Cx, text: &str) {
        let cat = text.trim();
        if cat.is_empty() {
            self.text_input(cx, ids!(cat_quick)).take_key_focus(cx);
            return;
        }
        let mut record = Record::blank(self.next_number(), Self::now());
        record.catalogue_number = cat.to_string();
        record.status = "captured".to_string();
        match self.persist_insert(&mut record) {
            Ok(()) => {
                let id = record.id;
                let cat = record.catalogue_number.clone();
                self.records.push(record);
                self.selected = Some(self.records.len() - 1);
                self.sync_editor(cx);
                let field = self.text_input(cx, ids!(cat_quick));
                field.set_text(cx, "");
                field.take_key_focus(cx);
                #[cfg(feature = "native")]
                self.enqueue_discogs_search(cx, id, &cat);
                #[cfg(not(feature = "native"))]
                self.set_status(cx, format!("Added {cat}"));
                self.redraw(cx);
            }
            Err(error) => {
                self.set_status(cx, format!("insert failed: {error}"));
                self.text_input(cx, ids!(cat_quick)).take_key_focus(cx);
            }
        }
    }

    fn focus_cat_quick_if_idle(&mut self, cx: &mut Cx) {
        let field = self.text_input(cx, ids!(cat_quick));
        if field.text().trim().is_empty() {
            field.take_key_focus(cx);
        }
    }

    fn release_cat_quick_if_idle(&mut self, cx: &mut Cx) {
        let field = self.text_input(cx, ids!(cat_quick));
        if field.text().trim().is_empty() && field.key_focus(cx) {
            cx.set_key_focus(Area::Empty);
        }
    }

    #[cfg(feature = "native")]
    fn enqueue_discogs_search(&mut self, cx: &mut Cx, record_id: i64, cat: &str) {
        if self.discogs_searching || self.confirm_kind != ConfirmKind::None {
            self.pending_search_ids.push(record_id);
            let n = self.pending_search_ids.len();
            self.set_status(cx, format!("Added {cat} — queued ({n} waiting)"));
            return;
        }
        self.start_master_search_for(cx, record_id);
    }

    #[cfg(feature = "native")]
    fn kick_pending_search(&mut self, cx: &mut Cx) {
        if self.discogs_searching || self.confirm_kind != ConfirmKind::None {
            return;
        }
        while !self.pending_search_ids.is_empty() {
            let id = self.pending_search_ids.remove(0);
            if !self.records.iter().any(|r| r.id == id) {
                continue;
            }
            self.start_master_search_for(cx, id);
            if self.discogs_searching || self.confirm_kind != ConfirmKind::None {
                return;
            }
        }
    }

    fn delete_selected(&mut self, cx: &mut Cx) {
        let Some(index) = self.selected else {
            self.set_status(cx, "Select a row to delete");
            return;
        };
        let id = self.records[index].id;
        match self.persist_delete(id) {
            Ok(()) => {
                self.records.remove(index);
                self.selected = if self.records.is_empty() {
                    None
                } else {
                    Some(index.min(self.records.len() - 1))
                };
                self.sync_editor(cx);
                self.refresh_status_counts(cx);
                self.redraw(cx);
            }
            Err(error) => self.set_status(cx, format!("delete failed: {error}")),
        }
    }

    fn export_csv(&mut self, cx: &mut Cx) {
        #[cfg(feature = "native")]
        {
            let mut dialog = FileDialog::new()
                .set_id(live_id!(export_csv))
                .set_title("Export catalogue CSV".into())
                .set_filename("vinyl-catalogue.csv".into())
                .add_filter("CSV".into(), vec!["csv".into()]);
            if let Some(downloads) = downloads_dir() {
                dialog = dialog.set_location(downloads);
            }
            cx.open_save_file_dialog(dialog);
            self.set_status(cx, "Choose where to save the CSV…");
            return;
        }
        #[cfg(not(feature = "native"))]
        {
            self.write_catalogue_csv(cx, Path::new("local/vinyl/catalogue.csv"));
        }
    }

    fn write_catalogue_csv(&mut self, cx: &mut Cx, path: &Path) {
        let path = ensure_csv_path(path);
        if let Some(dir) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(dir) {
                self.set_status(cx, format!("export failed: {error}"));
                return;
            }
        }
        match std::fs::write(&path, catalogue_csv(&self.records)) {
            Ok(()) => {
                self.set_status(
                    cx,
                    format!("Exported {} rows to {}", self.records.len(), path.display()),
                );
                #[cfg(target_os = "macos")]
                {
                    let _ = std::process::Command::new("open").arg(&path).spawn();
                }
            }
            Err(error) => self.set_status(cx, format!("export failed: {error}")),
        }
    }

    #[cfg(feature = "native")]
    fn start_discogs(&mut self, cx: &mut Cx) {
        let Some(index) = self.selected else {
            self.set_status(cx, "Select a row to search Discogs");
            return;
        };
        // Always search again — matched rows can be rematched (catno-only, wrong pick, etc.).
        // Prices button loads marketplace data for existing masters/releases.
        self.start_master_search_for(cx, self.records[index].id);
    }

    #[cfg(feature = "native")]
    fn start_master_search_for(&mut self, cx: &mut Cx, record_id: i64) {
        let Some(record) = self.records.iter().find(|r| r.id == record_id).cloned() else {
            return;
        };
        if self.discogs_searching {
            self.set_status(cx, "Discogs search already running");
            return;
        }
        let query = discogs::SearchQuery::from_record(&record);
        if discogs::search_attempts(&query).is_empty() {
            self.set_status(cx, "Nothing to search — fill artist, title, or catalogue number");
            return;
        }
        let token = match discogs::load_token() {
            Ok(token) => token,
            Err(error) => {
                self.set_status(cx, error);
                return;
            }
        };
        self.discogs_generation = self.discogs_generation.wrapping_add(1);
        let generation = self.discogs_generation;
        let heading = query.heading();
        let heading_status = heading.clone();
        let sender = self.discogs_rx.sender();
        match cx.task_pool().submit(Lane::Light, move || {
            let result = discogs::search_masters(&query, &token);
            let _ = sender.send(DiscogsMsg::Masters {
                generation,
                record_id,
                heading,
                result,
            });
        }) {
            Ok(handle) => {
                handle.detach();
                self.discogs_searching = true;
                self.set_status(cx, format!("Discogs: searching masters for {heading_status}…"));
            }
            Err(error) => self.set_status(cx, format!("Discogs: could not queue search ({error})")),
        }
    }

    /// Fill VG prices for matched rows. Uses the grid selection when present
    /// (shift/drag multi-row), otherwise the current row; if neither, every
    /// row that already has a Discogs master/release.
    #[cfg(feature = "native")]
    fn start_batch_prices(&mut self, cx: &mut Cx) {
        if self.discogs_searching {
            self.set_status(cx, "Discogs already busy — wait, then Prices again");
            return;
        }

        let selected_rows: Option<Vec<usize>> = {
            let grid = self.data_grid(cx, ids!(records_grid));
            if let Some(sel) = grid.selection() {
                let (r0, r1) = sel.row_range();
                let rows: Vec<usize> = (r0..=r1).filter(|r| *r < self.records.len()).collect();
                if !rows.is_empty() {
                    Some(rows)
                } else {
                    None
                }
            } else {
                self.selected
                    .filter(|i| *i < self.records.len())
                    .map(|i| vec![i])
            }
        };
        let used_selection = selected_rows.is_some();

        let scope_label;
        let jobs: Vec<(i64, i64, i64, String)> = if let Some(rows) = selected_rows {
            scope_label = if rows.len() == 1 {
                "selected row".to_string()
            } else {
                format!("{} selected rows", rows.len())
            };
            rows.into_iter()
                .filter_map(|i| price_job_for(&self.records[i]))
                .collect()
        } else {
            scope_label = "all matched".to_string();
            self.records.iter().filter_map(price_job_for).collect()
        };

        if jobs.is_empty() {
            let msg = if used_selection {
                "Selected row(s) have no Discogs match yet — confirm a master first"
            } else {
                "No masters yet — confirm some masters first"
            };
            self.set_status(cx, msg);
            return;
        }
        let token = match discogs::load_token() {
            Ok(token) => token,
            Err(error) => {
                self.set_status(cx, error);
                return;
            }
        };
        self.discogs_generation = self.discogs_generation.wrapping_add(1);
        let generation = self.discogs_generation;
        let total = jobs.len();
        let sender = self.discogs_rx.sender();
        match cx.task_pool().submit(Lane::Light, move || {
            let seller = discogs::probe_seller_pricing(&token);
            let suggestions_ok = matches!(seller, discogs::SellerPricing::Ready);
            if matches!(seller, discogs::SellerPricing::NeedsSetup) {
                discogs::open_seller_settings();
            }
            let mut results = Vec::with_capacity(jobs.len());
            for (i, (record_id, release_id, master_id, catno)) in jobs.into_iter().enumerate() {
                if i > 0 || !suggestions_ok {
                    // Probe already hit the API; always gap before first price row too
                    // when we probed (except Ready path still needs gap after probe).
                    std::thread::sleep(discogs::RATE_GAP);
                } else if i == 0 {
                    std::thread::sleep(discogs::RATE_GAP);
                }
                results.push((
                    record_id,
                    discogs::load_price_for_row(release_id, master_id, &catno, &token),
                ));
            }
            let _ = sender.send(DiscogsMsg::BatchPrices {
                generation,
                suggestions_ok,
                results,
            });
        }) {
            Ok(handle) => {
                handle.detach();
                self.discogs_searching = true;
                let status = if total == 1 {
                    "Prices: looking up…".to_string()
                } else {
                    format!("Prices: looking up {total} ({scope_label})…")
                };
                self.set_status(cx, status);
            }
            Err(error) => self.set_status(cx, format!("Prices: could not queue ({error})")),
        }
    }

    #[cfg(feature = "native")]
    fn drain_discogs(&mut self, cx: &mut Cx) {
        loop {
            let Ok(msg) = self.discogs_rx.try_recv() else { break };
            match msg {
                DiscogsMsg::Masters {
                    generation,
                    record_id,
                    heading,
                    result,
                } => {
                    if generation != self.discogs_generation {
                        continue;
                    }
                    self.discogs_searching = false;
                    self.discogs_record_id = record_id;
                    match result {
                        Ok(masters) => {
                            self.masters = masters;
                            self.show_master_modal(cx, &heading);
                        }
                        Err(error) => {
                            self.close_confirm(cx);
                            if !self.discogs_searching {
                                self.set_status(cx, error);
                            }
                        }
                    }
                }
                DiscogsMsg::BatchPrices {
                    generation,
                    suggestions_ok,
                    results,
                } => {
                    if generation != self.discogs_generation {
                        continue;
                    }
                    self.discogs_searching = false;
                    self.apply_batch_prices(cx, suggestions_ok, results);
                }
            }
        }
    }

    #[cfg(feature = "native")]
    fn apply_batch_prices(
        &mut self,
        cx: &mut Cx,
        suggestions_ok: bool,
        results: Vec<(i64, Result<VersionsBundle, String>)>,
    ) {
        let mut ok = 0usize;
        let mut high = 0usize;
        let mut err = 0usize;
        let lows = results
            .iter()
            .filter(|(_, r)| {
                r.as_ref()
                    .ok()
                    .and_then(|b| b.versions.first())
                    .map(|v| v.suggestions_json.is_empty() && v.vg_price > 0.0)
                    .unwrap_or(false)
            })
            .count();
        for (record_id, result) in results {
            let Some(pos) = self.records.iter().position(|r| r.id == record_id) else {
                continue;
            };
            match result {
                Ok(bundle) => {
                    let record = &mut self.records[pos];
                    if record.thumb_path.is_empty() && !bundle.master_thumb_path.is_empty() {
                        record.thumb_path = bundle.master_thumb_path.clone();
                    }
                    // Persist the specific release we priced (vinyl resolve from master).
                    if let Some(ver) = bundle.versions.first() {
                        if ver.id > 0 {
                            record.discogs_release_id = ver.id.to_string();
                            record.discogs_url =
                                format!("https://www.discogs.com/release/{}", ver.id);
                        }
                    }
                    if bundle.master_id > 0 && record.discogs_master_id.trim().is_empty() {
                        record.discogs_master_id = bundle.master_id.to_string();
                        record.discogs_master_url =
                            format!("https://www.discogs.com/master/{}", bundle.master_id);
                    }
                    if bundle.master_vg > 0.0 {
                        record.display_price = bundle.master_vg;
                    }
                    if let Some(ver) = bundle.versions.first() {
                        record.price_suggestions = ver.suggestions_json.clone();
                    }
                    if bundle.has_high_value() {
                        record.status = "high_value".to_string();
                        high += 1;
                    } else if record.status == "matched" || record.status.is_empty() {
                        record.status = "matched".to_string();
                    }
                    record.updated_at = Self::now();
                    let snapshot = record.clone();
                    if self.persist_update(&snapshot).is_ok() {
                        ok += 1;
                    } else {
                        err += 1;
                    }
                }
                Err(_) => err += 1,
            }
        }
        self.sync_editor(cx);
        let msg = if !suggestions_ok {
            format!(
                "Prices: {ok} updated as marketplace lows · fill Discogs seller settings for VG — {}",
                discogs::SELLER_SETTINGS_URL
            )
        } else if lows > 0 {
            format!("Prices: {ok} updated · {high} high-value · {lows} floor-only · {err} failed")
        } else {
            format!("Prices: {ok} updated · {high} high-value · {err} failed")
        };
        self.set_status(cx, msg);
        self.redraw(cx);
    }

    #[cfg(feature = "native")]
    fn show_master_modal(&mut self, cx: &mut Cx, heading: &str) {
        self.confirm_kind = ConfirmKind::Master;
        self.confirm_heading = heading.to_string();
        let n = self.masters.len();
        let head = if n == 0 {
            format!("{heading} — no Discogs masters. Say unresolved or skip.")
        } else if n == 1 {
            format!("{heading} — say OK to confirm, or skip")
        } else {
            format!("{heading} — say OK for #1, or 1–{n}, or skip")
        };
        self.label(cx, ids!(confirm_title)).set_text(cx, "Confirm release");
        self.label(cx, ids!(confirm_head)).set_text(cx, &head);
        self.fill_candidate_lines(cx);
        self.button(cx, ids!(ok_btn)).set_visible(cx, n >= 1);
        self.modal(cx, ids!(confirm_modal)).open(cx);
        self.update_phrase_mode(cx);
        self.release_cat_quick_if_idle(cx);
        self.set_status(cx, head);
        self.redraw(cx);
        self.flush_pending_confirm(cx);
    }

    #[cfg(feature = "native")]
    fn update_phrase_mode(&mut self, cx: &mut Cx) {
        let mode = if self.confirm_kind != ConfirmKind::None {
            PhraseMode::Quick
        } else if self.mic_on {
            PhraseMode::Dictation
        } else {
            PhraseMode::Normal
        };
        self.voice_wave(cx, ids!(capture_wave))
            .set_phrase_mode(cx, mode);
    }

    #[cfg(feature = "native")]
    fn flush_pending_confirm(&mut self, cx: &mut Cx) {
        if let Some(cmd) = self.pending_confirm.take() {
            self.apply_voice_cmd(cx, cmd);
        }
    }

    #[cfg(feature = "native")]
    fn fill_candidate_lines(&mut self, cx: &mut Cx) {
        let rows = [
            (ids!(cand0), ids!(cand0_text), ids!(cand0_thumb)),
            (ids!(cand1), ids!(cand1_text), ids!(cand1_thumb)),
            (ids!(cand2), ids!(cand2_text), ids!(cand2_thumb)),
            (ids!(cand3), ids!(cand3_text), ids!(cand3_thumb)),
        ];
        let picks = [ids!(pick1), ids!(pick2), ids!(pick3), ids!(pick4)];
        let n = self.masters.len().min(4);
        for (i, (row, text_id, thumb_id)) in rows.iter().enumerate() {
            let visible = i < n;
            self.view(cx, *row).set_visible(cx, visible);
            if !visible {
                continue;
            }
            let Some(cand) = self.masters.get(i) else { continue };
            self.label(cx, *text_id).set_text(cx, &cand.summary(i));
            let img = self.image(cx, *thumb_id);
            if !cand.thumb_path.is_empty() {
                let path = std::path::Path::new(&cand.thumb_path);
                let _ = img.load_image_file_by_path(cx, path);
                img.set_visible(cx, true);
            } else {
                img.set_visible(cx, false);
            }
        }
        for (i, id) in picks.iter().enumerate() {
            self.button(cx, *id).set_visible(cx, i < n);
        }
    }

    #[cfg(feature = "native")]
    fn close_confirm(&mut self, cx: &mut Cx) {
        self.confirm_kind = ConfirmKind::None;
        self.masters.clear();
        self.pending_confirm = None;
        self.update_phrase_mode(cx);
        self.modal(cx, ids!(confirm_modal)).close(cx);
        self.redraw(cx);
        self.focus_cat_quick_if_idle(cx);
        self.kick_pending_search(cx);
    }

    #[cfg(feature = "native")]
    fn apply_voice_cmd(&mut self, cx: &mut Cx, cmd: VoiceCmd) {
        match cmd {
            VoiceCmd::Ok => {
                if self.confirm_kind == ConfirmKind::Master && !self.masters.is_empty() {
                    self.pick_master(cx, 0);
                }
            }
            VoiceCmd::Pick(i) => {
                if self.confirm_kind == ConfirmKind::Master {
                    self.pick_master(cx, i);
                }
            }
            VoiceCmd::Skip => {
                self.set_status(cx, "Skipped — type the next cat number or speak a record");
                self.close_confirm(cx);
            }
            VoiceCmd::Unresolved => self.mark_unresolved(cx),
        }
    }

    #[cfg(feature = "native")]
    fn sync_mic_from_wave(&mut self, cx: &mut Cx) {
        let on = self.voice_wave(cx, ids!(capture_wave)).is_enabled();
        if on == self.mic_on {
            return;
        }
        self.mic_on = on;
        if on {
            self.capture_parts.clear();
            self.set_status(cx, "Armed — say artist, title, label (pauses OK); say go when done");
        } else {
            self.capture_parts.clear();
            self.refresh_status_counts(cx);
        }
        self.update_phrase_mode(cx);
    }

    #[cfg(feature = "native")]
    fn pick_master(&mut self, cx: &mut Cx, index: usize) {
        if self.confirm_kind != ConfirmKind::Master {
            return;
        }
        let Some(master) = self.masters.get(index).cloned() else { return };
        let Some(pos) = self.records.iter().position(|r| r.id == self.discogs_record_id) else {
            self.close_confirm(cx);
            self.set_status(cx, "Discogs: row is gone");
            return;
        };
        {
            let record = &mut self.records[pos];
            if master.release_id > 0 || master.is_release {
                let rid = if master.release_id > 0 {
                    master.release_id
                } else {
                    master.id
                };
                record.discogs_release_id = rid.to_string();
                record.discogs_url = format!("https://www.discogs.com/release/{rid}");
            }
            if master.master_id > 0 {
                record.discogs_master_id = master.master_id.to_string();
                record.discogs_master_url =
                    format!("https://www.discogs.com/master/{}", master.master_id);
            } else if !master.is_release {
                record.discogs_master_id = master.id.to_string();
                record.discogs_master_url = master.url();
                // Master-only confirm — Prices will resolve a vinyl pressing later.
            } else {
                record.discogs_master_id.clear();
                record.discogs_master_url.clear();
            }
            if !master.thumb_path.is_empty() {
                record.thumb_path = master.thumb_path.clone();
            }
            // Discogs titles are "Artist - Title" — fill blanks (catno-only search).
            let (disc_artist, disc_title) = discogs::split_discogs_title(&master.title);
            if record.artist.trim().is_empty() && !disc_artist.is_empty() {
                record.artist = disc_artist;
            }
            if record.title.trim().is_empty() {
                if !disc_title.is_empty() {
                    record.title = disc_title;
                } else if !master.title.is_empty() {
                    record.title = master.title.clone();
                }
            }
            if record.label.trim().is_empty() && !master.label.is_empty() {
                record.label = master.label.clone();
            }
            if record.catalogue_number.trim().is_empty() && !master.catno.is_empty() {
                record.catalogue_number = master.catno.clone();
            }
            record.status = "matched".to_string();
            record.updated_at = Self::now();
            let snapshot = record.clone();
            if let Err(error) = self.persist_update(&snapshot) {
                self.set_status(cx, format!("save failed: {error}"));
                return;
            }
        }
        self.selected = Some(pos);
        self.sync_editor(cx);
        let kind = if master.release_id > 0 || master.is_release {
            "Release"
        } else {
            "Master"
        };
        self.set_status(
            cx,
            format!("{kind} {} · type the next cat number (Prices fills marketplace later)", master.id),
        );
        self.close_confirm(cx);
        self.redraw(cx);
    }

    #[cfg(feature = "native")]
    fn mark_unresolved(&mut self, cx: &mut Cx) {
        if self.confirm_kind == ConfirmKind::None {
            return;
        }
        let Some(pos) = self.records.iter().position(|r| r.id == self.discogs_record_id) else {
            self.close_confirm(cx);
            return;
        };
        {
            let record = &mut self.records[pos];
            record.status = "unresolved".to_string();
            record.updated_at = Self::now();
            let snapshot = record.clone();
            if let Err(error) = self.persist_update(&snapshot) {
                self.set_status(cx, format!("save failed: {error}"));
                return;
            }
        }
        self.selected = Some(pos);
        self.sync_editor(cx);
        self.set_status(cx, "Unresolved — type the next cat number or speak a record");
        self.close_confirm(cx);
        self.redraw(cx);
    }

    #[cfg(feature = "native")]
    fn discogs_key(&mut self, cx: &mut Cx, event: &KeyEvent) {
        if self.confirm_kind == ConfirmKind::None || event.is_repeat || event.modifiers.any() {
            return;
        }
        if self.typing_in_field(cx) {
            return;
        }
        match event.key_code {
            KeyCode::ReturnKey => self.apply_voice_cmd(cx, VoiceCmd::Ok),
            KeyCode::Key1 | KeyCode::Numpad1 => self.apply_voice_cmd(cx, VoiceCmd::Pick(0)),
            KeyCode::Key2 | KeyCode::Numpad2 => self.apply_voice_cmd(cx, VoiceCmd::Pick(1)),
            KeyCode::Key3 | KeyCode::Numpad3 => self.apply_voice_cmd(cx, VoiceCmd::Pick(2)),
            KeyCode::Key4 | KeyCode::Numpad4 => self.apply_voice_cmd(cx, VoiceCmd::Pick(3)),
            KeyCode::Key5 | KeyCode::Numpad5 => self.apply_voice_cmd(cx, VoiceCmd::Pick(4)),
            KeyCode::KeyU => self.apply_voice_cmd(cx, VoiceCmd::Unresolved),
            KeyCode::Escape => self.apply_voice_cmd(cx, VoiceCmd::Skip),
            _ => {}
        }
    }

    #[cfg(feature = "native")]
    fn typing_in_field(&self, cx: &Cx) -> bool {
        self.text_input(cx, ids!(transcript)).key_focus(cx)
            || self.text_input(cx, ids!(cat_quick)).key_focus(cx)
            || self.text_input(cx, ids!(edit_artist)).key_focus(cx)
            || self.text_input(cx, ids!(edit_title)).key_focus(cx)
            || self.text_input(cx, ids!(edit_label)).key_focus(cx)
            || self.text_input(cx, ids!(edit_cat)).key_focus(cx)
            || self.text_input(cx, ids!(edit_notes)).key_focus(cx)
            || self.text_input(cx, ids!(edit_condition)).key_focus(cx)
    }

    fn append_transcript(&mut self, cx: &mut Cx, spoken: &str) {
        let spoken = spoken.trim();
        if spoken.is_empty() {
            return;
        }
        let mut text = self.widget(cx, ids!(transcript)).text();
        if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
            text.push(' ');
        }
        text.push_str(spoken);
        self.widget(cx, ids!(transcript)).set_text(cx, &text);
    }

    fn cell_text(&self, row: usize, col: usize) -> String {
        let Some(record) = self.records.get(row) else {
            return String::new();
        };
        match col {
            1 => record.record_number.to_string(),
            2 => record.artist.clone(),
            3 => record.title.clone(),
            4 => record.label.clone(),
            5 => record.catalogue_number.clone(),
            6 => record.shown_price_text(),
            7 => discogs_link_text(record),
            8 => record.status.clone(),
            _ => String::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum Field {
    Artist,
    Title,
    Label,
    Cat,
    Notes,
    Condition,
}

fn discogs_page_url(record: &Record) -> Option<&str> {
    let release = record.discogs_url.trim();
    if !release.is_empty() {
        return Some(release);
    }
    let master = record.discogs_master_url.trim();
    if !master.is_empty() {
        return Some(master);
    }
    None
}

fn discogs_link_text(record: &Record) -> String {
    if !record.discogs_release_id.is_empty() {
        format!("R {}", record.discogs_release_id)
    } else if !record.discogs_master_id.is_empty() {
        format!("M {}", record.discogs_master_id)
    } else {
        String::new()
    }
}

fn downloads_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let downloads = PathBuf::from(home).join("Downloads");
    downloads.is_dir().then_some(downloads)
}

fn ensure_csv_path(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("csv") => path.to_path_buf(),
        _ => path.with_extension("csv"),
    }
}

fn catalogue_csv(records: &[Record]) -> String {
    let mut out = String::from(
        "#,artist,title,label,catalogue_number,notes,condition,price,discogs,status\n",
    );
    for record in records {
        let discogs = discogs_page_url(record).unwrap_or("");
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            record.record_number,
            csv_field(&record.artist),
            csv_field(&record.title),
            csv_field(&record.label),
            csv_field(&record.catalogue_number),
            csv_field(&record.notes),
            csv_field(&record.condition),
            csv_field(&record.shown_price_text()),
            csv_field(discogs),
            csv_field(&record.status),
        ));
    }
    out
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(feature = "native")]
fn price_job_for(r: &Record) -> Option<(i64, i64, i64, String)> {
    // (record_id, release_id, master_id, catno)
    let release_id = r.discogs_release_id.trim().parse::<i64>().ok().filter(|n| *n > 0);
    let master_id = r.discogs_master_id.trim().parse::<i64>().ok().filter(|n| *n > 0);
    match (release_id, master_id) {
        (Some(rid), mid) => Some((r.id, rid, mid.unwrap_or(0), r.catalogue_number.clone())),
        (None, Some(mid)) => Some((r.id, 0, mid, r.catalogue_number.clone())),
        (None, None) => None,
    }
}

#[cfg(feature = "native")]
fn normalize_voice_cmd(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '\'' {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    out.trim().to_string()
}

#[cfg(feature = "native")]
fn parse_voice_command_exact(t: &str) -> Option<VoiceCmd> {
    match t {
        "ok" | "okay" | "yes" | "yep" | "confirm" | "alright" | "all right" => Some(VoiceCmd::Ok),
        "skip" | "next" | "pass" | "cancel" => Some(VoiceCmd::Skip),
        "unresolved" | "unknown" | "none" => Some(VoiceCmd::Unresolved),
        "1" | "one" => Some(VoiceCmd::Pick(0)),
        "2" | "two" | "to" | "too" => Some(VoiceCmd::Pick(1)),
        "3" | "three" | "tree" => Some(VoiceCmd::Pick(2)),
        "4" | "four" | "for" => Some(VoiceCmd::Pick(3)),
        "5" | "five" => Some(VoiceCmd::Pick(4)),
        "6" | "six" => Some(VoiceCmd::Pick(5)),
        "7" | "seven" => Some(VoiceCmd::Pick(6)),
        "8" | "eight" => Some(VoiceCmd::Pick(7)),
        _ if is_ok_utterance(t) => Some(VoiceCmd::Ok),
        _ => None,
    }
}

/// Parse a confirm utterance. Whisper often glues turns ("One. Okay.") — prefer
/// the last number pick if any, otherwise the last other command word.
#[cfg(feature = "native")]
fn parse_voice_command(text: &str) -> Option<VoiceCmd> {
    let t = normalize_voice_cmd(text);
    if t.is_empty() {
        return None;
    }
    if let Some(cmd) = parse_voice_command_exact(&t) {
        return Some(cmd);
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    let mut last_pick: Option<VoiceCmd> = None;
    let mut last_other: Option<VoiceCmd> = None;
    for i in 0..words.len() {
        if let Some(cmd) = parse_voice_command_exact(words[i]) {
            match cmd {
                VoiceCmd::Pick(_) => last_pick = Some(cmd),
                other => last_other = Some(other),
            }
        }
        if i + 1 < words.len() {
            let phrase = format!("{} {}", words[i], words[i + 1]);
            if let Some(cmd) = parse_voice_command_exact(&phrase) {
                match cmd {
                    VoiceCmd::Pick(_) => last_pick = Some(cmd),
                    other => last_other = Some(other),
                }
            }
        }
    }
    last_pick.or(last_other)
}

#[cfg(feature = "native")]
fn is_ok_utterance(t: &str) -> bool {
    const PHRASES: &[&str] = &[
        "ok then",
        "okay then",
        "that's fine",
        "thats fine",
        "that is fine",
        "sounds good",
        "looks good",
        "go ahead",
    ];
    if PHRASES.iter().any(|p| t == *p) {
        return true;
    }
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.is_empty() || words.len() > 4 {
        return false;
    }
    const OK: &[&str] = &["ok", "okay", "yes", "yep", "alright", "confirm"];
    const FILLER: &[&str] = &[
        "then", "please", "thanks", "thank", "you", "yeah", "yup", "fine", "good", "thats",
        "that's", "that", "is", "all", "right",
    ];
    let has_ok = words.iter().any(|w| OK.contains(w));
    let all_ok_or_filler = words.iter().all(|w| OK.contains(w) || FILLER.contains(w));
    has_ok && all_ok_or_filler
}

#[cfg(all(test, feature = "native"))]
mod voice_cmd_tests {
    use super::*;

    #[test]
    fn parses_okay_with_repeated_punctuation() {
        assert!(matches!(parse_voice_command("Okay. Okay."), Some(VoiceCmd::Ok)));
        assert!(matches!(parse_voice_command("okay."), Some(VoiceCmd::Ok)));
        assert!(matches!(parse_voice_command("  OK!  "), Some(VoiceCmd::Ok)));
        assert!(matches!(parse_voice_command("ok then"), Some(VoiceCmd::Ok)));
        assert!(matches!(parse_voice_command("that's fine"), Some(VoiceCmd::Ok)));
    }

    #[test]
    fn parses_one_then_okay_as_pick() {
        assert!(matches!(
            parse_voice_command("One. Okay."),
            Some(VoiceCmd::Pick(0))
        ));
        assert!(matches!(
            parse_voice_command("two okay"),
            Some(VoiceCmd::Pick(1))
        ));
    }

    #[test]
    fn does_not_treat_record_as_ok() {
        assert!(parse_voice_command("Moody Man, Don't Be Misled, KDJ.").is_none());
    }
}

impl Widget for Vinyl {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.start(cx);
        while let Some(step) = self.view.draw_walk(cx, scope, walk).step() {
            let grid_ref = step.as_data_grid();
            let Some(mut grid) = grid_ref.borrow_mut() else { continue };
            grid.set_grid_size(self.records.len(), COL_LABELS.len());
            grid.set_col_labels(COL_LABELS.iter().map(|s| s.to_string()).collect());
            for (index, width) in COL_WIDTHS.iter().enumerate() {
                grid.set_col_width(index, *width);
            }
            while let Some(cell) = grid.next_cell(cx) {
                if cell.col == 0 {
                    if let Some(item) = grid.item(cx, cell.row, cell.col, id!(ThumbCell)) {
                        if let Some(record) = self.records.get(cell.row) {
                            if !record.thumb_path.is_empty() {
                                let path = std::path::Path::new(&record.thumb_path);
                                let _ = item.image(cx, ids!(thumb)).load_image_file_by_path(cx, path);
                            }
                        }
                        grid.draw_item(cx, &cell, &item, None);
                    }
                    continue;
                }
                if cell.col == 7 {
                    if let Some(record) = self.records.get(cell.row) {
                        if let Some(url) = discogs_page_url(record) {
                            let label = discogs_link_text(record);
                            if let Some(item) = grid.item(cx, cell.row, cell.col, id!(DiscogsLink)) {
                                let link = item.link_label(cx, ids!(link));
                                link.set_text(cx, &label);
                                link.set_url(url);
                                grid.draw_item(cx, &cell, &item, None);
                            }
                            continue;
                        }
                    }
                }
                let text = self.cell_text(cell.row, cell.col);
                let style = if cell.col == 1 || cell.col == 6 || cell.col == 8 {
                    CellStyle {
                        color: Some(vec4(0.604, 0.655, 0.706, 1.0)),
                        align: if cell.col == 1 || cell.col == 6 { 1.0 } else { 0.0 },
                        ..CellStyle::default()
                    }
                } else {
                    CellStyle::default()
                };
                grid.cell_text_styled(cx, &cell, &text, style);
            }
        }
        DrawStep::done()
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        #[cfg(feature = "native")]
        {
            let events = self.agent.as_ref().map(|a| a.poll()).unwrap_or_default();
            for event in events {
                self.on_chat_event(cx, event);
            }
            self.drain_discogs(cx);
            if let Event::KeyDown(key) = event {
                self.discogs_key(cx, key);
            }
        }
        self.view.handle_event(cx, event, scope);
        self.widget_match_event(cx, event, scope);
        #[cfg(feature = "native")]
        self.sync_mic_from_wave(cx);
    }
}

impl WidgetMatchEvent for Vinyl {
    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions, _scope: &mut Scope) {
        #[cfg(feature = "native")]
        {
            let wave = self.voice_wave(cx, ids!(capture_wave));
            let uid = wave.widget_uid();
            for action in actions.filter_widget_actions_cast::<VoiceWaveAction>(uid) {
                match action {
                    VoiceWaveAction::InjectText(text) => {
                        let armed = self.mic_on
                            || self.voice_wave(cx, ids!(capture_wave)).is_enabled()
                            || self.confirm_kind != ConfirmKind::None;
                        if armed {
                            self.process_capture_utterance(cx, &text);
                        } else {
                            self.append_transcript(cx, &text);
                        }
                    }
                    VoiceWaveAction::RecordVoice(on) => {
                        self.mic_on = on;
                        if on {
                            self.capture_parts.clear();
                            self.set_status(
                                cx,
                                "Armed — say artist, title, label (pauses OK); say go when done",
                            );
                        } else {
                            self.capture_parts.clear();
                            self.set_status(
                                cx,
                                "Mic off — if arm failed, allow Microphone for Terminal in System Settings",
                            );
                        }
                        self.update_phrase_mode(cx);
                    }
                    _ => {}
                }
            }
            if self.modal(cx, ids!(confirm_modal)).dismissed(actions) {
                self.close_confirm(cx);
            }
        }

        if self.button(cx, ids!(process_btn)).clicked(actions) {
            self.process_transcript(cx);
        }
        if let Some((text, _)) = self.text_input(cx, ids!(cat_quick)).returned(actions) {
            self.add_from_catno(cx, &text);
        }
        if self.button(cx, ids!(add_btn)).clicked(actions) {
            self.add_row(cx);
        }
        if self.button(cx, ids!(discogs_btn)).clicked(actions) {
            #[cfg(feature = "native")]
            self.start_discogs(cx);
            #[cfg(not(feature = "native"))]
            self.set_status(cx, "Discogs needs the native build");
        }
        if self.button(cx, ids!(prices_btn)).clicked(actions) {
            #[cfg(feature = "native")]
            self.start_batch_prices(cx);
            #[cfg(not(feature = "native"))]
            self.set_status(cx, "Prices needs the native build");
        }
        if self.button(cx, ids!(export_btn)).clicked(actions) {
            self.export_csv(cx);
        }
        for action in actions {
            if let Some(FileDialogAction::SaveFileSelected { id, path }) =
                action.downcast_ref::<FileDialogAction>()
            {
                if *id == live_id!(export_csv) {
                    let path = path.clone();
                    self.write_catalogue_csv(cx, &path);
                }
            }
        }
        if self.button(cx, ids!(delete_btn)).clicked(actions) {
            self.delete_selected(cx);
        }
        #[cfg(feature = "native")]
        {
            if self.button(cx, ids!(ok_btn)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Ok);
            }
            if self.button(cx, ids!(pick1)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Pick(0));
            }
            if self.button(cx, ids!(pick2)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Pick(1));
            }
            if self.button(cx, ids!(pick3)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Pick(2));
            }
            if self.button(cx, ids!(pick4)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Pick(3));
            }
            if self.button(cx, ids!(unresolved_btn)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Unresolved);
            }
            if self.button(cx, ids!(skip_btn)).clicked(actions) {
                self.apply_voice_cmd(cx, VoiceCmd::Skip);
            }
        }

        if let Some(text) = self.text_input(cx, ids!(edit_artist)).changed(actions) {
            self.write_selected_field(cx, Field::Artist, text);
        }
        if let Some(text) = self.text_input(cx, ids!(edit_title)).changed(actions) {
            self.write_selected_field(cx, Field::Title, text);
        }
        if let Some(text) = self.text_input(cx, ids!(edit_label)).changed(actions) {
            self.write_selected_field(cx, Field::Label, text);
        }
        if let Some(text) = self.text_input(cx, ids!(edit_cat)).changed(actions) {
            self.write_selected_field(cx, Field::Cat, text);
        }
        if let Some(text) = self.text_input(cx, ids!(edit_notes)).changed(actions) {
            self.write_selected_field(cx, Field::Notes, text);
        }
        if let Some(text) = self.text_input(cx, ids!(edit_condition)).changed(actions) {
            self.write_selected_field(cx, Field::Condition, text);
        }

        let grid = self.data_grid(cx, ids!(records_grid));
        for (row, col, widget) in grid.cell_widgets_with_actions(actions) {
            if col == 7 && widget.link_label(cx, ids!(link)).clicked(actions) {
                if row < self.records.len() {
                    self.selected = Some(row);
                    self.sync_editor(cx);
                    if let Some(url) = discogs_page_url(&self.records[row]) {
                        self.set_status(cx, format!("Opened {url}"));
                    }
                }
            }
        }
        for action in grid.actions(actions) {
            match action {
                DataGridAction::CellClicked { row, .. }
                | DataGridAction::CellDoubleClicked { row, .. } => {
                    if row < self.records.len() {
                        self.selected = Some(row);
                        self.sync_editor(cx);
                    }
                }
                DataGridAction::SelectionChanged { selection } => {
                    if let Some(selection) = selection {
                        let row = selection.row_range().0;
                        if row < self.records.len() {
                            self.selected = Some(row);
                            self.sync_editor(cx);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
