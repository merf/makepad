//! Sidebar + Import / Searches / Listings / Ask.

use crate::ask_tools::{self, AskToolJob, AskToolRunner};
use crate::db::{Db, ImportResult};
use crate::llm::{self, ChatAgent, ChatEvent};
#[allow(unused_imports)]
use crate::model::{Listing, Store};
use crate::parse::parse_export;
use makepad_widgets::*;
use std::collections::HashSet;
use std::path::PathBuf;

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    mod.widgets.EtsyBase = #(Etsy::register_widget(vm))

    let Panel = SolidView{
        draw_bg +: { color: #x14161d }
    }

    let Title = Label{
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_bold{font_size: 12}
        }
    }

    let Body = Label{
        width: Fill
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_regular{font_size: 10}
        }
    }

    let Dim = Label{
        width: Fill
        draw_text +: {
            color: #x9aa7b4
            text_style: theme.font_regular{font_size: 9}
        }
    }

    let Mono = Label{
        width: Fill
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_code{font_size: 9}
        }
    }

    let NavItem = Button{
        width: Fill
        height: 34
        align: Align{x: 0.0, y: 0.5}
        padding: Inset{left: 12, right: 10, top: 6, bottom: 6}
        draw_bg +: {
            color: #x00000000
            color_hover: #x1b1e27
            color_down: #xc45c2633
            color_focus: #x00000000
            border_radius: 6.0
            border_size: 0.0
        }
        draw_text +: {
            color: #x9aa7b4
            color_hover: #xf2f4f8
            color_down: #xf2f4f8
            color_focus: #x9aa7b4
            text_style: theme.font_regular{font_size: 10}
        }
    }

    let Primary = Button{
        height: 32
        padding: Inset{left: 16, right: 16, top: 6, bottom: 6}
        draw_bg +: {
            color: #xc45c26
            color_hover: #xd46a32
            color_down: #xa84a1e
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

    let Chip = Button{
        height: 28
        padding: Inset{left: 12, right: 12, top: 4, bottom: 4}
        draw_bg +: {
            color: #x1b1e27
            color_hover: #x252833
            color_down: #xc45c2633
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

    let AskLine = View{
        width: Fill
        height: Fit
        padding: Inset{left: 12, right: 12, top: 6, bottom: 4}
        line_label := Body{ width: Fill }
    }

    let AskInput = TextInput{
        width: Fill
        height: 64
        is_multiline: true
        submit_on_enter: true
        empty_text: "Ask about prices, shops, searches… (Enter to send)"
        draw_bg +: {
            color: #x0c0d12
            border_radius: 8.0
            border_size: 1.0
            border_color: #x252833
        }
        draw_text +: {
            color: #xf2f4f8
            color_empty: #x5a6570
            text_style: theme.font_regular{font_size: 10}
        }
    }

    let PasteBox = TextInput{
        width: Fill
        height: Fill
        is_multiline: true
        empty_text: "Paste Export Etsy Results JSON here…"
        draw_bg +: {
            color: #x0c0d12
            border_radius: 8.0
            border_size: 1.0
            border_color: #x252833
        }
        draw_text +: {
            color: #xf2f4f8
            color_empty: #x5a6570
            text_style: theme.font_code{font_size: 9}
        }
    }

    let ListingsGrid = DataGrid{
        width: Fill
        height: Fill
        rows: 0
        cols: 9
        default_col_width: 120.0
        default_row_height: 28.0
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
        scroll_bar_h: mod.widgets.ScrollBar{}
        scroll_bar_v: mod.widgets.ScrollBar{}
    }

    let SearchRow = View{
        width: Fill
        height: Fit
        flow: Right
        align: Align{x: 0.0, y: 0.5}
        padding: Inset{left: 4, right: 4, top: 4, bottom: 4}
        spacing: 8
        expand_btn := Chip{
            width: Fit
            text: "▸"
        }
        search_meta := Body{ width: Fill }
        delete_btn := Chip{
            width: Fit
            text: "Delete"
        }
    }

    let ListingRow = View{
        width: Fill
        height: Fit
        flow: Right
        align: Align{x: 0.0, y: 0.5}
        padding: Inset{left: 36, right: 8, top: 4, bottom: 4}
        spacing: 10
        listing_titles := View{
            width: Fill
            height: Fit
            flow: Down
            spacing: 1
            listing_title := Body{ width: Fill }
            listing_title_raw := Dim{ width: Fill }
        }
        listing_shop := Dim{ width: 110 }
        listing_price := Body{ width: 110 }
        listing_reviews := Dim{ width: 72 }
    }

    mod.widgets.Etsy = set_type_default() do mod.widgets.EtsyBase{
        width: Fill
        height: Fill
        flow: Right
        show_bg: true
        draw_bg +: { color: #x0c0d12 }

        sidebar := Panel{
            width: 200
            height: Fill
            flow: Down
            padding: Inset{left: 10, right: 10, top: 14, bottom: 10}
            spacing: 4

            brand := Label{
                margin: Inset{left: 10, bottom: 10}
                draw_text +: {
                    color: #xf2f4f8
                    text_style: theme.font_bold{font_size: 13}
                }
                text: "Etsy Research"
            }

            nav_import := NavItem{ text: "Import" }
            nav_searches := NavItem{ text: "Searches" }
            nav_listings := NavItem{ text: "Listings" }
            nav_ask := NavItem{ text: "Ask" }

            Hr{ height: 18 }

            store_stats := Dim{
                margin: Inset{left: 10, top: 4}
                text: "0 listings"
            }
        }

        content := View{
            width: Fill
            height: Fill
            flow: Down

            topbar := Panel{
                width: Fill
                height: 48
                flow: Right
                align: Align{x: 0.0, y: 0.5}
                padding: Inset{left: 16, right: 16}
                screen_title := Title{ text: "Import" }
            }

            screens := View{
                width: Fill
                height: Fill
                flow: Overlay

                import_screen := View{
                    width: Fill
                    height: Fill
                    flow: Down
                    padding: Inset{left: 16, right: 16, top: 8, bottom: 16}
                    spacing: 10

                    paste_hint := Dim{
                        text: "Paste the bookmarklet JSON, then Import. Prior imports are kept."
                    }

                    paste_box := PasteBox{}

                    import_actions := View{
                        width: Fill
                        height: Fit
                        flow: Right
                        spacing: 8
                        import_btn := Primary{ text: "Import Etsy Export" }
                        clear_btn := Chip{ text: "Clear" }
                    }

                    import_status := Body{ text: "Waiting for paste…" }
                    import_meta := Dim{ text: "" }
                }

                searches_screen := View{
                    width: Fill
                    height: Fill
                    flow: Down
                    padding: Inset{left: 12, right: 12, top: 8, bottom: 12}
                    spacing: 8
                    visible: false

                    searches_toolbar := View{
                        width: Fill
                        height: Fit
                        flow: Right
                        align: Align{x: 0.0, y: 0.5}
                        spacing: 10
                        clean_titles_btn := Chip{
                            width: Fit
                            text: "Clean titles"
                        }
                        searches_hint := Dim{
                            text: "Expand a search to see its listings. Delete needs a second click."
                        }
                    }

                    searches_list := PortalList{
                        width: Fill
                        height: Fill
                        scroll_bar: mod.widgets.ScrollBar{}
                        SearchRow := SearchRow{}
                        ListingRow := ListingRow{}
                    }
                }

                listings_screen := View{
                    width: Fill
                    height: Fill
                    flow: Down
                    padding: Inset{left: 12, right: 12, top: 4, bottom: 12}
                    visible: false
                    listings_hint := Dim{
                        margin: Inset{left: 4, bottom: 6}
                        text: "Stored listings (deduped by URL). Flags: D=digital P=personalised B=bundle."
                    }
                    listings_grid := ListingsGrid{}
                }

                ask_screen := View{
                    width: Fill
                    height: Fill
                    flow: Down
                    padding: Inset{left: 12, right: 12, top: 8, bottom: 12}
                    spacing: 8
                    visible: false

                    ask_list := PortalList{
                        width: Fill
                        height: Fill
                        scroll_bar: mod.widgets.ScrollBar{}
                        UserLine := AskLine{
                            line_label +: {
                                draw_text +: {
                                    color: #xf2f4f8
                                    text_style: theme.font_bold{font_size: 10}
                                }
                            }
                        }
                        AssistantLine := AskLine{}
                        ToolLine := AskLine{
                            padding: Inset{left: 20, right: 12, top: 2, bottom: 2}
                            line_label +: {
                                draw_text +: {
                                    color: #x9aa7b4
                                    text_style: theme.font_regular{font_size: 9}
                                }
                            }
                        }
                        InfoLine := AskLine{
                            line_label +: {
                                draw_text +: {
                                    color: #xc45c26
                                    text_style: theme.font_regular{font_size: 9}
                                }
                            }
                        }
                    }

                    ask_confirm_row := View{
                        width: Fill
                        height: Fit
                        flow: Right
                        spacing: 8
                        align: Align{x: 0.0, y: 0.5}
                        visible: false
                        ask_confirm_label := Body{ width: Fill, text: "" }
                        ask_confirm_btn := Primary{ text: "Confirm" }
                        ask_cancel_btn := Chip{ text: "Cancel" }
                    }

                    ask_input_row := View{
                        width: Fill
                        height: Fit
                        flow: Right
                        spacing: 8
                        align: Align{x: 0.0, y: 0.0}
                        ask_input := AskInput{}
                        ask_btns := View{
                            width: Fit
                            height: Fit
                            flow: Down
                            spacing: 6
                            ask_send := Primary{ text: "Send" }
                            ask_stop := Chip{ text: "Stop" }
                        }
                    }

                    ask_status := Dim{ text: "Ask about your imported Etsy data." }
                }
            }
        }
    }
}

const COL_LABELS: [&str; 9] = [
    "Title",
    "Shop",
    "Price",
    "Postage",
    "Pack",
    "£/card",
    "Flags",
    "Shop reviews",
    "Query",
];
const COL_WIDTHS: [f64; 9] = [220.0, 110.0, 70.0, 80.0, 48.0, 64.0, 48.0, 88.0, 140.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Import,
    Searches,
    Listings,
    Ask,
}

impl Screen {
    const ALL: [Screen; 4] = [
        Screen::Import,
        Screen::Searches,
        Screen::Listings,
        Screen::Ask,
    ];

    fn title(self) -> &'static str {
        match self {
            Screen::Import => "Import",
            Screen::Searches => "Searches",
            Screen::Listings => "Listings",
            Screen::Ask => "Ask",
        }
    }

    fn view_id(self) -> &'static [LiveId] {
        match self {
            Screen::Import => ids!(import_screen),
            Screen::Searches => ids!(searches_screen),
            Screen::Listings => ids!(listings_screen),
            Screen::Ask => ids!(ask_screen),
        }
    }

    fn nav_id(self) -> &'static [LiveId] {
        match self {
            Screen::Import => ids!(nav_import),
            Screen::Searches => ids!(nav_searches),
            Screen::Listings => ids!(nav_listings),
            Screen::Ask => ids!(nav_ask),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlatRow {
    Search(i64),
    Listing(i64),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum AskVoice {
    User,
    #[default]
    Assistant,
    Tool,
    Info,
}

#[derive(Clone, Debug)]
struct AskLine {
    voice: AskVoice,
    text: String,
}

struct AskState {
    lines: Vec<AskLine>,
    pending: String,
}

const ASK_MAX_LINES: usize = 200;

impl Default for AskState {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            pending: String::new(),
        }
    }
}

impl AskState {
    fn push(&mut self, voice: AskVoice, text: impl Into<String>) {
        self.lines.push(AskLine {
            voice,
            text: text.into(),
        });
        if self.lines.len() > ASK_MAX_LINES {
            let cut = self.lines.len() - ASK_MAX_LINES;
            self.lines.drain(..cut);
        }
    }

    fn commit_pending(&mut self) -> bool {
        let text = std::mem::take(&mut self.pending);
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        self.push(AskVoice::Assistant, text.to_string());
        true
    }

    fn row_count(&self) -> usize {
        self.lines.len() + usize::from(!self.pending.trim().is_empty())
    }
}

struct PendingMutate {
    name: String,
    args_json: String,
    summary: String,
}

#[derive(Script, ScriptHook, Widget)]
pub struct Etsy {
    #[deref]
    view: View,
    #[rust]
    started: bool,
    #[rust(Screen::Import)]
    screen: Screen,
    #[rust]
    store: Store,
    #[rust]
    db: Option<Db>,
    #[rust]
    status: String,
    #[rust]
    last_import: Option<ImportResult>,
    #[rust]
    expanded: HashSet<i64>,
    #[rust]
    flat_rows: Vec<FlatRow>,
    #[rust]
    pending_delete: Option<i64>,
    #[rust]
    title_agent: Option<ChatAgent>,
    #[rust]
    title_agent_ready: bool,
    #[rust]
    title_busy: bool,
    #[rust]
    llm_delta: String,
    #[rust]
    pending_title_batch: Vec<i64>,
    #[rust]
    ask: AskState,
    #[rust]
    ask_agent: Option<ChatAgent>,
    #[rust]
    ask_agent_ready: bool,
    #[rust]
    ask_busy: bool,
    #[rust]
    ask_status: String,
    #[rust]
    ask_pending_mutate: Option<PendingMutate>,
    #[rust]
    ask_pending_user: Option<String>,
    #[rust]
    ask_awaiting_tools: usize,
    #[rust]
    ask_tool_replies: Vec<(String, bool)>,
    #[rust]
    tool_runner: Option<AskToolRunner>,
}

impl Etsy {
    fn db_path() -> PathBuf {
        PathBuf::from("local/etsy/etsy.db")
    }

    fn start(&mut self, cx: &mut Cx) {
        if self.started {
            return;
        }
        self.started = true;
        let path = Self::db_path();
        match Db::open(&path) {
            Ok(mut db) => match db.load() {
                Ok(store) => {
                    self.store = store;
                    self.status = format!(
                        "Loaded {} listings from {}",
                        self.store.listing_count(),
                        path.display()
                    );
                    self.db = Some(db);
                }
                Err(error) => {
                    self.status = format!("load failed: {error}");
                    self.db = Some(db);
                }
            },
            Err(error) => {
                self.status = format!("cannot open {}: {error}", path.display());
            }
        }
        self.rebuild_flat_rows();
        self.ask_status = "Ask about your imported Etsy data.".into();
        self.show_only_current_screen(cx);
        self.refresh_chrome(cx);
        // Warm the Ask model in the background so the first question is not a wait.
        let _ = self.ensure_ask_agent(cx);
    }

    fn set_screen(&mut self, cx: &mut Cx, screen: Screen) {
        self.screen = screen;
        self.show_only_current_screen(cx);
        self.refresh_chrome(cx);
        self.redraw(cx);
    }

    fn show_only_current_screen(&mut self, cx: &mut Cx) {
        for screen in Screen::ALL {
            self.view(cx, screen.view_id())
                .set_visible(cx, screen == self.screen);
        }
    }

    fn reload_store(&mut self, cx: &mut Cx) {
        let Some(db) = self.db.as_mut() else {
            return;
        };
        match db.load() {
            Ok(store) => {
                self.store = store;
                self.rebuild_flat_rows();
            }
            Err(error) => {
                self.status = format!("reload failed: {error}");
                self.refresh_chrome(cx);
            }
        }
    }

    fn rebuild_flat_rows(&mut self) {
        self.flat_rows.clear();
        for search in &self.store.searches {
            self.flat_rows.push(FlatRow::Search(search.id));
            if self.expanded.contains(&search.id) {
                for listing in self.store.listings_for_search(search.id) {
                    self.flat_rows.push(FlatRow::Listing(listing.id));
                }
            }
        }
    }

    fn toggle_expand(&mut self, cx: &mut Cx, search_id: i64) {
        if !self.expanded.remove(&search_id) {
            self.expanded.insert(search_id);
        }
        self.rebuild_flat_rows();
        self.redraw(cx);
    }

    fn delete_search(&mut self, cx: &mut Cx, search_id: i64) {
        if self.pending_delete != Some(search_id) {
            self.pending_delete = Some(search_id);
            self.status = "Click Delete again to confirm…".into();
            self.refresh_chrome(cx);
            self.redraw(cx);
            return;
        }
        self.pending_delete = None;
        let Some(db) = self.db.as_mut() else {
            self.status = "No database open — run from the repo root.".into();
            self.refresh_chrome(cx);
            return;
        };
        match db.delete_search(search_id) {
            Ok(_) => {
                self.expanded.remove(&search_id);
                self.reload_store(cx);
                self.status = format!("Deleted search {search_id}.");
                self.refresh_chrome(cx);
                self.redraw(cx);
            }
            Err(error) => {
                self.status = format!("Delete failed: {error}");
                self.refresh_chrome(cx);
                self.redraw(cx);
            }
        }
    }

    fn ensure_title_agent(&mut self, cx: &mut Cx) -> bool {
        if self.title_agent.is_some() {
            return self.title_agent_ready;
        }
        let Some(path) = llm::model_path() else {
            self.status = format!(
                "No LLM model — set {} or place {}",
                llm::MODEL_ENV,
                llm::MODEL_FILE
            );
            self.title_busy = false;
            self.refresh_chrome(cx);
            return false;
        };
        self.status = "Loading Qwen (title clean)…".into();
        self.title_agent = Some(ChatAgent::start_title_clean(path));
        self.title_agent_ready = false;
        self.refresh_chrome(cx);
        false
    }

    fn ensure_ask_agent(&mut self, cx: &mut Cx) -> bool {
        if self.tool_runner.is_none() {
            self.tool_runner = Some(AskToolRunner::new(&cx.thread_spawner()));
        }
        if self.ask_agent.is_some() {
            return self.ask_agent_ready;
        }
        let Some(path) = llm::model_path() else {
            self.ask_status = format!(
                "No LLM model — set {} or place {}",
                llm::MODEL_ENV,
                llm::MODEL_FILE
            );
            self.ask_busy = false;
            self.refresh_ask_chrome(cx);
            return false;
        };
        self.ask_status = "Loading Qwen…".into();
        self.ask.push(AskVoice::Info, "Loading model…");
        self.ask_agent = Some(ChatAgent::start_ask(path));
        self.ask_agent_ready = false;
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
        false
    }

    fn maybe_queue_title_clean(&mut self, cx: &mut Cx) {
        for listing in &self.store.listings {
            if listing.title_clean.trim().is_empty()
                && !self.pending_title_batch.contains(&listing.id)
            {
                self.pending_title_batch.push(listing.id);
            }
        }
        if self.pending_title_batch.is_empty() || self.title_busy {
            return;
        }
        self.start_clean_titles(cx);
    }

    fn start_clean_titles(&mut self, cx: &mut Cx) {
        if self.title_busy {
            return;
        }
        if self.pending_title_batch.is_empty() {
            for listing in &self.store.listings {
                if listing.title_clean.trim().is_empty() {
                    self.pending_title_batch.push(listing.id);
                }
            }
        }
        if self.pending_title_batch.is_empty() {
            self.status = "All titles already cleaned.".into();
            self.refresh_chrome(cx);
            return;
        }
        self.title_busy = true;
        if self.ensure_title_agent(cx) {
            self.send_title_batch(cx);
        }
    }

    fn send_title_batch(&mut self, cx: &mut Cx) {
        let batch: Vec<i64> = self.pending_title_batch.iter().copied().take(20).collect();
        if batch.is_empty() {
            self.title_busy = false;
            return;
        }
        let mut parts = Vec::new();
        for id in &batch {
            let Some(listing) = self.store.listings.iter().find(|l| l.id == *id) else {
                continue;
            };
            parts.push(format!(
                "{{\"url\":\"{}\",\"title\":\"{}\"}}",
                json_escape(&listing.url),
                json_escape(&listing.title)
            ));
        }
        if parts.is_empty() {
            self.pending_title_batch.clear();
            self.title_busy = false;
            self.status = "Title clean: nothing to send.".into();
            self.refresh_chrome(cx);
            return;
        }
        let payload = format!("[{}]", parts.join(","));
        self.llm_delta.clear();
        self.status = format!(
            "Cleaning titles… ({} remaining)",
            self.pending_title_batch.len()
        );
        if let Some(agent) = &self.title_agent {
            agent.send_user_turn(payload);
        }
        self.refresh_chrome(cx);
    }

    fn on_title_chat_event(&mut self, cx: &mut Cx, event: ChatEvent) {
        match event {
            ChatEvent::Loading { phase, fraction } => {
                self.status = format!("loading — {phase} {:.0}%", fraction * 100.0);
                self.refresh_chrome(cx);
            }
            ChatEvent::Ready { secs, .. } => {
                self.title_agent_ready = true;
                self.status = format!("Qwen ready in {secs:.1}s");
                self.refresh_chrome(cx);
                if self.title_busy && !self.pending_title_batch.is_empty() {
                    self.send_title_batch(cx);
                }
            }
            ChatEvent::Failed(error) => {
                self.title_agent = None;
                self.title_agent_ready = false;
                self.title_busy = false;
                self.status = format!("Qwen failed ({error})");
                self.refresh_chrome(cx);
            }
            ChatEvent::Delta(text) => {
                self.llm_delta.push_str(&text);
            }
            ChatEvent::ToolCall { name, .. } => {
                log!("etsy title-clean unexpected tool call: {name}");
            }
            ChatEvent::TurnDone { .. } => {
                let pairs = llm::parse_title_clean_batch(&self.llm_delta);
                self.llm_delta.clear();
                if pairs.is_empty() {
                    self.title_busy = false;
                    self.status = "Title clean: model returned no usable JSON.".into();
                    self.refresh_chrome(cx);
                    self.redraw(cx);
                    return;
                }
                let mut applied = 0usize;
                if let Some(db) = self.db.as_mut() {
                    for (url, title_clean) in &pairs {
                        let Some(listing) = self.store.listings.iter().find(|l| l.url == *url)
                        else {
                            continue;
                        };
                        let id = listing.id;
                        if db.set_title_clean(id, title_clean).is_ok() {
                            self.pending_title_batch.retain(|&x| x != id);
                            applied += 1;
                        }
                    }
                }
                self.reload_store(cx);
                if !self.pending_title_batch.is_empty() {
                    self.status = format!(
                        "Cleaned {applied}; {} remaining…",
                        self.pending_title_batch.len()
                    );
                    self.refresh_chrome(cx);
                    self.send_title_batch(cx);
                } else {
                    self.title_busy = false;
                    self.status = format!("Title clean done ({applied} this batch).");
                    self.refresh_chrome(cx);
                }
                self.redraw(cx);
            }
            ChatEvent::ContextFull => {
                self.title_busy = false;
                self.status = "Qwen context full — reopen the app.".into();
                self.refresh_chrome(cx);
            }
        }
    }

    fn send_ask(&mut self, cx: &mut Cx) {
        if self.ask_busy || self.ask_pending_mutate.is_some() {
            return;
        }
        let text = self.text_input(cx, ids!(ask_input)).text();
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.text_input(cx, ids!(ask_input)).set_text(cx, "");
        self.ask.push(AskVoice::User, text.clone());
        self.ask_busy = true;
        self.ask_awaiting_tools = 0;
        self.ask_tool_replies.clear();
        self.ask_status = "thinking…".into();
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
        if self.ensure_ask_agent(cx) {
            if let Some(agent) = &self.ask_agent {
                agent.send_user_turn(text);
            }
        } else {
            self.ask_pending_user = Some(text);
        }
    }

    fn stop_ask(&mut self, cx: &mut Cx) {
        if let Some(agent) = &self.ask_agent {
            agent.cancel();
        }
        self.ask.commit_pending();
        self.ask.push(AskVoice::Info, "stopped");
        self.ask_busy = false;
        self.ask_awaiting_tools = 0;
        self.ask_tool_replies.clear();
        self.ask_pending_mutate = None;
        self.ask_pending_user = None;
        self.ask_status = "stopped".into();
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
    }

    fn on_ask_chat_event(&mut self, cx: &mut Cx, event: ChatEvent) {
        match event {
            ChatEvent::Loading { phase, fraction } => {
                self.ask_status = format!("loading — {phase} {:.0}%", fraction * 100.0);
                self.refresh_ask_chrome(cx);
            }
            ChatEvent::Ready { secs, .. } => {
                self.ask_agent_ready = true;
                self.ask.push(
                    AskVoice::Info,
                    format!("Ready in {secs:.1}s — ask about prices, shops, searches."),
                );
                self.ask_status = "ready".into();
                if let Some(text) = self.ask_pending_user.take() {
                    if let Some(agent) = &self.ask_agent {
                        agent.send_user_turn(text);
                    }
                } else if !self.ask_busy {
                    // idle ready
                }
                self.refresh_ask_chrome(cx);
                self.redraw(cx);
            }
            ChatEvent::Failed(error) => {
                self.ask_agent = None;
                self.ask_agent_ready = false;
                self.ask_busy = false;
                self.ask_pending_user = None;
                self.ask.push(AskVoice::Info, format!("⚠ {error}"));
                self.ask_status = "model failed to load".into();
                self.refresh_ask_chrome(cx);
                self.redraw(cx);
            }
            ChatEvent::Delta(text) => {
                self.ask.pending.push_str(&text);
                self.redraw(cx);
            }
            ChatEvent::ToolCall { name, args } => {
                self.ask.commit_pending();
                let args_json = ask_tools::args_to_json(&args);
                self.ask_awaiting_tools += 1;
                if ask_tools::is_mutating(&name) {
                    if self.ask_pending_mutate.is_some() {
                        self.ask_awaiting_tools = self.ask_awaiting_tools.saturating_sub(1);
                        self.push_ask_tool_reply(
                            cx,
                            "another edit is already waiting for confirm — cancel or confirm it first"
                                .into(),
                            true,
                        );
                        return;
                    }
                    let summary = ask_tools::mutate_summary(&name, &args_json);
                    self.ask
                        .push(AskVoice::Tool, format!("⚙ {summary} (needs confirm)"));
                    self.ask_pending_mutate = Some(PendingMutate {
                        name,
                        args_json,
                        summary: summary.clone(),
                    });
                    self.ask_status = format!("Confirm: {summary}?");
                    self.refresh_ask_chrome(cx);
                    self.redraw(cx);
                    return;
                }
                self.ask.push(AskVoice::Tool, format!("⚙ {name}"));
                self.ask_status = format!("tool: {name}…");
                if let Some(runner) = &self.tool_runner {
                    runner.submit(AskToolJob {
                        name,
                        args_json,
                        store: self.store.clone(),
                    });
                } else {
                    self.push_ask_tool_reply(cx, "tool runner unavailable".into(), true);
                }
                self.refresh_ask_chrome(cx);
                self.redraw(cx);
            }
            ChatEvent::TurnDone {
                tool_calls,
                tokens,
                secs,
                ..
            } => {
                if tool_calls > 0 || self.ask_awaiting_tools > 0 || self.ask_pending_mutate.is_some()
                {
                    self.ask_status = "looking…".into();
                    self.refresh_ask_chrome(cx);
                    return;
                }
                self.ask.commit_pending();
                self.ask_busy = false;
                let rate = tokens as f64 / secs.max(0.001);
                self.ask_status = format!("{tokens} tok in {secs:.1}s ({rate:.1}/s)");
                self.refresh_ask_chrome(cx);
                self.redraw(cx);
            }
            ChatEvent::ContextFull => {
                self.ask.commit_pending();
                self.ask_busy = false;
                self.ask.push(
                    AskVoice::Info,
                    "⚠ context full — reopen the app for a fresh Ask session",
                );
                self.ask_status = "context full".into();
                self.refresh_ask_chrome(cx);
                self.redraw(cx);
            }
        }
    }

    fn push_ask_tool_reply(&mut self, cx: &mut Cx, text: String, is_error: bool) {
        let note: String = text.chars().take(120).collect();
        self.ask.push(
            AskVoice::Tool,
            if is_error {
                format!("⚠ {note}")
            } else {
                format!("→ {note}")
            },
        );
        self.ask_tool_replies.push((text, is_error));
        self.flush_ask_tool_results(cx);
    }

    fn flush_ask_tool_results(&mut self, cx: &mut Cx) {
        if self.ask_pending_mutate.is_some() {
            return;
        }
        if self.ask_awaiting_tools == 0 {
            return;
        }
        if self.ask_tool_replies.len() < self.ask_awaiting_tools {
            return;
        }
        let results = std::mem::take(&mut self.ask_tool_replies);
        self.ask_awaiting_tools = 0;
        if let Some(agent) = &self.ask_agent {
            agent.send_tool_results(results);
        }
        self.ask_status = "reading…".into();
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
    }

    fn drain_ask_tools(&mut self, cx: &mut Cx) {
        let Some(runner) = &self.tool_runner else {
            return;
        };
        let outcomes = runner.drain();
        if outcomes.is_empty() {
            return;
        }
        for outcome in outcomes {
            self.push_ask_tool_reply(cx, outcome.text, outcome.is_error);
        }
    }

    fn confirm_ask_mutate(&mut self, cx: &mut Cx) {
        let Some(pending) = self.ask_pending_mutate.take() else {
            return;
        };
        let store = self.store.clone();
        let result = match self.db.as_mut() {
            Some(db) => ask_tools::run_mutate(&pending.name, &pending.args_json, &store, db),
            None => Err("no database open".into()),
        };
        match result {
            Ok(text) => {
                self.reload_store(cx);
                self.ask.push(AskVoice::Info, format!("✓ {text}"));
                self.status = format!("Ask edit applied: {}", pending.summary);
                self.push_ask_tool_reply(cx, text, false);
            }
            Err(err) => {
                self.ask.push(AskVoice::Info, format!("⚠ {err}"));
                self.push_ask_tool_reply(cx, err, true);
            }
        }
        self.ask_status = "ready".into();
        self.refresh_chrome(cx);
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
    }

    fn cancel_ask_mutate(&mut self, cx: &mut Cx) {
        let Some(pending) = self.ask_pending_mutate.take() else {
            return;
        };
        self.ask
            .push(AskVoice::Info, format!("cancelled: {}", pending.summary));
        self.push_ask_tool_reply(cx, "user cancelled the edit".into(), true);
        self.ask_status = "ready".into();
        self.refresh_ask_chrome(cx);
        self.redraw(cx);
    }

    fn refresh_ask_chrome(&mut self, cx: &mut Cx) {
        self.label(cx, ids!(ask_status))
            .set_text(cx, &self.ask_status);
        let confirming = self.ask_pending_mutate.is_some();
        self.view(cx, ids!(ask_confirm_row))
            .set_visible(cx, confirming);
        if let Some(pending) = &self.ask_pending_mutate {
            self.label(cx, ids!(ask_confirm_label))
                .set_text(cx, &format!("Confirm: {}?", pending.summary));
        }
    }

    fn draw_ask_list(&mut self, cx: &mut Cx2d, list: &mut PortalList) {
        let total = self.ask.row_count();
        list.set_item_range(cx, 0, total);
        while let Some(index) = list.next_visible_item(cx) {
            if index >= total {
                continue;
            }
            let (voice, text) = match self.ask.lines.get(index) {
                Some(line) => (line.voice, line.text.clone()),
                None => (AskVoice::Assistant, self.ask.pending.trim_end().to_string()),
            };
            let template = match voice {
                AskVoice::User => id!(UserLine),
                AskVoice::Assistant => id!(AssistantLine),
                AskVoice::Tool => id!(ToolLine),
                AskVoice::Info => id!(InfoLine),
            };
            let item = list.item(cx, index, template);
            item.label(cx, ids!(line_label)).set_text(cx, &text);
            item.draw_all(cx, &mut Scope::empty());
        }
    }

    fn draw_searches_list(&mut self, cx: &mut Cx2d, list: &mut PortalList) {
        list.set_item_range(cx, 0, self.flat_rows.len());
        while let Some(index) = list.next_visible_item(cx) {
            let Some(row) = self.flat_rows.get(index).copied() else {
                continue;
            };
            match row {
                FlatRow::Search(search_id) => {
                    let item = list.item(cx, index, id!(SearchRow));
                    let Some(search) = self.store.searches.iter().find(|s| s.id == search_id)
                    else {
                        continue;
                    };
                    let expanded = self.expanded.contains(&search_id);
                    let count = self.store.listings_for_search(search_id).len();
                    let query = if search.query.is_empty() {
                        "(no query)"
                    } else {
                        &search.query
                    };
                    let meta = format!(
                        "{}{} · page {} · {} listings · {}",
                        if expanded { "▾ " } else { "▸ " },
                        query,
                        search.page,
                        count,
                        if search.exported_at.is_empty() {
                            "exported ?"
                        } else {
                            &search.exported_at
                        }
                    );
                    item.button(cx, ids!(expand_btn))
                        .set_text(cx, if expanded { "▾" } else { "▸" });
                    item.label(cx, ids!(search_meta)).set_text(cx, &meta);
                    let delete_label = if self.pending_delete == Some(search_id) {
                        "Confirm?"
                    } else {
                        "Delete"
                    };
                    item.button(cx, ids!(delete_btn))
                        .set_text(cx, delete_label);
                    item.draw_all(cx, &mut Scope::empty());
                }
                FlatRow::Listing(listing_id) => {
                    let item = list.item(cx, index, id!(ListingRow));
                    let Some(listing) = self.store.listings.iter().find(|l| l.id == listing_id)
                    else {
                        continue;
                    };
                    let clean = listing.title_clean.trim();
                    let raw = listing.title.trim();
                    let primary = if !clean.is_empty() { clean } else { raw };
                    item.label(cx, ids!(listing_title))
                        .set_text(cx, &truncate(primary, 72));
                    let raw_line = if !clean.is_empty() && !raw.is_empty() && clean != raw {
                        truncate(raw, 90)
                    } else {
                        String::new()
                    };
                    item.label(cx, ids!(listing_title_raw))
                        .set_text(cx, &raw_line);
                    item.label(cx, ids!(listing_shop)).set_text(
                        cx,
                        &truncate(&listing.shop_name, 18),
                    );
                    item.label(cx, ids!(listing_price))
                        .set_text(cx, &format_delivered_price(listing));
                    let reviews = match listing.review_count {
                        Some(n) => n.to_string(),
                        None => "—".into(),
                    };
                    item.label(cx, ids!(listing_reviews))
                        .set_text(cx, &reviews);
                    item.draw_all(cx, &mut Scope::empty());
                }
            }
        }
    }

    fn refresh_chrome(&mut self, cx: &mut Cx) {
        self.label(cx, ids!(screen_title))
            .set_text(cx, self.screen.title());
        self.label(cx, ids!(store_stats)).set_text(
            cx,
            &format!(
                "{} listings · {} searches",
                self.store.listing_count(),
                self.store.search_count()
            ),
        );
        self.label(cx, ids!(import_status))
            .set_text(cx, &self.status);
        let meta = match &self.last_import {
            Some(r) => format!(
                "Last import: {} listings in export · +{} new · {} updated · query “{}” · page {} · exported {}",
                r.listings_in_export,
                r.listings_inserted,
                r.listings_updated,
                if r.query.is_empty() { "(none)" } else { &r.query },
                r.page,
                if r.exported_at.is_empty() {
                    "(unknown)"
                } else {
                    &r.exported_at
                }
            ),
            None => String::new(),
        };
        self.label(cx, ids!(import_meta)).set_text(cx, &meta);

        for screen in Screen::ALL {
            let active = screen == self.screen;
            let color = if active {
                vec4(0.949, 0.957, 0.973, 1.0)
            } else {
                vec4(0.604, 0.655, 0.706, 1.0)
            };
            let mut nav = self.button(cx, screen.nav_id());
            script_apply_eval!(cx, nav, {
                draw_text +: { color: #(color) }
            });
        }

        if self.screen == Screen::Ask {
            self.refresh_ask_chrome(cx);
        }
    }

    fn clear_paste(&mut self, cx: &mut Cx) {
        self.text_input(cx, ids!(paste_box)).set_text(cx, "");
        self.status = "Paste cleared.".into();
        self.refresh_chrome(cx);
    }

    fn do_import(&mut self, cx: &mut Cx) {
        let text = self.text_input(cx, ids!(paste_box)).text();
        let payload = match parse_export(&text) {
            Ok(p) => p,
            Err(error) => {
                self.status = format!("Import failed: {error}");
                self.refresh_chrome(cx);
                self.redraw(cx);
                return;
            }
        };

        let Some(db) = self.db.as_mut() else {
            self.status = "No database open — run from the repo root.".into();
            self.refresh_chrome(cx);
            return;
        };

        match db.import_export(&payload) {
            Ok(result) => {
                match db.load() {
                    Ok(store) => {
                        self.store = store;
                        self.rebuild_flat_rows();
                    }
                    Err(error) => {
                        self.status = format!("imported but reload failed: {error}");
                        self.refresh_chrome(cx);
                        self.redraw(cx);
                        return;
                    }
                }
                self.status = format!(
                    "Imported {} listings ({} new, {} already known).",
                    result.listings_in_export, result.listings_inserted, result.listings_updated
                );
                self.last_import = Some(result);
                self.set_screen(cx, Screen::Searches);
                self.maybe_queue_title_clean(cx);
            }
            Err(error) => {
                self.status = format!("Database error: {error}");
                self.refresh_chrome(cx);
                self.redraw(cx);
            }
        }
    }

    fn cell_text(&self, row: usize, col: usize) -> String {
        let Some(listing) = self.store.listings.get(row) else {
            return String::new();
        };
        match col {
            0 => listing.display_title().to_string(),
            1 => listing.shop_name.clone(),
            2 => format_price(listing.price, &listing.currency, &listing.price_text),
            3 => {
                if listing.postage_known {
                    match listing.postage {
                        Some(0.0) => "Free".into(),
                        Some(n) => format!("£{n:.2}"),
                        None => listing.postage_text.clone(),
                    }
                } else if listing.postage_text.is_empty() {
                    "—".into()
                } else {
                    listing.postage_text.clone()
                }
            }
            4 => listing
                .pack_size
                .map(|n| n.to_string())
                .unwrap_or_else(|| "—".into()),
            5 => listing
                .price_per_card
                .map(|n| format!("£{n:.2}"))
                .unwrap_or_else(|| "—".into()),
            6 => listing.flag_text(),
            7 => match listing.review_count {
                Some(n) => n.to_string(),
                None => "—".into(),
            },
            8 => {
                if listing.queries.is_empty() {
                    "—".into()
                } else {
                    listing.queries.join("; ")
                }
            }
            _ => String::new(),
        }
    }
}

fn format_price(price: Option<f64>, currency: &str, price_text: &str) -> String {
    match price {
        Some(n) => {
            let sym = match currency {
                "GBP" => "£",
                "USD" => "$",
                "EUR" => "€",
                _ if !currency.is_empty() => return format!("{currency} {n:.2}"),
                _ => {
                    if !price_text.is_empty() {
                        return price_text.to_string();
                    }
                    "£"
                }
            };
            format!("{sym}{n:.2}")
        }
        None if !price_text.is_empty() => price_text.to_string(),
        None => "—".into(),
    }
}

/// Delivered total primary; paid postage in brackets when it is a separate amount.
fn format_delivered_price(listing: &Listing) -> String {
    let item = format_price(listing.price, &listing.currency, &listing.price_text);
    let primary = match listing.delivered_price {
        Some(n) if n.is_finite() => {
            let sym = match listing.currency.as_str() {
                "USD" => "$",
                "EUR" => "€",
                "GBP" | "" => "£",
                other => {
                    return format_delivered_with_postage(
                        format!("{other} {n:.2}"),
                        listing,
                    );
                }
            };
            format!("{sym}{n:.2}")
        }
        _ => item,
    };
    format_delivered_with_postage(primary, listing)
}

fn format_delivered_with_postage(primary: String, listing: &Listing) -> String {
    if !listing.postage_known {
        return primary;
    }
    match listing.postage {
        Some(p) if p > 0.0 && p.is_finite() => {
            let sym = match listing.currency.as_str() {
                "USD" => "$",
                "EUR" => "€",
                _ => "£",
            };
            format!("{primary} ({sym}{p:.2})")
        }
        Some(0.0) => primary, // free — already in delivered total
        _ => primary,
    }
}

fn truncate(s: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[allow(dead_code)]
fn short_url(url: &str) -> String {
    let trimmed = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("www.");
    truncate(trimmed, 48)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

impl Widget for Etsy {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.start(cx);
        while let Some(step) = self.view.draw_walk(cx, scope, walk).step() {
            if let Some(mut list) = step.as_portal_list().borrow_mut() {
                // Two portal lists share this path — disambiguate by screen.
                if self.screen == Screen::Ask {
                    self.draw_ask_list(cx, &mut list);
                } else {
                    self.draw_searches_list(cx, &mut list);
                }
                continue;
            }
            if let Some(mut grid) = step.as_data_grid().borrow_mut() {
                grid.set_grid_size(self.store.listings.len(), COL_LABELS.len());
                grid.set_col_labels(COL_LABELS.iter().map(|s| s.to_string()).collect());
                for (index, width) in COL_WIDTHS.iter().enumerate() {
                    grid.set_col_width(index, *width);
                }
                while let Some(cell) = grid.next_cell(cx) {
                    let text = self.cell_text(cell.row, cell.col);
                    grid.cell_text(cx, &cell, &text);
                }
            }
        }
        DrawStep::done()
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, scope: &mut Scope) {
        if let Event::Signal = event {
            let title_events = self
                .title_agent
                .as_ref()
                .map(|a| a.poll())
                .unwrap_or_default();
            for chat_event in title_events {
                self.on_title_chat_event(cx, chat_event);
            }
            let ask_events = self
                .ask_agent
                .as_ref()
                .map(|a| a.poll())
                .unwrap_or_default();
            for chat_event in ask_events {
                self.on_ask_chat_event(cx, chat_event);
            }
            self.drain_ask_tools(cx);
        }
        self.view.handle_event(cx, event, scope);
        self.widget_match_event(cx, event, scope);
    }
}

impl WidgetMatchEvent for Etsy {
    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions, _scope: &mut Scope) {
        for screen in Screen::ALL {
            if self.button(cx, screen.nav_id()).clicked(actions) {
                self.set_screen(cx, screen);
            }
        }
        if self.button(cx, ids!(import_btn)).clicked(actions) {
            self.do_import(cx);
        }
        if self.button(cx, ids!(clear_btn)).clicked(actions) {
            self.clear_paste(cx);
        }
        if self.button(cx, ids!(clean_titles_btn)).clicked(actions) {
            self.start_clean_titles(cx);
        }
        if self.button(cx, ids!(ask_send)).clicked(actions)
            || self.text_input(cx, ids!(ask_input)).returned(actions).is_some()
        {
            self.send_ask(cx);
        }
        if self.button(cx, ids!(ask_stop)).clicked(actions) {
            self.stop_ask(cx);
        }
        if self.button(cx, ids!(ask_confirm_btn)).clicked(actions) {
            self.confirm_ask_mutate(cx);
        }
        if self.button(cx, ids!(ask_cancel_btn)).clicked(actions) {
            self.cancel_ask_mutate(cx);
        }

        let rows = self.flat_rows.clone();
        for (index, row) in rows.iter().enumerate() {
            if let FlatRow::Search(search_id) = *row {
                let item = self
                    .portal_list(cx, ids!(searches_list))
                    .item(cx, index, id!(SearchRow));
                if item.button(cx, ids!(expand_btn)).clicked(actions) {
                    self.toggle_expand(cx, search_id);
                }
                if item.button(cx, ids!(delete_btn)).clicked(actions) {
                    self.delete_search(cx, search_id);
                }
            }
        }
    }
}
