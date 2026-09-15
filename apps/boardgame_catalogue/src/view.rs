//! Board game catalogue UI — Etsy-style left nav, Catalogue grid, What to play cards.

use crate::bgg::{self, BggAction, BggClient, SearchHit};
use crate::db::Db;
use crate::demo;
use crate::meters::{PlayerCountStrip, RatingMeter};
use crate::model::*;
use makepad_widgets::*;
use std::path::PathBuf;

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    mod.widgets.BoardGamesBase = #(BoardGames::register_widget(vm))

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

    let Body = Label{
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_regular{font_size: 10}
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
            border_size: 1.0
            border_color: #x252833
        }
        draw_text +: {
            color: #x9aa7b4
            color_hover: #xf2f4f8
            color_down: #xf2f4f8
            color_focus: #x9aa7b4
            text_style: theme.font_regular{font_size: 9}
        }
    }

    let ChipOn = Button{
        height: 28
        padding: Inset{left: 12, right: 12, top: 4, bottom: 4}
        draw_bg +: {
            color: #xc45c2633
            color_hover: #xc45c2644
            color_down: #xc45c2655
            color_focus: #xc45c2633
            border_radius: 6.0
            border_size: 1.0
            border_color: #xc45c26
        }
        draw_text +: {
            color: #xf2f4f8
            color_hover: #xffffff
            color_down: #xffffff
            color_focus: #xf2f4f8
            text_style: theme.font_bold{font_size: 9}
        }
    }

    let Field = TextInput{
        width: Fill
        height: 32
        empty_text: ""
        draw_bg +: {
            color: #x0c0d12
            color_hover: #x101219
            color_focus: #x101219
            border_radius: 6.0
            border_size: 1.0
            border_color: #x252833
            border_color_hover: #x3a4150
            border_color_focus: #xc45c26
        }
        draw_text +: {
            color: #xf2f4f8
            color_empty: #x5a6570
            text_style: theme.font_regular{font_size: 10}
        }
    }

    let CatalogueGrid = DataGrid{
        width: Fill
        height: Fill
        rows: 0
        cols: 10
        default_col_width: 100.0
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

    let GameCard = RoundedView{
        width: Fill
        height: Fit
        padding: Inset{left: 10, right: 10, top: 10, bottom: 10}
        flow: Down
        spacing: 8
        draw_bg +: {
            color: #x14161d
            border_radius: 10.0
            border_size: 1.0
            border_color: #x252833
        }

        cover_slot := View{
            width: Fill
            height: 140
            align: Align{x: 0.5, y: 0.5}
            cover := Image{
                width: Fill
                height: Fill
                fit: ImageFit.Smallest
            }
        }

        title_row := View{
            width: Fill
            // Fixed two-line slot so short titles don't collapse the row.
            height: 34
            flow: Right
            spacing: 8
            align: Align{y: 0.0}
            card_title := Title{
                width: Fill
                height: Fill
                max_lines: 2
                text_overflow: Ellipsis
                text: "Game"
            }
            card_time := Dim{ width: Fit, text: "" }
        }

        card_owners := Dim{ text: "" }

        score_row := View{
            width: Fill
            height: 36
            flow: Right
            spacing: 10
            align: Align{y: 0.5}
            rating_meter := RatingMeter{}
            player_strip := PlayerCountStrip{ width: Fill, height: Fill }
        }
    }

    let HitRow = View{
        width: Fill
        height: Fit
        flow: Right
        padding: Inset{left: 8, right: 8, top: 4, bottom: 4}
        spacing: 8
        align: Align{y: 0.5}
        hit_label := Body{ width: Fill, text: "" }
        hit_pick := Primary{ text: "Add" }
    }

    mod.widgets.BoardGames = set_type_default() do mod.widgets.BoardGamesBase{
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
                text: "Board Games"
            }

            nav_play := NavItem{ text: "What to play" }
            nav_catalogue := NavItem{ text: "Catalogue" }

            Hr{ height: 18 }

            store_stats := Dim{
                margin: Inset{left: 10, top: 4}
                text: "0 games"
            }
        }

        content := View{
            width: Fill
            height: Fill
            flow: Down

            topbar := Panel{
                width: Fill
                height: Fit
                flow: Down
                padding: Inset{left: 16, right: 16, top: 10, bottom: 10}
                spacing: 8

                title_row := View{
                    width: Fill
                    height: 32
                    flow: Right
                    align: Align{y: 0.5}
                    screen_title := Title{ text: "What to play" }
                    Fill{}
                    status_label := Dim{ text: "" }
                }

                play_bar := View{
                    width: Fill
                    height: Fit
                    flow: Right
                    spacing: 8
                    align: Align{y: 0.5}
                    players_label := Dim{ text: "Players" }
                    players_dec := Chip{ text: "−" }
                    players_value := Body{ text: "4" }
                    players_inc := Chip{ text: "+" }
                    venue_table := ChipOn{ text: "Table" }
                    venue_bga := Chip{ text: "BGA" }
                    venue_tts := Chip{ text: "TTS" }
                    owners_label := Dim{ text: "Boxes:" }
                    owners_me := ChipOn{ text: "Me" }
                    owners_all := Chip{ text: "Everyone" }
                    include_wishlist := Chip{ text: "Wishlist" }
                }

                catalogue_bar := View{
                    width: Fill
                    height: Fit
                    flow: Right
                    spacing: 8
                    align: Align{y: 0.5}
                    visible: false
                    filter_input := Field{ empty_text: "Filter catalogue…" }
                    add_query := Field{ empty_text: "Search BoardGameGeek…" }
                    search_btn := Primary{ text: "Search BGG" }
                    demo_btn := Chip{ text: "Load demo" }
                    import_btn := Chip{ text: "Import XML" }
                }
            }

            screens := View{
                width: Fill
                height: Fill
                flow: Overlay

                play_screen := View{
                    width: Fill
                    height: Fill
                    padding: Inset{left: 16, right: 16, top: 4, bottom: 16}
                    cards_list := PortalList{
                        width: Fill
                        height: Fill
                        scroll_bar: mod.widgets.ScrollBar{}
                        // Inline row template (asset-ui CatalogGrid pattern): named
                        // slots so Rust can show `card_columns` of them per row.
                        CardRow := View{
                            width: Fill
                            height: Fit
                            flow: Right
                            spacing: 12
                            padding: Inset{bottom: 12}
                            c1 := GameCard{}
                            c2 := GameCard{}
                            c3 := GameCard{}
                            c4 := GameCard{}
                        }
                    }
                }

                catalogue_screen := View{
                    width: Fill
                    height: Fill
                    flow: Overlay
                    visible: false

                    catalogue_grid := CatalogueGrid{}

                    add_overlay := RoundedView{
                        width: 420
                        height: Fill
                        visible: false
                        margin: Inset{left: 0, right: 16, top: 8, bottom: 16}
                        align: Align{x: 1.0, y: 0.0}
                        padding: Inset{left: 14, right: 14, top: 14, bottom: 14}
                        flow: Down
                        spacing: 8
                        draw_bg +: {
                            color: #x14161d
                            border_radius: 10.0
                            border_size: 1.0
                            border_color: #xc45c26
                        }
                        overlay_title := Title{ text: "BGG results" }
                        overlay_hint := Dim{ text: "Pick a game to add to your catalogue." }
                        hits_list := PortalList{
                            width: Fill
                            height: Fill
                            scroll_bar: mod.widgets.ScrollBar{}
                            HitRow := HitRow{}
                        }
                        close_overlay := Chip{ text: "Close" }
                    }

                    edit_overlay := RoundedView{
                        width: 360
                        height: Fit
                        visible: false
                        margin: Inset{left: 0, right: 16, top: 8, bottom: 16}
                        align: Align{x: 1.0, y: 0.0}
                        padding: Inset{left: 14, right: 14, top: 14, bottom: 14}
                        flow: Down
                        spacing: 8
                        draw_bg +: {
                            color: #x14161d
                            border_radius: 10.0
                            border_size: 1.0
                            border_color: #xc45c26
                        }
                        edit_title := Title{ text: "Edit game" }
                        edit_owners_label := Dim{ text: "Owners (click to toggle)" }
                        edit_owners_list := PortalList{
                            width: Fill
                            height: 120
                            scroll_bar: mod.widgets.ScrollBar{}
                            OwnerChip := Chip{ width: Fill, text: "Person" }
                        }
                        edit_platforms_label := Dim{ text: "Online" }
                        edit_bga := Chip{ text: "BGA" }
                        edit_tts := Chip{ text: "TTS" }
                        edit_notes := Field{ empty_text: "Local notes…" }
                        edit_actions := View{
                            width: Fill
                            height: Fit
                            flow: Right
                            spacing: 8
                            edit_save := Primary{ text: "Save" }
                            edit_close := Chip{ text: "Close" }
                        }
                    }
                }
            }
        }
    }
}

const COL_LABELS: [&str; 10] = [
    "Name",
    "Year",
    "Geek",
    "Weight",
    "Players",
    "Best at",
    "Owners",
    "BGA",
    "TTS",
    "Notes",
];
const COL_WIDTHS: [f64; 10] = [220.0, 56.0, 56.0, 64.0, 72.0, 90.0, 120.0, 44.0, 44.0, 140.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Play,
    Catalogue,
}

impl Screen {
    const ALL: [Screen; 2] = [Screen::Play, Screen::Catalogue];

    fn title(self) -> &'static str {
        match self {
            Screen::Play => "What to play",
            Screen::Catalogue => "Catalogue",
        }
    }

    fn view_id(self) -> &'static [LiveId] {
        match self {
            Screen::Play => ids!(play_screen),
            Screen::Catalogue => ids!(catalogue_screen),
        }
    }

    fn nav_id(self) -> &'static [LiveId] {
        match self {
            Screen::Play => ids!(nav_play),
            Screen::Catalogue => ids!(nav_catalogue),
        }
    }
}


#[derive(Script, ScriptHook, Widget)]
pub struct BoardGames {
    #[deref]
    view: View,
    #[rust]
    started: bool,
    #[rust(Screen::Play)]
    screen: Screen,
    #[rust]
    cat: Catalogue,
    #[rust]
    db: Option<Db>,
    #[rust]
    status: String,
    #[rust]
    bgg: BggClient,
    #[rust]
    search_hits: Vec<SearchHit>,
    #[rust]
    show_hits: bool,
    #[rust]
    edit_game_id: Option<i64>,
    #[rust]
    night: NightFilters,
    #[rust]
    night_cards: Vec<NightCard>,
    #[rust]
    grid_filter: String,
    #[rust]
    grid_ids: Vec<i64>,
    #[rust]
    tick_timer: Timer,
}

impl BoardGames {
    fn db_path() -> PathBuf {
        PathBuf::from("local/boardgames/catalogue.db")
    }

    fn start(&mut self, cx: &mut Cx) {
        if self.started {
            return;
        }
        self.started = true;
        self.tick_timer = cx.start_interval(0.25);
        let path = Self::db_path();
        match Db::open(&path) {
            Ok(mut db) => match db.load() {
                Ok(cat) => {
                    self.cat = cat;
                    self.db = Some(db);
                    self.status = format!(
                        "Loaded {} games from {}",
                        self.cat.games.len(),
                        path.display()
                    );
                }
                Err(error) => {
                    self.status = format!("Load failed: {error}");
                    self.db = Some(db);
                }
            },
            Err(error) => {
                self.status = format!("Cannot open {}: {error}", path.display());
            }
        }
        self.bgg.set_platforms(&self.cat.platforms);
        if let Some(token) = self.cat.settings.get("bgg_token") {
            if !token.is_empty() {
                self.bgg.set_token(token.clone());
            }
        }
        if !self.bgg.has_token() {
            self.status = format!(
                "Loaded {} games — set BGG_TOKEN or local/boardgames/bgg_token to search BGG",
                self.cat.games.len()
            );
        }
        if self.night.owner_ids.is_empty() && self.cat.me_person_id > 0 {
            self.night.owner_ids.insert(self.cat.me_person_id);
        }
        self.rebuild_night();
        self.rebuild_grid_ids();
        self.show_only_current_screen(cx);
        self.refresh_chrome(cx);
    }

    fn reload(&mut self, cx: &mut Cx) {
        let Some(db) = self.db.as_mut() else {
            return;
        };
        match db.load() {
            Ok(cat) => {
                self.cat = cat;
                self.bgg.set_platforms(&self.cat.platforms);
                self.rebuild_night();
                self.rebuild_grid_ids();
            }
            Err(error) => self.status = format!("reload failed: {error}"),
        }
        self.refresh_chrome(cx);
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
        let play = self.screen == Screen::Play;
        self.view(cx, ids!(play_bar)).set_visible(cx, play);
        self.view(cx, ids!(catalogue_bar))
            .set_visible(cx, !play);
    }

    fn rebuild_night(&mut self) {
        self.night_cards = night_cards(&self.cat, &self.night);
    }

    fn rebuild_grid_ids(&mut self) {
        let q = self.grid_filter.trim().to_ascii_lowercase();
        self.grid_ids.clear();
        for g in &self.cat.games {
            if q.is_empty() || g.name.to_ascii_lowercase().contains(&q) {
                self.grid_ids.push(g.id);
            }
        }
    }

    fn refresh_chrome(&mut self, cx: &mut Cx) {
        self.label(cx, ids!(screen_title))
            .set_text(cx, self.screen.title());
        self.label(cx, ids!(status_label)).set_text(cx, &self.status);
        self.label(cx, ids!(store_stats)).set_text(
            cx,
            &format!(
                "{} games · {} people",
                self.cat.games.len(),
                self.cat.people.len()
            ),
        );
        self.label(cx, ids!(players_value))
            .set_text(cx, &self.night.players.to_string());

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

        self.style_venue_chip(cx, ids!(venue_table), self.night.include_table);
        self.style_venue_chip(cx, ids!(venue_bga), self.night.include_bga);
        self.style_venue_chip(cx, ids!(venue_tts), self.night.include_tts);
        self.style_venue_chip(cx, ids!(include_wishlist), self.night.include_wishlist);

        let me_only = self.cat.me_person_id > 0
            && self.night.owner_ids.len() == 1
            && self.night.owner_ids.contains(&self.cat.me_person_id);
        let everyone = !self.cat.people.is_empty()
            && self
                .cat
                .people
                .iter()
                .all(|p| self.night.owner_ids.contains(&p.id));
        self.style_venue_chip(cx, ids!(owners_me), me_only);
        self.style_venue_chip(cx, ids!(owners_all), everyone && !me_only);

        self.view(cx, ids!(add_overlay))
            .set_visible(cx, self.show_hits);
        self.view(cx, ids!(edit_overlay))
            .set_visible(cx, self.edit_game_id.is_some());
    }

    fn style_venue_chip(&mut self, cx: &mut Cx, id: &[LiveId], on: bool) {
        // Set every animator color slot. Buttons keep hover/focus for a frame
        // after click; if only `color` updates, the chip looks off until you
        // click something else and hover clears.
        let (bg, border, text) = if on {
            (
                vec4(0.769, 0.361, 0.149, 0.2),
                vec4(0.769, 0.361, 0.149, 1.0),
                vec4(0.949, 0.957, 0.973, 1.0),
            )
        } else {
            (
                vec4(0.106, 0.118, 0.153, 1.0),
                vec4(0.145, 0.157, 0.2, 1.0),
                vec4(0.604, 0.655, 0.706, 1.0),
            )
        };
        let hover = if on {
            vec4(0.769, 0.361, 0.149, 0.27)
        } else {
            vec4(0.145, 0.157, 0.2, 1.0)
        };
        let down = if on {
            vec4(0.769, 0.361, 0.149, 0.33)
        } else {
            vec4(0.769, 0.361, 0.149, 0.2)
        };
        let mut btn = self.button(cx, id);
        script_apply_eval!(cx, btn, {
            draw_bg +: {
                color: #(bg)
                color_hover: #(hover)
                color_down: #(down)
                color_focus: #(bg)
                border_color: #(border)
            }
            draw_text +: {
                color: #(text)
                color_hover: #(text)
                color_down: #(text)
                color_focus: #(text)
            }
        });
    }

    fn do_search(&mut self, cx: &mut Cx) {
        let query = self.text_input(cx, ids!(add_query)).text();
        if query.trim().is_empty() {
            self.status = "Type a game name to search BGG.".into();
            self.refresh_chrome(cx);
            return;
        }
        if !self.bgg.has_token() {
            self.status = "BGG needs a token — use Load demo meanwhile, or set BGG_TOKEN / \
                local/boardgames/bgg_token"
                .into();
            self.refresh_chrome(cx);
            return;
        }
        self.status = format!("Searching BGG for “{query}”…");
        self.refresh_chrome(cx);
        self.bgg.search(cx, &query);
    }

    fn load_demo(&mut self, cx: &mut Cx) {
        let Some(db) = self.db.as_mut() else {
            self.status = "No database open.".into();
            self.refresh_chrome(cx);
            return;
        };
        match demo::load_demo_catalogue(db, &self.cat) {
            Ok(n) => {
                self.status = format!("Loaded {n} demo games (offline — no BGG token needed).");
                self.reload(cx);
                self.set_screen(cx, Screen::Play);
                self.redraw(cx);
            }
            Err(error) => {
                self.status = format!("Demo load failed: {error}");
                self.refresh_chrome(cx);
            }
        }
    }

    fn import_xml(&mut self, cx: &mut Cx) {
        let dir = PathBuf::from("local/boardgames/import");
        let Some(db) = self.db.as_mut() else {
            self.status = "No database open.".into();
            self.refresh_chrome(cx);
            return;
        };
        match demo::import_xml_dir(db, &self.cat, &dir, bgg::parse_thing) {
            Ok((0, 0)) => {
                self.status = format!(
                    "Drop BGG thing XML files into {} then click Import XML again.",
                    dir.display()
                );
                self.refresh_chrome(cx);
            }
            Ok((ok, fail)) => {
                self.status = format!("Imported {ok} XML file(s) from {} ({fail} failed).", dir.display());
                self.reload(cx);
                self.redraw(cx);
            }
            Err(error) => {
                self.status = format!("Import failed: {error}");
                self.refresh_chrome(cx);
            }
        }
    }

    fn on_bgg(&mut self, cx: &mut Cx, action: BggAction) {
        match action {
            BggAction::SearchResults(hits) => {
                self.search_hits = hits;
                self.show_hits = true;
                self.status = format!("{} BGG hits", self.search_hits.len());
                self.set_screen(cx, Screen::Catalogue);
                self.refresh_chrome(cx);
                self.redraw(cx);
            }
            BggAction::Thing(game) => {
                let name = game.name.clone();
                let me = self.cat.me_person_id;
                let Some(db) = self.db.as_mut() else {
                    self.status = "No database open.".into();
                    self.refresh_chrome(cx);
                    return;
                };
                match db.upsert_game_from_bgg(&game, me, true) {
                    Ok(_) => {
                        self.show_hits = false;
                        self.status = format!("Saved “{name}”.");
                        self.reload(cx);
                        self.redraw(cx);
                    }
                    Err(error) => {
                        self.status = format!("Save failed: {error}");
                        self.refresh_chrome(cx);
                    }
                }
            }
            BggAction::Error(msg) => {
                self.status = msg;
                self.refresh_chrome(cx);
            }
            BggAction::Status(msg) => {
                self.status = msg;
                self.refresh_chrome(cx);
            }
        }
    }

    fn open_edit(&mut self, cx: &mut Cx, game_id: i64) {
        self.edit_game_id = Some(game_id);
        if let Some(game) = self.cat.game(game_id) {
            self.label(cx, ids!(edit_title))
                .set_text(cx, &format!("Edit · {}", game.name));
            self.text_input(cx, ids!(edit_notes))
                .set_text(cx, &game.local_notes);
            let bga = self
                .cat
                .platform_by_slug("bga")
                .map(|p| game.has_platform(p.id))
                .unwrap_or(false);
            let tts = self
                .cat
                .platform_by_slug("tts")
                .map(|p| game.has_platform(p.id))
                .unwrap_or(false);
            self.style_venue_chip(cx, ids!(edit_bga), bga);
            self.style_venue_chip(cx, ids!(edit_tts), tts);
        }
        self.refresh_chrome(cx);
        self.redraw(cx);
    }

    fn save_edit(&mut self, cx: &mut Cx) {
        let Some(game_id) = self.edit_game_id else {
            return;
        };
        let notes = self.text_input(cx, ids!(edit_notes)).text();
        let owners = self
            .cat
            .game(game_id)
            .map(|g| g.owner_ids.clone())
            .unwrap_or_default();
        let bga_on = self
            .cat
            .game(game_id)
            .and_then(|g| {
                let id = self.cat.platform_by_slug("bga")?.id;
                Some(g.has_platform(id))
            })
            .unwrap_or(false);
        let tts_on = self
            .cat
            .game(game_id)
            .and_then(|g| {
                let id = self.cat.platform_by_slug("tts")?.id;
                Some(g.has_platform(id))
            })
            .unwrap_or(false);

        // Owners are toggled live on chips into cat; persist those.
        let Some(db) = self.db.as_mut() else {
            return;
        };
        if let Err(e) = db.set_local_notes(game_id, &notes) {
            self.status = e;
            self.refresh_chrome(cx);
            return;
        }
        if let Err(e) = db.set_owners(game_id, &owners) {
            self.status = e;
            self.refresh_chrome(cx);
            return;
        }
        if let Some(p) = self.cat.platform_by_slug("bga").map(|p| p.id) {
            let _ = db.set_manual_platform(game_id, p, bga_on);
        }
        if let Some(p) = self.cat.platform_by_slug("tts").map(|p| p.id) {
            let _ = db.set_manual_platform(game_id, p, tts_on);
        }
        self.edit_game_id = None;
        self.status = "Saved edits.".into();
        self.reload(cx);
        self.redraw(cx);
    }

    fn toggle_edit_owner(&mut self, cx: &mut Cx, person_id: i64) {
        let Some(game_id) = self.edit_game_id else {
            return;
        };
        if let Some(game) = self.cat.game_mut(game_id) {
            if let Some(pos) = game.owner_ids.iter().position(|id| *id == person_id) {
                game.owner_ids.remove(pos);
            } else {
                game.owner_ids.push(person_id);
            }
        }
        self.redraw(cx);
    }

    fn toggle_edit_platform(&mut self, cx: &mut Cx, slug: &str) {
        let Some(game_id) = self.edit_game_id else {
            return;
        };
        let Some(platform_id) = self.cat.platform_by_slug(slug).map(|p| p.id) else {
            return;
        };
        if let Some(game) = self.cat.game_mut(game_id) {
            if let Some(pos) = game
                .platforms
                .iter()
                .position(|p| p.platform_id == platform_id)
            {
                game.platforms.remove(pos);
            } else {
                game.platforms.push(GamePlatform {
                    platform_id,
                    url: String::new(),
                    source: "manual".into(),
                });
            }
            let on = game.has_platform(platform_id);
            if slug == "bga" {
                self.style_venue_chip(cx, ids!(edit_bga), on);
            } else {
                self.style_venue_chip(cx, ids!(edit_tts), on);
            }
        }
        self.redraw(cx);
    }

    fn cell_text(&self, row: usize, col: usize) -> String {
        let Some(&id) = self.grid_ids.get(row) else {
            return String::new();
        };
        let Some(game) = self.cat.game(id) else {
            return String::new();
        };
        let bga = self
            .cat
            .platform_by_slug("bga")
            .map(|p| game.has_platform(p.id))
            .unwrap_or(false);
        let tts = self
            .cat
            .platform_by_slug("tts")
            .map(|p| game.has_platform(p.id))
            .unwrap_or(false);
        match col {
            0 => game.name.clone(),
            1 => {
                if game.year > 0 {
                    game.year.to_string()
                } else {
                    String::new()
                }
            }
            2 => format!("{:.2}", game.rating_bayes),
            3 => format!("{:.2}", game.weight),
            4 => {
                if game.min_players > 0 || game.max_players > 0 {
                    format!("{}–{}", game.min_players, game.max_players)
                } else {
                    String::new()
                }
            }
            5 => game.best_at_summary(),
            6 => game
                .owner_ids
                .iter()
                .map(|id| self.cat.person_name(*id))
                .collect::<Vec<_>>()
                .join(", "),
            7 => if bga { "yes" } else { "" }.into(),
            8 => if tts { "yes" } else { "" }.into(),
            9 => game.local_notes.clone(),
            _ => String::new(),
        }
    }

    fn card_columns(width: f64) -> usize {
        const AIM: f64 = 240.0;
        const GAP: f64 = 12.0;
        const MAX: usize = 4;
        (((width + GAP) / (AIM + GAP)) as usize).clamp(1, MAX)
    }

    fn playtime_label(game: &Game) -> String {
        if game.min_playtime > 0
            && game.max_playtime > 0
            && game.min_playtime != game.max_playtime
        {
            format!("{}–{}m", game.min_playtime, game.max_playtime)
        } else if game.playtime > 0 {
            format!("{}m", game.playtime)
        } else if game.max_playtime > 0 {
            format!("{}m", game.max_playtime)
        } else {
            String::new()
        }
    }

    fn owners_line(&self, game: &Game) -> String {
        let mut marks = Vec::new();
        for id in &game.owner_ids {
            let name = self.cat.person_name(*id);
            if *id == self.cat.me_person_id || name == "Me" {
                marks.push("You".into());
            } else {
                marks.push(name);
            }
        }
        if self
            .cat
            .shelf_by_slug("wishlist")
            .map(|s| game.on_shelf(s.id))
            .unwrap_or(false)
        {
            marks.push("Wishlist".into());
        }
        if self
            .cat
            .platform_by_slug("bga")
            .map(|p| game.has_platform(p.id))
            .unwrap_or(false)
        {
            marks.push("BGA".into());
        }
        if self
            .cat
            .platform_by_slug("tts")
            .map(|p| game.has_platform(p.id))
            .unwrap_or(false)
        {
            marks.push("TTS".into());
        }
        marks.join(" · ")
    }

    fn fill_game_card(&self, cx: &mut Cx, card: &WidgetRef, game: &Game) {
        card.label(cx, ids!(card_title)).set_text(cx, &game.name);
        card.label(cx, ids!(card_time))
            .set_text(cx, &Self::playtime_label(game));
        card.label(cx, ids!(card_owners))
            .set_text(cx, &self.owners_line(game));
        if let Some(mut meter) = card
            .widget(cx, ids!(rating_meter))
            .borrow_mut::<RatingMeter>()
        {
            meter.set_rating(game.rating_bayes);
        }
        if let Some(mut strip) = card
            .widget(cx, ids!(player_strip))
            .borrow_mut::<PlayerCountStrip>()
        {
            strip.set_data(
                &game.player_counts,
                self.night.players,
                game.min_players,
                game.max_players,
            );
        }
        let url = game.cover_url().to_string();
        if !url.is_empty() {
            let _ = card
                .image(cx, ids!(cover))
                .load_image_http_by_url_async(cx, &url);
        }
    }

    fn draw_cards(&mut self, cx: &mut Cx2d, list: &mut PortalList) {
        const GAP: f64 = 12.0;
        let width = {
            let turtle = cx.turtle().rect().size.x;
            if turtle > 1.0 {
                turtle
            } else {
                self.view.area().rect(cx).size.x
            }
        };
        let inner = (width - 14.0).max(0.0);
        let cols = Self::card_columns(inner);
        let card_w = ((inner - GAP * (cols - 1) as f64) / cols as f64).max(160.0);
        let total = self.night_cards.len();
        let rows = total.div_ceil(cols).max(1);
        list.set_item_range(cx, 0, rows);
        let slots = [ids!(c1), ids!(c2), ids!(c3), ids!(c4)];
        while let Some(row_id) = list.next_visible_item(cx) {
            let mut item = list.item(cx, row_id, id!(CardRow));
            script_apply_eval!(cx, item, {
                spacing: #(GAP)
                padding: mod.turtle.Inset{bottom: #(GAP)}
            });
            for slot in 0..4 {
                let index = row_id * cols + slot;
                let visible = slot < cols && index < total;
                let mut cell = item.view(cx, slots[slot]);
                cell.set_visible(cx, visible);
                if !visible {
                    continue;
                }
                script_apply_eval!(cx, cell, {
                    width: #(card_w)
                });
                if let Some(night) = self.night_cards.get(index) {
                    if let Some(game) = self.cat.game(night.game_id) {
                        self.fill_game_card(cx, &cell, game);
                    }
                }
            }
            item.draw_all(cx, &mut Scope::empty());
        }
    }

    fn draw_hits(&mut self, cx: &mut Cx2d, list: &mut PortalList) {
        list.set_item_range(cx, 0, self.search_hits.len());
        while let Some(index) = list.next_visible_item(cx) {
            let item = list.item(cx, index, id!(HitRow));
            if let Some(hit) = self.search_hits.get(index) {
                let year = if hit.year > 0 {
                    format!(" ({})", hit.year)
                } else {
                    String::new()
                };
                item.label(cx, ids!(hit_label)).set_text(
                    cx,
                    &format!("{}{} · {}", hit.name, year, hit.bgg_type),
                );
            }
            item.draw_all(cx, &mut Scope::empty());
        }
    }

    fn draw_owner_chips(&mut self, cx: &mut Cx2d, list: &mut PortalList) {
        let game_id = self.edit_game_id;
        list.set_item_range(cx, 0, self.cat.people.len());
        while let Some(index) = list.next_visible_item(cx) {
            let item = list.item(cx, index, id!(OwnerChip));
            if let Some(person) = self.cat.people.get(index) {
                let owned = game_id
                    .and_then(|gid| self.cat.game(gid))
                    .map(|g| g.owner_ids.contains(&person.id))
                    .unwrap_or(false);
                let label = if owned {
                    format!("✓ {}", person.name)
                } else {
                    person.name.clone()
                };
                item.as_button().set_text(cx, &label);
            }
            item.draw_all(cx, &mut Scope::empty());
        }
    }
}

impl Widget for BoardGames {
    fn draw_walk(&mut self, cx: &mut Cx2d, scope: &mut Scope, walk: Walk) -> DrawStep {
        self.start(cx);
        while let Some(step) = self.view.draw_walk(cx, scope, walk).step() {
            if let Some(mut list) = step.as_portal_list().borrow_mut() {
                if self.screen == Screen::Play {
                    self.draw_cards(cx, &mut list);
                } else if self.show_hits {
                    self.draw_hits(cx, &mut list);
                } else if self.edit_game_id.is_some() {
                    self.draw_owner_chips(cx, &mut list);
                }
                continue;
            }
            if let Some(mut grid) = step.as_data_grid().borrow_mut() {
                grid.set_grid_size(self.grid_ids.len(), COL_LABELS.len());
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
        if self.tick_timer.is_event(event).is_some() {
            self.bgg.tick(cx);
        }
        for action in self.bgg.handle_event(cx, event) {
            self.on_bgg(cx, action);
        }
        self.view.handle_event(cx, event, scope);
        self.widget_match_event(cx, event, scope);
    }
}

impl WidgetMatchEvent for BoardGames {
    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions, _scope: &mut Scope) {
        for screen in Screen::ALL {
            if self.button(cx, screen.nav_id()).clicked(actions) {
                self.set_screen(cx, screen);
            }
        }
        if self.button(cx, ids!(players_dec)).clicked(actions) {
            self.night.players = (self.night.players - 1).max(1);
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(players_inc)).clicked(actions) {
            self.night.players = (self.night.players + 1).min(12);
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(venue_table)).clicked(actions) {
            self.night.include_table = !self.night.include_table;
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(venue_bga)).clicked(actions) {
            self.night.include_bga = !self.night.include_bga;
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(venue_tts)).clicked(actions) {
            self.night.include_tts = !self.night.include_tts;
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(include_wishlist)).clicked(actions) {
            self.night.include_wishlist = !self.night.include_wishlist;
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(search_btn)).clicked(actions)
            || self.text_input(cx, ids!(add_query)).returned(actions).is_some()
        {
            self.do_search(cx);
        }
        if self.button(cx, ids!(demo_btn)).clicked(actions) {
            self.load_demo(cx);
        }
        if self.button(cx, ids!(import_btn)).clicked(actions) {
            self.import_xml(cx);
        }
        if let Some(text) = self.text_input(cx, ids!(filter_input)).changed(actions) {
            self.grid_filter = text;
            self.rebuild_grid_ids();
            self.redraw(cx);
        }
        if self.button(cx, ids!(close_overlay)).clicked(actions) {
            self.show_hits = false;
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(edit_close)).clicked(actions) {
            self.edit_game_id = None;
            self.reload(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(edit_save)).clicked(actions) {
            self.save_edit(cx);
        }
        if self.button(cx, ids!(edit_bga)).clicked(actions) {
            self.toggle_edit_platform(cx, "bga");
        }
        if self.button(cx, ids!(edit_tts)).clicked(actions) {
            self.toggle_edit_platform(cx, "tts");
        }
        if self.button(cx, ids!(owners_me)).clicked(actions) {
            self.night.owner_ids.clear();
            if self.cat.me_person_id > 0 {
                self.night.owner_ids.insert(self.cat.me_person_id);
            }
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }
        if self.button(cx, ids!(owners_all)).clicked(actions) {
            self.night.owner_ids = self.cat.people.iter().map(|p| p.id).collect();
            self.rebuild_night();
            self.refresh_chrome(cx);
            self.redraw(cx);
        }

        for (index, hit) in self.search_hits.iter().enumerate() {
            let item = self
                .portal_list(cx, ids!(hits_list))
                .item(cx, index, id!(HitRow));
            if item.button(cx, ids!(hit_pick)).clicked(actions) {
                let bgg_id = hit.bgg_id;
                self.status = format!("Fetching BGG thing {bgg_id}…");
                self.refresh_chrome(cx);
                self.bgg.fetch_thing(cx, bgg_id);
                break;
            }
        }

        if self.edit_game_id.is_some() {
            for (index, person) in self.cat.people.iter().enumerate() {
                let item = self
                    .portal_list(cx, ids!(edit_owners_list))
                    .item(cx, index, id!(OwnerChip));
                if item.as_button().clicked(actions) {
                    let person_id = person.id;
                    self.toggle_edit_owner(cx, person_id);
                    break;
                }
            }
        }

        let grid = self.data_grid(cx, ids!(catalogue_grid));
        for action in grid.actions(actions) {
            match action {
                DataGridAction::CellDoubleClicked { row, .. }
                | DataGridAction::CellClicked { row, .. } => {
                    if let Some(&id) = self.grid_ids.get(row) {
                        self.open_edit(cx, id);
                    }
                }
                _ => {}
            }
        }

        if self.night.owner_ids.is_empty() && self.cat.me_person_id > 0 {
            self.night.owner_ids.insert(self.cat.me_person_id);
        }
    }
}
