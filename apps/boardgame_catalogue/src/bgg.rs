//! BoardGameGeek XML API2 client over `cx.http_request`.

use crate::db::now_unix;
use crate::model::{Game, GameLink, GamePlatform, PlayerCount, Platform};
use makepad_widgets::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const USER_AGENT: &str =
    "MakePadBoardGames/0.1 (local catalogue; +https://github.com/makepad/makepad)";
const SEARCH_URL: &str = "https://boardgamegeek.com/xmlapi2/search";
const THING_URL: &str = "https://boardgamegeek.com/xmlapi2/thing";
const MIN_GAP: Duration = Duration::from_millis(700);

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub bgg_id: i64,
    pub name: String,
    pub year: i32,
    pub bgg_type: String,
}

#[derive(Clone, Debug)]
pub enum BggAction {
    SearchResults(Vec<SearchHit>),
    Thing(Game),
    Error(String),
    Status(String),
}

#[derive(Clone, Debug)]
enum Pending {
    Search { query: String },
    Thing { bgg_id: i64 },
}

struct InFlight {
    kind: Pending,
    attempt: u32,
}

struct Queued {
    kind: Pending,
    attempt: u32,
    ready_at: Instant,
}

pub struct BggClient {
    in_flight: HashMap<LiveId, InFlight>,
    last_sent: Option<Instant>,
    queued: Option<Queued>,
    platform_ids: HashMap<String, i64>,
    token: String,
}

impl Default for BggClient {
    fn default() -> Self {
        Self {
            in_flight: HashMap::new(),
            last_sent: None,
            queued: None,
            platform_ids: HashMap::new(),
            token: load_token(),
        }
    }
}

impl BggClient {
    pub fn set_token(&mut self, token: impl Into<String>) {
        self.token = token.into().trim().to_string();
    }

    pub fn has_token(&self) -> bool {
        !self.token.is_empty()
    }

    pub fn set_platforms(&mut self, platforms: &[Platform]) {
        self.platform_ids.clear();
        for p in platforms {
            self.platform_ids.insert(p.slug.clone(), p.id);
        }
    }

    pub fn search(&mut self, cx: &mut Cx, query: &str) {
        let query = query.trim().to_string();
        if query.is_empty() {
            return;
        }
        if !self.has_token() {
            return;
        }
        self.enqueue(cx, Pending::Search { query }, 0, Instant::now());
    }

    pub fn fetch_thing(&mut self, cx: &mut Cx, bgg_id: i64) {
        if !self.has_token() {
            return;
        }
        self.enqueue(cx, Pending::Thing { bgg_id }, 0, Instant::now());
    }

    pub fn busy(&self) -> bool {
        !self.in_flight.is_empty() || self.queued.is_some()
    }

    fn enqueue(&mut self, cx: &mut Cx, kind: Pending, attempt: u32, ready_at: Instant) {
        if self.in_flight.is_empty() && self.can_send_at(ready_at) {
            self.send(cx, kind, attempt);
        } else {
            self.queued = Some(Queued {
                kind,
                attempt,
                ready_at: ready_at.max(self.earliest_send()),
            });
        }
    }

    fn earliest_send(&self) -> Instant {
        match self.last_sent {
            None => Instant::now(),
            Some(t) => t + MIN_GAP,
        }
    }

    fn can_send_at(&self, ready_at: Instant) -> bool {
        Instant::now() >= ready_at && Instant::now() >= self.earliest_send()
    }

    fn send(&mut self, cx: &mut Cx, kind: Pending, attempt: u32) {
        let url = match &kind {
            Pending::Search { query } => {
                format!(
                    "{SEARCH_URL}?type=boardgame,boardgameexpansion&query={}",
                    url_encode(query)
                )
            }
            Pending::Thing { bgg_id } => format!("{THING_URL}?id={bgg_id}&stats=1"),
        };
        let mut request = HttpRequest::new(url, HttpMethod::GET);
        request.set_header("User-Agent".into(), USER_AGENT.into());
        request.set_header("Accept".into(), "application/xml".into());
        if !self.token.is_empty() {
            request.set_header(
                "Authorization".into(),
                format!("Bearer {}", self.token),
            );
        }
        let request_id = LiveId::unique();
        self.in_flight
            .insert(request_id, InFlight { kind, attempt });
        self.last_sent = Some(Instant::now());
        cx.http_request(request_id, request);
    }

    /// Drain a delayed queue after the rate-limit / 202 backoff.
    pub fn tick(&mut self, cx: &mut Cx) {
        if !self.in_flight.is_empty() {
            return;
        }
        let Some(q) = self.queued.take() else {
            return;
        };
        if self.can_send_at(q.ready_at) {
            self.send(cx, q.kind, q.attempt);
        } else {
            self.queued = Some(q);
        }
    }

    pub fn handle_event(&mut self, cx: &mut Cx, event: &Event) -> Vec<BggAction> {
        let mut out = Vec::new();
        let Event::NetworkResponses(responses) = event else {
            return out;
        };
        for response in responses {
            match response {
                NetworkResponse::HttpResponse {
                    request_id,
                    response,
                } => {
                    let Some(pending) = self.in_flight.remove(request_id) else {
                        continue;
                    };
                    self.on_http(cx, pending, response, &mut out);
                }
                NetworkResponse::HttpError { request_id, error } => {
                    if self.in_flight.remove(request_id).is_some() {
                        out.push(BggAction::Error(error.message.clone()));
                        self.flush_queue(cx);
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn on_http(
        &mut self,
        cx: &mut Cx,
        pending: InFlight,
        response: &HttpResponse,
        out: &mut Vec<BggAction>,
    ) {
        let body = response.get_string_body().unwrap_or_default();
        match response.status_code {
            200 => {
                match pending.kind {
                    Pending::Search { .. } => match parse_search(&body) {
                        Ok(hits) => out.push(BggAction::SearchResults(hits)),
                        Err(e) => out.push(BggAction::Error(e)),
                    },
                    Pending::Thing { bgg_id } => {
                        match parse_thing(&body, bgg_id, &self.platform_ids) {
                            Ok(game) => out.push(BggAction::Thing(game)),
                            Err(e) => out.push(BggAction::Error(e)),
                        }
                    }
                }
                self.flush_queue(cx);
            }
            202 => {
                if pending.attempt < 6 {
                    let wait = MIN_GAP.saturating_mul(2 + pending.attempt);
                    out.push(BggAction::Status(format!(
                        "BGG still preparing… retry {} in {}ms",
                        pending.attempt + 1,
                        wait.as_millis()
                    )));
                    self.enqueue(
                        cx,
                        pending.kind,
                        pending.attempt + 1,
                        Instant::now() + wait,
                    );
                } else {
                    out.push(BggAction::Error(
                        "BGG returned 202 too many times — try again later".into(),
                    ));
                    self.flush_queue(cx);
                }
            }
            401 | 403 => {
                out.push(BggAction::Error(
                    "BGG unauthorized — set BGG_TOKEN (Bearer from boardgamegeek.com/applications) \
                     or put the token in local/boardgames/bgg_token"
                        .into(),
                ));
                self.flush_queue(cx);
            }
            code => {
                out.push(BggAction::Error(format!(
                    "BGG HTTP {code}: {}",
                    body.chars().take(160).collect::<String>()
                )));
                self.flush_queue(cx);
            }
        }
    }

    fn flush_queue(&mut self, cx: &mut Cx) {
        if let Some(q) = self.queued.take() {
            if self.can_send_at(q.ready_at) {
                self.send(cx, q.kind, q.attempt);
            } else {
                self.queued = Some(q);
            }
        }
    }
}

pub fn parse_search(xml: &str) -> Result<Vec<SearchHit>, String> {
    let mut hits = Vec::new();
    for item in iter_tags(xml, "item") {
        let bgg_id = attr_i64(&item.open, "id").unwrap_or(0);
        if bgg_id == 0 {
            continue;
        }
        let bgg_type = attr(&item.open, "type").unwrap_or("boardgame").to_string();
        let mut name = String::new();
        let mut year = 0i32;
        for name_tag in iter_tags(&item.body, "name") {
            let ty = attr(&name_tag.open, "type").unwrap_or("");
            if ty == "primary" || name.is_empty() {
                if let Some(v) = attr(&name_tag.open, "value") {
                    name = xml_decode(v);
                }
            }
        }
        if let Some(y) = iter_tags(&item.body, "yearpublished").next() {
            year = attr_i64(&y.open, "value").unwrap_or(0) as i32;
        }
        if name.is_empty() {
            continue;
        }
        hits.push(SearchHit {
            bgg_id,
            name,
            year,
            bgg_type,
        });
    }
    Ok(hits)
}

pub fn parse_thing(
    xml: &str,
    expected_id: i64,
    platform_ids: &HashMap<String, i64>,
) -> Result<Game, String> {
    let item = iter_tags(xml, "item")
        .find(|it| attr_i64(&it.open, "id") == Some(expected_id))
        .or_else(|| iter_tags(xml, "item").next())
        .ok_or_else(|| "no <item> in BGG thing response".to_string())?;

    let bgg_id = attr_i64(&item.open, "id").unwrap_or(expected_id);
    let bgg_type = attr(&item.open, "type")
        .unwrap_or("boardgame")
        .to_string();

    let mut game = Game {
        bgg_id: Some(bgg_id),
        bgg_type,
        bgg_synced_at: now_unix(),
        bgg_payload: xml.to_string(),
        ..Game::default()
    };

    for name_tag in iter_tags(&item.body, "name") {
        if attr(&name_tag.open, "type").unwrap_or("") == "primary" {
            if let Some(v) = attr(&name_tag.open, "value") {
                game.name = xml_decode(v);
            }
        }
    }
    if game.name.is_empty() {
        if let Some(name_tag) = iter_tags(&item.body, "name").next() {
            if let Some(v) = attr(&name_tag.open, "value") {
                game.name = xml_decode(v);
            }
        }
    }

    game.year = first_value_i32(&item.body, "yearpublished");
    game.min_players = first_value_i32(&item.body, "minplayers");
    game.max_players = first_value_i32(&item.body, "maxplayers");
    game.playtime = first_value_i32(&item.body, "playingtime");
    game.min_playtime = first_value_i32(&item.body, "minplaytime");
    game.max_playtime = first_value_i32(&item.body, "maxplaytime");
    game.min_age = first_value_i32(&item.body, "minage");
    game.thumbnail_url = first_text(&item.body, "thumbnail");
    game.image_url = first_text(&item.body, "image");

    // Player count poll.
    for poll in iter_tags(&item.body, "poll") {
        if attr(&poll.open, "name") != Some("suggested_numplayers") {
            continue;
        }
        game.poll_votes = attr_i64(&poll.open, "totalvotes").unwrap_or(0) as i32;
        for results in iter_tags(&poll.body, "results") {
            let Some(numplayers) = attr(&results.open, "numplayers") else {
                continue;
            };
            let (player_n, plus) = PlayerCount::parse_numplayers(numplayers);
            let mut best = 0i32;
            let mut rec = 0i32;
            let mut not_rec = 0i32;
            for result in iter_tags(&results.body, "result") {
                let value = attr(&result.open, "value").unwrap_or("");
                let votes = attr_i64(&result.open, "numvotes").unwrap_or(0) as i32;
                match value {
                    "Best" => best = votes,
                    "Recommended" => rec = votes,
                    "Not Recommended" => not_rec = votes,
                    _ => {}
                }
            }
            game.player_counts.push(PlayerCount {
                numplayers: numplayers.to_string(),
                player_n,
                plus,
                best_votes: best,
                rec_votes: rec,
                not_rec_votes: not_rec,
            });
        }
    }
    game.player_counts
        .sort_by_key(|p| (p.plus, p.player_n));

    // poll-summary (note BGG typo recommmendedwith)
    for summary in iter_tags(&item.body, "poll-summary") {
        if attr(&summary.open, "name") != Some("suggested_numplayers") {
            continue;
        }
        for result in iter_tags(&summary.body, "result") {
            let name = attr(&result.open, "name").unwrap_or("");
            let value = attr(&result.open, "value")
                .map(xml_decode)
                .unwrap_or_default();
            match name {
                "bestwith" => game.best_with = value,
                "recommmendedwith" | "recommendedwith" => game.recommended_with = value,
                _ => {}
            }
        }
    }

    // Statistics / ratings.
    if let Some(stats) = iter_tags(&item.body, "statistics").next() {
        if let Some(ratings) = iter_tags(&stats.body, "ratings").next() {
            game.rating_average = first_value_f64(&ratings.body, "average");
            game.rating_bayes = first_value_f64(&ratings.body, "bayesaverage");
            game.users_rated = first_value_i32(&ratings.body, "usersrated");
            game.weight = first_value_f64(&ratings.body, "averageweight");
            for ranks in iter_tags(&ratings.body, "ranks") {
                for rank in iter_tags(&ranks.body, "rank") {
                    if attr(&rank.open, "name") == Some("boardgame") {
                        let v = attr(&rank.open, "value").unwrap_or("Not Ranked");
                        game.rating_rank = if v.eq_ignore_ascii_case("Not Ranked") {
                            0
                        } else {
                            v.parse().unwrap_or(0)
                        };
                    }
                }
            }
        }
    }

    // Links + platform inference.
    let keep_kinds = [
        "boardgamecategory",
        "boardgamemechanic",
        "boardgamedesigner",
        "boardgamepublisher",
    ];
    for link in iter_tags(&item.body, "link") {
        let kind = attr(&link.open, "type").unwrap_or("").to_string();
        let id = attr_i64(&link.open, "id").unwrap_or(0);
        let name = attr(&link.open, "value")
            .map(xml_decode)
            .unwrap_or_default();
        if keep_kinds.iter().any(|k| *k == kind) {
            game.links.push(GameLink {
                kind: kind.clone(),
                bgg_id: id,
                name: name.clone(),
            });
        }
        infer_platforms(&mut game, platform_ids, &kind, &name);
    }

    // Also scan description / families text lightly via link names already done.
    if game.name.is_empty() {
        return Err("thing missing primary name".into());
    }
    Ok(game)
}

fn infer_platforms(
    game: &mut Game,
    platform_ids: &HashMap<String, i64>,
    kind: &str,
    name: &str,
) {
    let lower = name.to_ascii_lowercase();
    let is_family = kind.contains("family") || kind.contains("implementation");
    let looks_bga = lower.contains("board game arena") || lower == "bga";
    let looks_tts = lower.contains("tabletop simulator") || lower.contains("tabletopsim");
    if looks_bga {
        if let Some(&id) = platform_ids.get("bga") {
            if !game.platforms.iter().any(|p| p.platform_id == id) {
                game.platforms.push(GamePlatform {
                    platform_id: id,
                    url: String::new(),
                    source: "bgg".into(),
                });
            }
        }
    }
    if looks_tts || (is_family && lower.contains("simulator") && lower.contains("tabletop")) {
        if let Some(&id) = platform_ids.get("tts") {
            if !game.platforms.iter().any(|p| p.platform_id == id) {
                game.platforms.push(GamePlatform {
                    platform_id: id,
                    url: String::new(),
                    source: "bgg".into(),
                });
            }
        }
    }
}

// ---- tiny XML helpers (BGG responses are attribute-heavy, shallow nesting)

struct Tag<'a> {
    open: &'a str,
    body: &'a str,
}

fn iter_tags<'a>(xml: &'a str, name: &'a str) -> TagIter<'a> {
    TagIter {
        xml,
        name,
        pos: 0,
    }
}

struct TagIter<'a> {
    xml: &'a str,
    name: &'a str,
    pos: usize,
}

impl<'a> Iterator for TagIter<'a> {
    type Item = Tag<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let open_pat = format!("<{}", self.name);
        let rest = &self.xml[self.pos..];
        let rel = rest.find(&open_pat)?;
        let start = self.pos + rel;
        let after_name = start + open_pat.len();
        // Ensure tag name boundary (space, >, or /).
        let boundary = self.xml.as_bytes().get(after_name).copied()?;
        if boundary != b' ' && boundary != b'>' && boundary != b'/' && boundary != b'\n' && boundary != b'\r' && boundary != b'\t' {
            self.pos = after_name;
            return self.next();
        }
        let open_end = self.xml[after_name..]
            .find('>')
            .map(|i| after_name + i)
            ?;
        let open = &self.xml[start..=open_end];
        if open.ends_with("/>") {
            self.pos = open_end + 1;
            return Some(Tag { open, body: "" });
        }
        let close_pat = format!("</{}>", self.name);
        let close_rel = self.xml[open_end + 1..].find(&close_pat)?;
        let body_start = open_end + 1;
        let body_end = open_end + 1 + close_rel;
        let body = &self.xml[body_start..body_end];
        self.pos = body_end + close_pat.len();
        Some(Tag { open, body })
    }
}

fn attr<'a>(open_tag: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("{key}=\"");
    let start = open_tag.find(&pat)? + pat.len();
    let end = open_tag[start..].find('"')? + start;
    Some(&open_tag[start..end])
}

fn attr_i64(open_tag: &str, key: &str) -> Option<i64> {
    attr(open_tag, key)?.parse().ok()
}

fn first_value_i32(xml: &str, tag: &str) -> i32 {
    iter_tags(xml, tag)
        .next()
        .and_then(|t| attr_i64(&t.open, "value"))
        .unwrap_or(0) as i32
}

fn first_value_f64(xml: &str, tag: &str) -> f64 {
    iter_tags(xml, tag)
        .next()
        .and_then(|t| attr(&t.open, "value"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn first_text(xml: &str, tag: &str) -> String {
    iter_tags(xml, tag)
        .next()
        .map(|t| xml_decode(t.body.trim()))
        .unwrap_or_default()
}

fn xml_decode(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn load_token() -> String {
    if let Ok(token) = std::env::var("BGG_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return token;
        }
    }
    let path = std::path::Path::new("local/boardgames/bgg_token");
    if let Ok(token) = std::fs::read_to_string(path) {
        return token.trim().to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_search_sample() {
        let xml = r#"<?xml version="1.0"?>
        <items>
          <item type="boardgame" id="13">
            <name type="primary" value="CATAN"/>
            <yearpublished value="1995"/>
          </item>
          <item type="boardgameexpansion" id="325">
            <name type="primary" value="Catan: Cities &amp; Knights"/>
            <yearpublished value="1998"/>
          </item>
        </items>"#;
        let hits = parse_search(xml).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].bgg_id, 13);
        assert_eq!(hits[0].name, "CATAN");
        assert_eq!(hits[1].name, "Catan: Cities & Knights");
    }

    #[test]
    fn parses_thing_poll() {
        let xml = r#"<?xml version="1.0"?>
        <items>
          <item type="boardgame" id="13">
            <name type="primary" value="CATAN"/>
            <yearpublished value="1995"/>
            <minplayers value="3"/>
            <maxplayers value="4"/>
            <thumbnail>https://example.com/t.jpg</thumbnail>
            <poll name="suggested_numplayers" totalvotes="10">
              <results numplayers="3">
                <result value="Best" numvotes="4"/>
                <result value="Recommended" numvotes="5"/>
                <result value="Not Recommended" numvotes="1"/>
              </results>
              <results numplayers="4">
                <result value="Best" numvotes="8"/>
                <result value="Recommended" numvotes="2"/>
                <result value="Not Recommended" numvotes="0"/>
              </results>
              <results numplayers="5+">
                <result value="Best" numvotes="0"/>
                <result value="Recommended" numvotes="1"/>
                <result value="Not Recommended" numvotes="9"/>
              </results>
            </poll>
            <poll-summary name="suggested_numplayers">
              <result name="bestwith" value="Best with 4 players"/>
              <result name="recommmendedwith" value="Recommended with 3–4 players"/>
            </poll-summary>
            <statistics>
              <ratings>
                <average value="7.1"/>
                <bayesaverage value="6.9"/>
                <usersrated value="100"/>
                <averageweight value="2.3"/>
                <ranks>
                  <rank type="subtype" name="boardgame" value="400"/>
                </ranks>
              </ratings>
            </statistics>
            <link type="boardgamecategory" id="1" value="Strategy"/>
            <link type="boardgamefamily" id="2" value="Board Game Arena implementation"/>
          </item>
        </items>"#;
        let mut platforms = HashMap::new();
        platforms.insert("bga".into(), 7i64);
        platforms.insert("tts".into(), 8i64);
        let game = parse_thing(xml, 13, &platforms).unwrap();
        assert_eq!(game.name, "CATAN");
        assert_eq!(game.min_players, 3);
        assert_eq!(game.max_players, 4);
        assert!((game.rating_bayes - 6.9).abs() < 0.01);
        assert_eq!(game.player_counts.len(), 3);
        assert_eq!(game.best_with, "Best with 4 players");
        assert!(game.platforms.iter().any(|p| p.platform_id == 7));
        let fit = game.fit_at(4);
        assert_eq!(fit.fit, crate::model::FitAtN::Best);
    }
}
