//! Official Discogs API: master search, versions, price suggestions, thumbs.
//!
//! Token from `DISCOGS_TOKEN` or `local/vinyl/discogs.token`. Never logged.
//! Prices come from `/marketplace/price_suggestions` (seller settings required).

use crate::model::{self, Record};
use makepad_network::blocking_http::{self, Limits, Request};
use makepad_strict_json::Value;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

pub const TOKEN_ENV: &str = "DISCOGS_TOKEN";
pub const HIGH_VALUE_GBP: f64 = 40.0;
/// Discogs seller settings — required for condition price suggestions (VG/NM).
pub const SELLER_SETTINGS_URL: &str = "https://www.discogs.com/settings/seller";
const TOKEN_FILE: &str = "discogs.token";
const API: &str = "https://api.discogs.com";
const USER_AGENT: &str = "MakePadVinyl/0.1 +https://github.com/makepad/makepad";
/// Known release used only to probe whether seller settings unlock suggestions.
const PROBE_RELEASE_ID: i64 = 23599; // Plastikman – Sheet One
const MAX_MASTERS: usize = 4;
const MAX_VERSIONS: usize = 8;
const POOL_CAP: usize = 15;
pub const RATE_GAP: Duration = Duration::from_millis(1100);

#[derive(Clone, Debug, Default)]
pub struct SearchQuery {
    pub artist: String,
    pub title: String,
    pub label: String,
    pub catalogue_number: String,
    pub notes: String,
}

impl SearchQuery {
    pub fn from_record(record: &Record) -> Self {
        Self {
            artist: record.artist.clone(),
            title: record.title.clone(),
            label: record.label.clone(),
            catalogue_number: record.catalogue_number.clone(),
            notes: record.notes.clone(),
        }
    }

    pub fn from_extracted(row: &crate::model::Extracted) -> Self {
        Self {
            artist: row.artist.clone(),
            title: row.title.clone(),
            label: row.label.clone(),
            catalogue_number: row.catalogue_number.clone(),
            notes: row.notes.clone(),
        }
    }

    pub fn heading(&self) -> String {
        let artist = self.artist.trim();
        let title = self.title.trim();
        match (artist.is_empty(), title.is_empty()) {
            (false, false) => format!("{artist} – {title}"),
            (false, true) => artist.to_string(),
            (true, false) => title.to_string(),
            (true, true) => {
                let cat = self.catalogue_number.trim();
                if !cat.is_empty() {
                    format!("{} {cat}", self.label.trim()).trim().to_string()
                } else {
                    "this row".to_string()
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MasterCandidate {
    pub id: i64,
    /// True when `id` is a release id (specific pressing).
    pub is_release: bool,
    /// Linked master when known (0 if release-only).
    pub master_id: i64,
    /// Specific pressing to price (0 if master-only until resolved).
    pub release_id: i64,
    pub title: String,
    pub year: String,
    pub country: String,
    pub label: String,
    pub catno: String,
    pub format: String,
    pub thumb_url: String,
    pub thumb_path: String,
}

impl MasterCandidate {
    pub fn url(&self) -> String {
        if self.release_id > 0 {
            format!("https://www.discogs.com/release/{}", self.release_id)
        } else if self.is_release {
            format!("https://www.discogs.com/release/{}", self.id)
        } else {
            format!("https://www.discogs.com/master/{}", self.id)
        }
    }

    /// Discogs-style short card (no company soup).
    pub fn summary(&self, index: usize) -> String {
        let (artist, track) = split_discogs_title(&self.title);
        let title = if track.is_empty() {
            self.title.clone()
        } else {
            track
        };
        let artist = if artist.is_empty() {
            String::new()
        } else {
            artist
        };
        let badge = format_badge(&self.format);
        let specs = format_specs(&self.format);
        let mut line2 = title;
        if !artist.is_empty() {
            line2 = format!("{line2} — {artist}");
        }
        let mut meta = Vec::new();
        if !specs.is_empty() {
            meta.push(specs);
        }
        if !self.year.is_empty() {
            meta.push(self.year.clone());
        }
        if !self.country.is_empty() {
            meta.push(self.country.clone());
        }
        if !self.catno.is_empty() {
            meta.push(self.catno.clone());
        }
        if meta.is_empty() {
            format!("{}. {badge}\n{line2}", index + 1)
        } else {
            format!("{}. {badge}\n{line2}\n{}", index + 1, meta.join(" · "))
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VersionCandidate {
    pub id: i64,
    pub title: String,
    pub year: String,
    pub country: String,
    pub format: String,
    pub label: String,
    pub catno: String,
    pub thumb_url: String,
    pub thumb_path: String,
    pub suggestions_json: String,
    pub vg_price: f64,
    pub nm_price: f64,
    pub num_for_sale: i64,
    pub community_have: i64,
}

impl VersionCandidate {
    pub fn url(&self) -> String {
        format!("https://www.discogs.com/release/{}", self.id)
    }

    pub fn summary(&self, index: usize) -> String {
        let mut bits = Vec::new();
        if !self.year.is_empty() {
            bits.push(self.year.clone());
        }
        if !self.country.is_empty() {
            bits.push(self.country.clone());
        }
        if !self.format.is_empty() {
            bits.push(self.format.clone());
        }
        if !self.catno.is_empty() {
            bits.push(self.catno.clone());
        }
        if self.vg_price > 0.0 {
            bits.push(format!("VG £{:.0}", self.vg_price));
        }
        if self.nm_price > 0.0 && (self.nm_price - self.vg_price).abs() > 0.5 {
            bits.push(format!("NM £{:.0}", self.nm_price));
        }
        format!("{}. {}", index + 1, bits.join(" · "))
    }

    pub fn is_high_value(&self) -> bool {
        self.vg_price >= HIGH_VALUE_GBP
    }
}

#[derive(Clone, Debug, Default)]
pub struct VersionsBundle {
    pub master_id: i64,
    pub master_thumb_path: String,
    pub master_vg: f64,
    pub versions: Vec<VersionCandidate>,
}

impl VersionsBundle {
    pub fn high_value_versions(&self) -> Vec<VersionCandidate> {
        self.versions
            .iter()
            .filter(|v| v.is_high_value())
            .cloned()
            .collect()
    }

    pub fn has_high_value(&self) -> bool {
        self.versions.iter().any(|v| v.is_high_value())
    }
}

/// Catalogue dir (`local/vinyl`) in the MakePad checkout, even if vinyl was
/// launched from `target/release`.
pub fn vinyl_dir() -> PathBuf {
    checkout_root()
        .map(|root| root.join("local/vinyl"))
        .unwrap_or_else(|| PathBuf::from("local/vinyl"))
}

pub fn thumbs_dir() -> PathBuf {
    vinyl_dir().join("thumbs")
}

pub fn token_path() -> PathBuf {
    vinyl_dir().join(TOKEN_FILE)
}

fn checkout_root() -> Option<PathBuf> {
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            starts.push(parent.to_path_buf());
        }
    }
    if let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
    {
        starts.push(root.to_path_buf());
    }
    for start in starts {
        let mut dir = start;
        for _ in 0..8 {
            if dir.join("Cargo.toml").is_file() && dir.join("local").is_dir() {
                return Some(dir);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    None
}

pub fn load_token() -> Result<String, String> {
    if let Ok(env) = std::env::var(TOKEN_ENV) {
        let token = env.trim().to_string();
        if !token.is_empty() {
            return Ok(token);
        }
    }
    let path = token_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => first_token_line(&text).ok_or_else(|| {
            format!(
                "Put a Discogs personal token in {} (https://www.discogs.com/settings/developers)",
                path.display()
            )
        }),
        Err(_) => {
            write_token_stub(&path);
            Err(format!(
                "No Discogs token — paste one into {} or set {TOKEN_ENV}",
                path.display()
            ))
        }
    }
}

fn first_token_line(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        return Some(line.to_string());
    }
    None
}

fn write_token_stub(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(
        path,
        "# Discogs personal access token\n\
         # https://www.discogs.com/settings/developers\n\
         # For VG/NM price suggestions (not just marketplace lows), also fill seller settings:\n\
         # https://www.discogs.com/settings/seller  (listing currency GBP, address, accept policy)\n\
         # Paste the token on the next line (no quotes):\n",
    );
}

/// Whether `/marketplace/price_suggestions` works for this token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SellerPricing {
    Ready,
    NeedsSetup,
    Error(String),
}

/// One cheap probe — Discogs only returns condition suggestions after seller settings exist.
pub fn probe_seller_pricing(token: &str) -> SellerPricing {
    match fetch_price_suggestions(PROBE_RELEASE_ID, token) {
        Ok(_) => SellerPricing::Ready,
        Err(e) if e.to_ascii_lowercase().contains("seller settings") => SellerPricing::NeedsSetup,
        Err(e) => SellerPricing::Error(e),
    }
}

/// Open Discogs seller settings in the default browser (macOS/Linux/Windows).
pub fn open_seller_settings() {
    let url = SELLER_SETTINGS_URL;
    let _ = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
}

/// Find up to three master/release candidates for a spoken/edited row.
pub fn search_masters(query: &SearchQuery, token: &str) -> Result<Vec<MasterCandidate>, String> {
    let mut pool: Vec<MasterCandidate> = Vec::new();
    // (is_release, id) — release-only hits must not collide with masters.
    let mut seen: std::collections::HashSet<(bool, i64)> = std::collections::HashSet::new();

    let cat = trim(&query.catalogue_number);
    let label = trim(&query.label);
    let artist = trim(&query.artist);
    let title = trim(&query.title);

    // 1. Catalogue number → release hits (strongest for 12"s).
    if !cat.is_empty() {
        let mut params = vec![("type", "release".into()), ("catno", cat.clone())];
        if !label.is_empty() {
            params.push(("label", label.clone()));
        }
        push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
    }

    // 2. Title on vinyl releases first (masters often surface CD).
    if pool.len() < POOL_CAP && !title.is_empty() {
        let before = pool.len();
        thread::sleep(RATE_GAP);
        let params = vec![
            ("type", "release".into()),
            ("title", title.clone()),
            ("format", "Vinyl".into()),
        ];
        push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
        if pool.len() == before {
            thread::sleep(RATE_GAP);
            let params = vec![
                ("type", "release".into()),
                ("q", title.clone()),
                ("format", "Vinyl".into()),
            ];
            push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
        }
    }

    // 3. Artist + title on vinyl releases.
    if pool.len() < POOL_CAP && !artist.is_empty() && !title.is_empty() {
        let q = format!("{artist} {title}");
        thread::sleep(RATE_GAP);
        let params = vec![
            ("type", "release".into()),
            ("q", q.clone()),
            ("format", "Vinyl".into()),
        ];
        push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
        if pool.len() < POOL_CAP {
            thread::sleep(RATE_GAP);
            let params = vec![
                ("type", "release".into()),
                ("artist", artist.clone()),
                ("title", title.clone()),
                ("format", "Vinyl".into()),
            ];
            push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
        }
    }

    // 4. Label (+ title in q), vinyl.
    if pool.len() < POOL_CAP && !label.is_empty() {
        let mut params = vec![
            ("type", "release".into()),
            ("label", label.clone()),
            ("format", "Vinyl".into()),
        ];
        if !title.is_empty() {
            params.push(("q", title.clone()));
        }
        thread::sleep(RATE_GAP);
        push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
    }

    // 5. Loose full-text leftovers (label/cat bits) — still prefer vinyl releases.
    let q_bits: Vec<&str> = [&artist, &title, &label, &cat]
        .into_iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    if pool.len() < POOL_CAP && !q_bits.is_empty() {
        let q = q_bits.join(" ");
        let already = !artist.is_empty() && !title.is_empty() && q == format!("{artist} {title}");
        if !already {
            thread::sleep(RATE_GAP);
            let params = vec![
                ("type", "release".into()),
                ("q", q),
                ("format", "Vinyl".into()),
            ];
            push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
        }
    }

    // 6. Artist-only last — noisy; keep vinyl filter.
    if pool.len() < POOL_CAP && !artist.is_empty() {
        thread::sleep(RATE_GAP);
        let params = vec![
            ("type", "release".into()),
            ("q", artist.clone()),
            ("format", "Vinyl".into()),
        ];
        push_search_hits(&mut pool, &mut seen, &get_json(&search_url(&params), token)?, POOL_CAP);
    }

    let mut ranked = rank_masters(query, pool);
    // Prefer vinyl pressings when the pool has any.
    let vinyl: Vec<MasterCandidate> = ranked
        .iter()
        .filter(|m| is_vinyl_format(&m.format.to_ascii_lowercase()))
        .cloned()
        .collect();
    if !vinyl.is_empty() {
        ranked = vinyl;
    }
    ranked.truncate(MAX_MASTERS);

    for master in &mut ranked {
        if master.thumb_path.is_empty() && !master.thumb_url.is_empty() {
            let stem = if master.is_release {
                format!("release-{}", master.id)
            } else {
                format!("master-{}", master.id)
            };
            if let Ok(path) = cache_thumb(&master.thumb_url, &stem, token) {
                master.thumb_path = path;
            }
        }
    }
    Ok(ranked)
}

fn push_search_hits(
    pool: &mut Vec<MasterCandidate>,
    seen: &mut std::collections::HashSet<(bool, i64)>,
    value: &Value,
    cap: usize,
) {
    for row in value.get("results").and_then(Value::as_arr).unwrap_or(&[]) {
        if pool.len() >= cap {
            break;
        }
        let Some((id, is_release)) = row_candidate_id(row) else { continue };
        if !seen.insert((is_release, id)) {
            continue;
        }
        pool.push(master_from_search_row(row, id, is_release));
    }
}

/// Prefer the specific pressing. Collapsing release→master loses format (CD vs 12").
fn row_candidate_id(row: &Value) -> Option<(i64, bool)> {
    let id = json_id(row.get("id"))?;
    match json_text(row.get("type")).as_str() {
        "master" => Some((id, false)),
        "release" => Some((id, true)),
        _ => None,
    }
}

/// Rank a pool of masters against the spoken query (fuzzy).
pub(crate) fn rank_masters(query: &SearchQuery, pool: Vec<MasterCandidate>) -> Vec<MasterCandidate> {
    let mut scored: Vec<(i32, usize, MasterCandidate)> = pool
        .into_iter()
        .enumerate()
        .map(|(order, master)| {
            let score = fuzzy_master_score(query, &master);
            (score, order, master)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, m)| m).collect()
}

fn fuzzy_master_score(query: &SearchQuery, master: &MasterCandidate) -> i32 {
    let artist = normalize_alpha(&query.artist);
    let title = normalize_alpha(&query.title);
    let label = normalize_alpha(&query.label);
    let cat = normalize_alpha(&query.catalogue_number);
    let hay_title = normalize_alpha(&master.title);
    let hay_label = normalize_alpha(&master.label);
    let hay_cat = normalize_alpha(&master.catno);
    let (disc_artist, disc_track) = split_discogs_title(&master.title);
    let disc_artist_n = normalize_alpha(&disc_artist);
    let disc_track_n = normalize_alpha(&disc_track);

    let mut score = 0i32;
    if !cat.is_empty() {
        score += (fuzzy_pair_score(&cat, &hay_cat) * 120) / 100;
    }
    if !label.is_empty() {
        score += (fuzzy_pair_score(&label, &hay_label) * 90) / 100;
        score += (fuzzy_pair_score(&label, &hay_title) * 35) / 100;
        score += best_token_score(&label, &master.title) / 2;
    }
    if !artist.is_empty() {
        // Prefer Discogs' left-of-dash artist when present.
        let against_artist = fuzzy_pair_score(&artist, &disc_artist_n)
            .max(fuzzy_pair_score(&artist, &hay_title))
            .max(best_token_score(&artist, &master.title));
        score += against_artist;
    }
    if !title.is_empty() {
        let against_title = fuzzy_pair_score(&title, &disc_track_n)
            .max(fuzzy_pair_score(&title, &hay_title))
            .max(best_token_score(&title, &master.title));
        score += against_title;
    }
    // Combined artist+title against Discogs "Artist - Title" string.
    if !artist.is_empty() || !title.is_empty() {
        let combined = format!("{artist}{title}");
        if !combined.is_empty() {
            score += fuzzy_pair_score(&combined, &hay_title) / 2;
        }
    }
    // Soft bonus when every spoken field hits something.
    let hits = [
        !artist.is_empty() && best_token_score(&artist, &master.title) >= 60,
        !title.is_empty()
            && (fuzzy_pair_score(&title, &disc_track_n) >= 60
                || best_token_score(&title, &master.title) >= 60),
        !label.is_empty() && fuzzy_pair_score(&label, &hay_label) >= 60,
    ]
    .into_iter()
    .filter(|b| *b)
    .count();
    score += (hits as i32) * 15;
    // Strong title match should beat remix credits that only mention the artist.
    if !title.is_empty() && fuzzy_pair_score(&title, &disc_track_n) >= 85 {
        score += 40;
    }
    // Prefer ordinary vinyl LPs over picture discs / white labels / reissues.
    score += edition_score(&master.format);
    score
}

/// Boost standard vinyl pressings; demote specialty editions.
pub(crate) fn edition_score(format: &str) -> i32 {
    let f = format.to_ascii_lowercase();
    if f.is_empty() {
        return 0;
    }
    let mut score = 0i32;
    if is_vinyl_format(&f) {
        score += 50;
    } else {
        score -= 60;
    }
    let specials = [
        ("picture", -45),
        ("white label", -40),
        ("test press", -50),
        ("test pressing", -50),
        ("promo", -30),
        ("reissue", -25),
        ("bioplastic", -20),
        ("limited", -12),
        ("mispress", -30),
        ("unofficial", -40),
    ];
    for (needle, pen) in specials {
        if f.contains(needle) {
            score += pen;
        }
    }
    let clean = !specials.iter().any(|(n, _)| f.contains(n));
    if clean && (f.contains("lp") || f.contains("album") || f.contains("12")) {
        score += 40;
    }
    // Fewer format tags ≈ closer to the Discogs "LP, Album" card.
    let tags = f.split(',').filter(|t| !t.trim().is_empty()).count();
    if tags <= 3 {
        score += 15;
    } else if tags >= 5 {
        score -= 10;
    }
    score
}

fn format_badge(format: &str) -> String {
    let f = format.to_ascii_lowercase();
    let mut bits = Vec::new();
    if is_vinyl_format(&f) {
        bits.push("VINYL");
    } else if f.contains("cd") {
        bits.push("CD");
    } else if !format.is_empty() {
        bits.push("RELEASE");
    }
    if f.contains("limited") {
        bits.push("LTD");
    }
    if f.contains("reissue") {
        bits.push("REISSUE");
    }
    if bits.is_empty() {
        "MATCH".into()
    } else {
        bits.join(" · ")
    }
}

fn format_specs(format: &str) -> String {
    format
        .split(',')
        .map(str::trim)
        .filter(|t| {
            let l = t.to_ascii_lowercase();
            !l.is_empty()
                && l != "vinyl"
                && !l.starts_with("33")
                && !l.starts_with("45")
                && !l.starts_with("78")
        })
        .take(4)
        .collect::<Vec<_>>()
        .join(", ")
}

fn primary_label(value: Option<&Value>) -> String {
    match value {
        Some(Value::Arr(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::Str(s) => {
                    let t = s.trim();
                    (!t.is_empty()).then_some(t.to_string())
                }
                _ => None,
            })
            .next()
            .unwrap_or_default(),
        Some(Value::Str(s)) => s
            .split(',')
            .next()
            .map(str::trim)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Discogs master titles are usually `"Artist - Track"`.
pub(crate) fn split_discogs_title(title: &str) -> (String, String) {
    if let Some((left, right)) = title.split_once(" - ") {
        (left.trim().to_string(), right.trim().to_string())
    } else if let Some((left, right)) = title.split_once('–') {
        (left.trim().to_string(), right.trim().to_string())
    } else {
        (String::new(), title.trim().to_string())
    }
}

/// Lowercase and strip non-alphanumerics: "Moody Man" → "moodyman".
pub(crate) fn normalize_alpha(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Best fuzzy score of `needle` against whitespace / dash tokens in `hay`.
fn best_token_score(needle_norm: &str, hay: &str) -> i32 {
    if needle_norm.is_empty() {
        return 0;
    }
    let mut best = fuzzy_pair_score(needle_norm, &normalize_alpha(hay));
    for tok in hay.split(|c: char| !c.is_ascii_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        best = best.max(fuzzy_pair_score(needle_norm, &normalize_alpha(tok)));
    }
    // Adjacent token pairs: "Don't Be" → dontbe against moodymann-dontbemisled chunks.
    let words: Vec<&str> = hay
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    for w in words.windows(2) {
        let pair = normalize_alpha(&format!("{}{}", w[0], w[1]));
        best = best.max(fuzzy_pair_score(needle_norm, &pair));
    }
    for w in words.windows(3) {
        let triple = normalize_alpha(&format!("{}{}{}", w[0], w[1], w[2]));
        best = best.max(fuzzy_pair_score(needle_norm, &triple));
    }
    best
}

/// 0..=100 similarity for two normalized strings (graded, not all-or-nothing).
pub(crate) fn fuzzy_pair_score(a: &str, b: &str) -> i32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    if a == b {
        return 100;
    }
    if a.contains(b) || b.contains(a) {
        let shorter = a.len().min(b.len());
        let longer = a.len().max(b.len());
        return (75 + (25 * shorter / longer.max(1))) as i32;
    }
    let max_len = a.len().max(b.len()).max(1);
    // Allow up to ~30% edits (e.g. moodyman↔moodymann, misled↔mislead).
    let max_ed = ((max_len * 30) / 100).clamp(1, 4);
    if a.len().abs_diff(b.len()) > max_ed {
        return 0;
    }
    match levenshtein_distance(a.as_bytes(), b.as_bytes(), max_ed) {
        Some(0) => 100,
        Some(d) => {
            let ratio = 1.0 - (d as f32 / max_len as f32);
            if ratio < 0.55 {
                0
            } else {
                (ratio * 95.0) as i32
            }
        }
        None => 0,
    }
}

fn levenshtein_distance(a: &[u8], b: &[u8], max: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            row_min = row_min.min(cur[j + 1]);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b.len()];
    (d <= max).then_some(d)
}

fn master_from_search_row(row: &Value, id: i64, is_release: bool) -> MasterCandidate {
    let mut thumb = json_text(row.get("thumb"));
    if thumb.is_empty() {
        thumb = json_text(row.get("cover_image"));
    }
    let master_id = if is_release {
        json_id(row.get("master_id")).unwrap_or(0)
    } else {
        id
    };
    let release_id = if is_release { id } else { 0 };
    let mut format = json_join(row.get("format"));
    if format.is_empty() {
        format = json_join(row.get("major_formats"));
    }
    MasterCandidate {
        id,
        is_release,
        master_id,
        release_id,
        title: json_text(row.get("title")),
        year: json_text(row.get("year")),
        country: json_text(row.get("country")),
        label: primary_label(row.get("label")),
        catno: json_text(row.get("catno")),
        format,
        thumb_url: thumb,
        thumb_path: String::new(),
    }
}

/// Price a catalogue row from its specific release. Never uses master-level
/// marketplace totals (those mix CD / digital / vinyl).
///
/// Returns a bundle whose first version id is the release that was priced —
/// callers can persist that as `discogs_release_id`.
pub fn load_price_for_row(
    release_id: i64,
    master_id: i64,
    preferred_catno: &str,
    token: &str,
) -> Result<VersionsBundle, String> {
    let rid = if release_id > 0 {
        release_id
    } else if master_id > 0 {
        resolve_vinyl_release(master_id, preferred_catno, token)?
    } else {
        return Err("no Discogs release or master to price".into());
    };
    price_from_release_id(rid, master_id, token)
}

/// Among a master's versions, pick a vinyl pressing (prefer catno match).
pub fn resolve_vinyl_release(
    master_id: i64,
    preferred_catno: &str,
    token: &str,
) -> Result<i64, String> {
    let versions_json = get_json(
        &format!("{API}/masters/{master_id}/versions?per_page=100"),
        token,
    )?;
    let want_cat = normalize_alpha(preferred_catno);
    let mut best: Option<(i32, i64, i64)> = None; // score, have, id
    for row in versions_json
        .get("versions")
        .and_then(Value::as_arr)
        .unwrap_or(&[])
    {
        let Some(id) = json_id(row.get("id")) else { continue };
        let mut format = json_join(row.get("major_formats"));
        if format.is_empty() {
            format = json_join(row.get("format"));
        }
        let fmt_l = format.to_ascii_lowercase();
        if !is_vinyl_format(&fmt_l) {
            continue;
        }
        let have = nested_i64(row, &["stats", "community", "in_collection"])
            .or_else(|| nested_i64(row, &["community", "have"]))
            .unwrap_or(0);
        let cat = normalize_alpha(&json_text(row.get("catno")));
        let mut score = 10;
        if !want_cat.is_empty() && !cat.is_empty() {
            score += fuzzy_pair_score(&want_cat, &cat);
        }
        if fmt_l.contains("12") {
            score += 15;
        } else if fmt_l.contains("7") {
            score += 10;
        } else if fmt_l.contains("lp") {
            score += 8;
        }
        if fmt_l.contains("test") {
            score -= 20;
        }
        let key = (score, have, id);
        if best.map(|b| key > b).unwrap_or(true) {
            best = Some(key);
        }
    }
    best.map(|(_, _, id)| id)
        .ok_or_else(|| format!("master {master_id}: no vinyl version found"))
}

fn is_vinyl_format(fmt_lower: &str) -> bool {
    if fmt_lower.is_empty() {
        return false;
    }
    // Explicit non-vinyl first — "Vinyl, CD" still counts as vinyl-capable.
    let only_digital = !fmt_lower.contains("vinyl")
        && !fmt_lower.contains("12")
        && !fmt_lower.contains("7")
        && !fmt_lower.contains("10")
        && !fmt_lower.contains("lp")
        && !fmt_lower.contains("acetate");
    if only_digital {
        return false;
    }
    fmt_lower.contains("vinyl")
        || fmt_lower.contains("12\"")
        || fmt_lower.contains("12'")
        || fmt_lower.contains("12")
        || fmt_lower.contains("7\"")
        || fmt_lower.contains("7'")
        || fmt_lower.contains("lp")
        || fmt_lower.contains("acetate")
}

fn price_from_release_id(
    release_id: i64,
    master_id: i64,
    token: &str,
) -> Result<VersionsBundle, String> {
    let rel = get_json(&format!("{API}/releases/{release_id}"), token)?;
    let mut thumb_url = json_text(rel.get("thumb"));
    if thumb_url.is_empty() {
        if let Some(arr) = rel.get("images").and_then(Value::as_arr) {
            if let Some(img) = arr.first() {
                thumb_url = json_text(img.get("uri150"));
                if thumb_url.is_empty() {
                    thumb_url = json_text(img.get("uri"));
                }
            }
        }
    }
    let thumb_path = if thumb_url.is_empty() {
        String::new()
    } else {
        cache_thumb(&thumb_url, &format!("release-{release_id}"), token).unwrap_or_default()
    };

    // Prefer Discogs condition suggestions (closer to fair value) when the
    // token's account has seller settings. Median sold price on the website
    // is sales-history only — not in the public API.
    let mut vg = 0.0;
    let mut suggestions = String::new();
    let mut nm = 0.0;
    let mut used_low = false;
    thread::sleep(RATE_GAP);
    match fetch_price_suggestions(release_id, token) {
        Ok(json) => {
            suggestions = json.clone();
            vg = model::price_for_condition(&json, "VG").unwrap_or(0.0);
            nm = model::price_for_condition(&json, "NM")
                .or_else(|| model::price_for_condition(&json, "VG+"))
                .unwrap_or(0.0);
        }
        Err(_) => {}
    }
    if vg <= 0.0 {
        vg = json_f64(rel.get("lowest_price")).unwrap_or(0.0);
        if vg <= 0.0 {
            thread::sleep(RATE_GAP);
            if let Ok(stats) = get_json(&format!("{API}/marketplace/stats/{release_id}"), token) {
                vg = stats
                    .get("lowest_price")
                    .and_then(|lp| json_f64(lp.get("value")).or_else(|| json_f64(Some(lp))))
                    .unwrap_or(0.0);
            }
        }
        used_low = vg > 0.0;
    }
    let _used_low = used_low;

    let have = nested_i64(&rel, &["community", "have"]).unwrap_or(0);
    let for_sale = rel
        .get("num_for_sale")
        .and_then(|v| match v {
            Value::Int(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0);

    let linked_master = json_id(rel.get("master_id")).unwrap_or(master_id);

    Ok(VersionsBundle {
        master_id: linked_master,
        master_thumb_path: thumb_path.clone(),
        master_vg: vg,
        versions: vec![VersionCandidate {
            id: release_id,
            title: json_text(rel.get("title")),
            year: {
                let mut y = json_text(rel.get("released"));
                if y.is_empty() {
                    y = json_text(rel.get("year"));
                }
                y
            },
            country: json_text(rel.get("country")),
            format: {
                let mut format = String::new();
                if let Some(arr) = rel.get("formats").and_then(Value::as_arr) {
                    format = arr
                        .iter()
                        .filter_map(|f| {
                            let name = json_text(f.get("name"));
                            (!name.is_empty()).then_some(name)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                format
            },
            label: String::new(),
            catno: String::new(),
            thumb_url,
            thumb_path,
            suggestions_json: suggestions,
            vg_price: vg,
            nm_price: nm,
            num_for_sale: for_sale,
            community_have: have,
        }],
    })
}

/// Load versions for a master, attach price suggestions, sort by VG desc.
pub fn load_versions(master_id: i64, token: &str) -> Result<VersionsBundle, String> {
    thread::sleep(RATE_GAP);
    let master = get_json(&format!("{API}/masters/{master_id}"), token)?;
    let master_thumb_url = {
        let mut url = json_text(master.get("thumb"));
        if url.is_empty() {
            if let Some(arr) = master.get("images").and_then(Value::as_arr) {
                if let Some(img) = arr.first() {
                    url = json_text(img.get("uri150"));
                    if url.is_empty() {
                        url = json_text(img.get("uri"));
                    }
                }
            }
        }
        url
    };
    let master_thumb_path = if master_thumb_url.is_empty() {
        String::new()
    } else {
        cache_thumb(&master_thumb_url, &format!("master-{master_id}"), token).unwrap_or_default()
    };

    thread::sleep(RATE_GAP);
    let versions_json = get_json(
        &format!("{API}/masters/{master_id}/versions?per_page=50"),
        token,
    )?;
    let mut raw: Vec<VersionCandidate> = Vec::new();
    for row in versions_json
        .get("versions")
        .and_then(Value::as_arr)
        .unwrap_or(&[])
    {
        let Some(id) = json_id(row.get("id")) else { continue };
        let have = nested_i64(row, &["community", "have"]).unwrap_or(0);
        let for_sale = nested_i64(row, &["stats", "marketplace", "for_sale"])
            .or_else(|| nested_i64(row, &["stats", "num_for_sale"]))
            .unwrap_or(0);
        let mut format = json_join(row.get("format"));
        if format.is_empty() {
            format = json_join(row.get("major_formats"));
        }
        let mut year = json_text(row.get("released"));
        if year.is_empty() {
            year = json_text(row.get("year"));
        }
        raw.push(VersionCandidate {
            id,
            title: json_text(row.get("title")),
            year,
            country: json_text(row.get("country")),
            format,
            label: json_join(row.get("label")),
            catno: json_text(row.get("catno")),
            thumb_url: json_text(row.get("thumb")),
            thumb_path: String::new(),
            suggestions_json: String::new(),
            vg_price: 0.0,
            nm_price: 0.0,
            num_for_sale: for_sale,
            community_have: have,
        });
    }

    // Prefer well-held pressings, then take top MAX_VERSIONS for pricing.
    raw.sort_by(|a, b| b.community_have.cmp(&a.community_have));
    raw.truncate(MAX_VERSIONS);

    price_versions(master_id, master_thumb_path, raw, token)
}

/// Single release with no Discogs master — price suggestions for that pressing only.
pub fn load_release(release_id: i64, token: &str) -> Result<VersionsBundle, String> {
    thread::sleep(RATE_GAP);
    let rel = get_json(&format!("{API}/releases/{release_id}"), token)?;
    let mut thumb_url = json_text(rel.get("thumb"));
    if thumb_url.is_empty() {
        if let Some(arr) = rel.get("images").and_then(Value::as_arr) {
            if let Some(img) = arr.first() {
                thumb_url = json_text(img.get("uri150"));
                if thumb_url.is_empty() {
                    thumb_url = json_text(img.get("uri"));
                }
            }
        }
    }
    let thumb_path = if thumb_url.is_empty() {
        String::new()
    } else {
        cache_thumb(&thumb_url, &format!("release-{release_id}"), token).unwrap_or_default()
    };

    let have = nested_i64(&rel, &["community", "have"]).unwrap_or(0);
    let for_sale = rel
        .get("num_for_sale")
        .and_then(|v| match v {
            Value::Int(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0);
    let mut format = String::new();
    if let Some(arr) = rel.get("formats").and_then(Value::as_arr) {
        format = arr
            .iter()
            .filter_map(|f| {
                let name = json_text(f.get("name"));
                (!name.is_empty()).then_some(name)
            })
            .collect::<Vec<_>>()
            .join(", ");
    }
    let mut year = json_text(rel.get("released"));
    if year.is_empty() {
        year = json_text(rel.get("year"));
    }
    let label = rel
        .get("labels")
        .and_then(Value::as_arr)
        .map(|labels| {
            labels
                .iter()
                .filter_map(|l| {
                    let name = json_text(l.get("name"));
                    (!name.is_empty()).then_some(name)
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let catno = rel
        .get("labels")
        .and_then(Value::as_arr)
        .and_then(|labels| labels.first())
        .map(|l| json_text(l.get("catno")))
        .unwrap_or_default();

    let raw = vec![VersionCandidate {
        id: release_id,
        title: json_text(rel.get("title")),
        year,
        country: json_text(rel.get("country")),
        format,
        label,
        catno,
        thumb_url,
        thumb_path: thumb_path.clone(),
        suggestions_json: String::new(),
        vg_price: 0.0,
        nm_price: 0.0,
        num_for_sale: for_sale,
        community_have: have,
    }];
    price_versions(0, thumb_path, raw, token)
}

fn price_versions(
    master_id: i64,
    master_thumb_path: String,
    raw: Vec<VersionCandidate>,
    token: &str,
) -> Result<VersionsBundle, String> {
    let mut priced = Vec::new();
    let mut lowest_prices = Vec::new();
    for mut ver in raw {
        thread::sleep(RATE_GAP);
        match fetch_price_suggestions(ver.id, token) {
            Ok(json) => {
                ver.suggestions_json = json.clone();
                ver.vg_price = model::price_for_condition(&json, "VG").unwrap_or(0.0);
                ver.nm_price = model::price_for_condition(&json, "NM")
                    .or_else(|| model::price_for_condition(&json, "VG+"))
                    .unwrap_or(0.0);
            }
            Err(_) => {
                // fall through — try release lowest_price
            }
        }
        if ver.vg_price <= 0.0 {
            thread::sleep(RATE_GAP);
            if let Ok(rel) = get_json(&format!("{API}/releases/{}", ver.id), token) {
                if let Some(low) = json_f64(rel.get("lowest_price")) {
                    lowest_prices.push(low);
                    ver.vg_price = low;
                }
                if ver.thumb_url.is_empty() {
                    ver.thumb_url = json_text(rel.get("thumb"));
                }
                if ver.num_for_sale == 0 {
                    ver.num_for_sale = rel
                        .get("num_for_sale")
                        .and_then(|v| match v {
                            Value::Int(n) => Some(*n),
                            _ => None,
                        })
                        .unwrap_or(0);
                }
            }
        }
        if !ver.thumb_url.is_empty() && ver.thumb_path.is_empty() {
            if let Ok(path) = cache_thumb(&ver.thumb_url, &format!("release-{}", ver.id), token) {
                ver.thumb_path = path;
            }
        }
        priced.push(ver);
    }

    // If suggestions were empty and we only have lowests, use median of lows
    // for any still-zero VG.
    if !lowest_prices.is_empty() {
        let median = median_f64(&lowest_prices);
        for ver in &mut priced {
            if ver.vg_price <= 0.0 {
                ver.vg_price = median;
            }
        }
    }

    priced.sort_by(|a, b| {
        b.vg_price
            .partial_cmp(&a.vg_price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let vg_values: Vec<f64> = priced.iter().map(|v| v.vg_price).filter(|v| *v > 0.0).collect();
    let master_vg = if vg_values.is_empty() {
        0.0
    } else {
        median_f64(&vg_values)
    };

    Ok(VersionsBundle {
        master_id,
        master_thumb_path,
        master_vg,
        versions: priced,
    })
}

fn fetch_price_suggestions(release_id: i64, token: &str) -> Result<String, String> {
    let value = get_json(
        &format!("{API}/marketplace/price_suggestions/{release_id}"),
        token,
    )?;
    // Empty object means seller settings missing or no data.
    match &value {
        Value::Obj(pairs) if pairs.is_empty() => Err("empty price suggestions".into()),
        Value::Obj(_) => {
            let mut out = String::new();
            value.write_into(&mut out);
            Ok(out)
        }
        _ => Err("unexpected price suggestions shape".into()),
    }
}

fn cache_thumb(url: &str, stem: &str, token: &str) -> Result<String, String> {
    if url.is_empty() {
        return Err("no thumb url".into());
    }
    let dir = thumbs_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("thumbs dir: {e}"))?;
    let path = dir.join(format!("{stem}.jpg"));
    if path.is_file() {
        return Ok(path.display().to_string());
    }
    let bytes = get_bytes(url, token)?;
    std::fs::write(&path, bytes).map_err(|e| format!("write thumb: {e}"))?;
    Ok(path.display().to_string())
}

fn get_bytes(url: &str, token: &str) -> Result<Vec<u8>, String> {
    let limits = Limits {
        max_body_bytes: 2 * 1024 * 1024,
        total_timeout: Duration::from_secs(20),
        ..Limits::default()
    };
    let mut req = Request::get(url)
        .limits(limits)
        .user_agent(USER_AGENT)
        .map_err(|e| format!("discogs thumb: {e}"))?;
    // img.discogs.com may not need auth; still send token for api hosts
    if url.contains("api.discogs.com") {
        req = req
            .header("Authorization", &format!("Discogs token={token}"))
            .map_err(|e| format!("discogs thumb: {e}"))?;
    }
    let response = blocking_http::request_no_redirect(req).map_err(|e| format!("discogs thumb: {e}"))?;
    if response.status != 200 {
        return Err(format!("thumb HTTP {}", response.status));
    }
    Ok(response.body)
}

pub(crate) type Attempt = (&'static str, Vec<(&'static str, String)>);

pub(crate) fn search_attempts(query: &SearchQuery) -> Vec<Attempt> {
    let cat = trim(&query.catalogue_number);
    let label = trim(&query.label);
    let artist = trim(&query.artist);
    let title = trim(&query.title);
    let notes = trim(&query.notes);
    let mut out = Vec::new();
    if !cat.is_empty() {
        let mut params = vec![("catno", cat.clone())];
        if !label.is_empty() {
            params.push(("label", label.clone()));
        }
        out.push(("catalogue", params));
    }
    if !artist.is_empty() && !title.is_empty() {
        if !label.is_empty() {
            out.push((
                "artist+title+label",
                vec![
                    ("artist", artist.clone()),
                    ("title", title.clone()),
                    ("label", label.clone()),
                ],
            ));
        }
        out.push((
            "artist+title",
            vec![("artist", artist.clone()), ("title", title.clone())],
        ));
        if !notes.is_empty() {
            out.push((
                "artist+title+notes",
                vec![("q", format!("{artist} {title} {notes}"))],
            ));
        }
    } else if out.is_empty() {
        let mut bits = Vec::new();
        for part in [&artist, &title, &label, &cat, &notes] {
            if !part.is_empty() {
                bits.push(part.as_str());
            }
        }
        if !bits.is_empty() {
            out.push(("q", vec![("q", bits.join(" "))]));
        }
    }
    out
}

fn trim(value: &str) -> String {
    value.trim().to_string()
}

fn search_url(params: &[(&str, String)]) -> String {
    let mut url = format!("{API}/database/search?per_page=15");
    for (key, value) in params {
        if value.trim().is_empty() {
            continue;
        }
        url.push('&');
        url.push_str(key);
        url.push('=');
        url.push_str(&percent_encode(value.trim()));
    }
    url
}

fn percent_encode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push_str(&format!("{byte:02X}"));
        }
    }
    out
}

fn get_json(url: &str, token: &str) -> Result<Value, String> {
    let limits = Limits {
        max_body_bytes: 1024 * 1024,
        total_timeout: Duration::from_secs(25),
        ..Limits::default()
    };
    let req = Request::get(url)
        .limits(limits)
        .user_agent(USER_AGENT)
        .map_err(|e| format!("discogs request: {e}"))?
        .header("Authorization", &format!("Discogs token={token}"))
        .map_err(|e| format!("discogs request: {e}"))?;
    let response = blocking_http::request_no_redirect(req).map_err(|e| format!("discogs: {e}"))?;
    let body = String::from_utf8_lossy(&response.body);
    let api_message = makepad_strict_json::parse_depth(body.as_bytes(), 8)
        .ok()
        .and_then(|v| match v.get("message") {
            Some(Value::Str(s)) => Some(s.trim().to_string()),
            _ => None,
        });
    match response.status {
        200 => {}
        401 | 403 => {
            return Err(api_message.unwrap_or_else(|| {
                "Discogs refused auth or User-Agent (check the token)".into()
            }));
        }
        404 => {
            let msg = api_message.unwrap_or_else(|| "Discogs HTTP 404".into());
            return Err(msg);
        }
        429 => return Err("Discogs rate limit — wait a minute".into()),
        status => {
            return Err(api_message.unwrap_or_else(|| format!("Discogs HTTP {status}")));
        }
    }
    makepad_strict_json::parse_depth(body.as_bytes(), 16)
        .map_err(|e| format!("discogs json: {e}"))
}

fn nested_i64(value: &Value, path: &[&str]) -> Option<i64> {
    let mut cur = value;
    for key in path {
        cur = cur.get(key)?;
    }
    match cur {
        Value::Int(n) => Some(*n),
        Value::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn json_id(value: Option<&Value>) -> Option<i64> {
    match value? {
        Value::Int(n) if *n > 0 => Some(*n),
        Value::Str(s) => s.trim().parse().ok().filter(|n: &i64| *n > 0),
        _ => None,
    }
}

fn json_f64(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::F64(n) if n.is_finite() && *n > 0.0 => Some(*n),
        Value::Int(n) if *n > 0 => Some(*n as f64),
        Value::Str(s) => s.trim().parse().ok().filter(|n: &f64| n.is_finite() && *n > 0.0),
        _ => None,
    }
}

fn json_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::Str(s)) => s.trim().to_string(),
        Some(Value::Int(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn json_join(value: Option<&Value>) -> String {
    match value {
        Some(Value::Arr(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::Str(s) => {
                    let t = s.trim();
                    (!t.is_empty()).then_some(t.to_string())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(", "),
        Some(Value::Str(s)) => s.trim().to_string(),
        _ => String::new(),
    }
}

fn median_f64(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|n| n.is_finite() && *n > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_attempt_comes_first() {
        let query = SearchQuery {
            artist: "Jeff Mills".into(),
            title: "The Bells".into(),
            label: "Purpose Maker".into(),
            catalogue_number: "PM-005".into(),
            notes: String::new(),
        };
        let kinds: Vec<_> = search_attempts(&query).into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            kinds,
            ["catalogue", "artist+title+label", "artist+title"]
        );
    }

    #[test]
    fn empty_row_has_no_attempts() {
        assert!(search_attempts(&SearchQuery::default()).is_empty());
    }

    #[test]
    fn token_file_skips_comments() {
        let text = "# comment\n\n  tok_abc  \n";
        assert_eq!(first_token_line(text).as_deref(), Some("tok_abc"));
    }

    #[test]
    fn percent_encodes_spaces() {
        assert_eq!(percent_encode("Purpose Maker"), "Purpose%20Maker");
    }

    #[test]
    fn heckman_near_heckmann_ranks_title_hit() {
        let query = SearchQuery {
            artist: "Thomas P. Heckman".into(),
            title: "Titty Twisters Two".into(),
            ..Default::default()
        };
        let pool = vec![
            MasterCandidate {
                id: 1,
                title: "Emmanuel Top - Turkish Bazar 2000 (Thomas P. Heckmann Remixes)".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 126757,
                is_release: true,
                title: "Thomas P. Heckmann - Titty Twisters Two".into(),
                catno: "ef26-12".into(),
                ..Default::default()
            },
        ];
        let ranked = rank_masters(&query, pool);
        assert_eq!(ranked[0].id, 126757);
        assert!(fuzzy_pair_score(
            &normalize_alpha("Thomas P. Heckman"),
            &normalize_alpha("Thomas P. Heckmann")
        ) >= 70);
    }

    #[test]
    fn release_without_master_becomes_candidate() {
        // Adam X / tiva005 style: type=release, master_id=0.
        let row = makepad_strict_json::parse(
            br#"{"id":21260,"type":"release","master_id":0,"title":"Adam X - 100% Brooklyn EP","catno":"tiva005"}"#,
        )
        .unwrap();
        assert_eq!(row_candidate_id(&row), Some((21260, true)));

        // Release with a master still keeps the pressing id (format-specific pricing).
        let with_master = makepad_strict_json::parse(
            br#"{"id":1,"type":"release","master_id":99,"title":"Has Master"}"#,
        )
        .unwrap();
        assert_eq!(row_candidate_id(&with_master), Some((1, true)));
        let c = master_from_search_row(&with_master, 1, true);
        assert_eq!(c.release_id, 1);
        assert_eq!(c.master_id, 99);

        let master = makepad_strict_json::parse(br#"{"id":42,"type":"master","title":"M"}"#).unwrap();
        assert_eq!(row_candidate_id(&master), Some((42, false)));
    }

    #[test]
    fn vinyl_format_detects_12_and_rejects_cd() {
        assert!(is_vinyl_format("vinyl"));
        assert!(is_vinyl_format("12\""));
        assert!(is_vinyl_format("vinyl, lp"));
        assert!(!is_vinyl_format("cd"));
        assert!(!is_vinyl_format("file"));
        assert!(!is_vinyl_format("cassette"));
    }

    #[test]
    fn sheet_one_prefers_standard_lp_over_picture_disc() {
        let query = SearchQuery {
            artist: "Plastikman".into(),
            title: "Sheet One".into(),
            ..Default::default()
        };
        let pool = vec![
            MasterCandidate {
                id: 346183,
                is_release: true,
                release_id: 346183,
                title: "Plastikman - Sheet One".into(),
                year: "1993".into(),
                country: "UK".into(),
                format: "Vinyl, LP, Album, Limited Edition, Picture Disc".into(),
                catno: "L NOMU 22LP".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 29050348,
                is_release: true,
                release_id: 29050348,
                title: "Plastikman - Sheet One".into(),
                year: "2023".into(),
                country: "UK".into(),
                format: "Vinyl, LP, Album, Bioplastic, Reissue".into(),
                catno: "NOMU22LP".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 23599,
                is_release: true,
                release_id: 23599,
                title: "Plastikman - Sheet One".into(),
                year: "1993".into(),
                country: "UK".into(),
                format: "Vinyl, LP, Album".into(),
                catno: "NoMu 22 LP".into(),
                label: "NovaMute".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 3550505,
                is_release: true,
                release_id: 3550505,
                title: "Plastikman - Sheet One".into(),
                year: "1993".into(),
                country: "UK".into(),
                format: "Vinyl, LP, Album, White Label".into(),
                catno: "NoMu 22 LP".into(),
                ..Default::default()
            },
        ];
        let ranked = rank_masters(&query, pool);
        assert_eq!(ranked[0].id, 23599, "standard LP should rank first");
        let s = ranked[0].summary(0);
        assert!(s.contains("Sheet One"));
        assert!(s.contains("VINYL"));
        assert!(!s.contains("P.R. Records"));
        assert!(
            edition_score("Vinyl, LP, Album")
                > edition_score("Vinyl, LP, Album, Limited Edition, Picture Disc")
        );
    }

    #[test]
    fn catno_ranks_release_only_hit() {
        let query = SearchQuery {
            catalogue_number: "tiva005".into(),
            ..Default::default()
        };
        let pool = vec![
            MasterCandidate {
                id: 1,
                title: "Unrelated - Song".into(),
                catno: "XYZ001".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 21260,
                is_release: true,
                title: "Adam X - 100% Brooklyn EP".into(),
                catno: "tiva005".into(),
                label: "Sativae Recordings".into(),
                ..Default::default()
            },
        ];
        let ranked = rank_masters(&query, pool);
        assert_eq!(ranked[0].id, 21260);
        assert!(ranked[0].is_release);
        assert!(ranked[0].url().contains("/release/21260"));
    }

    #[test]
    fn median_of_odds_and_evens() {
        assert_eq!(median_f64(&[10.0, 40.0, 50.0]), 40.0);
        assert_eq!(median_f64(&[10.0, 50.0]), 30.0);
    }

    #[test]
    fn high_value_gate() {
        let v = VersionCandidate {
            vg_price: 45.0,
            ..Default::default()
        };
        assert!(v.is_high_value());
        let low = VersionCandidate {
            vg_price: 12.0,
            ..Default::default()
        };
        assert!(!low.is_high_value());
    }

    #[test]
    fn normalize_strips_spaces_and_punct() {
        assert_eq!(normalize_alpha("Moody Man"), "moodyman");
        assert_eq!(normalize_alpha("Moodymann"), "moodymann");
        assert_eq!(normalize_alpha("KDJ-12"), "kdj12");
    }

    #[test]
    fn fuzzy_near_miss_moodyman() {
        assert!(fuzzy_pair_score("moodyman", "moodymann") >= 70);
        assert_eq!(fuzzy_pair_score("moodyman", "moodyman"), 100);
        assert_eq!(fuzzy_pair_score("", "moodymann"), 0);
        // True edit distance (not a substring): one letter swap / insert.
        assert!(fuzzy_pair_score("moodyman", "moodymen") >= 70);
    }

    #[test]
    fn ranks_split_title_artist_and_track() {
        let query = SearchQuery {
            artist: "Moody Man".into(),
            title: "Don't Be Misled".into(),
            label: "KDJ".into(),
            ..Default::default()
        };
        let pool = vec![
            MasterCandidate {
                id: 1,
                title: "Random Act - Other Song".into(),
                label: "Warp".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 2,
                title: "Moodymann - Don't Be Misled".into(),
                label: "KDJ".into(),
                ..Default::default()
            },
        ];
        let ranked = rank_masters(&query, pool);
        assert_eq!(ranked[0].id, 2);
    }

    #[test]
    fn moodyman_ranks_above_unrelated() {
        let query = SearchQuery {
            artist: "Moody Man".into(),
            ..Default::default()
        };
        let pool = vec![
            MasterCandidate {
                id: 1,
                title: "Random Artist - Unrelated Track".into(),
                ..Default::default()
            },
            MasterCandidate {
                id: 2,
                title: "Moodymann - Dem Young Lights".into(),
                ..Default::default()
            },
        ];
        let ranked = rank_masters(&query, pool);
        assert_eq!(ranked[0].id, 2);
    }

    #[test]
    fn empty_query_scores_nothing_useful() {
        let query = SearchQuery::default();
        let pool = vec![MasterCandidate {
            id: 9,
            title: "Moodymann - Something".into(),
            label: "KDJ".into(),
            ..Default::default()
        }];
        let ranked = rank_masters(&query, pool);
        // Empty spoken fields → all scores 0; order preserved, no preference signal.
        assert_eq!(ranked[0].id, 9);
        assert_eq!(fuzzy_pair_score(&normalize_alpha(""), &normalize_alpha("Moodymann")), 0);
    }
}
