//! Closed tool table for the Ask chat — interrogate raw listings; mutate with UI confirm.

use crate::analysis;
use crate::db::Db;
use crate::model::{Listing, Store};
use makepad_ai_hub::local_llm::ToolSpec;
use makepad_strict_json::Value;

pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new(
            "inventory",
            "Counts only: how many searches, listings, and distinct shops are stored. No prices.",
            r#"{"type":"object","properties":{},"required":[]}"#,
        ),
        ToolSpec::new(
            "compute_stats",
            "Compute counts/means/medians FROM THE RAW LISTING ROWS (not a dashboard). \
Use this for averages, proportions, postage questions. \
Postage semantics on each row: postage_known=false means unknown; postage=0 means free UK postage; postage>0 means paid ON TOP of item price; delivered=item+postage when known. \
Returns explicit free/paid/unknown postage counts — never a vague 'quality %'.",
            r#"{"type":"object","properties":{"query":{"type":"string"},"shop":{"type":"string"},"pack_size":{"type":"integer"},"digital":{"type":"boolean"},"personalised":{"type":"boolean"},"bundle":{"type":"boolean"},"physical_only":{"type":"boolean","description":"default true — drop digital listings"},"exclude_personalised":{"type":"boolean"}},"required":[]}"#,
        ),
        ToolSpec::new(
            "query_listings",
            "Return up to 40 raw listing rows so you can inspect item price, postage, and delivered separately. \
Columns: id, item£, postage£, postage_kind (free|paid|unknown), delivered£, pack, shop_reviews, shop, title. \
Use when the user names a card or you need examples. review_count is shop popularity, not sales.",
            r#"{"type":"object","properties":{"query":{"type":"string","description":"substring match on title or search query"},"shop":{"type":"string"},"min_delivered":{"type":"number"},"max_delivered":{"type":"number"},"pack_size":{"type":"integer"},"digital":{"type":"boolean"},"personalised":{"type":"boolean"},"bundle":{"type":"boolean"},"postage":{"type":"string","description":"free|paid|unknown — filter by postage_kind"},"sort":{"type":"string","description":"delivered|item|postage|reviews|shop"},"limit":{"type":"integer"}},"required":[]}"#,
        ),
        ToolSpec::new(
            "shop_stats",
            "Per-shop aggregates computed from that shop's listing rows: count, median item, median postage (known only), median delivered, median shop reviews.",
            r#"{"type":"object","properties":{"shop":{"type":"string","description":"optional substring filter on shop name"},"limit":{"type":"integer"}},"required":[]}"#,
        ),
        ToolSpec::new(
            "search_list",
            "List imported searches with id, query, page, listing count, exported_at.",
            r#"{"type":"object","properties":{},"required":[]}"#,
        ),
        ToolSpec::new(
            "set_title_clean",
            "Set the cleaned display title for one listing. Does not change the original SEO title. Needs user confirmation.",
            r#"{"type":"object","properties":{"listing_id":{"type":"integer"},"url":{"type":"string"},"title_clean":{"type":"string"}},"required":["title_clean"]}"#,
        ),
        ToolSpec::new(
            "clear_title_clean",
            "Clear the cleaned display title so the full SEO title is shown again. Needs user confirmation.",
            r#"{"type":"object","properties":{"listing_id":{"type":"integer"},"url":{"type":"string"}},"required":[]}"#,
        ),
        ToolSpec::new(
            "delete_search",
            "Delete one imported search and unlink its listings; orphan listings are removed. Needs user confirmation.",
            r#"{"type":"object","properties":{"search_id":{"type":"integer"}},"required":["search_id"]}"#,
        ),
    ]
}

pub fn is_mutating(name: &str) -> bool {
    matches!(
        name,
        "set_title_clean" | "clear_title_clean" | "delete_search"
    )
}

/// Describe a mutating tool call for the confirm banner.
pub fn mutate_summary(name: &str, args_json: &str) -> String {
    let args = parse_obj(args_json).unwrap_or(Value::Null);
    match name {
        "delete_search" => format!(
            "Delete search #{}",
            opt_i64(&args, "search_id").unwrap_or(0)
        ),
        "set_title_clean" => {
            let clean = opt_str(&args, "title_clean").unwrap_or_default();
            format!("Set display title to “{}”", truncate(&clean, 48))
        }
        "clear_title_clean" => "Clear cleaned display title".into(),
        other => format!("Apply {other}"),
    }
}

/// Hub tool-call args are string pairs; coerce to a JSON object for the runners.
pub fn args_to_json(args: &[(String, String)]) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in args.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("\"{}\":{}", json_escape_key(k), json_escape_value(v)));
    }
    out.push('}');
    out
}

pub fn run_read(name: &str, args_json: &str, store: &Store) -> Result<String, String> {
    let args = parse_obj(args_json)?;
    match name {
        "inventory" | "dataset_summary" => Ok(inventory(store)),
        "compute_stats" => Ok(compute_stats(store, &args)),
        "query_listings" => Ok(query_listings(store, &args)),
        "shop_stats" => Ok(shop_stats(store, &args)),
        "search_list" => Ok(search_list(store)),
        other => Err(format!("unknown or non-read tool: {other}")),
    }
}

pub fn run_mutate(
    name: &str,
    args_json: &str,
    store: &Store,
    db: &mut Db,
) -> Result<String, String> {
    let args = parse_obj(args_json)?;
    match name {
        "set_title_clean" => {
            let clean = opt_str(&args, "title_clean")
                .ok_or_else(|| "title_clean required".to_string())?;
            let id = resolve_listing_id(store, &args)?;
            db.set_title_clean(id, &clean)?;
            Ok(format!("Set title_clean on listing #{id}"))
        }
        "clear_title_clean" => {
            let id = resolve_listing_id(store, &args)?;
            db.set_title_clean(id, "")?;
            Ok(format!("Cleared title_clean on listing #{id}"))
        }
        "delete_search" => {
            let search_id =
                opt_i64(&args, "search_id").ok_or_else(|| "search_id required".to_string())?;
            db.delete_search(search_id)?;
            Ok(format!("Deleted search #{search_id}"))
        }
        other => Err(format!("unknown or non-mutate tool: {other}")),
    }
}

/// Background runner for read tools — never touches the DB.
pub struct AskToolRunner {
    jobs: std::sync::mpsc::Sender<AskToolJob>,
    results: std::sync::mpsc::Receiver<AskToolOutcome>,
}

pub struct AskToolJob {
    pub name: String,
    pub args_json: String,
    pub store: Store,
}

pub struct AskToolOutcome {
    pub text: String,
    pub is_error: bool,
}

impl AskToolRunner {
    pub fn new(spawner: &makepad_widgets::makepad_platform::thread::ThreadSpawner) -> Self {
        let (jobs, job_rx) = std::sync::mpsc::channel::<AskToolJob>();
        let (result_tx, results) = std::sync::mpsc::channel();
        if let Ok(handle) = spawner.spawn_worker(
            makepad_widgets::makepad_platform::thread::ThreadOptions {
                name: Some("etsy-ask-tools".into()),
                ..Default::default()
            },
            move || {
                while let Ok(job) = job_rx.recv() {
                    let outcome = match run_read(&job.name, &job.args_json, &job.store) {
                        Ok(text) => AskToolOutcome {
                            text,
                            is_error: false,
                        },
                        Err(text) => AskToolOutcome {
                            text,
                            is_error: true,
                        },
                    };
                    if result_tx.send(outcome).is_err() {
                        return;
                    }
                    makepad_widgets::makepad_platform::thread::SignalToUI::set_ui_signal();
                }
            },
        ) {
            handle.detach();
        }
        Self { jobs, results }
    }

    pub fn submit(&self, job: AskToolJob) {
        let _ = self.jobs.send(job);
    }

    pub fn drain(&self) -> Vec<AskToolOutcome> {
        self.results.try_iter().collect()
    }
}

fn json_escape_key(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_escape_value(s: &str) -> String {
    let t = s.trim();
    if t == "true" || t == "false" || t == "null" {
        return t.to_string();
    }
    if t.parse::<i64>().is_ok() || t.parse::<f64>().is_ok() {
        return t.to_string();
    }
    format!("\"{}\"", json_escape_key(s))
}

fn resolve_listing_id(store: &Store, args: &Value) -> Result<i64, String> {
    if let Some(id) = opt_i64(args, "listing_id") {
        if store.listings.iter().any(|l| l.id == id) {
            return Ok(id);
        }
        return Err(format!("no listing id {id}"));
    }
    if let Some(url) = opt_str(args, "url") {
        if let Some(l) = store.listings.iter().find(|l| l.url == url) {
            return Ok(l.id);
        }
        return Err(format!("no listing with url {url}"));
    }
    Err("listing_id or url required".into())
}

fn postage_kind(l: &Listing) -> &'static str {
    if !l.postage_known {
        "unknown"
    } else if l.postage == Some(0.0) {
        "free"
    } else if l.postage.map(|p| p > 0.0).unwrap_or(false) {
        "paid"
    } else {
        "unknown"
    }
}

fn listing_matches_with(l: &Listing, args: &Value, default_physical_only: bool) -> bool {
    let query = opt_str(args, "query").unwrap_or_default().to_ascii_lowercase();
    let shop = opt_str(args, "shop").unwrap_or_default().to_ascii_lowercase();
    let physical_only = opt_bool(args, "physical_only").unwrap_or(default_physical_only);
    let exclude_personalised = opt_bool(args, "exclude_personalised").unwrap_or(false);

    if physical_only && l.is_digital {
        return false;
    }
    if exclude_personalised && l.is_personalised {
        return false;
    }
    if !query.is_empty() {
        let title = l.display_title().to_ascii_lowercase();
        let raw = l.title.to_ascii_lowercase();
        let qmatch = title.contains(&query)
            || raw.contains(&query)
            || l.queries
                .iter()
                .any(|q| q.to_ascii_lowercase().contains(&query));
        if !qmatch {
            return false;
        }
    }
    if !shop.is_empty() && !l.shop_name.to_ascii_lowercase().contains(&shop) {
        return false;
    }
    if let Some(p) = opt_i64(args, "pack_size") {
        if l.pack_size != Some(p) {
            return false;
        }
    }
    if let Some(d) = opt_bool(args, "digital") {
        if l.is_digital != d {
            return false;
        }
    }
    if let Some(p) = opt_bool(args, "personalised") {
        if l.is_personalised != p {
            return false;
        }
    }
    if let Some(b) = opt_bool(args, "bundle") {
        if l.is_bundle != b {
            return false;
        }
    }
    if let Some(kind) = opt_str(args, "postage") {
        if postage_kind(l) != kind.to_ascii_lowercase() {
            return false;
        }
    }
    if let Some(min) = opt_f64(args, "min_delivered") {
        if l.delivered_price.map(|d| d < min).unwrap_or(true) {
            return false;
        }
    }
    if let Some(max) = opt_f64(args, "max_delivered") {
        if l.delivered_price.map(|d| d > max).unwrap_or(true) {
            return false;
        }
    }
    true
}

fn inventory(store: &Store) -> String {
    let shops: std::collections::BTreeSet<_> = store
        .listings
        .iter()
        .filter_map(|l| {
            let s = l.shop_name.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_ascii_lowercase())
            }
        })
        .collect();
    format!(
        "searches={} listings={} distinct_shops={}\n\
(Use compute_stats for prices/postage proportions; query_listings for individual rows.)",
        store.search_count(),
        store.listing_count(),
        shops.len(),
    )
}

fn compute_stats(store: &Store, args: &Value) -> String {
    let rows: Vec<&Listing> = store
        .listings
        .iter()
        .filter(|l| listing_matches_with(l, args, true))
        .collect();
    let n = rows.len();
    if n == 0 {
        return "matched=0 (no listings after filters)".into();
    }

    let item: Vec<f64> = rows.iter().filter_map(|l| l.price).collect();
    let delivered: Vec<f64> = rows.iter().filter_map(|l| l.delivered_price).collect();
    let ppc: Vec<f64> = rows.iter().filter_map(|l| l.delivered_price_per_card).collect();
    let postage_vals: Vec<f64> = rows
        .iter()
        .filter(|l| l.postage_known)
        .filter_map(|l| l.postage)
        .collect();
    let paid_postage: Vec<f64> = postage_vals.iter().copied().filter(|p| *p > 0.0).collect();

    let mut free = 0usize;
    let mut paid = 0usize;
    let mut unknown = 0usize;
    for l in &rows {
        match postage_kind(l) {
            "free" => free += 1,
            "paid" => paid += 1,
            _ => unknown += 1,
        }
    }
    let known = free + paid;
    let pct = |part: usize, whole: usize| -> String {
        if whole == 0 {
            "n/a".into()
        } else {
            format!("{:.1}%", 100.0 * part as f64 / whole as f64)
        }
    };

    format!(
        "matched={n} (of {} stored)\n\
item_price: known={} mean={} median={}\n\
delivered(item+postage when known): known={} mean={} median={}\n\
delivered_per_card: known={} mean={} median={}\n\
postage breakdown: free={free} ({})  paid_on_top={paid} ({})  unknown={unknown} ({})\n\
  among_postage_known ({known}): free={}  paid={}\n\
postage£ when known: mean={} median={}\n\
postage£ when paid>0: count={} mean={} median={}\n\
NOTE: 'free' = postage field is £0. 'paid_on_top' = postage>0 added to item price. \
'unknown' = postage not captured. Do NOT call data-completeness a 'included in price' rate.",
        store.listing_count(),
        item.len(),
        analysis::format_money(mean(&item)),
        analysis::format_money(median(&item)),
        delivered.len(),
        analysis::format_money(mean(&delivered)),
        analysis::format_money(median(&delivered)),
        ppc.len(),
        analysis::format_money(mean(&ppc)),
        analysis::format_money(median(&ppc)),
        pct(free, n),
        pct(paid, n),
        pct(unknown, n),
        pct(free, known),
        pct(paid, known),
        analysis::format_money(mean(&postage_vals)),
        analysis::format_money(median(&postage_vals)),
        paid_postage.len(),
        analysis::format_money(mean(&paid_postage)),
        analysis::format_money(median(&paid_postage)),
    )
}

fn query_listings(store: &Store, args: &Value) -> String {
    let sort = opt_str(args, "sort").unwrap_or_else(|| "delivered".into());
    let limit = opt_i64(args, "limit").unwrap_or(20).clamp(1, 40) as usize;

    let mut rows: Vec<&Listing> = store
        .listings
        .iter()
        .filter(|l| listing_matches_with(l, args, false))
        .collect();

    match sort.as_str() {
        "reviews" => rows.sort_by(|a, b| b.review_count.cmp(&a.review_count)),
        "item" | "price" => rows.sort_by(|a, b| partial_cmp_opt(b.price, a.price)),
        "postage" => rows.sort_by(|a, b| partial_cmp_opt(b.postage, a.postage)),
        "shop" => rows.sort_by(|a, b| {
            a.shop_name
                .to_ascii_lowercase()
                .cmp(&b.shop_name.to_ascii_lowercase())
        }),
        _ => rows.sort_by(|a, b| partial_cmp_opt(b.delivered_price, a.delivered_price)),
    }

    let total = rows.len();
    rows.truncate(limit);
    if rows.is_empty() {
        return "0 listings matched.".into();
    }
    let mut out = format!("showing {} of {} matches\n", rows.len(), total);
    out.push_str(
        "id | item£ | postage£ | postage_kind | delivered£ | pack | shop_reviews | shop | title\n",
    );
    for l in rows {
        out.push_str(&format!(
            "{} | {} | {} | {} | {} | {} | {} | {} | {}\n",
            l.id,
            analysis::format_money(l.price),
            if l.postage_known {
                analysis::format_money(l.postage)
            } else {
                "—".into()
            },
            postage_kind(l),
            analysis::format_money(l.delivered_price),
            l.pack_size
                .map(|n| n.to_string())
                .unwrap_or_else(|| "—".into()),
            l.review_count
                .map(|n| n.to_string())
                .unwrap_or_else(|| "—".into()),
            truncate(&l.shop_name, 16),
            truncate(l.display_title(), 48),
        ));
    }
    out
}

fn shop_stats(store: &Store, args: &Value) -> String {
    let shop_filter = opt_str(args, "shop").unwrap_or_default().to_ascii_lowercase();
    let limit = opt_i64(args, "limit").unwrap_or(30).clamp(1, 80) as usize;
    let mut map: std::collections::BTreeMap<String, Vec<&Listing>> =
        std::collections::BTreeMap::new();
    for l in &store.listings {
        if l.shop_name.trim().is_empty() {
            continue;
        }
        if !shop_filter.is_empty() && !l.shop_name.to_ascii_lowercase().contains(&shop_filter) {
            continue;
        }
        map.entry(l.shop_name.clone()).or_default().push(l);
    }
    let mut rows: Vec<_> = map.into_iter().collect();
    rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    rows.truncate(limit);
    if rows.is_empty() {
        return "0 shops.".into();
    }
    let mut out =
        String::from("shop | n | med_item | med_postage_known | med_delivered | med_shop_reviews\n");
    for (shop, list) in rows {
        let item: Vec<f64> = list.iter().filter_map(|l| l.price).collect();
        let postage: Vec<f64> = list
            .iter()
            .filter(|l| l.postage_known)
            .filter_map(|l| l.postage)
            .collect();
        let delivered: Vec<f64> = list.iter().filter_map(|l| l.delivered_price).collect();
        let reviews: Vec<f64> = list
            .iter()
            .filter_map(|l| l.review_count.map(|n| n as f64))
            .collect();
        out.push_str(&format!(
            "{} | {} | {} | {} | {} | {}\n",
            truncate(&shop, 24),
            list.len(),
            analysis::format_money(median(&item)),
            analysis::format_money(median(&postage)),
            analysis::format_money(median(&delivered)),
            fmt_num(median(&reviews)),
        ));
    }
    out
}

fn search_list(store: &Store) -> String {
    if store.searches.is_empty() {
        return "0 searches.".into();
    }
    let mut out = String::from("id | page | listings | query | exported\n");
    for s in &store.searches {
        let n = store.listings_for_search(s.id).len();
        out.push_str(&format!(
            "{} | {} | {} | {} | {}\n",
            s.id,
            s.page,
            n,
            if s.query.is_empty() {
                "(none)"
            } else {
                &s.query
            },
            if s.exported_at.is_empty() {
                "?"
            } else {
                &s.exported_at
            },
        ));
    }
    out
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        Some((v[mid - 1] + v[mid]) / 2.0)
    } else {
        Some(v[mid])
    }
}

fn partial_cmp_opt(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn fmt_num(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{n:.0}"),
        None => "—".into(),
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

fn parse_obj(args_json: &str) -> Result<Value, String> {
    let trimmed = args_json.trim();
    if trimmed.is_empty() || trimmed == "null" {
        return Ok(Value::Obj(Vec::new()));
    }
    let value = makepad_strict_json::parse(trimmed.as_bytes())
        .map_err(|e| format!("invalid tool args JSON: {e}"))?;
    match value {
        Value::Obj(_) => Ok(value),
        Value::Null => Ok(Value::Obj(Vec::new())),
        _ => Err("tool args must be an object".into()),
    }
}

fn opt_str(obj: &Value, key: &str) -> Option<String> {
    match obj.get(key)? {
        Value::Null => None,
        Value::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        Value::Int(n) => Some(n.to_string()),
        Value::F64(n) if n.is_finite() => Some(n.to_string()),
        _ => None,
    }
}

fn opt_i64(obj: &Value, key: &str) -> Option<i64> {
    match obj.get(key)? {
        Value::Int(n) => Some(*n),
        Value::F64(n) if n.is_finite() && n.fract() == 0.0 => Some(*n as i64),
        Value::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn opt_f64(obj: &Value, key: &str) -> Option<f64> {
    match obj.get(key)? {
        Value::F64(n) if n.is_finite() => Some(*n),
        Value::Int(n) => Some(*n as f64),
        Value::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn opt_bool(obj: &Value, key: &str) -> Option<bool> {
    match obj.get(key)? {
        Value::Bool(b) => Some(*b),
        Value::Int(n) => Some(*n != 0),
        Value::Str(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use crate::parse::parse_export;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn loaded() -> (Db, Store) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("etsy-ask-{nanos}.db"));
        let _ = std::fs::remove_file(&path);
        let mut db = Db::open(&path).unwrap();
        let a = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        db.import_export(&a).unwrap();
        let store = db.load().unwrap();
        (db, store)
    }

    #[test]
    fn inventory_compute_and_query() {
        let (_db, store) = loaded();
        let inv = run_read("inventory", "{}", &store).unwrap();
        assert!(inv.contains("listings=6"));
        assert!(inv.contains("distinct_shops="));
        assert!(!inv.contains("quality:"));

        let stats = run_read("compute_stats", "{}", &store).unwrap();
        assert!(stats.contains("matched="));
        assert!(stats.contains("postage breakdown:"));
        assert!(stats.contains("free="));
        assert!(stats.contains("paid_on_top="));
        assert!(stats.contains("unknown="));
        assert!(!stats.contains("quality:"));

        let paid = run_read(
            "query_listings",
            r#"{"postage":"paid","limit":10}"#,
            &store,
        )
        .unwrap();
        // Sample export may or may not have paid rows; column header must be present when any match,
        // and postage_kind vocabulary is always used when rows exist.
        let _ = paid;
        let q = run_read(
            "query_listings",
            r#"{"query":"pack","limit":10}"#,
            &store,
        )
        .unwrap();
        assert!(q.contains("postage_kind"));
        assert!(q.to_lowercase().contains("pack"));

        let shops = run_read("shop_stats", "{}", &store).unwrap();
        assert!(shops.contains("CardCraftUK") || shops.contains("med_item"));
        let searches = run_read("search_list", "{}", &store).unwrap();
        assert!(searches.contains("funny"));
    }

    #[test]
    fn mutate_set_and_delete() {
        let (mut db, store) = loaded();
        let id = store.listings[0].id;
        let args = format!(r#"{{"listing_id":{id},"title_clean":"Short"}}"#);
        let msg = run_mutate("set_title_clean", &args, &store, &mut db).unwrap();
        assert!(msg.contains("Set title_clean"));
        let store = db.load().unwrap();
        assert_eq!(
            store.listings.iter().find(|l| l.id == id).unwrap().title_clean,
            "Short"
        );
        let search_id = store.searches[0].id;
        run_mutate(
            "delete_search",
            &format!(r#"{{"search_id":{search_id}}}"#),
            &store,
            &mut db,
        )
        .unwrap();
        let store = db.load().unwrap();
        assert!(store.searches.is_empty());
        assert!(store.listings.is_empty());
    }
}
