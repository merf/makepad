//! SQLite file at `local/etsy/etsy.db`. Load into memory; write through.

use crate::model::{ExportPayload, Listing, ParsedListing, Search, Store};
use crate::normalise;
use makepad_sqlite::{Connection, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS searches(
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'etsy',
    query TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    page INTEGER NOT NULL DEFAULT 1,
    exported_at TEXT NOT NULL DEFAULT '',
    imported_at INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS listings(
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    title_clean TEXT NOT NULL DEFAULT '',
    shop_name TEXT NOT NULL DEFAULT '',
    shop_url TEXT NOT NULL DEFAULT '',
    price REAL,
    price_text TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT '',
    postage_text TEXT NOT NULL DEFAULT '',
    postage REAL,
    postage_known INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER,
    image_url TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL DEFAULT '',
    first_seen INTEGER NOT NULL DEFAULT 0,
    last_seen INTEGER NOT NULL DEFAULT 0,
    delivered_price REAL,
    pack_size INTEGER,
    price_per_card REAL,
    delivered_price_per_card REAL,
    is_digital INTEGER NOT NULL DEFAULT 0,
    is_personalised INTEGER NOT NULL DEFAULT 0,
    is_bundle INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS listing_searches(
    listing_id INTEGER NOT NULL,
    search_id INTEGER NOT NULL,
    PRIMARY KEY(listing_id, search_id)
);
";

const MIGRATIONS: &[(&str, &str)] = &[
    ("delivered_price", "ALTER TABLE listings ADD COLUMN delivered_price REAL"),
    ("pack_size", "ALTER TABLE listings ADD COLUMN pack_size INTEGER"),
    ("price_per_card", "ALTER TABLE listings ADD COLUMN price_per_card REAL"),
    (
        "delivered_price_per_card",
        "ALTER TABLE listings ADD COLUMN delivered_price_per_card REAL",
    ),
    ("is_digital", "ALTER TABLE listings ADD COLUMN is_digital INTEGER NOT NULL DEFAULT 0"),
    (
        "is_personalised",
        "ALTER TABLE listings ADD COLUMN is_personalised INTEGER NOT NULL DEFAULT 0",
    ),
    ("is_bundle", "ALTER TABLE listings ADD COLUMN is_bundle INTEGER NOT NULL DEFAULT 0"),
    ("page", "ALTER TABLE searches ADD COLUMN page INTEGER NOT NULL DEFAULT 1"),
    ("shop_url", "ALTER TABLE listings ADD COLUMN shop_url TEXT NOT NULL DEFAULT ''"),
    ("title_clean", "ALTER TABLE listings ADD COLUMN title_clean TEXT NOT NULL DEFAULT ''"),
];

pub struct Db {
    conn: Connection,
}

#[derive(Clone, Debug, Default)]
pub struct ImportResult {
    #[allow(dead_code)]
    pub search_id: i64,
    pub listings_in_export: usize,
    pub listings_inserted: usize,
    pub listings_updated: usize,
    pub query: String,
    pub page: i64,
    pub exported_at: String,
}

impl Db {
    pub fn open(path: &Path) -> Result<Db, String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let mut conn = Connection::open(path, std::time::Duration::from_secs(5))
            .map_err(|e| format!("open {}: {e:?}", path.display()))?;
        conn.limits_mut().max_rows = 500_000;
        conn.limits_mut().max_steps = 50_000_000;
        conn.execute_batch(SCHEMA)
            .map_err(|e| format!("schema: {e:?}"))?;
        migrate(&mut conn)?;
        if listings_schema_is_corrupt(&mut conn) {
            drop(conn);
            repair_corrupt_db(path)?;
            let mut conn = Connection::open(path, std::time::Duration::from_secs(5))
                .map_err(|e| format!("reopen {}: {e:?}", path.display()))?;
            conn.limits_mut().max_rows = 500_000;
            conn.limits_mut().max_steps = 50_000_000;
            return Ok(Db { conn });
        }
        Ok(Db { conn })
    }

    pub fn load(&mut self) -> Result<Store, String> {
        let search_rows = self
            .conn
            .query(
                "SELECT id, source, query, source_url, page, exported_at, imported_at, raw_json \
                 FROM searches ORDER BY id",
                &[],
            )
            .map_err(|e| format!("load searches: {e:?}"))?;
        let searches: Vec<Search> = search_rows
            .rows
            .iter()
            .map(|row| Search {
                id: int(&row[0]),
                source: text(&row[1]),
                query: text(&row[2]),
                source_url: text(&row[3]),
                page: {
                    let p = int(&row[4]);
                    if p > 0 {
                        p
                    } else {
                        1
                    }
                },
                exported_at: text(&row[5]),
                imported_at: int(&row[6]),
                raw_json: text(&row[7]),
            })
            .collect();

        let listing_rows = self
            .conn
            .query(
                "SELECT id, url, title, title_clean, shop_name, shop_url, price, price_text, \
                 currency, postage_text, postage, postage_known, review_count, image_url, \
                 raw_text, first_seen, last_seen, delivered_price, pack_size, price_per_card, \
                 delivered_price_per_card, is_digital, is_personalised, is_bundle \
                 FROM listings ORDER BY id",
                &[],
            )
            .map_err(|e| format!("load listings: {e:?}"))?;

        let mut listings: Vec<Listing> = listing_rows
            .rows
            .iter()
            .map(|row| {
                let mut listing = Listing {
                    id: int(&row[0]),
                    url: text(&row[1]),
                    title: text(&row[2]),
                    title_clean: text(&row[3]),
                    shop_name: text(&row[4]),
                    shop_url: text(&row[5]),
                    price: opt_real(&row[6]),
                    price_text: text(&row[7]),
                    currency: text(&row[8]),
                    postage_text: text(&row[9]),
                    postage: opt_real(&row[10]),
                    postage_known: int(&row[11]) != 0,
                    review_count: opt_int(&row[12]),
                    image_url: text(&row[13]),
                    raw_text: text(&row[14]),
                    first_seen: int(&row[15]),
                    last_seen: int(&row[16]),
                    delivered_price: opt_real(&row[17]),
                    pack_size: opt_int(&row[18]),
                    price_per_card: opt_real(&row[19]),
                    delivered_price_per_card: opt_real(&row[20]),
                    is_digital: int(&row[21]) != 0,
                    is_personalised: int(&row[22]) != 0,
                    is_bundle: int(&row[23]) != 0,
                    search_ids: Vec::new(),
                    queries: Vec::new(),
                };
                normalise::apply(&mut listing);
                listing
            })
            .collect();

        let link_rows = self
            .conn
            .query(
                "SELECT listing_id, search_id FROM listing_searches ORDER BY listing_id, search_id",
                &[],
            )
            .map_err(|e| format!("load listing_searches: {e:?}"))?;

        let query_by_search: HashMap<i64, String> =
            searches.iter().map(|s| (s.id, s.query.clone())).collect();
        let mut index_by_id: HashMap<i64, usize> = HashMap::new();
        for (i, listing) in listings.iter().enumerate() {
            index_by_id.insert(listing.id, i);
        }
        for row in &link_rows.rows {
            let listing_id = int(&row[0]);
            let search_id = int(&row[1]);
            if let Some(&idx) = index_by_id.get(&listing_id) {
                let listing = &mut listings[idx];
                if !listing.search_ids.contains(&search_id) {
                    listing.search_ids.push(search_id);
                    if let Some(q) = query_by_search.get(&search_id) {
                        if !q.is_empty() && !listing.queries.contains(q) {
                            listing.queries.push(q.clone());
                        }
                    }
                }
            }
        }

        Ok(Store { searches, listings })
    }

    /// Insert a new search and upsert listings by URL. Never wipes prior imports.
    pub fn import_export(&mut self, payload: &ExportPayload) -> Result<ImportResult, String> {
        let now = now_secs();
        self.conn
            .execute(
                "INSERT INTO searches(source, query, source_url, page, exported_at, imported_at, raw_json) \
                 VALUES(?, ?, ?, ?, ?, ?, ?)",
                &[
                    Value::text(payload.source.as_str()),
                    Value::text(payload.query.as_str()),
                    Value::text(payload.source_url.as_str()),
                    Value::Integer(if payload.page > 0 { payload.page } else { 1 }),
                    Value::text(payload.exported_at.as_str()),
                    Value::Integer(now),
                    Value::text(payload.raw_json.as_str()),
                ],
            )
            .map_err(|e| format!("insert search: {e:?}"))?;
        let search_id = self.last_id("searches")?;

        let mut inserted = 0usize;
        let mut updated = 0usize;
        for parsed in &payload.listings {
            let url = listing_key(parsed);
            if url.is_empty() {
                continue;
            }
            let mut listing = Listing::from_parsed(parsed, now);
            listing.url = url.clone();
            match self.find_listing_id(&url)? {
                Some(listing_id) => {
                    listing.id = listing_id;
                    // Preserve title_clean only when the SEO title is unchanged.
                    if let Some((clean, old_title)) = self.load_title_pair(listing_id)? {
                        if old_title == listing.title {
                            listing.title_clean = clean;
                        }
                    }
                    listing.first_seen = 0;
                    self.update_listing(&listing, now)?;
                    self.link_listing_search(listing_id, search_id)?;
                    updated += 1;
                }
                None => {
                    let listing_id = self.insert_listing(&listing)?;
                    self.link_listing_search(listing_id, search_id)?;
                    inserted += 1;
                }
            }
        }

        Ok(ImportResult {
            search_id,
            listings_in_export: payload.listings.len(),
            listings_inserted: inserted,
            listings_updated: updated,
            query: payload.query.clone(),
            page: payload.page,
            exported_at: payload.exported_at.clone(),
        })
    }

    pub fn delete_search(&mut self, search_id: i64) -> Result<usize, String> {
        self.conn
            .execute(
                "DELETE FROM listing_searches WHERE search_id=?",
                &[Value::Integer(search_id)],
            )
            .map_err(|e| format!("unlink search: {e:?}"))?;
        self.conn
            .execute(
                "DELETE FROM searches WHERE id=?",
                &[Value::Integer(search_id)],
            )
            .map_err(|e| format!("delete search: {e:?}"))?;
        self.conn
            .execute(
                "DELETE FROM listings WHERE id NOT IN (SELECT listing_id FROM listing_searches)",
                &[],
            )
            .map_err(|e| format!("orphan cleanup: {e:?}"))?;
        Ok(search_id as usize)
    }

    pub fn set_title_clean(&mut self, listing_id: i64, title_clean: &str) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE listings SET title_clean=? WHERE id=?",
                &[Value::text(title_clean), Value::Integer(listing_id)],
            )
            .map_err(|e| format!("set title_clean: {e:?}"))?;
        Ok(())
    }

    fn load_title_pair(&mut self, listing_id: i64) -> Result<Option<(String, String)>, String> {
        let rows = self
            .conn
            .query(
                "SELECT title_clean, title FROM listings WHERE id=? LIMIT 1",
                &[Value::Integer(listing_id)],
            )
            .map_err(|e| format!("load title_clean: {e:?}"))?;
        Ok(rows
            .rows
            .first()
            .map(|row| (text(&row[0]), text(&row[1]))))
    }

    fn find_listing_id(&mut self, url: &str) -> Result<Option<i64>, String> {
        let rows = self
            .conn
            .query(
                "SELECT id FROM listings WHERE url=? LIMIT 1",
                &[Value::text(url)],
            )
            .map_err(|e| format!("find listing: {e:?}"))?;
        Ok(rows.rows.first().map(|row| int(&row[0])))
    }

    fn insert_listing(&mut self, listing: &Listing) -> Result<i64, String> {
        self.conn
            .execute(
                "INSERT INTO listings(url, title, title_clean, shop_name, shop_url, price, \
                 price_text, currency, postage_text, postage, postage_known, review_count, \
                 image_url, raw_text, first_seen, last_seen, delivered_price, pack_size, \
                 price_per_card, delivered_price_per_card, is_digital, is_personalised, is_bundle) \
                 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &listing_insert_values(listing),
            )
            .map_err(|e| format!("insert listing: {e:?}"))?;
        self.last_id("listings")
    }

    fn update_listing(&mut self, listing: &Listing, now: i64) -> Result<(), String> {
        let mut params = listing_update_values(listing, now);
        params.push(Value::Integer(listing.id));
        self.conn
            .execute(
                "UPDATE listings SET title=?, title_clean=?, shop_name=?, shop_url=?, price=?, \
                 price_text=?, currency=?, postage_text=?, postage=?, postage_known=?, \
                 review_count=?, image_url=?, raw_text=?, last_seen=?, delivered_price=?, \
                 pack_size=?, price_per_card=?, delivered_price_per_card=?, is_digital=?, \
                 is_personalised=?, is_bundle=? WHERE id=?",
                &params,
            )
            .map_err(|e| format!("update listing: {e:?}"))?;
        Ok(())
    }

    fn link_listing_search(&mut self, listing_id: i64, search_id: i64) -> Result<(), String> {
        let _ = self.conn.execute(
            "INSERT INTO listing_searches(listing_id, search_id) VALUES(?, ?)",
            &[Value::Integer(listing_id), Value::Integer(search_id)],
        );
        Ok(())
    }

    fn last_id(&mut self, table: &str) -> Result<i64, String> {
        let sql = format!("SELECT MAX(id) FROM {table}");
        let rows = self
            .conn
            .query(&sql, &[])
            .map_err(|e| format!("last id {table}: {e:?}"))?;
        Ok(rows.scalar().and_then(|v| v.as_integer()).unwrap_or(0))
    }
}

fn column_names(conn: &mut Connection, table: &str) -> Result<Vec<String>, String> {
    let rows = conn
        .query(&format!("PRAGMA table_info({table})"), &[])
        .map_err(|e| format!("table_info {table}: {e:?}"))?;
    Ok(rows
        .rows
        .iter()
        .filter_map(|row| row.get(1).map(text))
        .collect())
}

fn migrate(conn: &mut Connection) -> Result<(), String> {
    let listing_cols = column_names(conn, "listings").unwrap_or_default();
    let search_cols = column_names(conn, "searches").unwrap_or_default();
    for (col, sql) in MIGRATIONS {
        let exists = if *col == "page" {
            search_cols.iter().any(|c| c == col)
        } else {
            listing_cols.iter().any(|c| c == col)
        };
        if !exists {
            let _ = conn.execute(sql, &[]);
        }
    }
    Ok(())
}

fn listings_schema_is_corrupt(conn: &mut Connection) -> bool {
    let Ok(names) = column_names(conn, "listings") else {
        return false;
    };
    let mut seen = std::collections::HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return true;
        }
    }
    false
}

/// Rebuild a DB whose CREATE TABLE accumulated duplicate ALTER columns.
fn repair_corrupt_db(path: &Path) -> Result<(), String> {
    let bak = path.with_extension("db.corrupt.bak");
    let mut old = Connection::open(path, std::time::Duration::from_secs(5))
        .map_err(|e| format!("repair open: {e:?}"))?;
    old.limits_mut().max_rows = 500_000;
    old.limits_mut().max_steps = 50_000_000;
    // Read raw search JSON before replacing the file.
    let search_rows = old
        .query(
            "SELECT raw_json FROM searches ORDER BY id",
            &[],
        )
        .map_err(|e| format!("repair read searches: {e:?}"))?;
    let payloads: Vec<String> = search_rows.rows.iter().map(|row| text(&row[0])).collect();
    drop(old);

    let _ = std::fs::remove_file(&bak);
    std::fs::rename(path, &bak).map_err(|e| format!("backup corrupt db: {e}"))?;

    let mut fresh = Db::open(path)?;
    for raw in payloads {
        if raw.trim().is_empty() {
            continue;
        }
        match crate::parse::parse_export(&raw) {
            Ok(payload) => {
                let _ = fresh.import_export(&payload);
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

fn listing_key(parsed: &ParsedListing) -> String {
    let url = parsed.url.trim();
    if !url.is_empty() {
        return url.to_string();
    }
    format!("title:{}|shop:{}", parsed.title.trim(), parsed.shop_name.trim())
}

fn listing_insert_values(listing: &Listing) -> Vec<Value> {
    vec![
        Value::text(listing.url.as_str()),
        Value::text(listing.title.as_str()),
        Value::text(listing.title_clean.as_str()),
        Value::text(listing.shop_name.as_str()),
        Value::text(listing.shop_url.as_str()),
        opt_real_val(listing.price),
        Value::text(listing.price_text.as_str()),
        Value::text(listing.currency.as_str()),
        Value::text(listing.postage_text.as_str()),
        opt_real_val(listing.postage),
        Value::Integer(if listing.postage_known { 1 } else { 0 }),
        opt_int_val(listing.review_count),
        Value::text(listing.image_url.as_str()),
        Value::text(listing.raw_text.as_str()),
        Value::Integer(listing.first_seen),
        Value::Integer(listing.last_seen),
        opt_real_val(listing.delivered_price),
        opt_int_val(listing.pack_size),
        opt_real_val(listing.price_per_card),
        opt_real_val(listing.delivered_price_per_card),
        Value::Integer(if listing.is_digital { 1 } else { 0 }),
        Value::Integer(if listing.is_personalised { 1 } else { 0 }),
        Value::Integer(if listing.is_bundle { 1 } else { 0 }),
    ]
}

fn listing_update_values(listing: &Listing, now: i64) -> Vec<Value> {
    vec![
        Value::text(listing.title.as_str()),
        Value::text(listing.title_clean.as_str()),
        Value::text(listing.shop_name.as_str()),
        Value::text(listing.shop_url.as_str()),
        opt_real_val(listing.price),
        Value::text(listing.price_text.as_str()),
        Value::text(listing.currency.as_str()),
        Value::text(listing.postage_text.as_str()),
        opt_real_val(listing.postage),
        Value::Integer(if listing.postage_known { 1 } else { 0 }),
        opt_int_val(listing.review_count),
        Value::text(listing.image_url.as_str()),
        Value::text(listing.raw_text.as_str()),
        Value::Integer(now),
        opt_real_val(listing.delivered_price),
        opt_int_val(listing.pack_size),
        opt_real_val(listing.price_per_card),
        opt_real_val(listing.delivered_price_per_card),
        Value::Integer(if listing.is_digital { 1 } else { 0 }),
        Value::Integer(if listing.is_personalised { 1 } else { 0 }),
        Value::Integer(if listing.is_bundle { 1 } else { 0 }),
    ]
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn int(value: &Value) -> i64 {
    value.as_integer().unwrap_or(0)
}

fn opt_int(value: &Value) -> Option<i64> {
    if value.is_null() {
        None
    } else {
        value.as_integer()
    }
}

fn opt_real(value: &Value) -> Option<f64> {
    if value.is_null() {
        None
    } else {
        value.as_real().or_else(|| value.as_integer().map(|n| n as f64))
    }
}

fn text(value: &Value) -> String {
    value.as_text().unwrap_or("").to_string()
}

fn opt_real_val(v: Option<f64>) -> Value {
    match v {
        Some(n) if n.is_finite() => Value::Real(n),
        _ => Value::Null,
    }
}

fn opt_int_val(v: Option<i64>) -> Value {
    match v {
        Some(n) => Value::Integer(n),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_export;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db() -> Db {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("etsy-test-{nanos}.db"));
        let _ = std::fs::remove_file(&path);
        Db::open(&path).expect("open temp db")
    }

    #[test]
    fn import_and_dedupe_across_searches() {
        let mut db = temp_db();
        let a = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        let r1 = db.import_export(&a).unwrap();
        assert_eq!(r1.listings_inserted, 6);
        assert_eq!(r1.listings_updated, 0);

        let b = parse_export(include_str!("../testdata/export_second_search.json")).unwrap();
        let r2 = db.import_export(&b).unwrap();
        assert!(r2.listings_updated >= 1);
        assert_eq!(r2.query, "cute birthday card");

        let store = db.load().unwrap();
        assert_eq!(store.searches.len(), 2);
        let shared = store
            .listings
            .iter()
            .find(|l| l.url.contains("/listing/1001/"))
            .expect("shared listing");
        assert_eq!(shared.search_ids.len(), 2);
        assert!(shared.queries.iter().any(|q| q.contains("funny")));
        assert!(shared.queries.iter().any(|q| q.contains("cute")));
        assert!(store.listings.iter().any(|l| l.url.contains("/listing/2001/")));
    }

    #[test]
    fn delete_search_keeps_shared_removes_orphans() {
        let mut db = temp_db();
        let a = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        let r1 = db.import_export(&a).unwrap();
        let b = parse_export(include_str!("../testdata/export_second_search.json")).unwrap();
        let r2 = db.import_export(&b).unwrap();

        db.delete_search(r1.search_id).unwrap();
        let store = db.load().unwrap();
        assert_eq!(store.searches.len(), 1);
        assert_eq!(store.searches[0].id, r2.search_id);
        // Shared 1001 kept; sample-only listings gone; 2001 kept.
        assert!(store.listings.iter().any(|l| l.url.contains("/listing/1001/")));
        assert!(store.listings.iter().any(|l| l.url.contains("/listing/2001/")));
        assert!(!store.listings.iter().any(|l| l.url.contains("/listing/1002/")));
    }

    #[test]
    fn title_clean_round_trip() {
        let mut db = temp_db();
        let a = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        db.import_export(&a).unwrap();
        let store = db.load().unwrap();
        let id = store.listings[0].id;
        db.set_title_clean(id, "Short title").unwrap();
        let store = db.load().unwrap();
        let listing = store.listings.iter().find(|l| l.id == id).unwrap();
        assert_eq!(listing.title_clean, "Short title");
        assert_eq!(listing.display_title(), "Short title");
    }

    #[test]
    fn currency_other_than_gbp_kept() {
        let mut db = temp_db();
        let payload = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        db.import_export(&payload).unwrap();
        let store = db.load().unwrap();
        let usd = store
            .listings
            .iter()
            .find(|l| l.currency == "USD")
            .expect("usd listing");
        assert_eq!(usd.price, Some(4.99));
    }

    #[test]
    fn normalisation_persisted_on_import() {
        let mut db = temp_db();
        let payload = parse_export(include_str!("../testdata/export_sample.json")).unwrap();
        db.import_export(&payload).unwrap();
        let store = db.load().unwrap();
        let pack = store
            .listings
            .iter()
            .find(|l| l.title.to_lowercase().contains("pack of 4"))
            .expect("pack");
        assert_eq!(pack.pack_size, Some(4));
        assert!(pack.is_bundle);
        assert_eq!(pack.postage, Some(0.0));
        assert!(pack.delivered_price.is_some());
        let personal = store
            .listings
            .iter()
            .find(|l| l.title.to_lowercase().contains("personalised"))
            .expect("personalised");
        assert!(personal.is_personalised);
    }
}

