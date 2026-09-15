//! SQLite file at `local/vinyl/vinyl.db`. Load into memory; write through.

use crate::model::Record;
use makepad_sqlite::{Connection, Value};
use std::path::Path;
use std::time::Duration;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS records(
    id INTEGER PRIMARY KEY,
    record_number INTEGER NOT NULL,
    raw_transcript TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    catalogue_number TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    discogs_master_id TEXT NOT NULL DEFAULT '',
    discogs_master_url TEXT NOT NULL DEFAULT '',
    discogs_release_id TEXT NOT NULL DEFAULT '',
    discogs_url TEXT NOT NULL DEFAULT '',
    thumb_path TEXT NOT NULL DEFAULT '',
    condition TEXT NOT NULL DEFAULT '',
    price_suggestions TEXT NOT NULL DEFAULT '',
    display_price REAL NOT NULL DEFAULT 0,
    num_for_sale INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'captured',
    created_at INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL DEFAULT 0
);
";

const MIGRATIONS: &[&str] = &[
    "ALTER TABLE records ADD COLUMN discogs_master_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE records ADD COLUMN discogs_master_url TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE records ADD COLUMN thumb_path TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE records ADD COLUMN condition TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE records ADD COLUMN price_suggestions TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE records ADD COLUMN display_price REAL NOT NULL DEFAULT 0",
    "ALTER TABLE records ADD COLUMN num_for_sale INTEGER NOT NULL DEFAULT 0",
];

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Db, String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let mut conn = Connection::open(path, Duration::from_secs(5))
            .map_err(|e| format!("open {}: {e:?}", path.display()))?;
        conn.limits_mut().max_rows = 500_000;
        conn.limits_mut().max_steps = 50_000_000;
        conn.execute_batch(SCHEMA).map_err(|e| format!("schema: {e:?}"))?;
        for sql in MIGRATIONS {
            let _ = conn.execute(sql, &[]);
        }
        Ok(Db { conn })
    }

    pub fn load(&mut self) -> Result<Vec<Record>, String> {
        let rows = self
            .conn
            .query(
                "SELECT id, record_number, raw_transcript, artist, title, label, catalogue_number, \
                 notes, discogs_master_id, discogs_master_url, discogs_release_id, discogs_url, \
                 thumb_path, condition, price_suggestions, display_price, num_for_sale, status, \
                 created_at, updated_at \
                 FROM records ORDER BY record_number, id",
                &[],
            )
            .map_err(|e| format!("load records: {e:?}"))?;
        Ok(rows.rows.iter().map(|row| Record {
            id: int(&row[0]),
            record_number: int(&row[1]),
            raw_transcript: text(&row[2]),
            artist: text(&row[3]),
            title: text(&row[4]),
            label: text(&row[5]),
            catalogue_number: text(&row[6]),
            notes: text(&row[7]),
            discogs_master_id: text(&row[8]),
            discogs_master_url: text(&row[9]),
            discogs_release_id: text(&row[10]),
            discogs_url: text(&row[11]),
            thumb_path: text(&row[12]),
            condition: text(&row[13]),
            price_suggestions: text(&row[14]),
            display_price: float(&row[15]),
            num_for_sale: int(&row[16]),
            status: text(&row[17]),
            created_at: int(&row[18]),
            updated_at: int(&row[19]),
        }).collect())
    }

    pub fn insert(&mut self, record: &mut Record) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT INTO records(record_number, raw_transcript, artist, title, label, \
                 catalogue_number, notes, discogs_master_id, discogs_master_url, \
                 discogs_release_id, discogs_url, thumb_path, condition, price_suggestions, \
                 display_price, num_for_sale, status, created_at, updated_at) \
                 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                &values(record),
            )
            .map_err(|e| format!("insert: {e:?}"))?;
        record.id = self.last_id()?;
        Ok(())
    }

    pub fn update(&mut self, record: &Record) -> Result<(), String> {
        let mut params = values(record);
        params.push(Value::Integer(record.id));
        self.conn
            .execute(
                "UPDATE records SET record_number=?, raw_transcript=?, artist=?, title=?, label=?, \
                 catalogue_number=?, notes=?, discogs_master_id=?, discogs_master_url=?, \
                 discogs_release_id=?, discogs_url=?, thumb_path=?, condition=?, price_suggestions=?, \
                 display_price=?, num_for_sale=?, status=?, created_at=?, updated_at=? WHERE id=?",
                &params,
            )
            .map_err(|e| format!("update: {e:?}"))?;
        Ok(())
    }

    pub fn delete(&mut self, id: i64) -> Result<(), String> {
        self.conn
            .execute("DELETE FROM records WHERE id=?", &[Value::Integer(id)])
            .map_err(|e| format!("delete: {e:?}"))?;
        Ok(())
    }

    fn last_id(&mut self) -> Result<i64, String> {
        let rows = self
            .conn
            .query("SELECT MAX(id) FROM records", &[])
            .map_err(|e| format!("last id: {e:?}"))?;
        Ok(rows.scalar().and_then(|v| v.as_integer()).unwrap_or(0))
    }
}

fn values(record: &Record) -> Vec<Value> {
    vec![
        Value::Integer(record.record_number),
        Value::text(record.raw_transcript.as_str()),
        Value::text(record.artist.as_str()),
        Value::text(record.title.as_str()),
        Value::text(record.label.as_str()),
        Value::text(record.catalogue_number.as_str()),
        Value::text(record.notes.as_str()),
        Value::text(record.discogs_master_id.as_str()),
        Value::text(record.discogs_master_url.as_str()),
        Value::text(record.discogs_release_id.as_str()),
        Value::text(record.discogs_url.as_str()),
        Value::text(record.thumb_path.as_str()),
        Value::text(record.condition.as_str()),
        Value::text(record.price_suggestions.as_str()),
        Value::Real(record.display_price),
        Value::Integer(record.num_for_sale),
        Value::text(record.status.as_str()),
        Value::Integer(record.created_at),
        Value::Integer(record.updated_at),
    ]
}

fn int(value: &Value) -> i64 {
    value.as_integer().unwrap_or(0)
}

fn float(value: &Value) -> f64 {
    value.as_real().unwrap_or(0.0)
}

fn text(value: &Value) -> String {
    value.as_text().unwrap_or("").to_string()
}
