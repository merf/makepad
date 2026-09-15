//! SQLite file at `local/boardgames/catalogue.db`. Load into RAM; write through.

use crate::model::*;
use makepad_sqlite::{Connection, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS settings(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS people(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS people_name ON people(name);
CREATE TABLE IF NOT EXISTS games(
    id INTEGER PRIMARY KEY,
    bgg_id INTEGER,
    bgg_type TEXT NOT NULL DEFAULT 'boardgame',
    name TEXT NOT NULL,
    year INTEGER NOT NULL DEFAULT 0,
    min_players INTEGER NOT NULL DEFAULT 0,
    max_players INTEGER NOT NULL DEFAULT 0,
    playtime INTEGER NOT NULL DEFAULT 0,
    min_playtime INTEGER NOT NULL DEFAULT 0,
    max_playtime INTEGER NOT NULL DEFAULT 0,
    min_age INTEGER NOT NULL DEFAULT 0,
    thumbnail_url TEXT NOT NULL DEFAULT '',
    image_url TEXT NOT NULL DEFAULT '',
    rating_average REAL NOT NULL DEFAULT 0,
    rating_bayes REAL NOT NULL DEFAULT 0,
    rating_rank INTEGER NOT NULL DEFAULT 0,
    users_rated INTEGER NOT NULL DEFAULT 0,
    weight REAL NOT NULL DEFAULT 0,
    poll_votes INTEGER NOT NULL DEFAULT 0,
    best_with TEXT NOT NULL DEFAULT '',
    recommended_with TEXT NOT NULL DEFAULT '',
    local_notes TEXT NOT NULL DEFAULT '',
    bgg_synced_at INTEGER NOT NULL DEFAULT 0,
    bgg_payload TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS games_bgg_id ON games(bgg_id) WHERE bgg_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS games_name ON games(name);
CREATE TABLE IF NOT EXISTS player_counts(
    game_id INTEGER NOT NULL,
    numplayers TEXT NOT NULL,
    player_n INTEGER NOT NULL,
    plus INTEGER NOT NULL DEFAULT 0,
    best_votes INTEGER NOT NULL DEFAULT 0,
    rec_votes INTEGER NOT NULL DEFAULT 0,
    not_rec_votes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(game_id, numplayers)
);
CREATE INDEX IF NOT EXISTS player_counts_n ON player_counts(game_id, player_n, plus);
CREATE TABLE IF NOT EXISTS game_links(
    game_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    bgg_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY(game_id, kind, bgg_id)
);
CREATE TABLE IF NOT EXISTS owners(
    game_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    PRIMARY KEY(game_id, person_id)
);
CREATE INDEX IF NOT EXISTS owners_person ON owners(person_id);
CREATE TABLE IF NOT EXISTS platforms(
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL,
    name TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS platforms_slug ON platforms(slug);
CREATE TABLE IF NOT EXISTS game_platforms(
    game_id INTEGER NOT NULL,
    platform_id INTEGER NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    PRIMARY KEY(game_id, platform_id)
);
CREATE INDEX IF NOT EXISTS game_platforms_platform ON game_platforms(platform_id);
CREATE TABLE IF NOT EXISTS shelves(
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS shelves_slug ON shelves(slug);
CREATE TABLE IF NOT EXISTS shelf_items(
    shelf_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    added_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(shelf_id, game_id)
);
CREATE INDEX IF NOT EXISTS shelf_items_game ON shelf_items(game_id);
";

const MIGRATIONS: [&str; 0] = [];

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
        conn.execute_batch(SCHEMA)
            .map_err(|e| format!("schema: {e:?}"))?;
        let mut db = Db { conn };
        db.migrate()?;
        db.seed_if_needed()?;
        Ok(db)
    }

    fn migrate(&mut self) -> Result<(), String> {
        let from = self.conn.user_version().max(0) as usize;
        for (index, sql) in MIGRATIONS.iter().enumerate().skip(from) {
            self.conn
                .execute_batch(sql)
                .map_err(|e| format!("migration {}: {e:?}", index + 1))?;
        }
        let target = SCHEMA_VERSION.max(MIGRATIONS.len() as i64);
        if from < target as usize {
            self.conn
                .execute(&format!("PRAGMA user_version = {target}"), &[])
                .map_err(|e| format!("set user_version: {e:?}"))?;
        }
        Ok(())
    }

    fn seed_if_needed(&mut self) -> Result<(), String> {
        let people = self
            .conn
            .query("SELECT COUNT(*) FROM people", &[])
            .map_err(|e| format!("count people: {e:?}"))?;
        if int(people.scalar().unwrap_or(&Value::Integer(0))) == 0 {
            self.conn
                .execute(
                    "INSERT INTO people(name, sort_order) VALUES('Me', 0)",
                    &[],
                )
                .map_err(|e| format!("seed me: {e:?}"))?;
            let me = self.last_id("people")?;
            self.conn
                .execute(
                    "INSERT OR REPLACE INTO settings(key, value) VALUES('me_person_id', ?)",
                    &[Value::text(me.to_string())],
                )
                .map_err(|e| format!("seed me setting: {e:?}"))?;
        }

        let shelves = self
            .conn
            .query("SELECT COUNT(*) FROM shelves", &[])
            .map_err(|e| format!("count shelves: {e:?}"))?;
        if int(shelves.scalar().unwrap_or(&Value::Integer(0))) == 0 {
            for (slug, name, order) in [("wishlist", "Wishlist", 0), ("to_play", "To play", 1)] {
                self.conn
                    .execute(
                        "INSERT INTO shelves(slug, name, sort_order) VALUES(?, ?, ?)",
                        &[
                            Value::text(slug),
                            Value::text(name),
                            Value::Integer(order),
                        ],
                    )
                    .map_err(|e| format!("seed shelf {slug}: {e:?}"))?;
            }
        }

        let platforms = self
            .conn
            .query("SELECT COUNT(*) FROM platforms", &[])
            .map_err(|e| format!("count platforms: {e:?}"))?;
        if int(platforms.scalar().unwrap_or(&Value::Integer(0))) == 0 {
            for (slug, name) in [("tts", "Tabletop Simulator"), ("bga", "Board Game Arena")] {
                self.conn
                    .execute(
                        "INSERT INTO platforms(slug, name) VALUES(?, ?)",
                        &[Value::text(slug), Value::text(name)],
                    )
                    .map_err(|e| format!("seed platform {slug}: {e:?}"))?;
            }
        }

        // Ensure me_person_id always points at someone.
        let me_setting = self
            .conn
            .query(
                "SELECT value FROM settings WHERE key = 'me_person_id'",
                &[],
            )
            .map_err(|e| format!("read me: {e:?}"))?;
        if me_setting.rows.is_empty() {
            let first = self
                .conn
                .query("SELECT id FROM people ORDER BY id LIMIT 1", &[])
                .map_err(|e| format!("first person: {e:?}"))?;
            if let Some(row) = first.rows.first() {
                let id = int(&row[0]);
                self.conn
                    .execute(
                        "INSERT INTO settings(key, value) VALUES('me_person_id', ?)",
                        &[Value::text(id.to_string())],
                    )
                    .map_err(|e| format!("write me: {e:?}"))?;
            }
        }
        Ok(())
    }

    fn last_id(&mut self, table: &str) -> Result<i64, String> {
        let sql = format!("SELECT MAX(id) FROM {table}");
        let result = self
            .conn
            .query(&sql, &[])
            .map_err(|e| format!("last id {table}: {e:?}"))?;
        Ok(int(result.scalar().unwrap_or(&Value::Integer(0))))
    }

    pub fn load(&mut self) -> Result<Catalogue, String> {
        let mut cat = Catalogue::default();

        let rows = self
            .conn
            .query("SELECT key, value FROM settings", &[])
            .map_err(|e| format!("load settings: {e:?}"))?;
        for row in &rows.rows {
            cat.settings.insert(text(&row[0]), text(&row[1]));
        }
        cat.me_person_id = cat
            .settings
            .get("me_person_id")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let rows = self
            .conn
            .query(
                "SELECT id, name, sort_order FROM people ORDER BY sort_order, id",
                &[],
            )
            .map_err(|e| format!("load people: {e:?}"))?;
        for row in &rows.rows {
            cat.people.push(Person {
                id: int(&row[0]),
                name: text(&row[1]),
                sort_order: int(&row[2]) as i32,
            });
        }

        let rows = self
            .conn
            .query("SELECT id, slug, name FROM platforms ORDER BY id", &[])
            .map_err(|e| format!("load platforms: {e:?}"))?;
        for row in &rows.rows {
            cat.platforms.push(Platform {
                id: int(&row[0]),
                slug: text(&row[1]),
                name: text(&row[2]),
            });
        }

        let rows = self
            .conn
            .query(
                "SELECT id, slug, name, sort_order FROM shelves ORDER BY sort_order, id",
                &[],
            )
            .map_err(|e| format!("load shelves: {e:?}"))?;
        for row in &rows.rows {
            cat.shelves.push(Shelf {
                id: int(&row[0]),
                slug: text(&row[1]),
                name: text(&row[2]),
                sort_order: int(&row[3]) as i32,
            });
        }

        let rows = self
            .conn
            .query(
                "SELECT id, bgg_id, bgg_type, name, year, min_players, max_players, playtime, \
                 min_playtime, max_playtime, min_age, thumbnail_url, image_url, \
                 rating_average, rating_bayes, rating_rank, users_rated, weight, poll_votes, \
                 best_with, recommended_with, local_notes, bgg_synced_at, bgg_payload \
                 FROM games ORDER BY name, id",
                &[],
            )
            .map_err(|e| format!("load games: {e:?}"))?;

        let mut by_id: HashMap<i64, usize> = HashMap::new();
        for row in &rows.rows {
            let id = int(&row[0]);
            let game = Game {
                id,
                bgg_id: opt_int(&row[1]),
                bgg_type: text(&row[2]),
                name: text(&row[3]),
                year: int(&row[4]) as i32,
                min_players: int(&row[5]) as i32,
                max_players: int(&row[6]) as i32,
                playtime: int(&row[7]) as i32,
                min_playtime: int(&row[8]) as i32,
                max_playtime: int(&row[9]) as i32,
                min_age: int(&row[10]) as i32,
                thumbnail_url: text(&row[11]),
                image_url: text(&row[12]),
                rating_average: real(&row[13]),
                rating_bayes: real(&row[14]),
                rating_rank: int(&row[15]) as i32,
                users_rated: int(&row[16]) as i32,
                weight: real(&row[17]),
                poll_votes: int(&row[18]) as i32,
                best_with: text(&row[19]),
                recommended_with: text(&row[20]),
                local_notes: text(&row[21]),
                bgg_synced_at: int(&row[22]),
                bgg_payload: text(&row[23]),
                ..Game::default()
            };
            by_id.insert(id, cat.games.len());
            cat.games.push(game);
        }

        let rows = self
            .conn
            .query(
                "SELECT game_id, numplayers, player_n, plus, best_votes, rec_votes, not_rec_votes \
                 FROM player_counts",
                &[],
            )
            .map_err(|e| format!("load player_counts: {e:?}"))?;
        for row in &rows.rows {
            let game_id = int(&row[0]);
            if let Some(&idx) = by_id.get(&game_id) {
                cat.games[idx].player_counts.push(PlayerCount {
                    numplayers: text(&row[1]),
                    player_n: int(&row[2]) as i32,
                    plus: int(&row[3]) != 0,
                    best_votes: int(&row[4]) as i32,
                    rec_votes: int(&row[5]) as i32,
                    not_rec_votes: int(&row[6]) as i32,
                });
            }
        }
        for game in &mut cat.games {
            game.player_counts.sort_by_key(|p| (p.plus, p.player_n));
        }

        let rows = self
            .conn
            .query(
                "SELECT game_id, kind, bgg_id, name FROM game_links",
                &[],
            )
            .map_err(|e| format!("load game_links: {e:?}"))?;
        for row in &rows.rows {
            let game_id = int(&row[0]);
            if let Some(&idx) = by_id.get(&game_id) {
                cat.games[idx].links.push(GameLink {
                    kind: text(&row[1]),
                    bgg_id: int(&row[2]),
                    name: text(&row[3]),
                });
            }
        }

        let rows = self
            .conn
            .query("SELECT game_id, person_id FROM owners", &[])
            .map_err(|e| format!("load owners: {e:?}"))?;
        for row in &rows.rows {
            let game_id = int(&row[0]);
            if let Some(&idx) = by_id.get(&game_id) {
                cat.games[idx].owner_ids.push(int(&row[1]));
            }
        }

        let rows = self
            .conn
            .query(
                "SELECT game_id, platform_id, url, source FROM game_platforms",
                &[],
            )
            .map_err(|e| format!("load game_platforms: {e:?}"))?;
        for row in &rows.rows {
            let game_id = int(&row[0]);
            if let Some(&idx) = by_id.get(&game_id) {
                cat.games[idx].platforms.push(GamePlatform {
                    platform_id: int(&row[1]),
                    url: text(&row[2]),
                    source: text(&row[3]),
                });
            }
        }

        let rows = self
            .conn
            .query(
                "SELECT shelf_id, game_id, notes, added_at FROM shelf_items",
                &[],
            )
            .map_err(|e| format!("load shelf_items: {e:?}"))?;
        for row in &rows.rows {
            let game_id = int(&row[1]);
            if let Some(&idx) = by_id.get(&game_id) {
                cat.games[idx].shelves.push(ShelfItem {
                    shelf_id: int(&row[0]),
                    notes: text(&row[2]),
                    added_at: int(&row[3]),
                });
            }
        }

        Ok(cat)
    }

    pub fn upsert_game_from_bgg(
        &mut self,
        game: &Game,
        me_person_id: i64,
        mark_owned_by_me: bool,
    ) -> Result<i64, String> {
        let existing = if let Some(bgg_id) = game.bgg_id {
            let rows = self
                .conn
                .query(
                    "SELECT id FROM games WHERE bgg_id = ?",
                    &[Value::Integer(bgg_id)],
                )
                .map_err(|e| format!("find bgg: {e:?}"))?;
            rows.rows.first().map(|r| int(&r[0]))
        } else {
            None
        };

        let id = if let Some(id) = existing {
            self.conn
                .execute(
                    "UPDATE games SET bgg_type=?, name=?, year=?, min_players=?, max_players=?, \
                     playtime=?, min_playtime=?, max_playtime=?, min_age=?, thumbnail_url=?, \
                     image_url=?, rating_average=?, rating_bayes=?, rating_rank=?, users_rated=?, \
                     weight=?, poll_votes=?, best_with=?, recommended_with=?, bgg_synced_at=?, \
                     bgg_payload=? WHERE id=?",
                    &[
                        Value::text(game.bgg_type.as_str()),
                        Value::text(game.name.as_str()),
                        Value::Integer(game.year as i64),
                        Value::Integer(game.min_players as i64),
                        Value::Integer(game.max_players as i64),
                        Value::Integer(game.playtime as i64),
                        Value::Integer(game.min_playtime as i64),
                        Value::Integer(game.max_playtime as i64),
                        Value::Integer(game.min_age as i64),
                        Value::text(game.thumbnail_url.as_str()),
                        Value::text(game.image_url.as_str()),
                        Value::Real(game.rating_average),
                        Value::Real(game.rating_bayes),
                        Value::Integer(game.rating_rank as i64),
                        Value::Integer(game.users_rated as i64),
                        Value::Real(game.weight),
                        Value::Integer(game.poll_votes as i64),
                        Value::text(game.best_with.as_str()),
                        Value::text(game.recommended_with.as_str()),
                        Value::Integer(game.bgg_synced_at),
                        Value::text(game.bgg_payload.as_str()),
                        Value::Integer(id),
                    ],
                )
                .map_err(|e| format!("update game: {e:?}"))?;
            id
        } else {
            let bgg_val = match game.bgg_id {
                Some(v) => Value::Integer(v),
                None => Value::Null,
            };
            self.conn
                .execute(
                    "INSERT INTO games(bgg_id, bgg_type, name, year, min_players, max_players, \
                     playtime, min_playtime, max_playtime, min_age, thumbnail_url, image_url, \
                     rating_average, rating_bayes, rating_rank, users_rated, weight, poll_votes, \
                     best_with, recommended_with, local_notes, bgg_synced_at, bgg_payload) \
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    &[
                        bgg_val,
                        Value::text(game.bgg_type.as_str()),
                        Value::text(game.name.as_str()),
                        Value::Integer(game.year as i64),
                        Value::Integer(game.min_players as i64),
                        Value::Integer(game.max_players as i64),
                        Value::Integer(game.playtime as i64),
                        Value::Integer(game.min_playtime as i64),
                        Value::Integer(game.max_playtime as i64),
                        Value::Integer(game.min_age as i64),
                        Value::text(game.thumbnail_url.as_str()),
                        Value::text(game.image_url.as_str()),
                        Value::Real(game.rating_average),
                        Value::Real(game.rating_bayes),
                        Value::Integer(game.rating_rank as i64),
                        Value::Integer(game.users_rated as i64),
                        Value::Real(game.weight),
                        Value::Integer(game.poll_votes as i64),
                        Value::text(game.best_with.as_str()),
                        Value::text(game.recommended_with.as_str()),
                        Value::text(game.local_notes.as_str()),
                        Value::Integer(game.bgg_synced_at),
                        Value::text(game.bgg_payload.as_str()),
                    ],
                )
                .map_err(|e| format!("insert game: {e:?}"))?;
            self.last_id("games")?
        };

        self.replace_player_counts(id, &game.player_counts)?;
        self.replace_links(id, &game.links)?;
        self.merge_bgg_platforms(id, &game.platforms)?;

        if mark_owned_by_me && me_person_id > 0 {
            let _ = self.conn.execute(
                "INSERT OR IGNORE INTO owners(game_id, person_id) VALUES(?, ?)",
                &[Value::Integer(id), Value::Integer(me_person_id)],
            );
        }

        Ok(id)
    }

    fn replace_player_counts(
        &mut self,
        game_id: i64,
        counts: &[PlayerCount],
    ) -> Result<(), String> {
        self.conn
            .execute(
                "DELETE FROM player_counts WHERE game_id = ?",
                &[Value::Integer(game_id)],
            )
            .map_err(|e| format!("clear player_counts: {e:?}"))?;
        for c in counts {
            self.conn
                .execute(
                    "INSERT INTO player_counts(game_id, numplayers, player_n, plus, best_votes, \
                     rec_votes, not_rec_votes) VALUES(?,?,?,?,?,?,?)",
                    &[
                        Value::Integer(game_id),
                        Value::text(c.numplayers.as_str()),
                        Value::Integer(c.player_n as i64),
                        Value::Integer(if c.plus { 1 } else { 0 }),
                        Value::Integer(c.best_votes as i64),
                        Value::Integer(c.rec_votes as i64),
                        Value::Integer(c.not_rec_votes as i64),
                    ],
                )
                .map_err(|e| format!("insert player_count: {e:?}"))?;
        }
        Ok(())
    }

    fn replace_links(&mut self, game_id: i64, links: &[GameLink]) -> Result<(), String> {
        self.conn
            .execute(
                "DELETE FROM game_links WHERE game_id = ?",
                &[Value::Integer(game_id)],
            )
            .map_err(|e| format!("clear links: {e:?}"))?;
        for link in links {
            self.conn
                .execute(
                    "INSERT INTO game_links(game_id, kind, bgg_id, name) VALUES(?,?,?,?)",
                    &[
                        Value::Integer(game_id),
                        Value::text(link.kind.as_str()),
                        Value::Integer(link.bgg_id),
                        Value::text(link.name.as_str()),
                    ],
                )
                .map_err(|e| format!("insert link: {e:?}"))?;
        }
        Ok(())
    }

    /// Replace BGG-sourced platforms; keep manually added ones.
    fn merge_bgg_platforms(
        &mut self,
        game_id: i64,
        platforms: &[GamePlatform],
    ) -> Result<(), String> {
        self.conn
            .execute(
                "DELETE FROM game_platforms WHERE game_id = ? AND source = 'bgg'",
                &[Value::Integer(game_id)],
            )
            .map_err(|e| format!("clear bgg platforms: {e:?}"))?;
        for p in platforms {
            if p.source != "bgg" {
                continue;
            }
            self.conn
                .execute(
                    "INSERT OR REPLACE INTO game_platforms(game_id, platform_id, url, source) \
                     VALUES(?,?,?,?)",
                    &[
                        Value::Integer(game_id),
                        Value::Integer(p.platform_id),
                        Value::text(p.url.as_str()),
                        Value::text("bgg"),
                    ],
                )
                .map_err(|e| format!("insert platform: {e:?}"))?;
        }
        Ok(())
    }

    pub fn set_owners(&mut self, game_id: i64, owner_ids: &[i64]) -> Result<(), String> {
        self.conn
            .execute(
                "DELETE FROM owners WHERE game_id = ?",
                &[Value::Integer(game_id)],
            )
            .map_err(|e| format!("clear owners: {e:?}"))?;
        for person_id in owner_ids {
            self.conn
                .execute(
                    "INSERT INTO owners(game_id, person_id) VALUES(?, ?)",
                    &[Value::Integer(game_id), Value::Integer(*person_id)],
                )
                .map_err(|e| format!("insert owner: {e:?}"))?;
        }
        Ok(())
    }

    pub fn set_manual_platform(
        &mut self,
        game_id: i64,
        platform_id: i64,
        enabled: bool,
    ) -> Result<(), String> {
        if enabled {
            self.conn
                .execute(
                    "INSERT OR REPLACE INTO game_platforms(game_id, platform_id, url, source) \
                     VALUES(?,?, '', 'manual')",
                    &[Value::Integer(game_id), Value::Integer(platform_id)],
                )
                .map_err(|e| format!("enable platform: {e:?}"))?;
        } else {
            self.conn
                .execute(
                    "DELETE FROM game_platforms WHERE game_id = ? AND platform_id = ?",
                    &[Value::Integer(game_id), Value::Integer(platform_id)],
                )
                .map_err(|e| format!("disable platform: {e:?}"))?;
        }
        Ok(())
    }

    pub fn set_local_notes(&mut self, game_id: i64, notes: &str) -> Result<(), String> {
        self.conn
            .execute(
                "UPDATE games SET local_notes = ? WHERE id = ?",
                &[Value::text(notes), Value::Integer(game_id)],
            )
            .map_err(|e| format!("notes: {e:?}"))?;
        Ok(())
    }

    pub fn add_person(&mut self, name: &str) -> Result<i64, String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("name required".into());
        }
        self.conn
            .execute(
                "INSERT INTO people(name, sort_order) VALUES(?, 100)",
                &[Value::text(name)],
            )
            .map_err(|e| format!("add person: {e:?}"))?;
        self.last_id("people")
    }

    pub fn set_me_person(&mut self, person_id: i64) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES('me_person_id', ?)",
                &[Value::text(person_id.to_string())],
            )
            .map_err(|e| format!("set me: {e:?}"))?;
        Ok(())
    }

    pub fn find_game_id_by_bgg(&mut self, bgg_id: i64) -> Result<Option<i64>, String> {
        let rows = self
            .conn
            .query(
                "SELECT id FROM games WHERE bgg_id = ?",
                &[Value::Integer(bgg_id)],
            )
            .map_err(|e| format!("find bgg: {e:?}"))?;
        Ok(rows.rows.first().map(|r| int(&r[0])))
    }

    pub fn load_owner_ids(&mut self, game_id: i64) -> Result<Vec<i64>, String> {
        let rows = self
            .conn
            .query(
                "SELECT person_id FROM owners WHERE game_id = ?",
                &[Value::Integer(game_id)],
            )
            .map_err(|e| format!("load owners: {e:?}"))?;
        Ok(rows.rows.iter().map(|r| int(&r[0])).collect())
    }
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn int(value: &Value) -> i64 {
    value.as_integer().unwrap_or(0)
}

fn opt_int(value: &Value) -> Option<i64> {
    value.as_integer()
}

fn text(value: &Value) -> String {
    value.as_text().unwrap_or("").to_string()
}

fn real(value: &Value) -> f64 {
    value.as_real().unwrap_or(0.0)
}
