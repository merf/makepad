//! Offline demo catalogue + import of saved BGG thing XML files.
//!
//! Use while waiting for a BGG application token. Demo numbers are approximate
//! community-style polls so What to play ranking is exercisable. One-off page
//! dumps for hand-tuning live under `local/boardgames/fixtures/` — the app
//! never scrapes BGG HTML; live sync is XML API2 + Bearer token only.

use crate::db::{now_unix, Db};
use crate::model::{Catalogue, Game, PlayerCount};
use std::path::Path;

struct DemoSpec {
    game: Game,
    own_by_me: bool,
    on_bga: bool,
    on_tts: bool,
}

/// Seed eight well-known games (idempotent upsert by bgg_id).
pub fn load_demo_catalogue(db: &mut Db, cat: &Catalogue) -> Result<usize, String> {
    let me = cat.me_person_id;
    let bga = cat.platform_by_slug("bga").map(|p| p.id);
    let tts = cat.platform_by_slug("tts").map(|p| p.id);
    let mut inserted = 0;
    for spec in demo_specs() {
        let game = spec.game;
        let id = db.upsert_game_from_bgg(&game, me, spec.own_by_me)?;
        if spec.on_bga {
            if let Some(pid) = bga {
                db.set_manual_platform(id, pid, true)?;
            }
        }
        if spec.on_tts {
            if let Some(pid) = tts {
                db.set_manual_platform(id, pid, true)?;
            }
        }
        inserted += 1;
    }
    ensure_friend_and_share(db, cat)?;
    Ok(inserted)
}

fn ensure_friend_and_share(db: &mut Db, cat: &Catalogue) -> Result<(), String> {
    let alex_id = if let Some(p) = cat.people.iter().find(|p| p.name == "Alex") {
        p.id
    } else {
        db.add_person("Alex")?
    };
    // Catan + Azul: Alex also has a copy so "Everyone" differs from "Me".
    for bgg_id in [13i64, 230802] {
        if let Some(game_id) = db.find_game_id_by_bgg(bgg_id)? {
            let mut owners = db.load_owner_ids(game_id)?;
            if !owners.contains(&alex_id) {
                owners.push(alex_id);
                db.set_owners(game_id, &owners)?;
            }
        }
    }
    Ok(())
}

/// Import every `*.xml` in `dir` as a BGG thing payload.
pub fn import_xml_dir(
    db: &mut Db,
    cat: &Catalogue,
    dir: &Path,
    parse_thing: fn(&str, i64, &std::collections::HashMap<String, i64>) -> Result<Game, String>,
) -> Result<(usize, usize), String> {
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        return Ok((0, 0));
    }
    let mut platform_ids = std::collections::HashMap::new();
    for p in &cat.platforms {
        platform_ids.insert(p.slug.clone(), p.id);
    }
    let me = cat.me_person_id;
    let mut ok = 0usize;
    let mut fail = 0usize;
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("xml") {
            continue;
        }
        let xml = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                fail += 1;
                continue;
            }
        };
        let bgg_id = guess_bgg_id(&xml).unwrap_or(0);
        match parse_thing(&xml, bgg_id, &platform_ids) {
            Ok(game) => {
                if db.upsert_game_from_bgg(&game, me, true).is_ok() {
                    ok += 1;
                } else {
                    fail += 1;
                }
            }
            Err(_) => fail += 1,
        }
    }
    Ok((ok, fail))
}

fn guess_bgg_id(xml: &str) -> Option<i64> {
    let marker = "id=\"";
    let start = xml.find("<item")?;
    let slice = &xml[start..];
    let id_at = slice.find(marker)? + marker.len();
    let end = slice[id_at..].find('"')? + id_at;
    slice[id_at..end].parse().ok()
}

fn demo_specs() -> Vec<DemoSpec> {
    vec![
        DemoSpec {
            own_by_me: true,
            on_bga: true,
            on_tts: true,
            game: game(
                13,
                "CATAN",
                1995,
                3,
                4,
                60,
                120,
                7.1,
                6.9,
                2.3,
                "Best with 4 players",
                "Recommended with 3–4 players",
                &[
                    ("1", 0, 1, 40),
                    ("2", 0, 2, 38),
                    ("3", 12, 28, 5),
                    ("4", 35, 10, 1),
                    ("5+", 1, 3, 30),
                ],
            ),
        },
        DemoSpec {
            own_by_me: true,
            on_bga: true,
            on_tts: false,
            game: game(
                230802,
                "Azul",
                2017,
                2,
                4,
                30,
                45,
                7.8,
                7.5,
                1.8,
                "Best with 2 players",
                "Recommended with 2–4 players",
                &[
                    ("1", 0, 0, 20),
                    ("2", 40, 15, 2),
                    ("3", 18, 30, 4),
                    ("4", 12, 28, 6),
                    ("5+", 0, 1, 25),
                ],
            ),
        },
        DemoSpec {
            own_by_me: true,
            on_bga: false,
            on_tts: true,
            game: game(
                174430,
                "Gloomhaven",
                2017,
                1,
                4,
                60,
                120,
                8.7,
                8.5,
                3.9,
                "Best with 3 players",
                "Recommended with 1–4 players",
                &[
                    ("1", 20, 25, 8),
                    ("2", 30, 22, 5),
                    ("3", 45, 15, 3),
                    ("4", 18, 28, 10),
                    ("5+", 0, 0, 40),
                ],
            ),
        },
        DemoSpec {
            own_by_me: true,
            on_bga: true,
            on_tts: false,
            game: game(
                266192,
                "Wingspan",
                2019,
                1,
                5,
                40,
                70,
                8.1,
                7.9,
                2.4,
                "Best with 3 players",
                "Recommended with 1–4 players",
                &[
                    ("1", 15, 30, 8),
                    ("2", 25, 28, 4),
                    ("3", 40, 18, 2),
                    ("4", 20, 25, 5),
                    ("5", 5, 15, 20),
                    ("6+", 0, 2, 30),
                ],
            ),
        },
        DemoSpec {
            own_by_me: false,
            on_bga: true,
            on_tts: false,
            game: game(
                167791,
                "Terraforming Mars",
                2016,
                1,
                5,
                90,
                120,
                8.4,
                8.2,
                3.2,
                "Best with 3 players",
                "Recommended with 1–4 players",
                &[
                    ("1", 22, 28, 6),
                    ("2", 20, 30, 5),
                    ("3", 42, 16, 2),
                    ("4", 25, 22, 4),
                    ("5", 8, 18, 15),
                    ("6+", 0, 1, 35),
                ],
            ),
        },
        DemoSpec {
            own_by_me: true,
            on_bga: true,
            on_tts: true,
            game: game(
                342942,
                "Ark Nova",
                2021,
                1,
                4,
                90,
                150,
                8.5,
                8.3,
                3.7,
                "Best with 2 players",
                "Recommended with 1–3 players",
                &[
                    ("1", 28, 25, 5),
                    ("2", 45, 15, 2),
                    ("3", 20, 28, 6),
                    ("4", 8, 18, 20),
                    ("5+", 0, 1, 40),
                ],
            ),
        },
        DemoSpec {
            own_by_me: true,
            on_bga: false,
            on_tts: false,
            game: game(
                9209,
                "Ticket to Ride",
                2004,
                2,
                5,
                30,
                60,
                7.4,
                7.2,
                1.8,
                "Best with 4 players",
                "Recommended with 2–5 players",
                &[
                    ("1", 0, 0, 30),
                    ("2", 10, 25, 10),
                    ("3", 20, 30, 4),
                    ("4", 35, 18, 2),
                    ("5", 15, 25, 5),
                    ("6+", 1, 5, 25),
                ],
            ),
        },
        DemoSpec {
            own_by_me: false,
            on_bga: true,
            on_tts: false,
            game: game(
                237182,
                "Root",
                2018,
                2,
                4,
                60,
                90,
                8.1,
                7.8,
                3.7,
                "Best with 4 players",
                "Recommended with 3–4 players",
                &[
                    ("1", 0, 2, 35),
                    ("2", 5, 15, 25),
                    ("3", 20, 30, 8),
                    ("4", 40, 15, 3),
                    ("5+", 2, 5, 30),
                ],
            ),
        },
    ]
}

fn game(
    bgg_id: i64,
    name: &str,
    year: i32,
    min_p: i32,
    max_p: i32,
    min_pt: i32,
    max_pt: i32,
    avg: f64,
    bayes: f64,
    weight: f64,
    best_with: &str,
    rec_with: &str,
    polls: &[(&str, i32, i32, i32)],
) -> Game {
    let mut player_counts = Vec::new();
    let mut poll_votes = 0;
    for (label, best, rec, not_rec) in polls {
        let (player_n, plus) = PlayerCount::parse_numplayers(label);
        poll_votes += best + rec + not_rec;
        player_counts.push(PlayerCount {
            numplayers: (*label).into(),
            player_n,
            plus,
            best_votes: *best,
            rec_votes: *rec,
            not_rec_votes: *not_rec,
        });
    }
    let thumb = demo_thumb(bgg_id);
    let playtime = if max_pt > 0 { max_pt } else { min_pt };
    Game {
        bgg_id: Some(bgg_id),
        bgg_type: "boardgame".into(),
        name: name.into(),
        year,
        min_players: min_p,
        max_players: max_p,
        playtime,
        min_playtime: min_pt,
        max_playtime: max_pt,
        thumbnail_url: thumb.into(),
        image_url: thumb.into(),
        rating_average: avg,
        rating_bayes: bayes,
        weight,
        poll_votes,
        best_with: best_with.into(),
        recommended_with: rec_with.into(),
        bgg_synced_at: now_unix(),
        bgg_payload: format!("<!-- demo seed bgg_id={bgg_id} -->"),
        player_counts,
        local_notes: "demo".into(),
        ..Game::default()
    }
}

/// Stable BGG CDN thumbs for the offline demo (tall poster cards).
fn demo_thumb(bgg_id: i64) -> &'static str {
    match bgg_id {
        13 => "https://cf.geekdo-images.com/0XODRpReiZBFUffEcqT5-Q__itemrep/img/6Jf5G-bSvdOIMUSwxsJfZXl29B8=/fit-in/246x300/filters:strip_icc()/pic9156909.png",
        230802 => "https://cf.geekdo-images.com/aPSHJO0d0XOpQR5X-wJonw__itemrep/img/7EYDEVj3bLVkRHGZGM1LZ7KfA6w=/fit-in/246x300/filters:strip_icc()/pic3718960.jpg",
        266192 => "https://cf.geekdo-images.com/yLZJCVLiRRbnVlsZggB5yw__itemrep/img/uyFh-dgfBrH90UKm9fG_7J9YpgM=/fit-in/246x300/filters:strip_icc()/pic4458123.jpg",
        174430 => "https://cf.geekdo-images.com/s9fjRdxEkYhN1ZR0PIPqQA__itemrep/img/7h-L86Z_7H8=/fit-in/246x300/filters:strip_icc()/pic2437871.jpg",
        167791 => "https://cf.geekdo-images.com/wg9oOLcsKvDesSUdZQ4rxw__itemrep/img/8J5vY5w=/fit-in/246x300/filters:strip_icc()/pic3477869.jpg",
        342942 => "https://cf.geekdo-images.com/SoU8pYVpxqDORtVc2zt76A__itemrep/img/8J5vY5w=/fit-in/246x300/filters:strip_icc()/pic6293412.jpg",
        9209 => "https://cf.geekdo-images.com/ZW2o6172P8d4Z1jKjXjQKQ__itemrep/img/8J5vY5w=/fit-in/246x300/filters:strip_icc()/pic38668.jpg",
        237182 => "https://cf.geekdo-images.com/jwQszfAmevd6uLZSmKvDTQ__itemrep/img/8J5vY5w=/fit-in/246x300/filters:strip_icc()/pic4254509.jpg",
        _ => "",
    }
}
