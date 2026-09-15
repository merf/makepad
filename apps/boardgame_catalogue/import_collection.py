#!/usr/bin/env python3
"""Import BGG collection.csv into local/boardgames/catalogue.db.

CSV alone fills catalogue + ownership + wishlist. Cover art comes from the
public Geekdo geekitems JSON (no token). With a Bearer token in
local/boardgames/bgg_token (or $BGG_TOKEN), also pulls thing XML for
full metadata and real player-count polls.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = Path(__file__).resolve().parent / "collection.csv"
DB_PATH = REPO / "local" / "boardgames" / "catalogue.db"
TOKEN_PATH = REPO / "local" / "boardgames" / "bgg_token"
THING_URL = "https://boardgamegeek.com/xmlapi2/thing"
GEEKITEM_URL = "https://api.geekdo.com/api/geekitems"
USER_AGENT = "MakePadBoardGames/0.1 (local catalogue import)"
BATCH = 20
GAP_S = 0.8
GEEK_GAP_S = 0.25


def load_token() -> str:
    env = os.environ.get("BGG_TOKEN", "").strip()
    if env:
        return env
    if TOKEN_PATH.is_file():
        return TOKEN_PATH.read_text().strip()
    return ""


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_seed(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Return (me_person_id, wishlist_shelf_id, to_play_shelf_id)."""
    if conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 0:
        conn.execute("INSERT INTO people(name, sort_order) VALUES('Me', 0)")
    me = conn.execute(
        "SELECT value FROM settings WHERE key='me_person_id'"
    ).fetchone()
    if me is None:
        pid = conn.execute("SELECT id FROM people ORDER BY id LIMIT 1").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES('me_person_id', ?)",
            (str(pid),),
        )
        me_id = pid
    else:
        me_id = int(me[0])

    if conn.execute("SELECT COUNT(*) FROM shelves").fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO shelves(slug, name, sort_order) VALUES('wishlist','Wishlist',0)"
        )
        conn.execute(
            "INSERT INTO shelves(slug, name, sort_order) VALUES('to_play','To play',1)"
        )
    if conn.execute("SELECT COUNT(*) FROM platforms").fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO platforms(slug, name) VALUES('tts','Tabletop Simulator')"
        )
        conn.execute(
            "INSERT INTO platforms(slug, name) VALUES('bga','Board Game Arena')"
        )

    wish = conn.execute("SELECT id FROM shelves WHERE slug='wishlist'").fetchone()[0]
    to_play = conn.execute("SELECT id FROM shelves WHERE slug='to_play'").fetchone()[0]
    conn.commit()
    return me_id, wish, to_play


def parse_int(s: str, default: int = 0) -> int:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def parse_float(s: str, default: float = 0.0) -> float:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_num_list(s: str) -> list[int]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def synth_player_counts(
    min_p: int, max_p: int, best: list[int], rec: list[int]
) -> list[tuple[str, int, int, int, int, int]]:
    """Synthesize Best/Rec/Not votes from BGG collection export hints."""
    if min_p <= 0:
        min_p = 1
    if max_p < min_p:
        max_p = min_p
    best_s, rec_s = set(best), set(rec)
    rows = []
    for n in range(min_p, max_p + 1):
        label = str(n)
        if n in best_s:
            rows.append((label, n, 0, 24, 6, 1))
        elif n in rec_s:
            rows.append((label, n, 0, 6, 18, 3))
        else:
            rows.append((label, n, 0, 1, 3, 12))
    return rows


def upsert_game_from_csv(
    conn: sqlite3.Connection,
    row: dict,
    me_id: int,
    wish_id: int,
    to_play_id: int,
) -> None:
    bgg_id = parse_int(row["objectid"])
    if bgg_id <= 0:
        return
    name = (row.get("objectname") or row.get("originalname") or f"#{bgg_id}").strip()
    year = parse_int(row.get("yearpublished"))
    min_p = parse_int(row.get("minplayers"))
    max_p = parse_int(row.get("maxplayers"))
    playtime = parse_int(row.get("playingtime"))
    min_pt = parse_int(row.get("minplaytime"))
    max_pt = parse_int(row.get("maxplaytime"))
    rating_avg = parse_float(row.get("average"))
    rating_bayes = parse_float(row.get("baverage"))
    rank = parse_int(row.get("rank"))
    weight = parse_float(row.get("avgweight"))
    notes = (row.get("comment") or "").strip()
    own = row.get("own") == "1"
    wishlist = row.get("wishlist") == "1"
    want_play = row.get("wanttoplay") == "1"
    want_buy = row.get("wanttobuy") == "1"

    best = parse_num_list(row.get("bggbestplayers", ""))
    rec = parse_num_list(row.get("bggrecplayers", ""))
    if best:
        best_with = f"Best with {', '.join(str(n) for n in best)} players"
    else:
        best_with = ""
    if rec:
        lo, hi = min(rec), max(rec)
        recommended_with = (
            f"Recommended with {lo}–{hi} players" if lo != hi else f"Recommended with {lo} players"
        )
    else:
        recommended_with = ""

    existing = conn.execute(
        "SELECT id, thumbnail_url, image_url FROM games WHERE bgg_id=?", (bgg_id,)
    ).fetchone()
    now = int(time.time())
    if existing:
        game_id = existing["id"]
        # Keep any thumbs already fetched from BGG.
        conn.execute(
            """UPDATE games SET name=?, year=?, min_players=?, max_players=?,
               playtime=?, min_playtime=?, max_playtime=?,
               rating_average=?, rating_bayes=?, rating_rank=?, weight=?,
               best_with=?, recommended_with=?, local_notes=?,
               bgg_type='boardgame', bgg_synced_at=?
               WHERE id=?""",
            (
                name,
                year,
                min_p,
                max_p,
                playtime,
                min_pt,
                max_pt,
                rating_avg,
                rating_bayes,
                rank,
                weight,
                best_with,
                recommended_with,
                notes,
                now,
                game_id,
            ),
        )
    else:
        cur = conn.execute(
            """INSERT INTO games(
               bgg_id, bgg_type, name, year, min_players, max_players,
               playtime, min_playtime, max_playtime, min_age,
               thumbnail_url, image_url, rating_average, rating_bayes,
               rating_rank, users_rated, weight, poll_votes,
               best_with, recommended_with, local_notes, bgg_synced_at, bgg_payload)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                bgg_id,
                "boardgame",
                name,
                year,
                min_p,
                max_p,
                playtime,
                min_pt,
                max_pt,
                0,
                "",
                "",
                rating_avg,
                rating_bayes,
                rank,
                0,
                weight,
                0,
                best_with,
                recommended_with,
                notes,
                now,
                "collection.csv",
            ),
        )
        game_id = cur.lastrowid

    # Only replace polls when we have no real vote data yet.
    has_votes = conn.execute(
        "SELECT COALESCE(SUM(best_votes+rec_votes+not_rec_votes),0) FROM player_counts WHERE game_id=?",
        (game_id,),
    ).fetchone()[0]
    if not has_votes:
        conn.execute("DELETE FROM player_counts WHERE game_id=?", (game_id,))
        for label, n, plus, b, r, nr in synth_player_counts(min_p, max_p, best, rec):
            conn.execute(
                """INSERT INTO player_counts(
                   game_id, numplayers, player_n, plus, best_votes, rec_votes, not_rec_votes)
                   VALUES(?,?,?,?,?,?,?)""",
                (game_id, label, n, plus, b, r, nr),
            )

    if own:
        conn.execute(
            "INSERT OR IGNORE INTO owners(game_id, person_id) VALUES(?,?)",
            (game_id, me_id),
        )
    else:
        conn.execute(
            "DELETE FROM owners WHERE game_id=? AND person_id=?",
            (game_id, me_id),
        )
    if wishlist or want_buy:
        conn.execute(
            "INSERT OR IGNORE INTO shelf_items(shelf_id, game_id, notes, added_at) VALUES(?,?,?,?)",
            (wish_id, game_id, "", now),
        )
    else:
        conn.execute(
            "DELETE FROM shelf_items WHERE shelf_id=? AND game_id=?",
            (wish_id, game_id),
        )
    if want_play:
        conn.execute(
            "INSERT OR IGNORE INTO shelf_items(shelf_id, game_id, notes, added_at) VALUES(?,?,?,?)",
            (to_play_id, game_id, "", now),
        )
    else:
        conn.execute(
            "DELETE FROM shelf_items WHERE shelf_id=? AND game_id=?",
            (to_play_id, game_id),
        )


def import_csv(conn: sqlite3.Connection) -> tuple[int, int, int]:
    me_id, wish_id, to_play_id = ensure_seed(conn)
    # CSV is source of truth for *my* flags — drop prior Me ownership /
    # wishlist rows so demo leftovers don't linger as "owned".
    conn.execute("DELETE FROM owners WHERE person_id=?", (me_id,))
    conn.execute(
        "DELETE FROM shelf_items WHERE shelf_id IN (?, ?)", (wish_id, to_play_id)
    )
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    # Dedupe by objectid, preferring own=1 over wishlist-only duplicates.
    by_id: dict[int, dict] = {}
    for row in rows:
        oid = parse_int(row.get("objectid"))
        if oid <= 0:
            continue
        prev = by_id.get(oid)
        if prev is None or (row.get("own") == "1" and prev.get("own") != "1"):
            by_id[oid] = row
        elif prev is not None:
            # Merge flags.
            for flag in ("own", "wishlist", "wanttobuy", "wanttoplay"):
                if row.get(flag) == "1":
                    prev[flag] = "1"
            if (row.get("comment") or "").strip() and not (prev.get("comment") or "").strip():
                prev["comment"] = row["comment"]

    for row in by_id.values():
        upsert_game_from_csv(conn, row, me_id, wish_id, to_play_id)
    conn.commit()
    owned = conn.execute("SELECT COUNT(*) FROM owners").fetchone()[0]
    games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    return len(by_id), games, owned


def http_get(url: str, token: str) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body


def parse_thing_batch(xml_text: str) -> list[dict]:
    out = []
    root = ET.fromstring(xml_text)
    for item in root.findall("item"):
        bgg_id = parse_int(item.get("id", "0"))
        name = ""
        for n in item.findall("name"):
            if n.get("type") == "primary":
                name = n.get("value") or ""
                break
        if not name:
            n = item.find("name")
            name = (n.get("value") if n is not None else "") or f"#{bgg_id}"
        thumb = (item.findtext("thumbnail") or "").strip()
        image = (item.findtext("image") or "").strip()
        year = parse_int(
            (item.find("yearpublished").get("value") if item.find("yearpublished") is not None else "0")
        )
        min_p = parse_int(
            (item.find("minplayers").get("value") if item.find("minplayers") is not None else "0")
        )
        max_p = parse_int(
            (item.find("maxplayers").get("value") if item.find("maxplayers") is not None else "0")
        )
        playtime = parse_int(
            (item.find("playingtime").get("value") if item.find("playingtime") is not None else "0")
        )
        min_pt = parse_int(
            (item.find("minplaytime").get("value") if item.find("minplaytime") is not None else "0")
        )
        max_pt = parse_int(
            (item.find("maxplaytime").get("value") if item.find("maxplaytime") is not None else "0")
        )
        min_age = parse_int(
            (item.find("minage").get("value") if item.find("minage") is not None else "0")
        )

        rating_avg = rating_bayes = weight = 0.0
        users_rated = rank = 0
        stats = item.find("statistics")
        if stats is not None:
            ratings = stats.find("ratings")
            if ratings is not None:
                rating_avg = parse_float(
                    (ratings.find("average").get("value") if ratings.find("average") is not None else "0")
                )
                rating_bayes = parse_float(
                    (
                        ratings.find("bayesaverage").get("value")
                        if ratings.find("bayesaverage") is not None
                        else "0"
                    )
                )
                users_rated = parse_int(
                    (
                        ratings.find("usersrated").get("value")
                        if ratings.find("usersrated") is not None
                        else "0"
                    )
                )
                weight = parse_float(
                    (
                        ratings.find("averageweight").get("value")
                        if ratings.find("averageweight") is not None
                        else "0"
                    )
                )
                ranks = ratings.find("ranks")
                if ranks is not None:
                    for r in ranks.findall("rank"):
                        if r.get("name") == "boardgame":
                            rank = parse_int(r.get("value"), 0)
                            break

        counts = []
        poll_votes = 0
        best_with = recommended_with = ""
        for poll in item.findall("poll"):
            if poll.get("name") != "suggested_numplayers":
                continue
            for results in poll.findall("results"):
                label = results.get("numplayers") or ""
                if not label:
                    continue
                plus = 1 if label.endswith("+") else 0
                n = parse_int(label.rstrip("+"))
                votes = {"Best": 0, "Recommended": 0, "Not Recommended": 0}
                for res in results.findall("result"):
                    votes[res.get("value") or ""] = parse_int(res.get("numvotes"))
                b, r, nr = votes["Best"], votes["Recommended"], votes["Not Recommended"]
                poll_votes += b + r + nr
                counts.append((label, n, plus, b, r, nr))
        for summary in item.findall("poll-summary"):
            if summary.get("name") != "suggested_numplayers":
                continue
            for res in summary.findall("result"):
                if res.get("name") == "bestwith":
                    best_with = res.get("value") or ""
                if res.get("name") in ("recommmendedwith", "recommendedwith"):
                    recommended_with = res.get("value") or ""

        out.append(
            {
                "bgg_id": bgg_id,
                "name": name,
                "year": year,
                "min_players": min_p,
                "max_players": max_p,
                "playtime": playtime,
                "min_playtime": min_pt,
                "max_playtime": max_pt,
                "min_age": min_age,
                "thumbnail_url": thumb,
                "image_url": image,
                "rating_average": rating_avg,
                "rating_bayes": rating_bayes,
                "rating_rank": rank,
                "users_rated": users_rated,
                "weight": weight,
                "poll_votes": poll_votes,
                "best_with": best_with,
                "recommended_with": recommended_with,
                "counts": counts,
            }
        )
    return out


def apply_thing(conn: sqlite3.Connection, thing: dict) -> None:
    row = conn.execute(
        "SELECT id FROM games WHERE bgg_id=?", (thing["bgg_id"],)
    ).fetchone()
    if row is None:
        return
    game_id = row[0]
    now = int(time.time())
    conn.execute(
        """UPDATE games SET name=?, year=?, min_players=?, max_players=?,
           playtime=?, min_playtime=?, max_playtime=?, min_age=?,
           thumbnail_url=?, image_url=?,
           rating_average=?, rating_bayes=?, rating_rank=?, users_rated=?, weight=?,
           poll_votes=?, best_with=?, recommended_with=?, bgg_synced_at=?
           WHERE id=?""",
        (
            thing["name"],
            thing["year"],
            thing["min_players"],
            thing["max_players"],
            thing["playtime"],
            thing["min_playtime"],
            thing["max_playtime"],
            thing["min_age"],
            thing["thumbnail_url"],
            thing["image_url"],
            thing["rating_average"],
            thing["rating_bayes"],
            thing["rating_rank"],
            thing["users_rated"],
            thing["weight"],
            thing["poll_votes"],
            thing["best_with"],
            thing["recommended_with"],
            now,
            game_id,
        ),
    )
    if thing["counts"]:
        conn.execute("DELETE FROM player_counts WHERE game_id=?", (game_id,))
        for label, n, plus, b, r, nr in thing["counts"]:
            conn.execute(
                """INSERT INTO player_counts(
                   game_id, numplayers, player_n, plus, best_votes, rec_votes, not_rec_votes)
                   VALUES(?,?,?,?,?,?,?)""",
                (game_id, label, n, plus, b, r, nr),
            )


def fetch_things(conn: sqlite3.Connection, token: str) -> int:
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT bgg_id FROM games WHERE bgg_id IS NOT NULL ORDER BY bgg_id"
        )
    ]
    updated = 0
    for i in range(0, len(ids), BATCH):
        chunk = ids[i : i + BATCH]
        url = f"{THING_URL}?id={','.join(map(str, chunk))}&stats=1"
        for attempt in range(6):
            code, body = http_get(url, token)
            if code == 202:
                time.sleep(2.0 + attempt)
                continue
            if code == 401:
                raise SystemExit(
                    "BGG 401 Unauthorized — put a Bearer token in "
                    f"{TOKEN_PATH} (from https://boardgamegeek.com/applications)"
                )
            if code != 200:
                print(f"  batch {chunk[0]}… HTTP {code}: {body[:120]}", file=sys.stderr)
                break
            things = parse_thing_batch(body)
            for t in things:
                apply_thing(conn, t)
                updated += 1
            conn.commit()
            print(f"  fetched {len(things)} things ({i + len(chunk)}/{len(ids)})")
            break
        time.sleep(GAP_S)
    return updated


def geekitem_urls(bgg_id: int):
    """Public Geekdo JSON — cover URLs without a BGG Bearer token."""
    url = f"{GEEKITEM_URL}?objectid={bgg_id}&objecttype=thing"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        print(f"  geekitem {bgg_id}: HTTP {e.code}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  geekitem {bgg_id}: {e}", file=sys.stderr)
        return None

    item = data.get("item") or {}
    images = item.get("images") or {}
    # Prefer portrait itemrep for cards; fall back to preview / original.
    thumb = (
        (item.get("imageurl") or "").strip()
        or (images.get("previewthumb") or "").strip()
        or (images.get("thumb") or "").strip()
    )
    image = (
        (images.get("original") or "").strip()
        or (item.get("imageurl@2x") or "").strip()
        or thumb
    )
    if not thumb and not image:
        return None
    return thumb or image, image or thumb


def fill_missing_thumbs(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """SELECT id, bgg_id, name FROM games
           WHERE bgg_id IS NOT NULL
             AND (thumbnail_url IS NULL OR thumbnail_url = '')
           ORDER BY bgg_id"""
    ).fetchall()
    updated = 0
    for i, row in enumerate(rows, 1):
        game_id, bgg_id, name = row[0], row[1], row[2]
        got = geekitem_urls(int(bgg_id))
        if got:
            thumb, image = got
            conn.execute(
                "UPDATE games SET thumbnail_url=?, image_url=? WHERE id=?",
                (thumb, image, game_id),
            )
            updated += 1
            if updated % 10 == 0 or i == len(rows):
                conn.commit()
                print(f"  thumbs {updated}/{len(rows)} ({name})")
        time.sleep(GEEK_GAP_S)
    conn.commit()
    return updated


def main() -> None:
    if not CSV_PATH.is_file():
        raise SystemExit(f"missing {CSV_PATH}")
    conn = open_db()
    unique, games, owned = import_csv(conn)
    print(f"CSV import: {unique} unique games → db has {games} games, {owned} ownership rows")

    token = load_token()
    if token:
        print("Fetching BGG thing metadata + thumbnails…")
        n = fetch_things(conn, token)
        print(f"Updated {n} games from BGG XML")
    else:
        print(
            f"No BGG token — skipping XML polls.\n"
            f"  (optional) token → {TOKEN_PATH}"
        )

    missing = conn.execute(
        "SELECT COUNT(*) FROM games WHERE thumbnail_url IS NULL OR thumbnail_url = ''"
    ).fetchone()[0]
    if missing:
        print(f"Fetching {missing} missing covers from Geekdo…")
        n = fill_missing_thumbs(conn)
        print(f"Filled {n} thumbnail URLs from Geekdo")
    thumbs = conn.execute(
        "SELECT COUNT(*) FROM games WHERE thumbnail_url != ''"
    ).fetchone()[0]
    print(f"Done — {thumbs} games have thumbnails")


if __name__ == "__main__":
    main()
