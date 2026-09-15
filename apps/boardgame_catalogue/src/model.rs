//! In-memory catalogue — screens read this; the SQLite file is the format.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default)]
pub struct Person {
    pub id: i64,
    pub name: String,
    pub sort_order: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Platform {
    pub id: i64,
    pub slug: String,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct Shelf {
    pub id: i64,
    pub slug: String,
    pub name: String,
    pub sort_order: i32,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerCount {
    pub numplayers: String,
    pub player_n: i32,
    pub plus: bool,
    pub best_votes: i32,
    pub rec_votes: i32,
    pub not_rec_votes: i32,
}

impl PlayerCount {
    pub fn total_votes(&self) -> i32 {
        self.best_votes + self.rec_votes + self.not_rec_votes
    }

    /// Single 0..1 recommendation for this count — collapses Best/Rec/Not.
    pub fn recommend_score(&self) -> f64 {
        let total = self.total_votes();
        if total <= 0 {
            return 0.0;
        }
        ((self.best_votes as f64) + 0.5 * (self.rec_votes as f64)) / total as f64
    }

    pub fn fit_level(&self) -> FitAtN {
        let total = self.total_votes();
        if total <= 0 {
            return FitAtN::Empty;
        }
        if self.not_rec_votes >= self.best_votes + self.rec_votes {
            FitAtN::NotRecommended
        } else if self.best_votes >= self.rec_votes {
            FitAtN::Best
        } else {
            FitAtN::Recommended
        }
    }

    pub fn parse_numplayers(label: &str) -> (i32, bool) {
        let trimmed = label.trim();
        if let Some(rest) = trimmed.strip_suffix('+') {
            let n = rest.parse::<i32>().unwrap_or(0);
            (n, true)
        } else {
            (trimmed.parse::<i32>().unwrap_or(0), false)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GameLink {
    pub kind: String,
    pub bgg_id: i64,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct GamePlatform {
    pub platform_id: i64,
    pub url: String,
    pub source: String,
}

#[derive(Clone, Debug, Default)]
pub struct ShelfItem {
    pub shelf_id: i64,
    pub notes: String,
    pub added_at: i64,
}

#[derive(Clone, Debug, Default)]
pub struct Game {
    pub id: i64,
    pub bgg_id: Option<i64>,
    pub bgg_type: String,
    pub name: String,
    pub year: i32,
    pub min_players: i32,
    pub max_players: i32,
    pub playtime: i32,
    pub min_playtime: i32,
    pub max_playtime: i32,
    pub min_age: i32,
    pub thumbnail_url: String,
    pub image_url: String,
    pub rating_average: f64,
    pub rating_bayes: f64,
    pub rating_rank: i32,
    pub users_rated: i32,
    pub weight: f64,
    pub poll_votes: i32,
    pub best_with: String,
    pub recommended_with: String,
    pub local_notes: String,
    pub bgg_synced_at: i64,
    pub bgg_payload: String,
    pub player_counts: Vec<PlayerCount>,
    pub links: Vec<GameLink>,
    pub owner_ids: Vec<i64>,
    pub platforms: Vec<GamePlatform>,
    pub shelves: Vec<ShelfItem>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitAtN {
    Best = 0,
    Recommended = 1,
    Empty = 2,
    NotRecommended = 3,
}

impl FitAtN {
    pub fn label(self) -> &'static str {
        match self {
            Self::Best => "Best",
            Self::Recommended => "Recommended",
            Self::Empty => "No votes",
            Self::NotRecommended => "Not recommended",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NightFit {
    pub fit: FitAtN,
    pub box_ok: bool,
    pub best_pct: f64,
    pub poll: Option<PlayerCount>,
}

impl Game {
    pub fn cover_url(&self) -> &str {
        if !self.thumbnail_url.is_empty() {
            &self.thumbnail_url
        } else {
            &self.image_url
        }
    }

    pub fn has_platform(&self, platform_id: i64) -> bool {
        self.platforms.iter().any(|p| p.platform_id == platform_id)
    }

    pub fn on_shelf(&self, shelf_id: i64) -> bool {
        self.shelves.iter().any(|s| s.shelf_id == shelf_id)
    }

    pub fn owned_by_any(&self, people: &HashSet<i64>) -> bool {
        self.owner_ids.iter().any(|id| people.contains(id))
    }

    pub fn resolve_poll_for(&self, n: i32) -> Option<&PlayerCount> {
        if let Some(exact) = self
            .player_counts
            .iter()
            .find(|p| !p.plus && p.player_n == n)
        {
            return Some(exact);
        }
        self.player_counts
            .iter()
            .filter(|p| p.plus && p.player_n <= n)
            .max_by_key(|p| p.player_n)
    }

    pub fn fit_at(&self, n: i32) -> NightFit {
        let box_ok = if self.min_players == 0 || self.max_players == 0 {
            true
        } else {
            self.min_players <= n && n <= self.max_players
        };
        let Some(poll) = self.resolve_poll_for(n).cloned() else {
            return NightFit {
                fit: FitAtN::Empty,
                box_ok,
                best_pct: 0.0,
                poll: None,
            };
        };
        let total = poll.total_votes();
        if total == 0 {
            return NightFit {
                fit: FitAtN::Empty,
                box_ok,
                best_pct: 0.0,
                poll: Some(poll),
            };
        }
        let best_pct = poll.best_votes as f64 / total as f64;
        let fit = if poll.not_rec_votes >= poll.best_votes + poll.rec_votes {
            FitAtN::NotRecommended
        } else if poll.best_votes >= poll.rec_votes {
            FitAtN::Best
        } else {
            FitAtN::Recommended
        };
        NightFit {
            fit,
            box_ok,
            best_pct,
            poll: Some(poll),
        }
    }

    pub fn best_at_summary(&self) -> String {
        let mut bests: Vec<String> = self
            .player_counts
            .iter()
            .filter(|p| {
                let t = p.total_votes();
                t > 0
                    && p.not_rec_votes < p.best_votes + p.rec_votes
                    && p.best_votes >= p.rec_votes
            })
            .map(|p| p.numplayers.clone())
            .collect();
        if bests.is_empty() {
            if !self.best_with.is_empty() {
                return self.best_with.clone();
            }
            return String::new();
        }
        bests.sort_by(|a, b| {
            let (an, ap) = PlayerCount::parse_numplayers(a);
            let (bn, bp) = PlayerCount::parse_numplayers(b);
            (ap, an).cmp(&(bp, bn))
        });
        bests.join(", ")
    }
}

#[derive(Clone, Debug, Default)]
pub struct Catalogue {
    pub people: Vec<Person>,
    pub platforms: Vec<Platform>,
    pub shelves: Vec<Shelf>,
    pub games: Vec<Game>,
    pub me_person_id: i64,
    pub settings: HashMap<String, String>,
}

impl Catalogue {
    pub fn person_name(&self, id: i64) -> &str {
        self.people
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.as_str())
            .unwrap_or("?")
    }

    pub fn platform_by_slug(&self, slug: &str) -> Option<&Platform> {
        self.platforms.iter().find(|p| p.slug == slug)
    }

    pub fn shelf_by_slug(&self, slug: &str) -> Option<&Shelf> {
        self.shelves.iter().find(|s| s.slug == slug)
    }

    pub fn game(&self, id: i64) -> Option<&Game> {
        self.games.iter().find(|g| g.id == id)
    }

    pub fn game_mut(&mut self, id: i64) -> Option<&mut Game> {
        self.games.iter_mut().find(|g| g.id == id)
    }
}

#[derive(Clone, Debug)]
pub struct NightFilters {
    pub players: i32,
    pub owner_ids: HashSet<i64>,
    pub include_table: bool,
    pub include_bga: bool,
    pub include_tts: bool,
    pub include_wishlist: bool,
    pub include_empty: bool,
    pub include_not_recommended: bool,
}

impl Default for NightFilters {
    fn default() -> Self {
        Self {
            players: 4,
            owner_ids: HashSet::new(),
            include_table: true,
            include_bga: false,
            include_tts: false,
            include_wishlist: false,
            include_empty: false,
            include_not_recommended: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NightCard {
    pub game_id: i64,
    pub fit: NightFit,
}

pub fn night_cards(cat: &Catalogue, filters: &NightFilters) -> Vec<NightCard> {
    let bga_id = cat.platform_by_slug("bga").map(|p| p.id);
    let tts_id = cat.platform_by_slug("tts").map(|p| p.id);
    let wishlist_id = cat.shelf_by_slug("wishlist").map(|s| s.id);

    let mut out = Vec::new();
    for game in &cat.games {
        let table_ok = filters.include_table && game.owned_by_any(&filters.owner_ids);
        let wish_ok = filters.include_wishlist
            && wishlist_id
                .map(|id| game.on_shelf(id))
                .unwrap_or(false);
        let bga_ok = filters.include_bga
            && bga_id.map(|id| game.has_platform(id)).unwrap_or(false);
        let tts_ok = filters.include_tts
            && tts_id.map(|id| game.has_platform(id)).unwrap_or(false);
        if !(table_ok || wish_ok || bga_ok || tts_ok) {
            continue;
        }
        let fit = game.fit_at(filters.players);
        if !fit.box_ok {
            continue;
        }
        match fit.fit {
            FitAtN::Best | FitAtN::Recommended => {}
            FitAtN::Empty if filters.include_empty => {}
            FitAtN::NotRecommended if filters.include_not_recommended => {}
            _ => continue,
        }
        out.push(NightCard {
            game_id: game.id,
            fit,
        });
    }
    out.sort_by(|a, b| {
        a.fit
            .fit
            .cmp(&b.fit.fit)
            .then_with(|| {
                b.fit
                    .best_pct
                    .partial_cmp(&a.fit.best_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                let ga = cat.game(a.game_id).map(|g| g.rating_bayes).unwrap_or(0.0);
                let gb = cat.game(b.game_id).map(|g| g.rating_bayes).unwrap_or(0.0);
                gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.game_id.cmp(&b.game_id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_game(id: i64, name: &str, best_at_4: bool) -> Game {
        let mut g = Game {
            id,
            name: name.into(),
            min_players: 2,
            max_players: 5,
            rating_bayes: 7.0,
            ..Game::default()
        };
        g.player_counts.push(PlayerCount {
            numplayers: "4".into(),
            player_n: 4,
            plus: false,
            best_votes: if best_at_4 { 10 } else { 1 },
            rec_votes: if best_at_4 { 2 } else { 10 },
            not_rec_votes: 0,
        });
        g
    }

    #[test]
    fn night_prefers_best_and_respects_venues() {
        let mut cat = Catalogue::default();
        cat.platforms.push(Platform {
            id: 1,
            slug: "bga".into(),
            name: "BGA".into(),
        });
        cat.platforms.push(Platform {
            id: 2,
            slug: "tts".into(),
            name: "TTS".into(),
        });
        let mut table = sample_game(1, "Table Game", true);
        table.owner_ids.push(10);
        let mut online = sample_game(2, "BGA Game", true);
        online.platforms.push(GamePlatform {
            platform_id: 1,
            url: String::new(),
            source: "manual".into(),
        });
        cat.games.push(table);
        cat.games.push(online);

        let mut filters = NightFilters::default();
        filters.owner_ids.insert(10);
        filters.include_table = true;
        filters.include_bga = false;
        filters.include_tts = false;
        let cards = night_cards(&cat, &filters);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].game_id, 1);

        filters.include_table = false;
        filters.include_bga = true;
        let cards = night_cards(&cat, &filters);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].game_id, 2);
    }

    #[test]
    fn night_wishlist_toggle_adds_shelf_games() {
        let mut cat = Catalogue::default();
        cat.shelves.push(Shelf {
            id: 7,
            slug: "wishlist".into(),
            name: "Wishlist".into(),
            sort_order: 0,
        });
        let mut owned = sample_game(1, "Owned", true);
        owned.owner_ids.push(10);
        let mut wished = sample_game(2, "Wished", true);
        wished.shelves.push(ShelfItem {
            shelf_id: 7,
            notes: String::new(),
            added_at: 0,
        });
        cat.games.push(owned);
        cat.games.push(wished);

        let mut filters = NightFilters::default();
        filters.owner_ids.insert(10);
        filters.include_table = true;
        assert_eq!(night_cards(&cat, &filters).len(), 1);

        filters.include_wishlist = true;
        let cards = night_cards(&cat, &filters);
        assert_eq!(cards.len(), 2);
    }
}
