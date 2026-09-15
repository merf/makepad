//! In-memory shapes for searches and listings.

#![allow(dead_code)] // some fields reserved for later phases

#[derive(Clone, Debug, Default)]
pub struct Search {
    pub id: i64,
    pub source: String,
    pub query: String,
    pub source_url: String,
    pub page: i64,
    pub exported_at: String,
    pub imported_at: i64,
    pub raw_json: String,
}

#[derive(Clone, Debug, Default)]
pub struct Listing {
    pub id: i64,
    pub url: String,
    pub title: String,
    /// LLM-cleaned display title; empty until cleaned. Full SEO title stays in `title`.
    pub title_clean: String,
    pub shop_name: String,
    pub shop_url: String,
    pub price: Option<f64>,
    pub price_text: String,
    pub currency: String,
    pub postage_text: String,
    pub postage: Option<f64>,
    pub postage_known: bool,
    /// Shop review count from the search card (not listing-level reviews).
    pub review_count: Option<i64>,
    pub image_url: String,
    pub raw_text: String,
    pub first_seen: i64,
    pub last_seen: i64,
    /// Search ids this listing belongs to (may be more than one).
    pub search_ids: Vec<i64>,
    /// Queries from those searches, for the grid.
    pub queries: Vec<String>,
    // --- Phase 3 derived ---
    pub delivered_price: Option<f64>,
    pub pack_size: Option<i64>,
    pub price_per_card: Option<f64>,
    pub delivered_price_per_card: Option<f64>,
    pub is_digital: bool,
    pub is_personalised: bool,
    pub is_bundle: bool,
}

/// One listing row as parsed from the bookmarklet JSON (before DB ids).
#[derive(Clone, Debug, Default)]
pub struct ParsedListing {
    pub title: String,
    pub url: String,
    pub price_text: String,
    pub price: Option<f64>,
    pub currency: String,
    pub postage_text: String,
    pub postage: Option<f64>,
    pub postage_known: bool,
    pub review_count: Option<i64>,
    pub shop_name: String,
    pub shop_url: String,
    pub image_url: String,
    pub raw_text: String,
}

/// Validated bookmarklet payload.
#[derive(Clone, Debug, Default)]
pub struct ExportPayload {
    pub schema_version: i64,
    pub source: String,
    pub source_url: String,
    pub query: String,
    pub page: i64,
    pub exported_at: String,
    pub listings: Vec<ParsedListing>,
    pub raw_json: String,
}

#[derive(Clone, Debug, Default)]
pub struct Store {
    pub searches: Vec<Search>,
    pub listings: Vec<Listing>,
}

impl Store {
    pub fn listing_count(&self) -> usize {
        self.listings.len()
    }

    pub fn search_count(&self) -> usize {
        self.searches.len()
    }

    pub fn listings_for_search(&self, search_id: i64) -> Vec<&Listing> {
        self.listings
            .iter()
            .filter(|l| l.search_ids.contains(&search_id))
            .collect()
    }
}

impl Listing {
    pub fn from_parsed(parsed: &ParsedListing, now: i64) -> Listing {
        let mut listing = Listing {
            url: parsed.url.clone(),
            title: parsed.title.clone(),
            shop_name: parsed.shop_name.clone(),
            shop_url: parsed.shop_url.clone(),
            price: parsed.price,
            price_text: parsed.price_text.clone(),
            currency: parsed.currency.clone(),
            postage_text: parsed.postage_text.clone(),
            postage: parsed.postage,
            postage_known: parsed.postage_known,
            review_count: parsed.review_count,
            image_url: parsed.image_url.clone(),
            raw_text: parsed.raw_text.clone(),
            first_seen: now,
            last_seen: now,
            ..Listing::default()
        };
        crate::normalise::apply(&mut listing);
        listing
    }

    pub fn display_title(&self) -> &str {
        if self.title_clean.trim().is_empty() {
            &self.title
        } else {
            &self.title_clean
        }
    }

    pub fn flag_text(&self) -> String {
        let mut flags = String::new();
        if self.is_digital {
            flags.push('D');
        }
        if self.is_personalised {
            flags.push('P');
        }
        if self.is_bundle {
            flags.push('B');
        }
        if flags.is_empty() {
            "—".into()
        } else {
            flags
        }
    }
}
