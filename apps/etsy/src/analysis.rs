//! Filter listings and compute analysis metrics (pure; no UI).

use crate::model::Listing;
use crate::normalise;
use makepad_widgets::DataPoint;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug)]
pub struct Filters {
    pub physical_only: bool,
    pub exclude_personalised: bool,
    pub exclude_digital: bool,
    pub singles_only: bool,
    pub packs_only: bool,
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            physical_only: true,
            exclude_personalised: false,
            exclude_digital: true,
            singles_only: false,
            packs_only: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistMetric {
    ItemPrice,
    Delivered,
    PricePerCard,
}

impl HistMetric {
    #[allow(dead_code)]
    pub fn label(self) -> &'static str {
        match self {
            HistMetric::ItemPrice => "Item price",
            HistMetric::Delivered => "Delivered",
            HistMetric::PricePerCard => "£/card",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Percentiles {
    pub p25: Option<f64>,
    pub median: Option<f64>,
    pub p75: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct PackRow {
    pub pack_size: i64,
    pub count: usize,
    pub median_price: Option<f64>,
    pub median_delivered: Option<f64>,
    pub median_per_card: Option<f64>,
    pub free_postage_pct: f64,
    pub median_paid_postage: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct PriceBand {
    pub label: String,
    pub count: usize,
    pub median_reviews: Option<f64>,
    pub max_reviews: Option<i64>,
    pub free_postage_pct: f64,
}

#[derive(Clone, Debug)]
pub struct TopListing {
    pub title: String,
    pub shop_name: String,
    pub delivered_price: Option<f64>,
    pub review_count: i64,
}

#[derive(Clone, Debug, Default)]
pub struct DataQuality {
    pub total: usize,
    pub pct_price: f64,
    pub pct_postage: f64,
    pub pct_reviews: f64,
    pub pct_pack: f64,
    pub unknown_postage: usize,
    pub unknown_pack: usize,
    pub digital: usize,
    pub personalised: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub filtered_count: usize,
    pub physical_count: usize,
    pub shop_count: usize,
    pub price: Percentiles,
    pub delivered: Percentiles,
    pub postage: Percentiles,
    pub price_per_card: Percentiles,
    pub delivered_per_card: Percentiles,
    pub reviews: Percentiles,
    pub pack_rows: Vec<PackRow>,
    pub price_bands: Vec<PriceBand>,
    pub top_by_reviews: Vec<TopListing>,
    pub quality: DataQuality,
    pub hist_points: Vec<DataPoint>,
    pub scatter_points: Vec<DataPoint>,
}

pub fn filter_listings<'a>(listings: &'a [Listing], filters: &Filters) -> Vec<&'a Listing> {
    listings
        .iter()
        .filter(|l| passes(l, filters))
        .collect()
}

fn passes(l: &Listing, f: &Filters) -> bool {
    if f.physical_only && !normalise::is_physical(l) {
        return false;
    }
    if f.exclude_digital && l.is_digital {
        return false;
    }
    if f.exclude_personalised && l.is_personalised {
        return false;
    }
    if f.singles_only {
        match l.pack_size {
            Some(1) => {}
            _ => return false,
        }
    }
    if f.packs_only {
        match l.pack_size {
            Some(n) if n > 1 => {}
            _ => return false,
        }
    }
    true
}

pub fn build_report(listings: &[Listing], filters: &Filters, hist: HistMetric) -> Report {
    let filtered = filter_listings(listings, filters);
    let physical_count = filtered
        .iter()
        .filter(|l| normalise::is_physical(l))
        .count();
    let mut shops = HashSet::new();
    for l in &filtered {
        if !l.shop_name.is_empty() {
            shops.insert(l.shop_name.to_ascii_lowercase());
        }
    }

    let prices: Vec<f64> = filtered.iter().filter_map(|l| l.price).collect();
    let delivered: Vec<f64> = filtered.iter().filter_map(|l| l.delivered_price).collect();
    let postage: Vec<f64> = filtered
        .iter()
        .filter(|l| l.postage_known)
        .filter_map(|l| l.postage)
        .collect();
    let ppc: Vec<f64> = filtered.iter().filter_map(|l| l.price_per_card).collect();
    let dppc: Vec<f64> = filtered
        .iter()
        .filter_map(|l| l.delivered_price_per_card)
        .collect();
    let reviews: Vec<f64> = filtered
        .iter()
        .filter_map(|l| l.review_count.map(|n| n as f64))
        .collect();

    let pack_rows = pack_breakdown(&filtered);
    let price_bands = delivered_bands(&filtered);
    let top_by_reviews = top_reviews(&filtered, 10);
    let quality = data_quality(listings);
    let hist_points = histogram(&filtered, hist);
    let scatter_points = scatter_delivered_vs_reviews(&filtered);

    Report {
        filtered_count: filtered.len(),
        physical_count,
        shop_count: shops.len(),
        price: percentiles(&prices),
        delivered: percentiles(&delivered),
        postage: percentiles(&postage),
        price_per_card: percentiles(&ppc),
        delivered_per_card: percentiles(&dppc),
        reviews: percentiles(&reviews),
        pack_rows,
        price_bands,
        top_by_reviews,
        quality,
        hist_points,
        scatter_points,
    }
}

pub fn percentiles(values: &[f64]) -> Percentiles {
    if values.is_empty() {
        return Percentiles::default();
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Percentiles {
        p25: Some(quantile_sorted(&v, 0.25)),
        median: Some(quantile_sorted(&v, 0.5)),
        p75: Some(quantile_sorted(&v, 0.75)),
    }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

fn pack_breakdown(listings: &[&Listing]) -> Vec<PackRow> {
    let mut groups: BTreeMap<i64, Vec<&Listing>> = BTreeMap::new();
    for l in listings {
        if let Some(n) = l.pack_size {
            groups.entry(n).or_default().push(*l);
        }
    }
    groups
        .into_iter()
        .map(|(pack_size, rows)| {
            let prices: Vec<f64> = rows.iter().filter_map(|l| l.price).collect();
            let delivered: Vec<f64> = rows.iter().filter_map(|l| l.delivered_price).collect();
            let per_card: Vec<f64> = rows.iter().filter_map(|l| l.price_per_card).collect();
            let with_post: Vec<&&Listing> = rows.iter().filter(|l| l.postage_known).collect();
            let free = with_post
                .iter()
                .filter(|l| l.postage == Some(0.0))
                .count();
            let free_pct = if with_post.is_empty() {
                0.0
            } else {
                100.0 * free as f64 / with_post.len() as f64
            };
            let paid: Vec<f64> = with_post
                .iter()
                .filter_map(|l| l.postage)
                .filter(|p| *p > 0.0)
                .collect();
            PackRow {
                pack_size,
                count: rows.len(),
                median_price: percentiles(&prices).median,
                median_delivered: percentiles(&delivered).median,
                median_per_card: percentiles(&per_card).median,
                free_postage_pct: free_pct,
                median_paid_postage: percentiles(&paid).median,
            }
        })
        .collect()
}

fn delivered_bands(listings: &[&Listing]) -> Vec<PriceBand> {
    const EDGES: &[(f64, f64, &str)] = &[
        (3.0, 4.0, "£3–£4"),
        (4.0, 5.0, "£4–£5"),
        (5.0, 6.0, "£5–£6"),
        (6.0, 7.0, "£6–£7"),
    ];
    let mut bands = Vec::new();
    for &(lo, hi, label) in EDGES {
        bands.push(band_for(listings, lo, hi, label, false));
    }
    bands.push(band_for(listings, 7.0, f64::INFINITY, "£7+", true));
    bands
}

fn band_for(
    listings: &[&Listing],
    lo: f64,
    hi: f64,
    label: &str,
    open_ended: bool,
) -> PriceBand {
    let rows: Vec<&&Listing> = listings
        .iter()
        .filter(|l| {
            let Some(d) = l.delivered_price else {
                return false;
            };
            if open_ended {
                d >= lo
            } else {
                d >= lo && d < hi
            }
        })
        .collect();
    let reviews: Vec<f64> = rows
        .iter()
        .filter_map(|l| l.review_count.map(|n| n as f64))
        .collect();
    let max_reviews = rows.iter().filter_map(|l| l.review_count).max();
    let with_post: Vec<&&&Listing> = rows.iter().filter(|l| l.postage_known).collect();
    let free = with_post
        .iter()
        .filter(|l| l.postage == Some(0.0))
        .count();
    let free_pct = if with_post.is_empty() {
        0.0
    } else {
        100.0 * free as f64 / with_post.len() as f64
    };
    PriceBand {
        label: label.into(),
        count: rows.len(),
        median_reviews: percentiles(&reviews).median,
        max_reviews,
        free_postage_pct: free_pct,
    }
}

fn top_reviews(listings: &[&Listing], n: usize) -> Vec<TopListing> {
    let mut rows: Vec<&Listing> = listings
        .iter()
        .copied()
        .filter(|l| l.review_count.is_some())
        .collect();
    rows.sort_by(|a, b| {
        b.review_count
            .cmp(&a.review_count)
            .then_with(|| a.title.cmp(&b.title))
    });
    rows.into_iter()
        .take(n)
        .map(|l| TopListing {
            title: l.title.clone(),
            shop_name: l.shop_name.clone(),
            delivered_price: l.delivered_price,
            review_count: l.review_count.unwrap_or(0),
        })
        .collect()
}

fn data_quality(all: &[Listing]) -> DataQuality {
    let total = all.len();
    if total == 0 {
        return DataQuality::default();
    }
    let with_price = all.iter().filter(|l| l.price.is_some()).count();
    let with_post = all.iter().filter(|l| l.postage_known).count();
    let with_reviews = all.iter().filter(|l| l.review_count.is_some()).count();
    let with_pack = all.iter().filter(|l| l.pack_size.is_some()).count();
    DataQuality {
        total,
        pct_price: 100.0 * with_price as f64 / total as f64,
        pct_postage: 100.0 * with_post as f64 / total as f64,
        pct_reviews: 100.0 * with_reviews as f64 / total as f64,
        pct_pack: 100.0 * with_pack as f64 / total as f64,
        unknown_postage: total - with_post,
        unknown_pack: total - with_pack,
        digital: all.iter().filter(|l| l.is_digital).count(),
        personalised: all.iter().filter(|l| l.is_personalised).count(),
    }
}

fn histogram(listings: &[&Listing], metric: HistMetric) -> Vec<DataPoint> {
    let values: Vec<f64> = listings
        .iter()
        .filter_map(|l| match metric {
            HistMetric::ItemPrice => l.price,
            HistMetric::Delivered => l.delivered_price,
            HistMetric::PricePerCard => l.price_per_card,
        })
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    if values.is_empty() {
        return Vec::new();
    }
    let bin_width = 1.0;
    let max_v = values.iter().cloned().fold(0.0_f64, f64::max);
    let n_bins = ((max_v / bin_width).floor() as usize + 1).min(40).max(1);
    let mut counts = vec![0usize; n_bins];
    for v in values {
        let idx = ((v / bin_width).floor() as usize).min(n_bins - 1);
        counts[idx] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, c)| DataPoint {
            x: i as f64 * bin_width + bin_width * 0.5,
            y: c as f64,
        })
        .collect()
}

fn scatter_delivered_vs_reviews(listings: &[&Listing]) -> Vec<DataPoint> {
    listings
        .iter()
        .filter_map(|l| match (l.delivered_price, l.review_count) {
            (Some(x), Some(y)) if x.is_finite() => Some(DataPoint {
                x,
                y: y as f64,
            }),
            _ => None,
        })
        .collect()
}

pub fn format_money(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("£{n:.2}"),
        None => "—".into(),
    }
}

pub fn format_pct(v: f64) -> String {
    format!("{v:.0}%")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Listing;
    use crate::normalise;

    fn sample_store() -> Vec<Listing> {
        let mut rows = Vec::new();
        let specs = [
            ("Single A", 3.5, Some(0.0), true, Some(1), 100, false, false),
            ("Single B", 4.0, Some(1.8), true, Some(1), 50, false, false),
            ("Pack of 4 funny", 9.95, Some(0.0), true, Some(4), 20, false, false),
            ("Digital PDF download", 2.0, None, false, Some(1), 5, true, false),
            ("Personalised name card", 4.5, Some(1.5), true, Some(1), 10, false, true),
        ];
        for (i, (title, price, post, known, pack, reviews, dig, per)) in specs.into_iter().enumerate()
        {
            let mut l = Listing {
                id: i as i64 + 1,
                title: title.into(),
                raw_text: title.into(),
                shop_name: format!("Shop{i}"),
                price: Some(price),
                postage: post,
                postage_known: known,
                pack_size: pack,
                review_count: Some(reviews),
                is_digital: dig,
                is_personalised: per,
                currency: "GBP".into(),
                ..Listing::default()
            };
            // Force derived via apply using title cues where pack not preset.
            if title.contains("Pack of 4") {
                l.pack_size = None;
            }
            normalise::apply(&mut l);
            if dig {
                l.is_digital = true;
            }
            if per {
                l.is_personalised = true;
            }
            rows.push(l);
        }
        rows
    }

    #[test]
    fn default_filters_drop_digital() {
        let rows = sample_store();
        let report = build_report(&rows, &Filters::default(), HistMetric::ItemPrice);
        assert!(report.filtered_count >= 3);
        assert!(report.filtered_count < rows.len());
        assert!(report.shop_count >= 1);
        assert!(report.price.median.is_some());
    }

    #[test]
    fn pack_rows_include_four() {
        let rows = sample_store();
        let report = build_report(&rows, &Filters::default(), HistMetric::ItemPrice);
        assert!(report.pack_rows.iter().any(|r| r.pack_size == 4 && r.count >= 1));
    }

    #[test]
    fn percentiles_median() {
        let p = percentiles(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(p.median, Some(3.0));
        assert_eq!(p.p25, Some(2.0));
        assert_eq!(p.p75, Some(4.0));
    }
}
