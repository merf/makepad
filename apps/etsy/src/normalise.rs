//! Derive pack size, delivered price, flags from title/raw/postage text.

use crate::model::Listing;
use crate::parse::{parse_money, parse_postage};

/// Fill derived fields on a listing (safe to call repeatedly).
pub fn apply(listing: &mut Listing) {
    refine_postage(listing);

    listing.delivered_price = match (listing.price, listing.postage, listing.postage_known) {
        (Some(p), Some(post), true) if p.is_finite() && post.is_finite() => Some(p + post),
        _ => None,
    };

    listing.pack_size = infer_pack_size(&listing.title, &listing.raw_text);
    listing.is_digital = detect_digital(&listing.title, &listing.raw_text);
    listing.is_personalised = detect_personalised(&listing.title, &listing.raw_text);
    listing.is_bundle = match listing.pack_size {
        Some(n) if n > 1 => true,
        _ => detect_bundle_wording(&listing.title, &listing.raw_text),
    };

    listing.price_per_card = match (listing.price, listing.pack_size) {
        (Some(p), Some(n)) if n > 0 && p.is_finite() => Some(p / n as f64),
        _ => None,
    };
    listing.delivered_price_per_card = match (listing.delivered_price, listing.pack_size) {
        (Some(p), Some(n)) if n > 0 && p.is_finite() => Some(p / n as f64),
        _ => None,
    };
}

pub fn is_physical(listing: &Listing) -> bool {
    !listing.is_digital
}

fn refine_postage(listing: &mut Listing) {
    let blob = format!("{} {}", listing.postage_text, listing.raw_text);
    let (from_text, known) = parse_postage(&listing.postage_text);
    if known {
        listing.postage = from_text;
        listing.postage_known = true;
    }

    // Free delivery in raw even when postage_text empty/wrong.
    let lower = blob.to_ascii_lowercase();
    if lower.contains("free")
        && (lower.contains("delivery") || lower.contains("shipping") || lower.contains("postage"))
    {
        listing.postage = Some(0.0);
        listing.postage_known = true;
        return;
    }

    // "£4.95 incl. postage" with item price £3.15 → postage 1.80.
    if let Some(delivered) = find_incl_postage(&blob) {
        if let Some(price) = listing.price {
            if delivered >= price - 0.001 {
                let post = (delivered - price).max(0.0);
                listing.postage = Some(post);
                listing.postage_known = true;
                return;
            }
        }
    }

    // Paid delivery phrase in raw: "£1.80 delivery" / "£1.80 shipping".
    if !listing.postage_known {
        if let Some(amount) = find_paid_delivery(&blob) {
            listing.postage = Some(amount);
            listing.postage_known = true;
        }
    }
}

fn find_paid_delivery(text: &str) -> Option<f64> {
    let lower = text.to_ascii_lowercase();
    for key in ["delivery", "shipping", "postage"] {
        if let Some(idx) = lower.find(key) {
            let before = &text[..idx];
            // Take the nearest money amount just before the keyword.
            if let Some(amount) = last_money_in(before) {
                // Skip if this looks like "incl. postage" (handled elsewhere).
                let window = before.to_ascii_lowercase();
                if window.trim_end().ends_with("incl.") || window.trim_end().ends_with("incl") {
                    continue;
                }
                return Some(amount);
            }
        }
    }
    None
}

fn last_money_in(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        // skip trailing space
        while i > 0 && bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        let end = i;
        while i > 0 {
            let c = bytes[i - 1] as char;
            if c.is_ascii_digit() || c == '.' || c == ',' {
                i -= 1;
            } else {
                break;
            }
        }
        if i < end {
            if let Some(v) = parse_money(&text[i..end]) {
                return Some(v);
            }
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    None
}

fn find_incl_postage(text: &str) -> Option<f64> {
    let lower = text.to_ascii_lowercase();
    let Some(idx) = lower.find("incl") else {
        return None;
    };
    // Walk backward from "incl" for a money amount.
    let before = &text[..idx];
    let bytes = before.as_bytes();
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 {
        let c = bytes[start - 1] as char;
        if c.is_ascii_digit() || c == '.' || c == ',' {
            start -= 1;
        } else {
            break;
        }
    }
    if start == end {
        return None;
    }
    parse_money(&before[start..end])
}

/// Infer pack size only when clear; otherwise unknown.
pub fn infer_pack_size(title: &str, raw: &str) -> Option<i64> {
    let text = format!("{title} {raw}").to_ascii_lowercase();

    if let Some(n) = capture_after(&text, &["pack of ", "set of ", "bundle of ", "multipack of "]) {
        if (2..=50).contains(&n) {
            return Some(n);
        }
    }

    // "4 cards", "10 greeting cards", "6 birthday cards"
    if let Some(n) = capture_cards_count(&text) {
        if (2..=50).contains(&n) {
            return Some(n);
        }
    }

    // Explicit single — not bare "card".
    if text.contains("single card")
        || text.contains("a single ")
        || text.contains("one card")
        || text.contains("(single)")
        || text.split_whitespace().any(|w| w == "single")
            && !text.contains("pack")
            && !text.contains("set of")
            && !text.contains("multipack")
    {
        // "single" alone in title is enough if no pack wording.
        if !text.contains("pack of") && !text.contains("set of") && !text.contains("multipack") {
            return Some(1);
        }
    }

    // Default greeting-card listing with no pack signal → treat as single.
    // Spec: "Do not guess aggressively" / "single -> 1" when clear.
    // Bare product cards without pack words are overwhelmingly singles on Etsy
    // greeting-card search; only assign 1 when title looks like a card and
    // has no plural pack cue.
    if looks_like_single_card(&text) {
        return Some(1);
    }

    None
}

fn looks_like_single_card(text: &str) -> bool {
    if text.contains("pack of")
        || text.contains("set of")
        || text.contains("multipack")
        || text.contains("bundle of")
        || text.contains(" cards")
        || text.contains("greeting cards")
    {
        return false;
    }
    text.contains("card") || text.contains("greeting")
}

fn capture_after(text: &str, prefixes: &[&str]) -> Option<i64> {
    for prefix in prefixes {
        if let Some(pos) = text.find(prefix) {
            let rest = &text[pos + prefix.len()..];
            if let Some(n) = parse_leading_int(rest) {
                return Some(n);
            }
        }
    }
    None
}

fn capture_cards_count(text: &str) -> Option<i64> {
    // Find "N cards" / "N greeting cards"
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let n: i64 = text[start..i].parse().ok()?;
            let rest = text[i..].trim_start();
            if rest.starts_with("greeting cards")
                || rest.starts_with("cards")
                || rest.starts_with("card pack")
                || rest.starts_with("card multipack")
            {
                return Some(n);
            }
        } else {
            i += 1;
        }
    }
    None
}

fn parse_leading_int(s: &str) -> Option<i64> {
    let s = s.trim_start();
    let mut end = 0;
    for c in s.chars() {
        if c.is_ascii_digit() {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    s[..end].parse().ok()
}

pub fn detect_digital(title: &str, raw: &str) -> bool {
    let text = format!("{title} {raw}").to_ascii_lowercase();
    text.contains("instant download")
        || text.contains("digital download")
        || text.contains("digital file")
        || text.contains("printable")
        || text.contains("pdf download")
        || (text.contains("digital") && (text.contains("download") || text.contains("printable")))
        || text.contains("e-card")
        || text.contains("ecard")
}

pub fn detect_personalised(title: &str, raw: &str) -> bool {
    let text = format!("{title} {raw}").to_ascii_lowercase();
    text.contains("personalised")
        || text.contains("personalized")
        || text.contains("custom name")
        || text.contains("add a name")
        || text.contains("with name")
}

fn detect_bundle_wording(title: &str, raw: &str) -> bool {
    let text = format!("{title} {raw}").to_ascii_lowercase();
    text.contains("multipack")
        || text.contains("multi pack")
        || text.contains("card pack")
        || text.contains("card bundle")
        || text.contains("bundle of")
        || text.contains("pack of")
        || text.contains("set of")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Listing;

    fn listing(title: &str, raw: &str, price: Option<f64>, postage_text: &str) -> Listing {
        let (postage, postage_known) = parse_postage(postage_text);
        let mut l = Listing {
            title: title.into(),
            raw_text: raw.into(),
            price,
            postage_text: postage_text.into(),
            postage,
            postage_known,
            ..Listing::default()
        };
        apply(&mut l);
        l
    }

    #[test]
    fn pack_of_four() {
        let l = listing(
            "Funny Birthday Card Bundle Pack of 4",
            "",
            Some(9.95),
            "FREE UK delivery",
        );
        assert_eq!(l.pack_size, Some(4));
        assert!(l.is_bundle);
        assert_eq!(l.postage, Some(0.0));
        assert_eq!(l.delivered_price, Some(9.95));
        assert!((l.price_per_card.unwrap() - 9.95 / 4.0).abs() < 0.001);
    }

    #[test]
    fn single_card_default() {
        let l = listing("Funny Birthday Card for Friend", "", Some(3.5), "£1.80");
        assert_eq!(l.pack_size, Some(1));
        assert!(!l.is_bundle);
        assert_eq!(l.delivered_price, Some(5.3));
    }

    #[test]
    fn personalised_and_digital() {
        let p = listing(
            "Personalised Birthday Card Wife Name",
            "",
            Some(4.5),
            "£1.50",
        );
        assert!(p.is_personalised);
        assert!(!p.is_digital);

        let d = listing(
            "Printable Digital Download Birthday Card PDF",
            "instant download",
            Some(2.0),
            "",
        );
        assert!(d.is_digital);
        assert!(!is_physical(&d));
    }

    #[test]
    fn incl_postage_from_raw() {
        let l = listing(
            "Birthday Jokes Card",
            "£3.15 £4.95 incl. postage Sent from GB",
            Some(3.15),
            "",
        );
        assert!(l.postage_known);
        assert!((l.postage.unwrap() - 1.80).abs() < 0.01);
        assert!((l.delivered_price.unwrap() - 4.95).abs() < 0.01);
    }

    #[test]
    fn rejects_sale_price_as_postage() {
        let l = listing(
            "Sale card",
            "Sale Price £3.37 £4.50 Original Price £4.50 (25% off) Free UK delivery",
            Some(3.37),
            "£4.50 Original Price £4.50 (25% off)",
        );
        assert_eq!(l.postage, Some(0.0));
        assert!(l.postage_known);
    }

    #[test]
    fn n_cards_pattern() {
        let l = listing("Set of 6 funny cards", "6 greeting cards", Some(12.0), "free delivery");
        assert_eq!(l.pack_size, Some(6));
    }

    #[test]
    fn unknown_postage_stays_unknown() {
        let l = listing("Minimal Card", "Minimal Card £3.00 QuietPaper", Some(3.0), "");
        assert!(!l.postage_known);
        assert!(l.delivered_price.is_none());
    }
}
