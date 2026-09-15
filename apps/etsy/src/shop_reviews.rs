//! Shop review-count parsing (mirrors export_etsy_results.js).
//! Search cards show shop popularity, not listing reviews.

/// Parse "1,234", "1.2k", "12k", "1.5m".
pub fn parse_count_token(token: &str) -> Option<i64> {
    let s: String = token
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ',')
        .collect::<String>()
        .to_ascii_lowercase();
    if let Some(rest) = s.strip_suffix('k') {
        let n: f64 = rest.parse().ok()?;
        return Some((n * 1000.0).round() as i64);
    }
    if let Some(rest) = s.strip_suffix('m') {
        let n: f64 = rest.parse().ok()?;
        return Some((n * 1_000_000.0).round() as i64);
    }
    if !s.chars().all(|c| c.is_ascii_digit()) || s.is_empty() {
        return None;
    }
    s.parse().ok()
}

fn strip_rating_phrases(text: &str) -> String {
    let mut out = text.to_string();
    // crude case-insensitive replacements for the patterns we care about
    let lower = out.to_ascii_lowercase();
    // remove "4.8 out of 5 stars" / "1 out of 5"
    let mut cleaned = String::new();
    let bytes = lower.as_bytes();
    let orig = out.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // try match digit sequence then " out of 5"
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            let rest = &lower[i..];
            if rest.starts_with(" out of 5") || rest.starts_with(" / 5") {
                let skip = if rest.starts_with(" out of 5") { 9 } else { 4 };
                i += skip;
                if lower[i..].starts_with(" stars") {
                    i += 6;
                } else if lower[i..].starts_with(" star") {
                    i += 5;
                }
                cleaned.push(' ');
                continue;
            }
            cleaned.push_str(std::str::from_utf8(&orig[start..i]).unwrap_or(""));
            continue;
        }
        cleaned.push(orig[i] as char);
        i += 1;
    }
    out = cleaned;
    out = out.replace("Star seller", " ").replace("star seller", " ");
    out
}

/// Extract shop review count from card text. Never uses star-rating digits as the count.
pub fn parse_shop_review_count(text: &str, shop_name: &str) -> Option<i64> {
    if text.is_empty() {
        return None;
    }

    if let Some(n) = find_reviews_word_count(text) {
        return Some(n);
    }

    let stripped = strip_rating_phrases(text);

    if let Some(caps) = find_paren_km(&stripped) {
        if let Some(n) = parse_count_token(&caps) {
            return Some(n);
        }
    }

    if !shop_name.is_empty() {
        if let Some(inner) = find_shop_paren(&stripped, shop_name) {
            if let Some(n) = parse_count_token(&inner) {
                return Some(n);
            }
        }
    }

    for caps in find_all_parens(&stripped) {
        if caps.contains('%') {
            continue;
        }
        if let Some(n) = parse_count_token(&caps) {
            if n >= 6 {
                return Some(n);
            }
        }
    }

    // bare 1.2k after stripping rating
    if let Some(n) = find_bare_k(&stripped) {
        if n >= 100 {
            return Some(n);
        }
    }

    None
}

fn find_reviews_word_count(text: &str) -> Option<i64> {
    let lower = text.to_ascii_lowercase();
    for (needle, _) in [("reviews", 7usize), ("review", 6)] {
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(needle) {
            let at = search_from + rel;
            // word boundary-ish
            let after_ok = lower
                .as_bytes()
                .get(at + needle.len())
                .map(|c| !c.is_ascii_alphabetic())
                .unwrap_or(true);
            if !after_ok {
                search_from = at + 1;
                continue;
            }
            let before = text[..at].trim_end();
            // optional "from "
            let before = before.strip_suffix("from").unwrap_or(before).trim_end();
            let bytes = before.as_bytes();
            let mut i = before.len();
            while i > 0 {
                let c = bytes[i - 1] as char;
                if c.is_ascii_digit()
                    || c == ','
                    || c == '.'
                    || c == 'k'
                    || c == 'K'
                    || c == 'm'
                    || c == 'M'
                    || c.is_whitespace()
                {
                    i -= 1;
                } else {
                    break;
                }
            }
            let token = before[i..].trim();
            if let Some(n) = parse_count_token(token) {
                return Some(n);
            }
            search_from = at + 1;
        }
    }
    None
}

fn find_bare_k(text: &str) -> Option<i64> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] == b'k' || bytes[i] == b'm') {
                let end = i + 1;
                let token = &text[start..end];
                if let Some(n) = parse_count_token(token) {
                    return Some(n);
                }
            }
            continue;
        }
        i += 1;
    }
    None
}

/// Prefer explicit JSON count; heal star-rating false positives from raw_text.
pub fn resolve_shop_review_count(
    json_count: Option<i64>,
    raw_text: &str,
    shop_name: &str,
) -> Option<i64> {
    let from_raw = parse_shop_review_count(raw_text, shop_name);
    match (json_count, from_raw) {
        (Some(j), Some(r)) if looks_like_star_rating_digit(j) && r != j => Some(r),
        (Some(j), Some(r)) if j <= 5 && r >= 10 => Some(r),
        (Some(j), _) => Some(j),
        (None, r) => r,
    }
}

fn looks_like_star_rating_digit(n: i64) -> bool {
    (1..=5).contains(&n)
}

fn find_paren_km(text: &str) -> Option<String> {
    for caps in find_all_parens(text) {
        let lower = caps.to_ascii_lowercase();
        if lower.ends_with('k') || lower.ends_with('m') {
            return Some(caps);
        }
    }
    None
}

fn find_shop_paren(text: &str, shop: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let shop_l = shop.to_ascii_lowercase();
    let mut from = 0;
    while let Some(rel) = lower[from..].find(&shop_l) {
        let at = from + rel;
        let after = text[at + shop.len()..].trim_start();
        if let Some(inner) = after.strip_prefix('(').and_then(|s| s.split(')').next()) {
            return Some(inner.to_string());
        }
        from = at + 1;
    }
    None
}

fn find_all_parens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b')' {
                j += 1;
            }
            if j < bytes.len() {
                let inner = &text[start..j];
                if inner.chars().all(|c| {
                    c.is_ascii_digit()
                        || c == ','
                        || c == '.'
                        || c == 'k'
                        || c == 'K'
                        || c == 'm'
                        || c == 'M'
                        || c.is_whitespace()
                }) && !inner.is_empty()
                {
                    out.push(inner.to_string());
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_from_fixture_json() {
        assert_eq!(
            parse_shop_review_count(
                "Funny Birthday Card £3.50 FREE delivery CardCraftUK Rated 4.8 out of 5 stars (1,284)",
                "CardCraftUK"
            ),
            Some(1284)
        );
        assert_eq!(
            parse_shop_review_count(
                "Dad Joke Card £2.95 PunnyPrints Rating: 1 out of 5 stars (412)",
                "PunnyPrints"
            ),
            Some(412)
        );
        assert_eq!(
            parse_shop_review_count("4.8 out of 5 stars, 1,284 Reviews", ""),
            Some(1284)
        );
        assert_eq!(
            parse_shop_review_count("Rated 4.9 out of 5 stars from 15.4k reviews", ""),
            Some(15400)
        );
        assert_eq!(parse_shop_review_count("1284 reviews", ""), Some(1284));
        assert_eq!(
            parse_shop_review_count("ShopName (1.2k)", "ShopName"),
            Some(1200)
        );
        assert_eq!(
            parse_shop_review_count("Pack of (1) greeting card £3.00 QuietPaper", "QuietPaper"),
            None
        );
        assert_eq!(
            parse_shop_review_count("Star seller CardCraftUK Free delivery", "CardCraftUK"),
            None
        );
        assert_eq!(
            parse_shop_review_count("£3.50 FREE delivery BundleCards (88)", "BundleCards"),
            Some(88)
        );
    }

    #[test]
    fn heals_star_rating_json_count() {
        let raw = "CardCraftUK Rated 4.8 out of 5 stars (1,284)";
        assert_eq!(
            resolve_shop_review_count(Some(1), raw, "CardCraftUK"),
            Some(1284)
        );
    }
}
