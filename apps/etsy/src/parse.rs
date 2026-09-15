//! Parse bookmarklet JSON into [`ExportPayload`].

use crate::model::{ExportPayload, ParsedListing};
use crate::shop_reviews;
use makepad_strict_json::Value;

pub fn parse_export(text: &str) -> Result<ExportPayload, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("paste is empty".into());
    }
    let value = makepad_strict_json::parse(trimmed.as_bytes())
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = match &value {
        Value::Obj(_) => &value,
        _ => return Err("JSON root must be an object".into()),
    };

    let schema_version = require_i64(obj, "schema_version")?;
    if schema_version != 1 && schema_version != 2 {
        return Err(format!(
            "unsupported schema_version {schema_version} (want 1 or 2)"
        ));
    }

    let listings_val = obj
        .get("listings")
        .ok_or_else(|| "missing listings array".to_string())?;
    let items = listings_val
        .as_arr()
        .ok_or_else(|| "listings must be an array".to_string())?;

    let mut listings = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            Value::Obj(_) => listings.push(parse_listing(item, index)?),
            Value::Null => continue,
            _ => {
                return Err(format!("listings[{index}] must be an object"));
            }
        }
    }

    let page = opt_i64(obj, "page").filter(|p| *p > 0).unwrap_or(1);

    Ok(ExportPayload {
        schema_version,
        source: opt_str(obj, "source").unwrap_or_else(|| "etsy".into()),
        source_url: opt_str(obj, "source_url").unwrap_or_default(),
        query: opt_str(obj, "query").unwrap_or_default(),
        page,
        exported_at: opt_str(obj, "exported_at").unwrap_or_default(),
        listings,
        raw_json: trimmed.to_string(),
    })
}

fn parse_listing(obj: &Value, index: usize) -> Result<ParsedListing, String> {
    let title = opt_str(obj, "title").unwrap_or_default();
    let url = opt_str(obj, "url").unwrap_or_default();
    if url.trim().is_empty() && title.trim().is_empty() {
        return Err(format!("listings[{index}] has neither title nor url"));
    }

    let price_text = opt_str(obj, "price_text").unwrap_or_default();
    let mut price = opt_f64(obj, "price");
    if price.is_none() {
        price = parse_money(&price_text);
    }

    let postage_text = opt_str(obj, "postage_text").unwrap_or_default();
    let (postage, postage_known) = parse_postage(&postage_text);

    let currency = opt_str(obj, "currency")
        .or_else(|| currency_from_price_text(&price_text))
        .unwrap_or_default();

    let shop_name = opt_str(obj, "shop_name").unwrap_or_default();
    let shop_url = opt_str(obj, "shop_url").unwrap_or_default();
    let raw_text = opt_str(obj, "raw_text").unwrap_or_default();

    let json_reviews = opt_i64(obj, "shop_review_count").or_else(|| opt_i64(obj, "review_count"));
    let review_count =
        shop_reviews::resolve_shop_review_count(json_reviews, &raw_text, &shop_name);

    Ok(ParsedListing {
        title,
        url,
        price_text,
        price,
        currency,
        postage_text,
        postage,
        postage_known,
        review_count,
        shop_name,
        shop_url,
        image_url: opt_str(obj, "image_url").unwrap_or_default(),
        raw_text,
    })
}

fn require_i64(obj: &Value, key: &str) -> Result<i64, String> {
    match obj.get(key) {
        Some(Value::Int(n)) => Ok(*n),
        Some(Value::F64(n)) if n.is_finite() && n.fract() == 0.0 => Ok(*n as i64),
        Some(_) => Err(format!("{key} must be an integer")),
        None => Err(format!("missing {key}")),
    }
}

fn opt_str(obj: &Value, key: &str) -> Option<String> {
    match obj.get(key)? {
        Value::Null => None,
        Value::Str(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        Value::Int(n) => Some(n.to_string()),
        Value::F64(n) if n.is_finite() => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn opt_f64(obj: &Value, key: &str) -> Option<f64> {
    match obj.get(key)? {
        Value::Null => None,
        Value::F64(n) if n.is_finite() => Some(*n),
        Value::Int(n) => Some(*n as f64),
        Value::Str(s) => parse_money(s),
        _ => None,
    }
}

fn opt_i64(obj: &Value, key: &str) -> Option<i64> {
    match obj.get(key)? {
        Value::Null => None,
        Value::Int(n) => Some(*n),
        Value::F64(n) if n.is_finite() && n.fract() == 0.0 => Some(*n as i64),
        Value::Str(s) => {
            let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                None
            } else {
                digits.parse().ok()
            }
        }
        _ => None,
    }
}

/// Parse displayed money like `£3.50`, `GBP 12.99`, `$4.00`.
pub fn parse_money(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut num = String::new();
    let mut seen_dot = false;
    for c in trimmed.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else if c == '.' && !seen_dot {
            num.push('.');
            seen_dot = true;
        } else if c == ',' {
            continue;
        } else if !num.is_empty() {
            break;
        }
    }
    if num.is_empty() || num == "." {
        return None;
    }
    let value: f64 = num.parse().ok()?;
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn currency_from_price_text(text: &str) -> Option<String> {
    let t = text.trim();
    if t.contains('£') || t.to_ascii_uppercase().contains("GBP") {
        Some("GBP".into())
    } else if t.contains('$') || t.to_ascii_uppercase().contains("USD") {
        Some("USD".into())
    } else if t.contains('€') || t.to_ascii_uppercase().contains("EUR") {
        Some("EUR".into())
    } else {
        None
    }
}

/// Free delivery → postage 0 known; `£1.80` → known; sale/original price noise → unknown.
pub fn parse_postage(text: &str) -> (Option<f64>, bool) {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return (None, false);
    }

    let mentions_delivery = lower.contains("delivery")
        || lower.contains("shipping")
        || lower.contains("postage");
    let mentions_sale_noise = lower.contains("original price")
        || lower.contains("sale price")
        || lower.contains("% off")
        || lower.contains("percent off");

    if lower.contains("free") && (mentions_delivery || !mentions_sale_noise) {
        if mentions_delivery || !mentions_sale_noise {
            return (Some(0.0), true);
        }
    }

    if mentions_sale_noise {
        return (None, false);
    }

    if let Some(amount) = parse_money(text) {
        return (Some(amount), true);
    }
    (None, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../testdata/export_sample.json");

    #[test]
    fn parses_valid_export() {
        let payload = parse_export(FIXTURE).expect("fixture");
        assert_eq!(payload.schema_version, 1);
        assert_eq!(payload.query, "funny birthday card");
        assert_eq!(payload.page, 1);
        assert_eq!(payload.listings.len(), 6);
        let single = &payload.listings[0];
        assert_eq!(single.price, Some(3.50));
        assert_eq!(single.currency, "GBP");
        assert!(single.postage_known);
        assert_eq!(single.postage, Some(0.0));
        let paid = &payload.listings[1];
        assert_eq!(paid.postage, Some(1.80));
        assert_eq!(paid.review_count, Some(412));
        let pack = &payload.listings[2];
        assert!(pack.title.to_lowercase().contains("pack of 4"));
        let missing = &payload.listings[4];
        assert!(missing.review_count.is_none());
        assert!(!missing.postage_known);
        let usd = &payload.listings[5];
        assert_eq!(usd.currency, "USD");
        assert_eq!(usd.price, Some(4.99));
    }

    #[test]
    fn accepts_schema_v2_with_page_and_shop_review_count() {
        let payload = parse_export(
            r#"{
              "schema_version": 2,
              "source": "etsy",
              "query": "funny card",
              "page": 3,
              "listings": [{
                "title": "A card",
                "url": "https://etsy.com/listing/1",
                "shop_name": "ShopA",
                "shop_url": "https://etsy.com/shop/ShopA",
                "shop_review_count": 99,
                "raw_text": "ShopA (99)"
              }]
            }"#,
        )
        .unwrap();
        assert_eq!(payload.schema_version, 2);
        assert_eq!(payload.page, 3);
        assert_eq!(payload.listings[0].review_count, Some(99));
        assert_eq!(payload.listings[0].shop_url, "https://etsy.com/shop/ShopA");
    }

    #[test]
    fn heals_bad_review_from_raw_text() {
        let payload = parse_export(
            r#"{
              "schema_version": 1,
              "listings": [{
                "title": "Card",
                "url": "https://etsy.com/listing/9",
                "shop_name": "CardCraftUK",
                "review_count": 1,
                "raw_text": "CardCraftUK Rated 4.8 out of 5 stars (1,284)"
              }]
            }"#,
        )
        .unwrap();
        assert_eq!(payload.listings[0].review_count, Some(1284));
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_export("").is_err());
        assert!(parse_export("   ").is_err());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_export("{not json").is_err());
    }

    #[test]
    fn rejects_wrong_schema() {
        let err = parse_export(r#"{"schema_version":99,"listings":[]}"#).unwrap_err();
        assert!(err.contains("schema_version"));
    }

    #[test]
    fn accepts_empty_listings() {
        let payload = parse_export(
            r#"{"schema_version":1,"source":"etsy","listings":[],"query":"x"}"#,
        )
        .unwrap();
        assert!(payload.listings.is_empty());
        assert_eq!(payload.query, "x");
    }

    #[test]
    fn missing_fields_still_import() {
        let payload = parse_export(
            r#"{"schema_version":1,"listings":[{"title":"A card","url":"https://etsy.com/listing/1"}]}"#,
        )
        .unwrap();
        assert_eq!(payload.listings.len(), 1);
        assert!(payload.listings[0].price.is_none());
        assert!(!payload.listings[0].postage_known);
    }

    #[test]
    fn parse_money_examples() {
        assert_eq!(parse_money("£3.50"), Some(3.50));
        assert_eq!(parse_money("£12.99"), Some(12.99));
        assert_eq!(parse_money("GBP 1.80"), Some(1.80));
        assert_eq!(parse_money("$4.00"), Some(4.0));
        assert_eq!(parse_money(""), None);
    }

    #[test]
    fn parse_postage_free_and_paid() {
        assert_eq!(parse_postage("FREE delivery"), (Some(0.0), true));
        assert_eq!(parse_postage("£1.80"), (Some(1.80), true));
        assert_eq!(parse_postage(""), (None, false));
        assert_eq!(parse_postage("see listing"), (None, false));
        assert_eq!(
            parse_postage("£4.50 Original Price £4.50 (25% off)"),
            (None, false)
        );
        assert_eq!(
            parse_postage("Sale Price £3.37 Free UK delivery"),
            (Some(0.0), true)
        );
    }
}
