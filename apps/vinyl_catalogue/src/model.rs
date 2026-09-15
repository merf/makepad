//! One catalogue row. The grid and the SQLite file share this shape.

#[derive(Clone, Debug, Default)]
pub struct Record {
    pub id: i64,
    pub record_number: i64,
    pub raw_transcript: String,
    pub artist: String,
    pub title: String,
    pub label: String,
    pub catalogue_number: String,
    pub notes: String,
    pub discogs_master_id: String,
    pub discogs_master_url: String,
    pub discogs_release_id: String,
    pub discogs_url: String,
    pub thumb_path: String,
    /// Empty means display VG (or VG+ fallback).
    pub condition: String,
    /// JSON object of Discogs grade → {currency, value}.
    pub price_suggestions: String,
    pub display_price: f64,
    pub num_for_sale: i64,
    pub status: String,
    pub created_at: i64,
    pub updated_at: i64,
}

impl Record {
    pub fn blank(record_number: i64, now: i64) -> Self {
        Self {
            record_number,
            status: "captured".to_string(),
            created_at: now,
            updated_at: now,
            ..Self::default()
        }
    }

    /// Price shown in the grid: assigned condition, else VG, else VG+.
    pub fn shown_price(&self) -> Option<f64> {
        if self.display_price > 0.0 {
            return Some(self.display_price);
        }
        price_for_condition(&self.price_suggestions, &self.condition)
    }

    pub fn shown_price_text(&self) -> String {
        match self.shown_price() {
            // No suggestions JSON ⇒ we only have marketplace floor ("From £X" on the site).
            Some(v) if self.price_suggestions.trim().is_empty() => format!("from £{v:.0}"),
            Some(v) => format!("£{v:.0}"),
            None => String::new(),
        }
    }
}

/// Fields the LLM (or the comma/`next` fallback) is allowed to fill.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Extracted {
    pub artist: String,
    pub title: String,
    pub label: String,
    pub catalogue_number: String,
    pub notes: String,
}

impl Extracted {
    pub fn is_empty(&self) -> bool {
        self.artist.trim().is_empty()
            && self.title.trim().is_empty()
            && self.label.trim().is_empty()
            && self.catalogue_number.trim().is_empty()
            && self.notes.trim().is_empty()
    }
}

/// Pick a grade from a stored suggestions JSON blob.
pub fn price_for_condition(suggestions_json: &str, condition: &str) -> Option<f64> {
    let value = makepad_strict_json::parse_depth(suggestions_json.trim().as_bytes(), 8).ok()?;
    let want = normalize_condition(condition);
    let keys: &[&str] = if want.is_empty() {
        &[
            "Very Good (VG)",
            "VG",
            "Very Good Plus (VG+)",
            "VG+",
            "Near Mint (NM or M-)",
            "NM",
        ]
    } else {
        match want.as_str() {
            "m" | "mint" => &["Mint (M)", "Mint", "M"],
            "nm" | "nearmint" => &["Near Mint (NM or M-)", "Near Mint", "NM", "M-"],
            "vg+" | "vgplus" => &["Very Good Plus (VG+)", "VG+", "Very Good Plus"],
            "vg" => &["Very Good (VG)", "VG", "Very Good"],
            "g+" | "gplus" => &["Good Plus (G+)", "G+", "Good Plus"],
            "g" | "good" => &["Good (G)", "Good", "G"],
            "f" | "fair" => &["Fair (F)", "Fair", "F"],
            "p" | "poor" => &["Poor (P)", "Poor", "P"],
            _ => &[
                "Very Good (VG)",
                "VG",
                "Very Good Plus (VG+)",
                "VG+",
            ],
        }
    };
    for key in keys {
        if let Some(price) = grade_price(&value, key) {
            return Some(price);
        }
    }
    None
}

fn normalize_condition(condition: &str) -> String {
    condition
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '+')
        .collect::<String>()
        .replace("verygoodplus", "vg+")
        .replace("nearmint", "nm")
        .replace("goodplus", "g+")
}

fn grade_price(value: &makepad_strict_json::Value, key: &str) -> Option<f64> {
    let entry = value.get(key)?;
    match entry {
        makepad_strict_json::Value::F64(n) if n.is_finite() && *n > 0.0 => Some(*n),
        makepad_strict_json::Value::Int(n) if *n > 0 => Some(*n as f64),
        makepad_strict_json::Value::Obj(_) => match entry.get("value") {
            Some(makepad_strict_json::Value::F64(n)) if n.is_finite() && *n > 0.0 => Some(*n),
            Some(makepad_strict_json::Value::Int(n)) if *n > 0 => Some(*n as f64),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_vg() {
        let json = r#"{"Near Mint (NM or M-)":{"currency":"GBP","value":70.0},"Very Good Plus (VG+)":{"currency":"GBP","value":55.0},"Very Good (VG)":{"currency":"GBP","value":40.0}}"#;
        assert_eq!(price_for_condition(json, ""), Some(40.0));
        assert_eq!(price_for_condition(json, "NM"), Some(70.0));
        assert_eq!(price_for_condition(json, "vg+"), Some(55.0));
    }
}
