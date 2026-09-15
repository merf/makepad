//! Turn a spoken batch into structured rows without inventing fields.
//!
//! The LLM path (when Qwen is present) should call `emit_records`. This
//! module also understands a JSON array and a naive `next`/comma split so
//! the catalogue works with no model on disk.

use crate::model::Extracted;

pub fn naive_split(transcript: &str) -> Vec<Extracted> {
    let chunks: Vec<&str> = split_on_next(transcript);
    chunks
        .into_iter()
        .filter_map(|chunk| {
            let extracted = fields_from_chunk(chunk);
            (!extracted.is_empty()).then_some(extracted)
        })
        .collect()
}

/// One spoken utterance → one row (first `next` chunk, or the whole line).
pub fn one_record(transcript: &str) -> Option<Extracted> {
    naive_split(transcript).into_iter().next()
}

fn split_on_next(transcript: &str) -> Vec<&str> {
    let lower = transcript.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = transcript.as_bytes();
    let mut i = 0;
    while i + 4 <= lower.len() {
        if is_next_token(&lower, i) {
            let chunk = transcript[start..i].trim();
            if !chunk.is_empty() {
                out.push(chunk);
            }
            i += 4;
            while i < bytes.len() && (bytes[i] == b'.' || bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
                i += 1;
            }
            start = i;
            continue;
        }
        i += 1;
    }
    let tail = transcript[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    if out.is_empty() && !transcript.trim().is_empty() {
        out.push(transcript.trim());
    }
    out
}

fn is_next_token(lower: &str, i: usize) -> bool {
    let rest = &lower[i..];
    let Some(after) = rest.strip_prefix("next") else {
        return false;
    };
    let before_ok = i == 0 || !lower.as_bytes()[i - 1].is_ascii_alphanumeric();
    let after_ok = after.is_empty() || !after.as_bytes()[0].is_ascii_alphanumeric();
    before_ok && after_ok
}

fn fields_from_chunk(chunk: &str) -> Extracted {
    let cleaned = chunk.trim().trim_end_matches('.').trim();
    let parts: Vec<String> = cleaned
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    match parts.len() {
        0 => Extracted::default(),
        1 => Extracted { artist: parts[0].clone(), ..Extracted::default() },
        2 => Extracted { artist: parts[0].clone(), title: parts[1].clone(), ..Extracted::default() },
        3 => Extracted {
            artist: parts[0].clone(),
            title: parts[1].clone(),
            label: parts[2].clone(),
            ..Extracted::default()
        },
        4 => Extracted {
            artist: parts[0].clone(),
            title: parts[1].clone(),
            label: parts[2].clone(),
            catalogue_number: parts[3].clone(),
            ..Extracted::default()
        },
        _ => Extracted {
            artist: parts[0].clone(),
            title: parts[1].clone(),
            label: parts[2].clone(),
            catalogue_number: parts[3].clone(),
            notes: parts[4..].join(", "),
        },
    }
}

/// Pull a records array out of a tool-arg list or a free-form model reply.
pub fn from_llm_payload(tool_args: &[(String, String)], text: &str) -> Vec<Extracted> {
    for (key, value) in tool_args {
        if key == "records" {
            if let Some(rows) = parse_records_json(value) {
                return rows;
            }
        }
    }
    if tool_args.iter().any(|(k, _)| k == "artist" || k == "title") {
        let one = extracted_from_pairs(tool_args);
        if !one.is_empty() {
            return vec![one];
        }
    }
    if let Some(rows) = parse_records_json(text) {
        return rows;
    }
    if let Some(blob) = extract_json_blob(text) {
        if let Some(rows) = parse_records_json(&blob) {
            return rows;
        }
    }
    Vec::new()
}

fn extracted_from_pairs(args: &[(String, String)]) -> Extracted {
    let mut out = Extracted::default();
    for (key, value) in args {
        let v = value.trim().trim_matches('"').to_string();
        match key.as_str() {
            "artist" => out.artist = v,
            "title" => out.title = v,
            "label" => out.label = v,
            "catalogue_number" | "catno" | "catalog" => out.catalogue_number = v,
            "notes" => out.notes = v,
            _ => {}
        }
    }
    out
}

fn strip_fences(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if let Some(rest) = text.strip_prefix("```json") {
        text = rest.to_string();
    } else if let Some(rest) = text.strip_prefix("```") {
        text = rest.to_string();
    }
    if let Some(stripped) = text.strip_suffix("```") {
        text = stripped.to_string();
    }
    text.trim().to_string()
}

fn extract_json_blob(text: &str) -> Option<String> {
    let cleaned = strip_fences(text);
    if let Some(start) = cleaned.find('[') {
        if let Some(end) = cleaned.rfind(']') {
            if end > start {
                return Some(cleaned[start..=end].to_string());
            }
        }
    }
    if let Some(start) = cleaned.find('{') {
        if let Some(end) = cleaned.rfind('}') {
            if end > start {
                return Some(cleaned[start..=end].to_string());
            }
        }
    }
    None
}

fn parse_records_json(raw: &str) -> Option<Vec<Extracted>> {
    #[cfg(feature = "native")]
    {
        let cleaned = strip_fences(raw);
        parse_records_json_strict(&cleaned)
            .or_else(|| extract_json_blob(&cleaned).and_then(|blob| parse_records_json_strict(&blob)))
    }
    #[cfg(not(feature = "native"))]
    {
        let _ = raw;
        None
    }
}

#[cfg(feature = "native")]
fn parse_records_json_strict(raw: &str) -> Option<Vec<Extracted>> {
    let trimmed = raw.trim();
    let value = makepad_strict_json::parse(trimmed.as_bytes()).ok()?;
    records_from_json(&value)
}

#[cfg(feature = "native")]
fn json_text(value: &makepad_strict_json::Value) -> String {
    match value {
        makepad_strict_json::Value::Str(s) => s.trim().to_string(),
        makepad_strict_json::Value::Int(n) => n.to_string(),
        _ => String::new(),
    }
}

#[cfg(feature = "native")]
fn extracted_from_json_obj(value: &makepad_strict_json::Value) -> Extracted {
    let get = |key: &str| value.get(key).map(json_text).unwrap_or_default();
    let catalogue = get("catalogue_number");
    Extracted {
        artist: get("artist"),
        title: get("title"),
        label: get("label"),
        catalogue_number: if catalogue.is_empty() { get("catno") } else { catalogue },
        notes: get("notes"),
    }
}

#[cfg(feature = "native")]
fn records_from_json(value: &makepad_strict_json::Value) -> Option<Vec<Extracted>> {
    use makepad_strict_json::Value;
    match value {
        Value::Arr(items) => {
            let mut out = Vec::new();
            for item in items {
                let extracted = match item {
                    Value::Obj(_) => extracted_from_json_obj(item),
                    _ => continue,
                };
                if !extracted.is_empty() {
                    out.push(extracted);
                }
            }
            Some(out)
        }
        Value::Obj(_) => {
            if let Some(inner) = value.get("records").and_then(|v| v.as_arr()) {
                return records_from_json(&Value::Arr(inner.to_vec()));
            }
            let one = extracted_from_json_obj(value);
            if one.is_empty() { Some(Vec::new()) } else { Some(vec![one]) }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_next_and_commas() {
        let rows = naive_split(
            "Surgeon, Magneze, Tresor, T-179, next. Jeff Mills, The Bells, Purpose Maker, PM-001, next",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].artist, "Surgeon");
        assert_eq!(rows[0].title, "Magneze");
        assert_eq!(rows[0].label, "Tresor");
        assert_eq!(rows[0].catalogue_number, "T-179");
        assert_eq!(rows[1].artist, "Jeff Mills");
        assert_eq!(rows[1].catalogue_number, "PM-001");
    }

    #[test]
    fn missing_fields_stay_empty() {
        let rows = naive_split("Underground Resistance, Jupiter Jazz, next");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].artist, "Underground Resistance");
        assert_eq!(rows[0].title, "Jupiter Jazz");
        assert_eq!(rows[0].label, "");
        assert_eq!(rows[0].catalogue_number, "");
    }

    #[test]
    fn does_not_split_next_inside_a_word() {
        let rows = naive_split("The Nextgen, Track, Label, CAT");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].artist, "The Nextgen");
    }

    #[test]
    fn parses_json_array_from_model_text() {
        let rows = from_llm_payload(
            &[],
            "```json\n[{\"artist\":\"Surgeon\",\"title\":\"Magneze\",\"label\":\"Tresor\",\"catalogue_number\":\"T-179\",\"notes\":\"\"}]\n```",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].artist, "Surgeon");
        assert_eq!(rows[0].title, "Magneze");
        assert_eq!(rows[0].catalogue_number, "T-179");
    }
}
