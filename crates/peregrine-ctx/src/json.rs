//! A minimal JSON reader/writer for persistence.
//!
//! Objects keep insertion order so a saved store is byte-stable across runs —
//! which matters when the file is meant to be diffable and the runtime claims
//! to be deterministic.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn str(value: impl Into<String>) -> Json {
        Json::Str(value.into())
    }

    pub fn num(value: impl Into<f64>) -> Json {
        Json::Num(value.into())
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| n as u64)
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|n| n as usize)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn string_list(&self) -> Vec<String> {
        self.as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|i| i.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    /// The encoding that gets hashed.
    ///
    /// `to_json_string` preserves insertion order, which is stable *per run*
    /// but is a property of how the value was built. A digest has to be a
    /// property of the value itself, so the canonical form sorts object keys,
    /// drops repeated keys (keeping the first, which is the one [`Json::get`]
    /// returns), and writes non-finite numbers as `null` rather than emitting
    /// the invalid JSON tokens `NaN` and `inf`.
    ///
    /// Everything in the container is addressed by the digest of this form:
    /// re-serialising a loaded object must reproduce its id, or the store
    /// would quarantine its own contents on the next scrub.
    pub fn to_canonical_string(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Json::Num(n) if !n.is_finite() => out.push_str("null"),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_canonical(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                let mut sorted: Vec<&(String, Json)> = pairs.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                sorted.dedup_by(|a, b| a.0 == b.0);
                out.push('{');
                for (i, (key, value)) in sorted.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write_canonical(out);
                }
                out.push('}');
            }
            other => other.write(out),
        }
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(input: &str) -> Result<Json, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0usize;
    let value = parse_value(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos < chars.len() {
        return Err(format!("trailing input at char {pos}"));
    }
    Ok(value)
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_value(chars: &[char], pos: &mut usize) -> Result<Json, String> {
    skip_ws(chars, pos);
    match chars.get(*pos) {
        None => Err("unexpected end of input".to_string()),
        Some('{') => parse_object(chars, pos),
        Some('[') => parse_array(chars, pos),
        Some('"') => parse_string(chars, pos).map(Json::Str),
        Some('t') => expect(chars, pos, "true").map(|_| Json::Bool(true)),
        Some('f') => expect(chars, pos, "false").map(|_| Json::Bool(false)),
        Some('n') => expect(chars, pos, "null").map(|_| Json::Null),
        Some(_) => parse_number(chars, pos),
    }
}

fn expect(chars: &[char], pos: &mut usize, word: &str) -> Result<(), String> {
    for expected in word.chars() {
        if chars.get(*pos) != Some(&expected) {
            return Err(format!("expected {word} at char {pos}"));
        }
        *pos += 1;
    }
    Ok(())
}

fn parse_object(chars: &[char], pos: &mut usize) -> Result<Json, String> {
    *pos += 1; // '{'
    let mut pairs = Vec::new();
    skip_ws(chars, pos);
    if chars.get(*pos) == Some(&'}') {
        *pos += 1;
        return Ok(Json::Obj(pairs));
    }
    loop {
        skip_ws(chars, pos);
        let key = parse_string(chars, pos)?;
        skip_ws(chars, pos);
        if chars.get(*pos) != Some(&':') {
            return Err(format!("expected ':' at char {pos}"));
        }
        *pos += 1;
        let value = parse_value(chars, pos)?;
        pairs.push((key, value));
        skip_ws(chars, pos);
        match chars.get(*pos) {
            Some(',') => *pos += 1,
            Some('}') => {
                *pos += 1;
                return Ok(Json::Obj(pairs));
            }
            _ => return Err(format!("expected ',' or '}}' at char {pos}")),
        }
    }
}

fn parse_array(chars: &[char], pos: &mut usize) -> Result<Json, String> {
    *pos += 1; // '['
    let mut items = Vec::new();
    skip_ws(chars, pos);
    if chars.get(*pos) == Some(&']') {
        *pos += 1;
        return Ok(Json::Arr(items));
    }
    loop {
        items.push(parse_value(chars, pos)?);
        skip_ws(chars, pos);
        match chars.get(*pos) {
            Some(',') => *pos += 1,
            Some(']') => {
                *pos += 1;
                return Ok(Json::Arr(items));
            }
            _ => return Err(format!("expected ',' or ']' at char {pos}")),
        }
    }
}

fn parse_string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    if chars.get(*pos) != Some(&'"') {
        return Err(format!("expected string at char {pos}"));
    }
    *pos += 1;
    let mut out = String::new();
    while let Some(&c) = chars.get(*pos) {
        *pos += 1;
        match c {
            '"' => return Ok(out),
            '\\' => {
                let escape = chars.get(*pos).copied().ok_or("unterminated escape")?;
                *pos += 1;
                match escape {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let hex: String = chars
                            .get(*pos..*pos + 4)
                            .ok_or("short \\u")?
                            .iter()
                            .collect();
                        *pos += 4;
                        let code = u32::from_str_radix(&hex, 16).map_err(|e| e.to_string())?;
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                    }
                    other => return Err(format!("bad escape \\{other}")),
                }
            }
            c => out.push(c),
        }
    }
    Err("unterminated string".to_string())
}

fn parse_number(chars: &[char], pos: &mut usize) -> Result<Json, String> {
    let start = *pos;
    if chars.get(*pos) == Some(&'-') {
        *pos += 1;
    }
    while let Some(c) = chars.get(*pos) {
        if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-') {
            *pos += 1;
        } else {
            break;
        }
    }
    let text: String = chars[start..*pos].iter().collect();
    text.parse::<f64>()
        .map(Json::Num)
        .map_err(|_| format!("bad number {text:?}"))
}
