//! Text scanning: tokenisation, word boundaries, sentence and paragraph splits.
//!
//! The Python original leaned on `re` for all of this. Hand-written scanners
//! are the better trade here: no engine to pull in, the intent of each rule is
//! visible, and the one genuinely intricate pattern — the key/value extractor
//! in [`crate::indexer`] — reads as what it actually is, a small parser.

pub const TOKEN_CHARS_EXTRA: [char; 5] = ['_', '.', ':', '/', '-'];

pub const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "what",
    "which", "who", "whom", "this", "these", "those", "do", "does", "did", "not", "you", "your",
    "we", "our", "they", "their", "he", "she", "his", "her", "i", "me", "my", "but", "if", "then",
    "than", "so",
];

/// Words that describe the *form* of an answer rather than its content.
/// Scoring on them drags every query toward whichever line happens to contain
/// "message" or "exactly", so they are stripped alongside stopwords.
pub const INSTRUCTION_WORDS: &[&str] = &[
    "quote", "exactly", "exact", "verbatim", "literal", "tell", "show", "give", "me", "please",
    "line", "text", "word", "words", "message", "wording",
];

pub fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || TOKEN_CHARS_EXTRA.contains(&c)
}

pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Lowercased runs of token characters — the equivalent of `[A-Za-z0-9_.:/-]+`.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if is_token_char(c) {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn skipped(token: &str) -> bool {
    STOPWORDS.contains(&token) || INSTRUCTION_WORDS.contains(&token)
}

/// Tokens worth scoring on: no stopwords, no instruction words, no single chars.
pub fn content_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| t.chars().count() > 1 && !skipped(t))
        .collect()
}

/// Byte offset of `needle` in `hay`, matched on word boundaries.
/// Both sides are expected to be lowercase already.
pub fn find_word(hay: &str, needle: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(rel) = hay[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let before_ok = start == 0 || !hay[..start].chars().next_back().is_some_and(is_word_char);
        let after_ok = end == hay.len() || !hay[end..].chars().next().is_some_and(is_word_char);
        if before_ok && after_ok {
            return Some(start);
        }
        from = start + needle.chars().next().map_or(1, char::len_utf8);
        if from >= hay.len() {
            break;
        }
    }
    None
}

pub fn contains_word(hay: &str, needle: &str) -> bool {
    find_word(hay, needle).is_some()
}

/// True when any of the phrases appears on word boundaries.
pub fn contains_any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| contains_word(hay, n))
}

/// Half-open byte ranges of blank-line-separated paragraphs.
///
/// Mirrors `\n\s*\n` with greedy backtracking: a run of whitespace containing
/// at least two newlines is a separator, and the separator ends after its last
/// newline.
pub fn paragraph_ranges(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'\n' {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut last_newline = None;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            if bytes[j] == b'\n' {
                last_newline = Some(j);
            }
            j += 1;
        }
        match last_newline {
            Some(end) => {
                if i > pos {
                    out.push((pos, i));
                }
                pos = end + 1;
                i = end + 1;
            }
            None => i = j.max(i + 1),
        }
    }
    if pos < text.len() {
        out.push((pos, text.len()));
    }
    out
}

/// Half-open byte ranges of sentences within `text`, split after `.`/`!`/`?`
/// followed by whitespace, and at newlines.
pub fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        let terminator = matches!(c, b'.' | b'!' | b'?')
            && i + 1 < bytes.len()
            && (bytes[i + 1] as char).is_whitespace();
        if terminator || c == b'\n' {
            let end = if c == b'\n' { i } else { i + 1 };
            if end > start {
                out.push((start, end));
            }
            let mut j = end;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            start = j;
            i = j;
            continue;
        }
        i += 1;
    }
    if start < text.len() {
        out.push((start, text.len()));
    }
    out
}

/// Trim, then strip one layer of matching quotes.
pub fn strip_quotes(value: &str) -> &str {
    let value = value.trim();
    for (open, close) in [('"', '"'), ('\u{201c}', '\u{201d}'), ('\'', '\'')] {
        let mut chars = value.chars();
        if chars.next() == Some(open) && value.ends_with(close) && value.chars().count() > 1 {
            let start = open.len_utf8();
            let end = value.len() - close.len_utf8();
            if start <= end {
                return value[start..end].trim();
            }
        }
    }
    value
}

/// Clip to `max_chars`, appending an ellipsis when anything was removed.
pub fn clip(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let cut: String = value.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{}\u{2026}", cut.trim_end())
}
