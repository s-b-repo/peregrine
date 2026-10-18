//! OpenAI tool calling ↔ GLM `<tool_call>` bridge.
//!
//! An OpenAI client (opencode, and every other agent harness) sends tool
//! *schemas* in a `tools` array and expects tool *calls* back in
//! `choices[].message.tool_calls`. GLM-5.2 knows neither shape: it was trained
//! to read tool schemas from the system turn and to emit calls as the markup
//! its tokenizer carries dedicated tokens for —
//!
//! ```text
//! <tool_call>read
//! <arg_key>filePath</arg_key>
//! <arg_value>/etc/hosts</arg_value>
//! </tool_call>
//! ```
//!
//! so this module is the translation in both directions: [`render_preamble`]
//! puts the schemas where the model expects to find them, and [`OutputFilter`]
//! turns what it generates back into OpenAI `tool_calls`.
//!
//! The filter is *incremental* because the streaming path needs it: markup
//! arrives split across tokens, and no partial `<tool_call>` may ever reach the
//! client as visible content — a client that renders half a tag has already
//! shown the user something the model did not say.

use serde::Deserialize;
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dialect {
    #[default]
    Glm,
    Qwen,
}

pub const QWEN_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

/// One entry of the request's `tools` array. `type` is accepted and ignored:
/// every value OpenAI defines here is `"function"`, and rejecting an unknown
/// one would fail a request the model could have served.
#[derive(Deserialize, Clone, Debug)]
pub struct ToolDef {
    pub function: FunctionDef,
}

#[derive(Deserialize, Clone, Debug)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the arguments. Passed through verbatim — the model reads
    /// it, this server never validates against it.
    #[serde(default)]
    pub parameters: Option<Value>,
}

/// A tool call recovered from the model's output.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedCall {
    pub name: String,
    /// Always a JSON *object*, so `arguments` serializes to something a client
    /// can `JSON.parse` even when the model named no arguments at all.
    pub arguments: Value,
}

impl ParsedCall {
    /// The OpenAI wire form. `arguments` is a *string* holding JSON, which is
    /// the shape clients parse — not a nested object.
    pub fn to_openai(&self, index: usize, id: &str) -> Value {
        json!({
            "index": index,
            "id": id,
            "type": "function",
            "function": {
                // `Value::to_string` rather than `serde_json::to_string(..).unwrap_or_else(..)`:
                // serializing a `Value` that is already in memory cannot fail —
                // `Value` cannot hold a non-string map key or a NaN — so the
                // fallback was unreachable code standing in for a missing proof.
                "name": self.name,
                "arguments": self.arguments.to_string(),
            }
        })
    }
}

/// Render the tool schemas into the text the model was trained to read them in.
///
/// Appended to the system turn rather than sent as its own turn: GLM's own
/// template puts tools inside `<|system|>`, and a bare tools turn with no
/// system content trains the model to answer the schema instead of using it.
pub fn render_preamble(tools: &[ToolDef]) -> String {
    let mut s = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        let card = json!({
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description.as_deref().unwrap_or(""),
                "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({})),
            }
        });
        // Every tool renders. An earlier version skipped a card "that will not
        // serialize", but `card` is a `Value` built right here and `Value`
        // serialization is infallible, so that branch could never be taken —
        // it read as a considered failure policy while being dead code.
        s.push_str(&card.to_string());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, output the function name and arguments \
         within the following XML format:\n<tool_call>{function-name}\n\
         <arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n\
         <arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n\
         ...\n</tool_call>",
    );
    s
}

pub fn render_preamble_for(dialect: Dialect, tools: &[ToolDef]) -> String {
    match dialect {
        Dialect::Glm => render_preamble(tools),
        Dialect::Qwen => render_preamble_qwen_with_thinking(tools, false),
    }
}

pub fn render_preamble_for_with_thinking(
    dialect: Dialect,
    tools: &[ToolDef],
    reasoning_enabled: bool,
) -> String {
    if !reasoning_enabled {
        return render_preamble_for(dialect, tools);
    }
    match dialect {
        Dialect::Glm => render_preamble(tools),
        Dialect::Qwen => render_preamble_qwen_with_thinking(tools, true),
    }
}

#[cfg(test)]
pub fn render_preamble_qwen(tools: &[ToolDef]) -> String {
    render_preamble_qwen_with_thinking(tools, false)
}

pub fn render_preamble_qwen_with_thinking(tools: &[ToolDef], reasoning_enabled: bool) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    if reasoning_enabled {
        s.push_str(QWEN_REASONING_INSTRUCTION);
        s.push_str("\n\n");
    }
    s.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for t in tools {
        let card = json!({
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description.as_deref().unwrap_or(""),
                "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({})),
            }
        });
        s.push('\n');
        s.push_str(&card.to_string());
    }
    s.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
    );
    s
}

pub fn render_assistant_calls_for(dialect: Dialect, tool_calls: &[Value]) -> String {
    match dialect {
        Dialect::Glm => render_assistant_calls(tool_calls),
        Dialect::Qwen => render_assistant_calls_qwen(tool_calls),
    }
}

/// Render an assistant turn's tool calls back into GLM markup, so a
/// multi-turn conversation replays to the model in the form it emitted.
pub fn render_assistant_calls(tool_calls: &[Value]) -> String {
    let mut s = String::new();
    for c in tool_calls {
        let f = &c["function"];
        let Some(name) = f["name"].as_str() else { continue };
        s.push_str("\n<tool_call>");
        s.push_str(name);
        s.push('\n');
        // `arguments` is a JSON string by the OpenAI contract; a client that
        // sent an object anyway is accepted rather than dropped.
        let args: Value = match f["arguments"].as_str() {
            Some(raw) => serde_json::from_str(raw).unwrap_or(Value::Null),
            None => f["arguments"].clone(),
        };
        if let Some(obj) = args.as_object() {
            for (k, v) in obj {
                s.push_str("<arg_key>");
                s.push_str(k);
                s.push_str("</arg_key>\n<arg_value>");
                s.push_str(&value_to_text(v));
                s.push_str("</arg_value>\n");
            }
        }
        s.push_str("</tool_call>");
    }
    s
}

pub fn render_assistant_calls_qwen(tool_calls: &[Value]) -> String {
    let mut s = String::new();
    let mut first = true;
    for c in tool_calls {
        let f = c.get("function").unwrap_or(c);
        let Some(name) = f.get("name").and_then(Value::as_str) else { continue };
        if !first {
            s.push('\n');
        }
        first = false;
        s.push_str("<tool_call>\n<function=");
        s.push_str(name);
        s.push_str(">\n");
        let args: Value = match f.get("arguments").and_then(|value| value.as_str()) {
            Some(raw) => serde_json::from_str(raw).unwrap_or(Value::Null),
            None => f.get("arguments").cloned().unwrap_or(Value::Null),
        };
        if let Some(obj) = args.as_object() {
            for (k, v) in obj {
                s.push_str("<parameter=");
                s.push_str(k);
                s.push_str(">\n");
                s.push_str(&value_to_text(v));
                s.push_str("\n</parameter>\n");
            }
        }
        s.push_str("</function>\n</tool_call>");
    }
    s
}

/// Argument values render as their plain text, not as quoted JSON — the model
/// wrote `/etc/hosts`, not `"/etc/hosts"`, and feeding the quoted form back
/// teaches it the wrong shape on the next turn.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        // Infallible for the same reason as `ParsedCall::to_openai` — and the
        // `unwrap_or_default()` this replaces would have rendered the argument
        // as an empty string, which the model would read as a real value.
        other => other.to_string(),
    }
}

/// Markers the filter must never split across a delta boundary.
const MARKERS: [&str; 4] = ["<tool_call>", "</tool_call>", "<think>", "</think>"];

/// Incremental splitter over the model's output: visible text out one side,
/// [`ParsedCall`]s out the other, reasoning dropped.
///
/// Feed it decoded text as it arrives; it holds back any tail that could still
/// turn out to be the start of a marker, so a `<tool_call>` split across three
/// tokens is still recognized and never leaks a fragment to the client.
#[derive(Default)]
pub struct OutputFilter {
    /// Text seen but not yet classified — a possible partial marker.
    pending: String,
    /// Inside `<tool_call>…</tool_call>`: accumulating a call, emitting nothing.
    in_call: bool,
    /// Inside `<think>…</think>`: reasoning, dropped rather than shown.
    in_think: bool,
    call_buf: String,
    calls: Vec<ParsedCall>,
    /// Declared parameter types, so an argument's JSON type comes from the
    /// tool's own schema instead of from what its text happens to look like.
    schemas: SchemaMap,
    dialect: Dialect,
    capture_reasoning: bool,
    reasoning: String,
}

impl OutputFilter {
    /// A filter that types arguments from the request's tool schemas. Pass an
    /// empty slice for a request that declared none — the markup is still
    /// stripped, since a model can emit it whether or not it was asked to.
    #[cfg(test)]
    pub fn with_tools(tools: &[ToolDef]) -> Self {
        Self::with_tools_dialect(tools, Dialect::Glm, false)
    }

    #[cfg(test)]
    pub fn with_qwen_tools(tools: &[ToolDef], capture_reasoning: bool) -> Self {
        Self::with_tools_dialect(tools, Dialect::Qwen, capture_reasoning)
    }

    pub fn with_tools_dialect(tools: &[ToolDef], dialect: Dialect, capture_reasoning: bool) -> Self {
        Self {
            schemas: schema_map(tools),
            dialect,
            capture_reasoning,
            ..Self::default()
        }
    }

    /// Push a decoded chunk; returns the text safe to show the client now.
    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    /// End of generation: flush whatever is held back.
    ///
    /// A `<tool_call>` the model never closed is still parsed here — hitting
    /// the token cap or EOS mid-call is common, and dropping the call would
    /// turn "the agent did something" into "the agent said nothing".
    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    /// Tool calls recovered so far, cleared by the taking.
    pub fn take_calls(&mut self) -> Vec<ParsedCall> {
        std::mem::take(&mut self.calls)
    }

    pub fn take_reasoning(&mut self) -> String {
        std::mem::take(&mut self.reasoning)
    }

    /// The name of the call currently open, before any argument parsing —
    /// the cheap probe (one `find` over the buffered body) that lets a
    /// speculator skip non-hosted tools without ever building a snapshot.
    /// `None` when no `<tool_call>` is open or the body does not yet carry a
    /// name.
    pub fn open_call_name(&self) -> Option<&str> {
        if !self.in_call {
            return None;
        }
        match self.dialect {
            Dialect::Glm => {
                if looks_like_qwen_call(&self.call_buf) {
                    return qwen_call_name(&self.call_buf);
                }
                let cut = self.call_buf.find("<arg_key>").unwrap_or(self.call_buf.len());
                let name = self.call_buf[..cut].lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
                if name.is_empty() || name.contains('<') {
                    return None;
                }
                Some(name)
            }
            Dialect::Qwen => qwen_call_name(&self.call_buf),
        }
    }

    /// The call the still-open `<tool_call>` determines so far — the
    /// speculative view sPTC executes against while the model is still
    /// writing (see [`crate::sptc`]).
    ///
    /// Read-only over the accumulating buffer: what reaches the client is
    /// decided by the closing parse alone, so a snapshot can never change
    /// the stream. Costs one full re-parse of the buffered body per call, so
    /// callers gate it on [`Self::open_call_name`] naming a hosted tool —
    /// the body itself is bounded by the server's argument cap in the
    /// speculation path.
    pub fn speculation(&self) -> Option<SpecSnapshot> {
        if !self.in_call {
            return None;
        }
        parse_call_prefix_dialect(&self.call_buf, Some(&self.schemas), self.dialect)
            .map(|c| SpecSnapshot { name: c.name, arguments: c.arguments })
    }

    fn capture(&mut self, text: &str) {
        if self.capture_reasoning {
            self.reasoning.push_str(text);
        }
    }

    fn drain(&mut self, eof: bool) -> String {
        let mut out = String::new();
        loop {
            if self.in_call {
                if let Some(i) = self.pending.find("</tool_call>") {
                    self.call_buf.push_str(&self.pending[..i]);
                    self.pending = self.pending[i + "</tool_call>".len()..].to_string();
                    self.in_call = false;
                    let buf = std::mem::take(&mut self.call_buf);
                    if let Some(c) = parse_call_dialect(&buf, Some(&self.schemas), self.dialect) {
                        self.calls.push(c);
                    }
                    continue;
                }
                if eof {
                    self.call_buf.push_str(&self.pending);
                    self.pending.clear();
                    let buf = std::mem::take(&mut self.call_buf);
                    if let Some(c) = parse_call_dialect(&buf, Some(&self.schemas), self.dialect) {
                        self.calls.push(c);
                    }
                    self.in_call = false;
                } else {
                    let keep = tail_prefix_len(&self.pending, "</tool_call>");
                    let take = self.pending.len() - keep;
                    self.call_buf.push_str(&self.pending[..take]);
                    self.pending = self.pending[take..].to_string();
                }
                break;
            }
            if self.in_think {
                if let Some(i) = self.pending.find("</think>") {
                    let body = self.pending[..i].to_string();
                    self.pending = self.pending[i + "</think>".len()..].to_string();
                    self.capture(&body);
                    self.in_think = false;
                    continue;
                }
                if eof {
                    let body = std::mem::take(&mut self.pending);
                    self.capture(&body);
                    self.in_think = false;
                    continue;
                }
                let keep = tail_prefix_len(&self.pending, "</think>");
                let take = self.pending.len() - keep;
                let body = self.pending[..take].to_string();
                self.pending = self.pending[take..].to_string();
                self.capture(&body);
                break;
            }
            let open_call = self.pending.find("<tool_call>");
            let open_think = self.pending.find("<think>");
            match earliest(open_call, open_think) {
                Some((i, which)) => {
                    out.push_str(&self.pending[..i]);
                    if which == 0 {
                        self.pending = self.pending[i + "<tool_call>".len()..].to_string();
                        self.in_call = true;
                    } else {
                        self.pending = self.pending[i + "<think>".len()..].to_string();
                        self.in_think = true;
                    }
                    continue;
                }
                None => {
                    if eof {
                        out.push_str(&self.pending);
                        self.pending.clear();
                    } else {
                        let keep = MARKERS.iter().map(|m| tail_prefix_len(&self.pending, m)).max().unwrap_or(0);
                        let take = self.pending.len() - keep;
                        out.push_str(&self.pending[..take]);
                        self.pending = self.pending[take..].to_string();
                    }
                    break;
                }
            }
        }
        out
    }
}

/// Index of the earlier of two optional positions, with which one it was
/// (0 = first, 1 = second).
fn earliest(a: Option<usize>, b: Option<usize>) -> Option<(usize, u8)> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if x <= y { (x, 0) } else { (y, 1) }),
        (Some(x), None) => Some((x, 0)),
        (None, Some(y)) => Some((y, 1)),
        (None, None) => None,
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `marker`,
/// i.e. how much of `s` must be held back in case the marker is still arriving.
/// Counted in bytes, and only at char boundaries so the held-back tail stays
/// valid UTF-8.
fn tail_prefix_len(s: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(s.len());
    for n in (1..=max).rev() {
        let start = s.len() - n;
        if !s.is_char_boundary(start) {
            continue;
        }
        if marker.as_bytes().starts_with(&s.as_bytes()[start..]) {
            return n;
        }
    }
    0
}

fn looks_like_qwen_call(body: &str) -> bool {
    let function = body.find("<function=");
    let argument = body.find("<arg_key>");
    match (function, argument) {
        (Some(function), Some(argument)) => function < argument,
        (Some(_), None) => true,
        _ => false,
    }
}

fn qwen_call_name(body: &str) -> Option<&str> {
    let start = body.find("<function=")?;
    let after = &body[start + "<function=".len()..];
    let end = after.find('>')?;
    let name = after[..end].trim();
    if name.is_empty() || name.contains('<') {
        return None;
    }
    Some(name)
}

/// A call as far as the still-open `<tool_call>` markup determines it: the
/// name and every *terminated* argument pair, typed exactly as the closing
/// parse will type them. Produced only by [`OutputFilter::speculation`] and
/// consumed only by the sPTC layer (see [`crate::sptc`]) — it never touches
/// the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecSnapshot {
    pub name: String,
    /// Only complete `<arg_key>/<arg_value>` pairs; an argument the model is
    /// still writing is excluded, which makes these arguments a *prefix* of
    /// the final ones.
    pub arguments: Value,
}

/// Parse one `<tool_call>` body (the text between the tags) into a call.
///
/// Shape: a name, then `<arg_key>`/`<arg_value>` pairs. The name is whatever
/// precedes the first `<arg_key>` — not "the first line" — because a model that
/// puts the name and its first key on one line is writing something perfectly
/// unambiguous, and a body whose first line *is* markup has no name at all.
/// A call with no name yields `None`: dropping a malformed call beats handing a
/// client a tool invocation with an empty name.
fn parse_call(body: &str, schemas: Option<&SchemaMap>) -> Option<ParsedCall> {
    if looks_like_qwen_call(body) {
        parse_qwen_call(body, schemas, false)
    } else {
        parse_glm_call(body, schemas)
    }
}

fn parse_call_dialect(body: &str, schemas: Option<&SchemaMap>, dialect: Dialect) -> Option<ParsedCall> {
    match dialect {
        Dialect::Glm => parse_call(body, schemas),
        Dialect::Qwen => parse_qwen_call(body, schemas, false),
    }
}

fn parse_glm_call(body: &str, schemas: Option<&SchemaMap>) -> Option<ParsedCall> {
    let cut = body.find("<arg_key>").unwrap_or(body.len());
    let name = body[..cut].lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let props = schemas.and_then(|m| m.get(name));
    let mut args = Map::new();
    let mut cur = &body[cut..];
    while let Some(ks) = cur.find("<arg_key>") {
        let after_k = &cur[ks + "<arg_key>".len()..];
        let Some(ke) = after_k.find("</arg_key>") else { break };
        let key = after_k[..ke].trim().to_string();
        let after_key_close = &after_k[ke + "</arg_key>".len()..];
        let Some(vs) = after_key_close.find("<arg_value>") else { break };
        let after_v = &after_key_close[vs + "<arg_value>".len()..];
        // An unterminated final value is still taken: the model may have been
        // cut off at the token cap mid-argument, and the key it named is more
        // useful to the client than nothing.
        let (raw, next) = match after_v.find("</arg_value>") {
            Some(ve) => (&after_v[..ve], &after_v[ve + "</arg_value>".len()..]),
            None => (after_v, ""),
        };
        if !key.is_empty() {
            let declared = props.and_then(|p| p.get(&key)).map(String::as_str);
            args.insert(key, coerce(raw.trim(), declared));
        }
        if next.is_empty() {
            break;
        }
        cur = next;
    }
    Some(ParsedCall { name: name.to_string(), arguments: Value::Object(args) })
}

fn parse_qwen_call(body: &str, schemas: Option<&SchemaMap>, prefix_only: bool) -> Option<ParsedCall> {
    let name = qwen_call_name(body)?;
    let function_start = body.find("<function=")?;
    let after_function = &body[function_start + "<function=".len()..];
    let function_end = after_function.find('>')?;
    let mut cur = &after_function[function_end + 1..];
    let props = schemas.and_then(|m| m.get(name));
    let mut args = Map::new();
    while let Some(relative) = cur.find("<parameter=") {
        let after_open = &cur[relative + "<parameter=".len()..];
        let Some(key_end) = after_open.find('>') else { break };
        let key = after_open[..key_end].trim().to_string();
        let after_value = &after_open[key_end + 1..];
        let Some(value_end) = after_value.find("</parameter>") else {
            if !prefix_only && !key.is_empty() {
                let raw_end = after_value.find("</function>").unwrap_or(after_value.len());
                let declared = props.and_then(|p| p.get(&key)).map(String::as_str);
                args.insert(key, coerce(after_value[..raw_end].trim(), declared));
            }
            break;
        };
        if !key.is_empty() {
            let declared = props.and_then(|p| p.get(&key)).map(String::as_str);
            args.insert(key, coerce(after_value[..value_end].trim(), declared));
        }
        cur = &after_value[value_end + "</parameter>".len()..];
    }
    Some(ParsedCall { name: name.to_string(), arguments: Value::Object(args) })
}

/// The speculative twin of [`parse_call`]: parse a *still-open* call body
/// (no `</tool_call>` yet) into the call its markup determines so far —
/// the name, plus every **terminated** `<arg_key>/<arg_value>` pair, typed
/// by the same [`coerce`] rules the closing parse will apply.
///
/// The one behavioural difference from [`parse_call`] is the point: an
/// unterminated trailing `<arg_value>` is *excluded* here rather than taken,
/// because a value the model is still writing is not a value yet — running a
/// tool against it would be running it against text that is about to change.
/// Excluding it makes the snapshot's arguments a **prefix** of the final
/// arguments (every terminated pair survives to the close unchanged), which
/// is what lets a speculated result be verified by exact-argument match
/// instead of trusted: the sPTC layer compares the close-time key against
/// the speculated one, and a mismatch is a discard, never a wrong answer.
fn parse_call_prefix(body: &str, schemas: Option<&SchemaMap>) -> Option<ParsedCall> {
    if looks_like_qwen_call(body) {
        parse_qwen_call(body, schemas, true)
    } else {
        parse_glm_call_prefix(body, schemas)
    }
}

fn parse_call_prefix_dialect(body: &str, schemas: Option<&SchemaMap>, dialect: Dialect) -> Option<ParsedCall> {
    match dialect {
        Dialect::Glm => parse_call_prefix(body, schemas),
        Dialect::Qwen => parse_qwen_call(body, schemas, true),
    }
}

fn parse_glm_call_prefix(body: &str, schemas: Option<&SchemaMap>) -> Option<ParsedCall> {
    let cut = body.find("<arg_key>").unwrap_or(body.len());
    let name = body[..cut].lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let props = schemas.and_then(|m| m.get(name));
    let mut args = Map::new();
    let mut cur = &body[cut..];
    while let Some(ks) = cur.find("<arg_key>") {
        let after_k = &cur[ks + "<arg_key>".len()..];
        let Some(ke) = after_k.find("</arg_key>") else { break };
        let key = after_k[..ke].trim().to_string();
        let after_key_close = &after_k[ke + "</arg_key>".len()..];
        let Some(vs) = after_key_close.find("<arg_value>") else { break };
        let after_v = &after_key_close[vs + "<arg_value>".len()..];
        // The difference from `parse_call`: an unclosed value stops the scan.
        // The model has not finished saying it.
        let Some(ve) = after_v.find("</arg_value>") else { break };
        if !key.is_empty() {
            let declared = props.and_then(|p| p.get(&key)).map(String::as_str);
            args.insert(key, coerce(after_v[..ve].trim(), declared));
        }
        cur = &after_v[ve + "</arg_value>".len()..];
    }
    Some(ParsedCall { name: name.to_string(), arguments: Value::Object(args) })
}

/// Argument values arrive as text; this decides what JSON type they become.
///
/// **The schema decides when it can.** Guessing from the text alone turns a
/// shell command of `1.10` into the number `1.1` and a file named `true` into a
/// boolean — corruption a client cannot detect, because the wrong type is still
/// valid JSON. So a parameter the tool declared as `string` stays a string
/// verbatim, and only an undeclared parameter falls back to sniffing.
fn coerce(raw: &str, declared: Option<&str>) -> Value {
    match declared {
        Some("string") => return Value::String(raw.to_string()),
        Some("integer") | Some("number") => {
            return serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_number)
                .unwrap_or_else(|| Value::String(raw.to_string()));
        }
        Some("boolean") => {
            return match raw {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                other => Value::String(other.to_string()),
            }
        }
        Some("array") | Some("object") => {
            return serde_json::from_str::<Value>(raw)
                .ok()
                .filter(|v| v.is_array() || v.is_object())
                .unwrap_or_else(|| Value::String(raw.to_string()));
        }
        _ => {}
    }
    if raw.is_empty() {
        return Value::String(String::new());
    }
    // A quoted JSON string unquotes to its contents — the quotes were the
    // encoding, not part of the value; that is `from_str`'s own behaviour, so
    // the `Ok(Value::String(s)) => Value::String(s)` arm this replaces was the
    // identity of the arm below it. Unquoted prose (`hello world`,
    // `/etc/hosts`) is not JSON and stays text.
    serde_json::from_str::<Value>(raw).ok().unwrap_or_else(|| Value::String(raw.to_string()))
}

/// Per-tool map of parameter name → declared JSON Schema `type`, the only part
/// of the schema argument coercion needs.
pub type SchemaMap = std::collections::HashMap<String, std::collections::HashMap<String, String>>;

/// Extract the parameter types from the request's tool schemas.
pub fn schema_map(tools: &[ToolDef]) -> SchemaMap {
    let mut out = SchemaMap::new();
    for t in tools {
        let mut props = std::collections::HashMap::new();
        if let Some(p) = t.function.parameters.as_ref().and_then(|p| p.get("properties")).and_then(|p| p.as_object())
        {
            for (k, v) in p {
                // A union type (`["string","null"]`) is left undeclared rather
                // than half-honoured; the fallback sniff is the honest answer.
                if let Some(ty) = v.get("type").and_then(|t| t.as_str()) {
                    props.insert(k.clone(), ty.to_string());
                }
            }
        }
        out.insert(t.function.name.clone(), props);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through_unchanged() {
        let mut f = OutputFilter::with_tools(&[]);
        let mut out = f.push("hello ");
        out.push_str(&f.push("world"));
        out.push_str(&f.finish());
        assert_eq!(out, "hello world");
        assert!(f.calls.is_empty());
    }

    #[test]
    fn a_whole_tool_call_becomes_a_parsed_call_and_no_visible_text() {
        let mut f = OutputFilter::with_tools(&[]);
        let out = f.push(
            "<tool_call>read\n<arg_key>filePath</arg_key>\n<arg_value>/etc/hosts</arg_value>\n</tool_call>",
        );
        assert_eq!(out, "");
        let calls = f.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["filePath"], json!("/etc/hosts"));
    }

    #[test]
    fn markup_split_across_deltas_never_leaks_a_fragment() {
        // The streaming case that motivated the whole filter: one marker
        // arriving one character at a time must still be recognized, and no
        // prefix of it may be shown to the client on the way.
        let mut f = OutputFilter::with_tools(&[]);
        let mut visible = String::new();
        for ch in "ok<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>ls</arg_value>\n</tool_call>".chars() {
            visible.push_str(&f.push(&ch.to_string()));
        }
        visible.push_str(&f.finish());
        assert_eq!(visible, "ok", "no part of the markup may reach the client");
        let calls = f.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], json!("ls"));
    }

    #[test]
    fn an_unclosed_call_is_still_recovered_at_eof() {
        // Hitting the token cap mid-call is normal; the call is the whole point
        // of the turn, so it is parsed rather than discarded.
        let mut f = OutputFilter::with_tools(&[]);
        assert_eq!(f.push("<tool_call>glob\n<arg_key>pattern</arg_key>\n<arg_value>**/*.rs</arg_value>"), "");
        assert!(f.calls.is_empty(), "not yet — the close tag might still arrive");
        assert_eq!(f.finish(), "", "the recovered call is a call, not visible text");
        let calls = f.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], json!("**/*.rs"));
    }

    #[test]
    fn reasoning_is_dropped_not_shown() {
        let mut f = OutputFilter::with_tools(&[]);
        let mut out = f.push("<think>I should greet them</think>hi");
        out.push_str(&f.finish());
        assert_eq!(out, "hi");
    }

    #[test]
    fn two_calls_in_one_turn_both_survive() {
        let mut f = OutputFilter::with_tools(&[]);
        assert_eq!(
            f.push(
                "<tool_call>read\n<arg_key>filePath</arg_key>\n<arg_value>a</arg_value>\n</tool_call>\
                 <tool_call>read\n<arg_key>filePath</arg_key>\n<arg_value>b</arg_value>\n</tool_call>",
            ),
            ""
        );
        assert_eq!(f.finish(), "");
        let calls = f.take_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments["filePath"], json!("b"));
    }

    #[test]
    fn typed_argument_values_stay_typed_and_bare_words_stay_strings() {
        assert_eq!(coerce("7", None), json!(7));
        assert_eq!(coerce("true", None), json!(true));
        assert_eq!(coerce("[\"a\"]", None), json!(["a"]));
        assert_eq!(coerce("/etc/hosts", None), json!("/etc/hosts"));
        assert_eq!(coerce("hello world", None), json!("hello world"));
        // A quoted string must not lose its quotes and then be re-parsed as a
        // number on the next hop.
        assert_eq!(coerce("\"7\"", None), json!("7"));
    }

    #[test]
    fn the_schema_decides_the_type_not_the_text() {
        // The corruption this exists to prevent: a shell command that looks
        // like a number must not become one, and a version string must not be
        // rounded to a float by JSON parsing.
        assert_eq!(coerce("1.10", Some("string")), json!("1.10"));
        assert_eq!(coerce("true", Some("string")), json!("true"));
        assert_eq!(coerce("42", Some("integer")), json!(42));
        assert_eq!(coerce("yes", Some("boolean")), json!("yes"), "a non-boolean stays text rather than becoming false");
        assert_eq!(coerce("[1,2]", Some("array")), json!([1, 2]));
        // A declared array the model wrote as prose is not silently emptied.
        assert_eq!(coerce("one and two", Some("array")), json!("one and two"));
    }

    #[test]
    fn schema_typing_reaches_a_parsed_call() {
        let tools = vec![ToolDef {
            function: FunctionDef {
                name: "bash".into(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}
                })),
            },
        }];
        let mut f = OutputFilter::with_tools(&tools);
        assert_eq!(
            f.push(
                "<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>1.10</arg_value>\n\
                 <arg_key>timeout</arg_key>\n<arg_value>30</arg_value>\n</tool_call>",
            ),
            ""
        );
        assert_eq!(f.finish(), "");
        let calls = f.take_calls();
        assert_eq!(calls[0].arguments["command"], json!("1.10"), "declared string stays a string");
        assert_eq!(calls[0].arguments["timeout"], json!(30), "declared integer becomes a number");
    }

    #[test]
    fn a_name_on_the_same_line_as_its_first_key_still_parses() -> Result<(), &'static str> {
        let c = parse_call("read<arg_key>filePath</arg_key>\n<arg_value>/tmp/x</arg_value>", None)
            .ok_or("name precedes the first key")?;
        assert_eq!(c.name, "read");
        assert_eq!(c.arguments["filePath"], json!("/tmp/x"));
        Ok(())
    }

    #[test]
    fn a_call_with_no_name_is_dropped() {
        assert!(parse_call("\n<arg_key>a</arg_key>\n<arg_value>b</arg_value>", None).is_none());
    }

    #[test]
    fn openai_form_puts_arguments_in_a_string() -> Result<(), &'static str> {
        let c = ParsedCall { name: "read".into(), arguments: json!({"filePath": "/tmp/x"}) };
        let v = c.to_openai(0, "call_1");
        assert_eq!(v["type"], json!("function"));
        assert_eq!(v["function"]["name"], json!("read"));
        // Asserted through the Option/Result rather than unwrapped out of it:
        // `arguments` being a *string* holding JSON is the contract clients
        // parse against, so "it was not a string" must fail as a comparison
        // rather than as a panic.
        let args = v["function"]["arguments"].as_str().ok_or("arguments must be a string")?;
        let parsed: Value = serde_json::from_str(args).map_err(|e| {
            eprintln!("arguments did not hold valid json: {e}");
            "arguments must hold valid json"
        })?;
        assert_eq!(parsed["filePath"], json!("/tmp/x"));
        Ok(())
    }

    #[test]
    fn assistant_calls_round_trip_back_into_glm_markup() {
        // A multi-turn conversation replays the model's own calls to it; the
        // rendering must be the shape it emits, values unquoted.
        let calls = vec![json!({
            "id": "call_1", "type": "function",
            "function": {"name": "read", "arguments": "{\"filePath\":\"/etc/hosts\"}"}
        })];
        let s = render_assistant_calls(&calls);
        assert!(s.contains("<tool_call>read"), "{s}");
        assert!(s.contains("<arg_value>/etc/hosts</arg_value>"), "{s}");

        let mut f = OutputFilter::with_tools(&[]);
        assert_eq!(f.push(s.trim_start()), "");
        assert_eq!(f.finish(), "");
        let back = f.take_calls();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].arguments["filePath"], json!("/etc/hosts"));
    }

    #[test]
    fn the_preamble_carries_every_tool_name_and_its_schema() {
        let tools = vec![ToolDef {
            function: FunctionDef {
                name: "bash".into(),
                description: Some("run a command".into()),
                parameters: Some(json!({"type": "object", "properties": {"command": {"type": "string"}}})),
            },
        }];
        let s = render_preamble(&tools);
        assert!(s.contains("\"name\":\"bash\""), "{s}");
        assert!(s.contains("run a command"));
        assert!(s.contains("<tool_call>"), "the output format must be shown");
    }

    #[test]
    fn a_lone_open_angle_bracket_is_not_held_hostage() {
        // `a < b` must not stall forever waiting to see if it becomes a marker.
        let mut f = OutputFilter::with_tools(&[]);
        let mut out = f.push("a < b");
        out.push_str(&f.finish());
        assert_eq!(out, "a < b");
    }

    #[test]
    fn multibyte_text_around_a_marker_stays_valid_utf8() {
        let mut f = OutputFilter::with_tools(&[]);
        let mut out = f.push("héllo →");
        out.push_str(&f.push("<tool_call>x\n</tool_call>"));
        out.push_str(&f.finish());
        assert_eq!(out, "héllo →");
        assert_eq!(f.take_calls().len(), 1);
    }

    #[test]
    fn a_still_open_call_speculates_from_its_complete_pairs_only() -> Result<(), String> {
        // The sPTC view: the call is open, one pair is fully written, the
        // second is mid-value. The snapshot must carry the first pair and
        // nothing of the second — a value the model is still writing is not
        // a value yet.
        let mut f = OutputFilter::with_tools(&[]);
        assert_eq!(f.push("<tool_call>"), "", "the marker itself never shows");
        assert!(f.open_call_name().is_none(), "the open body has no name yet");

        f.push("write\n<arg_key>filePath</arg_key>\n<arg_value>/tmp/a</arg_value>\n");
        assert_eq!(f.open_call_name(), Some("write"), "the cheap name probe");
        let snap = f.speculation().ok_or_else(|| "open call must yield a snapshot".to_string())?;
        assert_eq!(snap.name, "write");
        assert_eq!(snap.arguments["filePath"], json!("/tmp/a"));

        f.push("<arg_key>content</arg_key>\n<arg_value>half a thought");
        let snap2 = f.speculation().ok_or_else(|| "still open".to_string())?;
        assert!(
            snap2.arguments.get("content").is_none(),
            "an unterminated value is excluded from speculation: {}",
            snap2.arguments
        );
        assert_eq!(snap2.arguments["filePath"], json!("/tmp/a"), "complete pairs survive");

        // The close-time parse is unchanged by any of this: the final call
        // carries both pairs, typed as ever.
        f.push(" done</arg_value>\n</tool_call>");
        assert_eq!(f.push(""), "");
        let calls = f.take_calls();
        let c = calls.first().ok_or_else(|| "the close must parse one call".to_string())?;
        assert_eq!(c.arguments["content"], json!("half a thought done"));
        assert!(f.speculation().is_none(), "no open call, no snapshot");
        Ok(())
    }

    #[test]
    fn the_speculated_prefix_key_equals_the_finished_call_key() -> Result<(), String> {
        // THE property sPTC verification rests on: parse the same call
        // partially and fully, and the canonical serializations agree —
        // otherwise every speculation would miss and the layer would be
        // pure overhead.
        let tools = vec![ToolDef {
            function: FunctionDef {
                name: "bash".into(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}
                })),
            },
        }];
        let full = "<tool_call>bash\n<arg_key>command</arg_key>\n<arg_value>1.10</arg_value>\n\
                    <arg_key>timeout</arg_key>\n<arg_value>30</arg_value>\n</tool_call>";
        let mut f = OutputFilter::with_tools(&tools);
        // Feed everything except the close tag: prefix state.
        let body = &full[..full.len() - "</tool_call>".len()];
        f.push(body);
        let snap = f
            .speculation()
            .ok_or_else(|| "open call must yield a snapshot".to_string())?;
        f.push("</tool_call>");
        let calls = f.take_calls();
        let c = calls.first().ok_or_else(|| "the close must parse one call".to_string())?;
        assert_eq!(snap.name, c.name);
        assert_eq!(
            snap.arguments.to_string(),
            c.arguments.to_string(),
            "canonical keys must agree, or speculation never verifies"
        );
        Ok(())
    }

    #[test]
    fn the_name_probe_costs_no_argument_parse_and_lies_about_nothing() -> Result<(), String> {
        let mut f = OutputFilter::with_tools(&[]);
        f.push("<tool_call>read\n");
        assert_eq!(f.open_call_name(), Some("read"));
        let snap = f.speculation().ok_or_else(|| "open call must yield a snapshot".to_string())?;
        assert_eq!(snap.name, "read");
        let empty = snap.arguments.as_object().ok_or_else(|| "arguments are an object".to_string())?;
        assert!(empty.is_empty(), "no complete pairs yet: {snap:?}");
        Ok(())
    }

    #[test]
    fn qwen_call_round_trip_uses_checkpoint_xml_and_schema_types() -> Result<(), String> {
        let tools = vec![ToolDef {
            function: FunctionDef {
                name: "read".into(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"}
                    }
                })),
            },
        }];
        let calls = vec![json!({
            "function": {
                "name": "read",
                "arguments": "{\"path\":\"1.10\",\"count\":\"30\",\"enabled\":\"true\"}"
            }
        })];
        let rendered = render_assistant_calls_qwen(&calls);
        assert!(rendered.contains("<function=read>"), "{rendered}");
        assert!(rendered.contains("<parameter=path>"), "{rendered}");
        let mut f = OutputFilter::with_qwen_tools(&tools, false);
        assert_eq!(f.push(&rendered), "");
        assert_eq!(f.finish(), "");
        let parsed = f.take_calls();
        let call = parsed.first().ok_or_else(|| "qwen call must parse".to_string())?;
        assert_eq!(call.name, "read");
        assert_eq!(call.arguments["path"], json!("1.10"));
        assert_eq!(call.arguments["count"], json!(30));
        assert_eq!(call.arguments["enabled"], json!(true));
        Ok(())
    }

    #[test]
    fn qwen_streaming_speculation_only_uses_complete_parameters() -> Result<(), String> {
        let mut f = OutputFilter::with_tools_dialect(&[], Dialect::Qwen, false);
        assert_eq!(f.push("<tool_call>\n<function=read>"), "");
        assert_eq!(f.open_call_name(), Some("read"));
        f.push("<parameter=path>\n/tmp/a</parameter>\n<parameter=count>\n7");
        let snap = f.speculation().ok_or_else(|| "qwen prefix must speculate".to_string())?;
        assert_eq!(snap.arguments["path"], json!("/tmp/a"));
        assert!(snap.arguments.get("count").is_none(), "{snap:?}");
        f.push("</parameter>\n</function>\n</tool_call>");
        let calls = f.take_calls();
        let call = calls.first().ok_or_else(|| "qwen close must parse".to_string())?;
        assert_eq!(call.arguments["count"], json!(7));
        Ok(())
    }

    #[test]
    fn qwen_reasoning_is_captured_only_when_enabled() {
        let mut enabled = OutputFilter::with_tools_dialect(&[], Dialect::Qwen, true);
        assert_eq!(enabled.push("<think>reason"), "");
        assert_eq!(enabled.take_reasoning(), "reason");
        assert_eq!(enabled.push("ing</think>answer"), "answer");
        assert_eq!(enabled.take_reasoning(), "ing");
        assert_eq!(enabled.finish(), "");

        let mut default = OutputFilter::with_tools_dialect(&[], Dialect::Qwen, false);
        assert_eq!(default.push("<think>hidden</think>answer"), "answer");
        assert_eq!(default.take_reasoning(), "");
        assert_eq!(default.finish(), "");
    }

    #[test]
    fn qwen_preamble_matches_checkpoint_wording() {
        let tools = vec![ToolDef {
            function: FunctionDef { name: "bash".into(), description: Some("run".into()), parameters: None },
        }];
        let s = render_preamble_qwen(&tools);
        assert!(s.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>\n"), "{s}");
        assert!(s.contains("<function=example_function_name>"), "{s}");
        assert!(s.contains("<parameter=example_parameter_1>"), "{s}");
        assert!(s.contains("</IMPORTANT>"), "{s}");
    }
}
