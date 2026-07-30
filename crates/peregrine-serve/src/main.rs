//! peregrine-serve — a native, OpenAI-compatible HTTP server for the peregrine
//! engine (axum/tokio). Exposes `POST /v1/chat/completions` (streaming SSE and
//! non-streaming), `GET /v1/models`, and `GET /health`.
//!
//! Design & safety:
//! - A single [`batch`] engine thread owns the model and continuously batches all
//!   in-flight requests (one decode token per active sequence per step), so
//!   concurrent requests share expert reads instead of serializing behind a lock.
//!   Each handler submits a request and streams decoded token deltas back over an
//!   async channel.
//! - No panics anywhere (deny-lints below); every error becomes an OpenAI-shaped
//!   JSON body. Binds `127.0.0.1` by default; optional bearer `--api-key`;
//!   `max_tokens` and prompt-length caps; graceful Ctrl-C shutdown.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod batch;
mod tok;

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use batch::{EngineHandle, EngineOut, EngineRequest, Priority};
use clap::Parser;
use peregrine_core::Error;
use peregrine_model::{Model, Sampler};
use serde::{Deserialize, Serialize};
use tok::TokenBackend;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// CLI configuration.
#[derive(Parser, Clone)]
#[command(name = "peregrine-serve", about = "OpenAI-compatible server for peregrine")]
struct Args {
    /// Model directory (config.json + *.safetensors + tokenizer.json). Falls back to $COLI_MODEL.
    #[arg(long, env = "COLI_MODEL")]
    model: String,
    /// Bind host (default localhost for safety).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Require this bearer token on every request when set.
    #[arg(long, env = "PEREGRINE_API_KEY")]
    api_key: Option<String>,
    /// Hard cap on generated tokens per request.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Hard cap on prompt tokens per request.
    #[arg(long, default_value_t = 8192)]
    max_prompt_tokens: usize,
    /// Public model id reported by /v1/models and in responses.
    #[arg(long, default_value = "glm-5.2")]
    model_id: String,
    /// Max sequences decoded together per batched step (continuous-batching width).
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Benchmark the tokenizer backends on a text file and exit: encodes the
    /// file through the gigatoken tokenizer, reports MB/s. Needs
    /// `<model>/tokenizer.json`; no model weights are loaded.
    #[arg(long, value_name = "TEXT_FILE")]
    bench_tokenizer: Option<std::path::PathBuf>,
}

/// Shared, cloneable server state.
#[derive(Clone)]
struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    engine: EngineHandle,
    tokenizer: Arc<TokenBackend>,
    args: Args,
}

// ---- OpenAI request/response shapes ----

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}
#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

// ---- error → OpenAI JSON ----

struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
}
impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> ApiError {
        ApiError { status, kind, message: message.into() }
    }
    fn bad_request(m: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", m)
    }
    fn internal(m: impl Into<String>) -> ApiError {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "message": self.message, "type": self.kind }
        });
        (self.status, Json(body)).into_response()
    }
}
impl From<Error> for ApiError {
    fn from(e: Error) -> ApiError {
        ApiError::internal(e.to_string())
    }
}

/// Convert a tokenizer error into our error type — the one boundary that
/// needs it, kept in a single helper.
fn tk<T>(r: Result<T, peregrine_core::Error>) -> Result<T, ApiError> {
    r.map_err(|e| ApiError::internal(format!("tokenizer: {e}")))
}

/// Build the GLM-5.2 prompt from chat messages (no chat_template ships in the
/// tokenizer): `[gMASK]<sop>` then `<|role|>\n{content}` per turn, ending with
/// an empty `<|assistant|>` turn to generate into.
fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::from("[gMASK]<sop>");
    for m in messages {
        let role = match m.role.as_str() {
            "system" => "<|system|>",
            "assistant" => "<|assistant|>",
            "user" => "<|user|>",
            // unknown roles are treated as user content, never trusted as markup
            _ => "<|user|>",
        };
        s.push_str(role);
        s.push('\n');
        s.push_str(&m.content);
    }
    s.push_str("<|assistant|>\n");
    s
}

/// A monotonically-unique-ish seed for the sampler without extra deps.
fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9E3779B9)
}

/// Submit a request to the batch engine and return its token-id stream. The
/// caller decodes ids to text — incrementally for streaming, in one shot
/// otherwise — so the engine stays tokenizer-free and both paths share it.
fn submit_request(
    state: &AppState,
    ids: &[u32],
    max_new: usize,
    temperature: f32,
    top_p: f32,
    priority: Priority,
    class: peregrine_model::TokenClass,
) -> Result<mpsc::Receiver<EngineOut>, ApiError> {
    let (tx, rx) = mpsc::channel::<EngineOut>(64);
    let prompt: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    let sampler = Sampler::new(temperature, top_p, seed());
    state.inner.engine.submit(EngineRequest { prompt, max_new, sampler, out: tx, priority, class })?;
    Ok(rx)
}

/// Classify the workload from the tail of the last user message — the part of
/// the conversation the model is about to continue, so the best signal for
/// what routing distribution decode will see. The tail is capped at 512 chars
/// (classification is a ratio heuristic; more text doesn't sharpen it).
fn classify_request(messages: &[ChatMessage]) -> peregrine_model::TokenClass {
    let last_user = messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.as_str()).unwrap_or("");
    let tail_start = last_user.len().saturating_sub(512);
    // step forward to a char boundary so the slice is valid UTF-8
    let mut start = tail_start;
    while start < last_user.len() && !last_user.is_char_boundary(start) {
        start += 1;
    }
    peregrine_model::classify_str(&last_user[start..])
}

/// Parse an `X-Peregrine-Priority` header value into a [`Priority`]. Unknown
/// values fall back to `Normal` — a deliberately-lax mapping so a client can't
/// break admission by sending an unrecognized string.
fn priority_from_header(v: Option<&str>) -> Priority {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("high") | Some("1") | Some("true") => Priority::High,
        _ => Priority::Normal,
    }
}

/// Resolve + validate common generation params against the server caps.
fn resolve_params(state: &AppState, req: &ChatRequest) -> Result<(Vec<u32>, usize, f32, f32), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let prompt = build_prompt(&req.messages);
    let ids = tk(state.inner.tokenizer.encode(&prompt))?;
    if ids.len() > state.inner.args.max_prompt_tokens {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!("prompt is {} tokens; max is {}", ids.len(), state.inner.args.max_prompt_tokens),
        ));
    }
    let max_new = req.max_tokens.unwrap_or(256).min(state.inner.args.max_tokens).max(1);
    let temperature = req.temperature.unwrap_or(0.0).clamp(0.0, 2.0);
    let top_p = req.top_p.unwrap_or(0.95).clamp(0.0, 1.0);
    Ok((ids, max_new, temperature, top_p))
}

/// A header value as UTF-8, treating a non-UTF-8 value as absent — the lax
/// parse both header consumers (auth, priority) want. The rejection is
/// reported (`COLI_DEBUG=1`) rather than silently `.ok()`-dropped.
fn header_utf8(v: &axum::http::HeaderValue) -> Option<&str> {
    match v.to_str() {
        Ok(s) => Some(s),
        Err(e) => {
            peregrine_core::note_advisory_err("non-UTF-8 request header ignored", &e);
            None
        }
    }
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(want) = state.inner.args.api_key.as_deref() else {
        return Ok(());
    };
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(header_utf8)
        .and_then(|s| s.strip_prefix("Bearer "));
    if got == Some(want) {
        Ok(())
    } else {
        Err(ApiError::new(StatusCode::UNAUTHORIZED, "invalid_request_error", "missing or invalid API key"))
    }
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "tokenizer": state.inner.tokenizer.name() }))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<ModelList>, ApiError> {
    check_auth(&state, &headers)?;
    Ok(Json(ModelList {
        object: "list",
        data: vec![ModelCard { id: state.inner.args.model_id.clone(), object: "model", owned_by: "peregrine" }],
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;
    let (ids, max_new, temperature, top_p) = resolve_params(&state, &req)?;
    let model_id = state.inner.args.model_id.clone();
    let priority = priority_from_header(headers.get("x-peregrine-priority").and_then(header_utf8));
    let class = classify_request(&req.messages);
    let mut rx = submit_request(&state, &ids, max_new, temperature, top_p, priority, class)?;
    let tokenizer = state.inner.tokenizer.clone();

    if req.stream {
        // SSE: an async task decodes engine token ids into text deltas and pushes
        // OpenAI chunk events; the response streams them.
        let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let mid = model_id.clone();
        tokio::spawn(async move {
            let mut out_ids: Vec<u32> = Vec::new();
            let mut prev = String::new();
            while let Some(msg) = rx.recv().await {
                match msg {
                    EngineOut::Token(t) => {
                        out_ids.push(t);
                        // incremental decode: emit the newly-completed suffix
                        match tokenizer.decode(&out_ids) {
                            Ok(text) => {
                                if text.len() > prev.len() {
                                    let ev = chunk_event(&mid, Some(&text[prev.len()..]), None);
                                    if sse_tx.send(Ok(ev)).await.is_err() {
                                        return; // client disconnected
                                    }
                                    prev = text;
                                }
                            }
                            Err(e) => {
                                // end the stream with an error event instead of
                                // silently dropping the delta
                                if sse_tx.send(Ok(sse_error(&format!("decode: {e}")))).await.is_err() {
                                    peregrine_core::note_advisory_err("SSE decode-error event", &"client already disconnected");
                                }
                                return;
                            }
                        }
                    }
                    EngineOut::Error(m) => {
                        if sse_tx.send(Ok(sse_error(&m))).await.is_err() {
                            peregrine_core::note_advisory_err("SSE error event", &"client already disconnected");
                        }
                        return;
                    }
                }
            }
            // tail frames: a send error just means the client hung up first
            if sse_tx.send(Ok(chunk_event(&mid, None, Some("stop")))).await.is_err()
                || sse_tx.send(Ok(Event::default().data("[DONE]"))).await.is_err()
            {
                peregrine_core::note_advisory_err("SSE stream tail", &"client disconnected before [DONE]");
            }
        });
        let stream = ReceiverStream::new(sse_rx);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming: collect the whole token stream, decode once, one JSON body.
        let mut out_ids: Vec<u32> = Vec::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                EngineOut::Token(t) => out_ids.push(t),
                EngineOut::Error(m) => return Err(ApiError::internal(m)),
            }
        }
        let text = tk(tokenizer.decode(&out_ids))?;
        let body = serde_json::json!({
            "id": format!("chatcmpl-{}", seed()),
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop"
            }]
        });
        Ok(Json(body).into_response())
    }
}

/// One streaming chunk in OpenAI `chat.completion.chunk` shape.
fn chunk_event(model_id: &str, delta: Option<&str>, finish: Option<&str>) -> Event {
    let delta_obj = match delta {
        Some(d) => serde_json::json!({ "content": d }),
        None => serde_json::json!({}),
    };
    let payload = serde_json::json!({
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [{ "index": 0, "delta": delta_obj, "finish_reason": finish }]
    });
    Event::default().data(payload.to_string())
}

fn sse_error(message: &str) -> Event {
    Event::default().data(serde_json::json!({ "error": { "message": message } }).to_string())
}

/// Encode `file` through the gigatoken tokenizer (per-line, mirroring the
/// serve pattern of many short encodes) and report MB/s + total ids.
/// Correctness is the parity test suite's job (`tests/tokenizer_parity.rs`,
/// HF oracle as a dev-dependency); this only measures throughput. Numbers are
/// local to this box; no docs-level claims.
fn bench_tokenizer(model_dir: &std::path::Path, file: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read(model_dir.join("tokenizer.json"))?;
    let text = std::fs::read_to_string(file)?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    if lines.is_empty() {
        return Err("bench file has no non-empty lines".into());
    }
    eprintln!("bench: {} lines, {:.1} MB", lines.len(), bytes as f64 / 1e6);

    let mut giga = peregrine_token::GigaTokenizer::from_hf_json_bytes(&json)
        .map_err(|e| format!("gigatoken: {e}"))?;

    // warmup (fills the pretoken memo cache the way a running server would;
    // the ids themselves are not needed here)
    for l in lines.iter().take(64) {
        giga.encode(l);
    }

    let t0 = std::time::Instant::now();
    let mut giga_ids = 0usize;
    for l in &lines {
        giga_ids += giga.encode(l).len();
    }
    let giga_s = t0.elapsed().as_secs_f64();

    let mbs = |s: f64| bytes as f64 / 1e6 / s.max(1e-9);
    println!("gigatoken     : {:8.2} MB/s  ({giga_ids} ids, {giga_s:.3}s)", mbs(giga_s));
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cap glibc arenas before the model spawns its worker pools, so the server no
    // longer needs `MALLOC_ARENA_MAX=2` in the environment to keep RSS flat.
    peregrine_model::cap_malloc_arenas();
    let args = Args::parse();
    let dir = std::path::PathBuf::from(&args.model);
    // Tokenizer throughput bench: encode a text file through both backends and
    // exit — no model weights loaded, so it runs anywhere tokenizer.json does.
    if let Some(file) = &args.bench_tokenizer {
        bench_tokenizer(&dir, file)?;
        return Ok(());
    }
    let model = Model::load(&dir)?;
    let tokenizer = TokenBackend::load(&dir).map_err(|e| format!("tokenizer: {e}"))?;

    // One engine thread owns the model and continuously batches all requests.
    let (engine, _engine_join) = batch::spawn(model, args.max_batch)?;

    let addr = format!("{}:{}", args.host, args.port);
    let state = AppState { inner: Arc::new(Inner { engine, tokenizer: Arc::new(tokenizer), args }) };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("peregrine-serve listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                eprintln!("peregrine-serve: ctrl-c handler failed ({e}); shutting down");
            }
            eprintln!("peregrine-serve shutting down");
        })
        .await?;
    Ok(())
}
