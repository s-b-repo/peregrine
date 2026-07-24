//! peregrine-serve — a native, OpenAI-compatible HTTP server for the peregrine
//! engine (axum/tokio). Exposes `POST /v1/chat/completions` (streaming SSE and
//! non-streaming), `GET /v1/models`, and `GET /health`.
//!
//! Design & safety:
//! - The model holds one KV cache, so generation is serialized behind a
//!   `std::sync::Mutex<Model>`; each request runs on `spawn_blocking` (inference
//!   is CPU-bound) and streams decoded token deltas back over an async channel.
//! - No panics anywhere (deny-lints below); every error becomes an OpenAI-shaped
//!   JSON body. Binds `127.0.0.1` by default; optional bearer `--api-key`;
//!   `max_tokens` and prompt-length caps; graceful Ctrl-C shutdown.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use peregrine_core::Error;
use peregrine_model::{Model, Sampler};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
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
}

/// Shared, cloneable server state.
#[derive(Clone)]
struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    model: Mutex<Model>,
    tokenizer: Tokenizer,
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

/// Convert a tokenizer error (boxed) into our error type — the one boundary that
/// needs it, kept in a single helper.
fn tk<T>(r: tokenizers::Result<T>) -> Result<T, ApiError> {
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

/// Streamed greedy/sampled generation over the locked model. Calls `on_delta`
/// with each new decoded text fragment; returns the number of tokens produced.
fn generate_stream(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
    temperature: f32,
    top_p: f32,
    mut on_delta: impl FnMut(&str) -> Result<(), ()>,
) -> Result<usize, ApiError> {
    model.reset();
    let vocab = model.cfg.vocab as usize;
    let stop = model.cfg.stop_ids.clone();
    let prompt: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
    if prompt.is_empty() {
        return Ok(0);
    }
    let mut sampler = Sampler::new(temperature, top_p, seed());
    let logits = model.forward_step(&prompt, 0)?;
    let last = (prompt.len() - 1) * vocab;
    let mut tok = sampler.pick(&logits[last..last + vocab], -1) as i32;

    let mut out_ids: Vec<u32> = Vec::new();
    let mut prev = String::new();
    let mut n = 0usize;
    loop {
        if stop.contains(&tok) {
            break;
        }
        out_ids.push(tok as u32);
        n += 1;
        // incremental decode: emit the newly-completed suffix
        let text = tk(tokenizer.decode(&out_ids, true))?;
        if text.len() > prev.len() {
            if on_delta(&text[prev.len()..]).is_err() {
                break; // client disconnected
            }
            prev = text;
        }
        if n >= max_new {
            break;
        }
        let pos = prompt.len() + n - 1;
        let lg = model.forward_step(&[tok], pos)?;
        tok = sampler.pick(&lg[..vocab], -1) as i32;
    }
    Ok(n)
}

/// Resolve + validate common generation params against the server caps.
fn resolve_params(state: &AppState, req: &ChatRequest) -> Result<(Vec<u32>, usize, f32, f32), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let prompt = build_prompt(&req.messages);
    let enc = tk(state.inner.tokenizer.encode(prompt, false))?;
    let ids = enc.get_ids().to_vec();
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

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(want) = state.inner.args.api_key.as_deref() else {
        return Ok(());
    };
    let got = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    if got == Some(want) {
        Ok(())
    } else {
        Err(ApiError::new(StatusCode::UNAUTHORIZED, "invalid_request_error", "missing or invalid API key"))
    }
}

async fn health() -> &'static str {
    "ok"
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

    if req.stream {
        // SSE: a blocking task generates and pushes deltas; the response streams them.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);
        let inner = state.inner.clone();
        let mid = model_id.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = match inner.model.lock() {
                Ok(m) => m,
                Err(_) => {
                    let _ = tx.blocking_send(Ok(sse_error("model lock poisoned")));
                    return;
                }
            };
            let send_chunk = |delta: &str| -> Result<(), ()> {
                let ev = chunk_event(&mid, Some(delta), None);
                tx.blocking_send(Ok(ev)).map_err(|_| ())
            };
            match generate_stream(&mut model, &inner.tokenizer, &ids, max_new, temperature, top_p, send_chunk) {
                Ok(_) => {
                    let _ = tx.blocking_send(Ok(chunk_event(&mid, None, Some("stop"))));
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                }
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_error(&e.message)));
                }
            }
        });
        let stream = ReceiverStream::new(rx);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming: collect the full completion, then return one JSON body.
        let inner = state.inner.clone();
        let text = tokio::task::spawn_blocking(move || -> Result<String, ApiError> {
            let mut model = inner.model.lock().map_err(|_| ApiError::internal("model lock poisoned"))?;
            let mut out = String::new();
            generate_stream(&mut model, &inner.tokenizer, &ids, max_new, temperature, top_p, |d| {
                out.push_str(d);
                Ok(())
            })?;
            Ok(out)
        })
        .await
        .map_err(|e| ApiError::internal(format!("worker join: {e}")))??;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let dir = std::path::PathBuf::from(&args.model);
    let model = Model::load(&dir)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer: {e}"))?;

    let addr = format!("{}:{}", args.host, args.port);
    let state = AppState { inner: Arc::new(Inner { model: Mutex::new(model), tokenizer, args }) };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("peregrine-serve listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("peregrine-serve shutting down");
        })
        .await?;
    Ok(())
}
