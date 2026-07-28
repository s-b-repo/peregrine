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

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use batch::{EngineHandle, EngineOut, EngineRequest};
use clap::Parser;
use peregrine_core::Error;
use peregrine_model::{Model, Sampler};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
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
}

/// Shared, cloneable server state.
#[derive(Clone)]
struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    engine: EngineHandle,
    tokenizer: Arc<Tokenizer>,
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

/// Submit a request to the batch engine and return its token-id stream. The
/// caller decodes ids to text — incrementally for streaming, in one shot
/// otherwise — so the engine stays tokenizer-free and both paths share it.
fn submit_request(
    state: &AppState,
    ids: &[u32],
    max_new: usize,
    temperature: f32,
    top_p: f32,
) -> Result<mpsc::Receiver<EngineOut>, ApiError> {
    let (tx, rx) = mpsc::channel::<EngineOut>(64);
    let prompt: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    let sampler = Sampler::new(temperature, top_p, seed());
    state.inner.engine.submit(EngineRequest { prompt, max_new, sampler, out: tx })?;
    Ok(rx)
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
    let mut rx = submit_request(&state, &ids, max_new, temperature, top_p)?;
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
                        if let Ok(text) = tokenizer.decode(&out_ids, true) {
                            if text.len() > prev.len() {
                                let ev = chunk_event(&mid, Some(&text[prev.len()..]), None);
                                if sse_tx.send(Ok(ev)).await.is_err() {
                                    return; // client disconnected
                                }
                                prev = text;
                            }
                        }
                    }
                    EngineOut::Error(m) => {
                        let _ = sse_tx.send(Ok(sse_error(&m))).await;
                        return;
                    }
                }
            }
            let _ = sse_tx.send(Ok(chunk_event(&mid, None, Some("stop")))).await;
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
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
        let text = tk(tokenizer.decode(&out_ids, true))?;
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
    // Cap glibc arenas before the model spawns its worker pools, so the server no
    // longer needs `MALLOC_ARENA_MAX=2` in the environment to keep RSS flat.
    peregrine_model::cap_malloc_arenas();
    let args = Args::parse();
    let dir = std::path::PathBuf::from(&args.model);
    let model = Model::load(&dir)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer: {e}"))?;

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
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("peregrine-serve shutting down");
        })
        .await?;
    Ok(())
}
