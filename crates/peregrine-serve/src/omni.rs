//! HTTP transport for the released Omni model roster. Each request sends a
//! bounded prompt or conversation to an isolated Omni worker.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use peregrine_core::Error;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWriteExt as _};
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::{bind, chunk_event, http, seed, shutdown_signal, unix_seconds};
use crate::{ApiError, Args, ChatRequest, ModelCard, ModelList};

const MAX_INPUT_BYTES: usize = 16_384;
const MAX_MESSAGES: usize = 33;
const MAX_BODY_BYTES: usize = 6 * MAX_INPUT_BYTES + 4096;
const DEADLINE: Duration = Duration::from_secs(115);
const PROBE_DEADLINE: Duration = Duration::from_secs(15);
const MAX_WORKER_STDOUT: usize = 4 * 1024 * 1024;
const MAX_WORKER_STDERR: usize = 64 * 1024;

struct Suite {
    root: PathBuf,
    binary: PathBuf,
    artifacts: PathBuf,
    store: Option<PathBuf>,
    models: BTreeMap<String, ModelRoute>,
    conversations: bool,
    chat: Option<NativeChat>,
    max_tokens: usize,
    permits: Semaphore,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelRoute {
    Legacy(String),
    Declared {
        family: String,
        #[serde(default)]
        input: InputMode,
    },
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputMode {
    #[default]
    Prompt,
    Messages,
}

impl ModelRoute {
    fn accepts_history(&self) -> bool {
        match self {
            Self::Declared { input, .. } => matches!(input, InputMode::Messages),
            Self::Legacy(family) => matches!(
                family.as_str(),
                "causal chat generation" | "automatic field selection"
            ),
        }
    }

    fn family(&self) -> &str {
        match self {
            Self::Legacy(family) | Self::Declared { family, .. } => family,
        }
    }
}

struct NativeChat {
    decoder: PathBuf,
    checkpoint: PathBuf,
    context: usize,
    sha256: String,
    step: u64,
    prose_checker: PathBuf,
    science_index: PathBuf,
}

#[derive(Deserialize)]
struct ChatManifest {
    generator: PathBuf,
    generator_sha256: String,
}

impl NativeChat {
    fn load(root: &Path, artifacts: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let folder = canonical_path(&artifacts.join("chat"), "suite chat bundle")?;
        let model: ChatManifest =
            serde_json::from_slice(&std::fs::read(folder.join("model.json"))?)?;
        let checkpoint = canonical_path(&folder.join(model.generator), "suite chat checkpoint")?;
        if !checkpoint.starts_with(&folder) || !checkpoint.is_file() {
            return Err(
                Error::Format("chat checkpoint must be a file in the chat bundle".into()).into(),
            );
        }
        let state: Value = serde_json::from_slice(&std::fs::read(
            checkpoint.with_extension("state").join("state.json"),
        )?)?;
        let context = state["config"]["model_config"]["context"]
            .as_u64()
            .and_then(|width| usize::try_from(width).ok())
            .filter(|width| *width > 0)
            .ok_or_else(|| {
                Error::Format("missing or invalid chat context in checkpoint state".into())
            })?;
        let step = state["step"].as_u64().ok_or_else(|| {
            Error::Format("missing training step in chat checkpoint state".into())
        })?;
        if Some(model.generator_sha256.as_str()) != state["model"]["sha256"].as_str() {
            return Err(Error::Format(
                "chat checkpoint manifest and training state have different hashes".into(),
            )
            .into());
        }
        Ok(Self {
            decoder: canonical_path(&artifacts.join("runtime/dblocks"), "suite Unifur decoder")?,
            checkpoint,
            context,
            sha256: model.generator_sha256,
            step,
            prose_checker: root.join("tools/science_reference.py"),
            science_index: artifacts.join("references/sciq.sqlite"),
        })
    }

    async fn complete(
        &self,
        reply: &mut ModelReply,
        prompt: &str,
        requested: Option<usize>,
        server_cap: usize,
    ) -> Result<&'static str, ApiError> {
        if prompt.is_empty() || prompt.len() > self.context || prompt.contains('\0') {
            return Err(ApiError::bad_request(
                "invalid prompt for the Nina chat checkpoint",
            ));
        }
        if reply.details["checkpoint"]["generator_step"].as_u64() != Some(self.step)
            || reply.details["checkpoint"]["generator_sha256"].as_str()
                != Some(self.sha256.as_str())
        {
            return Err(ApiError::internal(
                "the Nina chat checkpoint has changed; restart Peregrine with its current bundle",
            ));
        }
        let limit = requested
            .unwrap_or(server_cap)
            .min(server_cap)
            .min(self.context);
        let mut decoder = Command::new(&self.decoder);
        decoder
            .args(["lm", "generate", "--cached", "--checkpoint"])
            .arg(&self.checkpoint)
            .arg("--max-new")
            .arg(limit.to_string())
            .arg(format!("--prompt={prompt}"))
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .env_remove("PEREGRINE_API_KEY");
        let output = run_worker(&mut decoder, None)
            .await
            .map_err(|error| {
                eprintln!("peregrine-serve: Nina decoder: {error}");
                ApiError::internal("Peregrine could not start the Nina decoder")
            })?;
        if !output.status.success() {
            eprintln!(
                "peregrine-serve: Nina decoder exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
                    .chars()
                    .take(512)
                    .collect::<String>()
            );
            return Err(ApiError::internal("Peregrine's Nina decoder failed"));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            ApiError::internal(format!("Nina decoder output is not UTF-8: {error}"))
        })?;
        let (text, generated_tokens) = decode_chat(prompt, limit, &stdout)?;
        let details = reply
            .details
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("invalid Nina checkpoint metadata"))?;
        let generation = details
            .get_mut("generation")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| ApiError::internal("missing generation metadata from Omni"))?;
        generation.insert("generated_tokens".into(), json!(generated_tokens));
        generation.insert("executor".into(), json!("peregrine"));
        // A caller-requested token limit can end a sentence while it is still being generated.
        let should_check_prose = generated_tokens < limit || requested.is_none();
        if let Some(question) = details
            .remove("prose_question")
            .filter(|_| should_check_prose)
        {
            let question = question
                .as_str()
                .ok_or_else(|| ApiError::internal("invalid Nina prose question from Omni"))?;
            if let Some(reason) = self.prose_issue(question, &text).await? {
                details.insert("selection".into(), json!("abstained"));
                details.insert("answer".into(), Value::Null);
                details.insert("reason".into(), json!(reason));
                reply.text = "I couldn't produce a reliable answer to that question.".into();
                return Ok("stop");
            }
        }
        if text.is_empty() {
            details.insert("answer".into(), Value::Null);
            reply.text = "Nina's chat checkpoint returned an empty response.".into();
        } else {
            details.insert("answer".into(), json!(text));
            reply.text = text;
        }
        Ok(if generated_tokens == limit {
            "length"
        } else {
            "stop"
        })
    }

    async fn prose_issue(&self, question: &str, text: &str) -> Result<Option<String>, ApiError> {
        let mut checker = Command::new("python3");
        checker
            .arg("-B")
            .arg(&self.prose_checker)
            .arg("--check-prose")
            .arg(&self.science_index)
            .kill_on_drop(true)
            .env_remove("PEREGRINE_API_KEY");
        let input = json!({"question": question, "answer": text}).to_string();
        let output = run_worker(&mut checker, Some(input.as_bytes()))
            .await
            .map_err(|error| {
                eprintln!("peregrine-serve: Nina prose checker: {error}");
                ApiError::internal("Nina prose check could not complete")
            })?;
        if !output.status.success() {
            eprintln!(
                "peregrine-serve: Nina prose checker exited {}",
                output.status
            );
            return Err(ApiError::internal("Nina prose check failed"));
        }
        serde_json::from_slice(&output.stdout)
            .map_err(|error| ApiError::internal(format!("invalid Nina prose check reply: {error}")))
    }
}

fn decode_chat(prompt: &str, limit: usize, stdout: &str) -> Result<(String, usize), ApiError> {
    let (_, report) = stdout
        .split_once("\ndecoder=greedy+kv-cache | ")
        .ok_or_else(|| ApiError::internal("missing Nina decoder report"))?;
    let (metrics, body) = report
        .split_once("\n---\n")
        .ok_or_else(|| ApiError::internal("missing Nina output marker"))?;
    let count = metrics
        .split_once(" tokens in ")
        .and_then(|(count, _)| count.parse::<usize>().ok())
        .filter(|count| *count <= limit)
        .ok_or_else(|| ApiError::internal("invalid Nina token count"))?;
    let text = body
        .strip_prefix(prompt)
        .and_then(|text| text.strip_suffix("\n---\n"))
        .ok_or_else(|| {
            ApiError::internal("Nina decoder did not return the expected continuation")
        })?;
    Ok((text.trim().to_owned(), count))
}

#[derive(Deserialize)]
struct Request {
    model: String,
    #[serde(flatten)]
    chat: ChatRequest,
}

enum WorkerInput {
    Prompt(String),
    Chat(Vec<Value>),
}

impl Request {
    fn input(&self, accepts_history: bool) -> Result<WorkerInput, ApiError> {
        if self
            .chat
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        {
            return Err(ApiError::bad_request(
                "the Omni suite accepts questions and model inputs",
            ));
        }
        let messages = self.chat.messages.as_slice();
        // The plain invocation protocol also works with older Omni bundles.
        if !accepts_history {
            if let [message] = messages {
                let text = message.text();
                if message.role == "user"
                    && message.tool_calls.as_ref().is_none_or(Vec::is_empty)
                    && valid_text(&text, text.len())
                {
                    return Ok(WorkerInput::Prompt(text));
                }
            }
            return Err(ApiError::bad_request(format!(
                "model expects one nonempty user message of up to {MAX_INPUT_BYTES} UTF-8 bytes without NUL"
            )));
        }
        if messages.len() > MAX_MESSAGES || messages.last().is_none_or(|msg| msg.role != "user") {
            return Err(ApiError::bad_request(format!(
                "chat requires a final user message and at most {MAX_MESSAGES} messages"
            )));
        }
        let mut expected = "user";
        let mut bytes = 0;
        let mut turns = Vec::with_capacity(messages.len());
        for (index, message) in messages.iter().enumerate() {
            let text = message.text();
            bytes += text.len();
            if !valid_text(&text, bytes)
                || message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
            {
                return Err(ApiError::bad_request(format!(
                    "chat requires nonempty text messages without NUL totaling at most {MAX_INPUT_BYTES} UTF-8 bytes"
                )));
            }
            if message.role == expected {
                expected = if expected == "user" {
                    "assistant"
                } else {
                    "user"
                };
            } else if index != 0 || message.role != "system" {
                return Err(ApiError::bad_request("chat requires alternating user and assistant messages after an optional system message"));
            }
            turns.push(json!({"role": message.role, "content": text}));
        }
        Ok(WorkerInput::Chat(turns))
    }
}

fn valid_text(text: &str, total_bytes: usize) -> bool {
    total_bytes <= MAX_INPUT_BYTES && !text.trim().is_empty() && !text.contains('\0')
}

#[derive(Deserialize)]
struct ModelReply {
    model: String,
    text: String,
    details: Value,
    native_prompt: Option<String>,
}

impl Suite {
    fn worker(&self) -> Command {
        let mut command = Command::new(&self.binary);
        command
            .args(["suite", "--bundle"])
            .arg(&self.root)
            .arg("--artifacts")
            .arg(&self.artifacts)
            .env_remove("PEREGRINE_API_KEY")
            .kill_on_drop(true);
        command
    }

    async fn complete(
        &self,
        model: &str,
        input: WorkerInput,
        max_tokens: Option<usize>,
    ) -> Result<(ModelReply, &'static str), ApiError> {
        let permit = self.permits.try_acquire().map_err(|error| {
            peregrine_io::note_advisory_err("Omni worker admission", &error);
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "overloaded_error",
                "all Omni workers are busy",
            )
        })?;
        let mut worker = self.worker();
        worker.args(["invoke", model]);
        if self.chat.is_some() {
            worker.arg("--defer-chat");
        }
        let input = match input {
            WorkerInput::Prompt(text) => text.into_bytes(),
            WorkerInput::Chat(messages) => {
                worker.arg("--messages");
                serde_json::to_vec(&messages)
                    .map_err(|error| ApiError::internal(format!("encode Omni messages: {error}")))?
            }
        };
        if let Some(path) = &self.store {
            worker.arg("--store").arg(path);
        }
        let inference = async {
            let output = run_worker(&mut worker, Some(&input)).await.map_err(|error| {
                eprintln!("peregrine-serve: Omni worker: {error}");
                ApiError::internal("Omni worker could not complete")
            })?;
            let mut reply = decode_reply(model, output)?;
            let mut finish = "stop";
            if let Some(prompt) = reply.native_prompt.take() {
                let chat = self.chat.as_ref().ok_or_else(|| {
                    ApiError::internal("Peregrine has no registered Nina chat checkpoint")
                })?;
                finish = chat
                    .complete(&mut reply, &prompt, max_tokens, self.max_tokens)
                    .await?;
            }
            Ok((reply, finish))
        };
        let reply = tokio::time::timeout(DEADLINE, inference)
            .await
            .map_err(|error| {
                peregrine_io::note_advisory_err("Omni inference", &error);
                ApiError::new(
                    StatusCode::GATEWAY_TIMEOUT,
                    "timeout_error",
                    "Peregrine exceeded its inference deadline",
                )
            })?;
        drop(permit);
        reply
    }
}

async fn run_worker(worker: &mut Command, input: Option<&[u8]>) -> std::io::Result<Output> {
    use std::io::Error as IoError;

    let stdin_mode = if input.is_some() { Stdio::piped() } else { Stdio::null() };
    let mut child = worker
        .stdin(stdin_mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take().ok_or_else(|| IoError::other("worker stdout unavailable"))?;
    let stderr = child.stderr.take().ok_or_else(|| IoError::other("worker stderr unavailable"))?;
    let send_input = async move {
        let result = match input {
            Some(bytes) => {
                let mut pipe = stdin.ok_or_else(|| IoError::other("worker stdin unavailable"))?;
                pipe.write_all(bytes).await
            }
            None => Ok(()),
        };
        // Read the exit status even if the worker closed stdin early.
        Ok::<_, IoError>(result)
    };
    match tokio::try_join!(
        child.wait(),
        read_bounded(stdout, MAX_WORKER_STDOUT, "stdout"),
        read_bounded(stderr, MAX_WORKER_STDERR, "stderr"),
        send_input,
    ) {
        Ok((status, stdout, stderr, written)) => {
            if status.success() {
                written?;
            }
            Ok(Output { status, stdout, stderr })
        }
        Err(error) => {
            if let Err(cleanup) = child.kill().await {
                peregrine_io::note_advisory_err("terminate Omni child", &cleanup);
            }
            Err(error)
        }
    }
}

async fn read_bounded(
    pipe: impl AsyncRead + Unpin,
    limit: usize,
    name: &str,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.take(limit as u64 + 1).read_to_end(&mut bytes).await?;
    if bytes.len() > limit {
        return Err(std::io::Error::other(format!("worker {name} exceeds {limit} bytes")));
    }
    Ok(bytes)
}

fn decode_reply(model: &str, output: Output) -> Result<ModelReply, ApiError> {
    if !output.status.success() {
        let detail: String = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(512)
            .collect();
        if output.status.code() == Some(2) {
            return Err(ApiError::bad_request(detail.trim()));
        }
        eprintln!(
            "peregrine-serve: Omni worker exited {}: {}",
            output.status,
            detail.trim()
        );
        return Err(ApiError::internal("Omni model failed"));
    }
    let reply: ModelReply = serde_json::from_slice(&output.stdout)
        .map_err(|error| ApiError::internal(format!("invalid Omni reply: {error}")))?;
    if reply.model != model {
        return Err(ApiError::internal("Omni reply named a different model"));
    }
    Ok(reply)
}

async fn health(State(suite): State<Arc<Suite>>) -> Json<Value> {
    Json(json!({"status": "ok", "backend": "omni", "models": suite.models.len()}))
}

async fn models(State(suite): State<Arc<Suite>>) -> Result<Json<ModelList>, ApiError> {
    Ok(Json(ModelList {
        object: "list",
        data: suite
            .models
            .keys()
            .map(|id| ModelCard {
                id: id.clone(),
                object: "model",
                owned_by: "triskele",
            })
            .collect(),
    }))
}

async fn completions(
    State(suite): State<Arc<Suite>>,
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Response, ApiError> {
    let request = http::json_body(body)?;
    let Some(route) = suite.models.get(&request.model) else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "unknown Omni model",
        ));
    };
    let max_tokens = request.chat.token_limit()?;
    let accepts_history = suite.conversations && route.accepts_history();
    let input = request.input(accepts_history)?;
    let (reply, finish) = suite
        .complete(&request.model, input, max_tokens)
        .await?;
    let id = format!("chatcmpl-{}", seed());
    let created = unix_seconds();
    if request.chat.stream {
        let events = [
            chunk_event(&id, &reply.model, created, None, Some("assistant"), None),
            chunk_event(&id, &reply.model, created, Some(&reply.text), None, None),
            chunk_event(&id, &reply.model, created, None, None, Some(finish)),
            Event::default().data("[DONE]"),
        ];
        return Ok(Sse::new(tokio_stream::iter(
            events.map(Ok::<_, std::convert::Infallible>),
        ))
        .into_response());
    }
    Ok(Json(json!({
        "id": id, "created": created, "object": "chat.completion", "model": reply.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply.text}, "finish_reason": finish}],
        "omni": reply.details
    })).into_response())
}

pub(super) async fn serve(args: &Args, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=64).contains(&args.max_batch) {
        return Err(
            Error::Format("Omni serving requires --max-batch between 1 and 64".into()).into(),
        );
    }
    let root = canonical_path(root, "Omni suite")?;
    let artifacts = args
        .suite_artifacts
        .clone()
        .unwrap_or_else(|| root.join("artifacts"));
    let artifacts = canonical_path(&artifacts, "suite artifacts")?;
    let binary = args
        .omni_bin
        .clone()
        .unwrap_or_else(|| artifacts.join("runtime/omni"));
    let binary = canonical_path(&binary, "suite Omni executable")?;
    let store = args
        .context_store
        .as_deref()
        .map(|path| canonical_path(path, "suite context store"))
        .transpose()?;
    let mut suite = Suite {
        root,
        binary,
        artifacts,
        store,
        models: BTreeMap::new(),
        conversations: false,
        chat: None,
        max_tokens: args.max_tokens,
        permits: Semaphore::new(args.max_batch),
    };
    let mut worker = suite.worker();
    worker.args(["list", "--json"]);
    let listed = tokio::time::timeout(PROBE_DEADLINE, run_worker(&mut worker, None)).await??;
    if !listed.status.success() {
        return Err(Error::Format(format!(
            "listing Omni models: {}",
            String::from_utf8_lossy(&listed.stderr).trim()
        ))
        .into());
    }
    suite.models = serde_json::from_slice(&listed.stdout)?;
    suite.models.retain(|id, route| {
        !id.to_ascii_lowercase().contains("offens")
            && !route.family().to_ascii_lowercase().contains("offens")
    });
    if suite.models.is_empty() {
        return Err(Error::Format("Omni suite has no installed models".into()).into());
    }
    if suite.models.values().any(ModelRoute::accepts_history) {
        let mut worker = suite.worker();
        worker.args(["invoke", "--help"]);
        let help = tokio::time::timeout(PROBE_DEADLINE, run_worker(&mut worker, None)).await??;
        if help.status.success() {
            let help = String::from_utf8_lossy(&help.stdout);
            suite.conversations = help.split_whitespace().any(|word| word == "--messages");
            if suite.artifacts.join("chat/model.json").is_file()
                && help.split_whitespace().any(|word| word == "--defer-chat")
            {
                suite.chat = Some(NativeChat::load(&suite.root, &suite.artifacts)?);
            }
        }
    }
    let api_key = args.api_key.as_deref().map(Arc::<str>::from);
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completions))
        .route_layer(middleware::from_fn_with_state(api_key, http::authorize))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(Arc::new(suite));
    let listener = bind(&args.host, args.port).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn canonical_path(path: &Path, label: &str) -> Result<PathBuf, Error> {
    path.canonicalize()
        .map_err(|error| Error::Format(format!("{label} '{}': {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_user_message_uses_the_original_worker_protocol() -> serde_json::Result<()> {
        let request: Request = serde_json::from_value(json!({
            "model": "nina",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi Nina"}]}],
        }))?;

        assert!(matches!(request.input(false), Ok(WorkerInput::Prompt(text)) if text == "Hi Nina"));
        assert!(matches!(
            request.input(true),
            Ok(WorkerInput::Chat(turns)) if turns == [json!({"role": "user", "content": "Hi Nina"})]
        ));
        Ok(())
    }

    #[test]
    fn chat_routes_receive_history_in_order() -> serde_json::Result<()> {
        let messages = json!([
            {"role": "system", "content": "Keep it short."},
            {"role": "user", "content": "Name a page."},
            {"role": "assistant", "content": "Orbit"},
            {"role": "user", "content": "What was the name?"},
        ]);
        let request: Request =
            serde_json::from_value(json!({"model": "nina-chat", "messages": messages}))?;

        assert!(matches!(
            request.input(true),
            Ok(WorkerInput::Chat(turns)) if json!(turns) == messages
        ));
        assert!(matches!(
            request.input(false),
            Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn chat_input_limit_counts_utf8_bytes_across_turns() -> serde_json::Result<()> {
        let request: Request = serde_json::from_value(json!({
            "model": "nina-chat",
            "messages": [
                {"role": "user", "content": "é".repeat(MAX_INPUT_BYTES / 2)},
                {"role": "assistant", "content": "Orbit"},
                {"role": "user", "content": "What was the name?"},
            ],
        }))?;

        assert!(matches!(
            request.input(true),
            Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn native_decoder_preserves_multiline_output_and_reports_token_count() {
        let prompt = "User: show a markdown rule\nAssistant: ";
        let output = format!("loaded weights\ndecoder=greedy+kv-cache | 5 tokens in 10ms\n---\n{prompt}---\nbody\n---\n");
        assert!(matches!(decode_chat(prompt, 5, &output), Ok((text, 5)) if text == "---\nbody"));
    }

    #[test]
    fn independently_declared_chat_models_accept_history() -> serde_json::Result<()> {
        let routes: BTreeMap<String, ModelRoute> = serde_json::from_value(json!({
            "legacy": "causal chat generation",
            "conversation": {"family": "custom text model", "input": "messages"},
            "single-turn": {"family": "custom text model"}
        }))?;
        assert!(routes["legacy"].accepts_history());
        assert!(routes["conversation"].accepts_history());
        assert!(!routes["single-turn"].accepts_history());
        Ok(())
    }
}
