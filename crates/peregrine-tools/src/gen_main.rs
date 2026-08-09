//! `peregrine-gen` — watch token generation live, and time it.
//!
//! A streaming client for `peregrine-serve`'s `POST /v1/chat/completions`. It
//! prints the completion as it arrives and reports what the engine actually did:
//! time to first token, the per-token intervals, and their spread.
//!
//!   peregrine-gen "Explain how a mixture-of-experts layer routes a token."
//!   peregrine-gen --max-tokens 8 --json timings.json < prompt.txt
//!
//! **Text goes to stdout, statistics to stderr**, so a run can be piped without
//! the display contaminating it:
//!
//!   peregrine-gen "..." > completion.txt      # stats still visible on the terminal
//!
//! Three choices here are specific to this engine rather than generic:
//!
//! 1. **Seconds per token, not tokens per second.** Streaming experts off disk
//!    puts decode in the tens of seconds per token at GLM-5.2 shapes, where
//!    `tok/s` renders as `0.02` and carries no information. The unit flips
//!    automatically once generation is faster than 1 tok/s.
//! 2. **The live line is redrawn from its own thread.** The socket blocks for as
//!    long as a token takes, so a status line updated only on arrival would sit
//!    frozen for a minute at a time and look like a hang. It ticks instead, and
//!    shows how long the current token has been outstanding.
//! 3. **The inter-token *spread* is the headline, not the mean.** On a disk-bound
//!    MoE lane a token served from the warm cache and one that reads 11.3 GB of
//!    experts differ by orders of magnitude, so median and p95 say something the
//!    average hides. `--json` writes every interval for offline analysis.
//!
//! No new dependencies: raw HTTP/1.1 over `std::net::TcpStream`, including
//! chunked-transfer decoding, and `serde_json` (already a dependency) for the
//! event payloads.

use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn usage() -> i32 {
    eprintln!(
        "usage: peregrine-gen [options] [prompt ...]\n\
         \n\
         Prompt comes from the arguments, or from stdin when none are given.\n\
         \n\
         options:\n\
         \x20 --host <h>          server host                       (default 127.0.0.1)\n\
         \x20 --port <p>          server port                       (default 8137)\n\
         \x20 --model <id>        model id sent in the request      (default glm-5.2)\n\
         \x20 --max-tokens <n>    completion length cap             (default 64)\n\
         \x20 --temperature <f>   sampling temperature              (default 0.0)\n\
         \x20 --system <text>     prepend a system message\n\
         \x20 --json <file>       write per-token timing records as JSON\n\
         \x20 --quiet             no live status line (summary still printed)\n\
         \n\
         Text is written to stdout and statistics to stderr, so stdout can be\n\
         redirected on its own."
    );
    2
}

struct Opts {
    host: String,
    port: u16,
    model: String,
    max_tokens: u32,
    temperature: f64,
    system: Option<String>,
    json: Option<String>,
    quiet: bool,
    prompt: String,
}

fn parse_args() -> Result<Opts, i32> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut o = Opts {
        host: "127.0.0.1".into(),
        port: 8137,
        model: "glm-5.2".into(),
        max_tokens: 64,
        temperature: 0.0,
        system: None,
        json: None,
        quiet: false,
        prompt: String::new(),
    };
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0usize;
    // `next` exists so a missing value reports *which* flag was short rather than
    // silently consuming the following flag as its argument.
    macro_rules! next {
        ($flag:expr) => {{
            i += 1;
            match argv.get(i) {
                Some(v) => v.clone(),
                None => {
                    eprintln!("peregrine-gen: {} needs a value", $flag);
                    return Err(2);
                }
            }
        }};
    }
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--host" => o.host = next!("--host"),
            "--port" => match next!("--port").parse() {
                Ok(v) => o.port = v,
                Err(_) => {
                    eprintln!("peregrine-gen: --port must be a number");
                    return Err(2);
                }
            },
            "--model" => o.model = next!("--model"),
            "--max-tokens" => match next!("--max-tokens").parse() {
                Ok(v) => o.max_tokens = v,
                Err(_) => {
                    eprintln!("peregrine-gen: --max-tokens must be a number");
                    return Err(2);
                }
            },
            "--temperature" => match next!("--temperature").parse() {
                Ok(v) => o.temperature = v,
                Err(_) => {
                    eprintln!("peregrine-gen: --temperature must be a number");
                    return Err(2);
                }
            },
            "--system" => o.system = Some(next!("--system")),
            "--json" => o.json = Some(next!("--json")),
            "--quiet" => o.quiet = true,
            s if s.starts_with("--") => {
                eprintln!("peregrine-gen: unknown option {s}");
                return Err(2);
            }
            s => rest.push(s.to_string()),
        }
        i += 1;
    }
    o.prompt = if rest.is_empty() {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() || s.trim().is_empty() {
            eprintln!("peregrine-gen: no prompt given (arguments or stdin)");
            return Err(2);
        }
        s.trim().to_string()
    } else {
        rest.join(" ")
    };
    Ok(o)
}

/// `Transfer-Encoding: chunked` decoder over a buffered socket.
///
/// axum streams SSE chunked, so the hex size lines are interleaved with the
/// payload and scanning the raw socket for `data:` would parse framing bytes as
/// content. Small enough to carry rather than take an HTTP-client dependency.
struct Chunked<R: BufRead> {
    inner: R,
    remaining: usize,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done || buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut line = String::new();
            // Skips the CRLF that terminates the previous chunk's body as well as
            // the size line itself, so both are handled by one loop.
            loop {
                line.clear();
                if self.inner.read_line(&mut line)? == 0 {
                    self.done = true;
                    return Ok(0);
                }
                let t = line.trim();
                if t.is_empty() {
                    continue;
                }
                // A chunk size may carry `;ext=...` extensions; only the hex counts.
                let hex = t.split(';').next().unwrap_or("").trim();
                match usize::from_str_radix(hex, 16) {
                    Ok(n) => {
                        self.remaining = n;
                        break;
                    }
                    Err(_) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("bad chunk size {hex:?}"),
                        ))
                    }
                }
            }
            if self.remaining == 0 {
                self.done = true;
                return Ok(0);
            }
        }
        let want = buf.len().min(self.remaining);
        let n = self.inner.read(&mut buf[..want])?;
        if n == 0 {
            self.done = true;
            return Ok(0);
        }
        self.remaining -= n;
        Ok(n)
    }
}

/// `12.4 s` / `1m 42s` / `2h 05m` — a duration a human can read at a glance.
fn human(secs: f64) -> String {
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m {:02}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{}h {:02}m", (secs / 3600.0) as u64, ((secs % 3600.0) / 60.0) as u64)
    }
}

/// The rate, in whichever unit is informative at this speed. Below 1 tok/s —
/// which is where this engine lives when streaming experts off disk — `tok/s`
/// rounds to noise, so seconds per token leads and the reciprocal follows.
fn rate(tokens: usize, secs: f64) -> String {
    if tokens == 0 || secs <= 0.0 {
        return "n/a".into();
    }
    let tps = tokens as f64 / secs;
    if tps >= 1.0 {
        format!("{tps:.2} tok/s ({:.2} s/tok)", 1.0 / tps)
    } else {
        format!("{:.2} s/tok ({tps:.3} tok/s)", 1.0 / tps)
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> std::process::ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(c) => return std::process::ExitCode::from(c as u8),
    };

    let mut messages = Vec::new();
    if let Some(s) = &opts.system {
        messages.push(serde_json::json!({"role": "system", "content": s}));
    }
    messages.push(serde_json::json!({"role": "user", "content": opts.prompt}));
    let body = serde_json::json!({
        "model": opts.model,
        "messages": messages,
        "max_tokens": opts.max_tokens,
        "temperature": opts.temperature,
        "stream": true,
    })
    .to_string();

    let addr = format!("{}:{}", opts.host, opts.port);
    let stream = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "peregrine-gen: cannot reach {addr} ({e})\n\
                 Start the server first, e.g.\n\
                 \x20 target/release/peregrine-serve --model <dir> --port {}",
                opts.port
            );
            return std::process::ExitCode::FAILURE;
        }
    };
    // No read timeout on purpose: a single token legitimately takes tens of
    // seconds here, and any timeout short enough to catch a hang would also kill
    // healthy runs. The ticking status line is what distinguishes the two.
    let mut wsock = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("peregrine-gen: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: {}\r\n\
         Content-Type: application/json\r\n\
         Accept: text/event-stream\r\n\
         Connection: close\r\n\
         Content-Length: {}\r\n\r\n{}",
        opts.host,
        body.len(),
        body
    );
    if let Err(e) = wsock.write_all(req.as_bytes()).and_then(|_| wsock.flush()) {
        eprintln!("peregrine-gen: sending request: {e}");
        return std::process::ExitCode::FAILURE;
    }

    let t0 = Instant::now();
    let mut head = BufReader::new(stream);

    // Status line, then headers.
    let mut status = String::new();
    if head.read_line(&mut status).unwrap_or(0) == 0 {
        eprintln!("peregrine-gen: server closed without a response");
        return std::process::ExitCode::FAILURE;
    }
    let code = status.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut chunked = false;
    loop {
        let mut line = String::new();
        match head.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line.trim().is_empty() {
                    break;
                }
                let l = line.to_ascii_lowercase();
                if l.starts_with("transfer-encoding:") && l.contains("chunked") {
                    chunked = true;
                }
            }
            Err(e) => {
                eprintln!("peregrine-gen: reading headers: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    if code != "200" {
        let mut rest = String::new();
        let _ = head.read_to_string(&mut rest);
        eprintln!("peregrine-gen: server returned {}\n{}", status.trim(), rest.trim());
        return std::process::ExitCode::FAILURE;
    }

    // Live status. Its own thread because the read below blocks for as long as a
    // token takes, and a line that only moves when a token lands is
    // indistinguishable from a hung process at these latencies.
    let tokens = Arc::new(AtomicU64::new(0));
    let last_ms = Arc::new(AtomicU64::new(0));
    let ttft_ms = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    // Colour follows the terminal, not `--quiet`: `2> stats.txt` must capture
    // plain text, while a quiet run on a tty still reads well.
    let tty = std::io::stderr().is_terminal();
    let show_live = !opts.quiet && tty;
    let ticker = {
        let (tokens, last_ms, ttft_ms, done) =
            (tokens.clone(), last_ms.clone(), ttft_ms.clone(), done.clone());
        std::thread::spawn(move || {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut f = 0usize;
            while !done.load(Ordering::Relaxed) {
                if show_live {
                    let el = t0.elapsed().as_secs_f64();
                    let n = tokens.load(Ordering::Relaxed);
                    let since = el - last_ms.load(Ordering::Relaxed) as f64 / 1000.0;
                    let ttft = ttft_ms.load(Ordering::Relaxed);
                    let head = if n == 0 {
                        format!("prefill · waiting {}", human(el))
                    } else {
                        format!(
                            "{n} tok · {} · ttft {} · this token {}",
                            rate(n as usize, el - ttft as f64 / 1000.0),
                            human(ttft as f64 / 1000.0),
                            human(since)
                        )
                    };
                    let _ = write!(
                        std::io::stderr(),
                        "\r\x1b[2K\x1b[2m{}\x1b[0m {}  \x1b[2m[{}]\x1b[0m",
                        FRAMES[f % FRAMES.len()],
                        head,
                        human(el)
                    );
                    let _ = std::io::stderr().flush();
                }
                f += 1;
                std::thread::sleep(Duration::from_millis(120));
            }
        })
    };

    let mut reader: Box<dyn BufRead> = if chunked {
        Box::new(BufReader::new(Chunked { inner: head, remaining: 0, done: false }))
    } else {
        Box::new(head)
    };

    let mut text = String::new();
    let mut gaps: Vec<f64> = Vec::new();
    let mut records: Vec<serde_json::Value> = Vec::new();
    let mut prev = 0f64;
    let mut finish_reason = String::new();
    let mut line = String::new();
    let mut stdout = std::io::stdout();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                // Do not join here: the shutdown below owns the handle, and the
                // partial completion is still worth summarising.
                eprintln!("\rperegrine-gen: stream error: {e}");
                break;
            }
        }
        let l = line.trim_end();
        let Some(payload) = l.strip_prefix("data:") else { continue };
        let payload = payload.trim();
        if payload == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else { continue };
        let choice = &v["choices"][0];
        if let Some(fr) = choice["finish_reason"].as_str() {
            finish_reason = fr.to_string();
        }
        let Some(delta) = choice["delta"]["content"].as_str() else { continue };
        if delta.is_empty() {
            continue;
        }

        let now = t0.elapsed().as_secs_f64();
        let n = tokens.fetch_add(1, Ordering::Relaxed);
        // The first delta has no predecessor, so it contributes a ttft and not an
        // interval. Folding it into `gaps` would let prefill — tens of seconds of
        // work that happens once — masquerade as a decode interval and drag every
        // percentile with it.
        let gap = if n == 0 {
            ttft_ms.store((now * 1000.0) as u64, Ordering::Relaxed);
            None
        } else {
            let g = now - prev;
            gaps.push(g);
            Some(g)
        };
        last_ms.store((now * 1000.0) as u64, Ordering::Relaxed);
        prev = now;

        if opts.json.is_some() {
            records.push(serde_json::json!({
                "index": n,
                "at_s": (now * 1000.0).round() / 1000.0,
                "gap_s": match gap { Some(g) => serde_json::json!((g * 1000.0).round() / 1000.0),
                                     None => serde_json::Value::Null },
                "text": delta,
            }));
        }

        // Clear the status line before emitting text, or the two interleave on
        // the same row. stdout is flushed per delta so a pipe sees tokens live.
        if show_live {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
        let _ = stdout.write_all(delta.as_bytes());
        let _ = stdout.flush();
        text.push_str(delta);
    }

    done.store(true, Ordering::Relaxed);
    let _ = ticker.join();
    if show_live {
        let _ = write!(std::io::stderr(), "\r\x1b[2K");
    }
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();

    let total = t0.elapsed().as_secs_f64();
    let n = tokens.load(Ordering::Relaxed) as usize;
    let ttft = ttft_ms.load(Ordering::Relaxed) as f64 / 1000.0;
    let mut sorted = gaps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut err = std::io::stderr();
    let (d, r) = if tty { ("\x1b[2m", "\x1b[0m") } else { ("", "") };
    let _ = writeln!(err, "{d}──{r} peregrine-gen {d}{}{r}", "─".repeat(46));
    let _ = writeln!(err, "  generated    {n} tokens, {} chars", text.chars().count());
    let _ = writeln!(err, "  ttft         {}  {d}(prefill + first token){r}", human(ttft));
    let _ = writeln!(err, "  total        {}", human(total));
    if n > 1 {
        let decode = (total - ttft).max(0.0);
        let _ = writeln!(err, "  decode rate  {}  {d}(excludes ttft){r}", rate(n - 1, decode));
        let _ = writeln!(
            err,
            "  inter-token  min {}  p50 {}  p95 {}  max {}",
            human(sorted[0]),
            human(percentile(&sorted, 0.50)),
            human(percentile(&sorted, 0.95)),
            human(sorted[sorted.len() - 1]),
        );
        // The spread is the diagnostic: on a disk-bound MoE lane a warm-cache
        // token and one that streams a full routed union differ by an order of
        // magnitude, and an average hides exactly that.
        let ratio = sorted[sorted.len() - 1] / sorted[0].max(1e-9);
        if ratio >= 2.0 {
            let _ = writeln!(
                err,
                "  {d}spread       {ratio:.1}x slowest/fastest — uneven decode, \
                 consistent with cache hits vs full expert reads{r}"
            );
        }
    }
    if !finish_reason.is_empty() {
        let _ = writeln!(err, "  finish       {finish_reason}");
    }

    if let Some(path) = &opts.json {
        let doc = serde_json::json!({
            "model": opts.model,
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "prompt": opts.prompt,
            "ttft_s": (ttft * 1000.0).round() / 1000.0,
            "total_s": (total * 1000.0).round() / 1000.0,
            "tokens": n,
            "gaps_s": gaps.iter().map(|g| (g * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            "deltas": records,
            "text": text,
        });
        match serde_json::to_vec_pretty(&doc).map_err(|e| e.to_string()).and_then(|b| {
            std::fs::write(path, b).map_err(|e| e.to_string())
        }) {
            Ok(()) => {
                let _ = writeln!(err, "  timings      {path}");
            }
            Err(e) => {
                let _ = writeln!(err, "peregrine-gen: writing {path}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    if n == 0 {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
