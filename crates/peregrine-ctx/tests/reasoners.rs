//! The HTTP reasoner against a mock server: proves the wire format end to end
//! (request shape, response parsing) without a model. The validation ladder's
//! first rung is plumbing, and plumbing gets a test, not a hope.

use std::io::{Read, Write};
use std::net::TcpListener;

use peregrine_ctx::llm::{HttpReasoner, Reasoner};

/// Serve exactly one chat-completion response; return the request it saw.
fn serve_once(response_body: &str) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = listener.local_addr().expect("addr").to_string();
    let body = response_body.to_string();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let request = read_request(&mut stream);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).expect("write");
        request
    });
    (endpoint, handle)
}

/// Read exactly one request: headers, then exactly Content-Length bytes of
/// body. Never read to EOF — the client holds the socket open waiting for
/// the response, so waiting for its close deadlocks both sides.
fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]);
    let length: usize = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().to_string())
        })
        .and_then(|v| v.parse().ok())
        .expect("content-length");
    while buf.len() < header_end + length {
        let n = stream.read(&mut chunk).expect("read body");
        buf.extend_from_slice(&chunk[..n]);
    }
    String::from_utf8_lossy(&buf).to_string()
}

#[test]
fn http_reasoner_posts_a_chat_completion_and_reads_the_answer() {
    let (endpoint, handle) = serve_once(
        r#"{"choices":[{"message":{"role":"assistant","content":"the answer is 42 [clai_1]"}}]}"#,
    );
    let mut reasoner = HttpReasoner::new(endpoint, "peregrine");
    let answer = reasoner.complete("QUESTION: what is the answer?", "you are the reasoner");
    let request = handle.join().expect("server thread");

    assert_eq!(answer, "the answer is 42 [clai_1]");
    assert!(
        request.starts_with("POST /v1/chat/completions HTTP/1.1"),
        "wrong request line: {request:.120}"
    );
    assert!(request.contains("what is the answer?"), "prompt not sent");
    assert!(request.contains("you are the reasoner"), "system not sent");
    assert!(request.contains("\"model\":\"peregrine\""), "model not sent");
}

#[test]
fn http_reasoner_reports_transport_failure_as_failure() {
    // Nothing listens here: the answer must say so, not read as a refusal.
    let mut reasoner = HttpReasoner::new("127.0.0.1:1", "peregrine");
    reasoner.timeout = std::time::Duration::from_secs(2);
    let answer = reasoner.complete("anything", "system");
    assert!(
        answer.starts_with("[reasoner 127.0.0.1:1 failed:"),
        "transport failure must be visible: {answer}"
    );
}

/// Serve one raw response, bytes exactly as given — for the framings
/// `serve_once`'s fixed header cannot produce.
fn serve_raw(response: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = listener.local_addr().expect("addr").to_string();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        stream.write_all(&response).expect("write");
    });
    (endpoint, handle)
}

#[test]
fn http_reasoner_decodes_a_chunked_response() {
    let json = r#"{"choices":[{"message":{"role":"assistant","content":"chunked answer"}}]}"#;
    // Two chunks, split mid-JSON so partial framing cannot parse by luck.
    let (first, second) = json.split_at(20);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{first}\r\n{:x}\r\n{second}\r\n0\r\n\r\n",
        first.len(),
        second.len()
    );
    let (endpoint, handle) = serve_raw(response.into_bytes());
    let mut reasoner = HttpReasoner::new(endpoint, "peregrine");
    let answer = reasoner.complete("QUESTION: anything", "system");
    handle.join().expect("server thread");
    assert_eq!(answer, "chunked answer");
}

#[test]
fn http_reasoner_names_an_http_error_instead_of_parsing_the_error_page() {
    let response = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 11\r\nConnection: close\r\n\r\nnot json at";
    let (endpoint, handle) = serve_raw(response.as_bytes().to_vec());
    let mut reasoner = HttpReasoner::new(endpoint.clone(), "peregrine");
    let answer = reasoner.complete("QUESTION: anything", "system");
    handle.join().expect("server thread");
    assert!(
        answer.contains("500 Internal Server Error"),
        "the status line must surface: {answer}"
    );
    assert!(
        answer.starts_with(&format!("[reasoner {endpoint} failed:")),
        "an HTTP error is a failure, not an answer: {answer}"
    );
}
