use std::sync::Arc;

use axum::extract::{rejection::JsonRejection, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::Json;

use crate::{header_utf8, ApiError};

pub(super) async fn authorize(
    State(key): State<Option<Arc<str>>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if let Some(want) = key {
        let got = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(header_utf8)
            .and_then(|value| value.split_once(' '))
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, token)| token.trim_start());
        if !got.is_some_and(|token| constant_time_eq(token.as_bytes(), want.as_bytes())) {
            return Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                "missing or invalid API key",
            ));
        }
    }
    Ok(next.run(request).await)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub(super) fn json_body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    body.map(|Json(value)| value).map_err(|error| {
        ApiError::new(error.status(), "invalid_request_error", error.body_text())
    })
}
