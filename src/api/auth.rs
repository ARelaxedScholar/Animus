use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use std::env;

pub async fn auth_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    // Skip auth for health check
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }

    let api_key = env::var("ANIMUS_API_KEY").unwrap_or_else(|_| "animus_dev_key".to_string());

    let auth_header = req.headers().get("X-API-Key");

    match auth_header {
        Some(val)
            if val == HeaderValue::from_str(&api_key).unwrap_or(HeaderValue::from_static("")) =>
        {
            Ok(next.run(req).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
