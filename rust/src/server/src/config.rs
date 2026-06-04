use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, bail};
use axum::http::{HeaderName, HeaderValue, Method};
use serde::Serialize;
use serde_json::Value;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp { host: String, port: u16 },
    /// Bind a fresh Unix domain listener on the given filesystem path.
    BindUnix { path: String },
    /// Adopt an already-open listening socket inherited from a supervisor
    /// process.
    InheritedFd { fd: i32 },
}

/// Which coordinator implementation should be active when one is present for a
/// frontend client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if
    /// there are multiple engines and the model is MoE.
    MaybeInProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// CORS policy for the HTTP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorsConfig {
    /// Whether browser clients may include credentials in cross-origin
    /// requests.
    pub allow_credentials: bool,
    /// Allowed origins. A single `"*"` matches any origin.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods. A single `"*"` matches any method.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers. A single `"*"` matches any header.
    pub allowed_headers: Vec<String>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allow_credentials: false,
            allowed_origins: wildcard_list(),
            allowed_methods: wildcard_list(),
            allowed_headers: wildcard_list(),
        }
    }
}

impl CorsConfig {
    /// Validate CORS values before server startup.
    pub fn validate(&self) -> Result<()> {
        if self.allow_credentials {
            if has_wildcard(&self.allowed_origins) {
                bail!("cannot use wildcard CORS origin when allow_credentials is true");
            }
            if has_wildcard(&self.allowed_methods) {
                bail!("cannot use wildcard CORS methods when allow_credentials is true");
            }
            if has_wildcard(&self.allowed_headers) {
                bail!("cannot use wildcard CORS headers when allow_credentials is true");
            }
        }

        validate_origins(&self.allowed_origins)?;
        validate_methods(&self.allowed_methods)?;
        validate_headers(&self.allowed_headers)?;
        Ok(())
    }
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Config {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Requested frontend-side coordinator behavior.
    pub coordinator_mode: CoordinatorMode,
    /// Backend model identifier used for engine-core loading.
    pub model: String,
    /// Model name(s) exposed to clients via the OpenAI API. When non-empty,
    /// the first entry is used as the primary ID in responses and all entries
    /// are accepted in requests. When empty, falls back to `model`.
    pub served_model_name: Vec<String>,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Tool-call parser selection.
    pub tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    pub reasoning_parser: ParserSelection,
    /// Chat renderer selection.
    pub renderer: RendererSelection,
    /// Server-default chat template override, as a file path or inline
    /// template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, set `X-Request-Id` on every HTTP response.
    pub enable_request_id_headers: bool,
    /// When `true`, suppress periodic stats logging (throughput, queue depth,
    /// cache usage).
    pub disable_log_stats: bool,
    /// CORS policy for browser-based clients.
    pub cors: CorsConfig,
    /// TCP port for the gRPC Generate service. When `None`, no gRPC server is
    /// started.
    pub grpc_port: Option<u16>,
    /// Maximum time to wait for active HTTP/gRPC requests to drain on shutdown.
    pub shutdown_timeout: Duration,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine
    /// startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;
        self.cors.validate()?;

        Ok(())
    }

    /// Return the number of engines implied by the configured transport mode.
    pub fn engine_count(&self) -> usize {
        match &self.transport_mode {
            TransportMode::HandshakeOwner { engine_count, .. }
            | TransportMode::Bootstrapped { engine_count, .. } => *engine_count,
        }
    }

    /// Resolve the effective coordinator mode.
    pub fn effective_coordinator_mode(
        &self,
        model_is_moe: bool,
    ) -> Option<EngineCoreCoordinatorMode> {
        match &self.coordinator_mode {
            CoordinatorMode::None => None,
            CoordinatorMode::MaybeInProc => {
                if model_is_moe && self.engine_count() > 1 {
                    Some(EngineCoreCoordinatorMode::InProc)
                } else {
                    None
                }
            }
            CoordinatorMode::External { address } => Some(EngineCoreCoordinatorMode::External {
                address: address.clone(),
            }),
        }
    }
}

fn wildcard_list() -> Vec<String> {
    vec!["*".to_string()]
}

fn has_wildcard(values: &[String]) -> bool {
    values.iter().any(|value| value == "*")
}

fn validate_origins(origins: &[String]) -> Result<()> {
    for origin in origins {
        if origin == "*" {
            continue;
        }
        HeaderValue::from_str(origin)
            .map_err(|err| anyhow::anyhow!("invalid CORS origin `{origin}`: {err}"))?;
    }
    Ok(())
}

fn validate_methods(methods: &[String]) -> Result<()> {
    for method in methods {
        if method == "*" {
            continue;
        }
        Method::from_bytes(method.as_bytes())
            .map_err(|err| anyhow::anyhow!("invalid CORS method `{method}`: {err}"))?;
    }
    Ok(())
}

fn validate_headers(headers: &[String]) -> Result<()> {
    for header in headers {
        if header == "*" {
            continue;
        }
        HeaderName::from_bytes(header.as_bytes())
            .map_err(|err| anyhow::anyhow!("invalid CORS header `{header}`: {err}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CorsConfig;

    #[test]
    fn cors_config_default_matches_python_defaults() {
        assert_eq!(
            CorsConfig::default(),
            CorsConfig {
                allow_credentials: false,
                allowed_origins: vec!["*".to_string()],
                allowed_methods: vec!["*".to_string()],
                allowed_headers: vec!["*".to_string()],
            }
        );
    }

    #[test]
    fn cors_config_rejects_wildcard_values_with_credentials() {
        let error = CorsConfig {
            allow_credentials: true,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["*".to_string()],
            allowed_headers: vec!["*".to_string()],
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot use wildcard CORS origin when allow_credentials is true"
        );

        let error = CorsConfig {
            allow_credentials: true,
            allowed_origins: vec!["https://example.com".to_string()],
            allowed_methods: vec!["*".to_string()],
            allowed_headers: vec!["authorization".to_string()],
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot use wildcard CORS methods when allow_credentials is true"
        );

        let error = CorsConfig {
            allow_credentials: true,
            allowed_origins: vec!["https://example.com".to_string()],
            allowed_methods: vec!["POST".to_string()],
            allowed_headers: vec!["*".to_string()],
        }
        .validate()
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot use wildcard CORS headers when allow_credentials is true"
        );
    }

    #[test]
    fn cors_config_rejects_invalid_values() {
        let error = CorsConfig {
            allow_credentials: false,
            allowed_origins: vec!["bad\norigin".to_string()],
            allowed_methods: vec!["GET".to_string()],
            allowed_headers: vec!["content-type".to_string()],
        }
        .validate()
        .unwrap_err();

        assert!(error.to_string().contains("invalid CORS origin"));
    }
}
