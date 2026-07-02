use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct InitWeightTransferEngineRequest {
    init_info: Option<JsonValue>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UpdateWeightsRequest {
    update_info: Option<JsonValue>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MessageResponse {
    message: &'static str,
}

/// Initialize the RLHF weight-transfer engine with a backend-specific
/// `init_info` payload.
pub async fn init_weight_transfer_engine(
    State(state): State<Arc<AppState>>,
    body: Result<Json<InitWeightTransferEngineRequest>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let init_info = body.init_info.ok_or_else(|| {
        ApiError::invalid_request(
            "Missing 'init_info' in request body".to_string(),
            Some("init_info"),
        )
    })?;

    state
        .engine_core_client()
        .init_weight_transfer_engine(init_info)
        .await
        .map_err(|error| utility_call_error("init_weight_transfer_engine", error))?;

    Ok(Json(MessageResponse {
        message: "Weight transfer initialized",
    }))
}

/// Start a weight update on every connected engine.
pub async fn start_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    state
        .engine_core_client()
        .start_weight_update()
        .await
        .map_err(|error| utility_call_error("start_weight_update", error))?;

    Ok(Json(MessageResponse {
        message: "Weight update started",
    }))
}

/// Push a weight update carrying a backend-specific `update_info` payload.
pub async fn update_weights(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightsRequest>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let update_info = body.update_info.ok_or_else(|| {
        ApiError::invalid_request(
            "Missing 'update_info' in request body".to_string(),
            Some("update_info"),
        )
    })?;

    state
        .engine_core_client()
        .update_weights(update_info)
        .await
        .map_err(|error| utility_call_error("update_weights", error))?;

    Ok(Json(MessageResponse {
        message: "Weights updated",
    }))
}

/// Finish a weight update on every connected engine.
pub async fn finish_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    state
        .engine_core_client()
        .finish_weight_update()
        .await
        .map_err(|error| utility_call_error("finish_weight_update", error))?;

    Ok(Json(MessageResponse {
        message: "Weight update finished",
    }))
}
