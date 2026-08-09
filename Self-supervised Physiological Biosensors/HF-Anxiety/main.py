import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from huggingface_hub import hf_hub_download

# -------------------------------------------------------------
# STEP 1: PYTORCH MODEL CLASSES (Unchanged, fully functional)
# -------------------------------------------------------------

class MaskedLSTMAutoEncoder(nn.Module):
    def __init__(self, n_features=10, hidden_size=64, n_layers=1):
        super(MaskedLSTMAutoEncoder, self).__init__()
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.T           = 5

        self.encoder     = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder     = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        bottleneck   = h_n[-1]
        decoder_input = bottleneck.unsqueeze(1).repeat(1, self.T, 1)
        decoder_out, _ = self.decoder(decoder_input)
        return self.output_layer(decoder_out)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]


class Seq2SeqForecaster(nn.Module):
    def __init__(self, embed_dim=64, enc_hidden=128, dec_hidden=128, n_layers=1, forecast_steps=10):
        super(Seq2SeqForecaster, self).__init__()
        self.forecast_steps = forecast_steps
        self.dec_hidden     = dec_hidden
        self.n_layers       = n_layers

        self.encoder    = nn.LSTM(input_size=embed_dim, hidden_size=enc_hidden, num_layers=n_layers, batch_first=True)
        self.bridge_h   = nn.Linear(enc_hidden, dec_hidden)
        self.bridge_c   = nn.Linear(enc_hidden, dec_hidden)
        self.decoder    = nn.LSTM(input_size=1, hidden_size=dec_hidden, num_layers=n_layers, batch_first=True)
        self.output_proj = nn.Sequential(nn.Linear(dec_hidden, 1), nn.Softplus())

    def forward(self, x_emb, y_target=None, teacher_forcing_ratio=0.5):
        batch_size = x_emb.size(0)
        _, (h_n, c_n) = self.encoder(x_emb)
        h_dec = torch.tanh(self.bridge_h(h_n))
        c_dec = torch.tanh(self.bridge_c(c_n))
        dec_input = torch.zeros(batch_size, 1, 1).to(x_emb.device)

        predictions = []
        for t in range(self.forecast_steps):
            dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            pred = self.output_proj(dec_out.squeeze(1))
            predictions.append(pred)
            if y_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y_target[:, t].unsqueeze(1).unsqueeze(2)
            else:
                dec_input = pred.unsqueeze(1).detach()
        return torch.cat(predictions, dim=1)

    def predict(self, x_emb):
        self.eval()
        with torch.no_grad():
            return self.forward(x_emb, y_target=None, teacher_forcing_ratio=0.0)


# -------------------------------------------------------------
# STEP 2: APP INITIALIZATION
# -------------------------------------------------------------

app = FastAPI(title="Physiological Escalation API")

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
FEATURE_COUNT = 10
FEATURE_NAMES = (
    "mean_hr",
    "mean_rr",
    "sdnn",
    "rmssd",
    "mean_br",
    "std_br",
    "mean_temp",
    "std_temp",
    "mean_acc_mag",
    "std_acc_mag",
)
NORM_MEAN_FIELDS = tuple(f"{name}_baseline_mean" for name in FEATURE_NAMES)
NORM_STD_FIELDS = tuple(f"{name}_baseline_std" for name in FEATURE_NAMES)
DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD = 0.25
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")

INFLUX_URL    = os.getenv("INFLUX_URL",    "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN")
INFLUX_ORG    = os.getenv("INFLUX_ORG",   "Dewdu")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "sensor_data")

db_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api  = db_client.write_api(write_options=SYNCHRONOUS)
query_api  = db_client.query_api()

# Hugging Face Vault settings for permanent model weight storage
HF_TOKEN        = os.getenv("HF_TOKEN")
HF_WEIGHTS_REPO = os.getenv("HF_WEIGHTS_REPO", "Dewdu/physiological-anxiety-weights")

device = torch.device('cpu')

# The global default champion model for cold-starts and unpersonalized users
global_ae_model = MaskedLSTMAutoEncoder(n_features=10, hidden_size=64, n_layers=1)
global_ae_model.load_state_dict(torch.load(
    MODEL_DIR / "MASKED_LSTM_AE_LOSO_S10.pth",
    map_location=device,
))
global_ae_model.eval()

# Our fast in-memory storage to keep personalized user models alive in RAM
user_model_cache = {}

forecaster = Seq2SeqForecaster(embed_dim=64, enc_hidden=128, dec_hidden=128, n_layers=1, forecast_steps=10)
forecaster.load_state_dict(torch.load(
    MODEL_DIR / "SEQ2SEQ_LOSO_S11.pth",
    map_location=device,
))
forecaster.eval()


# -------------------------------------------------------------
# STEP 3: PYDANTIC MODELS
# -------------------------------------------------------------

class ChestStrapFeaturesPayload(BaseModel):
    user_id: str
    timestamp: datetime
    is_worn: bool
    mean_hr: float
    mean_rr: float
    sdnn: float
    rmssd: float
    mean_br: float
    std_br: float
    mean_temp: float
    std_temp: float
    mean_acc_mag: float
    std_acc_mag: float



class NormParamsPayload(BaseModel):
    """
    Per-user baseline normalization parameters, computed from the user's
    baseline windows exactly as in the notebook:
        b_mean = np.mean(base_feats, axis=0)
        b_std  = np.std(base_feats,  axis=0)
        b_std[b_std == 0] = 1e-8
    Both arrays must have exactly 10 values — one per feature:
    [mean_HR, mean_RR, SDNN, RMSSD, mean_BR, std_BR, mean_temp, std_temp, mean_acc_mag, std_acc_mag]
    """
    b_mean: list[float]
    b_std:  list[float]
    baseline_windows: list[list[float]] | None = None


class AnxietyFeedbackPayload(BaseModel):
    user_id: str
    event_id: str
    detected_at: datetime
    initial_risk_score: float
    initial_hr: float | None = None
    initial_br: float | None = None
    initial_motion: float | None = None
    risk_source: str = "physiological"
    confirmed_anxious: bool | None = None
    activity: str | None = None
    intervention: str | None = None
    intervention_at: datetime | None = None
    intervention_completed: bool | None = None
    alternative_action: str | None = None
    followup_at: datetime | None = None
    followup_risk_score: float | None = None
    followup_hr: float | None = None
    followup_br: float | None = None
    followup_motion: float | None = None
    felt_better: bool | None = None


# -------------------------------------------------------------
# STEP 4: ENDPOINTS
# -------------------------------------------------------------

def validate_user_id(user_id: str) -> None:
    if not USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "user_id must be 1-128 characters and contain only letters, "
                "numbers, underscores, full stops, or hyphens."
            ),
        )


def _record_feature(record: dict, index: int) -> float | None:
    """Read the descriptive field name, falling back to legacy f_0..f_9."""
    value = record.get(FEATURE_NAMES[index])
    if value is None:
        value = record.get(f"f_{index}")
    return value


def _record_norm_value(record: dict, index: int, *, std: bool) -> float | None:
    fields = NORM_STD_FIELDS if std else NORM_MEAN_FIELDS
    value = record.get(fields[index])
    if value is None:
        legacy_prefix = "std" if std else "mean"
        value = record.get(f"{legacy_prefix}_{index}")
    return value


def _risk_from_reconstruction_error(error: float, threshold: float) -> float:
    """Map model error to a user-facing risk index using calibration anchors.

    The result is an index from 0 to 100, not a diagnostic probability.
    The user's 90th-percentile calm calibration error maps to the High boundary
    (70), while lower anchors preserve the app's Low/Moderate/Elevated tiers.
    """
    safe_threshold = max(float(threshold), 1e-6)
    return float(np.interp(
        max(float(error), 0.0),
        [
            0.0,
            safe_threshold * 0.40,
            safe_threshold * 0.70,
            safe_threshold,
            safe_threshold * 1.60,
        ],
        [0.0, 20.0, 45.0, 70.0, 100.0],
    ))


def _physiological_risk_from_features(row: list[float]) -> float:
    """Mirror the app's transparent current-signal risk index for history."""
    mean_hr, _, _, rmssd, mean_br, _, mean_temp, _, _, _ = row

    if mean_hr > 110:
        hr_score = 100.0
    elif mean_hr > 90:
        hr_score = 40.0 + (mean_hr - 90.0) / 20.0 * 40.0
    elif mean_hr > 70:
        hr_score = (mean_hr - 70.0) / 20.0 * 40.0
    else:
        hr_score = 0.0

    if mean_br > 26:
        br_score = 100.0
    elif mean_br > 20:
        br_score = 40.0 + (mean_br - 20.0) / 6.0 * 40.0
    elif mean_br > 16:
        br_score = (mean_br - 16.0) / 4.0 * 40.0
    else:
        br_score = 0.0

    temp_deviation = abs(mean_temp - 36.75)
    if temp_deviation > 0.6:
        temp_score = 100.0
    elif temp_deviation > 0.3:
        temp_score = (temp_deviation - 0.3) / 0.3 * 50.0 + 50.0
    else:
        temp_score = 0.0

    if rmssd >= 40:
        hrv_score = 0.0
    elif rmssd >= 20:
        hrv_score = (40.0 - rmssd) / 20.0 * 50.0
    else:
        hrv_score = min(50.0 + (20.0 - rmssd) / 20.0 * 50.0, 100.0)

    return float(np.clip(
        hr_score * 0.35
        + br_score * 0.25
        + temp_score * 0.15
        + hrv_score * 0.25,
        0.0,
        100.0,
    ))

@app.get("/")
def home():
    return {"status": "running", "message": "Physiological Escalation API is fully operational"}


@app.post("/set_norm_params/{user_id}")
def store_norm_params(user_id: str, payload: NormParamsPayload):
    """
    Store per-user baseline normalization parameters in InfluxDB.
    Must be called once per user (after their baseline session) before /predict
    will return valid results. Equivalent to saving SX_norm_params_mean.npy and
    SX_norm_params_std.npy in the notebook pipeline.
    """
    validate_user_id(user_id)

    if len(payload.b_mean) != FEATURE_COUNT or len(payload.b_std) != FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail="b_mean and b_std must each contain exactly 10 values (one per feature)."
        )

    b_mean = np.asarray(payload.b_mean, dtype=np.float64)
    b_std = np.asarray(payload.b_std, dtype=np.float64)
    if not np.isfinite(b_mean).all() or not np.isfinite(b_std).all():
        raise HTTPException(
            status_code=400,
            detail="b_mean and b_std must contain only finite numbers.",
        )
    if (b_std <= 0).any():
        raise HTTPException(
            status_code=400,
            detail="Every b_std value must be greater than zero.",
        )

    reconstruction_threshold = DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD
    if payload.baseline_windows is not None:
        baseline_windows = np.asarray(payload.baseline_windows, dtype=np.float32)
        if (
            baseline_windows.ndim != 2
            or baseline_windows.shape[1] != FEATURE_COUNT
            or baseline_windows.shape[0] < 1
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "baseline_windows must contain at least one row of "
                    "exactly 10 physiological features."
                ),
            )
        if not np.isfinite(baseline_windows).all():
            raise HTTPException(
                status_code=400,
                detail="baseline_windows must contain only finite numbers.",
            )

        normalized_baseline = (baseline_windows - b_mean) / b_std
        # Calibration currently yields three calm one-minute summaries. Pad
        # the earlier context with normalized baseline zeros so the five-step
        # AE receives the same cold-start structure used by /predict.
        normalized_baseline = np.vstack([
            np.zeros((global_ae_model.T - 1, FEATURE_COUNT), dtype=np.float32),
            normalized_baseline,
        ])
        baseline_errors = []
        with torch.no_grad():
            for start in range(len(normalized_baseline) - global_ae_model.T + 1):
                window = torch.tensor(
                    normalized_baseline[start : start + global_ae_model.T],
                    dtype=torch.float32,
                ).unsqueeze(0)
                reconstructed = global_ae_model(window)
                baseline_errors.append(
                    torch.mean((window - reconstructed) ** 2).item()
                )
        if baseline_errors:
            reconstruction_threshold = max(
                float(np.percentile(baseline_errors, 90)),
                1e-6,
            )

    point = Point("norm_params").tag("user_id", user_id)
    for i in range(FEATURE_COUNT):
        point.field(NORM_MEAN_FIELDS[i], float(b_mean[i]))
        point.field(NORM_STD_FIELDS[i], float(b_std[i]))
    point.field("reconstruction_error_p90", reconstruction_threshold)

    try:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "message": f"Norm params stored for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store norm params: {str(e)}")


@app.post("/ingest")
def process_and_ingest_raw_data(payload: ChestStrapFeaturesPayload):
    validate_user_id(payload.user_id)

    # --- NEW GUARD: Reject if not being worn ---
    if not payload.is_worn:
        raise HTTPException(
            status_code=400,
            detail="Data window rejected: Chest strap is currently not worn."
        )

    # --- E. BUILD AND VALIDATE FEATURE VECTOR ---
    # Feature order matches the notebook exactly:
    # [mean_HR, mean_RR, SDNN, RMSSD, mean_BR, std_BR, mean_temp, std_temp, mean_acc_mag, std_acc_mag]
    features = [
        payload.mean_hr, payload.mean_rr, payload.sdnn, payload.rmssd, 
        payload.mean_br, payload.std_br, payload.mean_temp, payload.std_temp, 
        payload.mean_acc_mag, payload.std_acc_mag
    ]

    # Reject NaN and infinity before they can poison normalization/inference.
    if not np.isfinite(np.asarray(features, dtype=np.float64)).all():
        raise HTTPException(
            status_code=400,
            detail="Data window rejected: Payload contains invalid NaN or Infinite numerical values."
        )

    try:
        # --- F. WRITE RAW FEATURES TO INFLUXDB ---
        point = (
            Point("physiological_metrics")
            .tag("user_id", payload.user_id)
            .time(payload.timestamp)
        )
        for field_name, value in zip(FEATURE_NAMES, features):
            point.field(field_name, float(value))

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "message": "Pre-calculated feature window successfully saved to InfluxDB"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database write failure: {str(e)}")


@app.get("/predict/{user_id}")
def get_escalation_forecast(user_id: str):
    validate_user_id(user_id)

    # FIX 5 — Fetch per-user baseline normalization parameters.
    # Equivalent to loading SX_norm_params_mean.npy and SX_norm_params_std.npy
    # in the notebook, and applying: normalized = (raw - b_mean) / b_std
    # (notebook lines 759–763 and 773).
    norm_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1y)
      |> filter(fn: (r) => r["_measurement"] == "norm_params")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''
    try:
        norm_tables  = query_api.query(norm_query)
        norm_records = [record.values for table in norm_tables for record in table.records]

        if not norm_records:
            return {
                "status": "not_calibrated",
                "message": (
                    f"No normalization params found for user {user_id}. "
                    f"POST b_mean and b_std to /set_norm_params/{user_id} "
                    f"before calling /predict."
                ),
                "forecast": []
            }

        b_mean = np.array(
            [
                _record_norm_value(norm_records[0], i, std=False)
                for i in range(FEATURE_COUNT)
            ],
            dtype=np.float32,
        )
        b_std = np.array(
            [
                _record_norm_value(norm_records[0], i, std=True)
                for i in range(FEATURE_COUNT)
            ],
            dtype=np.float32,
        )
        reconstruction_threshold = float(
            norm_records[0].get(
                "reconstruction_error_p90",
                DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD,
            )
        )

        if not np.isfinite(b_mean).all() or not np.isfinite(b_std).all():
            raise ValueError("Stored normalization parameters are incomplete or non-finite.")

        # Exact notebook guard: b_std[b_std == 0] = 1e-8
        b_std[b_std <= 0] = 1e-8

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve norm params: {str(e)}")

    # Fetch the last 19 minutes of raw feature windows
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -19m)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    try:
        tables  = query_api.query(query)
        records = []
        for table in tables:
            for record in table.records:
                row = [
                    _record_feature(record.values, i)
                    for i in range(FEATURE_COUNT)
                ]
                if None not in row and np.isfinite(
                    np.asarray(row, dtype=np.float64)
                ).all():
                    records.append(row)

        current_length = len(records)
        
        # Absolute safety net: If there is zero data in InfluxDB, we must wait for the first 60-second block
        if current_length == 0:
            return {
                "status": "buffering",
                "message": "Waiting for the first 60-second data block from your chest strap to arrive.",
                "forecast": []
            }

        # Pull whatever records are available, up to the last 19 minutes
        raw_sequence = np.array(records[-19:], dtype=np.float32)

        # Turn our raw data into normalized data using the user's custom baseline stats
        normalized_sequence = (raw_sequence - b_mean) / b_std

        # Smart Cold-Start Fallback: Pad missing history with baseline zeros (0.0 = calm state)
        if current_length < 19:
            needed_padding = 19 - current_length
            # Create a block of normal baseline zeros for the missing minutes across all 10 features
            padding_block = np.zeros((needed_padding, 10), dtype=np.float32)
            # Stack the calm padding at the front (past) and our live data at the back (present)
            normalized_sequence = np.vstack([padding_block, normalized_sequence])

        # --- DYNAMIC PERSONALIZATION LOGIC ---
        # Look into our memory cache first to see if this user's model is already in RAM
        if user_id in user_model_cache:
            active_ae_model = user_model_cache[user_id]
        else:
            personalized_weight_path = MODEL_DIR / f"{user_id}.pth"
            
            # If the file is not on the local hard drive, try to fetch it from your permanent Dataset vault
            if not personalized_weight_path.exists():
                try:
                    # Reach into your private dataset repository and pull down their specific .pth file
                    hf_hub_download(
                        repo_id=HF_WEIGHTS_REPO,
                        filename=f"{user_id}.pth",
                        repo_type="dataset",
                        local_dir=str(MODEL_DIR),
                        token=HF_TOKEN
                    )
                except Exception:
                    # If the file isn't in the vault (like for a brand new user), fail silently and use the fallback
                    pass

            # Check if the file exists now (either because it was already here, or we just successfully downloaded it)
            if personalized_weight_path.exists():
                try:
                    # Create a fresh model structure and load their custom weights
                    personalized_model = MaskedLSTMAutoEncoder(n_features=10, hidden_size=64, n_layers=1)
                    personalized_model.load_state_dict(torch.load(personalized_weight_path, map_location=device))
                    personalized_model.eval()
                    
                    # Store it in the cache memory so the next 60-second request is instant
                    user_model_cache[user_id] = personalized_model
                    active_ae_model = personalized_model
                except Exception:
                    # If the file exists but fails to load for any reason, use the global fallback
                    active_ae_model = global_ae_model
            else:
                # No personalized weights found anywhere (new user), use the global default model
                active_ae_model = global_ae_model

        embeddings_list = []
        observed_errors = []

        with torch.no_grad():
            for i in range(15):
                window        = normalized_sequence[i : i + 5]
                window_tensor = torch.tensor(
                    window,
                    dtype=torch.float32,
                ).unsqueeze(0)
                # We use active_ae_model here, which automatically points to the right weights
                emb           = active_ae_model.encode(window_tensor)
                embeddings_list.append(emb)
                reconstruction = active_ae_model(window_tensor)
                observed_errors.append(
                    torch.mean((window_tensor - reconstruction) ** 2).item()
                )

            lookback_tensor = torch.cat(embeddings_list, dim=0).unsqueeze(0)
            predictions     = forecaster.predict(lookback_tensor)

        raw_forecast = predictions.squeeze(0).cpu().numpy().astype(np.float64)

        # Anchor the trained forecast shape to the user's most recent observed
        # anomaly error. Without this correction, a model bias learned from the
        # training cohort can keep every line artificially low even when the
        # latest personalized reconstruction error is extreme.
        latest_error = float(observed_errors[-1])
        offset = latest_error - float(raw_forecast[0])
        decay = np.linspace(0.90, 0.20, len(raw_forecast))
        recent_slope = 0.0
        if len(observed_errors) >= 6:
            recent_slope = (
                float(np.mean(observed_errors[-3:]))
                - float(np.mean(observed_errors[-6:-3]))
            ) / 3.0
        adjusted_error_forecast = np.maximum(
            raw_forecast
            + offset * decay
            + recent_slope * np.arange(1, len(raw_forecast) + 1),
            0.0,
        )
        risk_forecast = [
            _risk_from_reconstruction_error(value, reconstruction_threshold)
            for value in adjusted_error_forecast
        ]

        return {
            "status": "success",
            "message": "Personalized physiological forecast ready.",
            "forecast": raw_forecast.tolist(),
            "adjusted_error_forecast": adjusted_error_forecast.tolist(),
            "risk_forecast": risk_forecast,
            "current_reconstruction_error": latest_error,
            "current_risk_index": _risk_from_reconstruction_error(
                latest_error,
                reconstruction_threshold,
            ),
            "reconstruction_error_threshold": reconstruction_threshold,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")


@app.get("/history/{user_id}")
def get_physiological_history(user_id: str, days: int = 30):
    validate_user_id(user_id)
    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail="days must be between 1 and 90.")

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''

    try:
        tables = query_api.query(query)
        daily_rows: dict[str, list[list[float]]] = defaultdict(list)
        for table in tables:
            for record in table.records:
                row = [
                    _record_feature(record.values, i)
                    for i in range(FEATURE_COUNT)
                ]
                if None in row:
                    continue
                values = np.asarray(row, dtype=np.float64)
                if not np.isfinite(values).all():
                    continue
                day = record.get_time().date().isoformat()
                daily_rows[day].append(values.tolist())

        history = []
        for day in sorted(daily_rows):
            rows = np.asarray(daily_rows[day], dtype=np.float64)
            means = rows.mean(axis=0).tolist()
            history.append({
                "date": day,
                "samples": int(len(rows)),
                "mean_hr": means[0],
                "mean_br": means[4],
                "mean_temp": means[6],
                "mean_motion": means[9],
                "risk_index": float(np.mean([
                    _physiological_risk_from_features(row.tolist())
                    for row in rows
                ])),
            })

        return {"status": "success", "days": days, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History query failed: {str(e)}")


@app.post("/feedback/anxiety")
def store_anxiety_feedback(payload: AnxietyFeedbackPayload):
    validate_user_id(payload.user_id)
    if not EVENT_ID_PATTERN.fullmatch(payload.event_id):
        raise HTTPException(status_code=400, detail="Invalid event_id.")
    if not 0.0 <= payload.initial_risk_score <= 100.0:
        raise HTTPException(
            status_code=400,
            detail="initial_risk_score must be between 0 and 100.",
        )

    point = (
        Point("anxiety_feedback")
        .tag("user_id", payload.user_id)
        .tag("event_id", payload.event_id)
        .time(payload.detected_at)
        .field("initial_risk_score", float(payload.initial_risk_score))
    )

    optional_fields = {
        "initial_hr": payload.initial_hr,
        "initial_br": payload.initial_br,
        "initial_motion": payload.initial_motion,
        "risk_source": payload.risk_source,
        "confirmed_anxious": payload.confirmed_anxious,
        "activity": payload.activity,
        "intervention": payload.intervention,
        "intervention_at": (
            payload.intervention_at.isoformat() if payload.intervention_at else None
        ),
        "intervention_completed": payload.intervention_completed,
        "alternative_action": payload.alternative_action,
        "followup_at": (
            payload.followup_at.isoformat() if payload.followup_at else None
        ),
        "followup_risk_score": payload.followup_risk_score,
        "followup_hr": payload.followup_hr,
        "followup_br": payload.followup_br,
        "followup_motion": payload.followup_motion,
        "felt_better": payload.felt_better,
    }
    for field_name, value in optional_fields.items():
        if value is not None:
            point.field(field_name, value)

    try:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "event_id": payload.event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback write failed: {str(e)}")


@app.get("/feedback/weekly/{user_id}")
def get_weekly_feedback_summary(user_id: str):
    validate_user_id(user_id)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "anxiety_feedback")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    try:
        records = [
            record.values
            for table in query_api.query(query)
            for record in table.records
        ]
        answered = [
            record for record in records
            if record.get("confirmed_anxious") is not None
        ]
        confirmed = [
            record for record in answered
            if record.get("confirmed_anxious") is True
        ]
        activities: dict[str, int] = defaultdict(int)
        effective_actions: dict[str, int] = defaultdict(int)
        for record in confirmed:
            activity = record.get("activity")
            if activity:
                activities[str(activity)] += 1

            improved = record.get("felt_better") is True
            if not improved:
                initial = record.get("initial_risk_score")
                followup = record.get("followup_risk_score")
                improved = (
                    initial is not None
                    and followup is not None
                    and float(followup) <= float(initial) - 10.0
                )
            if improved:
                action = (
                    record.get("intervention")
                    or record.get("alternative_action")
                )
                if action:
                    effective_actions[str(action)] += 1

        common_activity = (
            max(activities, key=activities.get) if activities else None
        )
        most_effective_action = (
            max(effective_actions, key=effective_actions.get)
            if effective_actions
            else None
        )
        confirmation_rate = (
            len(confirmed) / len(answered) if answered else None
        )
        return {
            "status": "success",
            "alerts": len(records),
            "answered_alerts": len(answered),
            "confirmed_alerts": len(confirmed),
            "false_positive_alerts": len(answered) - len(confirmed),
            "confirmation_rate": confirmation_rate,
            "common_activity": common_activity,
            "most_effective_action": most_effective_action,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Weekly feedback query failed: {str(e)}",
        )
