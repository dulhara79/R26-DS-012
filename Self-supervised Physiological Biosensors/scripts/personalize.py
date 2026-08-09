#!/usr/bin/env python3
"""
personalize.py — Weekly Physiological Model Personalization Script
===================================================================
Runs every week automatically via GitHub Actions (configured in Step 10).

For every user who has physiological data in InfluxDB, this script:
  1. Pulls their last 7 days of feature windows
  2. Loads their personal baseline norm params and normalizes the data
  3. Loads their existing personalized model (or the global base model for new users)
  4. Fine-tunes the LSTM Autoencoder on their personal data
  5. Uploads the updated weights to Dewdu/physiological-anxiety-weights as {user_id}.pth

Your main.py in the HF Space already knows to download and use this file
automatically every time a user calls /predict.

Required Environment Variables (set in GitHub Actions secrets):
  INFLUX_URL     — e.g. https://us-east-1-1.aws.cloud2.influxdata.com
  INFLUX_TOKEN   — your InfluxDB API token
  INFLUX_ORG     — e.g. Dewdu
  INFLUX_BUCKET  — e.g. sensor_data
  HF_TOKEN       — your Hugging Face write token
"""

import os
import io
from datetime import timedelta
import torch
import torch.nn as nn
import numpy as np
from influxdb_client import InfluxDBClient
from huggingface_hub import hf_hub_download, HfApi

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# All values come from environment variables so no secrets are hardcoded here.
# ─────────────────────────────────────────────────────────────────────────────

INFLUX_URL    = os.environ["INFLUX_URL"]
INFLUX_TOKEN  = os.environ["INFLUX_TOKEN"]
INFLUX_ORG    = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]
HF_TOKEN      = os.environ["HF_TOKEN"]

# The HF Space that hosts your base model weights
HF_SPACE_REPO = "Dewdu/physiological-anxiety-escalation"

# The private HF Dataset where personalized per-user weights are stored
HF_WEIGHTS_REPO = "Dewdu/physiological-anxiety-weights"

FEATURE_NAMES = [
    "mean_hr", "mean_rr", "sdnn", "rmssd", "mean_br", "std_br",
    "mean_temp", "std_temp", "mean_acc_mag", "std_acc_mag",
]
NORM_MEAN_FIELDS = [f"{name}_baseline_mean" for name in FEATURE_NAMES]
NORM_STD_FIELDS = [f"{name}_baseline_std" for name in FEATURE_NAMES]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MODEL DEFINITION
# This must be an exact copy of the class in main.py so the weights are
# compatible when your HF Space loads them.
# Feature order (10 features):
#   [mean_HR, mean_RR, SDNN, RMSSD, mean_BR, std_BR,
#    mean_temp, std_temp, mean_acc_mag, std_acc_mag]
# ─────────────────────────────────────────────────────────────────────────────

class MaskedLSTMAutoEncoder(nn.Module):
    def __init__(self, n_features=10, hidden_size=64, n_layers=1):
        super().__init__()
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.T           = 5  # decoder sequence length — must match main.py

        self.encoder      = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True
        )
        self.decoder      = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h_n, _)   = self.encoder(x)
        bottleneck    = h_n[-1]
        decoder_input = bottleneck.unsqueeze(1).repeat(1, self.T, 1)
        decoder_out, _ = self.decoder(decoder_input)
        return self.output_layer(decoder_out)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA FETCHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_user_ids(query_api: object) -> list[str]:
    """
    Finds every unique user_id that has sent physiological data
    in the last 7 days. Returns a list of user ID strings.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> keep(columns: ["user_id"])
      |> distinct(column: "user_id")
    '''
    tables   = query_api.query(query)
    user_ids = set()
    for table in tables:
        for record in table.records:
            uid = record.values.get("user_id") or record.values.get("_value")
            if uid:
                user_ids.add(str(uid))
    return list(user_ids)


def get_user_feature_rows(
    query_api: object,
    user_id: str,
) -> tuple[np.ndarray, list]:
    """
    Pulls all 10-feature rows stored by /ingest for this user over the last 7 days.
    Returns a NumPy array of shape (N, 10).
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    tables  = query_api.query(query)
    records = []
    timestamps = []
    for table in tables:
        for record in table.records:
            row = [
                record.values.get(feature)
                if record.values.get(feature) is not None
                else record.values.get(f"f_{index}")
                for index, feature in enumerate(FEATURE_NAMES)
            ]
            if None not in row:
                records.append(row)
                timestamps.append(record.get_time())

    if not records:
        return np.empty((0, 10), dtype=np.float32), []

    return np.array(records, dtype=np.float32), timestamps


def get_user_features(query_api: object, user_id: str) -> np.ndarray:
    """Backward-compatible array-only helper used by existing notebooks."""
    records, _ = get_user_feature_rows(query_api, user_id)
    return records


def exclude_confirmed_anxiety_rows(
    query_api: object,
    user_id: str,
    data: np.ndarray,
    timestamps: list,
) -> tuple[np.ndarray, int]:
    """Do not teach confirmed anxiety episodes to the AE as normal baseline.

    False-positive (`confirmed_anxious=false`) rows remain in training. Fine-
    tuning on them lowers reconstruction error for that user's legitimate
    non-anxiety physiology. Confirmed episodes are excluded from 10 minutes
    before through 10 minutes after the alert.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "anxiety_feedback")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> filter(fn: (r) => r["_field"] == "confirmed_anxious")
      |> filter(fn: (r) => r["_value"] == true)
    '''
    confirmed_times = [
        record.get_time()
        for table in query_api.query(query)
        for record in table.records
    ]
    if not confirmed_times:
        return data, 0

    keep = []
    excluded = 0
    margin = timedelta(minutes=10)
    for timestamp in timestamps:
        is_confirmed_episode = any(
            event_time - margin <= timestamp <= event_time + margin
            for event_time in confirmed_times
        )
        keep.append(not is_confirmed_episode)
        excluded += int(is_confirmed_episode)

    return data[np.asarray(keep, dtype=bool)], excluded


def get_norm_params(query_api: object, user_id: str) -> tuple:
    """
    Pulls the user's baseline normalization params (b_mean, b_std) from InfluxDB.
    These were uploaded by the app after the user's calibration session.
    Returns (b_mean, b_std) as numpy arrays, or (None, None) if not found.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1y)
      |> filter(fn: (r) => r["_measurement"] == "norm_params")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''
    tables  = query_api.query(query)
    records = [r.values for t in tables for r in t.records]

    if not records:
        return None, None

    b_mean = np.array([
        records[0].get(NORM_MEAN_FIELDS[i], records[0].get(f"mean_{i}"))
        for i in range(10)
    ], dtype=np.float32)
    b_std = np.array([
        records[0].get(NORM_STD_FIELDS[i], records[0].get(f"std_{i}"))
        for i in range(10)
    ], dtype=np.float32)
    if not np.isfinite(b_mean).all() or not np.isfinite(b_std).all():
        return None, None
    b_std[b_std <= 0] = 1e-8  # defensive guard matching main.py
    return b_mean, b_std


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MODEL LOADING
# Tries to load personalized weights first. Falls back to the base model.
# ─────────────────────────────────────────────────────────────────────────────

def load_starting_model(user_id: str) -> MaskedLSTMAutoEncoder | None:
    """
    Priority:
      1. User's existing personalized .pth from Dewdu/physiological-anxiety-weights
         (continue improving from last week's fine-tune)
      2. Global base model from the HF Space
         (brand new user — start from scratch with the LOSO-trained weights)
    """
    model = MaskedLSTMAutoEncoder()

    # Try personalized weights first
    try:
        path = hf_hub_download(
            repo_id=HF_WEIGHTS_REPO,
            filename=f"{user_id}.pth",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"    ✓ Loaded existing personalized model for {user_id}")
        return model
    except Exception:
        pass  # file doesn't exist yet — this is a new user

    # Fall back to the global base model stored in the HF Space
    try:
        path = hf_hub_download(
            repo_id=HF_SPACE_REPO,
            filename="models/MASKED_LSTM_AE_LOSO_S10.pth",
            repo_type="space",
            token=HF_TOKEN,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"    ✓ Loaded global base model for {user_id} (first personalization)")
        return model
    except Exception as e:
        print(f"    ✗ Could not load any model for {user_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FINE-TUNING
# Slides a 5-timestep window across the user's data and trains the
# autoencoder to reconstruct each window. This teaches the model exactly
# what THIS user's body looks like so deviations stand out more clearly.
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    model: MaskedLSTMAutoEncoder,
    normalized_data: np.ndarray,
    n_epochs: int = 5,
    learning_rate: float = 1e-4,
) -> MaskedLSTMAutoEncoder | None:
    """
    Fine-tunes the LSTM Autoencoder on the user's normalized weekly data.

    - Window size = 5 (matches model.T in main.py)
    - Loss = MSE between input and reconstructed output
    - A small learning rate (1e-4) prevents catastrophic forgetting of the
      base model's general knowledge while still adapting to this user.
    """
    WINDOW = 5  # must match model.T

    if len(normalized_data) < WINDOW:
        print(f"    ✗ Not enough data ({len(normalized_data)} rows, need at least {WINDOW}). Skipping.")
        return None

    # Build all overlapping 5-step windows from the weekly data
    windows = []
    for i in range(len(normalized_data) - WINDOW + 1):
        windows.append(normalized_data[i : i + WINDOW])

    windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32)  # shape: (N, 5, 10)
    print(f"    Training on {len(windows)} windows over {n_epochs} epochs...")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for window in windows_tensor:
            x   = window.unsqueeze(0)   # (1, 5, 10)
            optimizer.zero_grad()
            out  = model(x)             # (1, 5, 10) — reconstructed window
            loss = criterion(out, x)    # how different is the reconstruction?
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(windows_tensor)
        print(f"    Epoch {epoch + 1}/{n_epochs} — avg reconstruction loss: {avg_loss:.6f}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: UPLOAD PERSONALIZED WEIGHTS
# Saves the fine-tuned model weights as {user_id}.pth and pushes to HF.
# Your main.py will pick this up automatically on the next /predict call.
# ─────────────────────────────────────────────────────────────────────────────

def upload_weights(model: MaskedLSTMAutoEncoder, user_id: str, api: HfApi) -> None:
    """
    Serializes model weights to an in-memory buffer and uploads to
    Dewdu/physiological-anxiety-weights/{user_id}.pth without writing to disk.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo=f"{user_id}.pth",
        repo_id=HF_WEIGHTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"    ✓ Uploaded {user_id}.pth → {HF_WEIGHTS_REPO}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Orchestrates the full weekly personalization run
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Weekly Physiological Personalization Script")
    print("=" * 60)

    # Connect to InfluxDB and Hugging Face
    db_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api  = db_client.query_api()
    hf_api     = HfApi()

    # Find all users who have sent data this week
    user_ids = get_all_user_ids(query_api)
    if not user_ids:
        print("No users with data in the last 7 days. Nothing to do.")
        return

    print(f"\nFound {len(user_ids)} user(s) to personalize: {user_ids}\n")

    for user_id in user_ids:
        print(f"─── Processing user: {user_id} ───")

        # 1. Get raw feature data from InfluxDB
        raw_data, timestamps = get_user_feature_rows(query_api, user_id)
        if len(raw_data) == 0:
            print(f"    No feature rows found. Skipping.\n")
            continue
        print(f"    Retrieved {len(raw_data)} feature rows from InfluxDB")

        # 2. Get the calm calibration. Live weekly data must not silently
        # become the baseline because it may include stress episodes.
        b_mean, b_std = get_norm_params(query_api, user_id)
        if b_mean is None:
            print("    No valid calm calibration params found. Skipping.\n")
            continue

        # Confirmed anxiety windows are labels for abnormal physiology, not
        # examples the reconstruction model should learn as normal.
        raw_data, excluded = exclude_confirmed_anxiety_rows(
            query_api,
            user_id,
            raw_data,
            timestamps,
        )
        if excluded:
            print(f"    Excluded {excluded} row(s) around confirmed anxiety alerts")
        if len(raw_data) < 5:
            print("    Not enough non-anxiety data after feedback filtering. Skipping.\n")
            continue

        # 3. Normalize
        normalized = (raw_data - b_mean) / b_std

        # 4. Load the best available model weights
        model = load_starting_model(user_id)
        if model is None:
            print(f"    Could not load a model. Skipping.\n")
            continue

        # 5. Fine-tune on this user's personal data
        model = fine_tune(model, normalized, n_epochs=5, learning_rate=1e-4)
        if model is None:
            print(f"    Fine-tuning skipped.\n")
            continue

        # 6. Upload the updated weights to Hugging Face
        upload_weights(model, user_id, hf_api)
        print(f"    Done!\n")

    db_client.close()
    print("=" * 60)
    print("  Personalization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
