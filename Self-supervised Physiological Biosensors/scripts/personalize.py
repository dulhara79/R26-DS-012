#!/usr/bin/env python3
"""
personalize.py — Weekly Physiological Model Personalization Script
===================================================================
Runs every week automatically via GitHub Actions.

For every user who has physiological data in InfluxDB, this script:
  1. Pulls their last 7 days of feature windows
  2. Loads their personal baseline norm params and normalizes the data
  3. Loads their existing personalized unmasked model
     (or the final unmasked base model for new users)
  4. Fine-tunes the LSTM Autoencoder on their personal data
  5. Uploads the updated unmasked AE weights to
     Dewdu/physiological-anxiety-weights as unmasked_v2_{user_id}.pth

Required Environment Variables:
  INFLUX_URL     — e.g. https://us-east-1-1.aws.cloud2.influxdata.com
  INFLUX_TOKEN
  INFLUX_ORG     — e.g. Dewdu
  INFLUX_BUCKET  — e.g. sensor_data
  HF_TOKEN
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
# ─────────────────────────────────────────────────────────────────────────────

INFLUX_URL = os.environ["INFLUX_URL"]
INFLUX_TOKEN = os.environ["INFLUX_TOKEN"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]
HF_TOKEN = os.environ["HF_TOKEN"]

# HF Space containing the final base model weights
HF_SPACE_REPO = "Dewdu/physiological-anxiety-escalation"

# Private HF Dataset containing personalized per-user weights
HF_WEIGHTS_REPO = "Dewdu/physiological-anxiety-weights"

# Version prefix prevents old masked-model personalized weights
# from being accidentally loaded by the new unmasked pipeline.
PERSONALIZED_MODEL_PREFIX = "unmasked_v2"

FEATURE_NAMES = [
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
]

NORM_MEAN_FIELDS = [
    f"{name}_baseline_mean"
    for name in FEATURE_NAMES
]

NORM_STD_FIELDS = [
    f"{name}_baseline_std"
    for name in FEATURE_NAMES
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MODEL DEFINITION
#
# This architecture must exactly match LSTM_AE_FINAL.pth and the LSTM
# autoencoder definition used by the inference server.
#
# Input shape:
#     (batch, 5, 10)
#
# Feature order:
#     [mean_HR, mean_RR, SDNN, RMSSD, mean_BR, std_BR,
#      mean_temp, std_temp, mean_acc_mag, std_acc_mag]
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoEncoder(nn.Module):

    def __init__(
        self,
        n_features=10,
        hidden_size=64,
        n_layers=1,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Five one-minute observations per AE sequence
        self.T = 5

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            n_features,
        )

    def forward(self, x):

        _, (h_n, _) = self.encoder(x)

        bottleneck = h_n[-1]

        decoder_input = (
            bottleneck
            .unsqueeze(1)
            .repeat(1, self.T, 1)
        )

        decoder_out, _ = self.decoder(
            decoder_input
        )

        return self.output_layer(
            decoder_out
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA FETCHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_user_ids(
    query_api: object,
) -> list[str]:
    """
    Finds every unique user_id that has sent physiological data
    during the last seven days.
    """

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> keep(columns: ["user_id"])
      |> distinct(column: "user_id")
    '''

    tables = query_api.query(query)

    user_ids = set()

    for table in tables:
        for record in table.records:

            uid = (
                record.values.get("user_id")
                or record.values.get("_value")
            )

            if uid:
                user_ids.add(str(uid))

    return list(user_ids)


def get_user_feature_rows(
    query_api: object,
    user_id: str,
) -> tuple[np.ndarray, list]:
    """
    Pulls all 10-feature physiological rows for one user
    during the last seven days.

    Returns:
        data        -> NumPy array of shape (N, 10)
        timestamps  -> matching timestamps
    """

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(
            rowKey:["_time"],
            columnKey:["_field"],
            valueColumn:"_value"
         )
      |> sort(columns: ["_time"])
    '''

    tables = query_api.query(query)

    records = []
    timestamps = []

    for table in tables:
        for record in table.records:

            row = [
                (
                    record.values.get(feature)
                    if record.values.get(feature) is not None
                    else record.values.get(f"f_{index}")
                )
                for index, feature
                in enumerate(FEATURE_NAMES)
            ]

            if None not in row:
                records.append(row)
                timestamps.append(
                    record.get_time()
                )

    if not records:
        return (
            np.empty(
                (0, 10),
                dtype=np.float32,
            ),
            [],
        )

    return (
        np.array(
            records,
            dtype=np.float32,
        ),
        timestamps,
    )


def get_user_features(
    query_api: object,
    user_id: str,
) -> np.ndarray:
    """
    Backward-compatible array-only helper.
    """

    records, _ = get_user_feature_rows(
        query_api,
        user_id,
    )

    return records


def exclude_confirmed_anxiety_rows(
    query_api: object,
    user_id: str,
    data: np.ndarray,
    timestamps: list,
) -> tuple[np.ndarray, int]:
    """
    Prevent confirmed anxiety episodes from being learned
    as normal user physiology.

    False-positive alerts remain available for fine-tuning.

    Confirmed anxiety periods are excluded from:
        10 minutes before
        through
        10 minutes after
    the confirmed event.
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

    margin = timedelta(
        minutes=10
    )

    for timestamp in timestamps:

        is_confirmed_episode = any(
            event_time - margin
            <= timestamp
            <= event_time + margin
            for event_time in confirmed_times
        )

        keep.append(
            not is_confirmed_episode
        )

        excluded += int(
            is_confirmed_episode
        )

    return (
        data[
            np.asarray(
                keep,
                dtype=bool,
            )
        ],
        excluded,
    )


def get_norm_params(
    query_api: object,
    user_id: str,
) -> tuple:
    """
    Loads the participant's calm baseline normalization parameters.

    Returns:
        b_mean
        b_std

    Each array contains 10 values matching FEATURE_NAMES.
    """

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -1y)
      |> filter(fn: (r) => r["_measurement"] == "norm_params")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(
            rowKey:["_time"],
            columnKey:["_field"],
            valueColumn:"_value"
         )
      |> sort(
            columns:["_time"],
            desc:true
         )
      |> limit(n: 1)
    '''

    tables = query_api.query(query)

    records = [
        record.values
        for table in tables
        for record in table.records
    ]

    if not records:
        return None, None

    latest = records[0]

    b_mean = np.array(
        [
            latest.get(
                NORM_MEAN_FIELDS[i],
                latest.get(f"mean_{i}"),
            )
            for i in range(10)
        ],
        dtype=np.float32,
    )

    b_std = np.array(
        [
            latest.get(
                NORM_STD_FIELDS[i],
                latest.get(f"std_{i}"),
            )
            for i in range(10)
        ],
        dtype=np.float32,
    )

    if (
        not np.isfinite(b_mean).all()
        or not np.isfinite(b_std).all()
    ):
        return None, None

    # Defensive guard against division by zero
    b_std[b_std <= 0] = 1e-8

    return b_mean, b_std


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MODEL LOADING
#
# Priority:
#
# 1. Existing personalized UNMASKED model
#    unmasked_v2_{user_id}.pth
#
# 2. Final population-trained UNMASKED model
#    LSTM_AE_FINAL.pth
# ─────────────────────────────────────────────────────────────────────────────

def load_starting_model(
    user_id: str,
) -> LSTMAutoEncoder | None:

    model = LSTMAutoEncoder(
        n_features=10,
        hidden_size=64,
        n_layers=1,
    )

    personalized_filename = (
        f"{PERSONALIZED_MODEL_PREFIX}_{user_id}.pth"
    )

    # -------------------------------------------------------------------------
    # 1. Try the participant's existing personalized unmasked model
    # -------------------------------------------------------------------------

    try:

        path = hf_hub_download(
            repo_id=HF_WEIGHTS_REPO,
            filename=personalized_filename,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        state_dict = torch.load(
            path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        print(
            f"    ✓ Loaded existing personalized "
            f"unmasked model for {user_id}"
        )

        return model

    except Exception:
        # New users will not have a personalized
        # unmasked_v2 file yet.
        pass

    # -------------------------------------------------------------------------
    # 2. Fall back to the final population-trained unmasked AE
    # -------------------------------------------------------------------------

    try:

        path = hf_hub_download(
            repo_id=HF_SPACE_REPO,
            filename="models/LSTM_AE_FINAL.pth",
            repo_type="space",
            token=HF_TOKEN,
        )

        state_dict = torch.load(
            path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        print(
            f"    ✓ Loaded final unmasked base model "
            f"for {user_id} (first personalization)"
        )

        return model

    except Exception as e:

        print(
            f"    ✗ Could not load any unmasked "
            f"model for {user_id}: {e}"
        )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FINE-TUNING
#
# Uses complete 5-minute physiological windows.
#
# IMPORTANT:
# No masking is applied here.
# The model reconstructs the complete input sequence.
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    model: LSTMAutoEncoder,
    normalized_data: np.ndarray,
    n_epochs: int = 5,
    learning_rate: float = 1e-4,
) -> LSTMAutoEncoder | None:
    """
    Fine-tunes the final unmasked LSTM Autoencoder
    on the participant's normalized weekly data.

    Window size:
        5 one-minute observations

    Input shape:
        (N, 5, 10)

    Loss:
        MSE(input, reconstruction)
    """

    WINDOW = 5

    if len(normalized_data) < WINDOW:

        print(
            f"    ✗ Not enough data "
            f"({len(normalized_data)} rows, "
            f"need at least {WINDOW}). Skipping."
        )

        return None

    # -------------------------------------------------------------------------
    # Build overlapping five-minute sequences
    # -------------------------------------------------------------------------

    windows = []

    for i in range(
        len(normalized_data)
        - WINDOW
        + 1
    ):
        windows.append(
            normalized_data[
                i : i + WINDOW
            ]
        )

    windows_tensor = torch.tensor(
        np.array(windows),
        dtype=torch.float32,
    )

    print(
        f"    Training on "
        f"{len(windows)} windows "
        f"over {n_epochs} epochs..."
    )

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------------------------

    for epoch in range(n_epochs):

        total_loss = 0.0

        for window in windows_tensor:

            # Shape:
            # (1, 5, 10)
            x = window.unsqueeze(0)

            optimizer.zero_grad()

            # Full unmasked reconstruction
            output = model(x)

            loss = criterion(
                output,
                x,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss
            / len(windows_tensor)
        )

        print(
            f"    Epoch "
            f"{epoch + 1}/{n_epochs} "
            f"— avg reconstruction loss: "
            f"{avg_loss:.6f}"
        )

    model.eval()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: UPLOAD PERSONALIZED WEIGHTS
#
# New personalized models use:
#
#     unmasked_v2_{user_id}.pth
#
# Old {user_id}.pth files are intentionally ignored.
# ─────────────────────────────────────────────────────────────────────────────

def upload_weights(
    model: LSTMAutoEncoder,
    user_id: str,
    api: HfApi,
) -> None:
    """
    Uploads personalized unmasked LSTM-AE weights to:

    Dewdu/physiological-anxiety-weights/
        unmasked_v2_{user_id}.pth
    """

    personalized_filename = (
        f"{PERSONALIZED_MODEL_PREFIX}_{user_id}.pth"
    )

    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer,
    )

    buffer.seek(0)

    api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo=personalized_filename,
        repo_id=HF_WEIGHTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print(
        f"    ✓ Uploaded "
        f"{personalized_filename} "
        f"→ {HF_WEIGHTS_REPO}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    print("=" * 60)
    print(
        "  Weekly Physiological Personalization Script"
    )
    print(
        "  Final Unmasked LSTM Autoencoder"
    )
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Connect to InfluxDB and Hugging Face
    # -------------------------------------------------------------------------

    db_client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
    )

    query_api = db_client.query_api()

    hf_api = HfApi()

    try:

        # ---------------------------------------------------------------------
        # Find users with physiological data during the last seven days
        # ---------------------------------------------------------------------

        user_ids = get_all_user_ids(
            query_api
        )

        if not user_ids:

            print(
                "No users with data in the last "
                "7 days. Nothing to do."
            )

            return

        print(
            f"\nFound {len(user_ids)} "
            f"user(s) to personalize: "
            f"{user_ids}\n"
        )

        # ---------------------------------------------------------------------
        # Personalize each user independently
        # ---------------------------------------------------------------------

        for user_id in user_ids:

            print(
                f"─── Processing user: "
                f"{user_id} ───"
            )

            # -----------------------------------------------------------------
            # 1. Get raw physiological data
            # -----------------------------------------------------------------

            raw_data, timestamps = (
                get_user_feature_rows(
                    query_api,
                    user_id,
                )
            )

            if len(raw_data) == 0:

                print(
                    "    No feature rows found. "
                    "Skipping.\n"
                )

                continue

            print(
                f"    Retrieved "
                f"{len(raw_data)} "
                f"feature rows from InfluxDB"
            )

            # -----------------------------------------------------------------
            # 2. Load participant's calm calibration
            # -----------------------------------------------------------------

            b_mean, b_std = get_norm_params(
                query_api,
                user_id,
            )

            if b_mean is None:

                print(
                    "    No valid calm calibration "
                    "params found. Skipping.\n"
                )

                continue

            # -----------------------------------------------------------------
            # 3. Exclude confirmed anxiety periods
            # -----------------------------------------------------------------

            raw_data, excluded = (
                exclude_confirmed_anxiety_rows(
                    query_api,
                    user_id,
                    raw_data,
                    timestamps,
                )
            )

            if excluded:

                print(
                    f"    Excluded {excluded} "
                    f"row(s) around confirmed "
                    f"anxiety alerts"
                )

            if len(raw_data) < 5:

                print(
                    "    Not enough non-anxiety "
                    "data after feedback filtering. "
                    "Skipping.\n"
                )

                continue

            # -----------------------------------------------------------------
            # 4. Apply participant-specific normalization
            # -----------------------------------------------------------------

            normalized = (
                raw_data - b_mean
            ) / b_std

            if not np.isfinite(
                normalized
            ).all():

                print(
                    "    Normalized data contains "
                    "NaN or infinite values. "
                    "Skipping.\n"
                )

                continue

            # -----------------------------------------------------------------
            # 5. Load personalized model or final base model
            # -----------------------------------------------------------------

            model = load_starting_model(
                user_id
            )

            if model is None:

                print(
                    "    Could not load a model. "
                    "Skipping.\n"
                )

                continue

            # -----------------------------------------------------------------
            # 6. Fine-tune the unmasked AE
            # -----------------------------------------------------------------

            model = fine_tune(
                model,
                normalized,
                n_epochs=5,
                learning_rate=1e-4,
            )

            if model is None:

                print(
                    "    Fine-tuning skipped.\n"
                )

                continue

            # -----------------------------------------------------------------
            # 7. Upload updated personalized unmasked AE
            # -----------------------------------------------------------------

            upload_weights(
                model,
                user_id,
                hf_api,
            )

            print(
                "    Done!\n"
            )

    finally:

        db_client.close()

    print("=" * 60)
    print(
        "  Personalization complete."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()