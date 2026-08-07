import os
import torch
import torch.nn as nn
import numpy as np
import neurokit2 as nk
from scipy import signal
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
write_api  = db_client.write_api(write_options=SYNCHRONOUS)
query_api  = db_client.query_api()

device = torch.device('cpu')

# The global default champion model for cold-starts and unpersonalized users
global_ae_model = MaskedLSTMAutoEncoder(n_features=10, hidden_size=64, n_layers=1)
global_ae_model.load_state_dict(torch.load("models/MASKED_LSTM_AE_LOSO_S10.pth", map_location=device))
global_ae_model.eval()

# Our fast in-memory storage to keep personalized user models alive in RAM
user_model_cache = {}

forecaster = Seq2SeqForecaster(embed_dim=64, enc_hidden=128, dec_hidden=128, n_layers=1, forecast_steps=10)
forecaster.load_state_dict(torch.load("models/SEQ2SEQ_LOSO_S11.pth", map_location=device))
forecaster.eval()


# -------------------------------------------------------------
# STEP 3: PYDANTIC MODELS
# -------------------------------------------------------------

class ChestStrapFeaturesPayload(BaseModel):
    user_id: str
    timestamp: str
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


# -------------------------------------------------------------
# STEP 4: ENDPOINTS
# -------------------------------------------------------------

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
    if len(payload.b_mean) != 10 or len(payload.b_std) != 10:
        raise HTTPException(
            status_code=400,
            detail="b_mean and b_std must each contain exactly 10 values (one per feature)."
        )

    point = Point("norm_params").tag("user_id", user_id)
    for i in range(10):
        point.field(f"mean_{i}", float(payload.b_mean[i]))
        point.field(f"std_{i}",  float(payload.b_std[i]))

    try:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "message": f"Norm params stored for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store norm params: {str(e)}")


@app.post("/ingest")
def process_and_ingest_raw_data(payload: ChestStrapFeaturesPayload):
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

    import numpy as np
    # FIX 4 — Final feature vector NaN check
    if any(np.isnan(features)) or any(np.isinf(features)):
        raise HTTPException(
            status_code=400,
            detail="Data window rejected: Payload contains invalid NaN or Infinite numerical values."
        )

    try:
        # --- F. WRITE RAW FEATURES TO INFLUXDB ---
        point = Point("physiological_metrics")\
            .tag("user_id", payload.user_id)\
            .time(payload.timestamp)
        for idx, value in enumerate(features):
            point.field(f"f_{idx}", float(value))

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "message": "Pre-calculated feature window successfully saved to InfluxDB"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database write failure: {str(e)}")

        for idx, value in enumerate(features):
            point.field(f"f_{idx}", float(value))

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        return {"status": "success", "message": "Raw data window successfully processed and features saved to InfluxDB"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Signal processing failure: {str(e)}")


@app.get("/predict/{user_id}")
def get_escalation_forecast(user_id: str):

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

        b_mean = np.array([norm_records[0].get(f"mean_{i}", 0.0) for i in range(10)], dtype=np.float32)
        b_std  = np.array([norm_records[0].get(f"std_{i}",  1.0) for i in range(10)], dtype=np.float32)

        # Exact notebook guard: b_std[b_std == 0] = 1e-8
        b_std[b_std == 0] = 1e-8

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve norm params: {str(e)}")

    # Fetch the last 19 minutes of raw feature windows
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -19m)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> filter(fn: (r) => r["user_id"] == "{user_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        tables  = query_api.query(query)
        records = []
        for table in tables:
            for record in table.records:
                row = [record.values.get(f"f_{i}") for i in range(10)]
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
            personalized_weight_path = f"models/{user_id}.pth"
            
            # If the file is not on the local hard drive, try to fetch it from your permanent Dataset vault
            if not os.path.exists(personalized_weight_path):
                try:
                    # Reach into your private dataset repository and pull down their specific .pth file
                    hf_hub_download(
                        repo_id=HF_WEIGHTS_REPO,
                        filename=f"{user_id}.pth",
                        repo_type="dataset",
                        local_dir="models",
                        token=HF_TOKEN
                    )
                except Exception:
                    # If the file isn't in the vault (like for a brand new user), fail silently and use the fallback
                    pass

            # Check if the file exists now (either because it was already here, or we just successfully downloaded it)
            if os.path.exists(personalized_weight_path):
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

        with torch.no_grad():
            for i in range(15):
                window        = normalized_sequence[i : i + 5]
                window_tensor = torch.tensor(window).unsqueeze(0)
                # We use active_ae_model here, which automatically points to the right weights
                emb           = active_ae_model.encode(window_tensor)
                embeddings_list.append(emb)

            lookback_tensor = torch.cat(embeddings_list, dim=0).unsqueeze(0)
            predictions     = forecaster.predict(lookback_tensor)

        return {
            "status": "success",
            "forecast": predictions.squeeze(0).tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")