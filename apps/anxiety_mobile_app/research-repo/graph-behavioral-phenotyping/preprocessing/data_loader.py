"""
preprocessing/data_loader.py
=============================
Loads all raw StudentLife modality files for a given participant ID.
Each loader returns a tidy DataFrame (or None if the file is missing /
does not meet minimum size requirements).
"""

import os
import json
import numpy as np
import pandas as pd

from config import DATASET_PATH


# ── Participant discovery ─────────────────────────────────────────────────────

def get_users():
    """Return sorted list of participant IDs found in the GPS sensing folder."""
    files = os.listdir(os.path.join(DATASET_PATH, "sensing", "gps"))
    return sorted([
        f.replace("gps_", "").replace(".csv", "")
        for f in files
        if f.startswith("gps_") and f.endswith(".csv")
    ])


# ── Per-modality loaders ──────────────────────────────────────────────────────

def load_gps(uid: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "sensing", "gps", f"gps_{uid}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=0, index_col=False)
        df.columns = df.columns.str.strip().str.lower()
        if len(df) < 10 or "time" not in df.columns:
            return None
        df["user_id"]     = uid
        df["timestamp"]   = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp", "latitude", "longitude"])
        df["hour"]        = df["timestamp"].dt.hour
        df["date"]        = df["timestamp"].dt.date
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        return df
    except Exception:
        return None


def load_activity(uid: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "sensing", "activity", f"activity_{uid}.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            first = f.readline().strip()
        has_hdr = not first.split(",")[0].strip().isdigit()
        df = pd.read_csv(path, header=0 if has_hdr else None, index_col=False)
        if not has_hdr:
            df.columns = ["timestamp", "activity_inference"]
        else:
            df.columns = df.columns.str.strip().str.lower()
            rename = {}
            for c in df.columns:
                if "time"  in c: rename[c] = "timestamp"
                if "activ" in c: rename[c] = "activity_inference"
            df = df.rename(columns=rename)
        if len(df) < 10:
            return None
        df["user_id"]   = uid
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"])
        act_map = {0: "STATIONARY", 1: "WALKING", 2: "RUNNING", 3: "UNKNOWN"}
        df["activity"] = df["activity_inference"].map(act_map).fillna("UNKNOWN")
        return df
    except Exception:
        return None


def load_stress(uid: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "EMA", "response", "Stress", f"Stress_{uid}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data or len(data) < 3:
            return None
        records = []
        for entry in data:
            ts    = pd.to_datetime(entry.get("resp_time", 0), unit="s")
            level = entry.get("level", None)
            if level is None:
                continue
            records.append({
                "user_id"     : uid,
                "timestamp"   : ts,
                "stress_level": float(level),
                "hour"        : ts.hour,
                "date"        : ts.date(),
            })
        return pd.DataFrame(records) if len(records) >= 3 else None
    except Exception:
        return None


def load_conversation(uid: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "sensing", "conversation", f"conversation_{uid}.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            first = f.readline().strip()
        has_hdr = not first.split(",")[0].strip().isdigit()
        df = pd.read_csv(path, header=0 if has_hdr else None, index_col=False)
        if not has_hdr:
            df.columns = ["start_time", "end_time", "inference"]
        else:
            df.columns = df.columns.str.strip().str.lower()
        df["user_id"]      = uid
        col0               = df.columns[0]
        col1               = df.columns[1]
        df["start_time"]   = pd.to_datetime(df[col0], unit="s", errors="coerce")
        df["end_time"]     = pd.to_datetime(df[col1], unit="s", errors="coerce")
        df["duration_min"] = (df["end_time"] - df["start_time"]).dt.seconds / 60
        df["hour"]         = df["start_time"].dt.hour
        return df.dropna(subset=["start_time"])
    except Exception:
        return None


def load_phonelock(uid: str) -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "sensing", "phonelock", f"phonelock_{uid}.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            first = f.readline().strip()
        has_hdr = not first.split(",")[0].strip().isdigit()
        df = pd.read_csv(path, header=0 if has_hdr else None, index_col=False)
        if not has_hdr:
            df.columns = ["start_timestamp", "end_timestamp", "lock"]
        else:
            df.columns = df.columns.str.strip().str.lower()
        df["user_id"]    = uid
        df["start_time"] = pd.to_datetime(df[df.columns[0]], unit="s", errors="coerce")
        df["hour"]       = df["start_time"].dt.hour
        return df.dropna(subset=["start_time"])
    except Exception:
        return None


def load_phq9() -> pd.DataFrame | None:
    path = os.path.join(DATASET_PATH, "survey", "PHQ-9.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return None


# ── Batch loader ─────────────────────────────────────────────────────────────

def load_all_users(users: list[str]) -> dict:
    """
    Load all modalities for every participant in *users*.

    Returns
    -------
    dict with keys:
        gps, activity, stress, conversation, phonelock
    Each value is a dict mapping uid → DataFrame.
    """
    all_gps, all_activity, all_stress = {}, {}, {}
    all_conversation, all_phonelock   = {}, {}

    for uid in users:
        g = load_gps(uid)
        a = load_activity(uid)
        s = load_stress(uid)
        c = load_conversation(uid)
        p = load_phonelock(uid)
        if g is not None: all_gps[uid]          = g
        if a is not None: all_activity[uid]     = a
        if s is not None: all_stress[uid]       = s
        if c is not None: all_conversation[uid] = c
        if p is not None: all_phonelock[uid]    = p

    return {
        "gps"         : all_gps,
        "activity"    : all_activity,
        "stress"      : all_stress,
        "conversation": all_conversation,
        "phonelock"   : all_phonelock,
    }
