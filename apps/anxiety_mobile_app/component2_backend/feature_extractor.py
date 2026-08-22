"""Component 2 screen+location feature extractor.

Builds the exact 240-column order expected by M2_mobile_screen_location_v1:
20 bases x 4 GLOBEM segments x (28-day mean, std, availability).

The location features are mobile proxies, not an exact RAPIDS reproduction.
The score produced downstream must therefore remain EXPERIMENTAL.
"""
from __future__ import annotations

import json, math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOCAL_TZ = ZoneInfo("Asia/Colombo")
WINDOW_DAYS = 28
MIN_HISTORY_DAYS = 14
USE_SEGMENTS = ["morning", "afternoon", "evening", "night"]

LOCATION_BASES = [
    "f_loc:phone_locations_doryab_avglengthstayatclusters",
    "f_loc:phone_locations_doryab_avgspeed",
    "f_loc:phone_locations_doryab_homelabel",
    "f_loc:phone_locations_doryab_locationentropy",
    "f_loc:phone_locations_doryab_locationvariance",
    "f_loc:phone_locations_doryab_maxlengthstayatclusters",
    "f_loc:phone_locations_doryab_minlengthstayatclusters",
    "f_loc:phone_locations_doryab_movingtostaticratio",
    "f_loc:phone_locations_doryab_normalizedlocationentropy",
    "f_loc:phone_locations_doryab_numberlocationtransitions",
]

SCREEN_BASES = [
    "f_screen:phone_screen_rapids_avgdurationunlock",
    "f_screen:phone_screen_rapids_countepisodeunlock",
    "f_screen:phone_screen_rapids_countepisodeunlock_norm",
    "f_screen:phone_screen_rapids_firstuseafter00unlock",
    "f_screen:phone_screen_rapids_maxdurationunlock",
    "f_screen:phone_screen_rapids_maxdurationunlock_norm",
    "f_screen:phone_screen_rapids_mindurationunlock",
    "f_screen:phone_screen_rapids_mindurationunlock_norm",
    "f_screen:phone_screen_rapids_sumdurationunlock",
    "f_screen:phone_screen_rapids_sumdurationunlock_norm",
]

MOBILE_BASES = LOCATION_BASES + SCREEN_BASES
SCREEN_NORM_MAP = {
    "f_screen:phone_screen_rapids_countepisodeunlock_norm": "f_screen:phone_screen_rapids_countepisodeunlock",
    "f_screen:phone_screen_rapids_maxdurationunlock_norm": "f_screen:phone_screen_rapids_maxdurationunlock",
    "f_screen:phone_screen_rapids_mindurationunlock_norm": "f_screen:phone_screen_rapids_mindurationunlock",
    "f_screen:phone_screen_rapids_sumdurationunlock_norm": "f_screen:phone_screen_rapids_sumdurationunlock",
}

FEATURE_NAMES: List[str] = []
for stat in ("mean", "std", "availability"):
    for seg in USE_SEGMENTS:
        for base in MOBILE_BASES:
            FEATURE_NAMES.append(f"{base}:{seg}__{stat}")
assert len(FEATURE_NAMES) == 240


@dataclass
class FeatureExtractionResult:
    vector: np.ndarray
    feature_names: List[str]
    daily_features: pd.DataFrame
    coverage: Dict[str, Any]
    window_start: str
    window_end: str

    def as_model_input(self) -> np.ndarray:
        return self.vector.reshape(1, -1)


def _decode(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            x = json.loads(v)
            return x if isinstance(x, dict) else {"value": x}
        except Exception:
            return {"value": v}
    return {} if v is None else {"value": v}


def _segment(ts: pd.Timestamp) -> str:
    h = ts.hour
    if h < 6: return "night"
    if h < 12: return "morning"
    if h < 18: return "afternoon"
    return "evening"


def prepare_sensor_events(rows, participant_id: Optional[str] = None) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df = df.rename(columns={
        "timestamp": "event_time", "dataType": "event_type",
        "value": "value_json", "userId": "participant_code",
    })
    required = {"event_time", "event_type", "value_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if participant_id is not None and "participant_code" in df.columns:
        df = df[df["participant_code"].astype(str) == str(participant_id)].copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time", "event_type"]).copy()
    df["local_time"] = df["event_time"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["local_time"].dt.date
    df["segment"] = df["local_time"].map(_segment)
    df["value_json"] = df["value_json"].map(_decode)
    return df.sort_values("local_time").reset_index(drop=True)


# ----------------------------- Screen -------------------------------------

def build_unlock_episodes(events: pd.DataFrame) -> pd.DataFrame:
    s = events[events.event_type == "Screen_Event"].copy()
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "duration_min"])
    s["state"] = s.value_json.map(lambda d: d.get("state"))
    pending = None
    rows = []
    for r in s.sort_values("local_time").itertuples():
        if r.state == "Screen_Unlocked":
            pending = r.local_time
        elif r.state == "Screen_Off":
            if pending is not None and r.local_time > pending:
                mins = (r.local_time - pending).total_seconds() / 60
                if 0 < mins <= 360:  # RAPIDS default max episode = 6h
                    rows.append({"start": pending, "end": r.local_time, "duration_min": mins})
            pending = None
    return pd.DataFrame(rows)


def _next_segment_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    day = ts.normalize()
    for hour in (6, 12, 18, 24):
        b = day + pd.Timedelta(hours=hour)
        if b > ts:
            return b
    return day + pd.Timedelta(days=1)


def _split_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ep in episodes.itertuples():
        cur, end = ep.start, ep.end
        while cur < end:
            nxt = min(end, _next_segment_boundary(cur))
            out.append({
                "local_date": cur.date(), "segment": _segment(cur),
                "episode_start": cur,
                "duration_min": (nxt-cur).total_seconds()/60,
            })
            cur = nxt
    return pd.DataFrame(out)


def extract_daily_screen(events: pd.DataFrame, days: List[date]) -> pd.DataFrame:
    raw = [
        "f_screen:phone_screen_rapids_avgdurationunlock",
        "f_screen:phone_screen_rapids_countepisodeunlock",
        "f_screen:phone_screen_rapids_firstuseafter00unlock",
        "f_screen:phone_screen_rapids_maxdurationunlock",
        "f_screen:phone_screen_rapids_mindurationunlock",
        "f_screen:phone_screen_rapids_sumdurationunlock",
    ]
    idx = pd.MultiIndex.from_product([days, USE_SEGMENTS], names=["date","segment"])
    out = pd.DataFrame(index=idx, columns=raw, dtype=float)
    eps = _split_episodes(build_unlock_episodes(events))

    # Some collection evidence means zero unlocks can be represented as count=0.
    observed = events[events.event_type.isin(["Screen_Event","Location_Grid_100m","Service_Heartbeat"])]
    observed = observed.groupby(["local_date","segment"]).size()

    if not eps.empty:
        for key, g in eps.groupby(["local_date","segment"]):
            if key not in out.index: continue
            d = g.duration_min.to_numpy(float)
            first = min(x.hour*60 + x.minute + x.second/60 for x in pd.to_datetime(g.episode_start))
            out.loc[key, raw] = [d.mean(), len(d), first, d.max(), d.min(), d.sum()]

    for key in out.index:
        if pd.isna(out.loc[key, raw[1]]) and int(observed.get(key, 0)) > 0:
            out.loc[key, raw[1]] = 0.0
            out.loc[key, raw[5]] = 0.0
    return out


# ----------------------------- Location -----------------------------------

def _location_rows(events: pd.DataFrame) -> pd.DataFrame:
    x = events[events.event_type == "Location_Grid_100m"].copy()
    if x.empty: return x
    def num(d,k):
        try: return float(d.get(k))
        except Exception: return np.nan
    for c in ("lat","lng","speed_mps","accuracy_m"):
        x[c] = x.value_json.map(lambda d, c=c: num(d,c))
    x = x.dropna(subset=["lat","lng"]).copy()
    x["grid"] = x.lat.round(3).map(lambda z:f"{z:.3f}") + "," + x.lng.round(3).map(lambda z:f"{z:.3f}")
    return x.sort_values("local_time")


def _haversine_m(a,b,c,d):
    R=6371000.0
    p1,p2=math.radians(a),math.radians(c)
    dp,dl=math.radians(c-a),math.radians(d-b)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(h),math.sqrt(max(0,1-h)))


def _speed_kmh(g: pd.DataFrame) -> np.ndarray:
    sp = g.speed_mps.to_numpy(float)*3.6
    ts = g.local_time.tolist(); lat=g.lat.to_numpy(float); lng=g.lng.to_numpy(float)
    for i in range(len(g)-1):
        if np.isfinite(sp[i]): continue
        hours=(ts[i+1]-ts[i]).total_seconds()/3600
        if hours>0: sp[i]=_haversine_m(lat[i],lng[i],lat[i+1],lng[i+1])/1000/hours
    return sp


def _weights(g: pd.DataFrame, max_gap_min=30.0) -> np.ndarray:
    ts=g.local_time.tolist()
    if len(ts)==1: return np.array([15.0])
    ds=np.array([(ts[i+1]-ts[i]).total_seconds()/60 for i in range(len(ts)-1)])
    valid=ds[(ds>0)&(ds<=max_gap_min)]
    fallback=float(np.median(valid)) if len(valid) else 15.0
    w=np.array([d if 0<d<=max_gap_min else fallback for d in ds]+[fallback], float)
    return w


def _stays(g: pd.DataFrame) -> List[float]:
    grids=g.grid.tolist(); w=_weights(g)
    if not grids: return []
    out=[]; current=grids[0]; total=0.0
    for grid,dt in zip(grids,w):
        if grid!=current:
            out.append(total); current=grid; total=0.0
        total+=float(dt)
    out.append(total)
    return [v for v in out if v>0]


def extract_daily_location(events: pd.DataFrame, days: List[date]) -> pd.DataFrame:
    idx=pd.MultiIndex.from_product([days,USE_SEGMENTS],names=["date","segment"])
    out=pd.DataFrame(index=idx,columns=LOCATION_BASES,dtype=float)
    loc=_location_rows(events)
    if loc.empty: return out
    home = None if loc[loc.segment=="night"].empty else loc[loc.segment=="night"].grid.mode().iloc[0]

    for key,g in loc.groupby(["local_date","segment"]):
        if key not in out.index: continue
        g=g.sort_values("local_time").reset_index(drop=True)
        grids=g.grid.tolist(); vc=pd.Series(grids).value_counts(); p=vc.to_numpy(float)/vc.sum()
        ent=float(-(p*np.log(p+1e-12)).sum()); k=len(vc)
        trans=float(sum(a!=b for a,b in zip(grids[:-1],grids[1:])))
        stays=_stays(g); sp=_speed_kmh(g); w=_weights(g)
        moving=np.isfinite(sp)&(sp>=1.0)
        avgspeed=float(np.nanmean(sp[moving])) if moving.any() else 0.0
        stationary=np.where(np.isfinite(sp),sp<1.0,True)
        statratio=float(w[stationary].sum()/w.sum()) if w.sum()>0 else np.nan
        locvar=float(np.nanvar(g.lat.to_numpy(float))+np.nanvar(g.lng.to_numpy(float)))
        vals={
            LOCATION_BASES[0]: float(np.mean(stays)) if stays else np.nan,
            LOCATION_BASES[1]: avgspeed,
            LOCATION_BASES[2]: 1.0 if home is not None else np.nan,
            LOCATION_BASES[3]: ent,
            LOCATION_BASES[4]: locvar,
            LOCATION_BASES[5]: float(np.max(stays)) if stays else np.nan,
            LOCATION_BASES[6]: float(np.min(stays)) if stays else np.nan,
            LOCATION_BASES[7]: statratio,
            LOCATION_BASES[8]: float(ent/k) if k else np.nan,
            LOCATION_BASES[9]: trans,
        }
        for c,v in vals.items(): out.loc[key,c]=v
    return out


# ----------------------- Normalize + aggregate -----------------------------

def _apply_screen_norm(daily: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out=daily.copy()
    for norm,raw in SCREEN_NORM_MAP.items():
        out[norm]=np.nan
        for seg in USE_SEGMENTS:
            vals=pd.to_numeric(ref.xs(seg,level="segment")[raw],errors="coerce").dropna()
            if len(vals)<5: continue
            med=float(vals.median()); q05=float(vals.quantile(.05)); q95=float(vals.quantile(.95)); den=q95-q05
            if not np.isfinite(den) or abs(den)<1e-9: continue
            idx=pd.IndexSlice[:,seg]
            x=pd.to_numeric(out.loc[idx,raw],errors="coerce")
            out.loc[idx,norm]=(x-med)/den
    return out


def build_daily_features(events: pd.DataFrame, start: date, end: date,
                         normalization_events: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    days=[start+timedelta(days=i) for i in range((end-start).days)]
    daily=extract_daily_location(events,days).join(extract_daily_screen(events,days))
    if normalization_events is None:
        ref=daily
    else:
        ref_start=normalization_events.local_date.min()
        ref_days=[ref_start+timedelta(days=i) for i in range((end-ref_start).days)]
        ref=extract_daily_location(normalization_events,ref_days).join(extract_daily_screen(normalization_events,ref_days))
    daily=_apply_screen_norm(daily,ref)
    for b in MOBILE_BASES:
        if b not in daily: daily[b]=np.nan
    return daily[MOBILE_BASES].sort_index()


def aggregate_28_day_vector(daily: pd.DataFrame) -> np.ndarray:
    out=[]
    for stat in ("mean","std","availability"):
        for seg in USE_SEGMENTS:
            s=daily.xs(seg,level="segment")
            for base in MOBILE_BASES:
                x=pd.to_numeric(s[base],errors="coerce").to_numpy(float)
                if stat=="mean": v=float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
                elif stat=="std": v=float(np.nanstd(x,ddof=0)) if np.isfinite(x).any() else np.nan
                else: v=float(np.mean(np.isfinite(x)))
                out.append(v)
    arr=np.asarray(out,np.float32); assert arr.shape==(240,)
    return arr


def calculate_coverage(events: pd.DataFrame, daily: pd.DataFrame) -> Dict[str,Any]:
    loc=events[events.event_type=="Location_Grid_100m"]
    scr=events[events.event_type=="Screen_Event"]
    expected=WINDOW_DAYS*24*4  # 15-min target
    arr=daily[MOBILE_BASES].to_numpy(float)
    return {
        "window_days": WINDOW_DAYS,
        "days_with_any_data": int(events.local_date.nunique()),
        "days_with_screen_events": int(scr.local_date.nunique()),
        "days_with_location": int(loc.local_date.nunique()),
        "location_events": int(len(loc)),
        "expected_location_events_15min": int(expected),
        "location_sampling_coverage": float(min(1.0,len(loc)/expected)),
        "daily_feature_availability": float(np.mean(np.isfinite(arr))),
        "minimum_history_met": bool(events.local_date.nunique()>=MIN_HISTORY_DAYS),
    }


def build_feature_vector(rows, participant_id: Optional[str], window_end_date,
                         normalization_rows=None) -> FeatureExtractionResult:
    """window_end_date is exclusive local date; only prior 28 days are scored."""
    if isinstance(window_end_date,datetime): end=window_end_date.date()
    elif isinstance(window_end_date,date): end=window_end_date
    else: end=date.fromisoformat(str(window_end_date))
    start=end-timedelta(days=WINDOW_DAYS)

    all_events=prepare_sensor_events(rows,participant_id)
    events=all_events[(all_events.local_date>=start)&(all_events.local_date<end)].copy()

    norm=None
    if normalization_rows is not None:
        norm=prepare_sensor_events(normalization_rows,participant_id)
        norm=norm[norm.local_date<end].copy()  # no future data

    daily=build_daily_features(events,start,end,norm)
    vector=aggregate_28_day_vector(daily)
    coverage=calculate_coverage(events,daily)
    return FeatureExtractionResult(vector,FEATURE_NAMES.copy(),daily,coverage,start.isoformat(),end.isoformat())


def validate_against_model_metadata(result: FeatureExtractionResult, metadata: Dict[str,Any]):
    expected=metadata.get("model_feature_names")
    if expected is None: raise ValueError("metadata missing model_feature_names")
    if list(expected)!=result.feature_names:
        raise ValueError("Feature order does not match exported model metadata")
    print(f"Feature contract PASS: {len(expected)} columns match")


if __name__ == "__main__":
    print("Expected model input:",len(FEATURE_NAMES),"features")
    print("Segments:",USE_SEGMENTS)
