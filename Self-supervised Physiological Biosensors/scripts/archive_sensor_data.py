#!/usr/bin/env python3
"""Archive the last seven days of physiological features to Hugging Face."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from influxdb_client import InfluxDBClient


INFLUX_URL = os.environ["INFLUX_URL"]
INFLUX_TOKEN = os.environ["INFLUX_TOKEN"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]
HF_TOKEN = os.environ["HF_TOKEN"]

HF_DATASET_REPO = os.getenv(
    "HF_SENSOR_ARCHIVE_REPO",
    "Dewdu/physiological-sensor-archive",
)
FEATURE_COLUMNS = [
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


def fetch_recent_rows(query_api: object) -> pd.DataFrame:
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "physiological_metrics")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''
    tables = query_api.query(query)
    rows: list[dict[str, object]] = []

    for table in tables:
        for record in table.records:
            row = {
                "timestamp": record.get_time(),
                "user_id": record.values.get("user_id"),
            }
            row.update(
                {
                    feature: (
                        record.values.get(feature)
                        if record.values.get(feature) is not None
                        else record.values.get(f"f_{index}")
                    )
                    for index, feature in enumerate(FEATURE_COLUMNS)
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "user_id", *FEATURE_COLUMNS])

    dataframe = pd.DataFrame(rows)
    complete_rows = dataframe.dropna(
        subset=["timestamp", "user_id", *FEATURE_COLUMNS]
    ).copy()
    skipped = len(dataframe) - len(complete_rows)
    if skipped:
        print(f"Skipped {skipped} incomplete physiological row(s).")

    return complete_rows.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)


def upload_archive(dataframe: pd.DataFrame, api: HfApi) -> str:
    date = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    filename = f"sensor_archive_{date}.parquet"

    with tempfile.TemporaryDirectory() as directory:
        local_path = Path(directory) / filename
        dataframe.to_parquet(
            local_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )

    return filename


def main() -> None:
    print("Querying the last seven days of physiological data...")
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
        timeout=30_000,
    )

    try:
        dataframe = fetch_recent_rows(client.query_api())
        if dataframe.empty:
            print("No complete physiological rows found. Nothing to archive.")
            return

        print(f"Found {len(dataframe)} complete physiological row(s).")
        filename = upload_archive(dataframe, HfApi())
        print(f"Uploaded data/{filename} to {HF_DATASET_REPO}.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
