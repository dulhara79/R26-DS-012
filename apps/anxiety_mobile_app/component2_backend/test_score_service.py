
"""
test_score_service.py

Usage:
    python test_score_service.py
"""

import json
import pandas as pd

from score_service import score_participant_events

CSV_PATH = "sensor_events.csv"
PARTICIPANT_ID = "P_2648DB2EA754E775"
WINDOW_END_DATE = "2026-08-20"

events = pd.read_csv(CSV_PATH)

result = score_participant_events(
    rows=events,
    participant_id=PARTICIPANT_ID,
    window_end_date=WINDOW_END_DATE,
)

print(json.dumps(result, indent=2))
