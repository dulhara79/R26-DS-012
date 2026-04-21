# Dataset

This component uses the **StudentLife** dataset. You must download it
separately and point `DATASET_PATH` in `config.py` (or via the env-var)
to the extracted folder.

**Download:** https://studentlife.cs.dartmouth.edu/dataset.html

## Required folder layout

After extracting, the folder pointed to by `DATASET_PATH` must contain:

```
dataset/
├── sensing/
│   ├── gps/
│   │   └── gps_<uid>.csv          # latitude, longitude, altitude, accuracy, time
│   ├── activity/
│   │   └── activity_<uid>.csv     # timestamp, activity_inference (0-3)
│   ├── conversation/
│   │   └── conversation_<uid>.csv # start_time, end_time, inference
│   └── phonelock/
│       └── phonelock_<uid>.csv    # start_timestamp, end_timestamp, lock
├── EMA/
│   └── response/
│       └── Stress/
│           └── Stress_<uid>.json  # resp_time, level (1-5)
└── survey/
    └── PHQ-9.csv                  # PHQ-9 questionnaire responses (optional)
```

The `<uid>` placeholders are participant identifiers (e.g. `u00`, `u01`, …).

## Minimum data requirements

A participant is included in training only if **all three** of these
modalities are present and contain sufficient data:

| Modality | Minimum records |
|----------|----------------|
| GPS      | 100 clean points after filtering |
| Activity | 10 rows |
| Stress   | 3 EMA responses |

Conversation and phone-lock data are used as optional social features;
participants missing these files are still included.
