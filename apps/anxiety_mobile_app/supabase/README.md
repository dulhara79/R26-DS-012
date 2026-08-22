# Component 2 Supabase setup

The Flutter app uploads research events directly to Supabase using a publishable
client key, an authenticated participant session, and Row Level Security (RLS).
Google Apps Script is not required for Component 2.

## 1. Create the database schema

In the Supabase Dashboard open **SQL Editor**, paste the complete contents of:

`supabase/component2_schema.sql`

and run it once.

This creates:

- `participants`
- `sensor_events`
- `daily_behavior_features`
- `behavioral_observations`
- indexes, grants, and RLS policies

Raw `sensor_events` intentionally has no participant-facing SELECT policy.
Participant clients can insert only rows whose `auth_user_id` matches their own
Supabase authenticated user and whose participant code matches their registered
participant row.

## 2. Enable anonymous authentication for development

In Supabase Dashboard open the Authentication settings and enable **Allow
anonymous sign-ins**.

The app uses an anonymous authenticated Supabase user so no participant email or
phone number is required during the research setup. The pseudonymous app code
(e.g. `P_7F3A9C2E4B10D6C1`) is stored separately in `participants`.

For final recruitment, add an account recovery/re-enrollment plan before relying
on anonymous identities across reinstalls or device changes.

## 3. Run Flutter with the publishable credentials

Preferred Flutter variable names:

```bash
flutter run -d <android-device> \
  --dart-define=SUPABASE_URL=https://YOUR_PROJECT.supabase.co \
  --dart-define=SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
```

The app also accepts the aliases:

```bash
--dart-define=NEXT_PUBLIC_SUPABASE_URL=...
--dart-define=NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=...
```

Do not pass a `service_role`, `sb_secret_...`, or database password to Flutter.

## 4. Data flow

```text
Android sensors
   -> local SharedPreferences offline queue
   -> main Flutter isolate
   -> authenticated Supabase client
   -> public.sensor_events
```

The Android background isolate is collection-only. It never needs a Supabase
session. Its queued events are uploaded by the main isolate on app startup,
foreground activity, or connectivity restoration.

Each event receives a persistent client-generated `event_id`; uploads use an
idempotent upsert so retrying the same queued event does not create duplicates.

## 5. Multiple participants

One Supabase project is used for all study participants. Each installation gets a
separate Supabase Auth UUID, mapped to a pseudonymous participant code:

```text
auth.users UUID A -> P_...
auth.users UUID B -> P_...
auth.users UUID C -> P_...
```

RLS uses the authenticated UUID to prevent one participant session from writing
rows for another participant.

## 6. First verification

After running the SQL and enabling anonymous sign-ins:

1. Run the Android app with the two `--dart-define` values.
2. Complete consent/login/profile setup.
3. Keep the app running long enough to collect events.
4. In Supabase Table Editor, check `participants` for one participant mapping.
5. Check `sensor_events` for event types such as `Screen_Event`,
   `Location_Grid_100m`, `App_Usage_Category_15m`, and `Movement_Window_5m`.
6. Confirm `value_json` contains JSON, not a JSON string.

The next backend step is to aggregate `sensor_events` into one
`daily_behavior_features` row per participant per day and then calculate the
personal-baseline behavioral observations.
