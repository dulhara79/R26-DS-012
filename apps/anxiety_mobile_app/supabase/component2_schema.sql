-- Aura / Component 2 research storage
-- Run once in Supabase Dashboard -> SQL Editor.
-- Mobile clients use ONLY the publishable key + authenticated user session.

begin;

create table if not exists public.participants (
  auth_user_id uuid primary key references auth.users(id) on delete cascade,
  participant_code text not null unique,
  enrolled_at timestamptz not null default now(),
  consent_version text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint participant_code_format
    check (participant_code ~ '^P_[A-F0-9]{16}$')
);

create table if not exists public.sensor_events (
  id bigint generated always as identity primary key,
  event_id text not null unique,
  auth_user_id uuid not null references auth.users(id) on delete cascade,
  participant_code text not null,
  event_time timestamptz not null,
  event_type text not null,
  value_json jsonb not null default '{}'::jsonb,
  source text not null default 'android',
  received_at timestamptz not null default now(),
  constraint sensor_event_participant_fk
    foreign key (participant_code)
    references public.participants(participant_code)
    on update cascade
    on delete cascade
);

create index if not exists idx_sensor_events_user_time
  on public.sensor_events(auth_user_id, event_time desc);
create index if not exists idx_sensor_events_participant_time
  on public.sensor_events(participant_code, event_time desc);
create index if not exists idx_sensor_events_type_time
  on public.sensor_events(event_type, event_time desc);

create table if not exists public.daily_behavior_features (
  auth_user_id uuid not null references auth.users(id) on delete cascade,
  participant_code text not null,
  feature_date date not null,

  screen_minutes double precision,
  unlock_count integer,
  night_screen_minutes double precision,

  distance_km double precision,
  home_minutes double precision,
  significant_places integer,
  location_entropy double precision,

  movement_mean double precision,
  movement_variability double precision,
  high_motion_fraction double precision,

  social_media_minutes double precision,
  entertainment_minutes double precision,
  education_minutes double precision,

  incoming_calls integer,
  outgoing_calls integer,
  missed_calls integer,
  rejected_calls integer,
  sms_sent integer,
  sms_received integer,

  routine_regularity double precision,

  location_coverage double precision,
  screen_coverage double precision,
  movement_coverage double precision,
  usable_day boolean not null default false,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  primary key (auth_user_id, feature_date),
  constraint daily_feature_participant_fk
    foreign key (participant_code)
    references public.participants(participant_code)
    on update cascade
    on delete cascade
);

create index if not exists idx_daily_features_participant_date
  on public.daily_behavior_features(participant_code, feature_date desc);

create table if not exists public.behavioral_observations (
  id bigint generated always as identity primary key,
  auth_user_id uuid not null references auth.users(id) on delete cascade,
  participant_code text not null,
  window_start date not null,
  window_end date not null,
  baseline_ready boolean not null default false,
  reportable boolean not null default false,
  observations jsonb not null default '{}'::jsonb,
  data_quality jsonb not null default '{}'::jsonb,
  change_detection jsonb,
  model_output jsonb,
  model_status text not null default 'withheld_pending_validation',
  created_at timestamptz not null default now(),
  constraint observation_participant_fk
    foreign key (participant_code)
    references public.participants(participant_code)
    on update cascade
    on delete cascade
);

create index if not exists idx_observations_participant_created
  on public.behavioral_observations(participant_code, created_at desc);

-- RLS -----------------------------------------------------------------------
alter table public.participants enable row level security;
alter table public.sensor_events enable row level security;
alter table public.daily_behavior_features enable row level security;
alter table public.behavioral_observations enable row level security;

-- Re-running this migration is safe.
drop policy if exists "participant insert own profile" on public.participants;
create policy "participant insert own profile"
on public.participants
for insert
to authenticated
with check ((select auth.uid()) = auth_user_id);

drop policy if exists "participant read own profile" on public.participants;
create policy "participant read own profile"
on public.participants
for select
to authenticated
using ((select auth.uid()) = auth_user_id);

drop policy if exists "participant update own profile" on public.participants;
create policy "participant update own profile"
on public.participants
for update
to authenticated
using ((select auth.uid()) = auth_user_id)
with check ((select auth.uid()) = auth_user_id);

drop policy if exists "participant insert own sensor events" on public.sensor_events;
create policy "participant insert own sensor events"
on public.sensor_events
for insert
to authenticated
with check (
  (select auth.uid()) = auth_user_id
  and exists (
    select 1
    from public.participants p
    where p.auth_user_id = (select auth.uid())
      and p.participant_code = sensor_events.participant_code
      and p.active = true
  )
);

-- Participants can read processed summaries only. Raw sensor_events deliberately
-- has no SELECT policy for authenticated mobile clients.
drop policy if exists "participant read own daily features" on public.daily_behavior_features;
create policy "participant read own daily features"
on public.daily_behavior_features
for select
to authenticated
using ((select auth.uid()) = auth_user_id);

drop policy if exists "participant read own behavioral observations" on public.behavioral_observations;
create policy "participant read own behavioral observations"
on public.behavioral_observations
for select
to authenticated
using ((select auth.uid()) = auth_user_id);

-- Data API privileges. RLS remains the row-level gate.
revoke all on public.participants from anon;
revoke all on public.sensor_events from anon;
revoke all on public.daily_behavior_features from anon;
revoke all on public.behavioral_observations from anon;

grant select, insert, update on public.participants to authenticated;
grant insert on public.sensor_events to authenticated;
grant select on public.daily_behavior_features to authenticated;
grant select on public.behavioral_observations to authenticated;

grant usage, select on sequence public.sensor_events_id_seq to authenticated;
grant usage, select on sequence public.behavioral_observations_id_seq to authenticated;

-- Trusted backend processing can use the service_role. Never ship that key in
-- Flutter. The backend will write daily_behavior_features and observations.
grant all on public.participants to service_role;
grant all on public.sensor_events to service_role;
grant all on public.daily_behavior_features to service_role;
grant all on public.behavioral_observations to service_role;
grant all on sequence public.sensor_events_id_seq to service_role;
grant all on sequence public.behavioral_observations_id_seq to service_role;

commit;
