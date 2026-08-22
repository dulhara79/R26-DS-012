begin;

-- Allows the trusted processor to keep one observation row per participant
-- and observation window when recalculating the same day.
create unique index if not exists uq_behavioral_observation_user_window
  on public.behavioral_observations(auth_user_id, window_end);

commit;
