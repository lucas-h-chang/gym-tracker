-- 012_prediction_snapshots.sql
--
-- Keep every forecast we publish, not just the latest one.
--
-- THE PROBLEM
-- -----------
-- `today_summary` is ONE ROW per day. today_builder.py upserts it every 15
-- minutes, and it carries only the slots still ahead of the current time. So by
-- the evening there is no record anywhere of what the site told people that
-- morning, and the question "how accurate was the forecast at 9 AM?" can only
-- be answered by re-deriving it with replay_day.py — which rebuilds the entire
-- curve table from history to answer one day's question, and still cannot
-- reproduce what was served bit for bit.
--
-- WHAT THIS ADDS
-- --------------
--   prediction_snapshots         append-only, one row per published forecast
--   prediction_snapshot_points   that row unnested to one row per (slot, pct)
--   prediction_accuracy          those points joined to what actually happened
--
-- With the views in place, live accuracy tracking is a single select:
--
--   select cut_slot / 4.0 as checked_at,
--          count(*) as n,
--          round(avg(abs_err)::numeric, 2)      as live_mae,
--          round(avg(base_abs_err)::numeric, 2) as day_ahead_mae
--   from prediction_accuracy
--   where date = '2026-09-03' and source = 'live'
--   group by 1 order by 1;
--
-- SHAPE NOTES
-- -----------
-- `preds` is stored exactly as today_summary receives it, [{x, y, w, label}],
-- so a snapshot is literally what a client rendered. An EMPTY array is valid
-- and meaningful: below carry_model.MIN_OBSERVED readings today_builder
-- publishes nothing and every client falls back to the bare base curve. Keeping
-- that row means a missing row always means "the job did not run", which is the
-- distinction an audit table exists to make.
--
-- `base` is the uncorrected baseline the correction was applied to. It is
-- duplicated on every row of the day (~40 floats x ~40 writes, about 6 MB a
-- year) on purpose: predictions_builder rebuilds `predictions` daily and purges
-- past slots, so without this copy the baseline a day was actually served from
-- is gone too, and a self-contained row can be scored years later without
-- reasoning about what else was rebuilt in between.
--
-- ACCESS
-- ------
-- Backend-only, matching 006's stance on day_profiles. `prediction_accuracy`
-- joins capacity_log, which 006 deliberately walled off from anon at 3 days;
-- granting these to anon would hand the publishable key a way around that wall.
-- Views are created WITHOUT security_invoker, so they run as owner — which is
-- exactly why they must not be reachable by anon.
--
-- Rollback is at the bottom of this file.

begin;

-- 1) The table ---------------------------------------------------------------

create table if not exists public.prediction_snapshots (
    id           bigint generated always as identity primary key,

    -- PT calendar date being forecast.
    date         date        not null,
    -- When this forecast was published. Also the natural key within a source.
    computed_at  timestamptz not null,

    -- 'live'          = what today_builder.py actually served.
    -- 'reconstructed' = rebuilt after the fact by backfill_snapshots.py from a
    --                   replay_day.py run. Close but NOT identical: replay
    --                   builds the curve table as of the day rather than the
    --                   preceding Sunday, and fits carry coefficients as of the
    --                   month. Filter on this before quoting production
    --                   accuracy — the two are not interchangeable.
    source       text        not null default 'live',
    -- Which forecaster produced it, so a future model swap stays comparable
    -- rather than silently mixing into the same average.
    model        text        not null default 'carry',

    -- Diagnostics from carry_model.compute_gaps. Nullable because a row with no
    -- correction is a real state, not a missing row.
    cut_slot     smallint,   -- last observed quarter-hour slot, 0-95
    last_slot    smallint,   -- last slot that fed the gaps (horizons measure from here)
    n_obs        smallint,   -- readings the correction was computed from
    gap_day      real,       -- today-so-far vs base, in percentage points
    gap_recent   real,       -- last hour vs base
    gap_last     real,       -- last single reading vs base

    preds        jsonb       not null default '[]'::jsonb,  -- [{x, y, w, label}]
    base         jsonb       not null default '{}'::jsonb,  -- {slot: pct}

    inserted_at  timestamptz not null default now(),

    constraint prediction_snapshots_source_chk
        check (source in ('live', 'reconstructed')),
    constraint prediction_snapshots_cut_slot_chk
        check (cut_slot  is null or cut_slot  between 0 and 95),
    constraint prediction_snapshots_last_slot_chk
        check (last_slot is null or last_slot between 0 and 95),
    -- Note what this does and does not buy. It makes the BACKFILL idempotent:
    -- its computed_at is deterministic (top of the cut hour, PT), so re-running
    -- it updates the same rows instead of laying down a duplicate day that
    -- would silently double-weight it in every average.
    --
    -- It does NOT deduplicate the live job, whose computed_at carries
    -- microseconds and is therefore unique on every run. That is deliberate:
    -- this table records forecasts PUBLISHED, and if the 15-min job genuinely
    -- fired twice, two rows is the honest answer. Two near-identical rows a
    -- second apart shift a daily average by a negligible amount; a table that
    -- silently dropped one of two real publishes would be lying about history,
    -- which is worse for the one thing this table is for.
    constraint prediction_snapshots_natural_key unique (source, computed_at)
);

comment on table public.prediction_snapshots is
    'Append-only record of every forecast today_builder.py published. today_summary keeps only the latest; this keeps all of them so accuracy can be measured after the fact.';

create index if not exists prediction_snapshots_date_source_idx
    on public.prediction_snapshots (date, source, computed_at);

-- 2) One row per predicted slot ----------------------------------------------

create or replace view public.prediction_snapshot_points as
select
    s.id            as snapshot_id,
    s.date,
    s.computed_at,
    s.source,
    s.model,
    s.cut_slot,
    s.last_slot,
    s.n_obs,
    -- `x` is the fractional hour the clients plot on; slots are quarter-hours.
    ((p->>'x')::numeric * 4)::int                             as slot,
    (p->>'y')::numeric                                        as pct,
    -- The baseline for the same slot, pulled out of the row's own snapshot.
    (s.base ->> (((p->>'x')::numeric * 4)::int)::text)::numeric as base_pct,
    -- Hours ahead, measured from last_slot — the same origin carry_model uses,
    -- so these bucket the same way replay_day.py's `horizon` column does.
    (((p->>'x')::numeric * 4) - s.last_slot) / 4.0            as horizon_h
from public.prediction_snapshots s
cross join lateral jsonb_array_elements(s.preds) p;

comment on view public.prediction_snapshot_points is
    'prediction_snapshots.preds unnested to one row per forecast slot, with the matching baseline and the horizon in hours.';

-- 3) Those points scored against what actually happened ----------------------

create or replace view public.prediction_accuracy as
with actual as (
    select
        (c.timestamp at time zone 'America/Los_Angeles')::date as d,
        -- Mirrors academic_calendar.slot_of: round to the NEAREST quarter-hour
        -- and clip at 95. Flooring instead would file a 10:40 scrape at 10:30
        -- and disagree with what today_builder scored itself against.
        least(
            round(
                extract(hour   from c.timestamp at time zone 'America/Los_Angeles') * 4
              + extract(minute from c.timestamp at time zone 'America/Los_Angeles') / 15.0
            ), 95
        )::int as slot,
        -- people_count / MAX_CAPACITY, matching carry_data.py rather than the
        -- stored percent_full, which is pre-rounded to one decimal.
        avg(c.people_count) / 150.0 * 100.0 as actual_pct
    from public.capacity_log c
    -- Drop readings taken while the counter was dead (migration 008), the same
    -- filter carry_data.py and build_curves.py apply. No `> 5` floor here: that
    -- one exists to drop closed-gym rows from TRAINING, and applying it to
    -- scoring would quietly delete the genuinely near-empty opening slot.
    where c.sensor_ok is not false
    group by 1, 2
)
select
    p.*,
    a.actual_pct,
    p.pct      - a.actual_pct  as err,
    abs(p.pct  - a.actual_pct) as abs_err,
    p.base_pct - a.actual_pct  as base_err,
    abs(p.base_pct - a.actual_pct) as base_abs_err
from public.prediction_snapshot_points p
join actual a
  on a.d = p.date
 and a.slot = p.slot;

comment on view public.prediction_accuracy is
    'Every published forecast joined to the reading that later landed in that slot. abs_err scores the live model, base_abs_err the day-ahead baseline it corrected.';

-- 4) Backend-only access -----------------------------------------------------

-- RLS on the table with no policy: anon/authenticated get nothing, service_role
-- bypasses RLS entirely, so today_builder.py and backfill_snapshots.py are
-- unaffected.
alter table public.prediction_snapshots enable row level security;

revoke all on public.prediction_snapshots       from anon, authenticated, public;
revoke all on public.prediction_snapshot_points from anon, authenticated, public;
revoke all on public.prediction_accuracy        from anon, authenticated, public;

grant select, insert, update on public.prediction_snapshots to service_role;
grant select on public.prediction_snapshot_points to service_role;
grant select on public.prediction_accuracy        to service_role;

commit;

-- PostgREST caches schema + permissions; reload so this takes effect now rather
-- than on the next periodic refresh.
notify pgrst, 'reload schema';

-- ─────────────────────────────────────────────────────────────────────────────
-- VERIFY
--
-- 1) The objects exist (SQL editor):
--      select table_name, table_type from information_schema.tables
--      where table_schema = 'public'
--        and table_name in ('prediction_snapshots','prediction_snapshot_points',
--                           'prediction_accuracy');
--    -- expect 1 BASE TABLE and 2 VIEWs
--
-- 2) anon cannot reach any of them (terminal, PUBLISHABLE key):
--      for t in prediction_snapshots prediction_snapshot_points prediction_accuracy; do
--        curl -s "$SUPABASE_URL/rest/v1/$t?select=*&limit=1" \
--          -H "apikey: $SUPABASE_ANON_KEY" -H "Authorization: Bearer $SUPABASE_ANON_KEY"
--        echo
--      done
--    -- expect a permission-denied / empty result for each, NOT rows
--
-- 3) After the next today_builder.py run (within 15 min, gym open), rows appear:
--      select computed_at, source, n_obs, jsonb_array_length(preds) as slots
--      from prediction_snapshots order by computed_at desc limit 5;
--
-- 4) The payoff query works end to end:
--      select source, count(*) as n,
--             round(avg(abs_err)::numeric, 2)      as live_mae,
--             round(avg(base_abs_err)::numeric, 2) as day_ahead_mae
--      from prediction_accuracy group by source;
--
-- ─────────────────────────────────────────────────────────────────────────────
-- ROLLBACK (paste and run if you need to undo this):
--
--   begin;
--   drop view  if exists public.prediction_accuracy;
--   drop view  if exists public.prediction_snapshot_points;
--   drop table if exists public.prediction_snapshots;
--   commit;
--   notify pgrst, 'reload schema';
--
-- (Destructive: it discards every snapshot recorded so far. Nothing else reads
-- this table — today_summary still drives the site — so dropping it degrades
-- accuracy tracking only, never the forecast itself.)
-- ─────────────────────────────────────────────────────────────────────────────
