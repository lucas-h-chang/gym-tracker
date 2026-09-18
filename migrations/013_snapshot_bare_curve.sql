-- 013_snapshot_bare_curve.sql
-- Preserve the bare curve beside the served curve + trailing baseline.

begin;

alter table public.prediction_snapshots
    add column if not exists curve jsonb not null default '{}'::jsonb;

create or replace view public.prediction_snapshot_points as
select
    s.id as snapshot_id, s.date, s.computed_at, s.source, s.model,
    s.cut_slot, s.last_slot, s.n_obs,
    ((p->>'x')::numeric * 4)::int as slot,
    (p->>'y')::numeric as pct,
    (s.base ->> (((p->>'x')::numeric * 4)::int)::text)::numeric as base_pct,
    (((p->>'x')::numeric * 4) - s.last_slot) / 4.0 as horizon_h,
    -- Appended after every pre-013 column so CREATE OR REPLACE preserves the
    -- existing view contract instead of trying to rename horizon_h in place.
    (s.curve ->> (((p->>'x')::numeric * 4)::int)::text)::numeric as curve_pct
from public.prediction_snapshots s
cross join lateral jsonb_array_elements(s.preds) p;

create or replace view public.prediction_accuracy as
with actual as (
    select
        (c.timestamp at time zone 'America/Los_Angeles')::date as d,
        least(round(
            extract(hour from c.timestamp at time zone 'America/Los_Angeles') * 4
          + extract(minute from c.timestamp at time zone 'America/Los_Angeles') / 15.0
        ), 95)::int as slot,
        avg(c.people_count) / 150.0 * 100.0 as actual_pct
    from public.capacity_log c
    where c.sensor_ok is not false
    group by 1, 2
)
select
    -- Preserve every pre-013 column in its original position. PostgreSQL only
    -- allows CREATE OR REPLACE VIEW to append columns, not insert them midway.
    p.snapshot_id, p.date, p.computed_at, p.source, p.model,
    p.cut_slot, p.last_slot, p.n_obs, p.slot, p.pct, p.base_pct, p.horizon_h,
    a.actual_pct,
    p.pct - a.actual_pct as err,
    abs(p.pct - a.actual_pct) as abs_err,
    p.base_pct - a.actual_pct as base_err,
    abs(p.base_pct - a.actual_pct) as base_abs_err,
    p.curve_pct,
    p.curve_pct - a.actual_pct as curve_err,
    abs(p.curve_pct - a.actual_pct) as curve_abs_err
from public.prediction_snapshot_points p
join actual a on a.d = p.date and a.slot = p.slot;

revoke all on public.prediction_snapshot_points from anon, authenticated, public;
revoke all on public.prediction_accuracy from anon, authenticated, public;
grant select on public.prediction_snapshot_points to service_role;
grant select on public.prediction_accuracy to service_role;

commit;
notify pgrst, 'reload schema';
