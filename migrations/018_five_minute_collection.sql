-- Isolated cadence experiment. No production table, view, or function changes.
begin;

create table public.capacity_samples_5m (
    tick_at timestamptz primary key,
    observed_at timestamptz not null,
    people_count integer not null check (people_count >= 0),
    experiment text not null default 'cadence-5m-v1'
        check (experiment = 'cadence-5m-v1'),
    -- The production forecast available BEFORE the sensor request started.
    -- Failure to capture it must not discard an irreplaceable sensor reading.
    production jsonb not null check (
        jsonb_typeof(production) = 'object'
        and production ? 'status'
        and production->>'status' in ('ok', 'missing', 'stale', 'error')
    ),
    inserted_at timestamptz not null default now(),
    constraint capacity_samples_5m_tick check (
        mod(extract(epoch from tick_at), 300) = 0
        and observed_at >= tick_at
        and observed_at < tick_at + interval '5 minutes'
    )
);

comment on table public.capacity_samples_5m is
    'Raw 5-minute experiment readings and contemporaneous production controls. Never an input to production training, forecasts, sensor checks, or notifications. One immutable first observation per tick.';

alter table public.capacity_samples_5m enable row level security;
revoke all on public.capacity_samples_5m from public, anon, authenticated, service_role;
grant select, insert on public.capacity_samples_5m to service_role;

insert into public.schema_migrations(version, name)
values ('018', 'five_minute_collection');

commit;
notify pgrst, 'reload schema';

-- Stop the cron job and unset FIVE_MINUTE_EXPERIMENT_ENABLED to stop collection.
-- Retain this table for analysis. No production rollback or model rebuild needed.
