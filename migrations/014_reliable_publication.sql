-- Apply after 013, before deploying builders that call these RPCs.
-- Additive client schema; all writers/functions below are backend-only.
begin;

create table if not exists public.pipeline_status (
    product text primary key,
    built_at timestamptz not null,
    metadata jsonb not null default '{}'::jsonb
);
alter table public.pipeline_status enable row level security;
revoke all on public.pipeline_status from public, anon, authenticated;
grant all on public.pipeline_status to service_role;

alter table public.predictions add column if not exists curve_pct double precision;
alter table public.predictions add column if not exists curve_version text;
alter table public.prediction_snapshots add column if not exists metadata jsonb not null default '{}'::jsonb;

create or replace function public.publish_weekly_averages(p_rows jsonb, p_built_at timestamptz)
returns boolean language plpgsql set search_path = public, pg_temp as $$
begin
    perform pg_advisory_xact_lock(hashtext('publish_weekly_averages'));
    if exists (select 1 from pipeline_status where product = 'weekly_averages' and built_at >= p_built_at) then
        return false;
    end if;
    if p_built_at is null or jsonb_typeof(p_rows) <> 'array' or jsonb_array_length(p_rows) < 100 then
        raise exception 'Incomplete weekly publication';
    end if;
    if exists (select 1 from jsonb_to_recordset(p_rows) as r(
        day_of_week text, hour_slot float8, avg_pct float8, range_type text, semester_only boolean)
        where day_of_week is null or day_of_week not in ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
        or hour_slot is null or not (hour_slot >= 0 and hour_slot <= 24)
        or avg_pct is null or not (avg_pct >= 0 and avg_pct < 'Infinity'::float8)
        or range_type is null or semester_only is null) then
        raise exception 'Invalid weekly row';
    end if;
    if (select count(distinct r->>'day_of_week') from jsonb_array_elements(p_rows) r
        where r->>'range_type' = 'all_time') <> 7 then
        raise exception 'Weekly publication missing all-time weekdays';
    end if;
    if exists (select 1 from jsonb_array_elements(p_rows) r
        group by r->>'day_of_week', r->>'hour_slot', r->>'range_type', r->>'semester_only' having count(*) > 1) then
        raise exception 'Duplicate weekly keys';
    end if;
    delete from weekly_averages;
    insert into weekly_averages(day_of_week, hour_slot, avg_pct, range_type, semester_only)
        select day_of_week, hour_slot, avg_pct, range_type, semester_only
        from jsonb_to_recordset(p_rows) as r(day_of_week text, hour_slot float8, avg_pct float8, range_type text, semester_only boolean);
    insert into pipeline_status values ('weekly_averages', p_built_at, jsonb_build_object('rows', jsonb_array_length(p_rows)))
        on conflict (product) do update set built_at = excluded.built_at, metadata = excluded.metadata;
    return true;
end $$;

create or replace function public.publish_predictions(p_rows jsonb, p_built_at timestamptz, p_metadata jsonb)
returns boolean language plpgsql set search_path = public, pg_temp as $$
declare first_slot timestamptz;
begin
    perform pg_advisory_xact_lock(hashtext('publish_predictions'));
    if exists (select 1 from pipeline_status where product = 'predictions' and built_at >= p_built_at) then
        return false;
    end if;
    if p_built_at is null or p_metadata is null or jsonb_typeof(p_rows) <> 'array' or jsonb_array_length(p_rows) < 1000 then
        raise exception 'Incomplete predictions publication';
    end if;
    if exists (select 1 from jsonb_to_recordset(p_rows) as r(slot_ts timestamptz, pct float8, curve_pct float8, curve_version text)
        where slot_ts is null or pct is null or not (pct >= 0 and pct <= 100)
        or curve_pct is null or not (curve_pct >= 0 and curve_pct < 'Infinity'::float8) or curve_version is null) then
        raise exception 'Invalid prediction row';
    end if;
    select min((r->>'slot_ts')::timestamptz) into first_slot from jsonb_array_elements(p_rows) r;
    if first_slot < p_built_at - interval '1 day' or first_slot > p_built_at + interval '7 days' then
        raise exception 'Prediction horizon starts outside publication window';
    end if;
    -- Retain recent history needed by clients. Replace the generated horizon,
    -- removing obsolete closure/after-hours slots and obsolete far-future rows.
    delete from predictions where slot_ts >= date_trunc('day', p_built_at at time zone 'America/Los_Angeles') at time zone 'America/Los_Angeles'
        or slot_ts < p_built_at - interval '3 days';
    insert into predictions(slot_ts, pct, curve_pct, curve_version)
        select slot_ts, pct, curve_pct, curve_version from jsonb_to_recordset(p_rows)
        as r(slot_ts timestamptz, pct float8, curve_pct float8, curve_version text)
        on conflict (slot_ts) do update set pct=excluded.pct, curve_pct=excluded.curve_pct, curve_version=excluded.curve_version;
    insert into pipeline_status values ('predictions', p_built_at, p_metadata)
        on conflict (product) do update set built_at=excluded.built_at, metadata=excluded.metadata;
    return true;
end $$;

create or replace function public.publish_today_summary(p_date text, p_preds jsonb, p_computed_at timestamptz, p_metadata jsonb)
returns boolean language plpgsql set search_path = public, pg_temp as $$
begin
    perform pg_advisory_xact_lock(hashtext('publish_today_summary'));
    if p_date is null or p_computed_at is null or p_metadata is null or jsonb_typeof(p_preds) <> 'array' then
        raise exception 'Invalid today publication';
    end if;
    if exists (select 1 from today_summary where date=p_date and computed_at >= p_computed_at) then
        return false;
    end if;
    if p_date::date <> (p_computed_at at time zone 'America/Los_Angeles')::date then
        raise exception 'Today publication date does not match timestamp';
    end if;
    insert into today_summary(date, similarity_preds, blend_weight, computed_at)
        values (p_date, p_preds, 1.0, p_computed_at)
        on conflict (date) do update set similarity_preds=excluded.similarity_preds,
            blend_weight=excluded.blend_weight, computed_at=excluded.computed_at;
    insert into pipeline_status values ('today_summary', p_computed_at, p_metadata)
        on conflict (product) do update set built_at=excluded.built_at, metadata=excluded.metadata
        where pipeline_status.built_at < excluded.built_at;
    return true;
end $$;

revoke all on function public.publish_weekly_averages(jsonb,timestamptz) from public, anon, authenticated;
revoke all on function public.publish_predictions(jsonb,timestamptz,jsonb) from public, anon, authenticated;
revoke all on function public.publish_today_summary(text,jsonb,timestamptz,jsonb) from public, anon, authenticated;
grant execute on function public.publish_weekly_averages(jsonb,timestamptz) to service_role;
grant execute on function public.publish_predictions(jsonb,timestamptz,jsonb) to service_role;
grant execute on function public.publish_today_summary(text,jsonb,timestamptz,jsonb) to service_role;

create table if not exists public.notification_deliveries (
    token text not null,
    kind text not null,
    scheduled_at timestamptz not null,
    claimed_at timestamptz not null default now(),
    status text not null default 'claimed' check (status in ('claimed','sent','failed','unknown','invalid')),
    primary key(token, kind, scheduled_at)
);
alter table public.notification_deliveries enable row level security;
revoke all on public.notification_deliveries from public, anon, authenticated;
grant all on public.notification_deliveries to service_role;

create or replace function public.claim_notification(p_token text, p_kind text, p_scheduled_at timestamptz)
returns boolean language plpgsql set search_path = public, pg_temp as $$
declare n integer;
begin
    insert into notification_deliveries(token,kind,scheduled_at) values(p_token,p_kind,p_scheduled_at)
        on conflict do nothing;
    get diagnostics n = row_count;
    return n = 1;
end $$;
revoke all on function public.claim_notification(text,text,timestamptz) from public, anon, authenticated;
grant execute on function public.claim_notification(text,text,timestamptz) to service_role;

commit;
notify pgrst, 'reload schema';
