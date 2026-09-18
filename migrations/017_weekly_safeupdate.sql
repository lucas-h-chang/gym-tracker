-- Supabase safeupdate requires a WHERE clause, including inside invoker RPCs.
-- day_of_week is NOT NULL in the live table, so this still replaces all rows.
-- Keep the surrounding publication transaction and rollback guarantees intact.
begin;
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
    delete from weekly_averages where day_of_week is not null;
    insert into weekly_averages(day_of_week, hour_slot, avg_pct, range_type, semester_only)
        select day_of_week, hour_slot, avg_pct, range_type, semester_only
        from jsonb_to_recordset(p_rows) as r(day_of_week text, hour_slot float8, avg_pct float8, range_type text, semester_only boolean);
    insert into pipeline_status values ('weekly_averages', p_built_at, jsonb_build_object('rows', jsonb_array_length(p_rows)))
        on conflict (product) do update set built_at = excluded.built_at, metadata = excluded.metadata;
    return true;
end $$;
insert into public.schema_migrations(version,name)
values ('017','weekly_safeupdate') on conflict (version) do nothing;
commit;
notify pgrst, 'reload schema';
