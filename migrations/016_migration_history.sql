-- Start a database-owned migration ledger without replaying historical migrations.
begin;
create table if not exists public.schema_migrations (
    version text primary key,
    name text not null,
    applied_at timestamptz not null default now()
);
alter table public.schema_migrations enable row level security;
revoke all on public.schema_migrations from public, anon, authenticated;
grant select, insert on public.schema_migrations to service_role;
-- Assert the new baseline really exists before registering it.
do $$ begin
    if to_regprocedure('public.publish_weekly_averages(jsonb,timestamp with time zone)') is null
       or to_regprocedure('public.publish_predictions(jsonb,timestamp with time zone,jsonb)') is null
       or to_regprocedure('public.claim_notification(text,text,timestamp with time zone)') is null
       or position('jsonb_each_text' in pg_get_viewdef('public.prediction_snapshot_points'::regclass, true)) = 0
       or not exists (select 1 from information_schema.columns where table_schema='public'
           and table_name='prediction_snapshots' and column_name='curve') then
        raise exception 'Apply migrations 013 through 015 before initializing this ledger';
    end if;
end $$;
insert into public.schema_migrations(version,name) values
    ('013','snapshot_bare_curve'),
    ('014','reliable_publication'),
    ('015','accuracy_publication_horizon'),
    ('016','migration_history')
on conflict (version) do nothing;
commit;
notify pgrst, 'reload schema';
