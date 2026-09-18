-- Keep existing view columns/types while measuring time from publication.
-- Empty carry predictions mean clients served the baseline: score those too.
begin;
create or replace view public.prediction_snapshot_points as
select s.id as snapshot_id, s.date, s.computed_at, s.source, s.model,
    s.cut_slot, s.last_slot, s.n_obs,
    ((p->>'x')::numeric * 4)::int as slot,
    (p->>'y')::numeric as pct,
    (s.base ->> (((p->>'x')::numeric * 4)::int)::text)::numeric as base_pct,
    extract(epoch from (
        (s.date::timestamp + (p->>'x')::numeric * interval '1 hour') at time zone 'America/Los_Angeles'
        - s.computed_at)) / 3600.0 as horizon_h,
    (s.curve ->> (((p->>'x')::numeric * 4)::int)::text)::numeric as curve_pct
from public.prediction_snapshots s
cross join lateral jsonb_array_elements(
    case when jsonb_array_length(s.preds) > 0 then s.preds else
        (select coalesce(jsonb_agg(jsonb_build_object('x', k::numeric/4, 'y', v::numeric)), '[]'::jsonb)
         from jsonb_each_text(s.base) b(k,v)) end
) p
where (s.date::timestamp + (p->>'x')::numeric * interval '1 hour') at time zone 'America/Los_Angeles' > s.computed_at;
revoke all on public.prediction_snapshot_points from public, anon, authenticated;
grant select on public.prediction_snapshot_points to service_role;
commit;
notify pgrst, 'reload schema';
