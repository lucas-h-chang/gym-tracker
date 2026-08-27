-- 011_device_tokens_anon_upsert.sql
--
-- Lets the iOS app register its APNs device token. Without this, EVERY push
-- notification feature the app advertises is silently dead.
--
-- Symptom: a user enables "Daily Summary" or "Workout Reminder" in the Alerts
-- tab, the toggle sticks, and no notification ever arrives. Nothing in the app
-- reports an error, because NotificationManager.uploadDeviceToken() discarded
-- the response (`_ = try? await ...`) — fixed in the same change as this file,
-- which now logs the status code.
--
-- Diagnosis: both notification types are SERVER-sent. NotificationManager
-- schedules nothing locally any more (scheduleAllNotifications only clears a
-- legacy identifier); send_workout_notifications.py fans them out from
-- `device_tokens` as service_role. The app's only job is to POST its token to
-- PostgREST with the publishable key. That POST returns:
--
--   HTTP 401
--   {"code":"42501",
--    "message":"new row violates row-level security policy for table \"device_tokens\""}
--
-- RLS is enabled on the table with no INSERT policy, so anon can never write.
-- No prior migration ever granted it — the table predates migrations/ and RLS
-- was most likely switched on out of band via the dashboard's security advisor
-- (the same way weekly_averages lost its reads in 007), which enables RLS
-- without creating any policy.
--
-- Fix: add write-only access for anon. Deliberately asymmetric:
--
--   INSERT + UPDATE  yes  — the app upserts with `Prefer: resolution=merge-duplicates`,
--                           which is an INSERT ... ON CONFLICT and needs BOTH.
--   SELECT           NO   — the token list stays unreadable with the public key.
--                           This is what stops the publishable key (which ships
--                           inside the app binary and is trivially extractable)
--                           from becoming a way to enumerate every user's device.
--   DELETE           NO   — nothing in the client deletes; a stale token just
--                           starts failing at APNs and is pruned server-side.
--
-- The residual risk is accepted and bounded: anyone holding the publishable key
-- can write a row, but to overwrite an EXISTING user's prefs they must already
-- know that user's 64-hex APNs token, which they cannot read back out. Worst
-- case is junk rows, which cost one rejected APNs call each.
--
-- Column-level grants keep the writable surface to exactly the three columns
-- the app sends, so a future column (say an internal flag) is not writable by
-- the public key just because it was added.

begin;

-- Should already be on; make it explicit so the end state is unambiguous.
alter table public.device_tokens enable row level security;

-- Write-only surface. Note WITH CHECK on INSERT and BOTH USING + WITH CHECK on
-- UPDATE: an ON CONFLICT DO UPDATE is checked against the UPDATE policy's
-- USING clause for the row it lands on, so omitting USING would make every
-- second registration from the same device fail.
drop policy if exists device_tokens_insert_anon on public.device_tokens;
create policy device_tokens_insert_anon
    on public.device_tokens
    for insert
    to anon, authenticated
    with check (true);

drop policy if exists device_tokens_update_anon on public.device_tokens;
create policy device_tokens_update_anon
    on public.device_tokens
    for update
    to anon, authenticated
    using (true)
    with check (true);

-- Belt and braces: RLS filters rows, GRANTs decide which COLUMNS are reachable
-- at all. Revoke first so this is idempotent and so no previously granted
-- column stays writable.
revoke all on public.device_tokens from anon, authenticated;
grant insert (token, prefs, last_seen) on public.device_tokens to anon, authenticated;
grant update (token, prefs, last_seen) on public.device_tokens to anon, authenticated;
-- No `grant select` on purpose. PostgREST needs no SELECT for a plain upsert as
-- long as the request does not ask for the row back — the app sends no
-- `Prefer: return=representation`, so it gets 201 with an empty body.

commit;

-- PostgREST caches schema + permissions; reload so this takes effect now
-- rather than on the next periodic refresh.
notify pgrst, 'reload schema';

-- ─────────────────────────────────────────────────────────────────────────────
-- VERIFY
--
-- 1) Policies are in place (run in the SQL editor):
--      select policyname, cmd, roles from pg_policies
--      where schemaname = 'public' and tablename = 'device_tokens';
--    -- expect device_tokens_insert_anon (INSERT) and device_tokens_update_anon (UPDATE)
--
-- 2) Column grants are exactly the three (SQL editor):
--      select grantee, privilege_type, column_name
--      from information_schema.column_privileges
--      where table_name = 'device_tokens' and grantee in ('anon','authenticated')
--      order by grantee, privilege_type, column_name;
--    -- expect INSERT + UPDATE on token, prefs, last_seen only; NO SELECT rows
--
-- 3) The write actually works, from a terminal with the PUBLISHABLE key.
--    This inserts a throwaway row; step 4 removes it.
--      curl -s -o /dev/null -w '%{http_code}\n' \
--        -X POST "$SUPABASE_URL/rest/v1/device_tokens" \
--        -H "apikey: $SUPABASE_ANON_KEY" \
--        -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
--        -H "Content-Type: application/json" \
--        -H "Prefer: resolution=merge-duplicates" \
--        -d '{"token":"00migration011verify000000000000000000000000000000000000000000",
--             "prefs":"{}","last_seen":"2026-08-27T00:00:00Z"}'
--    -- expect 201 (was 401 before this migration). Run it TWICE — the second
--    -- call exercises the ON CONFLICT path and must also return 201.
--
-- 4) Reads are still shut, and clean up (terminal, publishable key):
--      curl -s "$SUPABASE_URL/rest/v1/device_tokens?select=token&limit=5" \
--        -H "apikey: $SUPABASE_ANON_KEY" -H "Authorization: Bearer $SUPABASE_ANON_KEY"
--    -- expect [] — anon must NOT be able to read tokens back
--    then, in the SQL editor:
--      delete from public.device_tokens
--      where token = '00migration011verify000000000000000000000000000000000000000000';
--
-- 5) End to end: this can only be confirmed on a TestFlight or App Store build.
--    Debug builds get aps-environment=development and therefore SANDBOX APNs
--    tokens, which send_workout_notifications.py's production host
--    (https://api.push.apple.com) rejects. A token registered from Xcode will
--    land in this table and still never receive anything.
--
-- ─────────────────────────────────────────────────────────────────────────────
-- ROLLBACK (paste and run if you need to undo this):
--
--   begin;
--   drop policy if exists device_tokens_insert_anon on public.device_tokens;
--   drop policy if exists device_tokens_update_anon on public.device_tokens;
--   revoke all on public.device_tokens from anon, authenticated;
--   commit;
--   notify pgrst, 'reload schema';
--
-- (This returns the table to its pre-migration state: RLS on, no anon policy,
-- every write 401ing — i.e. push notifications dead again.)
-- ─────────────────────────────────────────────────────────────────────────────
