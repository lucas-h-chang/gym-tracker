-- 010_holiday_closures.sql
--
-- Extends is_rsf_closed_day() (from 009) beyond Caltopia to the holiday
-- closures: Thanksgiving Day, Christmas Eve + Day, and New Year's Day.
--
-- WHY THIS MATTERS BEYOND DISPLAY
-- Density keeps answering on a day the building is shut, reporting 0-2 people
-- for all 96 readings. Unguarded, those days are indistinguishable from a very
-- quiet real day, so they drag down every weekday baseline that averages them
-- and they read to the sensor-stall detector as dead hardware. Verified against
-- capacity_log: on each date below the day's PEAK occupancy stayed at or under
-- 5 people for the entire day.
--
-- ENUMERATED, NOT DERIVED. The policy widened over time and a blanket
-- "this holiday is always closed" rule would wrongly exclude five real days:
--   Christmas Eve  was OPEN 2022 (peak 25.5%) and 2023 (31%), closed from 2024
--   New Year's Day was OPEN 2022 (20.5%), 2023 (14%), 2024 (38%), closed from 2025
-- Only Thanksgiving and Christmas Day are closed in every year on record.
--
-- Generated from academic_calendar.py::CLOSURES. Keep the two in step; this
-- SQL copy is the one mirror sync_calendar.py does NOT write.
--
-- The day_profiles view created in 009 calls this function rather than
-- inlining the dates, so replacing the function is enough. No view rebuild.

create or replace function is_rsf_closed_day(d date)
returns boolean
language sql
immutable
as $$
  select exists (
    select 1
    from (values
      (date '2021-08-22', date '2021-08-23'),  -- Caltopia
      (date '2022-08-21', date '2022-08-22'),  -- Caltopia
      (date '2023-08-20', date '2023-08-21'),  -- Caltopia
      (date '2024-08-25', date '2024-08-26'),  -- Caltopia
      (date '2025-08-24', date '2025-08-25'),  -- Caltopia
      (date '2026-08-23', date '2026-08-25'),  -- Caltopia
      (date '2027-08-22', date '2027-08-23'),  -- Caltopia
      (date '2022-11-24', date '2022-11-24'),  -- Thanksgiving
      (date '2022-12-25', date '2022-12-25'),  -- Christmas
      (date '2023-11-23', date '2023-11-23'),  -- Thanksgiving
      (date '2023-12-25', date '2023-12-25'),  -- Christmas
      (date '2024-11-28', date '2024-11-28'),  -- Thanksgiving
      (date '2024-12-24', date '2024-12-25'),  -- Christmas
      (date '2025-01-01', date '2025-01-01'),  -- New Year's Day
      (date '2025-11-27', date '2025-11-27'),  -- Thanksgiving
      (date '2025-12-24', date '2025-12-25'),  -- Christmas
      (date '2026-01-01', date '2026-01-01'),  -- New Year's Day
      (date '2026-11-26', date '2026-11-26'),  -- Thanksgiving
      (date '2026-12-24', date '2026-12-25'),  -- Christmas
      (date '2027-01-01', date '2027-01-01'),  -- New Year's Day
      (date '2027-11-25', date '2027-11-25'),  -- Thanksgiving
      (date '2027-12-24', date '2027-12-25'),  -- Christmas
      (date '2028-01-01', date '2028-01-01')   -- New Year's Day
    ) as r(start_date, end_date)
    where d between r.start_date and r.end_date
  );
$$;

comment on function is_rsf_closed_day(date) is
  'SQL mirror of academic_calendar.py::is_closed_day(). Days the RSF is shut '
  'entirely: Caltopia, Thanksgiving, Christmas Eve + Day, New Year''s Day. '
  'Density still reports 0-2 people on these days, so they must be excluded '
  'from any occupancy average.';
