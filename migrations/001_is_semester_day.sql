-- 001_is_semester_day.sql
--
-- SQL equivalent of academic_calendar.py::is_semester_day().
--
-- Python source (gym-tracker/academic_calendar.py):
--   BREAK_RANGES = WINTER_BREAK_RANGES + SPRING_BREAK_RANGES + SUMMER_BREAK_RANGES
--   def is_semester_day(d):
--       return not _in_any(d, BREAK_RANGES)
--   def _in_any(d, ranges):
--       return any(start <= d <= end for start, end in ranges)
--
-- i.e. "in session" iff the date is NOT inside any winter, spring, or summer
-- break range. All three range lists below are copied verbatim from
-- academic_calendar.py (WINTER_BREAK_RANGES, SPRING_BREAK_RANGES,
-- SUMMER_BREAK_RANGES) -- do not retype from memory if this ever needs
-- updating; copy the new tuples from that file.
--
-- NOTE: this is intentionally the same "break-range membership" the Python
-- docstring describes, not classify_date()'s season-aware label. It matches
-- classify_date() on every day except the ~1-2/year where the last finals
-- day coincides with the first listed day of the following break; that seam
-- is immaterial to the semester-only aggregates this gates (see the Python
-- docstring for the full rationale).

create or replace function is_semester_day(d date)
returns boolean
language sql
immutable
as $$
  select not exists (
    select 1
    from (
      values
        -- WINTER_BREAK_RANGES
        (date '2020-12-18', date '2021-01-18'),
        (date '2021-12-17', date '2022-01-17'),
        (date '2022-12-16', date '2023-01-16'),
        (date '2023-12-15', date '2024-01-15'),
        (date '2024-12-20', date '2025-01-20'),
        (date '2025-12-19', date '2026-01-19'),
        (date '2026-12-18', date '2027-01-18'),
        (date '2027-12-17', date '2028-01-17'),
        -- SPRING_BREAK_RANGES
        (date '2021-03-20', date '2021-03-28'),
        (date '2022-03-19', date '2022-03-27'),
        (date '2023-03-25', date '2023-04-02'),
        (date '2024-03-23', date '2024-03-31'),
        (date '2025-03-22', date '2025-03-30'),
        (date '2026-03-21', date '2026-03-29'),
        (date '2027-03-20', date '2027-03-28'),
        (date '2028-03-25', date '2028-04-02'),
        -- SUMMER_BREAK_RANGES
        (date '2021-05-14', date '2021-08-24'),
        (date '2022-05-13', date '2022-08-23'),
        (date '2023-05-12', date '2023-08-22'),
        (date '2024-05-10', date '2024-08-27'),
        (date '2025-05-16', date '2025-08-26'),
        (date '2026-05-15', date '2026-08-25'),
        (date '2027-05-14', date '2027-08-24')
    ) as breaks(start_date, end_date)
    where d between breaks.start_date and breaks.end_date
  );
$$;

comment on function is_semester_day(date) is
  'In-session gate translated from academic_calendar.py::is_semester_day(). '
  'Returns true unless d falls inside a winter/spring/summer break range '
  '(academic_calendar.BREAK_RANGES). Update this function whenever '
  'academic_calendar.py''s WINTER_BREAK_RANGES / SPRING_BREAK_RANGES / '
  'SUMMER_BREAK_RANGES gain a new year -- copy the new tuples verbatim, '
  'do not retype from memory.';
