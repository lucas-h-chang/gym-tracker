# RSF Gym Tracker — Technical Feature Specification

## Overview

The app shows UC Berkeley RSF weight room occupancy data: live status, ML-powered predictions, and historical patterns. All prediction data is pre-computed server-side every 15 minutes and served as a single JSON file (`data.json`, ~500 KB). Live count is fetched directly from the Density.io sensor API at runtime. The gym max capacity is **150 people**.

---

## Data Sources

### 1. `data.json` (primary, pre-computed)
Fetched from the static file host on app launch. Rebuilt every 15 minutes by a GitHub Actions scraper. Structure:

```
{
  built_at:                "2026-04-20 14:15 PT"         // timestamp string
  today_date:              "2026-04-20"                   // YYYY-MM-DD
  today_actuals:           [{x, y, label}, ...]           // today's real readings
  today_similarity_preds:  [{x, y, label}, ...]           // similarity model preds
  today_blend_weight:      0.45                           // 0.0–0.9, how much to trust similarity
  predictions:             { "2026-04-20_07:00": [rf, mlp], ... }  // 180-day ML predictions
  weekly:                  { "Monday|Last 7 days|true": [{x, y, label}, ...], ... }
  metrics:                 { trained_at, training_rows, rf_mae, mlp_mae }
}
```

**`predictions` key format:** `"{YYYY-MM-DD}_{HH:MM}"` → `[rf_prediction, mlp_prediction]` both as floats (percent full, 0–100). Values exist for every 15-minute slot within open hours for the next 180 days.

**`weekly` key format:** `"{DayName}|{RangeName}|{true|false}"` where range is one of: `Last 7 days`, `Last month`, `Last 6 months`, `Last year`, `All time`, `This semester`. The boolean is the semester-only filter. Value is an array of `{x: hour_numeric, y: percent_full, label: "7:00 AM"}`.

**`today_actuals` and `today_similarity_preds`:** Arrays of `{x: hour_numeric, y: percent_full, label}`. `x` is decimal hour (7.5 = 7:30 AM), `y` is percent 0–100.

### 2. Density.io Live API (secondary, runtime fetch)
Endpoint: `GET https://api.density.io/v2/spaces/spc_863128347956216317/count`
Auth header: `Authorization: Bearer shr_o69HxjQ0BYrY2FPD9HxdirhJYcFDCeRolEd744Uj88e`

Response: `{ count: <integer> }` — number of people currently in the weight room.

Live capacity: `pct = round((count / 150) * 100)`

This fetch only happens when the gym is currently open. If the fetch fails, live data is shown as unavailable (shown as "—").

---

## Gym Operating Hours

| Day | Open | Close |
|---|---|---|
| Mon–Fri | 7 AM | 11 PM |
| Saturday | 8 AM | 6 PM |
| Sunday | 8 AM | 11 PM |

All times are **Pacific Time (America/Los_Angeles)**.

---

## Feature 1: Insight Cards (3 cards, row layout)

Cards are shown for the currently selected prediction date. Cards are always rendered; future dates show "—" for any live-data-dependent values.

### Card 1 — Live Status

**Purpose:** Show the current state of the gym (open/closed, live occupancy, capacity bar).

**Inputs:**
- Current PT time
- Currently selected date (today vs. future)
- `liveCapacity` from Density.io API: `{ count, pct }`
- `data.predictions` (for future-day average/peak)

**Display logic (Today, gym is OPEN):**
- If live data available: shows animated "live dot" + `"{pct}% Full"` at 34px bold, colored by capacity threshold
- If live data unavailable: shows "—" in muted color
- Capacity bar: fills to `pct`%, color thresholds:
  - `< 60%` → green (`#6ee7b7`)
  - `60–79%` → amber (`#fbbf24`)
  - `80–89%` → red (`#f87171`)
  - `≥ 90%` → dark red (`#b91c1c`)
- Bar label: `"{count} / 150 people"`

**Display logic (Today, gym is CLOSED):**
- Shows "CLOSED" in red
- Separator "—"
- Next open time: `"Opens at {time}"` (if before open) or `"Opens tomorrow at {time}"` (after close)
- Capacity bar empty, label "—"

**Display logic (Future day):**
- Shows opening hours as `"{open time} – {close time}"` (e.g., "7:00 AM – 11:00 PM") at smaller font in accent blue
- Capacity bar fills to **average predicted occupancy** across the whole day
- Bar label: `"avg {avg}% · peak {peak}%"`

**Live dot animation:** Solid 11px center circle + 20px pulsing ring, both `currentColor`, ring scales 0.6→1.4 with opacity 0→0 over 1.8s.

---

### Card 2 — Comparison to Previous Days

**Purpose:** Tell the user whether the gym is busier or quieter than usual, compared to historical same-day data.

**Header:** `"NOW VS PREVIOUS {DAYNAME}S"` (e.g., "NOW VS PREVIOUS MONDAYS")

**Inputs:**
- Currently selected day name (e.g., "Monday")
- Reference hour:
  - Today: current fractional hour (e.g., 14.75 for 2:45 PM), rounded to nearest 0.25
  - Future day: 12.0 (noon, as a representative hour)
- `liveCapacity.pct` (reference percentage; only available on today)
- `data.weekly` entries for the selected day at the reference hour:
  - `lastWeek`: from `"{day}|Last 7 days|false"`
  - `thisSemester`: from `"{day}|This semester|true"`
  - `allTime`: from `"{day}|All time|true"`

**Baseline for verdict:** `thisSemester ?? allTime` (prefer this semester, fall back to all time).

**Verdict thresholds** (delta = `livePct - baseline`):

| Delta | Verdict | Color |
|---|---|---|
| > +15% | "Much busier than usual" | Red `#f87171` |
| +5% to +15% | "Busier than usual" | Red `#f87171` |
| -5% to +5% | "About average" | Default text |
| -15% to -5% | "Quieter than usual" | Green `#6ee7b7` |
| < -15% | "Much quieter than usual" | Green `#6ee7b7` |
| No live data | "—" | — |

**Delta row** (below verdict): Three items separated by "·"
Each item: `"{+/-X%} vs {label}"`
- `+2% vs last week`
- `-5% vs this sem`
- `+1% vs all time`

Delta value is `round(livePct - historicalPct)`; colored red if positive, green if negative. Text "vs {label}" is muted color. If `livePct` is null (no live data or future day), all deltas show "—".

---

### Card 3 — Crowd in Next Hour

**Purpose:** Predict whether occupancy will increase or decrease over the next hour.

**Header:** `"CROWD IN NEXT HOUR"`

**Only shown on today** (future days show "—").

**Inputs:**
- `liveCapacity.pct` as `startY` (live occupancy right now)
- `data.predictions` + `today_similarity_preds` + `today_blend_weight` via `getBlendedSlots()`

**`getBlendedSlots(data, fromH, toH, date)` — core helper:**
For each 15-min slot from `fromH` to `toH`:
1. Look up ML prediction: `data.predictions["{date}_{HH:MM}"]` → average of `[rf, mlp]`
2. If today and similarity prediction exists for that slot: blend
   - `blended = (1 - blend_weight) * mlAvg + blend_weight * simPred`
3. Returns array of `{h, m, slotH, y}`

**`computeTrend(data, livePct)`:**
- `fromH` = current time (fractional hour)
- `toH` = `min(nowH + 1, closeH)` — one hour ahead, capped at closing
- Returns null if less than 1 hour until close, or fewer than 2 prediction slots
- `startY` = `livePct` (live occupancy; if unavailable, falls back to first predicted slot)
- `endY` = last blended slot value in the hour window
- `delta` = `endY - startY`
- Direction:
  - `abs(delta) < 5%` → "flat" / "Holding steady" → `→` arrow, white
  - `delta > 0` → "up" / "Getting busier" → `↑` arrow + label, red `#f87171`
  - `delta < 0` → "down" / "Getting quieter" → `↓` arrow + label, green `#6ee7b7`
- Delta percentage shown if not flat: `"+{abs(delta)}%"`
- Sub-label: `"from {livePct}% now"`

---

## Feature 2: Predicted Occupancy Chart

**Purpose:** Line chart showing actual (past) and predicted (future) occupancy for a selected date.

### Date Navigation

A pill control with ← date → buttons inline with the "Predicted Occupancy" header.

- Range: today through today + 179 days (6 months forward)
- Format: `"Today, Apr 20"` for today; `"Monday, Apr 21"` for other days
- ← disabled on today; → disabled at max date

**Calendar popup:** Clicking the date label opens a monthly calendar grid. Disabled dates (outside range) are dimmed. Clicking a date selects it and closes the popup. Navigate months with ← → buttons.

### Chart Data

**Two datasets:**

1. **Actual** (solid line, accent blue `#93a4f5`, with gradient fill): `data.today_actuals` filtered to `x ≤ nowHour`. Only shown on today.
2. **Predicted** (dashed grey line, `#9ca3af`): blended ML predictions for slots `x ≥ nowHour` (today) or all slots (future days).

**Blending for predicted line (today only):**
- For each predicted point: `finalY = (1 - blend_weight) * mlAvg + blend_weight * simPred`
- `blend_weight` grows from 0 to max 0.9 as more of the day is observed (0 at open, 0.9 after 6 hours)

**Chart axes:**
- X: linear, min=7, max=23, integer ticks formatted as "7 AM", "12 PM", "11 PM"
- Y: linear, min=0, max=110, ticks as "0%", "50%", etc.

**"Now" line:** Vertical dashed line at current fractional hour (accent color), labelled "Now" above. Only shown on today.

**Tooltip:**
- Standard data point: `"{time}" / "{dataset}: {y}%"`
- Hovering near Now line: `"{current time}" / "Actual: {livePct}%"` or `"Live data unavailable"`

---

## Feature 3: Weekly Patterns Chart

**Purpose:** Show the typical occupancy curve for a given day of week, filtered by date range and semester status.

### Controls (inline with "Weekly Patterns" header)

| Control | Type | Options |
|---|---|---|
| DAY | Dropdown | Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday |
| DATE RANGE | Dropdown | Last 7 days, Last month, Last 6 months, Last year, All time |
| Semester only | Checkbox | Default: checked |

On any change, chart updates immediately. Default day = current day of week.

### Chart Data

Key lookup: `data.weekly["{day}|{range}|{true/false}"]` → array of `{x, y, label}`.

If the array has no values above 0 (no data for that filter combo), the chart shows empty.

**Single dataset** (solid blue line, same style as Predicted Occupancy Actual line).

**Axes:** Same as Predicted Occupancy chart (X: 7–23, Y: 0–110%).

**Tooltip:** `"{time}" / "{y}% capacity"`

### How Weekly Averages Are Computed (backend, for reference)

For each combination of day × range × semester_flag:
1. Filter `capacity_log` to the date range
2. Optionally filter to semester days only (excludes June/July/August, Dec 16 – Jan 12, Spring Recess)
3. Filter to the given day of week, within open hours
4. Group by 15-minute bin, take mean `percent_full`
5. Append a closing zero point at closing hour

"This semester" range: from the start of the current semester (walk back from today until a non-semester day) to now.

---

## Header

Shows the app name "RSF Crowd Level Predictor". No dynamic state.

---

## Footer

Shows `"UPDATED {data.built_at}"` — the timestamp when `data.json` was last rebuilt, formatted as `"YYYY-MM-DD HH:MM PT"`.

---

## Data Warning Banner

Shown at the top of the page in red if:
- `data.json` fails to load → "Could not load gym data..."
- `data.predictions` is empty → "Prediction data is unavailable..."
- `data.built_at` is more than 48 hours old → "Data may be outdated..."

---

## Key Calculations Summary

| Calculation | Where used | Formula |
|---|---|---|
| Live pct | Cards 1, 2, 3 | `round(count / 150 * 100)` |
| Blended prediction | Charts, Card 3 | `(1 - w) * mlAvg + w * simPred` |
| Comparison delta | Card 2 | `livePct - historicalPct` at same hour |
| Trend direction | Card 3 | `endY - startY` over next hour |
| Capacity bar color | Card 1 | <60 green, 60–79 amber, 80–89 red, ≥90 dark red |
| Verdict threshold | Card 2 | ±5% = average, ±15% = "much" |
| blend_weight | Pred chart, Card 3 | `min((nowHour - openHour) / 6.0, 0.9)` |

---

## iOS Implementation Notes

- All timezone logic must use Pacific Time (`America/Los_Angeles`)
- `data.json` should be fetched fresh on app launch and cached with a 15-minute TTL
- Live Density.io fetch should only happen when gym is currently open; handle failure gracefully
- `today_similarity_preds` and `today_blend_weight` are only meaningful on the current PT date — discard if `data.today_date ≠ today PT date`
- Prediction keys use 24-hour zero-padded format: `"2026-04-20_07:00"` through `"2026-04-20_22:45"`
- Selecting a date affects all 3 insight cards and the prediction chart simultaneously
- The Weekly Patterns chart is independent of the selected prediction date
