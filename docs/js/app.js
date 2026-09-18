// ── Boot ───────────────────────────────────────────────────────
async function init() {
  // chart.js loads with `defer`, so it isn't available until DOMContentLoaded
  // (which gates this init call). Register plugins here, not at parse time.
  Chart.register(capacityGradientPlugin, closedHoursPlugin, nowLinePlugin, scrubLinePlugin);
  updateStatusBar();
  // Live capacity is shared through a 30s Supabase cache (see
  // api/live-capacity.js), so
  // polling any faster than that just re-fetches the same cached value.
  setInterval(updateStatusBar, 30000);

  const today     = todayPT();
  const yesterday = shiftDate(today, -1);

  let predRows, actualsRows, todaySummaryRows;
  try {
    [predRows, actualsRows, todaySummaryRows] = await Promise.all([
      // First paint only needs the visible pills (today .. today+4). The rest
      // of the 90-day horizon (calendar "More" days) loads in the background
      // below so it can't slow the initial render.
      sbFetch(`predictions?select=slot_ts,pct&slot_ts=gte.${today}T00:00:00&slot_ts=lte.${shiftDate(today, 5)}T12:00:00Z&order=slot_ts.asc&limit=10000`),
      sbFetch(`capacity_log?timestamp=gte.${yesterday}T12:00:00Z&order=timestamp.asc&limit=2000`),
      sbFetch(`today_summary?date=eq.${today}&select=similarity_preds,blend_weight,computed_at`),
    ]);
  } catch (e) {
    console.error('Supabase load failed:', e);
    document.body.classList.remove('loading');
    showDataWarning(`Could not load gym data: ${e.message}`);
    return;
  }

  const weekly = {};
  const weeklyTask = sbFetch('weekly_averages?select=day_of_week,hour_slot,avg_pct,range_type,semester_only&limit=10000')
    .catch(e => { console.error('weekly averages unavailable:', e); return []; });

  const predictions = {};
  predRows.forEach(r => { predictions[slotTsToKey(r.slot_ts)] = r.pct; });

  const ts = todaySummaryRows[0] || {};

  const data = {
    predictions,
    weekly,
    today_actuals:           buildTodayActuals(actualsRows, today),
    today_computed_at: ts.computed_at,
    today_similarity_preds:  ts.similarity_preds || [],
    today_blend_weight:      ts.blend_weight || 0,
  };

  if (Object.keys(predictions).length === 0) {
    showDataWarning('Prediction data is unavailable — the daily build may not have run yet.');
  }

  // predDate follows the day pills; bootDate is pinned to the day this page
  // actually loaded, so refreshTodayData() can tell "user is browsing Friday"
  // apart from "the clock rolled past midnight".
  predDate = today;
  bootDate = today;

  window._insightData = data;

  // Render day pills
  renderPills(data);

  // Calendar popup
  const calPopup = document.getElementById('cal-popup');
  const minDate  = today;
  const maxDate  = shiftDate(today, 90);
  let calYear, calMonth;

  function renderCalendar() {
    const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
    const firstDay = new Date(calYear, calMonth, 1).getDay();
    const daysInMonth = new Date(calYear, calMonth + 1, 0).getDate();
    const prevMonthHasDays = new Date(calYear, calMonth - 1, 28) >= new Date(minDate + 'T00:00:00');
    const nextMonthHasDays = new Date(calYear, calMonth + 1, 1) <= new Date(maxDate + 'T00:00:00');

    let html = `<div class="cal-header">
      <button class="cal-nav-btn" id="cal-prev-month" ${!prevMonthHasDays ? 'disabled' : ''}>←</button>
      <span class="cal-month-label">${months[calMonth]} ${calYear}</span>
      <button class="cal-nav-btn" id="cal-next-month" ${!nextMonthHasDays ? 'disabled' : ''}>→</button>
    </div>
    <div class="cal-grid">`;
    ['Su','Mo','Tu','We','Th','Fr','Sa'].forEach(d => html += `<div class="cal-dow">${d}</div>`);
    for (let i = 0; i < firstDay; i++) html += `<div class="cal-day cal-day-empty"></div>`;
    for (let d = 1; d <= daysInMonth; d++) {
      const ds = `${calYear}-${String(calMonth+1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
      const disabled = ds < minDate || ds > maxDate;
      const selected = ds === predDate;
      const isToday  = ds === today;
      const cls = ['cal-day', disabled ? 'cal-day-disabled' : '', selected ? 'cal-day-selected' : '', isToday ? 'cal-day-today' : ''].filter(Boolean).join(' ');
      html += `<div class="${cls}" data-date="${ds}">${d}</div>`;
    }
    html += '</div>';
    calPopup.innerHTML = html;

    document.getElementById('cal-prev-month').addEventListener('click', () => { calMonth--; if (calMonth < 0) { calMonth = 11; calYear--; } renderCalendar(); });
    document.getElementById('cal-next-month').addEventListener('click', () => { calMonth++; if (calMonth > 11) { calMonth = 0; calYear++; } renderCalendar(); });
    calPopup.querySelectorAll('.cal-day:not(.cal-day-disabled):not(.cal-day-empty)').forEach(el => {
      el.addEventListener('click', () => {
        predDate = el.dataset.date;
        calPopup.style.display = 'none';
        updatePredChart(data);
        renderInsightCards(data);
        updatePillStates();
      });
    });
  }

  // Paired with the document-level pointerdown dismiss below: without this the
  // trigger's own tap would bubble up, close the popup, and then the click
  // handler would immediately re-open it, so tapping "More" while open could
  // never close it.
  document.getElementById('day-pills').addEventListener('pointerdown', (e) => {
    if (e.target.closest('#cal-trigger')) e.stopPropagation();
  });

  document.getElementById('day-pills').addEventListener('click', (e) => {
    const calTrigger = e.target.closest('#cal-trigger');
    if (!calTrigger) return;
    e.stopPropagation();
    if (calPopup.style.display !== 'none') { calPopup.style.display = 'none'; return; }
    const d = new Date(predDate + 'T12:00:00');
    calYear = d.getFullYear(); calMonth = d.getMonth();
    renderCalendar();
    const pillsEl = document.getElementById('day-pills');
    const rect = pillsEl.getBoundingClientRect();
    const sectionRect = document.getElementById('section-predict').getBoundingClientRect();
    calPopup.style.top  = (rect.bottom - sectionRect.top + 8) + 'px';
    calPopup.style.right = '0';
    calPopup.style.left = 'auto';
    calPopup.style.display = 'block';
  });

  // pointerdown, not click: Safari on iOS doesn't deliver click from taps on
  // non-interactive elements, so tapping the page background to dismiss the
  // calendar silently did nothing on an iPhone. pointerdown fires identically
  // for desktop mouse input.
  document.addEventListener('pointerdown', () => { calPopup.style.display = 'none'; });
  calPopup.addEventListener('pointerdown', e => e.stopPropagation());
  calPopup.addEventListener('click', e => e.stopPropagation());

  updatePredChart(data);
  renderInsightCards(data);

  document.body.classList.remove('loading');
  weeklyTask.then(rows => {
    rows.forEach(r => {
      const disp = RANGE_DB_TO_DISPLAY[r.range_type] || r.range_type;
      const key = `${r.day_of_week}|${disp}|${r.semester_only}`;
      (weekly[key] ||= []).push({ x: r.hour_slot, y: r.avg_pct, label: slotToLabel(r.hour_slot) });
    });
    Object.values(weekly).forEach(points => points.sort((a, b) => a.x - b.x));
    renderInsightCards(data);
  });

  // Two triggers, because neither covers the other. The interval handles a
  // tab left open in the foreground (visibilitychange never fires there);
  // the visibility handler catches the backgrounded-tab case, where browsers
  // throttle timers hard enough that a tab can sit hours behind. New
  // capacity_log rows only land every 15 min, so 5 min is well inside the
  // useful resolution without polling for its own sake.
  setInterval(refreshTodayData, 5 * 60 * 1000);

  // No single event covers every way a tab comes back into view, so listen for
  // all three and let refreshTodayData() de-dupe. visibilitychange misses
  // app/window switches where the tab never technically hides; focus misses
  // same-window tab switches; mobile Safari can restore from bfcache and fire
  // only pageshow. Live capacity and the curve refresh together here so the
  // dot can't jump to a new value while the line it sits on stays behind.
  // init() just loaded all of this, and pageshow fires on first load too, so
  // start the cooldown here to stop the boot pageshow re-fetching immediately.
  lastRefreshAt = Date.now();

  const wake = async () => {
    if (document.visibilityState === 'hidden') return;
    // Settle both before the final render so the live % and the forecast the
    // trend card subtracts it from always describe the same moment.
    await Promise.allSettled([updateStatusBar(), refreshTodayData()]);
    if (window._insightData) renderInsightCards(window._insightData);
  };
  document.addEventListener('visibilitychange', wake);
  window.addEventListener('focus', wake);
  window.addEventListener('pageshow', wake);

  // Background-load the rest of the 90-day horizon (calendar "More" days) after
  // first paint. Merges into the same `predictions` object the rendered view
  // already references, then re-renders so a day the user may have navigated to
  // via the calendar fills in automatically once it arrives.
  sbFetch(`predictions?select=slot_ts,pct&slot_ts=gte.${shiftDate(today, 5)}T00:00:00&slot_ts=lte.${shiftDate(today, 92)}T23:59:59&order=slot_ts.asc&limit=10000`)
    .then(rows => {
      rows.forEach(r => { predictions[slotTsToKey(r.slot_ts)] = r.pct; });
      updatePredChart(data);
      renderInsightCards(data);
    })
    .catch(e => console.error('background predictions load failed:', e));
}

// chart.js is loaded with `defer`, which executes before DOMContentLoaded but
// after this inline script parses. Gating init on DOMContentLoaded guarantees
// the Chart global exists before updatePredChart runs.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
