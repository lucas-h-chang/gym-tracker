function pctColor(p) {
  if (p >= 90) return '#b91c1c';
  if (p >= 80) return '#f87171';
  if (p >= 60) return '#fbbf24';
  return '#6ee7b7';
}

function pctColorRGB(p) {
  if (p >= 90) return '185, 28, 28';
  if (p >= 80) return '248, 113, 113';
  if (p >= 60) return '251, 191, 36';
  return '110, 231, 183';
}

// Custom HTML tooltip renderer, replacing Chart.js's canvas-drawn tooltip.
// Needed so (a) it can sit above the pulse dot via z-index instead of the
// dot always painting over it, and (b) the numeric value can be bolded —
// canvas tooltips render each line in one uniform font, no per-word styling.
function makeExternalTooltip(tooltipId) {
  return (context) => {
    const el = document.getElementById(tooltipId);
    if (!el) return;
    const tt = context.tooltip;

    if (tt.opacity === 0) {
      el.style.opacity = 0;
      return;
    }

    // Every row can be filtered away while the tooltip still counts as active:
    // Chart.js applies the filter when it builds the items but never re-checks
    // whether any survived, so it would happily show a title over an empty box.
    // Reachable via the Tail dataset, whose points are hoverable scaffolding
    // with no row of their own (see the filter, and the dataset's comment).
    const rows = (tt.body || []).reduce((n, b) => n + b.lines.length, 0);
    if (!rows) {
      el.style.opacity = 0;
      return;
    }

    let html = '';
    if (tt.title?.length) {
      html += `<div class="tt-title">${tt.title[0]}</div>`;
    }
    (tt.body || []).forEach(b => {
      b.lines.forEach(line => {
        const bolded = line.replace(/\d+/, (m) => `<b>${m}</b>`);
        html += `<div class="tt-row">${bolded}</div>`;
      });
    });
    el.innerHTML = html;

    // Measure off-screen first so we know which side to place the box on.
    el.style.left = '0px';
    el.style.top  = '0px';
    const w = el.offsetWidth;
    const h = el.offsetHeight;

    const gap = 14;
    const chartArea = context.chart.chartArea;
    // The wrapper, not the plot area, is what the box can actually overflow —
    // it's the positioning context, and anything past its right edge scrolls
    // the whole page sideways on a phone.
    const wrap = el.parentElement;
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;

    let left, top;
    if (wrapW < 480) {
      // Too narrow to put a ~170px nowrap box beside the caret on either side,
      // so sit it above the point and centre it instead.
      left = tt.caretX - w / 2;
      top  = tt.caretY - h - gap;
      // Above the plot area there's only the 36px of layout padding, so if the
      // point is near the top the box has to flip below it.
      if (top < 0) top = tt.caretY + gap;
    } else {
      const preferLeft = tt.caretX - w - gap >= chartArea.left;
      left = preferLeft ? (tt.caretX - w - gap) : (tt.caretX + gap);
      top  = tt.caretY - h / 2;
    }

    // Clamp last, so every branch above is guaranteed to land inside the wrapper.
    el.style.left = Math.max(0, Math.min(left, wrapW - w)) + 'px';
    el.style.top  = Math.max(0, Math.min(top,  wrapH - h)) + 'px';
    el.style.opacity = 1;
  };
}

// ── Prediction Chart ───────────────────────────────────────────
let predChart = null;
let predDate  = '';

function isPhone() { return window.innerWidth <= 768; }

// iOS keeps every 2nd hour on a phone (GymChart: stride(from:through:by: 2))
// and affords it by dropping the space: "7AM", not "7 AM". Same trick here, so
// the web shows the app's tick density instead of thinning to every 3rd hour.
function formatHourTick(h) {
  return isPhone() ? formatHour(h).replace(' ', '') : formatHour(h);
}

// The y ceiling doubles as headroom above the 100% tick, so it is trimmed on
// the phone alongside chartTopPad; see the y scale config for why not 100.
function yCeiling() { return isPhone() ? 105 : 110; }

// Top padding gives the pulse dot's halo room above a near-100% curve. 36px of
// a 340px chart is fine; the same 36px eats an eighth of the phone's box.
//
// The phone value is 12 rather than 22 because the clearance is paid for twice:
// the y domain runs to 110 while ticks stop at 100, so there is already an empty
// band above the curve worth ~29px at the taller phone height. The halo is 22px
// (.pulse-dot 14px + ::before inset -4px) peaking at scale(2.2), so it needs
// ~24px of radius, and 29 + 12 clears that with room to spare.
function chartTopPad() { return isPhone() ? 6 : 36; }

// Re-resolve the breakpoint-dependent options on rotation or a window drag.
// Chart.js re-renders on resize via its own ResizeObserver, but the top padding
// is read once into the options object, and the tick callbacks
// (formatHourTick, the y-axis afterBuildTicks) only re-run on an update — so
// crossing 768 has to force one.
let _wasPhone = isPhone();
window.addEventListener('resize', () => {
  if (document.visibilityState === 'hidden') return;
  if (!predChart) return;
  const pad     = chartTopPad();
  const phoneNow = isPhone();
  if (predChart.options.layout.padding.top === pad && phoneNow === _wasPhone) return;
  _wasPhone = phoneNow;
  predChart.options.layout.padding.top = pad;
  predChart.options.scales.y.max = yCeiling();
  predChart.update('none');
});

function shiftDate(dateStr, delta) {
  const d = new Date(dateStr + 'T12:00:00');
  d.setDate(d.getDate() + delta);
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

function chartTitle(dateStr) {
  const today = todayPT();
  if (dateStr === today) return "Today's Crowd";
  if (dateStr === shiftDate(today, 1)) return "Tomorrow's Crowd";
  const d = new Date(dateStr + 'T12:00:00');
  return `${DAYS[d.getDay()]}'s Crowd`;
}

function buildPredPoints(data, dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  const { open, close } = getOpenHours(d.getDay(), d);
  const points = [];
  for (let h = open; h < close; h++) {
    for (const m of [0, 15, 30, 45]) {
      const key = `${dateStr}_${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
      const pred = data.predictions[key];
      if (pred == null) continue;
      points.push({ x: h + m / 60, y: Math.round(pred * 10) / 10, label: formatTime(h, m) });
    }
  }
  // A closure day has the empty interval [0, 0): no slots, and no closing
  // zero-point either — pushing one would drop a stray marker at midnight,
  // outside the chart's fixed 7-23 axis.
  if (close <= open) return [];
  points.push({ x: close, y: 0, label: formatTime(close, 0) });
  return points;
}

// ── Chart.js Plugins ─────────────────────────────────────────

// Gradient that maps chart color to capacity value (green → yellow → red)
const capacityGradientPlugin = {
  id: 'capacityGradient',
  // Chart.js resolves each line's fill color from dataset.backgroundColor
  // during the *update* phase (before Filler ever draws), not the draw
  // phase — so this must run in beforeDatasetsUpdate, not beforeDatasetsDraw,
  // or the gradient lags one full update cycle behind (invisible on first load).
  beforeDatasetsUpdate(chart) {
    if (!chart.options.plugins?.capacityGradient?.enabled) return;
    const { ctx, scales } = chart;
    if (!scales?.y) return;

    const yBottom = scales.y.getPixelForValue(0);
    const yTop    = scales.y.getPixelForValue(110);

    chart.data.datasets.forEach((ds, i) => {
      const lineOp = ds._predicted ? 0.55 : 1.0;
      const fillOp = ds._predicted ? 0.15 : 0.35;

      const lg = ctx.createLinearGradient(0, yBottom, 0, yTop);
      lg.addColorStop(0,       `rgba(110,231,183,${lineOp})`);
      lg.addColorStop(45/110,  `rgba(110,231,183,${lineOp})`);
      lg.addColorStop(60/110,  `rgba(251,191,36,${lineOp})`);
      lg.addColorStop(80/110,  `rgba(248,113,113,${lineOp})`);
      lg.addColorStop(1,       `rgba(185,28,28,${lineOp})`);
      ds.borderColor = lg;

      const fg = ctx.createLinearGradient(0, yBottom, 0, yTop);
      fg.addColorStop(0,       `rgba(110,231,183,${fillOp * 0.1})`);
      fg.addColorStop(0.08,    `rgba(110,231,183,${fillOp * 0.45})`);
      fg.addColorStop(45/110,  `rgba(110,231,183,${fillOp})`);
      fg.addColorStop(60/110,  `rgba(251,191,36,${fillOp})`);
      fg.addColorStop(80/110,  `rgba(248,113,113,${fillOp})`);
      fg.addColorStop(1,       `rgba(185,28,28,${fillOp})`);
      ds.backgroundColor = fg;
    });
  }
};

// Closed-hours shading. The x-axis is pinned 7 AM to 11 PM on every day so that
// any two days can be read against one scale, which leaves dead space whenever
// the RSF opens late or shuts early (a Saturday closes at 6 PM, a summer weekday
// at 8). Left bare, that stretch reads as "the gym was empty" rather than "the
// gym was shut". Mirrored on iOS by GymChart's closed-hours Canvas pass.
const closedHoursPlugin = {
  id: 'closedHours',
  // Under the datasets, never over them: the curve, its fill and the pulse dot
  // must not be dimmed by the wash.
  beforeDatasetsDraw(chart) {
    const opts = chart.options.plugins?.closedHours;
    if (!opts?.show) return;
    const { ctx, chartArea: area, scales } = chart;
    if (!scales?.x) return;

    const clamp  = (px) => Math.min(Math.max(px, area.left), area.right);
    const openX  = clamp(scales.x.getPixelForValue(opts.open));
    const closeX = clamp(scales.x.getPixelForValue(opts.close));

    // [start, end, boundary]: boundary is the edge that touches open hours, and
    // is skipped when it lands on a plot edge (there is nothing to divide there).
    // A closure day arrives as the empty interval [0, 0), so both pixels clamp to
    // area.left and the second band washes the entire plot, which is right: the
    // building is shut all day, and .chart-closed says so on top of it.
    const bands = [
      [area.left, openX,      openX ],
      [closeX,    area.right, closeX],
    ];

    ctx.save();
    ctx.fillStyle   = 'rgba(255,255,255,0.04)';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth   = 1;
    for (const [x0, x1, edge] of bands) {
      if (x1 - x0 < 0.5) continue;
      ctx.fillRect(x0, area.top, x1 - x0, area.bottom - area.top);
      if (edge > area.left + 0.5 && edge < area.right - 0.5) {
        // Half-pixel offset so the 1px divider lands on one device pixel
        // instead of straddling two and rendering as a 2px smudge.
        const x = Math.round(edge) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
      }
    }
    ctx.restore();
  }
};

// NOW indicator: pulsing dot at the live point (positioned via CSS element).
// The live point itself — same x/y the actual line is extended to and the
// predicted line bridges from in updatePredChart() — is what's rendered
// here, so the dot always sits exactly where the two lines meet.
const nowLinePlugin = {
  id: 'nowLine',
  // afterDraw, not afterDatasetsDraw: this is the one mark that has to sit over
  // everything else on the chart, including hover markers and the ghost tail.
  afterDraw(chart) {
    const opts = chart.options.plugins.nowLine;
    const pulseDot = document.getElementById('pulse-dot');

    if (!opts?.show) {
      if (pulseDot) pulseDot.style.display = 'none';
      return;
    }

    const actualData = chart.data.datasets[0]?.data || [];
    const lastPt = actualData[actualData.length - 1];
    if (!lastPt) {
      if (pulseDot) pulseDot.style.display = 'none';
      return;
    }

    // Read the ANIMATED element, not the data value. Chart.js tweens
    // meta.data[i].x/.y across the 320ms update, so asking the scale for the
    // data's pixel would pin the HALO to its final spot on the first frame
    // while the native core it rings travels in with the line.
    const meta = chart.getDatasetMeta(0);
    const el   = meta && meta.data && meta.data[meta.data.length - 1];
    const dotX = el ? el.x : chart.scales.x.getPixelForValue(lastPt.x);
    const dotY = el ? el.y : chart.scales.y.getPixelForValue(lastPt.y);

    positionPulseDot(chart, lastPt, dotX, dotY);
  }
};

function positionPulseDot(chart, lastPt, dotX, dotY) {
  const pulseDot = document.getElementById('pulse-dot');
  if (!pulseDot) return;

  // The solid core is drawn on the canvas as a native point; this element is
  // only the pulsing halo. Keep it hidden until the chart is actually visible
  // (canvas is visibility:hidden during body.loading) so it never flashes in
  // empty space before the chart's entrance animation runs.
  if (document.body.classList.contains('loading')) {
    pulseDot.style.display = 'none';
    return;
  }

  pulseDot.style.display = 'block';
  pulseDot.style.left = (dotX - 7) + 'px';
  pulseDot.style.top  = (dotY - 7) + 'px';
  pulseDot.style.setProperty('--dot-rgb', pctColorRGB(lastPt.y));
}

// The chart's data only advances when updatePredChart() re-runs (page load,
// day switch), so without this the live point — and the dot sitting on it —
// would freeze at whatever time the page happened to load. Recompute it
// every 15s so the line, its fill, and the dot all keep creeping forward
// together and never drift apart.
setInterval(() => {
  if (!predChart || predDate !== todayPT()) return;
  if (!window._insightData) return;
  updatePredChart(window._insightData);
  // computeTrend() subtracts a forecast slot from the *live* reading, so the
  // moment those two describe different instants the card doesn't just go
  // stale, it inverts: a live 42% against a forecast still holding 31% reads
  // as "Getting quieter" while the gym is filling. Re-render on the same tick
  // as the chart so both halves of that subtraction always agree on "now".
  renderInsightCards(window._insightData);
}, 15000);

// Scrub line on hover
const scrubLinePlugin = {
  id: 'scrubLine',
  afterDraw(chart) {
    const active = chart.tooltip?.getActiveElements();
    if (!active || !active.length) return;
    const { ctx, chartArea: a } = chart;
    const x = active[0].element.x;
    if (x < a.left || x > a.right) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }
};

// ── Day Pills ────────────────────────────────────────────────
function buildPillDates() {
  const today = todayPT();
  const pills = [
    { label: 'Today', date: today },
    { label: 'Tomorrow', date: shiftDate(today, 1) },
  ];
  for (let i = 2; i <= 4; i++) {
    const d = new Date(shiftDate(today, i) + 'T12:00:00');
    pills.push({
      label: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()],
      date: shiftDate(today, i)
    });
  }
  return pills;
}

function renderPills(data) {
  const container = document.getElementById('day-pills');
  const pills = buildPillDates();
  container.innerHTML = '';

  pills.forEach(p => {
    const btn = document.createElement('button');
    btn.className = `pill${p.date === predDate ? ' active' : ''}`;
    btn.textContent = p.label;
    btn.dataset.date = p.date;
    btn.addEventListener('click', () => {
      predDate = p.date;
      updatePredChart(data);
      renderInsightCards(data);
      updatePillStates();
    });
    container.appendChild(btn);
  });

  const calBtn = document.createElement('button');
  calBtn.className = 'pill pill-cal';
  calBtn.textContent = 'More';
  calBtn.id = 'cal-trigger';
  if (!pills.some(p => p.date === predDate)) calBtn.classList.add('active');
  container.appendChild(calBtn);
}

function updatePillStates() {
  const pills = buildPillDates();
  document.querySelectorAll('.pill').forEach(el => {
    if (el.classList.contains('pill-cal')) {
      el.classList.toggle('active', !pills.some(p => p.date === predDate));
    } else {
      el.classList.toggle('active', el.dataset.date === predDate);
    }
  });
}

// ── Prediction Chart ─────────────────────────────────────────
function updatePredChart(data) {
  const today    = todayPT();
  const isToday  = predDate === today;
  const pt       = getPTNow();
  const nowVal   = pt.getHours() + pt.getMinutes() / 60;
  const predDay  = new Date(predDate + 'T12:00:00');
  const xMin     = 7;
  const xMax     = 23;

  // The pulse dot sits on the live point at "now". Past closing (or before
  // open) there's no actual line for it to cap, so it would float alone in
  // empty space — only show it while the gym is actually open.
  const { open: openH, close: closeH } = getOpenHours(predDay.getDay(), predDay);
  const gymIsOpen = pt.getHours() >= openH && pt.getHours() < closeH;
  const showNow   = isToday && gymIsOpen;

  const allPredPoints = buildPredPoints(data, predDate);
  // During an outage the logged readings are a flat line just above zero.
  // Drawing it would render a hardware failure as a genuinely empty gym, so
  // the actual series is dropped and the forecast is drawn across the whole
  // day instead of only from now onward.
  const liveToday = isToday && !sensorIsDown();
  const actualPoints = liveToday
    ? (data.today_actuals || []).filter(p => p.x <= nowVal)
    : [];
  // The forecast line begins at the last COMPLETED 15-min reading rather than
  // at "now", and takes that reading's own value as its first point, so the two
  // series leave a shared origin: from 3:30 the solid line shows what actually
  // happened and the dashed line shows what was predicted, and the space that
  // opens between them IS the forecast error. Starting it at "now" is what
  // produced the cliff, because the only way to reach the 3:45 slot from a 3:44
  // live reading is one near-vertical segment (a 20pp move across ~1px of
  // x-axis, and worse the closer "now" sat to the boundary).
  //
  // Anchoring on the READING rather than on the base curve's own 3:30 value
  // also keeps the dashed line kink-free. today_builder's level correction is
  // computed FROM that reading and only ever covers slots after it (see
  // carry_model.apply_to_day, which returns range(cut_slot + 1, hi)), so a
  // base-curve anchor would put an uncorrected point next to a corrected one
  // and reintroduce the same jump one slot to the left.
  const ghostAnchor = (liveToday && gymIsOpen && actualPoints.length)
    ? actualPoints[actualPoints.length - 1]
    : null;
  let predPoints = liveToday
    ? allPredPoints.filter(p => ghostAnchor ? p.x > ghostAnchor.x : p.x >= nowVal)
    : allPredPoints;

  if (isToday && correctedPoints(data).length) {
    // Blend weight is per-slot ('w'): full trust near the last observation,
    // decaying to the base curve within ~2h (see today_builder.py). Fall back
    // to the scalar blend_weight for older cached rows that lack per-point w.
    const simMap = {};
    correctedPoints(data).forEach(p => { simMap[p.x] = p; });
    predPoints = predPoints.map(p => {
      const sim = simMap[p.x];
      if (sim == null) return p;
      const w = sim.w != null ? sim.w : (data.today_blend_weight || 0);
      return { ...p, y: Math.round(((1 - w) * p.y + w * sim.y) * 10) / 10 };
    });
  }

  // Extend the actual line all the way to "right now" (not just the last
  // completed 15-min bin) using the live capacity reading. This is also
  // what the pulse dot renders on top of, so the line, its fill and the dot
  // all end at one spot instead of the dot floating off the end of it, and the
  // sliver of time between the last bin and now gets shaded as "actual"
  // since it's already passed, instead of being left blank.
  let tailPoints = [];
  if (isToday && actualPoints.length && gymIsOpen) {
    const lastActual = actualPoints[actualPoints.length - 1];
    const liveX = Math.min(Math.max(nowVal, lastActual.x), xMax);
    const live = usableLive();
    const liveY = live ? live.pct : lastActual.y;
    const livePoint = { x: liveX, y: liveY, label: formatTime(pt.getHours(), pt.getMinutes()) };

    if (liveX > lastActual.x) {
      actualPoints.push(livePoint);
    } else {
      actualPoints[actualPoints.length - 1] = livePoint;
    }

    // The forecast dataset starts at "now", carrying the forecast's own value
    // there (interpolated between the anchor reading and the next slot), so the
    // two fills TILE the day: actual owns [open, now], predicted owns
    // [now, close]. Letting the predicted dataset start back at the anchor
    // instead made both of them cover the minutes since the last reading, and
    // 0.15 alpha stacked on 0.35 painted that stretch as a bright vertical band
    // under the live dot.
    //
    // The line still has to cross those minutes, or the dashed curve would
    // begin at "now" and the divergence since the last reading would be
    // invisible. The Tail dataset carries that stretch on its own with fill
    // off, so the eye sees one continuous dashed line from the anchor while
    // only one fill ever covers a given x.
    if (predPoints.length) {
      // Forecast slots that "now" has already passed. Normally none: a reading
      // lands every 15 minutes, so the anchor sits between the last reading and
      // the next slot. But a missed or late scrape leaves whole slots behind
      // "now", and they have to come OUT of the predicted dataset — the anchor
      // is its first point, so a 9:15 slot sitting in front of a 9:30 anchor
      // makes the dataset's x run BACKWARDS. Chart.js draws points in array
      // order, so that folded the dashed line back over itself (two dashed
      // strands over the same minutes) and, worse, handed the tension-0.35
      // bezier a negative x-step, whose control points then swept a straight
      // ramp across the whole plot and dragged the forecast fill with it.
      const passed = predPoints.filter(p => p.x <= liveX);
      const future = predPoints.filter(p => p.x >  liveX);
      // Interpolate along whatever brackets "now": the last passed slot when
      // there is one, otherwise the reading itself, which is the shared origin
      // the comment above describes.
      const prev = passed.length ? passed[passed.length - 1] : lastActual;
      const nxt  = future.length ? future[0] : null;
      const span = nxt ? nxt.x - prev.x : 0;
      const t    = span > 0 ? Math.max(0, Math.min(1, (liveX - prev.x) / span)) : 1;
      const yAt  = nxt ? Math.round((prev.y + (nxt.y - prev.y) * t) * 10) / 10 : prev.y;
      // _anchor marks this as scaffolding, not a forecast slot. Real forecast
      // points sit on 15-min boundaries; this one sits at "now" purely so the
      // predicted fill starts where the actual fill stops. The tooltip filter
      // keys off the flag so it never offers "Predicted: 67%" for an 11:47.
      predPoints = [{ x: liveX, y: yAt, label: livePoint.label, _anchor: true }, ...future];
      const tail = [{ x: lastActual.x, y: lastActual.y }, ...passed];
      if (liveX > tail[tail.length - 1].x) tail.push({ x: liveX, y: yAt });
      // Only worth drawing when it actually spans something. A single point
      // renders as nothing, but it would still put a stray hover target under
      // the live dot.
      if (tail.length >= 2 && tail[tail.length - 1].x > tail[0].x) tailPoints = tail;
    }
  } else if (isToday && actualPoints.length && nowVal >= closeH) {
    // Day is over. There's no live point to extend to, and the predicted
    // (close, 0) point was filtered out by p.x >= nowVal, so the actual line
    // would otherwise end flat at its last non-zero reading. Drop it to 0 at
    // close so today looks like every other finished day.
    const lastActual = actualPoints[actualPoints.length - 1];
    if (lastActual.x < closeH) {
      actualPoints.push({ x: closeH, y: 0, label: formatTime(closeH, 0) });
    } else {
      actualPoints[actualPoints.length - 1] = { x: closeH, y: 0, label: formatTime(closeH, 0) };
    }
  }

  document.getElementById('chart-title').textContent = chartTitle(predDate);
  // "every 15 minutes" is only true of today: other days come from the nightly
  // predictions_builder.py run in daily.yml, so the dot is hidden on them
  // rather than stating a cadence that does not apply.
  document.getElementById('chart-info').hidden = !isToday;

  const closedFor = closureReason(predDay);
  const closedEl  = document.getElementById('chart-closed');
  if (closedFor) {
    closedEl.innerHTML =
      `<span class="cc-title">Closed for ${closedFor}</span>` +
      `<span class="cc-sub">The RSF is shut all day</span>`;
    closedEl.style.display = 'flex';
  } else {
    closedEl.style.display = 'none';
  }

  updatePillStates();

  if (predChart) {
    predChart.data.datasets[0].data = actualPoints;
    predChart.data.datasets[1].data = predPoints;
    predChart.data.datasets[2].data = tailPoints;
    predChart.options.plugins.nowLine.show  = showNow;
    predChart.options.plugins.nowLine.value = nowVal;
    predChart.options.plugins.closedHours.open  = openH;
    predChart.options.plugins.closedHours.close = closeH;
    predChart.options.scales.x.min          = xMin;
    predChart.options.scales.x.max          = xMax;
    predChart.update();
    return;
  }

  const canvasEl = document.getElementById('pred-chart');
  const ctx = canvasEl.getContext('2d');

  // Chart.js's default event list is
  // ['mousemove','mouseout','click','touchstart','touchmove'] — no touchend.
  // Nothing ever tells it a touch finished, so the last touched point stays
  // active and the readout hangs there after you lift your finger. iOS clears
  // it in DragGesture's .onEnded; this is the equivalent.
  //
  // Two things are needed, and an earlier pointerup version had only the
  // second: preventDefault on touchend suppresses the synthetic mousemove
  // Safari fires afterwards at the same coordinates, which would otherwise
  // re-activate the point immediately after it was cleared. mouseout is then
  // Chart.js's own reset path (_handleEvent forces active = [] for it), which
  // clears the tooltip and the scrub line together.
  if (!canvasEl.dataset.liftBound) {
    canvasEl.dataset.liftBound = '1';
    const endTouch = (e) => {
      if (e.cancelable) e.preventDefault();
      canvasEl.dispatchEvent(new MouseEvent('mouseout', { bubbles: true }));
    };
    canvasEl.addEventListener('touchend', endTouch, { passive: false });
    canvasEl.addEventListener('touchcancel', endTouch, { passive: false });
  }

  predChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Actual',
          data: actualPoints,
          borderColor: '#6ee7b7',
          backgroundColor: 'transparent',
          borderWidth: 3.25,
          // The live dot is a native point on the last actual point, so it
          // animates in with the line and stays glued to it. Nothing paints over
          // it, and that is arranged rather than hacked around:
          //   - Actual has the LOWER `order`. Verified against chart.umd 4.4.3:
          //     _sortedMetasets sorts ascending by order, then _drawDatasets
          //     iterates it BACKWARDS, so the lowest order is painted last and
          //     ends up on top. Raising this number sinks the dot under the
          //     dashed line, which is exactly what happened when it was tried.
          //   - drawActiveElementsOnTop is off below, so the white hover marker
          //     on an earlier actual point cannot jump above the live point.
          //   - The Tail dataset takes the HIGHEST order, so it is painted
          //     first and cannot land over the dot either.
          pointRadius: (c) => {
            const show = c.chart.options.plugins?.nowLine?.show;
            return (show && c.dataIndex === c.dataset.data.length - 1) ? 5 : 0;
          },
          pointHoverRadius: 5,
          pointBackgroundColor: (c) => {
            const isLast = c.dataIndex === c.dataset.data.length - 1;
            const y = c.dataset.data[c.dataIndex]?.y;
            return (isLast && y != null) ? `rgb(${pctColorRGB(y)})` : '#fff';
          },
          pointBorderColor: 'transparent',
          pointBorderWidth: 0,
          tension: 0.35,
          fill: true,
          // Lower order = painted last = on top. See the pointRadius comment.
          order: 1,
          // Chart.js collects hovered elements and redraws them in a SECOND pass
          // after every other point in the same dataset (DatasetController.draw,
          // gated on this flag, default true). That is what put the white marker
          // on the last completed reading over the live dot a few pixels away.
          // `order` cannot reach it, because both points live in this dataset.
          // Off, so points paint in index order and the live point, being last,
          // stays on top.
          drawActiveElementsOnTop: false,
        },
        {
          label: 'Predicted',
          data: predPoints,
          borderColor: '#9ca3af',
          backgroundColor: 'transparent',
          borderWidth: 2.6,
          borderDash: [6, 5],
          // Round caps and joins. The dash pattern already matched GymChart's
          // dash: [6, 5], but Chart.js defaults to butt caps and miter joins
          // while iOS strokes with StrokeStyle(lineCap: .round, lineJoin:
          // .round) — which is the whole reason the same numbers read as hard
          // rectangles here and soft lozenges in the app.
          borderCapStyle: 'round',
          borderJoinStyle: 'round',
          pointRadius: 0,
          // The _anchor point shares its x with the live dot but carries the
          // FORECAST's value, so hovering there drew this dataset's white hover
          // marker a few pixels off the yellow live point and read as the two
          // dots failing to line up. It is scaffolding, not a forecast slot
          // (see the tooltip filter, which drops its row for the same reason),
          // so it gets no marker at all. Note the tooltip filter alone is not
          // enough: it decides what the tooltip PRINTS, not which points count
          // as active and draw a hover radius.
          pointHoverRadius: (c) => (c.dataset.data[c.dataIndex]?._anchor ? 0 : 5),
          pointBackgroundColor: '#fff',
          pointBorderColor: 'transparent',
          tension: 0.35,
          fill: true,
          // Higher than Actual, so this is painted first and sits underneath it.
          order: 2,
          _predicted: true,
        },
        {
          // The minutes between the last completed reading and "now". This is
          // the forecast's line continuing across a stretch the Predicted
          // dataset deliberately no longer spans, so that its fill cannot stack
          // on top of the actual line's fill there (see the tailPoints block in
          // updatePredChart). Same dashes, same gradient, no fill of its own:
          // those minutes already happened, so the shading under them is the
          // actual series' to own.
          //
          // A dataset rather than a hand-stroked plugin, because a plugin
          // painting through getPixelForValue has no idea the 320ms entrance
          // tween is running. It snapped to its final position on the first
          // frame while both real lines were still rising off the baseline,
          // which read as a dashed stub floating over an empty chart.
          label: 'Tail',
          data: tailPoints,
          borderColor: '#9ca3af',
          backgroundColor: 'transparent',
          borderWidth: 2.6,
          borderDash: [6, 5],
          borderCapStyle: 'round',
          borderJoinStyle: 'round',
          // Scaffolding, not readable data: it re-plots forecast slots the
          // Predicted dataset already gave up, so a hover here would offer a
          // second row for a slot that is already in the tooltip. No markers,
          // and the tooltip filter drops it outright.
          pointRadius: 0,
          pointHoverRadius: 0,
          // Matches Predicted, so the bridge curves into the dashed line
          // instead of kinking where the two meet.
          tension: 0.35,
          fill: false,
          // Highest order = painted first = furthest back. That is where the
          // old plugin already put it: it stroked after the Predicted dataset
          // but before Actual, so the green fill has always washed over these
          // dashes. Keeping it last preserves that exactly.
          order: 3,
          // Picks up the forecast's 0.55-opacity gradient from
          // capacityGradientPlugin, which walks every dataset.
          _predicted: true,
        }
      ]
    },
    options: {
      parsing: false,
      // The .chart-wrap height is the single source of truth for how tall the
      // chart is (see its CSS comment). Without this Chart.js would re-derive a
      // height from an aspect ratio and fight the wrapper.
      maintainAspectRatio: false,
      animation: { duration: 320 },
      layout: { padding: { top: chartTopPad() } },
      interaction: { mode: 'nearest', intersect: false, axis: 'x' },
      plugins: {
        capacityGradient: { enabled: true },
        closedHours: { show: true, open: openH, close: closeH },
        nowLine: { show: showNow, value: nowVal },
        legend:  { display: false },
        tooltip: {
          enabled: false,
          external: makeExternalTooltip('pred-tooltip'),
          displayColors: false,
          // Decoupled from `order`, which now runs the other way for z-order.
          itemSort: (a, b) => a.datasetIndex - b.datasetIndex,
          // Drop the synthetic point that starts the predicted dataset at "now"
          // (see _anchor in updatePredChart). It exists so the two fills tile
          // rather than overlap, and it is not a forecast for anything: real
          // forecast slots land on 15-min boundaries, so surfacing it offered a
          // "Predicted: 67%" for an 11:47 that the model never produced.
          //
          // Flagged rather than identified by coordinates. The previous version
          // matched x AND y against the last actual reading, which held only
          // while the connector carried the live VALUE; the moment it started
          // carrying the forecast's value instead, the y stopped matching and
          // the row came back.
          filter: (item) => item.datasetIndex !== 2 &&
                            !(item.datasetIndex === 1 && item.raw && item.raw._anchor),
          callbacks: {
            title: (items) => items[0].raw.label || '',
            label: (item) => `${item.dataset.label}: ${Math.round(item.raw.y)}%`,
          }
        }
      },
      scales: {
        x: {
          type: 'linear', min: xMin, max: xMax,
          grid: { display: false },
          border: { display: false },
          ticks: {
            // Scriptable so a tick sitting in a closed band reads as dimmed as
            // the band itself. The opening and closing hours stay at full
            // strength: they label the boundary, not the dead time.
            color: (c) => {
              const o = c.chart.options.plugins?.closedHours;
              const v = c.tick?.value;
              return (o?.show && v != null && (v < o.open || v > o.close))
                ? '#3a3733' : '#5a5550';
            },
            font: { size: 11, family: "'Inter', sans-serif", weight: '500' },
            stepSize: 2,
            padding: 8,
            callback: (v) => Number.isInteger(v) ? formatHourTick(v) : ''
          }
        },
        y: {
          min: 0,
          // 110 on desktop, but on the phone that spare 10pp is ~29px of empty
          // plot sitting between the day pills and the 100% gridline, on top of
          // the header margin and chartTopPad. Three separate clearances stacked
          // into one visible void. 105 keeps a band above the tick without the
          // void, and still leaves room over 100% rather than clipping at it:
          // capacity genuinely exceeds 100 (livePct is count/150, and
          // apply_to_day clamps at 110), so a hard 100 ceiling would flatten a
          // genuinely overfull gym against the top of the chart.
          max: yCeiling(),
          grid: { color: 'rgba(255,255,255,0.04)', drawTicks: false },
          border: { display: false },
          // The 110 ceiling is headroom for the curve, not a value worth
          // labelling, but Chart.js includes the bound by default and prints
          // "110%" jammed against "100%". iOS lists its ticks explicitly
          // (AxisMarks(values: [25, 50, 75, 100])); match that on the phone,
          // where the collision is worst. Desktop is left as-is deliberately.
          afterBuildTicks: (axis) => {
            if (isPhone()) {
              axis.ticks = [25, 50, 75, 100].map(value => ({ value }));
            }
          },
          ticks: {
            color: '#5a5550',
            font: { size: 10, family: "'Inter', sans-serif", weight: '500' },
            stepSize: 25,
            padding: 8,
            callback: (v) => v > 0 ? `${v}%` : ''
          }
        }
      }
    }
  });
}
