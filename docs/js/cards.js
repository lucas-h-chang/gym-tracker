// ── Data Warning ──────────────────────────────────────────────
function showDataWarning(msg) {
  const el = document.getElementById('data-warning');
  el.textContent = msg;
  el.style.display = 'block';
}

const SENSOR_WARNING =
  "RSF's occupancy sensor looks stuck \u2014 the live count and the usual-day " +
  "comparison are paused until it recovers. The forecast is unaffected.";

function sensorIsDown() {
  return !!(liveCapacity && liveCapacity.sensorOk === false);
}

// Separate from showDataWarning's one-shot load errors because this one has to
// clear itself when the hardware comes back — and must not wipe a load-failure
// message that's already on screen, hence the ownership check before hiding.
function setSensorWarning(down) {
  const el = document.getElementById('data-warning');
  if (down) {
    el.textContent = SENSOR_WARNING;
    el.style.display = 'block';
  } else if (el.textContent === SENSOR_WARNING) {
    el.textContent = '';
    el.style.display = 'none';
  }
}

// ── Insight Cards ─────────────────────────────────────────────
// Half-open [fromH, toH) by default, which is what the full-day curve wants:
// called with the closing hour, it must stop at the last slot INSIDE opening
// hours (10:45 PM), not draw a point at the moment the doors shut.
//
// `includeEnd` closes the interval to [fromH, toH], for the "next 30 min" card.
// That window is 30 minutes because the card says so, and excluding its far edge
// meant that on an exact quarter-hour — 4:00, 4:15, … — the last slot considered
// was only 15 minutes out while the label still promised 30.
function getBlendedSlots(data, fromH, toH, date, includeEnd = false) {
  const dateStr = date || todayPT();
  const isToday = dateStr === todayPT();
  const simMap = {};
  if (isToday && correctedPoints(data).length) {
    correctedPoints(data).forEach(p => { simMap[p.x] = p; });
  }
  const slots = [];
  for (let h = Math.floor(fromH); h <= Math.floor(toH); h++) {
    for (const m of [0, 15, 30, 45]) {
      const slotH = h + m / 60;
      if (slotH < fromH || (includeEnd ? slotH > toH : slotH >= toH)) continue;
      const key  = `${dateStr}_${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`;
      const pred = data.predictions[key];
      if (pred == null) continue;
      const sim     = simMap[slotH];
      const w       = sim ? (sim.w != null ? sim.w : (data.today_blend_weight || 0)) : 0;
      const blended = sim ? (1 - w) * pred + w * sim.y : pred;
      slots.push({ h, m, slotH, y: blended });
    }
  }
  return slots;
}

function computeTrend(data, livePct) {
  const isToday = predDate === todayPT();
  const d = new Date(predDate + 'T12:00:00');
  const { open, close } = getOpenHours(d.getDay(), d);

  let fromH, toH, sub;
  if (isToday) {
    const pt  = getPTNow();
    const nowH = pt.getHours() + pt.getMinutes() / 60;
    if (close - nowH < 0.5) return null;
    fromH = nowH;
    toH   = Math.min(nowH + 0.5, close);
    sub   = null;
  } else {
    fromH = open;
    toH   = Math.min(open + 0.5, close);
    sub   = 'first 30 min · projected';
  }

  const slots = getBlendedSlots(data, fromH, toH, predDate, true);
  if (slots.length < 2) return null;

  const startY = (isToday && livePct != null) ? livePct : slots[0].y;
  const endY   = slots[slots.length - 1].y;
  const delta  = endY - startY;

  const dir = Math.abs(delta) < 5 ? 'flat' : delta > 0 ? 'up' : 'down';
  const lastSlot = slots[slots.length - 1];
  return { dir, delta: Math.round(delta), startY: Math.round(startY), endY: Math.round(endY), toH: lastSlot.h, toMin: lastSlot.m, sub };
}

function computeDayComparisonAt(dayName, qH, data, forDate) {
  function getWeeklyAt(key, h) {
    const arr = data.weekly[key];
    if (!arr) return null;
    const pt = arr.find(p => p.x === h);
    return pt ? pt.y : null;
  }
  // Compare against the same KIND of day: a summer Thursday vs usual summer
  // Thursdays, an in-session one vs usual semester Thursdays, a break day vs
  // usual break days. (The old "This semester only" slice went blank every
  // summer, since there are no in-session days to average.)
  const period    = periodTypeOf(forDate);
  const rangeType = PERIOD_RANGE_TYPE[period];
  const disp      = RANGE_DB_TO_DISPLAY[rangeType] || rangeType;
  return {
    period,
    baseline: getWeeklyAt(`${dayName}|${disp}|false`, qH),
  };
}

function renderInsightCards(data) {
  const row = document.getElementById('insight-row');
  row.classList.remove('hidden');

  const isToday   = predDate === todayPT();
  const pt        = getPTNow();
  const predDay   = new Date(predDate + 'T12:00:00');
  const dayName   = DAYS[predDay.getDay()];
  // A stalled counter is routed into the "no live reading" path rather than
  // given a new UI state of its own: every card already has a designed
  // fallback for that (— for capacity, forecast-only trend, — for the
  // comparison), and reusing them is what stops the page asserting 1% as fact
  // and then stacking "Filling up" and "Much quieter than usual Sundays" on
  // top of it. See api/_sensor.js.
  const sensorDown = sensorIsDown();
  setSensorWarning(sensorDown);
  const live = usableLive();
  const livePct = live?.pct ?? null;
  const liveCount = live?.count ?? null;

  const { open: openH, close: closeH } = getOpenHours(predDay.getDay(), predDay);
  // Caltopia shuts the whole building, so the day has no forecast to show and
  // no live count to take — the card names the reason instead of showing a
  // bare dash the reader has to interpret.
  const closure = closureReason(predDay);

  // ── Card 1: Status / Capacity ──
  const pctEl       = document.getElementById('insight-pct');
  const pctTagEl    = document.getElementById('insight-pct-tag');
  const capFill     = document.getElementById('insight-cap-fill');
  const capLabel    = document.getElementById('insight-cap-label');
  const liveDotEl   = document.getElementById('insight-live-dot');
  pctTagEl.textContent = '';

  // Only rendered on the phone layout (see .live-dot). display, not opacity:
  // iOS reserves the dot's slot with .opacity(0), but on the web that left
  // "CLOSED" indented 16px past the LIVE CAPACITY label above it and the
  // "Opens tomorrow at" line below it, which read as a misalignment bug.
  // Removing it from the flow keeps every closed/error state flush left.
  // A class, not an inline display: an inline style would beat the base
  // `.live-dot { display: none }` and leak the dot onto desktop, which never
  // shows it.
  const setLiveDot = (color) => {
    liveDotEl.style.background = color || 'transparent';
    liveDotEl.classList.toggle('on', !!color);
  };
  setLiveDot(null);

  if (isToday) {
    const gymIsOpen = pt.getHours() >= openH && pt.getHours() < closeH;
    if (gymIsOpen && livePct != null) {
      pctEl.textContent = `${livePct.toFixed(0)}%`;
      pctEl.style.color = pctColor(livePct);
      setLiveDot(pctColor(livePct));

      const fillCls = livePct >= 90 ? 'critical' : livePct >= 80 ? 'high' : livePct >= 60 ? 'mid' : '';
      capFill.style.width = `${Math.min(livePct, 100)}%`;
      capFill.className = `cap-bar-fill${fillCls ? ' ' + fillCls : ''}`;

      if (livePct >= 95) {
        capLabel.innerHTML = `${liveCount} / 150 people · <a href="https://417804.waitwell.us/join/48" target="_blank" rel="noopener">Join virtual queue</a>`;
      } else {
        capLabel.textContent = `${liveCount} / 150 people`;
      }
    } else if (gymIsOpen) {
      pctEl.textContent = '—';
      pctEl.style.color = 'var(--sub)';
      capFill.style.width = '0%';
      capLabel.textContent = sensorDown ? 'RSF sensor offline' : liveCapacity ? 'Live reading is stale' : 'Live count unavailable';
    } else {
      pctEl.textContent = 'CLOSED';
      pctEl.style.color = '#f87171';
      pctEl.style.fontSize = '';
      capFill.style.width = '0%';

      const nowH = pt.getHours() + pt.getMinutes() / 60;
      const isBeforeOpen = closeH > openH && nowH < openH;
      if (closure) {
        capLabel.textContent = `Closed for ${closure}`;
      } else if (isBeforeOpen) {
        capLabel.textContent = `Opens at ${formatTime(openH, 0)}`;
      } else {
        // Searched, not assumed: on the Saturday night before a Caltopia
        // closure the gym does NOT open tomorrow, and the previous
        // "Opens tomorrow at 8:00 AM" stated that as fact.
        const next = nextOpenDate(pt);
        capLabel.textContent = next
          ? `Opens ${next.isTomorrow ? 'tomorrow' : DAYS[next.date.getDay()]} at ${formatTime(next.open, 0)}`
          : 'Closed';
      }
    }
  } else {
    const allSlots = getBlendedSlots(data, openH, closeH, predDate);
    if (closure) {
      pctEl.textContent = 'CLOSED';
      pctEl.style.color = '#f87171';
      pctEl.style.fontSize = '';
      capFill.style.width = '0%';
      capLabel.textContent = `Closed for ${closure}`;
    } else if (allSlots.length) {
      const peak = allSlots.reduce((a, b) => b.y > a.y ? b : a);
      pctEl.textContent = `${Math.round(peak.y)}%`;
      pctEl.style.color = pctColor(peak.y);
      pctTagEl.textContent = 'PEAK';
      const fillCls = peak.y >= 90 ? 'critical' : peak.y >= 80 ? 'high' : peak.y >= 60 ? 'mid' : '';
      capFill.style.width = `${Math.min(peak.y, 100)}%`;
      capFill.className = `cap-bar-fill${fillCls ? ' ' + fillCls : ''}`;
      capLabel.textContent = `${formatTime(openH, 0)} – ${formatTime(closeH, 0)}`;
    } else {
      pctEl.textContent = '—';
      pctEl.style.color = 'var(--sub)';
      capFill.style.width = '0%';
      capLabel.textContent = '—';
    }
  }

  // ── Card 2: Trend (Next 30 min) ──
  const trendValEl = document.getElementById('insight-trend-val');
  const trendSubEl = document.getElementById('insight-trend-sub');
  const trend = computeTrend(data, livePct);

  if (trend && isToday) {
    const labels  = { up: 'Filling up', down: 'Clearing out', flat: 'Holding steady' };
    const dirColor = trend.dir === 'up' ? '#f87171' : trend.dir === 'down' ? '#6ee7b7' : 'var(--text)';
    trendValEl.textContent = labels[trend.dir];
    trendValEl.style.color = dirColor;

    // Magnitude only, since the arrow already carries the direction. Naming the
    // window instead of its end time is safe here: computeTrend returns null
    // when fewer than 30 minutes remain before close, so this can never claim a
    // 30-minute window that actually got clipped at closing time.
    // NOTE: no longer identical to RSFApp2.0's TodayView trendCard, which still
    // renders "N% by <time>". The card label above it is gone on web only too.
    const arrows = { up: '↑', down: '↓', flat: '→' };
    const subColor = trend.dir === 'up' ? '#f87171' : trend.dir === 'down' ? '#6ee7b7' : 'var(--sub)';
    trendSubEl.innerHTML = `<span style="color:${subColor}">${arrows[trend.dir]} ${Math.abs(trend.delta)}% in next 30 mins</span>`;
  } else {
    trendValEl.textContent = '—';
    trendValEl.style.color = 'var(--sub)';
    trendSubEl.textContent = '';
  }

  // ── Card 3: Comparison (vs usual day of this same period type) ──
  const compVerdictEl = document.getElementById('insight-comp-verdict');
  const compDeltaEl   = document.getElementById('insight-comp-delta');

  const cmpH = isToday
    ? Math.round((pt.getHours() + pt.getMinutes() / 60) * 4) / 4
    : 12;
  const cmp = computeDayComparisonAt(dayName, cmpH, data, new Date(predDate + 'T12:00:00'));

  // The delta line now names the baseline the comparison actually used (summer /
  // semester / break Tuesdays) instead of a card header doing it, so a reader
  // never sees a bare "5% below" without knowing below WHAT. Built AFTER cmp so
  // the phrase and the number can never describe different periods.
  const periodWord = PERIOD_LABEL[cmp.period] ? PERIOD_LABEL[cmp.period].toLowerCase() : null;
  const vsPhrase = periodWord ? `${dayName}s this ${periodWord}` : `usual ${dayName}s`;
  const refPct = isToday ? livePct : null;
  const baseline = cmp.baseline;
  const avgDelta = (baseline != null && refPct != null) ? Math.round(refPct - baseline) : null;

  if (avgDelta != null) {
    const absDelta = Math.abs(avgDelta);
    const higher = avgDelta > 0;

    // Sentence case, kept identical to RSFApp2.0's computeComparison verdicts.
    let verdict, verdictClass;
    if (absDelta > 15) {
      verdict = higher ? 'Much busier' : 'Much quieter';
      verdictClass = higher ? 'much-busier' : 'much-quieter';
    } else if (absDelta > 6) {
      verdict = higher ? 'Busier' : 'Quieter';
      verdictClass = higher ? 'busier' : 'quieter';
    } else if (absDelta > 3) {
      verdict = higher ? 'A bit busier' : 'A bit quieter';
      verdictClass = higher ? 'bit-busier' : 'bit-quieter';
    } else {
      verdict = 'About average';
      verdictClass = 'average';
    }

    // At exactly 0 the above/below split has nothing to describe, and
    // "0% below" reads as a rendering bug rather than a dead-on match.
    // NOTE: no longer identical to RSFApp2.0's computeComparison deltaText.
    const deltaText = absDelta === 0
      ? `Right on ${vsPhrase}`
      : `${absDelta}% ${higher ? 'above' : 'below'} ${vsPhrase}`;

    compVerdictEl.textContent = verdict;
    compVerdictEl.className = `card-value comp-verdict ${verdictClass}`;
    compVerdictEl.style.color = '';
    compDeltaEl.innerHTML = `<span class="comp-verdict ${verdictClass}">${deltaText}</span>`;
  } else {
    compVerdictEl.textContent = '—';
    compVerdictEl.className = 'card-value comp-verdict';
    compVerdictEl.style.color = 'var(--sub)';
    compDeltaEl.textContent = '';
  }
}
