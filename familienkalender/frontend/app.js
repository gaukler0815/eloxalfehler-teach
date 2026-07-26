/* =========================================================================
   Familienkalender – Frontend (PWA)
   ========================================================================= */
"use strict";

const State = {
  token: localStorage.getItem("fk_token") || null,
  user: null,
  view: localStorage.getItem("fk_view") || "month",
  anchor: new Date(),           // Bezugsdatum der aktuellen Ansicht
  persons: [],
  occurrences: [],
  feiertage: [],
  ferien: [],
  vapidKey: null,
};

// Bundesländer für die Einstellungen (Code -> Name)
const STATES_DE = [
  ["BW", "Baden-Württemberg"], ["BY", "Bayern"], ["BE", "Berlin"],
  ["BB", "Brandenburg"], ["HB", "Bremen"], ["HH", "Hamburg"], ["HE", "Hessen"],
  ["MV", "Mecklenburg-Vorpommern"], ["NI", "Niedersachsen"],
  ["NW", "Nordrhein-Westfalen"], ["RP", "Rheinland-Pfalz"], ["SL", "Saarland"],
  ["SN", "Sachsen"], ["ST", "Sachsen-Anhalt"], ["SH", "Schleswig-Holstein"],
  ["TH", "Thüringen"],
];

const WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"];
const MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
  "August", "September", "Oktober", "November", "Dezember"];
const COLORS = ["#4f7cff", "#e05a5a", "#3bb273", "#f0a202", "#9b5de5",
  "#ff7b54", "#00b4d8", "#e83e8c"];
const REMINDER_OPTIONS = [
  [0, "Zum Zeitpunkt"], [5, "5 Minuten vorher"], [10, "10 Minuten vorher"],
  [15, "15 Minuten vorher"], [30, "30 Minuten vorher"], [60, "1 Stunde vorher"],
  [120, "2 Stunden vorher"], [180, "3 Stunden vorher"], [360, "6 Stunden vorher"],
  [720, "12 Stunden vorher"], [1440, "1 Tag vorher"], [2880, "2 Tage vorher"],
  [4320, "3 Tage vorher"], [10080, "1 Woche vorher"], [20160, "2 Wochen vorher"],
];

/* ------------------------- Hilfsfunktionen ---------------------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const pad = (n) => String(n).padStart(2, "0");
const dateKey = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const toLocalInput = (d) => `${dateKey(d)}T${pad(d.getHours())}:${pad(d.getMinutes())}`;

function parseLocal(str) {
  if (!str) return null;
  const [datePart, timePart] = str.split("T");
  const [y, m, d] = datePart.split("-").map(Number);
  if (!timePart) return new Date(y, m - 1, d);
  const [hh, mm] = timePart.split(":").map(Number);
  return new Date(y, m - 1, d, hh || 0, mm || 0);
}

function startOfWeek(d) {
  const x = new Date(d);
  const day = (x.getDay() + 6) % 7; // Montag = 0
  x.setDate(x.getDate() - day);
  x.setHours(0, 0, 0, 0);
  return x;
}
function addDays(d, n) { const x = new Date(d); x.setDate(x.getDate() + n); return x; }
function sameDay(a, b) { return dateKey(a) === dateKey(b); }

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

/* ------------------------- API ---------------------------------------- */
async function api(path, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}) } };
  if (State.token) opts.headers["Authorization"] = "Bearer " + State.token;
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch("/api" + path, opts);
  if (res.status === 401) { logout(); throw new Error("Nicht angemeldet"); }
  if (!res.ok) {
    let detail = "Fehler";
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res;
}

/* ------------------------- Auth --------------------------------------- */
function showAuth() {
  $("#app").classList.add("hidden");
  $("#auth-screen").classList.remove("hidden");
}
function showApp() {
  $("#auth-screen").classList.add("hidden");
  $("#app").classList.remove("hidden");
}
function logout() {
  State.token = null; State.user = null;
  localStorage.removeItem("fk_token");
  showAuth();
}

$$("[data-authtab]").forEach((btn) => btn.addEventListener("click", () => {
  $$("[data-authtab]").forEach((b) => b.classList.toggle("active", b === btn));
  const tab = btn.dataset.authtab;
  $("#login-form").classList.toggle("hidden", tab !== "login");
  $("#register-form").classList.toggle("hidden", tab !== "register");
  $("#auth-error").textContent = "";
}));

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    const data = await api("/auth/login", {
      method: "POST",
      body: { email: fd.get("email"), password: fd.get("password") },
    });
    onLoggedIn(data);
  } catch (err) { $("#auth-error").textContent = err.message; }
});

$("#register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    const data = await api("/auth/register", {
      method: "POST",
      body: { name: fd.get("name"), email: fd.get("email"),
        password: fd.get("password"), family_code: fd.get("family_code") || "" },
    });
    onLoggedIn(data);
  } catch (err) { $("#auth-error").textContent = err.message; }
});

function onLoggedIn(data) {
  State.token = data.token;
  State.user = data.user;
  localStorage.setItem("fk_token", data.token);
  showApp();
  init();
}

/* ------------------------- Datenladen --------------------------------- */
function viewWindow() {
  let start, end;
  if (State.view === "month") {
    const first = new Date(State.anchor.getFullYear(), State.anchor.getMonth(), 1);
    start = startOfWeek(first);
    end = addDays(start, 41);
  } else if (State.view === "week") {
    start = startOfWeek(State.anchor);
    end = addDays(start, 6);
  } else if (State.view === "day") {
    start = new Date(State.anchor); start.setHours(0, 0, 0, 0);
    end = new Date(start);
  } else { // agenda
    start = new Date(State.anchor); start.setHours(0, 0, 0, 0);
    end = addDays(start, 60);
  }
  end.setHours(23, 59, 0, 0);
  return { start, end };
}

async function loadData() {
  const { start, end } = viewWindow();
  const q = `?start=${toLocalInput(start)}&end=${toLocalInput(end)}`;
  const [data, persons] = await Promise.all([
    api("/occurrences" + q),
    api("/persons"),
  ]);
  State.occurrences = data.events || [];
  State.feiertage = data.feiertage || [];
  State.ferien = data.ferien || [];
  State.persons = persons;
}

function personById(id) { return State.persons.find((p) => p.id === id); }

/* Feiertage nach Datum + Schulferien-Tage (Datum -> Name) aufbereiten */
function holidayMaps() {
  const fMap = {};
  (State.feiertage || []).forEach((h) => {
    (fMap[h.date] = fMap[h.date] || []).push(h.name);
  });
  const ferienDays = {};
  (State.ferien || []).forEach((p) => {
    let d = parseLocal(p.start + "T00:00");
    const end = parseLocal(p.end + "T00:00");
    let guard = 0;
    while (d <= end && guard < 400) {
      ferienDays[dateKey(d)] = p.name;
      d = addDays(d, 1); guard += 1;
    }
  });
  return { fMap, ferienDays };
}

/* ------------------------- Rendering ---------------------------------- */
function setPeriodLabel() {
  const a = State.anchor;
  if (State.view === "month") {
    $("#period-label").textContent = `${MONTHS[a.getMonth()]} ${a.getFullYear()}`;
  } else if (State.view === "week") {
    const s = startOfWeek(a), e = addDays(s, 6);
    $("#period-label").textContent = `${s.getDate()}.–${e.getDate()}. ${MONTHS[e.getMonth()]}`;
  } else if (State.view === "day") {
    $("#period-label").textContent =
      `${WEEKDAYS[(a.getDay() + 6) % 7]}, ${a.getDate()}. ${MONTHS[a.getMonth()]}`;
  } else {
    $("#period-label").textContent = "Kommende Termine";
  }
}

function occByDay() {
  const map = {};
  for (const o of State.occurrences) {
    const key = o.start.split("T")[0];
    (map[key] = map[key] || []).push(o);
  }
  for (const k in map) map[k].sort((a, b) => a.start.localeCompare(b.start));
  return map;
}

async function render() {
  setPeriodLabel();
  $$(".view-tab").forEach((t) => t.classList.toggle("active", t.dataset.view === State.view));
  const el = $("#calendar");
  el.innerHTML = '<div class="empty">Lädt…</div>';
  try { await loadData(); } catch (e) { el.innerHTML = `<div class="empty">${e.message}</div>`; return; }
  if (State.view === "month") renderMonth(el);
  else if (State.view === "week") renderWeek(el);
  else if (State.view === "day") renderDay(el);
  else renderAgenda(el);
}

function renderMonth(el) {
  const byDay = occByDay();
  const { fMap, ferienDays } = holidayMaps();
  const first = new Date(State.anchor.getFullYear(), State.anchor.getMonth(), 1);
  const gridStart = startOfWeek(first);
  const today = new Date();
  let html = '<div class="month-grid"><div class="weekday-row">' +
    WEEKDAYS.map((w) => `<div>${w}</div>`).join("") + '</div><div class="days-grid">';
  for (let i = 0; i < 42; i++) {
    const d = addDays(gridStart, i);
    const key = dateKey(d);
    const items = byDay[key] || [];
    const other = d.getMonth() !== State.anchor.getMonth();
    const isToday = sameDay(d, today);
    const ferienName = ferienDays[key];
    const prevFerien = ferienDays[dateKey(addDays(d, -1))];
    const cls = ["day-cell", other ? "other-month" : "", isToday ? "today" : "",
                 ferienName ? "ferien-day" : ""].filter(Boolean).join(" ");
    html += `<div class="${cls}" data-date="${key}">
      <span class="day-num">${d.getDate()}</span>`;
    if (ferienName && prevFerien !== ferienName) {
      html += `<div class="ferien-label" title="${escapeHtml(ferienName)}">${escapeHtml(ferienName)}</div>`;
    }
    html += '<div class="day-events">';
    (fMap[key] || []).forEach((n) => {
      html += `<div class="evt-pill hol" title="${escapeHtml(n)}">${escapeHtml(n)}</div>`;
    });
    items.slice(0, 3).forEach((o) => {
      const time = o.all_day ? "" : parseLocal(o.start).getHours() + ":" + pad(parseLocal(o.start).getMinutes()) + " ";
      html += `<div class="evt-pill" style="background:${o.color}" data-event="${o.event_id}">${time}${escapeHtml(o.title)}</div>`;
    });
    if (items.length > 3) html += `<div class="evt-more">+${items.length - 3} mehr</div>`;
    html += "</div></div>";
  }
  html += "</div></div>";
  el.innerHTML = html;
  $$(".day-cell", el).forEach((c) => c.addEventListener("click", (e) => {
    if (e.target.dataset.event) { openEventDetail(+e.target.dataset.event); return; }
    const d = parseLocal(c.dataset.date + "T09:00");
    openEventForm(null, d);
  }));
  $$(".evt-pill", el).forEach((p) => p.addEventListener("click", (e) => {
    if (!p.dataset.event) return;   // Feiertag-Pille -> nicht anklickbar
    e.stopPropagation(); openEventDetail(+p.dataset.event);
  }));
}

function renderWeek(el) {
  const start = startOfWeek(State.anchor);
  const days = Array.from({ length: 7 }, (_, i) => addDays(start, i));
  renderTimeGrid(el, days);
}
function renderDay(el) {
  const d = new Date(State.anchor); d.setHours(0, 0, 0, 0);
  renderTimeGrid(el, [d]);
}

function renderTimeGrid(el, days) {
  const byDay = occByDay();
  const today = new Date();
  const multi = days.length > 1;
  // Ganztägige Termine + Feiertage/Ferien als Band oben
  const { fMap, ferienDays } = holidayMaps();
  let bandInner = "";
  const ferienShown = new Set();
  days.forEach((d) => {
    const key = dateKey(d);
    (fMap[key] || []).forEach((n) => {
      bandInner += `<span class="evt-pill hol">${escapeHtml(n)}</span>`;
    });
    const fn = ferienDays[key];
    if (fn && !ferienShown.has(fn)) {
      ferienShown.add(fn);
      bandInner += `<span class="evt-pill ferien">${escapeHtml(fn)}</span>`;
    }
  });
  days.forEach((d) => (byDay[dateKey(d)] || []).forEach((o) => {
    if (o.all_day) bandInner +=
      `<span class="evt-pill" style="background:${o.color}" data-event="${o.event_id}">${escapeHtml(o.title)}</span>`;
  }));
  const allday = bandInner ? `<div class="allday-band">${bandInner}</div>` : "";
  let head = "";
  if (multi) {
    head = '<div class="tg-week-head"><div></div>' + days.map((d) =>
      `<div class="${sameDay(d, today) ? "wd-today" : ""}">${WEEKDAYS[(d.getDay() + 6) % 7]}<br>${d.getDate()}</div>`).join("") + "</div>";
  }
  let rows = "";
  for (let h = 0; h < 24; h++) {
    if (multi) {
      rows += `<div class="tg-week"><div class="tg-hour">${pad(h)}:00</div>`;
      days.forEach((d) => {
        const evs = (byDay[dateKey(d)] || []).filter((o) => !o.all_day && parseLocal(o.start).getHours() === h);
        rows += `<div class="tg-slot" data-date="${dateKey(d)}" data-hour="${h}">` +
          evs.map((o) => tgEvent(o)).join("") + "</div>";
      });
      rows += "</div>";
    } else {
      const d = days[0];
      const evs = (byDay[dateKey(d)] || []).filter((o) => !o.all_day && parseLocal(o.start).getHours() === h);
      rows += `<div class="tg-row"><div class="tg-hour">${pad(h)}:00</div>
        <div class="tg-slot" data-date="${dateKey(d)}" data-hour="${h}">${evs.map((o) => tgEvent(o)).join("")}</div></div>`;
    }
  }
  el.innerHTML = allday + `<div class="timegrid">${head}${rows}</div>`;
  $$(".tg-event", el).forEach((p) => p.addEventListener("click", (e) => {
    e.stopPropagation(); openEventDetail(+p.dataset.event);
  }));
  $$(".evt-pill", el).forEach((p) => p.addEventListener("click", (e) => {
    if (!p.dataset.event) return;   // Feiertag/Ferien -> nicht anklickbar
    e.stopPropagation(); openEventDetail(+p.dataset.event);
  }));
  $$(".tg-slot", el).forEach((s) => s.addEventListener("click", () => {
    const d = parseLocal(s.dataset.date + "T" + pad(+s.dataset.hour) + ":00");
    openEventForm(null, d);
  }));
}
function tgEvent(o) {
  const t = parseLocal(o.start);
  return `<div class="tg-event" style="background:${o.color}" data-event="${o.event_id}">${pad(t.getHours())}:${pad(t.getMinutes())} ${escapeHtml(o.title)}</div>`;
}

function renderAgenda(el) {
  const byDay = occByDay();
  const { fMap } = holidayMaps();
  // Feiertage nach Tag, Ferien als Zeiträume (am Starttag einsortiert)
  const feByDay = {};
  Object.keys(fMap).forEach((k) => { feByDay[k] = fMap[k]; });
  const ferienByStart = {};
  (State.ferien || []).forEach((p) => {
    (ferienByStart[p.start] = ferienByStart[p.start] || []).push(p);
  });
  const keys = Array.from(new Set([
    ...Object.keys(byDay), ...Object.keys(feByDay), ...Object.keys(ferienByStart),
  ])).sort();
  if (!keys.length) { el.innerHTML = '<div class="empty">Keine kommenden Termine.<br>Tippe auf ＋, um einen Termin anzulegen.</div>'; return; }
  let html = '<div class="agenda">';
  keys.forEach((k) => {
    const d = parseLocal(k + "T00:00");
    const label = `${WEEKDAYS[(d.getDay() + 6) % 7]}, ${d.getDate()}. ${MONTHS[d.getMonth()]}`;
    html += `<div class="agenda-day"><div class="agenda-date">${label}</div>`;
    (feByDay[k] || []).forEach((n) => {
      html += `<div class="agenda-item hol-item">
        <div class="bar" style="background:#d99400"></div>
        <div class="agenda-time">🎌</div>
        <div style="flex:1"><div class="agenda-title">${escapeHtml(n)}</div>
          <div class="agenda-meta">Feiertag</div></div></div>`;
    });
    (ferienByStart[k] || []).forEach((p) => {
      const e = parseLocal(p.end + "T00:00");
      const until = `${e.getDate()}. ${MONTHS[e.getMonth()]}`;
      html += `<div class="agenda-item hol-item">
        <div class="bar" style="background:#2f9e63"></div>
        <div class="agenda-time">🏖️</div>
        <div style="flex:1"><div class="agenda-title">${escapeHtml(p.name)}</div>
          <div class="agenda-meta">Schulferien · bis ${until}</div></div></div>`;
    });
    (byDay[k] || []).forEach((o) => {
      const t = parseLocal(o.start);
      const time = o.all_day ? "ganztägig" : `${pad(t.getHours())}:${pad(t.getMinutes())}`;
      const persons = o.person_ids.map((id) => personById(id)).filter(Boolean).map((p) => p.name).join(", ");
      html += `<div class="agenda-item" data-event="${o.event_id}">
        <div class="bar" style="background:${o.color}"></div>
        <div class="agenda-time">${time}</div>
        <div style="flex:1">
          <div class="agenda-title">${escapeHtml(o.title)}${o.recurring ? " 🔁" : ""}</div>
          ${(o.location || persons) ? `<div class="agenda-meta">${escapeHtml([o.location, persons].filter(Boolean).join(" · "))}</div>` : ""}
        </div></div>`;
    });
    html += "</div>";
  });
  html += "</div>";
  el.innerHTML = html;
  $$(".agenda-item", el).forEach((i) => i.addEventListener("click", () => {
    if (i.dataset.event) openEventDetail(+i.dataset.event);
  }));
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------- Navigation --------------------------------- */
function navigate(dir) {
  const a = State.anchor;
  if (State.view === "month") a.setMonth(a.getMonth() + dir);
  else if (State.view === "week") a.setDate(a.getDate() + 7 * dir);
  else if (State.view === "day") a.setDate(a.getDate() + dir);
  else a.setDate(a.getDate() + 30 * dir);
  State.anchor = new Date(a);
  render();
}
$("#prev-btn").addEventListener("click", () => navigate(-1));
$("#next-btn").addEventListener("click", () => navigate(1));
$("#today-btn").addEventListener("click", () => { State.anchor = new Date(); render(); });
$$(".view-tab").forEach((t) => t.addEventListener("click", () => {
  State.view = t.dataset.view;
  localStorage.setItem("fk_view", State.view);
  render();
}));
$("#add-btn").addEventListener("click", () => openEventForm(null, null));

/* ------------------------- Termin abfotografieren --------------------- */
$("#scan-btn").addEventListener("click", () => $("#scan-input").click());
$("#scan-input").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (file) startScan(file);
});

async function startScan(file) {
  toast("📷 Foto wird ausgewertet…");
  let res;
  try {
    const fd = new FormData(); fd.append("file", file);
    res = await api("/scan", { method: "POST", body: fd });
  } catch (err) { toast(err.message); return; }
  const drafts = res.events || [];
  if (!drafts.length) {
    toast("Keine Termine erkannt – bitte von Hand anlegen"); return;
  }
  // Nutzer nacheinander durch die erkannten Termine führen
  let i = 0;
  const step = () => {
    if (i >= drafts.length) { toast("Fertig – alle Termine geprüft"); render(); return; }
    const draft = drafts[i];
    const badge = `Termin ${i + 1} von ${drafts.length}`;
    i += 1;
    openEventForm(null, null, {
      prefill: draft,
      attachFiles: [file],   // Originalfoto an jeden Termin anhängen
      badge,
      onClose: step,
    });
  };
  toast(`${drafts.length} Termin(e) erkannt – bitte prüfen`);
  step();
}

/* ------------------------- Suche -------------------------------------- */
$("#search-btn").addEventListener("click", () => {
  $("#search-bar").classList.toggle("hidden");
  if (!$("#search-bar").classList.contains("hidden")) $("#search-input").focus();
});
$("#search-close").addEventListener("click", () => {
  $("#search-bar").classList.add("hidden"); $("#search-input").value = ""; render();
});
let searchTimer;
$("#search-input").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  const q = e.target.value.trim();
  searchTimer = setTimeout(() => runSearch(q), 250);
});
async function runSearch(q) {
  const el = $("#calendar");
  if (!q) { render(); return; }
  try {
    const hits = await api("/search?q=" + encodeURIComponent(q));
    if (!hits.length) { el.innerHTML = '<div class="empty">Nichts gefunden.</div>'; return; }
    let html = '<div class="agenda"><div class="agenda-date" style="margin:8px 0">' +
      hits.length + ' Treffer</div>';
    hits.forEach((e2) => {
      const d = parseLocal(e2.start);
      html += `<div class="agenda-item" data-event="${e2.id}">
        <div class="bar" style="background:${e2.color}"></div>
        <div style="flex:1"><div class="agenda-title">${escapeHtml(e2.title)}</div>
        <div class="agenda-meta">${d.toLocaleDateString("de-DE")}${e2.location ? " · " + escapeHtml(e2.location) : ""}</div></div></div>`;
    });
    html += "</div>";
    el.innerHTML = html;
    $$(".agenda-item", el).forEach((i) => i.addEventListener("click", () => openEventDetail(+i.dataset.event)));
  } catch (err) { el.innerHTML = `<div class="empty">${err.message}</div>`; }
}

/* ------------------------- Modal-Basis -------------------------------- */
function openModal(title, bodyHtml, actionsHtml) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="modal-overlay">
    <div class="modal">
      <div class="modal-head"><h3>${title}</h3><button class="icon-btn" data-close>✕</button></div>
      <div class="modal-body">${bodyHtml}</div>
      ${actionsHtml ? `<div class="modal-actions">${actionsHtml}</div>` : ""}
    </div></div>`;
  const overlay = $(".modal-overlay", root);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) closeModal(); });
  $("[data-close]", root).addEventListener("click", closeModal);
  return root;
}
function closeModal() { $("#modal-root").innerHTML = ""; }

/* ------------------------- Termin-Formular ---------------------------- */
function reminderRow(minutes) {
  const opts = REMINDER_OPTIONS.map(([v, l]) =>
    `<option value="${v}" ${v === minutes ? "selected" : ""}>${l}</option>`).join("");
  return `<div class="reminder-item"><select>${opts}</select>
    <button type="button" class="rm">✕</button></div>`;
}

async function openEventForm(eventId, presetDate, opts = {}) {
  let ev = null;
  if (eventId) { try { ev = await api("/events/" + eventId); } catch (e) {} }
  if (!State.persons.length) { try { State.persons = await api("/persons"); } catch (e) {} }

  // base = bestehender Termin, sonst Vorbelegung (z. B. aus einem Foto)
  const base = ev || opts.prefill || {};
  const startSrc = ev ? ev.start : (opts.prefill && opts.prefill.start);
  const start = startSrc ? parseLocal(startSrc) : (presetDate || roundedNow());
  const endSrc = ev ? ev.end : (opts.prefill && opts.prefill.end);
  const end = endSrc ? parseLocal(endSrc) : new Date(start.getTime() + 60 * 60 * 1000);
  const rec = parseRRule(ev ? ev.rrule : null);
  const selPersons = new Set(ev ? ev.person_ids : []);
  const color = base.color || COLORS[0];

  const personPills = State.persons.map((p) =>
    `<button type="button" class="opt ${selPersons.has(p.id) ? "selected" : ""}" data-pid="${p.id}" style="${selPersons.has(p.id) ? `background:${p.color};border-color:${p.color}` : ""}">${escapeHtml(p.name)}</button>`).join("");
  const colorPills = COLORS.map((c) =>
    `<button type="button" class="opt color-opt" data-color="${c}" style="background:${c};width:32px;height:32px;padding:0;border-radius:50%;${c === color ? "outline:3px solid var(--text);outline-offset:2px" : ""}"></button>`).join("");
  const remindersHtml = (ev && ev.reminders.length ? ev.reminders : []).map(reminderRow).join("");

  const body = `
    <div class="form-section">
      <div class="sec-title">🗓️ Was & wann</div>
      <div class="field"><label>Titel</label>
        <input id="f-title" value="${escapeHtml(base.title || "")}" placeholder="z. B. Zahnarzt Mia" /></div>
      <div class="field checkbox-row"><input type="checkbox" id="f-allday" ${base.all_day ? "checked" : ""}/>
        <label for="f-allday" style="margin:0">Ganztägig</label></div>
      <div class="row2">
        <div class="field"><label>Beginn</label><input type="datetime-local" id="f-start" value="${toLocalInput(start)}" /></div>
        <div class="field"><label>Ende</label><input type="datetime-local" id="f-end" value="${toLocalInput(end)}" /></div>
      </div>
    </div>

    <div class="form-section">
      <div class="sec-title">📍 Ort & Notizen</div>
      <div class="field"><label>Ort</label>
        <input id="f-location" value="${escapeHtml(base.location || "")}" placeholder="Ort (optional)" /></div>
      <div class="field"><label>Notiz</label>
        <textarea id="f-desc" placeholder="Beschreibung (optional)">${escapeHtml(base.description || "")}</textarea>
        <div class="agenda-meta" style="margin-top:6px">Sehen alle beteiligten Personen.</div></div>
      <div class="field"><label>🔒 Private Notiz</label>
        <textarea id="f-private" placeholder="Nur für dich sichtbar – z. B. Geschenkideen">${escapeHtml(ev ? (ev.private_note || "") : "")}</textarea>
        <div class="agenda-meta" style="margin-top:6px">Diese Notiz sieht <b>niemand sonst</b> – auch nicht die beteiligten Personen.</div></div>
    </div>

    <div class="form-section">
      <div class="sec-title">🎨 Farbe & Wiederholung</div>
      <div class="field"><label>Farbe</label><div class="pill-select" id="f-colors">${colorPills}</div></div>
      <div class="field"><label>Wiederholung</label>
        <select id="f-freq">
          <option value="">Keine</option>
          <option value="DAILY" ${rec.freq === "DAILY" ? "selected" : ""}>Täglich</option>
          <option value="WEEKLY" ${rec.freq === "WEEKLY" ? "selected" : ""}>Wöchentlich</option>
          <option value="MONTHLY" ${rec.freq === "MONTHLY" ? "selected" : ""}>Monatlich</option>
          <option value="YEARLY" ${rec.freq === "YEARLY" ? "selected" : ""}>Jährlich</option>
        </select></div>
      <div class="row2" id="f-recur-extra" style="${rec.freq ? "" : "display:none"}">
        <div class="field"><label>Alle … (Intervall)</label>
          <input type="number" id="f-interval" min="1" value="${rec.interval || 1}" /></div>
        <div class="field"><label>Enddatum (optional)</label>
          <input type="date" id="f-until" value="${rec.until || ""}" /></div>
      </div>
    </div>

    <div class="form-section">
      <div class="sec-title">👥 Wer & Erinnerung</div>
      <div class="field"><label>Betrifft (Personen)</label>
        <div class="pill-select" id="f-persons">${personPills || '<span class="agenda-meta">Noch keine Personen – über 👥 anlegen</span>'}</div></div>
      <div class="field"><label>Erinnerungen (Push)</label>
        <div class="reminder-list" id="f-reminders">${remindersHtml}</div>
        <button type="button" class="btn-add-reminder" id="f-add-reminder">＋ Erinnerung hinzufügen</button>
        <div class="agenda-meta" style="margin-top:6px">Du kannst mehrere Erinnerungen setzen – z. B. 2 Tage und 2 Stunden vorher.</div></div>
    </div>

    <div class="form-section">
      <div class="sec-title">📎 Anhänge</div>
      <div class="field" id="f-attach-field">
        <label class="file-input-label">📎 Datei auswählen<input type="file" id="f-file" multiple /></label>
        <div class="attach-list" id="f-attach-list"></div></div>
    </div>`;

  const actions = `${eventId ? '<button class="btn-danger" id="f-delete">Löschen</button>' : ""}
    <button class="btn-primary" id="f-save">Speichern</button>`;

  const heading = (eventId ? "Termin bearbeiten" : "Neuer Termin")
    + (opts.badge ? ` · ${opts.badge}` : "");
  openModal(heading, body, actions);

  // Fortführung (z. B. nächster Termin aus einem Foto)
  let finished = false;
  function finish() { if (finished) return; finished = true; if (opts.onClose) opts.onClose(); }
  if (opts.onClose) {
    $("[data-close]").addEventListener("click", finish);
    $(".modal-overlay").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) finish();
    });
  }

  // Interaktionen
  let selColor = color;
  $$("#f-colors .color-opt").forEach((b) => b.addEventListener("click", () => {
    selColor = b.dataset.color;
    $$("#f-colors .color-opt").forEach((x) => x.style.outline = "none");
    b.style.outline = "3px solid var(--text)"; b.style.outlineOffset = "2px";
  }));
  $$("#f-persons .opt[data-pid]").forEach((b) => b.addEventListener("click", () => {
    const pid = +b.dataset.pid; const p = personById(pid);
    if (selPersons.has(pid)) { selPersons.delete(pid); b.classList.remove("selected"); b.style.background = ""; b.style.borderColor = ""; }
    else { selPersons.add(pid); b.classList.add("selected"); b.style.background = p.color; b.style.borderColor = p.color; }
  }));
  $("#f-freq").addEventListener("change", (e) => {
    $("#f-recur-extra").style.display = e.target.value ? "" : "none";
  });
  $("#f-allday").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("#f-start").type = on ? "date" : "datetime-local";
    $("#f-end").type = on ? "date" : "datetime-local";
  });
  if (base.all_day) { $("#f-start").type = "date"; $("#f-end").type = "date"; }
  $("#f-add-reminder").addEventListener("click", () => {
    $("#f-reminders").insertAdjacentHTML("beforeend", reminderRow(60));
    bindReminderRemovers();
  });
  bindReminderRemovers();

  // Anhänge – vorbelegte Dateien (z. B. das abfotografierte Bild) übernehmen
  const pendingFiles = (opts.attachFiles || []).slice();
  renderAttachList(ev, pendingFiles);
  $("#f-file").addEventListener("change", (e) => {
    for (const f of e.target.files) pendingFiles.push(f);
    e.target.value = "";
    renderAttachList(ev, pendingFiles);
  });

  if (eventId) $("#f-delete").addEventListener("click", async () => {
    if (!confirm("Diesen Termin wirklich löschen?")) return;
    await api("/events/" + eventId, { method: "DELETE" });
    closeModal(); finish(); toast("Termin gelöscht"); render();
  });

  $("#f-save").addEventListener("click", async () => {
    const title = $("#f-title").value.trim();
    if (!title) { toast("Bitte einen Titel eingeben"); return; }
    const allDay = $("#f-allday").checked;
    const startVal = $("#f-start").value;
    let endVal = $("#f-end").value || null;
    const startNorm = allDay ? startVal.split("T")[0] : startVal;
    const endNorm = endVal ? (allDay ? endVal.split("T")[0] : endVal) : null;
    const reminders = $$("#f-reminders select").map((s) => +s.value);
    const payload = {
      title, description: $("#f-desc").value, private_note: $("#f-private").value,
      location: $("#f-location").value,
      color: selColor, category: (ev && ev.category === "birthday") ? "birthday" : "general",
      start: startNorm, end: endNorm, all_day: allDay,
      rrule: buildRRule(), person_ids: [...selPersons], reminders,
    };
    try {
      let saved;
      if (eventId) saved = await api("/events/" + eventId, { method: "PUT", body: payload });
      else saved = await api("/events", { method: "POST", body: payload });
      for (const f of pendingFiles) {
        const fd = new FormData(); fd.append("file", f);
        await api(`/events/${saved.id}/attachments`, { method: "POST", body: fd });
      }
      closeModal(); finish(); toast("Gespeichert"); render();
    } catch (err) { toast(err.message); }
  });

  function buildRRule() {
    const freq = $("#f-freq").value;
    if (!freq) return null;
    let rule = `FREQ=${freq}`;
    const interval = +$("#f-interval").value || 1;
    if (interval > 1) rule += `;INTERVAL=${interval}`;
    const until = $("#f-until").value;
    if (until) rule += `;UNTIL=${until.replace(/-/g, "")}T235959`;
    return rule;
  }
}

function bindReminderRemovers() {
  $$("#f-reminders .rm").forEach((b) => b.onclick = () => b.parentElement.remove());
}
function roundedNow() {
  const d = new Date(); d.setMinutes(0, 0, 0); d.setHours(d.getHours() + 1); return d;
}
function attachmentUrl(a) {
  return a.url || `/api/attachments/${a.id}?token=${encodeURIComponent(State.token)}`;
}
function parseRRule(rrule) {
  if (!rrule) return { freq: "", interval: 1, until: "" };
  const parts = Object.fromEntries(rrule.split(";").map((p) => p.split("=")));
  let until = "";
  if (parts.UNTIL) {
    const u = parts.UNTIL;
    until = `${u.slice(0, 4)}-${u.slice(4, 6)}-${u.slice(6, 8)}`;
  }
  return { freq: parts.FREQ || "", interval: +(parts.INTERVAL || 1), until };
}

function renderAttachList(ev, pendingFiles) {
  const box = $("#f-attach-list");
  if (!box) return;
  let html = "";
  if (ev && ev.attachments) {
    ev.attachments.forEach((a) => {
      const url = attachmentUrl(a);
      const thumb = a.content_type.startsWith("image/")
        ? `<img class="attach-thumb" src="${url}" alt="" />` : "<span>📄</span>";
      html += `<div class="attach-item">${thumb}
        <a href="${url}" target="_blank">${escapeHtml(a.filename)}</a>
        <button type="button" class="icon-btn" data-att="${a.id}">🗑️</button></div>`;
    });
  }
  pendingFiles.forEach((f, i) => {
    html += `<div class="attach-item"><span>⬆️</span><a>${escapeHtml(f.name)} (neu)</a>
      <button type="button" class="icon-btn" data-pending="${i}">✕</button></div>`;
  });
  box.innerHTML = html;
  $$("[data-att]", box).forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Anhang löschen?")) return;
    await api("/attachments/" + b.dataset.att, { method: "DELETE" });
    ev.attachments = ev.attachments.filter((a) => a.id !== +b.dataset.att);
    renderAttachList(ev, pendingFiles);
  }));
  $$("[data-pending]", box).forEach((b) => b.addEventListener("click", () => {
    pendingFiles.splice(+b.dataset.pending, 1); renderAttachList(ev, pendingFiles);
  }));
}

/* ------------------------- Termin-Detail ------------------------------ */
async function openEventDetail(eventId, focusDate = false) {
  let ev;
  try { ev = await api("/events/" + eventId); } catch (e) { toast(e.message); return; }
  // Personen sicherstellen, damit die Namen im Detail erscheinen
  if (!State.persons.length) { try { State.persons = await api("/persons"); } catch (e) {} }
  // Beim Öffnen aus einer Benachrichtigung zum passenden Datum springen
  if (focusDate && ev.start) { State.anchor = parseLocal(ev.start); render(); }
  const start = parseLocal(ev.start);
  const end = ev.end ? parseLocal(ev.end) : null;
  const dateStr = ev.all_day
    ? start.toLocaleDateString("de-DE", { weekday: "long", day: "numeric", month: "long", year: "numeric" })
    : `${start.toLocaleDateString("de-DE", { weekday: "long", day: "numeric", month: "long" })}, ${pad(start.getHours())}:${pad(start.getMinutes())}${end ? "–" + pad(end.getHours()) + ":" + pad(end.getMinutes()) : ""}`;
  const persons = ev.person_ids.map((id) => personById(id)).filter(Boolean);
  const rec = parseRRule(ev.rrule);
  const recLabel = { DAILY: "täglich", WEEKLY: "wöchentlich", MONTHLY: "monatlich", YEARLY: "jährlich" }[rec.freq];

  let body = `<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
      <span style="width:14px;height:14px;border-radius:50%;background:${ev.color}"></span>
      <h2 style="margin:0;font-size:20px">${escapeHtml(ev.title)}</h2></div>
    <div class="detail-row"><span class="ic">🕐</span><span>${dateStr}${ev.all_day ? " · ganztägig" : ""}</span></div>`;
  if (recLabel) body += `<div class="detail-row"><span class="ic">🔁</span><span>Wiederholt sich ${recLabel}${rec.interval > 1 ? ` (alle ${rec.interval})` : ""}${rec.until ? `, bis ${parseLocal(rec.until + "T00:00").toLocaleDateString("de-DE")}` : ""}</span></div>`;
  if (ev.location) body += `<div class="detail-row"><span class="ic">📍</span><span>${escapeHtml(ev.location)}</span></div>`;
  if (ev.description) body += `<div class="detail-row"><span class="ic">📝</span><span>${escapeHtml(ev.description)}</span></div>`;
  if (ev.private_note) body += `<div class="detail-row"><span class="ic">🔒</span><div style="flex:1"><div style="white-space:pre-wrap">${escapeHtml(ev.private_note)}</div><div class="agenda-meta" style="margin-top:2px">Private Notiz – nur du siehst das</div></div></div>`;
  if (persons.length) body += `<div class="detail-row"><span class="ic">👤</span><div class="detail-persons">${persons.map((p) => `<span class="person-chip"><span class="person-dot" style="width:18px;height:18px;font-size:10px;background:${p.color}">${p.name[0]}</span>${escapeHtml(p.name)}</span>`).join("")}</div></div>`;
  if (ev.reminders.length) body += `<div class="detail-row"><span class="ic">🔔</span><span>${ev.reminders.map(reminderLabel).join(", ")}</span></div>`;
  if (ev.attachments.length) {
    body += `<div class="detail-row"><span class="ic">📎</span><div style="flex:1"><div class="attach-list">` +
      ev.attachments.map((a) => {
        const url = attachmentUrl(a);
        if (a.content_type.startsWith("image/")) {
          return `<a href="${url}" target="_blank" class="attach-photo">
            <img src="${url}" alt="${escapeHtml(a.filename)}">
            <span class="attach-name">${escapeHtml(a.filename)}</span></a>`;
        }
        return `<div class="attach-item"><span>📄</span><a href="${url}" target="_blank">${escapeHtml(a.filename)}</a></div>`;
      }).join("") + "</div></div></div>";
  }
  const actions = `<button class="btn-ghost" id="d-edit">Bearbeiten</button>
    <button class="btn-primary" id="d-close">Schließen</button>`;
  openModal("Termin", body, actions);
  $("#d-close").addEventListener("click", closeModal);
  $("#d-edit").addEventListener("click", () => { closeModal(); openEventForm(eventId, null); });
}
function reminderLabel(m) {
  const found = REMINDER_OPTIONS.find(([v]) => v === m);
  return found ? found[1] : m + " Min vorher";
}

/* ------------------------- Personen ----------------------------------- */
async function openPersons() {
  let persons = [];
  try { persons = await api("/persons"); State.persons = persons; } catch (e) {}
  const body = `<div id="persons-list">${persons.map(personRowHtml).join("") || '<div class="empty">Noch keine Personen.</div>'}</div>
    <button class="btn-add-reminder" id="p-add" style="margin-top:10px">＋ Person hinzufügen</button>`;
  openModal("Personen", body, `<button class="btn-primary" id="p-done">Fertig</button>`);
  $("#p-done").addEventListener("click", () => { closeModal(); render(); });
  $("#p-add").addEventListener("click", () => openPersonForm(null));
  bindPersonRows();
}
function personRowHtml(p) {
  return `<div class="person-row" data-pid="${p.id}">
    <div class="person-dot" style="background:${p.color}">${escapeHtml(p.name[0] || "?")}</div>
    <div class="person-info"><div class="pn">${escapeHtml(p.name)}</div>
      ${p.birthday ? `<div class="pm">🎂 ${parseLocal(p.birthday + "T00:00").toLocaleDateString("de-DE")}</div>` : ""}</div>
    ${p.has_app ? '<span class="badge-app">App</span>' : ""}
    <button class="icon-btn" data-edit="${p.id}">✏️</button></div>`;
}
function bindPersonRows() {
  $$("[data-edit]").forEach((b) => b.addEventListener("click", () => openPersonForm(+b.dataset.edit)));
}
async function openPersonForm(personId) {
  const persons = State.persons;
  const p = personId ? persons.find((x) => x.id === personId) : null;
  let users = [];
  try { users = await api("/users"); } catch (e) {}
  const linkedIds = persons.filter((x) => x.user_id && (!p || x.id !== p.id)).map((x) => x.user_id);
  const userOpts = ['<option value="">— kein App-Konto —</option>'].concat(
    users.map((u) => {
      const taken = linkedIds.includes(u.id) && (!p || p.user_id !== u.id);
      return `<option value="${u.id}" ${p && p.user_id === u.id ? "selected" : ""} ${taken ? "disabled" : ""}>${escapeHtml(u.name)} (${escapeHtml(u.email)})${taken ? " – schon verknüpft" : ""}</option>`;
    })).join("");
  const colorPills = COLORS.map((c) =>
    `<button type="button" class="opt color-opt" data-color="${c}" style="background:${c};width:32px;height:32px;padding:0;border-radius:50%;${(p ? p.color : COLORS[0]) === c ? "outline:3px solid var(--text);outline-offset:2px" : ""}"></button>`).join("");
  const body = `
    <div class="field"><label>Name</label><input id="pf-name" value="${escapeHtml(p ? p.name : "")}" /></div>
    <div class="field"><label>Geburtstag (optional)</label><input type="date" id="pf-bday" value="${p && p.birthday ? p.birthday : ""}" />
      <div class="agenda-meta" style="margin-top:6px">Wird automatisch als jährlicher Termin angelegt.</div></div>
    <div class="field"><label>Farbe</label><div class="pill-select" id="pf-colors">${colorPills}</div></div>
    <div class="field"><label>Mit App-Konto verknüpfen</label><select id="pf-user">${userOpts}</select>
      <div class="agenda-meta" style="margin-top:6px">Nur verknüpfte Personen erhalten Push-Nachrichten.</div></div>`;
  openModal(personId ? "Person bearbeiten" : "Neue Person", body,
    `${personId ? '<button class="btn-danger" id="pf-del">Löschen</button>' : ""}<button class="btn-primary" id="pf-save">Speichern</button>`);
  let selColor = p ? p.color : COLORS[0];
  $$("#pf-colors .color-opt").forEach((b) => b.addEventListener("click", () => {
    selColor = b.dataset.color;
    $$("#pf-colors .color-opt").forEach((x) => x.style.outline = "none");
    b.style.outline = "3px solid var(--text)"; b.style.outlineOffset = "2px";
  }));
  $("#pf-save").addEventListener("click", async () => {
    const name = $("#pf-name").value.trim();
    if (!name) { toast("Bitte Namen eingeben"); return; }
    const payload = { name, color: selColor, birthday: $("#pf-bday").value || null,
      user_id: $("#pf-user").value ? +$("#pf-user").value : null };
    try {
      if (personId) await api("/persons/" + personId, { method: "PUT", body: payload });
      else await api("/persons", { method: "POST", body: payload });
      State.persons = await api("/persons");
      openPersons();
    } catch (err) { toast(err.message); }
  });
  if (personId) $("#pf-del").addEventListener("click", async () => {
    if (!confirm("Person löschen?")) return;
    await api("/persons/" + personId, { method: "DELETE" });
    State.persons = await api("/persons");
    openPersons();
  });
}
$("#persons-btn").addEventListener("click", openPersons);

/* ------------------------- Einstellungen ------------------------------ */
async function openSettings() {
  const permission = ("Notification" in window) ? Notification.permission : "unsupported";
  const isSub = await currentSubscription() ? true : false;
  let hs = { state: "", public_holidays: false, school_holidays: false };
  try { hs = await api("/settings/holidays"); } catch (e) {}
  const stateOptions = ['<option value="">— keins —</option>'].concat(
    STATES_DE.map(([c, n]) =>
      `<option value="${c}" ${hs.state === c ? "selected" : ""}>${n}</option>`)).join("");
  const body = `
    <div class="detail-row"><span class="ic">👤</span><span>${escapeHtml(State.user.name)}<br><span class="agenda-meta">${escapeHtml(State.user.email)}</span></span></div>
    <div class="field" style="margin-top:16px"><label>Push-Benachrichtigungen</label>
      <div class="agenda-meta" style="margin-bottom:8px">Status: ${pushStatusText(permission, isSub)}</div>
      ${permission === "unsupported"
        ? '<div class="agenda-meta">Dieses Gerät unterstützt keine Web-Push-Nachrichten. Auf dem iPhone: App über „Zum Home-Bildschirm“ installieren und von dort öffnen.</div>'
        : (isSub
          ? '<button class="btn-ghost" id="s-push-off" style="width:100%">Benachrichtigungen deaktivieren</button><button class="btn-primary" id="s-push-test" style="width:100%;margin-top:8px">Test-Benachrichtigung senden</button>'
          : '<button class="btn-primary" id="s-push-on" style="width:100%">Benachrichtigungen aktivieren</button>')}
    </div>
    <div class="field"><label>Standard-Ansicht</label>
      <select id="s-view">
        <option value="month" ${State.view === "month" ? "selected" : ""}>Monat</option>
        <option value="week" ${State.view === "week" ? "selected" : ""}>Woche</option>
        <option value="day" ${State.view === "day" ? "selected" : ""}>Tag</option>
        <option value="agenda" ${State.view === "agenda" ? "selected" : ""}>Liste</option>
      </select></div>
    <div class="field"><label>🗓️ Feiertage & Schulferien</label>
      <select id="s-state">${stateOptions}</select>
      <div class="agenda-meta" style="margin:6px 0 10px">Bundesland wählen, dann unten aktivieren.</div>
      <div class="checkbox-row" style="margin-bottom:8px"><input type="checkbox" id="s-feiertage" ${hs.public_holidays ? "checked" : ""}/>
        <label for="s-feiertage" style="margin:0">Gesetzliche Feiertage anzeigen</label></div>
      <div class="checkbox-row"><input type="checkbox" id="s-ferien" ${hs.school_holidays ? "checked" : ""}/>
        <label for="s-ferien" style="margin:0">Schulferien anzeigen</label></div>
    </div>
    <button class="btn-ghost" id="s-logout" style="width:100%;margin-top:10px">Abmelden</button>
    <div class="agenda-meta" style="text-align:center;margin-top:14px">Familienkalender · alle Geräte teilen denselben Kalender</div>`;
  openModal("Einstellungen", body, `<button class="btn-primary" id="s-close">Fertig</button>`);
  $("#s-close").addEventListener("click", closeModal);
  $("#s-view").addEventListener("change", (e) => {
    State.view = e.target.value; localStorage.setItem("fk_view", State.view);
  });
  const saveHolidays = async () => {
    try {
      await api("/settings/holidays", { method: "PUT", body: {
        state: $("#s-state").value,
        public_holidays: $("#s-feiertage").checked,
        school_holidays: $("#s-ferien").checked,
      } });
      toast("Gespeichert");
      render();
    } catch (e) { toast(e.message); }
  };
  $("#s-state").addEventListener("change", saveHolidays);
  $("#s-feiertage").addEventListener("change", saveHolidays);
  $("#s-ferien").addEventListener("change", saveHolidays);
  $("#s-logout").addEventListener("click", () => { closeModal(); logout(); });
  const on = $("#s-push-on"); if (on) on.addEventListener("click", async () => { await enablePush(); closeModal(); openSettings(); });
  const off = $("#s-push-off"); if (off) off.addEventListener("click", async () => { await disablePush(); closeModal(); openSettings(); });
  const test = $("#s-push-test"); if (test) test.addEventListener("click", async () => {
    try { const r = await api("/push/test", { method: "POST" }); toast(`Test an ${r.sent} Gerät(e) gesendet`); }
    catch (e) { toast(e.message); }
  });
}
function pushStatusText(permission, isSub) {
  if (permission === "granted" && isSub) return "aktiv ✅";
  if (permission === "denied") return "blockiert (in den Browsereinstellungen erlauben)";
  return "nicht aktiv";
}
$("#settings-btn").addEventListener("click", openSettings);

/* ------------------------- Web-Push ----------------------------------- */
function urlBase64ToUint8Array(base64String) {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  return Uint8Array.from([...raw].map((c) => c.charCodeAt(0)));
}
async function swRegistration() {
  if (!navigator.serviceWorker) return null;
  return navigator.serviceWorker.ready;
}
async function currentSubscription() {
  const reg = await swRegistration();
  if (!reg) return null;
  return reg.pushManager.getSubscription();
}
async function enablePush() {
  if (!("Notification" in window) || !("serviceWorker" in navigator)) {
    toast("Push wird hier nicht unterstützt"); return;
  }
  const perm = await Notification.requestPermission();
  if (perm !== "granted") { toast("Benachrichtigungen nicht erlaubt"); return; }
  try {
    if (!State.vapidKey) State.vapidKey = (await api("/config")).vapid_public_key;
    const reg = await swRegistration();
    const sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(State.vapidKey),
    });
    const json = sub.toJSON();
    await api("/push/subscribe", { method: "POST", body: {
      endpoint: sub.endpoint, p256dh: json.keys.p256dh, auth: json.keys.auth } });
    toast("Benachrichtigungen aktiviert");
  } catch (e) { toast("Fehler: " + e.message); }
}
async function disablePush() {
  const sub = await currentSubscription();
  if (sub) {
    const json = sub.toJSON();
    try { await api("/push/unsubscribe", { method: "POST", body: {
      endpoint: sub.endpoint, p256dh: json.keys.p256dh, auth: json.keys.auth } }); } catch (e) {}
    await sub.unsubscribe();
  }
  toast("Benachrichtigungen deaktiviert");
}

/* ------------------------- Start -------------------------------------- */
async function init() {
  try {
    State.user = await api("/me");
  } catch (e) { showAuth(); return; }
  showApp();
  render();
  // Konfiguration laden (Foto-Erkennung nur zeigen, wenn eingerichtet)
  try {
    const cfg = await api("/config");
    State.vapidKey = cfg.vapid_public_key;
    $("#scan-btn").classList.toggle("hidden", !cfg.scan_enabled);
  } catch (e) {}
  // Bei bestehender Erlaubnis Abo im Hintergrund erneuern
  if ("Notification" in window && Notification.permission === "granted"
      && "serviceWorker" in navigator) {
    enablePush().catch(() => {});
  }
  // Deep-Link ?event=ID (z. B. aus einer Push-Benachrichtigung)
  const params = new URLSearchParams(location.search);
  const evId = params.get("event");
  if (evId) {
    history.replaceState({}, "", "/");   // URL säubern, damit Neuladen nicht erneut öffnet
    openEventDetail(+evId, true);
  }
}

async function configureAuthScreen() {
  // Familien-Code-Feld nur zeigen, wenn der Server ihn verlangt
  try {
    const cfg = await api("/config");
    State.vapidKey = cfg.vapid_public_key;
    const field = $("#register-familycode");
    if (field) {
      if (cfg.family_code_required) { field.classList.remove("hidden"); field.required = true; }
      else { field.classList.add("hidden"); field.required = false; }
    }
  } catch (e) {}
}

function setupSwipe() {
  const cal = $("#calendar");
  if (!cal) return;
  let sx = null, sy = null, moved = false;
  cal.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1) { sx = null; return; }
    sx = e.touches[0].clientX; sy = e.touches[0].clientY; moved = false;
  }, { passive: true });
  cal.addEventListener("touchmove", () => { moved = true; }, { passive: true });
  cal.addEventListener("touchend", (e) => {
    if (sx === null || !moved) return;
    const dx = e.changedTouches[0].clientX - sx;
    const dy = e.changedTouches[0].clientY - sy;
    sx = null;
    // Nur klare horizontale Wischer (nicht beim Hoch-/Runterscrollen)
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.6) {
      navigate(dx < 0 ? 1 : -1);   // links wischen = vor, rechts = zurück
    }
  }, { passive: true });
}

async function boot() {
  if ("serviceWorker" in navigator) {
    try { await navigator.serviceWorker.register("/service-worker.js"); } catch (e) {}
  }
  // Antippen einer Push-Nachricht bei bereits offener App -> Termin öffnen
  if (navigator.serviceWorker) {
    navigator.serviceWorker.addEventListener("message", (e) => {
      const d = e.data || {};
      if (d.type === "open-event" && d.eventId && State.token) {
        openEventDetail(+d.eventId, true);
      }
    });
  }
  setupSwipe();
  configureAuthScreen();
  if (State.token) init();
  else showAuth();
}
boot();
