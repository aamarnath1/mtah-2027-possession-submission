"use strict";
// Match viewer: tactical video with tracking overlay (left), digital pitch (right), label timelines.
// Frame numbers are 0-based video frames everywhere (= submission rows = homography `frame`).

const CODES = { H: 0, A: 1, D: 2, X: 3 };
const NONE = 255;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = {
  0: css("--home"), 1: css("--away"), 2: css("--dead"), 3: "#555c66", [NONE]: css("--none"),
};
const CODE_NAME = { 0: "H", 1: "A", 2: "D", 3: "X", [NONE]: "–" };
const $ = (id) => document.getElementById(id);

const state = {
  meta: null, nSlots: 0, nFrames: 0, fps: 25,
  pos: null, ballSpeed: null, H: null, tags: [], slotTeam: null,
  frame: 0, playing: false, rate: 1, offset: 0,
  hasVideo: false, videoW: 0, videoH: 0,
  labels: { A: null, B: null }, labelNames: { A: "", B: "" },
  show: { overlay: true, numbers: false, trail: true, nearest: true, prediction: true },
  zoomFrames: 750, fullCache: null,
};

// ---------- loading ----------

async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.arrayBuffer();
}

async function loadData() {
  const meta = await (await fetch("/data/meta.json")).json();
  if (meta.byte_order && meta.byte_order !== "little") {
    throw new Error("Viewer data was written big-endian; rebuild it on this machine.");
  }
  state.meta = meta;
  state.nFrames = meta.n_frames;
  state.fps = meta.fps;
  state.nSlots = meta.slot_teams.length;
  state.slotTeam = Uint8Array.from(meta.slot_teams, (t) => (t === "H" ? 0 : 1));
  const [pos, speed, hom, tags] = await Promise.all([
    fetchBuffer("/data/positions.f32"),
    fetchBuffer("/data/ball_speed.f32"),
    fetchBuffer("/data/homography.f64"),
    fetch("/data/tags.json").then((r) => (r.ok ? r.json() : [])),
  ]);
  state.pos = new Float32Array(pos);
  state.ballSpeed = new Float32Array(speed);
  state.H = new Float64Array(hom);
  state.tags = tags;
  const expect = state.nFrames * (1 + state.nSlots) * 2;
  if (state.pos.length !== expect) throw new Error(`positions.f32 has ${state.pos.length} values, expected ${expect}`);
  if (state.H.length !== state.nFrames * 9) throw new Error("homography.f64 has the wrong size");
}

function parseLabels(text) {
  const out = new Uint8Array(state.nFrames).fill(NONE);
  const lines = text.split(/\r?\n/);
  let start = 0;
  if (lines.length && !/^\s*\d/.test(lines[0])) start = 1; // header
  let bad = 0;
  for (let i = start; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const comma = line.indexOf(",");
    const frame = parseInt(line.slice(0, comma), 10);
    const code = CODES[line.slice(comma + 1).trim().toUpperCase()];
    if (Number.isInteger(frame) && frame >= 0 && frame < state.nFrames && code !== undefined) out[frame] = code;
    else bad++;
  }
  if (bad) console.warn(`labels: ${bad} rows skipped`);
  return out;
}

async function refreshLabelList() {
  const names = await (await fetch("/api/labels", { cache: "no-store" })).json();
  for (const key of ["A", "B"]) {
    const select = $(`labels${key}`);
    const keep = select.value;
    select.innerHTML = '<option value="">none</option>' +
      names.map((n) => `<option value="${encodeURIComponent(n)}">${n.replace(/</g, "&lt;")}</option>`).join("");
    if (names.map(encodeURIComponent).includes(keep)) select.value = keep;
  }
}

function remember(key, value) { try { localStorage.setItem(`viewer.${key}`, value); } catch (_) { /* private mode */ } }
function recall(key) { try { return localStorage.getItem(`viewer.${key}`); } catch (_) { return null; } }

async function selectLabels(key, encodedName) {
  remember(`labels${key}`, encodedName || "");
  state.labelNames[key] = encodedName ? decodeURIComponent(encodedName) : "";
  state.labels[key] = encodedName
    ? parseLabels(await (await fetch(`/labels/${encodedName}`, { cache: "no-store" })).text())
    : null;
  state.fullCache = null;
  render();
}

// ---------- geometry ----------

function slotXY(frame, slot) { // slot 0 = ball, 1..n = players
  const i = (frame * (1 + state.nSlots) + slot) * 2;
  return [state.pos[i], state.pos[i + 1]];
}

function project(frame, x, y) {
  const h = state.H, o = frame * 9;
  const d = h[o + 6] * x + h[o + 7] * y + h[o + 8];
  if (!(d > 1e-9)) return null; // NaN (invalid frame) or behind the camera
  return [(h[o] * x + h[o + 1] * y + h[o + 2]) / d, (h[o + 3] * x + h[o + 4] * y + h[o + 5]) / d];
}

function nearestPerTeam(frame) {
  const [bx, by] = slotXY(frame, 0);
  if (Number.isNaN(bx)) return null;
  const [L, W] = state.meta.pitch_m;
  const best = [{ slot: -1, d: Infinity }, { slot: -1, d: Infinity }];
  for (let s = 1; s <= state.nSlots; s++) {
    const [x, y] = slotXY(frame, s);
    if (Number.isNaN(x)) continue;
    const d = Math.hypot((x - bx) * L, (y - by) * W);
    const t = state.slotTeam[s - 1];
    if (d < best[t].d) best[t] = { slot: s, d };
  }
  return best;
}

// ---------- canvases ----------

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, canvas.clientWidth, canvas.clientHeight];
}

function dot(ctx, x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.lineWidth = 1.5; ctx.strokeStyle = stroke; ctx.stroke(); }
}

function drawOverlay() {
  const [ctx, w, h] = fitCanvas($("overlay"));
  ctx.clearRect(0, 0, w, h);
  paintStageBorder();
  if (!state.hasVideo || !state.videoW) return;
  const f = state.frame;
  const scale = Math.min(w / state.videoW, h / state.videoH);
  const ox = (w - state.videoW * scale) / 2, oy = (h - state.videoH * scale) / 2;
  if (state.show.overlay) drawTracking(ctx, f, scale, ox, oy);
  if (predictionOn()) drawPredictionBanner(ctx, ox + 10, oy + 10);
}

function drawTracking(ctx, f, scale, ox, oy) {
  const toScreen = (x, y) => {
    const p = project(f, x, y);
    return p ? [ox + p[0] * scale, oy + p[1] * scale] : null;
  };
  // pitch outline: a quick visual check that the homography fits this frame
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => toScreen(x, y));
  if (corners.every(Boolean)) {
    ctx.beginPath();
    corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(242, 201, 76, 0.5)";
    ctx.stroke();
  }
  for (let s = 1; s <= state.nSlots; s++) {
    const [x, y] = slotXY(f, s);
    if (Number.isNaN(x)) continue;
    const p = toScreen(x, y);
    if (!p) continue;
    playerDot(ctx, p[0], p[1], 5, state.slotTeam[s - 1], "#fff");
    if (state.show.numbers) label(ctx, p[0] + 7, p[1] - 6, String(s));
  }
  const [bx, by] = slotXY(f, 0);
  if (!Number.isNaN(bx)) {
    const p = toScreen(bx, by);
    if (p) { dot(ctx, p[0], p[1], 7, "transparent", css("--accent")); dot(ctx, p[0], p[1], 2.5, css("--accent")); }
  }
}

// ---------- prediction layer (Labels A on the video) ----------

function predictionOn() { return state.show.prediction && !!state.labels.A; }

function predictedCode(f = state.frame) { return state.labels.A ? state.labels.A[f] : NONE; }

function codeTitle(code) {
  if (code === 0) return `${state.meta.teams.H} · H`;
  if (code === 1) return `${state.meta.teams.A} · A`;
  if (code === 2) return "Dead ball · D";
  if (code === 3) return "No answer · X";
  return "–";
}

// The predicted team's players get a bright ring; the other team (or everyone,
// on a dead ball) is dimmed, so the call is readable at a glance while playing.
function playerDot(ctx, x, y, r, team, stroke) {
  if (!predictionOn()) { dot(ctx, x, y, r, COLORS[team], stroke); return; }
  const code = predictedCode();
  const owner = code === 0 || code === 1 ? code : -1;
  ctx.save();
  if (owner !== team) ctx.globalAlpha = 0.45;
  dot(ctx, x, y, r, COLORS[team], stroke);
  ctx.restore();
  if (owner === team) {
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,.95)";
    ctx.stroke();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawPredictionBanner(ctx, x, y) {
  const f = state.frame;
  const code = predictedCode(f);
  const run = runAround(state.labels.A, f);
  const elapsed = ((f - run.a + 1) / state.fps).toFixed(1);
  const lines = [{ text: codeTitle(code), font: "700 18px -apple-system, sans-serif", color: "#fff" },
                 { text: `Labels A · ${elapsed} s into this run`, font: "12px -apple-system, sans-serif", color: "rgba(255,255,255,.85)" }];
  let bCode = null;
  if (state.labels.B) {
    bCode = state.labels.B[f];
    lines.push({ text: `Labels B: ${codeTitle(bCode)} · ${bCode === code ? "agree" : "differ"}`,
                 font: "600 12px -apple-system, sans-serif", color: bCode === code ? "rgba(255,255,255,.85)" : "#ffd166" });
  }
  ctx.save();
  let width = 0;
  for (const l of lines) { ctx.font = l.font; width = Math.max(width, ctx.measureText(l.text).width); }
  const pad = 10, lineH = [24, 17, 17], height = pad * 2 + lines.reduce((a, _, i) => a + lineH[i], 0) - 4;
  roundRect(ctx, x, y, width + pad * 2 + 8, height, 8);
  ctx.fillStyle = "rgba(10,12,16,.72)";
  ctx.fill();
  ctx.fillStyle = COLORS[code];
  ctx.fillRect(x, y + 6, 6, height - 12);           // colour bar = the call
  let ty = y + pad + 15;
  lines.forEach((l, i) => {
    ctx.font = l.font; ctx.fillStyle = l.color;
    ctx.fillText(l.text, x + pad + 8, ty);
    ty += lineH[i + 1] || 0;
  });
  ctx.restore();
}

function paintStageBorder() {
  const stage = $("videoStage");
  stage.style.boxShadow = predictionOn() ? `inset 0 0 0 4px ${COLORS[predictedCode()]}` : "none";
}

function label(ctx, x, y, text) {
  ctx.font = "600 10px ui-monospace, Menlo, monospace";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,.8)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x, y);
}

function drawPitch() {
  const [ctx, w, h] = fitCanvas($("pitch"));
  const [L, W] = state.meta.pitch_m;
  const pad = 18;
  const s = Math.min((w - 2 * pad) / L, (h - 2 * pad) / W);
  const ox = (w - L * s) / 2, oy = (h - W * s) / 2;
  const X = (m) => ox + m * s, Y = (m) => oy + m * s;

  ctx.fillStyle = css("--grass");
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 10; i++) { // mowing stripes
    if (i % 2) { ctx.fillStyle = css("--grass-2"); ctx.fillRect(X((L / 10) * i), Y(0), (L / 10) * s, W * s); }
  }
  ctx.strokeStyle = css("--pitch-line");
  ctx.lineWidth = 1.5;
  const rect = (x, y, rw, rh) => ctx.strokeRect(X(x), Y(y), rw * s, rh * s);
  const circle = (x, y, r, a0 = 0, a1 = Math.PI * 2) => { ctx.beginPath(); ctx.arc(X(x), Y(y), r * s, a0, a1); ctx.stroke(); };
  rect(0, 0, L, W);
  ctx.beginPath(); ctx.moveTo(X(L / 2), Y(0)); ctx.lineTo(X(L / 2), Y(W)); ctx.stroke();
  circle(L / 2, W / 2, 9.15);
  for (const side of [0, 1]) {
    const x0 = side ? L - 16.5 : 0, g0 = side ? L - 5.5 : 0, spot = side ? L - 11 : 11;
    rect(x0, W / 2 - 20.16, 16.5, 40.32);
    rect(g0, W / 2 - 9.16, 5.5, 18.32);
    const a = Math.acos(5.5 / 9.15);
    if (side) circle(spot, W / 2, 9.15, Math.PI - a, Math.PI + a); else circle(spot, W / 2, 9.15, -a, a);
    ctx.fillStyle = css("--pitch-line");
    ctx.beginPath(); ctx.arc(X(spot), Y(W / 2), 2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeRect(side ? X(L) : X(-1.5), Y(W / 2 - 3.66), 1.5 * s, 7.32 * s);
  }
  ctx.beginPath(); ctx.arc(X(L / 2), Y(W / 2), 2, 0, Math.PI * 2); ctx.fill();

  const f = state.frame;
  const [bx, by] = slotXY(f, 0);
  const near = state.show.nearest ? nearestPerTeam(f) : null;
  if (near) {
    for (const t of [0, 1]) {
      if (near[t].slot < 0) continue;
      const [x, y] = slotXY(f, near[t].slot);
      ctx.beginPath(); ctx.moveTo(X(bx * L), Y(by * W)); ctx.lineTo(X(x * L), Y(y * W));
      ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = COLORS[t]; ctx.stroke(); ctx.setLineDash([]);
    }
  }
  const r = Math.max(4, 0.9 * s);
  for (let slot = 1; slot <= state.nSlots; slot++) {
    const [x, y] = slotXY(f, slot);
    if (Number.isNaN(x)) continue;
    playerDot(ctx, X(x * L), Y(y * W), r, state.slotTeam[slot - 1], "rgba(255,255,255,.9)");
    if (state.show.numbers) label(ctx, X(x * L) + r + 2, Y(y * W) - r, String(slot));
  }
  if (state.show.trail) {
    ctx.beginPath();
    let pen = false;
    for (let k = Math.max(0, f - 50); k <= f; k++) {
      const [x, y] = slotXY(k, 0);
      if (Number.isNaN(x)) { pen = false; continue; }
      if (pen) ctx.lineTo(X(x * L), Y(y * W)); else ctx.moveTo(X(x * L), Y(y * W));
      pen = true;
    }
    ctx.lineWidth = 2; ctx.strokeStyle = "rgba(242, 201, 76, .55)"; ctx.stroke();
  }
  if (!Number.isNaN(bx)) dot(ctx, X(bx * L), Y(by * W), Math.max(3.5, 0.6 * s), css("--accent"), "#000");
  else { ctx.fillStyle = "rgba(255,255,255,.6)"; ctx.font = "12px sans-serif"; ctx.fillText("ball not seen", X(1), Y(W) - 6); }
  if (predictionOn() && !state.hasVideo) drawPredictionBanner(ctx, 10, 10);

  const badge = $("pitchBadge");
  const code = state.labels.A ? state.labels.A[f] : NONE;
  badge.textContent = state.labels.A ? `A: ${CODE_NAME[code]}` : "";
  badge.style.background = COLORS[code];
  badge.style.display = state.labels.A ? "" : "none";
}

// ---------- timelines ----------

const TRACKS = [
  { key: "A", name: "Labels A" },
  { key: "B", name: "Labels B" },
  { key: "tags", name: "Tag phases" },
  { key: "set", name: "Set pieces" },
];

function trackRows(h) {
  const top = 4, gap = 3, rh = (h - top * 2 - gap * (TRACKS.length - 1)) / TRACKS.length;
  return TRACKS.map((t, i) => ({ ...t, y: top + i * (rh + gap), h: rh }));
}

function majorityCodes(labels, width, f0, f1) {
  const out = new Uint8Array(width).fill(NONE);
  const span = (f1 - f0) / width;
  const counts = new Uint32Array(5);
  for (let px = 0; px < width; px++) {
    const a = Math.floor(f0 + px * span), b = Math.max(a + 1, Math.floor(f0 + (px + 1) * span));
    counts.fill(0);
    for (let f = a; f < b && f < state.nFrames; f++) counts[labels[f] === NONE ? 4 : labels[f]]++;
    let best = 4;
    for (let c = 0; c < 4; c++) if (counts[c] > counts[best] || (best === 4 && counts[c] > 0)) best = c;
    out[px] = best === 4 ? NONE : best;
  }
  return out;
}

function drawTimeline(canvas, f0, f1, cacheKey) {
  const [ctx, w, h] = fitCanvas(canvas);
  const left = 78, plotW = Math.max(1, w - left - 6);
  const rows = trackRows(h);
  const fx = (f) => left + ((f - f0) / (f1 - f0)) * plotW;
  ctx.clearRect(0, 0, w, h);

  // graded blocks as a faint background band
  ctx.fillStyle = "rgba(255,255,255,.04)";
  for (const [a, b] of state.meta.graded_blocks || []) {
    const x0 = Math.max(left, fx(a)), x1 = Math.min(left + plotW, fx(b + 1));
    if (x1 > x0) ctx.fillRect(x0, 0, x1 - x0, h);
  }
  ctx.font = "11px -apple-system, sans-serif";
  for (const row of rows) {
    ctx.fillStyle = css("--muted");
    ctx.fillText(row.name, 6, row.y + row.h / 2 + 4);
    ctx.fillStyle = "rgba(255,255,255,.03)";
    ctx.fillRect(left, row.y, plotW, row.h);
    if (row.key === "A" || row.key === "B") {
      const labels = state.labels[row.key];
      if (!labels) continue;
      const px = Math.round(plotW);
      let codes;
      const key = `${cacheKey}|${row.key}|${px}|${state.labelNames[row.key]}`;
      if (cacheKey && state.fullCache && state.fullCache[key]) codes = state.fullCache[key];
      else {
        codes = majorityCodes(labels, px, f0, f1);
        if (cacheKey) (state.fullCache ||= {})[key] = codes;
      }
      let run = 0;
      for (let i = 1; i <= px; i++) {
        if (i === px || codes[i] !== codes[run]) {
          if (codes[run] !== NONE) { ctx.fillStyle = COLORS[codes[run]]; ctx.fillRect(left + run, row.y, i - run, row.h); }
          run = i;
        }
      }
    } else {
      const sets = row.key === "set";
      for (const tag of state.tags) {
        const isSet = tag.code.toUpperCase() === "SET PIECES";
        if (isSet !== sets || tag.end <= f0 || tag.start >= f1 || tag.end <= tag.start) continue;
        const x0 = Math.max(left, fx(tag.start)), x1 = Math.min(left + plotW, fx(tag.end));
        ctx.fillStyle = tag.team === "H" ? COLORS[0] : tag.team === "A" ? COLORS[1] : "#9aa3ad";
        ctx.globalAlpha = sets ? 0.95 : 0.35;
        ctx.fillRect(x0, row.y, Math.max(1, x1 - x0), row.h);
        ctx.globalAlpha = 1;
      }
    }
  }
  // playhead
  const x = fx(state.frame);
  if (x >= left && x <= left + plotW) {
    ctx.fillStyle = "#fff";
    ctx.fillRect(Math.round(x) - 1, 0, 2, h);
  }
  canvas._map = { f0, f1, left, plotW };
}

function drawTimelines() {
  drawTimeline($("tlFull"), 0, state.nFrames, "full");
  const half = state.zoomFrames;
  let f0 = state.frame - half, f1 = state.frame + half;
  if (f0 < 0) { f1 -= f0; f0 = 0; }
  if (f1 > state.nFrames) { f0 = Math.max(0, f0 - (f1 - state.nFrames)); f1 = state.nFrames; }
  drawTimeline($("tlZoom"), f0, f1, null);
}

function seekFromTimeline(canvas, clientX) {
  const m = canvas._map;
  if (!m) return;
  const rect = canvas.getBoundingClientRect();
  const t = (clientX - rect.left - m.left) / m.plotW;
  seek(Math.round(m.f0 + Math.min(1, Math.max(0, t)) * (m.f1 - m.f0)));
}

// ---------- info ----------

function timecode(frame) {
  const total = frame / state.fps;
  const hh = Math.floor(total / 3600), mm = Math.floor((total % 3600) / 60), ss = Math.floor(total % 60);
  const ff = frame % state.fps;
  return `${hh}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ff).padStart(2, "0")}`;
}

function runAround(labels, f) {
  const code = labels[f];
  let a = f, b = f;
  while (a > 0 && labels[a - 1] === code) a--;
  while (b < state.nFrames - 1 && labels[b + 1] === code) b++;
  return { code, a, b };
}

function chip(code) { return `<span class="chip" style="background:${COLORS[code]}">${CODE_NAME[code]}</span>`; }

function renderInfo() {
  const f = state.frame;
  const graded = (state.meta.graded_blocks || []).findIndex(([a, b]) => f >= a && f <= b);
  const cards = [];
  cards.push(`<div class="card"><h3>Frame</h3><div class="big">${f.toLocaleString()}</div>
    <div class="muted">${timecode(f)} video time · ${graded >= 0 ? `graded, half ${graded + 1}` :
      (state.meta.graded_blocks || []).length ? "not graded" : "graded range unknown"}</div></div>`);
  for (const key of ["A", "B"]) {
    const labels = state.labels[key];
    if (!labels) continue;
    const run = runAround(labels, f);
    cards.push(`<div class="card"><h3>Labels ${key} · ${state.labelNames[key].replace(/</g, "&lt;")}</h3>
      <div class="big">${chip(run.code)}</div>
      <div class="muted">run ${run.a.toLocaleString()}–${run.b.toLocaleString()} (${((run.b - run.a + 1) / state.fps).toFixed(1)} s)</div></div>`);
  }
  if (state.labels.A && state.labels.B) {
    const same = state.labels.A[f] === state.labels.B[f];
    cards.push(`<div class="card"><h3>A vs B</h3><div class="big">${same ? "agree" : "differ"}</div></div>`);
  }
  const [bx] = slotXY(f, 0);
  const speed = state.ballSpeed[f];
  let seen = [0, 0];
  for (let s = 1; s <= state.nSlots; s++) if (!Number.isNaN(slotXY(f, s)[0])) seen[state.slotTeam[s - 1]]++;
  const near = nearestPerTeam(f);
  cards.push(`<div class="card"><h3>Tracking</h3>
    <div>Ball: ${Number.isNaN(bx) ? "not seen" : `seen · ${Number.isNaN(speed) ? "?" : speed.toFixed(1)} m/s`}</div>
    <div>Players seen: ${chip(0)} ${seen[0]} &nbsp; ${chip(1)} ${seen[1]}</div>
    ${near ? `<div class="muted">Nearest to ball: H ${near[0].slot > 0 ? near[0].d.toFixed(1) + " m" : "–"} · A ${near[1].slot > 0 ? near[1].d.toFixed(1) + " m" : "–"}</div>` : ""}</div>`);
  const active = state.tags.filter((t) => t.start <= f && f < t.end);
  cards.push(`<div class="card"><h3>Smart Tagging (${active.length} active)</h3>${active.length ? "<ul>" +
    active.map((t) => `<li>${t.code}${t.type ? ` · ${t.type}` : ""} · ${t.team === "?" ? (t.team_raw || "?") : chip(CODES[t.team])}</li>`).join("") +
    "</ul>" : '<span class="muted">none</span>'}</div>`);
  $("info").innerHTML = cards.join("");
  $("clock").textContent = `${timecode(f)} · ${state.rate}×`;
  if (document.activeElement !== $("frameInput")) $("frameInput").value = f;
}

// ---------- playback ----------

function render() {
  drawOverlay();
  drawPitch();
  drawTimelines();
  renderInfo();
  $("playBtn").textContent = state.playing ? "❚❚" : "▶";
}

function clampFrame(f) { return Math.min(state.nFrames - 1, Math.max(0, f)); }

function seek(frame) {
  state.frame = clampFrame(frame);
  if (state.hasVideo) {
    const videoFrame = state.frame - state.offset;
    $("video").currentTime = Math.max(0, (videoFrame + 0.5) / state.fps);
  }
  render();
}

function step(n) { if (state.playing) togglePlay(); seek(state.frame + n); }

function togglePlay() {
  state.playing = !state.playing;
  const video = $("video");
  if (state.hasVideo) {
    if (state.playing) video.play().catch(() => { state.playing = false; render(); }); else video.pause();
  } else if (state.playing) {
    clock.last = performance.now();
    requestAnimationFrame(clock);
  }
  render();
}

function clock(now) { // drives playback when there is no video
  if (!state.playing || state.hasVideo) return;
  clock.acc = (clock.acc || 0) + ((now - clock.last) / 1000) * state.fps * state.rate;
  clock.last = now;
  const whole = Math.floor(clock.acc);
  if (whole) {
    clock.acc -= whole;
    state.frame = clampFrame(state.frame + whole);
    if (state.frame >= state.nFrames - 1) state.playing = false;
    render();
  }
  requestAnimationFrame(clock);
}

function setupVideo() {
  const video = $("video");
  return new Promise((resolve) => {
    let done = false;
    const finish = (ok) => { if (!done) { done = true; resolve(ok); } };
    video.addEventListener("loadedmetadata", () => {
      state.hasVideo = true;
      state.videoW = video.videoWidth; state.videoH = video.videoHeight;
      const frames = Math.round(video.duration * state.fps);
      $("videoStatus").textContent = `${state.videoW}×${state.videoH} · ${frames.toLocaleString()} frames`;
      if (Math.abs(frames - state.nFrames) > 2) {
        $("videoStatus").textContent += ` ⚠ data has ${state.nFrames.toLocaleString()}`;
      }
      finish(true);
    }, { once: true });
    video.addEventListener("error", () => { $("noVideo").hidden = false; finish(false); }, { once: true });
    setTimeout(() => finish(false), 8000);
    video.src = "/video";
  }).then((ok) => {
    if (!ok) { $("noVideo").hidden = false; video.removeAttribute("src"); video.load(); return; }
    video.playbackRate = state.rate;
    const onFrame = (_now, meta) => {
      const f = clampFrame(Math.round(meta.mediaTime * state.fps) + state.offset);
      if (f !== state.frame) { state.frame = f; render(); }
      video.requestVideoFrameCallback(onFrame);
    };
    if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(onFrame);
    else {
      const poll = () => {
        const f = clampFrame(Math.floor(video.currentTime * state.fps + 1e-3) + state.offset);
        if (f !== state.frame) { state.frame = f; render(); }
        requestAnimationFrame(poll);
      };
      requestAnimationFrame(poll);
    }
    video.addEventListener("ended", () => { state.playing = false; render(); });
    video.addEventListener("pause", () => { if (state.playing) { state.playing = false; render(); } });
  });
}

function setRate(rate) {
  state.rate = rate;
  $("speed").value = String(rate);
  if (state.hasVideo) $("video").playbackRate = rate;
  render();
}

// ---------- wiring ----------

function bind() {
  $("playBtn").onclick = togglePlay;
  document.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => step(Number(b.dataset.step))));
  $("prevSwitch").onclick = () => jumpSwitch(-1);
  $("nextSwitch").onclick = () => jumpSwitch(1);
  $("speed").onchange = (e) => setRate(Number(e.target.value));
  $("frameInput").onchange = (e) => { if (state.playing) togglePlay(); seek(Number(e.target.value) || 0); };
  $("offset").onchange = (e) => { state.offset = Math.round(Number(e.target.value) || 0); seek(state.frame); };
  $("labelsA").onchange = (e) => selectLabels("A", e.target.value);
  $("labelsB").onchange = (e) => selectLabels("B", e.target.value);
  $("reloadLabels").onclick = async () => {
    await refreshLabelList();
    for (const k of ["A", "B"]) if ($(`labels${k}`).value) await selectLabels(k, $(`labels${k}`).value);
  };
  for (const [id, key] of [["tOverlay", "overlay"], ["tNumbers", "numbers"], ["tTrail", "trail"], ["tNearest", "nearest"], ["tPrediction", "prediction"]]) {
    $(id).onchange = (e) => { state.show[key] = e.target.checked; render(); };
  }
  $("helpBtn").onclick = () => $("help").showModal();
  for (const id of ["tlFull", "tlZoom"]) {
    const c = $(id);
    let dragging = false;
    c.addEventListener("pointerdown", (e) => { dragging = true; c.setPointerCapture(e.pointerId); if (state.playing) togglePlay(); seekFromTimeline(c, e.clientX); });
    c.addEventListener("pointermove", (e) => { if (dragging && id === "tlFull") seekFromTimeline(c, e.clientX); });
    c.addEventListener("pointerup", () => (dragging = false));
  }
  $("tlZoom").addEventListener("wheel", (e) => {
    e.preventDefault();
    state.zoomFrames = Math.round(Math.min(state.fps * 600, Math.max(state.fps * 2, state.zoomFrames * (e.deltaY > 0 ? 1.25 : 0.8))));
    $("zoomSecs").textContent = Math.round(state.zoomFrames / state.fps);
    drawTimelines();
  }, { passive: false });
  window.addEventListener("resize", () => { state.fullCache = null; render(); });
  document.addEventListener("keydown", onKey);
}

function onKey(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const toggles = { o: "tOverlay", n: "tNumbers", t: "tTrail", l: "tNearest", p: "tPrediction" };
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); step(e.shiftKey ? -state.fps : -1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); step(e.shiftKey ? state.fps : 1); }
  else if (e.key === "ArrowDown") { e.preventDefault(); step(-10 * state.fps); }
  else if (e.key === "ArrowUp") { e.preventDefault(); step(10 * state.fps); }
  else if (e.key === ",") jumpSwitch(-1);
  else if (e.key === ".") jumpSwitch(1);
  else if (e.key === "[" || e.key === "]") {
    const rates = [0.25, 0.5, 1, 2, 4, 8];
    const i = rates.indexOf(state.rate) + (e.key === "]" ? 1 : -1);
    if (rates[i]) setRate(rates[i]);
  } else if (e.key === "g") { e.preventDefault(); $("frameInput").focus(); $("frameInput").select(); }
  else if (e.key === "?") $("help").showModal();
  else if (toggles[e.key.toLowerCase()]) { const box = $(toggles[e.key.toLowerCase()]); box.checked = !box.checked; box.onchange({ target: box }); }
}

function jumpSwitch(direction) {
  const labels = state.labels.A;
  if (!labels) return;
  const code = labels[state.frame];
  let f = state.frame;
  while (f + direction >= 0 && f + direction < state.nFrames && labels[f + direction] === code) f += direction;
  if (direction > 0) f += 1; // first frame of the next run
  else { // start of the previous run
    f -= 1;
    if (f < 0) return;
    const prev = labels[f];
    while (f > 0 && labels[f - 1] === prev) f--;
  }
  step(clampFrame(f) - state.frame);
}

function renderLegend() {
  $("legend").innerHTML = [[0, `H · ${state.meta.teams.H}`], [1, `A · ${state.meta.teams.A}`], [2, "D · dead"], [3, "X"]]
    .map(([c, n]) => `<span><i style="background:${COLORS[c]}"></i>${n.replace(/</g, "&lt;")}</span>`).join("") +
    '<span><i style="background:rgba(255,255,255,.15)"></i>graded</span>';
}

(async function main() {
  try {
    await loadData();
    renderLegend();
    bind();
    await refreshLabelList();
    for (const key of ["A", "B"]) { // restore last session's label choices if the files still exist
      const saved = recall(`labels${key}`);
      const select = $(`labels${key}`);
      if (saved && [...select.options].some((o) => o.value === saved)) {
        select.value = saved;
        await selectLabels(key, saved);
      }
    }
    await setupVideo();
    $("loading").hidden = true;
    seek(Number(recall("frame")) || 0);
    setInterval(() => remember("frame", String(state.frame)), 2000);
  } catch (err) {
    $("loading").textContent = `Could not load viewer data: ${err.message}`;
    console.error(err);
  }
})();
