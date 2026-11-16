// Replays of past games: pick a game, press play, scrub the timeline.
// Data: /api/replay/<name> = "FBTR1\n" + u32 header length + JSON header + pad,
// then float32 times (s from start) and int16 joints in millimetres (n x 26 x 3).
"use strict";
const $ = s => document.querySelector(s);
const cv = $("#view"), ctx = cv.getContext("2d");
const lc = $("#line"), lx = lc.getContext("2d");
const COL = { left: "#3caaff", right: "#ffa03c", mid: "#78e678" };
const SPEEDS = [1, 2, 5, 10];

let G = null;                 // the loaded game: {h, t, p}
let playing = false, speed = 1, pos = 0, lastNow = null;
const view = { yaw: null, pitch: 0.32, dist: 4.2, tgt: null };

// ---------------------------------------------------------------- games
function fmt(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${x}` : `${m}:${x}`;
}
function dayWord(d) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((today - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 864e5);
  return diff === 0 ? "Сьогодні" : diff === 1 ? "Вчора" : d.toLocaleDateString("uk-UA", { day: "numeric", month: "short" });
}
function hm(d) { return d.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }); }

async function loadList() {
  const list = await (await fetch("/api/sessions")).json();
  const box = $("#games");
  if (!list.length) { msg("Тут з'являться твої ігри.<br><small>Зберігаються останні 5.</small>"); return; }
  box.innerHTML = list.map(s => {
    const d = new Date(s.start);
    return `<button class="chip game" data-n="${s.name}">${dayWord(d)} ${hm(d)} <span class="bat">· ${Math.max(1, Math.round(s.duration_s / 60))} хв</span></button>`;
  }).join("");
  for (const b of box.querySelectorAll(".game")) b.onclick = () => open(b.dataset.n);
  open(list[0].name);
}

async function open(name) {
  for (const b of document.querySelectorAll(".game")) b.classList.toggle("on", b.dataset.n === name);
  setPlaying(false);
  G = null;
  msg("Готую запис…");
  try {
    const r = await fetch(`/api/replay/${name}`);
    if (!r.ok) throw new Error(await r.text());
    const buf = await r.arrayBuffer();
    const dv = new DataView(buf), hl = dv.getUint32(6, true);
    const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 10, hl)));
    const off = 10 + hl + ((4 - ((10 + hl) % 4)) % 4);
    G = { h, t: new Float32Array(buf, off, h.n), p: new Int16Array(buf, off + h.n * 4, h.n * 26 * 3) };
    G.act = smoothAct(h.activity);
    pos = 0; view.tgt = null;
    if (h.cameras.length) {
      let cx = 0, cz = 0;
      for (const c of h.cameras) { cx += c.c[0] / h.cameras.length; cz += c.c[2] / h.cameras.length; }
      view.yaw = Math.atan2(-cx, -cz);            // start behind the player, facing the cameras
    } else view.yaw = 0.6;
    msg("");
    setPlaying(true);
  } catch (e) {
    msg("Не вдалося відкрити запис");
  }
}
function msg(html) { const m = $("#msg"); m.innerHTML = html; m.classList.toggle("hidden", !html); }
function smoothAct(a) {
  const s = a.slice().sort((x, y) => x - y), top = s[Math.floor(s.length * 0.97)] || 1;
  return a.map(v => Math.min(1, v / top));
}

// ---------------------------------------------------------------- playback
function setPlaying(v) {
  playing = v && !!G;
  if (G && playing && pos >= G.h.duration_s - 0.05) pos = 0;
  $("#icPlay").classList.toggle("hidden", playing);
  $("#icPause").classList.toggle("hidden", !playing);
  lastNow = null;
}
$("#play").onclick = () => setPlaying(!playing);
$("#speed").onclick = () => { speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; $("#speed").textContent = speed + "×"; };
addEventListener("keydown", e => {
  if (!G) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
  else if (e.code === "ArrowRight") pos = Math.min(G.h.duration_s, pos + (e.shiftKey ? 30 : 5));
  else if (e.code === "ArrowLeft") pos = Math.max(0, pos - (e.shiftKey ? 30 : 5));
});

function frameAt(t) {         // index i with G.t[i] <= t < G.t[i+1]
  const T = G.t; let lo = 0, hi = T.length - 1;
  if (t <= T[0]) return 0;
  if (t >= T[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
  return lo;
}
function joints(t) {          // 26 x [x,y,z] in metres, interpolated between frames
  const i = frameAt(t), j = Math.min(i + 1, G.t.length - 1);
  const gap = G.t[j] - G.t[i], a = gap > 0 && gap < 0.4 ? (t - G.t[i]) / gap : 0;
  const P = G.p, out = [];
  for (let k = 0; k < 26; k++) {
    const o = (i * 26 + k) * 3, q = (j * 26 + k) * 3;
    out.push([0, 1, 2].map(c => (P[o + c] + (P[q + c] - P[o + c]) * a) / 1000));
  }
  return out;
}

// ---------------------------------------------------------------- 3D view
let drag = null;
cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  view.yaw -= (e.clientX - drag.x) * 0.008;
  view.pitch = Math.min(1.45, Math.max(0.02, view.pitch + (e.clientY - drag.y) * 0.006));
  drag = { x: e.clientX, y: e.clientY };
});
cv.addEventListener("pointerup", () => (drag = null));
cv.addEventListener("wheel", e => { e.preventDefault(); view.dist = Math.min(10, Math.max(1.5, view.dist * (1 + e.deltaY * 0.001))); }, { passive: false });

function fit(c) {
  const dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  return [W, H];
}

function draw(now) {
  requestAnimationFrame(draw);
  if (G && playing && lastNow !== null) {
    pos += (now - lastNow) / 1000 * speed;
    if (pos >= G.h.duration_s) { pos = G.h.duration_s; setPlaying(false); }
  }
  lastNow = now;
  drawScene();
  drawLine();
}

function drawScene() {
  const [W, H] = fit(cv);
  ctx.clearRect(0, 0, W, H);
  if (!G) return;
  const h = G.h, J = joints(pos);
  const hip = J[19];
  const want = [hip[0], Math.max(0.35, hip[1] * 0.6 + 0.3), hip[2]];   // lower the view when you lie down
  if (!view.tgt) view.tgt = want;
  else for (let c = 0; c < 3; c++) view.tgt[c] += (want[c] - view.tgt[c]) * 0.06;   // follow the player gently
  const tgt = view.tgt, yaw = view.yaw;
  const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch), f = H * 1.05;
  const pr = p => {
    const dx = p[0] - tgt[0], dy = p[1] - tgt[1], dz = p[2] - tgt[2];
    const x = cyw * dx - syw * dz, z = syw * dx + cyw * dz;
    const y2 = cp * dy - sp * z, z2 = sp * dy + cp * z, d = view.dist - z2;
    return d < 0.2 ? null : [W / 2 + f * x / d, H / 2 - f * y2 / d, f / d];
  };
  const line = (a, b, col, w) => { const A = pr(a), B = pr(b); if (!A || !B) return; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
  ctx.lineCap = "round";
  // floor grid around the play area
  const gx = Math.round(tgt[0] * 2) / 2, gz = Math.round(tgt[2] * 2) / 2;
  for (let v = -3; v <= 3.001; v += 0.5) {
    line([gx + v, 0, gz - 3], [gx + v, 0, gz + 3], "#272c35", 1);
    line([gx - 3, 0, gz + v], [gx + 3, 0, gz + v], "#272c35", 1);
  }
  // cameras
  ctx.font = "12px -apple-system, sans-serif";
  for (const c of h.cameras) {
    line(c.c, [c.c[0] + c.f[0] * 0.3, c.c[1] + c.f[1] * 0.3, c.c[2] + c.f[2] * 0.3], "#5b6370", 2);
    const A = pr(c.c); if (!A) continue;
    ctx.fillStyle = "#8b93a1"; ctx.beginPath(); ctx.arc(A[0], A[1], 4, 0, 7); ctx.fill();
  }
  // footsteps: where the feet were over the last second
  for (const [k, col] of [[15, COL.left], [16, COL.right]]) {
    ctx.strokeStyle = col;
    for (let s = 1.0; s > 0; s -= 0.1) {
      const a = joints(Math.max(0, pos - s)), b = joints(Math.max(0, pos - s + 0.1));
      const A = pr([a[k][0], 0.005, a[k][2]]), B = pr([b[k][0], 0.005, b[k][2]]);
      if (!A || !B) continue;
      ctx.globalAlpha = 0.5 * (1 - s); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  // shadow
  const S = pr([hip[0], 0, hip[2]]);
  if (S) { ctx.fillStyle = "rgba(61,220,132,.16)"; ctx.beginPath(); ctx.ellipse(S[0], S[1], S[2] * 0.28, S[2] * 0.09, 0, 0, 7); ctx.fill(); }
  // body: far bones first, thickness by distance
  const bones = h.bones.map(([a, b]) => {
    const na = h.joints[a], nb = h.joints[b];
    const col = na.startsWith("left") && nb.startsWith("left") ? COL.left : na.startsWith("right") && nb.startsWith("right") ? COL.right : COL.mid;
    const A = pr(J[a]), B = pr(J[b]);
    return A && B ? { A, B, col, d: (A[2] + B[2]) / 2 } : null;
  }).filter(Boolean).sort((x, y) => x.d - y.d);
  for (const b of bones) {
    ctx.strokeStyle = b.col; ctx.lineWidth = Math.max(2, b.d * 0.045);
    ctx.beginPath(); ctx.moveTo(b.A[0], b.A[1]); ctx.lineTo(b.B[0], b.B[1]); ctx.stroke();
  }
  // round head between the ears
  const ear = [0, 1, 2].map(c => (J[3][c] + J[4][c]) / 2), E = pr(ear);
  if (E) {
    ctx.fillStyle = "#e9ecf1"; ctx.beginPath(); ctx.arc(E[0], E[1], E[2] * 0.095, 0, 7); ctx.fill();
  }
  // clock
  const start = new Date(h.start);
  $("#clock").textContent = new Date(start.getTime() + pos * 1000).toLocaleTimeString("uk-UA");
}

// ---------------------------------------------------------------- timeline
let scrub = false;
const seek = e => { if (!G) return; const r = lc.getBoundingClientRect(); pos = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * G.h.duration_s; };
lc.addEventListener("pointerdown", e => { scrub = true; lc.setPointerCapture(e.pointerId); seek(e); });
lc.addEventListener("pointermove", e => { if (scrub) seek(e); });
lc.addEventListener("pointerup", () => (scrub = false));

function drawLine() {
  const [W, H] = fit(lc);
  lx.clearRect(0, 0, W, H);
  if (!G) return;
  const a = G.act, n = a.length, x = pos / G.h.duration_s * W, mid = H / 2;
  // waveform of how much you moved
  for (let px = 0; px < W; px += 3) {
    const i0 = Math.floor(px / W * n), i1 = Math.max(i0 + 1, Math.floor((px + 3) / W * n));
    let m = 0; for (let i = i0; i < i1 && i < n; i++) m = Math.max(m, a[i]);
    const hh = Math.max(1.5, m * (H / 2 - 3));
    lx.fillStyle = px < x ? "#3ddc84" : "#3a414c";
    lx.fillRect(px, mid - hh, 2, hh * 2);
  }
  lx.fillStyle = "#e9ecf1";
  lx.fillRect(Math.round(x) - 1, 2, 2, H - 4);
  $("#time").textContent = `${fmt(pos)} / ${fmt(G.h.duration_s)}`;
}

loadList();
requestAnimationFrame(draw);
