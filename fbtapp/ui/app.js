"use strict";
const $ = (s, el = document) => el.querySelector(s);
let S = null, meta = null, lastMode = null, tilesKey = "", panelKey = "", order = [];
const view = { yaw: null, pitch: 0.42, dist: 4.4 };

fetch("/api/meta").then(r => r.json()).then(m => (meta = m));
function connect() {
  const es = new EventSource("/events");
  es.onmessage = e => { S = JSON.parse(e.data); render(); };
  es.onerror = () => { es.close(); setTimeout(connect, 1000); };
}
connect();

function post(path, body) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
    .then(r => r.json()).then(r => { if (r && r.error) toast(r.error); return r; })
    .catch(() => toast("Немає зв'язку з програмою"));
}
let toastTimer = null;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), 3500);
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- helpers
function camOrder(st) {
  // Tiles in the order the user sees them standing in front of the table:
  // the phone on the user's left first, the MacBook in the middle.
  const n = st.cams.length, idx = [...Array(n).keys()];
  const mac = idx.find(i => st.cams[i].extra);       // the third camera (MacBook or a phone) stands in the middle
  const phones = idx.filter(i => !st.cams[i].extra);
  const P = st.cam_poses || [];
  if (mac !== undefined && P[mac] && P[0] && P[1] && P[mac].x) {
    const c = P[mac].c, x = P[mac].x;
    const side = i => (P[i].c[0] - c[0]) * x[0] + (P[i].c[1] - c[1]) * x[1] + (P[i].c[2] - c[2]) * x[2];
    phones.sort((a, b) => side(b) - side(a));   // right of the Mac's image = the user's left
  }
  if (mac === undefined) return phones;
  return [phones[0], mac, ...phones.slice(1)].filter(i => i !== undefined);
}
function bat(c) {
  if (c.battery == null || c.battery < 0) return "";
  return `${c.battery}%${c.charging ? " ⚡" : ""}`;
}
// Light a phone has, from the exposure it holds (engine.light_level): 10 / 20 / 30 ms.
const LIGHT = { good: ["світло добре", "ok"], fair: ["світло середнє", "warn"], low: ["мало світла", "bad"] };
function lightText(c) { return c.light && c.state === "online" ? LIGHT[c.light.level][0] : ""; }
function lightTitle(c) { return c.light ? `витримка ${c.light.ms} мс, ISO ${c.light.iso}` : ""; }
function dotClass(c) { return c.state === "online" ? "ok" : c.state === "connecting" ? "warn" : "bad"; }
function when(t) {
  if (!t) return "";
  const d = new Date(t * 1000), now = new Date();
  const hm = d.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" });
  const day = 864e5, a = new Date(d.getFullYear(), d.getMonth(), d.getDate()), b = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const diff = Math.round((b - a) / day);
  if (diff === 0) return `сьогодні о ${hm}`;
  if (diff === 1) return `учора о ${hm}`;
  return `${d.toLocaleDateString("uk-UA", { day: "2-digit", month: "2-digit" })} о ${hm}`;
}
function camWord(n) { return n === 1 ? "камера" : n < 5 ? "камери" : "камер"; }
function offlineHelp(c) {
  if (c.mac) {
    if (/authori|permission|denied|not permitted|доступ/i.test(c.error)) return "Немає дозволу на камеру. Дай доступ FBT у Системних параметрах → Конфіденційність → Камера.";
    return "Камера MacBook недоступна";
  }
  return "Відкрий на телефоні FBT Cam і перевір Wi-Fi";
}

// ---------------------------------------------------------------- render
function render() {
  if (!S) return;
  const game = S.mode === "server", asleep = S.mode === "sleep";
  $("#setup").classList.toggle("hidden", game || asleep);
  $("#game").classList.toggle("hidden", !game);
  $("#sleep").classList.toggle("hidden", !asleep);
  if (S.mode !== lastMode) {
    tilesKey = ""; panelKey = ""; lastMode = S.mode;
    if (asleep) { const b = $("#wakeBtn"); b.disabled = false; b.textContent = "Увімкнути камери"; b.focus(); }
  }
  if (asleep) {
    $("#sleepWhy").textContent = S.slept_auto
      ? "5 хвилин перед камерами нікого не було, тож я їх вимкнув. MacBook відпочиває."
      : "MacBook відпочиває, програма лишається відкритою.";
    return;
  }
  order = camOrder(S);
  if (game) renderGame(); else renderSetup();
}

function renderSetup() {
  // chips
  const chips = order.map(i => {
    const c = S.cams[i];
    return `<span class="chip"><span class="dot ${dotClass(c)}"></span>${esc(c.name)}${bat(c) ? ` <span class="bat">${bat(c)}</span>` : ""}</span>`;
  });
  chips.push(`<span class="chip"><span class="dot ${S.headset == null ? "" : S.headset ? "ok" : "bad"}"></span>Шолом${S.hmd && S.hmd.ok ? ' <span class="bat">дані ✓</span>' : ""}</span>`);
  const mb = S.mac_battery;
  if (mb && !mb.charging && mb.percent <= 30) chips.push(`<span class="chip"><span class="dot ${mb.percent <= 15 ? "bad" : "warn"}"></span>Mac ${mb.percent}% — підключи зарядку</span>`);
  if (S.thermal >= 2) chips.push(`<span class="chip"><span class="dot ${S.thermal >= 3 ? "bad" : "warn"}"></span>Mac гарячий</span>`);
  const ch = chips.join("");
  if ($("#chips").innerHTML !== ch) $("#chips").innerHTML = ch;

  // video tiles
  const key = order.map(i => S.cams[i].host).join("|");
  if (key !== tilesKey) {
    tilesKey = key;
    $("#videos").innerHTML = order.map(i => `
      <div class="tile ${S.cams[i].mac ? "mac" : "phone"}" data-i="${i}">
        <img alt="">
        <div class="label"><span class="dot"></span><span class="nm"></span><span class="bat"></span><span class="lt"></span></div>
        <div class="nosee hidden">не бачу тебе</div>
        <div class="off"><b></b><span></span></div>
      </div>`).join("");
  }
  for (const el of document.querySelectorAll(".tile")) {
    const i = +el.dataset.i, c = S.cams[i];
    if (!c) continue;
    const img = $("img", el), on = c.state === "online";
    if (on && img.dataset.on !== "1") { img.src = `/video/${i}?t=${Date.now()}`; img.dataset.on = "1"; }
    if (!on && img.dataset.on === "1") { img.removeAttribute("src"); img.dataset.on = "0"; }
    $(".off", el).classList.toggle("hidden", on);
    $(".off b", el).textContent = c.state === "connecting" ? "Підключаюсь…" : "Не в мережі";
    $(".off span", el).textContent = c.state === "connecting" ? c.name : offlineHelp(c);
    $(".label .dot", el).className = `dot ${dotClass(c)}`;
    $(".label .nm", el).textContent = c.name;
    $(".label .bat", el).textContent = bat(c);
    const lt = $(".label .lt", el);
    lt.textContent = lightText(c); lt.className = `lt ${c.light ? LIGHT[c.light.level][1] : ""}`; lt.title = lightTitle(c);
    el.classList.toggle("person", on && c.person);
    $(".nosee", el).classList.toggle("hidden", !on || c.person);
  }
  renderPanel();
}

function renderPanel() {
  const cal = S.calib, st = cal.state, p = $("#panel");
  const phonesOn = S.cams.slice(0, 2).every(c => c.state === "online");
  const k = `${st}|${phonesOn}|${S.calibrated}`;
  if (k !== panelKey) {
    panelKey = k;
    if (st === "countdown") {
      p.innerHTML = `<div class="count" id="cnt"></div><div class="timer-text" style="text-align:center"><p>Приготуйся — стань у центр</p></div>`;
    } else if (st === "recording") {
      p.innerHTML = `
        <div class="big-timer">
          <div class="ring"><svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" stroke="var(--line)" stroke-width="8" fill="none"/>
            <circle id="arc" cx="50" cy="50" r="44" stroke="var(--accent)" stroke-width="8" fill="none" stroke-linecap="round" stroke-dasharray="276.5" stroke-dashoffset="0"/></svg>
            <div class="num" id="cnt"></div></div>
          <div class="timer-text"><h3>Рухайся</h3><p>Ходи, присідай, піднімай руки, підходь ближче й далі</p></div>
        </div>
        <div class="actions"><button class="btn ghost" id="bCancel">Скасувати</button></div>`;
      $("#bCancel").onclick = () => post("/api/calibrate/cancel");
    } else if (st === "computing") {
      p.innerHTML = `<div style="display:flex;align-items:center;gap:18px"><div class="spin"></div><div class="timer-text"><h3>Рахую калібрування…</h3><p>Кілька секунд</p></div></div>`;
    } else if (st === "done") {
      const r = cal.result || {};
      const xn = esc(r.extra_name || "MacBook");
      const mac = r.mac ? `${xn}: ${r.mac.px.toFixed(1)} px` : r.mac_error ? `${xn}: ${esc(r.mac_error)}` : "";
      p.innerHTML = `<div class="result"><h3><span class="good">✓</span> Готово</h3>
        <p class="q">Точність: <b>${esc(r.quality)}</b></p>
        <p>Телефони: ${r.phones_px.toFixed(1)} px · між ними ${r.baseline_m.toFixed(2)} м, кут ${Math.round(r.angle_deg)}°${mac ? "<br>" + mac : ""}</p></div>
        <div class="actions"><button class="btn primary" id="bOk">Добре</button></div>`;
      $("#bOk").onclick = () => post("/api/calibrate/dismiss");
    } else if (st === "failed") {
      p.innerHTML = `<div class="result"><h3><span class="badc">✗</span> Не вдалося</h3><p>${esc(cal.message)}</p></div>
        <div class="actions"><button class="btn ghost" id="bClose">Закрити</button><button class="btn primary" id="bRetry">Спробувати ще</button></div>`;
      $("#bClose").onclick = () => post("/api/calibrate/dismiss");
      $("#bRetry").onclick = () => post("/api/calibrate");
    } else {
      p.innerHTML = `<div class="calib-line" id="cline"></div>
        <div class="actions">
          <button class="btn" id="bCal">Калібрувати</button>
          <button class="btn primary" id="bStart">Почати гру</button>
        </div>
        <div class="hint" id="hint"></div>`;
      $("#bCal").onclick = () => post("/api/calibrate");
      $("#bStart").onclick = () => post("/api/server/start");
    }
  }
  if (st === "countdown" || st === "recording") {
    const left = cal.left ?? 0;
    $("#cnt").textContent = Math.ceil(left);
    const arc = $("#arc");
    if (arc) arc.setAttribute("stroke-dashoffset", String(276.5 * (1 - left / (cal.seconds || 30))));
  }
  if (st === "idle") {
    const ci = S.calib_info, cl = $("#cline");
    if (S.calibrated) {
      cl.className = "calib-line";
      cl.innerHTML = `Калібровано ${esc(when(ci.time))} · точність <b>${esc(ci.quality || "—")}</b> · ${ci.n} ${camWord(ci.n)}`;
    } else {
      cl.className = "calib-line none";
      cl.innerHTML = `<b>Ще не калібровано.</b> Натисни «Калібрувати» і рухайся ${S.settings ? 30 : 30} секунд.`;
    }
    $("#bCal").disabled = !phonesOn;
    $("#bStart").disabled = !phonesOn || !S.calibrated;
    const h = $("#hint");
    const off = S.cams.filter(c => c.state !== "online" && !c.mac).map(c => c.name);
    const seen = S.cams.filter(c => c.state === "online" && c.person);
    const unseen = S.cams.filter(c => c.state === "online" && !c.person).map(c => c.name);
    const lightOf = lv => S.cams.filter(c => c.state === "online" && c.light && c.light.level === lv).map(c => c.name);
    let txt = "", muted = false;
    if (off.length) txt = `Не в мережі: ${off.join(", ")} — відкрий FBT Cam на телефоні`;
    else if (S.cams.some(c => c.size_changed)) txt = `Змінилась роздільність: ${S.cams.filter(c => c.size_changed).map(c => c.name).join(", ")} — відкалібруй ще раз`;
    else if (S.calibrated && S.tracking.mismatch) txt = "Схоже, камери зрушили — відкалібруй ще раз";
    else if (lightOf("low").length) txt = `Мало світла: ${lightOf("low").join(", ")} — увімкни більше світла, інакше ноги в русі змазуються й тремтять`;
    else if (seen.length && unseen.length) txt = `Тебе не видно: ${unseen.join(", ")}`;
    else if (lightOf("fair").length) txt = `Світла можна більше: ${lightOf("fair").join(", ")} — буде менше тремтіння`;
    else { txt = "У грі стань у Т-позу на секунду (зручно разом з Calibrate FBT у VRChat) — трекери вирівняються"; muted = true; }
    h.textContent = txt; h.classList.toggle("muted", muted);
  }
}

function renderGame() {
  const t = S.tracking, phones = S.cams.filter(c => !c.extra);
  const offPhones = phones.filter(c => c.state !== "online").map(c => c.name);
  let cls = "ok", text = "Працює", sub = "", subOk = false;
  if (offPhones.length) { cls = "bad"; text = "Камера відключилась"; sub = offPhones.join(", "); }
  else if (!t.bones_ready) { cls = "info"; text = "Стань рівно"; sub = `Вчу довжини кісток… ${Math.round(t.learn * 100)}%`; }
  else if (!t.visible) { cls = "warn"; text = "Не бачу тебе"; sub = "Стань перед камерами"; }
  else if (S.mac_battery && !S.mac_battery.charging && S.mac_battery.percent <= 15) sub = `Mac розряджається (${S.mac_battery.percent}%) — підключи зарядку, інакше трекінг гальмуватиме`;
  else if (S.hmd && S.hmd.ok && S.settings.hmd_anchor && !t.align_ready) sub = "Стань у Т-позу на секунду — прив'яжу трекери до шолома";
  else if (S.hmd && S.hmd.ok && S.settings.hmd_anchor && t.align_lost) sub = "Прив'язка до шолома збилась — стань у Т-позу на секунду";
  else if (t.anchor === "hmd") { if (t.yaw_ago != null && t.yaw_ago < 4) { sub = "Прив'язано до шолома ✓"; subOk = true; } }
  else if (t.yaw_ago == null) sub = "Стань у Т-позу на секунду — вирівняю трекери";
  else if (t.yaw_ago < 4) { sub = "Поворот вирівняно ✓"; subOk = true; }
  const gs = $("#gStatus");
  gs.className = `g-status ${cls}`;
  $(".g-text", gs).textContent = text;
  const sb = $("#gSub"); sb.textContent = sub; sb.className = `g-sub${subOk ? " ok" : ""}`;

  const cards = order.map(i => {
    const c = S.cams[i], on = c.state === "online";
    const b = c.battery != null && c.battery >= 0
      ? `<div class="b${c.battery <= 15 && !c.charging ? " low" : ""}">${c.battery}%${c.charging ? "<small>⚡ заряджається</small>" : ""}</div>` : "";
    return `<div class="g-card${on ? "" : " bad"}"><div class="n"><span class="dot ${dotClass(c)}"></span>${esc(c.name)}</div>
      <div class="s">${on ? "на зв'язку" : "немає зв'язку"}</div>${b}${c.hot ? `<div class="w">телефон гарячий</div>` : ""}${on && c.light && c.light.level !== "good" ? `<div class="w" title="${esc(lightTitle(c))}">${lightText(c)}</div>` : ""}</div>`;
  });
  const hs = S.headset;
  cards.push(`<div class="g-card${hs === false ? " bad" : ""}"><div class="n"><span class="dot ${hs == null ? "" : hs ? "ok" : "bad"}"></span>Шолом</div>
    <div class="s">${hs == null ? "перевіряю…" : hs ? "у мережі" : "не відповідає"}</div>${S.hmd && S.hmd.ok ? '<div class="s" style="color:var(--ok)">дані шолома ✓</div>' : S.hmd && S.hmd.queries ? '<div class="w">дані шолома не йдуть</div>' : ""}</div>`);
  const ch = cards.join("");
  if ($("#gCards").innerHTML !== ch) $("#gCards").innerHTML = ch;

  const fpsCls = t.fps >= 25 ? "" : t.fps >= 15 ? "warn" : "bad";
  const parts = [`Трекінг <b class="${fpsCls}">${Math.round(t.fps)} к/с</b>`];
  parts.push(t.anchor === "hmd" ? `Прив'язка <b>до шолома</b>` : `Прив'язка <b>за камерами</b>`);
  if (S.thermal != null && S.thermal >= 0) {
    parts.push(S.thermal >= 3 ? `Mac <span class="bad">перегрівся</span>` : S.thermal === 2 ? `Mac <span class="warn">нагрівся</span>` : `Mac <b>у нормі</b>`);
  }
  if (S.mac_battery && !S.mac_battery.charging) {
    const p = S.mac_battery.percent;
    parts.push(`Mac на батареї <span class="${p <= 15 ? "bad" : p <= 30 ? "warn" : ""}">${p}%</span>`);
  }
  if (t.uptime > 60) parts.push(`Граєш <b>${Math.floor(t.uptime / 60)} хв</b>`);
  const info = parts.join("");
  if ($("#gInfo").innerHTML !== info) $("#gInfo").innerHTML = info;
}
$("#btnStop").onclick = () => post("/api/server/stop");
$("#sleepBtn").onclick = () => post("/api/sleep");
$("#wakeBtn").onclick = () => { const b = $("#wakeBtn"); b.disabled = true; b.textContent = "Вмикаю…"; post("/api/wake"); };

// ---------------------------------------------------------------- 3D view
const cv = $("#render"), ctx = cv.getContext("2d");
let drag = null;
cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  view.yaw -= (e.clientX - drag.x) * 0.008;
  view.pitch = Math.min(1.45, Math.max(0.02, view.pitch + (e.clientY - drag.y) * 0.006));
  drag = { x: e.clientX, y: e.clientY };
});
cv.addEventListener("pointerup", () => (drag = null));
cv.addEventListener("wheel", e => { e.preventDefault(); view.dist = Math.min(10, Math.max(2, view.dist * (1 + e.deltaY * 0.001))); }, { passive: false });

const COL = { left: "#3caaff", right: "#ffa03c", mid: "#78e678" };
function draw3d() {
  requestAnimationFrame(draw3d);
  if (!S || S.mode !== "setup") return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const P = S.cam_poses || [];
  let cx = 0, cz = 0;
  for (const c of P) { cx += c.c[0] / Math.max(P.length, 1); cz += c.c[2] / Math.max(P.length, 1); }
  if (view.yaw === null && P.length) view.yaw = Math.atan2(-cx, -cz);   // start behind the player, facing the cameras
  const yaw = view.yaw ?? 0.6, tgt = [cx * 0.45, 0.75, cz * 0.45];
  const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch), f = H * 1.05;
  const pr = p => {
    const dx = p[0] - tgt[0], dy = p[1] - tgt[1], dz = p[2] - tgt[2];
    const x = cyw * dx - syw * dz, z = syw * dx + cyw * dz;
    const y2 = cp * dy - sp * z, z2 = sp * dy + cp * z, d = view.dist - z2;
    return d < 0.2 ? null : [W / 2 + f * x / d, H / 2 - f * y2 / d];
  };
  const line = (a, b, col, w) => { const A = pr(a), B = pr(b); if (!A || !B) return; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); };
  // floor grid
  ctx.lineCap = "round";
  for (let v = -3; v <= 3.001; v += 0.5) {
    const strong = Math.abs(v) < 1e-6;
    line([tgt[0] + v, 0, tgt[2] - 3], [tgt[0] + v, 0, tgt[2] + 3], strong ? "#353b45" : "#252a31", 1);
    line([tgt[0] - 3, 0, tgt[2] + v], [tgt[0] + 3, 0, tgt[2] + v], strong ? "#353b45" : "#252a31", 1);
  }
  // cameras
  ctx.font = "12px -apple-system, sans-serif";
  P.forEach((c, i) => {
    const tip = [c.c[0] + c.f[0] * 0.35, c.c[1] + c.f[1] * 0.35, c.c[2] + c.f[2] * 0.35];
    line(c.c, tip, "#8b93a1", 2);
    const A = pr(c.c); if (!A) return;
    ctx.fillStyle = "#e9ecf1"; ctx.beginPath(); ctx.arc(A[0], A[1], 5, 0, 7); ctx.fill();
    const nm = S.cams[i] ? S.cams[i].name : `cam${i}`;
    ctx.fillStyle = "#8b93a1"; ctx.fillText(nm, A[0] + 8, A[1] - 8);
  });
  // skeleton
  const sk = S.skeleton, va = S.valid;
  if (sk && meta) {
    meta.bones.forEach(([a, b]) => {
      if (!va[a] || !va[b]) return;
      const na = meta.names[a], nb = meta.names[b];
      const col = na.startsWith("left") && nb.startsWith("left") ? COL.left : na.startsWith("right") && nb.startsWith("right") ? COL.right : COL.mid;
      line(sk[a], sk[b], col, 4);
    });
    ctx.fillStyle = "#fff";
    sk.forEach((p, k) => { if (!va[k]) return; const A = pr(p); if (A) { ctx.beginPath(); ctx.arc(A[0], A[1], 2.6, 0, 7); ctx.fill(); } });
    // shadow on the floor helps judge where the player stands
    const hip = sk[19]; if (hip) { const A = pr([hip[0], 0, hip[2]]); if (A) { ctx.fillStyle = "rgba(61,220,132,.18)"; ctx.beginPath(); ctx.ellipse(A[0], A[1], 26, 9, 0, 0, 7); ctx.fill(); } }
  } else {
    ctx.fillStyle = "#5b6370"; ctx.font = "15px -apple-system, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(S.calibrated ? "Стань перед камерами — тут з'явиться 3D" : "Після калібрування тут буде 3D", W / 2, H / 2);
    ctx.textAlign = "left";
  }
}
draw3d();

// ---------------------------------------------------------------- settings
const SLIDERS = [
  ["smooth_hz", "Плавність тіла", "менше — плавніше, але з запізненням", 0.5, 6, 0.1, "Гц"],
  ["predict_ms", "Передбачення руху тіла", "ховає затримку, але може «перелітати»", 0, 150, 10, "мс"],
  ["head_offset_cm", "Трекери вперед / назад", "якщо кубики в грі зсунуті від тіла", -20, 20, 1, "см"],
  ["head_predict_ms", "Передбачення голови", "щоб ноги не відставали при нахилах", 0, 300, 10, "мс"],
  ["head_smooth_hz", "Плавність голови", "менше — плавніше", 0.5, 6, 0.1, "Гц"],
  ["yaw_trim", "Поправка повороту", "якщо тіло в грі трохи розвернуте", -30, 30, 1, "°"],
  ["extra_fps", "Частота третьої камери", "телефон на ПК: 30 (від 25 його вага як у телефонів); MacBook: більше — точніше, але Mac сильніше гріється", 5, 30, 5, "к/с"],
];
const CHECKS = ["hmd_anchor", "track_chest", "track_knees", "track_elbows", "foot_flat", "use_mac", "auto_sleep", "voice_alerts"];
$("#sliders").innerHTML = SLIDERS.map(([k, t, hint, mn, mx, st, u]) => `
  <div class="slider" data-k="${k}">
    <div class="t"><b>${t}</b><div class="num"><input type="number" min="${mn}" max="${mx}" step="${st}" aria-label="${t}, ${u}"><small>${u}</small></div></div>
    <input type="range" min="${mn}" max="${mx}" step="${st}" aria-label="${t}">
    <div class="h">${hint}</div>
  </div>`).join("");
for (const el of document.querySelectorAll(".slider")) {
  const [r, n] = el.querySelectorAll("input");
  r.oninput = () => (n.value = r.value);
  n.oninput = () => (r.value = n.value);
}
function setSlider(k, v) {
  const el = document.querySelector(`.slider[data-k="${k}"]`); if (!el) return;
  const [r, n] = el.querySelectorAll("input"); r.value = v; n.value = v;
}
const modal = $("#settings"), form = $("#setForm");
$("#gear").onclick = () => {
  if (!S) return;
  const s = S.settings;
  form.height_m.value = s.height_m; form.quest_ip.value = s.quest_ip;
  form.cam0.value = s.cameras[0] || ""; form.cam1.value = s.cameras[1] || "";
  for (const k of CHECKS) form[k].checked = !!s[k];
  for (const [k] of SLIDERS) setSlider(k, s[k]);
  modal.classList.remove("hidden");
};
$("#setReset").onclick = () => { for (const [k] of SLIDERS) setSlider(k, S.defaults[k]); };
$("#setCancel").onclick = () => modal.classList.add("hidden");
document.addEventListener("keydown", e => { if (e.key === "Escape") modal.classList.add("hidden"); });
form.onsubmit = e => {
  e.preventDefault();
  const cams = [...S.settings.cameras];
  cams[0] = form.cam0.value.trim(); cams[1] = form.cam1.value.trim();
  const body = { height_m: parseFloat(form.height_m.value) || 1.73, quest_ip: form.quest_ip.value.trim(), cameras: cams };
  for (const k of CHECKS) body[k] = form[k].checked;
  for (const [k, , , mn, mx] of SLIDERS) {
    const v = parseFloat(document.querySelector(`.slider[data-k="${k}"] input[type=number]`).value);
    body[k] = isNaN(v) ? S.settings[k] : Math.min(mx, Math.max(mn, v));
  }
  post("/api/settings", body).then(() => { modal.classList.add("hidden"); toast("Збережено"); });
};
