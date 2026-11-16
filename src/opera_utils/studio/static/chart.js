// Small SVG time-series chart with hover read-out. Values come in metres; shown in cm.

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const t = (iso) => Date.parse(iso.length <= 10 ? iso + "T00:00:00Z" : iso);
const fmtDay = (ms) => new Date(ms).toISOString().slice(0, 10);

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/**
 * series: [{ key, label, color, points: [[iso, metres|null]], segments?: [[[iso, m], ...]], dashed, dots }]
 * marks:  [iso] vertical markers (reference-date changes)
 */
export function drawChart(host, { series, marks = [] }) {
  host.innerHTML = "";
  const W = host.clientWidth || 600, H = host.clientHeight || 220;
  const m = { l: 46, r: 12, t: 10, b: 24 };
  const all = series.flatMap((s) => (s.segments ? s.segments.flat() : s.points)).filter((p) => p[1] != null);
  if (!all.length) {
    host.innerHTML = '<p class="empty">No valid data at this point.</p>';
    return;
  }
  const xs = all.map((p) => t(p[0])), ys = all.map((p) => p[1] * 100);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(0, ...ys), y1 = Math.max(0, ...ys);
  const pad = (y1 - y0) * 0.08 || 0.5;
  y0 -= pad; y1 += pad;
  if (x1 === x0) { x0 -= 864e5; x1 += 864e5; }
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, host);
  const grid = el("g", { class: "grid" }, svg);
  const axis = el("g", { class: "axis" }, svg);
  for (const v of niceTicks(y0, y1, 5)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
    el("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, axis).textContent = v.toFixed(Math.abs(v) < 10 && v % 1 ? 1 : 0);
  }
  el("text", { x: 10, y: m.t + 8, "text-anchor": "start" }, axis).textContent = "cm";
  const y0d = new Date(x0).getUTCFullYear(), y1d = new Date(x1).getUTCFullYear();
  const every = Math.max(1, Math.ceil((y1d - y0d + 1) / 8));
  for (let y = y0d; y <= y1d + 1; y += every) {
    const ms = Date.UTC(y, 0, 1);
    if (ms < x0 || ms > x1) continue;
    el("line", { x1: X(ms), x2: X(ms), y1: m.t, y2: H - m.b }, grid);
    el("text", { x: X(ms), y: H - 6, "text-anchor": "middle" }, axis).textContent = y;
  }
  el("line", { class: "zero", x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0) }, svg);
  for (const d of marks) el("line", { class: "refchg", x1: X(t(d)), x2: X(t(d)), y1: m.t, y2: H - m.b }, svg);

  const path = (pts) => pts.filter((p) => p[1] != null).map((p, i) => `${i ? "L" : "M"}${X(t(p[0])).toFixed(1)},${Y(p[1] * 100).toFixed(1)}`).join("");
  for (const s of series) {
    const segs = s.segments || [s.points];
    for (const seg of segs) {
      el("path", { d: path(seg), fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-dasharray": s.dashed ? "4 3" : "none", opacity: s.opacity ?? 1, "vector-effect": "non-scaling-stroke" }, svg);
    }
    if (s.dots) for (const p of s.points) if (p[1] != null) el("circle", { cx: X(t(p[0])), cy: Y(p[1] * 100), r: 1.9, fill: s.color, opacity: s.opacity ?? 1 }, svg);
  }

  // hover
  const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "rgba(255,255,255,.25)", visibility: "hidden" }, svg);
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  host.appendChild(tip);
  const keyed = series.map((s) => ({ s, pts: s.points.filter((p) => p[1] != null).map((p) => [t(p[0]), p[1]]) }));
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const ms = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
    let best = null;
    for (const k of keyed) for (const p of k.pts) if (!best || Math.abs(p[0] - ms) < Math.abs(best - ms)) best = p[0];
    if (best == null) return;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
    const rows = keyed.map((k) => { const p = k.pts.find((q) => q[0] === best); return p ? `<div><span style="color:${k.s.color}">●</span> ${(p[1] * 100).toFixed(2)} cm <span style="opacity:.6">${k.s.short || ""}</span></div>` : ""; }).join("");
    tip.innerHTML = `<div style="opacity:.7">${fmtDay(best)}</div>${rows}`;
    tip.hidden = false;
    const left = (X(best) / W) * r.width;
    tip.style.left = Math.min(left + 12, r.width - tip.offsetWidth - 4) + "px";
    tip.style.top = "8px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

/**
 * Archive timeline: acquisitions per month (bars), cumulative count (line),
 * one tick per acquisition (rug), gaps shaded, reference-date changes dashed.
 * tl: { dates: [YYYY-MM-DD], reference_dates?: [...], gaps?: [[a, b, days]] }
 */
export function drawTimeline(host, tl) {
  host.innerHTML = "";
  const dates = tl.dates || [];
  if (!dates.length) {
    host.innerHTML = '<p class="empty">No granules in this range.</p>';
    return;
  }
  const W = host.clientWidth || 600, H = host.clientHeight || 180;
  const m = { l: 40, r: 40, t: 10, b: 30 };
  const ms = dates.map(t);
  const start = new Date(Math.min(...ms)), end = new Date(Math.max(...ms));
  const x0 = Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1);
  const x1 = Date.UTC(end.getUTCFullYear(), end.getUTCMonth() + 1, 1);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const months = new Map();
  for (const v of ms) {
    const d = new Date(v);
    const k = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
    months.set(k, (months.get(k) || 0) + 1);
  }
  const maxM = Math.max(...months.values());
  const rugH = 10;
  const Yb = (v) => H - m.b - rugH - 4 - (v / maxM) * (H - m.t - m.b - rugH - 4);
  const Yc = (v) => H - m.b - rugH - 4 - (v / ms.length) * (H - m.t - m.b - rugH - 4);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, host);
  const grid = el("g", { class: "grid" }, svg);
  const axis = el("g", { class: "axis" }, svg);

  for (const [a, b] of tl.gaps || []) el("rect", { class: "gapband", x: X(t(a)), y: m.t, width: Math.max(1, X(t(b)) - X(t(a))), height: H - m.t - m.b }, svg);
  for (const v of niceTicks(0, maxM, 3)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Yb(v), y2: Yb(v) }, grid);
    el("text", { x: m.l - 6, y: Yb(v) + 3.5, "text-anchor": "end" }, axis).textContent = v;
  }
  el("text", { x: 4, y: m.t + 8 }, axis).textContent = "/mo";
  el("text", { x: W - 4, y: m.t + 8, "text-anchor": "end" }, axis).textContent = "total";
  el("text", { x: W - 4, y: Yc(ms.length) + 12, "text-anchor": "end" }, axis).textContent = ms.length;
  const yrs = end.getUTCFullYear() - start.getUTCFullYear() + 1;
  const every = Math.max(1, Math.ceil(yrs / 10));
  for (let y = start.getUTCFullYear() + 1; y <= end.getUTCFullYear(); y += every) {
    const v = Date.UTC(y, 0, 1);
    el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, grid);
    el("text", { x: X(v), y: H - 8, "text-anchor": "middle" }, axis).textContent = y;
  }
  const bw = Math.max(1, ((W - m.l - m.r) / ((x1 - x0) / (30.44 * 864e5))) - 1.5);
  for (const [k, n] of months) el("rect", { class: "tl-bar", x: X(k) + 0.75, y: Yb(n), width: bw, height: Yb(0) - Yb(n), rx: 1.5 }, svg).appendChild(Object.assign(document.createElementNS(NS, "title"), { textContent: `${fmtDay(k).slice(0, 7)}: ${n}` }));
  const sorted = [...ms].sort((a, b) => a - b);
  el("path", { class: "cum", d: sorted.map((v, i) => `${i ? "L" : "M"}${X(v).toFixed(1)},${Yc(i + 1).toFixed(1)}`).join(""), "vector-effect": "non-scaling-stroke" }, svg);
  for (const r of tl.reference_dates || []) el("line", { class: "refchg", x1: X(t(r)), x2: X(t(r)), y1: m.t, y2: H - m.b }, svg);
  const ry = H - m.b - rugH + 2;
  for (const v of ms) el("line", { class: "rug", x1: X(v), x2: X(v), y1: ry, y2: ry + rugH - 2 }, svg);
}
