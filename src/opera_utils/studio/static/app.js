import { Draw, areaKm2, bounds, parseArea } from "./draw.js";
import { drawChart, drawTimeline } from "./chart.js";

// ---------------------------------------------------------------- helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const day = (iso) => (iso ? iso.slice(0, 10) : "–");
function fmtBytes(b) {
  if (b == null || !isFinite(b)) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b.toFixed(b < 10 && i ? 1 : 0)} ${u[i]}`;
}
const fmtArea = (km2) => (km2 >= 1000 ? `${(km2 / 1000).toFixed(1)}k km²` : km2 >= 10 ? `${km2.toFixed(0)} km²` : `${km2.toFixed(2)} km²`);
const store = {
  get(k, d) { try { const v = localStorage.getItem("opera-studio:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("opera-studio:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ---------------------------------------------------------------- server
// Served by `opera-utils studio` itself, the API is on the same origin. Hosted as a
// static page (GitHub Pages), it talks to the studio running on the user's machine:
// http://localhost:8765 unless ?server=... or the Connect dialog says otherwise.
const HOSTED = !["localhost", "127.0.0.1", "[::1]", ""].includes(location.hostname) && location.protocol === "https:";
const DEFAULT_SERVER = "http://localhost:8765";
function serverFromUrl() {
  const q = new URLSearchParams(location.search).get("server");
  if (!q) return null;
  try {
    const u = new URL(q);
    // only a studio on this machine: never send requests (or credentials) elsewhere
    return ["localhost", "127.0.0.1", "[::1]"].includes(u.hostname) && u.protocol === "http:" ? u.origin : null;
  } catch { return null; }
}
let SERVER = HOSTED ? serverFromUrl() || localStorage.getItem("opera-studio:server") || DEFAULT_SERVER : "";

async function api(path, { method = "GET", body } = {}) {
  const r = await fetch(SERVER + path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  return data;
}

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "bad" ? 7000 : 4000);
}

// ---------------------------------------------------------------- products
const G = {
  disp: '<path d="M3 17c3 0 3-6 6-6s3 4 6 4 3-8 6-8"/>',
  cslc: '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 10h16M9 5v14"/>',
  nisar: '<circle cx="12" cy="12" r="3"/><path d="M4.5 16.5a9 9 0 0 1 0-9M19.5 7.5a9 9 0 0 1 0 9"/>',
  tropo: '<path d="M7 17a4 4 0 0 1-.5-8A6 6 0 0 1 18 8.5a3.5 3.5 0 0 1-.5 8.5z"/>',
  static: '<path d="M3 19l6-10 4 6 3-4 5 8z"/>',
};
const PRODUCTS = {
  disp: { name: "DISP-S1", short: "DISP", color: "var(--c-disp)", hex: "#2cc2bd", noun: "frame", desc: "Sentinel-1 displacement time series (North America). Cropped per frame, rebased to the first date and re-referenced to one point, as in the DISP portal." },
  cslc: { name: "CSLC-S1", short: "CSLC", color: "var(--c-cslc)", hex: "#50a0e0", noun: "burst", desc: "Sentinel-1 coregistered SLC bursts. Each burst is downloaded and cropped to your area, so only the subset is kept." },
  nisar: { name: "NISAR", short: "NISAR", color: "var(--c-nisar)", hex: "#90d0c0", noun: "track · frame", desc: "NISAR L-band GSLC and RSLC (cropped while downloading) and GUNW interferograms (whole granules)." },
  tropo: { name: "TROPO", short: "TROPO", color: "var(--c-tropo)", hex: "#90d0f0", noun: "day", desc: "OPERA zenith tropospheric delay, global every 6 hours. Cropped to your area plus a margin, up to 10 km height." },
  static: { name: "Static", short: "STATIC", color: "var(--c-static)", hex: "#6fc7a9", noun: "frame / burst", desc: "Static layers, both kinds at once: DISP-S1-STATIC per frame (line-of-sight ENU, DEM, layover/shadow; windowed reads) and CSLC-S1-STATIC per burst (cropped)." },
};

const state = {
  product: store.get("product", "disp"),
  aoi: store.get("aoi", null),
  filters: { direction: "both", nisarKind: "gslc", staticKinds: ["disp", "cslc"], hours: [], ...store.get("filters", {}) },
  opts: store.get("opts", {}),
  result: null,
  selected: new Set(),
  hover: null,
  jobs: [],
  drawerOpen: false,
  auth: null,
  explore: null,
  refPick: null, // DISP: reference lon/lat chosen before running
};

const DEFAULT_OPTS = {
  disp: { reference: "auto", solid_earth: true, ionosphere: false, recommended_mask: false, static: true, keep_downloads: true, workers: 4 },
  cslc: { subset: true, workers: 3 },
  nisar: { subset: true, polarizations: "", workers: 4 },
  tropo: { height_max: 10000, margin_deg: 0.3, workers: 2 },
  static: { layers: ["line_of_sight_enu", "dem", "layover_shadow_mask"], full_frame: false, subset: true },
};
const opts = () => (state.opts[state.product] ||= structuredClone(DEFAULT_OPTS[state.product]));

// ---------------------------------------------------------------- map
// Keyless tile services (CARTO now watermarks requests without an API key).
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
const BASEMAPS = {
  dark: { tiles: [`${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`], attr: "Esri, HERE, Garmin, © OpenStreetMap contributors", maxzoom: 16 },
  light: { tiles: [`${ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`], attr: "Esri, HERE, Garmin, © OpenStreetMap contributors", maxzoom: 16 },
  satellite: { tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`], attr: "Imagery © Esri, Maxar, Earthstar Geographics", maxzoom: 19 },
};
const baseSource = (k) => ({ type: "raster", tiles: BASEMAPS[k].tiles, tileSize: 256, attribution: BASEMAPS[k].attr, maxzoom: BASEMAPS[k].maxzoom });
const bm0 = BASEMAPS[store.get("basemap", "dark")] ? store.get("basemap", "dark") : "dark";
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: { base: baseSource(bm0) },
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#06090f" } }, { id: "base", type: "raster", source: "base", paint: { "raster-saturation": -0.15 } }],
  },
  center: store.get("center", [-98, 38]),
  zoom: store.get("zoom", 3.4),
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "bottom-right");

// Globe / flat map switch (MapLibre 5 projections); the choice is remembered.
class ProjectionControl {
  onAdd(m) {
    this.map = m;
    this.el = document.createElement("div");
    this.el.className = "maplibregl-ctrl maplibregl-ctrl-group proj-ctrl";
    this.btn = document.createElement("button");
    this.btn.type = "button";
    this.btn.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 3.9 5.6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-5.6-3.9-9s1.3-6.4 3.9-9z"/></svg>';
    this.btn.addEventListener("click", () => this.set(this.type === "globe" ? "mercator" : "globe"));
    this.el.appendChild(this.btn);
    this.type = store.get("projection", "mercator") === "globe" ? "globe" : "mercator";
    this._label();
    return this.el;
  }
  onRemove() { this.el.remove(); }
  apply() { try { this.map.setProjection({ type: this.type }); } catch (e) { console.warn("projection", e); } }
  set(type) {
    this.type = type;
    store.set("projection", type);
    this.apply();
    this._label();
    if (type === "globe" && this.map.getZoom() > 4) this.map.easeTo({ zoom: 2.6, duration: 900 });
  }
  _label() {
    const g = this.type === "globe";
    this.btn.classList.toggle("on", g);
    this.btn.title = g ? "Switch to the flat map" : "Switch to the globe";
    this.btn.setAttribute("aria-label", this.btn.title);
    this.btn.setAttribute("aria-pressed", String(g));
  }
}
const projection = new ProjectionControl();
map.addControl(projection, "bottom-right");
map.on("style.load", () => projection.apply());
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
map.on("moveend", () => { store.set("center", map.getCenter().toArray()); store.set("zoom", map.getZoom()); });

let draw;
map.on("load", () => {
  map.addSource("foot", { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "id" });
  const dirColor = ["match", ["get", "direction"], "ascending", "#90d0f0", "descending", "#f2a27e", "#2cc2bd"];
  map.addLayer({ id: "foot-fill", type: "fill", source: "foot", paint: { "fill-color": dirColor, "fill-opacity": ["case", ["boolean", ["feature-state", "sel"], false], 0.2, ["boolean", ["feature-state", "hover"], false], 0.14, 0.03] } });
  map.addLayer({ id: "foot-line", type: "line", source: "foot", paint: { "line-color": dirColor, "line-width": ["case", ["boolean", ["feature-state", "sel"], false], 2.2, ["boolean", ["feature-state", "hover"], false], 2, 1], "line-opacity": ["case", ["boolean", ["feature-state", "sel"], false], 1, 0.55] } });
  map.addSource("vel", { type: "image", url: "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==", coordinates: [[0, 0.001], [0.001, 0.001], [0.001, 0], [0, 0]] });
  map.addLayer({ id: "vel", type: "raster", source: "vel", layout: { visibility: "none" }, paint: { "raster-opacity": 0.85, "raster-fade-duration": 0, "raster-resampling": "nearest" } });

  draw = new Draw(map, {
    color: "#2cc2bd",
    onHint: (h) => { const el = $("#drawHint"); el.hidden = !h; if (h) el.innerHTML = h; $$(".tool[data-draw]").forEach((b) => b.classList.toggle("on", !!h && b.dataset.draw === draw?.mode)); },
    onArea: (g) => { if (g) setAoi(g, false); },
  });
  if (state.aoi) draw.setArea(state.aoi);

  map.on("mousemove", "foot-fill", (e) => { if (!draw.active && !state.explore) setHover(e.features[0]?.properties?.id); });
  map.on("mouseleave", "foot-fill", () => setHover(null));
  map.on("click", (e) => {
    if (draw.active) return;
    if (state.explore) return exploreClick(e.lngLat);
    const f = map.queryRenderedFeatures(e.point, { layers: ["foot-fill"] });
    if (f.length) toggleGroup(f[0].properties.id);
  });
  renderAll();
});

function setHover(id) {
  if (state.hover === id) return;
  if (state.hover != null) map.setFeatureState({ source: "foot", id: state.hover }, { hover: false });
  state.hover = id;
  if (id != null) map.setFeatureState({ source: "foot", id }, { hover: true });
  $$(".res").forEach((el) => el.classList.toggle("hover", el.dataset.id === id));
  map.getCanvas().style.cursor = id ? "pointer" : "";
}

function fitTo(g, pad = 80) {
  const [w, s, e, n] = bounds(g);
  map.fitBounds([[w, s], [e, n]], { padding: { top: pad, bottom: pad, left: 420 + pad / 2, right: pad }, maxZoom: 12, duration: 700 });
}

// ---------------------------------------------------------------- dock rendering
function renderProducts() {
  $("#products").innerHTML = Object.entries(PRODUCTS).map(([k, p]) => `
    <button class="prod ${k === state.product ? "on" : ""}" data-p="${k}" role="tab" aria-selected="${k === state.product}" style="--c:${p.color}">
      <span class="glyph"><svg viewBox="0 0 24 24">${G[k]}</svg></span><b>${p.short}</b>
    </button>`).join("") + `<p class="prod-desc" style="grid-column:1/-1">${esc(PRODUCTS[state.product].desc)}</p>`;

}

function seg(name, value, choices) {
  return `<div class="seg" data-seg="${name}">${choices.map(([v, l]) => `<button data-v="${v}" class="${v === value ? "on" : ""}">${l}</button>`).join("")}</div>`;
}

function renderFilters() {
  const f = state.filters, p = state.product;
  let h = "";
  if (p === "nisar") h += `<div class="opt"><span class="opt-label">Product</span>${seg("nisarKind", f.nisarKind, [["gslc", "GSLC"], ["rslc", "RSLC"], ["gunw", "GUNW"]])}</div>`;
  if (p === "static") {
    const k = new Set(f.staticKinds || []);
    h += `<div class="opt"><span class="opt-label">Static layers of</span><div class="chips" data-static-kinds>${[["disp", "DISP-S1 frames"], ["cslc", "CSLC-S1 bursts"]].map(([v, l]) => `<button data-k="${v}" class="${k.has(v) ? "on" : ""}">${l}</button>`).join("")}</div></div>`;
  }
  if (p !== "tropo") h += `<div class="opt"><span class="opt-label">Orbit direction</span>${seg("direction", f.direction, [["both", "Both"], ["ascending", "Ascending"], ["descending", "Descending"]])}</div>`;
  if (p === "tropo") {
    const hrs = new Set(f.hours || []);
    h += `<div class="opt"><span class="opt-label">Hours (UTC) · none = all four</span><div class="chips" data-hours>${[0, 6, 12, 18].map((x) => `<button data-h="${x}" class="${hrs.has(x) ? "on" : ""}">${String(x).padStart(2, "0")}:00</button>`).join("")}</div></div>`;
  }
  if (p === "static") h += `<p class="opt-note">Static layers do not depend on dates.</p>`;
  $("#filters").innerHTML = h;
}

function renderAoi() {
  const g = state.aoi;
  $("#aoiCard").hidden = !g;
  $("#aoiHint").textContent = g ? "" : "draw on the map";
  if (!g) return;
  const [w, s, e, n] = bounds(g);
  $("#aoiArea").textContent = fmtArea(areaKm2(g));
  $("#aoiBounds").textContent = `${w.toFixed(3)}, ${s.toFixed(3)}, ${e.toFixed(3)}, ${n.toFixed(3)}`;
}

function setAoi(g, fit = true) {
  state.aoi = g;
  store.set("aoi", g);
  draw?.setArea(g);
  renderAoi();
  clearResults();
  if (g && fit) fitTo(g);
  if (g && areaKm2(g) > 40000 && state.product !== "tropo") toast("That is a big area: subsets of whole frames can run to tens of GB.", "");
}

function clearResults() {
  state.result = null;
  state.selected.clear();
  map.getSource("foot")?.setData({ type: "FeatureCollection", features: [] });
  renderResults();
}

function groupMeta(g, p) {
  const bits = [];
  if (p === "static" && g.kind === "disp") bits.push(g.n ? `${(g.layers || []).length} layers` : "not published");
  else bits.push(`<span>${g.n} ${p === "tropo" ? "times" : "granules"}</span>`);
  if (g.first && p !== "static") bits.push(`<span>${day(g.first)} → ${day(g.last)}</span>`);
  if (g.n_reference_dates) bits.push(`<span title="ministacks: reference date changes that rebasing stitches">${g.n_reference_dates} ref. dates</span>`);
  if (g.polarizations?.length) bits.push(`<span>${esc(g.polarizations.join(" "))}</span>`);
  if (g.coverage != null) bits.push(`<span class="cov" title="share of your area inside this footprint"><i><b style="width:${Math.round(g.coverage * 100)}%"></b></i>${Math.round(g.coverage * 100)}%</span>`);
  return bits.join("");
}

function renderResults() {
  const r = state.result;
  $("#resultsStep").hidden = !r;
  $("#optionsStep").hidden = !r || !r.groups.length;
  $("#dockFoot").hidden = !r || !r.groups.length;
  if (!r) return;
  const p = r.product;
  $("#resultHint").textContent = `${r.groups.length} ${PRODUCTS[p].noun}${r.groups.length === 1 ? "" : "s"} · ${r.items.length} files`;
  $("#searchTime").textContent = r.seconds != null ? `${r.seconds}s` : "";
  const MAX = 300;
  const rows = r.groups.slice(0, MAX).map((g) => {
    const dir = g.direction === "ascending" ? "asc" : g.direction === "descending" ? "desc" : "global";
    const size = p === "disp" ? `~${fmtBytes(g.est_bytes)}` : fmtBytes(g.size_bytes);
    return `<div class="res ${dir} ${state.selected.has(g.id) ? "on" : ""} ${g.n ? "" : "empty-group"}" data-id="${esc(g.id)}">
      <input type="checkbox" ${state.selected.has(g.id) ? "checked" : ""} ${g.n ? "" : "disabled"} tabindex="-1" />
      <div class="t">${esc(g.label)} ${p === "static" ? `<span class="kind-tag">${g.kind === "cslc" ? "CSLC" : "DISP"}</span>` : ""}${g.direction && g.direction !== "global" ? `<span class="dir">${g.direction === "ascending" ? "ASC" : "DESC"}</span>` : ""}</div>
      <span class="size" title="${p === "disp" ? "estimated subset size" : "size of the source files"}">${size}</span>
      <div class="m">${groupMeta(g, p)}</div>
    </div>`;
  });
  if (r.groups.length > MAX) rows.push(`<div class="more">+ ${r.groups.length - MAX} more (use Select all, or narrow the area)</div>`);
  if (!r.groups.length) rows.push(`<p class="empty">Nothing found here${p === "disp" ? " (DISP-S1 covers North America)" : ""}. Try a wider date range or another area.</p>`);
  $("#results").innerHTML = rows.join("");
  $("#selectAll").checked = r.groups.length > 0 && r.groups.filter((g) => g.n).every((g) => state.selected.has(g.id));
  for (const g of r.groups) map.setFeatureState({ source: "foot", id: g.id }, { sel: state.selected.has(g.id) });
  renderOptions();
  renderSummary();
}

function renderSummary() {
  const r = state.result;
  if (!r) return;
  const groups = r.groups.filter((g) => state.selected.has(g.id));
  const items = r.items.filter((i) => state.selected.has(i.group));
  const src = items.reduce((s, i) => s + (i.size_bytes || 0), 0);
  const p = r.product;
  let est = "";
  if (p === "disp") est = `<span>est. subset <b>~${fmtBytes(groups.reduce((s, g) => s + (g.est_bytes || 0), 0))}</b></span>`;
  else if (p === "cslc" || p === "static") est = `<span>source <b>${fmtBytes(src)}</b>${opts().subset !== false ? " → cropped" : ""}</span>`;
  else if (p === "nisar") est = `<span>source <b>${fmtBytes(src)}</b>${r.kind !== "gunw" && opts().subset ? " → cropped" : ""}</span>`;
  else if (p === "tropo") est = `<span>source <b>${fmtBytes(src)}</b> → cropped</span>`;
  $("#selSummary").innerHTML = `<span><b>${groups.length}</b> ${PRODUCTS[p].noun}${groups.length === 1 ? "" : "s"} · <b>${items.length}</b> files</span>${est}`;
  $("#runBtn").disabled = !groups.length;
}

function sw(key, label, sub, checked) {
  return `<label class="switch"><span>${label}${sub ? `<small>${sub}</small>` : ""}</span><input type="checkbox" data-opt="${key}" ${checked ? "checked" : ""} /></label>`;
}

function renderOptions() {
  const o = opts(), p = state.product, r = state.result;
  let h = "";
  if (p === "disp") {
    h += `<div class="opt"><span class="opt-label">Reference point</span>${seg("reference", o.reference, [["auto", "Most coherent"], ["point", "Pick on map"], ["high_coherence", "Area median"], ["none", "None"]])}
      <p class="opt-note">${{
        auto: "Like the DISP portal: the most coherent 5×5 spot observed in every epoch. You can move it afterwards in Explore.",
        point: state.refPick ? `Reference near <span class="mono">${state.refPick[0].toFixed(4)}, ${state.refPick[1].toFixed(4)}</span>; it snaps to the best pixel within ~300 m. <a href="#" data-act="repick">Pick again</a>` : `<a href="#" data-act="repick">Click the map</a> to place the reference.`,
        high_coherence: "Each epoch minus the median of high-coherence pixels (reformat_stack's default).",
        none: "No spatial reference: values are as published, rebased in time only.",
      }[o.reference]}</p></div>`;
    h += `<div class="opt">${sw("solid_earth", "Solid-earth tide correction", "applied while rebasing", o.solid_earth)}${sw("ionosphere", "Ionospheric correction", "", o.ionosphere)}${sw("recommended_mask", "Mask with recommended_mask", "when accumulating reference changes", o.recommended_mask)}${sw("static", "Line of sight + layover/shadow", "DISP-S1-STATIC, cropped", o.static)}${sw("keep_downloads", "Keep the NetCDF subsets", "needed for the raw series in Explore", o.keep_downloads)}</div>`;
  } else if (p === "cslc") {
    h += sw("subset", "Crop each burst to the area", "the full burst is downloaded, cropped, then deleted", o.subset ?? true);
  } else if (p === "nisar") {
    if (r?.kind === "gunw") h += `<p class="opt-note">GUNW granules are downloaded whole (≈2–3 GB each).</p>`;
    else h += sw("subset", "Crop to the area while downloading", "remote windowed reads; only the subset is written", o.subset);
    if (r?.kind !== "gunw") h += `<div class="opt"><span class="opt-label">Polarizations (blank = all)</span><input type="text" data-opt-text="polarizations" value="${esc(o.polarizations || "")}" placeholder="HH VV" /></div>`;
  } else if (p === "tropo") {
    h += `<div class="opt dates"><label><span>Max height (m)</span><input type="number" data-opt-num="height_max" value="${o.height_max}" /></label><label><span>Margin (°)</span><input type="number" step="0.1" data-opt-num="margin_deg" value="${o.margin_deg}" /></label></div>`;
  } else if (p === "static") {
    const kinds = new Set((r?.groups || []).map((g) => g.kind));
    if (kinds.has("cslc")) h += `<span class="opt-label" style="margin-top:8px">CSLC-S1-STATIC bursts</span>` + sw("subset", "Crop each burst to the area", "", o.subset ?? true);
    if (kinds.has("disp")) h += `<span class="opt-label" style="margin-top:8px">DISP-S1-STATIC frames</span>`;
    if (!kinds.has("disp")) { $("#options").innerHTML = h; return; }
    const L = [["line_of_sight_enu", "Line of sight (east, north, up)"], ["dem", "DEM"], ["layover_shadow_mask", "Layover / shadow mask"]];
    h += L.map(([k, l]) => `<label class="switch"><span>${l}</span><input type="checkbox" data-layer="${k}" ${o.layers.includes(k) ? "checked" : ""} /></label>`).join("");
    h += sw("full_frame", "Whole frames", "instead of cropping to the area", o.full_frame);
  }
  $("#options").innerHTML = h;
}

function renderAll() {
  renderProducts();
  renderFilters();
  renderAoi();
  renderResults();
}

// ---------------------------------------------------------------- dock events
$("#products").addEventListener("click", (e) => {
  const b = e.target.closest(".prod");
  if (!b || b.dataset.p === state.product) return;
  state.product = b.dataset.p;
  store.set("product", state.product);
  clearResults();
  renderAll();
});

$("#dock").addEventListener("click", (e) => {
  const sb = e.target.closest(".seg[data-seg] button");
  if (sb) {
    const name = sb.parentElement.dataset.seg;
    if (name in state.filters) {
      state.filters[name] = sb.dataset.v;
      store.set("filters", state.filters);
      renderFilters();
      clearResults();
    } else {
      opts()[name] = sb.dataset.v;
      store.set("opts", state.opts);
      if (name === "reference" && sb.dataset.v === "point" && !state.refPick) pickRefBeforeRun();
      if (name === "reference") showRefPickMarker();
      renderOptions();
    }
    return;
  }
  const kb = e.target.closest("[data-static-kinds] button");
  if (kb) {
    const k = new Set(state.filters.staticKinds || []);
    k.has(kb.dataset.k) ? k.delete(kb.dataset.k) : k.add(kb.dataset.k);
    state.filters.staticKinds = [...k];
    store.set("filters", state.filters);
    renderFilters();
    clearResults();
    return;
  }
  const hb = e.target.closest("[data-hours] button");
  if (hb) {
    const h = +hb.dataset.h, s = new Set(state.filters.hours || []);
    s.has(h) ? s.delete(h) : s.add(h);
    state.filters.hours = [...s].sort((a, b) => a - b);
    store.set("filters", state.filters);
    renderFilters();
    clearResults();
    return;
  }
  if (e.target.closest('[data-act="repick"]')) { e.preventDefault(); pickRefBeforeRun(); }
});

$("#dock").addEventListener("change", (e) => {
  const t = e.target;
  if (t.dataset.opt) opts()[t.dataset.opt] = t.checked;
  else if (t.dataset.optText) opts()[t.dataset.optText] = t.value;
  else if (t.dataset.optNum) opts()[t.dataset.optNum] = +t.value;
  else if (t.dataset.layer) {
    const L = new Set(opts().layers);
    t.checked ? L.add(t.dataset.layer) : L.delete(t.dataset.layer);
    opts().layers = [...L];
  } else return;
  store.set("opts", state.opts);
  renderSummary();
});

$("#results").addEventListener("click", (e) => { const r = e.target.closest(".res"); if (r) toggleGroup(r.dataset.id); });
$("#results").addEventListener("mouseover", (e) => { const r = e.target.closest(".res"); setHover(r ? r.dataset.id : null); });
$("#results").addEventListener("mouseleave", () => setHover(null));
$("#results").addEventListener("dblclick", (e) => {
  const r = e.target.closest(".res");
  const f = r && state.result.footprints.features.find((x) => x.properties.id === r.dataset.id);
  if (f) fitTo(f.geometry, 40);
});
$("#selectAll").addEventListener("change", (e) => {
  for (const g of state.result?.groups || []) if (g.n) e.target.checked ? state.selected.add(g.id) : state.selected.delete(g.id);
  renderResults();
});

function toggleGroup(id) {
  const g = state.result?.groups.find((x) => x.id === id);
  if (!g || !g.n) return;
  state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
  renderResults();
}

$$(".tool[data-draw]").forEach((b) => b.addEventListener("click", () => (draw.mode === b.dataset.draw ? draw.cancel() : draw.start(b.dataset.draw))));
$("#clearAoi").addEventListener("click", () => { draw.cancel(false); setAoi(null); });
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, dialog") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "p" || e.key === "P") draw.start("polygon");
  else if (e.key === "r" || e.key === "R") draw.start("rectangle");
  else if (e.key === "j" || e.key === "J") toggleDrawer();
  else if (e.key === "Enter" && !draw.active) search();
});

// dates
const today = () => new Date().toISOString().slice(0, 10);
$("#start").value = store.get("start", "");
$("#end").value = store.get("end", "");
for (const id of ["start", "end"]) $("#" + id).addEventListener("change", (e) => { store.set(id, e.target.value); $$("#dateChips button").forEach((b) => b.classList.remove("on")); clearResults(); });
$("#dateChips").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const r = b.dataset.range;
  const d = new Date();
  if (r === "all") { $("#start").value = ""; $("#end").value = ""; }
  else {
    if (r === "90d") d.setUTCDate(d.getUTCDate() - 90);
    if (r === "1y") d.setUTCFullYear(d.getUTCFullYear() - 1);
    if (r === "3y") d.setUTCFullYear(d.getUTCFullYear() - 3);
    $("#start").value = d.toISOString().slice(0, 10);
    $("#end").value = today();
  }
  store.set("start", $("#start").value);
  store.set("end", $("#end").value);
  $$("#dateChips button").forEach((x) => x.classList.toggle("on", x === b));
  clearResults();
});

// import dialog
$("#importBtn").addEventListener("click", () => { $("#importErr").hidden = true; $("#importDlg").showModal(); $("#importText").focus(); });
$("#importFile").addEventListener("change", async (e) => { const f = e.target.files[0]; if (f) $("#importText").value = await f.text(); });
$("#importDlg").addEventListener("drop", async (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) $("#importText").value = await f.text(); });
$("#importDlg").addEventListener("dragover", (e) => e.preventDefault());
$("#importForm").addEventListener("submit", (e) => {
  if (e.submitter?.value === "cancel") return;
  try {
    const g = parseArea($("#importText").value);
    setAoi(g);
  } catch (err) {
    e.preventDefault();
    $("#importErr").textContent = err.message;
    $("#importErr").hidden = false;
  }
});

// ---------------------------------------------------------------- search
async function search() {
  if (!state.aoi) { toast("Draw or import an area first.", "bad"); return; }
  const btn = $("#searchBtn");
  if (btn.disabled) return;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.querySelector(".btn-label").textContent = "Searching";
  const f = state.filters, p = state.product;
  const options = { direction: f.direction };
  if (p === "nisar") options.kind = f.nisarKind;
  if (p === "static") {
    if (!(f.staticKinds || []).length) { toast("Pick DISP-S1 frames, CSLC-S1 bursts, or both.", "bad"); btn.disabled = false; btn.classList.remove("busy"); btn.querySelector(".btn-label").textContent = "Search"; return; }
    options.kinds = f.staticKinds;
  }
  if (p === "tropo") options.hours = f.hours;
  try {
    const r = await api("/api/search", { method: "POST", body: { product: p, aoi: state.aoi, start: $("#start").value || null, end: $("#end").value || null, options } });
    if (p !== state.product) return;
    state.result = r;
    state.selected = new Set();
    const live = r.groups.filter((g) => g.n);
    if (live.length && live.length <= 4) live.forEach((g) => state.selected.add(g.id));
    map.getSource("foot").setData(r.footprints);
    renderResults();
    if (r.footprints.features.length && p !== "tropo") {
      const all = { type: "MultiPolygon", coordinates: r.footprints.features.flatMap((x) => (x.geometry.type === "Polygon" ? [x.geometry.coordinates] : x.geometry.coordinates)) };
      fitTo(all, 50);
    }
    $("#resultsStep").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.querySelector(".btn-label").textContent = "Search";
  }
}
$("#searchBtn").addEventListener("click", search);

// DISP: choose the reference before running
let refPickMarker;
function showRefPickMarker() {
  refPickMarker?.remove();
  refPickMarker = null;
  if (state.product === "disp" && opts().reference === "point" && state.refPick && !state.explore) {
    refPickMarker = new maplibregl.Marker({ element: refEl() }).setLngLat(state.refPick).addTo(map);
  }
}
function pickRefBeforeRun() {
  draw.pick("Click the reference point (stable ground, e.g. bedrock or a GNSS station)", (ll) => {
    state.refPick = [ll.lng, ll.lat];
    showRefPickMarker();
    renderOptions();
  });
}
function refEl() {
  const d = document.createElement("div");
  d.className = "ref-marker";
  d.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 3l9 16H3z"/></svg>';
  d.title = "Reference point";
  return d;
}

// ---------------------------------------------------------------- run
async function run() {
  const r = state.result;
  if (!r) return;
  if (!state.auth?.signed_in) { toast("Sign in to Earthdata first: downloads need it.", "bad"); openAuth(); return; }
  const o = structuredClone(opts());
  if (r.product === "disp" && o.reference === "point") {
    if (!state.refPick) { toast("Pick the reference point on the map first.", "bad"); pickRefBeforeRun(); return; }
    o.reference_lonlat = state.refPick;
  }
  if (r.product === "nisar" && o.polarizations) o.polarizations = o.polarizations.split(/[\s,]+/).filter(Boolean).map((s) => s.toUpperCase());
  else if (r.product === "nisar") o.polarizations = null;
  const groups = r.groups.filter((g) => state.selected.has(g.id));
  const btn = $("#runBtn");
  btn.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: {
        product: r.product,
        kind: r.kind,
        aoi: state.aoi,
        start: $("#start").value || null,
        end: $("#end").value || null,
        groups: groups.map((g) => g.id),
        group_info: groups.map((g) => ({ id: g.id, direction: g.direction, label: g.label })),
        items: r.items.filter((i) => state.selected.has(i.group)),
        footprints: r.footprints.features.filter((f) => state.selected.has(f.properties.id)),
        options: o,
      },
    });
    toast(`Job started: ${job.title}`, "ok");
    await refreshJobs();
    toggleDrawer(true);
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    btn.disabled = false;
  }
}
$("#runBtn").addEventListener("click", run);

// ---------------------------------------------------------------- auth
async function refreshAuth() {
  try { state.auth = await api("/api/auth"); } catch { state.auth = null; }
  const a = state.auth;
  $("#authChip").classList.toggle("ok", !!a?.signed_in);
  $("#authText").textContent = a?.signed_in ? a.username : "Sign in to Earthdata";
  $("#authChip").title = a?.signed_in ? `Earthdata: ${a.label}` : "Downloads need an Earthdata Login";
}
function openAuth() {
  const a = state.auth;
  $("#authErr").hidden = true;
  $("#authNow").innerHTML = a?.signed_in
    ? `Using <b>${esc(a.username)}</b> · ${esc(a.label)}.${a.source === "netrc" || a.source === "env" ? " Sign in below to use another account for this session." : ""}`
    : "Searches work without an account; downloads need one. The password goes only to urs.earthdata.nasa.gov and stays on this machine.";
  $("#signOut").hidden = !a?.can_sign_out;
  $("#authForm").reset();
  $("#authDlg").showModal();
}
$("#authChip").addEventListener("click", openAuth);
$("#authForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value === "cancel") return;
  e.preventDefault();
  const fd = new FormData(e.target);
  const btn = $("#authSubmit");
  btn.disabled = true;
  btn.textContent = "Checking…";
  try {
    await api("/api/auth", { method: "POST", body: { username: fd.get("username"), password: fd.get("password"), remember: fd.get("remember") === "on" } });
    $("#authDlg").close();
    await refreshAuth();
    toast(`Signed in as ${state.auth.username}`, "ok");
  } catch (err) {
    $("#authErr").textContent = err.message;
    $("#authErr").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Sign in";
  }
});
$("#signOut").addEventListener("click", async () => {
  await api("/api/auth", { method: "DELETE" });
  $("#authDlg").close();
  await refreshAuth();
  toast(state.auth?.signed_in ? `Signed out; now using ${state.auth.label.toLowerCase()}` : "Signed out");
});

// ---------------------------------------------------------------- basemaps
function setBasemap(k) {
  // rebuild the source: tile services differ in max zoom and attribution
  map.removeLayer("base");
  map.removeSource("base");
  map.addSource("base", baseSource(k));
  map.addLayer({ id: "base", type: "raster", source: "base", paint: { "raster-saturation": -0.15 } }, map.getLayer("foot-fill") ? "foot-fill" : undefined);
  $$("#basemaps button").forEach((b) => b.classList.toggle("on", b.dataset.bm === k));
  store.set("basemap", k);
}
$("#basemaps").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setBasemap(b.dataset.bm); });
$$("#basemaps button").forEach((b) => b.classList.toggle("on", b.dataset.bm === bm0));

// ---------------------------------------------------------------- jobs
const openJobs = new Set(store.get("openJobs", []));
const openLogs = new Set();
function toggleDrawer(force) {
  state.drawerOpen = force ?? !state.drawerOpen;
  $("#drawer").classList.toggle("open", state.drawerOpen);
  $("#drawer").setAttribute("aria-hidden", String(!state.drawerOpen));
  if (state.drawerOpen) refreshJobs();
}
$("#jobsBtn").addEventListener("click", () => toggleDrawer());
$("#closeDrawer").addEventListener("click", () => toggleDrawer(false));

let jobsTimer;
async function refreshJobs() {
  clearTimeout(jobsTimer);
  try {
    const { jobs } = await api("/api/jobs");
    const before = new Map(state.jobs.map((j) => [j.id, j.state]));
    state.jobs = jobs;
    for (const j of jobs) {
      const was = before.get(j.id);
      if (was && was !== j.state && j.state === "done") toast(`Done: ${j.title}`, "ok");
      if (was && was !== j.state && j.state === "failed") toast(`Failed: ${j.title}`, "bad");
    }
    for (const id of openLogs) {
      const j = jobs.find((x) => x.id === id);
      if (j) j.log = (await api(`/api/jobs/${id}?log=60`)).log;
    }
    renderJobs();
  } catch { /* server restarting */ }
  const live = state.jobs.some((j) => j.state === "running" || j.state === "queued");
  jobsTimer = setTimeout(refreshJobs, live ? 2500 : 10000);
}

function renderJobs() {
  const live = state.jobs.filter((j) => j.state === "running" || j.state === "queued").length;
  $("#jobsBadge").hidden = !live;
  $("#jobsBadge").textContent = live;
  $("#jobsBadge").classList.toggle("live", live > 0);
  if (!state.drawerOpen) return;
  if (!state.jobs.length) { $("#jobs").innerHTML = '<p class="empty">No jobs yet. Search, pick results, then <b>Subset &amp; download</b>.</p>'; return; }
  const scroll = $("#jobs").scrollTop;
  $("#jobs").innerHTML = state.jobs.map(jobCard).join("");
  $("#jobs").scrollTop = scroll;
}

function jobCard(j) {
  const P = PRODUCTS[j.product] || PRODUCTS.disp;
  const pct = Math.round((j.progress || 0) * 100);
  const outs = j.outputs || [];
  const total = outs.reduce((s, o) => s + (o.size || 0), 0);
  const open = openJobs.has(j.id);
  const indet = j.state === "running" && (j.progress || 0) === 0;
  const canExplore = j.product === "disp" && (j.frames || []).length;
  return `<article class="job" data-id="${esc(j.id)}" style="--c:${P.color}">
    <div class="job-h">
      <h3>${esc(j.title || j.id)}</h3><span class="state ${esc(j.state)}">${esc(j.state)}</span>
      <div class="sub">${day(j.created)} · ${j.aoi_km2 != null ? fmtArea(j.aoi_km2) : ""}${j.start || j.end ? ` · ${day(j.start)} → ${day(j.end)}` : ""}${outs.length ? ` · ${outs.length} outputs, ${fmtBytes(total)}` : ""}</div>
    </div>
    ${j.state === "running" || j.state === "queued" ? `<div class="bar ${indet ? "indet" : ""}"><b style="width:${pct}%"></b></div><div class="job-step">${esc(j.step || "queued")}${pct && !indet ? ` · ${pct}%` : ""}</div>` : ""}
    ${j.error ? `<div class="job-err">${esc(j.error)}</div>` : ""}
    <div class="job-a">
      ${canExplore ? `<button class="ghost-btn accent" data-a="explore">Explore &amp; rebase</button>` : ""}
      ${outs.length ? `<button class="ghost-btn" data-a="files">${open ? "Hide files" : "Files"}</button><a class="ghost-btn" href="${SERVER}/api/jobs/${encodeURIComponent(j.id)}/bundle.zip" download>Download .zip</a>` : ""}
      <button class="ghost-btn" data-a="log">${openLogs.has(j.id) ? "Hide log" : "Log"}</button>
      <span class="grow"></span>
      ${j.state === "running" || j.state === "queued" ? `<button class="ghost-btn danger" data-a="cancel">Cancel</button>` : `<button class="ghost-btn danger" data-a="delete" title="Delete the job and its files">Delete</button>`}
    </div>
    ${open && outs.length ? `<div class="files">${outs.map((o) => `<div class="file"><span class="kind">${esc(o.kind)}</span>${o.dir ? `<a title="${esc(o.path)} (folder: in the .zip)">${esc(o.path)}/</a>` : `<a href="${SERVER}/api/jobs/${encodeURIComponent(j.id)}/files/${o.path.split("/").map(encodeURIComponent).join("/")}" download title="${esc(o.path)}">${esc(o.path)}</a>`}<span class="sz">${fmtBytes(o.size)}</span></div>`).join("")}
      <div class="path"><code title="${esc(j.dir)}">${esc(j.dir)}</code><button class="icon" data-a="copy" title="Copy the folder path"><svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/></svg></button></div></div>` : ""}
    ${openLogs.has(j.id) ? `<pre class="log">${esc((j.log || ["(no log yet)"]).join("\n"))}</pre>` : ""}
  </article>`;
}

$("#jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-a]");
  if (!b) return;
  const id = b.closest(".job").dataset.id;
  const j = state.jobs.find((x) => x.id === id);
  const a = b.dataset.a;
  try {
    if (a === "files") { openJobs.has(id) ? openJobs.delete(id) : openJobs.add(id); store.set("openJobs", [...openJobs]); renderJobs(); }
    else if (a === "log") { openLogs.has(id) ? openLogs.delete(id) : openLogs.add(id); await refreshJobs(); }
    else if (a === "cancel") { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); await refreshJobs(); }
    else if (a === "delete") { if (confirm(`Delete this job and its ${(j.outputs || []).length} outputs from disk?`)) { await api(`/api/jobs/${id}`, { method: "DELETE" }); if (state.explore?.job.id === id) closeExplore(); await refreshJobs(); } }
    else if (a === "copy") { await navigator.clipboard.writeText(j.dir); toast("Folder path copied"); }
    else if (a === "explore") openExplore(j);
  } catch (err) { toast(err.message, "bad"); }
});

// ---------------------------------------------------------------- explore (DISP)
let refMarker, ptMarker;
async function openExplore(job) {
  closeExplore(false);
  state.explore = { job, frameId: job.frames[0].id, point: null, series: null, show: store.get("seriesShow", { referenced: true, rebased: false, raw: true }) };
  refPickMarker?.remove();
  toggleDrawer(false);
  $("#timeline").hidden = true;
  $("#inspector").hidden = false;
  $("#inspTitle").textContent = job.title;
  $("#frameTabs").innerHTML = job.frames.map((f) => `<button data-f="${f.id}">${f.id}${f.direction ? ` · ${f.direction.slice(0, 3)}` : ""}</button>`).join("");
  $$("#seriesToggles input").forEach((i) => (i.checked = !!state.explore.show[i.dataset.s]));
  await loadFrame(state.explore.frameId, true);
}

async function loadFrame(fid, fit) {
  const ex = state.explore;
  ex.frameId = fid;
  $$("#frameTabs button").forEach((b) => b.classList.toggle("on", b.dataset.f === fid));
  ex.meta = await api(`/api/jobs/${ex.job.id}/disp/${fid}/meta`);
  ex.vmax = ex.meta.vmax;
  const mm = ex.vmax * 1000;
  $("#vmax").max = Math.max(5, Math.ceil(mm * 4));
  $("#vmax").value = mm.toFixed(1);
  updateVelocity();
  showRef();
  ex.point = null;
  ptMarker?.remove();
  $("#chart").innerHTML = '<p class="empty">Click the map inside the frame to plot a time series.</p>';
  $("#chartStats").innerHTML = "";
  $("#csvBtn").disabled = true;
  if (fit) {
    const c = ex.meta.corners;
    const lons = c.map((p) => p[0]), lats = c.map((p) => p[1]);
    map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: { top: 70, bottom: 380, left: 440, right: 60 }, duration: 700 });
  }
}

function updateVelocity() {
  const ex = state.explore;
  if (!ex) return;
  const vmax = +$("#vmax").value / 1000;
  const coh = +$("#minCoh").value;
  ex.vmax = vmax;
  $("#vmaxVal").textContent = `±${(+$("#vmax").value).toFixed(1)}`;
  $("#legMin").textContent = `−${(+$("#vmax").value).toFixed(1)}`;
  $("#legMax").textContent = `+${(+$("#vmax").value).toFixed(1)}`;
  $("#cohVal").textContent = coh.toFixed(2);
  $("#opVal").textContent = (+$("#opacity").value).toFixed(2);
  const url = `${SERVER}/api/jobs/${ex.job.id}/disp/${ex.frameId}/velocity.png?vmax=${vmax}&min_coherence=${coh}&t=${Date.now()}`;
  map.getSource("vel").updateImage({ url, coordinates: ex.meta.corners });
  map.setLayoutProperty("vel", "visibility", "visible");
  map.setPaintProperty("vel", "raster-opacity", +$("#opacity").value);
}
let velTimer;
for (const id of ["vmax", "minCoh", "opacity"]) $("#" + id).addEventListener("input", () => { clearTimeout(velTimer); velTimer = setTimeout(updateVelocity, id === "opacity" ? 0 : 180); });

function showRef() {
  const ex = state.explore, ref = ex.meta.reference || {};
  refMarker?.remove();
  refMarker = null;
  let h;
  if (ref.mode === "point") {
    refMarker = new maplibregl.Marker({ element: refEl(), draggable: true }).setLngLat([ref.lon, ref.lat]).addTo(map);
    refMarker.on("dragend", () => { const ll = refMarker.getLngLat(); setReference({ mode: "point", lon: ll.lng, lat: ll.lat }); });
    h = `<div class="rt"><svg viewBox="0 0 24 24"><path d="M12 3l9 16H3z"/></svg>Reference point</div>
      <div class="mono">${ref.lon.toFixed(5)}, ${ref.lat.toFixed(5)}</div>
      <div class="dim">${esc(ref.method)} · coherence ${ref.coherence} · 5×5 median</div>
      <div class="dim">its own trend, removed: <span class="mono">${(ref.slope_m_yr * 1000).toFixed(2)} mm/yr</span></div>`;
  } else {
    h = `<div class="rt">Area reference</div><div class="dim">${esc(ref.method || "median of high-coherence pixels")}</div>`;
  }
  if (ref.error) h += `<div class="warn">${esc(ref.error)}</div>`;
  $("#refCard").innerHTML = h + `<div class="dim small">Drag the marker or pick below; GeoTIFFs are rewritten to match.</div>`;
  $$("#refModes button").forEach((b) => b.classList.toggle("on", (b.dataset.ref === "auto" && ref.method === "most coherent spot") || (b.dataset.ref === "area" && ref.mode === "area") || (b.dataset.ref === "pick" && ref.method === "user point")));
}

async function setReference(body) {
  const ex = state.explore;
  try {
    const r = await api(`/api/jobs/${ex.job.id}/disp/${ex.frameId}/reference`, { method: "POST", body });
    ex.meta = r.meta;
    showRef();
    updateVelocity();
    if (ex.point) await loadSeries(ex.point);
    toast(r.reference.mode === "point" ? `Reference moved (${r.reference.method})` : "Area reference", "ok");
  } catch (err) { toast(err.message, "bad"); showRef(); }
}
$("#refModes").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || !state.explore) return;
  if (b.dataset.ref === "pick") draw.pick("Click the new reference point", (ll) => setReference({ mode: "point", lon: ll.lng, lat: ll.lat }));
  else setReference({ mode: b.dataset.ref });
});
$("#frameTabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) loadFrame(b.dataset.f, true); });

async function exploreClick(ll) {
  state.explore.point = [ll.lng, ll.lat];
  ptMarker?.remove();
  const d = document.createElement("div");
  d.className = "pt-marker";
  ptMarker = new maplibregl.Marker({ element: d }).setLngLat(ll).addTo(map);
  await loadSeries(state.explore.point);
}

async function loadSeries([lon, lat]) {
  const ex = state.explore;
  $("#chart").innerHTML = '<p class="empty">Reading…</p>';
  try {
    ex.series = await api(`/api/jobs/${ex.job.id}/disp/${ex.frameId}/series?lon=${lon}&lat=${lat}`);
    $("#csvBtn").disabled = false;
    renderSeries();
  } catch (err) {
    ex.series = null;
    $("#chart").innerHTML = `<p class="empty">${esc(err.message)}</p>`;
    $("#chartStats").innerHTML = "";
  }
}

function renderSeries() {
  const ex = state.explore, s = ex?.series;
  if (!s) return;
  const show = ex.show, series = [];
  const pts = (vals, dates = s.dates) => dates.map((d, i) => [d, vals[i]]);
  let marks = [];
  if (show.raw && s.raw) {
    // one segment per ministack, starting at 0 on its reference date: the saw-tooth rebasing removes
    const segs = [];
    let cur = null;
    s.raw.dates.forEach((d, i) => {
      const ref = s.raw.reference_dates[i];
      if (!cur || cur.ref !== ref) { cur = { ref, pts: [[ref, 0]] }; segs.push(cur); }
      cur.pts.push([d, s.raw.values[i]]);
    });
    marks = segs.slice(1).map((x) => x.ref);
    series.push({ key: "raw", short: "raw", color: "#94a3b8", points: pts(s.raw.values, s.raw.dates), segments: segs.map((x) => x.pts), dashed: true, width: 1.3, opacity: 0.85 });
  }
  if (show.rebased) series.push({ key: "rebased", short: "rebased", color: "#a78bfa", points: pts(s.rebased), width: 1.6, dots: true });
  if (show.referenced) series.push({ key: "referenced", short: "re-referenced", color: getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2dd4bf", points: pts(s.referenced), width: 2.2, dots: true });
  drawChart($("#chart"), { series, marks });
  const v = s.velocity_m_yr;
  $("#chartStats").innerHTML = `<span>${s.lon.toFixed(5)}, ${s.lat.toFixed(5)}</span><span>velocity <b>${v == null ? "–" : (v * 1000).toFixed(2)}</b> mm/yr</span><span>coh <b>${s.coherence == null ? "–" : s.coherence.toFixed(2)}</b></span>`;
}
$("#seriesToggles").addEventListener("change", (e) => {
  const k = e.target.dataset.s;
  if (!k || !state.explore) return;
  state.explore.show[k] = e.target.checked;
  store.set("seriesShow", state.explore.show);
  renderSeries();
});
new ResizeObserver(() => renderSeries()).observe($("#chart"));

$("#csvBtn").addEventListener("click", () => {
  const ex = state.explore, s = ex?.series;
  if (!s) return;
  const raw = new Map((s.raw?.dates || []).map((d, i) => [d.slice(0, 10), [s.raw.values[i], s.raw.reference_dates[i]]]));
  const ref = ex.meta.reference || {};
  const lines = [
    `# OPERA DISP-S1 · ${ex.frameId} · point ${s.lon.toFixed(6)},${s.lat.toFixed(6)} · reference ${ref.mode === "point" ? `${ref.lon},${ref.lat} (${ref.method})` : ref.method || "area"}`,
    "date,referenced_m,rebased_area_m,raw_m,raw_reference_date",
    ...s.dates.map((d, i) => { const r = raw.get(d.slice(0, 10)) || ["", ""]; return [d.slice(0, 10), s.referenced[i] ?? "", s.rebased[i] ?? "", r[0] ?? "", (r[1] || "").slice(0, 10)].join(","); }),
  ];
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = `disp_${ex.frameId}_${s.lon.toFixed(4)}_${s.lat.toFixed(4)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
});

function closeExplore(restore = true) {
  if (!state.explore && restore) return;
  state.explore = null;
  $("#inspector").hidden = true;
  refMarker?.remove();
  ptMarker?.remove();
  if (map.getLayer("vel")) map.setLayoutProperty("vel", "visibility", "none");
  if (restore) showRefPickMarker();
}
$("#closeInsp").addEventListener("click", () => closeExplore());

// ---------------------------------------------------------------- boot
async function connect() {
  try {
    const h = await api("/api/health");
    $("#workdir").textContent = h.workdir;
    $("#workdir").title = h.workdir;
    document.body.classList.remove("offline");
    return true;
  } catch {
    document.body.classList.add("offline");
    return false;
  }
}

function openConnect(failed = false) {
  $("#serverUrl").value = SERVER || DEFAULT_SERVER;
  $("#pagesOrigin").textContent = location.origin;
  $("#connectErr").hidden = !failed;
  if (!$("#connectDlg").open) $("#connectDlg").showModal();
}

$("#connectForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value === "cancel") return;
  e.preventDefault();
  let url;
  try {
    url = new URL($("#serverUrl").value.trim());
  } catch {
    $("#connectErr").textContent = "That is not a URL (e.g. http://localhost:8765).";
    $("#connectErr").hidden = false;
    return;
  }
  if (!["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) || url.protocol !== "http:") {
    $("#connectErr").textContent = "The studio only connects to a server on this machine (http://localhost:PORT).";
    $("#connectErr").hidden = false;
    return;
  }
  SERVER = url.origin;
  try { localStorage.setItem("opera-studio:server", SERVER); } catch { /* private mode */ }
  if (await connect()) {
    $("#connectDlg").close();
    toast("Connected to your OPERA Studio", "ok");
    await refreshAuth();
    refreshJobs();
    if (map.loaded()) refreshOutlines();
  } else {
    $("#connectErr").textContent = `No studio answered at ${SERVER}. Is it running with --allow-origin ${location.origin}, and did the browser allow this site to access apps on this device?`;
    $("#connectErr").hidden = false;
  }
});
$("#connectChip").addEventListener("click", () => openConnect());

(async () => {
  $("#connectChip").hidden = !HOSTED;
  const ok = await connect();
  await refreshAuth();
  if (!ok && HOSTED) openConnect();
  refreshJobs();
})();

// ---------------------------------------------------------------- reference outlines + archive depth
const OUTLINES = {
  frames: { color: "#2cc2bd", minzoom: 2, label: "DISP-S1 frame" },
  bursts: { color: "#50a0e0", minzoom: 6.5, label: "S1 burst" },
  nisar: { color: "#90d0c0", minzoom: 2, label: "NISAR frame" },
};
const lay = {
  on: { frames: false, bursts: false, nisar: false, ...store.get("layersOn", {}) },
  depth: store.get("depthOn", false),
  counts: new Map(), // layer|kind|start|end|id -> n
  gen: 0,
  max: 1,
  visible: { frames: [], bursts: [], nisar: [] },
  dir: ["both", "ascending", "descending"].includes(store.get("layerDir")) ? store.get("layerDir") : "both",
};
// Every layer carries its pass direction; outlines without one (NISAR boxes
// from the fallback database) stay visible whatever the toggle says.
const DIRECTIONAL = new Set(["frames", "bursts", "nisar"]);
const dirShown = (name, f) => lay.dir === "both" || !f.properties?.direction || f.properties.direction === lay.dir;

function applyDirFilter() {
  for (const name of DIRECTIONAL) {
    const filter = lay.dir === "both" ? null : ["any", ["==", ["get", "direction"], lay.dir], ["==", ["get", "direction"], ""]];
    for (const part of ["fill", "line"]) if (map.getLayer(`ol-${name}-${part}`)) map.setFilter(`ol-${name}-${part}`, filter);
  }
  $$("#layDir button").forEach((b) => b.classList.toggle("on", b.dataset.d === lay.dir));
}
const DEPTH_STOPS = [[0, "#22393a"], [0.02, "#007570"], [0.2, "#10b0b0"], [0.45, "#90d0c0"], [0.75, "#90d0f0"], [1, "#f2fbff"]];

// What each layer's depth counts follows the product and filters chosen in the dock.
function depthKind(layer) {
  if (layer === "nisar") return state.filters.nisarKind || "gslc";
  return state.product === "static" ? "static" : null;
}
const depthKey = (layer, id) => [layer, depthKind(layer), $("#start").value, $("#end").value, id].join("|");

function depthPaint() {
  const max = Math.max(lay.max, 1);
  let prev = -1;
  const stops = [];
  for (const [f, c] of DEPTH_STOPS) {
    const v = Math.max(f * max, prev + 0.001);
    stops.push(v, c);
    prev = v;
  }
  return ["case", ["<", ["coalesce", ["feature-state", "n"], -1], 0], "rgba(0,0,0,0)", ["interpolate", ["linear"], ["feature-state", "n"], ...stops]];
}

const visibleFills = () => Object.keys(OUTLINES).map((n) => `ol-${n}-fill`).filter((l) => map.getLayer(l) && map.getLayoutProperty(l, "visibility") === "visible");

function initLayers() {
  for (const [name, o] of Object.entries(OUTLINES)) {
    map.addSource(`ol-${name}`, { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "id" });
    map.addLayer({ id: `ol-${name}-fill`, type: "fill", source: `ol-${name}`, layout: { visibility: "none" }, paint: { "fill-color": depthPaint(), "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.75, 0.32] } }, "foot-fill");
    map.addLayer({ id: `ol-${name}-line`, type: "line", source: `ol-${name}`, layout: { visibility: "none" }, paint: { "line-color": o.color, "line-width": ["case", ["boolean", ["feature-state", "hover"], false], 2.4, name === "bursts" ? 0.7 : 1], "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, 0.7] } }, "foot-fill");
  }
  applyDirFilter();
  $$("[data-layer-toggle]").forEach((i) => (i.checked = !!lay.on[i.dataset.layerToggle]));
  $("#depthToggle").checked = lay.depth;
  $("#layers").classList.toggle("folded", store.get("layersFolded", false));
  let hovered = null;
  map.on("mousemove", (e) => {
    if (draw?.active) return;
    const f = map.queryRenderedFeatures(e.point, { layers: visibleFills() })[0];
    const key = f ? [f.layer.id.split("-")[1], f.id] : null;
    if (hovered && (!key || key[0] !== hovered[0] || key[1] !== hovered[1])) map.setFeatureState({ source: `ol-${hovered[0]}`, id: hovered[1] }, { hover: false });
    hovered = key;
    const tip = $("#hoverTip");
    if (!key) { tip.hidden = true; return; }
    map.setFeatureState({ source: `ol-${key[0]}`, id: key[1] }, { hover: true });
    const n = lay.counts.get(depthKey(key[0], String(key[1])));
    const dir = f.properties.direction ? ` · ${f.properties.direction.slice(0, 3)}` : "";
    tip.innerHTML = `${esc(f.properties.label)}${dir}${n != null ? ` · <b>${n}</b> granules` : ""}`;
    tip.style.left = e.originalEvent.clientX + 14 + "px";
    tip.style.top = e.originalEvent.clientY + 12 + "px";
    tip.hidden = false;
  });
  map.getCanvas().addEventListener("mouseleave", () => ($("#hoverTip").hidden = true));
  map.on("click", (e) => {
    if (draw?.active || state.explore) return;
    if (map.queryRenderedFeatures(e.point, { layers: ["foot-fill"] }).length) return;
    const f = map.queryRenderedFeatures(e.point, { layers: visibleFills() })[0];
    if (f) openTimeline(f.layer.id.split("-")[1], String(f.id), f.properties.label);
  });
  map.on("moveend", scheduleOutlines);
  refreshOutlines();
}
map.on("load", initLayers);

let olTimer;
function scheduleOutlines() { clearTimeout(olTimer); olTimer = setTimeout(refreshOutlines, 250); }

async function refreshOutlines() {
  const gen = ++lay.gen;
  const z = map.getZoom();
  const b = map.getBounds();
  const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(4)).join(",");
  const notes = [];
  for (const [name, o] of Object.entries(OUTLINES)) {
    const vis = lay.on[name] ? "visible" : "none";
    map.setLayoutProperty(`ol-${name}-fill`, "visibility", vis);
    map.setLayoutProperty(`ol-${name}-line`, "visibility", vis);
    if (!lay.on[name]) { lay.visible[name] = []; continue; }
    if (z < o.minzoom) {
      map.getSource(`ol-${name}`).setData({ type: "FeatureCollection", features: [] });
      lay.visible[name] = [];
      notes.push(`zoom in for ${o.label.toLowerCase()}s`);
      continue;
    }
    try {
      const fc = await api(`/api/outlines/${name}?bbox=${bbox}&limit=${name === "bursts" ? 1500 : 2500}`);
      if (gen !== lay.gen) return;
      map.getSource(`ol-${name}`).setData(fc);
      // hidden directions are not counted and do not stretch the colour ramp
      lay.visible[name] = fc.features.filter((f) => dirShown(name, f)).map((f) => String(f.id));
      lay.data = { ...(lay.data || {}), [name]: fc.features };
      if (fc.truncated) notes.push(`${o.label}s: first ${fc.features.length} shown`);
      if (fc.note) notes.push(fc.note);
      for (const id of lay.visible[name]) {
        const n = lay.counts.get(depthKey(name, id));
        map.setFeatureState({ source: `ol-${name}`, id }, { n: lay.depth && n != null ? n : -1 });
      }
    } catch (err) {
      notes.push(`${o.label}s: ${err.message}`);
    }
  }
  lay.notes = notes;
  $("#layStatus").textContent = notes.join(" · ");
  updateRamp();
  if (lay.depth) await refreshDepth(gen);
}

async function refreshDepth(gen) {
  const c = map.getCenter();
  let total = 0, done = 0;
  const queues = {};
  for (const name of Object.keys(OUTLINES)) {
    const feats = (lay.data?.[name] || []).filter((f) => lay.visible[name].includes(String(f.id)) && !lay.counts.has(depthKey(name, String(f.id))));
    const dist = (f) => { const [w, s, e, n] = bounds(f.geometry); return ((w + e) / 2 - c.lng) ** 2 + ((s + n) / 2 - c.lat) ** 2; };
    queues[name] = feats.sort((a, b) => dist(a) - dist(b)).map((f) => String(f.id)); // nearest the view centre first
    total += queues[name].length;
  }
  for (const [name, ids] of Object.entries(queues)) {
    for (let i = 0; i < ids.length; i += 24) {
      if (gen !== lay.gen) return;
      const chunk = ids.slice(i, i + 24);
      $("#layStatus").textContent = `counting granules ${done}/${total}…`;
      try {
        const r = await api(`/api/depth/${name}`, { method: "POST", body: { ids: chunk, kind: depthKind(name), start: $("#start").value || null, end: $("#end").value || null } });
        if (gen !== lay.gen) return;
        for (const [id, n] of Object.entries(r.counts)) {
          if (n == null) continue;
          lay.counts.set(depthKey(name, id), n);
          map.setFeatureState({ source: `ol-${name}`, id }, { n });
        }
      } catch (err) { $("#layStatus").textContent = err.message; return; }
      done += chunk.length;
      updateRamp();
    }
  }
  if (gen === lay.gen) $("#layStatus").textContent = [...(lay.notes || []), "hover for counts · click an outline for its timeline"].join(" · ");
}

function updateRamp() {
  const vals = [];
  for (const name of Object.keys(OUTLINES)) for (const id of lay.visible[name]) { const n = lay.counts.get(depthKey(name, id)); if (n != null) vals.push(n); }
  vals.sort((a, b) => a - b);
  const p95 = vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.95))] : 1;
  if (Math.abs(p95 - lay.max) > 0.5) {
    lay.max = Math.max(p95, 1);
    for (const name of Object.keys(OUTLINES)) if (map.getLayer(`ol-${name}-fill`)) map.setPaintProperty(`ol-${name}-fill`, "fill-color", depthPaint());
  }
  $("#ramp").hidden = !lay.depth;
  $("#rampMid").textContent = Math.round(lay.max / 2);
  $("#rampMax").textContent = `${Math.round(lay.max)}+`;
  const what = { frames: depthKind("frames") === "static" ? "DISP-S1-STATIC" : "DISP-S1", bursts: depthKind("bursts") === "static" ? "CSLC-S1-STATIC" : "CSLC-S1", nisar: `NISAR ${(depthKind("nisar") || "").toUpperCase()}` };
  const on = Object.keys(OUTLINES).filter((n) => lay.on[n]).map((n) => what[n]);
  $("#depthWhat").textContent = on.length ? `granules in CMR: ${on.join(", ")}` : "turn on a layer above";
}

$("#layers").addEventListener("change", (e) => {
  const t = e.target;
  if (t.dataset.layerToggle) { lay.on[t.dataset.layerToggle] = t.checked; store.set("layersOn", lay.on); }
  else if (t.id === "depthToggle") {
    lay.depth = t.checked;
    store.set("depthOn", lay.depth);
    if (!lay.depth) for (const name of Object.keys(OUTLINES)) for (const id of lay.visible[name]) map.setFeatureState({ source: `ol-${name}`, id }, { n: -1 });
  } else return;
  refreshOutlines();
});
$("#layDir").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-d]");
  if (!b || b.dataset.d === lay.dir) return;
  lay.dir = b.dataset.d;
  store.set("layerDir", lay.dir);
  applyDirFilter();
  refreshOutlines();
});
$("#layersFold").addEventListener("click", () => store.set("layersFolded", $("#layers").classList.toggle("folded")));
// counts depend on the product, the NISAR kind and the dates: recolour when they change
for (const sel of ["#products", "#filters", "#dateChips"]) $(sel).addEventListener("click", () => setTimeout(() => lay.depth && map.getLayer("ol-frames-fill") && refreshOutlines(), 50));
for (const sel of ["#start", "#end"]) $(sel).addEventListener("change", () => lay.depth && refreshOutlines());

// ---------------------------------------------------------------- archive timeline of one outline
let tlCurrent = null;
const TL_NAMES = { frames: ["DISP-S1", "DISP-S1-STATIC"], bursts: ["CSLC-S1", "CSLC-S1-STATIC"] };
async function openTimeline(layer, id, label) {
  closeExplore();
  const kind = depthKind(layer);
  tlCurrent = { layer, id, geometry: null };
  $("#timeline").hidden = false;
  $("#tlTitle").textContent = label || id;
  $("#tlPill").textContent = layer === "nisar" ? `NISAR ${(kind || "gslc").toUpperCase()}` : TL_NAMES[layer][kind === "static" ? 1 : 0];
  $("#tlStats").innerHTML = "reading the archive…";
  $("#tlChart").innerHTML = "";
  const q = new URLSearchParams();
  if (kind) q.set("kind", kind);
  if ($("#start").value) q.set("start", $("#start").value);
  if ($("#end").value) q.set("end", $("#end").value);
  try {
    const tl = await api(`/api/timeline/${layer}/${encodeURIComponent(id)}?${q}`);
    if (tlCurrent?.id !== id) return;
    tlCurrent.geometry = tl.geometry;
    tlCurrent.data = tl;
    const gapDays = (tl.gaps || []).reduce((s, g) => s + g[2], 0);
    $("#tlStats").innerHTML = [
      `<span><b>${tl.n_files}</b> files</span>`,
      `<span><b>${tl.n_dates}</b> dates</span>`,
      tl.first ? `<span><b>${tl.first}</b> → <b>${tl.last}</b></span>` : "",
      tl.revisit_days ? `<span>revisit <b>${tl.revisit_days}</b> d</span>` : "",
      (tl.reference_dates || []).length > 1 ? `<span>reference dates <b>${tl.reference_dates.length}</b></span>` : "",
      (tl.gaps || []).length ? `<span class="gap">gaps <b>${tl.gaps.length}</b> (${gapDays} d)</span>` : "",
      `<span class="dim">${esc(tl.collection)}</span>`,
    ].join("");
    drawTimeline($("#tlChart"), tl);
  } catch (err) {
    $("#tlStats").innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
}
new ResizeObserver(() => tlCurrent?.data && drawTimeline($("#tlChart"), tlCurrent.data)).observe($("#tlChart"));
$("#tlClose").addEventListener("click", () => { $("#timeline").hidden = true; tlCurrent = null; });
$("#tlUseArea").addEventListener("click", () => {
  if (!tlCurrent?.geometry) return;
  setAoi(tlCurrent.geometry);
  toast(`Area set to ${tlCurrent.id}`, "ok");
});
