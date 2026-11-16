// Polygon / rectangle drawing and single-point picking on a MapLibre map.
// No plugin: two GeoJSON sources ("aoi" for the finished area, "draft" while drawing).

const EMPTY = { type: "FeatureCollection", features: [] };

export class Draw {
  constructor(map, { onArea, onHint, color = "#2dd4bf" }) {
    this.map = map;
    this.onArea = onArea;
    this.onHint = onHint;
    this.mode = null;
    this.pts = [];
    this.color = color;
    this._bound = {};
    map.addSource("aoi", { type: "geojson", data: EMPTY });
    map.addSource("draft", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": color, "fill-opacity": 0.08 } });
    map.addLayer({ id: "aoi-glow", type: "line", source: "aoi", paint: { "line-color": color, "line-width": 8, "line-opacity": 0.18, "line-blur": 4 } });
    map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": color, "line-width": 2 } });
    map.addLayer({ id: "draft-fill", type: "fill", source: "draft", filter: ["==", "$type", "Polygon"], paint: { "fill-color": color, "fill-opacity": 0.1 } });
    map.addLayer({ id: "draft-line", type: "line", source: "draft", filter: ["!=", "$type", "Point"], paint: { "line-color": color, "line-width": 2, "line-dasharray": [2, 1.5] } });
    map.addLayer({ id: "draft-pts", type: "circle", source: "draft", filter: ["==", "$type", "Point"], paint: { "circle-radius": ["case", ["get", "first"], 6, 4], "circle-color": "#06090f", "circle-stroke-color": color, "circle-stroke-width": 2 } });
  }

  setColor(color) {
    this.color = color;
    for (const [id, prop] of [["aoi-fill", "fill-color"], ["aoi-glow", "line-color"], ["aoi-line", "line-color"], ["draft-fill", "fill-color"], ["draft-line", "line-color"], ["draft-pts", "circle-stroke-color"]]) {
      this.map.setPaintProperty(id, prop, color);
    }
  }

  setArea(geometry) {
    this.map.getSource("aoi").setData(geometry ? { type: "Feature", geometry, properties: {} } : EMPTY);
  }

  start(mode) {
    this.cancel(false);
    this.mode = mode;
    this.pts = [];
    document.body.classList.add("drawing");
    this.map.doubleClickZoom.disable();
    const m = this.map;
    if (mode === "polygon") {
      this.onHint("Click to add corners · <kbd>Enter</kbd> or click the first corner to finish · <kbd>⌫</kbd> undo · <kbd>Esc</kbd> cancel");
      this._on("click", (e) => this._polyClick(e));
      this._on("mousemove", (e) => this._preview(e.lngLat));
      this._on("dblclick", (e) => { e.preventDefault(); this._finishPolygon(); });
    } else if (mode === "rectangle") {
      this.onHint("Press and drag to draw a box · <kbd>Esc</kbd> cancel");
      m.dragPan.disable();
      this._on("mousedown", (e) => { this.pts = [e.lngLat]; });
      this._on("mousemove", (e) => { if (this.pts.length) this._rectPreview(this.pts[0], e.lngLat); });
      this._on("mouseup", (e) => {
        if (!this.pts.length) return;
        const g = rectGeom(this.pts[0], e.lngLat);
        if (Math.abs(this.pts[0].lng - e.lngLat.lng) < 1e-6) { this.pts = []; return; }
        this._done(g);
      });
    }
    this._key = (e) => {
      if (e.key === "Escape") this.cancel();
      else if (e.key === "Enter" && this.mode === "polygon") this._finishPolygon();
      else if (e.key === "Backspace" && this.mode === "polygon" && this.pts.length) { this.pts.pop(); this._preview(null); }
    };
    window.addEventListener("keydown", this._key);
  }

  // One click on the map -> callback(lngLat). Used for reference points.
  pick(hint, callback) {
    this.cancel(false);
    this.mode = "pick";
    document.body.classList.add("picking");
    this.onHint(hint + " · <kbd>Esc</kbd> cancel");
    this._on("click", (e) => { const ll = e.lngLat; this.cancel(); callback(ll); });
    this._key = (e) => { if (e.key === "Escape") this.cancel(); };
    window.addEventListener("keydown", this._key);
  }

  get active() { return this.mode !== null; }

  cancel(notify = true) {
    for (const [ev, fn] of Object.entries(this._bound)) this.map.off(ev, fn);
    this._bound = {};
    if (this._key) window.removeEventListener("keydown", this._key);
    this._key = null;
    this.map.dragPan.enable();
    this.map.doubleClickZoom.enable();
    document.body.classList.remove("drawing", "picking");
    this.map.getSource("draft").setData(EMPTY);
    const was = this.mode;
    this.mode = null;
    this.pts = [];
    this.onHint(null);
    if (notify && was) this.onArea?.(undefined);
  }

  _on(ev, fn) { this._bound[ev] = fn; this.map.on(ev, fn); }

  _polyClick(e) {
    if (this.pts.length >= 3) {
      const p0 = this.map.project(this.pts[0]);
      const p = this.map.project(e.lngLat);
      if (Math.hypot(p.x - p0.x, p.y - p0.y) < 10) return this._finishPolygon();
    }
    this.pts.push(e.lngLat);
    this._preview(e.lngLat);
  }

  _preview(cursor) {
    const ring = this.pts.map((p) => [p.lng, p.lat]);
    if (cursor) ring.push([cursor.lng, cursor.lat]);
    const feats = this.pts.map((p, i) => ({ type: "Feature", properties: { first: i === 0 }, geometry: { type: "Point", coordinates: [p.lng, p.lat] } }));
    if (ring.length >= 3) feats.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] } });
    else if (ring.length === 2) feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: ring } });
    this.map.getSource("draft").setData({ type: "FeatureCollection", features: feats });
  }

  _rectPreview(a, b) {
    this.map.getSource("draft").setData({ type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: rectGeom(a, b) }] });
  }

  _finishPolygon() {
    // a dblclick also fires two clicks: drop the duplicate last corner
    const pts = this.pts.filter((p, i, a) => i === 0 || p.lng !== a[i - 1].lng || p.lat !== a[i - 1].lat);
    if (pts.length < 3) return;
    const ring = pts.map((p) => [p.lng, p.lat]);
    this._done({ type: "Polygon", coordinates: [[...ring, ring[0]]] });
  }

  _done(geom) {
    this.cancel(false);
    this.setArea(geom);
    this.onArea?.(geom);
  }
}

function rectGeom(a, b) {
  const w = Math.min(a.lng, b.lng), e = Math.max(a.lng, b.lng), s = Math.min(a.lat, b.lat), n = Math.max(a.lat, b.lat);
  return { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] };
}

// ---- geometry helpers -------------------------------------------------------
const R = 6371.0088;
function ringArea(ring) {
  // spherical excess approximation (good to <0.5% for areas we use)
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i], [x2, y2] = ring[i + 1];
    a += ((x2 - x1) * Math.PI / 180) * (2 + Math.sin(y1 * Math.PI / 180) + Math.sin(y2 * Math.PI / 180));
  }
  return Math.abs(a * R * R / 2);
}
export function areaKm2(g) {
  if (!g) return 0;
  const polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
  return polys.reduce((s, p) => s + ringArea(p[0]) - p.slice(1).reduce((h, r) => h + ringArea(r), 0), 0);
}
export function bounds(g) {
  let w = 180, s = 90, e = -180, n = -90;
  const walk = (c) => { if (typeof c[0] === "number") { w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]); } else c.forEach(walk); };
  walk(g.coordinates);
  return [w, s, e, n];
}

// WKT (POLYGON / MULTIPOLYGON), GeoJSON (geometry / Feature / FeatureCollection) or "w,s,e,n".
export function parseArea(text) {
  const t = text.trim();
  if (!t) throw new Error("Nothing to read.");
  if (t.startsWith("{")) {
    const j = JSON.parse(t);
    const geoms = j.type === "FeatureCollection" ? j.features.map((f) => f.geometry) : j.type === "Feature" ? [j.geometry] : [j];
    const polys = [];
    for (const g of geoms) {
      if (g.type === "Polygon") polys.push(g.coordinates);
      else if (g.type === "MultiPolygon") polys.push(...g.coordinates);
    }
    if (!polys.length) throw new Error("No polygon in this GeoJSON.");
    return polys.length === 1 ? { type: "Polygon", coordinates: polys[0] } : { type: "MultiPolygon", coordinates: polys };
  }
  const nums = t.split(/[\s,]+/).map(Number);
  if (nums.length === 4 && nums.every(Number.isFinite)) {
    const [w, s, e, n] = nums;
    if (w >= e || s >= n) throw new Error("Use west, south, east, north (west < east, south < north).");
    return { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] };
  }
  const m = /^\s*(MULTIPOLYGON|POLYGON)\s*(.*)$/is.exec(t);
  if (!m) throw new Error("Not WKT, GeoJSON or a bounding box.");
  const body = m[2].trim();
  const ring = (s) => s.replace(/[()]/g, "").split(",").map((p) => p.trim().split(/\s+/).slice(0, 2).map(Number));
  const poly = (s) => s.replace(/^\(|\)$/g, "").split(/\)\s*,\s*\(/).map(ring);
  if (m[1].toUpperCase() === "POLYGON") return { type: "Polygon", coordinates: poly(body.replace(/^\(|\)$/g, "")) };
  const parts = body.replace(/^\(\s*\(|\)\s*\)$/g, "").split(/\)\s*\)\s*,\s*\(\s*\(/);
  return { type: "MultiPolygon", coordinates: parts.map((p) => poly(p)) };
}
