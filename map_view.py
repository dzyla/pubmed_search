"""
The "Map" tab: every paper in the index on one zoomable map (Leaflet over the
backend's density tiles), with the query, the results and any example papers
placed on it. Clicking the map lists the papers at that spot.
"""
import json
import math
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui_components import EOSIN, INK, SOURCE_COLORS, SOURCE_NAMES

# Where the browser reaches the backend's /map/tiles and /v1/map/nearby. Empty
# means the page's own origin (nginx routes both); set it for local development.
MAP_BASE_URL = os.environ.get("MSS_MAP_BASE_URL", "").rstrip("/")
HEIGHT = 620


def _xy(value):
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        return [float(value[0]), float(value[1])]
    return None


def _points(results) -> list:
    pts = []
    for _, row in results.iterrows():
        xy = _xy(row.get("map_xy"))
        if xy:
            pts.append({"xy": xy, "rank": int(row["rank"]), "title": str(row.get("title") or ""),
                        "url": row.get("url") or "", "ref": row.get("ref") or "",
                        "source": str(row.get("source") or ""),
                        "year": int(row["year"]) if str(row.get("year") or "").isdigit() else None,
                        "journal": str(row.get("journal") or "")})
    return pts


def render_map(info: dict, results, meta: dict, height: int = HEIGHT):
    seeds = [{"xy": _xy(s.get("map_xy")), "title": s.get("title", ""), "url": s.get("url") or ""}
             for s in meta.get("seeds", []) if _xy(s.get("map_xy"))]
    data = {
        "map": {k: info[k] for k in ("world", "max_zoom", "tile", "tiles", "labels", "colors", "counts")},
        "base": MAP_BASE_URL,
        "query": _xy(meta.get("query_map_xy")),
        "results": _points(results),
        "seeds": seeds,
        "sourceColors": SOURCE_COLORS,
        "sourceNames": SOURCE_NAMES,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    components.html(_TEMPLATE.replace("__DATA__", payload).replace("__EOSIN__", EOSIN)
                    .replace("__INK__", INK).replace("__HEIGHT__", str(height))
                    .replace("__BASE__", MAP_BASE_URL), height=height + 4)


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="__BASE__/map/static/leaflet.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter:wght@400;500&display=swap">
<script src="__BASE__/map/static/leaflet.js"></script>
<style>
  :root { --eosin: __EOSIN__; --ink: __INK__; --field: #14102A; --paper: #FBFAFD; --muted: #6D6884; }
  html, body { margin: 0; height: 100%; background: transparent; font-family: Inter, system-ui, sans-serif; }
  #map { height: __HEIGHT__px; border-radius: 6px; background: var(--field); }
  .leaflet-container { background: var(--field); font-family: Inter, system-ui, sans-serif; }
  .lbl span { display: block; transform: translate(-50%, -50%); white-space: nowrap; text-align: center;
              font-family: Newsreader, serif; font-style: italic; font-size: 13px; line-height: 1.15;
              color: rgba(255,255,255,.86); text-shadow: 0 0 3px #000, 0 0 6px #000; pointer-events: none; }
  .lbl.fine span { font-size: 12px; color: rgba(255,255,255,.78); }
  .pin { width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid #fff; box-sizing: border-box;
         color: #fff; font: 500 10px/17px Inter, sans-serif; text-align: center; box-shadow: 0 0 0 1px rgba(0,0,0,.45); }
  .seed { width: 14px; height: 14px; background: var(--eosin); border: 2px solid #fff; transform: rotate(45deg);
          box-shadow: 0 0 0 1px rgba(0,0,0,.5); }
  .legend { background: rgba(20,16,42,.82); color: #e9e6f4; padding: 8px 10px; border-radius: 6px;
            font-size: 11.5px; line-height: 1.6; }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
  .leaflet-popup-content-wrapper { border-radius: 6px; }
  .leaflet-popup-content { margin: 12px 14px; font-size: 12.5px; color: var(--ink); width: 300px; }
  .pop h4 { margin: 0 0 6px; font: 500 15px/1.25 Newsreader, serif; }
  .pop a { color: var(--ink); }
  .pop .meta { color: var(--muted); margin-bottom: 6px; }
  .pop ol { margin: 4px 0 8px; padding-left: 18px; max-height: 230px; overflow-y: auto; }
  .pop li { margin-bottom: 5px; }
  .pop li a { font: 14px/1.25 Newsreader, serif; text-decoration: none; }
  .pop li a:hover { text-decoration: underline; }
  .pop .act { display: inline-block; margin-top: 2px; padding: 5px 10px; border-radius: 5px; background: var(--eosin);
              color: #fff !important; text-decoration: none; font-weight: 500; }
</style></head>
<body><div id="map" role="region" aria-label="Map of all indexed papers"></div>
<script>
const D = __DATA__;
const M = D.map, Z = M.max_zoom, W = M.tile * Math.pow(2, Z), w = M.world;
const map = L.map("map", {crs: L.CRS.Simple, minZoom: 0, maxZoom: Z + 2, zoomSnap: 0.5,
                          attributionControl: false, worldCopyJump: false});
const toLL = (x, y) => map.unproject([(x - w.x0) / w.size * W, (1 - (y - w.y0) / w.size) * W], Z);
const fromLL = ll => { const p = map.project(ll, Z); return [w.x0 + p.x / W * w.size, w.y0 + (1 - p.y / W) * w.size]; };
const world = L.latLngBounds(map.unproject([0, W], Z), map.unproject([W, 0], Z));
L.tileLayer(D.base + M.tiles, {maxNativeZoom: Z, maxZoom: Z + 2, bounds: world, noWrap: true}).addTo(map);
map.setMaxBounds(world.pad(0.3));

const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let page = "/";
try { page = window.parent.location.origin + window.parent.location.pathname; } catch (e) {}
const similarLink = refs => page + "?similar=" + encodeURIComponent(refs.join(","));
const color = s => M.colors[s] || D.sourceColors[s] || "#888";   // same hues as the tiles
const sourceName = s => D.sourceNames[s] || s;

// topic labels: broad regions when zoomed out, finer ones from zoom 4; larger
// regions first, and a label that would overlap one already shown is left out
const labelLayer = L.layerGroup().addTo(map);
const labelSets = {};
for (const level of ["coarse", "fine"]) {
  labelSets[level] = (M.labels[level] || []).slice().sort((a, b) => b.n - a.n).map(l => {
    const lines = String(l.text).split("\\n");
    return {l, ll: toLL(l.x, l.y), w: Math.max(...lines.map(t => t.length)) * 6.6 + 8, h: lines.length * 15 + 4,
            marker: L.marker(toLL(l.x, l.y), {interactive: false, keyboard: false,
              icon: L.divIcon({className: "lbl " + level, iconSize: null,
                               html: "<span>" + lines.map(esc).join("<br>") + "</span>"})})};
  });
}
function showLabels() {
  const z = map.getZoom(), shown = [];
  labelLayer.clearLayers();
  for (const item of labelSets[z >= 4 ? "fine" : "coarse"]) {
    const p = map.project(item.ll, z);
    const box = [p.x - item.w / 2, p.y - item.h / 2, p.x + item.w / 2, p.y + item.h / 2];
    if (shown.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
    shown.push(box);
    labelLayer.addLayer(item.marker);
  }
}
map.on("zoomend", showLabels);

const placed = [];
for (const r of D.results) {
  const ll = toLL(r.xy[0], r.xy[1]); placed.push(ll);
  const meta = [sourceName(r.source), r.journal && r.journal !== "N/A" ? r.journal : "", r.year || ""].filter(Boolean).join(", ");
  L.marker(ll, {riseOnHover: true, icon: L.divIcon({className: "", iconSize: [20, 20], iconAnchor: [10, 10],
      html: '<div class="pin" style="background:' + color(r.source) + '">' + r.rank + "</div>"})})
    .bindTooltip(r.rank + ". " + esc(r.title.length > 90 ? r.title.slice(0, 88) + "…" : r.title), {direction: "top", offset: [0, -8]})
    .bindPopup('<div class="pop"><h4>' + (r.url ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">' + esc(r.title) + "</a>" : esc(r.title)) +
      '</h4><div class="meta">Result ' + r.rank + ". " + esc(meta) + "</div>" +
      (r.ref ? '<a class="act" target="_blank" rel="noopener" href="' + esc(similarLink([r.ref])) + '">Similar papers</a>' : "") + "</div>")
    .addTo(map);
}
for (const s of D.seeds) {
  const ll = toLL(s.xy[0], s.xy[1]); placed.push(ll);
  L.marker(ll, {zIndexOffset: 900, icon: L.divIcon({className: "", iconSize: [14, 14], iconAnchor: [7, 7], html: '<div class="seed"></div>'})})
    .bindTooltip("Example: " + esc(s.title), {direction: "top", offset: [0, -8]}).addTo(map);
}
if (D.query) {
  const ll = toLL(D.query[0], D.query[1]); placed.push(ll);
  L.marker(ll, {zIndexOffset: 1000, icon: L.divIcon({className: "", iconSize: [30, 30], iconAnchor: [15, 15],
    html: '<svg width="30" height="30" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5l2.9 6.1 6.6.8-4.9 4.6 1.3 6.6L12 17.3 6.1 20.6l1.3-6.6L2.5 9.4l6.6-.8z" fill="' + getComputedStyle(document.documentElement).getPropertyValue("--eosin") + '" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>'})})
    .bindTooltip(D.seeds.length ? "The examples combined" : "Your query", {direction: "top", offset: [0, -12]}).addTo(map);
}
// Streamlit builds the tab while it is hidden (zero size): fit the view once it is shown
// Leaflet ignores size changes until a view exists, so set one right away
map.setView(world.getCenter(), 0);
let fitted = false;
function fitView() {
  if (fitted || !document.getElementById("map").clientWidth) return;
  map.invalidateSize();
  fitted = true;
  const regions = (M.labels.coarse || []).map(l => toLL(l.x, l.y));
  if (placed.length) map.fitBounds(L.latLngBounds(placed).pad(0.35), {maxZoom: 5});
  else if (regions.length) map.fitBounds(L.latLngBounds(regions).pad(0.12));   // where the papers are
  else map.fitBounds(world);
  showLabels();
}
new ResizeObserver(() => { map.invalidateSize(); fitView(); }).observe(document.getElementById("map"));
fitView();

// what is here? the papers nearest to the clicked point
map.on("click", e => {
  const [x, y] = fromLL(e.latlng);
  const pop = L.popup({maxWidth: 340}).setLatLng(e.latlng).setContent('<div class="pop">Looking up papers here…</div>').openOn(map);
  fetch(D.base + "/v1/map/nearby?x=" + x.toFixed(4) + "&y=" + y.toFixed(4) + "&k=8")
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(d => {
      const ps = d.papers || [];
      if (!ps.length) { pop.setContent('<div class="pop">No papers here. Try a brighter area.</div>'); return; }
      const items = ps.map(p => "<li>" + (p.url ? '<a href="' + esc(p.url) + '" target="_blank" rel="noopener">' + esc(p.title) + "</a>" : esc(p.title)) +
        '<div class="meta">' + esc([sourceName(p.source), p.year].filter(Boolean).join(", ")) + "</div></li>").join("");
      const refs = ps.map(p => p.ref).filter(Boolean).slice(0, 5);
      pop.setContent('<div class="pop"><div class="meta">Papers at this spot</div><ol>' + items + "</ol>" +
        (refs.length ? '<a class="act" target="_blank" rel="noopener" href="' + esc(similarLink(refs)) + '">Find papers like these</a>' : "") + "</div>");
    })
    .catch(status => pop.setContent('<div class="pop">' + (status === 429 ? "Too many lookups at once. Wait a few seconds and click again." : "Could not look up this spot. Try again in a moment.") + "</div>"));
});

const legend = L.control({position: "bottomleft"});
legend.onAdd = () => {
  const div = L.DomUtil.create("div", "legend");
  div.innerHTML = Object.entries(M.colors).map(([s, c]) => '<div><i style="background:' + c + '"></i>' + esc(sourceName(s)) + "</div>").join("") +
    '<div style="margin-top:4px;opacity:.75">Brighter = more papers</div>';
  return div;
};
legend.addTo(map);
</script></body></html>
"""


def render_landing_preview(total_papers: int):
    """The map's doorway on the landing page: a still preview that opens the live map."""
    count = f"{total_papers / 1e6:.1f} million" if total_papers >= 1e6 else "all"
    st.markdown(
        f"""<a class="mss-mapcard" href="?map=1" target="_self" aria-label="Explore the map of all papers">
        <img src="{MAP_BASE_URL}/map/preview.jpg" alt="" loading="lazy">
        <span>Explore the map of {count} papers</span></a>
        <p class="mss-mapnote">Papers on similar topics sit together. Zoom in, and click any spot to see
        the papers there.</p>""",
        unsafe_allow_html=True,
    )


def render_explore(info: dict):
    """The live map on its own, before any search."""
    head, close = st.columns([5, 1], vertical_alignment="bottom")
    head.markdown("### Map of all papers")
    close.markdown('<a class="mss-mapclose" href="/" target="_self">Close the map</a>', unsafe_allow_html=True)
    render_map(info, pd.DataFrame(), {}, height=680)
    st.caption("Each point of light is a paper, coloured by database; brighter areas hold more papers. "
               "Zoom in for finer topics, click any spot to see its papers, and from there find papers "
               "like them. Search above to see where your own question lands.")
