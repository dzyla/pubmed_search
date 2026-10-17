"""
Per-browser settings, search history and saved alerts, plus the views that show them.

Everything here lives in the visitor's browser (localStorage, key "mss_prefs_v1"):
the server never stores search history or which alerts belong to whom. An alert itself
(query + bar) is stored on the server under a random token; the browser keeps the
token, and the token's RSS feed URL is the only way to read it.

Streamlit components v2 run their JavaScript in the page (not in an iframe), so one
small component reads/writes localStorage and another copies text to the clipboard.
"""
import json
from datetime import datetime

import streamlit as st

import backend_client
import citations
from ui_components import SOURCE_NAMES, _e, _safe_url

HISTORY_MAX = 100

_STORE_JS = """
export default function(component) {
  const { data, setStateValue } = component;
  const KEY = "mss_prefs_v1";
  try {
    if (data && data.write != null) {   // a save: write only (the session already has the value)
      localStorage.setItem(KEY, data.write);
      return;
    }
    setStateValue("value", localStorage.getItem(KEY) || "{}");
  } catch (e) {
    setStateValue("value", "{}");       // storage blocked (private mode): start empty
  }
}
"""
_store = st.components.v2.component("mss_store", js=_STORE_JS)

_COPY_JS = """
export default function(component) {
  const { data, parentElement } = component;
  const btn = parentElement.querySelector("button");
  const label = parentElement.querySelector(".lbl");
  label.textContent = data.label;
  btn.title = data.title || "";
  btn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(data.text);
    } catch (e) {                         // older browsers / insecure context
      const t = document.createElement("textarea");
      t.value = data.text; document.body.appendChild(t); t.select();
      document.execCommand("copy"); t.remove();
    }
    label.textContent = "Copied";
    setTimeout(() => { label.textContent = data.label; }, 1500);
  };
}
"""
_COPY_HTML = ('<button type="button"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 17h3l2-4V7H5v6h3'
              'zm8 0h3l2-4V7h-6v6h3z"/></svg><span class="lbl"></span></button>')
_COPY_CSS = """
button { display:inline-flex; align-items:center; gap:.3rem; border:none; background:none; cursor:pointer;
         color: var(--st-text-color); font: inherit; font-size: .875rem; padding: .25rem .25rem; border-radius:.5rem; }
button:hover { color: var(--st-primary-color); }
svg { width: 1.1rem; height: 1.1rem; fill: currentColor; }
"""
_copy = st.components.v2.component("mss_copy", html=_COPY_HTML, css=_COPY_CSS, js=_COPY_JS)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def prefs() -> dict:
    """This browser's saved preferences ({} until the browser has answered, on the first run)."""
    p = st.session_state.get("prefs")
    if p is None:
        state = st.session_state.get("mss_store")
        raw = getattr(state, "value", None) if state is not None else None
        if raw is not None:
            try:
                p = json.loads(raw)
            except ValueError:
                p = {}
            st.session_state["prefs"] = p if isinstance(p, dict) else {}
            return st.session_state["prefs"]
        return {}
    return p


def _save():
    if st.session_state.get("prefs") is not None:      # never overwrite what has not been read yet
        st.session_state["_prefs_pending"] = json.dumps(st.session_state["prefs"])


def mount():
    """Reads localStorage (first run) and writes pending changes. Call once per run, after all changes."""
    _store(data={"write": st.session_state.pop("_prefs_pending", None)}, key="mss_store",
           on_value_change=lambda: None)


def copy_button(text: str, key: str, label: str = "Copy citation", title: str = None):
    _copy(data={"text": text, "label": label, "title": title or ""}, key=key)


def cite_style() -> str:
    style = prefs().get("cite_style")
    return style if style in citations.STYLES else citations.DEFAULT_STYLE


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def render_settings():
    p = prefs()
    with st.popover("Settings", icon=":material/settings:", help="Citation style, precise ranking and search history"):
        style = st.selectbox("Citation style", citations.STYLES, index=citations.STYLES.index(cite_style()),
                             help="Used by “Copy citation” on each result and by the Export tab.")
        precise = st.toggle("Precise ranking (slower)", value=p.get("precise", False),
                            help="Re-checks the top 20 matches with a biomedical model that reads each "
                                 "abstract together with your query, and puts the best 10 first. Adds "
                                 "about half a second per search.")
        keep = st.toggle("Keep my search history on this device", value=p.get("history_on", True),
                         help="Your last 100 searches, stored only in this browser.")
        if style != cite_style() or keep != p.get("history_on", True) or precise != p.get("precise", False):
            p["cite_style"], p["history_on"], p["precise"] = style, keep, precise
            if not keep:
                p["history"] = []
            _save()
        st.caption("Settings, search history and your list of alerts are kept only in this browser; "
                   "they are not sent to or stored on the server.")


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------

def add_history(query: str, sources, start_date, end_date, n_results: int):
    p = prefs()
    if st.session_state.get("prefs") is None or not p.get("history_on", True) or not query:
        return
    entry = {"q": query[:2000], "sources": list(sources or []), "start": start_date, "end": end_date,
             "n": int(n_results), "t": datetime.now().isoformat(timespec="minutes")}
    history = [e for e in p.get("history", []) if e.get("q") != entry["q"]]
    p["history"] = [entry] + history[:HISTORY_MAX - 1]
    _save()


def _run_again(entry: dict):
    """on_click: put a saved search back into the form and run it."""
    st.session_state["query_text"] = entry.get("q", "")
    if entry.get("sources"):
        st.session_state["sources_sel"] = entry["sources"]
    has_dates = bool(entry.get("start") or entry.get("end"))
    st.session_state["date_filter_toggle"] = has_dates
    if has_dates:
        from datetime import date
        st.session_state["start_d"] = date.fromisoformat(entry["start"]) if entry.get("start") else date(1800, 1, 1)
        st.session_state["end_d"] = date.fromisoformat(entry["end"]) if entry.get("end") else date.today()
    st.session_state["run_saved_search"] = True
    st.session_state["page"] = "Search"


def _remove_history(i: int):
    p = prefs()
    history = p.get("history", [])
    if 0 <= i < len(history):
        history.pop(i)
        _save()


def _clear_history():
    prefs()["history"] = []
    _save()


def _sources_text(sources) -> str:
    return ", ".join(SOURCE_NAMES.get(s, s) for s in sources) if sources else "default databases"


def render_history():
    p = prefs()
    history = p.get("history", [])
    if st.session_state.get("prefs") is None:
        st.info("Loading your history…")
        return
    if not p.get("history_on", True):
        st.info("Search history is off. Turn it on under Settings.")
        return
    if not history:
        st.info("No searches yet. Your searches will appear here, saved only in this browser.")
        return
    top_l, top_r = st.columns([3, 1], vertical_alignment="bottom")
    flt = top_l.text_input("Filter", placeholder="Filter your searches", label_visibility="collapsed")
    top_r.button("Clear history", on_click=_clear_history, type="tertiary", icon=":material/delete:",
                 width="stretch")
    shown = 0
    for i, e in enumerate(history):
        if flt and flt.lower() not in e.get("q", "").lower():
            continue
        shown += 1
        with st.container(border=True):
            q = e.get("q", "")
            st.markdown(f"**{_e(q[:300])}{'…' if len(q) > 300 else ''}**", unsafe_allow_html=True)
            when = e.get("t", "")[:16].replace("T", " ")
            dates = f" · published {e.get('start') or '…'} to {e.get('end') or '…'}" if (e.get("start") or e.get("end")) else ""
            c1, c2, c3 = st.columns([4, 1.4, 0.9], vertical_alignment="center")
            c1.caption(f"{when} · {e.get('n', 0)} results · {_sources_text(e.get('sources'))}{dates}")
            c2.button("Search again", key=f"hist_run_{i}", on_click=_run_again, args=(e,), icon=":material/search:",
                      type="tertiary", width="stretch")
            c3.button("Remove", key=f"hist_rm_{i}", on_click=_remove_history, args=(i,), type="tertiary",
                      width="stretch")
    if not shown:
        st.caption("No saved search contains that text.")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

SENSITIVITY = {
    "focused": ("Focused", "Only the closest matches (papers that would rank in the top 500 for this search)."),
    "balanced": ("Balanced", "Close matches (top 2,000). A good default."),
    "broad": ("Broad", "Anything clearly related (top 10,000). Can be several papers a day for busy topics."),
}

_RSS_HELP = ("Each alert is a private RSS feed. Paste its link into a feed reader such as Feedly, Inoreader, "
             "Thunderbird or Outlook (Add a feed), or into Slack with `/feed subscribe <link>`. New papers "
             "are matched every night after the databases update. Anyone with the link can read the feed, so "
             "share it only on purpose. Alerts nobody reads for 6 months are removed.")


def _visitor_ip() -> str:
    """The visitor's address as nginx passed it (X-Real-IP), for the per-visitor alert limits."""
    try:
        return st.context.headers.get("X-Real-Ip") or ""
    except Exception:
        return ""


def render_create_alert(query: str, sources):
    """'Create alert' popover for the current search."""
    with st.popover("Create alert", icon=":material/notifications:",
                    help="Get new papers matching this search as an RSS feed"):
        st.markdown("Get **new papers** that match this search as they are added (checked nightly).")
        sens = st.radio("How close should they be?", list(SENSITIVITY), index=1, key="alert_sensitivity",
                        format_func=lambda k: SENSITIVITY[k][0], captions=[v[1] for v in SENSITIVITY.values()])
        if st.button("Save alert", type="primary", icon=":material/notifications_active:", width="stretch"):
            if st.session_state.get("prefs") is None:
                st.warning("Your browser storage is not available, so the alert could not be remembered here.")
                return
            try:
                with st.spinner("Saving…"):
                    res = backend_client.create_alert(query, sources, sens, _visitor_ip())
            except backend_client.BackendError as exc:
                st.error(str(exc))
                return
            p = prefs()
            p.setdefault("alerts", []).insert(0, {
                "token": res["token"], "query": query[:500], "sources": list(sources or []),
                "sensitivity": sens, "created": datetime.now().isoformat(timespec="minutes"),
                "feed_url": res["feed_url"]})
            _save()
            st.session_state["alert_created"] = res
        res = st.session_state.get("alert_created")
        if res:
            st.success("Alert saved. Copy its feed link into your feed reader:")
            st.code(res["feed_url"], language=None, wrap_lines=True)
            per_month = res.get("per_month_estimate") or 0
            st.caption(f"At this setting, about {per_month:g} papers a month were published last year. "
                       f"{res.get('seeded', 0)} recent papers are already in the feed. "
                       "Manage your alerts under Alerts at the top.")


@st.cache_data(ttl=300, show_spinner=False)
def _alert_info(token: str):
    return backend_client.get_alert(token)


def _delete_alert(token: str):
    backend_client.delete_alert(token)
    p = prefs()
    p["alerts"] = [a for a in p.get("alerts", []) if a.get("token") != token]
    _alert_info.clear()
    _save()


def _forget_alert(token: str):
    p = prefs()
    p["alerts"] = [a for a in p.get("alerts", []) if a.get("token") != token]
    _save()


def _import_alert():
    import re
    text = (st.session_state.get("alert_import") or "").strip()
    m = re.search(r"([A-Za-z0-9_-]{16,64})(?:/feed\.xml)?/?$", text)
    st.session_state["alert_import"] = ""
    if not m:
        st.session_state["alert_import_msg"] = "That does not look like an alert feed link."
        return
    token = m.group(1)
    try:
        info = backend_client.get_alert(token)
    except backend_client.BackendError as exc:
        st.session_state["alert_import_msg"] = str(exc)
        return
    if info is None:
        st.session_state["alert_import_msg"] = "No alert with that link exists (it may have expired)."
        return
    p = prefs()
    if any(a.get("token") == token for a in p.get("alerts", [])):
        st.session_state["alert_import_msg"] = "That alert is already in your list."
        return
    p.setdefault("alerts", []).insert(0, {
        "token": token, "query": info["query"][:500], "sources": info.get("sources", []),
        "sensitivity": info.get("sensitivity"), "created": datetime.fromtimestamp(info["created"]).isoformat(timespec="minutes"),
        "feed_url": info["feed_url"]})
    st.session_state["alert_import_msg"] = None
    _save()


def render_alerts():
    if st.session_state.get("prefs") is None:
        st.info("Loading your alerts…")
        return
    alerts = prefs().get("alerts", [])
    st.caption(_RSS_HELP)
    if not alerts:
        st.info("No alerts yet. Run a search, then choose **Create alert** above the results.")
    for a in alerts:
        token = a.get("token", "")
        with st.container(border=True):
            st.markdown(f"**{_e(a.get('query', ''))}**", unsafe_allow_html=True)
            label = SENSITIVITY.get(a.get("sensitivity"), ("",))[0]
            st.caption(f"{label} · {_sources_text(a.get('sources'))} · created {a.get('created', '')[:10]}")
            try:
                info = _alert_info(token)
            except backend_client.BackendError as exc:
                st.warning(str(exc))
                continue
            if info is None:
                st.warning("This alert no longer exists on the server (unused for 6 months, or deleted).")
                st.button("Remove from this list", key=f"forget_{token}", on_click=_forget_alert, args=(token,),
                          type="tertiary")
                continue
            st.code(info["feed_url"], language=None, wrap_lines=True)
            items = info.get("items", [])
            with st.expander(f"{len(items)} paper{'s' if len(items) != 1 else ''} in the feed" if items
                             else "No papers yet. New matches arrive with the nightly update."):
                for it in items[:20]:
                    url = _safe_url(it.get("url") or (f"https://doi.org/{it['doi']}" if it.get("doi") else ""))
                    title = _e(it.get("title") or "Untitled")
                    meta = " · ".join(_e(x) for x in (it.get("journal"), it.get("date"), SOURCE_NAMES.get(it.get("source"), it.get("source"))) if x)
                    st.markdown((f'<a href="{url}" target="_blank" rel="noopener">{title}</a>' if url else title)
                                + f'<br><span style="font-size:.85em;opacity:.75">{meta}</span>',
                                unsafe_allow_html=True)
            c1, c2, _ = st.columns([1.3, 1.1, 2])
            c1.button("Search again", key=f"alert_run_{token}", on_click=_run_again,
                      args=({"q": a.get("query", ""), "sources": a.get("sources")},),
                      icon=":material/search:", type="tertiary", width="stretch")
            c2.button("Delete alert", key=f"alert_del_{token}", on_click=_delete_alert, args=(token,),
                      icon=":material/delete:", type="tertiary", width="stretch")
    with st.expander("Add an alert from another device"):
        st.text_input("Paste the alert's feed link", key="alert_import", on_change=_import_alert,
                      placeholder="https://manuscript-search.org/v1/alerts/…/feed.xml")
        if st.session_state.get("alert_import_msg"):
            st.warning(st.session_state["alert_import_msg"])
