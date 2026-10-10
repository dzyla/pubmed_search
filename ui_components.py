"""
Presentation for the Streamlit UI: page styling, header, query fingerprint,
result entries, chart and exports. Fonts and palette come from
.streamlit/config.toml (H&E stain: hematoxylin ink, eosin accent).
All data-derived text is HTML-escaped before it is rendered.
"""
import html
import logging
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import paper_links

LOGGER = logging.getLogger(__name__)

INK, EOSIN, MUTED, RULE, GLASS = "#2A2250", "#C8336B", "#6D6884", "#DDD9E8", "#F4F3F8"
# Methyl green (a classic counterstain) for trials, keeping the histology palette.
SOURCE_COLORS = {"PubMed": INK, "BioRxiv": EOSIN, "MedRxiv": "#7A6FB0", "arXiv": "#B7832F",
                 "ClinicalTrials": "#3E7D63", "Preprints": "#A0507A", "Grants": "#56708F",
                 "OpenAlex": "#8C5A3C"}
SOURCE_NAMES = {"PubMed": "PubMed", "BioRxiv": "bioRxiv", "MedRxiv": "medRxiv", "arXiv": "arXiv",
                "ClinicalTrials": "ClinicalTrials.gov", "Preprints": "Other preprints", "OpenAlex": "Other journals",
                "Grants": "NIH grants"}
_NO_CITATIONS = {"arXiv", "ClinicalTrials", "Grants"}   # no Crossref DOI to count citations for
_SERVER_NAMES = {"biorxiv": "bioRxiv", "medrxiv": "medRxiv"}

_CSS = f"""
<style>
  .block-container {{ padding-top: 4.5rem; max-width: 46rem; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  /* "Read the full abstract": a quiet text toggle aligned with the entry text */
  [class*="st-key-abstract_"] {{ margin-left: 2.6rem; }}
  [class*="st-key-abstract_"] details {{ border: none !important; background: transparent; }}
  [class*="st-key-abstract_"] summary {{ padding: 0.1rem 0 !important; font-size: 0.86rem; color: {MUTED}; }}
  [class*="st-key-abstract_"] summary:hover {{ color: {EOSIN}; }}
  [class*="st-key-abstract_"] [data-testid="stExpanderDetails"] {{ padding: 0.2rem 0 0.4rem 0; }}

  .mss-head h1 {{
    font-family: "Newsreader", serif; font-weight: 500; font-size: 3rem;
    letter-spacing: -0.02em; line-height: 1; margin: 0 0 0.6rem 0; padding: 0; color: {INK};
  }}
  .mss-head h1 a {{ color: inherit !important; text-decoration: none; }}
  .mss-head h1 a:hover {{ color: {EOSIN} !important; }}
  .mss-head h1 a:focus-visible {{ outline: 2px solid {EOSIN}; outline-offset: 4px; border-radius: 2px; }}
  .mss-head p {{ margin: 0; color: {MUTED}; font-size: 1.02rem; line-height: 1.5; max-width: 44rem; }}
  .mss-head p strong {{ color: {INK}; font-weight: 600; }}
  .mss-head .mss-fresh {{ font-size: 0.82rem; margin-top: 0.35rem; }}
  .mss-head {{ margin-bottom: 1.6rem; }}

  .mss-print {{ margin: 0.4rem 0 1.4rem 0; }}
  .mss-print svg {{ display: block; max-width: 100%; height: auto; }}
  .mss-print figcaption {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.35rem; }}

  .mss-entry {{
    display: grid; grid-template-columns: 2.2rem 1fr; column-gap: 0.4rem;
    padding: 1.1rem 0 0.2rem 0; border-top: 1px solid {RULE};
  }}
  .mss-rank {{
    font-family: "Newsreader", serif; font-size: 1.35rem; color: {EOSIN};
    line-height: 1.25; font-variant-numeric: tabular-nums;
  }}
  .mss-title {{
    font-family: "Newsreader", serif; font-size: 1.22rem; font-weight: 500; line-height: 1.3;
    color: {INK} !important; text-decoration: none;
  }}
  .mss-title:hover {{ text-decoration: underline; text-decoration-color: {EOSIN}; text-underline-offset: 3px; }}
  .mss-cite {{ margin: 0.3rem 0 0 0; font-size: 0.88rem; color: {MUTED}; line-height: 1.45; }}
  .mss-cite em {{ color: {INK}; }}
  .mss-facts {{ margin: 0.45rem 0 0 0; display: flex; flex-wrap: wrap; gap: 0.35rem 0.5rem;
               align-items: center; font-size: 0.8rem; color: {MUTED}; }}
  .mss-src {{ font-weight: 600; }}
  .mss-tag {{ border: 1px solid {RULE}; border-radius: 0.3rem; padding: 0.05rem 0.4rem; color: {INK}; }}
  .mss-exact {{ border-color: {EOSIN}; color: {EOSIN}; font-weight: 600; }}

  /* Per-result actions (select / similar), aligned with the entry text */
  [class*="st-key-actions_"] {{ margin-left: 2.6rem; margin-top: -0.2rem; }}
  [class*="st-key-actions_"] label p, [class*="st-key-actions_"] button p {{ font-size: 0.84rem; }}
  .mss-seeds {{ margin: 0 0 0.8rem 0; padding-left: 1.1rem; color: {MUTED}; font-size: 0.92rem; }}
  .mss-seeds a {{ font-family: "Newsreader", serif; font-size: 1.02rem; color: {INK} !important; }}
  .mss-retracted {{ display: inline-block; background: #B42318; color: #fff; border-radius: 0.3rem;
                    padding: 0.05rem 0.45rem; font-family: "Instrument Sans", sans-serif;
                    font-size: 0.75rem; font-weight: 600; margin-right: 0.4rem; vertical-align: 0.15em; }}
  .mss-meter {{ display: inline-block; width: 4.5rem; height: 0.3rem; background: {RULE};
               border-radius: 0.2rem; overflow: hidden; vertical-align: middle; }}
  .mss-meter span {{ display: block; height: 100%; background: {EOSIN}; }}
  .mss-snippet {{ font-family: "Newsreader", serif; font-size: 1rem; line-height: 1.6;
                 color: #3d3660; margin: 0.55rem 0 0 0; max-width: 40rem; }}
  .mss-links {{ margin: 0.5rem 0 0 0; display: flex; flex-wrap: wrap; gap: 0.2rem 1rem; font-size: 0.86rem; }}
  .mss-links a {{ font-weight: 500; }}
  .mss-abstract {{ font-family: "Newsreader", serif; font-size: 1.02rem; line-height: 1.65;
                  color: {INK}; max-width: 40rem; }}

  /* The map's doorway on the landing page */
  .mss-mapcard {{ position: relative; display: block; margin: 2.2rem 0 0.4rem 0; border-radius: 6px;
                 overflow: hidden; background: #14102A; text-decoration: none !important; }}
  .mss-mapcard img {{ display: block; width: 100%; aspect-ratio: 2.6; object-fit: cover;
                     transition: transform 0.6s ease, filter 0.6s ease; filter: brightness(0.92); }}
  .mss-mapcard:hover img, .mss-mapcard:focus-visible img {{ transform: scale(1.025); filter: brightness(1.05); }}
  .mss-mapcard span {{ position: absolute; left: 1.1rem; bottom: 0.9rem; color: #fff;
                      font-family: "Newsreader", serif; font-size: 1.35rem; font-style: italic;
                      text-shadow: 0 1px 8px rgba(0,0,0,0.75); }}
  .mss-mapcard:focus-visible {{ outline: 2px solid {EOSIN}; outline-offset: 3px; }}
  .mss-mapnote {{ font-size: 0.85rem; color: {MUTED}; margin: 0; }}
  .mss-mapclose {{ display: block; text-align: right; font-size: 0.88rem; padding-bottom: 0.7rem; }}
  @media (prefers-reduced-motion: reduce) {{ .mss-mapcard img {{ transition: none; }} }}

  .mss-foot {{ font-size: 0.82rem; color: {MUTED}; line-height: 1.6; margin-top: 2.5rem;
              padding-top: 1rem; border-top: 1px solid {RULE}; }}
  .mss-foot a {{ font-weight: 500; }}

  @media (max-width: 640px) {{
    .mss-head h1 {{ font-size: 2.3rem; }}
    .mss-entry {{ grid-template-columns: 1.7rem 1fr; }}
  }}
</style>
"""


def _e(value) -> str:
    """Escapes data for HTML; NaN/None become empty."""
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    return html.escape(text.strip(), quote=True)


def _safe_url(url) -> str:
    url = str(url or "")
    return html.escape(url, quote=True) if url.startswith(("https://", "http://")) else ""


def apply_style():
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _fmt_date(value) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%-d %b %Y")
    except ValueError:
        return ""


def _round_millions(n: int) -> str:
    return f"{n / 1e6:.1f} million" if n >= 1e6 else f"{n:,}"


def render_header(stats: dict, active_users: int):
    total = stats.get("total_papers", 0)
    lead = (f"Search <strong>{_round_millions(total)}</strong> papers, preprints, clinical trials and "
            f"grants by meaning, not keywords." if total else
            "Search papers, preprints, clinical trials and grants by meaning, not keywords.")
    people = f"{active_users} {'person' if active_users == 1 else 'people'} searching now"
    # The title starts a fresh search (a full reload drops the query and results).
    st.markdown(
        f"""<header class="mss-head"><h1><a href="/" target="_self" title="Start a new search">Manuscript Search</a></h1>
        <p>{lead}</p><p class="mss-fresh">{people}</p></header>""",
        unsafe_allow_html=True,
    )


def freshness_line(stats: dict) -> str:
    fresh = [f"{SOURCE_NAMES.get(s, s)} {_fmt_date(i.get('updated'))}"
             for s, i in stats.get("sources", {}).items() if _fmt_date(i.get("updated"))]
    return ("Last updated: " + ", ".join(fresh) + ".") if fresh else ""


# ---------------------------------------------------------------------------
# Query fingerprint — the one decorative element, and it is real data
# ---------------------------------------------------------------------------

def fingerprint_svg(bits_hex: str, cell: int = 6, gap: int = 1) -> str:
    """384-bit query code as a 48 x 8 grid (one column per byte)."""
    try:
        data = bytes.fromhex(bits_hex)
    except (TypeError, ValueError):
        return ""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    cols, rows = len(data), 8
    step = cell + gap
    rects = [
        f'<rect x="{c * step}" y="{r * step}" width="{cell}" height="{cell}" rx="1" '
        f'fill="{EOSIN if bits[c * 8 + r] else RULE}"/>'
        for c in range(cols) for r in range(rows)
    ]
    w, h = cols * step - gap, rows * step - gap
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="Binary code of your query, {len(bits)} bits">{"".join(rects)}</svg>')


def render_seeds(seeds: list):
    """List of the example papers a similarity search was based on."""
    items = []
    for sd in seeds:
        url = _safe_url(sd.get("url"))
        title = _e(sd.get("title")) or "Untitled"
        src = _e(SOURCE_NAMES.get(sd.get("source"), sd.get("source", "")))
        link = f'<a href="{url}" target="_blank" rel="noopener">{title}</a>' if url else title
        items.append(f"<li>{link} <span>({src}{', ' + _e(sd.get('date')) if sd.get('date') else ''})</span></li>")
    st.markdown(f'<ul class="mss-seeds">{"".join(items)}</ul>', unsafe_allow_html=True)


def render_fingerprint(bits_hex: str, similar: bool = False):
    svg = fingerprint_svg(bits_hex)
    if svg:
        lead = ("The example papers combined into one code" if similar
                else "Your query as the index reads it")
        st.markdown(
            f'<figure class="mss-print">{svg}<figcaption>{lead}: '
            f'{len(bits_hex) * 4} bits, compared against the code of every paper.</figcaption></figure>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Result entry
# ---------------------------------------------------------------------------

def _authors_short(authors: str, keep: int = 6) -> str:
    names = [a.strip() for a in re.split(r"\s*[;,]\s*(?=[A-Z])|\s*;\s*", str(authors or "")) if a.strip()]
    if not names or names == ["N/A"]:
        return ""
    text = ", ".join(names[:keep]) + (", et al" if len(names) > keep else "")
    return text.rstrip(".") + "."


# Structured-abstract headings that bioRxiv/medRxiv deliver glued to the text
# ("BackgroundHydroxychloroquine is…").
_SECTION_GLUE = re.compile(
    r"\b(Background|Objectives?|Aims?|Purpose|Importance|Introduction|Methods?|Design|Setting|"
    r"Participants|Interventions|Main outcome measures|Data sources|Study selection|Results|"
    r"Findings|Conclusions?|Interpretation|Significance|Funding)(?=[A-Z][a-z])"
)


def tidy_abstract(text: str) -> str:
    return _SECTION_GLUE.sub(r"\1: ", str(text or ""))


def _links(row) -> list:
    links = row.get("links")
    if isinstance(links, list) and links:
        return [(link["label"], link["url"]) for link in links]
    return paper_links.build_links(row)


def render_entry(row, rank: int, citations, top_score: float, term_display: dict = None):
    links = _links(row)
    url = _safe_url(links[0][1]) if links else ""
    title = _e(row.get("title")) or "Untitled"
    title_html = f'<a class="mss-title" href="{url}" target="_blank" rel="noopener">{title}</a>' if url \
        else f'<span class="mss-title">{title}</span>'
    retracted = bool(row.get("retracted"))
    if retracted:
        title_html = f'<span class="mss-retracted">Retracted</span>{title_html}'

    journal = _e(row.get("journal"))
    journal = "" if journal in ("N/A", "nan") else _SERVER_NAMES.get(journal.lower(), journal)
    when = _e(row.get("date"))
    if str(row.get("source")) == "Grants":
        pis = _e(_authors_short(row.get("authors"), keep=3)).rstrip(".")
        cite = (f"{pis + '. ' if pis else ''}<em>{journal or 'NIH RePORTER'}</em> "
                f"{_e(row.get('registry_id') or row.get('grant_id') or '')}{', funded ' + when if when else ''}.")
    elif str(row.get("source")) == "ClinicalTrials":
        nct = _e(row.get("registry_id") or row.get("nct_id")) or (_e(row.get("url", "")).rsplit("/", 1)[-1])
        sponsor = _e(row.get("authors"))
        cite = (f"{sponsor + '. ' if sponsor and sponsor != 'N/A' else ''}<em>ClinicalTrials.gov</em> "
                f"{nct}{', registered ' + when if when else ''}.")
    else:
        cite = None
    cite = cite or " ".join(p for p in (
        _e(_authors_short(row.get("authors"))),
        f"<em>{journal}</em>," if journal and when else (f"<em>{journal}</em>." if journal else ""),
        f"{when}." if when else "",
    ) if p)

    source = str(row.get("source", ""))
    labels = [lab for lab in (row.get("labels") or []) if lab != "Retracted"]
    score = float(row.get("score") or 0)
    pct = max(4, min(100, round(100 * (score - 0.5) / max(top_score - 0.5, 1e-6))))
    facts = [f'<span class="mss-src" style="color:{SOURCE_COLORS.get(source, INK)}">'
             f'{_e(SOURCE_NAMES.get(source, source))}</span>']
    matched = row.get("matched_terms")
    if isinstance(matched, (list, tuple, np.ndarray)) and len(matched):
        shown = ", ".join((term_display or {}).get(t, t) for t in matched)
        facts.append(f'<span class="mss-tag mss-exact" title="Contains these terms from your query">'
                     f'matches {_e(shown)}</span>')
    facts += [f'<span class="mss-tag">{_e(lab)}</span>' for lab in labels]
    if source in _NO_CITATIONS:
        pass
    elif citations is None:
        facts.append("<span>counting citations…</span>")
    else:
        facts.append(f"<span>cited {int(citations):,} time{'s' if citations != 1 else ''}</span>")
    facts.append(f'<span class="mss-meter" title="Relevance relative to the best match '
                 f'(similarity {score:.3f})" aria-hidden="true"><span style="width:{pct}%"></span></span>')

    abstract = tidy_abstract(row.get("abstract"))
    snippet = abstract if len(abstract) <= 320 else abstract[:300].rsplit(" ", 1)[0] + " …"
    link_html = "".join(
        f'<a href="{_safe_url(u)}" target="_blank" rel="noopener">{_e(label)}</a>'
        for label, u in links if _safe_url(u)
    )
    st.markdown(
        f"""<article class="mss-entry"><div class="mss-rank">{rank}</div><div>
        {title_html}
        <p class="mss-cite">{cite}</p>
        <div class="mss-facts">{''.join(facts)}</div>
        {f'<p class="mss-snippet">{_e(snippet)}</p>' if snippet and snippet != 'N/A' else ''}
        <div class="mss-links">{link_html}</div>
        </div></article>""",
        unsafe_allow_html=True,
    )
    if len(abstract) > 320:
        with st.container(key=f"abstract_{rank}"), st.expander("Read the full abstract"):
            st.markdown(f'<div class="mss-abstract">{_e(abstract)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_score_vs_year(sorted_results: pd.DataFrame) -> go.Figure:
    """Publication date × relevance, sized by citations, coloured by database."""
    try:
        df = sorted_results.copy()
        df["Date_Parsed"] = pd.to_datetime(df["date"], errors="coerce")
        min_valid = df["Date_Parsed"].min()
        if pd.isnull(min_valid):
            min_valid = pd.Timestamp.now()
        df["Date_Plot"] = df["Date_Parsed"].fillna(min_valid)
        df["citations"] = pd.to_numeric(df.get("citations", 0), errors="coerce").fillna(0)
        df["marker_size"] = np.log1p(df["citations"]) * 5 + 6
        df["Database"] = df["source"].map(SOURCE_NAMES).fillna(df["source"])

        fig = px.scatter(
            df, x="Date_Plot", y="score", size="marker_size", color="Database",
            hover_name="title",
            hover_data={"Date_Plot": False, "score": ":.3f", "citations": True,
                        "marker_size": False, "Database": False},
            labels={"score": "Relevance", "Date_Plot": "Published"},
            color_discrete_map={SOURCE_NAMES[k]: v for k, v in SOURCE_COLORS.items()},
        )
        for trace in fig.data:
            trace.update(
                marker=dict(line=dict(width=0), opacity=0.85),
                hovertemplate=("<b>%{hovertext}</b><br>Relevance %{y:.3f}<br>"
                               "Published %{x|%Y-%m-%d}<br>Cited %{customdata[0]} times"
                               f"<extra>{trace.name}</extra>"),
            )
        fig.update_layout(
            font=dict(family="Instrument Sans, sans-serif", color=INK, size=13),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
            hoverlabel=dict(bgcolor="white", bordercolor=RULE, font=dict(color=INK)),
        )
        fig.update_xaxes(showgrid=False, linecolor=RULE, title=None)
        fig.update_yaxes(gridcolor=RULE, zeroline=False)
        return fig
    except Exception as e:
        LOGGER.error(f"Plot error: {e}")
        return go.Figure()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _row_url(row):
    url = row.get("url")
    return url if isinstance(url, str) and url else paper_links.primary_link(row)


def generate_bibtex(df: pd.DataFrame) -> str:
    """Generates a BibTeX string from a results DataFrame."""
    entries = []
    for idx, row in df.iterrows():
        authors = str(row.get("authors", "Unknown")).replace(";", " and")
        authors = authors.replace("[", "").replace("]", "").replace("'", "")
        year = paper_links.year_of(row) or "n.d."
        first_author = re.sub(r"\W+", "", authors.split(" ")[0].split(",")[0].strip()) or "Anon"
        title = str(row.get("title", "No Title"))
        title_words = re.sub(r"[^\w\s]", "", title).split()
        first_word = title_words[0] if title_words else "Untitled"
        key = f"{first_author}{year}{first_word}_{idx}"

        source = str(row.get("source", "")).lower()
        journal = str(row.get("journal", ""))
        if journal.lower() in ("nan", "", "n/a"):
            journal = source.capitalize()

        if source in ("clinicaltrials", "grants"):
            url = _row_url(row) or ""
            entries.append(f"@misc{{{key},\n  author = {{{authors}}},\n  title = {{{title}}},\n"
                           f"  howpublished = {{{journal}, {row.get('nct_id') or row.get('grant_id') or ''}}},\n"
                           f"  year = {{{year}}},\n  url = {{{url}}},\n}}\n")
            continue

        entry = f"@article{{{key},\n"
        entry += f"  author = {{{authors}}},\n"
        entry += f"  title = {{{title}}},\n"
        entry += f"  journal = {{{journal}}},\n"
        entry += f"  year = {{{year}}},\n"
        clean = paper_links.linkable_doi(row.get("doi"))
        if clean:
            entry += f"  doi = {{{clean}}},\n"
            entry += f"  url = {{https://doi.org/{clean}}},\n"
        entry += "}\n"
        entries.append(entry)
    return "\n".join(entries)


def generate_ris(df: pd.DataFrame) -> str:
    """RIS export (Zotero, EndNote, Mendeley)."""
    records = []
    for _, row in df.iterrows():
        source = str(row.get("source", ""))
        ris_type = {"BioRxiv": "UNPB", "MedRxiv": "UNPB", "arXiv": "UNPB", "Preprints": "UNPB",
                    "ClinicalTrials": "GEN", "Grants": "GRANT"}.get(source, "JOUR")
        lines = [f"TY  - {ris_type}"]
        lines.append(f"TI  - {row.get('title', '')}")
        for author in re.split(r"\s*;\s*", str(row.get("authors") or "")):
            if author and author != "N/A":
                lines.append(f"AU  - {author}")
        journal = str(row.get("journal") or "")
        if journal and journal.lower() not in ("nan", "none", "n/a"):
            lines.append(f"JO  - {journal}")
        year = paper_links.year_of(row)
        if year:
            lines.append(f"PY  - {year}")
        if row.get("date"):
            lines.append(f"DA  - {str(row['date']).replace('-', '/')}")
        doi = paper_links.linkable_doi(row.get("doi"))
        if doi:
            lines.append(f"DO  - {doi}")
        url = _row_url(row)
        if url:
            lines.append(f"UR  - {url}")
        abstract = str(row.get("abstract") or "")
        if abstract and abstract.lower() not in ("nan", "n/a"):
            lines.append(f"AB  - {abstract}")
        lines.append("ER  - ")
        records.append("\n".join(lines))
    return "\n\n".join(records) + "\n"


def results_csv(df: pd.DataFrame) -> str:
    out = df.copy()
    out["url"] = [_row_url(row) for _, row in out.iterrows()]
    if "labels" in out.columns:
        out["labels"] = out["labels"].map(lambda v: "; ".join(v) if isinstance(v, list) else v)
    cols = [c for c in ("rank", "title", "authors", "journal", "date", "source", "doi", "pmid",
                        "url", "labels", "citations", "score", "abstract") if c in out.columns]
    return out[cols].to_csv(index=False)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def render_footer(freshness: str = ""):
    st.markdown(
        f"""<footer class="mss-foot">{_e(freshness) + "<br>" if freshness else ""}
        Built by Dawid Zyla at the
        <a href="https://zylalab.org" target="_blank" rel="noopener">Zyla Lab (zylalab.org)</a>.
        Not affiliated with PubMed, bioRxiv, medRxiv, arXiv or ClinicalTrials.gov.
        Search from your own code or AI agent with the
        <a href="/docs" target="_blank" rel="noopener">REST API and MCP endpoint</a>
        (<a href="/signup" target="_blank" rel="noopener">get a free API key</a>).
        <a href="https://www.buymeacoffee.com/dzyla" target="_blank" rel="noopener">Support the server costs</a>.
        </footer>""",
        unsafe_allow_html=True,
    )
