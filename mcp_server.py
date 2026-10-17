"""
MCP (Model Context Protocol) tools for AI agents.

build_mcp(search, stats) returns a FastMCP server whose tools call the given
async functions. The backend (search_api.py) mounts it at /mcp with in-process
search; mcp_stdio.py runs the same tools over stdio against the public REST API.
"""
from datetime import date
from typing import Any, Awaitable, Callable, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

SOURCES = ("PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints", "Grants", "OpenAlex")

INSTRUCTIONS = (
    "Manuscript Search: semantic search over ~50 million scientific abstracts, "
    "clinical trial registrations and NIH grant abstracts from PubMed, bioRxiv, "
    "medRxiv, arXiv, other preprint servers (Research Square, Preprints.org, ChemRxiv, "
    "PsyArXiv, ...), biology and medicine journals outside PubMed (via OpenAlex), "
    "ClinicalTrials.gov and NIH RePORTER. Grants are searched only "
    "when sources includes 'Grants'. Describe what you are looking for in natural "
    "language (a sentence, a research question, or a pasted abstract); bare keyword "
    "lists work worse. Each result has title, authors, date, journal, abstract, a "
    "relevance score, links (DOI, PubMed, PDF) and labels such as Retracted, Review, "
    "Meta-analysis, RCT or Preprint; trials carry their phase and recruitment status. "
    "Always surface the 'Retracted' label to the user."
)


class SearchFailed(Exception):
    """Raised by search callables with a message that is safe to show to the agent."""


SearchFn = Callable[..., Awaitable[dict]]
StatsFn = Callable[[], Awaitable[dict]]


def _parse_date(value: Optional[str], name: str) -> Optional[str]:
    if not value:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ToolError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc


def build_mcp(search: SearchFn, stats: StatsFn, similar: SearchFn = None, max_top_k: int = 10,
              library: SearchFn = None, citations: SearchFn = None) -> FastMCP:
    mcp = FastMCP(
        "Manuscript Search",
        instructions=INSTRUCTIONS,
        website_url="https://manuscript-search.org",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        # Requests are authenticated by API key and arrive through nginx with the
        # public Host header, so the localhost-only DNS-rebinding guard is off.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @mcp.tool()
    async def search_papers(
        query: str,
        top_k: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sources: Optional[list[str]] = None,
        include_abstracts: bool = True,
    ) -> dict[str, Any]:
        """
        Semantic search for scientific papers.

        Args:
            query: What you are looking for, in natural language (3-2000 characters).
            top_k: Number of results, 1-10.
            start_date: Only papers published on/after this date (YYYY-MM-DD).
            end_date: Only papers published on/before this date (YYYY-MM-DD).
            sources: Subset of ["PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints",
                "OpenAlex", "Grants"]; all except Grants if omitted. OpenAlex = biology and medicine
                journals outside PubMed, including meeting abstracts.
            include_abstracts: Set false for a compact list (titles, links, labels only).
        """
        if not 3 <= len(query.strip()) <= 2000:
            raise ToolError("query must be 3-2000 characters")
        if not 1 <= top_k <= max_top_k:
            raise ToolError(f"top_k must be between 1 and {max_top_k}")
        start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
        if start and end and start > end:
            raise ToolError("start_date must not be after end_date")
        if sources:
            lookup = {s.lower(): s for s in SOURCES}
            unknown = [s for s in sources if s.lower() not in lookup]
            if unknown:
                raise ToolError(f"unknown source(s) {unknown}; choose from {list(SOURCES)}")
            sources = [lookup[s.lower()] for s in sources]
        try:
            result = await search(query=query.strip(), top_k=top_k, start_date=start,
                                  end_date=end, sources=sources)
        except SearchFailed as exc:
            raise ToolError(str(exc)) from exc
        if not include_abstracts:
            for paper in result.get("results", []):
                paper.pop("abstract", None)
        return result

    if similar is not None:
        @mcp.tool()
        async def find_similar(
            paper_refs: list[str],
            top_k: int = 10,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            sources: Optional[list[str]] = None,
            include_abstracts: bool = True,
        ) -> dict[str, Any]:
            """
            Papers similar to one or more example papers ("more like these").

            Args:
                paper_refs: 1-20 'ref' values from search_papers results, e.g. ["PubMed:123456"].
                    Several examples are combined into one query.
                top_k: Number of results, 1-10.
                start_date / end_date: Optional publication date range (YYYY-MM-DD).
                sources: Optional subset of databases to search.
                include_abstracts: Set false for a compact list.
            """
            if not 1 <= len(paper_refs) <= 20:
                raise ToolError("paper_refs must contain 1-20 references")
            if not 1 <= top_k <= max_top_k:
                raise ToolError(f"top_k must be between 1 and {max_top_k}")
            start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
            try:
                result = await similar(refs=paper_refs, top_k=top_k, start_date=start, end_date=end,
                                       sources=sources)
            except SearchFailed as exc:
                raise ToolError(str(exc)) from exc
            if not include_abstracts:
                for paper in result.get("results", []) + result.get("seeds", []):
                    paper.pop("abstract", None)
            return result

    if library is not None:
        @mcp.tool()
        async def find_papers_like(
            paper_ids: list[str],
            top_k: int = 10,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            include_abstracts: bool = False,
        ) -> dict[str, Any]:
            """
            Papers like a set of known papers (e.g. a user's library or a reference list) that are
            not in the set. The papers are grouped into themes and each theme is searched; every
            result's 'theme' indexes the response's 'themes'. 'not_found' lists ids not in the index.

            Args:
                paper_ids: 1-500 DOIs, PubMed ids or arXiv ids, e.g. ["10.1038/s41586-020-2012-7", "32015507"].
                top_k: Number of results, 1-10.
                start_date / end_date: Optional publication date range (YYYY-MM-DD).
                include_abstracts: Set true to include abstracts.
            """
            if not 1 <= len(paper_ids) <= 500:
                raise ToolError("paper_ids must contain 1-500 identifiers")
            if not 1 <= top_k <= max_top_k:
                raise ToolError(f"top_k must be between 1 and {max_top_k}")
            start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
            try:
                result = await library(ids=[str(i) for i in paper_ids], top_k=top_k, start_date=start,
                                       end_date=end)
            except SearchFailed as exc:
                raise ToolError(str(exc)) from exc
            for paper in result.get("results", []) + result.get("seeds", []):
                if not include_abstracts:
                    paper.pop("abstract", None)
            return result

    if citations is not None:
        @mcp.tool()
        async def get_citations(
            paper: str,
            direction: str = "cited_by",
            top_k: int = 10,
            include_abstracts: bool = False,
        ) -> dict[str, Any]:
            """
            Papers that cite a paper ('cited_by', most cited first) or that it cites
            ('references', newest first), from OpenAlex. Papers in this index carry a 'ref' for
            find_similar; others have in_index=false. total_results is the full count.

            Args:
                paper: A result 'ref' (e.g. "PubMed:123456"), a DOI, or a PubMed id.
                direction: "cited_by" or "references".
                top_k: Number of papers, 1-10.
                include_abstracts: Set true to include abstracts of indexed papers.
            """
            if direction not in ("cited_by", "references"):
                raise ToolError("direction must be 'cited_by' or 'references'")
            if not 1 <= top_k <= max_top_k:
                raise ToolError(f"top_k must be between 1 and {max_top_k}")
            paper = paper.strip()
            kind = ("ref" if ":" in paper and paper.split(":", 1)[0] in SOURCES
                    else "pmid" if paper.isdigit() else "doi")
            try:
                result = await citations(**{kind: paper}, direction=direction, top_k=top_k)
            except SearchFailed as exc:
                raise ToolError(str(exc)) from exc
            for p in result.get("results", []) + result.get("seeds", []):
                if not include_abstracts:
                    p.pop("abstract", None)
            return result

    @mcp.tool()
    async def database_info() -> dict[str, Any]:
        """Which databases are searchable, how many papers each holds, and when each was last updated."""
        try:
            return await stats()
        except SearchFailed as exc:
            raise ToolError(str(exc)) from exc

    return mcp
