"""
Local MCP bridge (stdio) for agents that cannot connect to the remote /mcp
endpoint. Exposes the same tools, backed by the public REST API.

Claude Desktop / Claude Code config:
    {
      "mcpServers": {
        "manuscript-search": {
          "command": "python",
          "args": ["/path/to/mcp_stdio.py"],
          "env": {"MSS_API_KEY": "<your key>"}
        }
      }
    }

Environment:
    MSS_API_KEY   your API key (optional; without one the anonymous daily limit applies)
    MSS_API_URL   default https://manuscript-search.org
"""
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_server import SearchFailed, build_mcp  # noqa: E402

API_URL = os.environ.get("MSS_API_URL", "https://manuscript-search.org").rstrip("/")
API_KEY = os.environ.get("MSS_API_KEY", "")


async def _post(path: str, body: dict) -> dict:
    body = {k: v for k, v in body.items() if v is not None}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{API_URL}{path}", json=body, headers=headers)
    if r.status_code == 200:
        return r.json()
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    raise SearchFailed(f"Search API returned {r.status_code}: {detail}")


async def _get_stats() -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{API_URL}/v1/stats")
    if r.status_code != 200:
        raise SearchFailed(f"Stats unavailable ({r.status_code}).")
    return r.json()


async def _post_search(**kwargs) -> dict:
    return await _post("/search", kwargs)


async def _post_similar(**kwargs) -> dict:
    return await _post("/v1/similar", kwargs)


mcp = build_mcp(search=_post_search, stats=_get_stats, similar=_post_similar)

if __name__ == "__main__":
    mcp.run()   # stdio
