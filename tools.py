"""Tool schemas (as the OpenAI function-calling API wants them) and their
server-side executors. Two tools:

  * run_python  — the workhorse: fetch + parse + compute in one place.
  * web_search  — discovery: find the right dataset page / current figure.

Keeping executors here means app/agent code just routes name -> function.
"""
from __future__ import annotations

import html
import re
from typing import Any

import httpx

from sandbox import run_python as _run_python

# ---------------------------------------------------------------------------
# Schemas advertised to the model
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python 3 in a sandbox that HAS internet access and these "
                "libraries: pandas, numpy, requests, httpx, beautifulsoup4, lxml, "
                "openpyxl, xlrd, pdfplumber, python-dateutil. Use it to download "
                "datasets (MOSPI and other public sources), parse HTML/CSV/Excel/"
                "PDF/JSON, and compute the answer. IMPORTANT: only what you print() "
                "to stdout is returned to you. Variables do NOT persist between "
                "calls, but files you save to the current directory DO."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web (DuckDuckGo) and get back the top results as "
                "title / url / snippet. Use it to locate the correct dataset page "
                "or verify a current figure, then fetch the details with run_python."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# web_search executor (dependency-free DuckDuckGo scrape, best effort)
# ---------------------------------------------------------------------------
_DDG_HTML = "https://html.duckduckgo.com/html/"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)


def _clean(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def web_search(query: str, max_results: int = 6) -> str:
    max_results = max(1, min(int(max_results or 6), 10))
    try:
        with httpx.Client(timeout=20, follow_redirects=True,
                          headers={"User-Agent": _UA}) as client:
            resp = client.post(_DDG_HTML, data={"q": query})
            resp.raise_for_status()
            body = resp.text
    except Exception as e:
        return (
            f"web_search failed ({type(e).__name__}: {e}). "
            f"Fall back to run_python with requests/httpx to fetch a known URL directly."
        )

    results = []
    # Each organic result: <a class="result__a" href="...">Title</a> ... snippet
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.S
    ):
        href, title = m.group(1), _clean(m.group(2))
        # DDG wraps target in a redirect; pull out uddg= if present.
        um = re.search(r"uddg=([^&]+)", href)
        if um:
            from urllib.parse import unquote

            href = unquote(um.group(1))
        results.append({"title": title, "url": href})
        if len(results) >= max_results:
            break

    snippets = [
        _clean(s) for s in re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', body, re.S
        )
    ]
    for i, snip in enumerate(snippets[: len(results)]):
        results[i]["snippet"] = snip

    if not results:
        return "No results parsed. Try run_python to fetch a source URL directly."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title','(no title)')}\n   {r.get('url','')}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def execute_tool(name: str, args: dict[str, Any], *, workdir: str,
                 python_timeout: int, sandbox_mem_mb: int) -> str:
    if name == "run_python":
        res = _run_python(
            args.get("code", ""),
            workdir=workdir,
            timeout=python_timeout,
            mem_mb=sandbox_mem_mb,
        )
        return res.as_tool_output()
    if name == "web_search":
        return web_search(args.get("query", ""), args.get("max_results", 6))
    return f"Unknown tool: {name}"
