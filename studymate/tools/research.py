"""Tools for the Researcher agent: web search, page fetching, Wikipedia."""
import ipaddress
import socket
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from . import Tool

HEADERS = {"User-Agent": "StudyMate/1.0 (university coursework project)"}


def web_search(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    max_results = max(1, min(int(max_results), 8))
    hits = DDGS().text(query, max_results=max_results)
    if not hits:
        return "No results found."
    return "\n\n".join(f"{i}. {h['title']}\n   {h['href']}\n   {h['body']}" for i, h in enumerate(hits, 1))


def _is_public_host(host: str) -> bool:
    try:
        return all(ipaddress.ip_address(info[4][0]).is_global for info in socket.getaddrinfo(host, None))
    except (socket.gaierror, ValueError):
        return False


def fetch_webpage(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed.")
    if not _is_public_host(parsed.hostname):  # keeps the agent away from localhost / internal addresses
        raise ValueError("Refusing to fetch a non-public address.")
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    text = " ".join(soup.get_text(" ", strip=True).split())
    return f"Title: {title}\nURL: {url}\n\n{text}"


def wikipedia_summary(topic: str) -> str:
    r = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ', '_'))}",
        headers=HEADERS,
        timeout=15,
    )
    if r.status_code == 404:
        return f"No Wikipedia article titled '{topic}'. Try web_search instead."
    r.raise_for_status()
    d = r.json()
    return f"{d.get('title')}: {d.get('extract')}\nSource: {d.get('content_urls', {}).get('desktop', {}).get('page', '')}"


RESEARCH_TOOLS = [
    Tool(
        "web_search",
        "Search the web (DuckDuckGo). Returns titles, URLs and snippets. Use for current or specific information.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results, 1-8 (default 5)"},
            },
            "required": ["query"],
        },
        web_search,
    ),
    Tool(
        "fetch_webpage",
        "Download a public web page and return its readable text. Use after web_search to read a promising result in full.",
        {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
            "required": ["url"],
        },
        fetch_webpage,
    ),
    Tool(
        "wikipedia_summary",
        "Get the lead summary of a Wikipedia article. Good for definitions and background on a concept.",
        {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Article title, e.g. 'Gradient descent'"}},
            "required": ["topic"],
        },
        wikipedia_summary,
    ),
]
