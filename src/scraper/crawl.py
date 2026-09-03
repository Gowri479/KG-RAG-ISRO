"""Scrape and save ISRO pages to the local raw-data store.

The scraper supports both the Firecrawl API (preferred when an API key is set)
and a simple HTTP fallback for local development or offline debugging.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw"
DEFAULT_SEEDS = [
    "https://www.isro.gov.in/",
    "https://www.isro.gov.in/Spacecraft",
    "https://www.isro.gov.in/Missions",
]

logger = logging.getLogger(__name__)
load_dotenv(ROOT / ".env")


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from an HTML page without external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self.parts)


def _extract_text_from_html(html_text: str) -> str:
    """Convert HTML into a readable markdown-like text block."""
    parser = _HTMLTextExtractor()
    parser.feed(html_text)
    text = parser.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_url(url: str) -> str:
    """Normalize page URLs for deduplicated storage."""
    return url.strip().rstrip("/") or url


def extract_internal_links(html_text: str, base_url: str) -> list[str]:
    """Extract valid internal ISRO links from a fetched page HTML."""
    links: set[str] = set()
    seen = set()
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html_text, flags=re.I):
        raw_link = match.group(1).strip()
        if not raw_link or raw_link.startswith(("javascript:", "mailto:", "tel:")):
            continue
        resolved = urljoin(base_url, raw_link)
        parsed = urlparse(resolved)
        if parsed.netloc not in {"www.isro.gov.in", "isro.gov.in"}:
            continue
        if parsed.fragment:
            parsed = parsed._replace(fragment="")
        cleaned = parsed.geturl().rstrip("/")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        links.add(cleaned)
    return sorted(links)


def crawl_discovered_links(seed_url: str, output_dir: Path = DEFAULT_OUTPUT_DIR, max_pages: int | None = None) -> List[Path]:
    """Fetch a seed page, discover internal links, and crawl the valid site pages."""
    base_html = fetch_page_html(seed_url)
    candidates = extract_internal_links(base_html, seed_url)
    if not candidates:
        logger.warning("No internal links discovered from %s", seed_url)
        return crawl_urls([seed_url], output_dir=output_dir, max_pages=max_pages)

    filtered = [
        link
        for link in candidates
        if all(token not in link.lower() for token in ("search", "calendar", "logout", "login", "privacy", "terms"))
    ]
    logger.info("Discovered %d candidate links from %s", len(filtered), seed_url)
    return crawl_urls(filtered[:max_pages] if max_pages else filtered, output_dir=output_dir, max_pages=max_pages)


def _fetch_firecrawl_markdown(url: str, api_key: str) -> str:
    """Fetch page content using the Firecrawl API when configured."""
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": url,
        "pageOptions": {"onlyMainContent": True},
        "formats": ["markdown"],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "data" not in payload:
        raise ValueError(f"Firecrawl response missing 'data' for url: {url}")
    markdown = payload["data"].get("markdown") or payload["data"].get("content") or ""
    if not markdown:
        raise ValueError(f"Firecrawl returned empty markdown for url: {url}")
    return markdown


def fetch_page_html(url: str) -> str:
    """Fetch the raw HTML for a page so that link discovery can inspect href attributes."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.text


def fetch_page_text(url: str) -> str:
    """Fetch page text from Firecrawl or, if unavailable, from a plain HTTP request."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if api_key:
        try:
            return _fetch_firecrawl_markdown(url, api_key)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Firecrawl fetch failed for %s: %s", url, exc)

    return _extract_text_from_html(fetch_page_html(url))


def save_markdown(url: str, text: str, output_dir: Path) -> Path:
    """Save a cleaned markdown page to disk using a URL-derived filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", _normalize_url(url).lower()).strip("_") or "page"
    file_path = output_dir / f"{safe_name}.md"
    content = f"# Source: {url}\n\n{text.strip()}\n"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def crawl_urls(urls: Iterable[str], output_dir: Path = DEFAULT_OUTPUT_DIR, max_pages: int | None = None) -> List[Path]:
    """Scrape a list of seed URLs and save Markdown files to disk."""
    seen: set[str] = set()
    written: List[Path] = []
    for idx, url in enumerate(tqdm(list(urls), desc="Scraping ISRO pages", unit="page")):
        if max_pages is not None and idx >= max_pages:
            break
        normalized_url = _normalize_url(url)
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        try:
            text = fetch_page_text(normalized_url)
            if not text or not text.strip():
                raise ValueError(f"No page content retrieved for {normalized_url}")
            saved = save_markdown(normalized_url, text, output_dir)
            written.append(saved)
            logger.info("Saved %s", saved)
        except Exception as exc:
            logger.exception("Failed to crawl %s: %s", normalized_url, exc)
    return written


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the local scraper entry point."""
    parser = argparse.ArgumentParser(description="Scrape ISRO pages and save raw Markdown to the data/raw folder.")
    parser.add_argument("--url", action="append", default=[], help="A single ISRO URL to scrape; may be repeated")
    parser.add_argument("--input-file", type=Path, default=None, help="Optional text file containing a URL per line")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved raw Markdown files")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to crawl")
    parser.add_argument("--discover", action="store_true", help="Fetch a seed page and crawl valid internal links discovered on it")
    return parser.parse_args()


def main() -> None:
    """Run the scraper from the command line."""
    args = _parse_args()
    urls: List[str] = list(args.url)
    if args.input_file:
        urls.extend(line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not urls:
        urls = DEFAULT_SEEDS

    if args.discover:
        collected: List[Path] = []
        for url in urls:
            collected.extend(crawl_discovered_links(url, output_dir=args.output_dir, max_pages=args.max_pages))
        return

    crawl_urls(urls, output_dir=args.output_dir, max_pages=args.max_pages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()

