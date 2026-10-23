"""Collect AI articles from public sources as rows that follow the project's data contract.

- OpenAI: the official RSS feed, then the article page for the full text.
- Anthropic: no RSS feed, so the sitemap lists the news pages and each page is scraped.
- arXiv: the official API through the ``arxiv`` package; the abstract is the text.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urldefrag

import arxiv
import requests
from bs4 import BeautifulSoup
from loguru import logger

OPENAI_RSS_URL = "https://openai.com/news/rss.xml"
ANTHROPIC_SITEMAP_URL = "https://www.anthropic.com/sitemap.xml"
ANTHROPIC_NEWS_PREFIX = "https://www.anthropic.com/news/"
ARXIV_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"

USER_AGENT = "ai-daily-digest/0.1 (personal LLMOps course project)"
REQUEST_TIMEOUT_SECONDS = 30
SECONDS_BETWEEN_PAGES = 1.0  # one page per second, to stay polite to the sites we scrape
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@dataclass(frozen=True)
class Article:
    """One collected article, in the columns of the data contract."""

    source: str
    item_id: str
    title: str
    url: str
    published: datetime
    text: str
    authors: list[str]
    ingested_at: datetime


@dataclass(frozen=True)
class Page:
    """What an article page tells us about itself."""

    title: str
    text: str
    published: datetime | None


def make_item_id(url: str) -> str:
    """Return a stable id for an article: the SHA-256 of its URL without fragment or trailing slash."""
    clean_url, _fragment = urldefrag(url.strip())
    return hashlib.sha256(clean_url.rstrip("/").encode()).hexdigest()


def parse_rss(xml: bytes, source: str, ingested_at: datetime) -> list[Article]:
    """Turn an RSS 2.0 feed into articles; items without a link or date are skipped."""
    articles = []
    for item in ET.fromstring(xml).iter("item"):
        url = (item.findtext("link") or "").strip()
        pub_date = item.findtext("pubDate")
        if not url or not pub_date:
            continue
        articles.append(
            Article(
                source=source,
                item_id=make_item_id(url),
                title=(item.findtext("title") or "").strip(),
                url=url,
                published=parsedate_to_datetime(pub_date),
                text=_html_to_text(item.findtext("description") or ""),
                authors=[],
                ingested_at=ingested_at,
            )
        )
    return articles


def parse_sitemap(xml: bytes, url_prefix: str) -> list[tuple[str, datetime]]:
    """Return ``(url, lastmod)`` for sitemap entries under ``url_prefix`` that have a lastmod.

    Category pages such as ``/news/policy`` have no lastmod, so this also leaves them out.
    """
    entries = []
    for entry in ET.fromstring(xml).findall("sm:url", SITEMAP_NS):
        url = (entry.findtext("sm:loc", default="", namespaces=SITEMAP_NS)).strip()
        lastmod = _parse_iso(entry.findtext("sm:lastmod", namespaces=SITEMAP_NS))
        if url.startswith(url_prefix) and lastmod is not None:
            entries.append((url, lastmod))
    return entries


def extract_page(html: str) -> Page:
    """Pull the title, the main text, and the publication time out of an article page."""
    soup = BeautifulSoup(html, "html.parser")
    title = _meta(soup, "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    body = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in body.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    paragraphs = (p.get_text(" ", strip=True) for p in body.find_all("p"))
    return Page(
        title=title,
        text="\n\n".join(p for p in paragraphs if p),
        published=_parse_iso(_meta(soup, "article:published_time")),
    )


def fetch_openai(session: requests.Session, since: datetime, ingested_at: datetime) -> list[Article]:
    """Collect OpenAI articles published since ``since`` from the RSS feed, with the page text."""
    feed = parse_rss(_get(session, OPENAI_RSS_URL).content, "openai", ingested_at)
    return [_with_page_text(session, article) for article in feed if article.published >= since]


def fetch_anthropic(session: requests.Session, since: datetime, ingested_at: datetime) -> list[Article]:
    """Collect Anthropic news pages changed since ``since``, found through the sitemap."""
    entries = parse_sitemap(_get(session, ANTHROPIC_SITEMAP_URL).content, ANTHROPIC_NEWS_PREFIX)
    articles = []
    for url, lastmod in entries:
        if lastmod < since:
            continue
        time.sleep(SECONDS_BETWEEN_PAGES)
        try:
            page = extract_page(_get(session, url).text)
        except requests.RequestException as error:
            logger.warning(f"Skipping {url}: {error}")
            continue
        articles.append(
            Article(
                source="anthropic",
                item_id=make_item_id(url),
                title=page.title,
                url=url,
                published=page.published or lastmod,
                text=page.text,
                authors=[],
                ingested_at=ingested_at,
            )
        )
    return articles


def fetch_arxiv(since: datetime, ingested_at: datetime, max_results: int = 100) -> list[Article]:
    """Collect the newest arXiv papers in AI, ML, and NLP submitted since ``since``."""
    search = arxiv.Search(
        query=ARXIV_QUERY,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    articles = []
    for result in arxiv.Client().results(search):
        if result.published < since:
            continue
        # Drop the version (v1, v2, ...) so a revised paper keeps the same id.
        url = re.sub(r"v\d+$", "", result.entry_id)
        articles.append(
            Article(
                source="arxiv",
                item_id=make_item_id(url),
                title=result.title,
                url=url,
                published=result.published,
                text=result.summary,
                authors=[author.name for author in result.authors],
                ingested_at=ingested_at,
            )
        )
    return articles


def collect_articles(since: datetime) -> list[Article]:
    """Collect articles published since ``since`` (timezone-aware, e.g. UTC) from every source.

    A source that fails is logged and skipped, so one broken site does not stop the daily run.
    """
    ingested_at = datetime.now(UTC)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    collectors: dict[str, Callable[[], list[Article]]] = {
        "openai": lambda: fetch_openai(session, since, ingested_at),
        "anthropic": lambda: fetch_anthropic(session, since, ingested_at),
        "arxiv": lambda: fetch_arxiv(since, ingested_at),
    }
    articles = []
    for source, collect in collectors.items():
        try:
            found = collect()
        except Exception:
            logger.exception(f"Source {source} failed")
            continue
        logger.info(f"{source}: {len(found)} articles")
        articles.extend(found)
    return articles


def _get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def _with_page_text(session: requests.Session, article: Article) -> Article:
    """Replace the short feed description with the page's full text when the page can be read."""
    time.sleep(SECONDS_BETWEEN_PAGES)
    try:
        page = extract_page(_get(session, article.url).text)
    except requests.RequestException as error:
        logger.warning(f"Keeping the feed description for {article.url}: {error}")
        return article
    return replace(article, text=page.text) if page.text else article


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
    content = tag.get("content") if tag else None
    return content.strip() if content else None


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO 8601 time; a time without a zone is taken as UTC, and an invalid one as missing."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
