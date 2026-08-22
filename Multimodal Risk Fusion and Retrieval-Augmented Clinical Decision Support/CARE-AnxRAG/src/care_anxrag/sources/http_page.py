from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from ..models import RawDocument, Section, SourceConfig, SourceState
from ..util import normalize_whitespace, parse_datetime, stable_id, utc_now
from .base import FetchResult, ResilientHttpClient


class HttpPageConnector:
    def __init__(self, user_agent: str, timeout_seconds: float = 60.0):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def fetch(
        self,
        source: SourceConfig,
        state: SourceState,
        since: datetime | None,
        until: datetime,
    ) -> FetchResult:
        del since, until
        configured_urls = source.settings.get("urls")
        if configured_urls is None:
            configured_urls = source.settings.get("url")
        if isinstance(configured_urls, str):
            urls = [configured_urls.strip()] if configured_urls.strip() else []
        elif isinstance(configured_urls, (list, tuple, set)):
            urls = [str(url).strip() for url in configured_urls if str(url).strip()]
        elif configured_urls is None:
            urls = []
        else:
            raise ValueError("settings.urls must be a URL string or a list of URL strings")
        if not urls:
            raise ValueError(f"Source {source.id} requires settings.url or settings.urls")

        prior_states = _load_cursor(state.cursor)
        new_states: dict[str, dict[str, str]] = {}
        documents: list[RawDocument] = []
        warnings: list[str] = []

        with ResilientHttpClient(
            user_agent=self.user_agent,
            timeout_seconds=self.timeout_seconds,
            minimum_interval_seconds=float(source.settings.get("request_interval_seconds", 0.5)),
        ) as client:
            for url in urls:
                headers: dict[str, str] = {
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"
                }
                prior = prior_states.get(url, {})
                if prior.get("etag"):
                    headers["If-None-Match"] = prior["etag"]
                if prior.get("last_modified"):
                    headers["If-Modified-Since"] = prior["last_modified"]
                response = client.request(
                    "GET", url, headers=headers, acceptable_statuses={304}
                )
                if response.status_code == 304:
                    new_states[url] = prior
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    warnings.append(f"Skipped non-HTML page {url}: {content_type}")
                    new_states[url] = {
                        "etag": response.headers.get("etag", "") or prior.get("etag", ""),
                        "last_modified": response.headers.get("last-modified", "")
                        or prior.get("last_modified", ""),
                    }
                    continue
                document = _parse_html_document(source.id, url, response.text, response.headers)
                documents.append(document)
                new_states[url] = {
                    "etag": response.headers.get("etag", ""),
                    "last_modified": response.headers.get("last-modified", ""),
                }

        return FetchResult(
            documents=documents,
            etag=new_states.get(urls[0], {}).get("etag") or None if len(urls) == 1 else None,
            last_modified=(
                new_states.get(urls[0], {}).get("last_modified") or None
                if len(urls) == 1
                else None
            ),
            cursor=json.dumps(new_states, sort_keys=True),
            changed=bool(documents),
            warnings=warnings,
        )


def _load_cursor(cursor: str | None) -> dict[str, dict[str, str]]:
    if not cursor:
        return {}
    try:
        value = json.loads(cursor)
        if not isinstance(value, dict):
            return {}
        output: dict[str, dict[str, str]] = {}
        for raw_url, raw_state in value.items():
            if not isinstance(raw_state, dict):
                continue
            output[str(raw_url)] = {
                "etag": str(raw_state.get("etag", "")),
                "last_modified": str(raw_state.get("last_modified", "")),
            }
        return output
    except json.JSONDecodeError:
        return {}


def _parse_html_document(
    source_id: str,
    url: str,
    html: str,
    headers: Any,
) -> RawDocument:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "script, style, noscript, svg, nav, footer, form, button, iframe, [aria-hidden='true']"
    ):
        tag.decompose()

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    title = _extract_title(soup, root)
    sections = _extract_sections(root, title)
    text = normalize_whitespace("\n\n".join(section.text for section in sections))
    metadata_dates = _extract_dates(soup)
    published_at = metadata_dates.get("published")
    updated_at = metadata_dates.get("modified") or parse_datetime(headers.get("last-modified"))
    language = "en"
    if soup.html and soup.html.get("lang"):
        language = str(soup.html.get("lang")).split("-")[0].lower()

    return RawDocument(
        source_id=source_id,
        external_id=stable_id(url, length=32),
        title=title,
        text=text,
        url=url,
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=utc_now(),
        language=language,
        sections=sections,
        metadata={
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "content_type": headers.get("content-type"),
        },
    )


def _extract_title(soup: BeautifulSoup, root: Tag) -> str:
    h1 = root.find("h1")
    if h1:
        value = normalize_whitespace(h1.get_text(" ", strip=True))
        if value:
            return value
    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        return normalize_whitespace(str(meta["content"]))
    if soup.title:
        return normalize_whitespace(soup.title.get_text(" ", strip=True))
    return "Untitled source page"


def _extract_sections(root: Tag, title: str) -> list[Section]:
    sections: list[Section] = []
    heading = title
    path = "root"
    buffer: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal buffer
        text = normalize_whitespace("\n".join(buffer))
        if text:
            sections.append(
                Section(path=path, heading=heading, text=text, ordinal=len(sections))
            )
        buffer = []

    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if not isinstance(element, Tag):
            continue
        value = normalize_whitespace(element.get_text(" ", strip=True))
        if not value:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            level = int(element.name[1])
            heading_stack = [(old_level, old) for old_level, old in heading_stack if old_level < level]
            heading_stack.append((level, value))
            heading = value
            path = " / ".join(item for _, item in heading_stack)
        else:
            buffer.append(value)
    flush()
    if not sections:
        fallback = normalize_whitespace(root.get_text("\n", strip=True))
        if fallback:
            sections.append(Section(path="root", heading=title, text=fallback, ordinal=0))
    return sections


def _extract_dates(soup: BeautifulSoup) -> dict[str, datetime | None]:
    values: dict[str, datetime | None] = {"published": None, "modified": None}
    meta_candidates = {
        "published": ["article:published_time", "datePublished", "date"],
        "modified": ["article:modified_time", "dateModified", "last-modified"],
    }
    for kind, names in meta_candidates.items():
        for name in names:
            meta = soup.find("meta", attrs={"property": name}) or soup.find(
                "meta", attrs={"name": name}
            )
            if meta and meta.get("content"):
                parsed = parse_datetime(str(meta["content"]))
                if parsed:
                    values[kind] = parsed
                    break

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "{}")
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            values["published"] = values["published"] or parse_datetime(item.get("datePublished"))
            values["modified"] = values["modified"] or parse_datetime(item.get("dateModified"))
    return values
