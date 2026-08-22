from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from ..models import RawDocument, Section, SourceConfig, SourceState
from ..util import normalize_whitespace, parse_datetime, stable_id, utc_now, xml_text
from .base import FetchResult, ResilientHttpClient


class NiceSyndicationConnector:
    """NICE licensed syndication connector.

    The registry supplies a licensed resource/feed URL. The connector uses the official
    API-Key header, conditional requests, and parses Atom or NICE JSON/XML representations.
    It deliberately does not scrape public NICE pages for RAG ingestion.
    """

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
        feed_url = str(source.settings.get("feed_url", "")).strip()
        if not feed_url:
            raise ValueError(f"Source {source.id} requires settings.feed_url")
        api_key_env = str(source.settings.get("api_key_env", "NICE_API_KEY"))
        api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
        if not api_key:
            raise RuntimeError(
                f"NICE API key is missing. Set {api_key_env} after obtaining the required licence."
            )
        accept = str(
            source.settings.get(
                "accept", "application/vnd.nice.syndication.services+json, application/atom+xml;q=0.9"
            )
        )
        headers = {"API-Key": api_key, "Accept": accept}
        if state.etag:
            headers["If-None-Match"] = state.etag
        if state.last_modified:
            headers["If-Modified-Since"] = state.last_modified

        with ResilientHttpClient(
            user_agent=self.user_agent,
            timeout_seconds=self.timeout_seconds,
            minimum_interval_seconds=float(source.settings.get("request_interval_seconds", 0.5)),
        ) as client:
            response = client.request(
                "GET",
                feed_url,
                headers=headers,
                acceptable_statuses={304, 410},
            )
        if response.status_code == 304:
            return FetchResult(
                documents=[],
                etag=state.etag,
                last_modified=state.last_modified,
                changed=False,
            )
        if response.status_code == 410:
            return FetchResult(
                documents=[],
                changed=True,
                warnings=["NICE resource returned 410 Gone; manual withdrawal review required."],
            )

        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type or response.text.lstrip().startswith(("{", "[")):
            documents = _parse_nice_json(source.id, response.json(), feed_url)
        else:
            documents = _parse_nice_xml(source.id, response.content, feed_url)
        return FetchResult(
            documents=documents,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            changed=bool(documents),
        )


def _parse_nice_json(source_id: str, payload: Any, feed_url: str) -> list[RawDocument]:
    candidates = _find_json_resources(payload)
    documents: list[RawDocument] = []
    for index, item in enumerate(candidates):
        title = _first_value(item, "title", "Title", "name", "Name")
        content = _first_value(
            item,
            "content",
            "Content",
            "body",
            "Body",
            "description",
            "Description",
            "summary",
            "Summary",
        )
        if isinstance(content, dict):
            content = _first_value(content, "value", "Value", "text", "Text")
        if not title or not content:
            continue
        text, sections = _html_or_text_sections(str(content), str(title))
        resource_id = str(
            _first_value(item, "id", "Id", "guidanceId", "GuidanceId", "identifier", "Identifier")
            or stable_id(feed_url, str(index), str(title), length=32)
        )
        url = _first_value(item, "uri", "Uri", "url", "Url", "href", "Href") or feed_url
        published = parse_datetime(_first_value(item, "published", "Published", "publicationDate"))
        updated = parse_datetime(_first_value(item, "updated", "Updated", "lastModified"))
        documents.append(
            RawDocument(
                source_id=source_id,
                external_id=resource_id,
                title=str(title),
                text=text,
                url=str(url),
                published_at=published,
                updated_at=updated,
                retrieved_at=utc_now(),
                language="en",
                publication_types=["Clinical Guideline"],
                topics=["anxiety"],
                sections=sections,
                metadata={"licensed_source": "NICE syndication API"},
            )
        )
    return documents


def _parse_nice_xml(source_id: str, content: bytes, feed_url: str) -> list[RawDocument]:
    root = ET.fromstring(content)
    entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    if entries:
        documents: list[RawDocument] = []
        for index, entry in enumerate(entries):
            title = xml_text(entry.find("{http://www.w3.org/2005/Atom}title"))
            content_node = entry.find("{http://www.w3.org/2005/Atom}content")
            summary_node = entry.find("{http://www.w3.org/2005/Atom}summary")
            body = xml_text(content_node) or xml_text(summary_node)
            if not title or not body:
                continue
            text, sections = _html_or_text_sections(body, title)
            external_id = xml_text(entry.find("{http://www.w3.org/2005/Atom}id")) or stable_id(
                feed_url, str(index), title, length=32
            )
            link = feed_url
            for link_node in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link_node.attrib.get("rel", "alternate") in {"alternate", "self"}:
                    link = link_node.attrib.get("href", link)
                    break
            documents.append(
                RawDocument(
                    source_id=source_id,
                    external_id=external_id,
                    title=title,
                    text=text,
                    url=link,
                    published_at=parse_datetime(
                        xml_text(entry.find("{http://www.w3.org/2005/Atom}published"))
                    ),
                    updated_at=parse_datetime(
                        xml_text(entry.find("{http://www.w3.org/2005/Atom}updated"))
                    ),
                    retrieved_at=utc_now(),
                    language="en",
                    publication_types=["Clinical Guideline"],
                    topics=["anxiety"],
                    sections=sections,
                    metadata={"licensed_source": "NICE syndication API"},
                )
            )
        return documents

    title = xml_text(root.find(".//Title")) or xml_text(root.find(".//title"))
    resource_id = xml_text(root.find(".//Id")) or xml_text(root.find(".//id"))
    body_parts = [xml_text(node) for node in root.findall(".//Text") if xml_text(node)]
    if not body_parts:
        body_parts = [xml_text(node) for node in root.findall(".//text") if xml_text(node)]
    body = normalize_whitespace("\n\n".join(body_parts))
    if not title or not body:
        return []
    text, sections = _html_or_text_sections(body, title)
    return [
        RawDocument(
            source_id=source_id,
            external_id=resource_id or stable_id(feed_url, title, length=32),
            title=title,
            text=text,
            url=feed_url,
            retrieved_at=utc_now(),
            language="en",
            publication_types=["Clinical Guideline"],
            topics=["anxiety"],
            sections=sections,
            metadata={"licensed_source": "NICE syndication API"},
        )
    ]


def _find_json_resources(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_find_json_resources(item))
        return result
    if not isinstance(value, dict):
        return []
    has_title = any(key.lower() in {"title", "name"} for key in value)
    has_content = any(
        key.lower() in {"content", "body", "description", "summary"} for key in value
    )
    if has_title and has_content:
        return [value]
    result = []
    for child in value.values():
        result.extend(_find_json_resources(child))
    return result


def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _html_or_text_sections(content: str, title: str) -> tuple[str, list[Section]]:
    soup = BeautifulSoup(content, "html.parser")
    if soup.find():
        for tag in soup.select("script,style,noscript,nav,footer"):
            tag.decompose()
        sections: list[Section] = []
        heading = title
        buffer: list[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            value = normalize_whitespace(element.get_text(" ", strip=True))
            if not value:
                continue
            if element.name and element.name.startswith("h"):
                if buffer:
                    sections.append(
                        Section(
                            path=heading,
                            heading=heading,
                            text=normalize_whitespace("\n".join(buffer)),
                            ordinal=len(sections),
                        )
                    )
                    buffer = []
                heading = value
            else:
                buffer.append(value)
        if buffer:
            sections.append(
                Section(
                    path=heading,
                    heading=heading,
                    text=normalize_whitespace("\n".join(buffer)),
                    ordinal=len(sections),
                )
            )
        text = normalize_whitespace("\n\n".join(section.text for section in sections))
        if text:
            return text, sections
    text = normalize_whitespace(content)
    return text, [Section(path="root", heading=title, text=text, ordinal=0)]
