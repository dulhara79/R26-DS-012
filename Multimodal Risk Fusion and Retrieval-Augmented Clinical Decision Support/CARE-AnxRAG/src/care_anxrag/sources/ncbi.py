from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from ..models import RawDocument, Section, SourceConfig, SourceState
from ..util import normalize_whitespace, parse_datetime, utc_now, xml_text
from .base import FetchResult, ResilientHttpClient


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NcbiConnectorBase:
    database_name: str

    def __init__(self, user_agent: str, timeout_seconds: float = 60.0):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def _credentials(self, source: SourceConfig) -> dict[str, str]:
        api_key_env = str(source.settings.get("api_key_env", "NCBI_API_KEY"))
        email_env = str(source.settings.get("email_env", "NCBI_EMAIL"))
        tool = str(source.settings.get("tool", "CARE-AnxRAG"))
        credentials = {"tool": tool}
        email = os.getenv(email_env, str(source.settings.get("email", ""))).strip()
        api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
        if email:
            credentials["email"] = email
        if api_key:
            credentials["api_key"] = api_key
        return credentials

    def _client(self, source: SourceConfig) -> ResilientHttpClient:
        has_key = bool(self._credentials(source).get("api_key"))
        minimum_interval = 0.11 if has_key else 0.35
        return ResilientHttpClient(
            user_agent=self.user_agent,
            timeout_seconds=self.timeout_seconds,
            minimum_interval_seconds=minimum_interval,
        )

    def _search_ids(
        self,
        client: ResilientHttpClient,
        source: SourceConfig,
        since: datetime | None,
        until: datetime,
    ) -> list[str]:
        query = str(source.settings.get("query", "anxiety disorders")).strip()
        if not query:
            raise ValueError("NCBI settings.query must not be empty")
        lookback_days = _positive_int(source.settings.get("initial_lookback_days", 365), "initial_lookback_days")
        overlap_days = int(source.settings.get("overlap_days", 1))
        if overlap_days < 0:
            raise ValueError("overlap_days cannot be negative")
        maximum_records = _positive_int(
            source.settings.get("max_records_per_sync", 500),
            "max_records_per_sync",
        )
        batch_size = min(
            500,
            _positive_int(source.settings.get("search_batch_size", 200), "search_batch_size"),
        )
        start = (since - timedelta(days=overlap_days)) if since else (until - timedelta(days=lookback_days))
        credentials = self._credentials(source)
        ids: list[str] = []
        retstart = 0
        total = None
        while len(ids) < maximum_records and (total is None or retstart < total):
            params: dict[str, Any] = {
                "db": self.database_name,
                "term": query,
                "datetype": "mdat",
                "mindate": start.strftime("%Y/%m/%d"),
                "maxdate": until.strftime("%Y/%m/%d"),
                "retmode": "json",
                "retstart": retstart,
                "retmax": min(batch_size, maximum_records - len(ids)),
                "sort": "pub date",
                **credentials,
            }
            response = client.request("GET", f"{EUTILS_BASE}/esearch.fcgi", params=params)
            result = response.json().get("esearchresult", {})
            page_ids = [str(value) for value in result.get("idlist", [])]
            if total is None:
                total = int(result.get("count", len(page_ids)))
                if total > maximum_records:
                    raise RuntimeError(
                        f"NCBI update window contains {total} records, exceeding "
                        f"max_records_per_sync={maximum_records}. Increase the cap or reduce "
                        "initial_lookback_days; the source watermark was not advanced."
                    )
            ids.extend(page_ids)
            if not page_ids:
                break
            retstart += len(page_ids)
        return ids[:maximum_records]

    def _fetch_xml(
        self,
        client: ResilientHttpClient,
        source: SourceConfig,
        ids: Iterable[str],
    ) -> list[ET.Element]:
        id_list = list(ids)
        credentials = self._credentials(source)
        batch_size = _positive_int(
            source.settings.get("fetch_batch_size", 100),
            "fetch_batch_size",
        )
        roots: list[ET.Element] = []
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start : start + batch_size]
            response = client.request(
                "POST",
                f"{EUTILS_BASE}/efetch.fcgi",
                data={
                    "db": self.database_name,
                    "id": ",".join(batch),
                    "retmode": "xml",
                    **credentials,
                },
            )
            roots.append(ET.fromstring(response.content))
        return roots


class PubMedConnector(NcbiConnectorBase):
    database_name = "pubmed"

    def fetch(
        self,
        source: SourceConfig,
        state: SourceState,
        since: datetime | None,
        until: datetime,
    ) -> FetchResult:
        del state
        with self._client(source) as client:
            ids = self._search_ids(client, source, since, until)
            roots = self._fetch_xml(client, source, ids)
        documents: list[RawDocument] = []
        for root in roots:
            for article in root.findall(".//PubmedArticle"):
                parsed = _parse_pubmed_article(source.id, article)
                if parsed is not None:
                    documents.append(parsed)
        return FetchResult(
            documents=documents,
            changed=bool(documents),
            cursor=until.astimezone(UTC).isoformat(),
        )


class PmcConnector(NcbiConnectorBase):
    database_name = "pmc"

    def fetch(
        self,
        source: SourceConfig,
        state: SourceState,
        since: datetime | None,
        until: datetime,
    ) -> FetchResult:
        del state
        with self._client(source) as client:
            ids = self._search_ids(client, source, since, until)
            roots = self._fetch_xml(client, source, ids)
        raw_allowlist = source.settings.get(
            "license_allowlist",
            [
                "creativecommons.org/licenses/by/",
                "creativecommons.org/publicdomain/zero/",
                "creativecommons.org/publicdomain/mark/",
                "cc0",
                "public domain",
            ],
        )
        if isinstance(raw_allowlist, str):
            raw_allowlist = [raw_allowlist]
        if not isinstance(raw_allowlist, (list, tuple, set)):
            raise ValueError("license_allowlist must be a string or a list of strings")
        allowlist = [str(value).strip().lower() for value in raw_allowlist if str(value).strip()]
        if not allowlist:
            raise ValueError("license_allowlist must not be empty")
        documents: list[RawDocument] = []
        for root in roots:
            for article in root.findall(".//article"):
                parsed = _parse_pmc_article(source.id, article, allowlist)
                if parsed is not None:
                    documents.append(parsed)
        return FetchResult(
            documents=documents,
            changed=bool(documents),
            cursor=until.astimezone(UTC).isoformat(),
        )


def _parse_pubmed_article(source_id: str, item: ET.Element) -> RawDocument | None:
    citation = item.find("MedlineCitation")
    article = item.find("MedlineCitation/Article")
    if citation is None or article is None:
        return None
    pmid = xml_text(citation.find("PMID"))
    title = xml_text(article.find("ArticleTitle"))
    if not pmid or not title:
        return None

    abstract_sections: list[Section] = []
    abstract_parts: list[str] = []
    for index, abstract_text in enumerate(article.findall("Abstract/AbstractText")):
        value = xml_text(abstract_text)
        if not value:
            continue
        label = str(abstract_text.attrib.get("Label") or abstract_text.attrib.get("NlmCategory") or "Abstract")
        abstract_parts.append(f"{label}: {value}" if label.lower() != "abstract" else value)
        abstract_sections.append(
            Section(path=f"abstract/{index}", heading=label, text=value, ordinal=index)
        )
    abstract = normalize_whitespace("\n\n".join(abstract_parts))
    if not abstract:
        return None

    publication_types = [
        xml_text(node) for node in article.findall("PublicationTypeList/PublicationType") if xml_text(node)
    ]
    authors: list[str] = []
    for author in article.findall("AuthorList/Author"):
        collective = xml_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        surname = xml_text(author.find("LastName"))
        initials = xml_text(author.find("Initials"))
        name = normalize_whitespace(f"{surname} {initials}")
        if name:
            authors.append(name)

    topics = [
        xml_text(node)
        for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
        if xml_text(node)
    ]
    topics.extend(
        xml_text(node)
        for node in citation.findall("KeywordList/Keyword")
        if xml_text(node)
    )
    doi = ""
    pmc_id = ""
    for article_id in item.findall("PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.attrib.get("IdType", "")
        if id_type == "doi":
            doi = xml_text(article_id)
        elif id_type == "pmc":
            pmc_id = xml_text(article_id)

    published_at = _parse_pub_date(article.find("Journal/JournalIssue/PubDate"))
    article_date = article.find("ArticleDate")
    if article_date is not None:
        published_at = _parse_ymd(article_date) or published_at
    updated_at = _parse_ymd(citation.find("DateRevised"))
    journal = xml_text(article.find("Journal/Title"))
    language = xml_text(article.find("Language")) or "eng"
    publication_status = xml_text(item.find("PubmedData/PublicationStatus"))

    return RawDocument(
        source_id=source_id,
        external_id=pmid,
        title=title,
        text=normalize_whitespace(f"{title}\n\n{abstract}"),
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=utc_now(),
        authors=authors,
        language=language,
        publication_types=publication_types,
        topics=sorted(set(topics)),
        sections=abstract_sections,
        metadata={
            "pmid": pmid,
            "pmcid": pmc_id,
            "doi": doi,
            "journal": journal,
            "publication_status": publication_status,
            "record_source": "PubMed abstract",
        },
    )


def _parse_pmc_article(
    source_id: str,
    article: ET.Element,
    license_allowlist: list[str],
) -> RawDocument | None:
    front = article.find("front")
    if front is None:
        return None
    article_meta = front.find("article-meta")
    if article_meta is None:
        return None
    pmcid = ""
    doi = ""
    for node in article_meta.findall("article-id"):
        id_type = node.attrib.get("pub-id-type", "")
        if id_type == "pmc":
            pmcid = xml_text(node)
        elif id_type == "doi":
            doi = xml_text(node)
    if not pmcid:
        return None
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    title = xml_text(article_meta.find("title-group/article-title"))
    if not title:
        return None

    sections: list[Section] = []
    abstract = article_meta.find("abstract")
    if abstract is not None:
        abstract_text = normalize_whitespace(" ".join(xml_text(p) for p in abstract.findall(".//p")))
        if abstract_text:
            sections.append(
                Section(path="abstract", heading="Abstract", text=abstract_text, ordinal=len(sections))
            )

    body = article.find("body")
    if body is not None:
        top_sections = body.findall("sec")
        if top_sections:
            for sec in top_sections:
                _collect_pmc_sections(sec, [], sections)
        else:
            body_text = normalize_whitespace(" ".join(xml_text(p) for p in body.findall(".//p")))
            if body_text:
                sections.append(
                    Section(path="body", heading="Body", text=body_text, ordinal=len(sections))
                )

    text = normalize_whitespace("\n\n".join(section.text for section in sections))
    if not text:
        return None

    license_nodes = article_meta.findall("permissions/license")
    license_text = normalize_whitespace(" ".join(xml_text(node) for node in license_nodes))
    license_urls = [
        value
        for node in license_nodes
        for key, value in node.attrib.items()
        if key.endswith("href") and value
    ]
    license_haystack = f"{license_text} {' '.join(license_urls)}".lower()
    restricted_markers = (
        "all rights reserved",
        "noncommercial",
        "non-commercial",
        "no derivatives",
        "no-derivatives",
        "/by-nc",
        "/by-nd",
    )
    reuse_allowed = not any(
        marker in license_haystack for marker in restricted_markers
    ) and any(marker in license_haystack for marker in license_allowlist)

    publication_type = article.attrib.get("article-type", "journal-article")
    published_at = None
    for pub_date in article_meta.findall("pub-date"):
        if pub_date.attrib.get("pub-type") in {"epub", "ppub", "collection"}:
            published_at = _parse_ymd(pub_date)
            if published_at:
                break
    if published_at is None:
        published_at = _parse_ymd(article_meta.find("pub-date"))
    updated_at = None
    for date_node in article_meta.findall("history/date"):
        if date_node.attrib.get("date-type") in {"rev-recd", "corrected", "updated"}:
            updated_at = _parse_ymd(date_node)

    authors: list[str] = []
    for contrib in article_meta.findall("contrib-group/contrib"):
        if contrib.attrib.get("contrib-type") != "author":
            continue
        surname = xml_text(contrib.find("name/surname"))
        given = xml_text(contrib.find("name/given-names"))
        collective = xml_text(contrib.find("collab"))
        name = collective or normalize_whitespace(f"{surname} {given}")
        if name:
            authors.append(name)
    topics = [xml_text(node) for node in article_meta.findall("kwd-group/kwd") if xml_text(node)]
    language = article.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "en")

    return RawDocument(
        source_id=source_id,
        external_id=pmcid,
        title=title,
        text=normalize_whitespace(f"{title}\n\n{text}"),
        url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=utc_now(),
        authors=authors,
        language=language,
        publication_types=[publication_type],
        topics=sorted(set(topics)),
        sections=sections,
        metadata={
            "pmcid": pmcid,
            "doi": doi,
            "license_text": license_text,
            "license_urls": license_urls,
            "reuse_allowed": reuse_allowed,
            "record_source": "PMC full text",
        },
    )


def _collect_pmc_sections(
    sec: ET.Element,
    parents: list[str],
    output: list[Section],
) -> None:
    heading = xml_text(sec.find("title")) or "Untitled section"
    path_parts = [*parents, heading]
    direct_paragraphs = [xml_text(node) for node in sec.findall("p") if xml_text(node)]
    if direct_paragraphs:
        output.append(
            Section(
                path=" / ".join(path_parts),
                heading=heading,
                text=normalize_whitespace("\n".join(direct_paragraphs)),
                ordinal=len(output),
            )
        )
    for child in sec.findall("sec"):
        _collect_pmc_sections(child, path_parts, output)


def _parse_pub_date(element: ET.Element | None) -> datetime | None:
    if element is None:
        return None
    medline = xml_text(element.find("MedlineDate"))
    if medline:
        year = next((part for part in medline.split() if part[:4].isdigit()), "")[:4]
        if year.isdigit():
            return datetime(int(year), 1, 1, tzinfo=UTC)
    return _parse_ymd(element)


def _parse_ymd(element: ET.Element | None) -> datetime | None:
    if element is None:
        return None
    year_text = xml_text(element.find("Year")) or xml_text(element.find("year"))
    month_text = xml_text(element.find("Month")) or xml_text(element.find("month")) or "1"
    day_text = xml_text(element.find("Day")) or xml_text(element.find("day")) or "1"
    if not year_text.isdigit():
        return None
    month_lookup = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    month = int(month_text) if month_text.isdigit() else month_lookup.get(month_text[:3].lower(), 1)
    day = int(day_text) if day_text.isdigit() else 1
    try:
        return datetime(int(year_text), month, day, tzinfo=UTC)
    except ValueError:
        return datetime(int(year_text), 1, 1, tzinfo=UTC)


def _positive_int(value: Any, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed
