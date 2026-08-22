from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import pytest
import yaml

from care_anxrag.models import EvidenceLevel, KnowledgeLayer, SourceConfig, SourceState
from care_anxrag.registry import load_source_registry
from care_anxrag.sources.http_page import HttpPageConnector, _load_cursor
from care_anxrag.sources.ncbi import PubMedConnector, _parse_pmc_article


def _source_payload() -> dict:
    return {
        "id": "local_test",
        "name": "Local Test",
        "connector": "local_files",
        "layer": "clinical_core",
        "authority_score": 0.8,
        "evidence_level": "general_information",
        "settings": {"path": "data/local"},
    }


def test_registry_accepts_list_root(tmp_path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump([_source_payload()]), encoding="utf-8")
    sources = load_source_registry(path)
    assert [source.id for source in sources] == ["local_test"]


def test_registry_rejects_invalid_root(tmp_path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text("not-a-source-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be"):
        load_source_registry(path)


def test_registry_rejects_mapping_without_sources_key(tmp_path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"source": [_source_payload()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a 'sources' key"):
        load_source_registry(path)


def test_registry_rejects_inline_credentials(tmp_path) -> None:
    payload = _source_payload()
    payload["settings"]["api_key"] = "plaintext-secret"
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": [payload]}), encoding="utf-8")
    with pytest.raises(ValueError, match="inline credential"):
        load_source_registry(path)


def test_registry_rejects_nested_inline_credentials_but_allows_env_references(tmp_path) -> None:
    payload = _source_payload()
    payload["settings"]["headers"] = {"Authorization": "Bearer plaintext-secret"}
    payload["settings"]["api_key_env"] = "SERVICE_API_KEY"
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": [payload]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"settings\.headers\.Authorization"):
        load_source_registry(path)

    payload["settings"]["headers"] = {"Accept": "application/json"}
    path.write_text(yaml.safe_dump({"sources": [payload]}), encoding="utf-8")
    sources = load_source_registry(path)
    assert sources[0].settings["api_key_env"] == "SERVICE_API_KEY"


def test_ncbi_connector_never_reads_inline_api_key(monkeypatch) -> None:
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    source = SourceConfig(
        id="pubmed_test",
        name="PubMed Test",
        connector="pubmed",
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
        authority_score=0.75,
        evidence_level=EvidenceLevel.UNKNOWN,
        settings={
            "api_key_env": "NCBI_API_KEY",
            "api_key": "must-not-be-used",
            "email": "researcher@example.org",
        },
    )
    credentials = PubMedConnector("test-agent")._credentials(source)
    assert "api_key" not in credentials
    assert credentials["email"] == "researcher@example.org"


def test_http_cursor_ignores_malformed_per_url_state() -> None:
    assert _load_cursor('{"https://example.org": "invalid"}') == {}


def test_http_connector_accepts_single_url_string(monkeypatch) -> None:
    requested: list[str] = []

    class Response:
        status_code = 304
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def request(self, method, url, **kwargs):
            requested.append(url)
            return Response()

    monkeypatch.setattr("care_anxrag.sources.http_page.ResilientHttpClient", Client)
    source = SourceConfig(
        id="web_test",
        name="Web Test",
        connector="http_page",
        layer=KnowledgeLayer.CLINICAL_CORE,
        authority_score=0.8,
        evidence_level=EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION,
        settings={"urls": "https://example.org/anxiety"},
    )
    result = HttpPageConnector("test-agent").fetch(
        source,
        SourceState(source_id=source.id),
        None,
        datetime(2026, 8, 17, tzinfo=UTC),
    )
    assert requested == ["https://example.org/anxiety"]
    assert result.changed is False


def test_pmc_restricted_cc_license_is_not_allowlisted() -> None:
    xml = """
    <article article-type="research-article" xml:lang="en">
      <front><article-meta>
        <article-id pub-id-type="pmc">123</article-id>
        <title-group><article-title>Anxiety intervention study</article-title></title-group>
        <abstract><p>This study examines anxiety treatment and outcomes.</p></abstract>
        <permissions><license xlink:href="https://creativecommons.org/licenses/by-nc/4.0/"
          xmlns:xlink="http://www.w3.org/1999/xlink">
          <license-p>Creative Commons Attribution-NonCommercial.</license-p>
        </license></permissions>
      </article-meta></front>
      <body><sec><title>Results</title><p>Anxiety symptoms were measured.</p></sec></body>
    </article>
    """
    document = _parse_pmc_article("pmc", ET.fromstring(xml), ["cc by", "creativecommons.org/licenses/by/"])
    assert document is not None
    assert document.metadata["reuse_allowed"] is False
