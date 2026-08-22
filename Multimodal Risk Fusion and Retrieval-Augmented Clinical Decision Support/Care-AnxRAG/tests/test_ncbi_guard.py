from __future__ import annotations

from datetime import UTC, datetime

import pytest

from care_anxrag.models import EvidenceLevel, KnowledgeLayer, SourceConfig
from care_anxrag.sources.ncbi import PubMedConnector


class Response:
    def json(self):
        return {"esearchresult": {"count": "1001", "idlist": ["1"]}}


class Client:
    def request(self, *args, **kwargs):
        return Response()


def test_ncbi_refuses_to_silently_truncate_update_window() -> None:
    connector = PubMedConnector("test-agent")
    source = SourceConfig(
        id="pubmed_test",
        name="PubMed Test",
        connector="pubmed",
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
        authority_score=0.7,
        evidence_level=EvidenceLevel.UNKNOWN,
        settings={
            "query": "anxiety",
            "max_records_per_sync": 1000,
            "initial_lookback_days": 30,
        },
    )
    with pytest.raises(RuntimeError, match="watermark was not advanced"):
        connector._search_ids(
            Client(),
            source,
            None,
            datetime(2026, 8, 17, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_lookback_days", 0),
        ("max_records_per_sync", 0),
        ("search_batch_size", 0),
    ],
)
def test_ncbi_rejects_non_positive_paging_configuration(field: str, value: int) -> None:
    connector = PubMedConnector("test-agent")
    settings = {
        "query": "anxiety",
        "max_records_per_sync": 10,
        "initial_lookback_days": 30,
        "search_batch_size": 10,
        field: value,
    }
    source = SourceConfig(
        id="pubmed_test",
        name="PubMed Test",
        connector="pubmed",
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
        authority_score=0.7,
        evidence_level=EvidenceLevel.UNKNOWN,
        settings=settings,
    )
    with pytest.raises(ValueError, match="must be positive"):
        connector._search_ids(
            Client(),
            source,
            None,
            datetime(2026, 8, 17, tzinfo=UTC),
        )
