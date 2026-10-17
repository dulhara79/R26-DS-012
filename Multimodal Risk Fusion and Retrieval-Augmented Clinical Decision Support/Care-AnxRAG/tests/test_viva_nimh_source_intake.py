"""The new NIMH pages are staged only, not automatically clinically approved."""
from pathlib import Path

from care_anxrag.models import EvidenceLevel, KnowledgeLayer
from care_anxrag.registry import load_source_registry


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_nimh_sources_are_registered_as_staging_only():
    sources = {item.id: item for item in load_source_registry(ROOT / "config" / "sources.yaml")}
    for name, suffix in (
        ("nimh_social_anxiety", "social-anxiety-disorder-more-than-just-shyness"),
        ("nimh_panic_disorder", "panic-disorder-when-fear-overwhelms"),
    ):
        source = sources[name]
        assert source.connector == "http_page"
        assert source.enabled
        assert source.publish_to_rag
        assert source.auto_promote is False
        assert source.layer == KnowledgeLayer.CLINICAL_CORE
        assert source.evidence_level == EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
        assert len(source.settings["urls"]) == 1
        assert source.settings["urls"][0] == (
            f"https://www.nimh.nih.gov/health/publications/{suffix}"
        )


def test_existing_broad_sources_remain_unchanged_and_unapproved():
    sources = {item.id: item for item in load_source_registry(ROOT / "config" / "sources.yaml")}
    for name in ("nimh_anxiety", "pubmed_anxiety", "nice_update_monitor"):
        assert name in sources
        assert sources[name].auto_promote is False
    assert sources["nice_syndication"].enabled is False
    assert sources["nice_update_monitor"].publish_to_rag is False
