"""Keyword corpus coverage numbers must be labelled as heuristic, not gold."""
from datetime import UTC, datetime
from pathlib import Path
import runpy


OPS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "clinician_corpus_coverage.py")
)


def row(title, text, *, source_id="pubmed_anxiety", published_at="2026-08-01T00:00:00Z"):
    return {
        "title": title,
        "text": text,
        "source_id": source_id,
        "published_at": published_at,
        "evidence_level": "systematic_review",
        "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
    }


def test_coverage_counts_source_content_but_never_claims_adjudication():
    rows = [
        row(
            "CBT for social anxiety disorder: systematic review",
            "CBT was evaluated for social phobia."
        ),
        row(
            "A review of biomarkers and anxiety",
            "Biomarkers are associated with anxiety conditions."
        ),
    ]
    report = OPS["summarize"](rows, now=datetime(2026, 10, 16, tzinfo=UTC))
    assert report["active_document_versions"] == 2
    assert report["by_source"]["pubmed_anxiety"] == 2
    assert report["topics"]["P02_social_anxiety_CBT"]["heuristic_matching_document_count"] == 1
    assert len(report["recent_treatment_reviews"]) == 1
    assert "Keyword coverage only" in report["disclaimer"]


def test_missing_or_old_publication_date_excluded_from_recent_review_list():
    rows = [
        row("Treatment of anxiety disorders: systematic review",
            "CBT was studied.", published_at="2011-01-01T00:00:00Z"),
        row("Treatment of anxiety disorders: systematic review",
            "CBT was studied.", published_at=None),
    ]
    report = OPS["summarize"](rows, now=datetime(2026, 10, 16, tzinfo=UTC))
    assert report["recent_treatment_reviews"] == []
