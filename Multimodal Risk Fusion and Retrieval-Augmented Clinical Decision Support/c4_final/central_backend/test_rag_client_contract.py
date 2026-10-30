from __future__ import annotations

import unittest

import rag_client


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_post = None
        self.last_get = None

    def post(self, url, **kwargs):
        self.last_post = (url, kwargs)
        return FakeResponse(200, self.payload)

    def get(self, url, **kwargs):
        self.last_get = (url, kwargs)
        return FakeResponse(
            200,
            {"status": "ok", "database": True, "vector_store": True, "ollama": None},
        )


class RagClientContractTests(unittest.TestCase):
    def test_local_prescreen_does_not_capture_academic_or_negated_queries(self):
        self.assertFalse(
            rag_client.local_crisis_prescreen(
                "What does research say about suicide risk in anxiety disorders?"
            )
        )
        self.assertFalse(rag_client.local_crisis_prescreen("I am not suicidal"))
        self.assertTrue(rag_client.local_crisis_prescreen("I want to kill myself"))
        self.assertTrue(
            rag_client.local_crisis_prescreen(
                "patient mentioned wanting to end his life"
            )
        )

    def test_call_rag_preserves_current_care_wire_contract(self):
        payload = {
            "answer": "Evidence sentence. [S1]",
            "citations": [
                {
                    "citation_id": "S1",
                    "chunk_id": "chunk-1",
                    "title": "Guideline",
                    "source_name": "World Health Organization",
                    "source_id": "who_anxiety",
                    "url": "https://example.org/source",
                    "published_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "evidence_level": "government_health_information",
                    "excerpt": "Evidence sentence.",
                }
            ],
            "confidence": 0.81,
            "conflict_score": 0.0,
            "abstained": False,
            "abstention_reason": None,
            "safety_level": "normal",
            "safety_message": None,
            "latest_evidence_at": "2026-01-01T00:00:00Z",
            "knowledge_base_last_sync_at": "2026-09-27T00:00:00Z",
            "timings_ms": {"total": 12.4},
            "retrieval": None,
        }
        client = FakeClient(payload)
        original_token = rag_client.RAG_TOKEN
        try:
            rag_client.RAG_TOKEN = "test-admin-key"
            result = rag_client.call_rag("What does the evidence say?", client=client)
        finally:
            rag_client.RAG_TOKEN = original_token

        self.assertTrue(result.available)
        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "Evidence sentence. [S1]")
        self.assertEqual(result.citations[0].citation_id, "S1")
        self.assertEqual(result.latest_evidence_at, "2026-01-01T00:00:00Z")
        self.assertEqual(
            result.knowledge_base_last_sync_at,
            "2026-09-27T00:00:00Z",
        )
        self.assertEqual(
            client.last_post[1]["headers"]["X-Admin-Key"],
            "test-admin-key",
        )
        wire = result.to_wire()
        self.assertEqual(wire["latest_evidence_at"], "2026-01-01T00:00:00Z")
        self.assertNotIn("Authorization", client.last_post[1]["headers"])

    def test_health_contract_accepts_care_health_shape(self):
        client = FakeClient({})
        result = rag_client.check_rag_health(client=client)

        self.assertTrue(result["configured"])
        self.assertTrue(result["reachable"])
        self.assertEqual(result["detail"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
