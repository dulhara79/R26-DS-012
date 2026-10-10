import sys
import types

from care_anxrag.nli import CrossEncoderNliClassifier


def test_nli_cross_encoder_uses_cpu_on_macos(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append(kwargs)
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
                )
            )

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(nn=types.SimpleNamespace(Identity=object)))

    CrossEncoderNliClassifier("cross-encoder/nli-deberta-v3-base")

    assert calls[0]["device"] == "cpu"
