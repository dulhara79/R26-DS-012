from __future__ import annotations

import re
from dataclasses import dataclass

from .models import SafetyLevel


@dataclass(slots=True)
class SafetyAssessment:
    level: SafetyLevel
    reason: str | None = None


class SafetyRouter:
    _crisis_patterns = [
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\bi(?:'m| am| feel| have been)?\s+suicidal\b",
        r"\bmy suicidal (?:thoughts|plan|intent)\b",
        r"\b(?:want|plan|intend|going) to (?:die|kill myself|end my life)\b",
        r"\bself[- ]?harm(?:ing)? myself\b",
        r"\bhurt myself\b",
        r"\boverdose myself\b",
        r"\bno reason to live\b",
    ]
    _urgent_patterns = [
        r"\bsevere chest pain\b",
        r"\bcan(?:not|'t) breathe\b",
        r"\bfainted\b",
        r"\bpassed out\b",
        r"\bmedical emergency\b",
        r"\bimmediate danger\b",
    ]

    def assess(self, query: str) -> SafetyAssessment:
        normalized = query.lower()
        if self._has_non_negated_match(normalized, self._crisis_patterns):
            return SafetyAssessment(SafetyLevel.CRISIS, "self_harm_or_suicide_signal")
        if self._has_non_negated_match(normalized, self._urgent_patterns):
            return SafetyAssessment(SafetyLevel.URGENT, "possible_medical_emergency")
        return SafetyAssessment(SafetyLevel.NORMAL)

    @staticmethod
    def _has_non_negated_match(text: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                prefix = text[max(0, match.start() - 24) : match.start()]
                if re.search(r"\b(not|never|no longer|without|do not|don't|am not|i'm not)\s+$", prefix):
                    continue
                return True
        return False


def safety_message(level: SafetyLevel, crisis_resource_text: str) -> str | None:
    if level == SafetyLevel.CRISIS:
        return (
            "Your message may indicate immediate risk. I cannot safely handle this as a normal "
            f"information-search question. {crisis_resource_text}"
        )
    if level == SafetyLevel.URGENT:
        return (
            "Severe or new physical symptoms can have causes other than anxiety. Seek urgent medical "
            "assessment, especially for severe chest pain, fainting, or difficulty breathing."
        )
    return None
