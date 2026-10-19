"""Narrow, deterministic A1 weak-verb/V2 fallback for the internal demo.

This is an I01 runtime rule path, not a data generator or E01 scorer. It only
returns text that matches the declared input-derived sentence and Chinese rule
template exactly; unsupported inputs remain unverified.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


RULE_VERSION = "i01-a1-weak-v2-fallback-1.0"
ENDINGS = {"1S": "e", "2S": "st", "3S": "t", "1P": "en", "2P": "t", "3P": "en"}
SUBJECTS = {"ich": "1S", "du": "2S", "er": "3S", "sie": {"3S", "3P"}, "wir": "1P", "ihr": "2P"}
PERSON_ZH = {
    "1S": "第一人称单数", "2S": "第二人称单数", "3S": "第三人称单数",
    "1P": "第一人称复数", "2P": "第二人称复数", "3P": "第三人称复数",
}
SUPPORTED_LEMMAS = {
    "fragen", "hören", "kaufen", "kochen", "lernen", "lieben", "machen",
    "malen", "sagen", "spielen", "suchen", "wohnen",
}
SAFE_FRAGMENT = re.compile(r"^[\wäöüÄÖÜßẞ]+(?:[ '\u2019-][\wäöüÄÖÜßẞ]+)*$", re.UNICODE)


def _expected(task: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(task, dict):
        return None
    if task.get("task_type") != "exercise_feedback" or task.get("grammar_id") not in {
        "PRESENT_WEAK_VERB", "MAIN_CLAUSE_V2"
    }:
        return None
    subject = task.get("subject")
    person_number = task.get("subject_person_number")
    lemma = task.get("required_lemma")
    time_phrase = task.get("required_time_phrase")
    complement = task.get("complement")
    if not isinstance(subject, str) or subject not in SUBJECTS:
        return None
    if person_number not in ENDINGS:
        return None
    allowed = SUBJECTS[subject]
    if person_number not in (allowed if isinstance(allowed, set) else {allowed}):
        return None
    if not isinstance(lemma, str) or lemma not in SUPPORTED_LEMMAS:
        return None
    if not isinstance(time_phrase, str) or not time_phrase or not SAFE_FRAGMENT.fullmatch(time_phrase):
        return None
    if task.get("time_required") is not True:
        return None
    if not isinstance(complement, str) or not complement or not SAFE_FRAGMENT.fullmatch(complement):
        return None
    if task.get("level", "A1") != "A1":
        return None
    finite = lemma[:-2] + ENDINGS[person_number]
    sentence = f"{time_phrase} {finite} {subject} {complement}."
    explanation = (
        f"规则：句首时间状语后的主句中，有限动词位于动词第二位（V2）；"
        f"主语“{subject}”对应{PERSON_ZH[person_number]}，"
        f"“{lemma}”的现在时变位形式为“{finite}”。"
    )
    return sentence, explanation


def verify_rule_fallback(task: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Verify exact fragments, finite form, V2 scaffold and fixed rule template."""
    expected = _expected(task)
    if expected is None:
        return {"status": "NOT_VERIFIED", "reason_codes": ["UNSUPPORTED_OR_AMBIGUOUS_TASK"],
                "rule_version": RULE_VERSION}
    expected_sentence, expected_explanation = expected
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if len(lines) != 2:
        return {"status": "FAIL", "reason_codes": ["FALLBACK_LINE_STRUCTURE_INVALID"],
                "rule_version": RULE_VERSION}
    if lines[0] != expected_sentence:
        return {"status": "FAIL", "reason_codes": ["FALLBACK_SENTENCE_CONTRACT_MISMATCH"],
                "rule_version": RULE_VERSION}
    if lines[1] != expected_explanation:
        return {"status": "FAIL", "reason_codes": ["FALLBACK_EXPLANATION_CONTRACT_MISMATCH"],
                "rule_version": RULE_VERSION}
    return {
        "status": "PASS",
        "rule_version": RULE_VERSION,
        "components": {
            "task_constraints": "PASS", "fragment_preservation": "PASS", "grammar": "PASS",
            "explanation_facts": "PASS", "punctuation": "PASS",
        },
        "coverage": {name: "COMPLETE_FOR_SUPPORTED_RULE" for name in (
            "task_constraints", "fragment_preservation", "grammar", "explanation_facts", "punctuation"
        )},
    }


def build_verified_rule_fallback(task: dict[str, Any]) -> dict[str, Any]:
    expected = _expected(task)
    if expected is None:
        return {"status": "NOT_VERIFIED", "response": None,
                "reason_codes": ["RULE_FALLBACK_UNSUPPORTED_TASK"], "rule_version": RULE_VERSION}
    response = "\n".join(expected)
    verification = verify_rule_fallback(task, response)
    if verification["status"] != "PASS":
        return {"status": "NOT_VERIFIED", "response": None,
                "reason_codes": verification["reason_codes"], "rule_version": RULE_VERSION}
    evidence_value = {"task": task, "response": response, "rule_version": RULE_VERSION}
    digest = hashlib.sha256(json.dumps(evidence_value, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"status": "VERIFIED_RULE_FALLBACK", "response": response,
            "reason_codes": [], "rule_version": RULE_VERSION,
            "evidence_sha256": digest, "validation": verification}
