"""Independent, conservative Oracle for the frozen A1 weak-verb E01 contract.

This module consumes input-only task records and model predictions. It does not
read Silver references, training data, or generator code, and never changes the
existing E01 scoring artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path


ENDINGS = {"1S": "e", "2S": "st", "3S": "t", "1P": "en", "2P": "t", "3P": "en"}
SUBJECTS = {"1S": {"ich"}, "2S": {"du"}, "3S": {"er", "sie"},
            "1P": {"wir"}, "2P": {"ihr"}, "3P": {"sie"}}
SUPPORTED_LEMMAS = {
    "fragen", "hören", "kaufen", "kochen", "lernen", "lieben", "machen",
    "malen", "sagen", "spielen", "suchen", "wohnen",
}
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
HAN_RE = re.compile(r"[\u3400-\u9fff]")
ORACLE_VERSION = "e01-oracle-1.2"


def _coverage(grammar_complete: bool = False) -> dict[str, str]:
    return {
        "fragment_preservation": "COMPLETE_FOR_DECLARED_FRAGMENTS",
        "task_constraints": "COMPLETE_FOR_DECLARED_FIELDS",
        "grammar": "COMPLETE_FOR_SUPPORTED_SCAFFOLD" if grammar_complete else "PARTIAL",
        "explanation_facts": "PARTIAL_EXPLICIT_PATTERN_CHECKS",
        "punctuation": "PARTIAL_TERMINAL_MARK_ONLY",
    }


def _finite_form(lemma: str, person_number: str) -> str | None:
    if lemma not in SUPPORTED_LEMMAS:
        return None
    ending = ENDINGS.get(person_number)
    return lemma[:-2] + ending if ending else None


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _first_nonempty_line(response: str) -> str:
    return next((line.strip() for line in response.splitlines() if line.strip()), "")


def _explanation_checks(explanation: str, task: dict, finite: str, time_phrase: str) -> tuple[list[str], bool]:
    """Return only explicit contradictions; unparsed claims remain UNKNOWN."""
    failures: list[str] = []
    checked = False

    subj_claims = re.findall(r'主语\s*[“「"]\s*([^”」"]+?)\s*[”」"]\s*(?:为|是|位于)', explanation)
    for claim in subj_claims:
        checked = True
        if claim.strip().casefold() != task["subject"].casefold():
            failures.append("EXPLANATION_SUBJECT_CONFLICT")
            break

    expected_person = {
        "1S": "第一人称单数", "2S": "第二人称单数", "3S": "第三人称单数",
        "3S_SIE": "第三人称单数", "1P": "第一人称复数", "2P": "第二人称复数",
        "3P": "第三人称复数",
    }.get(task["subject_person_number"])
    person_phrases = re.findall(
        r"(?:第一|第二|第三)人称(?:单数|复数)|(?:一|二|三)(?:单|复)", explanation
    )
    if person_phrases:
        checked = True
        canonical = set()
        for phrase in person_phrases:
            canonical.add({
                "第一人称单数": "第一人称单数", "一单": "第一人称单数",
                "第二人称单数": "第二人称单数", "二单": "第二人称单数",
                "第三人称单数": "第三人称单数", "三单": "第三人称单数",
                "第一人称复数": "第一人称复数", "一复": "第一人称复数",
                "第二人称复数": "第二人称复数", "二复": "第二人称复数",
                "第三人称复数": "第三人称复数", "三复": "第三人称复数",
            }[phrase])
        if len(canonical) != 1 or expected_person not in canonical:
            failures.append("EXPLANATION_PERSON_NUMBER_CONFLICT")

    lemma_claims = re.findall(r'动词\s*[“「"]\s*([^”」"]+?)\s*[”」"]', explanation)
    if lemma_claims:
        checked = True
        if any(claim.strip().casefold() != task["required_lemma"].casefold() for claim in lemma_claims):
            failures.append("EXPLANATION_LEMMA_CONFLICT")

    ending_claims = re.findall(r"词尾(?:为|是)\s*[-－]\s*([a-zäöüß]+)", explanation, flags=re.I)
    expected_ending = ENDINGS.get(task["subject_person_number"])
    if ending_claims:
        checked = True
        if expected_ending is None or any(
            claim.casefold() != expected_ending.casefold() for claim in ending_claims
        ):
            failures.append("EXPLANATION_ENDING_CONFLICT")

    # Catch explicit constituent-position claims that contradict the task's
    # required initial time phrase. Do not infer truth from a Chinese-character count.
    if re.search(r"主语.{0,20}(?:位于|在).{0,8}(?:句首|句子开头)|主语开头", explanation):
        checked = True
        failures.append("EXPLANATION_SUBJECT_POSITION_CONFLICT")
    if re.search(r"宾语.{0,24}(?:第三句子成分|第三个?成分|第三位)", explanation):
        checked = True
        failures.append("EXPLANATION_OBJECT_POSITION_CONFLICT")
    if re.search(r"(?:变为|变成).{0,8}主语开头|主语开头变序", explanation):
        checked = True
        failures.append("EXPLANATION_WORD_ORDER_CONFLICT")

    expected_v2_claim = re.search(
        r"时间状语.{0,10}句首.{0,20}(?:变位动词|动词).{0,10}第二(?:句子成分|位)", explanation
    )
    if expected_v2_claim:
        checked = True
    elif re.search(r"(?:主语开头|主语位于句首|变为主语开头)", explanation):
        checked = True
        failures.append("EXPLANATION_WORD_ORDER_CONFLICT")

    if not HAN_RE.search(explanation):
        failures.append("EXPLANATION_NOT_CHINESE")
    if not checked and not failures:
        return ["EXPLANATION_FACTS_UNPARSED"], False
    return sorted(set(failures)), checked


def assess(task: dict, prediction: dict) -> dict:
    """Assess one response against input metadata; no reference answer is read."""
    reasons: list[str] = []
    unknown: list[str] = []
    review: list[str] = []

    required = ("grammar_id", "template_family", "subject", "subject_person_number",
                "required_lemma", "required_time_phrase", "complement", "time_required")
    missing = [key for key in required if key not in task]
    if missing:
        return {"sample_id": task.get("sample_id"), "verdict": "UNKNOWN",
                "reason_codes": ["TASK_FIELDS_MISSING:" + ",".join(missing)],
                "fragment_status": "UNKNOWN", "grammar_status": "UNKNOWN",
                "explanation_status": "UNKNOWN", "adjudication": "NOT_ASSESSED",
                "oracle_version": ORACLE_VERSION, "coverage": _coverage()}

    for key in ("subject", "subject_person_number", "required_lemma", "required_time_phrase", "grammar_id"):
        if key in prediction and prediction[key] != task[key]:
            reasons.append("PREDICTION_TASK_METADATA_CONFLICT:" + key)

    person = task["subject_person_number"]
    ending = ENDINGS.get(person)
    lemma = task["required_lemma"]
    if ending is None or str(task["subject"]).casefold() not in SUBJECTS.get(person, set()):
        return {"sample_id": task.get("sample_id"), "verdict": "UNKNOWN",
                "reason_codes": sorted(set(reasons + ["UNSUPPORTED_OR_CONFLICTING_PERSON_NUMBER"])),
                "fragment_status": "UNKNOWN", "grammar_status": "UNKNOWN",
                "explanation_status": "UNKNOWN", "adjudication": "NOT_ASSESSED",
                "oracle_version": ORACLE_VERSION, "coverage": _coverage()}
    finite = _finite_form(lemma, person)
    if finite is None:
        return {"sample_id": task.get("sample_id"), "verdict": "UNKNOWN",
                "reason_codes": sorted(set(reasons + ["UNSUPPORTED_LEMMA_OR_RULE"])),
                "fragment_status": "UNKNOWN", "grammar_status": "UNKNOWN",
                "explanation_status": "UNKNOWN", "adjudication": "NOT_ASSESSED",
                "oracle_version": ORACLE_VERSION, "coverage": _coverage()}
    time_phrase = task["required_time_phrase"]
    complement = task["complement"]
    if task["time_required"] is not True or not isinstance(time_phrase, str) or not time_phrase.strip():
        return {"sample_id": task.get("sample_id"), "verdict": "UNKNOWN",
                "reason_codes": sorted(set(reasons + ["TIME_CONTRACT_UNSUPPORTED"])),
                "fragment_status": "UNKNOWN", "grammar_status": "UNKNOWN",
                "explanation_status": "UNKNOWN", "adjudication": "NOT_ASSESSED",
                "oracle_version": ORACLE_VERSION, "coverage": _coverage()}
    if not isinstance(complement, str) or not isinstance(prediction.get("response"), str):
        return {"sample_id": task.get("sample_id"), "verdict": "UNKNOWN",
                "reason_codes": sorted(set(reasons + ["INVALID_INPUT_OR_RESPONSE_TYPE"])),
                "fragment_status": "UNKNOWN", "grammar_status": "UNKNOWN",
                "explanation_status": "UNKNOWN", "adjudication": "NOT_ASSESSED",
                "oracle_version": ORACLE_VERSION, "coverage": _coverage()}

    response = prediction["response"]
    first = _first_nonempty_line(response)
    words = _words(first)
    term = re.search(r"([.!?])$", first)
    punctuation_status = "PASS" if term else "UNKNOWN"
    if not term:
        unknown.append("TERMINAL_PUNCTUATION_NOT_VERIFIED")
    if re.search(r"[.!?]{2,}$", first):
        reasons.append("INVALID_TERMINAL_PUNCTUATION")
        punctuation_status = "FAIL"
    if re.search(r"[,;:]", first):
        unknown.append("INTERNAL_PUNCTUATION_NOT_VERIFIED")
        punctuation_status = "UNKNOWN"

    # Sentence-initial capitalization is a separate formatting check. It does
    # not change lexical fragment comparison.
    first_alpha = next((c for c in first if c.isalpha()), "")
    if first_alpha and not first_alpha.isupper():
        reasons.append("SENTENCE_INITIAL_CAPITALIZATION")

    task_reasons: list[str] = []
    grammar_reasons: list[str] = []
    time_words = _words(time_phrase)
    complement_words = _words(complement)
    finite_out = ""
    subject_out = ""
    if len(words) < len(time_words) + len(complement_words) + 2:
        task_reasons.append("SENTENCE_TOO_SHORT")
        unknown.append("V2_STRUCTURE_TOO_SHORT_TO_CHECK")
    # Compare required fragments independent of their order. This is a
    # fragment-reordering task: positions are checked by the grammar
    # component, while lexical additions/deletions/replacements are task
    # fidelity failures. Case is ignored here; capitalization is separate.
    expected_tokens = Counter(
        token.casefold()
        for token in time_words + [finite, str(task["subject"])] + complement_words
    )
    actual_tokens = Counter(token.casefold() for token in words)
    token_delta = actual_tokens - expected_tokens
    missing_tokens = expected_tokens - actual_tokens
    if token_delta and not missing_tokens:
        task_reasons.append("FRAGMENT_TOKEN_ADDED")
        if task.get("template_family") == "ood_word_order":
            review.append("LEGACY_FRAGMENT_POLICY_ADJUDICATION_REQUIRED")
    if not _contains_contiguous(words, time_words):
        task_reasons.append("TIME_FRAGMENT_MISSING_CHANGED_OR_MISPLACED")
    if not _contains_contiguous(words, [str(task["subject"])]):
        task_reasons.append("SUBJECT_CHANGED_OR_MISPLACED")
    if not _contains_contiguous(words, [finite]):
        task_reasons.append("SPECIFIED_VERB_FORM_MISMATCH")
    if complement_words and not _contains_contiguous(words, complement_words):
        task_reasons.append("COMPLEMENT_OR_FRAGMENT_REPLACED_OR_REORDERED")
    if token_delta and missing_tokens:
        task_reasons.append("FRAGMENT_TOKEN_REPLACED_OR_ADDED_DELETED")
    fragment_status = "FAIL" if task_reasons else "PASS"

    # Validate the narrow known pattern only when its constituent order is
    # present. Other fragment permutations stay UNKNOWN, not false FAIL.
    time_start = _find_contiguous(words, time_words)
    supported_scaffold = time_start == 0 and len(words) > len(time_words) + 1
    if supported_scaffold:
        finite_out = words[len(time_words)]
        subject_out = words[len(time_words) + 1]
        actual_subject = subject_out.casefold()
        possible_people = [pn for pn, subjects in SUBJECTS.items() if actual_subject in subjects]
        matching_forms = {_finite_form(lemma, pn) for pn in possible_people}
        if not possible_people:
            unknown.append("ACTUAL_SUBJECT_OUTSIDE_ORACLE_COVERAGE")
        elif finite_out not in matching_forms:
            # The known lemma's wrong person ending is deterministically bad;
            # an entirely substituted verb cannot be judged for grammar here.
            alternative_forms = {
                _finite_form(lemma, pn) for pn in ENDINGS
            }
            if finite_out in alternative_forms:
                grammar_reasons.append("FINITE_VERB_DOES_NOT_AGREE_WITH_OUTPUT_SUBJECT")
            else:
                unknown.append("OUTPUT_VERB_OUTSIDE_DECLARED_LEMMA_COVERAGE")
        if finite_out != finite and finite in words and words.index(finite) != len(time_words):
            grammar_reasons.append("EXPECTED_FINITE_VERB_NOT_IN_V2_SLOT")
    else:
        unknown.append("V2_CONSTITUENT_ORDER_OUTSIDE_ORACLE_COVERAGE")
    variable_tokens = {str(task["subject"]).casefold()} | {
        subject.casefold() for subject_set in SUBJECTS.values() for subject in subject_set
    } | {_finite_form(lemma, person).casefold() for person in ENDINGS}
    unexpected_added = set(token_delta) - variable_tokens
    unexpected_missing = set(missing_tokens) - variable_tokens
    if unexpected_added or unexpected_missing or not _contains_contiguous(words, time_words):
        unknown.append("ALTERED_FRAGMENT_SYNTAX_OUTSIDE_ORACLE_COVERAGE")

    lines = [line for line in response.splitlines() if line.strip()]
    explanation = "\n".join(lines[1:]) if len(lines) > 1 else ""
    explanation_errors, explanation_checked = _explanation_checks(explanation, task, finite, time_phrase)
    reasons.extend(task_reasons)
    reasons.extend(grammar_reasons)
    reasons.extend(explanation_errors)
    explanation_status = "FAIL" if explanation_errors and explanation_errors != ["EXPLANATION_FACTS_UNPARSED"] else (
        "PASS" if explanation_checked and not explanation_errors else "UNKNOWN"
    )
    if explanation_status == "UNKNOWN":
        unknown.append("EXPLANATION_FACTS_NOT_FULLY_VERIFIED")

    # A hard content/fact contradiction takes precedence. Legacy task-meaning
    # questions are still recorded independently for adjudication.
    if reasons:
        verdict = "FAIL"
    elif review:
        verdict = "REVIEW_REQUIRED"
    elif unknown:
        verdict = "UNKNOWN"
    else:
        verdict = "PASS"
    return {
        "sample_id": task.get("sample_id"),
        "verdict": verdict,
        "reason_codes": sorted(set(reasons + unknown + review)),
        "fragment_status": fragment_status,
        "task_status": "FAIL" if task_reasons else "PASS",
        "grammar_status": "FAIL" if grammar_reasons else (
            "UNKNOWN" if any(code in unknown for code in (
                "ACTUAL_SUBJECT_OUTSIDE_ORACLE_COVERAGE",
                "OUTPUT_VERB_OUTSIDE_DECLARED_LEMMA_COVERAGE",
                "V2_STRUCTURE_TOO_SHORT_TO_CHECK",
                "V2_CONSTITUENT_ORDER_OUTSIDE_ORACLE_COVERAGE",
                "ALTERED_FRAGMENT_SYNTAX_OUTSIDE_ORACLE_COVERAGE",
            )) else "PASS"
        ),
        "explanation_status": explanation_status,
        "punctuation_status": punctuation_status,
        "adjudication": "TASK_CONTRACT_REVIEW" if review else "NOT_REQUIRED_BY_ORACLE",
        "oracle_version": ORACLE_VERSION,
        "coverage": _coverage(grammar_complete=(
            supported_scaffold and not any(code in unknown for code in (
                "ACTUAL_SUBJECT_OUTSIDE_ORACLE_COVERAGE",
                "OUTPUT_VERB_OUTSIDE_DECLARED_LEMMA_COVERAGE",
                "V2_STRUCTURE_TOO_SHORT_TO_CHECK",
                "V2_CONSTITUENT_ORDER_OUTSIDE_ORACLE_COVERAGE",
                "ALTERED_FRAGMENT_SYNTAX_OUTSIDE_ORACLE_COVERAGE",
            ))
        )),
    }


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(item == candidate for candidate in it) for item in needle)


def _find_contiguous(haystack: list[str], needle: list[str]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    target = [word.casefold() for word in needle]
    for start in range(len(haystack) - len(needle) + 1):
        if [word.casefold() for word in haystack[start:start + len(needle)]] == target:
            return start
    return None


def _contains_contiguous(haystack: list[str], needle: list[str]) -> bool:
    return _find_contiguous(haystack, needle) is not None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="input-only JSONL files; never Silver")
    parser.add_argument("--predictions", type=Path, required=True, help="E01 predictions JSONL")
    parser.add_argument("--output", type=Path, help="new JSONL report; refuses overwrite; default stdout")
    args = parser.parse_args()
    inputs = {}
    for source in args.inputs:
        for row in _read_jsonl(source):
            sid = row.get("sample_id")
            if not sid or sid in inputs:
                print(f"E01_ORACLE_DUPLICATE_INPUT_ID: {sid}", file=sys.stderr)
                return 2
            inputs[sid] = row
    predictions = {row["sample_id"]: row for row in _read_jsonl(args.predictions)}
    if not inputs or inputs.keys() != predictions.keys():
        print("E01_ORACLE_INPUT_PAIRING_FAILED", file=sys.stderr)
        return 2
    results = [assess(inputs[sid], predictions[sid]) for sid in sorted(inputs)]
    lines = [json.dumps(result, ensure_ascii=False, sort_keys=True) for result in results]
    if args.output:
        if args.output.exists() or args.output.resolve() in {p.resolve() for p in args.inputs + [args.predictions]}:
            print(f"E01_ORACLE_REFUSES_OVERWRITE_OR_INPUT_PATH: {args.output}", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
        print(f"E01_ORACLE_OUTPUT={args.output} ROWS={len(results)}", file=sys.stderr)
    else:
        print("\n".join(lines))
    from collections import Counter
    print("E01_ORACLE_COMPLETED", dict(Counter(row["verdict"] for row in results)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
