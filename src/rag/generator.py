"""LLM-based clinical pre-report generator (Block 15 — RAG pipeline).

Drafts a structured clinical pre-report from a triage decision (the
output of src/rules/triage.py's triage_decision()) plus retrieved
guideline chunks (the top-k results from the RAG retriever). The LLM
only drafts narrative text — it never decides the triage outcome.
Fields that carry clinical/safety weight (predicted_grade, confidence,
action, interval_months, requires_human_review) are copied directly
from the already-computed triage_decision into the final output and are
never taken from the LLM's response, even if it tries to alter them.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

load_dotenv()

_client = None


def get_client():
    """Lazily build (and cache) the OpenAI client from OPENAI_API_KEY."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to a .env file at the repo root."
            )
        _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a clinical documentation assistant drafting a pre-report for an \
ophthalmologist reviewing a diabetic retinopathy screening case.

Rules you MUST follow:
1. Use ONLY information from the provided guideline passages. Do not use \
any outside medical knowledge, even if you believe it to be correct. If \
the provided passages do not cover something, say so explicitly rather \
than filling the gap.
2. The triage decision (action, follow-up interval, and whether human \
review is required) has ALREADY been made by a separate rule engine. \
You must NEVER change, override, or second-guess that decision. Your \
job is only to explain and justify it in clinical language, citing the \
relevant guideline passage(s).
3. Write findings and justification in clear, professional clinical \
language suitable for an ophthalmologist reviewing the case — concise, \
factual, no hedging filler.
4. Respond with a single JSON object only, no prose outside the JSON, \
matching exactly this schema:
{
  "findings": "<string: brief clinical findings summary, grounded ONLY in the provided passages>",
  "guideline_applied": "<string: which guideline section(s) justify this decision, citing chunk_id(s)>",
  "source_chunks": ["<chunk_id>", ...]
}
Do not include any other keys. Do not include the triage decision fields \
(action, interval, grade, etc.) — those are added separately, after your \
response, and are not yours to set."""


def _build_user_prompt(triage_decision, retrieved_chunks):
    chunks_text = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    return (
        f"Triage decision (already final — do not modify):\n"
        f"- Predicted grade: {triage_decision['predicted_grade']}\n"
        f"- Model confidence: {triage_decision['confidence']}\n"
        f"- Action: {triage_decision['action']}\n"
        f"- Follow-up interval (months): {triage_decision['interval_months']}\n"
        f"- Requires human review: {triage_decision['requires_human_review']}\n"
        f"- Rule engine notes: {triage_decision.get('notes', [])}\n\n"
        f"Retrieved guideline passages:\n\n{chunks_text}\n\n"
        f"Draft the findings and guideline justification as instructed."
    )


def _fallback_report(triage_decision, error_message):
    """Returned when the API call or JSON parsing fails, so a broken
    generation step degrades gracefully instead of crashing the pipeline.

    Fails safe: requires_human_review is forced to True regardless of
    what the rule engine decided, since an ophthalmologist should see
    the case directly if the drafting step itself is broken.
    """
    return {
        "predicted_grade": triage_decision.get("predicted_grade"),
        "confidence": triage_decision.get("confidence"),
        "findings": None,
        "guideline_applied": None,
        "action": triage_decision.get("action"),
        "interval_months": triage_decision.get("interval_months"),
        "requires_human_review": True,
        "macular_status_assessed": False,
        "source_chunks": [],
        "error": error_message,
    }


def generate_report(triage_decision, retrieved_chunks, client=None):
    """Generate a structured clinical pre-report.

    triage_decision: dict — the output of src/rules/triage.py's
        triage_decision(). Its predicted_grade, confidence, action,
        interval_months, and requires_human_review are copied verbatim
        into the result; the LLM never sets them.
    retrieved_chunks: list of dicts, each with at least 'chunk_id' and
        'text' — the top-k results from the RAG retriever.
    client: optional pre-built OpenAI-compatible client. Inject a mock
        here for testing without hitting the real API (see __main__).

    Returns a dict matching the fixed output schema. Never raises on an
    API error or a malformed LLM response — falls back to an
    error-flagged dict instead.
    """
    try:
        active_client = client or get_client()
        user_prompt = _build_user_prompt(triage_decision, retrieved_chunks)

        response = active_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        return _fallback_report(triage_decision, f"API call failed: {e}")

    try:
        parsed = json.loads(raw_content)
        findings = parsed["findings"]
        guideline_applied = parsed["guideline_applied"]
        llm_source_chunks = parsed.get("source_chunks", [])

        # Defensive: only keep chunk_ids that were actually retrieved —
        # never trust a hallucinated citation from the LLM.
        valid_chunk_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}
        source_chunks = [cid for cid in llm_source_chunks if cid in valid_chunk_ids]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return _fallback_report(triage_decision, f"Failed to parse LLM response: {e}")

    return {
        "predicted_grade": triage_decision["predicted_grade"],
        "confidence": triage_decision["confidence"],
        "findings": findings,
        "guideline_applied": guideline_applied,
        "action": triage_decision["action"],
        "interval_months": triage_decision["interval_months"],
        "requires_human_review": triage_decision["requires_human_review"],
        "macular_status_assessed": False,
        "source_chunks": source_chunks,
    }


if __name__ == "__main__":
    # Hardcoded example with the API call mocked/stubbed out entirely —
    # lets us review the prompt structure and output schema without
    # spending any API budget or needing OPENAI_API_KEY to be set.
    from unittest.mock import MagicMock

    example_triage_decision = {
        "predicted_grade": 3,
        "confidence": 0.90,
        "requires_human_review": False,
        "action": "urgent_referral",
        "interval_months": 1,
        "notes": [],
    }

    example_retrieved_chunks = [
        {
            "chunk_id": "grade_definitions_icdrss",
            "text": (
                "Grade 3 — Severe Non-Proliferative DR (NPDR): more than 20 "
                "intraretinal haemorrhages in all four quadrants, definite venous "
                "beading in two or more quadrants, or prominent IRMA."
            ),
        },
        {
            "chunk_id": "urgent_referral_criteria",
            "text": (
                "Extensive intraretinal haemorrhage (Grade 3 criteria) — high risk "
                "of progression to proliferative disease within a short timeframe."
            ),
        },
        {
            "chunk_id": "recommended_follow_up_intervals",
            "text": (
                "Grade 3 | Severe NPDR | 1 month (urgent referral). This system "
                "applies a stricter 1-month interval by design, given the real risk "
                "of progression from severe NPDR to proliferative disease."
            ),
        },
    ]

    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "findings": (
            "Fundus imaging shows extensive intraretinal haemorrhage consistent "
            "with severe non-proliferative diabetic retinopathy (Grade 3), meeting "
            "the 4-2-1 rule criteria for this grade."
        ),
        "guideline_applied": (
            "Per [grade_definitions_icdrss], Grade 3 is defined by >20 intraretinal "
            "haemorrhages, venous beading, or IRMA. Per [urgent_referral_criteria] "
            "and [recommended_follow_up_intervals], this grade carries a high risk "
            "of progression to proliferative disease and warrants a shortened "
            "1-month interval rather than the standard 3-4 month severe-NPDR "
            "follow-up, as an intentional safety margin."
        ),
        "source_chunks": [
            "grade_definitions_icdrss",
            "urgent_referral_criteria",
            "recommended_follow_up_intervals",
        ],
    })
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_report(example_triage_decision, example_retrieved_chunks, client=mock_client)

    print(json.dumps(result, indent=2))
