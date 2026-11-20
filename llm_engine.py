"""
llm_engine.py
================================================================================
AI-Powered OBE Syllabus Generator Microservice
Milestone 1 — Deliverable 2: Two-Pass Ollama Query Engine & Error Retry Logic

Architecture
------------
For EVERY course, the engine runs a TWO-PASS local LLM conversation with
Qwen 2.5 (served by Ollama):

    PASS 1 — DRAFT
        The model freely drafts 4-6 Course Outcomes distributed across the
        Knowledge / Skills / Attitude (K/S/A) domains, given the course
        metadata.

    PASS 2 — SELF-AUDIT & REFINE
        The Pass-1 draft is fed straight back to the model together with an
        explicit OBE compliance checklist (banned verbs, domain verb banks,
        PO range, K/S/A coverage, sequential numbering). The model must
        audit its own draft and return a corrected, final JSON object.

The Pass-2 output is then parsed and validated against the strict Pydantic
contracts in `obe_schemas.py`. If validation fails (bad JSON syntax OR a
Pydantic ValidationError), the concrete error message is injected back into
the conversation as a new user turn and the model is asked to correct it —
this is the automated ERROR RETRY LOOP, bounded by `max_retries`.

This module also performs BATCH PROCESSING for the four 3rd Year, 1st
Semester CS/IT subjects:
    1. Computer Graphics Programming
    2. Database Systems 1
    3. Data Mining and Analytics
    4. Software Engineering

Usage
-----
    # Real local Ollama run — auto-detects whichever model you already have
    # installed (prefers a "qwen" model if more than one is installed).
    python llm_engine.py

    # Or pin a specific model explicitly:
    python llm_engine.py --model qwen2.5:7b --retries 3

    # Offline / CI-safe demo run using the built-in MockProvider
    # (exercises the exact same validation + retry code path, no Ollama needed)
    python llm_engine.py --mock --output sample_validated_output.json
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import ValidationError

from obe_schemas import (
    ALLOWED_PO_RANGE,
    BANNED_VERBS,
    DOMAIN_LEVEL_DEFAULT,
    DOMAIN_LEVELS,
    DOMAIN_VERB_BANKS,
    LEVEL_VERB_BANKS,
    BatchGenerationResult,
    CourseMetadata,
    CourseOutcomePayload,
    GenerationAudit,
    ValidatedCourseRecord,
)


# ==============================================================================
# 1. BATCH COURSE CATALOG — 3rd Year, 1st Semester CS/IT
# ==============================================================================

COURSE_CATALOG: List[CourseMetadata] = [
    CourseMetadata(
        course_code="CS321",
        course_title="Computer Graphics Programming",
        course_description=(
            "Covers the mathematical and programmatic foundations of 2D and 3D "
            "computer graphics, including transformation pipelines, rasterization, "
            "shader programming, lighting models, and real-time rendering using "
            "modern graphics APIs."
        ),
        target_pos=[1, 2, 3, 5],
    ),
    CourseMetadata(
        course_code="CS322",
        course_title="Database Systems 1",
        course_description=(
            "Introduces relational database design theory, normalization, "
            "SQL query authoring, transaction management, and the implementation "
            "of a normalized relational database for a real-world application."
        ),
        target_pos=[1, 2, 3, 4],
    ),
    CourseMetadata(
        course_code="CS323",
        course_title="Data Mining and Analytics",
        course_description=(
            "Explores data preprocessing, exploratory data analysis, classification, "
            "clustering, and association-rule mining techniques, applying analytical "
            "pipelines to real-world datasets to extract actionable insight."
        ),
        target_pos=[1, 2, 3, 6],
    ),
    CourseMetadata(
        course_code="CS324",
        course_title="Software Engineering",
        course_description=(
            "Covers the software development life cycle, requirements engineering, "
            "software design principles, agile project management, quality assurance, "
            "and team-based delivery of a working software system."
        ),
        target_pos=[1, 2, 3, 4, 7],
    ),
]


# ==============================================================================
# 2. ENGINE CONFIGURATION
# ==============================================================================

@dataclass
class EngineConfig:
    model_name: str = "qwen2.5:7b"
    pass1_temperature: float = 0.4   # drafting: a bit of creative variance
    pass2_temperature: float = 0.1   # self-audit: deterministic correction
    max_retries: int = 3
    request_timeout_seconds: float = 300.0
    outcomes_min: int = 4
    outcomes_max: int = 6
    generation_preset: str = "Balanced"
    week_count: int = 8
    concurrency: int = 1
    keep_alive: str = "10m"
    num_predict: int = 1800

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if not 0.0 <= self.pass1_temperature <= 2.0:
            raise ValueError("pass1_temperature must be between 0 and 2.")
        if not 0.0 <= self.pass2_temperature <= 2.0:
            raise ValueError("pass2_temperature must be between 0 and 2.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if not self.generation_preset:
            raise ValueError("generation_preset cannot be empty.")
        if self.generation_preset not in ("Fast", "Balanced", "Quality"):
            raise ValueError("generation_preset must be Fast, Balanced, or Quality.")
        if self.concurrency < 1 or self.concurrency > 2:
            raise ValueError("concurrency must be 1 or 2.")
        if self.num_predict < 256:
            raise ValueError("num_predict must be at least 256.")
        if self.week_count < 1 or self.week_count > 12:
            raise ValueError("week_count must be between 1 and 12.")


def apply_generation_preset(
    preset: str,
    model_name: str,
    retries: Optional[int] = None,
    week_count: int = 8,
) -> EngineConfig:
    preset = preset.strip().title()
    if preset not in ("Fast", "Balanced", "Quality"):
        raise ValueError("Preset must be Fast, Balanced, or Quality.")

    if preset == "Fast":
        max_retries = 1
        timeout = 180.0
        num_predict = 900
        pass1_temperature = 0.25
        pass2_temperature = 0.05
    elif preset == "Balanced":
        max_retries = 2
        timeout = 300.0
        num_predict = 1800
        pass1_temperature = 0.35
        pass2_temperature = 0.10
    else:
        max_retries = 3
        timeout = 600.0
        num_predict = 2600
        pass1_temperature = 0.40
        pass2_temperature = 0.10

    if retries is not None:
        max_retries = retries

    num_predict += week_count * 140

    return EngineConfig(
        model_name=model_name,
        max_retries=max_retries,
        request_timeout_seconds=timeout,
        pass1_temperature=pass1_temperature,
        pass2_temperature=pass2_temperature,
        generation_preset=preset,
        week_count=week_count,
        num_predict=num_predict,
    )


# ==============================================================================
# 3. LLM PROVIDER ABSTRACTION (real Ollama + offline Mock for testing)
# ==============================================================================

class LLMProvider(ABC):
    """Thin abstraction over 'send chat messages, get JSON-mode text back'.
    Lets the exact same two-pass + retry pipeline run against either the
    real local Ollama/Qwen2.5 runtime, or a scripted MockProvider for
    dependency-free testing of the schema/retry logic."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        ...


def _get_ollama_module():
    try:
        import ollama  # local import: keeps the module importable without the dep
    except ImportError as exc:
        raise RuntimeError(
            "The 'ollama' Python package is required. Install with "
            "`pip install ollama` and ensure `ollama serve` is running locally."
        ) from exc
    return ollama


def _extract_model_name(entry: Any) -> Optional[str]:
    """Handles both dict-style and object-style entries returned by
    different `ollama` python client versions."""
    if isinstance(entry, dict):
        return entry.get("model") or entry.get("name")
    return getattr(entry, "model", None) or getattr(entry, "name", None)


def auto_detect_model(prefer_substring: str = "qwen") -> str:
    """Auto-detects a locally installed Ollama model so the user never has
    to type a model name. Prefers a model whose name contains `prefer_substring`
    (default 'qwen'); otherwise falls back to the first installed model.
    Raises RuntimeError with a helpful message if no models are installed
    or the Ollama daemon isn't reachable."""
    ollama = _get_ollama_module()
    try:
        listing = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach the local Ollama daemon. Make sure Ollama is "
            "installed and running (`ollama serve`), then try again."
        ) from exc

    raw_models = listing.get("models") if isinstance(listing, dict) else getattr(listing, "models", None)
    raw_models = raw_models or []
    names = [n for n in (_extract_model_name(m) for m in raw_models) if n]

    if not names:
        raise RuntimeError(
            "No local Ollama models found. Pull one first, e.g.:\n"
            "  ollama pull qwen2.5:7b"
        )

    for name in names:
        if prefer_substring.lower() in name.lower():
            return name

    return names[0]


def list_installed_models(prefer_substring: str = "qwen") -> List[str]:
    """Return every installed Ollama model, with the preferred model first."""
    ollama = _get_ollama_module()
    try:
        listing = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach the local Ollama daemon. Make sure Ollama is "
            "installed and running (`ollama serve`), then try again."
        ) from exc

    raw_models = listing.get("models") if isinstance(listing, dict) else getattr(listing, "models", None)
    raw_models = raw_models or []
    names = [n for n in (_extract_model_name(m) for m in raw_models) if n]
    if not names:
        raise RuntimeError(
            "No local Ollama models found. Pull one first, e.g.\n"
            "  ollama pull qwen2.5:7b"
        )

    preferred = [name for name in names if prefer_substring.lower() in name.lower()]
    return preferred + [name for name in names if name not in preferred]


class OllamaProvider(LLMProvider):
    """Queries a locally running Ollama daemon in strict JSON mode. This is
    the production provider for the microservice. If no model_name is given,
    it auto-detects whichever model the user already has installed."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout_seconds: float = 300.0,
        keep_alive: str = "10m",
        num_predict: int = 1800,
    ):
        self._ollama = _get_ollama_module()
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self._timeout_seconds = timeout_seconds
        if hasattr(self._ollama, "Client"):
            try:
                self._ollama = self._ollama.Client(timeout=timeout_seconds)
            except Exception:
                pass
        self.model_name = model_name or auto_detect_model()

    def chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        try:
            response = self._ollama.chat(
                model=self.model_name,
                messages=messages,
                format="json",
                options={
                    "temperature": temperature,
                    "num_predict": self.num_predict,
                },
                keep_alive=self.keep_alive,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "timed out" in message or "timeout" in message:
                raise RuntimeError(
                    f"Ollama timed out after {self._timeout_seconds}s. "
                    "Try Fast mode, increase timeout, or use a smaller model."
                ) from exc
            if "connection" in message or "refused" in message:
                raise RuntimeError(
                    "Could not connect to the Ollama daemon. Make sure Ollama is running "
                    "(run `ollama serve`) and the model is installed."
                ) from exc
            raise
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned an unexpected response without message content.")
        return content


class MockProvider(LLMProvider):
    """Deterministic, offline stand-in for Ollama. Returns a pre-scripted
    queue of raw JSON-text responses, one per `chat()` call, in order.
    Used for local/CI verification of the schema-validation and error-retry
    logic without requiring a running Ollama daemon or GPU."""

    def __init__(
        self,
        scripted_responses: List[str],
        description_responses: Optional[List[str]] = None,
        custom_responses: Optional[List[str]] = None,
    ):
        self._queue = list(scripted_responses)
        self._description_queue = list(description_responses or [])
        self._custom_queue = list(custom_responses or [])
        self.calls_made = 0

    @staticmethod
    def _add_mock_weekly_schedule(raw: str, messages: List[Dict[str, str]]) -> str:
        user_text = "\n".join(
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        )
        match = re.search(
            r"(?:Requested weekly schedule entries:|Keep exactly)\s*(\d+)",
            user_text,
            re.IGNORECASE,
        )
        if not match:
            return raw
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not isinstance(payload, dict) or "weekly_schedule" in payload:
            return raw
        outcomes = payload.get("course_outcomes")
        if not isinstance(outcomes, list):
            return raw
        clo_numbers = [
            outcome["clo_number"]
            for outcome in outcomes
            if isinstance(outcome, dict) and isinstance(outcome.get("clo_number"), int)
        ]
        if not clo_numbers:
            return raw
        title = payload.get("course_title") or "course"
        week_count = int(match.group(1))
        payload["weekly_schedule"] = [
            {
                "week_number": week,
                "topic": f"{title}: week {week} focus",
                "learning_outcome": (
                    f"Apply a core {title} concept through a focused weekly exercise."
                ),
                "activities": [
                    "Review a guided example.",
                    "Complete a short applied exercise.",
                ],
                "mapped_clos": [clo_numbers[(week - 1) % len(clo_numbers)]],
            }
            for week in range(1, week_count + 1)
        ]
        return json.dumps(payload)

    def chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        self.calls_made += 1
        improvement_prompt = bool(
            messages
            and "Improve a course description" in "\n".join(
                message.get("content", "") for message in messages
            )
        )
        if improvement_prompt:
            if self._description_queue:
                return self._description_queue.pop(0)
            user_text = "\n".join(
                message.get("content", "")
                for message in messages
                if message.get("role") == "user"
            )
            draft = ""
            marker = "User-written description:\n"
            if marker in user_text:
                draft = user_text.split(marker, 1)[1].split("\n\n", 1)[0].strip()
            return json.dumps({
                "course_description": (
                    f"{draft.rstrip('.')} It emphasizes practical understanding, "
                    "ethical application, and relevant academic concepts."
                )
            })
        description_prompt = bool(
            messages
            and (
                "course description" in messages[0]["content"].lower()
                or "academic cs/it course description" in messages[0]["content"].lower()
            )
        )
        if description_prompt:
            if not self._description_queue:
                user_text = " ".join(
                    message.get("content", "")
                    for message in messages
                    if message.get("role") == "user"
                )
                title = "Custom Subject"
                marker = "Course Title:"
                if marker in user_text:
                    title = user_text.split(marker, 1)[1].splitlines()[0].strip()
                return json.dumps({
                    "course_description": (
                        f"Introduces the foundations, core concepts, and practical applications "
                        f"of {title}. Students examine relevant methods and develop applied "
                        f"skills for academic and professional contexts."
                    )
                })
            return self._description_queue.pop(0)
        if self._custom_queue:
            user_text = " ".join(
                message.get("content", "")
                for message in messages
                if message.get("role") == "user"
            ).lower()
            if "cybersecurity fundamentals" in user_text:
                return self._add_mock_weekly_schedule(
                    self._custom_queue.pop(0), messages)
        if not self._queue:
            raise RuntimeError("MockProvider response queue exhausted.")
        return self._add_mock_weekly_schedule(self._queue.pop(0), messages)


# ==============================================================================
# 4. PROMPT ENGINEERING
# ==============================================================================

_SCHEMA_REFERENCE = """{
  "course_code": string,
  "course_title": string,
  "course_description": string,
  "target_pos": [int],
  "course_outcomes": [
    {
      "clo_number": 1,
      "domain": "Knowledge" | "Skills" | "Attitude",
      "bloom_level": string (must match the declared domain's level list below),
      "co_description": "Active domain-appropriate verb + measurable outcome statement.",
      "mapped_po": [int]
    }
  ],
  "weekly_schedule": [
    {
      "week_number": 1,
      "topic": "string",
      "learning_outcome": "measurable outcome",
      "activities": ["string"],
      "mapped_clos": [1],
      "contact_hours": 3.0
    }
  ]
}"""

_LEVEL_VERB_REFERENCE = "\n".join(
    f"{domain} / {level}: {', '.join(verbs)}"
    for domain, levels in LEVEL_VERB_BANKS.items()
    for level, verbs in levels.items()
)

SYSTEM_PROMPT_PASS1_DRAFT = f"""Create OBE Course Outcomes for a CS/IT course.
RULES: 4-6 CLOs; CLO 1..N; domains Knowledge/Skills/Attitude; all 3 domains required.
Also create exactly the requested number of sequential weekly_schedule entries, each with a topic, one measurable learning_outcome, activities, and mapped_clos referencing valid CLO numbers.
Knowledge levels: {DOMAIN_LEVELS['Knowledge']}; Skills: {DOMAIN_LEVELS['Skills']}; Attitude: {DOMAIN_LEVELS['Attitude']}.
Each co_description MUST start with an exact verb from the matching domain AND bloom_level below.
Do not substitute synonyms, use gerunds, or add a different verb before the approved verb.
{_LEVEL_VERB_REFERENCE}
Never use: {sorted(BANNED_VERBS)}. mapped_po must be in PO range {ALLOWED_PO_RANGE} and the course target_pos.
Schema:
{_SCHEMA_REFERENCE}
Return pure JSON only."""


SYSTEM_PROMPT_PASS2_AUDIT = f"""Audit and repair this OBE JSON.
Rules: 4-6 CLOs; sequential 1..N; Knowledge/Skills/Attitude; all 3 present.
Preserve exactly the requested number of sequential weekly_schedule entries; ensure each includes a topic, measurable learning_outcome, activities, and valid mapped_clos.
Allowed levels — Knowledge: {DOMAIN_LEVELS['Knowledge']}; Skills: {DOMAIN_LEVELS['Skills']}; Attitude: {DOMAIN_LEVELS['Attitude']}.
The first word of each co_description must exactly match a verb for its domain and bloom_level:
{_LEVEL_VERB_REFERENCE}
Rewrite non-matching verbs as a complete grammatical sentence; do not just swap a word if the sentence becomes awkward.
Banned: {sorted(BANNED_VERBS)}.
mapped_po: PO range {ALLOWED_PO_RANGE}, and within target_pos.
Return corrected pure JSON only."""


def _build_pass1_user_prompt(meta: CourseMetadata, week_count: int) -> str:
    return f"""Draft Course Outcomes for the following course:
Course Code: {meta.course_code}
Course Title: {meta.course_title}
Course Description: {meta.course_description}
Year/Semester: {meta.year_level}, {meta.semester}
Target Program Outcomes (POs): {meta.target_pos}
Requested weekly schedule entries: {week_count}

Use only a verb listed for the outcome's exact domain and bloom_level. The
description's first word must be that verb, not a synonym. Include 4-6 CLOs
and all three K/S/A domains. Create exactly {week_count} numbered weeks, each
with one measurable weekly learning outcome, activities, and CLO mappings.
Return pure JSON only."""


def _build_pass2_user_prompt(draft_json_text: str, week_count: int) -> str:
    return f"""Audit and correct every outcome. The first word must be an exact
verb listed for its domain and bloom_level; rewrite any invalid sentence so it
reads naturally while preserving its course meaning. Keep exactly {week_count}
sequential weekly_schedule entries. Each week must have a measurable
learning_outcome, activities, and mapped_clos that refer to existing CLOs.

{draft_json_text}

Return corrected pure JSON only."""


def find_builtin_subject(identifier: str) -> Optional[CourseMetadata]:
    """Find a built-in subject by exact course code or title (case-insensitive)."""
    normalized = identifier.strip().lower()
    if not normalized:
        return None
    for meta in COURSE_CATALOG:
        if normalized in (meta.course_code.lower(), meta.course_title.lower()):
            return meta
    return None


def next_custom_course_code(existing_codes: Iterable[str] = ()) -> str:
    """Return the next unused code in the built-in catalog's prefix/number series."""
    parsed = []
    for meta in COURSE_CATALOG:
        match = re.fullmatch(r"([A-Z]+)(\d+)", meta.course_code.strip().upper())
        if match:
            parsed.append((match.group(1), int(match.group(2))))
    if not parsed:
        raise ValueError("Built-in course codes do not define a prefix-and-number pattern.")

    prefixes = [prefix for prefix, _number in parsed]
    prefix = max(dict.fromkeys(prefixes), key=prefixes.count)
    next_number = max(number for code_prefix, number in parsed if code_prefix == prefix) + 1
    used = {str(code).strip().upper() for code in existing_codes}
    while f"{prefix}{next_number}" in used:
        next_number += 1
    return f"{prefix}{next_number}"


def build_custom_subject(
    title: str,
    course_code: str,
    course_description: Optional[str] = None,
    year_level: str = "3rd Year",
    semester: str = "1st Semester",
    units: float = 3.0,
    target_pos: Optional[List[int]] = None,
) -> CourseMetadata:
    """Build and validate a user-entered subject."""
    title = title.strip()
    if not title:
        raise ValueError("A subject title is required.")
    code = course_code.strip().upper()
    if not code:
        code = "CUSTOM"
    description = course_description.strip() if course_description else None
    if description is not None and len(description) < 10:
        raise ValueError("The supplied course description must contain at least 10 characters.")
    return CourseMetadata(
        course_code=code,
        course_title=title,
        course_description=description,
        year_level=year_level.strip() or "3rd Year",
        semester=semester.strip() or "1st Semester",
        units=units,
        target_pos=target_pos if target_pos is not None else [1, 2, 3, 4, 5],
    )


DESCRIPTION_SYSTEM_PROMPT = """You are a course description generator for a College of Computer Studies.
Write a concise, academic course description in 2-4 sentences. Cover the subject purpose,
core topics, practical or application context, and relevance to IT/CS students.
Return ONLY pure JSON: {"course_description": "..."}."""


def generate_course_description(
    meta: CourseMetadata,
    provider: LLMProvider,
    temperature: float = 0.4,
    debug: bool = False,
    emit: "Callable[[str], None]" = print,
) -> str:
    """Generate and validate a course description for a custom or rewritten subject."""
    if meta.course_description:
        return meta.course_description

    messages = [
        {
            "role": "system",
            "content": "Write a concise academic CS/IT course description.",
        },
        {
            "role": "user",
            "content": (
                f"{meta.course_code} | {meta.course_title} | "
                f"{meta.year_level}, {meta.semester} | POs {meta.target_pos}\n"
                "Write 2-3 sentences covering purpose, core topics, and practical relevance.\n"
                "Return ONLY pure JSON: {\"course_description\": \"...\"}"
            ),
        },
    ]
    raw = provider.chat(messages, temperature=temperature)
    if debug:
        emit(
            f"----- RAW COURSE DESCRIPTION OUTPUT ({meta.course_code}) "
            "— verbatim from the model -----"
        )
        emit(raw)
        emit("----- END RAW COURSE DESCRIPTION OUTPUT -----\n")

    parsed = _parse_json_or_raise(raw)
    description = parsed.get("course_description")
    if not isinstance(description, str):
        raise ValueError("Course description JSON must contain a string 'course_description' field.")
    description = description.strip()
    if len(description) < 10:
        raise ValueError("Generated course description must contain at least 10 characters.")
    return description


def improve_course_description(
    meta: CourseMetadata,
    provider: LLMProvider,
    temperature: float = 0.3,
    debug: bool = False,
    emit: "Callable[[str], None]" = print,
) -> str:
    """Improve a user-written description without changing its subject, scope, or intent."""
    if not meta.course_description:
        raise ValueError("A course description is required before it can be improved.")

    messages = [
        {
            "role": "system",
            "content": (
                "Improve a course description for a College of Computer Studies syllabus. "
                "Preserve the author's intent and scope. Write 2-3 concise academic "
                "sentences covering purpose, core topics, and practical relevance."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Course Code: {meta.course_code}\n"
                f"Course Title: {meta.course_title}\n"
                f"Year/Semester: {meta.year_level}, {meta.semester}\n"
                f"Target POs: {meta.target_pos}\n"
                f"User-written description:\n{meta.course_description}\n\n"
                "Improve this description without changing its meaning. "
                "Return ONLY pure JSON: {\"course_description\": \"...\"}"
            ),
        },
    ]
    raw = provider.chat(messages, temperature=temperature)
    if debug:
        emit(
            f"----- RAW IMPROVED DESCRIPTION OUTPUT ({meta.course_code}) "
            "— verbatim from the model -----"
        )
        emit(raw)
        emit("----- END RAW IMPROVED DESCRIPTION OUTPUT -----\n")

    parsed = _parse_json_or_raise(raw)
    description = parsed.get("course_description")
    if not isinstance(description, str):
        raise ValueError(
            "Improved course description JSON must contain a string "
            "'course_description' field."
        )
    description = description.strip()
    if len(description) < 10:
        raise ValueError("Improved course description must contain at least 10 characters.")
    return description


# ==============================================================================
# 5. JSON CLEANUP HELPERS
# ==============================================================================

def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def _parse_json_or_raise(raw: str) -> Dict[str, Any]:
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON syntax generated by model: {exc}") from exc


# ==============================================================================
# 6. AUTO-REPAIR LAYER — mechanically fixes the mistakes small/weak local
#    models make most often, BEFORE Pydantic validation runs. This does not
#    weaken the schema (Pydantic still rejects anything malformed or
#    incomplete) — it just prevents predictable, mechanical mismatches from
#    burning through retries on a model too small to track every rule at once.
# ==============================================================================

# Verbs the model commonly borrows from the WRONG domain, remapped to the
# closest equivalent verb in each of the three domains.
_VERB_CROSS_MAP: Dict[str, Dict[str, str]] = {
    "apply":      {"Skills": "implement", "Attitude": "practice"},
    "analyze":    {"Skills": "test",      "Attitude": "assess"},
    "evaluate":   {"Skills": "test",      "Attitude": "assess"},
    "assess":     {"Skills": "test",      "Knowledge": "evaluate"},
    "formulate":  {"Skills": "construct", "Attitude": "advocate"},
    "design":     {"Skills": "construct", "Attitude": "advocate"},
    "develop":    {"Skills": "build",     "Attitude": "cultivate"},
    "recognize":  {"Skills": "demonstrate", "Attitude": "value"},
    "describe":   {"Skills": "demonstrate", "Attitude": "observe"},
    "explain":    {"Skills": "demonstrate", "Attitude": "advocate"},
    "identify":   {"Skills": "demonstrate", "Attitude": "observe"},
    "solve":      {"Skills": "implement",   "Attitude": "practice"},
    "compare":    {"Skills": "test",        "Attitude": "assess"},
    "construct":  {"Knowledge": "design",   "Attitude": "cultivate"},
    "implement":  {"Knowledge": "apply",    "Attitude": "practice"},
    "demonstrate": {"Knowledge": "apply",   "Attitude": "exhibit"},
    "value":      {"Knowledge": "assess",   "Skills": "prioritize"},
    "advocate":   {"Knowledge": "justify",  "Skills": "champion"},
    "respond":    {"Attitude": "practice"},
}


def _repair_bloom_level(domain: str, level: Any) -> str:
    valid_levels = DOMAIN_LEVELS.get(domain, [])
    if isinstance(level, str) and level in valid_levels:
        return level
    return DOMAIN_LEVEL_DEFAULT.get(domain, valid_levels[0] if valid_levels else "")


def _repair_description_verb(domain: str, level: Any, description: Any) -> Any:
    if not isinstance(description, str) or not description.strip():
        return description
    words = description.strip().split()
    if not words:
        return description
    first_raw = words[0]
    first = first_raw.lower().rstrip(",.:;")
    rest = " ".join(words[1:])
    bank = DOMAIN_VERB_BANKS.get(domain, [])

    if domain == "Knowledge" and first == "apply" and level == "Remember":
        prefix = "understanding of "
        if rest.lower().startswith(prefix):
            body = rest[len(prefix):]
            body_lower = body.lower()
            for suffix in (" through recall", " through memorization"):
                if body_lower.endswith(suffix):
                    body = body[:-len(suffix)]
                    break
            body = body.strip().rstrip(".!?")
            if body:
                return f"Describe {body}."

    if domain == "Knowledge" and first == "break" and len(words) > 1 and words[1].lower() == "down":
        subject = " ".join(words[2:])
        if " into " in subject:
            subject, components = subject.split(" into ", 1)
            components = components.strip().rstrip(".!?")
            component_phrase = (
                f"its {components}"
                if components.lower() in ("components", "parts", "elements")
                else f"the {components}"
            )
            return f"Analyze {subject.strip()} by identifying {component_phrase}."

    if domain == "Attitude" and first == "use":
        if level == "Valuing":
            return "Value critical thinking as an approach to solving problems."
        if level == "Organizing":
            return "Prioritize critical thinking when solving problems."
        if level == "Characterizing":
            return "Model critical thinking when solving problems."
        return "Practice critical thinking to solve problems."

    if domain == "Attitude" and first == "assess":
        topic = rest.strip().rstrip(".!?")
        if topic.lower().startswith("the validity of "):
            topic = topic[len("the validity of "):]
        return f"Advocate for evidence-based evaluation of {topic}."

    if domain == "Attitude" and first == "respond":
        if rest.lower().startswith("to "):
            return f"Practice appropriate responses {rest}"
        if rest:
            return f"Practice {rest[0].lower() + rest[1:]}"
        return "Practice appropriate responses to course-related situations."

    replacement: Optional[str] = None

    # 1) Known cross-domain confusion -> direct remap
    if first in _VERB_CROSS_MAP and domain in _VERB_CROSS_MAP[first]:
        replacement = _VERB_CROSS_MAP[first][domain]
    # 2) Banned/non-measurable verb -> map to a sensible domain-appropriate verb
    elif first in BANNED_VERBS:
        replacement = bank[0] if bank else first
    # 3) Domain-correct but wrong for this proficiency level -> remap narrowly
    elif isinstance(level, str) and first in bank:
        level_bank = LEVEL_VERB_BANKS.get(domain, {}).get(level, [])
        if level_bank and first not in level_bank:
            replacement = level_bank[0]
    # Unknown verbs are left for Pydantic and the model retry loop.

    if replacement is None:
        return description

    replacement = replacement.capitalize() if first_raw[0:1].isupper() else replacement
    return f"{replacement} {rest}".strip() if rest else replacement


def auto_repair_payload_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates a shallow copy of the parsed draft/audit JSON, repairing
    mechanical domain/level/verb mismatches before Pydantic sees it."""
    if not isinstance(parsed, dict):
        return parsed
    repaired = dict(parsed)
    outcomes = repaired.get("course_outcomes")
    if not isinstance(outcomes, list):
        return repaired

    fixed_outcomes = []
    for oc in outcomes:
        if not isinstance(oc, dict):
            fixed_outcomes.append(oc)
            continue
        oc = dict(oc)
        domain = oc.get("domain")
        if domain in DOMAIN_LEVELS:
            oc["bloom_level"] = _repair_bloom_level(domain, oc.get("bloom_level"))
            description = oc.get("co_description")
            if isinstance(description, str):
                leading_words = description.strip().lower().split()
                if domain == "Knowledge" and leading_words[:2] == ["break", "down"]:
                    oc["bloom_level"] = "Analyze"
                elif domain == "Attitude" and leading_words[:1] == ["assess"]:
                    oc["bloom_level"] = "Valuing"
                elif domain == "Attitude" and leading_words[:1] == ["use"]:
                    oc["bloom_level"] = "Responding"
            oc["co_description"] = _repair_description_verb(
                domain,
                oc["bloom_level"],
                description,
            )
        fixed_outcomes.append(oc)

    repaired["course_outcomes"] = fixed_outcomes
    return repaired


def _normalize_payload_for_course(
    parsed: Dict[str, Any],
    meta: CourseMetadata,
) -> Dict[str, Any]:
    normalized = dict(parsed)
    normalized["course_code"] = meta.course_code
    normalized["course_title"] = meta.course_title
    normalized["course_description"] = meta.course_description
    normalized["year_level"] = meta.year_level
    normalized["semester"] = meta.semester
    normalized["target_pos"] = meta.target_pos
    outcomes = normalized.get("course_outcomes")
    if isinstance(outcomes, list):
        normalized_outcomes = []
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                normalized_outcomes.append(outcome)
                continue
            outcome = dict(outcome)
            mapped = outcome.get("mapped_po")
            if isinstance(mapped, int) and not isinstance(mapped, bool):
                mapped = [mapped]
            if isinstance(mapped, list):
                valid = sorted(set(
                    po for po in mapped
                    if isinstance(po, int) and not isinstance(po, bool)
                    and po in meta.target_pos
                ))
                outcome["mapped_po"] = valid or [meta.target_pos[0]]
            normalized_outcomes.append(outcome)
        normalized["course_outcomes"] = normalized_outcomes
    return normalized


def _validate_requested_week_count(payload: CourseOutcomePayload, week_count: int) -> None:
    actual = len(payload.weekly_schedule)
    if actual != week_count:
        raise ValueError(
            f"Expected exactly {week_count} weekly_schedule entries, found {actual}."
        )


def _diff_repair(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """Human-readable list of exactly what auto_repair_payload_dict changed,
    field by field, for a given course's outcomes. Used only for --debug
    output so you can see the model's original words vs. the correction."""
    diffs: List[str] = []
    before_list = before.get("course_outcomes") if isinstance(before, dict) else None
    after_list = after.get("course_outcomes") if isinstance(after, dict) else None
    if not isinstance(before_list, list) or not isinstance(after_list, list):
        return diffs

    for b, a in zip(before_list, after_list):
        if not isinstance(b, dict) or not isinstance(a, dict):
            continue
        clo = a.get("clo_number", b.get("clo_number", "?"))
        if b.get("bloom_level") != a.get("bloom_level"):
            diffs.append(f"CLO {clo}: bloom_level '{b.get('bloom_level')}' -> '{a.get('bloom_level')}'")
        if b.get("co_description") != a.get("co_description"):
            diffs.append(f"CLO {clo}: co_description '{b.get('co_description')}' -> '{a.get('co_description')}'")
    return diffs

# ==============================================================================
# 6. TWO-PASS GENERATION + ERROR RETRY LOOP
# ==============================================================================

@dataclass
class GenerationOutcome:
    payload: Optional[CourseOutcomePayload]
    audit: GenerationAudit
    success: bool
    error: Optional[str] = None
    log: List[str] = field(default_factory=list)


def generate_course_outcomes(
    meta: CourseMetadata,
    provider: LLMProvider,
    cfg: EngineConfig,
    debug: bool = False,
    emit: "Callable[[str], None]" = print,
    is_cancelled: "Optional[Callable[[], bool]]" = None,
) -> GenerationOutcome:
    """Runs the full two-pass pipeline (draft -> self-audit) for ONE course,
    then validates the result against `CourseOutcomePayload`. On a Pydantic
    or JSON failure, re-prompts the model (pass 2 style) with the concrete
    error message and retries, up to `cfg.max_retries` times.

    debug=True emits the RAW, UNEDITED text returned by the model at every
    pass, plus a before/after diff of anything the auto-repair layer
    touched — so you can directly verify the output is coming from the live
    model and see exactly what (if anything) got mechanically corrected.

    emit: a callable that receives each progress/log line as it happens
    (defaults to `print` for CLI use). The GUI passes a thread-safe queue
    writer here instead, so it can stream this same output live."""

    log: List[str] = []
    started_at = time.monotonic()
    provider_calls = 0
    retries_used = 0
    pass1_completed = False
    pass2_completed = False
    repair_details: List[str] = []
    description_completed = bool(meta.course_description)

    if meta.course_description and meta.improve_description:
        if is_cancelled and is_cancelled():
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
                description_source="ai_improved",
            )
            return GenerationOutcome(
                payload=None,
                audit=audit,
                success=False,
                error="Cancelled by user.",
            )
        try:
            description = improve_course_description(
                meta,
                provider,
                temperature=cfg.pass1_temperature,
                debug=debug,
                emit=emit,
            )
            provider_calls += 1
            meta = meta.model_copy(update={
                "course_description": description,
                "description_source": "ai_improved",
            })
            description_completed = True
            log.append(
                f"[DESCRIPTION IMPROVED] {meta.course_code}: refined the user-written "
                f"description ({len(description)} chars)."
            )
        except (ValueError, ValidationError) as exc:
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
                description_source="manual",
            )
            return GenerationOutcome(
                payload=None,
                audit=audit,
                success=False,
                error=f"Course description improvement error: {exc}",
                log=log,
            )
        except Exception as exc:
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
                description_source="manual",
            )
            return GenerationOutcome(
                payload=None,
                audit=audit,
                success=False,
                error=f"Course description improvement provider error: {exc}",
                log=log,
            )
    elif not meta.course_description:
        if is_cancelled and is_cancelled():
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
            )
            return GenerationOutcome(payload=None, audit=audit, success=False, error="Cancelled by user.")
        try:
            description = generate_course_description(
                meta,
                provider,
                temperature=cfg.pass1_temperature,
                debug=debug,
                emit=emit,
            )
            provider_calls += 1
            meta = meta.model_copy(update={"course_description": description})
            description_completed = True
            log.append(
                f"[DESCRIPTION] {meta.course_code}: generated course description "
                f"({len(description)} chars)."
            )
        except (ValueError, ValidationError) as exc:
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
            )
            return GenerationOutcome(
                payload=None,
                audit=audit,
                success=False,
                error=f"Course description error: {exc}",
                log=log,
            )
        except Exception as exc:
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=False,
                pass2_completed=False,
                description_completed=False,
            )
            return GenerationOutcome(
                payload=None,
                audit=audit,
                success=False,
                error=f"Course description provider error: {exc}",
                log=log,
            )

    # ---- PASS 1: DRAFT -------------------------------------------------
    pass1_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PASS1_DRAFT},
        {"role": "user", "content": _build_pass1_user_prompt(meta, cfg.week_count)},
    ]
    try:
        if is_cancelled and is_cancelled():
            audit = GenerationAudit(model_used=cfg.model_name, pass1_completed=False, pass2_completed=False)
            return GenerationOutcome(payload=None, audit=audit, success=False, error="Cancelled by user.")
        draft_raw = provider.chat(pass1_messages, temperature=cfg.pass1_temperature)
        provider_calls += 1
        pass1_completed = True
        log.append(f"[PASS 1 / DRAFT] {meta.course_code}: received draft ({len(draft_raw)} chars).")
        if debug:
            emit(f"\n----- RAW PASS 1 OUTPUT ({meta.course_code}) — verbatim from the model -----")
            emit(draft_raw)
            emit("----- END RAW PASS 1 OUTPUT -----\n")
    except Exception as exc:  # provider-level failure (e.g. Ollama unreachable)
        audit = GenerationAudit(model_used=cfg.model_name, pass1_completed=False, pass2_completed=False)
        return GenerationOutcome(payload=None, audit=audit, success=False, error=f"Pass 1 provider error: {exc}", log=log)

    if cfg.generation_preset == "Fast":
        try:
            parsed = _normalize_payload_for_course(
                _parse_json_or_raise(draft_raw),
                meta,
            )
            repaired = auto_repair_payload_dict(parsed)
            diffs = _diff_repair(parsed, repaired)
            repair_details.extend(diff for diff in diffs if diff not in repair_details)
            payload = CourseOutcomePayload.model_validate(repaired)
            _validate_requested_week_count(payload, cfg.week_count)
            elapsed = round(time.monotonic() - started_at, 3)
            log.append(
                f"[FAST VALIDATED] {meta.course_code}: schema OK, K/S/A = "
                f"{payload.ksa_summary()}, elapsed={elapsed}s."
            )
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=True,
                pass2_completed=False,
                retries_used=0,
                repair_count=len(repair_details),
                repair_details=repair_details,
                description_completed=description_completed,
                description_source=meta.description_source,
                elapsed_seconds=elapsed,
                provider_calls=provider_calls,
            )
            return GenerationOutcome(payload=payload, audit=audit, success=True, log=log)
        except (ValueError, ValidationError) as err:
            last_error = str(err)
            log.append(f"[FAST VALIDATION FAILURE] {meta.course_code}: {last_error}")

    # ---- PASS 2: SELF-AUDIT & REFINE -----------------------------------
    audit_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PASS2_AUDIT},
        {"role": "user", "content": _build_pass2_user_prompt(draft_raw, cfg.week_count)},
    ]

    attempt = 0
    last_error = ""
    final_raw = ""

    if cfg.generation_preset == "Fast":
        audit_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PASS1_DRAFT},
            {"role": "user", "content": _build_pass1_user_prompt(meta, cfg.week_count)},
        ]

    while attempt <= cfg.max_retries:
        try:
            if is_cancelled and is_cancelled():
                audit = GenerationAudit(
                    model_used=cfg.model_name,
                    pass1_completed=pass1_completed,
                    pass2_completed=pass2_completed,
                    retries_used=retries_used,
                    repair_count=len(repair_details),
                    repair_details=repair_details,
                    description_completed=description_completed,
                    elapsed_seconds=round(time.monotonic() - started_at, 3),
                    provider_calls=provider_calls,
                )
                return GenerationOutcome(payload=None, audit=audit, success=False, error="Cancelled by user.", log=log)
            try:
                final_raw = provider.chat(audit_messages, temperature=cfg.pass2_temperature)
                provider_calls += 1
            except Exception as exc:
                pass2_completed = False
                last_error = f"Pass 2 provider error: {exc}"
                log.append(f"[PROVIDER FAILURE] {meta.course_code}: {last_error}")
                break
            pass2_completed = True
            log.append(f"[PASS 2 / AUDIT attempt {attempt + 1}] {meta.course_code}: received audited JSON.")
            if debug:
                emit(f"----- RAW PASS 2 OUTPUT ({meta.course_code}, attempt {attempt + 1}) — verbatim from the model -----")
                emit(final_raw)
                emit("----- END RAW PASS 2 OUTPUT -----\n")

            parsed = _normalize_payload_for_course(
                _parse_json_or_raise(final_raw),
                meta,
            )
            repaired = auto_repair_payload_dict(parsed)
            diffs = _diff_repair(parsed, repaired)
            for diff in diffs:
                if diff not in repair_details:
                    repair_details.append(diff)
            if debug:
                if diffs:
                    emit(f"----- AUTO-REPAIR CHANGES ({meta.course_code}) -----")
                    for d in diffs:
                        emit(f"  {d}")
                    emit("----- END AUTO-REPAIR CHANGES -----\n")
                else:
                    emit(f"[AUTO-REPAIR] {meta.course_code}: nothing to fix, model output already domain-compliant.\n")

            payload = CourseOutcomePayload.model_validate(repaired)
            _validate_requested_week_count(payload, cfg.week_count)

            log.append(
                f"[VALIDATED] {meta.course_code}: schema OK, K/S/A = {payload.ksa_summary()}, "
                f"retries_used={retries_used}."
            )
            audit = GenerationAudit(
                model_used=cfg.model_name,
                pass1_completed=pass1_completed,
                pass2_completed=pass2_completed,
                retries_used=retries_used,
                repair_count=len(repair_details),
                repair_details=repair_details,
                description_completed=description_completed,
            )
            return GenerationOutcome(payload=payload, audit=audit, success=True, log=log)

        except (ValueError, ValidationError) as err:
            attempt += 1
            retries_used = attempt
            last_error = str(err)
            log.append(f"[VALIDATION FAILURE attempt {attempt}] {meta.course_code}: {last_error}")

            if attempt > cfg.max_retries:
                log.append(f"[ABORT] {meta.course_code}: exceeded max_retries={cfg.max_retries}.")
                break

            # Automated error-recovery: inject the concrete Pydantic/JSON
            # error back into the conversation and ask the model to fix it.
            audit_messages.append({"role": "assistant", "content": final_raw})
            audit_messages.append({
                "role": "user",
                "content": (
                    f"Your JSON failed schema validation with this error:\n{last_error}\n\n"
                    f"Correct ONLY what is necessary to satisfy the checklist and the error "
                    f"above, keep everything else intact, and return the corrected pure JSON "
                    f"object (no markdown fences, no commentary)."
                ),
            })

    audit = GenerationAudit(
        model_used=cfg.model_name,
        pass1_completed=pass1_completed,
        pass2_completed=pass2_completed,
        retries_used=retries_used,
        repair_count=len(repair_details),
        repair_details=repair_details,
    )
    return GenerationOutcome(payload=None, audit=audit, success=False, error=last_error, log=log)


# ==============================================================================
# 7. BATCH PROCESSING — all 4 CS/IT subjects
# ==============================================================================

def run_batch(
    provider: LLMProvider,
    cfg: EngineConfig,
    catalog: Optional[List[CourseMetadata]] = None,
    verbose: bool = True,
    debug: bool = False,
    emit: "Callable[[str], None]" = print,
    on_course_done: "Optional[Callable[[CourseMetadata, GenerationOutcome], None]]" = None,
    is_cancelled: "Optional[Callable[[], bool]]" = None,
) -> BatchGenerationResult:
    """emit: receives every progress/log line as it happens (defaults to
    `print`). on_course_done: optional callback fired after each course
    finishes (success or failure) with (meta, outcome) — the GUI uses this
    to update its results table incrementally, course by course, rather
    than waiting for the whole batch to finish."""
    catalog = catalog if catalog is not None else COURSE_CATALOG
    records: List[ValidatedCourseRecord] = []
    failures: List[Dict[str, Any]] = []
    description_cache: Dict[str, str] = {}
    attempted = 0

    for meta in catalog:
        if is_cancelled and is_cancelled():
            break
        attempted += 1
        if not meta.course_description:
            cache_key = meta.course_title.strip().lower()
            cached_description = description_cache.get(cache_key)
            if cached_description:
                meta = meta.model_copy(update={"course_description": cached_description})
                emit(f"[CACHE] {meta.course_code}: reused description for '{meta.course_title}'.")
        if verbose:
            emit(f"\n{'=' * 70}\nGenerating: [{meta.course_code}] {meta.course_title}\n{'=' * 70}")

        outcome = generate_course_outcomes(
            meta,
            provider,
            cfg,
            debug=debug,
            emit=emit,
            is_cancelled=is_cancelled,
        )
        for line in outcome.log:
            if verbose:
                emit(f"  {line}")

        if outcome.success and outcome.payload is not None:
            records.append(ValidatedCourseRecord(payload=outcome.payload, audit=outcome.audit))
            if not meta.course_description and outcome.payload.course_description:
                description_cache[meta.course_title.strip().lower()] = outcome.payload.course_description
        else:
            if outcome.error != "Cancelled by user.":
                failures.append({
                    "course_code": meta.course_code,
                    "course_title": meta.course_title,
                    "error": outcome.error,
                    "retries_used": outcome.audit.retries_used,
                })
                if verbose:
                    emit(f"  [FAILED] {meta.course_code}: {outcome.error}")
            elif verbose:
                emit(f"  [CANCELLED] {meta.course_code}: generation cancelled by user.")

        if on_course_done is not None:
            on_course_done(meta, outcome)

    return BatchGenerationResult(
        total_courses_attempted=attempted,
        total_courses_validated=len(records),
        records=records,
        failures=failures,
    )


# ==============================================================================
# 8. OFFLINE MOCK SCRIPTING — proves the retry loop actually recovers
# ==============================================================================

def _build_mock_provider() -> MockProvider:
    """Builds a MockProvider whose scripted responses exercise the REAL
    validation + retry code path end-to-end without needing Ollama:
      - Course 1 (CGP): Pass-2 output first violates K/S/A coverage
        (no Attitude outcome) -> triggers a retry -> corrected version passes.
      - Courses 2-4: clean two-pass runs (Pass 1 draft, Pass 2 audit passes
        validation immediately).
    """

    def co(clo, domain, level, desc, po):
        return {"clo_number": clo, "domain": domain, "bloom_level": level, "co_description": desc, "mapped_po": po}

    # ---- Course 1: Computer Graphics Programming (demonstrates the retry) ---
    cgp_draft = {
        "course_code": "CS321", "course_title": "Computer Graphics Programming",
        "course_description": COURSE_CATALOG[0].course_description,
        "target_pos": [1, 2, 3, 5],
        "course_outcomes": [
            co(1, "Knowledge", "Analyze", "Analyze the mathematical foundations of 2D and 3D transformation matrices.", [1, 2]),
            co(2, "Skills", "Manipulation", "Implement shader programs using OpenGL to render textured 3D models.", [2, 3]),
            co(3, "Skills", "Precision", "Optimize rendering pipelines for real-time frame rate performance.", [3]),
            co(4, "Knowledge", "Evaluate", "Evaluate trade-offs between rasterization and ray-tracing rendering techniques.", [1, 5]),
        ],
    }
    # First audit pass INTENTIONALLY still has no "Attitude" outcome -> fails K/S/A coverage.
    cgp_audit_bad = dict(cgp_draft)
    # Second audit pass (after retry feedback) adds the missing Attitude outcome -> passes.
    cgp_audit_fixed = {
        **cgp_draft,
        "course_outcomes": cgp_draft["course_outcomes"] + [
            co(5, "Attitude", "Valuing", "Advocate for accessible and inclusive visual design practices in graphics applications.", [5]),
        ],
    }

    # ---- Course 2: Database Systems 1 (clean run) ---------------------------
    dbs_final = {
        "course_code": "CS322", "course_title": "Database Systems 1",
        "course_description": COURSE_CATALOG[1].course_description,
        "target_pos": [1, 2, 3, 4],
        "course_outcomes": [
            co(1, "Knowledge", "Analyze", "Analyze normalization requirements up to Third Normal Form for a relational schema.", [1, 2]),
            co(2, "Skills", "Manipulation", "Construct SQL queries that join, aggregate, and filter multi-table relational data.", [2, 3]),
            co(3, "Skills", "Precision", "Implement transaction control and indexing strategies for query performance.", [3, 4]),
            co(4, "Attitude", "Responding", "Comply with data integrity and referential-integrity best practices during schema design.", [4]),
        ],
    }

    # ---- Course 3: Data Mining and Analytics (clean run) ---------------------
    dm_final = {
        "course_code": "CS323", "course_title": "Data Mining and Analytics",
        "course_description": COURSE_CATALOG[2].course_description,
        "target_pos": [1, 2, 3, 6],
        "course_outcomes": [
            co(1, "Knowledge", "Understand-Cognitively", "Describe the data preprocessing pipeline of cleaning, transformation, and feature selection.", [1, 2]),
            co(2, "Skills", "Manipulation", "Implement classification and clustering algorithms on real-world datasets using Python.", [2, 3]),
            co(3, "Knowledge", "Evaluate", "Evaluate model performance using precision, recall, and cross-validation metrics.", [3, 6]),
            co(4, "Attitude", "Valuing", "Advocate for ethical and privacy-conscious handling of sensitive datasets.", [6]),
        ],
    }

    # ---- Course 4: Software Engineering (clean run) --------------------------
    se_final = {
        "course_code": "CS324", "course_title": "Software Engineering",
        "course_description": COURSE_CATALOG[3].course_description,
        "target_pos": [1, 2, 3, 4, 7],
        "course_outcomes": [
            co(1, "Knowledge", "Apply", "Apply requirements-engineering techniques to elicit and document functional specifications.", [1, 2]),
            co(2, "Skills", "Articulation", "Model a modular software architecture using established design patterns.", [2, 3]),
            co(3, "Skills", "Manipulation", "Construct a working software increment following an agile Scrum workflow.", [3, 4]),
            co(4, "Attitude", "Organizing", "Prioritize code quality and peer code-review practices within a development team.", [4, 7]),
        ],
    }

    responses = [
        json.dumps(cgp_draft),        # Pass 1 draft (Course 1)
        json.dumps(cgp_audit_bad),    # Pass 2 audit — still missing Attitude -> validation fails
        json.dumps(cgp_audit_fixed),  # Pass 2 retry — corrected -> validation passes

        json.dumps(dbs_final),        # Pass 1 draft (Course 2)
        json.dumps(dbs_final),        # Pass 2 audit — already compliant -> passes

        json.dumps(dm_final),         # Pass 1 draft (Course 3)
        json.dumps(dm_final),         # Pass 2 audit — already compliant -> passes

        json.dumps(se_final),         # Pass 1 draft (Course 4)
        json.dumps(se_final),         # Pass 2 audit — already compliant -> passes
    ]
    description_responses = [
        json.dumps({
            "course_description": (
                "Introduces the principles and practices of computer graphics and "
                "real-time visual applications. Students explore transformation "
                "pipelines, rasterization, rendering, and shader programming."
            )
        }),
    ]
    custom_payload = {
        "course_code": "CS399",
        "course_title": "Cybersecurity Fundamentals",
        "course_description": (
            "Introduces cybersecurity principles, threats, controls, and defensive "
            "practices across modern information systems."
        ),
        "target_pos": [1, 2, 4, 7],
        "course_outcomes": [
            co(1, "Knowledge", "Analyze", "Analyze common cybersecurity threats and attack vectors.", [1, 2]),
            co(2, "Skills", "Manipulation", "Implement defensive controls for information systems.", [2, 4]),
            co(3, "Skills", "Precision", "Configure network security mechanisms to mitigate threats.", [4, 7]),
            co(4, "Attitude", "Valuing", "Value secure practices in organizational technology use.", [7]),
        ],
    }
    custom_responses = [json.dumps(custom_payload), json.dumps(custom_payload)]
    return MockProvider(
        responses,
        description_responses=description_responses,
        custom_responses=custom_responses,
    )


# ==============================================================================
# 9. CLI ENTRYPOINT
# ==============================================================================

def write_json_atomic(path: str, content: str) -> None:
    """Write JSON through a same-directory temporary file, replacing atomically."""
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".obe-", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, absolute)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 1 — Two-Pass Ollama/Qwen2.5 OBE Course Outcomes Engine (batch)."
    )
    parser.add_argument(
        "--model", default=None,
        help="Ollama model name (e.g. qwen2.5:7b). If omitted, the engine "
             "auto-detects whichever model is already installed locally.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Max validation-error retries per course")
    parser.add_argument("--output", default="sample_validated_output.json", help="Batch output JSON path")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use the built-in offline MockProvider instead of a real Ollama daemon "
             "(exercises the identical validation + retry pipeline for demo/testing).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print the RAW, verbatim text returned by the model at every pass, plus a "
             "before/after diff of anything the auto-repair layer touched. Use this to "
             "verify the output is genuinely coming from your live model.",
    )
    parser.add_argument(
        "--subject",
        "--course",
        dest="subject",
        action="append",
        default=None,
        help="Repeat to load courses: --course CS321 --course CS323. Use 'all' for every built-in course.",
    )
    parser.add_argument(
        "--subject-title",
        "--course-title",
        dest="subject_title",
        action="append",
        default=None,
        help="Repeat with --course custom for multiple custom courses.",
    )
    parser.add_argument(
        "--course-code",
        action="append",
        default=None,
        help="Optional code for a custom course. Repeat to specify codes in custom-course order.",
    )
    parser.add_argument("--course-description", default=None)
    parser.add_argument(
        "--improve-course-description",
        action="store_true",
        help="Improve a supplied custom course description with the selected LLM.",
    )
    parser.add_argument("--year-level", default="3rd Year")
    parser.add_argument("--semester", default="1st Semester")
    parser.add_argument("--units", type=float, default=3.0)
    parser.add_argument("--target-pos", default="1,2,3,4,5")
    parser.add_argument("--weeks", type=int, choices=range(1, 13), default=8,
                        help="Weekly learning plan length per course (1-12; default: 8).")
    parser.add_argument(
        "--preset",
        choices=["Fast", "Balanced", "Quality"],
        default="Balanced",
    )
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.improve_course_description and not args.course_description:
        parser.error(
            "--improve-course-description requires --course-description"
        )

    if args.mock:
        cfg = apply_generation_preset(
            args.preset,
            "mock-scripted",
            retries=args.retries,
            week_count=args.weeks,
        )
        print("[*] Running in OFFLINE MOCK mode (no Ollama daemon required).")
        provider: LLMProvider = _build_mock_provider()
    else:
        print("[*] Connecting to local Ollama daemon...")
        cfg = apply_generation_preset(
            args.preset,
            args.model or "auto",
            retries=args.retries,
            week_count=args.weeks,
        )
        if args.timeout is not None:
            cfg.request_timeout_seconds = args.timeout
        provider = OllamaProvider(
            args.model,
            timeout_seconds=cfg.request_timeout_seconds,
            keep_alive=cfg.keep_alive,
            num_predict=cfg.num_predict,
        )
        print(f"[+] Using locally installed model: '{provider.model_name}'")
        cfg.model_name = provider.model_name

    try:
        target_pos = [
            int(part.strip())
            for part in args.target_pos.split(",")
            if part.strip()
        ]
    except ValueError:
        parser.error("--target-pos must be comma-separated integers (for example: 1,2,3)")

    subject_args = args.subject or ["all"]
    catalog = []
    custom_titles = args.subject_title or []
    custom_title_index = 0
    custom_codes = args.course_code or []
    custom_code_index = 0
    for subject_arg in subject_args:
        normalized_subject = subject_arg.strip().lower()
        if normalized_subject == "all":
            for meta in COURSE_CATALOG:
                if meta not in catalog:
                    catalog.append(meta)
            continue
        if normalized_subject == "custom":
            if custom_title_index >= len(custom_titles):
                parser.error("--course custom requires a matching --course-title")
            custom_title = custom_titles[custom_title_index]
            custom_title_index += 1
            if custom_code_index < len(custom_codes):
                custom_code = custom_codes[custom_code_index]
            else:
                custom_code = next_custom_course_code(meta.course_code for meta in catalog)
            custom_code_index += 1
            custom_meta = build_custom_subject(
                title=custom_title,
                course_code=custom_code,
                course_description=args.course_description,
                year_level=args.year_level,
                semester=args.semester,
                units=args.units,
                target_pos=target_pos,
            )
            if args.course_description:
                custom_meta = custom_meta.model_copy(update={
                    "description_source": "ai_improved" if args.improve_course_description else "manual",
                    "improve_description": bool(args.improve_course_description),
                })
            catalog.append(custom_meta)
            continue
        selected = find_builtin_subject(subject_arg)
        if selected is None:
            parser.error(
                "--course must be 'all', 'custom', or a built-in code/title "
                "(for example: CS321 or Computer Graphics Programming)"
            )
        if selected not in catalog:
            catalog.append(selected)

    if not catalog:
        parser.error("At least one course must be loaded.")
    if custom_code_index < len(custom_codes):
        parser.error("--course-code was provided without a matching --course custom.")

    print("=" * 70)
    print("CCS OBE SYLLABUS GENERATOR — MILESTONE 1: COURSE OUTCOMES ENGINE")
    print(f"Courses: {[meta.course_code for meta in catalog]}")
    print("=" * 70)

    result = run_batch(provider, cfg, catalog=catalog, debug=args.debug)

    print("\n" + "=" * 70)
    print(f"BATCH SUMMARY: {result.total_courses_validated}/{result.total_courses_attempted} courses validated.")
    if result.failures:
        print(f"Failures: {[f['course_code'] for f in result.failures]}")
    print("=" * 70)

    write_json_atomic(args.output, result.model_dump_json(indent=2))
    print(f"\n[+] Batch output saved to: {args.output}")

    if result.total_courses_validated == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
