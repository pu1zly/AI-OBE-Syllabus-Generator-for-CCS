"""
obe_schemas.py
================================================================================
AI-Powered OBE Syllabus Generator Microservice
Milestone 1 — Deliverable 1: Structured LLM Engine & Schema Validation

Pydantic v2 data contracts that enforce Outcome-Based Education (OBE)
compliance for Course Outcomes (COs) across all three Bloom's Taxonomy
DOMAINS:

    K  ->  Knowledge   (Cognitive domain   — Bloom/Anderson)
    S  ->  Skills      (Psychomotor domain — Simpson)
    A  ->  Attitude    (Affective domain   — Krathwohl)

Every CourseOutcome must declare which domain it belongs to, and its verb
must come from that domain's approved action-verb bank at the declared
proficiency level. A batch of Course Outcomes for a course is only accepted
if it contains at least one outcome from EACH of the three domains (K/S/A
coverage), in addition to the usual Bloom-verb / PO-mapping business rules.

College of Computer Studies (CCS) — 3rd Year, 1st Semester CS/IT batch.
================================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# ==============================================================================
# 1. NEGATIVE CONSTRAINTS — non-measurable / banned verbs (all domains)
# ==============================================================================

BANNED_VERBS = {
    "understand", "understands", "understanding",
    "learn", "learns", "learning",
    "know", "knows", "knowing",
    "be exposed to", "expose", "exposed",
    "study", "studies", "studying",
    "appreciate", "appreciates", "appreciating",
    "become familiar with", "familiarize", "familiarized",
    "cover", "covers", "covering",
}

# ==============================================================================
# 2. DOMAIN-SPECIFIC ACTION VERB BANKS
# ------------------------------------------------------------------------------
# Knowledge  -> Anderson & Krathwohl's revised cognitive taxonomy (6 levels)
# Skills     -> Simpson's psychomotor taxonomy (levels grouped low/mid/high)
# Attitude   -> Krathwohl's affective taxonomy (levels grouped low/mid/high)
# ==============================================================================

DOMAIN_LEVELS: Dict[str, List[str]] = {
    "Knowledge": [
        "Remember", "Understand-Cognitively", "Apply", "Analyze", "Evaluate", "Create",
    ],
    "Skills": [
        "Imitation", "Manipulation", "Precision", "Articulation", "Naturalization",
    ],
    "Attitude": [
        "Receiving", "Responding", "Valuing", "Organizing", "Characterizing",
    ],
}

DOMAIN_VERB_BANKS: Dict[str, List[str]] = {
    "Knowledge": [
        "identify", "define", "recall", "describe", "state", "list", "name",
        "label", "match", "recognize", "outline", "explain", "classify",
        "summarize", "apply", "implement", "calculate", "solve", "analyze",
        "compare", "contrast", "differentiate", "examine", "evaluate",
        "assess", "critique", "justify", "design", "develop", "formulate",
        "synthesize", "construct", "derive",
    ],
    "Skills": [
        "operate", "execute", "implement", "construct", "assemble",
        "calibrate", "configure", "install", "demonstrate", "manipulate",
        "code", "build", "compile", "debug", "test", "optimize", "render",
        "model", "simulate", "integrate", "deploy", "troubleshoot",
        "fabricate", "reproduce", "perform",
    ],
    "Attitude": [
        "value", "advocate", "adhere", "comply", "internalize", "exhibit",
        "demonstrate", "practice", "commit", "uphold", "observe",
        "prioritize", "embrace", "cultivate", "champion", "model",
        "exercise",
    ],
}

DOMAIN_LABELS = {"Knowledge": "K", "Skills": "S", "Attitude": "A"}
ALLOWED_PO_RANGE = (1, 9)  # CCS Program Outcomes PO1..PO9

# Safe fallback level per domain, used by the engine's auto-repair layer
# when a weaker local model reports a level that doesn't belong to the
# domain it declared (e.g. "Apply" tagged under "Skills").
DOMAIN_LEVEL_DEFAULT: Dict[str, str] = {
    "Knowledge": "Apply",
    "Skills": "Manipulation",
    "Attitude": "Valuing",
}

# Tight level-specific verb groups. These supplement the domain banks and keep
# outcomes from pairing a low-level verb with an advanced Bloom level.
LEVEL_VERB_BANKS: Dict[str, Dict[str, List[str]]] = {
    "Knowledge": {
        "Remember": ["identify", "define", "recall", "describe", "state", "list", "name", "label", "match", "recognize", "outline"],
        "Understand-Cognitively": ["describe", "explain", "classify", "summarize", "compare"],
        "Apply": ["apply", "implement", "calculate", "solve", "construct", "develop"],
        "Analyze": ["analyze", "compare", "contrast", "differentiate", "examine", "assess"],
        "Evaluate": ["evaluate", "assess", "critique", "justify", "compare"],
        "Create": ["design", "develop", "formulate", "synthesize", "construct", "derive"],
    },
    "Skills": {
        "Imitation": ["operate", "execute", "demonstrate", "manipulate", "reproduce", "perform"],
        "Manipulation": ["implement", "assemble", "configure", "install", "code", "build", "compile", "debug", "test", "model", "simulate"],
        "Precision": ["calibrate", "configure", "test", "optimize", "troubleshoot", "integrate"],
        "Articulation": ["demonstrate", "optimize", "render", "model", "simulate", "integrate"],
        "Naturalization": ["deploy", "fabricate", "reproduce", "perform", "troubleshoot", "integrate"],
    },
    "Attitude": {
        "Receiving": ["observe", "practice", "exercise", "comply"],
        "Responding": ["comply", "practice", "adhere", "observe", "exercise"],
        "Valuing": ["value", "advocate", "internalize", "embrace", "prioritize"],
        "Organizing": ["prioritize", "adhere", "uphold", "commit", "cultivate", "integrate"],
        "Characterizing": ["champion", "model", "exhibit", "advocate", "internalize"],
    },
}


def _first_verb(text: str) -> str:
    words = text.strip().split()
    if not words:
        return ""
    return words[0].lower().rstrip(",.:;")


def _check_not_banned(text: str) -> None:
    lowered = text.strip().lower()
    first = _first_verb(text)
    for banned in BANNED_VERBS:
        if first == banned or lowered.startswith(banned + " "):
            raise ValueError(
                f"Non-measurable verb '{banned}' violates OBE compliance rules. "
                f"Use an active, domain-appropriate Bloom's verb instead."
            )


# ==============================================================================
# 3. COURSE OUTCOME (single CLO, tagged to one K/S/A domain)
# ==============================================================================

class CourseOutcome(BaseModel):
    """A single Course Learning Outcome (CLO), classified into exactly one
    Bloom's Taxonomy domain (Knowledge / Skills / Attitude) and validated
    against that domain's approved verb bank and proficiency level."""

    clo_number: int = Field(..., ge=1, description="Sequential CLO number (1, 2, 3...)")
    domain: Literal["Knowledge", "Skills", "Attitude"] = Field(
        ..., description="Bloom's Taxonomy domain this outcome belongs to (K/S/A)"
    )
    bloom_level: str = Field(
        ..., description="Proficiency level within the declared domain"
    )
    co_description: str = Field(
        ..., min_length=10,
        description="Action outcome statement starting with a domain-appropriate active verb",
    )
    mapped_po: List[int] = Field(
        ..., description=f"Mapped Program Outcome numbers, range {ALLOWED_PO_RANGE}"
    )

    @field_validator("bloom_level")
    @classmethod
    def validate_level_belongs_to_domain(cls, v: str, info) -> str:
        domain = info.data.get("domain")
        if domain and v not in DOMAIN_LEVELS[domain]:
            raise ValueError(
                f"bloom_level '{v}' is not valid for domain '{domain}'. "
                f"Allowed levels: {DOMAIN_LEVELS.get(domain, [])}"
            )
        return v

    @field_validator("co_description")
    @classmethod
    def validate_domain_verb_compliance(cls, v: str, info) -> str:
        text = v.strip()
        if not text:
            raise ValueError("Course outcome description cannot be empty.")

        _check_not_banned(text)

        domain = info.data.get("domain")
        if domain:
            first = _first_verb(text)
            bank = DOMAIN_VERB_BANKS[domain]
            if first not in bank:
                raise ValueError(
                    f"Verb '{first}' does not belong to the '{domain}' "
                    f"({DOMAIN_LABELS[domain]}) verb bank. Expected one of: "
                    f"{bank[:8]}..."
                )
            level = info.data.get("bloom_level")
            level_bank = LEVEL_VERB_BANKS.get(domain, {}).get(level, [])
            if level_bank and first not in level_bank:
                raise ValueError(
                    f"Verb '{first}' is not appropriate for {domain} level "
                    f"'{level}'. Allowed verbs: {level_bank}"
                )
        return text

    @field_validator("mapped_po")
    @classmethod
    def validate_po_range(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("Each Course Outcome must map to at least one Program Outcome (PO).")
        lo, hi = ALLOWED_PO_RANGE
        for po in v:
            if not (lo <= po <= hi):
                raise ValueError(f"Program Outcome index {po} is out of standard range [{lo}-{hi}].")
        return sorted(set(v))


# ==============================================================================
# 4. COURSE METADATA (batch-processing input contract)
# ==============================================================================

class CourseMetadata(BaseModel):
    """Input contract describing a single CCS course to be processed in the batch."""

    course_code: str = Field(..., min_length=2)
    course_title: str = Field(..., min_length=3)
    course_description: Optional[str] = Field(default=None, min_length=10)
    year_level: str = Field(default="3rd Year")
    semester: str = Field(default="1st Semester")
    units: float = Field(default=3.0, gt=0)
    target_pos: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    description_source: Literal[
        "generated",
        "manual",
        "ai_improved",
    ] = "generated"
    improve_description: bool = False

    @field_validator("target_pos")
    @classmethod
    def validate_target_pos(cls, v: List[int]) -> List[int]:
        lo, hi = ALLOWED_PO_RANGE
        if not v:
            raise ValueError("A course must declare at least one target Program Outcome.")
        for po in v:
            if not (lo <= po <= hi):
                raise ValueError(f"Program Outcome index {po} is out of standard range [{lo}-{hi}].")
        return sorted(set(v))


class WeeklySchedule(BaseModel):
    """One scheduled instructional week with a measurable learning outcome."""

    week_number: int = Field(..., ge=1)
    topic: str = Field(..., min_length=3)
    learning_outcome: str = Field(..., min_length=10)
    activities: List[str] = Field(..., min_length=1)
    mapped_clos: List[int] = Field(default_factory=list)
    contact_hours: float = Field(default=3.0, ge=0)

    @field_validator("mapped_clos")
    @classmethod
    def validate_mapped_clos(cls, value: List[int]) -> List[int]:
        if any(clo < 1 for clo in value):
            raise ValueError("CLO mappings must use positive CLO numbers.")
        return sorted(set(value))


# ==============================================================================
# 5. COURSE OUTCOME PAYLOAD — the full validated K/S/A batch for one course
# ==============================================================================

class CourseOutcomePayload(BaseModel):
    """Complete, Pydantic-validated Course Outcomes payload for a single
    course, produced by the two-pass LLM engine. Enforces:
      - 4 to 6 total Course Outcomes
      - unique, sequential clo_number values
      - at least ONE outcome from each of the K / S / A domains
    """

    course_code: str
    course_title: str
    course_description: str
    year_level: str = "3rd Year"
    semester: str = "1st Semester"
    target_pos: List[int]
    course_outcomes: List[CourseOutcome] = Field(
        ..., description="4 to 6 Course Outcomes covering Knowledge, Skills, and Attitude domains"
    )
    weekly_schedule: List[WeeklySchedule] = Field(
        default_factory=list,
        description="Sequential weekly topics and measurable outcomes mapped to course outcomes",
    )

    @field_validator("course_outcomes")
    @classmethod
    def validate_outcome_count(cls, outcomes: List[CourseOutcome]) -> List[CourseOutcome]:
        if len(outcomes) < 4 or len(outcomes) > 6:
            raise ValueError(f"Course outcomes must contain between 4 and 6 outcomes. Found {len(outcomes)}.")
        return outcomes

    @model_validator(mode="after")
    def validate_unique_sequential_clo(self) -> "CourseOutcomePayload":
        numbers = [co.clo_number for co in self.course_outcomes]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Duplicate clo_number values detected: {numbers}")
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"clo_number values must be sequential starting at 1 (e.g. 1..{len(numbers)}). Got: {sorted(numbers)}"
            )
        return self

    @model_validator(mode="after")
    def validate_ksa_domain_coverage(self) -> "CourseOutcomePayload":
        """CCS Business Rule: the batch of Course Outcomes must cover ALL
        THREE Bloom's Taxonomy domains — Knowledge, Skills, AND Attitude —
        at least once each. This is the core 'K/S/A validation' rule."""
        present_domains = {co.domain for co in self.course_outcomes}
        required = {"Knowledge", "Skills", "Attitude"}
        missing = required - present_domains
        if missing:
            missing_labels = sorted(DOMAIN_LABELS[d] for d in missing)
            raise ValueError(
                f"K/S/A coverage error: missing domain(s) {sorted(missing)} "
                f"(labels: {missing_labels}). Every course must have at least one "
                f"Knowledge (K), one Skills (S), and one Attitude (A) outcome."
            )
        return self

    @model_validator(mode="after")
    def validate_mappings_against_course_targets(self) -> "CourseOutcomePayload":
        allowed = set(self.target_pos)
        for co in self.course_outcomes:
            outside = sorted(set(co.mapped_po) - allowed)
            if outside:
                raise ValueError(
                    f"CLO {co.clo_number} maps to PO(s) {outside}, outside this "
                    f"course's target_pos {sorted(allowed)}."
                )
        return self

    @model_validator(mode="after")
    def validate_weekly_schedule(self) -> "CourseOutcomePayload":
        if not self.weekly_schedule:
            return self
        week_numbers = [week.week_number for week in self.weekly_schedule]
        expected = list(range(1, len(week_numbers) + 1))
        if week_numbers != expected:
            raise ValueError(
                f"Weekly schedule numbers must be sequential starting at 1. Got: {week_numbers}"
            )
        valid_clos = {outcome.clo_number for outcome in self.course_outcomes}
        for week in self.weekly_schedule:
            if not week.mapped_clos:
                raise ValueError(f"Week {week.week_number} must map to at least one CLO.")
            unknown = sorted(set(week.mapped_clos) - valid_clos)
            if unknown:
                raise ValueError(
                    f"Week {week.week_number} references unknown CLO(s): {unknown}"
                )
        return self

    def ksa_summary(self) -> Dict[str, int]:
        """Convenience helper: count of outcomes per K/S/A domain."""
        counts = {"K": 0, "S": 0, "A": 0}
        for co in self.course_outcomes:
            counts[DOMAIN_LABELS[co.domain]] += 1
        return counts


# Backwards/forward-compatible alias matching common lab naming conventions
CourseOutcomeSchema = CourseOutcomePayload


# ==============================================================================
# 6. BATCH GENERATION AUDIT RECORD (microservice-level metadata)
# ==============================================================================

class GenerationAudit(BaseModel):
    """Audit trail attached to every generated course payload — records how
    many LLM passes and retries were needed, for institutional traceability."""

    model_used: str
    pass1_completed: bool = True
    pass2_completed: bool = True
    retries_used: int = Field(default=0, ge=0)
    description_completed: bool = True
    description_source: Literal[
        "generated",
        "manual",
        "ai_improved",
    ] = "generated"
    repair_count: int = Field(default=0, ge=0)
    repair_details: List[str] = Field(default_factory=list)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    provider_calls: int = Field(default=0, ge=0)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ValidatedCourseRecord(BaseModel):
    """One fully validated course record, ready for SQLite persistence /
    Jinja2 rendering in later milestones."""

    payload: CourseOutcomePayload
    audit: GenerationAudit


class BatchGenerationResult(BaseModel):
    """Top-level container for a full batch run across the 3rd Year,
    1st Semester CS/IT subject set."""

    batch_label: str = "3rd Year - 1st Semester CS/IT"
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    total_courses_attempted: int
    total_courses_validated: int
    records: List[ValidatedCourseRecord]
    failures: List[Dict[str, Any]] = Field(default_factory=list)


# ==============================================================================
# 7. MILESTONE 2 — COMPLETE SYLLABUS STRUCTURES
# ==============================================================================

class Assessment(BaseModel):
    """A measurable syllabus assessment linked to one or more CLOs."""

    assessment_id: str = Field(..., min_length=2)
    name: str = Field(..., min_length=3)
    assessment_type: Literal[
        "Quiz", "Exam", "Laboratory", "Project", "Presentation",
        "Case Study", "Portfolio", "Class Activity",
    ]
    weight_percent: float = Field(..., gt=0, le=100)
    mapped_clos: List[int] = Field(..., min_length=1)

    @field_validator("mapped_clos")
    @classmethod
    def validate_mapped_clos(cls, value: List[int]) -> List[int]:
        for clo in value:
            if clo < 1:
                raise ValueError("CLO mappings must use positive CLO numbers.")
        return sorted(set(value))


class LearningResource(BaseModel):
    """A textbook, article, software tool, or other learning resource."""

    title: str = Field(..., min_length=2)
    kind: Literal["Textbook", "Reference", "Article", "Website", "Software", "Video"]
    author_or_source: Optional[str] = None
    required: bool = True


class CLOAssessmentMapping(BaseModel):
    """One CLO-to-assessment weight or attainment mapping."""

    clo_number: int = Field(..., ge=1)
    assessment_id: str = Field(..., min_length=2)
    weight_percent: float = Field(..., ge=0, le=100)


class SyllabusDetails(BaseModel):
    """Milestone 2 structured syllabus fields attached to a validated CLO payload."""

    instructor: Optional[str] = None
    course_description_detailed: Optional[str] = None
    prerequisite_courses: List[str] = Field(default_factory=list)
    learning_resources: List[LearningResource] = Field(default_factory=list)
    assessments: List[Assessment] = Field(default_factory=list)
    weekly_schedule: List[WeeklySchedule] = Field(default_factory=list)
    clo_assessment_mapping: List[CLOAssessmentMapping] = Field(default_factory=list)
    grading_policy: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "SyllabusDetails":
        assessment_ids = {assessment.assessment_id for assessment in self.assessments}
        unknown_assessments = sorted(
            mapping.assessment_id
            for mapping in self.clo_assessment_mapping
            if mapping.assessment_id not in assessment_ids
        )
        if unknown_assessments:
            raise ValueError(
                f"CLO-assessment mappings reference unknown assessment IDs: {unknown_assessments}"
            )
        if self.assessments:
            total = sum(assessment.weight_percent for assessment in self.assessments)
            if abs(total - 100.0) > 0.01:
                raise ValueError(
                    f"Assessment weights must total 100 percent; found {total:.2f}."
                )
        return self


class CompleteSyllabusPayload(BaseModel):
    """Complete syllabus container preserving the validated Milestone 1 payload."""

    course_outcomes: CourseOutcomePayload
    syllabus: SyllabusDetails

    @model_validator(mode="after")
    def validate_clo_references(self) -> "CompleteSyllabusPayload":
        valid_clos = {co.clo_number for co in self.course_outcomes.course_outcomes}
        for assessment in self.syllabus.assessments:
            unknown = sorted(set(assessment.mapped_clos) - valid_clos)
            if unknown:
                raise ValueError(
                    f"Assessment '{assessment.assessment_id}' references unknown CLO(s): {unknown}"
                )
        for week in self.syllabus.weekly_schedule:
            unknown = sorted(set(week.mapped_clos) - valid_clos)
            if unknown:
                raise ValueError(
                    f"Week {week.week_number} references unknown CLO(s): {unknown}"
                )
        for mapping in self.syllabus.clo_assessment_mapping:
            if mapping.clo_number not in valid_clos:
                raise ValueError(
                    f"CLO-assessment mapping references unknown CLO {mapping.clo_number}"
                )
        return self
