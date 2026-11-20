import json
import os
import tempfile
import unittest

from pydantic import ValidationError

from llm_engine import (
    EngineConfig,
    MockProvider,
    _build_mock_provider,
    auto_repair_payload_dict,
    build_custom_subject,
    find_builtin_subject,
    generate_course_description,
    apply_generation_preset,
    improve_course_description,
    _normalize_payload_for_course,
    next_custom_course_code,
    run_batch,
    write_json_atomic,
)
from obe_schemas import (
    Assessment,
    CompleteSyllabusPayload,
    CourseOutcomePayload,
    SyllabusDetails,
)


def valid_payload():
    return {
        "course_code": "CS321",
        "course_title": "Computer Graphics Programming",
        "course_description": "Covers graphics pipelines, rendering, shaders, and real-time graphics applications.",
        "target_pos": [1, 2, 3],
        "course_outcomes": [
            {"clo_number": 1, "domain": "Knowledge", "bloom_level": "Remember", "co_description": "Identify graphics pipeline stages.", "mapped_po": [1]},
            {"clo_number": 2, "domain": "Skills", "bloom_level": "Manipulation", "co_description": "Implement shader programs.", "mapped_po": [2]},
            {"clo_number": 3, "domain": "Attitude", "bloom_level": "Valuing", "co_description": "Value accessible visual design.", "mapped_po": [3]},
            {"clo_number": 4, "domain": "Knowledge", "bloom_level": "Analyze", "co_description": "Analyze rendering performance.", "mapped_po": [1, 3]},
        ],
    }


class SchemaTests(unittest.TestCase):
    def test_valid_payload(self):
        payload = CourseOutcomePayload.model_validate(valid_payload())
        self.assertEqual(payload.ksa_summary(), {"K": 2, "S": 1, "A": 1})

    def test_mapped_po_must_be_in_target_pos(self):
        data = valid_payload()
        data["course_outcomes"][0]["mapped_po"] = [9]
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_missing_ksa_domain_is_rejected(self):
        data = valid_payload()
        data["course_outcomes"][2]["domain"] = "Knowledge"
        data["course_outcomes"][2]["bloom_level"] = "Remember"
        data["course_outcomes"][2]["co_description"] = "Identify graphics standards."
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_weekly_schedule_requires_sequential_weeks_and_valid_clos(self):
        data = valid_payload()
        data["weekly_schedule"] = [
            {
                "week_number": 1,
                "topic": "Animal welfare foundations",
                "learning_outcome": "Describe core animal welfare principles.",
                "activities": ["Review a guided case."],
                "mapped_clos": [1],
            },
            {
                "week_number": 3,
                "topic": "Ethical care decisions",
                "learning_outcome": "Analyze ethical choices in animal care.",
                "activities": ["Discuss a case study."],
                "mapped_clos": [99],
            },
        ]
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)
        data["weekly_schedule"][1]["week_number"] = 2
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_duplicate_clo_numbers_are_rejected(self):
        data = valid_payload()
        data["course_outcomes"][3]["clo_number"] = 1
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_level_specific_verb_is_enforced(self):
        data = valid_payload()
        data["course_outcomes"][0]["bloom_level"] = "Create"
        data["course_outcomes"][0]["co_description"] = "Identify graphics pipeline stages."
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_banned_verb_is_rejected(self):
        data = valid_payload()
        data["course_outcomes"][0]["co_description"] = "Understand graphics pipeline stages."
        with self.assertRaises(ValidationError):
            CourseOutcomePayload.model_validate(data)

    def test_complete_syllabus_cross_references_clos(self):
        payload = CourseOutcomePayload.model_validate(valid_payload())
        assessment = Assessment(
            assessment_id="LAB1",
            name="Graphics Laboratory",
            assessment_type="Laboratory",
            weight_percent=100,
            mapped_clos=[1, 2],
        )
        syllabus = SyllabusDetails(
            assessments=[assessment],
            clo_assessment_mapping=[
                {"clo_number": 1, "assessment_id": "LAB1", "weight_percent": 100}
            ],
        )
        CompleteSyllabusPayload(course_outcomes=payload, syllabus=syllabus)


class EngineTests(unittest.TestCase):
    def test_manual_description_is_preserved(self):
        course = build_custom_subject(
            "Animal Welfare",
            "CS325",
            course_description="Covers animal welfare principles and ethical care practices.",
        )
        course = course.model_copy(update={
            "description_source": "manual",
            "improve_description": False,
        })
        provider = MockProvider([], description_responses=[
            '{"course_description":"This response must not be used."}'
        ])
        self.assertEqual(
            course.course_description,
            "Covers animal welfare principles and ethical care practices.",
        )
        self.assertFalse(course.improve_description)

    def test_description_improvement_function_validates_output(self):
        course = build_custom_subject(
            "Animal Welfare",
            "CS325",
            course_description="Animal welfare course.",
        )
        provider = MockProvider(
            [],
            description_responses=[
                '{"course_description":"Examines animal welfare principles, ethical care practices, and practical approaches to responsible treatment across institutions."}'
            ],
        )
        improved = improve_course_description(course, provider)
        self.assertIn("animal welfare principles", improved.lower())

    def test_improve_description_flag_is_applied_before_course_generation(self):
        course = build_custom_subject(
            "Animal Welfare",
            "CS325",
            course_description="Animal welfare course.",
            target_pos=[1, 2, 3],
        ).model_copy(update={
            "description_source": "ai_improved",
            "improve_description": True,
        })
        improved_description = (
            "Examines animal welfare principles, ethical care practices, and "
            "practical approaches to responsible treatment across institutions."
        )
        provider = MockProvider(
            [json.dumps(valid_payload())],
            description_responses=[
                json.dumps({"course_description": improved_description})
            ],
        )
        result = run_batch(
            provider,
            EngineConfig(
                model_name="mock",
                generation_preset="Fast",
                week_count=8,
            ),
            catalog=[course],
            verbose=False,
        )

        self.assertEqual(result.total_courses_validated, 1, result.failures)
        payload = result.records[0].payload
        self.assertEqual(payload.course_description, improved_description)
        self.assertEqual(result.records[0].audit.description_source, "ai_improved")
        self.assertEqual(provider.calls_made, 2)

    def test_week_count_default_bounds_and_token_budget(self):
        default = apply_generation_preset("Balanced", "mock")
        maximum = apply_generation_preset("Balanced", "mock", week_count=12)
        self.assertEqual(default.week_count, 8)
        self.assertGreater(maximum.num_predict, default.num_predict)
        for invalid in (0, 13):
            with self.assertRaises(ValueError):
                EngineConfig(week_count=invalid)

    def test_generation_rejects_missing_requested_weeks(self):
        class NoWeeklyScheduleProvider:
            def chat(self, _messages, temperature):
                return json.dumps(valid_payload())

        course = build_custom_subject(
            "Animal Welfare",
            "CS325",
            course_description="Covers animal welfare principles and ethical care practices.",
        )
        result = run_batch(
            NoWeeklyScheduleProvider(),
            EngineConfig(week_count=2, max_retries=0),
            catalog=[course],
            verbose=False,
        )
        self.assertEqual(result.total_courses_validated, 0)
        self.assertIn("Expected exactly 2 weekly_schedule entries", result.failures[0]["error"])

    def test_next_custom_course_code_follows_catalog_and_skips_used_codes(self):
        self.assertEqual(next_custom_course_code(), "CS325")
        self.assertEqual(next_custom_course_code(["CS325"]), "CS326")
        self.assertEqual(next_custom_course_code(["cs325", "CS326"]), "CS327")

    def test_repairs_common_verb_and_level_mismatches(self):
        data = {
            "course_code": "CS326",
            "course_title": "Animal Welfare",
            "course_description": "Covers animal welfare principles and ethical care practices.",
            "target_pos": [1, 2, 3],
            "course_outcomes": [
                {
                    "clo_number": 1,
                    "domain": "Knowledge",
                    "bloom_level": "Remember",
                    "co_description": "Apply understanding of animal welfare concepts through recall.",
                    "mapped_po": [1],
                },
                {
                    "clo_number": 2,
                    "domain": "Skills",
                    "bloom_level": "Manipulation",
                    "co_description": "Implement practical animal care procedures.",
                    "mapped_po": [2],
                },
                {
                    "clo_number": 3,
                    "domain": "Attitude",
                    "bloom_level": "Responding",
                    "co_description": "Use critical thinking to solve problems.",
                    "mapped_po": [3],
                },
                {
                    "clo_number": 4,
                    "domain": "Knowledge",
                    "bloom_level": "Remember",
                    "co_description": "Break down complex information into components.",
                    "mapped_po": [1],
                },
                {
                    "clo_number": 5,
                    "domain": "Attitude",
                    "bloom_level": "Responding",
                    "co_description": "Assess the validity of information and arguments.",
                    "mapped_po": [3],
                },
            ],
        }
        repaired = auto_repair_payload_dict(data)
        payload = CourseOutcomePayload.model_validate(repaired)
        descriptions = [outcome.co_description for outcome in payload.course_outcomes]
        self.assertEqual(payload.course_outcomes[3].bloom_level, "Analyze")
        self.assertEqual(payload.course_outcomes[4].bloom_level, "Valuing")
        self.assertTrue(descriptions[0].startswith("Describe "))
        self.assertTrue(descriptions[2].startswith("Practice "))
        self.assertTrue(descriptions[3].startswith("Analyze "))
        self.assertTrue(descriptions[4].startswith("Advocate "))

    def test_repairs_scalar_program_outcomes_and_attitude_respond_verb(self):
        data = valid_payload()
        data["course_outcomes"][0]["mapped_po"] = 1
        data["course_outcomes"][1]["mapped_po"] = 2
        data["course_outcomes"][2]["mapped_po"] = 3
        data["course_outcomes"][2]["bloom_level"] = "Responding"
        data["course_outcomes"][2]["co_description"] = (
            "Respond to real-world cases involving project management."
        )
        meta = build_custom_subject(
            "Animal Welfare",
            "CS326",
            course_description=(
                "Covers animal welfare principles and ethical care practices."
            ),
            target_pos=[1, 2, 3],
        )
        normalized = _normalize_payload_for_course(data, meta)
        repaired = auto_repair_payload_dict(normalized)
        payload = CourseOutcomePayload.model_validate(repaired)
        self.assertEqual(payload.course_outcomes[0].mapped_po, [1])
        self.assertEqual(payload.course_outcomes[1].mapped_po, [2])
        self.assertEqual(payload.course_outcomes[2].mapped_po, [3])
        self.assertEqual(
            payload.course_outcomes[2].co_description,
            "Practice appropriate responses to real-world cases involving project management.",
        )

    def test_generation_presets_get_different_limits(self):
        fast = apply_generation_preset("Fast", "mock")
        balanced = apply_generation_preset("Balanced", "mock")
        quality = apply_generation_preset("Quality", "mock")
        self.assertEqual(fast.num_predict, 900 + 8 * 140)
        self.assertEqual(fast.max_retries, 1)
        self.assertGreater(balanced.num_predict, fast.num_predict)
        self.assertGreater(quality.num_predict, balanced.num_predict)

    def test_subject_queue_accepts_mixed_entries(self):
        queue = []
        builtin = find_builtin_subject("CS321")
        custom = build_custom_subject("Cybersecurity Fundamentals", "CS399")
        queue.append(builtin)
        queue.append(custom)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue[0].course_code, "CS321")
        self.assertEqual(queue[1].course_code, "CS399")

    def test_subject_queue_removal_and_clear(self):
        queue = [
            find_builtin_subject("CS321"),
            build_custom_subject("Cybersecurity Fundamentals", "CS399"),
        ]
        queue.pop(0)
        self.assertEqual(len(queue), 1)
        queue.clear()
        self.assertEqual(queue, [])

    def test_builtin_subject_lookup(self):
        by_code = find_builtin_subject("CS321")
        by_title = find_builtin_subject("Computer Graphics Programming")
        self.assertIsNotNone(by_code)
        self.assertEqual(by_code.course_code, "CS321")
        self.assertEqual(by_title.course_title, "Computer Graphics Programming")

    def test_custom_subject_requires_title(self):
        with self.assertRaises(ValueError):
            build_custom_subject(title="  ", course_code="CUSTOM")

    def test_custom_subject_defaults_metadata(self):
        subject = build_custom_subject(
            title="Cybersecurity Fundamentals",
            course_code="cs399",
        )
        self.assertEqual(subject.course_code, "CS399")
        self.assertEqual(subject.course_title, "Cybersecurity Fundamentals")
        self.assertIsNone(subject.course_description)

    def test_generated_description_is_validated(self):
        subject = build_custom_subject(
            title="Cybersecurity Fundamentals",
            course_code="CS399",
        )
        provider = MockProvider(
            [],
            description_responses=[
                '{"course_description":"Introduces core cybersecurity principles, threats, controls, and defensive practices. Students apply practical safeguards to realistic IT systems."}'
            ],
        )
        description = generate_course_description(subject, provider)
        self.assertIn("cybersecurity", description.lower())

    def test_supplied_description_bypasses_generation(self):
        subject = build_custom_subject(
            title="Cybersecurity Fundamentals",
            course_code="CS399",
            course_description="Covers network security, cryptography, incident response, and defensive system design.",
        )
        provider = MockProvider([], description_responses=[])
        description = generate_course_description(subject, provider)
        self.assertEqual(description, subject.course_description)

    def test_mock_batch_validates(self):
        cfg = EngineConfig(model_name="mock-scripted", max_retries=3)
        result = run_batch(
            _build_mock_provider(),
            cfg,
            verbose=False,
        )
        self.assertEqual(result.total_courses_attempted, 4)
        self.assertEqual(result.total_courses_validated, 4)
        self.assertEqual(result.failures, [])
        for record in result.records:
            self.assertEqual(len(record.payload.weekly_schedule), 8)
            self.assertEqual(
                [week.week_number for week in record.payload.weekly_schedule],
                list(range(1, 9)),
            )

    def test_fast_mode_uses_assigned_custom_course_metadata(self):
        course = build_custom_subject(
            "Animal Welfare",
            "CS325",
            course_description="Covers animal welfare principles and ethical care practices.",
        )
        result = run_batch(
            MockProvider([json.dumps(valid_payload())]),
            EngineConfig(model_name="mock", generation_preset="Fast", max_retries=0),
            catalog=[course],
            verbose=False,
        )
        self.assertEqual(result.total_courses_validated, 1)
        self.assertEqual(result.records[0].payload.course_code, "CS325")
        self.assertEqual(result.records[0].payload.course_title, "Animal Welfare")

    def test_cancelled_batch_stops_before_next_course(self):
        result = run_batch(
            _build_mock_provider(),
            EngineConfig(model_name="mock-scripted", max_retries=3),
            verbose=False,
            is_cancelled=lambda: True,
        )
        self.assertEqual(result.total_courses_attempted, 0)
        self.assertEqual(result.total_courses_validated, 0)

    def test_mid_course_cancellation_is_not_reported_as_failure(self):
        state = {"cancelled": False}
        outcomes = []

        class CancellingProvider:
            def chat(self, messages, temperature):
                state["cancelled"] = True
                return "{}"

        result = run_batch(
            CancellingProvider(),
            EngineConfig(
                model_name="cancelling-provider",
                max_retries=0,
                generation_preset="Fast",
            ),
            catalog=[find_builtin_subject("CS321")],
            verbose=False,
            is_cancelled=lambda: state["cancelled"],
            on_course_done=lambda _meta, outcome: outcomes.append(outcome),
        )
        self.assertEqual(result.total_courses_attempted, 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(outcomes[0].error, "Cancelled by user.")

    def test_negative_retries_are_rejected(self):
        with self.assertRaises(ValueError):
            EngineConfig(max_retries=-1)

    def test_atomic_json_writer_replaces_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "result.json")
            write_json_atomic(destination, json.dumps({"ok": True}))
            with open(destination, "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"ok": True})


if __name__ == "__main__":
    unittest.main()
