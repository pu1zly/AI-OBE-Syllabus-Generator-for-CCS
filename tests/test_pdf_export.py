import os
import re
import tempfile
import unittest

from pdf_export import write_course_summary_pdf


class PdfExportTests(unittest.TestCase):
    def test_course_summary_pdf_contains_outcomes_and_weekly_plan(self):
        payload = {
            "course_code": "CS325",
            "course_title": "Animal Welfare",
            "course_description": "Covers welfare principles and ethical care practices.",
            "year_level": "3rd Year",
            "semester": "1st Semester",
            "target_pos": [1, 2],
            "course_outcomes": [
                {
                    "clo_number": 1,
                    "domain": "Knowledge",
                    "bloom_level": "Remember",
                    "co_description": "Describe animal welfare principles.",
                    "mapped_po": [1],
                }
            ],
            "weekly_schedule": [
                {
                    "week_number": week,
                    "topic": f"Animal welfare topic {week}",
                    "learning_outcome": f"Describe animal welfare practice {week}.",
                    "activities": ["Review a guided case.", "Complete a short exercise."],
                    "mapped_clos": [1],
                    "contact_hours": 3.0,
                }
                for week in range(1, 18)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "summary.pdf")
            write_course_summary_pdf(path, payload)
            with open(path, "rb") as handle:
                pdf = handle.read()

        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"Animal Welfare", pdf)
        self.assertIn(b"Weekly learning plan", pdf)
        page_count = int(re.search(rb"/Count (\d+)", pdf).group(1))
        self.assertGreater(page_count, 1)


if __name__ == "__main__":
    unittest.main()
