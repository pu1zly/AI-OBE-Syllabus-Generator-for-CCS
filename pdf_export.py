"""Create a readable, dependency-free PDF of one generated course summary."""

from __future__ import annotations

import os
import tempfile
import textwrap
from typing import Dict, List, Tuple


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN_X = 48
MARGIN_TOP = 52
MARGIN_BOTTOM = 52
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_X * 2

STYLES: Dict[str, Tuple[str, int, int, Tuple[float, float, float]]] = {
    "title": ("F2", 20, 25, (0.10, 0.14, 0.24)),
    "code": ("F2", 10, 14, (0.55, 0.34, 0.21)),
    "meta": ("F1", 9, 13, (0.42, 0.48, 0.56)),
    "section": ("F2", 12, 17, (0.10, 0.14, 0.24)),
    "subhead": ("F2", 10, 14, (0.29, 0.39, 0.51)),
    "body": ("F1", 10, 14, (0.10, 0.14, 0.24)),
    "small": ("F1", 9, 12, (0.29, 0.39, 0.51)),
}


def _literal_string(text: str) -> str:
    encoded = text.encode("cp1252", errors="replace")
    escaped = []
    for byte in encoded:
        if byte in (ord("("), ord(")"), ord("\\")):
            escaped.append("\\" + chr(byte))
        elif byte < 32 or byte > 126:
            escaped.append(f"\\{byte:03o}")
        else:
            escaped.append(chr(byte))
    return "(" + "".join(escaped) + ")"


def _wrap(text: str, style: str, indent: int) -> List[str]:
    _font, size, _leading, _color = STYLES[style]
    available = CONTENT_WIDTH - indent
    max_chars = max(24, int(available / (size * 0.52)))
    return textwrap.wrap(
        str(text),
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=True,
    ) or [""]


def _summary_blocks(payload: dict) -> List[dict]:
    blocks: List[dict] = []

    def add(text: str, style: str = "body", indent: int = 0, before: int = 0):
        blocks.append({
            "style": style,
            "indent": indent,
            "before": before,
            "lines": _wrap(text, style, indent),
        })

    course_code = payload.get("course_code", "—")
    course_title = payload.get("course_title", "Course summary")
    add(course_title, "title")
    add(f"Course code: {course_code}", "code", before=2)
    target_pos = ", ".join(f"PO{po}" for po in payload.get("target_pos", [])) or "—"
    add(
        f"{payload.get('year_level', '—')}  |  {payload.get('semester', '—')}  |  "
        f"Target POs: {target_pos}",
        "meta",
        before=4,
    )

    add("Course description", "section", before=16)
    add(payload.get("course_description", "—"))

    domain_names = {
        "Knowledge": "Knowledge outcomes",
        "Skills": "Skills outcomes",
        "Attitude": "Attitude outcomes",
    }
    outcomes = payload.get("course_outcomes", [])
    add("Course outcomes", "section", before=16)
    for domain in ("Knowledge", "Skills", "Attitude"):
        domain_outcomes = [outcome for outcome in outcomes if outcome.get("domain") == domain]
        add(domain_names[domain], "subhead", before=8)
        if not domain_outcomes:
            add("No outcomes.", "small", indent=12)
            continue
        for outcome in domain_outcomes:
            add(
                f"CLO {outcome.get('clo_number', '—')}  |  "
                f"{outcome.get('bloom_level', '—')}",
                "subhead",
                indent=12,
                before=4,
            )
            add(outcome.get("co_description", "—"), "body", indent=24)
            mapped = ", ".join(
                f"PO{po}" for po in outcome.get("mapped_po", [])
            ) or "—"
            add(f"Mapped POs: {mapped}", "small", indent=24)

    weekly_schedule = payload.get("weekly_schedule", [])
    if weekly_schedule:
        add("Weekly learning plan", "section", before=16)
        for week in weekly_schedule:
            add(
                f"Week {week.get('week_number', '—')}  |  {week.get('topic', '—')}",
                "subhead",
                before=8,
            )
            add(
                f"Learning outcome: {week.get('learning_outcome', '—')}",
                "body",
                indent=12,
            )
            for activity in week.get("activities", []):
                add(f"Activity: {activity}", "small", indent=12)
            mapped = ", ".join(
                f"CLO {clo}" for clo in week.get("mapped_clos", [])
            ) or "—"
            add(
                f"Mapped CLOs: {mapped}  |  Contact hours: "
                f"{week.get('contact_hours', 3.0)}",
                "small",
                indent=12,
            )

    return blocks


def _make_page_stream(lines: List[Tuple[str, str, int, int]], page: int, total: int) -> bytes:
    commands = [
        "q",
        "0.88 0.90 0.93 RG",
        f"{MARGIN_X} 38 m {PAGE_WIDTH - MARGIN_X} 38 l S",
        "Q",
    ]
    for text, style, indent, y in lines:
        font, size, _leading, color = STYLES[style]
        commands.append(
            f"BT /{font} {size} Tf "
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
            f"{MARGIN_X + indent} {y} Td {_literal_string(text)} Tj ET"
        )
    commands.extend([
        "BT /F1 8 Tf 0.42 0.48 0.56 rg "
        f"{PAGE_WIDTH - MARGIN_X - 58} 24 Td "
        f"{_literal_string(f'Page {page} of {total}')} Tj ET",
    ])
    return ("\n".join(commands) + "\n").encode("ascii")


def _build_pdf(payload: dict) -> bytes:
    blocks = _summary_blocks(payload)
    pages: List[List[Tuple[str, str, int, int]]] = [[]]
    y = PAGE_HEIGHT - MARGIN_TOP
    bottom = MARGIN_BOTTOM + 8

    for block in blocks:
        style = block["style"]
        _font, _size, leading, _color = STYLES[style]
        lines = block["lines"]
        required = block["before"] + len(lines) * leading
        if pages[-1] and y - required < bottom:
            pages.append([])
            y = PAGE_HEIGHT - MARGIN_TOP
        y -= block["before"]
        for line in lines:
            if y - leading < bottom:
                pages.append([])
                y = PAGE_HEIGHT - MARGIN_TOP
            y -= leading
            pages[-1].append((line, style, block["indent"], y))

    streams = [
        _make_page_stream(lines, index + 1, len(pages))
        for index, lines in enumerate(pages)
    ]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{5 + index * 2} 0 R" for index in range(len(streams)))
            + f"] /Count {len(streams)} >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
        b"/Encoding /WinAnsiEncoding >>",
    ]
    for index, stream in enumerate(streams):
        page_id = 5 + index * 2
        stream_id = page_id + 1
        objects.extend([
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
                f"{PAGE_WIDTH} {PAGE_HEIGHT}] /Resources "
                f"<< /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode("ascii"),
            (
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"endstream"
            ),
        ])

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def write_course_summary_pdf(path: str, payload: dict) -> None:
    """Write a paginated, selectable-text PDF summary for a course payload."""
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    os.makedirs(directory, exist_ok=True)
    content = _build_pdf(payload)
    fd, temp_path = tempfile.mkstemp(
        prefix=".course-summary-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as handle:
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
