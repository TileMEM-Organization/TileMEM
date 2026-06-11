#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
from typing import Iterable

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "paper" / "tilemem_tilepo_v0_1_technical_report.md"
DEFAULT_OUTPUT = ROOT / "paper" / "TileMEM_TilePO_V0_1_Technical_Report.pdf"

rl_config.invariant = True


def build_pdf(source: Path, output: Path) -> None:
    styles = _styles()
    story = _markdown_to_flowables(source.read_text(), styles)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=0.52 * inch,
        leftMargin=0.52 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="TileMEM / TilePO V0.1 Technical Report",
        author="TerminusAkivili",
        subject="BF16 profile-guided tile-level placement/admission for MoE serving",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TileMEMTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "TileMEMH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "TileMEMH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "TileMEMBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.6,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "meta": ParagraphStyle(
            "TileMEMMeta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "TileMEMBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.3,
            leftIndent=14,
            firstLineIndent=-9,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "TileMEMCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.6,
            leading=9.4,
            textColor=colors.HexColor("#1f2933"),
            backColor=colors.HexColor("#f4f6f8"),
            borderColor=colors.HexColor("#d9e2ec"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TileMEMTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=9.1,
        ),
        "table_header": ParagraphStyle(
            "TileMEMTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.3,
            leading=8.8,
            textColor=colors.white,
        ),
        "footer": ParagraphStyle(
            "TileMEMFooter",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER,
        ),
    }


def _markdown_to_flowables(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = markdown.splitlines()
    flowables: list = []
    paragraph: list[str] = []
    code: list[str] | None = None
    table: list[str] = []
    first_heading = True

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(line.strip() for line in paragraph).strip()
            if text:
                flowables.append(Paragraph(_inline(text), styles["body"]))
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            rendered = _render_table(table, styles)
            if rendered is not None:
                flowables.append(rendered)
                flowables.append(Spacer(1, 5))
            table = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if code is None:
                code = []
            else:
                flowables.append(Preformatted("\n".join(code), styles["code"])); code = None
            continue
        if code is not None:
            code.append(line)
            continue

        if _is_table_line(line):
            flush_paragraph()
            table.append(line)
            continue
        flush_table()

        if not line.strip():
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            title = line[2:].strip()
            if not first_heading:
                flowables.append(PageBreak())
            flowables.append(Paragraph(_inline(title), styles["title"]))
            first_heading = False
            continue
        if line.startswith("## "):
            flush_paragraph()
            flowables.append(Paragraph(_inline(line[3:].strip()), styles["h1"]))
            continue
        if line.startswith("### "):
            flush_paragraph()
            flowables.append(Paragraph(_inline(line[4:].strip()), styles["h2"]))
            continue
        if line.startswith("**") and line.endswith("  "):
            flush_paragraph()
            flowables.append(Paragraph(_inline(line.rstrip()), styles["meta"]))
            continue
        if line.startswith("- "):
            flush_paragraph()
            flowables.append(Paragraph("- " + _inline(line[2:].strip()), styles["bullet"]))
            continue
        if re.match(r"^\d+\.\s+", line):
            flush_paragraph()
            match = re.match(r"^(\d+)\.\s+(.*)$", line)
            assert match is not None
            flowables.append(Paragraph(f"{match.group(1)}. " + _inline(match.group(2)), styles["bullet"]))
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_table()
    return flowables


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _render_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table | None:
    rows = [_split_table_row(line) for line in lines]
    rows = [row for row in rows if not _is_separator_row(row)]
    if not rows:
        return None
    col_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))
    data = []
    for r_index, row in enumerate(rows):
        style = styles["table_header"] if r_index == 0 else styles["table"]
        data.append([Paragraph(_inline(cell), style) for cell in row])
    col_widths = _column_widths(col_count)
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fbfcfd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9e2ec")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(row: Iterable[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in row)


def _column_widths(col_count: int) -> list[float]:
    available = A4[0] - 2 * 0.52 * inch
    if col_count == 6:
        return [82, 43, 112, 80, 112, 80]
    if col_count == 5:
        return [92, 60, 126, 126, 82]
    if col_count == 4:
        return [120, 70, 170, 120]
    return [available / col_count] * col_count


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = escaped.replace("  ", " ")
    return escaped


def _footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    text = f"TileMEM / TilePO V0.1 Technical Report  |  Page {doc.page}"
    canvas.drawCentredString(A4[0] / 2, 0.32 * inch, text)
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TileMEM / TilePO V0.1 technical report PDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
