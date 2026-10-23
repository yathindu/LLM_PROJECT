"""Convert docs/report.md to docs/Technical_Report.pdf using reportlab.  Run: python docs/build_report.py (needs reportlab, see requirements-dev.txt)"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).parent
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, keepWithNext=1)
CODE = ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10, backColor=colors.whitesmoke, borderPadding=4)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=11, spaceAfter=0)


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', text)


COLUMNS = {  # first header cell -> column widths in cm (17 cm of text width)
    "Tool (agent)": [4.0, 6.0, 7.0],
    "Prompt": [4.2, 6.6, 6.2],
    "Member": [3.0, 8.4, 5.6],
}


def build() -> None:
    lines = (HERE / "report.md").read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story += [Preformatted("\n".join(block), CODE), Spacer(1, 6)]
        elif line.startswith("!["):
            alt, src = re.match(r"!\[(.*?)\]\((.*?)\)", line).groups()
            w, h = ImageReader(str(HERE / src)).getSize()
            story += [Image(str(HERE / src), width=16.5 * cm, height=16.5 * cm * h / w), Spacer(1, 4), Paragraph(inline(alt), CELL), Spacer(1, 6)]
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), H1))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append([Paragraph(inline(c), CELL) for c in cells])
                i += 1
            i -= 1
            first = re.sub(r"<[^>]+>", "", rows[0][0].text)
            t = Table(rows, colWidths=[w * cm for w in COLUMNS.get(first, [5.6, 5.7, 5.7])], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 8)]
        elif re.match(r"(- |\d+\. )", line):
            ordered = line[0].isdigit()
            items = []
            while i < len(lines) and re.match(r"(- |\d+\. )", lines[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^(- |\d+\. )", "", lines[i])), BODY)))
                i += 1
            i -= 1
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet", leftIndent=14))
        elif line.strip():
            story.append(Paragraph(inline(line), BODY))
        i += 1
    SimpleDocTemplate(str(HERE / "Technical_Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm, title="StudyMate Technical Report").build(story)
    words = len(re.findall(r"\w+", (HERE / "report.md").read_text(encoding="utf-8")))
    print(f"Wrote docs/Technical_Report.pdf  (~{words} words incl. placeholders and diagram)")


if __name__ == "__main__":
    build()
