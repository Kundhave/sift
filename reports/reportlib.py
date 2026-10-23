"""Shared reportlab helpers. Content layout follows the generic structure (title, one-row header table,
numbered sections starting with '1. Executive Summary', tables, figures); no content is copied from any other project."""
from __future__ import annotations
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
                                Preformatted)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "CSE23037"
STUDENT = "Kundhave S (CB.SC.U4CSE23037)"
NAVY = colors.HexColor("#1f3a5f")
GREY = colors.HexColor("#eef1f5")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#444444"), spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=11, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#333333"), spaceBefore=5, spaceAfter=2)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=SMALL, textColor=colors.HexColor("#555555"), spaceAfter=6)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=7.4, leading=9)


def J(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def P(t, st=BODY): return Paragraph(t, st)
def H(t): return Paragraph(t, H2)
def H3_(t): return Paragraph(t, H3)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths=None, header=True, font=None, zebra=True):
    data = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0: st.append(("BACKGROUND", (0, r), (-1, r), GREY))
    t.setStyle(TableStyle(st))
    return t


def header_table(project, deliverable, team, member, extra_label, extra_val, width=17.4 * cm):
    lab = ParagraphStyle("hl", parent=CELL, fontSize=6.6, leading=8, textColor=colors.HexColor("#6b7380"), alignment=1)
    val = ParagraphStyle("hv", parent=CELL, fontName="Helvetica-Bold", fontSize=8.2, leading=10, textColor=colors.HexColor("#333333"), alignment=1)
    rows = [[Paragraph(x.upper(), lab) for x in ("Project", "Deliverable", "Team No", "Team Member", extra_label)],
            [Paragraph(x, val) for x in (project, deliverable, team, member, extra_val)]]
    t = Table(rows, [width * f for f in (0.16, 0.24, 0.1, 0.22, 0.28)])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d3e3")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def figure(path, width_cm=16, caption=None):
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, CAP)] if caption else [img])


def fmt(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def pct(x, d=1): return "-" if x is None else f"{100*x:.{d}f}%"


def _footer(title):
    def f(c, doc):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#666666"))
        c.drawString(2 * cm, 1.1 * cm, f"Sift | {title} | Kundhave S, CB.SC.U4CSE23037 | 23CSE473")
        c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}"); c.restoreState()
    return f


def build(path, story, title, cover=None):
    OUT.mkdir(exist_ok=True)
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                          title=title, author="Kundhave S")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=_footer(title))])
    doc.build(story)
    return path


def title_block(title, subtitle):
    return [Paragraph(title, H1), Paragraph(subtitle, SUB)]


def sci(x, d=3): return f"{x:.{d}f}"


# ---------------------------------------------------------------- shared helpers for build_reports.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = ROOT
TRACE_PATH = ROOT / "reports" / "number_trace.json"
_TRACE: dict = {}
PALETTE = ["#1f3a5f", "#d9822b", "#2a9d8f", "#b5446e", "#7b8794", "#6a4c93"]


def RJ(rel):
    """Load a JSON relative to the repo root (e.g. 'results/xai.json')."""
    return json.load(open(ROOT / rel))


def trace(deliverable, item, sources, note=None):
    """Record which results file(s) a table/figure/number group was generated from."""
    _TRACE.setdefault(deliverable, {})[item] = {"sources": list(sources), **({"note": note} if note else {})}


def save_trace():
    old = json.load(open(TRACE_PATH)) if TRACE_PATH.exists() else {}
    old.update(_TRACE)
    json.dump(old, open(TRACE_PATH, "w"), indent=2)


def fig_save(fig, name, dpi=200):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return p


def equation(latex, name, size=13):
    """Render one equation to a PNG with matplotlib mathtext; returns a flowable."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, latex, fontsize=size)
    p = fig_save(fig, f"eq_{name}.png", dpi=220)
    from PIL import Image as PI
    w, h = PI.open(p).size
    sc = 72 / 220 * 0.78
    img = Image(str(p), width=w * sc, height=h * sc); img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 2), img, Spacer(1, 3)])


def not_executed(cmd, what):
    return P(f"{what} Command: <font face='Courier' size='7.5'>{cmd}</font>", SMALL)
