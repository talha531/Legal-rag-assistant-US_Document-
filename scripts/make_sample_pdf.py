"""Create a small SYNTHETIC CFR-style PDF for smoke-testing (needs `pip install reportlab`).
The content is invented ("Title 99") and is NOT real law."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_cfr_excerpt.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=10, leading=13, spaceAfter=5)
bold = ParagraphStyle("bold", parent=body, fontName="Helvetica-Bold", fontSize=11, spaceBefore=8)
big = ParagraphStyle("big", parent=body, fontName="Helvetica-Bold", fontSize=13, spaceBefore=10)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)

story = [
    Paragraph("SAMPLE / SYNTHETIC DOCUMENT - NOT REAL LAW - for software testing only", small),
    Paragraph("Title 99\u2014Sample Regulations", big),
    Paragraph("Revised as of April 1, 1996", body),
    Spacer(1, 8),
    Paragraph("CHAPTER I\u2014DEPARTMENT OF SAMPLE AFFAIRS", big),
    Paragraph("PART 900\u2014WIDGET MANUFACTURING RECORDS", big),
    Paragraph("Subpart A\u2014General Provisions", bold),
    Paragraph("\u00a7 900.1 Scope.", bold),
    Paragraph("(a) This part applies to every person who manufactures widgets for sale in the sample jurisdiction.", body),
    Paragraph("(b) A person who only repairs widgets is not a manufacturer for purposes of this part.", body),
    Paragraph("\u00a7 900.2 Definitions.", bold),
    Paragraph("(a) Widget means a small mechanical device intended for consumer use.", body),
    Paragraph("(b) Manufacturer means a person who makes, assembles, or packages widgets.", body),
    Paragraph("Subpart B\u2014Records", bold),
    Paragraph("\u00a7 900.3 Record retention.", bold),
    Paragraph("(a) Each manufacturer shall keep batch records for the period listed in the table below.", body),
    Paragraph("(1) Records shall be legible and stored at the manufacturing site.", body),
    Paragraph("(2) Records shall be made available to the inspector on request.", body),
    Table(
        [["Record type", "Retention period"], ["Batch records", "2 years"], ["Complaint files", "3 years"], ["Training logs", "1 year"]],
        style=TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]),
    ),
    Paragraph("Table 1\u2014Retention periods by record type", body),
    PageBreak(),
    Paragraph("\u00a7 900.4 Labeling.", bold),
    Paragraph("(a) Every widget package shall bear the manufacturer name and a batch number.", body),
    Paragraph("(b) A manufacturer that fails to comply with paragraph (a) of this section commits a prohibited act.", body),
    Paragraph("\u00a7 900.5 [Reserved]", bold),
]
SimpleDocTemplate(str(OUT), pagesize=letter, title="Sample Regulations (synthetic)").build(story)
print("wrote", OUT)
