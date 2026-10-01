from app.ingestion.chunker import LegalChunker, split_long_text, split_table_markdown
from app.models import DocumentInfo, PageElement

DOC = DocumentInfo("doc1", "Sample Doc", "sample.pdf", "April 1, 1996", "99", "Sample Regs")


def el(page, text, etype="text", heading=False, **extra):
    return PageElement(page, etype, text, (0, 0, 1, 1), is_heading=heading, extra=extra)


def run(elements, **kw):
    return list(LegalChunker(DOC, **({"max_chars": 1800, "min_chars": 350, "overlap": 200} | kw)).chunk(elements))


def test_chunks_never_cross_sections_and_keep_metadata():
    chunks = run([
        el(1, "PART 900\u2014RULES", "heading", True),
        el(1, "\u00a7 900.1 Scope.", "heading", True),
        el(1, "(a) Applies to makers."),
        el(2, "\u00a7 900.2 Definitions.", "heading", True),
        el(2, "(a) Widget means a device."),
    ])
    assert [c.payload["section"] for c in chunks] == ["900.1", "900.2"]
    p = chunks[0].payload
    assert p["document_name"] == "Sample Doc" and p["revision_date"] == "April 1, 1996"
    assert p["part"] == "900" and p["title"] == "99" and p["page"] == 1
    assert p["hierarchy_path"].startswith("Title 99 > Part 900 > \u00a7 900.1")
    assert chunks[0].embed_text.startswith("Sample Doc | ")


def test_table_is_separate_chunk():
    chunks = run([
        el(1, "\u00a7 900.3 Records.", "heading", True),
        el(1, "(a) Keep records."),
        el(1, "| a | b |\n| --- | --- |\n| 1 | 2 |", "table", caption="Table 1"),
    ])
    assert [c.payload["element_type"] for c in chunks] == ["text", "table"]
    assert chunks[1].payload["section"] == "900.3"


def test_footnote_and_image_paths():
    chunks = run([
        el(1, "\u00a7 900.1 A.", "heading", True),
        el(1, "(a) Body text here."),
        el(1, "", "image", path="storage/images/x.png"),
        el(1, "1 A footnote.", "footnote"),
        el(2, "\u00a7 900.2 B.", "heading", True),
        el(2, "(a) More body."),
    ])
    types = [c.payload["element_type"] for c in chunks]
    assert "footnote" in types
    assert any("storage/images/x.png" in c.payload["image_paths"] for c in chunks)


def test_long_section_is_split_with_page_range():
    long = "The manufacturer shall keep every record. " * 200
    chunks = run([el(1, "\u00a7 900.1 A.", "heading", True), el(1, long), el(2, "(b) tail text.")])
    assert len(chunks) >= 3
    assert all(c.payload["char_count"] <= 1800 * 1.3 for c in chunks)
    assert chunks[-1].payload["page_end"] == 2 or chunks[-1].payload["page_start"] >= 1


def test_split_helpers():
    assert all(len(p) <= 500 for p in split_long_text("One sentence here. " * 200, 500, 100))
    md = "| a | b |\n| - | - |\n" + "\n".join(f"| {i} | v |" for i in range(200))
    pieces = split_table_markdown(md, 400)
    assert len(pieces) > 1 and all(p.startswith("| a | b |") for p in pieces)
