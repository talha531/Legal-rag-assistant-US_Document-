from app.agent.citations import build_citations, build_context, citation_label, revision_disclaimer

HITS = [
    {"document_name": "Doc", "revision_date": "April 1, 1996", "title": "99", "part": "900", "section": "900.3",
     "subsection": "(a)", "page_start": 2, "page_end": 2, "text": "Keep batch records 2 years. See 900.4.", "score": 0.8},
    {"document_name": "Doc", "revision_date": "April 1, 1996", "part": "900", "section": "900.4", "page_start": 3,
     "text": "Labeling.", "score": 0.6},
]


def test_label():
    assert citation_label(HITS[0]) == "Title 99, Part 900, \u00a7 900.3(a), p. 2"


def test_invalid_markers_removed_and_valid_cited():
    ctx, used = build_context(HITS, 7000)
    assert "[S1]" in ctx and "[S2]" in ctx
    answer, cites, warns = build_citations("Keep for 2 years [S1]. Bogus [S9].", used)
    assert "[S9]" not in answer and "[S1]" in answer
    assert [c["marker"] for c in cites if c["cited"]] == ["S1"]
    assert any("non-existent" in w for w in warns)


def test_no_markers_warns():
    _, used = build_context(HITS, 7000)
    _, cites, warns = build_citations("An answer with no markers.", used)
    assert not any(c["cited"] for c in cites) and warns


def test_unsupported_section_flagged():
    _, used = build_context(HITS, 7000)
    _, _, warns = build_citations("Also see \u00a7 900.99 [S1].", used)
    assert any("900.99" in w for w in warns)


def test_disclaimer_mentions_revision():
    _, used = build_context(HITS, 7000)
    _, cites, _ = build_citations("x [S1]", used)
    assert "April 1, 1996" in revision_disclaimer(cites)
