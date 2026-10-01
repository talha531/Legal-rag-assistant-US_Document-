from app.ingestion.legal_structure import (
    LegalStructureTracker,
    extract_part_ref,
    extract_section_ref,
    parse_revision_date,
    parse_title_info,
)


def feed(tracker, items):
    for text, heading in items:
        tracker.update(text, heading)


def test_full_hierarchy():
    t = LegalStructureTracker()
    feed(t, [
        ("TITLE 21\u2014FOOD AND DRUGS", True),
        ("CHAPTER I\u2014FOOD AND DRUG ADMINISTRATION", True),
        ("PART 101\u2014FOOD LABELING", True),
        ("Subpart B\u2014General Food Labeling", True),
        ("\u00a7 101.9 Nutrition labeling of food.", True),
        ("(a) Nutrition information.", False),
        ("(1) First item.", False),
        ("(i) Roman item.", False),
    ])
    s = t.snapshot()
    assert (s["title"], s["chapter"], s["part"], s["subpart"], s["section"]) == ("21", "I", "101", "B", "101.9")
    assert LegalStructureTracker.subsection_ref(s) == "(a)(1)(i)"
    assert s["part_name"] == "FOOD LABELING"
    assert s["section_heading"] == "Nutrition labeling of food"


def test_new_section_resets_lower_levels():
    t = LegalStructureTracker()
    feed(t, [("PART 101\u2014X", True), ("\u00a7 101.9 A.", True), ("(a) x", False), ("\u00a7 101.10 B.", True)])
    s = t.snapshot()
    assert s["section"] == "101.10" and s["subsection"] == ""


def test_cross_reference_is_not_a_heading():
    t = LegalStructureTracker()
    feed(t, [("\u00a7 101.9 A.", True), ("See \u00a7 101.10 of this chapter for details.", False)])
    assert t.snapshot()["section"] == "101.9"


def test_toc_line_ignored():
    t = LegalStructureTracker()
    t.update("\u00a7 101.9 Nutrition labeling ........ 45", True)
    assert t.snapshot()["section"] == ""


def test_letter_i_after_h_is_subsection():
    t = LegalStructureTracker()
    feed(t, [("\u00a7 1.1 A.", True)] + [(f"({c}) text", False) for c in "abcdefgh"] + [("(i) ninth letter", False)])
    assert t.snapshot()["subsection"] == "i"


def test_section_derives_missing_part():
    t = LegalStructureTracker()
    t.update("\u00a7 900.1 Scope.", True)
    assert t.snapshot()["part"] == "900"


def test_references_and_dates():
    assert extract_section_ref("What does 21 CFR 101.9(c) say?") == "101.9"
    assert extract_section_ref("see section 900.3") == "900.3"
    assert extract_part_ref("rules in Part 101") == "101"
    assert parse_revision_date("Revised as of April 1, 1996") == "April 1, 1996"
    assert parse_revision_date("21 (4-1-96 Edition)") == "April 1, 1996"
    assert parse_title_info("Title 21\u2014Food and Drugs")[0] == "21"
