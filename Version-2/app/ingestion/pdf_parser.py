"""PDF extraction with PyMuPDF: text, headings, footnotes, captions, tables, images, OCR, layout.

Every extracted element keeps its page number and bounding box (layout / coordinates).
Two-column pages (typical for the CFR) are re-ordered column by column.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator, Optional

from app.config import Settings
from app.ingestion.legal_structure import (
    looks_like_structural_heading,
    parse_revision_date,
    parse_title_info,
)
from app.models import DocumentInfo, PageElement

logger = logging.getLogger(__name__)

HEADER_RATIO = 0.065   # blocks entirely above this fraction of page height = running header
FOOTER_RATIO = 0.94    # blocks starting below this fraction = running footer / page number
FOOTNOTE_RATIO = 0.78  # small text below this fraction of the page = footnote
RE_CAPTION = re.compile(r"^(Figure|Fig\.|Table|Chart|Illustration|Exhibit)\s+[\w\-\.]+", re.I)

ProgressCB = Callable[[int, int, str], None]


def _fitz():
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:  # older PyMuPDF
        import fitz  # type: ignore
    return fitz


def _overlap(a: tuple, b: tuple) -> float:
    """Fraction of rectangle `a` covered by rectangle `b`."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return ((x1 - x0) * (y1 - y0)) / area


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.replace("\u00ad", "").strip()
        if not ln:
            continue
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}".strip()
    return re.sub(r"\s+", " ", out)


def _table_to_markdown(rows: list[list]) -> str:
    clean = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] for r in rows]
    ncols = max(len(r) for r in clean)
    clean = [r + [""] * (ncols - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def _is_real_table(rows: list[list]) -> bool:
    if not rows or len(rows) < 2:
        return False
    ncols = max(len(r) for r in rows)
    if ncols < 2:
        return False
    cells = [c for r in rows for c in r]
    filled = sum(1 for c in cells if c and str(c).strip())
    return filled / max(len(cells), 1) >= 0.4


class PDFParser:
    def __init__(self, settings: Settings, ocr_enabled: Optional[bool] = None):
        self.s = settings
        self.ocr_enabled = settings.ocr_enabled if ocr_enabled is None else ocr_enabled
        self._ocr_ok = False
        if self.ocr_enabled:
            try:
                import pytesseract

                pytesseract.get_tesseract_version()
                self._ocr_ok = True
            except Exception as exc:  # binary missing etc.
                logger.warning("OCR requested but Tesseract is unavailable (%s). OCR disabled.", exc)

    # ---------------------------------------------------------------- document info
    def read_info(self, pdf_path: str | Path, document_name: str = "") -> DocumentInfo:
        fitz = _fitz()
        path = Path(pdf_path)
        sha = hashlib.sha1()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)
        with fitz.open(str(path)) as doc:
            meta = {k: v for k, v in (doc.metadata or {}).items() if v}
            intro = "\n".join(doc[i].get_text("text") for i in range(min(6, len(doc))))
            page_count = len(doc)
        revision = parse_revision_date(intro) or self.s.default_revision_date or "Unknown"
        title_no, title_name = parse_title_info(intro)
        name = document_name.strip()
        if not name:
            if title_no:
                name = f"Code of Federal Regulations - Title {title_no}"
            else:
                name = meta.get("title") or path.stem.replace("_", " ").replace("-", " ").title()
        return DocumentInfo(
            document_id=sha.hexdigest()[:12],
            document_name=name,
            source_file=path.name,
            revision_date=revision,
            title_number=title_no,
            title_name=title_name,
            page_count=page_count,
            pdf_metadata=meta,
        )

    # ------------------------------------------------------------------- extraction
    def iter_elements(
        self, pdf_path: str | Path, info: DocumentInfo, progress: Optional[ProgressCB] = None
    ) -> Iterator[PageElement]:
        fitz = _fitz()
        with fitz.open(str(pdf_path)) as doc:
            body_size = self._body_font_size(doc)
            total = len(doc)
            for idx in range(total):
                page = doc[idx]
                try:
                    yield from self._page_elements(page, idx + 1, info, body_size)
                except Exception as exc:  # never let one bad page kill a 1000-page ingest
                    logger.exception("Page %s failed: %s", idx + 1, exc)
                if progress:
                    progress(idx + 1, total, f"Parsed page {idx + 1}/{total}")

    def _body_font_size(self, doc) -> float:
        sizes: Counter = Counter()
        n = len(doc)
        step = max(1, n // 60)
        for i in range(0, n, step):
            for block in doc[i].get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes[round(span["size"], 1)] += len(span["text"].strip())
        return sizes.most_common(1)[0][0] if sizes else 10.0

    # --------------------------------------------------------------- page handling
    def _page_elements(self, page, pno: int, info: DocumentInfo, body_size: float) -> list[PageElement]:
        fitz = _fitz()
        W, H = page.rect.width, page.rect.height
        els: list[PageElement] = []
        table_boxes: list[tuple] = []

        # 1) tables ---------------------------------------------------------------
        try:
            for t in page.find_tables().tables:
                rows = t.extract()
                if not _is_real_table(rows):
                    continue
                bbox = tuple(t.bbox)
                table_boxes.append(bbox)
                els.append(
                    PageElement(
                        pno, "table", _table_to_markdown(rows), bbox,
                        extra={"rows": len(rows), "cols": max(len(r) for r in rows)},
                    )
                )
        except Exception as exc:
            logger.debug("Table detection failed on page %s: %s", pno, exc)

        # 2) text blocks ----------------------------------------------------------
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            bbox = tuple(block["bbox"])
            if any(_overlap(bbox, tb) > 0.6 for tb in table_boxes):
                continue
            if bbox[3] < H * HEADER_RATIO or bbox[1] > H * FOOTER_RATIO:
                continue  # running header / footer / page number
            els.extend(self._block_to_elements(block, bbox, pno, H, body_size))

        text_chars = sum(len(e.text) for e in els if e.element_type in ("text", "heading"))

        # 3) scanned page -> OCR the whole page -------------------------------------
        if text_chars < 30 and not table_boxes:
            if self._ocr_ok:
                return self._ocr_page(page, pno, W, H)
            if self.ocr_enabled is False:
                logger.warning("Page %s has no text layer (scanned?). Enable OCR_ENABLED=true.", pno)

        # 4) images ---------------------------------------------------------------
        els.extend(self._image_elements(page, pno, info, W, H))

        # 5) reading order + caption linking ------------------------------------------
        els = self._order(els, W)
        return self._link_captions(els)

    def _block_to_elements(self, block: dict, bbox: tuple, pno: int, H: float, body_size: float) -> list[PageElement]:
        spans = [(s["text"], bool(s["flags"] & 16) or "bold" in s["font"].lower(), s["size"])
                 for ln in block["lines"] for s in ln["spans"]]
        if not spans:
            return []
        line_texts = ["".join(s["text"] for s in ln["spans"]) for ln in block["lines"]]
        text = _join_lines(line_texts)
        if not text or re.fullmatch(r"\d{1,4}", text):
            return []
        total = sum(len(t.strip()) for t, _, _ in spans) or 1
        avg_size = sum(sz * len(t.strip()) for t, _, sz in spans) / total
        bold_ratio = sum(len(t.strip()) for t, b, _ in spans if b) / total
        starts_bold = next((b for t, b, _ in spans if t.strip()), False)

        # heading text fused with body text in one block, e.g. bold "§ 101.9 Foo." + regular text
        if starts_bold and bold_ratio < 0.85:
            lead = ""
            for t, b, _ in spans:
                if not t.strip():
                    lead += t
                    continue
                if not b:
                    break
                lead += t
            lead_norm = " ".join(lead.split())
            if lead_norm and text.startswith(lead_norm) and looks_like_structural_heading(lead_norm):
                rest = text[len(lead_norm):].strip()
                out = [PageElement(pno, "heading", lead_norm, bbox, True, True, avg_size)]
                if rest:
                    out.append(PageElement(pno, "text", rest, bbox, False, False, avg_size))
                return out

        is_heading = (bold_ratio >= 0.85 or avg_size >= body_size * 1.12) and len(text) <= 240
        if RE_CAPTION.match(text) and len(text) <= 300:
            etype = "caption"
            is_heading = False
        elif bbox[1] > H * FOOTNOTE_RATIO and avg_size <= body_size * 0.93 and not is_heading:
            etype = "footnote"
        elif is_heading:
            etype = "heading"
        else:
            etype = "text"
        return [PageElement(pno, etype, text, bbox, is_heading, starts_bold, avg_size)]

    # ----------------------------------------------------------------- images / OCR
    def _ocr_pix(self, pix) -> str:
        if not self._ocr_ok:
            return ""
        try:
            import pytesseract
            from PIL import Image

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            return re.sub(r"\n{3,}", "\n\n", pytesseract.image_to_string(img, lang=self.s.ocr_lang)).strip()
        except Exception as exc:
            logger.debug("OCR failed: %s", exc)
            return ""

    def _ocr_page(self, page, pno: int, W: float, H: float) -> list[PageElement]:
        pix = page.get_pixmap(dpi=200)
        text = self._ocr_pix(pix)
        out = []
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if para:
                out.append(PageElement(pno, "ocr", para, (0.0, 0.0, W, H), extra={"ocr": True}))
        return out

    def _image_elements(self, page, pno: int, info: DocumentInfo, W: float, H: float) -> list[PageElement]:
        fitz = _fitz()
        out: list[PageElement] = []
        try:
            infos = page.get_image_info()
        except Exception:
            return out
        img_dir = self.s.images_dir / info.document_id
        n = 0
        for meta in infos:
            rect = fitz.Rect(meta["bbox"])
            if rect.width < 40 or rect.height < 40 or rect.width * rect.height > 0.85 * W * H:
                continue
            try:
                pix = page.get_pixmap(clip=rect, dpi=150)
            except Exception:
                continue
            if pix.width < self.s.min_image_px or pix.height < self.s.min_image_px:
                continue
            img_dir.mkdir(parents=True, exist_ok=True)
            path = img_dir / f"p{pno:04d}_{n:02d}.png"
            pix.save(str(path))
            n += 1
            ocr_text = self._ocr_pix(pix)
            out.append(
                PageElement(
                    pno, "image", ocr_text, tuple(rect),
                    extra={"path": str(path), "width": pix.width, "height": pix.height, "ocr": bool(ocr_text)},
                )
            )
        return out

    # ------------------------------------------------------------ order + captions
    @staticmethod
    def _order(els: list[PageElement], W: float) -> list[PageElement]:
        mid = W / 2

        def col(e: PageElement) -> str:
            x0, _, x1, _ = e.bbox
            if (x1 - x0) > 0.6 * W:
                return "F"
            if x1 <= mid + 0.05 * W:
                return "L"
            if x0 >= mid - 0.05 * W:
                return "R"
            return "F"

        fulls = sorted([e for e in els if col(e) == "F"], key=lambda e: e.bbox[1])
        cols = [e for e in els if col(e) != "F"]
        out: list[PageElement] = []
        prev = float("-inf")
        for f in fulls + [None]:  # type: ignore[list-item]
            limit = f.bbox[1] if f else float("inf")
            band = [c for c in cols if prev <= c.bbox[1] < limit]
            out += sorted([c for c in band if col(c) == "L"], key=lambda e: e.bbox[1])
            out += sorted([c for c in band if col(c) == "R"], key=lambda e: e.bbox[1])
            if f:
                out.append(f)
                prev = f.bbox[1]
        return out

    @staticmethod
    def _link_captions(els: list[PageElement]) -> list[PageElement]:
        targets = [e for e in els if e.element_type in ("table", "image")]
        keep: list[PageElement] = []
        for e in els:
            if e.element_type != "caption":
                keep.append(e)
                continue
            best, best_gap = None, 70.0
            for t in targets:
                gap = min(abs(e.bbox[1] - t.bbox[3]), abs(t.bbox[1] - e.bbox[3]))
                horiz = min(e.bbox[2], t.bbox[2]) - max(e.bbox[0], t.bbox[0])
                if horiz > 0 and gap < best_gap and not t.extra.get("caption"):
                    best, best_gap = t, gap
            if best is None:
                keep.append(e)  # standalone caption stays as text-like element
            else:
                best.extra["caption"] = e.text
                best.text = f"{e.text}\n{best.text}".strip() if best.text else e.text
        return keep
