"""Text extraction helpers for SERGDATA Knowledge. OCR is intentionally disabled."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".texi", ".texinfo", ".info", ".text", ".csv", ".tsv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".conf", ".sh", ".go", ".rs",
    ".c", ".h", ".cpp", ".java", ".xml", ".sql", ".log", ".service",
    ".lua", ".uci", ".mk", ".cmake", ".proto", ".sum", ".mod", ".s",
    ".patch", ".diff", ".in", ".config", ".default",
}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".odt", ".epub", ".ipynb"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS


def _clean_text(text: str) -> str:
    # GNU Info may use overstrike sequences for bold/underline.
    text = re.sub(r"(.)\x08\1", r"\1", text)
    text = text.replace("\x0c", "\n").replace("\x1f", "\n")
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters except tabs/newlines; preserve Unicode source text.
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Extract existing text layer only. Never OCR page images."""
    pages: list[tuple[int, str]] = []
    errors = []
    try:
        import pymupdf  # type: ignore
        with pymupdf.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                pages.append((page_number, _clean_text(page.get_text("text"))))
        return pages
    except ImportError:
        errors.append("PyMuPDF не установлен")
    except Exception as exc:
        errors.append(f"PyMuPDF: {exc}")

    try:
        from pypdf import PdfReader  # type: ignore
        pdf = PdfReader(str(path))
        for page_number, page in enumerate(pdf.pages, start=1):
            pages.append((page_number, _clean_text(page.extract_text() or "")))
        return pages
    except ImportError:
        errors.append("pypdf не установлен")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    raise RuntimeError("Не удалось прочитать PDF: " + "; ".join(errors))



def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return physical PDF page numbers and text-layer content, rejecting scans."""
    pages = _pdf_pages(path)
    usable = [(number, text) for number, text in pages if text]
    joined = "\n\n".join(text for _number, text in usable)
    alnum = sum(ch.isalnum() for ch in joined)
    if len(joined.strip()) < 80 or alnum < 40:
        raise ValueError("в PDF не обнаружен достаточный текстовый слой; OCR отключён")
    return usable


def _zip_xml_text(path: Path, kind: str) -> str:
    """Extract readable text from DOCX/ODT/EPUB without OCR."""
    chunks: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if kind == ".docx":
            targets = ["word/document.xml"]
            targets += [n for n in names if re.fullmatch(r"word/(header|footer)\d*\.xml", n)]
        elif kind == ".odt":
            targets = ["content.xml"]
        else:
            # EPUB: read HTML/XHTML entries in package order when possible.
            targets = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]
            try:
                container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
                rootfile = next((el.attrib.get("full-path") for el in container.iter() if el.tag.endswith("rootfile")), None)
                if rootfile and rootfile in names:
                    opf = ElementTree.fromstring(archive.read(rootfile))
                    manifest = {el.attrib.get("id"): el.attrib.get("href") for el in opf.iter() if el.tag.endswith("item")}
                    spine_ids = [el.attrib.get("idref") for el in opf.iter() if el.tag.endswith("itemref")]
                    rootdir = str(Path(rootfile).parent).replace(".", "")
                    ordered = []
                    for item_id in spine_ids:
                        href = manifest.get(item_id)
                        if href:
                            candidate = f"{rootdir}/{href}" if rootdir else href
                            candidate = candidate.replace("\\", "/")
                            if candidate in names:
                                ordered.append(candidate)
                    if ordered:
                        targets = ordered
            except (KeyError, ElementTree.ParseError, StopIteration):
                pass
        for name in targets:
            if name not in names:
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
            except (ElementTree.ParseError, KeyError):
                continue
            # Insert line breaks around common block elements.
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1].lower() in {
                    "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                    "li", "tr", "table", "section", "article", "title"
                }:
                    element.text = (element.text or "") + "\n"
            text = " ".join(part.strip() for part in root.itertext() if part.strip())
            if text.strip():
                chunks.append(text)
    return _clean_text("\n\n".join(chunks))


def extract_text(path: Path) -> str:
    """Return extracted text. For PDFs, only accept an existing text layer."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = extract_pdf_pages(path)
        return "\n\n".join(f"[PDF page {number}]\n{text}" for number, text in pages)
    if suffix == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"не удалось разобрать Jupyter Notebook: {exc}") from exc
        cells = []
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if source.strip():
                kind = cell.get("cell_type", "cell")
                cells.append(f"[Cell {index}: {kind}]\n{source}")
        text = "\n\n".join(cells)
        if not text:
            raise ValueError("в Jupyter Notebook нет текста или кода для индексации")
        return _clean_text(text)
    if suffix in {".docx", ".odt", ".epub"}:
        text = _zip_xml_text(path, suffix)
        if not text:
            raise ValueError("из документа не удалось извлечь текст")
        return text
    # Texinfo and GNU Info are text formats. Preserve source markup/commands.
    return _clean_text(path.read_text(encoding="utf-8", errors="replace"))


def extract_pdf_page_range(path: Path, start_page: int, end_page: int = 0) -> str:
    """Read a PDF text layer by physical page number; no OCR."""
    pages = extract_pdf_pages(path)
    if end_page == 0:
        end_page = len(pages)
    selected = [(number, text) for number, text in pages if start_page <= number <= end_page and text]
    if not selected:
        return ""
    return "\n\n".join(f"[PDF page {number}]\n{text}" for number, text in selected)
