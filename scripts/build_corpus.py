#!/usr/bin/env python
"""Build a multi-format documentation corpus on a single theme:
**Modern Python Web Development** — FastAPI plus the typing/async PEPs it builds on.

Sources (all public, permissive):
  - FastAPI docs (Markdown, MIT)                 -> corpus/fastapi/*.md
  - selected Python PEPs (reStructuredText, PSF) -> corpus/peps/*.rst
  - one PEP rendered to PDF (page-cite demo)     -> corpus/pdf/*.pdf

Reproducible: re-run to rebuild the corpus from scratch.
    python scripts/build_corpus.py
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

FASTAPI_TREE = "https://api.github.com/repos/fastapi/fastapi/git/trees/master?recursive=1"
FASTAPI_RAW = "https://raw.githubusercontent.com/fastapi/fastapi/master/"
DOCS_PREFIX = "docs/en/docs/"
FASTAPI_EXCLUDE_DIRS = {"reference", "img", "css", "js", "em"}
FASTAPI_EXCLUDE_FILES = {
    "release-notes.md", "newsletter.md", "management.md", "management-tasks.md",
    "contributing.md", "external-links.md", "fastapi-people.md",
    "history-design-future.md", "benchmarks.md", "help-fastapi.md",
    "project-generation.md", "alternatives.md", "repository-management.md",
}

PEP_RAW = "https://raw.githubusercontent.com/python/peps/main/peps/"
# typing / async / style PEPs — strongly related to how FastAPI works
PEPS_RST = ["pep-0008", "pep-0020", "pep-0257", "pep-0484", "pep-0492",
            "pep-0585", "pep-0604"]
PEP_PDF = "pep-0544"  # Protocols (structural subtyping) — rendered to PDF


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "docagent-corpus"})
    return urllib.request.urlopen(req, timeout=30).read()


def build_fastapi() -> int:
    out = CORPUS / "fastapi"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.md"):
        if f.name != "SOURCE.md":
            f.unlink()
    tree = json.loads(_get(FASTAPI_TREE))["tree"]
    n = 0
    for node in tree:
        p = node.get("path", "")
        if not (p.startswith(DOCS_PREFIX) and p.endswith(".md")):
            continue
        rel = p[len(DOCS_PREFIX):]
        top = rel.split("/")[0] if "/" in rel else ""
        if top in FASTAPI_EXCLUDE_DIRS or Path(rel).name in FASTAPI_EXCLUDE_FILES:
            continue
        (out / rel.replace("/", "-")).write_bytes(_get(FASTAPI_RAW + p))
        n += 1
    # MIT license for attribution
    try:
        (out / "LICENSE").write_bytes(
            _get("https://raw.githubusercontent.com/fastapi/fastapi/master/LICENSE")
        )
    except Exception:  # noqa: BLE001
        pass
    print(f"FastAPI md: {n}")
    return n


def build_peps() -> int:
    out = CORPUS / "peps"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.rst"):
        f.unlink()
    n = 0
    for pep in PEPS_RST:
        try:
            (out / f"{pep}.rst").write_bytes(_get(f"{PEP_RAW}{pep}.rst"))
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"skip {pep}: {e}")
    print(f"PEPs rst: {n}")
    return n


def build_pdf() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    out = CORPUS / "pdf"
    out.mkdir(parents=True, exist_ok=True)
    text = _get(f"{PEP_RAW}{PEP_PDF}.rst").decode("utf-8", "ignore")
    pdf_path = out / f"{PEP_PDF}-protocols.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin
    c.setFont("Helvetica", 9)
    for raw in text.splitlines():
        line = raw[:110]
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - margin
        c.drawString(margin, y, line)
        y -= 12
    c.save()
    print(f"PDF: {pdf_path.name}")


def write_source_note() -> None:
    (CORPUS / "SOURCE.md").write_text(
        "# Corpus sources & attribution\n\n"
        "Theme: **Modern Python Web Development**. Used only as sample / evaluation\n"
        "data for docagent; each source keeps its own license.\n\n"
        "- `fastapi/*.md` — FastAPI documentation subset. MIT "
        "(see `fastapi/LICENSE`). (c) 2018 Sebastian Ramirez.\n"
        "- `peps/*.rst` — Python Enhancement Proposals (typing/async/style). "
        "Public, authored for the Python Software Foundation.\n"
        "- `pdf/pep-0544-protocols.pdf` — PEP 544 (Protocols) rendered to PDF to "
        "demonstrate PDF ingestion + page-level citations. Original text public (PSF).\n",
        encoding="utf-8",
    )


def main():
    build_fastapi()
    build_peps()
    build_pdf()
    write_source_note()
    print("Done.")


if __name__ == "__main__":
    main()
