#!/usr/bin/env python3
"""Download/resolve research papers listed in papers/themes.json.

Workflow:
1. Read development themes and paper titles from themes.json
2. Resolve candidate papers from arXiv API by title/query
3. Download PDFs when available
4. Save machine-readable index for downstream processing
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
THEMES_FILE = ROOT / "themes.json"
DOWNLOAD_DIR = ROOT / "downloads"
INDEX_FILE = ROOT / "index.json"

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:120]


@dataclass
class PaperRecord:
    title: str
    query: str
    year_hint: Optional[int]
    theme_id: str
    source: str
    resolved: bool
    arxiv_id: Optional[str] = None
    arxiv_title: Optional[str] = None
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    notes: Optional[str] = None


def arxiv_search(query: str, max_results: int = 5) -> List[dict]:
    q = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{q}"
    with urllib.request.urlopen(url, timeout=30) as r:
        xml_data = r.read()

    root = ET.fromstring(xml_data)
    entries = []
    for e in root.findall("a:entry", NS):
        arxiv_id = e.find("a:id", NS).text.strip().split("/")[-1]
        title = (e.find("a:title", NS).text or "").replace("\n", " ").strip()
        summary = (e.find("a:summary", NS).text or "").replace("\n", " ").strip()
        pdf_url = None
        for link in e.findall("a:link", NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "pdf_url": pdf_url,
            }
        )
    return entries


def choose_best_match(title: str, candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None
    title_l = title.lower()
    # Prefer substring overlap on words from target title.
    words = [w for w in re.findall(r"[a-z0-9]+", title_l) if len(w) > 3]
    best = None
    best_score = -1
    for c in candidates:
        t = c["title"].lower()
        score = sum(1 for w in words if w in t)
        if score > best_score:
            best_score = score
            best = c
    # If no overlap at all, still return top candidate as weak match.
    return best


def download_pdf(pdf_url: str, out_file: Path) -> bool:
    try:
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "openmythos-swarm-paper-fetcher/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out_file.write_bytes(r.read())
        return True
    except Exception:
        return False


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    themes = json.loads(THEMES_FILE.read_text(encoding="utf-8"))
    out: List[PaperRecord] = []

    for theme in themes.get("themes", []):
        theme_id = theme.get("id", "unknown")
        for p in theme.get("papers", []):
            title = p.get("title", "")
            query = p.get("query", title)
            year_hint = p.get("year")

            try:
                candidates = arxiv_search(query)
                chosen = choose_best_match(title, candidates)
                if not chosen:
                    out.append(
                        PaperRecord(
                            title=title,
                            query=query,
                            year_hint=year_hint,
                            theme_id=theme_id,
                            source="arxiv",
                            resolved=False,
                            notes="No arXiv candidates returned",
                        )
                    )
                    continue

                filename = f"{slugify(title)}__{chosen['arxiv_id'].replace('/', '_')}.pdf"
                pdf_path = DOWNLOAD_DIR / filename
                ok = download_pdf(chosen["pdf_url"], pdf_path)

                out.append(
                    PaperRecord(
                        title=title,
                        query=query,
                        year_hint=year_hint,
                        theme_id=theme_id,
                        source="arxiv",
                        resolved=ok,
                        arxiv_id=chosen["arxiv_id"],
                        arxiv_title=chosen["title"],
                        abstract=chosen["summary"],
                        pdf_url=chosen["pdf_url"],
                        pdf_path=str(pdf_path) if ok else None,
                        notes=None if ok else "Failed PDF download",
                    )
                )
            except Exception as e:
                out.append(
                    PaperRecord(
                        title=title,
                        query=query,
                        year_hint=year_hint,
                        theme_id=theme_id,
                        source="arxiv",
                        resolved=False,
                        notes=f"Resolver error: {e}",
                    )
                )

    summary = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "total_requested": len(out),
        "resolved_count": sum(1 for r in out if r.resolved),
        "unresolved_count": sum(1 for r in out if not r.resolved),
        "records": [asdict(r) for r in out],
    }

    INDEX_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "total_requested": summary["total_requested"],
        "resolved_count": summary["resolved_count"],
        "unresolved_count": summary["unresolved_count"],
        "index": str(INDEX_FILE),
        "downloads_dir": str(DOWNLOAD_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
