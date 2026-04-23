#!/usr/bin/env python3
"""Build processed development themes from downloaded paper index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.json"
THEMES_FILE = ROOT / "themes.json"
OUT_FILE = ROOT / "development_themes.json"


def load() -> tuple[dict, dict]:
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    themes = json.loads(THEMES_FILE.read_text(encoding="utf-8"))
    return index, themes


def paper_confidence(requested_title: str, resolved_title: str) -> str:
    req = set(w for w in requested_title.lower().replace("-", " ").split() if len(w) > 3)
    got = set(w for w in resolved_title.lower().replace("-", " ").split() if len(w) > 3)
    if not req:
        return "low"
    overlap = len(req & got) / max(1, len(req))
    if overlap >= 0.5:
        return "high"
    if overlap >= 0.25:
        return "medium"
    return "low"


def build(index: dict, themes: dict) -> dict:
    by_title: Dict[str, dict] = {r["title"]: r for r in index.get("records", [])}

    out_themes: List[dict] = []
    for t in themes.get("themes", []):
        papers_out = []
        for p in t.get("papers", []):
            rec = by_title.get(p["title"], {})
            resolved_title = rec.get("arxiv_title") or p["title"]
            papers_out.append(
                {
                    "requested_title": p["title"],
                    "resolved": rec.get("resolved", False),
                    "resolved_title": resolved_title,
                    "arxiv_id": rec.get("arxiv_id"),
                    "pdf_path": rec.get("pdf_path"),
                    "confidence": paper_confidence(p["title"], resolved_title),
                    "abstract": rec.get("abstract"),
                }
            )

        out_themes.append(
            {
                "id": t["id"],
                "name": t["name"],
                "goal": t["goal"],
                "integration_targets": t.get("integration_targets", []),
                "roadmap": {
                    "phase_1": "prototype",
                    "phase_2": "ablation",
                    "phase_3": "production hardening",
                },
                "papers": papers_out,
            }
        )

    return {
        "generated_at": index.get("generated_at"),
        "total_themes": len(out_themes),
        "total_papers": len(index.get("records", [])),
        "resolved_count": index.get("resolved_count", 0),
        "themes": out_themes,
    }


def main() -> None:
    index, themes = load()
    output = build(index, themes)
    OUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_file": str(OUT_FILE),
        "total_themes": output["total_themes"],
        "total_papers": output["total_papers"],
        "resolved_count": output["resolved_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
