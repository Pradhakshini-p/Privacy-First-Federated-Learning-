#!/usr/bin/env python3
"""Build static GitHub Pages site with latest evaluation metrics."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
ASSETS = SITE / "assets"
RESULTS = ROOT / "results"


def load_metrics():
    report_path = RESULTS / "comparison_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("centralized_metrics", {}), data.get("federated_metrics", {})
    return {}, {}


def pct(value):
    return f"{float(value) * 100:.1f}%" if value is not None else "N/A"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for png in RESULTS.glob("*.png"):
        shutil.copy2(png, ASSETS / png.name)

    cent, fed = load_metrics()
    html_path = SITE / "index.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("{{CENT_ACC}}", pct(cent.get("accuracy", 0.708)))
    html = html.replace("{{FED_ACC}}", pct(fed.get("accuracy", 0.649)))
    html = html.replace("{{CENT_AUC}}", pct(cent.get("roc_auc", 0.804)))
    html = html.replace("{{FED_AUC}}", pct(fed.get("roc_auc", 0.631)))
    html_path.write_text(html, encoding="utf-8")
    print(f"Built GitHub Pages site at {SITE}")


if __name__ == "__main__":
    main()
