"""Aggregate multiple variance summaries into a simple markdown report."""

from __future__ import annotations

import glob
import json
import os


def aggregate_summaries(runs_glob: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(runs_glob))
    rows = []
    for path in files:
        with open(path, "r") as f:
            payload = json.load(f)
        rows.append({"path": path, "payload": payload})

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# Variance report\n\n")
        for row in rows:
            f.write(f"## {row['path']}\n\n")
            f.write("```json\n")
            f.write(json.dumps(row["payload"], indent=2))
            f.write("\n```\n\n")
    return {"report_path": report_path, "n_runs": len(rows)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = aggregate_summaries(args.runs_glob, args.out_dir)
    print(json.dumps(out, indent=2))
