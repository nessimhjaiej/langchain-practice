"""Pull traced runs and LLM-judge feedback from LangSmith into one local report.

Usage:
    python report.py                 # last 24h, up to 100 runs
    python report.py --hours 72 --limit 200
Writes reports/report_<timestamp>.md and .json
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dotenv
from langsmith import Client

dotenv.load_dotenv()


def text_of(payload):
    """Best-effort readable text from a run's inputs/outputs dict."""
    if not payload:
        return ""
    for key in ("answer", "output", "input", "question"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return json.dumps(payload, default=str)[:500]


def fetch(project, hours, limit):
    client = Client()
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    runs = list(client.list_runs(project_name=project, is_root=True, start_time=start, limit=limit))
    feedback = defaultdict(list)
    ids = [r.id for r in runs]
    for i in range(0, len(ids), 50):  # chunk to keep requests small
        for fb in client.list_feedback(run_ids=ids[i:i + 50]):
            feedback[fb.run_id].append(fb)
    rows = []
    for r in runs:
        rows.append({
            "run_id": str(r.id),
            "time": r.start_time.isoformat() if r.start_time else None,
            "status": r.status,
            "latency_s": round((r.end_time - r.start_time).total_seconds(), 2) if r.end_time and r.start_time else None,
            "total_tokens": r.total_tokens,
            "question": text_of(r.inputs),
            "answer": text_of(r.outputs),
            "url": getattr(r, "url", None),
            "judges": [
                {"judge": fb.key, "score": fb.score if fb.score is not None else fb.value,
                 "comment": fb.comment}
                for fb in feedback.get(r.id, [])
            ],
        })
    return sorted(rows, key=lambda x: x["time"] or "")


def to_markdown(project, rows):
    judges = sorted({j["judge"] for row in rows for j in row["judges"]})
    out = [f"# LLM report: {project}", f"_Generated {datetime.now():%Y-%m-%d %H:%M} — {len(rows)} runs_", ""]
    if judges:
        out += ["## Judge summary", "| Judge | Scored runs | Avg score |", "|---|---|---|"]
        for name in judges:
            scores = [j["score"] for row in rows for j in row["judges"] if j["judge"] == name]
            nums = [s for s in scores if isinstance(s, (int, float))]
            avg = f"{sum(nums) / len(nums):.2f}" if nums else "n/a"
            out.append(f"| {name} | {len(scores)} | {avg} |")
        out.append("")
    out.append("## Runs")
    for n, row in enumerate(rows, 1):
        out += [f"### {n}. {row['question'][:80]}",
                f"- time: {row['time']} | status: {row['status']} | latency: {row['latency_s']}s | tokens: {row['total_tokens']}",
                f"- trace: {row['url'] or row['run_id']}", "",
                f"**Question:** {row['question']}", "", f"**Answer:** {row['answer']}", ""]
        if row["judges"]:
            out += ["| Judge | Score | Reasoning |", "|---|---|---|"]
            for j in row["judges"]:
                comment = (j["comment"] or "").replace("\n", " ").replace("|", "\\|")
                out.append(f"| {j['judge']} | {j['score']} | {comment} |")
        else:
            out.append("_No judge feedback yet (online evaluators may still be running)._")
        out.append("")
    return "\n".join(out)


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT"))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = fetch(args.project, args.hours, args.limit)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"report_{stamp}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    (out_dir / f"report_{stamp}.md").write_text(to_markdown(args.project, rows), encoding="utf-8")
    print(f"{len(rows)} runs -> reports/report_{stamp}.md (+ .json)")


if __name__ == "__main__":
    main()
