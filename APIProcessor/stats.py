"""
Summary statistics for a finished run (models + judge), against the ground-truth labels.

Reports:
  - Per model: accuracy, precision / recall / F1 for "Scam", confusion matrix
  - Accuracy split by context (with / without) and by scam_type
  - Judge rubric % yes per model, and how often the judge's classification_correct
    matches the truth
  - Every row a model got wrong

Usage:
    python stats.py -f older_adults_scams_dataset.json
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

from config import RUBRIC_FIELDS
from eval_utils import yes_rate
from main import HAS_RICH, console, load_json_file, print_error, print_success

if HAS_RICH:
    from rich.table import Table


def pct(n: int, d: int) -> Optional[float]:
    return 100.0 * n / d if d else None


def fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.1f}%" if isinstance(v, float) else str(v)


def show_table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    if HAS_RICH and console:
        tbl = Table(title=title)
        for c in columns:
            tbl.add_column(c)
        for r in rows:
            tbl.add_row(*r)
        console.print(tbl)
    else:
        print(f"\n{title}")
        print(" | ".join(columns))
        for r in rows:
            print(" | ".join(r))


def classification_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(r["model_verdict"] == "Scam" and r["label"] == "Scam" for r in recs)
    fp = sum(r["model_verdict"] == "Scam" and r["label"] == "Not a scam" for r in recs)
    fn = sum(r["model_verdict"] == "Not a scam" and r["label"] == "Scam" for r in recs)
    tn = sum(r["model_verdict"] == "Not a scam" and r["label"] == "Not a scam" for r in recs)
    precision = pct(tp, tp + fp)
    recall = pct(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"n": len(recs), "accuracy": pct(tp + tn, len(recs)), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    parser = argparse.ArgumentParser(description="Summary statistics for a finished run.")
    parser.add_argument("-f", "--file", default="older_adults_scams_dataset.json", help="Labeled dataset JSON used for the run")
    parser.add_argument("-j", "--judge-dir", default=os.path.join("output", "judge"))
    parser.add_argument("--report", default=os.path.join("output", "stats_report.json"))
    args = parser.parse_args()

    rows = load_json_file(args.file)
    recs = []
    for path in sorted(glob.glob(os.path.join(args.judge_dir, "judge_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            recs.extend(json.load(f))
    recs = [r for r in recs if r.get("label") and r.get("model_verdict")]
    if not recs:
        print_error("No judged items with labels found. Run main.py and judge.py on a labeled dataset first.")
        sys.exit(1)
    for r in recs:
        r["scam_type"] = rows[r["row_index"]].get("scam_type", "")

    models = sorted({r["model_name"] for r in recs})
    report: Dict[str, Any] = {"per_model": {}, "by_context": {}, "by_scam_type": {}, "judge": {}, "errors": []}

    # --- Per-model classification ---------------------------------------------
    table = []
    for m in models:
        s = classification_stats([r for r in recs if r["model_name"] == m])
        report["per_model"][m] = s
        table.append([m, str(s["n"]), fmt(s["accuracy"]), fmt(s["precision"]), fmt(s["recall"]), fmt(s["f1"]),
                      f"{s['tp']} / {s['fp']} / {s['fn']} / {s['tn']}"])
    show_table("Model verdicts vs ground truth (positive class = Scam)",
               ["Model", "N", "Accuracy", "Precision", "Recall", "F1", "TP / FP / FN / TN"], table)

    # --- By context and scam type ---------------------------------------------
    for key, title in (("context", "Accuracy by context"), ("scam_type", "Accuracy by scam_type")):
        groups = sorted({r[key] for r in recs})
        table = []
        for g in groups:
            row = [g, str(sum(1 for r in recs if r[key] == g and r["model_name"] == models[0]))]
            for m in models:
                sub = [r for r in recs if r[key] == g and r["model_name"] == m]
                acc = pct(sum(r["model_verdict"] == r["label"] for r in sub), len(sub))
                report[f"by_{key}"].setdefault(g, {})[m] = acc
                row.append(fmt(acc))
            table.append(row)
        show_table(title, [key, "Prompts", *models], table)

    # --- Judge -----------------------------------------------------------------
    table = []
    for m in models:
        sub = [r for r in recs if r["model_name"] == m]
        truth = ["yes" if r["model_verdict"] == r["label"] else "no" for r in sub]
        judge_cc = [r.get("judge_classification_correct") for r in sub]
        judge_acc = pct(sum(j == t for j, t in zip(judge_cc, truth)), len(sub))
        rates = {f: yes_rate([r.get(f"judge_{f}") for r in sub if r.get(f"judge_{f}")]) for f in RUBRIC_FIELDS}
        report["judge"][m] = {"classification_correct_matches_truth": judge_acc, "percent_yes": rates}
        table.append([m, fmt(judge_acc)] + [fmt(rates[f]) for f in RUBRIC_FIELDS])
    show_table("LLM judge: does classification_correct match truth? + rubric % yes",
               ["Model", "Judge right on correctness", *RUBRIC_FIELDS], table)

    # --- Errors ----------------------------------------------------------------
    wrong = [r for r in recs if r["model_verdict"] != r["label"]]
    report["errors"] = [{k: r[k] for k in ("model_name", "row_index", "label", "model_verdict", "scam_type", "prompt")}
                        for r in wrong]
    if wrong:
        show_table(f"Wrong verdicts ({len(wrong)})", ["Model", "Row", "Truth", "Model said", "Type", "Prompt"],
                   [[r["model_name"], str(r["row_index"]), r["label"], r["model_verdict"], r["scam_type"],
                     r["prompt"][:70]] for r in sorted(wrong, key=lambda r: (r["row_index"], r["model_name"]))])

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print_success(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
